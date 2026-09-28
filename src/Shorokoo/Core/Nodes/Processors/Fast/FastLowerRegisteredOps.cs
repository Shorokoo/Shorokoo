using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Lowering;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Rewrites every node named by the caller's list of op codes into its registered
    /// <see cref="OpLowering"/>'s decomposition, in one walk over the graph.
    ///
    /// <para><b>The caller names the operators outright.</b> Each domain keeps its own list, at
    /// its own call site: the QuickExecutionEngine's names what it cannot compute, the autodiff
    /// pass's what it cannot differentiate, the ONNX builder's what it cannot emit. A list, not a
    /// predicate over what the domain happens to have a kernel or a rule for — those tables also
    /// hold entries that exist only to report the operator unsupported, and deriving the set from
    /// them would refuse to lower exactly the operators lowering was meant to rescue.</para>
    ///
    /// <para><b>This rewrites the graph it is handed</b>, and what that costs the caller is the
    /// caller's to decide. The QuickExecutionEngine clones first, so the decomposition is private
    /// to one run and the graph handed to it keeps its <c>Softsign</c>; the autodiff pass lowers
    /// the training graph it is expanding, which from then on carries the decomposition in place
    /// of the operator; the ONNX builder lowers the copy it already made to prepare the export, so
    /// only the written file is decomposed. A graph with no <c>AUTO_GRAD</c> node never reaches
    /// the autodiff pass, and <c>Softsign</c> is on no export list, so an inference model still
    /// exports its <c>Softsign</c> as a <c>Softsign</c>.</para>
    ///
    /// <para><b>Keys are preserved.</b> Like the other <c>FastLower*</c> passes, this one does not
    /// rewire consumers. The decomposition's non-terminal nodes are inserted at the lowered node's
    /// own index, and the node itself is mutated into the terminal one — same
    /// <see cref="FastNode.Key"/>, same <see cref="FastNode.FullOutputs"/> — so every consumer,
    /// and every key a caller looks the run's results up by, stays valid. Everything else it takes
    /// from the terminal, the open node, function and identifier template its operator comes with
    /// included: a decomposition ending in an <c>If</c> makes the node the <c>IF_CLOSE</c> paired
    /// with the spliced <c>IF_OPEN</c> (<see cref="FastSplice"/>). Inserting at the node's
    /// index rather than appending is what keeps a decomposition inside the loop or branch scope
    /// its operator sat in; nodes after the scope's close would never be re-executed by a
    /// loop-back.</para>
    ///
    /// <para><b>Every spliced node is re-keyed.</b> One decomposition is built per distinct shape
    /// of call and then reused, and the nodes it was converted from carry keys derived from the
    /// built <see cref="Node"/>s — so two occurrences of the same operator would otherwise be
    /// spliced under one set of <see cref="FastNodeKey"/>s and collide in the graph. Each
    /// occurrence therefore gets a fresh key per node, with every reference to one — input and
    /// output tensor keys, a close node's open-node key — remapped to match.</para>
    /// </summary>
    internal static class FastLowerRegisteredOps
    {
        /// <summary>
        /// Whether <paramref name="graph"/> holds anything this pass would rewrite for a caller
        /// whose list is <paramref name="opCodes"/>. A caller runs this first so a graph with no
        /// lowerable operator — the overwhelming majority — pays one linear scan instead of a
        /// clone and a Variable-level rebuild of every tensor; an empty list is answered without
        /// even that, so a domain that lowers nothing yet costs nothing.
        /// </summary>
        public static bool HasLowerableOp(InternalComputationGraph graph, IReadOnlySet<string> opCodes)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (opCodes is null) throw new ArgumentNullException(nameof(opCodes));
            if (opCodes.Count == 0) return false;

            foreach (var node in graph.Nodes)
                if (IsLowerable(node, opCodes)) return true;
            return false;
        }

        /// <summary>
        /// Lowers every node in <paramref name="graph"/> whose op code <paramref name="opCodes"/>
        /// names and that has a registered lowering.
        /// </summary>
        public static void Process(InternalComputationGraph graph, IReadOnlySet<string> opCodes)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (opCodes is null) throw new ArgumentNullException(nameof(opCodes));
            if (!HasLowerableOp(graph, opCodes)) return;

            var tensorInfo = BuildTensorInfo(graph);

            // Local to this call, not to the process: a decomposition is spliced into one graph
            // and re-keyed per occurrence, so keeping it past the pass would only hold the
            // built nodes alive for a graph that no longer exists.
            var plans = new Dictionary<string, FastSplice.Plan>(StringComparer.Ordinal);

            var newNodes = new List<FastNode>(graph.Nodes.Count);
            foreach (var node in graph.Nodes)
            {
                var plan = IsLowerable(node, opCodes) && OpLoweringRegistry.TryGet(node.OpCode, out var lowering)
                    ? TryPlan(lowering, node, tensorInfo, plans)
                    : null;

                if (plan is null) newNodes.Add(node);
                else FastSplice.SpliceInPlace(node, plan, newNodes);
            }
            graph.Nodes = newNodes;

            Debug.Assert(graph.TryValidateLinearOrder(out var orderError),
                "graph.IsLinearOrderValid(): " + orderError);
        }

        /// <summary>
        /// The decomposition of <paramref name="lowering"/> for a call whose input slots carry
        /// <paramref name="inputs"/> (a null entry is an omitted optional input) and whose
        /// operator carries <paramref name="attributes"/>, as nodes ready to splice: the terminal
        /// node — the one producing all <paramref name="declaredOutputs"/> outputs — last.
        ///
        /// <para>Null when the decomposition cannot stand in for the node key-preservingly: it
        /// produced a different number of outputs than the operator declares, handed one of its
        /// inputs straight back, or spread its outputs over more than one node. Rewiring the
        /// consumers would be the alternative, and this pass does not rewire.</para>
        ///
        /// <para>Throws when the decomposition is wrong rather than merely unusable: a lowering
        /// that builds its own op code cannot be built from itself, and one that builds an
        /// operator with no <see cref="QuickOp"/> has not reached the framework's primitives —
        /// <see cref="OpRegistry"/> is the roster of those, whichever caller is lowering, and a
        /// lowering is a single step down that never consults <see cref="OpLoweringRegistry"/>
        /// again.</para>
        ///
        /// <para>That second test asks whether the QuickExecutionEngine has a kernel for the
        /// operator, which is its own question rather than every caller's: the exporter's is
        /// whether the operator can be emitted. The two answers agree over every operator the
        /// framework defines today, so the one test serves all three callers.</para>
        ///
        /// <para>It also happens to stop a lowering being built out of another lowerable
        /// operator, since no operator a caller lowers has a kernel today. That is a coincidence
        /// and not a guard: a caller's list is stated rather than derived from the kernel table,
        /// so it may name an operator that does have one, and such an operator would pass this
        /// test, be spliced in, and stay — <see cref="Process"/> walks the node list once and
        /// never revisits what it splices. Nothing reaches that shape today.</para>
        /// </summary>
        internal static FastSplice.Plan? Decompose(
            OpLowering lowering,
            (DType DType, int? Rank)?[] inputs,
            OnnxCSharpAttributes attributes,
            int declaredOutputs)
        {
            var standIns = FastSplice.StandIns(inputs);
            var outputs = lowering.Build(standIns, attributes);
            if (outputs.Length != declaredOutputs || outputs.Any(x => x is null)) return null;

            // The stand-ins are the built graph's input prefix and its output nodes its suffix; the
            // decomposition is what lies between.
            var plan = FastSplice.Build(standIns, [.. outputs.Select(x => x!)]);
            if (plan is null) return null;

            foreach (var node in plan.Body)
            {
                if (string.Equals(node.OpCode, lowering.OpCode, StringComparison.Ordinal))
                    throw new InvalidOperationException(
                        $"Operator lowering for '{lowering.OpCode}' built '{lowering.OpCode}', "
                        + "which it cannot be built from.");

                if (OpRegistry.Get(node.OpCode) is null)
                    throw new InvalidOperationException(
                        $"Operator lowering for '{lowering.OpCode}' built '{node.OpCode}', which the "
                        + "QuickExecutionEngine has no operator for.");
            }

            return plan.TerminalProducesOutputs ? plan : null;
        }

        private static bool IsLowerable(FastNode node, IReadOnlySet<string> opCodes)
            => opCodes.Contains(node.OpCode) && OpLoweringRegistry.TryGet(node.OpCode, out _);

        /// <summary>
        /// Dtype and rank for every tensor in <paramref name="graph"/>, or nothing when the graph
        /// cannot be rebuilt at the Variable level — a partial graph handed to the engine mid-pass
        /// is one the engine still runs as far as it can, so a lookup that cannot be built falls
        /// back to <see cref="StandInDType"/> per slot rather than failing the run.
        /// </summary>
        private static Dictionary<FastTensorKey, FastTensorInfo> BuildTensorInfo(
            InternalComputationGraph graph)
        {
            try { return FastTensorInfoProcessor.BuildTensorInfoLookup(graph); }
            catch { return []; }
        }

        /// <summary>
        /// The plan for <paramref name="node"/>, built on first sight of its shape and kept for
        /// the rest of the pass, or null when the operator cannot be lowered — which leaves the
        /// node exactly as it was, for the engine to give up on as it would any operator it has no
        /// kernel for. Only a plan that built cleanly is kept, so a decomposition that is refused
        /// is refused at every node it appears on.
        /// </summary>
        private static FastSplice.Plan? TryPlan(
            OpLowering lowering,
            FastNode node,
            Dictionary<FastTensorKey, FastTensorInfo> tensorInfo,
            Dictionary<string, FastSplice.Plan> plans)
        {
            var inputs = node.Inputs;
            var declaredOutputs = node.Outputs.Count;

            var descriptors = new (DType DType, int? Rank)?[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] is not { } key || key.IsEmpty) continue;
                descriptors[i] = tensorInfo.TryGetValue(key, out var info)
                    ? (info.DType, info.Rank)
                    : (StandInDType(lowering, i), null);
            }

            var cacheKey = TryBuildKey(lowering, descriptors, node.Attributes, declaredOutputs);
            if (cacheKey is not null && plans.TryGetValue(cacheKey, out var cached)) return cached;

            FastSplice.Plan? plan;
            try { plan = Decompose(lowering, descriptors, node.Attributes, declaredOutputs); }
            catch { plan = null; }

            if (plan is not null && cacheKey is not null) plans[cacheKey] = plan;
            return plan;
        }

        /// <summary>
        /// The dtype to build slot <paramref name="slot"/>'s stand-in at when the graph does not
        /// say what that slot holds — every slot of every node, when the tensor-info lookup could
        /// not be built at all.
        ///
        /// <para><see cref="DType.Invalid"/> is the reading a missing entry invites and it is the
        /// wrong one: <c>invalid</c> is not the absence of a type but a type that fails every
        /// constraint, so a decomposition whose operands must be numeric throws before it builds
        /// anything, the node is left as it stands, and the engine — which now has no kernel for
        /// it — writes a placeholder. That loses a value the engine can perfectly well compute,
        /// which is the opposite of running a partial graph as far as it can.</para>
        ///
        /// <para>The lowering's own signature answers the question instead. A slot it declares as
        /// <c>Tensor&lt;int64&gt;</c> holds int64 whatever the graph says; a slot it declares
        /// generically is one it is written to handle at any dtype, so any concrete dtype builds
        /// the same decomposition and float32 — which satisfies every type constraint a lowering's
        /// operands carry — is the one taken.</para>
        ///
        /// <para>What a stand-in's dtype decides is what can be BUILT, not what is computed. A
        /// <see cref="FastNode"/> carries no per-tensor dtype, and the engine computes each node
        /// from the runtime dtypes of the tensors actually in its store, so building a
        /// <c>Softsign</c>'s decomposition at float32 does not make a float64 <c>Softsign</c>
        /// compute in float32. That holds only because a decomposition reads its operands' types
        /// at runtime rather than in C# — see <see cref="OpLowerings"/>.</para>
        /// </summary>
        private static DType StandInDType(OpLowering lowering, int slot)
        {
            var parameters = lowering.Method.GetParameters();
            if (slot >= parameters.Length) return DType.Float32;

            var declared = Nullable.GetUnderlyingType(parameters[slot].ParameterType)
                ?? parameters[slot].ParameterType;
            return declared.IsGenericType
                && declared.GenericTypeArguments.Length == 1
                && Shorokoo.Core.Utils.OnnxUtils.GetDType(declared.GenericTypeArguments[0]) is { } dtype
                ? dtype
                : DType.Float32;
        }

        /// <summary>
        /// What a plan may be reused for: the lowering itself and, as
        /// <see cref="FastSplice.TryBuildKey"/> renders them, the inputs, the declared output
        /// count and the attribute values. Null when some attribute has no stable rendering.
        /// </summary>
        internal static string? TryBuildKey(
            OpLowering lowering,
            (DType DType, int? Rank)?[] inputs,
            OnnxCSharpAttributes attributes,
            int declaredOutputs)
            => FastSplice.TryBuildKey($"{lowering.OpCode}/{lowering.Method.MethodHandle.Value}",
                inputs, attributes, declaredOutputs)?.ToString();
    }
}
