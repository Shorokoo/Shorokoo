using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Lowering;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Rewrites every node whose operator has a registered <see cref="OpLowering"/> but that the
    /// caller cannot carry out itself into that lowering's decomposition, in one walk over the
    /// graph.
    ///
    /// <para>Which operators those are is the caller's to say, and the two callers say different
    /// things: the QuickExecutionEngine lowers what it has no <see cref="QuickOp"/> for, the
    /// autodiff engine what it has no <c>[AutoDiff]</c> rule for. An operator the caller can
    /// carry out keeps it either way: a lowering is the fallback for an operator that cannot be,
    /// not a preferred spelling of one that can.</para>
    ///
    /// <para><b>This rewrites the graph it is handed</b>, and what that costs the caller is the
    /// caller's to decide. The QuickExecutionEngine clones first, so the decomposition is private
    /// to one run and the graph handed to it keeps its <c>Softsign</c>; the autodiff pass lowers
    /// the training graph it is expanding, which from then on carries the decomposition in place
    /// of the operator. A graph with no <c>AUTO_GRAD</c> node never reaches the autodiff pass, so
    /// an inference model still exports its <c>Softsign</c> as a <c>Softsign</c>.</para>
    ///
    /// <para><b>Keys are preserved.</b> Like the other <c>FastLower*</c> passes, this one does not
    /// rewire consumers. The decomposition's non-terminal nodes are inserted at the lowered node's
    /// own index, and the node itself is mutated into the terminal one — same
    /// <see cref="FastNode.Key"/>, same <see cref="FastNode.FullOutputs"/> — so every consumer,
    /// and every key a caller looks the run's results up by, stays valid. Inserting at the node's
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
        // Field and group separators for the plan cache key: control characters, so no op code,
        // attribute name, dtype name or attribute value can be mistaken for one.
        private const char Sep = '\u0001';
        private const char Group = '\u0002';

        /// <summary>
        /// Whether <paramref name="graph"/> holds anything this pass would rewrite for a caller
        /// that carries out the operators <paramref name="handledDirectly"/> accepts. A caller
        /// runs this first so a graph with no lowerable operator — the overwhelming majority —
        /// pays one linear scan instead of a clone and a Variable-level rebuild of every tensor.
        /// </summary>
        public static bool HasLowerableOp(InternalComputationGraph graph, Func<string, bool> handledDirectly)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (handledDirectly is null) throw new ArgumentNullException(nameof(handledDirectly));

            foreach (var node in graph.Nodes)
                if (IsLowerable(node, handledDirectly)) return true;
            return false;
        }

        /// <summary>
        /// Lowers every node in <paramref name="graph"/> that has a lowering and that
        /// <paramref name="handledDirectly"/> says the caller cannot carry out as it stands.
        /// </summary>
        public static void Process(InternalComputationGraph graph, Func<string, bool> handledDirectly)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (handledDirectly is null) throw new ArgumentNullException(nameof(handledDirectly));
            if (!HasLowerableOp(graph, handledDirectly)) return;

            var tensorInfo = BuildTensorInfo(graph);

            // Local to this call, not to the process: a decomposition is spliced into one graph
            // and re-keyed per occurrence, so keeping it past the pass would only hold the
            // built nodes alive for a graph that no longer exists.
            var plans = new Dictionary<string, LoweredPlan>(StringComparer.Ordinal);

            var newNodes = new List<FastNode>(graph.Nodes.Count);
            foreach (var node in graph.Nodes)
            {
                var plan = IsLowerable(node, handledDirectly) && OpLoweringRegistry.TryGet(node.OpCode, out var lowering)
                    ? TryPlan(lowering, node, tensorInfo, plans)
                    : null;

                if (plan is null) newNodes.Add(node);
                else Splice(node, plan, newNodes);
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
        /// </summary>
        internal static LoweredPlan? Decompose(
            OpLowering lowering,
            (DType DType, int? Rank)?[] inputs,
            OnnxCSharpAttributes attributes,
            int declaredOutputs)
        {
            var standIns = new Variable?[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                if (inputs[i] is { } descriptor)
                    standIns[i] = InternalOp.RuntimeInput(descriptor.DType, descriptor.Rank);

            var outputs = lowering.Build(standIns, attributes);
            if (outputs.Length != declaredOutputs || outputs.Any(x => x is null)) return null;

            ImmutableArray<Variable> presentStandIns = [.. standIns.Where(x => x is not null).Select(x => x!)];
            var built = new InternalComputationGraph(presentStandIns, [.. outputs.Select(x => x!)]);

            var standInKeyBySlot = new FastTensorKey?[inputs.Length];
            for (int i = 0, present = 0; i < standIns.Length; i++)
                if (standIns[i] is not null) standInKeyBySlot[i] = built.Inputs[present++];

            var standInKeys = new HashSet<FastTensorKey>(built.Inputs);
            List<FastNode> body = [.. built.Nodes.Where(n => !ProducesAny(n, standInKeys))];
            if (body.Count == 0) return null;

            foreach (var node in body)
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

            var terminalOutputs = body[^1].Outputs;
            if (terminalOutputs.Count != declaredOutputs) return null;
            for (int i = 0; i < declaredOutputs; i++)
                if (terminalOutputs[i] != built.Outputs[i]) return null;

            return new LoweredPlan(body, standInKeyBySlot);
        }

        /// <summary>
        /// Everything about one lowered operator that does not depend on which occurrence of it is
        /// being rewritten: the decomposition's nodes, terminal last, and the tensor key standing
        /// in for each of the operator's input slots.
        /// </summary>
        internal sealed record LoweredPlan(List<FastNode> Body, FastTensorKey?[] StandInKeyBySlot);

        private static bool IsLowerable(FastNode node, Func<string, bool> handledDirectly)
            => OpLoweringRegistry.TryGet(node.OpCode, out _) && !handledDirectly(node.OpCode);

        private static bool ProducesAny(FastNode node, HashSet<FastTensorKey> keys)
        {
            foreach (var group in node.FullOutputs)
                foreach (var key in group.Value)
                    if (key is { } k && keys.Contains(k)) return true;
            return false;
        }

        /// <summary>
        /// Dtype and rank for every tensor in <paramref name="graph"/>, or nothing when the graph
        /// cannot be rebuilt at the Variable level — a partial graph handed to the engine mid-pass
        /// is one the engine still runs as far as it can, so a lookup that cannot be built leaves
        /// each stand-in untyped rather than failing the run.
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
        private static LoweredPlan? TryPlan(
            OpLowering lowering,
            FastNode node,
            Dictionary<FastTensorKey, FastTensorInfo> tensorInfo,
            Dictionary<string, LoweredPlan> plans)
        {
            var inputs = node.Inputs;
            var declaredOutputs = node.Outputs.Count;

            var descriptors = new (DType DType, int? Rank)?[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] is not { } key || key.IsEmpty) continue;
                descriptors[i] = tensorInfo.TryGetValue(key, out var info)
                    ? (info.DType, info.Rank)
                    : (DType.Invalid, null);
            }

            var cacheKey = TryBuildKey(lowering, descriptors, node.Attributes, declaredOutputs);
            if (cacheKey is not null && plans.TryGetValue(cacheKey, out var cached)) return cached;

            LoweredPlan? plan;
            try { plan = Decompose(lowering, descriptors, node.Attributes, declaredOutputs); }
            catch { plan = null; }

            if (plan is not null && cacheKey is not null) plans[cacheKey] = plan;
            return plan;
        }

        /// <summary>
        /// Inserts <paramref name="plan"/>'s decomposition for <paramref name="node"/> into
        /// <paramref name="newNodes"/>, under keys minted for this occurrence, and mutates
        /// <paramref name="node"/> into the terminal step.
        /// </summary>
        private static void Splice(FastNode node, LoweredPlan plan, List<FastNode> newNodes)
        {
            var hostInputs = node.Inputs;

            var tensorMap = new Dictionary<FastTensorKey, FastTensorKey>();
            for (int i = 0; i < plan.StandInKeyBySlot.Length; i++)
                if (plan.StandInKeyBySlot[i] is { } standIn && hostInputs[i] is { } host)
                    tensorMap[standIn] = host;

            var nodeMap = new Dictionary<FastNodeKey, FastNodeKey>(plan.Body.Count);
            foreach (var built in plan.Body) nodeMap[built.Key] = FastNodeKey.New();
            foreach (var built in plan.Body)
                foreach (var group in built.FullOutputs)
                    foreach (var key in group.Value)
                        if (key is { } k && !k.IsEmpty)
                            tensorMap[k] = new FastTensorKey(nodeMap[k.FastNodeKey], k.OutputIndex);

            for (int i = 0; i < plan.Body.Count - 1; i++)
            {
                var built = plan.Body[i];
                var copy = new FastNode
                {
                    Key = nodeMap[built.Key],
                    OpCode = built.OpCode,
                    Attributes = built.Attributes,
                    FriendlyName = built.FriendlyName,
                    StackTrace = built.StackTrace,
                    GraphOpenNodeKey = built.GraphOpenNodeKey is { } open && nodeMap.TryGetValue(open, out var remapped)
                        ? remapped
                        : built.GraphOpenNodeKey,
                    IdentifierTemplate = built.IdentifierTemplate,
                    TargetFunction = built.TargetFunction,
                };
                foreach (var group in built.FullInputs) copy.FullInputs[group.Key] = Remap(group.Value, tensorMap);
                foreach (var group in built.FullOutputs) copy.FullOutputs[group.Key] = Remap(group.Value, tensorMap);
                newNodes.Add(copy);
            }

            var terminal = plan.Body[^1];
            node.OpCode = terminal.OpCode;
            node.Attributes = terminal.Attributes;
            node.FullInputs = terminal.FullInputs.ToDictionary(g => g.Key, g => Remap(g.Value, tensorMap));
            newNodes.Add(node);
        }

        private static List<FastTensorKey?> Remap(
            List<FastTensorKey?> keys, Dictionary<FastTensorKey, FastTensorKey> tensorMap)
        {
            var remapped = new List<FastTensorKey?>(keys.Count);
            foreach (var key in keys)
                remapped.Add(key is { } k && tensorMap.TryGetValue(k, out var mapped) ? mapped : key);
            return remapped;
        }

        /// <summary>
        /// What a plan may be reused for: the lowering itself, the dtype and rank of each input
        /// (and which inputs are absent), how many outputs the operator declares, and the
        /// attribute VALUES. The attributes belong in the key because a lowering is ordinary C#
        /// and may branch on them, so the same operator with different attributes can legitimately
        /// decompose into different nodes. Null when some attribute has no stable rendering — a
        /// tensor or a subgraph — since a key that ignored it would serve one node's plan to
        /// another that differs only there.
        /// </summary>
        internal static string? TryBuildKey(
            OpLowering lowering,
            (DType DType, int? Rank)?[] inputs,
            OnnxCSharpAttributes attributes,
            int declaredOutputs)
        {
            // The slot and output counts are part of the key, so a cached plan's stand-in-per-slot
            // table always lines up with the node it is reused for.
            var key = new StringBuilder(lowering.OpCode)
                .Append(Sep).Append(lowering.Method.MethodHandle.Value)
                .Append(Sep).Append(inputs.Length)
                .Append(Sep).Append(declaredOutputs);

            foreach (var input in inputs)
            {
                key.Append(Group);
                if (input is not { } descriptor) { key.Append('~'); continue; }
                key.Append(descriptor.DType).Append(Sep).Append(descriptor.Rank ?? -1);
            }

            foreach (var (name, value) in attributes.GetAttributeVals().OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                key.Append(Group).Append(name).Append(Sep);
                if (!TryAppendValue(key, value)) return null;
            }
            return key.ToString();
        }

        private static bool TryAppendValue(StringBuilder key, object? value)
        {
            switch (value)
            {
                case null: key.Append('~'); return true;
                case bool b: key.Append(b ? 'T' : 'F'); return true;
                case long l: key.Append(l); return true;
                case int i: key.Append(i); return true;
                // By bits, so the rendering is exact and carries no culture or rounding of its own.
                case float f: key.Append(BitConverter.SingleToInt32Bits(f)); return true;
                case double d: key.Append(BitConverter.DoubleToInt64Bits(d)); return true;
                // Length-prefixed: a string is the one value that could otherwise hold a separator.
                case string s: key.Append(s.Length).Append('"').Append(s); return true;
                case DType t: key.Append(t); return true;
                case Enum e: key.Append(e.GetType().FullName).Append('.').Append(e); return true;
                case Array a:
                    key.Append('[');
                    foreach (var item in a)
                    {
                        if (!TryAppendValue(key, item)) return false;
                        key.Append(Sep);
                    }
                    key.Append(']');
                    return true;
                default: return false;
            }
        }
    }
}
