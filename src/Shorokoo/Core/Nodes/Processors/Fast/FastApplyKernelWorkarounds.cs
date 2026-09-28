using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Lowering.KernelWorkarounds;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Applies a backend's <see cref="KernelWorkaroundSet"/> to the graph a model for its session
    /// is built from: every call a workaround of the set <see cref="KernelWorkaround.Applies">applies
    /// to</see> is replaced by what the workaround <see cref="KernelWorkaround.Rewrite">builds</see>.
    ///
    /// <para><b>Where it runs.</b> <see cref="Factory.FastOnnxModelBuilder"/> runs it over its own
    /// copy of the graph, and only for the model a backend's session is built from — never for an
    /// exported file or the <c>.srk</c> dialect — so the graph, what is exported, generated C# and
    /// what is saved keep every operator as written. It runs over the main graph and every function
    /// body alike; a loop or branch body sits in its graph's node list, so its calls are rewritten
    /// in place, inside their scope.</para>
    ///
    /// <para><b>Order.</b> One walk per workaround, in the set's order. A walk never revisits what
    /// it spliced in; the next walk sees it like any other node.</para>
    ///
    /// <para><b>Splicing.</b> A replacement whose last node produces all its outputs takes the
    /// call's place: the call's node becomes that last node, keeping its key and output keys, so
    /// every consumer stays wired (a replacement ending in an <c>If</c> makes the call's node the
    /// <c>IF_CLOSE</c>, paired with the <c>IF_OPEN</c> spliced before it). A single-output
    /// replacement whose last node does not is ended with an <c>Identity</c> to the same effect.
    /// Otherwise — the outputs come from different nodes — the replacement is spliced beside the
    /// call, the call is dropped, and every reference to one of its outputs is rewired to the
    /// replacement's.</para>
    ///
    /// <para><b>Names.</b> What the pass did is returned as <see cref="Splices"/>, which says how
    /// <see cref="FastUseUniqueNames"/> numbers the graph: the spliced nodes, and the nodes the
    /// later pre-passes add only for them, last; and, where a call was dropped, the number it had
    /// left unused. Every other value then keeps the name it has in the model built without the
    /// workarounds.</para>
    ///
    /// <para><b>Plans.</b> A replacement is built once per distinct shape of call — workaround,
    /// input dtypes and ranks, attributes, outputs, and what it read of constants: the shape of one
    /// read through <see cref="WorkaroundSite.ConstantShapeOf"/>, the value of one read through
    /// <see cref="WorkaroundSite.ConstantOf"/> — and reused for the rest of the call; nothing
    /// outlives it.</para>
    ///
    /// <para><b>Types.</b> The tensor-info lookup the sites are read from is built once, when the
    /// first walk finds a call it looks at, and kept current as the walks splice: each plan carries
    /// the types of what it builds, and every splice adds them. The builder takes the lookup the
    /// pass ends with (<see cref="Splices.TensorInfo"/>) rather than building its own.</para>
    ///
    /// <para><b>Failures.</b> A workaround that throws, or builds what cannot stand in for the call,
    /// is a defect of the workaround; the build fails, naming the workaround and the call.</para>
    /// </summary>
    internal static class FastApplyKernelWorkarounds
    {
        /// <summary>
        /// Applies <paramref name="set"/> to <paramref name="graph"/> in place, and returns what it
        /// spliced in. <paramref name="shapesAreConcrete"/> is what every site reports as
        /// <see cref="WorkaroundSite.ShapesAreConcrete"/>.
        /// </summary>
        public static Splices Process(InternalComputationGraph graph, KernelWorkaroundSet? set, bool shapesAreConcrete = false)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (set is null || set.IsEmpty) return Splices.None;
            var minted = new HashSet<FastNodeKey>();
            var hosts = new HashSet<FastNodeKey>();
            var gaps = new Dictionary<FastNodeKey, int>();
            int leadingGap = 0;

            Dictionary<FastTensorKey, FastTensorInfo>? tensorInfo = null;
            foreach (var workaround in set.Workarounds)
            {
                if (!graph.Nodes.Any(n => workaround.OpCodes.Contains(n.OpCode))) continue;

                tensorInfo ??= FastTensorInfoProcessor.BuildTensorInfoLookup(graph);
                var producers = graph.BuildProducerByOutputMap();
                var read = ReadKeys(graph);
                var plans = new Dictionary<string, List<CachedPlan>>(StringComparer.Ordinal);
                var rewired = new Dictionary<FastTensorKey, FastTensorKey>();
                var newNodes = new List<FastNode>(graph.Nodes.Count);
                // The last node kept so far that is numbered in place, which a dropped call's
                // number is left unused after.
                FastNodeKey? numbered = null;

                foreach (var node in graph.Nodes)
                {
                    if (!workaround.OpCodes.Contains(node.OpCode)
                        || WorkaroundSite.TryCreate(node, tensorInfo, producers, read, shapesAreConcrete) is not { } site
                        || !Applies(workaround, site, node))
                    {
                        newNodes.Add(node);
                        if (!minted.Contains(node.Key) && !InternalOpCodes.IsGraphOutputOp(node.OpCode)) numbered = node.Key;
                        continue;
                    }

                    var plan = PlanFor(workaround, site, node, plans);
                    if (plan.TerminalProducesOutputs)
                    {
                        FastSplice.SpliceInPlace(node, plan.Splice, newNodes, minted, tensorInfo);
                        hosts.Add(node.Key);
                        if (!minted.Contains(node.Key)) numbered = node.Key;
                        continue;
                    }

                    if (!minted.Contains(node.Key))
                    {
                        var unused = 1 + (gaps.Remove(node.Key, out var own) ? own : 0);
                        if (numbered is { } before) gaps[before] = gaps.GetValueOrDefault(before) + unused;
                        else leadingGap += unused;
                    }
                    var outputs = FastSplice.SpliceBeside(node, plan.Splice, newNodes, minted, tensorInfo);
                    for (int i = 0; i < site.OutputKeys.Length; i++)
                        if (site.OutputKeys[i] is { } key)
                        {
                            tensorInfo.Remove(key);
                            if (plan.Slots[i] >= 0) rewired[key] = outputs[plan.Slots[i]];
                        }
                }

                if (rewired.Count > 0)
                    foreach (var node in newNodes)
                        foreach (var group in node.FullInputs.Values)
                            for (int i = 0; i < group.Count; i++)
                                if (group[i] is { } key && rewired.TryGetValue(key, out var to))
                                    group[i] = to;

                graph.Nodes = newNodes;

                Debug.Assert(graph.TryValidateLinearOrder(out var orderError),
                    "graph.IsLinearOrderValid(): " + orderError);
            }
            return tensorInfo is null
                ? Splices.None
                : new Splices(minted, hosts, [.. graph.Nodes.Select(n => n.Key)], gaps, leadingGap, tensorInfo);
        }

        private static bool Applies(KernelWorkaround workaround, WorkaroundSite site, FastNode node)
        {
            try { return workaround.Applies(site); }
            catch (Exception ex) { throw Failed(workaround, node, ex); }
        }

        /// <summary>The error a workaround that failed at <paramref name="node"/> fails the build
        /// with.</summary>
        private static InvalidOperationException Failed(KernelWorkaround workaround, FastNode node, Exception ex)
            => new($"Kernel workaround '{workaround.Name}' failed on the '{node.OpCode}' call "
                + $"'{node.FriendlyName ?? node.Key.ToString()}': {ex.Message}", ex);

        /// <summary>
        /// What one <see cref="Process"/> spliced into a graph: the nodes it minted, the nodes it
        /// turned into a replacement's last node, every node the graph held once it was done, and
        /// the numbers of the calls it dropped — how many after each node numbered in place, and
        /// how many before the first.
        /// </summary>
        internal sealed class Splices
        {
            private readonly HashSet<FastNodeKey> minted;
            private readonly HashSet<FastNodeKey> hosts;
            private readonly HashSet<FastNodeKey> present;
            private readonly IReadOnlyDictionary<FastNodeKey, int> gaps;
            private readonly int leadingGap;
            private readonly IReadOnlyDictionary<FastTensorKey, FastTensorInfo>? tensorInfo;

            /// <summary>Nothing spliced, and no call looked at.</summary>
            public static Splices None { get; } = new([], [], [], new Dictionary<FastNodeKey, int>(), 0, null);

            internal Splices(HashSet<FastNodeKey> minted, HashSet<FastNodeKey> hosts, HashSet<FastNodeKey> present,
                IReadOnlyDictionary<FastNodeKey, int> gaps, int leadingGap, IReadOnlyDictionary<FastTensorKey, FastTensorInfo>? tensorInfo)
            {
                this.minted = minted;
                this.hosts = hosts;
                this.present = present;
                this.gaps = gaps;
                this.leadingGap = leadingGap;
                this.tensorInfo = tensorInfo;
            }

            /// <summary>
            /// The tensor-info lookup of <paramref name="graph"/> as the later pre-passes leave it,
            /// taken from the one the pass kept; null when the pass built none, or the later passes
            /// added a value it cannot type. Those passes add only <c>Identity</c> nodes — around a
            /// value an <c>If</c> or <c>Loop</c> body hands out, say — and an identity's output is
            /// typed as its input.
            /// </summary>
            public Dictionary<FastTensorKey, FastTensorInfo>? TensorInfo(InternalComputationGraph graph)
            {
                if (tensorInfo is null) return null;
                var info = new Dictionary<FastTensorKey, FastTensorInfo>(tensorInfo);
                foreach (var node in graph.Nodes)
                    foreach (var group in node.FullOutputs.Values)
                        foreach (var output in group)
                        {
                            if (output is not { IsEmpty: false } key || info.ContainsKey(key)) continue;
                            if (node.OpCode != OpCodes.IDENTITY || node.Inputs is not [{ } source, ..]
                                || !info.TryGetValue(source, out var of))
                                return null;
                            info[key] = new FastTensorInfo { Key = key, DType = of.DType, Structure = of.Structure, Rank = of.Rank };
                        }
                return info;
            }

            /// <summary>
            /// How <see cref="FastUseUniqueNames"/> numbers <paramref name="graph"/>, as the later
            /// pre-passes leave it; null when nothing was spliced. Numbered last: every minted node
            /// still in it, and every node added since the workarounds whose output only a node of
            /// the replacement reads, however many such nodes lie in between — the identities that
            /// carry an outer-scope value into a spliced branch, say, or wrap a value leaving one.
            /// Left unused: the number of each call a replacement spliced beside it took the place
            /// of, where the call stood.
            /// </summary>
            public FastUseUniqueNames.Numbering? Numbering(InternalComputationGraph graph)
            {
                if (minted.Count == 0 && hosts.Count == 0) return null;

                var last = new HashSet<FastNodeKey>(graph.Nodes.Select(n => n.Key).Where(minted.Contains));
                var readers = new Dictionary<FastNodeKey, List<FastNodeKey>>();
                foreach (var node in graph.Nodes)
                    foreach (var group in node.FullInputs.Values)
                        foreach (var key in group)
                            if (key is { IsEmpty: false } k)
                            {
                                if (!readers.TryGetValue(k.FastNodeKey, out var list)) readers[k.FastNodeKey] = list = [];
                                list.Add(node.Key);
                            }

                bool grew = true;
                while (grew)
                {
                    grew = false;
                    foreach (var node in graph.Nodes)
                        if (!present.Contains(node.Key) && !last.Contains(node.Key)
                            && readers.TryGetValue(node.Key, out var list)
                            && list.All(r => last.Contains(r) || hosts.Contains(r)))
                        {
                            last.Add(node.Key);
                            grew = true;
                        }
                }
                return new(last, gaps, leadingGap);
            }
        }

        /// <summary>A built replacement, and what the site it was built for read of constants.</summary>
        private sealed record CachedPlan(IReadOnlyDictionary<int, WorkaroundSite.ConstantRead> ConstantsRead, WorkaroundPlan Plan);

        /// <summary>
        /// A splice plan and, for each output slot of the call, the index among the plan's outputs
        /// of the value that replaces it (-1 for a slot the replacement leaves absent).
        /// </summary>
        private sealed record WorkaroundPlan(FastSplice.Plan Splice, int[] Slots)
        {
            public bool TerminalProducesOutputs => Splice.TerminalProducesOutputs
                && Slots.Length == Splice.OutputKeys.Length
                && Slots.Select((s, i) => s == i).All(x => x);
        }

        private static WorkaroundPlan PlanFor(
            KernelWorkaround workaround, WorkaroundSite site, FastNode node, Dictionary<string, List<CachedPlan>> plans)
        {
            var key = FastSplice.TryBuildKey($"{workaround.Name}/{site.OpCode}", site.InputDescriptors, site.Attributes, site.OutputCount)
                ?.Append('\u0002').Append(site.OutputFingerprint()).ToString();

            if (key is not null && plans.TryGetValue(key, out var candidates))
                foreach (var candidate in candidates)
                    if (site.Reads(candidate.ConstantsRead))
                        return candidate.Plan;

            WorkaroundPlan plan;
            try { plan = Build(workaround, site); }
            catch (Exception ex) { throw Failed(workaround, node, ex); }
            if (key is not null)
            {
                if (!plans.TryGetValue(key, out candidates)) plans[key] = candidates = [];
                candidates.Add(new CachedPlan(new Dictionary<int, WorkaroundSite.ConstantRead>(site.ConstantsRead), plan));
            }
            return plan;
        }

        private static WorkaroundPlan Build(KernelWorkaround workaround, WorkaroundSite site)
        {
            var standIns = FastSplice.StandIns(site.InputDescriptors);
            var outputs = workaround.Rewrite(site, standIns);
            if (outputs.Length != site.OutputCount)
                throw new InvalidOperationException($"It built {outputs.Length} output(s) for a call with {site.OutputCount}.");

            var slots = new int[outputs.Length];
            List<Variable> built = [];
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] is not { } value)
                {
                    if (site.IsOutputPresent(i))
                        throw new InvalidOperationException($"It built no value for output {i}, which the call produces.");
                    slots[i] = -1;
                    continue;
                }
                // A value handed straight back from an input, or already given for another slot,
                // is taken through an Identity, so every output has a node of its own.
                if (standIns.Any(s => ReferenceEquals(s, value)) || built.Any(b => ReferenceEquals(b, value)))
                    value = OnnxOp.Identity(value, null);
                slots[i] = built.Count;
                built.Add(value);
            }

            if (built.Count == 0)
                throw new InvalidOperationException("It built no value at all.");

            var plan = new WorkaroundPlan(FastSplice.Build(standIns, [.. built], typed: true)!, slots);
            if (plan.TerminalProducesOutputs || built.Count != 1) return plan;

            built[0] = OnnxOp.Identity(built[0], null);
            return new WorkaroundPlan(FastSplice.Build(standIns, [.. built], typed: true)!, slots);
        }

        /// <summary>Every tensor key some node of <paramref name="graph"/> reads.</summary>
        private static HashSet<FastTensorKey> ReadKeys(InternalComputationGraph graph)
        {
            var read = new HashSet<FastTensorKey>();
            foreach (var node in graph.Nodes)
                foreach (var group in node.FullInputs.Values)
                    foreach (var key in group)
                        if (key is { IsEmpty: false } k) read.Add(k);
            return read;
        }
    }
}
