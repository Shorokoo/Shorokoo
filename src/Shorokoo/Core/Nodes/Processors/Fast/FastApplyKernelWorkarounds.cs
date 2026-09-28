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
    /// input dtypes and ranks, attributes, outputs, and the constants it read — and reused for
    /// the rest of the call; nothing outlives it.</para>
    /// </summary>
    internal static class FastApplyKernelWorkarounds
    {
        /// <summary>Whether <paramref name="graph"/> holds a call of an operator some workaround
        /// of <paramref name="set"/> looks at.</summary>
        public static bool HasCandidate(InternalComputationGraph graph, KernelWorkaroundSet? set)
        {
            if (set is null || set.IsEmpty) return false;
            foreach (var node in graph.Nodes)
                if (set.OpCodes.Contains(node.OpCode)) return true;
            return false;
        }

        /// <summary>
        /// Applies <paramref name="set"/> to <paramref name="graph"/> in place, and returns what it
        /// spliced in. <paramref name="shapesAreConcrete"/> is what every site reports as
        /// <see cref="WorkaroundSite.ShapesAreConcrete"/>.
        /// </summary>
        public static Splices Process(InternalComputationGraph graph, KernelWorkaroundSet? set, bool shapesAreConcrete = false)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (!HasCandidate(graph, set)) return Splices.None;
            var minted = new HashSet<FastNodeKey>();
            var hosts = new HashSet<FastNodeKey>();
            var gaps = new Dictionary<FastNodeKey, int>();
            int leadingGap = 0;

            Dictionary<FastTensorKey, FastTensorInfo>? tensorInfo = null;
            foreach (var workaround in set!.Workarounds)
            {
                if (!graph.Nodes.Any(n => workaround.OpCodes.Contains(n.OpCode))) continue;

                tensorInfo ??= FastTensorInfoProcessor.BuildTensorInfoLookup(graph);
                var producers = graph.BuildProducerByOutputMap();
                var read = ReadKeys(graph);
                var plans = new Dictionary<string, List<CachedPlan>>(StringComparer.Ordinal);
                var rewired = new Dictionary<FastTensorKey, FastTensorKey>();
                var newNodes = new List<FastNode>(graph.Nodes.Count);
                bool changed = false;
                // The last node kept so far that is numbered in place, which a dropped call's
                // number is left unused after.
                FastNodeKey? numbered = null;

                foreach (var node in graph.Nodes)
                {
                    if (!workaround.OpCodes.Contains(node.OpCode)
                        || WorkaroundSite.TryCreate(node, tensorInfo, producers, read, shapesAreConcrete) is not { } site
                        || !workaround.Applies(site))
                    {
                        newNodes.Add(node);
                        if (!minted.Contains(node.Key) && !InternalOpCodes.IsGraphOutputOp(node.OpCode)) numbered = node.Key;
                        continue;
                    }

                    var plan = PlanFor(workaround, site, plans);
                    changed = true;
                    if (plan.TerminalProducesOutputs)
                    {
                        FastSplice.SpliceInPlace(node, plan.Splice, newNodes, minted);
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
                    var outputs = FastSplice.SpliceBeside(node, plan.Splice, newNodes, minted);
                    for (int i = 0; i < site.OutputKeys.Length; i++)
                        if (plan.Slots[i] >= 0 && site.OutputKeys[i] is { } key)
                            rewired[key] = outputs[plan.Slots[i]];
                }

                if (rewired.Count > 0)
                    foreach (var node in newNodes)
                        foreach (var group in node.FullInputs.Values)
                            for (int i = 0; i < group.Count; i++)
                                if (group[i] is { } key && rewired.TryGetValue(key, out var to))
                                    group[i] = to;

                graph.Nodes = newNodes;
                if (changed) tensorInfo = null;

                Debug.Assert(graph.TryValidateLinearOrder(out var orderError),
                    "graph.IsLinearOrderValid(): " + orderError);
            }
            return minted.Count == 0 && hosts.Count == 0
                ? Splices.None
                : new Splices(minted, hosts, [.. graph.Nodes.Select(n => n.Key)], gaps, leadingGap);
        }

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

            /// <summary>Nothing spliced.</summary>
            public static Splices None { get; } = new([], [], [], new Dictionary<FastNodeKey, int>(), 0);

            internal Splices(HashSet<FastNodeKey> minted, HashSet<FastNodeKey> hosts, HashSet<FastNodeKey> present,
                IReadOnlyDictionary<FastNodeKey, int> gaps, int leadingGap)
            {
                this.minted = minted;
                this.hosts = hosts;
                this.present = present;
                this.gaps = gaps;
                this.leadingGap = leadingGap;
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

        /// <summary>A built replacement, and the constants the site it was built for read.</summary>
        private sealed record CachedPlan(IReadOnlyDictionary<int, string> ConstantsRead, WorkaroundPlan Plan);

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
            KernelWorkaround workaround, WorkaroundSite site, Dictionary<string, List<CachedPlan>> plans)
        {
            var key = FastSplice.TryBuildKey($"{workaround.Name}/{site.OpCode}", site.InputDescriptors, site.Attributes, site.OutputCount)
                ?.Append('\u0002').Append(site.OutputFingerprint()).ToString();

            if (key is not null && plans.TryGetValue(key, out var candidates))
                foreach (var candidate in candidates)
                    if (candidate.ConstantsRead.All(read => site.FingerprintOf(read.Key) == read.Value))
                        return candidate.Plan;

            var plan = Build(workaround, site);
            if (key is not null)
            {
                if (!plans.TryGetValue(key, out candidates)) plans[key] = candidates = [];
                candidates.Add(new CachedPlan(new Dictionary<int, string>(site.ConstantsRead), plan));
            }
            return plan;
        }

        private static WorkaroundPlan Build(KernelWorkaround workaround, WorkaroundSite site)
        {
            var standIns = FastSplice.StandIns(site.InputDescriptors);
            var outputs = workaround.Rewrite(site, standIns);
            if (outputs.Length != site.OutputCount)
                throw new InvalidOperationException(
                    $"Kernel workaround '{workaround.Name}' built {outputs.Length} output(s) for '{site.OpCode}', "
                    + $"which has {site.OutputCount}.");

            var slots = new int[outputs.Length];
            List<Variable> built = [];
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] is not { } value)
                {
                    if (site.IsOutputPresent(i))
                        throw new InvalidOperationException(
                            $"Kernel workaround '{workaround.Name}' built no value for output {i} of '{site.OpCode}'.");
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
                throw new InvalidOperationException(
                    $"Kernel workaround '{workaround.Name}' built no value for '{site.OpCode}'.");

            var plan = new WorkaroundPlan(FastSplice.Build(standIns, [.. built])!, slots);
            if (plan.TerminalProducesOutputs || built.Count != 1) return plan;

            built[0] = OnnxOp.Identity(built[0], null);
            return new WorkaroundPlan(FastSplice.Build(standIns, [.. built])!, slots);
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
