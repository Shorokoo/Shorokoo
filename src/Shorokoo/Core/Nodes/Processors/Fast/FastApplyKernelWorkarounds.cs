using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Interpreter.Helpers;
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
    /// input dtypes and ranks, attributes, outputs,
    /// <see cref="WorkaroundSite.ShapesAreConcrete"/>, and what it read of constants: the shape of one
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
        /// spliced in. <paramref name="shapesAreConcrete"/> is what every site outside a loop body
        /// reports as <see cref="WorkaroundSite.ShapesAreConcrete"/>; a site inside one reports
        /// false (<see cref="WorkaroundSite.IsInLoopBody"/>). <paramref name="isFunctionBody"/> says
        /// the graph is a function's body, whose inputs are values from wherever it is called
        /// (<see cref="WorkaroundSite.IsFromOutsideBody"/>). <paramref name="shapes"/> are what
        /// <see cref="ConcreteShapes"/> worked out of the graph, which
        /// <see cref="WorkaroundSite.ShapeOf"/> answers from.
        /// </summary>
        public static Splices Process(
            InternalComputationGraph graph, KernelWorkaroundSet? set, bool shapesAreConcrete = false, bool isFunctionBody = false,
            IReadOnlyDictionary<FastTensorKey, Shape>? shapes = null)
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
                var outsideBody = workaround.ReadsBodies ? OutsideBody(graph, producers, tensorInfo, isFunctionBody) : null;
                var read = ReadKeys(graph);
                var plans = new Dictionary<string, List<CachedPlan>>(StringComparer.Ordinal);
                var rewired = new Dictionary<FastTensorKey, FastTensorKey>();
                var newNodes = new List<FastNode>(graph.Nodes.Count);
                // The last node kept so far that is numbered in place, which a dropped call's
                // number is left unused after.
                FastNodeKey? numbered = null;
                // Per enclosing scope, whether it runs its body once per iteration or element.
                var scopes = new Stack<bool>();
                int loops = 0;

                foreach (var node in graph.Nodes)
                {
                    if (FastOpsetResolver.IsOpenOpCode(node.OpCode))
                    {
                        bool loop = node.OpCode is OpCodes.LOOP_OPEN or OpCodes.SEQUENCE_MAP_OPEN;
                        scopes.Push(loop);
                        if (loop) loops++;
                    }
                    else if (FastOpsetResolver.IsCloseOpCode(node.OpCode) && scopes.TryPop(out var closed) && closed)
                        loops--;

                    if (!workaround.OpCodes.Contains(node.OpCode)
                        || WorkaroundSite.TryCreate(node, tensorInfo, producers, read, shapesAreConcrete, inLoopBody: loops > 0, outsideBody: outsideBody?.Invoke(node), shapes: workaround.ReadsShapes ? shapes : null) is not { } site
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

        /// <summary>
        /// The dimensions of the values of <paramref name="graph"/> that follow from
        /// <paramref name="inputDims"/>, the dimensions of its inputs, positionally: what
        /// <see cref="WorkaroundSite.ShapeOf"/> answers. Shorokoo's interpreter works them out from
        /// the dimensions alone, leaving out every value whose shape it cannot tell without the
        /// data. Null where <paramref name="set"/> holds no workaround that asks
        /// (<see cref="KernelWorkaround.ReadsShapes"/>) of an operator the graph calls, or an
        /// input's dimensions are not stated.
        ///
        /// <para>Left out too is every value computed from a scope's result the interpreter
        /// estimated rather than worked out (<see cref="Estimated"/>): an <c>If</c> it could not
        /// tell the branch of, and a <c>Loop</c> it could not tell the end of. The interpreter
        /// gives each a shape all the same — the one branch's it knows, the shape a loop carries
        /// after the few iterations it walks — which a run can contradict.</para>
        ///
        /// <para>Read off the graph before the pre-passes rewrite it: they keep each value they do
        /// not replace under its key, and a value they add has no shape here.</para>
        /// </summary>
        public static IReadOnlyDictionary<FastTensorKey, Shape>? ConcreteShapes(
            InternalComputationGraph graph, KernelWorkaroundSet? set, IReadOnlyList<long[]?>? inputDims)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (set is null || inputDims is null || inputDims.Any(d => d is null)
                || !set.Workarounds.Any(w => w.ReadsShapes && graph.Nodes.Any(n => w.OpCodes.Contains(n.OpCode))))
                return null;

            var inputNodes = graph.InputNodes;
            if (inputNodes.Count != inputDims.Count) return null;
            var initial = new Dictionary<FastTensorKey, IRuntimeTensor>();
            for (int i = 0; i < inputNodes.Count; i++)
                if (inputNodes[i].OpCode == InternalOpCodes.MODEL_TENSOR_INPUT
                    && inputNodes[i].Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) is { IsGenericType: false } dtype)
                    initial[InternalComputationGraph.InputKeyOf(inputNodes[i])] = RuntimeTensorFactory.Create(dtype, new Shape(inputDims[i]!));

            Dictionary<FastTensorKey, IRuntimeTensor> values;
            try
            {
                values = new QuickExecutionEngine().Run(graph, initial);
            }
            // The shapes only spare the backend decisions it would otherwise make itself: a graph the
            // interpreter cannot walk leaves every one of them to it.
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                return null;
            }

            var estimated = Estimated(graph, values);
            var shapes = new Dictionary<FastTensorKey, Shape>(values.Count);
            foreach (var (key, value) in values)
                if (!estimated.Contains(key) && value is RuntimeTensor { Shape: { } shape } && shape.Dims.All(d => d >= 0))
                    shapes[key] = shape;
            return shapes;
        }

        /// <summary>
        /// The values of <paramref name="graph"/> whose shape in <paramref name="values"/> the
        /// interpreter estimated: each output of a scope it did not work out, and every value
        /// computed from one. An <c>If</c> is worked out where its condition's value is known
        /// and the branch that holds gives the output; a <c>Loop</c> where its trip count is
        /// known and each condition it has is too, so its walk ran to the loop's end. Any other
        /// scope is taken as estimated.
        /// </summary>
        private static HashSet<FastTensorKey> Estimated(
            InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> values)
        {
            var nodeByKey = new Dictionary<FastNodeKey, FastNode>(graph.Nodes.Count);
            foreach (var node in graph.Nodes) nodeByKey[node.Key] = node;
            var estimated = new HashSet<FastTensorKey>();

            bool Known(FastTensorKey? key, Func<RuntimeTensor, bool> hasValue)
                => key is { IsEmpty: false } k && values.TryGetValue(k, out var v) && v is RuntimeTensor t && hasValue(t);
            static bool IsBool(RuntimeTensor t) => t.BoolData is { Length: > 0 };
            static bool IsInt(RuntimeTensor t) => t.IntData is { Length: > 0 };
            bool AbsentOrBool(IReadOnlyList<FastTensorKey?> keys, int slot)
                => slot >= keys.Count || keys[slot] is not { IsEmpty: false } || Known(keys[slot], IsBool);

            foreach (var node in graph.Nodes)
            {
                var outputs = node.Outputs;
                if (node.Inputs.Any(k => k is { } key && estimated.Contains(key)))
                {
                    foreach (var key in outputs) if (key is { } k) estimated.Add(k);
                    continue;
                }
                if (!FastOpsetResolver.IsCloseOpCode(node.OpCode)) continue;

                var open = node.GraphOpenNodeKey is { } openKey && nodeByKey.TryGetValue(openKey, out var o) ? o.Inputs : null;
                if (node.OpCode == OpCodes.IF_CLOSE && open is [{ } condition, ..]
                    && values.TryGetValue(condition, out var c) && c is RuntimeTensor { BoolData: { Length: > 0 } taken })
                {
                    var branch = node.FullInputs.TryGetValue(taken[0] ? OnnxOpAttributeNames.AttrThenBranch : OnnxOpAttributeNames.AttrElseBranch, out var b) ? b : [];
                    for (int i = 0; i < outputs.Count; i++)
                        if (outputs[i] is { } key && (i >= branch.Count || branch[i] is not { } from
                            || estimated.Contains(from) || !values.ContainsKey(from)))
                            estimated.Add(key);
                    continue;
                }
                if (node.OpCode == OpCodes.LOOP_CLOSE && open is { Count: >= 2 }
                    && Known(open[0], IsInt) && AbsentOrBool(open, 1) && AbsentOrBool(node.Inputs, 0))
                    continue;
                foreach (var key in outputs) if (key is { } k) estimated.Add(k);
            }
            return estimated;
        }

        /// <summary>
        /// For a node of <paramref name="graph"/>, the test <see cref="WorkaroundSite.IsFromOutsideBody"/>
        /// answers of its inputs: whether a value comes from outside the innermost loop or branch
        /// body the node is in — its producer stands no later than the body's open node, which
        /// itself produces the body's inputs, or it has none. At a function body's top level, where
        /// the node is in no body of its own, the function's inputs are the values from outside: the
        /// runtime puts the body in place wherever the function is called, a loop body among them.
        /// Null for a node in no body of a graph that is not a function's.
        ///
        /// <para>Bodies are read off the graph as the model is emitted from it: a copy with its scopes
        /// configured (<see cref="InternalComputationGraph.ConfigureScopes"/>), which moves each
        /// branch's nodes inside its <c>If</c> — they are traced before the <c>IF_OPEN</c> — and
        /// every other node into the widest scope its data flow allows. The copy is made on the first
        /// question asked of a node in the walk.</para>
        ///
        /// <para>A value is traced back through what the runtime removes before it runs — an
        /// <c>Identity</c>, and a <c>Cast</c> to the type it already has — since the call then reads
        /// what those pass on.</para>
        /// </summary>
        private static Func<FastNode, Func<FastTensorKey, bool>?> OutsideBody(
            InternalComputationGraph graph, IReadOnlyDictionary<FastTensorKey, FastNode> producers,
            IReadOnlyDictionary<FastTensorKey, FastTensorInfo> tensorInfo, bool isFunctionBody)
        {
            var layout = new Lazy<(Dictionary<FastNodeKey, int> Position, Dictionary<FastNodeKey, int> Enclosing)>(() =>
            {
                IReadOnlyList<FastNode> nodes = graph.Nodes;
                if (nodes.Any(n => FastOpsetResolver.IsOpenOpCode(n.OpCode)))
                    try
                    {
                        var emitted = graph.Clone();
                        emitted.ConfigureScopes();
                        nodes = emitted.Nodes;
                    }
                    // A graph whose scopes cannot be configured yet is read as it stands.
                    catch (InvalidOperationException) { }

                var position = new Dictionary<FastNodeKey, int>(nodes.Count);
                var enclosing = new Dictionary<FastNodeKey, int>(nodes.Count);
                var open = new Stack<int>();
                for (int i = 0; i < nodes.Count; i++)
                {
                    var node = nodes[i];
                    if (FastOpsetResolver.IsCloseOpCode(node.OpCode) && open.Count > 0) open.Pop();
                    position[node.Key] = i;
                    enclosing[node.Key] = open.Count > 0 ? open.Peek() : -1;
                    if (FastOpsetResolver.IsOpenOpCode(node.OpCode)) open.Push(i);
                }
                return (position, enclosing);
            });

            FastNode? Source(FastTensorKey key)
            {
                for (int hops = 0; hops <= graph.Nodes.Count; hops++)
                {
                    if (!producers.TryGetValue(key, out var producer)) return null;
                    if (!PassesThrough(producer, tensorInfo) || producer.Inputs is not [{ } input, ..]) return producer;
                    key = input;
                }
                return null;
            }

            return node =>
            {
                var (position, enclosing) = layout.Value;
                if (enclosing.TryGetValue(node.Key, out var body) && body >= 0)
                    return key => Source(key) is not { } producer
                                  || !position.TryGetValue(producer.Key, out var at) || at <= body;
                if (isFunctionBody)
                    return key => Source(key) is not { } producer || InternalOpCodes.IsModelInputOp(producer.OpCode);
                return null;
            };
        }

        /// <summary>Whether the runtime removes <paramref name="node"/> before it runs, so what reads
        /// its output reads its input: an <c>Identity</c>, or a <c>Cast</c> to the type its input
        /// already has.</summary>
        private static bool PassesThrough(FastNode node, IReadOnlyDictionary<FastTensorKey, FastTensorInfo> tensorInfo)
            => node.OpCode == OpCodes.IDENTITY
            || node.OpCode == OpCodes.CAST && node.Inputs is [{ } input, ..] && node.Outputs is [{ } output, ..]
               && tensorInfo.TryGetValue(input, out var from) && tensorInfo.TryGetValue(output, out var to)
               && from.DType != DType.Invalid && from.DType.IsSameElementTypeAs(to.DType);

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
