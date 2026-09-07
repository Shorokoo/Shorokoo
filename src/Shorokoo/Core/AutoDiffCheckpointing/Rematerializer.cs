using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Identifies tensors that should be recomputed rather than stored in memory, then
/// transforms the graph by inserting recomputation nodes for those tensors.
///
/// <para>Recomputation only pays when it recomputes a <b>chain</b> back to tensors that are
/// alive at the recompute point anyway. Cloning a single producer and letting it read its
/// original inputs forces those inputs to stay alive to the recompute point, and for a MatMul
/// or any elementwise op the inputs are at least as large as the output: the lifetime extended
/// cancels the lifetime shortened. So the walk back from a target stops at a tensor that is
/// live across the recompute point for its own reasons, or that is a graph input, parameter or
/// constant, and everything strictly inside the walk is recomputed. A transformer's attention
/// is the archetype: the softmax output and the scores it came from both span the forward-to-
/// backward gap, and both recompute from Q and K, which the gradient needs anyway.</para>
///
/// <para>A target is recomputed <b>once</b>, and every backward consumer of it reads that one
/// clone (Checkmate and Rockmate recompute a block once per backward stage; the earlier
/// per-consumer clones here rebuilt the same forward sub-chain <i>k</i> times for <i>k</i>
/// consumers). Two placements of the clone in the linear order and a split of the chain at an
/// intermediate tensor — kept as a checkpoint in Rockmate's sense, so only the tail is
/// recomputed — are each evaluated on the candidate graph, and the pass keeps the measured
/// best; the estimate only ranks and pre-filters.</para>
///
/// <para>Targets are found in the <b>peak region</b> — every point of the schedule within
/// <see cref="PeakRegionFraction"/> of the peak — and ranked by bytes times the number of peak
/// positions they are live across, the way XLA's rematerializer scores candidates by memory
/// reduced at the points over the limit. A committed chain moves the peak; the region is then
/// re-read from the new evaluation and the ranking rebuilt, with targets already tried in this
/// call skipped, so the search converges instead of re-proposing what it just rejected. The
/// budget of full-graph evaluations per call is explicit (<see cref="MaxEvaluationsPerCall"/>)
/// and what the instance has spent across its calls is readable (<see cref="EvaluationsUsed"/>),
/// as is whether the budget rather than the candidate list is what stopped it
/// (<see cref="BudgetBound"/>).</para>
///
/// <para>Two invariants hold for every commit: the candidate graph scores strictly better
/// under the objective and its evaluated peak does not exceed the peak before the commit; and
/// the shape info returned covers every clone, so nothing is ever priced at zero.</para>
///
/// <para>A user's <c>[Module(Checkpoint = true)]</c> is a different contract, applied by
/// <see cref="ApplyCheckpointSegments"/>: recompute every interior tensor of the segment that
/// the backward pass reads, from the segment's boundary, once, regardless of the objective.</para>
/// </summary>
internal class Rematerializer
{
    private readonly GraphEvaluator _evaluator;
    private readonly ComputeMemoryObjective _objective;

    public Rematerializer(
        ComputeMemoryObjective objective,
        int maxIterations = MemoryAwareGraphOptimizer.DefaultRematerializationIterations,
        GraphEvaluator? evaluator = null)
    {
        _evaluator = evaluator ?? new GraphEvaluator();
        _objective = objective;
        _ = maxIterations;
    }

    /// <summary>
    /// Computes the combined metric for a given evaluation result.
    /// </summary>
    public double ComputeCombinedMetric(GraphEvaluationResult eval) => _objective.Score(eval);

    /// <summary>Full-graph evaluations spent by every <see cref="Apply"/> on this instance.</summary>
    internal int EvaluationsUsed { get; private set; }

    /// <summary>
    /// Whether any call ran out of evaluations with candidates still ranked ahead of it, rather
    /// than exhausting the list. The two look identical in the result otherwise, and on the
    /// larger models the budget is what stops the search: a two-layer encoder commits nothing
    /// within it and eleven candidates beyond it.
    /// </summary>
    internal bool BudgetBound { get; private set; }

    /// <summary>One entry per target committed by <see cref="Apply"/> on this instance.</summary>
    internal List<CommitRecord> CommitLog { get; } = new();

    /// <summary>
    /// A committed recomputation: one clone of a <paramref name="ChainLength"/>-node chain
    /// rebuilding <paramref name="Target"/>, read by <paramref name="RewiredConsumers"/>
    /// consumers, and the evaluated peak either side of the commit.
    /// </summary>
    internal sealed record CommitRecord(
        FastTensorKey Target, int ChainLength, int RewiredConsumers, Placement Placement, Variant Variant,
        long PeakBefore, long PeakAfter);

    /// <summary>Where a chain's clones go in the linear order.</summary>
    internal enum Placement
    {
        /// <summary>Right after the target's original producer.</summary>
        AfterProducer,
        /// <summary>Right before the first consumer rewired to read the clone.</summary>
        BeforeFirstConsumer,
    }

    /// <summary>Which recomputation of a target a candidate is.</summary>
    internal enum Variant
    {
        /// <summary>One clone at the earliest backward consumer, shared by every later one.</summary>
        Shared,
        /// <summary>One clone for one backward consumer, dying right after it; a target with
        /// several consumers yields one such candidate per consumer.</summary>
        PerConsumer,
        /// <summary>One clone at the latest consumer only; earlier readers keep the original.
        /// (A one-reader suffix of <see cref="Shared"/>, kept for the record's readability.)</summary>
        LastOnly,
        /// <summary>Shared, with the chain cut at an intermediate tensor that is kept instead.</summary>
        Split,
    }

    /// <summary>
    /// Applies rematerialization to the graph, returning the rewritten graph together with
    /// shape information that covers it. Returns the very same graph instance when nothing
    /// was committed.
    ///
    /// <para>Returning the shape info is not a convenience. Every recomputed tensor gets a
    /// fresh key, absent from the caller's original <paramref name="shapeInfo"/>, and
    /// <see cref="GraphEvaluator"/> skips an output it has no shape for while the per-op
    /// estimators cost a null output shape at zero. Evaluating the rewritten graph against
    /// the ORIGINAL shape info therefore prices every clone this pass just inserted at zero
    /// bytes and zero time — which makes rematerialization look free and lets the caller
    /// accept transforms that are strictly worse on both axes.</para>
    /// </summary>
    public (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) Apply(
        InternalComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        var current = new State(graph, shapeInfo, _evaluator.Evaluate(graph, shapeInfo));
        var currentScore = _objective.Score(current.Eval);
        var tried = new HashSet<(FastTensorKey, bool, FastNodeKey, int)>();
        var evaluations = 0;

        State? Trial(List<RematCandidate> batch, Placement placement)
        {
            if (evaluations >= MaxEvaluationsPerCall) return null;
            var (candidateGraph, mapping) = ApplyCandidates(current.Graph, batch, placement);
            if (mapping.Count == 0) return null;
            var candidateShapeInfo = AugmentShapeInfo(current.ShapeInfo, mapping);
            var eval = _evaluator.Evaluate(candidateGraph, candidateShapeInfo);
            evaluations++;
            return new State(candidateGraph, candidateShapeInfo, eval);
        }

        double Score(State s) => _objective.Score(s.Eval);

        // Commit-worthy: a strictly better score AND no higher peak (the objective alone would
        // accept a higher peak paid for by compute, which is never what recomputation is for).
        bool Better(State? s)
            => s is not null && Score(s) < currentScore && s.Eval.PeakMemoryBytes <= current.Eval.PeakMemoryBytes;

        void Commit(State next, List<RematCandidate> batch, Placement placement)
        {
            foreach (var c in batch)
                CommitLog.Add(new CommitRecord(c.Rewires[0].Target, c.Chain.Count,
                    c.Rewires.Sum(r => r.Consumers.Count), placement, c.Variant,
                    current.Eval.PeakMemoryBytes, next.Eval.PeakMemoryBytes));
            current = next;
            currentScore = Score(next);
        }

        // Peak memory is a MAXIMUM over the schedule: freeing one tensor lowers one node and
        // exposes the next-highest, so single commits can stall on a plateau only a group
        // crosses. Each round therefore first looks for the best ranked PREFIX of the
        // candidates — halving the size each time, so a group that carries most of the drop
        // at a fraction of the compute is found in log(n) evaluations — and only when no
        // prefix improves does it walk the ranked list one candidate at a time, one
        // evaluation each, committing the first that measures better. Every variant of a
        // target (shared, per-consumer, latest-only, split) is its own candidate, so the
        // choice among them is made by the same measurement.
        var batchPhase = true;
        while (evaluations < MaxEvaluationsPerCall)
        {
            var liveness = Liveness.Build(current.Graph, current.Eval, current.ShapeInfo);
            var candidates = FindCandidates(liveness, current.ShapeInfo, tried);
            if (candidates.Count == 0)
                break;

            if (batchPhase)
            {
                // Variants of one target overlap in the readers they rewire; a group keeps
                // only the best-ranked one for each reader, so no reader is rewired twice.
                var batchable = Disjoint(candidates);
                State? best = null;
                var bestSize = 0;
                for (int size = batchable.Count; size >= 1; size /= 2)
                {
                    var trial = Trial(batchable.Take(size).ToList(), Placement.AfterProducer);
                    if (Better(trial) && (best is null || Score(trial!) < Score(best)))
                        (best, bestSize) = (trial, size);
                    if (size == 1) break;
                }

                if (best is not null)
                {
                    var prefix = batchable.Take(bestSize).ToList();
                    var placement = Placement.AfterProducer;
                    if (prefix.All(c => c.CanPlaceBeforeConsumer))
                    {
                        var high = Trial(prefix, Placement.BeforeFirstConsumer);
                        if (Better(high) && Score(high!) < Score(best))
                            (best, placement) = (high, Placement.BeforeFirstConsumer);
                    }
                    Commit(best!, prefix, placement);
                    foreach (var c in prefix) tried.Add(c.Identity);
                    continue;
                }
                batchPhase = false;
            }

            // Every ranked single is worth an evaluation: the ranking is by ESTIMATED relief,
            // and the estimate is wrong often enough that the committed candidate can sit well
            // down the list. An earlier version stopped after twelve rejections in a row to save
            // the evaluations a family like encoder2 spends without committing; that stop also
            // walked past the one candidate the one-layer encoder had, turning the pass off on
            // it entirely. The budget is the only stop.
            var accepted = false;
            foreach (var candidate in candidates)
            {
                if (evaluations >= MaxEvaluationsPerCall) break;
                if (!tried.Add(candidate.Identity)) continue;
                var trial = Trial([candidate], Placement.AfterProducer);
                if (!Better(trial)) continue;
                Commit(trial!, [candidate], Placement.AfterProducer);
                accepted = true;
                break;
            }

            if (!accepted)
                break;
        }

        EvaluationsUsed += evaluations;
        BudgetBound |= evaluations >= MaxEvaluationsPerCall;
        return (current.Graph, current.ShapeInfo);
    }

    /// <summary>
    /// Honours every <c>[Module(Checkpoint = true)]</c> segment in <paramref name="graph"/>:
    /// each interior tensor of a segment that a node outside the segment reads is recomputed
    /// from the segment's boundary inputs, once per segment, and every such reader is rewired
    /// to the recomputed copy, so the originals die at their last use inside the segment.
    /// The segment's own outputs are its boundary and stay. Unconditional — this is what the
    /// user asked for — except for the choice between the two clone placements, which is
    /// made on the evaluated peak. A graph without segments is returned as the same instance;
    /// re-applying is a no-op, since the rewired readers no longer read interior tensors.
    /// </summary>
    internal static (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) ApplyCheckpointSegments(
        InternalComputationGraph graph, ShapeInferenceResult shapeInfo, GraphEvaluator evaluator)
    {
        var members = new Dictionary<long, List<FastNode>>();
        foreach (var node in graph.Nodes)
            if (CheckpointSegment.IdOf(node) is long id)
                (members.TryGetValue(id, out var list) ? list : members[id] = new List<FastNode>()).Add(node);
        if (members.Count == 0)
            return (graph, shapeInfo);

        var liveness = Liveness.Build(graph, evaluator.Evaluate(graph, shapeInfo), shapeInfo);
        var candidates = new List<RematCandidate>();
        foreach (var (_, nodes) in members)
        {
            var candidate = BuildSegmentCandidate(nodes, liveness, shapeInfo);
            if (candidate is not null) candidates.Add(candidate);
        }
        if (candidates.Count == 0)
            return (graph, shapeInfo);

        (InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo, GraphEvaluationResult Eval)? best = null;
        foreach (var placement in (Placement[])[Placement.AfterProducer, Placement.BeforeFirstConsumer])
        {
            if (placement == Placement.BeforeFirstConsumer && !candidates.All(c => c.CanPlaceBeforeConsumer)) continue;
            var (candidateGraph, mapping) = ApplyCandidates(graph, candidates, placement);
            if (mapping.Count == 0) continue;
            var candidateShapeInfo = AugmentShapeInfo(shapeInfo, mapping);
            var eval = evaluator.Evaluate(candidateGraph, candidateShapeInfo);
            if (best is null || eval.PeakMemoryBytes < best.Value.Eval.PeakMemoryBytes
                || (eval.PeakMemoryBytes == best.Value.Eval.PeakMemoryBytes && eval.TotalComputeTime < best.Value.Eval.TotalComputeTime))
                best = (candidateGraph, candidateShapeInfo, eval);
        }

        return best is null ? (graph, shapeInfo) : (best.Value.Graph, best.Value.ShapeInfo);
    }

    private sealed record State(InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo, GraphEvaluationResult Eval);

    private static ShapeInferenceResult AugmentShapeInfo(
        ShapeInferenceResult baseInfo,
        Dictionary<FastTensorKey, FastTensorKey> newToOriginalMapping)
    {
        if (newToOriginalMapping.Count == 0)
            return baseInfo;

        var builder = baseInfo.TensorInfos.ToBuilder();
        foreach (var (newKey, originalKey) in newToOriginalMapping)
        {
            var originalInfo = baseInfo.GetTensorInfo(originalKey);
            if (originalInfo is not null)
                builder[newKey] = originalInfo;
        }

        return new ShapeInferenceResult(builder.ToImmutable());
    }

    /// <summary>Longest producer chain the objective-driven walk will recompute. A chain that
    /// reaches this deep is either not anchored to live tensors or is duplicating too much work
    /// to be worth it, and every extra link multiplies both the clone cost and the search. A
    /// user's checkpoint segment is not subject to it.</summary>
    private const int MaxChainNodes = 12;

    /// <summary>
    /// Ceiling on full-graph evaluations spent per <see cref="Apply"/>. Each trial costs one
    /// evaluation, committed or not, so this bounds the pass's cost to something proportional
    /// to graph size rather than to how many candidates happen to look profitable — which on a
    /// transformer is hundreds.
    /// </summary>
    internal const int MaxEvaluationsPerCall = 48;

    /// <summary>A schedule position is in the peak region when its live bytes are at least
    /// this fraction of the peak. Freeing memory at one position exposes the next-highest,
    /// so targets are ranked over the whole region rather than the single peak node.</summary>
    private const double PeakRegionFraction = 0.95;

    /// <summary>
    /// The liveness picture of one evaluated graph, indexed by position in the evaluation's
    /// walk (<see cref="NodeEvaluationInfo.NodeIndex"/> maps a position to its node).
    /// Positions are for analysis only; every rewrite is keyed by node identity.
    /// </summary>
    /// <summary>
    /// The peak <see cref="Liveness"/> reads out of an evaluation. It must equal the evaluator's
    /// own peak: the liveness profile is what the candidate search targets, and a profile that
    /// disagrees puts the peak region on the wrong nodes.
    /// </summary>
    internal static long LivenessPeakFor(InternalComputationGraph graph, GraphEvaluationResult eval, ShapeInferenceResult shapeInfo)
        => Liveness.Build(graph, eval, shapeInfo).Peak;

    private sealed class Liveness
    {
        public required IList<FastNode> Nodes { get; init; }
        public required Dictionary<FastTensorKey, int> ProducerPos { get; init; }
        public required Dictionary<FastTensorKey, FastNode> Producer { get; init; }
        public required Dictionary<FastTensorKey, int> LastUse { get; init; }
        public required Dictionary<FastTensorKey, List<(int Pos, FastNode Node)>> Consumers { get; init; }
        public required HashSet<FastTensorKey> Rooted { get; init; }
        public required Dictionary<FastNodeKey, double> NodeCost { get; init; }
        public required Dictionary<FastNodeKey, int> PosOf { get; init; }
        /// <summary>Node key to index in the graph's linear order (where clones are inserted).</summary>
        public required Dictionary<FastNodeKey, int> LinearIndexOf { get; init; }
        public required long[] MemoryAt { get; init; }
        /// <summary>Bytes live after each position, without the op's own workspace.</summary>
        public required long[] LiveAt { get; init; }
        public required int[] ScopeDepth { get; init; }
        public required long Peak { get; init; }

        public static Liveness Build(InternalComputationGraph graph, GraphEvaluationResult eval, ShapeInferenceResult shapeInfo)
        {
            // Scopes are contiguous in the linear order, so depth is read there and carried
            // over to the walk; everything else is positioned in the evaluation's walk — ORT's
            // execution order — so that "live across the peak" and "free to keep at the
            // recompute point" mean what they will mean at runtime.
            var linear = graph.Nodes;
            var linearDepth = new int[linear.Count];
            var linearIndexOf = new Dictionary<FastNodeKey, int>(linear.Count);
            for (int i = 0, d = 0; i < linear.Count; i++)
            {
                linearIndexOf[linear[i].Key] = i;
                if (linear[i].IsCloseNode()) d--;
                linearDepth[i] = d;
                if (linear[i].IsOpenNode()) d++;
            }

            var details = eval.NodeDetails;
            var walked = details.Count == linear.Count;
            var nodes = walked ? details.Select(d => linear[d.NodeIndex]).ToList() : linear;
            var producerPos = new Dictionary<FastTensorKey, int>();
            var producer = new Dictionary<FastTensorKey, FastNode>();
            var lastUse = new Dictionary<FastTensorKey, int>();
            var consumers = new Dictionary<FastTensorKey, List<(int, FastNode)>>();
            var rooted = new HashSet<FastTensorKey>();
            var nodeCost = new Dictionary<FastNodeKey, double>();
            var posOf = new Dictionary<FastNodeKey, int>();
            var memoryAt = new long[nodes.Count];
            var liveAt = new long[nodes.Count];
            var depth = new int[nodes.Count];
            long peak = 0;

            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                posOf[node.Key] = i;
                nodeCost[node.Key] = i < details.Count ? details[i].ComputeTime : 0;
                // The bytes an op needs while it runs — its workspace included, which is what
                // makes a convolution the peak — not only what is live after it.
                liveAt[i] = i < details.Count ? details[i].CurrentMemoryBytes : 0;
                memoryAt[i] = i < details.Count ? liveAt[i] + details[i].ExtraMemoryBytes : 0;
                depth[i] = linearDepth[walked ? details[i].NodeIndex : i];

                foreach (var input in node.Inputs)
                {
                    if (input is null) continue;
                    lastUse[input.Value] = i;
                    (consumers.TryGetValue(input.Value, out var list) ? list : consumers[input.Value] = new List<(int, FastNode)>()).Add((i, node));
                }

                foreach (var output in node.Outputs)
                {
                    if (output is null) continue;
                    producerPos[output.Value] = i;
                    producer[output.Value] = node;
                    // A graph input, parameter or constant costs nothing to keep: it is resident
                    // for the whole step regardless, so a walk can anchor on it for free.
                    if (node.IsModelInput() || node.IsModelParamData() || node.Inputs.Count == 0
                        || NonDeterministicOps.Contains(node.OpCode))
                        rooted.Add(output.Value);
                }
            }

            // What an op needs while it runs is its occupancy plus its workspace. The
            // evaluator's occupancy already holds a buffer through the position that releases
            // it, so the inputs an op is the last reader of are in it; adding them again would
            // put the peak region at the wrong nodes.
            for (int i = 0; i < nodes.Count; i++)
                if (memoryAt[i] > peak) peak = memoryAt[i];

            return new Liveness
            {
                Nodes = nodes, ProducerPos = producerPos, Producer = producer, LastUse = lastUse,
                Consumers = consumers, Rooted = rooted, NodeCost = nodeCost, PosOf = posOf,
                LinearIndexOf = linearIndexOf, MemoryAt = memoryAt, LiveAt = liveAt, ScopeDepth = depth, Peak = peak,
            };
        }

        /// <summary>Live at position <paramref name="at"/> for its own reasons, so anchoring on
        /// it extends nothing.</summary>
        public bool FreeToKeep(FastTensorKey key, int at)
            => Rooted.Contains(key)
            || (LastUse.TryGetValue(key, out var last) && last >= at)
            || !ProducerPos.ContainsKey(key);

        /// <summary>Positions whose live bytes are within <see cref="PeakRegionFraction"/> of the peak.</summary>
        public List<int> PeakRegion()
        {
            var threshold = (long)(Peak * PeakRegionFraction);
            var region = new List<int>();
            for (int i = 0; i < MemoryAt.Length; i++)
                if (MemoryAt[i] >= threshold) region.Add(i);
            return region;
        }
    }

    /// <summary>
    /// One recomputation: a producer chain to clone once, and for each target tensor it
    /// rebuilds, the consumers to rewire onto the clone.
    /// </summary>
    private sealed class RematCandidate
    {
        /// <summary>Producers to clone, in topological order — deepest dependency first.</summary>
        public required List<FastNode> Chain { get; init; }
        public required List<(FastTensorKey Target, List<FastNode> Consumers)> Rewires { get; init; }
        /// <summary>Original node the clones follow under <see cref="Placement.AfterProducer"/>.</summary>
        public required FastNode Anchor { get; init; }
        /// <summary>Consumer the clones precede under <see cref="Placement.BeforeFirstConsumer"/>.</summary>
        public required FastNode FirstConsumer { get; init; }
        /// <summary>False when the first consumer sits inside a scope, where inserting the
        /// clones before it would put them in the scope while other readers are outside.</summary>
        public required bool CanPlaceBeforeConsumer { get; init; }
        public Variant Variant { get; init; }
        public double ExtraCompute { get; init; }
        /// <summary>Estimated objective change; negative is worth trying, lower ranks first.</summary>
        public double ExpectedDelta { get; init; }

        /// <summary>
        /// What makes two proposals the same trial, so one is never evaluated twice. It is the
        /// rewrite, not the variant that proposed it: the one-reader suffix of Shared and the
        /// PerConsumer form of that same reader build the identical graph, and a reader that
        /// takes the tensor in two input slots proposes itself twice.
        /// </summary>
        public (FastTensorKey Target, bool Split, FastNodeKey FirstConsumer, int Consumers) Identity
            => (Rewires[0].Target, Variant == Variant.Split, FirstConsumer.Key, Rewires[0].Consumers.Count);
    }

    /// <summary>The ranked candidates with any whose (target, reader) pairs overlap an
    /// earlier one dropped.</summary>
    private static List<RematCandidate> Disjoint(List<RematCandidate> ranked)
    {
        var taken = new HashSet<(FastTensorKey, FastNodeKey)>();
        var result = new List<RematCandidate>();
        foreach (var candidate in ranked)
        {
            var pairs = candidate.Rewires.SelectMany(r => r.Consumers.Select(c => (r.Target, c.Key))).ToList();
            if (pairs.Any(taken.Contains)) continue;
            foreach (var pair in pairs) taken.Add(pair);
            result.Add(candidate);
        }
        return result;
    }

    private List<RematCandidate> FindCandidates(
        Liveness live, ShapeInferenceResult shapeInfo,
        HashSet<(FastTensorKey, bool, FastNodeKey, int)> tried)
    {
        var region = live.PeakRegion();
        if (region.Count == 0) return new List<RematCandidate>();

        var candidates = new List<RematCandidate>();
        foreach (var (target, producerPos) in live.ProducerPos)
        {
            if (live.Rooted.Contains(target)) continue;
            if (!live.LastUse.TryGetValue(target, out var lastUse)) continue;
            if (!live.Consumers.TryGetValue(target, out var consumers)) continue;
            var bytes = shapeInfo.GetTensorInfo(target)?.MemoryBytes ?? 0;
            if (bytes <= 0) continue;
            if (!region.Any(r => producerPos < r && r <= lastUse)) continue;

            // One entry per reading NODE: Liveness.Consumers records one per input slot, and a
            // node taking the tensor in two slots is still one rewire.
            var ordered = consumers.GroupBy(c => c.Node.Key).Select(g => g.First()).OrderBy(c => c.Pos).ToList();

            void Propose(Variant variant, List<(int Pos, FastNode Node)> rewired, FastTensorKey? keep)
            {
                var first = rewired[0].Pos;
                var chain = BuildChain(target, first, live, shapeInfo, keep, MaxChainNodes);
                if (chain is null) return;
                if (keep is not null && chain.Value.Nodes.Count >= MaxChainNodes) return;

                // The original now dies at its last reader that was not rewired, and the clone
                // lives from the first rewired reader to the last: the relief is the share of
                // the region where neither is alive, weighted like XLA weights memory reduced
                // at the points over the limit.
                var keptUntil = consumers.Where(c => !rewired.Contains(c)).Select(c => c.Pos).DefaultIfEmpty(producerPos).Max();
                var cloneFrom = first;
                var cloneTo = rewired[^1].Pos;
                var relief = region.Count(r => r > keptUntil && !(cloneFrom <= r && r <= cloneTo));
                if (relief == 0) return;
                var keptBytes = keep is FastTensorKey k ? shapeInfo.GetTensorInfo(k)?.MemoryBytes ?? 0 : 0;
                var saved = (bytes * relief) / region.Count - keptBytes;

                // A chain frees bytes spread over a lifetime but pays for them in one burst,
                // rebuilding every intermediate at the recompute point. If that point is
                // already near the peak the burst simply moves the peak there, so require
                // headroom for it before proposing the chain at all.
                if (live.LiveAt[first] + chain.Value.Freed > live.Peak) return;

                // Admit on the optimistic estimate (the whole tensor off the peak) and rank on
                // the relief-weighted one: the evaluation, not the estimate, decides, and a
                // pre-filter that is too strict here costs more than a wasted evaluation.
                if (_objective.TradeDelta(chain.Value.Cost, bytes - keptBytes) >= 0) return;
                var delta = _objective.TradeDelta(chain.Value.Cost, saved);

                // Clones placed before a consumer go before the LINEARLY first rewired one,
                // which is not necessarily the first ORT runs: every rewired reader must
                // follow them in the linear order.
                var linearFirst = rewired.MinBy(c => live.LinearIndexOf[c.Node.Key]).Node;
                var candidate = new RematCandidate
                {
                    Chain = chain.Value.Nodes,
                    Rewires = [(target, rewired.Select(c => c.Node).ToList())],
                    Anchor = live.Producer[target],
                    FirstConsumer = linearFirst,
                    CanPlaceBeforeConsumer = live.ScopeDepth[live.PosOf[linearFirst.Key]] == 0,
                    Variant = variant,
                    ExtraCompute = chain.Value.Cost,
                    ExpectedDelta = delta,
                };
                if (!tried.Contains(candidate.Identity))
                    candidates.Add(candidate);
            }

            // Every suffix of the readers is a candidate: rewiring readers k.. onto one clone
            // frees the original after reader k-1. The full suffix is the recompute at the
            // earliest reader, the last one alone is the recompute at the latest, and each
            // single reader is the per-consumer form. The relief test discards a suffix that
            // starts before the peak (its clone would span it) and a single whose original
            // survives it.
            for (int k = 0; k < ordered.Count; k++)
                Propose(Variant.Shared, ordered.Skip(k).ToList(), null);

            if (ordered.Count > 1)
                foreach (var c in ordered)
                    Propose(Variant.PerConsumer, [c], null);
            foreach (var k in ((int[])[0, ordered.Count - 1]).Distinct())
                ProposeSplit(Variant.Split, ordered.Skip(k).ToList());

            // The Rockmate-style split of the shared chain: the interior tensor whose keeping
            // buys the most is treated as a checkpoint, kept alive to the recompute point, and
            // only the chain beyond it is recomputed.
            void ProposeSplit(Variant variant, List<(int Pos, FastNode Node)> rewired)
            {
                var full = BuildChain(target, rewired[0].Pos, live, shapeInfo, null, MaxChainNodes);
                if (full is null || full.Value.Nodes.Count < 3) return;
                FastTensorKey? bestKeep = null;
                double bestDelta = 0;
                foreach (var node in full.Value.Nodes.Take(full.Value.Nodes.Count - 1))
                {
                    var kept = node.Outputs.FirstOrDefault(o => o is not null);
                    if (kept is null) continue;
                    var part = BuildChain(target, rewired[0].Pos, live, shapeInfo, kept.Value, MaxChainNodes);
                    if (part is null || part.Value.Nodes.Count >= full.Value.Nodes.Count) continue;
                    var keptBytes = shapeInfo.GetTensorInfo(kept.Value)?.MemoryBytes ?? 0;
                    var delta = _objective.TradeDelta(part.Value.Cost, bytes - keptBytes);
                    if (delta < bestDelta) (bestKeep, bestDelta) = (kept.Value, delta);
                }
                if (bestKeep is FastTensorKey keep) Propose(variant, rewired, keep);
            }
        }

        candidates.Sort((a, b) => a.ExpectedDelta != b.ExpectedDelta
            ? a.ExpectedDelta.CompareTo(b.ExpectedDelta)
            : a.ExtraCompute.CompareTo(b.ExtraCompute));
        return candidates;
    }

    /// <summary>
    /// Walks back from <paramref name="target"/> collecting the producers needed to rebuild it
    /// at <paramref name="recomputeAt"/>, stopping at any tensor that is free to keep there (or
    /// at <paramref name="keep"/>). Null when the chain cannot be built: it runs past
    /// <paramref name="maxNodes"/>, bottoms out on something that cannot be cloned, enters a
    /// scope, or holds a tensor with no shape info (a clone priced at zero would be a fake
    /// gain).
    /// </summary>
    private static (List<FastNode> Nodes, long Freed, double Cost)? BuildChain(
        FastTensorKey target, int recomputeAt, Liveness live, ShapeInferenceResult shapeInfo,
        FastTensorKey? keep, int? maxNodes)
    {
        var order = new List<FastNode>();
        var visited = new HashSet<FastNodeKey>();
        long freed = 0;
        double cost = 0;
        var ok = true;

        void Walk(FastTensorKey key)
        {
            if (!ok) return;
            if (keep is FastTensorKey k && k.Equals(key)) return;
            if (live.FreeToKeep(key, recomputeAt)) return;
            if (!Visit(key)) ok = false;
        }

        bool Visit(FastTensorKey key)
        {
            if (!live.Producer.TryGetValue(key, out var producer)) return false;
            if (visited.Contains(producer.Key)) return true;
            if (!IsRecomputable(producer) || live.ScopeDepth[live.PosOf[producer.Key]] != 0) return false;
            if (shapeInfo.GetTensorInfo(key) is null) return false;
            if (maxNodes is int max && order.Count >= max) return false;
            visited.Add(producer.Key);

            foreach (var input in producer.Inputs)
                if (input is not null) Walk(input.Value);
            if (!ok) return false;

            order.Add(producer);
            cost += live.NodeCost.GetValueOrDefault(producer.Key, 0);
            freed += shapeInfo.GetTensorInfo(key)!.MemoryBytes;
            return true;
        }

        // The target itself always needs recomputing, even though it is live at recomputeAt
        // (that is the consumption being replaced), so start from its producer directly.
        if (!Visit(target) || !ok || order.Count == 0 || freed <= 0) return null;
        return (order, freed, cost);
    }

    /// <summary>
    /// The recomputation a checkpoint segment asks for: every interior tensor read outside
    /// the segment, rebuilt from the segment's boundary. The boundary is whatever the walk
    /// cannot or must not recompute — the segment's inputs, its declared outputs, and any
    /// interior tensor whose producer cannot be cloned — and it simply stays alive.
    /// </summary>
    private static RematCandidate? BuildSegmentCandidate(List<FastNode> members, Liveness live, ShapeInferenceResult shapeInfo)
    {
        var memberKeys = new HashSet<FastNodeKey>(members.Select(m => m.Key));
        var interior = new HashSet<FastTensorKey>();
        foreach (var node in members)
            if (!CheckpointSegment.ProducesSegmentOutput(node))
                foreach (var output in node.Outputs)
                    if (output is not null) interior.Add(output.Value);

        bool Recomputable(FastTensorKey key)
            => interior.Contains(key)
            && live.Producer.TryGetValue(key, out var p)
            && IsRecomputable(p)
            && live.ScopeDepth[live.PosOf[p.Key]] == 0
            && shapeInfo.GetTensorInfo(key) is not null;

        var order = new List<FastNode>();
        var visited = new HashSet<FastNodeKey>();
        void Walk(FastTensorKey key)
        {
            if (!Recomputable(key)) return;
            var producer = live.Producer[key];
            if (!visited.Add(producer.Key)) return;
            foreach (var input in producer.Inputs)
                if (input is not null) Walk(input.Value);
            order.Add(producer);
        }

        var rewires = new List<(FastTensorKey, List<FastNode>)>();
        FastNode? linearFirst = null;
        foreach (var key in interior.OrderBy(k => live.ProducerPos[k]))
        {
            if (!Recomputable(key) || !live.Consumers.TryGetValue(key, out var consumers)) continue;
            var outside = consumers.Where(c => !memberKeys.Contains(c.Node.Key)).ToList();
            if (outside.Count == 0) continue;
            Walk(key);
            rewires.Add((key, outside.Select(c => c.Node).ToList()));
            var first = outside.MinBy(c => live.LinearIndexOf[c.Node.Key]).Node;
            if (linearFirst is null || live.LinearIndexOf[first.Key] < live.LinearIndexOf[linearFirst.Key])
                linearFirst = first;
        }
        if (rewires.Count == 0) return null;

        // Clones go after the segment's last node in the LINEAR order: every chain original
        // then precedes them, whatever order the evaluation walked the segment in.
        var anchor = members.OrderBy(m => live.LinearIndexOf[m.Key]).Last();
        if (live.ScopeDepth[live.PosOf[anchor.Key]] != 0) return null;

        return new RematCandidate
        {
            Chain = order,
            Rewires = rewires,
            Anchor = anchor,
            FirstConsumer = linearFirst!,
            CanPlaceBeforeConsumer = live.ScopeDepth[live.PosOf[linearFirst!.Key]] == 0,
            Variant = Variant.Shared,
            ExtraCompute = order.Sum(n => live.NodeCost.GetValueOrDefault(n.Key, 0)),
        };
    }

    /// <summary>
    /// Clones each candidate's chain once with fresh keys, rewires its consumers to read the
    /// clone, and inserts the clones at the chosen placement. Returns the rewritten graph and
    /// a map from every minted key to the tensor it replicates, so the caller can extend shape
    /// information to cover them.
    /// </summary>
    private static (InternalComputationGraph Graph, Dictionary<FastTensorKey, FastTensorKey> Mapping) ApplyCandidates(
        InternalComputationGraph graph, List<RematCandidate> candidates, Placement placement)
    {
        var copy = graph.Clone();
        var newToOriginal = new Dictionary<FastTensorKey, FastTensorKey>();
        var insertBefore = new Dictionary<FastNodeKey, List<FastNode>>();
        var insertAfter = new Dictionary<FastNodeKey, List<FastNode>>();
        var rewires = new Dictionary<FastNodeKey, List<(FastTensorKey Target, FastTensorKey Recomputed)>>();

        foreach (var candidate in candidates)
        {
            // Fresh key per cloned producer; interior references are remapped so the chain
            // reads its own copies and only its frontier reads the original graph.
            var remap = new Dictionary<FastTensorKey, FastTensorKey>();
            var emitted = new List<FastNode>(candidate.Chain.Count);
            foreach (var producer in candidate.Chain)
            {
                var clone = CloneProducerForRecompute(producer);
                if (clone is null) { emitted.Clear(); break; }

                foreach (var slot in clone.FullInputs.Values)
                    for (int i = 0; i < slot.Count; i++)
                        if (slot[i] is FastTensorKey k && remap.TryGetValue(k, out var mapped))
                            slot[i] = mapped;

                foreach (var (slotName, slot) in producer.FullOutputs)
                {
                    var cloneSlot = clone.FullOutputs[slotName];
                    for (int i = 0; i < slot.Count && i < cloneSlot.Count; i++)
                        if (slot[i] is FastTensorKey orig && cloneSlot[i] is FastTensorKey fresh)
                        {
                            remap[orig] = fresh;
                            newToOriginal[fresh] = orig;
                        }
                }
                emitted.Add(clone);
            }
            if (emitted.Count == 0) continue;

            var slot2 = placement == Placement.AfterProducer ? insertAfter : insertBefore;
            var at = placement == Placement.AfterProducer ? candidate.Anchor.Key : candidate.FirstConsumer.Key;
            (slot2.TryGetValue(at, out var list) ? list : slot2[at] = new List<FastNode>()).AddRange(emitted);

            foreach (var (target, consumers) in candidate.Rewires)
            {
                if (!remap.TryGetValue(target, out var recomputed)) continue;
                foreach (var consumer in consumers)
                    (rewires.TryGetValue(consumer.Key, out var l) ? l : rewires[consumer.Key] = new List<(FastTensorKey, FastTensorKey)>())
                        .Add((target, recomputed));
            }
        }

        var newNodes = new List<FastNode>(copy.Nodes.Count + newToOriginal.Count);
        foreach (var node in copy.Nodes)
        {
            if (rewires.TryGetValue(node.Key, out var pending))
                foreach (var (target, recomputed) in pending)
                    foreach (var slot in node.FullInputs.Values)
                        for (int i = 0; i < slot.Count; i++)
                            if (slot[i] is FastTensorKey k && k.Equals(target))
                                slot[i] = recomputed;

            if (insertBefore.TryGetValue(node.Key, out var before)) newNodes.AddRange(before);
            newNodes.Add(node);
            if (insertAfter.TryGetValue(node.Key, out var after)) newNodes.AddRange(after);
        }

        copy.Nodes = newNodes;
        return (copy, newToOriginal);
    }

    /// <summary>
    /// Whether a second instance of <paramref name="producer"/> computes the same value: a
    /// single-output executable op that is not a scope, function or input, and not a draw. A
    /// clone of RandomUniformLike or Dropout is a second sample, so readers rewired to it see a
    /// value the un-rewired readers never saw. The keyed <c>shrk_Rng*</c> draws are deterministic
    /// in their key and counter and may be recomputed; the unkeyed <c>shrk_Random*</c> ones are
    /// lowered to the ONNX random ops and may not.
    /// </summary>
    internal static bool IsRecomputable(FastNode producer)
        => !(producer.IsOpenNode() || producer.IsCloseNode() || producer.IsFunction()
             || producer.IsModelInput() || producer.IsModelParamData())
        && IsDeterministicOpCode(producer.OpCode)
        && producer.FullOutputs.Values.Sum(s => s.Count(k => k is not null)) == 1;

    /// <summary>
    /// Whether a second instance of <paramref name="opCode"/> on the same inputs computes the
    /// same value. False for the draws, whose clone is a second sample. An op whose result on a
    /// tie is implementation-defined is fine: the clone runs the same kernel on the same inputs
    /// and breaks the tie the same way. The hazard is a value that depends on state other than
    /// the node's inputs.
    ///
    /// <para>Denial is by op code, which is coarser than it could be: a training step's dropout
    /// draw reaches the pass as a <b>keyed</b> <c>shrk_RandomUniform</c>, whose verbatim clone
    /// would be value-identical, and keyed and unkeyed draws share the op code. So this gives up
    /// a rematerialization at every dropout layer to stay safe on the unkeyed ones.</para>
    /// </summary>
    internal static bool IsDeterministicOpCode(string opCode) => !NonDeterministicOps.Contains(opCode);

    private static readonly HashSet<string> NonDeterministicOps =
    [
        OpCodes.RANDOM_UNIFORM, OpCodes.RANDOM_NORMAL, OpCodes.RANDOM_UNIFORM_LIKE, OpCodes.RANDOM_NORMAL_LIKE,
        OpCodes.BERNOULLI, OpCodes.MULTINOMIAL, OpCodes.DROPOUT,
        InternalOpCodes.SHRK_RANDOM_UNIFORM, InternalOpCodes.SHRK_RANDOM_NORMAL, InternalOpCodes.SHRK_RANDOM_BITS,
    ];

    /// <summary>
    /// Mirror of <see cref="SimpleBackpropOptimizer"/>'s clone logic: a fresh FastNode for the
    /// same op with new keys, preserving inputs verbatim. A clone carries no checkpoint stamp —
    /// it is the recomputation, not a member of the user's segment.
    /// </summary>
    private static FastNode? CloneProducerForRecompute(FastNode producer)
    {
        if (!IsRecomputable(producer)) return null;

        var freshKey = FastNodeKey.New();

        var newFullInputs = new Dictionary<string, List<FastTensorKey?>>();
        foreach (var (slotName, slot) in producer.FullInputs)
            newFullInputs[slotName] = new List<FastTensorKey?>(slot);

        var newFullOutputs = new Dictionary<string, List<FastTensorKey?>>();
        foreach (var (slotName, slot) in producer.FullOutputs)
        {
            var remapped = new List<FastTensorKey?>(slot.Count);
            foreach (var k in slot)
            {
                if (k is null) { remapped.Add(null); continue; }
                if (k.Value.IsEmpty) { remapped.Add(k); continue; }
                remapped.Add(new FastTensorKey(freshKey, k.Value.OutputIndex));
            }
            newFullOutputs[slotName] = remapped;
        }

        return new FastNode
        {
            Key = freshKey,
            OpCode = producer.OpCode,
            Attributes = CheckpointSegment.Strip(producer.Attributes),
            FullInputs = newFullInputs,
            FullOutputs = newFullOutputs,
            FriendlyName = producer.FriendlyName,
            StackTrace = producer.StackTrace,
        };
    }
}
