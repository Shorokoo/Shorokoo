using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The result of a memory-aware graph optimization pass: the optimized graph and
/// evaluation metrics for every strategy considered.
/// </summary>
public class GraphOptimizationResult
{
    /// <summary>
    /// The name of the strategy that was selected.
    /// </summary>
    public required string StrategyName { get; init; }

    /// <summary>
    /// What the backend the step runs on holds at the peak of a run of each strategy's graph, in
    /// <see cref="AllStrategies"/>' order, where that backend models a run and the pass chose by
    /// it; null where it chose by its own evaluation.
    /// </summary>
    internal IReadOnlyList<long>? BackendPeakBytes { get; init; }

    /// <summary>
    /// The optimized <see cref="InternalComputationGraph"/> produced by the selected strategy.
    /// </summary>
    /// <summary>The winning strategy's rewritten graph. Internal: the rig freezes it
    /// into the readonly <c>TrainingStepPureGraph</c>; exposing the same instance as
    /// mutable public state would invalidate that wrapper's kind stamp.</summary>
    internal InternalComputationGraph OptimizedGraph { get; init; } = null!;

    /// <summary>
    /// Shape information covering <see cref="OptimizedGraph"/>, including every tensor
    /// rematerialization minted; <see cref="Evaluation"/> was computed against it, or, where a
    /// training rig's pass rewrote its step with the step's shape arithmetic baked, against that
    /// rewrite's.
    /// </summary>
    internal ShapeInferenceResult ShapeInfo { get; init; } = null!;

    /// <summary>
    /// The evaluation result for the selected strategy.
    /// </summary>
    public required GraphEvaluationResult Evaluation { get; init; }

    /// <summary>
    /// All strategies that were evaluated, ordered by effectiveness.
    /// </summary>
    public required IReadOnlyList<(string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph)> AllStrategies { get; init; }

    public override string ToString()
        => $"Strategy={StrategyName}, Compute={Evaluation.TotalComputeTime:F2}, " +
           $"PeakMemory={Evaluation.PeakMemoryBytes / (1024.0 * 1024.0):F2} MB";
}

/// <summary>
/// Optimizes a <see cref="InternalComputationGraph"/> for the compute–memory tradeoff by
/// combining memory-aware scheduling (<see cref="MemoryAwareScheduler"/>) with
/// rematerialization (<see cref="Rematerializer"/>). The optimizer evaluates two
/// alternating strategies — <c>RematReorder</c> and <c>ReorderRemat</c> — and a third
/// committing the rematerializer's batches whole (<see cref="UnprunedRematReorder"/>), and selects
/// the one with the best combined metric
/// (see <see cref="ComputeMemoryObjective"/>: both terms as ratios to the baseline graph).
/// Where its evaluator carries a step's state written in place, both start from the graph with
/// each state update ordered after the state's readers (<see cref="OrderedStateReads"/>) when that
/// scores better than the graph as it came.
///
/// <para>
/// Not strictly "gradient checkpointing" in the narrow sense (that's what
/// <see cref="Rematerializer"/> alone implements); the umbrella optimizer is a more
/// general memory-aware graph rewriter that uses rematerialization as one tool.
/// </para>
/// </summary>
internal class MemoryAwareGraphOptimizer
{
    /// <summary>
    /// Default weight on the compute term of <see cref="ComputeMemoryObjective"/>.
    /// </summary>
    public const double DefaultComputeWeight = 1.0;

    /// <summary>
    /// Default weight on the memory term. Read against <see cref="DefaultComputeWeight"/>:
    /// at 1.0 a 1% peak-memory reduction is worth exactly a 1% compute increase.
    ///
    /// <para>Set from measurement, and set conservatively. The pass is automatic and has no
    /// opt-out, so it must not buy memory at a price the user did not ask to pay: at 4.0 it
    /// accepts a rematerialization that halves one graph's peak for TWICE its modelled
    /// compute, which is the right trade only for someone who would otherwise not fit at all.
    /// At 2.0 it takes the cheap trades — a free 1.7% on a transformer encoder, 16% for 15%
    /// more compute on chunked attention — and refuses that one. A knob for the aggressive
    /// setting is what issue #197 asks for and this does not provide.</para>
    ///
    /// <para>Note this is a weight on a RATIO, not on bytes. A constant multiplying raw byte
    /// counts would be meaningful only for graphs whose peak happened to be around a million
    /// times their compute-time figure — on everything else the memory term would vanish and
    /// the pass would silently degenerate into a compute-only optimizer. The normalization is
    /// what prevents that, so a byte-scaled constant does not belong here.</para>
    /// </summary>
    public const double DefaultMemoryWeight = 2.0;

    /// <summary>
    /// Graphs whose peak is below this are handed back untouched.
    ///
    /// <para>This pass buys memory, and it is not free: it clones and re-evaluates the whole
    /// graph several times. Below a megabyte there is nothing worth buying — the absolute
    /// saving is smaller than one tensor of a real model — so the work is pure overhead on
    /// exactly the small graphs where rig-construction latency is most visible. (Its
    /// modeled compute term does improve on small graphs, but measured steady-state training
    /// throughput does not move, so that figure is not a reason to spend the time.)</para>
    ///
    /// <para>A <c>[Module(Checkpoint = true)]</c> segment is not subject to it: the user asked,
    /// so it is applied (<see cref="Rematerializer.ApplyCheckpointSegments"/>) before the
    /// threshold is read.</para>
    /// </summary>
    public const long MinimumPeakBytesToOptimize = 1L << 20;

    /// <summary>
    /// The strategy that orders each updated state output's writer after every reader of the state
    /// it replaces (<see cref="StateReadOrdering"/>), and nothing more; the rematerializing and
    /// reordering strategies start from it where it is kept.
    /// </summary>
    public const string OrderedStateReads = "OrderedStateReads";

    /// <summary>
    /// The strategy alternating rematerialization and reordering as <c>RematReorder</c> does, with
    /// a rematerializer that commits each batch that improves whole, rather than pruned to the
    /// members that pay for themselves where the peak stands.
    ///
    /// <para>Pruning saves the compute of the members that ride along, but the graph it commits
    /// can set the search on a path that ends higher: on a two-layer transformer encoder's step
    /// charged as PyTorch's translation lays it out, the pruned search stops at 78% of the step's
    /// peak for 4% more compute, and the whole batches reach 58% for 20% more, which scores better
    /// by the objective both are judged by. Neither a larger evaluation budget nor retrying the
    /// pruned members later reaches it. Where pruning does as well, it scores better, at less
    /// compute, and is chosen.</para>
    /// </summary>
    public const string UnprunedRematReorder = "RematReorderUnpruned";

    private readonly GraphEvaluator _evaluator;
    private readonly ShapeInferenceInterpreter _shapeInference;
    private readonly double _computeFactor;
    private readonly double _memoryFactor;
    private readonly Func<InternalComputationGraph, long?>? _backendPeak;
    private readonly bool _weighPlateaus;

    /// <summary>
    /// A pass scoring every graph it considers — the baseline, each checkpoint segment, each
    /// rematerialization and each reordering — with <paramref name="evaluator"/>, the default
    /// evaluator when null. Give that evaluator the step's <see cref="StepState"/> and every
    /// candidate is scored with the state it will run with, written in place where the candidate
    /// still proves it. Where <paramref name="backendPeak"/> answers — what the backend the step
    /// runs on holds at the peak of a run of a graph (<see cref="Shorokoo.Core.Backends.IShorokooBackend"/>'s
    /// modelled run peak) — the search is judged by that peak in place of the evaluator's
    /// (<see cref="BackendJudge"/>): each strategy takes a step only where it scores better so, the
    /// rematerializer weighs the trials its own figures leave on a plateau by it, and the strategies
    /// are chosen among by it, so that the graph handed back holds no more on that backend than the
    /// one handed over, at the same objective. A plateau's trials are weighed by it only where
    /// <paramref name="weighPlateaus"/> says it is quick to ask.
    /// </summary>
    public MemoryAwareGraphOptimizer(
        double computeFactor = DefaultComputeWeight,
        double memoryFactor = DefaultMemoryWeight,
        GraphEvaluator? evaluator = null,
        ShapeInferenceInterpreter? shapeInference = null,
        Func<InternalComputationGraph, long?>? backendPeak = null,
        bool weighPlateaus = true)
    {
        _backendPeak = backendPeak;
        _weighPlateaus = weighPlateaus;
        _evaluator = evaluator ?? new GraphEvaluator();
        _shapeInference = shapeInference ?? new ShapeInferenceInterpreter();
        _computeFactor = computeFactor;
        _memoryFactor = memoryFactor;
    }

    /// <summary>
    /// The objective for a candidate evaluation, relative to the <paramref name="baseline"/>
    /// graph the optimization started from. Lower is better; the baseline itself scores
    /// <c>computeWeight + memoryWeight</c>.
    /// </summary>
    public double ComputeCombinedMetric(GraphEvaluationResult eval, GraphEvaluationResult baseline)
        => new ComputeMemoryObjective(_computeFactor, _memoryFactor, baseline).Score(eval);

    /// <summary>
    /// Finds the best optimization strategy for the given graph.
    /// </summary>
    public GraphOptimizationResult Optimize(InternalComputationGraph graph, params TensorData[] sampleInputs)
    {
        var shapeInfo = _shapeInference.Infer(graph, sampleInputs);
        return OptimizeWithShapeInfo(graph, shapeInfo);
    }

    /// <summary>
    /// Optimizes with pre-computed shape inference results.
    /// </summary>
    public GraphOptimizationResult OptimizeWithShapeInfo(InternalComputationGraph graph, ShapeInferenceResult shapeInfo)
    {
        // A user's [Module(Checkpoint = true)] is honoured first and unconditionally: it is
        // applied before the objective exists, so "doing nothing" below already includes it,
        // and before the size threshold, so a small graph gets it too.
        (graph, shapeInfo) = Rematerializer.ApplyCheckpointSegments(graph, shapeInfo, _evaluator);

        var baselineEval = _evaluator.Evaluate(graph, shapeInfo);

        // The one objective every candidate is finally judged by, normalized to the graph we
        // started from so the weights behave identically at every model size.
        var selection = new ComputeMemoryObjective(_computeFactor, _memoryFactor, baselineEval);
        var judge = _backendPeak is { } peakOf ? new BackendJudge(peakOf, _computeFactor, _memoryFactor, graph, baselineEval, _weighPlateaus) : null;

        // Doing nothing is always a candidate, so the pass can never return a graph that
        // scores worse than the one it was handed.
        var strategies = new List<(string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo)>
        {
            ("Baseline", baselineEval, graph, shapeInfo),
        };

        // A step that writes its state in place keeps a pair only where every reader of the state
        // runs before its update, whatever the backend folds. Ordering the updates after those
        // readers (StateReadOrdering) is a candidate of its own, in two forms -- as placed, and with
        // each update's search run before any other reaches its readers -- the better kept where it
        // scores better than the graph as it came, and then what every strategy below starts from.
        // It is weighed before the size threshold, so a small step keeps its pairs too: the trial
        // is two evaluations, and each pair is a buffer the size of the state it writes over.
        var start = new Candidate(graph, shapeInfo);
        var startEval = baselineEval;
        if (_evaluator.State is { WrittenInPlace: true } state)
        {
            foreach (var writersWithTheirReaders in (bool[])[false, true])
            {
                var (ordered, orderedShapeInfo) = StateReadOrdering.Apply(graph, shapeInfo, state.Pairs, writersWithTheirReaders);
                if (ReferenceEquals(ordered, graph)) break;
                var orderedEval = _evaluator.Evaluate(ordered, orderedShapeInfo);
                if (selection.Score(orderedEval) < selection.Score(startEval))
                    (start, startEval) = (new Candidate(ordered, orderedShapeInfo), orderedEval);
            }
            if (!ReferenceEquals(start.Graph, graph))
                strategies.Add((OrderedStateReads, startEval, start.Graph, start.ShapeInfo));
        }

        // Below the threshold there is nothing worth buying, nor worth asking the backend about.
        if (baselineEval.PeakMemoryBytes < MinimumPeakBytesToOptimize)
            return Chosen(strategies, strategies.Count - 1, judge: null);

        // The evaluator walks a Loop/If body once, but a backend runs a Loop body per iteration and
        // allocates per iteration; the pass would be optimizing a number that is not what runs.
        // Measured on the LSTM training step with no judge, letting it act raised ONNX Runtime's
        // real peak by a quarter while the evaluator claimed a small saving. So a graph with a scope
        // is handed back as it came, unless the backend's own model of a run answers for it -- one
        // that runs each loop's body per iteration, as torch's does -- and then every step the
        // search takes is judged by that model, and a step it cannot judge is not taken.
        var scoped = graph.Nodes.Any(n => n.IsOpenNode());
        if (scoped && judge?.Peak(graph) is null)
            return Chosen(strategies, strategies.Count - 1, judge);

        var scheduler = new MemoryAwareScheduler();
        var rematerializer = new Rematerializer(selection, _evaluator, judge: judge);

        // Each pass carries shape info forward alongside the graph it produced. Rematerialization
        // mints new tensor keys, and evaluating against shape info that predates them prices the
        // new tensors at nothing — so the pair must travel together.
        Candidate Remat(Candidate c)
        {
            var (g, si) = rematerializer.Apply(c.Graph, c.ShapeInfo);
            return new Candidate(g, si);
        }

        Candidate Reorder(Candidate c) => new(scheduler.Reorder(c.Graph, c.ShapeInfo), c.ShapeInfo);

        strategies.Add(RunAlternatingStrategy("RematReorder", selection, judge, scoped, start, startEval, Remat, Reorder));
        strategies.Add(RunAlternatingStrategy("ReorderRemat", selection, judge, scoped, start, startEval, Reorder, Remat));

        var unpruned = new Rematerializer(selection, _evaluator, pruneBatches: false, judge: judge);
        Candidate RematWhole(Candidate c)
        {
            var (g, si) = unpruned.Apply(c.Graph, c.ShapeInfo);
            return new Candidate(g, si);
        }
        strategies.Add(RunAlternatingStrategy(UnprunedRematReorder, selection, judge, scoped, start, startEval, RematWhole, Reorder));

        var best = Enumerable.Range(0, strategies.Count).MinBy(i => selection.Score(strategies[i].Evaluation));
        return Chosen(strategies, best, judge);
    }

    /// <summary>
    /// The result handing back <paramref name="strategies"/>' entry at <paramref name="chosen"/>,
    /// the pass's own choice — or, where there is more than one and the backend the step runs on
    /// models a run of every strategy's graph (<paramref name="judge"/>), the strategy scoring least
    /// by the same objective with that backend's peaks, its first entry (the graph as handed over)
    /// scoring the weights' sum: so the step handed back never holds more on that backend than the
    /// one handed over, unless it buys that with less compute at the objective's rate. Where the pass kept the
    /// graph with its state's updates ordered after their readers (<see cref="OrderedStateReads"/>),
    /// that graph stands for the one handed over unless the backend's model has it holding more:
    /// the ordering costs a few empty kernels and keeps state pairs written in place, which the run's
    /// peak shows only where it falls at the updates.
    /// </summary>
    private GraphOptimizationResult Chosen(
        List<(string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo)> strategies,
        int chosen, BackendJudge? judge)
    {
        List<long>? peaks = null;
        List<double>? scores = null;
        if (judge is not null && strategies.Count > 1)
        {
            (peaks, scores) = (new List<long>(strategies.Count), new List<double>(strategies.Count));
            foreach (var strategy in strategies)
            {
                if (judge.Judge(strategy.Graph, strategy.Evaluation) is not { } judged)
                {
                    (peaks, scores) = (null, null);
                    break;
                }
                peaks.Add(judged.Peak);
                scores.Add(judged.Score);
            }
        }
        if (peaks is not null && scores is not null)
        {
            var ordered = strategies.FindIndex(s => s.Name == OrderedStateReads);
            chosen = Enumerable.Range(0, strategies.Count).Where(i => i != 0 || ordered < 0 || peaks[0] < peaks[ordered]).MinBy(i => scores[i]);
        }
        var best = strategies[chosen];
        return new GraphOptimizationResult
        {
            StrategyName = best.Name,
            OptimizedGraph = best.Graph,
            ShapeInfo = best.ShapeInfo,
            Evaluation = best.Evaluation,
            AllStrategies = strategies.Select(s => (s.Name, s.Evaluation, s.Graph)).ToList(),
            BackendPeakBytes = peaks,
        };
    }

    /// <summary>A graph and the shape information that describes it. The two are only
    /// meaningful together: a pass that mints new tensor keys invalidates older info.</summary>
    private readonly record struct Candidate(InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo);

    /// <summary>
    /// A strategy: <paramref name="firstPass"/> and <paramref name="secondPass"/> applied in turn
    /// from <paramref name="initial"/>, each result taken where it scores better than the graph it
    /// was made from — by <paramref name="judge"/> where the backend's model answers for both, else
    /// by <paramref name="objective"/>, unless the graph holds a scope (<paramref name="judgedOnly"/>),
    /// whose steps only the backend's model may take — until a pass's result is not taken.
    /// </summary>
    private (string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph, ShapeInferenceResult ShapeInfo) RunAlternatingStrategy(
        string name,
        ComputeMemoryObjective objective,
        BackendJudge? judge,
        bool judgedOnly,
        Candidate initial,
        GraphEvaluationResult initialEval,
        Func<Candidate, Candidate> firstPass,
        Func<Candidate, Candidate> secondPass)
    {
        var current = initial;
        var currentEval = initialEval;
        var currentMetric = objective.Score(currentEval);

        TryApply(objective, judge, judgedOnly, firstPass, ref current, ref currentEval, ref currentMetric);
        TryApply(objective, judge, judgedOnly, secondPass, ref current, ref currentEval, ref currentMetric);

        while (true)
        {
            if (!TryApply(objective, judge, judgedOnly, firstPass, ref current, ref currentEval, ref currentMetric))
                break;
            if (!TryApply(objective, judge, judgedOnly, secondPass, ref current, ref currentEval, ref currentMetric))
                break;
        }

        return (name, currentEval, current.Graph, current.ShapeInfo);
    }

    private bool TryApply(
        ComputeMemoryObjective objective,
        BackendJudge? judge,
        bool judgedOnly,
        Func<Candidate, Candidate> pass,
        ref Candidate current,
        ref GraphEvaluationResult currentEval,
        ref double currentMetric)
    {
        var candidate = pass(current);
        if (ReferenceEquals(candidate.Graph, current.Graph))
            return false;

        var candidateEval = _evaluator.Evaluate(candidate.Graph, candidate.ShapeInfo);
        var candidateMetric = objective.Score(candidateEval);
        if (judge?.Judge(candidate.Graph, candidateEval) is { } judged && judge.Judge(current.Graph, currentEval) is { } now)
        {
            if (judged.Score >= now.Score)
                return false;
        }
        else if (judgedOnly || candidateMetric >= currentMetric)
            return false;

        current = candidate;
        currentEval = candidateEval;
        currentMetric = candidateMetric;
        return true;
    }

    /// <summary>
    /// Evaluates a single graph configuration without trying different strategies.
    /// </summary>
    public GraphEvaluationResult EvaluateGraph(InternalComputationGraph graph, params TensorData[] sampleInputs)
    {
        var shapeInfo = _shapeInference.Infer(graph, sampleInputs);
        return _evaluator.Evaluate(graph, shapeInfo);
    }
}
