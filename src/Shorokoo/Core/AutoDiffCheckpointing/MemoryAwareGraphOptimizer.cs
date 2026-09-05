using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Graph;
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
    /// The optimized <see cref="InternalComputationGraph"/> produced by the selected strategy.
    /// </summary>
    /// <summary>The winning strategy's rewritten graph. Internal: the rig freezes it
    /// into the readonly <c>TrainingStepPureGraph</c>; exposing the same instance as
    /// mutable public state would invalidate that wrapper's kind stamp.</summary>
    internal InternalComputationGraph OptimizedGraph { get; init; } = null!;

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
/// alternating strategies — <c>RematReorder</c> and <c>ReorderRemat</c> — and selects
/// the one with the best combined metric
/// (<c>computeFactor × computeTime + memoryFactor × peakMemory</c>).
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
    /// <para>Note this is a weight on a RATIO, not on bytes: the predecessor of this
    /// constant multiplied raw byte counts, which made it meaningful only for graphs whose
    /// peak happened to be around a million times their compute-time figure — on everything
    /// else the memory term vanished and the pass silently degenerated into a compute-only
    /// optimizer. That is the bug the normalization fixes, so do not reintroduce a
    /// byte-scaled constant here.</para>
    /// </summary>
    public const double DefaultMemoryWeight = 1.0;

    /// <summary>
    /// Multipliers on <see cref="DefaultMemoryWeight"/> that the search is restarted from.
    ///
    /// <para>These do NOT change what counts as a good graph — every candidate is finally
    /// judged by one objective at the configured weights. They exist because the search is
    /// a strict hill-climb over two alternating transforms, so it is path-dependent: an
    /// early transform that improves the objective a little can block a later one that
    /// would have improved it a lot, and how hard the memory term pushes decides which path
    /// is taken. Measured across MLP, conv, one- and two-layer transformer encoders, LSTM
    /// and dense/chunked attention graphs, no single pressure won everywhere — a light
    /// setting was best on the MLP and left half the available reduction on the table for
    /// the two-layer encoder, while a heavy one inverted that. Running the search from
    /// several pressures and keeping the best result beats every fixed choice on the same
    /// suite, and it degrades gracefully: a restart that finds nothing simply loses.</para>
    ///
    /// <para>Kept short deliberately — each entry re-runs both strategies, so this is a
    /// direct multiplier on optimization time.</para>
    /// </summary>
    /// <summary>
    /// How many times <see cref="Rematerializer"/> is re-run within one strategy step.
    ///
    /// <para>Each run already applies EVERY candidate it finds, so a second exists only to
    /// catch candidates the first one created. Past that it is mostly churn, and expensive
    /// churn: every iteration clones the graph and re-evaluates it end to end. Measured over
    /// the same spread of training graphs, raising this from 2 to 20 left every peak figure
    /// unchanged but one, made a transformer encoder's peak WORSE (the extra transforms walk
    /// the hill-climb into a poorer neighbourhood), and cost 4x the optimization time —
    /// 9.6s to 39s on one graph, 4.6s to 37.5s on another.</para>
    /// </summary>
    public const int DefaultRematerializationIterations = 2;

    /// <summary>
    /// Graphs whose peak is below this are handed back untouched.
    ///
    /// <para>This pass buys memory, and it is not free: it clones and re-evaluates the whole
    /// graph several times. Below a megabyte there is nothing worth buying — the absolute
    /// saving is smaller than one tensor of a real model — so the work is pure overhead on
    /// exactly the small graphs where rig-construction latency is most visible. (Its
    /// modeled compute term does improve on small graphs, but measured steady-state training
    /// throughput does not move, so that figure is not a reason to spend the time.)</para>
    /// </summary>
    public const long MinimumPeakBytesToOptimize = 1L << 20;

    private readonly GraphEvaluator _evaluator;
    private readonly ShapeInferenceInterpreter _shapeInference;
    private readonly double _computeFactor;
    private readonly double _memoryFactor;
    private readonly int _maxRematerializationIterations;

    public MemoryAwareGraphOptimizer(
        double computeFactor = DefaultComputeWeight,
        double memoryFactor = DefaultMemoryWeight,
        int maxRematerializationIterations = DefaultRematerializationIterations,
        GraphEvaluator? evaluator = null,
        ShapeInferenceInterpreter? shapeInference = null)
    {
        _evaluator = evaluator ?? new GraphEvaluator();
        _shapeInference = shapeInference ?? new ShapeInferenceInterpreter();
        _computeFactor = computeFactor;
        _memoryFactor = memoryFactor;
        _maxRematerializationIterations = maxRematerializationIterations;
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
        var baselineEval = _evaluator.Evaluate(graph, shapeInfo);

        if (baselineEval.PeakMemoryBytes < MinimumPeakBytesToOptimize)
        {
            return new GraphOptimizationResult
            {
                StrategyName = "Baseline",
                OptimizedGraph = graph,
                Evaluation = baselineEval,
                AllStrategies = [("Baseline", baselineEval, graph)],
            };
        }

        // The one objective every candidate is finally judged by, normalized to the graph we
        // started from so the weights behave identically at every model size.
        var selection = new ComputeMemoryObjective(_computeFactor, _memoryFactor, baselineEval);

        // Doing nothing is always a candidate, so the pass can never return a graph that
        // scores worse than the one it was handed.
        var strategies = new List<(string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph)>
        {
            ("Baseline", baselineEval, graph),
        };

        var scheduler = new MemoryAwareScheduler();
        var rematerializer = new Rematerializer(selection, _maxRematerializationIterations, _evaluator);
        InternalComputationGraph Reorder(InternalComputationGraph g) => scheduler.Reorder(g, shapeInfo);
        InternalComputationGraph Remat(InternalComputationGraph g) => rematerializer.Apply(g, shapeInfo);

        strategies.Add(RunAlternatingStrategy(
            "RematReorder", selection, graph, baselineEval, shapeInfo, Remat, Reorder));
        strategies.Add(RunAlternatingStrategy(
            "ReorderRemat", selection, graph, baselineEval, shapeInfo, Reorder, Remat));

        var best = strategies.OrderBy(s => selection.Score(s.Evaluation)).First();

        return new GraphOptimizationResult
        {
            StrategyName = best.Name,
            OptimizedGraph = best.Graph,
            Evaluation = best.Evaluation,
            AllStrategies = strategies,
        };
    }

    private (string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph) RunAlternatingStrategy(
        string name,
        ComputeMemoryObjective objective,
        InternalComputationGraph initialGraph,
        GraphEvaluationResult initialEval,
        ShapeInferenceResult shapeInfo,
        Func<InternalComputationGraph, InternalComputationGraph> firstPass,
        Func<InternalComputationGraph, InternalComputationGraph> secondPass)
    {
        var currentGraph = initialGraph;
        var currentEval = initialEval;
        var currentMetric = objective.Score(currentEval);

        TryApply(objective, firstPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo);
        TryApply(objective, secondPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo);

        while (true)
        {
            if (!TryApply(objective, firstPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo))
                break;
            if (!TryApply(objective, secondPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo))
                break;
        }

        return (name, currentEval, currentGraph);
    }

    private bool TryApply(
        ComputeMemoryObjective objective,
        Func<InternalComputationGraph, InternalComputationGraph> pass,
        ref InternalComputationGraph currentGraph,
        ref GraphEvaluationResult currentEval,
        ref double currentMetric,
        ShapeInferenceResult shapeInfo)
    {
        var candidate = pass(currentGraph);
        if (ReferenceEquals(candidate, currentGraph))
            return false;

        var candidateEval = _evaluator.Evaluate(candidate, shapeInfo);
        var candidateMetric = objective.Score(candidateEval);
        if (candidateMetric >= currentMetric)
            return false;

        currentGraph = candidate;
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
