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
    private readonly GraphEvaluator _evaluator;
    private readonly ShapeInferenceInterpreter _shapeInference;
    private readonly double _computeFactor;
    private readonly double _memoryFactor;
    private readonly int _maxRematerializationIterations;

    public MemoryAwareGraphOptimizer(
        double computeFactor = 1.0,
        double memoryFactor = 1e-6,
        int maxRematerializationIterations = 20,
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
    /// Computes the combined metric for a given evaluation result.
    /// </summary>
    public double ComputeCombinedMetric(GraphEvaluationResult eval)
        => _computeFactor * eval.TotalComputeTime + _memoryFactor * eval.PeakMemoryBytes;

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
        var scheduler = new MemoryAwareScheduler();
        var rematerializer = new Rematerializer(
            _computeFactor, _memoryFactor, _maxRematerializationIterations, _evaluator);

        InternalComputationGraph Remat(InternalComputationGraph g) => rematerializer.Apply(g, shapeInfo);
        InternalComputationGraph Reorder(InternalComputationGraph g) => scheduler.Reorder(g, shapeInfo);

        var baselineEval = _evaluator.Evaluate(graph, shapeInfo);

        var s1 = RunAlternatingStrategy("RematReorder", graph, baselineEval, shapeInfo, Remat, Reorder);
        var s2 = RunAlternatingStrategy("ReorderRemat", graph, baselineEval, shapeInfo, Reorder, Remat);

        var strategies = new List<(string Name, GraphEvaluationResult Evaluation, InternalComputationGraph Graph)>
        {
            s1, s2,
        };

        var best = strategies.OrderBy(s => ComputeCombinedMetric(s.Evaluation)).First();

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
        InternalComputationGraph initialGraph,
        GraphEvaluationResult initialEval,
        ShapeInferenceResult shapeInfo,
        Func<InternalComputationGraph, InternalComputationGraph> firstPass,
        Func<InternalComputationGraph, InternalComputationGraph> secondPass)
    {
        var currentGraph = initialGraph;
        var currentEval = initialEval;
        var currentMetric = ComputeCombinedMetric(currentEval);

        TryApply(firstPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo);
        TryApply(secondPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo);

        while (true)
        {
            if (!TryApply(firstPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo))
                break;
            if (!TryApply(secondPass, ref currentGraph, ref currentEval, ref currentMetric, shapeInfo))
                break;
        }

        return (name, currentEval, currentGraph);
    }

    private bool TryApply(
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
        var candidateMetric = ComputeCombinedMetric(candidateEval);
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
