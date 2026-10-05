using Shorokoo.Graph;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The memory-aware pass's objective with the peak the backend a step runs on holds, as that
/// backend's own model of a run says (<see cref="Shorokoo.Core.Backends.IShorokooBackend"/>'s
/// modelled run peak), in place of the peak the pass's evaluator charges: the compute the
/// evaluator models, the memory the backend's model. It is normalized to the graph the pass was
/// handed — its modelled compute and the backend's peak of it — as the pass's own objective is to
/// the pass's figures, so the two weigh a trade alike. The backend is asked about each graph once.
/// Where its model is quick to ask (<see cref="WeighsPlateaus"/>), the rematerializer asks it about
/// the trials a plateau of its own figures leaves; otherwise only the steps each strategy takes and the
/// strategies' ends are weighed by it.
/// </summary>
internal sealed class BackendJudge
{
    private readonly Func<InternalComputationGraph, long?> _peakOf;
    private readonly Dictionary<InternalComputationGraph, long?> _peaks = new(ReferenceEqualityComparer.Instance);
    private readonly double _computeWeight;
    private readonly double _memoryWeight;
    private readonly InternalComputationGraph _handed;
    private readonly double _handedComputeTime;
    private ComputeMemoryObjective? _objective;

    /// <summary>Whether the rematerializer weighs a plateau's trials by the backend's model.</summary>
    internal bool WeighsPlateaus { get; }

    /// <param name="peakOf">The backend's peak of a run of a graph, or null where its model cannot
    /// tell.</param>
    /// <param name="computeWeight">The objective's weight on compute.</param>
    /// <param name="memoryWeight">The objective's weight on memory.</param>
    /// <param name="handed">The graph the pass was handed, which the objective is normalized to.</param>
    /// <param name="handedEval">The evaluator's figures for <paramref name="handed"/>.</param>
    /// <param name="weighsPlateaus">Whether the backend's model is quick enough to weigh a plateau's
    /// trials by.</param>
    internal BackendJudge(Func<InternalComputationGraph, long?> peakOf, double computeWeight, double memoryWeight,
        InternalComputationGraph handed, GraphEvaluationResult handedEval, bool weighsPlateaus = true)
    {
        WeighsPlateaus = weighsPlateaus;
        _peakOf = peakOf;
        _computeWeight = computeWeight;
        _memoryWeight = memoryWeight;
        _handed = handed;
        _handedComputeTime = handedEval.TotalComputeTime;
    }

    /// <summary>The backend's peak of a run of <paramref name="graph"/>, or null where its model
    /// cannot tell.</summary>
    internal long? Peak(InternalComputationGraph graph)
    {
        if (!_peaks.TryGetValue(graph, out var peak))
            _peaks[graph] = peak = _peakOf(graph);
        return peak;
    }

    /// <summary>The objective on <paramref name="graph"/>, evaluated as <paramref name="eval"/>,
    /// with the backend's peak of it, and that peak; null where the backend's model cannot tell the
    /// peak of the graph or of the graph the pass was handed.</summary>
    internal (double Score, long Peak)? Judge(InternalComputationGraph graph, GraphEvaluationResult eval)
    {
        if (Peak(graph) is not { } peak) return null;
        if (_objective is null)
        {
            if (Peak(_handed) is not { } handedPeak) return null;
            _objective = new ComputeMemoryObjective(_computeWeight, _memoryWeight,
                new GraphEvaluationResult { TotalComputeTime = _handedComputeTime, PeakMemoryBytes = handedPeak, NodeDetails = [] });
        }
        return (_objective.Value.Score(new GraphEvaluationResult { TotalComputeTime = eval.TotalComputeTime, PeakMemoryBytes = peak, NodeDetails = [] }), peak);
    }
}
