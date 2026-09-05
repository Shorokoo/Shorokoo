namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The compute–memory objective the memory-aware pass minimizes, expressed as ratios to
/// the graph it started from.
///
/// <para>Compute time and peak bytes are unrelated quantities in unrelated units, so a
/// fixed coefficient on bytes cannot balance them across models: the coefficient that
/// makes memory count for a multi-gigabyte transformer step makes it invisible on a
/// kilobyte-scale one, and vice versa. Both terms are therefore divided by the baseline
/// graph's own value, so each reads 1 at the baseline and the weights mean what they look
/// like they mean — at <c>computeWeight == memoryWeight</c> a 1% compute increase exactly
/// pays for a 1% peak reduction, at any model size.</para>
///
/// <para>Lower is better. The baseline scores <c>computeWeight + memoryWeight</c>, so a
/// transform is worth committing exactly when it scores below that.</para>
/// </summary>
internal readonly struct ComputeMemoryObjective
{
    private readonly double _computeWeight;
    private readonly double _memoryWeight;
    private readonly double _baselineComputeTime;
    private readonly double _baselinePeakBytes;

    public ComputeMemoryObjective(double computeWeight, double memoryWeight, GraphEvaluationResult baseline)
    {
        _computeWeight = computeWeight;
        _memoryWeight = memoryWeight;

        // A degenerate baseline — an empty graph, or one whose ops the perf model costs at
        // zero — would make the ratio undefined. Fall back to 1 so the term degrades to the
        // raw value rather than producing NaN and poisoning every comparison.
        _baselineComputeTime = baseline.TotalComputeTime > 0 ? baseline.TotalComputeTime : 1.0;
        _baselinePeakBytes = baseline.PeakMemoryBytes > 0 ? baseline.PeakMemoryBytes : 1.0;
    }

    /// <summary>The objective value for a candidate graph's evaluation. Lower is better.</summary>
    public double Score(GraphEvaluationResult eval)
        => _computeWeight * (eval.TotalComputeTime / _baselineComputeTime)
         + _memoryWeight * (eval.PeakMemoryBytes / _baselinePeakBytes);

    /// <summary>
    /// The objective change from spending <paramref name="extraComputeTime"/> to avoid
    /// holding <paramref name="savedPeakBytes"/>, in the same normalized units as
    /// <see cref="Score"/>. Negative means the trade is worth making. Used to pre-filter
    /// rematerialization candidates before the cost of building and re-evaluating a graph.
    /// </summary>
    public double TradeDelta(double extraComputeTime, long savedPeakBytes)
        => _computeWeight * (extraComputeTime / _baselineComputeTime)
         - _memoryWeight * (savedPeakBytes / _baselinePeakBytes);
}
