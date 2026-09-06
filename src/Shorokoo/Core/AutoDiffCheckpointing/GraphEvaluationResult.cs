namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The result of evaluating a computation graph's performance characteristics.
/// Contains total compute time, peak memory usage, and per-node details.
/// </summary>
public class GraphEvaluationResult
{
    /// <summary>
    /// Total compute time for the entire graph, in normalized units
    /// where 256 float32 add operations = 1 unit.
    /// </summary>
    public double TotalComputeTime { get; init; }

    /// <summary>
    /// Peak memory usage (in bytes) across the entire graph execution.
    /// This represents the maximum amount of tensor memory simultaneously alive
    /// at any point during execution.
    /// </summary>
    public long PeakMemoryBytes { get; init; }

    /// <summary>
    /// How much of the graph's linear order ONNX Runtime will actually run — the fraction of
    /// consecutive kernel pairs in ORT's execution order whose node indices ascend (see
    /// <see cref="OrtExecutionOrder.Fidelity"/>). 1 means ORT runs the linear order verbatim.
    /// </summary>
    public double OrderFidelity { get; init; } = 1.0;

    /// <summary>
    /// Per-node evaluation details, in the order the evaluation walked the graph
    /// (<see cref="EvaluationOrder"/>); <see cref="NodeEvaluationInfo.NodeIndex"/> maps each
    /// entry back to its node.
    /// </summary>
    public required IReadOnlyList<NodeEvaluationInfo> NodeDetails { get; init; }

    public override string ToString()
        => $"ComputeTime={TotalComputeTime:F2}, PeakMemory={PeakMemoryBytes / (1024.0 * 1024.0):F2} MB";
}

/// <summary>
/// Evaluation details for a single node in the graph.
/// </summary>
public class NodeEvaluationInfo
{
    /// <summary>
    /// The operation code of this node.
    /// </summary>
    public required string OpCode { get; init; }

    /// <summary>
    /// The node's index in the evaluated graph's <c>Nodes</c> list.
    /// </summary>
    public int NodeIndex { get; init; }

    /// <summary>
    /// Compute time for this node in normalized units.
    /// </summary>
    public double ComputeTime { get; init; }

    /// <summary>
    /// Extra temporary memory used by this node during execution (in bytes).
    /// </summary>
    public long ExtraMemoryBytes { get; init; }

    /// <summary>
    /// Current memory usage (in bytes) after this node completes,
    /// accounting for all live tensors at this point.
    /// </summary>
    public long CurrentMemoryBytes { get; init; }

    /// <summary>
    /// Cumulative compute time up to and including this node.
    /// </summary>
    public double CumulativeComputeTime { get; init; }
}
