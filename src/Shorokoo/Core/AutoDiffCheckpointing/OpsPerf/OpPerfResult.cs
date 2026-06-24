namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance characteristics result for a single operation execution.
/// </summary>
internal class OpPerfResult
{
    /// <summary>
    /// Compute time in normalized units where 256 float32 add operations = 1 unit.
    /// </summary>
    public double ComputeTime { get; init; }

    /// <summary>
    /// Extra memory allocated during this operation beyond input/output buffers, in bytes.
    /// This includes any temporary workspace or intermediate buffers the op needs.
    /// </summary>
    public long ExtraMemoryBytes { get; init; }

    /// <summary>
    /// Maps output index → input index, indicating which input buffer is reused
    /// for which output (in-place operation). For example, {0: 0} means output 0
    /// reuses the buffer of input 0. When an input buffer is reused in-place,
    /// the output occupies the same memory so no additional allocation is needed.
    /// Empty if no in-place reuse is possible.
    /// </summary>
    public IReadOnlyDictionary<int, int> InPlaceBufferReuse { get; init; } =
        new Dictionary<int, int>();

    public static OpPerfResult Zero => new() { ComputeTime = 0, ExtraMemoryBytes = 0 };
}
