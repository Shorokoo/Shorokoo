namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Interface for computing performance characteristics of a specific ONNX operation.
/// Each implementation handles one or more related operations.
/// </summary>
internal interface IOpPerf
{
    /// <summary>
    /// Returns the set of operation codes this estimator handles.
    /// </summary>
    IReadOnlySet<string> SupportedOpCodes { get; }

    /// <summary>
    /// Estimates the performance characteristics of executing this operation
    /// with the given input context.
    /// </summary>
    OpPerfResult Estimate(OpPerfInput input);
}
