namespace Shorokoo.Core.Inference.Abstractions;

public interface IShorokooInferenceSession : IDisposable
{
    IReadOnlyList<string> InputNames { get; }
    IReadOnlyList<string> OutputNames { get; }

    // Runs the session. The returned values are owned by the caller and must
    // be disposed.
    IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames);

    // Whether this session's execution provider produces its outputs somewhere other than
    // host memory -- true for a GPU provider, false for a CPU one. When it is false
    // RunRetainingOutputs has nothing to retain and behaves exactly like Run.
    bool HasDeviceMemory { get; }

    // Runs the session leaving the outputs named in retainedOutputNames in the execution
    // provider's own memory instead of fetching them back to the host, so they can be fed
    // straight into the next run without crossing the bus. Every other output comes back
    // host-resident as Run's do. The returned values are owned by the caller and must be
    // disposed; a retained one is not host-readable (IShorokooTensorValue.IsHostAccessible).
    //
    // Inputs may themselves be device-resident values from an earlier run: the provider
    // uses them where they are.
    IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames);
}
