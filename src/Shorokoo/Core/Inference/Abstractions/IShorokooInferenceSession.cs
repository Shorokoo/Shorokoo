namespace Shorokoo.Core.Inference.Abstractions;

public interface IShorokooInferenceSession : IDisposable
{
    IReadOnlyList<string> InputNames { get; }
    IReadOnlyList<string> OutputNames { get; }

    // Runs the session. The returned values are owned by the caller and must
    // be disposed. runSettings applies to this run alone -- ORT reads these off the run, not
    // the session, so two runs of one session may differ.
    IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings);

    // Whether this session's execution provider produces its outputs somewhere other than
    // host memory -- true for a GPU provider, false for a CPU one. When it is false
    // RunRetainingOutputs has nothing to retain and behaves exactly like Run.
    //
    // Defaulted so that a backend outside this repository keeps compiling when this interface
    // grows: a backend that does not answer is one that produces everything on the host, and the
    // default below follows from that rather than papering over it.
    bool HasDeviceMemory => false;

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
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings) => Run(inputs, outputNames, runSettings);
}
