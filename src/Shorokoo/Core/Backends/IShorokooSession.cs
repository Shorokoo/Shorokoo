namespace Shorokoo.Core.Backends;

public interface IShorokooSession : IDisposable
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

    // This session's own memory arena as its runtime reports it, or null when the backend has no
    // such figures to give. Cheap enough to call either side of a run, which is how a run's peak
    // is attributed; see ArenaStatistics for why MaxInUseBytes alone cannot be.
    //
    // Defaulted, like HasDeviceMemory above, so a backend outside this repository keeps compiling:
    // a backend that does not answer has no arena figures, and null is what no figures reads as
    // everywhere else here -- DeviceMemory.Read on a machine with no card answers the same way.
    ArenaStatistics? ReadArenaStatistics() => null;

    // The pinned host arena this session's execution provider stages its host-device crossings
    // through, where it has one -- a separate arena with figures of its own, never folded into the
    // ones above, since a byte of pinned host memory and a byte of device memory are not the same
    // thing and adding them would quietly change what either figure means.
    //
    // Defaulted to null, which is also what a backend with no such arena answers: a CPU session
    // stages nothing, so there is nothing to report rather than an arena that read as empty.
    ArenaStatistics? ReadPinnedArenaStatistics() => null;

    // Where this session produces its outputs. The free half of the placement question: the
    // session already knows, so no profiling and no extra run is needed, and on a device backend a
    // host-memory output is the tail of a graph that ran on the host.
    //
    // Defaulted to Unknown rather than Host: a backend that does not answer has not said its
    // outputs are host-resident, and reading silence as Host would report a device backend's
    // fallback as a deliberate CPU session.
    SessionOutputPlacement OutputPlacement => SessionOutputPlacement.Unknown;

    // Which execution provider ran each node, or null when this session was not built to record it
    // -- which is the default, since recording costs every run the session makes. Asked for
    // through DiagnosticSettings.TraceNodePlacement on the context that compiles the session.
    NodePlacement? ReadNodePlacement() => null;
}
