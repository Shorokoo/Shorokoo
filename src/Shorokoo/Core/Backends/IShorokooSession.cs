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

    // Runs the session -- as Run, or as RunRetainingOutputs when retainedOutputNames names any --
    // with the values in `consumed` handed over rather than lent. Every run calls the overload
    // below, which may also write outputs into consumed memory, and whose default is this.
    //
    // A consumed value is one the caller has given up: a tensor fed to the run as it is, which the
    // run took when it started. From the moment this is called it belongs to this session's
    // backend alone, ON EVERY PATH -- a run that returns, one that throws, one that is stopped, and
    // one that fails before it reaches the native call -- and the caller never touches it again,
    // not even to release it. The backend must release each consumed value exactly once, through
    // its own IShorokooBackend.Release, as soon as the run no longer reads it: before this returns
    // its outputs, and before it rethrows a failure. That is the contract
    // IShorokooBackend.CreateSequence has for the values it is handed, and for the same reason:
    // a caller that hands memory over cannot also be the one to free it. A value may appear under
    // several input names; it appears in `consumed` once.
    //
    // Every value in `consumed` is also in `inputs`, and every other value in `inputs` is lent:
    // read it, and leave it to the caller.
    //
    // Defaulted so a backend outside this repository keeps compiling, and so that one that does not
    // implement it keeps the contract anyway: the default runs the ordinary way and disposes each
    // consumed value in a finally, which is what IShorokooBackend.Release's own default does. A
    // backend whose Release does more than dispose implements this and releases through it.
    IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
    {
        try
        {
            return retainedOutputNames.Count == 0
                ? Run(inputs, outputNames, runSettings)
                : RunRetainingOutputs(inputs, outputNames, retainedOutputNames, runSettings);
        }
        finally
        {
            foreach (var value in consumed) value.Dispose();
        }
    }

    // RunConsuming, able to write an output into the memory of a value this run consumed rather
    // than into memory of its own -- output aliasing, for the pairs the session was built with (see
    // IShorokooBackend.CreateSession and OutputAlias) -- and saying which it did: `aliasedInputs`
    // holds, per output in `outputNames` order, the input name whose consumed value's memory that
    // output was written into, or null -- or is empty, where no output was written into anything,
    // which is what a run that aliases nothing answers without allocating. This is the call every
    // run makes.
    //
    // The contract on `consumed` is RunConsuming's, unchanged: each value is released exactly once,
    // through the backend, before this returns or rethrows. An aliased output is a value of its own
    // that holds the memory it was written into, so releasing the consumed value leaves the output
    // whole -- ONNX Runtime counts the references to a buffer, and releases it with the last.
    //
    // A session binds a pair only on a run that consumed the input, where no other input is fed the
    // same value, and where the value is in the memory the output is produced in, of the output's
    // element type and its shape: an output that could not be written there is produced as usual.
    //
    // Defaulted to RunConsuming, aliasing nothing, so a backend outside this repository keeps
    // compiling and keeps its contract.
    IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        return RunConsuming(inputs, consumed, outputNames, retainedOutputNames, runSettings);
    }

    // The inputs, by this session's names for them, that a run of it may write an output into the
    // consumed memory of (see RunConsuming): those of the pairs it was built with that it can bind at
    // all. The framework feeds a consumed tensor of one of these through a copy in memory it holds,
    // which the output can then be written into; one of any other input that the run cannot read
    // where it is goes to the session as the host has it, for the runtime to copy into its own arena
    // -- inside the limit the session was built with, rather than outside it cutting that limit.
    //
    // Defaulted to none, so a backend outside this repository keeps compiling: one that aliases
    // nothing has no such input.
    IReadOnlySet<string> AliasableInputs => System.Collections.Frozen.FrozenSet<string>.Empty;

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
