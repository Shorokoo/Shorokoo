namespace Shorokoo.Core.Backends;

// A session reads its inputs where its backend's runs read them and leaves its outputs there, and
// does nothing else with where a value is. Every value a run hands it is in the backend's run
// memory for that value -- IShorokooBackend.RunMemoryOf for a tensor of its element type,
// IShorokooBackend.SequenceRunMemory for a sequence -- because the framework places each input
// there before the run, through the backend's own moves. A session refuses any value outside that
// memory by throwing, before it runs anything: it never copies one in, so a value in the wrong
// place surfaces as an error rather than as a copy nobody asked for. Every output it returns is in
// that memory too, wherever the runtime computed it, and goes to the caller as it is.
public interface IShorokooSession : IDisposable
{
    IReadOnlyList<string> InputNames { get; }
    IReadOnlyList<string> OutputNames { get; }

    // Runs the session on `inputs`, every one of them in this backend's run memory, and returns the
    // outputs named in `outputNames`, every one of them in that memory too. The returned values are
    // owned by the caller and must be disposed. runSettings applies to this run alone -- ORT reads
    // these off the run, not the session, so two runs of one session may differ.
    IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings);

    // Runs the session as Run does, with the values in `consumed` handed over rather than lent.
    // Every run calls the overload below, which may also write outputs into consumed memory, and
    // whose default is this.
    //
    // A consumed value is one the caller has given up: a tensor fed to the run as it is, which the
    // run took when it started. From the moment this is called it belongs to this session's
    // backend alone, ON EVERY PATH -- a run that returns, one that throws, one that is stopped, and
    // one that fails before it reaches the native call, its refusal of an input outside its run
    // memory included -- and the caller never touches it again, not even to release it. The backend
    // must release each consumed value exactly once, through its own IShorokooBackend.Release, as
    // soon as the run no longer reads it: before this returns its outputs, and before it rethrows a
    // failure. That is the contract IShorokooBackend.CreateSequence has for the values it is handed,
    // and for the same reason: a caller that hands memory over cannot also be the one to free it. A
    // value may appear under several input names; it appears in `consumed` once.
    //
    // Every value in `consumed` is also in `inputs`, and every other value in `inputs` is lent:
    // read it, and leave it to the caller.
    //
    // The default runs the ordinary way and disposes each consumed value in a finally, which is what
    // IShorokooBackend.Release's own default does. A backend whose Release does more than dispose
    // implements this and releases through it.
    IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        try
        {
            return Run(inputs, outputNames, runSettings);
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
    // same value, and where the value is of the output's element type and its shape: an output that
    // could not be written there is produced as usual. The consumed value and the output are both in
    // this backend's run memory, so where they are never stands in the way.
    //
    // The default is RunConsuming, aliasing nothing.
    IReadOnlyList<IShorokooTensorValue> RunConsuming(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyCollection<IShorokooTensorValue> consumed,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings,
        out IReadOnlyList<string?> aliasedInputs)
    {
        aliasedInputs = [];
        return RunConsuming(inputs, consumed, outputNames, runSettings);
    }

    // The pairs of those it was built with that a run of this session can bind at all (see
    // RunConsuming), by this session's names: each output, and the input whose consumed memory it
    // may be written into. The framework feeds a consumed tensor in the run memory of the session's
    // backend -- as it is, where it is there already, and otherwise through a copy it makes there
    // -- so every pair is open to every run that consumes its input.
    //
    // The default is none: a session that aliases nothing binds no pair.
    IReadOnlyList<OutputAlias> BindableAliases => [];

    // This session's own memory arena as its runtime reports it, or null when the backend has no
    // such figures to give. Cheap enough to call either side of a run, which is how a run's peak
    // is attributed; see ArenaStatistics for why MaxInUseBytes alone cannot be.
    //
    // The default is null, which is what no figures reads as everywhere else here --
    // DeviceMemory.Read on a machine with no card answers the same way.
    ArenaStatistics? ReadArenaStatistics() => null;

    // The pinned host arena this session's execution provider stages its host-device crossings
    // through, where it has one -- a separate arena with figures of its own, never folded into the
    // ones above, since a byte of pinned host memory and a byte of device memory are not the same
    // thing and adding them would quietly change what either figure means.
    //
    // The default is null, which is also what a backend with no such arena answers: a CPU session
    // stages nothing, so there is nothing to report rather than an arena that read as empty.
    ArenaStatistics? ReadPinnedArenaStatistics() => null;

    // Where this session's runtime computes its outputs, before they are left in the backend's run
    // memory. The free half of the placement question: the session already knows, so no profiling
    // and no extra run is needed, and on a device backend an output computed in host memory is the
    // tail of a graph that ran on the host.
    //
    // The default is Unknown rather than Host: a backend that does not answer has not said its
    // outputs are computed on the host, and reading silence as Host would report a device backend's
    // fallback as a deliberate CPU session.
    SessionOutputPlacement OutputPlacement => SessionOutputPlacement.Unknown;

    // Which execution provider ran each node, or null when this session was not built to record it
    // -- which is the default, since recording costs every run the session makes. Asked for
    // through DiagnosticSettings.TraceNodePlacement on the context that compiles the session.
    NodePlacement? ReadNodePlacement() => null;
}
