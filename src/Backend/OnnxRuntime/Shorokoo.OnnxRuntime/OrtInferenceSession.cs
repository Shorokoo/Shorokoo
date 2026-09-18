using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtInferenceSession : IShorokooInferenceSession
{
    private readonly InferenceSession _session;
    private readonly int? _cudaDeviceId;
    private readonly IShorokooInferenceBackend _backend;

    // ORT's memory info for this session's own (non-host) output memory, or null when it
    // produces everything on the host. Held in a field, not a local: OrtMemoryInfo owns a
    // native handle and the binding below takes it as a bare pointer. Probed on first ask,
    // because most sessions never retain anything and a session is a common object here --
    // parameter initialization and every eager Eval build one.
    private readonly Lazy<OrtMemoryInfo?> _deviceMemoryInfo;

    public OrtInferenceSession(
        InferenceSession session, int? cudaDeviceId, IShorokooInferenceBackend backend)
    {
        _session = session;
        _cudaDeviceId = cudaDeviceId;
        _backend = backend;
        _deviceMemoryInfo = new Lazy<OrtMemoryInfo?>(() => DiscoverDeviceMemoryInfo(session));
    }

    /// <summary>
    /// The ORT value behind <paramref name="value"/>, which this session may feed.
    ///
    /// <para>An <see cref="OrtTensorValue"/> <i>of this assembly</i> is one of ours already. That
    /// test is exact rather than approximate: a backend loaded by
    /// <see cref="IsolatedBackend"/> gets a private copy of this assembly, so its
    /// <c>OrtTensorValue</c> is a different type from this one — and its handles point into a
    /// native runtime this session knows nothing about, which is precisely when feeding them
    /// would be a wild pointer rather than a mistake ORT could catch. Two backends over one
    /// loaded runtime share this assembly and so share the type, which is right too: their values
    /// are interchangeable and ORT moves them to the device itself.</para>
    ///
    /// <para>Anything else is rebuilt here, by this backend, from the source's contents. The copy
    /// belongs to nobody, so it is added to <paramref name="borrowed"/> for the caller to release
    /// once the run has read it. That list is created only when there is something to put in it:
    /// every feed of an ordinary single-backend run takes the first branch, and a training loop
    /// calls this once per input per step.</para>
    /// </summary>
    private OrtValue Unwrap(IShorokooTensorValue value, ref List<IShorokooTensorValue>? borrowed)
    {
        if (value is OrtTensorValue own) return own.Inner;
        var copy = BackendTransfer.CopyTo(_backend, value);
        (borrowed ??= []).Add(copy);
        return ((OrtTensorValue)copy).Inner;
    }

    public IReadOnlyList<string> InputNames => _session.InputNames;
    public IReadOnlyList<string> OutputNames => _session.OutputNames;

    public bool HasDeviceMemory => _deviceMemoryInfo.Value is not null;

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);
        List<IShorokooTensorValue>? borrowed = null;
        try
        {
            var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
            foreach (var (k, v) in inputs)
                ortInputs[k] = Unwrap(v, ref borrowed);

            using var runOptions = new RunOptions();
            ConfigureRun(runOptions, runSettings);
            var results = _session.Run(runOptions, ortInputs, outputNames);

            // ORT snapshots each input's handle into an IntPtr[] and keeps no reference to the OrtValue
            // wrappers, so from that point on `ortInputs` is their only root -- and the JIT retires it
            // at the call. OrtValue has an ordinary finalizer that calls OrtReleaseValue, so a GC inside
            // the native Run would free the feeds while it is still reading them. `_session` is rooted
            // by this instance and `runOptions` by the using; the inputs need this.
            GC.KeepAlive(ortInputs);

            // `results` is deliberately not disposed. It is a container whose Dispose would dispose
            // the values inside it, and those are exactly what this returns: each one is handed to an
            // OrtTensorValue, and from there to the TensorData that owns it and releases it when
            // disposed (Shorokoo/Shorokoo#180). The container itself holds nothing else to release.
            var wrapped = new List<IShorokooTensorValue>(results.Count);
            foreach (var r in results) wrapped.Add(new OrtTensorValue(r));
            return wrapped;
        }
        finally
        {
            // Copies made for feeds that came from another backend's runtime. They exist only for
            // the duration of the run: the outputs above are this session's own values, so nothing
            // the caller keeps points into one of these.
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
        }
    }

    public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames,
        RunSettings runSettings)
    {
        ArgumentNullException.ThrowIfNull(runSettings);

        // Nothing to retain, or nowhere to retain it: an unbound Run is the same thing and
        // costs one native call less.
        var deviceMemoryInfo = _deviceMemoryInfo.Value;
        if (deviceMemoryInfo is null || retainedOutputNames.Count == 0)
            return Run(inputs, outputNames, runSettings);

        List<IShorokooTensorValue>? borrowed = null;
        try
        {
            using var binding = _session.CreateIoBinding();
            foreach (var (k, v) in inputs)
                binding.BindInput(k, Unwrap(v, ref borrowed));

            // An output bound to a device is allocated there by ORT and left there; one bound to
            // the host allocator is fetched back exactly as an unbound Run fetches it. ORT sizes
            // both itself, so a shape it only learns while running is fine.
            var hostMemoryInfo = OrtMemoryInfo.DefaultInstance;
            foreach (var name in outputNames)
                binding.BindOutputToDevice(
                    name, retainedOutputNames.Contains(name) ? deviceMemoryInfo : hostMemoryInfo);

            using var runOptions = new RunOptions();
            ConfigureRun(runOptions, runSettings);
            var results = _session.RunWithBoundResults(runOptions, binding);

            // Same rooting hazard as Run: the binding holds the feeds' raw handles, not the managed
            // wrappers, so nothing but `inputs` keeps them alive across the native run. `borrowed`
            // roots any feed that had to be rebuilt here, which `inputs` does not hold.
            GC.KeepAlive(inputs);
            GC.KeepAlive(borrowed);

            // RunWithBoundResults returns the bound outputs in the binding's own order, which is the
            // order they were bound in -- ask it rather than assume, and hand them back in the order
            // the caller named.
            var boundNames = binding.GetOutputNames();
            // Nothing owns these values until each is wrapped and handed to a TensorData, and the
            // collection is deliberately not disposed, so anything that goes wrong between here and the
            // return leaks a device allocation apiece. Establish the shape first, and dispose the lot
            // if it is not what it must be.
            if (boundNames.Length != results.Count || results.Count != outputNames.Count)
            {
                foreach (var value in results) value.Dispose();
                throw new InvalidOperationException(
                    $"The run bound {boundNames.Length} outputs and returned {results.Count} values for "
                    + $"{outputNames.Count} requested names; they must agree one for one.");
            }

            var byName = new Dictionary<string, OrtValue>(results.Count);
            for (int i = 0; i < boundNames.Length; i++)
                byName[boundNames[i]] = results[i];

            var wrapped = new List<IShorokooTensorValue>(outputNames.Count);
            foreach (var name in outputNames)
            {
                // Same reason as the count check above, and the same handling: a name that does not
                // come back is a bad run, not an excuse to drop every device allocation it made.
                if (!byName.TryGetValue(name, out var value))
                {
                    foreach (var orphan in results) orphan.Dispose();
                    throw new InvalidOperationException(
                        $"The run bound no output named '{name}'; it bound "
                        + $"{string.Join(", ", boundNames)}.");
                }
                wrapped.Add(new OrtTensorValue(value));
            }
            return wrapped;
        }
        finally
        {
            if (borrowed is not null) foreach (var copy in borrowed) copy.Dispose();
        }
    }

    /// <summary>
    /// The memory the session's execution provider produces its outputs in, when that is not host
    /// memory; null when every output lands on the host (a CPU provider), and null too when the
    /// native build does not answer the question — which costs the retention, never correctness.
    /// The infos ORT reports are owned by the collection it returns, so this copies the one it
    /// keeps rather than outliving its source.
    /// </summary>
    private static OrtMemoryInfo? DiscoverDeviceMemoryInfo(InferenceSession session)
    {
        try
        {
            using var infos = session.GetMemoryInfosForOutputs();
            foreach (var info in infos)
            {
                if (info.Name == OrtTensorValue.CpuAllocatorName) continue;
                return new OrtMemoryInfo(info.Name, info.GetAllocatorType(), info.Id, info.GetMemoryType());
            }
        }
        // Catching broadly is the point: the doc above promises a failed probe costs the retention
        // and nothing else, and Lazy caches an escaping exception and rethrows it on every later
        // access -- which would fail every run of this session rather than fall back to the host.
        catch (Exception) { }
        return null;
    }

    /// <summary>Applies what <i>this</i> run runs with. The settings arrive per call rather than
    /// being held by the session, which is ORT's own shape for them: turning arena shrinkage on
    /// takes effect on a session that is already compiled, and on that run alone. Applied by both
    /// run paths, because a retaining run is the one that most wants the arena it keeps its state
    /// in bounded. It configures options the caller owns rather than returning new ones, so the
    /// handle stays inside a `using` at each call site.</summary>
    private void ConfigureRun(RunOptions runOptions, RunSettings runSettings)
    {
        if (OrtBackend.ArenaShrinkageRunConfig(_cudaDeviceId, runSettings.ShrinkArenaAfterRun)
            is { } arena)
            runOptions.AddRunConfigEntry("memory.enable_memory_arena_shrinkage", arena);
    }

    public void Dispose()
    {
        if (_deviceMemoryInfo.IsValueCreated) _deviceMemoryInfo.Value?.Dispose();
        _session.Dispose();
    }
}
