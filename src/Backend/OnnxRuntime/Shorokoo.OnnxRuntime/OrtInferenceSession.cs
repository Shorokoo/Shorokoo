using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtInferenceSession : IShorokooInferenceSession
{
    private readonly InferenceSession _session;

    // ORT's memory info for this session's own (non-host) output memory, or null when it
    // produces everything on the host. Held in a field, not a local: OrtMemoryInfo owns a
    // native handle and the binding below takes it as a bare pointer. Probed on first ask,
    // because most sessions never retain anything and a session is a common object here --
    // parameter initialization and every eager Eval build one.
    private readonly Lazy<OrtMemoryInfo?> _deviceMemoryInfo;

    public OrtInferenceSession(InferenceSession session)
    {
        _session = session;
        _deviceMemoryInfo = new Lazy<OrtMemoryInfo?>(() => DiscoverDeviceMemoryInfo(session));
    }

    public IReadOnlyList<string> InputNames => _session.InputNames;
    public IReadOnlyList<string> OutputNames => _session.OutputNames;

    public bool HasDeviceMemory => _deviceMemoryInfo.Value is not null;

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames)
    {
        var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
        foreach (var (k, v) in inputs)
            ortInputs[k] = ((OrtTensorValue)v).Inner;

        using var runOptions = new RunOptions();
        var results = _session.Run(runOptions, ortInputs, outputNames);

        // ORT snapshots each input's handle into an IntPtr[] and keeps no reference to the OrtValue
        // wrappers, so from that point on `ortInputs` is their only root -- and the JIT retires it
        // at the call. OrtValue has an ordinary finalizer that calls OrtReleaseValue, so a GC inside
        // the native Run would free the feeds while it is still reading them. `_session` is rooted
        // by this instance and `runOptions` by the using; the inputs need this.
        GC.KeepAlive(ortInputs);

        var wrapped = new List<IShorokooTensorValue>(results.Count);
        foreach (var r in results) wrapped.Add(new OrtTensorValue(r));
        return wrapped;
    }

    public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames,
        IReadOnlySet<string> retainedOutputNames)
    {
        // Nothing to retain, or nowhere to retain it: an unbound Run is the same thing and
        // costs one native call less.
        var deviceMemoryInfo = _deviceMemoryInfo.Value;
        if (deviceMemoryInfo is null || retainedOutputNames.Count == 0)
            return Run(inputs, outputNames);

        using var binding = _session.CreateIoBinding();
        foreach (var (k, v) in inputs)
            binding.BindInput(k, ((OrtTensorValue)v).Inner);

        // An output bound to a device is allocated there by ORT and left there; one bound to
        // the host allocator is fetched back exactly as an unbound Run fetches it. ORT sizes
        // both itself, so a shape it only learns while running is fine.
        var hostMemoryInfo = OrtMemoryInfo.DefaultInstance;
        foreach (var name in outputNames)
            binding.BindOutputToDevice(
                name, retainedOutputNames.Contains(name) ? deviceMemoryInfo : hostMemoryInfo);

        using var runOptions = new RunOptions();
        var results = _session.RunWithBoundResults(runOptions, binding);

        // Same rooting hazard as Run: the binding holds the feeds' raw handles, not the managed
        // wrappers, so nothing but `inputs` keeps them alive across the native run.
        GC.KeepAlive(inputs);

        // RunWithBoundResults returns the bound outputs in the binding's own order, which is the
        // order they were bound in -- ask it rather than assume, and hand them back in the order
        // the caller named.
        var boundNames = binding.GetOutputNames();
        var byName = new Dictionary<string, OrtValue>(results.Count);
        for (int i = 0; i < boundNames.Length && i < results.Count; i++)
            byName[boundNames[i]] = results[i];

        var wrapped = new List<IShorokooTensorValue>(outputNames.Count);
        foreach (var name in outputNames) wrapped.Add(new OrtTensorValue(byName[name]));
        return wrapped;
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
        catch (OnnxRuntimeException) { }
        catch (EntryPointNotFoundException) { }
        return null;
    }

    public void Dispose()
    {
        if (_deviceMemoryInfo.IsValueCreated) _deviceMemoryInfo.Value?.Dispose();
        _session.Dispose();
    }
}
