using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtInferenceSession : IShorokooInferenceSession
{
    private readonly InferenceSession _session;
    private readonly int? _cudaDeviceId;

    public OrtInferenceSession(InferenceSession session, int? cudaDeviceId)
    {
        _session = session;
        _cudaDeviceId = cudaDeviceId;
    }

    public IReadOnlyList<string> InputNames => _session.InputNames;
    public IReadOnlyList<string> OutputNames => _session.OutputNames;

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames)
    {
        var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
        foreach (var (k, v) in inputs)
            ortInputs[k] = ((OrtTensorValue)v).Inner;

        // Read per run, not per session, so turning arena shrinkage on takes effect on sessions
        // that are already compiled.
        using var runOptions = new RunOptions();
        if (OrtSessionFactory.ArenaShrinkageRunConfig(_cudaDeviceId, DeviceMemory.ShrinkArenaAfterRun)
            is { } arena)
            runOptions.AddRunConfigEntry("memory.enable_memory_arena_shrinkage", arena);
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

    public void Dispose() => _session.Dispose();
}
