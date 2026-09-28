namespace Shorokoo.Core.Backends;

/// <summary>
/// The names of the kernel-workaround sets a backend's sessions can be built with, as
/// <see cref="IShorokooBackend.KernelWorkaroundSet"/> answers them. A set rewrites operator calls
/// a runtime's kernels compute otherwise than the ONNX spec into equivalent calls they compute as
/// the spec does, in the model built for that runtime's sessions only.
/// </summary>
public static class KernelWorkaroundSets
{
    /// <summary>The rewrites around ONNX Runtime's kernels.</summary>
    public const string OnnxRuntime = "onnxruntime";
}
