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

    /// <summary>The rewrites around ONNX Runtime's kernels on its CUDA execution provider: those of
    /// <see cref="OnnxRuntime"/>, and those around what that provider alone does otherwise.</summary>
    public const string OnnxRuntimeCuda = "onnxruntime-cuda";

    /// <summary>What the ONNX node name of a call a set keeps on the host ends with
    /// (<see cref="Shorokoo.Core.Lowering.KernelWorkarounds.KernelWorkaround.KeepsOnHost"/>). A CUDA
    /// session of the ONNX Runtime backend assigns every node whose name holds it to the CPU
    /// execution provider; no other node name does.</summary>
    internal const string HostNodeNameSuffix = "@host";
}
