namespace Shorokoo.Core.Backends;

/// <summary>
/// The floating-point precision the sessions of a <see cref="Shorokoo.Runtime.ComputeContext"/>
/// compute in. <b>By default <c>float32</c> is computed in full <c>float32</c> precision</b>, on every
/// backend and every device: a product, a convolution or a recurrent layer of <c>float32</c>
/// operands rounds nothing below <c>float32</c>'s 24-bit significand. Anything less precise is asked
/// for here, and only here.
///
/// <para>Like <see cref="DeviceMemorySettings"/>, <see cref="RunSettings"/> and
/// <see cref="DiagnosticSettings"/> it is a record, so a variation is a <c>with</c> expression rather
/// than a mutation something else can see:</para>
/// <code>
/// using Shorokoo.Core.Backends;
/// using Shorokoo.Runtime;
///
/// var fast = new ComputeContext(new WinGpuBackend())
/// {
///     Precision = new PrecisionSettings { AllowTensorFloat32 = true },
/// };
/// </code>
///
/// <para>It is read when a session is built, and the session keeps it for life: a graph compiled on
/// a context, and a training rig's steps on its <c>runtimeContext</c>, compute in what that context
/// carries.</para>
/// </summary>
public sealed record PrecisionSettings
{
    /// <summary>What a context gets when it names nothing: <c>float32</c> in full <c>float32</c>
    /// precision.</summary>
    public static PrecisionSettings Default { get; } = new();

    /// <summary>
    /// Whether a CUDA card may compute <c>float32</c> products, convolutions and recurrent layers in
    /// TensorFloat-32: on its tensor cores, with each operand's significand rounded to 11 bits and
    /// the sums kept in <c>float32</c>. Off by default. On, a card from Ampere on runs them several
    /// times faster where they are large, and the result is about three decimal digits
    /// short of <c>float32</c>'s.
    ///
    /// <para>It allows, and does not require: what each backend does with it is in the user guide
    /// (<c>inference.md</c>, "Precision"). On the ONNX Runtime CUDA backend it is the provider's
    /// <c>use_tf32</c>; on the PyTorch CUDA backend, torch's switches for cuBLAS and cuDNN, set for
    /// each run from its session; on the JAX CUDA backend, the precision XLA compiles each product
    /// and convolution in. It changes nothing on a CPU backend, which computes <c>float32</c> in
    /// full precision either way, and nothing for any other element type.</para>
    /// </summary>
    public bool AllowTensorFloat32 { get; init; }
}
