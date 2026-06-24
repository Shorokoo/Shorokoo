using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>
/// Local Response Normalization helper (Krizhevsky, Sutskever &amp; Hinton 2012,
/// AlexNet). Like <see cref="GatedLinear"/> / <see cref="Pooling"/>, a plain
/// graph-building helper rather than a <c>[Module]</c>, because the window
/// <c>size</c> is a compile-time structural integer (an ONNX attribute), not a
/// hyperparameter. Cross-channel (ONNX/Caffe <c>ACROSS_CHANNELS</c>) only.
/// </summary>
public static class LRNHelper
{
    /// <summary>
    /// Local Response Normalization over the channel axis (axis 1):
    /// <c>b_c = a_c · (k + (α/size)·Σ_{c'∈window(c)} a_{c'}²)^(−β)</c>, where
    /// <c>window(c) = [max(0, c−⌊(size−1)/2⌋), min(C−1, c+⌈(size−1)/2⌉)]</c>
    /// (the asymmetric floor/ceil split for even <paramref name="size"/> matches
    /// ONNX). Each activation is divided by a power of the squared sum of its
    /// neighbouring <b>channels</b> — the AlexNet "brightness normalization" /
    /// lateral-inhibition rescale; output has the same shape as the input.
    /// <para>
    /// Thin wrapper over the native, fully differentiable <c>OnnxOp.Lrn</c>.
    /// <paramref name="size"/> is the window width — a compile-time ONNX
    /// <i>attribute</i>, hence a plain C# <c>long</c> (this static helper is the
    /// arbitrary-<c>size</c> surface the <see cref="LocalResponseNorm"/> module,
    /// which bakes <c>size = 5</c>, lacks). <paramref name="k"/> is PyTorch's
    /// additive constant, mapped to ONNX's <c>bias</c> attribute. Defaults match
    /// ONNX / PyTorch (<c>size=5, α=1e-4, β=0.75, k=1</c>).
    /// </para>
    /// </summary>
    public static Tensor<T> Lrn<T>(Tensor<T> x, long size = 5,
        float alpha = 1e-4f, float beta = 0.75f, float k = 1.0f) where T : FloatLike
        => (Tensor<T>)OnnxOp.Lrn(x, alpha, beta, bias: k, size: size);   // k → ONNX 'bias'
}

/// <summary>
/// Local Response Normalization (Krizhevsky, Sutskever &amp; Hinton 2012, AlexNet)
/// as a param-free module: cross-channel normalization with the window width baked
/// to <c>size = 5</c> (the ONNX / PyTorch / Caffe default). For input
/// <c>a</c> shaped <c>[N, C, *spatial]</c> (channel = axis 1):
/// <para>
/// <c>b_c = a_c · (k + (α/5)·Σ_{c'∈window(c)} a_{c'}²)^(−β)</c>,
/// <c>window(c) = [max(0, c−2), min(C−1, c+2)]</c>,
/// </para>
/// i.e. each activation is rescaled by a power of the squared sum of its
/// neighbouring channels (the AlexNet lateral-inhibition rescale). This matches
/// PyTorch <c>nn.LocalResponseNorm(5, alpha, beta, k)</c> and ONNX
/// <c>LRN(size=5)</c>. <c>alpha</c>/<c>beta</c>/<c>k</c> are hyperparameters
/// (<c>k</c> = PyTorch's additive constant = ONNX's <c>bias</c>); the window width
/// is <b>fixed</b> because it is an ONNX compile-time attribute (the structural-int
/// story shared with <see cref="GLU"/>'s split axis and the pooling geometry) — for
/// a different width use the static helper <see cref="LRNHelper.Lrn{T}"/>.
/// <para>
/// The forward is built in-graph from the same primitives as the registered
/// <c>LRN</c> gradient (a channel-axis <c>Pad</c> plus an unrolled
/// channel-window <c>Slice</c>-sum of <c>x²</c>, then <c>x · pool^(−β)</c>), so
/// that <c>alpha</c>/<c>beta</c>/<c>k</c> enter as live <see cref="Scalar{T}"/>
/// hyperparameters (the norm-family convention: the fused ONNX norm/LRN ops take
/// these as static attributes, so the modules construct the math from differentiable
/// primitives instead, exactly as <see cref="LayerNorm"/>/<see cref="GroupNorm"/>
/// do for <c>epsilon</c>). It is state-free, so it runs on the plain inference
/// pipeline.
/// </para>
/// <para>
/// LRN is <b>largely superseded by Batch Normalization</b> (Ioffe &amp; Szegedy
/// 2015) and the later norm family; it is provided for AlexNet-era parity and
/// legacy-model loading, not as a recommended default for new models.
/// </para>
/// </summary>
[Module]
public partial class LocalResponseNorm
{
    public static Tensor<float32> Inline(
        Tensor<float32> x,                       // [N, C, *spatial], channel = axis 1
        [Hyper(1e-4f)] Scalar<float32> alpha,
        [Hyper(0.75f)] Scalar<float32> beta,
        [Hyper(1.0f)] Scalar<float32> k)         // PyTorch 'k' / ONNX 'bias' (additive constant)
    {
        // size is baked to the ONNX/PyTorch/Caffe default 5 (a compile-time
        // attribute; the channel window is unrolled at build, so it must be a
        // literal long — the same structural-int constraint as GLU's split axis).
        const long size = 5;

        // Asymmetric floor/ceil window split (matches the ONNX spec and the shipped
        // LRN gradient): size=5 → left=2, right=2 (symmetric here).
        const long leftHalf = (size - 1) / 2;
        const long rightHalf = size - 1 - leftHalf;

        // Channel count C (axis 1 of the shape) for the per-window slice bounds.
        var cDim = (Tensor<int64>)OnnxOp.Slice(OnnxOp.Shape(x), Vector(1L), Vector(2L));

        // ChannelWindowSum: zero-pad along the channel axis, then accumulate the
        // `size` shifted channel windows (Pad + unrolled Slice-and-Add — the same
        // primitive the registered LRN gradient uses; channel-axis, not spatial,
        // because ONNX pooling windows are spatial only).
        Tensor<float32> ChannelWindowSum(Tensor<float32> t)
        {
            var pads = Vector(leftHalf, rightHalf);
            var padded = (Tensor<float32>)OnnxOp.Pad(t, pads, null, axes: Vector(1L), mode: PadMode.Constant);

            var result = (Tensor<float32>)OnnxOp.Slice(padded, Vector(0L), cDim, Vector(1L));
            for (long i = 1; i < size; i++)
            {
                var start = Vector(i);
                var end = (Tensor<int64>)OnnxOp.Add(cDim, Vector(i));
                result = result + (Tensor<float32>)OnnxOp.Slice(padded, start, end, Vector(1L));
            }
            return result;
        }

        // pool = k + (α/size)·Σ_{c'∈window(c)} x_{c'}²   (live alpha/beta/k hypers).
        var windowSumSq = ChannelWindowSum(x * x);
        var pool = k + (alpha / Scalar((float)size)) * windowSumSq;

        // y = x · pool^(−β).
        return x * (Tensor<float32>)OnnxOp.Pow(pool, -beta);
    }
}
