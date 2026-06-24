using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>
/// N-D pooling helpers over NC… input (channels at axis 1), in the same plain
/// C#-argument graph-building style as <see cref="Convolution"/>. ONNX pooling
/// geometry (kernel, stride, padding, dilation) only exists as static node
/// attributes — there is no attribute-tensorized pooling variant (the registry
/// only covers Conv) — so these are plain helpers rather than
/// hyperparameter-driven <c>[Module]</c>s, and unlike <see cref="Convolution"/>
/// they stay generic in the float element type (pooling owns no parameters).
/// </summary>
/// <remarks>
/// <para>
/// The per-axis <c>MaxPool</c>/<c>AvgPool</c>/<c>LpPool</c> helpers infer the
/// spatial rank from <c>kernelSize.Length</c> and take per-axis
/// <c>stride</c>/<c>padding</c>/<c>dilation</c> as plain <c>long[]</c> (length 1
/// broadcasts to every spatial axis; <c>padding</c> may also be
/// <c>2*spatialRank</c> ONNX <c>[begin₁…beginₙ, end₁…endₙ]</c>). Scalar-square
/// convenience overloads broadcast one scalar per knob (rank from
/// <c>x.Rank() - 2</c>, which must be known at build time), and the
/// <c>*1d/2d/3d</c> aliases assert the rank. In every form <c>stride</c> defaults
/// to <c>kernelSize</c> (the PyTorch convention). The historical scalar
/// <see cref="MaxPool2d{T}(Tensor{T}, long, long?, long, long, bool)"/> /
/// <see cref="AvgPool2d{T}(Tensor{T}, long, long?, long, bool, bool)"/> signatures
/// are kept verbatim for source compatibility and coexist with the per-axis
/// <c>long[]</c> aliases (C# overload resolution distinguishes <c>long</c> from
/// <c>long[]</c>).
/// </para>
/// <para>
/// <b>Defaults &amp; divergences.</b> <c>stride</c> defaults to <c>kernelSize</c>;
/// <c>AvgPool</c>'s <c>countIncludePad</c> defaults to <b>false</b> (divide by the
/// count of real cells) — this matches the pre-existing
/// <see cref="AvgPool2d{T}(Tensor{T}, long, long?, long, bool, bool)"/> contract
/// and <b>diverges from PyTorch's <c>count_include_pad=True</c></b>; pass
/// <c>countIncludePad: true</c> to recover the PyTorch denominator.
/// <c>LpPool</c>'s norm order <c>p</c> defaults to <c>2</c> (L2) and is an
/// <b>integer</b> — ONNX <c>LpPool</c> has no fractional norm, so PyTorch's float
/// <c>norm_type</c> is not expressible.
/// </para>
/// <para>
/// <b>Inference-grade backward caveats.</b> The windowed pools expose their full
/// forward attribute surface, but some gradients are restricted:
/// <list type="bullet">
/// <item><description><b>AvgPool</b>: <c>ceilMode: true</c> <b>throws</b> in the
/// backward pass — use it for inference only.</description></item>
/// <item><description><b>LpPool</b>: the gradient <b>ignores</b> <c>ceilMode</c>,
/// <c>dilation</c>, and <c>autoPad</c> — training an LpPool with non-default
/// geometry gives an incorrect gradient for those knobs.</description></item>
/// <item><description><b>MaxPool</b>: exact for every attribute (ties route the
/// gradient to the first max). <b>MaxUnpool</b> is fully differentiable.</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Out of scope (not expressible).</b> Adaptive pooling
/// (<c>AdaptiveAvg/MaxPool*</c>) beyond the <c>output_size == 1</c> case — which
/// <b>is</b> <see cref="GlobalAvgPool2d{T}"/>/<see cref="GlobalMaxPool2d{T}"/>/
/// <see cref="GlobalLpPool{T}"/> — has no general ONNX operator; fractional
/// (stochastic-window) max pooling has no core op. Both are deferred.
/// </para>
/// </remarks>
public static class Pooling
{
    // =====================================================================
    //  MaxPool
    // =====================================================================

    /// <summary>
    /// N-D max pooling over NC… input with per-axis geometry. The spatial rank is
    /// inferred as <c>kernelSize.Length</c>; <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>. Differentiable (ties route the gradient to
    /// the first max).
    /// </summary>
    /// <param name="x">Input <c>[N, C, d₁…dₙ]</c>.</param>
    /// <param name="kernelSize">Per-axis window size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride; length 1 (broadcast) or spatialRank. Defaults to <paramref name="kernelSize"/>.</param>
    /// <param name="padding">Length spatialRank (symmetric) or 2*spatialRank (ONNX <c>[begin₁…beginₙ, end₁…endₙ]</c>). Default all-0; ignored when <paramref name="autoPad"/> is set.</param>
    /// <param name="dilation">Per-axis dilation; length 1 (broadcast) or spatialRank. Default all-1.</param>
    /// <param name="ceilMode">Round the output size up instead of down.</param>
    /// <param name="autoPad">ONNX auto-pad mode; <see cref="AutoPad.SameUpper"/> matches PyTorch/TF <c>"same"</c>.</param>
    public static Tensor<T> MaxPool<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = RequireSpatialRank(kernelSize);
        long[] strides = Convolution.BroadcastAxes(stride ?? kernelSize, 1L, spatialRank, nameof(stride));
        long[] dilations = Convolution.BroadcastAxes(dilation, 1L, spatialRank, nameof(dilation));
        long[]? pads = autoPad == AutoPad.NotSet ? Convolution.ResolvePads(padding, spatialRank) : null;

        return NN.MaxPool(x, ceilMode,
            dilations: dilations,
            kernelShape: kernelSize,
            pads: pads,
            storageOrder: 0L,
            strides: strides,
            autoPad: autoPad);
    }

    /// <summary>
    /// Square/cubic max-pool convenience overload: one scalar per geometry knob,
    /// broadcast to every spatial axis (rank from <c>x.Rank() - 2</c>, which must
    /// be known at graph-build time). <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>. See the per-axis
    /// <see cref="MaxPool{T}(Tensor{T}, long[], long[], long[], long[], bool, AutoPad)"/>.
    /// </summary>
    public static Tensor<T> MaxPool<T>(
        Tensor<T> x, long kernelSize,
        long? stride = null, long padding = 0, long dilation = 1,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = Convolution.SpatialRankOf(x);
        return MaxPool(x,
            Convolution.Repeat(kernelSize, spatialRank),
            Convolution.Repeat(stride ?? kernelSize, spatialRank),
            Convolution.Repeat(padding, spatialRank),
            Convolution.Repeat(dilation, spatialRank),
            ceilMode, autoPad);
    }

    /// <summary>1-D max pooling over NCL input. <paramref name="kernelSize"/> must have length 1.</summary>
    public static Tensor<T> MaxPool1d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 1, nameof(MaxPool1d));
        return MaxPool(x, kernelSize, stride, padding, dilation, ceilMode, autoPad);
    }

    /// <summary>2-D max pooling over NCHW input. <paramref name="kernelSize"/> must have length 2.</summary>
    public static Tensor<T> MaxPool2d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 2, nameof(MaxPool2d));
        return MaxPool(x, kernelSize, stride, padding, dilation, ceilMode, autoPad);
    }

    /// <summary>3-D max pooling over NCDHW input. <paramref name="kernelSize"/> must have length 3.</summary>
    public static Tensor<T> MaxPool3d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 3, nameof(MaxPool3d));
        return MaxPool(x, kernelSize, stride, padding, dilation, ceilMode, autoPad);
    }

    /// <summary>
    /// Max pooling with a square <paramref name="kernelSize"/> window over NCHW
    /// input. Historical scalar 2-D helper, retained verbatim for source
    /// compatibility; <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>.
    /// </summary>
    public static Tensor<T> MaxPool2d<T>(Tensor<T> x, long kernelSize, long? stride = null,
        long padding = 0, long dilation = 1, bool ceilMode = false)
        where T : FloatLike
    {
        long s = stride ?? kernelSize;
        return NN.MaxPool(x, ceilMode,
            dilations: [dilation, dilation],
            kernelShape: [kernelSize, kernelSize],
            pads: [padding, padding, padding, padding],
            storageOrder: 0L,
            strides: [s, s]);
    }

    // =====================================================================
    //  AvgPool
    // =====================================================================

    /// <summary>
    /// N-D average pooling over NC… input with per-axis geometry. The spatial rank
    /// is inferred as <c>kernelSize.Length</c>; <paramref name="stride"/> defaults
    /// to <paramref name="kernelSize"/>.
    /// </summary>
    /// <param name="x">Input <c>[N, C, d₁…dₙ]</c>.</param>
    /// <param name="kernelSize">Per-axis window size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride; length 1 (broadcast) or spatialRank. Defaults to <paramref name="kernelSize"/>.</param>
    /// <param name="padding">Length spatialRank (symmetric) or 2*spatialRank (ONNX begin..end). Default all-0; ignored when <paramref name="autoPad"/> is set.</param>
    /// <param name="dilation">Per-axis dilation; length 1 (broadcast) or spatialRank. Default all-1 (ONNX ≥ opset 19; PyTorch AvgPool has none).</param>
    /// <param name="ceilMode">Round the output size up instead of down. <b>Throws in the backward pass</b> when <c>true</c> (forward / inference-grade only).</param>
    /// <param name="countIncludePad">When <c>true</c>, padded cells count toward the window denominator (PyTorch's default); the default <c>false</c> divides by the count of real cells only (matching the historical 2-D helper).</param>
    /// <param name="autoPad">ONNX auto-pad mode.</param>
    public static Tensor<T> AvgPool<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, bool countIncludePad = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = RequireSpatialRank(kernelSize);
        long[] strides = Convolution.BroadcastAxes(stride ?? kernelSize, 1L, spatialRank, nameof(stride));
        long[] dilations = Convolution.BroadcastAxes(dilation, 1L, spatialRank, nameof(dilation));
        long[]? pads = autoPad == AutoPad.NotSet ? Convolution.ResolvePads(padding, spatialRank) : null;

        return (Tensor<T>)OnnxOp.AveragePool(x, autoPad,
            ceilMode ? true : null,
            countIncludePad ? true : null,
            dilations: dilations,
            kernelShape: kernelSize,
            pads: pads,
            strides: strides);
    }

    /// <summary>
    /// Square/cubic average-pool convenience overload: one scalar per geometry
    /// knob, broadcast to every spatial axis (rank from <c>x.Rank() - 2</c>, which
    /// must be known at graph-build time). <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>. See the per-axis
    /// <see cref="AvgPool{T}(Tensor{T}, long[], long[], long[], long[], bool, bool, AutoPad)"/>.
    /// </summary>
    public static Tensor<T> AvgPool<T>(
        Tensor<T> x, long kernelSize,
        long? stride = null, long padding = 0, long dilation = 1,
        bool ceilMode = false, bool countIncludePad = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = Convolution.SpatialRankOf(x);
        return AvgPool(x,
            Convolution.Repeat(kernelSize, spatialRank),
            Convolution.Repeat(stride ?? kernelSize, spatialRank),
            Convolution.Repeat(padding, spatialRank),
            Convolution.Repeat(dilation, spatialRank),
            ceilMode, countIncludePad, autoPad);
    }

    /// <summary>1-D average pooling over NCL input. <paramref name="kernelSize"/> must have length 1.</summary>
    public static Tensor<T> AvgPool1d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, bool countIncludePad = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 1, nameof(AvgPool1d));
        return AvgPool(x, kernelSize, stride, padding, dilation, ceilMode, countIncludePad, autoPad);
    }

    /// <summary>2-D average pooling over NCHW input. <paramref name="kernelSize"/> must have length 2.</summary>
    public static Tensor<T> AvgPool2d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, bool countIncludePad = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 2, nameof(AvgPool2d));
        return AvgPool(x, kernelSize, stride, padding, dilation, ceilMode, countIncludePad, autoPad);
    }

    /// <summary>3-D average pooling over NCDHW input. <paramref name="kernelSize"/> must have length 3.</summary>
    public static Tensor<T> AvgPool3d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        bool ceilMode = false, bool countIncludePad = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 3, nameof(AvgPool3d));
        return AvgPool(x, kernelSize, stride, padding, dilation, ceilMode, countIncludePad, autoPad);
    }

    /// <summary>
    /// Average pooling with a square <paramref name="kernelSize"/> window over NCHW
    /// input. Historical scalar 2-D helper, retained verbatim for source
    /// compatibility; <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/> and <paramref name="countIncludePad"/> defaults
    /// to <c>false</c>.
    /// </summary>
    public static Tensor<T> AvgPool2d<T>(Tensor<T> x, long kernelSize, long? stride = null,
        long padding = 0, bool ceilMode = false, bool countIncludePad = false)
        where T : FloatLike
    {
        long s = stride ?? kernelSize;
        return (Tensor<T>)OnnxOp.AveragePool(x, AutoPad.NotSet,
            ceilMode ? true : null,
            countIncludePad ? true : null,
            dilations: null,
            kernelShape: [kernelSize, kernelSize],
            pads: [padding, padding, padding, padding],
            strides: [s, s]);
    }

    // =====================================================================
    //  LpPool
    // =====================================================================

    /// <summary>
    /// N-D Lᵖ-norm pooling over NC… input with per-axis geometry: each window
    /// reduces to <c>(Σ |x|ᵖ)^(1/p)</c> (<c>p = 2</c> ⇒ L2 = √(Σ x²),
    /// <c>p = 1</c> ⇒ Σ |x|). The spatial rank is inferred as
    /// <c>kernelSize.Length</c>; <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>. <paramref name="p"/> is an integer — ONNX
    /// <c>LpPool</c> has no fractional norm.
    /// </summary>
    /// <param name="x">Input <c>[N, C, d₁…dₙ]</c>.</param>
    /// <param name="kernelSize">Per-axis window size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride; length 1 (broadcast) or spatialRank. Defaults to <paramref name="kernelSize"/>.</param>
    /// <param name="padding">Length spatialRank (symmetric) or 2*spatialRank (ONNX begin..end). Default all-0; ignored when <paramref name="autoPad"/> is set.</param>
    /// <param name="dilation">Per-axis dilation; length 1 (broadcast) or spatialRank. Default all-1.</param>
    /// <param name="p">Norm order (integer); default <c>2</c> (L2).</param>
    /// <param name="ceilMode">Round the output size up instead of down.</param>
    /// <param name="autoPad">ONNX auto-pad mode.</param>
    /// <remarks>
    /// <b>Inference-grade backward caveat:</b> the LpPool gradient <b>ignores</b>
    /// <paramref name="ceilMode"/>, <paramref name="dilation"/>, and
    /// <paramref name="autoPad"/> — these knobs are forward-correct but produce an
    /// incorrect gradient when set to non-default values during training.
    /// </remarks>
    public static Tensor<T> LpPool<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long p = 2, bool ceilMode = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = RequireSpatialRank(kernelSize);
        long[] strides = Convolution.BroadcastAxes(stride ?? kernelSize, 1L, spatialRank, nameof(stride));
        long[] dilations = Convolution.BroadcastAxes(dilation, 1L, spatialRank, nameof(dilation));
        long[]? pads = autoPad == AutoPad.NotSet ? Convolution.ResolvePads(padding, spatialRank) : null;

        return (Tensor<T>)OnnxOp.LpPool(x, autoPad,
            ceilMode ? true : null,
            dilations: dilations,
            kernelShape: kernelSize,
            p: p,
            pads: pads,
            strides: strides);
    }

    /// <summary>
    /// Square/cubic Lᵖ-pool convenience overload: one scalar per geometry knob,
    /// broadcast to every spatial axis (rank from <c>x.Rank() - 2</c>, which must
    /// be known at graph-build time). <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>. See the per-axis
    /// <see cref="LpPool{T}(Tensor{T}, long[], long[], long[], long[], long, bool, AutoPad)"/>.
    /// </summary>
    public static Tensor<T> LpPool<T>(
        Tensor<T> x, long kernelSize,
        long? stride = null, long padding = 0, long dilation = 1,
        long p = 2, bool ceilMode = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = Convolution.SpatialRankOf(x);
        return LpPool(x,
            Convolution.Repeat(kernelSize, spatialRank),
            Convolution.Repeat(stride ?? kernelSize, spatialRank),
            Convolution.Repeat(padding, spatialRank),
            Convolution.Repeat(dilation, spatialRank),
            p, ceilMode, autoPad);
    }

    /// <summary>1-D Lᵖ pooling over NCL input. <paramref name="kernelSize"/> must have length 1.</summary>
    public static Tensor<T> LpPool1d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long p = 2, bool ceilMode = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 1, nameof(LpPool1d));
        return LpPool(x, kernelSize, stride, padding, dilation, p, ceilMode, autoPad);
    }

    /// <summary>2-D Lᵖ pooling over NCHW input. <paramref name="kernelSize"/> must have length 2.</summary>
    public static Tensor<T> LpPool2d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long p = 2, bool ceilMode = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 2, nameof(LpPool2d));
        return LpPool(x, kernelSize, stride, padding, dilation, p, ceilMode, autoPad);
    }

    /// <summary>3-D Lᵖ pooling over NCDHW input. <paramref name="kernelSize"/> must have length 3.</summary>
    public static Tensor<T> LpPool3d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long p = 2, bool ceilMode = false,
        AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 3, nameof(LpPool3d));
        return LpPool(x, kernelSize, stride, padding, dilation, p, ceilMode, autoPad);
    }

    // =====================================================================
    //  Global pools
    // =====================================================================

    /// <summary>Global average pooling: <c>[N, C, d₁…dₙ] → [N, C, 1…1]</c> (collapses every spatial axis).</summary>
    public static Tensor<T> GlobalAvgPool2d<T>(Tensor<T> x) where T : FloatLike
        => NN.GlobalAveragePool(x);

    /// <summary>Global max pooling: <c>[N, C, d₁…dₙ] → [N, C, 1…1]</c> (collapses every spatial axis).</summary>
    public static Tensor<T> GlobalMaxPool2d<T>(Tensor<T> x) where T : FloatLike
        => NN.GlobalMaxPool(x);

    /// <summary>
    /// Global Lᵖ pooling: <c>[N, C, d₁…dₙ] → [N, C, 1…1]</c>, the Lᵖ norm over all
    /// spatial axes at once. <paramref name="p"/> is an integer; default <c>2</c>.
    /// </summary>
    public static Tensor<T> GlobalLpPool<T>(Tensor<T> x, long p = 2) where T : FloatLike
        => NN.GlobalLpPool(x, p);

    // =====================================================================
    //  MaxUnpool flow
    // =====================================================================

    /// <summary>
    /// Max pooling that also returns the flat spatial indices of the selected
    /// elements (the ONNX MaxPool 2nd output), for feeding a later
    /// <see cref="MaxUnpool{T}(Tensor{T}, Tensor{int64}, long[], long[], long[], Vector{int64})"/>.
    /// The spatial rank is inferred as <c>kernelSize.Length</c>;
    /// <paramref name="stride"/> defaults to <paramref name="kernelSize"/>.
    /// </summary>
    /// <param name="x">Input <c>[N, C, d₁…dₙ]</c>.</param>
    /// <param name="kernelSize">Per-axis window size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride; length 1 (broadcast) or spatialRank. Defaults to <paramref name="kernelSize"/>.</param>
    /// <param name="padding">Length spatialRank (symmetric) or 2*spatialRank (ONNX begin..end). Default all-0; ignored when <paramref name="autoPad"/> is set.</param>
    /// <param name="ceilMode">Round the output size up instead of down.</param>
    /// <param name="autoPad">ONNX auto-pad mode.</param>
    /// <returns>The pooled <c>values</c> and the <c>int64</c> flat <c>indices</c> of the per-window maxima.</returns>
    /// <remarks>
    /// The companion indices-returning pool omits the <c>dilations</c> and
    /// <c>storage_order</c> attributes (kernel/pads/strides only) — sufficient for
    /// the <see cref="MaxUnpool{T}(Tensor{T}, Tensor{int64}, long[], long[], long[], Vector{int64})"/>
    /// round-trip, whose geometry is kernel/pads/strides.
    /// </remarks>
    public static (Tensor<T> values, Tensor<int64> indices) MaxPoolWithIndices<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = RequireSpatialRank(kernelSize);
        long[] strides = Convolution.BroadcastAxes(stride ?? kernelSize, 1L, spatialRank, nameof(stride));
        long[] pads = Convolution.ResolvePads(padding, spatialRank);

        return NN.MaxPoolWithIndices(x, ceilMode,
            kernelShape: kernelSize,
            pads: pads,
            strides: strides,
            autoPad: autoPad);
    }

    /// <summary>
    /// Square/cubic <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>
    /// convenience overload: one scalar per geometry knob, broadcast to every
    /// spatial axis (rank from <c>x.Rank() - 2</c>, which must be known at
    /// graph-build time). <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>.
    /// </summary>
    public static (Tensor<T> values, Tensor<int64> indices) MaxPoolWithIndices<T>(
        Tensor<T> x, long kernelSize,
        long? stride = null, long padding = 0,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        int spatialRank = Convolution.SpatialRankOf(x);
        return MaxPoolWithIndices(x,
            Convolution.Repeat(kernelSize, spatialRank),
            Convolution.Repeat(stride ?? kernelSize, spatialRank),
            Convolution.Repeat(padding, spatialRank),
            ceilMode, autoPad);
    }

    /// <summary>1-D <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>. <paramref name="kernelSize"/> must have length 1.</summary>
    public static (Tensor<T> values, Tensor<int64> indices) MaxPoolWithIndices1d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 1, nameof(MaxPoolWithIndices1d));
        return MaxPoolWithIndices(x, kernelSize, stride, padding, ceilMode, autoPad);
    }

    /// <summary>2-D <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>. <paramref name="kernelSize"/> must have length 2.</summary>
    public static (Tensor<T> values, Tensor<int64> indices) MaxPoolWithIndices2d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 2, nameof(MaxPoolWithIndices2d));
        return MaxPoolWithIndices(x, kernelSize, stride, padding, ceilMode, autoPad);
    }

    /// <summary>3-D <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>. <paramref name="kernelSize"/> must have length 3.</summary>
    public static (Tensor<T> values, Tensor<int64> indices) MaxPoolWithIndices3d<T>(
        Tensor<T> x, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        bool ceilMode = false, AutoPad autoPad = AutoPad.NotSet)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 3, nameof(MaxPoolWithIndices3d));
        return MaxPoolWithIndices(x, kernelSize, stride, padding, ceilMode, autoPad);
    }

    /// <summary>
    /// Scatters each pooled value back to its recorded position in a larger zero
    /// tensor (the partial inverse of <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>;
    /// ONNX MaxUnpool). Fully differentiable. The spatial rank is inferred as
    /// <c>kernelSize.Length</c>; <paramref name="stride"/> defaults to
    /// <paramref name="kernelSize"/>.
    /// </summary>
    /// <param name="values">The pooled values (1st output of <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>).</param>
    /// <param name="indices">The flat per-window max indices (2nd output of <see cref="MaxPoolWithIndices{T}(Tensor{T}, long[], long[], long[], bool, AutoPad)"/>).</param>
    /// <param name="kernelSize">Per-axis window size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride; length 1 (broadcast) or spatialRank. Defaults to <paramref name="kernelSize"/>.</param>
    /// <param name="padding">Length spatialRank (symmetric) or 2*spatialRank (ONNX begin..end). Default all-0.</param>
    /// <param name="outputShape">
    /// Optional in-graph target shape (full <c>[N, C, d₁…dₙ]</c>). When given it
    /// <b>overrides</b> the <c>(in_i-1)·stride_i - padBegin_i - padEnd_i + k_i</c>
    /// sizing; when <c>null</c> the op omits the optional input and uses the
    /// formula.
    /// </param>
    /// <remarks>
    /// MaxUnpool has no dilation or ceil-mode knobs (the core op takes none).
    /// </remarks>
    public static Tensor<T> MaxUnpool<T>(
        Tensor<T> values, Tensor<int64> indices, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        Vector<int64>? outputShape = null)
        where T : FloatLike
    {
        int spatialRank = RequireSpatialRank(kernelSize);
        long[] strides = Convolution.BroadcastAxes(stride ?? kernelSize, 1L, spatialRank, nameof(stride));
        long[] pads = Convolution.ResolvePads(padding, spatialRank);

        return (Tensor<T>)OnnxOp.MaxUnpool(values, indices,
            kernelShape: kernelSize,
            pads: pads,
            strides: strides,
            outputShape: outputShape);
    }

    /// <summary>1-D <see cref="MaxUnpool{T}(Tensor{T}, Tensor{int64}, long[], long[], long[], Vector{int64})"/>. <paramref name="kernelSize"/> must have length 1.</summary>
    public static Tensor<T> MaxUnpool1d<T>(
        Tensor<T> values, Tensor<int64> indices, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        Vector<int64>? outputShape = null)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 1, nameof(MaxUnpool1d));
        return MaxUnpool(values, indices, kernelSize, stride, padding, outputShape);
    }

    /// <summary>2-D <see cref="MaxUnpool{T}(Tensor{T}, Tensor{int64}, long[], long[], long[], Vector{int64})"/>. <paramref name="kernelSize"/> must have length 2.</summary>
    public static Tensor<T> MaxUnpool2d<T>(
        Tensor<T> values, Tensor<int64> indices, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        Vector<int64>? outputShape = null)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 2, nameof(MaxUnpool2d));
        return MaxUnpool(values, indices, kernelSize, stride, padding, outputShape);
    }

    /// <summary>3-D <see cref="MaxUnpool{T}(Tensor{T}, Tensor{int64}, long[], long[], long[], Vector{int64})"/>. <paramref name="kernelSize"/> must have length 3.</summary>
    public static Tensor<T> MaxUnpool3d<T>(
        Tensor<T> values, Tensor<int64> indices, long[] kernelSize,
        long[]? stride = null, long[]? padding = null,
        Vector<int64>? outputShape = null)
        where T : FloatLike
    {
        Convolution.AssertRank(kernelSize, 3, nameof(MaxUnpool3d));
        return MaxUnpool(values, indices, kernelSize, stride, padding, outputShape);
    }

    // =====================================================================
    //  Misc
    // =====================================================================

    /// <summary>Flattens dimensions from <paramref name="startAxis"/> onward: <c>[N, d1, d2, ...] → [N, d1*d2*...]</c>.</summary>
    public static Tensor<T> Flatten<T>(Tensor<T> x, long startAxis = 1) where T : IVarType
        => x.Flatten(startAxis);

    // =====================================================================
    //  Helpers
    // =====================================================================

    /// <summary>Spatial rank from a non-empty <paramref name="kernelSize"/> (its length).</summary>
    private static int RequireSpatialRank(long[] kernelSize)
    {
        if (kernelSize is null || kernelSize.Length == 0)
            throw new ArgumentException("kernelSize must be non-empty; its length defines the spatial rank.", nameof(kernelSize));
        return kernelSize.Length;
    }
}
