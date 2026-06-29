using System;
using System.Collections.Generic;
using System.Linq;
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
/// How the implicit border pixels are filled for the generalized
/// <see cref="Convolution"/> helpers. Maps onto Shorokoo's
/// <see cref="PadMode"/> for the non-zero modes, which are composed as a
/// separate <c>Pad</c> op applied to the input before a zero-pad conv.
/// </summary>
/// <remarks>
/// <para>
/// <b>Non-differentiability caveat.</b> Only <see cref="Zeros"/> uses the conv's
/// own (differentiable) implicit padding. Every other mode is realized by an
/// explicit <c>Tensor.Pad</c> step, and the gradient of <c>Pad</c> is
/// <b>constant-mode only</b>: <see cref="Reflect"/>, <see cref="Replicate"/>,
/// <see cref="Circular"/> (and the reflect/edge/wrap pad they map to)
/// <b>throw in autodiff</b> and have no QEE values. These modes are therefore
/// forward / inference-grade only — use them for inference, not for training the
/// pad stage. <see cref="Causal"/> is a zero-pad (constant mode) and so is
/// itself differentiable, but it is <b>1-D only</b>.
/// </para>
/// </remarks>
public enum PaddingMode
{
    /// <summary>Constant 0 border, via the conv's own implicit padding (differentiable).</summary>
    Zeros,
    /// <summary>Mirror the input without repeating the edge pixel (<see cref="PadMode.Reflect"/>); non-differentiable.</summary>
    Reflect,
    /// <summary>Repeat the edge pixel (<see cref="PadMode.Edge"/>); non-differentiable.</summary>
    Replicate,
    /// <summary>Wrap around toroidally (<see cref="PadMode.Wrap"/>); non-differentiable.</summary>
    Circular,
    /// <summary>
    /// Left-pad the single spatial axis with <c>(k-1)*dilation</c> zeros so
    /// <c>out[t]</c> never sees future input (WaveNet-style temporal conv).
    /// <b>1-D only</b> — rejected for higher spatial ranks.
    /// </summary>
    Causal,
}

/// <summary>
/// Generalized N-D convolution and transposed-convolution graph-building helpers
/// over NC… input (channels at axis 1), mirroring the <see cref="Pooling"/>
/// static-helper style. Unlike the per-dim <c>[Module]</c> layers
/// (<see cref="Conv1d"/>/<see cref="Conv2d"/>/<see cref="Conv3d"/> and
/// <see cref="ConvTranspose2d"/>, which keep a scalar-square,
/// hyperparameter-driven signature), these take geometry as plain C# array
/// arguments and expose the full ONNX attribute surface: per-axis
/// kernel/stride/padding/dilation, ONNX-style asymmetric padding, <c>auto_pad</c>,
/// <c>groups</c> (including depthwise), <c>padding_mode</c>, and — for
/// transposed conv — <c>output_padding</c> / <c>output_shape</c>.
/// </summary>
/// <remarks>
/// <para>
/// Geometry here is a plain C# value, not a <c>[Hyper]</c>: convolution geometry
/// is shape-determining (it sizes the weight <c>[outC, inC/groups, k…]</c>) and
/// is therefore baked at concretization regardless, so the <c>[Hyper]</c>
/// machinery buys nothing for it. The <c>inChannels</c> axis of the weight is
/// still read in-graph from <c>x.ShapeTensor()[1]</c>, so these helpers remain
/// lazy in the input channel count exactly like the per-dim modules.
/// </para>
/// <para>
/// <b>padding_mode</b> values other than <see cref="PaddingMode.Zeros"/> are
/// composed from <c>Tensor.Pad</c> and are <b>non-differentiable</b> in the pad
/// stage (reflect/edge/wrap have no autodiff and no QEE values) — see
/// <see cref="PaddingMode"/>. They are forward / inference-grade.
/// </para>
/// </remarks>
public static class Convolution
{
    // -- internal geometry helpers ------------------------------------------

    /// <summary>Broadcasts a length-1 array to <paramref name="spatialRank"/>, or returns it if already that length.</summary>
    internal static long[] BroadcastAxes(long[]? value, long defaultVal, int spatialRank, string name)
    {
        if (value is null || value.Length == 0)
            return Enumerable.Repeat(defaultVal, spatialRank).ToArray();
        if (value.Length == 1)
            return Enumerable.Repeat(value[0], spatialRank).ToArray();
        if (value.Length != spatialRank)
            throw new ArgumentException(
                $"{name} must have length 1 or {spatialRank} (spatialRank), but had {value.Length}.", name);
        return value;
    }

    /// <summary>
    /// Resolves <paramref name="padding"/> into an ONNX begin..end pads array of
    /// length <c>2*spatialRank</c>: a length-<c>spatialRank</c> input is treated
    /// as symmetric (begin == end per axis); a length-<c>2*spatialRank</c> input
    /// is used verbatim as <c>[begin₁…beginₙ, end₁…endₙ]</c>.
    /// </summary>
    internal static long[] ResolvePads(long[]? padding, int spatialRank)
    {
        if (padding is null || padding.Length == 0)
            return new long[2 * spatialRank];
        if (padding.Length == spatialRank)
        {
            var pads = new long[2 * spatialRank];
            for (int i = 0; i < spatialRank; i++)
            {
                pads[i] = padding[i];
                pads[spatialRank + i] = padding[i];
            }
            return pads;
        }
        if (padding.Length == 2 * spatialRank)
            return padding;
        throw new ArgumentException(
            $"padding must have length {spatialRank} (symmetric) or {2 * spatialRank} (ONNX begin..end), but had {padding.Length}.",
            nameof(padding));
    }

    /// <summary>Maps a non-causal <see cref="PaddingMode"/> onto the core <see cref="PadMode"/>.</summary>
    private static PadMode ToPadMode(PaddingMode mode) => mode switch
    {
        PaddingMode.Reflect => PadMode.Reflect,
        PaddingMode.Replicate => PadMode.Edge,
        PaddingMode.Circular => PadMode.Wrap,
        _ => throw new ArgumentException($"{mode} has no direct PadMode mapping.", nameof(mode)),
    };

    /// <summary>
    /// Builds the forward-conv weight shape <c>[outChannels, inChannels/groups, k…]</c>
    /// where the <c>inChannels/groups</c> axis is an in-graph scalar.
    /// </summary>
    private static Vector<int64> ConvWeightShape(long outChannels, Scalar<int64> inChannelsPerGroup, long[] kernelSize)
    {
        var dims = new List<VectorExpressionHelper<int64>> { Scalar(outChannels), inChannelsPerGroup };
        foreach (var k in kernelSize)
            dims.Add(Scalar(k));
        return TensorCollectionBuilder.CreateVector<int64>(dims.ToArray());
    }

    /// <summary>
    /// Builds the transposed-conv weight shape <c>[inChannels, outChannels/groups, k…]</c>
    /// where the in-channels axis is the in-graph <paramref name="inChannels"/> scalar.
    /// </summary>
    private static Vector<int64> ConvTransposeWeightShape(Scalar<int64> inChannels, long outChannels, long groups, long[] kernelSize)
    {
        var dims = new List<VectorExpressionHelper<int64>> { inChannels, Scalar(outChannels / groups) };
        foreach (var k in kernelSize)
            dims.Add(Scalar(k));
        return TensorCollectionBuilder.CreateVector<int64>(dims.ToArray());
    }

    // -- forward convolution -------------------------------------------------

    /// <summary>
    /// N-D convolution (cross-correlation) over NC… input with per-axis geometry.
    /// The spatial rank is inferred as <c>kernelSize.Length</c>.
    /// </summary>
    /// <param name="x">Input <c>[N, inC, d₁…dₙ]</c>.</param>
    /// <param name="outChannels">Number of output channels (weight axis 0).</param>
    /// <param name="kernelSize">Per-axis kernel size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride; length 1 (broadcast) or spatialRank. Default all-1.</param>
    /// <param name="padding">
    /// Either length spatialRank (symmetric per axis) or 2*spatialRank
    /// (ONNX <c>[begin₁…beginₙ, end₁…endₙ]</c>, allowing asymmetric pads).
    /// Default all-0; ignored when <paramref name="autoPad"/> is not
    /// <see cref="AutoPad.NotSet"/>.
    /// </param>
    /// <param name="dilation">Per-axis dilation; length 1 (broadcast) or spatialRank. Default all-1.</param>
    /// <param name="groups">Channel groups; <c>1</c> is dense, <c>inC</c> is depthwise. Weight axis 1 is <c>inC/groups</c>.</param>
    /// <param name="bias">When true, a trainable zero-initialized bias <c>[outChannels]</c>; otherwise an all-zero constant.</param>
    /// <param name="autoPad">ONNX auto-pad mode; <see cref="AutoPad.SameUpper"/> matches TF/PyTorch <c>"same"</c>.</param>
    /// <param name="paddingMode">
    /// Border fill mode. <see cref="PaddingMode.Zeros"/> uses the conv's own
    /// (differentiable) padding; the others compose a non-differentiable
    /// <c>Tensor.Pad</c> step (see <see cref="PaddingMode"/>).
    /// <see cref="PaddingMode.Causal"/> is 1-D only.
    /// </param>
    /// <remarks>
    /// Weight <c>[outChannels, inChannels/groups, k…]</c> is
    /// <see cref="KaimingUniform"/>-initialized (fan-in <c>inC/groups·∏k</c>);
    /// <c>inChannels</c> is read in-graph from <c>x.ShapeTensor()[1]</c>.
    /// </remarks>
    public static Tensor<float32> Conv(
        Tensor<float32> x,
        long outChannels,
        long[] kernelSize,
        long[]? stride = null,
        long[]? padding = null,
        long[]? dilation = null,
        long groups = 1,
        bool bias = true,
        AutoPad autoPad = AutoPad.NotSet,
        PaddingMode paddingMode = PaddingMode.Zeros)
    {
        if (kernelSize is null || kernelSize.Length == 0)
            throw new ArgumentException("kernelSize must be non-empty; its length defines the spatial rank.", nameof(kernelSize));

        int spatialRank = kernelSize.Length;
        long[] strides = BroadcastAxes(stride, 1L, spatialRank, nameof(stride));
        long[] dilations = BroadcastAxes(dilation, 1L, spatialRank, nameof(dilation));

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var w = KaimingUniform.Init(ConvWeightShape(outChannels, inChannels / Scalar(groups), kernelSize));

        var bTrainable = Zeros.Init([Scalar(outChannels)]).Vec();
        Vector<float32> b = bias ? bTrainable : VectorFill(outChannels, 0f);

        if (paddingMode == PaddingMode.Causal)
        {
            if (spatialRank != 1)
                throw new ArgumentException(
                    $"PaddingMode.Causal is 1-D only (spatialRank == 1), but spatialRank was {spatialRank}.", nameof(paddingMode));
            if (autoPad != AutoPad.NotSet)
                throw new ArgumentException("PaddingMode.Causal cannot be combined with auto_pad; it sets the padding itself.", nameof(autoPad));

            // Left-pad (k-1)*dilation zeros on the single spatial axis (axis 2), then a VALID conv.
            long leftPad = (kernelSize[0] - 1) * dilations[0];
            var causalPads = Vector(new long[] { leftPad, 0L }); // [begin, end] for the one padded axis
            var xPad = x.Pad(PadMode.Constant, causalPads, val: 0f, axes: Vector(new long[] { 2L }));

            return NN.Conv(xPad, w, b, AutoPad.NotSet,
                dilations: dilations, group: groups, kernelShape: kernelSize,
                pads: new long[2 * spatialRank], strides: strides);
        }

        if (paddingMode == PaddingMode.Zeros)
        {
            long[]? pads = autoPad == AutoPad.NotSet ? ResolvePads(padding, spatialRank) : null;
            return NN.Conv(x, w, b, autoPad,
                dilations: dilations, group: groups, kernelShape: kernelSize,
                pads: pads, strides: strides);
        }

        // Non-zero padding mode: compose an explicit (non-differentiable) Pad over the
        // spatial axes only, then a zero-pad conv. auto_pad is incompatible with an
        // explicit border, so it must stay NotSet here.
        if (autoPad != AutoPad.NotSet)
            throw new ArgumentException(
                $"padding_mode {paddingMode} composes an explicit Pad and cannot be combined with auto_pad ({autoPad}).", nameof(autoPad));

        long[] resolved = ResolvePads(padding, spatialRank);   // [begin₁…beginₙ, end₁…endₙ]
        long[] spatialAxes = Enumerable.Range(2, spatialRank).Select(i => (long)i).ToArray();
        var padsVec = Vector(resolved);
        var axesVec = Vector(spatialAxes);
        var xPadded = x.Pad(ToPadMode(paddingMode), padsVec, val: null, axes: axesVec);

        return NN.Conv(xPadded, w, b, AutoPad.NotSet,
            dilations: dilations, group: groups, kernelShape: kernelSize,
            pads: new long[2 * spatialRank], strides: strides);
    }

    /// <summary>
    /// Square/cubic convenience overload: one scalar per geometry knob, broadcast
    /// to every spatial axis. The spatial rank is taken from the input's
    /// structural rank (<c>x.Rank - 2</c>), which must be known at graph-build
    /// time; pass the per-axis <c>long[]</c> form (or the per-rank
    /// <see cref="Conv1d"/>/<see cref="Conv2d"/>/<see cref="Conv3d"/> aliases) when
    /// it is not. See the per-axis
    /// <see cref="Conv(Tensor{float32}, long, long[], long[], long[], long[], long, bool, AutoPad, PaddingMode)"/>
    /// for the full semantics.
    /// </summary>
    public static Tensor<float32> Conv(
        Tensor<float32> x,
        long outChannels,
        long kernelSize,
        long stride = 1,
        long padding = 0,
        long dilation = 1,
        long groups = 1,
        bool bias = true,
        AutoPad autoPad = AutoPad.NotSet,
        PaddingMode paddingMode = PaddingMode.Zeros)
    {
        int spatialRank = SpatialRankOf(x);
        return Conv(x, outChannels,
            Repeat(kernelSize, spatialRank),
            Repeat(stride, spatialRank),
            Repeat(padding, spatialRank),
            Repeat(dilation, spatialRank),
            groups, bias, autoPad, paddingMode);
    }

    /// <summary>Spatial rank (<c>structuralRank - 2</c>) of an NC… input, required at build time for scalar broadcasting.</summary>
    internal static int SpatialRankOf<T>(Tensor<T> x) where T : IVarType
    {
        int? rank = x.Rank;
        if (rank is null || rank.Value < 3)
            throw new ArgumentException(
                "The scalar-kernel overload needs the input's spatial rank at graph-build time " +
                "(x.Rank - 2, with rank >= 3). Use the long[] kernelSize overload or a 1d/2d/3d alias instead.",
                nameof(x));
        return rank.Value - 2;
    }

    internal static long[] Repeat(long value, int count) => Enumerable.Repeat(value, count).ToArray();

    // -- forward convolution: per-rank aliases ------------------------------

    /// <summary>1-D convolution over NCL input. <paramref name="kernelSize"/> must have length 1.</summary>
    public static Tensor<float32> Conv1d(
        Tensor<float32> x, long outChannels, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long groups = 1, bool bias = true,
        AutoPad autoPad = AutoPad.NotSet, PaddingMode paddingMode = PaddingMode.Zeros)
    {
        AssertRank(kernelSize, 1, nameof(Conv1d));
        return Conv(x, outChannels, kernelSize, stride, padding, dilation, groups, bias, autoPad, paddingMode);
    }

    /// <summary>2-D convolution over NCHW input. <paramref name="kernelSize"/> must have length 2.</summary>
    public static Tensor<float32> Conv2d(
        Tensor<float32> x, long outChannels, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long groups = 1, bool bias = true,
        AutoPad autoPad = AutoPad.NotSet, PaddingMode paddingMode = PaddingMode.Zeros)
    {
        AssertRank(kernelSize, 2, nameof(Conv2d));
        if (paddingMode == PaddingMode.Causal)
            throw new ArgumentException("PaddingMode.Causal is 1-D only; use Conv1d.", nameof(paddingMode));
        return Conv(x, outChannels, kernelSize, stride, padding, dilation, groups, bias, autoPad, paddingMode);
    }

    /// <summary>3-D convolution over NCDHW input. <paramref name="kernelSize"/> must have length 3.</summary>
    public static Tensor<float32> Conv3d(
        Tensor<float32> x, long outChannels, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? dilation = null,
        long groups = 1, bool bias = true,
        AutoPad autoPad = AutoPad.NotSet, PaddingMode paddingMode = PaddingMode.Zeros)
    {
        AssertRank(kernelSize, 3, nameof(Conv3d));
        if (paddingMode == PaddingMode.Causal)
            throw new ArgumentException("PaddingMode.Causal is 1-D only; use Conv1d.", nameof(paddingMode));
        return Conv(x, outChannels, kernelSize, stride, padding, dilation, groups, bias, autoPad, paddingMode);
    }

    // -- transposed convolution ---------------------------------------------

    /// <summary>
    /// N-D transposed (fractionally-strided) convolution over NC… input with
    /// per-axis geometry, the adjoint of the forward conv's input→output map.
    /// The spatial rank is inferred as <c>kernelSize.Length</c>.
    /// </summary>
    /// <param name="x">Input <c>[N, inC, d₁…dₙ]</c>.</param>
    /// <param name="outChannels">Number of output channels (weight axis 1, scaled by <c>1/groups</c>).</param>
    /// <param name="kernelSize">Per-axis kernel size; its length is the spatial rank.</param>
    /// <param name="stride">Per-axis stride (the upsampling factor); length 1 or spatialRank. Default all-1.</param>
    /// <param name="padding">
    /// Input-side implicit padding cropped from the output; length spatialRank
    /// (symmetric) or 2*spatialRank (ONNX begin..end). Default all-0.
    /// </param>
    /// <param name="outputPadding">
    /// Per-axis high-side disambiguator added to the computed output size when
    /// <c>stride &gt; 1</c> maps several input sizes to the same output. Length 1
    /// or spatialRank. Default all-0. (PyTorch's <c>&lt; max(stride, dilation)</c>
    /// guard is <b>not</b> re-imposed here — ORT validates the geometry.)
    /// </param>
    /// <param name="dilation">Per-axis dilation; length 1 or spatialRank. Default all-1.</param>
    /// <param name="groups">Channel groups; weight axis 1 is <c>outC/groups</c>.</param>
    /// <param name="bias">When true, a trainable zero-initialized bias <c>[outChannels]</c>; otherwise an all-zero constant.</param>
    /// <param name="outputShape">
    /// Names the target spatial output size directly; when given, the op derives
    /// the implied padding and <paramref name="outputPadding"/> is ignored.
    /// </param>
    /// <param name="autoPad">ONNX auto-pad mode.</param>
    /// <remarks>
    /// Weight <c>[inChannels, outChannels/groups, k…]</c> (in/out axes swapped vs
    /// forward conv) is <see cref="KaimingUniform"/>-initialized;
    /// <c>inChannels</c> is read in-graph from <c>x.ShapeTensor()[1]</c>. There is
    /// no <c>padding_mode</c> here — transposed conv is zeros-only (its "padding"
    /// is an output-shape crop, not an input border).
    /// </remarks>
    public static Tensor<float32> ConvTranspose(
        Tensor<float32> x,
        long outChannels,
        long[] kernelSize,
        long[]? stride = null,
        long[]? padding = null,
        long[]? outputPadding = null,
        long[]? dilation = null,
        long groups = 1,
        bool bias = true,
        long[]? outputShape = null,
        AutoPad autoPad = AutoPad.NotSet)
    {
        if (kernelSize is null || kernelSize.Length == 0)
            throw new ArgumentException("kernelSize must be non-empty; its length defines the spatial rank.", nameof(kernelSize));

        int spatialRank = kernelSize.Length;
        long[] strides = BroadcastAxes(stride, 1L, spatialRank, nameof(stride));
        long[] dilations = BroadcastAxes(dilation, 1L, spatialRank, nameof(dilation));
        long[] outPad = BroadcastAxes(outputPadding, 0L, spatialRank, nameof(outputPadding));
        long[]? pads = autoPad == AutoPad.NotSet ? ResolvePads(padding, spatialRank) : null;

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var w = KaimingUniform.Init(ConvTransposeWeightShape(inChannels, outChannels, groups, kernelSize));

        var bTrainable = Zeros.Init([Scalar(outChannels)]).Vec();
        Vector<float32> b = bias ? bTrainable : VectorFill(outChannels, 0f);

        return NN.ConvTranspose(x, w, b, autoPad,
            dilations: dilations, group: groups, kernelShape: kernelSize,
            outputPadding: outPad, outputShape: outputShape, pads: pads, strides: strides);
    }

    /// <summary>
    /// Square/cubic convenience overload for transposed convolution: one scalar
    /// per geometry knob, broadcast to every spatial axis (rank from
    /// <c>x.Rank - 2</c>, which must be known at graph-build time). See the
    /// per-axis
    /// <see cref="ConvTranspose(Tensor{float32}, long, long[], long[], long[], long[], long[], long, bool, long[], AutoPad)"/>
    /// for the full semantics.
    /// </summary>
    public static Tensor<float32> ConvTranspose(
        Tensor<float32> x,
        long outChannels,
        long kernelSize,
        long stride = 1,
        long padding = 0,
        long outputPadding = 0,
        long dilation = 1,
        long groups = 1,
        bool bias = true,
        long[]? outputShape = null,
        AutoPad autoPad = AutoPad.NotSet)
    {
        int spatialRank = SpatialRankOf(x);
        return ConvTranspose(x, outChannels,
            Repeat(kernelSize, spatialRank),
            Repeat(stride, spatialRank),
            Repeat(padding, spatialRank),
            Repeat(outputPadding, spatialRank),
            Repeat(dilation, spatialRank),
            groups, bias, outputShape, autoPad);
    }

    // -- transposed convolution: per-rank aliases ---------------------------

    /// <summary>1-D transposed convolution over NCL input. <paramref name="kernelSize"/> must have length 1.</summary>
    public static Tensor<float32> ConvTranspose1d(
        Tensor<float32> x, long outChannels, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? outputPadding = null,
        long[]? dilation = null, long groups = 1, bool bias = true,
        long[]? outputShape = null, AutoPad autoPad = AutoPad.NotSet)
    {
        AssertRank(kernelSize, 1, nameof(ConvTranspose1d));
        return ConvTranspose(x, outChannels, kernelSize, stride, padding, outputPadding, dilation, groups, bias, outputShape, autoPad);
    }

    /// <summary>2-D transposed convolution over NCHW input. <paramref name="kernelSize"/> must have length 2.</summary>
    public static Tensor<float32> ConvTranspose2d(
        Tensor<float32> x, long outChannels, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? outputPadding = null,
        long[]? dilation = null, long groups = 1, bool bias = true,
        long[]? outputShape = null, AutoPad autoPad = AutoPad.NotSet)
    {
        AssertRank(kernelSize, 2, nameof(ConvTranspose2d));
        return ConvTranspose(x, outChannels, kernelSize, stride, padding, outputPadding, dilation, groups, bias, outputShape, autoPad);
    }

    /// <summary>3-D transposed convolution over NCDHW input. <paramref name="kernelSize"/> must have length 3.</summary>
    public static Tensor<float32> ConvTranspose3d(
        Tensor<float32> x, long outChannels, long[] kernelSize,
        long[]? stride = null, long[]? padding = null, long[]? outputPadding = null,
        long[]? dilation = null, long groups = 1, bool bias = true,
        long[]? outputShape = null, AutoPad autoPad = AutoPad.NotSet)
    {
        AssertRank(kernelSize, 3, nameof(ConvTranspose3d));
        return ConvTranspose(x, outChannels, kernelSize, stride, padding, outputPadding, dilation, groups, bias, outputShape, autoPad);
    }

    internal static void AssertRank(long[] kernelSize, int expected, string alias)
    {
        if (kernelSize is null || kernelSize.Length != expected)
            throw new ArgumentException(
                $"{alias} requires kernelSize.Length == {expected}, but it was {(kernelSize?.Length ?? 0)}.", nameof(kernelSize));
    }
}
