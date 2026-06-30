using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Self-checking [Module]s for the baseline NN library (Shorokoo.Modules
// Initializers/Layers/Losses). Each module returns a single Scalar<bit> so
// AutoTest.AdvancedTestGraph treats it as a self-checking graph (the bit must
// be true), keeping the xUnit tests one-liners.
//
// Reference-vs-layer value checks exploit that the library initializers are
// seeded/deterministic: two trainable params of the same shape created by the
// same initializer class materialize to identical values, so a hand-built
// reference using the same initializer reproduces the layer's weights exactly.
//
// BatchNorm2d carries Globals.StateUpdate links (STATE_UPDATE_LINK is not an
// executable ORT op in the plain inference pipeline), so it is exercised via
// TrainingRig-based tests (NNLibraryTrainingCoverageTests) instead of AutoTest.
// ---------------------------------------------------------------------------

/// <summary>Linear must equal the manual flatten + MatMul against an identically initialized weight (bias is zero-init).</summary>
[Module]
public partial class NNLinearMatchesManualMatMul
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outFeatures = Scalar(4L);
        var y = Linear.Call(outFeatures, Scalar(true), x);

        var inFeatures = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();
        var wRef = KaimingUniform.Init([outFeatures, inFeatures]);
        var yRef = x.Reshape([x.DimTensor(0), inFeatures]).MatMul(wRef.Transpose(1L, 0L));

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

// ---------------------------------------------------------------------------
// Bilinear (src/Shorokoo.Modules/Layers/Bilinear.cs) — PyTorch nn.Bilinear:
// y[..., k] = Σ_{i,j} x1[..., i]·A[k,i,j]·x2[..., j] (+ b[k]), via a single
// explicit-label einsum "ni,kij,nj->nk". Design §7. The closed-form modules
// re-materialize the SAME seeded RecurrentUniform.Init weight A [out,in1,in2]
// and bias b [out] (both U(±1/√in1), seeded ⇒ identical tensors) and recompute
// the bilinear form with an INDEPENDENT op path — a manual Σ_{i,j} via broadcast
// Mul + ReduceSum, NOT a second einsum — so a bug in the einsum equation can't
// hide behind itself. Relative-L1 match, the NNLinearMatchesManualMatMul idiom.
// ---------------------------------------------------------------------------

/// <summary>§7.1 Closed-form (load-bearing): Bilinear(useBias:true) must equal the bilinear form
/// recomputed by hand from the SAME seeded weight/bias via an independent broadcast Mul + ReduceSum
/// (x1[N,1,in1,1]·A[1,out,in1,in2]·x2[N,1,1,in2] summed over i,j, + b). Relative-L1, NNLinear idiom.
/// The bias being RecurrentUniform (not Zeros) is load-bearing — a Zeros bias would fail the check.</summary>
[Module]
public partial class NNBilinearMatchesManualForm
{
    public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2)   // x1 [N,in1], x2 [N,in2]
    {
        var in1 = Scalar(3L);
        var in2 = Scalar(4L);
        var outF = Scalar(2L);
        var y = Bilinear.Call(in1, in2, outF, Scalar(true), x1, x2);           // [N, out]

        // Reference: SAME seeded weight/bias, contracted by hand (broadcast Mul + ReduceSum).
        var a = RecurrentUniform.Init([outF, in1, in2], in1);                  // [out, in1, in2]
        var b = RecurrentUniform.Init([outF], in1).Vec();                      // [out]
        var x1e = x1.Unsqueeze(1L).Unsqueeze(3L);                              // [N,1,in1,1]
        var x2e = x2.Unsqueeze(1L).Unsqueeze(1L);                              // [N,1,1,in2]
        var ae = a.Unsqueeze(0L);                                              // [1,out,in1,in2]
        var prod = x1e * ae * x2e;                                             // [N,out,in1,in2]
        Vector<int64> ijAxes = [Scalar(2L), Scalar(3L)];
        var yRef = prod.Reduce(ReduceKind.Sum, ijAxes, keepDims: false) + b;   // [N, out]

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7.2 useBias gating: Bilinear(useBias:false) must equal the manual bilinear form WITHOUT
/// the bias, AND the (true − false) difference must equal exactly the seeded bias b (broadcast over the
/// batch). Guards the IfElse branch selection — a Zeros bias or a wrong branch would fail one of the two.</summary>
[Module]
public partial class NNBilinearUseBiasFalseAndDiff
{
    public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2)   // x1 [N,in1], x2 [N,in2]
    {
        var in1 = Scalar(3L);
        var in2 = Scalar(4L);
        var outF = Scalar(2L);
        var yTrue = Bilinear.Call(in1, in2, outF, Scalar(true), x1, x2);       // [N, out]
        var yFalse = Bilinear.Call(in1, in2, outF, Scalar(false), x1, x2);     // [N, out]

        // SAME seeded weight/bias, manual contraction (no bias) as the reference for the false case.
        var a = RecurrentUniform.Init([outF, in1, in2], in1);
        var b = RecurrentUniform.Init([outF], in1).Vec();                      // [out]
        var x1e = x1.Unsqueeze(1L).Unsqueeze(3L);
        var x2e = x2.Unsqueeze(1L).Unsqueeze(1L);
        var prod = x1e * a.Unsqueeze(0L) * x2e;                                // [N,out,in1,in2]
        Vector<int64> ijAxes = [Scalar(2L), Scalar(3L)];
        var yRefNoBias = prod.Reduce(ReduceKind.Sum, ijAxes, keepDims: false); // [N, out] (no bias)

        // (a) useBias:false equals the no-bias reference.
        var falsePen = (yFalse - yRefNoBias).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // (b) the true − false difference is exactly the bias b (broadcast over the batch).
        var diffMinusBias = (yTrue - yFalse - b).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var scale = Scalar(1f)
            + yRefNoBias.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
            + b.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return falsePen + diffMinusBias < Scalar(1e-3f) * scale;
    }
}

/// <summary>§7.3 Batch broadcasting: with x1 [B,T,in1], x2 [B,T,in2] the output shape is [B,T,out]
/// (asserted via ShapeTensor == [2,2,2]) AND equals the per-row manual bilinear reference (same
/// broadcast Mul + ReduceSum, now with two leading dims). Exercises the flatten→einsum→restore path.</summary>
[Module]
public partial class NNBilinearBatchBroadcasts
{
    public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2)   // x1 [B,T,in1], x2 [B,T,in2]
    {
        var in1 = Scalar(3L);
        var in2 = Scalar(4L);
        var outF = Scalar(2L);
        var y = Bilinear.Call(in1, in2, outF, Scalar(true), x1, x2);           // [B, T, out]

        // Shape assertion: y.shape == [2, 2, 2].
        var shape = y.ShapeTensor();
        Scalar<int64> d0 = shape[0];
        Scalar<int64> d1 = shape[1];
        Scalar<int64> d2 = shape[2];
        var shapeOk = (d0 == Scalar(2L)) & (d1 == Scalar(2L)) & (d2 == Scalar(2L));

        // Per-row reference: SAME seeded weight/bias, broadcast over the two leading dims [B,T].
        var a = RecurrentUniform.Init([outF, in1, in2], in1);
        var b = RecurrentUniform.Init([outF], in1).Vec();                      // [out]
        var x1e = x1.Unsqueeze(2L).Unsqueeze(4L);                             // [B,T,1,in1,1]
        var x2e = x2.Unsqueeze(2L).Unsqueeze(2L);                             // [B,T,1,1,in2]
        var ae = a.Unsqueeze(0L).Unsqueeze(0L);                               // [1,1,out,in1,in2]
        var prod = x1e * ae * x2e;                                            // [B,T,out,in1,in2]
        Vector<int64> ijAxes = [Scalar(3L), Scalar(4L)];
        var yRef = prod.Reduce(ReduceKind.Sum, ijAxes, keepDims: false) + b;   // [B, T, out]

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var valueOk = diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
        return shapeOk & valueOk;
    }
}

/// <summary>§7.4 Rig train-step smoke model: wraps Bilinear on two fixed inputs (x1 [N,in1], x2 [N,in2])
/// and reduces to a per-row scalar so the rig can take an L2 loss. The rank-3 A weight [out,in1,in2] is
/// the first Einsum-autodiff exercise at the module level — the driving [Fact] asserts A moved.</summary>
[Module]
public partial class BilinearRigModel
{
    public static Tensor<float32> Inline(Tensor<float32> x1, Tensor<float32> x2)
    {
        var y = Bilinear.Model(Scalar(3L), Scalar(4L), Scalar(2L), Scalar(true)).Call(x1, x2);   // [N, out]
        Vector<int64> outAxis = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, outAxis, keepDims: false);   // [N]
    }
}

/// <summary>Conv2d (dynamic SHRK_CONV geometry) must equal the static-attribute NN.Conv with identical geometry and weights.</summary>
[Module]
public partial class NNConv2dMatchesStaticConv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = Scalar(3L);
        var y = Conv2d.Call(outChannels, Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true), x);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([outChannels, inChannels, Scalar(3L), Scalar(3L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: [1L, 1L, 1L, 1L], strides: [2L, 2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>Conv1d (rank-3 dynamic geometry) must equal the static-attribute NN.Conv with one spatial dim.</summary>
[Module]
public partial class NNConv1dMatchesStaticConv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = Scalar(3L);
        var y = Conv1d.Call(outChannels, Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true), x);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([outChannels, inChannels, Scalar(3L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L], group: 1L, kernelShape: [3L], pads: [1L, 1L], strides: [2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>ConvTranspose2d must equal the static NN.ConvTranspose with default geometry and identical weights.</summary>
[Module]
public partial class NNConvTranspose2dMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = Scalar(3L);
        var y = ConvTranspose2d.Call(outChannels, Scalar(2L), Scalar(true), x);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([inChannels, outChannels, Scalar(2L), Scalar(2L)]);
        var yRef = NN.ConvTranspose(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: null, group: 1L, kernelShape: null,
            outputPadding: null, outputShape: null, pads: null, strides: null);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

// ---------------------------------------------------------------------------
// Generalized Convolution helper coverage (Convolution.Conv / ConvTranspose,
// src/Shorokoo.Modules/Layers/Convolution.cs) — design §7 groups 1–9. Each
// self-checking [Module] returns a Scalar<bit> that AutoTest.AdvancedTestGraph
// requires to be true: it computes y = Convolution.Conv(x, …) and a reference
// yRef = NN.Conv(x, KaimingUniform.Init(SAME_WEIGHT_SHAPE), zeroBias, …same
// geometry…) and asserts a relative-L1 match. The weight shapes MUST coincide
// exactly ([outC, inC/groups, k…] forward; [inC, outC/groups, k…] transpose) so
// the seeded KaimingUniform inits materialize identically (the same idiom as
// NNConv2dMatchesStaticConv above). Conv has no QEE values, so value
// correctness comes from the ORT backend inside AdvancedTestGraph.
// ---------------------------------------------------------------------------

/// <summary>§7-1 Non-square kernel: Convolution.Conv(kernelSize:[3,5]) must equal the
/// static NN.Conv(kernelShape:[3,5], pads:[1,2,1,2]) with identical weights.</summary>
[Module]
public partial class ConvNonSquareKernelMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 3L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 5L], padding: [1L, 2L]);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(3L), Scalar(5L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 1L, kernelShape: [3L, 5L], pads: [1L, 2L, 1L, 2L], strides: [1L, 1L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-2 Per-axis stride &amp; dilation: stride:[1,2], dilation:[2,1] must equal the
/// matching static NN.Conv arrays (with explicit pads so output is well-formed).</summary>
[Module]
public partial class ConvPerAxisStrideDilationMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L],
            stride: [1L, 2L], padding: [1L, 1L], dilation: [2L, 1L]);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(3L), Scalar(3L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [2L, 1L], group: 1L, kernelShape: [3L, 3L], pads: [1L, 1L, 1L, 1L], strides: [1L, 2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-3 Asymmetric pad: padding:[1,2,0,1] (ONNX begin..end) must equal the
/// static NN.Conv with the same pads verbatim.</summary>
[Module]
public partial class ConvAsymmetricPadMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 2L, 0L, 1L]);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(3L), Scalar(3L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: [1L, 2L, 0L, 1L], strides: [1L, 1L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-4 auto_pad SAME_UPPER and VALID: both must equal the static NN.Conv with
/// the same autoPad and pads:null (forward value only — SAME has no Conv backward).</summary>
[Module]
public partial class ConvAutoPadMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(3L), Scalar(3L)]);

        var ySame = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], autoPad: AutoPad.SameUpper);
        var ySameRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.SameUpper,
            dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: null, strides: [1L, 1L]);

        var yValid = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], autoPad: AutoPad.Valid);
        var yValidRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.Valid,
            dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: null, strides: [1L, 1L]);

        var samePen = (ySame - ySameRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var validPen = (yValid - yValidRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f)
            + ySameRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
            + yValidRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return samePen + validPen < Scalar(1e-3f) * scale;
    }
}

/// <summary>§7-5 Groups / depthwise: groups:inC (depthwise, here inC=4 → outC=4) and a mid
/// groups:2 must each equal the static NN.Conv group: reference. The reference weight second
/// axis is inC/groups (1 for depthwise, 2 for groups:2).</summary>
[Module]
public partial class ConvGroupsMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, 4, H, W]
    {
        Scalar<int64> inChannels = x.ShapeTensor()[1];

        // Depthwise: groups == inC (4), outChannels == inC, weight second axis inC/groups == 1.
        var outDw = 4L;
        var yDw = Convolution.Conv(x, outDw, kernelSize: [3L, 3L], padding: [1L, 1L], groups: 4L);
        var wDw = KaimingUniform.Init([Scalar(outDw), Scalar(1L), Scalar(3L), Scalar(3L)]);
        var yDwRef = NN.Conv(x, wDw, VectorFill(outDw, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 4L, kernelShape: [3L, 3L], pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);

        // Mid groups:2 — weight second axis inC/groups == 4/2 == 2.
        var outG2 = 4L;
        var yG2 = Convolution.Conv(x, outG2, kernelSize: [3L, 3L], padding: [1L, 1L], groups: 2L);
        var wG2 = KaimingUniform.Init([Scalar(outG2), Scalar(2L), Scalar(3L), Scalar(3L)]);
        var yG2Ref = NN.Conv(x, wG2, VectorFill(outG2, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 2L, kernelShape: [3L, 3L], pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);

        var dwPen = (yDw - yDwRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var g2Pen = (yG2 - yG2Ref).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f)
            + yDwRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
            + yG2Ref.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return dwPen + g2Pen < Scalar(1e-3f) * scale;
    }
}

/// <summary>§7-6 padding_mode Reflect/Replicate/Circular: each must equal a hand-built
/// x.Pad(&lt;PadMode&gt;, pads, axes:spatial) then NN.Conv(pads:0) reference. Forward only —
/// reflect/edge/wrap Pad is non-differentiable and has no QEE values, so QEE / CS-roundtrip
/// are disabled for this check in the driving [Fact].</summary>
[Module]
public partial class ConvPaddingModeMatchesHandPad
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var w = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(3L), Scalar(3L)]);
        var b = VectorFill(outChannels, 0f);

        Vector<int64> pads = [Scalar(1L), Scalar(1L), Scalar(1L), Scalar(1L)];   // [begin_h, begin_w, end_h, end_w]
        Vector<int64> spatialAxes = [Scalar(2L), Scalar(3L)];
        long[] zeroPads = { 0L, 0L, 0L, 0L };

        // Relative-L1 penalty for one mode: (diff - tol*scale), negative when matching.
        Scalar<float32> ModePen(PaddingMode mode, PadMode padMode)
        {
            var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L], paddingMode: mode);
            var xPad = x.Pad(padMode, pads, val: null, axes: spatialAxes);
            var yRef = NN.Conv(xPad, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: zeroPads, strides: [1L, 1L]);
            var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return (diff - Scalar(1e-3f) * scale).Relu();   // 0 when matching, positive otherwise
        }

        var pen = ModePen(PaddingMode.Reflect, PadMode.Reflect)
                + ModePen(PaddingMode.Replicate, PadMode.Edge)
                + ModePen(PaddingMode.Circular, PadMode.Wrap);
        return pen < Scalar(1e-6f);
    }
}

/// <summary>§7-6 Causal (1D): Convolution.Conv1d(paddingMode:Causal) must equal a hand-built
/// left-Pad((k-1)*dilation) + VALID conv. Forward only (the Pad here is constant-mode, so it is
/// differentiable, but kept inference-grade for symmetry with the other padding-mode check).</summary>
[Module]
public partial class ConvCausalMatchesLeftPadValid
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, L]
    {
        var outChannels = 2L;
        var k = 3L;
        var dilation = 2L;
        var y = Convolution.Conv1d(x, outChannels, kernelSize: [k],
            dilation: [dilation], paddingMode: PaddingMode.Causal);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var w = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(k)]);
        var b = VectorFill(outChannels, 0f);

        long leftPad = (k - 1) * dilation;   // 4
        var xPad = x.Pad(PadMode.Constant, Vector(new long[] { leftPad, 0L }),
            val: Scalar(0f), axes: Vector(new long[] { 2L }));
        var yRef = NN.Conv(xPad, w, b, AutoPad.Valid,
            dilations: [dilation], group: 1L, kernelShape: [k], pads: null, strides: [1L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-7 ConvTranspose output_padding: ConvTranspose(kernelSize:[2,2], stride:[2,2],
/// outputPadding:[1,1]) must equal the static NN.ConvTranspose with identical weights/geometry.</summary>
[Module]
public partial class ConvTransposeOutputPaddingMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 3L;
        var y = Convolution.ConvTranspose(x, outChannels, kernelSize: [2L, 2L],
            stride: [2L, 2L], outputPadding: [1L, 1L]);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([inChannels, Scalar(outChannels), Scalar(2L), Scalar(2L)]);
        var yRef = NN.ConvTranspose(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 1L, kernelShape: [2L, 2L],
            outputPadding: [1L, 1L], outputShape: null, pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-7 ConvTranspose output_shape: ConvTranspose(stride:[2,2], outputShape:[…]) must
/// equal the static NN.ConvTranspose given the same outputShape.</summary>
[Module]
public partial class ConvTransposeOutputShapeMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, C, 3, 3] → stride 2 base out 6, target 7
    {
        var outChannels = 2L;
        long[] outShape = { 7L, 7L };
        var y = Convolution.ConvTranspose(x, outChannels, kernelSize: [2L, 2L],
            stride: [2L, 2L], outputShape: outShape);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([inChannels, Scalar(outChannels), Scalar(2L), Scalar(2L)]);
        var yRef = NN.ConvTranspose(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 1L, kernelShape: [2L, 2L],
            outputPadding: [0L, 0L], outputShape: outShape, pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-7 ConvTranspose1d rank alias smoke: ConvTranspose1d must equal the static
/// NN.ConvTranspose at rank 3 with one spatial dim.</summary>
[Module]
public partial class ConvTranspose1dMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, L]
    {
        var outChannels = 2L;
        var y = Convolution.ConvTranspose1d(x, outChannels, kernelSize: [2L], stride: [2L]);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([inChannels, Scalar(outChannels), Scalar(2L)]);
        var yRef = NN.ConvTranspose(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L], group: 1L, kernelShape: [2L],
            outputPadding: [0L], outputShape: null, pads: [0L, 0L], strides: [2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-7 ConvTranspose3d rank alias smoke: ConvTranspose3d must equal the static
/// NN.ConvTranspose at rank 5 with three spatial dims.</summary>
[Module]
public partial class ConvTranspose3dMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, D, H, W]
    {
        var outChannels = 2L;
        var y = Convolution.ConvTranspose3d(x, outChannels, kernelSize: [2L, 2L, 2L], stride: [2L, 2L, 2L]);

        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([inChannels, Scalar(outChannels), Scalar(2L), Scalar(2L), Scalar(2L)]);
        var yRef = NN.ConvTranspose(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L, 1L], group: 1L, kernelShape: [2L, 2L, 2L],
            outputPadding: [0L, 0L, 0L], outputShape: null, pads: [0L, 0L, 0L, 0L, 0L, 0L], strides: [2L, 2L, 2L]);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-8 Alias equivalence (forward + transpose): Conv2d(x, outC, [3,3]) == Conv(x, outC, [3,3]),
/// and ConvTranspose2d(x, outC, [2,2]) == the generic ConvTranspose. Bit-for-bit (same seeded weights).
/// The scalar→per-axis broadcast equivalence is pinned separately in TestConvScalarBroadcastEquivalence
/// (the scalar overload needs a build-time-known rank, which a symbolic [Module] input lacks).</summary>
[Module]
public partial class ConvAliasAndScalarEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;

        var perAxis = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L]);
        var alias2d = Convolution.Conv2d(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L]);

        var ctPerAxis = Convolution.ConvTranspose(x, outChannels, kernelSize: [2L, 2L], stride: [2L, 2L]);
        var ctAlias = Convolution.ConvTranspose2d(x, outChannels, kernelSize: [2L, 2L], stride: [2L, 2L]);

        var aliasPen = (perAxis - alias2d).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var ctPen = (ctPerAxis - ctAlias).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return aliasPen + ctPen < Scalar(1e-4f);
    }
}

/// <summary>§7-8 Scalar-overload broadcast: the scalar Conv(c, outC, 3, padding:1) must equal the
/// per-axis Conv(c, outC, [3,3], padding:[1,1]) — pinning the scalar→per-axis broadcast. The scalar
/// overload reads c.Rank()-2 at build time, which is statically known only for a concretely-shaped
/// tensor (a symbolic graph input has no build-time rank), so the conv input is a literal constant
/// [1,1,3,3] tensor built in-module. The runtime input <paramref name="x"/> only gates the result so
/// AutoTest has a graph input to drive.</summary>
[Module]
public partial class ConvScalarBroadcastMatchesPerAxis
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // Literal constant input → build-time-known rank 4, so the scalar overload can read Rank()-2.
        var c = Tensor(new long[] { 1L, 1L, 3L, 3L }, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f);
        var outChannels = 2L;

        var scalar = Convolution.Conv(c, outChannels, kernelSize: 3L, padding: 1L);
        var perAxis = Convolution.Conv(c, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L]);

        var diff = (scalar - perAxis).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // Fold a trivial dependence on x so AutoTest has a runtime input to feed.
        var xTouch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff + xTouch.Abs() < Scalar(1e-4f);
    }
}

/// <summary>§7-9 bias on/off: a bias:false conv must equal a zero-bias reference; and a bias:true
/// conv (bias zero at init) must ALSO equal the same zero reference — proving the init bias is zero,
/// so on/off agree at init. Both compared against the same KaimingUniform-weight, zero-bias NN.Conv.</summary>
[Module]
public partial class ConvBiasOnOffMatchesZeroBias
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        Scalar<int64> inChannels = x.ShapeTensor()[1];
        var wRef = KaimingUniform.Init([Scalar(outChannels), inChannels, Scalar(3L), Scalar(3L)]);
        var yRef = NN.Conv(x, wRef, VectorFill(outChannels, 0f), AutoPad.NotSet,
            dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: [1L, 1L, 1L, 1L], strides: [1L, 1L]);

        var yNoBias = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L], bias: false);
        var yBias = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L], bias: true);

        var noBiasPen = (yNoBias - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var biasPen = (yBias - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return noBiasPen + biasPen < Scalar(1e-3f) * scale;
    }
}

/// <summary>§7-10 Trainability smoke model: a tiny groups:1, explicit-pad, Zeros-mode
/// Convolution.Conv → ReLU → GlobalAvgPool → [N, 2] logits, for a TrainingRig FromScratch /
/// TrainStep (the supported differentiable corner: 2-D weight grad, explicit pads, zeros mode).</summary>
[Module]
public partial class ConvGeneralizedTrainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var x = Convolution.Conv(input, outChannels: 2L, kernelSize: [3L, 3L], padding: [1L, 1L]);
        x = x.Relu();
        x = Pooling.GlobalAvgPool2d(x);
        return x.Reshape([input.DimTensor(0), Scalar(2L)]);
    }
}

/// <summary>LayerNorm over the last dim must produce ~zero mean / ~unit variance rows (gamma=1, beta=0 at init).</summary>
[Module]
public partial class NNLayerNormNormalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = LayerNorm.Call(Scalar(1L), Scalar(1e-5f), x);
        Vector<int64> lastAxis = [Scalar(1L)];
        var mean = y.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);
        var variance = (y * y).Reduce(ReduceKind.Mean, lastAxis, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

/// <summary>GroupNorm (NCHW, 2 groups) must produce ~zero mean / ~unit variance per (sample, group).</summary>
[Module]
public partial class NNGroupNormNormalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(true), Scalar(1e-5f), x);
        var n = x.DimTensor(0);
        var yg = y.Reshape([n, Scalar(2L), Scalar(-1L)]);
        Vector<int64> groupAxis = [Scalar(2L)];
        var mean = yg.Reduce(ReduceKind.Mean, groupAxis, keepDims: false);
        var variance = (yg * yg).Reduce(ReduceKind.Mean, groupAxis, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

/// <summary>InstanceNorm2d must produce ~zero mean / ~unit variance per (sample, channel) spatial slice.</summary>
[Module]
public partial class NNInstanceNorm2dNormalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = InstanceNorm2d.Call(Scalar(1e-5f), x);
        Vector<int64> spatialAxes = [Scalar(2L), Scalar(3L)];
        var mean = y.Reduce(ReduceKind.Mean, spatialAxes, keepDims: false);
        var variance = (y * y).Reduce(ReduceKind.Mean, spatialAxes, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

// ---------------------------------------------------------------------------
// Generalized rank-generic InstanceNorm + GroupNorm coverage (design §7).
// Both are STATE-FREE (no Globals.StateUpdate), so they run on the plain
// inference pipeline and are exercised via AutoTest.AdvancedTestGraph here.
// Each self-checking [Module] mirrors the existing NNGroupNormNormalizes /
// NNInstanceNorm2dNormalizes idiom: reduce the output over the per-(sample,
// channel)/(sample, group) normalization region and assert ~zero mean /
// ~unit variance (meanPen + varPen < 1e-2), OR compare against a hand-built
// reference / a sibling norm with a relative-L1 penalty. Affine is off in the
// value/equivalence checks so the bare normalization is what is measured; the
// affine PARAM-COUNT discrimination needs a rig (NNLibraryTrainingCoverageTests,
// mirroring TestBatchNormAffineOnOff) and lives in the *Model modules below.
// ---------------------------------------------------------------------------

// --- §7-1: per-region zero-mean/unit-var, InstanceNorm at ranks 3/4/5 ---
// One module per rank (the reduction axis set [2..rank) is spelled per rank,
// like NNInstanceNorm2dNormalizes spells [2,3]). affine:false so γ/β do not
// perturb the bare x̂.

/// <summary>§7-1 InstanceNorm rank-3 [N,C,L]: ~zero mean / ~unit variance per (sample, channel) over axis 2.</summary>
[Module]
public partial class NNInstanceNormRank3Normalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = InstanceNorm.Call(Scalar(false), Scalar(1e-5f), x);
        Vector<int64> spatialAxes = [Scalar(2L)];
        var mean = y.Reduce(ReduceKind.Mean, spatialAxes, keepDims: false);
        var variance = (y * y).Reduce(ReduceKind.Mean, spatialAxes, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

/// <summary>§7-1 InstanceNorm rank-4 [N,C,H,W]: ~zero mean / ~unit variance per (sample, channel) over axes 2,3.</summary>
[Module]
public partial class NNInstanceNormRank4Normalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = InstanceNorm.Call(Scalar(false), Scalar(1e-5f), x);
        Vector<int64> spatialAxes = [Scalar(2L), Scalar(3L)];
        var mean = y.Reduce(ReduceKind.Mean, spatialAxes, keepDims: false);
        var variance = (y * y).Reduce(ReduceKind.Mean, spatialAxes, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

/// <summary>§7-1 InstanceNorm rank-5 [N,C,D,H,W]: ~zero mean / ~unit variance per (sample, channel) over axes 2,3,4.</summary>
[Module]
public partial class NNInstanceNormRank5Normalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = InstanceNorm.Call(Scalar(false), Scalar(1e-5f), x);
        Vector<int64> spatialAxes = [Scalar(2L), Scalar(3L), Scalar(4L)];
        var mean = y.Reduce(ReduceKind.Mean, spatialAxes, keepDims: false);
        var variance = (y * y).Reduce(ReduceKind.Mean, spatialAxes, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

// --- §7-1: per-region zero-mean/unit-var, GroupNorm (G=2) at ranks 3/4/5 ---
// Reduce the [N, G, -1] reshape's axis 2 (the whole per-(sample, group)
// region at any rank), exactly as NNGroupNormNormalizes does. affine:false.

/// <summary>§7-1 GroupNorm (G=2) rank-3 [N,C,L]: ~zero mean / ~unit variance per (sample, group).</summary>
[Module]
public partial class NNGroupNormRank3Normalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var n = x.DimTensor(0);
        var yg = y.Reshape([n, Scalar(2L), Scalar(-1L)]);
        Vector<int64> groupAxis = [Scalar(2L)];
        var mean = yg.Reduce(ReduceKind.Mean, groupAxis, keepDims: false);
        var variance = (yg * yg).Reduce(ReduceKind.Mean, groupAxis, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

/// <summary>§7-1 GroupNorm (G=2) rank-4 [N,C,H,W]: ~zero mean / ~unit variance per (sample, group).</summary>
[Module]
public partial class NNGroupNormRank4Normalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var n = x.DimTensor(0);
        var yg = y.Reshape([n, Scalar(2L), Scalar(-1L)]);
        Vector<int64> groupAxis = [Scalar(2L)];
        var mean = yg.Reduce(ReduceKind.Mean, groupAxis, keepDims: false);
        var variance = (yg * yg).Reduce(ReduceKind.Mean, groupAxis, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

/// <summary>§7-1 GroupNorm (G=2) rank-5 [N,C,D,H,W]: ~zero mean / ~unit variance per (sample, group).</summary>
[Module]
public partial class NNGroupNormRank5Normalizes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var n = x.DimTensor(0);
        var yg = y.Reshape([n, Scalar(2L), Scalar(-1L)]);
        Vector<int64> groupAxis = [Scalar(2L)];
        var mean = yg.Reduce(ReduceKind.Mean, groupAxis, keepDims: false);
        var variance = (yg * yg).Reduce(ReduceKind.Mean, groupAxis, keepDims: false) - mean * mean;
        var meanPen = mean.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var varPen = (variance - Scalar(1f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return meanPen + varPen < Scalar(1e-2f);
    }
}

// --- §7-2(a): affine:false output == hand-built manualXHat reference ---
// γ=1/β=0 at init means affine on/off coincide numerically, so the
// discriminating value check is against an independent hand-built x̂ (reduce/
// sqrt over the same region), NOT against the affine:true output. Relative-L1.

/// <summary>§7-2(a) InstanceNorm(affine:false) over a rank-4 input equals the hand-built
/// (x − mean)/sqrt(var + eps) reference (biased variance over axes [2..rank)). Relative-L1 &lt; 1e-5.</summary>
[Module]
public partial class NNInstanceNormAffineFalseMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var eps = Scalar(1e-5f);
        var y = InstanceNorm.Call(Scalar(false), eps, x);

        Vector<int64> spatialAxes = [Scalar(2L), Scalar(3L)];
        var mean = x.Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);
        var diff = x - mean;
        var variance = (diff * diff).Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);
        var manualXHat = diff / (variance + eps).Sqrt();

        var pen = (y - manualXHat).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + manualXHat.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f) * scale;
    }
}

/// <summary>§7-2(a) GroupNorm(G=2, affine:false) over a rank-4 input equals the hand-built
/// per-(sample,group) (x − mean)/sqrt(var + eps) reference via the [N,G,-1] reshape. Relative-L1 &lt; 1e-5.</summary>
[Module]
public partial class NNGroupNormAffineFalseMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var eps = Scalar(1e-5f);
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), eps, x);

        var shape = x.ShapeTensor();
        var n = x.DimTensor(0);
        var xg = x.Reshape([n, Scalar(2L), Scalar(-1L)]);
        Vector<int64> groupAxis = [Scalar(2L)];
        var mean = xg.Reduce(ReduceKind.Mean, groupAxis, keepDims: true);
        var diff = xg - mean;
        var variance = (diff * diff).Reduce(ReduceKind.Mean, groupAxis, keepDims: true);
        var manualXHat = (diff / (variance + eps).Sqrt()).Reshape(shape);

        var pen = (y - manualXHat).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + manualXHat.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f) * scale;
    }
}

// --- §7-3: GroupNorm(G=1) ≡ LayerNorm-over-CHW (rank-4) ---
// normalizedDims = rank-1 = 3 (the last C·H·W block). Both affine off → only
// the normalization is compared. Biased variance uniform ⇒ exact up to float.

/// <summary>§7-3 GroupNorm(G=1, affine:false) ≡ LayerNorm over the last C·H·W (normalizedDims=rank−1=3)
/// on a rank-4 input. Relative-L1 &lt; 1e-3.</summary>
[Module]
public partial class NNGroupNormG1MatchesLayerNorm
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, H, W]
    {
        var eps = Scalar(1e-5f);
        var g1 = GroupNorm.Call(Scalar(1L), Scalar(false), eps, x);
        var ln = LayerNorm.Call(Scalar(3L), eps, x);   // last C·H·W = rank-1 dims

        var pen = (g1 - ln).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + ln.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-3f) * scale;
    }
}

// --- §7-4: GroupNorm(G=C) ≡ InstanceNorm (rank-4 and rank-3) ---
// Each channel is its own group, so GN with G=C reduces over each channel's
// spatial extent — exactly InstanceNorm. C is read from the runtime shape.

/// <summary>§7-4 GroupNorm(G=C, affine:false) ≡ InstanceNorm(affine:false) on a rank-4 input
/// (C read from the runtime shape). Relative-L1 &lt; 1e-3.</summary>
[Module]
public partial class NNGroupNormGCMatchesInstanceNormRank4
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, H, W]
    {
        var eps = Scalar(1e-5f);
        Scalar<int64> c = x.ShapeTensor()[1];
        var gc = GroupNorm.Call(c, Scalar(false), eps, x);
        var inorm = InstanceNorm.Call(Scalar(false), eps, x);

        var pen = (gc - inorm).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + inorm.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-3f) * scale;
    }
}

/// <summary>§7-4 GroupNorm(G=C, affine:false) ≡ InstanceNorm(affine:false) on a rank-3 input
/// (C read from the runtime shape). Relative-L1 &lt; 1e-3.</summary>
[Module]
public partial class NNGroupNormGCMatchesInstanceNormRank3
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, L]
    {
        var eps = Scalar(1e-5f);
        Scalar<int64> c = x.ShapeTensor()[1];
        var gc = GroupNorm.Call(c, Scalar(false), eps, x);
        var inorm = InstanceNorm.Call(Scalar(false), eps, x);

        var pen = (gc - inorm).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + inorm.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-3f) * scale;
    }
}

// --- §7-5: alias equivalence — InstanceNorm{1,2,3}d ≡ InstanceNorm(affine:false) ---
// The aliases forward to InstanceNorm.Call(Scalar(false), eps, x), so they must
// agree bit-for-bit on the matching-rank input.

/// <summary>§7-5 InstanceNorm1d.Call(eps, x) == InstanceNorm.Call(affine:false, eps, x) on a rank-3 input (bit-for-bit).</summary>
[Module]
public partial class NNInstanceNorm1dAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, L]
    {
        var eps = Scalar(1e-5f);
        var alias = InstanceNorm1d.Call(eps, x);
        var generic = InstanceNorm.Call(Scalar(false), eps, x);
        var pen = (alias - generic).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-6f);
    }
}

/// <summary>§7-5 InstanceNorm2d.Call(eps, x) == InstanceNorm.Call(affine:false, eps, x) on a rank-4 input (bit-for-bit).</summary>
[Module]
public partial class NNInstanceNorm2dAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, H, W]
    {
        var eps = Scalar(1e-5f);
        var alias = InstanceNorm2d.Call(eps, x);
        var generic = InstanceNorm.Call(Scalar(false), eps, x);
        var pen = (alias - generic).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-6f);
    }
}

/// <summary>§7-5 InstanceNorm3d.Call(eps, x) == InstanceNorm.Call(affine:false, eps, x) on a rank-5 input (bit-for-bit).</summary>
[Module]
public partial class NNInstanceNorm3dAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, D, H, W]
    {
        var eps = Scalar(1e-5f);
        var alias = InstanceNorm3d.Call(eps, x);
        var generic = InstanceNorm.Call(Scalar(false), eps, x);
        var pen = (alias - generic).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-6f);
    }
}

// --- §7-2(b): affine param-count discrimination (rig models) ---
// Mirror NNBatchNormAffineFalseEvalGradModel: a live scalar pre-weight gives
// the rig a trainable param to count, and InstanceNorm/GroupNorm.Model(affine,
// …) fixes the affine bit. With affine:false, γ/β are pruned (dead branch, no
// gradient) ⇒ the model's only trainable param is the scalar weight (1). With
// affine:true, γ/β survive ⇒ 3 trainable params (scalar weight + γ + β). The
// rig [Fact] in NNLibraryTrainingCoverageTests asserts the field counts.

/// <summary>§7-2(b) InstanceNorm(affine:false) rig model: scalar pre-weight → InstanceNorm → per-(sample,channel) mean.
/// γ/β are pruned (dead branch), so the only trainable param is the scalar weight.</summary>
[Module]
public partial class NNInstanceNormAffineFalseParamModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = InstanceNorm.Model(Scalar(false), Scalar(1e-5f)).Call(input * w);
        Vector<int64> axes = [Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>§7-2(b) InstanceNorm(affine:true) rig model: scalar pre-weight → InstanceNorm → per-(sample,channel) mean.
/// γ/β survive, so the model exposes 3 trainable params (scalar weight + γ + β).</summary>
[Module]
public partial class NNInstanceNormAffineTrueParamModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = InstanceNorm.Model(Scalar(true), Scalar(1e-5f)).Call(input * w);
        Vector<int64> axes = [Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>§7-2(b) GroupNorm(G=2, affine:false) rig model: scalar pre-weight → GroupNorm → per-channel mean.
/// γ/β are pruned (dead branch), so the only trainable param is the scalar weight.</summary>
[Module]
public partial class NNGroupNormAffineFalseParamModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = GroupNorm.Model(Scalar(2L), Scalar(false), Scalar(1e-5f)).Call(input * w);
        Vector<int64> axes = [Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>§7-2(b) GroupNorm(G=2, affine:true) rig model: scalar pre-weight → GroupNorm → per-channel mean.
/// γ/β survive, so the model exposes 3 trainable params (scalar weight + γ + β).</summary>
[Module]
public partial class NNGroupNormAffineTrueParamModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = GroupNorm.Model(Scalar(2L), Scalar(true), Scalar(1e-5f)).Call(input * w);
        Vector<int64> axes = [Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>
/// Dropout: eval mode is the exact identity; training mode (ratio 0.5) makes
/// every element either 0 or exactly 2x, so y * (y - 2x) == 0 elementwise.
/// </summary>
[Module]
public partial class NNDropoutChecks
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var evalY = Dropout.Call(Scalar(0.5f), Scalar(false), x);
        var evalPen = (evalY - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var trainY = Dropout.Call(Scalar(0.5f), Scalar(true), x);
        var maskPen = (trainY * (trainY - x * Scalar(2f))).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return evalPen + maskPen < Scalar(1e-5f);
    }
}

// ---------------------------------------------------------------------------
// SpatialDropout (channel-wise dropout) coverage — src/Shorokoo.Modules/Layers/
// Dropout.cs. ONNX Dropout in TRAINING mode is QEE-value-blocked (random draw →
// no concrete values), so train-mode self-checks use only mask-AGNOSTIC
// invariants (each element is 0 or survivor-scale·x; per-channel uniformity over
// the spatial axes) — exactly the y·(y−2x)==0 idiom NNDropoutChecks uses. EVAL
// mode (identity) IS QEE-computable, so eval self-checks assert exact values.
// x is built STRICTLY POSITIVE (Range/positive tensor) so y/x is a clean mask.
// ---------------------------------------------------------------------------

/// <summary>
/// SpatialDropout channel-wise behavior on [N,C,H,W] (train mode, ratio 0.5,
/// mask-agnostic). With x strictly positive: (1) every element is 0 or exactly
/// 2x (survivor scale 1/(1-0.5)=2), so y·(y−2x)==0; (2) CHANNEL UNIFORMITY — the
/// per-channel mask m=y/x is constant across the spatial axes {2,3} within each
/// (n,c), so max(m)−min(m) over the spatial axes is 0 per (n,c). Check (2) passes
/// ONLY for channel-wise dropout and would FAIL for elementwise Dropout (which
/// draws per element, so m would vary across spatial positions) — it pins the
/// whole deliverable.
/// </summary>
[Module]
public partial class NNSpatialDropoutChannelWise
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = SpatialDropout.Call(Scalar(0.5f), Scalar(true), x);

        // (1) Each element is 0 or 2x (mask-agnostic).
        var maskPen = (y * (y - x * Scalar(2f))).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // (2) Channel uniformity: m = y/x constant over spatial axes {2,3} per (n,c).
        var m = y / x;
        Vector<int64> spatial = [Scalar(2L), Scalar(3L)];
        var hi = m.Reduce(ReduceKind.Max, spatial, keepDims: false);
        var lo = m.Reduce(ReduceKind.Min, spatial, keepDims: false);
        var uniformPen = (hi - lo).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return maskPen + uniformPen < Scalar(1e-4f);
    }
}

/// <summary>
/// SpatialDropout survivor scaling 1/(1-ratio) at a second ratio (train mode,
/// mask-agnostic): ratio 0.75 ⇒ survivors are exactly 4x, so y·(y−4x)==0. Pins
/// the survivor-scale coefficient (distinguishes the inverted-dropout rescale
/// from a plain unscaled mask).
/// </summary>
[Module]
public partial class NNSpatialDropoutSurvivorScale75
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = SpatialDropout.Call(Scalar(0.75f), Scalar(true), x);
        var maskPen = (y * (y - x * Scalar(4f))).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return maskPen < Scalar(1e-4f);
    }
}

/// <summary>
/// SpatialDropout eval-mode (training=false) is the exact identity, independent
/// of ratio (QEE-computable, exact values): SpatialDropout(0.5,false,x)==x and
/// SpatialDropout(0.9,false,x)==x, so Σ|y−x|==0 for both ratios.
/// </summary>
[Module]
public partial class NNSpatialDropoutEvalIdentity
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y05 = SpatialDropout.Call(Scalar(0.5f), Scalar(false), x);
        var y09 = SpatialDropout.Call(Scalar(0.9f), Scalar(false), x);
        var pen = (y05 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (y09 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// SpatialDropout rank-genericity — eval-mode exact identity at a single rank
/// (driven at rank 3/4/5 by separate AdvancedTestGraph one-liners). Identity is
/// QEE-computable, so Σ|y−x|==0 exactly. Proves the in-graph rank read handles
/// 1-D/2-D/3-D inputs (and the rank-2 degenerate path) uniformly.
/// </summary>
[Module]
public partial class NNSpatialDropoutEvalIdentityAnyRank
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = SpatialDropout.Call(Scalar(0.5f), Scalar(false), x);
        var pen = (y - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// SpatialDropout rank-3 [N,C,L] channel uniformity (train mode, mask-agnostic):
/// proves the [N,C,1] broadcast-mask shape spreads one draw over the single
/// spatial axis {2}. m=y/x must be constant over axis 2 per (n,c), AND each
/// element is 0 or 2x. FAILS for elementwise dropout.
/// </summary>
[Module]
public partial class NNSpatialDropoutChannelWiseRank3
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = SpatialDropout.Call(Scalar(0.5f), Scalar(true), x);
        var maskPen = (y * (y - x * Scalar(2f))).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var m = y / x;
        Vector<int64> spatial = [Scalar(2L)];
        var hi = m.Reduce(ReduceKind.Max, spatial, keepDims: false);
        var lo = m.Reduce(ReduceKind.Min, spatial, keepDims: false);
        var uniformPen = (hi - lo).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return maskPen + uniformPen < Scalar(1e-4f);
    }
}

/// <summary>
/// SpatialDropout rank-5 [N,C,D,H,W] channel uniformity (train mode, mask-agnostic):
/// proves the [N,C,1,1,1] broadcast-mask shape spreads one draw over all three
/// spatial axes {2,3,4}. m=y/x must be constant over axes {2,3,4} per (n,c), AND
/// each element is 0 or 2x. FAILS for elementwise dropout.
/// </summary>
[Module]
public partial class NNSpatialDropoutChannelWiseRank5
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = SpatialDropout.Call(Scalar(0.5f), Scalar(true), x);
        var maskPen = (y * (y - x * Scalar(2f))).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var m = y / x;
        Vector<int64> spatial = [Scalar(2L), Scalar(3L), Scalar(4L)];
        var hi = m.Reduce(ReduceKind.Max, spatial, keepDims: false);
        var lo = m.Reduce(ReduceKind.Min, spatial, keepDims: false);
        var uniformPen = (hi - lo).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return maskPen + uniformPen < Scalar(1e-4f);
    }
}

/// <summary>
/// SpatialDropout rank-2 [N,C] degenerate path (eval mode, exact identity): the
/// empty ones-run makes the mask shape [N,C] and the layer is still the identity
/// in eval, confirming the degenerate rank-2 branch builds and runs.
/// </summary>
[Module]
public partial class NNSpatialDropoutRank2Degenerate
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = SpatialDropout.Call(Scalar(0.5f), Scalar(false), x);
        var pen = (y - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// Dropout1d alias equivalence (eval mode, exact): Dropout1d.Call(r,t,x) must
/// equal SpatialDropout.Call(r,t,x) bit-for-bit on [N,C,L]. Eval mode keeps the
/// values QEE-computable so equality is exact — proves the alias is a faithful
/// forwarder.
/// </summary>
[Module]
public partial class NNDropout1dAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var alias = Dropout1d.Call(Scalar(0.5f), Scalar(false), x);
        var canon = SpatialDropout.Call(Scalar(0.5f), Scalar(false), x);
        var pen = (alias - canon).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-6f);
    }
}

/// <summary>
/// Dropout2d alias equivalence (eval mode, exact): Dropout2d.Call(r,t,x) ==
/// SpatialDropout.Call(r,t,x) bit-for-bit on [N,C,H,W].
/// </summary>
[Module]
public partial class NNDropout2dAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var alias = Dropout2d.Call(Scalar(0.5f), Scalar(false), x);
        var canon = SpatialDropout.Call(Scalar(0.5f), Scalar(false), x);
        var pen = (alias - canon).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-6f);
    }
}

/// <summary>
/// Dropout3d alias equivalence (eval mode, exact): Dropout3d.Call(r,t,x) ==
/// SpatialDropout.Call(r,t,x) bit-for-bit on [N,C,D,H,W].
/// </summary>
[Module]
public partial class NNDropout3dAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var alias = Dropout3d.Call(Scalar(0.5f), Scalar(false), x);
        var canon = SpatialDropout.Call(Scalar(0.5f), Scalar(false), x);
        var pen = (alias - canon).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-6f);
    }
}

// ---------------------------------------------------------------------------
// AlphaDropout + FeatureAlphaDropout (SELU-paired dropout) coverage —
// src/Shorokoo.Modules/Layers/Dropout.cs (Klambauer et al. 2017). ONNX Dropout
// in TRAINING mode draws a random mask (QEE-value-blocked; computed on the ORT
// backend inside AdvancedTestGraph), so train-mode self-checks use only
// mask-AGNOSTIC invariants. The KEY invariant: for a fixed ratio each output
// element is one of exactly TWO known closed forms — kept → a·x+b, dropped →
// a·α'+b (a constant) — so (y−(a·x+b))·(y−(a·α'+b))==0 elementwise, whatever the
// mask (the AlphaDropout analog of NNDropoutChecks' y·(y−2x)==0). α' =
// −1.7580993408473766f (PyTorch's fused SELU negative-saturation constant); a,b
// are computed from ratio (closed form below). x is built STRICTLY POSITIVE so
// kept values a·x+b > 0 never collide with the negative dropped constant a·α'+b,
// keeping the per-element drop indicator unambiguous. EVAL mode (identity) IS
// QEE-computable, so eval self-checks assert exact values.
//   p=0.5  : a=0.8864053f  b=0.7791938f   (vdrop = a·α'+b = −b = −0.7791938)
//   p=0.25 : a=0.8672579f  b=0.3811814f   (vdrop = a·α'+b = −1.1435442)
// ---------------------------------------------------------------------------

/// <summary>
/// AlphaDropout train-mode per-element invariant (ratio 0.5, mask-AGNOSTIC) — the
/// load-bearing correctness check. For a fixed ratio every output element is either
/// the KEPT closed form <c>vkeep = a·x + b</c> (per-element) or the DROPPED closed
/// form <c>vdrop = a·α' + b</c> (a constant, x-independent), so the product of the two
/// residuals <c>(y−vkeep)·(y−vdrop)==0</c> elementwise whatever the random mask —
/// the AlphaDropout analog of NNDropoutChecks' <c>y·(y−2x)==0</c>. Constants for
/// ratio 0.5: <c>α'=−1.7580993408473766f, a=0.8864053f, b=0.7791938f</c>. Tolerance
/// is loose (1e-3) because a,b are float consts vs the in-graph Pow/Sqrt rounding.
/// </summary>
[Module]
public partial class NNAlphaDropoutPerElementInvariant
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        const float alphaP = -1.7580993408473766f;
        const float a = 0.8864053f;
        const float b = 0.7791938f;

        var y = AlphaDropout.Call(Scalar(0.5f), Scalar(true), x);
        var vkeep = x * Scalar(a) + Scalar(b);          // a·x + b   (per-element)
        var vdrop = Scalar(a * alphaP + b);             // a·α' + b  (constant)
        var pen = ((y - vkeep) * (y - vdrop)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-3f);
    }
}

/// <summary>
/// AlphaDropout train-mode per-element invariant at a SECOND ratio (0.25,
/// mask-AGNOSTIC) — pins that a,b TRACK ratio (not a constant): the two closed forms
/// are now <c>a=0.8672579f, b=0.3811814f</c> (vdrop = a·α'+b = −1.1435442), distinct
/// from the ratio-0.5 pair. Same <c>(y−vkeep)·(y−vdrop)==0</c> elementwise invariant.
/// </summary>
[Module]
public partial class NNAlphaDropoutPerElementInvariantRatio25
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        const float alphaP = -1.7580993408473766f;
        const float a = 0.8672579f;
        const float b = 0.3811814f;

        var y = AlphaDropout.Call(Scalar(0.25f), Scalar(true), x);
        var vkeep = x * Scalar(a) + Scalar(b);
        var vdrop = Scalar(a * alphaP + b);
        var pen = ((y - vkeep) * (y - vdrop)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-3f);
    }
}

/// <summary>
/// AlphaDropout moment preservation (the defining SNN property), train mode, ratio
/// 0.5, over a LARGE tensor — an APPROXIMATE / statistical check. The affine a·x'+b
/// restores mean and variance only IN EXPECTATION over the mask (PyTorch issue
/// #74004), not per-realization, so this is a LOOSE sanity bound on a single fixed
/// seed-42 draw: |mean(y)−mean(x)| and |var(y)−var(x)| stay within a generous
/// tolerance (and y is finite). Demonstrates WHY the affine renorm exists — plain
/// dropout (which zeros) would shift both moments far more. Mask-agnostic in spirit
/// (no exact value), evaluated on the ORT backend's real random draw.
/// </summary>
[Module]
public partial class NNAlphaDropoutMomentPreservation
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = AlphaDropout.Call(Scalar(0.5f), Scalar(true), x);

        var meanX = x.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        var meanY = y.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        var varX = (x * x).Reduce(ReduceKind.Mean, keepDims: false).Scalar() - meanX * meanX;
        var varY = (y * y).Reduce(ReduceKind.Mean, keepDims: false).Scalar() - meanY * meanY;

        var meanPen = (meanY - meanX).Abs();
        var varPen = (varY - varX).Abs();
        // Generous statistical tolerances (single fixed-mask realization, #74004).
        return meanPen < Scalar(0.3f) & varPen < Scalar(0.6f);
    }
}

/// <summary>
/// AlphaDropout eval-mode exact identity (QEE-computable), ratio-independent:
/// <c>AlphaDropout(0.5,false,x)==x</c> AND <c>AlphaDropout(0.9,false,x)==x</c>, so
/// Σ|y−x|==0 for both. The <c>training.IfElse(affine, x)</c> gate yields x bit-exact
/// in eval (the affine is NOT the identity and the eval mask is all-ones), so this is
/// an exact closed-form check — mirrors NNSpatialDropoutEvalIdentity.
/// </summary>
[Module]
public partial class NNAlphaDropoutEvalIdentity
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y05 = AlphaDropout.Call(Scalar(0.5f), Scalar(false), x);
        var y09 = AlphaDropout.Call(Scalar(0.9f), Scalar(false), x);
        var pen = (y05 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (y09 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// FeatureAlphaDropout eval-mode exact identity at a single rank (driven at rank
/// 2/3/4/5 by separate AdvancedTestGraph one-liners): <c>FeatureAlphaDropout(0.5,
/// false,x)==x</c> and at ratio 0.9, so Σ|y−x|==0 for both. Proves the in-graph rank
/// read + the [N,C,1,…,1] mask-shape build collapse to the identity in eval for every
/// rank (incl. the rank-2 [N,C] degenerate path) — mirrors
/// NNSpatialDropoutEvalIdentityAnyRank.
/// </summary>
[Module]
public partial class NNFeatureAlphaDropoutEvalIdentityAnyRank
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y05 = FeatureAlphaDropout.Call(Scalar(0.5f), Scalar(false), x);
        var y09 = FeatureAlphaDropout.Call(Scalar(0.9f), Scalar(false), x);
        var pen = (y05 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (y09 - x).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// FeatureAlphaDropout CHANNEL-UNIFORMITY on [N,C,H,W] (train mode, ratio 0.5,
/// mask-AGNOSTIC) — the distinguishing channel-wise check. With x strictly positive:
/// (1) the per-element invariant <c>(y−vkeep)·(y−vdrop)==0</c> holds (each element is
/// a·x+b or a·α'+b); (2) the drop decision is per (sample,channel), so the per-element
/// "is-dropped" indicator <c>dropped = ((y−vdrop) near 0) ? 1 : 0</c> is CONSTANT
/// across the spatial axes {2,3} within each (n,c) — max−min over {2,3} is 0 per
/// (n,c). The indicator is built via the residual <c>(y−vdrop)</c>: dropped elements
/// share the constant vdrop (residual ≈ 0) while kept elements have residual
/// a·x+b−vdrop = a·(x−α') > 0 (since x>0>α'), so a Sign-of-clamped-residual cleanly
/// separates the two classes. Check (2) PASSES for FeatureAlphaDropout and FAILS for
/// elementwise AlphaDropout — it pins the channel-wise behavior. Constants for ratio
/// 0.5: a=0.8864053f, b=0.7791938f, α'=−1.7580993408473766f.
/// </summary>
[Module]
public partial class NNFeatureAlphaDropoutChannelUniform
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        const float alphaP = -1.7580993408473766f;
        const float a = 0.8864053f;
        const float b = 0.7791938f;

        var y = FeatureAlphaDropout.Call(Scalar(0.5f), Scalar(true), x);

        var vkeep = x * Scalar(a) + Scalar(b);
        var vdrop = Scalar(a * alphaP + b);

        // (1) Per-element two-value invariant (mask-agnostic): each element equals
        // vkeep or vdrop, so the product (y−vkeep)(y−vdrop) is ~0.
        var maskPen = ((y - vkeep) * (y - vdrop)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // (2) Channel uniformity: per-element is-kept indicator constant over {2,3} per (n,c).
        // residual = y − vdrop is ≈0 for dropped, ≥ a·(min x − α') > 0 for kept; clamp to {0,1}.
        var kept = (y - vdrop).Abs().Min(Scalar(1f));   // dropped → ≈0, kept → 1 (residual ≥ ~2.4)
        Vector<int64> spatial = [Scalar(2L), Scalar(3L)];
        var hi = kept.Reduce(ReduceKind.Max, spatial, keepDims: false);
        var lo = kept.Reduce(ReduceKind.Min, spatial, keepDims: false);
        var uniformPen = (hi - lo).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Relative-L1 (NNInstanceNormAffineFalseMatchesManual idiom): compare the
        // should-be-zero product against the squared value gap (vkeep−vdrop)² so the
        // bound is independent of tensor size and value magnitude. The old absolute
        // Σ|·| < 1e-3 grew with element count and tripped on WinCPU ORT FP; a relative
        // bound holds identically across platforms.
        var scale = Scalar(1f) + ((vkeep - vdrop) * (vkeep - vdrop)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return maskPen + uniformPen < Scalar(1e-4f) * scale;
    }
}

/// <summary>
/// FeatureAlphaDropout rank-3 [N,C,L] channel uniformity (train mode, ratio 0.5,
/// mask-agnostic): proves the [N,C,1] broadcast-mask shape spreads one draw over the
/// single spatial axis {2}. The per-element is-kept indicator must be constant over
/// axis 2 per (n,c), AND the two-value invariant holds. FAILS for elementwise
/// AlphaDropout.
/// </summary>
[Module]
public partial class NNFeatureAlphaDropoutChannelUniformRank3
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        const float alphaP = -1.7580993408473766f;
        const float a = 0.8864053f;
        const float b = 0.7791938f;

        var y = FeatureAlphaDropout.Call(Scalar(0.5f), Scalar(true), x);
        var vkeep = x * Scalar(a) + Scalar(b);
        var vdrop = Scalar(a * alphaP + b);
        var maskPen = ((y - vkeep) * (y - vdrop)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var kept = (y - vdrop).Abs().Min(Scalar(1f));
        Vector<int64> spatial = [Scalar(2L)];
        var hi = kept.Reduce(ReduceKind.Max, spatial, keepDims: false);
        var lo = kept.Reduce(ReduceKind.Min, spatial, keepDims: false);
        var uniformPen = (hi - lo).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Relative-L1 vs the squared value gap (see NNFeatureAlphaDropoutChannelUniform):
        // size/magnitude-independent so it holds identically on WinCPU and Linux.
        var scale = Scalar(1f) + ((vkeep - vdrop) * (vkeep - vdrop)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return maskPen + uniformPen < Scalar(1e-4f) * scale;
    }
}

/// <summary>
/// FeatureAlphaDropout rank-5 [N,C,D,H,W] channel uniformity (train mode, ratio 0.5,
/// mask-agnostic): proves the [N,C,1,1,1] broadcast-mask shape spreads one draw over
/// all three spatial axes {2,3,4}. The per-element is-kept indicator must be constant
/// over {2,3,4} per (n,c), AND the two-value invariant holds. FAILS for elementwise
/// AlphaDropout.
/// </summary>
[Module]
public partial class NNFeatureAlphaDropoutChannelUniformRank5
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        const float alphaP = -1.7580993408473766f;
        const float a = 0.8864053f;
        const float b = 0.7791938f;

        var y = FeatureAlphaDropout.Call(Scalar(0.5f), Scalar(true), x);
        var vkeep = x * Scalar(a) + Scalar(b);
        var vdrop = Scalar(a * alphaP + b);
        var maskPen = ((y - vkeep) * (y - vdrop)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var kept = (y - vdrop).Abs().Min(Scalar(1f));
        Vector<int64> spatial = [Scalar(2L), Scalar(3L), Scalar(4L)];
        var hi = kept.Reduce(ReduceKind.Max, spatial, keepDims: false);
        var lo = kept.Reduce(ReduceKind.Min, spatial, keepDims: false);
        var uniformPen = (hi - lo).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Relative-L1 vs the squared value gap (see NNFeatureAlphaDropoutChannelUniform):
        // size/magnitude-independent so it holds identically on WinCPU and Linux.
        var scale = Scalar(1f) + ((vkeep - vdrop) * (vkeep - vdrop)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return maskPen + uniformPen < Scalar(1e-4f) * scale;
    }
}

/// <summary>Embedding must equal a manual Gather over an identically initialized weight table.</summary>
[Module]
public partial class NNEmbeddingMatchesGather
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var y = Embedding.Call(numEmbeddings, dim, Scalar(-1L), Scalar(0f), Scalar(2f), indices);

        var wRef = Normal.Init([numEmbeddings, dim]);
        var yRef = wRef.Gather(indices, axis: 0);

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-4f);
    }
}

// ---------------------------------------------------------------------------
// Embedding knob coverage (paddingIdx / maxNorm / normType / init choice,
// src/Shorokoo.Modules/Layers/Embedding.cs — embedding-knobs design §8). Each
// self-checking [Module] returns a Scalar<bit> that AutoTest.AdvancedTestGraph
// requires true, mirroring NNEmbeddingMatchesGather above: the seeded Normal /
// XavierUniform init re-materializes identically inside the reference, so the
// gathered weight rows are reproduced exactly. The reference computes the
// padding mask / shrink-only p-norm clamp by hand and compares against
// Embedding.Call / EmbeddingHelpers.Embed.
// ---------------------------------------------------------------------------

/// <summary>
/// §8-2/§8-3 paddingIdx forward mask. With paddingIdx:2 over indices [0,1,2,2,3]
/// (V=5, D=4): every output row at a pad position (the two `2`s) must be all-zero,
/// and every non-pad row must equal the plain Normal.Init(...).Gather of that
/// index (unchanged). Also folds the off-sentinel no-op (§8-3): with paddingIdx:-1
/// the SAME call must equal the plain Gather of ALL rows (the IfElse(-1) gate
/// disables masking). The pad-row L2 mass and the non-pad/off-sentinel diffs are
/// rolled into one Scalar&lt;bit&gt; penalty.
/// </summary>
[Module]
public partial class NNEmbeddingPaddingIdxZeros
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);

        // paddingIdx:2 on indices [0,1,2,2,3] -> rows at the two `2`s masked to zero.
        var padded = Embedding.Call(numEmbeddings, dim, Scalar(2L), Scalar(0f), Scalar(2f), indices);
        // off-sentinel: paddingIdx:-1 must be a no-op == plain Gather.
        var offPad = Embedding.Call(numEmbeddings, dim, Scalar(-1L), Scalar(0f), Scalar(2f), indices);

        var wRef = Normal.Init([numEmbeddings, dim]);
        var gatherRef = wRef.Gather(indices, axis: 0);                 // [5, 4] plain lookup

        // Build the expected pad mask in the reference: rows where indices == 2 -> 0.
        var isPad = (indices == Scalar(2L)).Unsqueeze(-1);             // [5, 1] bit
        var zeros = gatherRef * Scalar(0f);
        var expected = isPad.Where(zeros, gatherRef);                  // pad rows -> 0, else gather

        // (a) padded output matches the hand-masked reference exactly (covers
        //     both the all-zero pad rows AND the unchanged non-pad rows).
        var padDiff = (padded - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // (b) explicit check that the pad positions carry zero L2 mass (independent of (a)).
        var padMass = (isPad.Where(padded, zeros)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // (c) off-sentinel is the plain Gather (no masking).
        var offDiff = (offPad - gatherRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return (padDiff + padMass + offDiff) < Scalar(1e-4f);
    }
}

/// <summary>
/// §8-4/§8-6 maxNorm L2 clamp (shrink-only). The seeded Normal.Init([5,4]) rows are
/// N(0,1)×4, so each row's L2 norm is ≈2 ≫ a 0.5 cap (the cap binds every gathered
/// row). The reference re-materializes the seeded weight and applies the hand-built
/// shrink-only clamp out = gather·min(1, maxNorm/‖row‖₂). Asserts:
///   (a) the over-cap output row's L2 norm ≈ maxNorm (clamped DOWN to the cap);
///   (b) the clamped output matches the hand reference exactly (per-row);
///   (c) under-cap rows are UNCHANGED — a huge maxNorm:1000f leaves the output ==
///       plain Gather (scale clipped to 1); and the §8-6 off-sentinel maxNorm:0f
///       likewise == plain Gather (renorm gate disabled).
/// Indices come from the runtime tensor (row 0 is the over-cap probe).
/// </summary>
[Module]
public partial class NNEmbeddingMaxNormClampsL2
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var maxNorm = Scalar(0.5f);   // ≪ the ~2.0 row norms, so it binds every gathered row

        var wRef = Normal.Init([numEmbeddings, dim]);
        var gatherRef = wRef.Gather(indices, axis: 0);                 // [n, 4]

        var y = Embedding.Call(numEmbeddings, dim, Scalar(-1L), maxNorm, Scalar(2f), indices);
        var bigCap = Embedding.Call(numEmbeddings, dim, Scalar(-1L), Scalar(1000f), Scalar(2f), indices);
        var off = Embedding.Call(numEmbeddings, dim, Scalar(-1L), Scalar(0f), Scalar(2f), indices);

        // Hand-built shrink-only L2 clamp reference: out = gather * min(1, maxNorm/‖row‖₂).
        var rowL2 = gatherRef.Reduce(ReduceKind.L2, [Scalar(-1L)], keepDims: true);  // [n, 1]
        var scale = (maxNorm / rowL2).Clip(Scalar(0f), Scalar(1f));                  // [n, 1]
        var expected = gatherRef * scale;                                           // [n, 4]

        // (b) the clamped output matches the reference for every row.
        var diff = (y - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // (a) the over-cap probe row (index 0 in the tensor) output L2 norm == maxNorm.
        var yOverRow = (Tensor<float32>)OnnxOp.Slice(y, Vector(0L), Vector(1L), Vector(0L)); // [1,4]
        var yOverNorm = yOverRow.Reduce(ReduceKind.L2, keepDims: false).Scalar();
        var capDiff = (yOverNorm - maxNorm).Abs();

        // (c) under-cap is a no-op: huge cap AND off-sentinel both == plain Gather.
        var bigDiff = (bigCap - gatherRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var offDiff = (off - gatherRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return (diff + capDiff + bigDiff + offDiff) < Scalar(1e-3f);
    }
}

/// <summary>
/// §8-5 normType honored (L1 vs L2). On the SAME over-cap row with maxNorm:0.5,
/// normType:1 must clamp the OUTPUT row's L1 norm to maxNorm, normType:2 must clamp
/// its L2 norm to maxNorm, and the two outputs must DIFFER — proving normType flows
/// into the p-norm (not a hardcoded 2). The over-cap index comes from the runtime
/// index tensor [kOver].
/// </summary>
[Module]
public partial class NNEmbeddingNormTypeL1VsL2
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var maxNorm = Scalar(0.5f);

        var yL1 = Embedding.Call(numEmbeddings, dim, Scalar(-1L), maxNorm, Scalar(1f), indices);  // [1,4]
        var yL2 = Embedding.Call(numEmbeddings, dim, Scalar(-1L), maxNorm, Scalar(2f), indices);  // [1,4]

        // (a) the chosen-p norm of each output row ≈ maxNorm.
        var l1Norm = yL1.Reduce(ReduceKind.L1, keepDims: false).Scalar();
        var l2Norm = yL2.Reduce(ReduceKind.L2, keepDims: false).Scalar();
        var l1CapDiff = (l1Norm - maxNorm).Abs();
        var l2CapDiff = (l2Norm - maxNorm).Abs();

        // (b) the two outputs DIFFER (normType actually changes the result).
        var differ = (yL1 - yL2).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return (l1CapDiff < Scalar(1e-3f)) & (l2CapDiff < Scalar(1e-3f)) & (differ > Scalar(1e-3f));
    }
}

/// <summary>
/// §8-7 init choice (static EmbeddingHelpers.Embed). Asserts (a)
/// Embed(idx, 5, 4, shape => XavierUniform.Init(shape)) equals
/// XavierUniform.Init([5,4]).Gather(idx), and (b) the DEFAULT Embed(idx, 5, 4)
/// (no init selector) equals Normal.Init([5,4]).Gather(idx) — i.e. the selector is
/// wired AND defaults to Normal (== the [Module] Embedding). Both seeded inits
/// re-materialize identically inside the reference.
/// </summary>
[Module]
public partial class NNEmbeddingInitChoice
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var xavier = EmbeddingHelpers.Embed(indices, 5L, 4L, shape => XavierUniform.Init(shape));
        var xavierRef = XavierUniform.Init([Scalar(5L), Scalar(4L)]).Gather(indices, axis: 0);
        var xDiff = (xavier - xavierRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var dflt = EmbeddingHelpers.Embed(indices, 5L, 4L);
        var dfltRef = Normal.Init([Scalar(5L), Scalar(4L)]).Gather(indices, axis: 0);
        var dDiff = (dflt - dfltRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        return (xDiff + dDiff) < Scalar(1e-4f);
    }
}

/// <summary>
/// §8-8 train-step rig model: a tiny Embedding-based model with paddingIdx set, so
/// TrainingRig.FromScratch + one TrainStep can assert the trainable embedding weight
/// MOVES (the masked lookup is differentiable). Embeds the in-graph constant indices
/// [0,1,2] over a trainable weight [4,3] (V=4, D=3) with paddingIdx:2 (so the last
/// gathered row is masked to zero and contributes no gradient), per-row-means to [3],
/// and adds the float input x [3] so the rig has a flowing graph input. The driving
/// [Fact] asserts a finite loss and that ≥1 embedding-weight element moved.
/// </summary>
[Module]
public partial class EmbeddingPaddingRigModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        Tensor<int64> indices = Vector(0L, 1L, 2L);                 // non-pad 0,1; pad 2
        var emb = Embedding.Call(Scalar(4L), Scalar(3L), Scalar(2L), Scalar(0f), Scalar(2f), indices); // [3,3]
        Vector<int64> lastAxis = [Scalar(1L)];
        var pooled = emb.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);   // [3]
        return pooled + x;                                          // [3] — x flows so the rig has an input
    }
}

// ---------------------------------------------------------------------------
// EmbeddingBag (src/Shorokoo.Modules/Layers/Embedding.cs — embedding-bag design
// §8). EmbeddingBag.Bag(indices [B,L], V, D, mode) is mathematically
// Embedding(indices).Reduce(mode, axis=1) → [B, D]. Each self-checking [Module]
// re-materializes the SAME seeded Normal.Init([V,D]) (deterministic, the
// NNEmbeddingMatchesGather idiom) and compares Bag against an INDEPENDENT
// Gather→Reduce reference by relative-L1. V=5, D=4, indices [B=2, L=3] with
// distinct ids so Sum ≠ Mean ≠ Max are all non-trivial.
// ---------------------------------------------------------------------------

/// <summary>
/// §8-1 (Sum, load-bearing): EmbeddingBag.Bag(indices, 5, 4, BagMode.Sum) must equal the
/// independent Normal.Init([5,4]).Gather(indices, axis:0).Reduce(Sum, axes:[1], keepDims:false)
/// — the seeded Normal table re-materializes identically, so the bag-of-words sum over axis 1
/// is reproduced by hand. Relative-L1 (the NNEmbeddingMatchesGather idiom plus the reduce).
/// </summary>
[Module]
public partial class NNEmbeddingBagSumMatchesGatherReduce
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);   // [B, D]

        var wRef = Normal.Init([numEmbeddings, dim]);
        var gathered = wRef.Gather(indices, axis: 0);            // [B, L, D]
        var yRef = gathered.Reduce(ReduceKind.Sum, Vector(1L), keepDims: false);   // [B, D]

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>
/// §8-1 (Mean): EmbeddingBag.Bag(indices, 5, 4, BagMode.Mean) must equal the independent
/// Normal.Init([5,4]).Gather(indices, axis:0).Reduce(Mean, axes:[1], keepDims:false). Confirms
/// the enum→ReduceKind.Mean dispatch and the documented full-L denominator (the reference also
/// divides by the full bag length L). Relative-L1.
/// </summary>
[Module]
public partial class NNEmbeddingBagMeanMatchesGatherReduce
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Mean);   // [B, D]

        var wRef = Normal.Init([numEmbeddings, dim]);
        var gathered = wRef.Gather(indices, axis: 0);             // [B, L, D]
        var yRef = gathered.Reduce(ReduceKind.Mean, Vector(1L), keepDims: false);   // [B, D]

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>
/// §8-1 (Max): EmbeddingBag.Bag(indices, 5, 4, BagMode.Max) must equal the independent
/// Normal.Init([5,4]).Gather(indices, axis:0).Reduce(Max, axes:[1], keepDims:false) — the
/// per-feature max over the bag. Confirms the BagMode.Max → ReduceKind.Max dispatch. Relative-L1.
/// </summary>
[Module]
public partial class NNEmbeddingBagMaxMatchesGatherReduce
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Max);   // [B, D]

        var wRef = Normal.Init([numEmbeddings, dim]);
        var gathered = wRef.Gather(indices, axis: 0);            // [B, L, D]
        var yRef = gathered.Reduce(ReduceKind.Max, Vector(1L), keepDims: false);   // [B, D]

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>
/// §8-2 shape: a [B, L] = [2, 3] index tensor → a [B, D] = [2, 4] EmbeddingBag output (D=4).
/// Asserts y.ShapeTensor()[0] == 2 (B) and [1] == 4 (D), pinning that the bag axis (L=3) is
/// reduced away and the output is exactly [batch, embeddingDim] (the NNBilinearBatchBroadcasts
/// ShapeTensor idiom).
/// </summary>
[Module]
public partial class NNEmbeddingBagShapeCheck
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [2, 3]
    {
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);   // expected [2, 4]
        var shape = y.ShapeTensor();
        Scalar<int64> d0 = shape[0];
        Scalar<int64> d1 = shape[1];
        return (d0 == Scalar(2L)) & (d1 == Scalar(4L));
    }
}

/// <summary>
/// §8-3 paddingIdx zeroes pad rows for Sum (the EXACT case). With paddingIdx:k and a bag whose
/// entries include k, EmbeddingBag.Bag(..., BagMode.Sum, paddingIdx:k) must equal the reference
/// that gathers, zeroes the pad rows (isPad.Where(0, gathered)) and sums over axis 1 — i.e. the
/// pad rows contribute zero to the Sum. Also asserts the masked Sum DIFFERS from the unmasked Sum
/// (the bag genuinely contains the pad id, so the mask is load-bearing). Per the design, only the
/// Sum-exact case is asserted (Mean/Max + paddingIdx have documented caveats). paddingIdx:2.
/// </summary>
[Module]
public partial class NNEmbeddingBagPaddingIdxSumExact
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L], contains the pad id 2
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        var padded = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum, paddingIdx: 2L);   // [B, D]
        var unmasked = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);                  // [B, D] (no pad mask)

        // Reference: gather, zero the pad rows (indices == 2), then sum over the bag axis.
        var wRef = Normal.Init([numEmbeddings, dim]);
        var gathered = wRef.Gather(indices, axis: 0);                  // [B, L, D]
        var isPad = (indices == Scalar(2L)).Unsqueeze(-1);             // [B, L, 1] bit
        var zeros = gathered * Scalar(0f);
        var masked = isPad.Where(zeros, gathered);                     // pad rows -> 0
        var yRef = masked.Reduce(ReduceKind.Sum, Vector(1L), keepDims: false);   // [B, D]

        // (a) the pad-masked Sum equals the hand-zeroed reference (pad rows excluded).
        var diff = (padded - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // (b) the pad mask is load-bearing: masked Sum != unmasked Sum (the bag contains a pad id).
        var differ = (padded - unmasked).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var scale = Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return (diff < Scalar(1e-3f) * scale) & (differ > Scalar(1e-3f));
    }
}

/// <summary>
/// §8-4 init choice. (a) EmbeddingBag.Bag(idx, 5, 4, BagMode.Sum, shape => XavierUniform.Init(shape))
/// must equal XavierUniform.Init([5,4]).Gather(idx, axis:0).Reduce(Sum, axis:1); and (b) the DEFAULT
/// EmbeddingBag.Bag(idx, 5, 4, BagMode.Sum) (no selector) must equal Normal.Init([5,4]).Gather(idx)
/// .Reduce(Sum, axis:1) — i.e. the embeddingInit selector is wired AND defaults to Normal. Both
/// seeded inits re-materialize identically. Relative-L1.
/// </summary>
[Module]
public partial class NNEmbeddingBagInitChoice
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var numEmbeddings = Scalar(5L);
        var dim = Scalar(4L);
        Vector<int64> bagAxis = [Scalar(1L)];

        var xavier = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum, shape => XavierUniform.Init(shape));
        var xavierRef = XavierUniform.Init([numEmbeddings, dim]).Gather(indices, axis: 0)
            .Reduce(ReduceKind.Sum, bagAxis, keepDims: false);
        var xDiff = (xavier - xavierRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var dflt = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);
        var dfltRef = Normal.Init([numEmbeddings, dim]).Gather(indices, axis: 0)
            .Reduce(ReduceKind.Sum, bagAxis, keepDims: false);
        var dDiff = (dflt - dfltRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var scale = Scalar(1f)
            + xavierRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
            + dfltRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return (xDiff + dDiff) < Scalar(1e-3f) * scale;
    }
}

/// <summary>
/// §8-5 train-step rig model: a tiny model ending in EmbeddingBag.Bag so TrainingRig.FromScratch +
/// one TrainStep can assert the owned trainable table MOVES (the bag lookup is differentiable
/// through Gather + Reduce). Bags the in-graph constant indices [[0,1],[2,3]] ([B=2,L=2]) over a
/// trainable weight [5,3] (V=5, D=3) with BagMode.Sum → [2,3], per-row-means to [2], and adds the
/// float input x [2] so the rig has a flowing graph input. The driving [Fact] asserts a finite loss
/// and that ≥1 embedding-table element moved.
/// </summary>
[Module]
public partial class EmbeddingBagRigModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        Tensor<int64> indices = Tensor(new long[] { 2L, 2L }, 0L, 1L, 2L, 3L);   // [2, 2] bags
        var bag = EmbeddingBag.Bag(indices, 5L, 3L, BagMode.Sum);                // [2, 3]
        Vector<int64> lastAxis = [Scalar(1L)];
        var pooled = bag.Reduce(ReduceKind.Mean, lastAxis, keepDims: false);     // [2]
        return pooled + x;                                                       // [2] — x flows so the rig has an input
    }
}

/// <summary>LeakyReLU and ELU hyper-alpha modules must match their Where-based closed forms.</summary>
[Module]
public partial class NNLeakyReLUAndELUClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var zeros = x * Scalar(0f);

        var leaky = LeakyReLU.Call(Scalar(0.1f), x);
        var leakyRef = (x > zeros).Where(x, x * Scalar(0.1f));

        var elu = ELU.Call(Scalar(0.7f), x);
        var eluRef = (x > zeros).Where(x, (x.Exp() - Scalar(1f)) * Scalar(0.7f));

        var pen = (leaky - leakyRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (elu - eluRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-5f);
    }
}

/// <summary>
/// Pooling helpers: full-window MaxPool2d/AvgPool2d must equal the global
/// pools; Flatten must produce [N, C*H*W]; a strided MaxPool preserves the
/// global maximum.
/// </summary>
[Module]
public partial class NNPoolingHelpersChecks
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var mp = Pooling.MaxPool2d(x, kernelSize: 4);
        var gmp = Pooling.GlobalMaxPool2d(x);
        var ap = Pooling.AvgPool2d(x, kernelSize: 4);
        var gap = Pooling.GlobalAvgPool2d(x);

        var mp2 = Pooling.MaxPool2d(x, kernelSize: 2);
        var strideMaxPen = (mp2.Reduce(ReduceKind.Max, keepDims: false).Scalar()
                            - x.Reduce(ReduceKind.Max, keepDims: false).Scalar()).Abs();

        var f = Pooling.Flatten(x);
        var flatPen = (f.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>()
                    + (f.DimTensor(1) - Scalar(32L)).Abs().Cast<float32>();

        var pen = (mp - gmp).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (ap - gap).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + strideMaxPen + flatPen;
        return pen < Scalar(1e-4f);
    }
}

// ---------------------------------------------------------------------------
// Generalized Pooling helper coverage (Pooling.MaxPool/AvgPool/LpPool + 1d/2d/3d
// aliases, GlobalLpPool, MaxPoolWithIndices/MaxUnpool, src/Shorokoo.Modules/
// Layers/Pooling.cs) — design §7 cases 1–8. Each self-checking [Module] returns
// a Scalar<bit> that AutoTest.AdvancedTestGraph requires true. ALL pool ops have
// NO QEE values (shape/dtype only), so the numeric Scalar<bit> is computed on the
// ORT backend inside AdvancedTestGraph — exactly like NNPoolingHelpersChecks. The
// closed-form modules build the pool input as an in-module constant tensor (so the
// hand-computed expected values are exact) and fold a 0*x touch so AutoTest has a
// runtime input to drive; the full-window==global / alias-equivalence / geometry
// modules pool the runtime input x directly (as NNPoolingHelpersChecks does).
// NaN-safe ok-counting (Within → 1/0) with the correct > (N-1) threshold is used
// wherever values are compared to constants.
// ---------------------------------------------------------------------------

/// <summary>§7-1 1D/3D closed forms. MaxPool1d([2]) on x=[1,3,2,4] (stride 2) → [3,4];
/// AvgPool1d([2]) → [2,3]; a full-window MaxPool3d([2,2,2]) / AvgPool3d on a [1,1,2,2,2]
/// cube of 1..8 → max 8 / mean 4.5. Inputs are in-module constants (exact hand math);
/// a 0*x touch gates on the runtime input.</summary>
[Module]
public partial class NNPool1d3dClosedForm
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // 1-D: x = [[[1,3,2,4]]], kernel [2], stride defaults to 2.
        var x1 = Tensor(new long[] { 1L, 1L, 4L }, 1f, 3f, 2f, 4f);
        var mp1 = Pooling.MaxPool1d(x1, new long[] { 2L });   // [3,4]
        var ap1 = Pooling.AvgPool1d(x1, new long[] { 2L });   // [2,3]
        var mp1Ref = Tensor(new long[] { 1L, 1L, 2L }, 3f, 4f);
        var ap1Ref = Tensor(new long[] { 1L, 1L, 2L }, 2f, 3f);

        // 3-D: [1,1,2,2,2] cube of 1..8, full window [2,2,2] → max 8, mean 4.5.
        var x3 = Tensor(new long[] { 1L, 1L, 2L, 2L, 2L }, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        var mp3 = Pooling.MaxPool3d(x3, new long[] { 2L, 2L, 2L }).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var ap3 = Pooling.AvgPool3d(x3, new long[] { 2L, 2L, 2L }).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((mp1 - mp1Ref).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar(), 1e-4f)
               + Within((ap1 - ap1Ref).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar(), 1e-4f)
               + Within((mp3 - Scalar(8f)).Abs(), 1e-4f)
               + Within((ap3 - Scalar(4.5f)).Abs(), 1e-4f);

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(4L);   // all 4 closed-form ok-bits + touch; > (5-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>§7-2 LpPool p=2 == √(Σx²). LpPool1d([2]) on x=[3,4] → √(9+16)=5; and a full-window
/// LpPool2d([H,W]) == GlobalLpPool(p=2) on the runtime input (relative-L1 ≈ 0). The closed-form
/// half uses an in-module constant; the equivalence half pools x.</summary>
[Module]
public partial class NNLpPoolClosedFormAndGlobal
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // Closed form: x = [[[3,4]]], kernel [2] → sqrt(9+16) = 5.
        var x1 = Tensor(new long[] { 1L, 1L, 2L }, 3f, 4f);
        var lp1 = Pooling.LpPool1d(x1, new long[] { 2L }).Reduce(ReduceKind.Sum, keepDims: false).Scalar();   // 5
        var okClosed = Within((lp1 - Scalar(5f)).Abs(), 1e-4f);

        // Full-window LpPool2d == GlobalLpPool (p=2) on the runtime input x = [1, C, H, W].
        long[] fullHW = { 4L, 4L };   // runtime input is [1,2,4,4]
        var lpFull = Pooling.LpPool2d(x, fullHW);
        var glp = Pooling.GlobalLpPool(x);
        var equivPen = (lpFull - glp).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + glp.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var okEquiv = (equivPen < Scalar(1e-3f) * scale).IfElse(Scalar(1L), Scalar(0L));

        return okClosed + okEquiv > Scalar(1L);   // 2 ok-bits; > (2-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>§7-3 full-window pool == global pool on the runtime input x=[1,C,H,W]:
/// MaxPool2d(x, [H,W]) == GlobalMaxPool2d(x); AvgPool2d(x, [H,W]) == GlobalAvgPool2d(x);
/// LpPool2d(x, [H,W]) == GlobalLpPool(x). Relative-L1 ≈ 0.</summary>
[Module]
public partial class NNFullWindowEqualsGlobal
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // runtime input [1,2,4,4]
    {
        long[] fullHW = { 4L, 4L };
        var maxFull = Pooling.MaxPool2d(x, fullHW);
        var avgFull = Pooling.AvgPool2d(x, fullHW);
        var lpFull = Pooling.LpPool2d(x, fullHW);
        var gMax = Pooling.GlobalMaxPool2d(x);
        var gAvg = Pooling.GlobalAvgPool2d(x);
        var gLp = Pooling.GlobalLpPool(x);

        var pen = (maxFull - gMax).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (avgFull - gAvg).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (lpFull - gLp).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f)
                  + gMax.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                  + gAvg.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                  + gLp.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return pen < Scalar(1e-3f) * scale;
    }
}

/// <summary>§7-4 scalar ↔ per-axis alias equivalence on the runtime input:
/// MaxPool2d(x, 2) == MaxPool2d(x, [2,2]); same for AvgPool/LpPool; and the per-rank alias
/// MaxPool1d(x1,[2]) == MaxPool(x1,[2]). The scalar overload reads x.Rank()-2 at build time,
/// which a symbolic [Module] input lacks, so the scalar pools use an in-module constant
/// [1,1,4,4] tensor; x only gates the result via a touch. The 1d per-rank alias uses an
/// in-module [1,1,4] constant.</summary>
[Module]
public partial class NNPoolScalarPerAxisAliasEquiv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // Literal constant input → build-time-known rank 4 for the scalar overloads.
        var c = Tensor(new long[] { 1L, 1L, 4L, 4L },
            1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f);

        var maxS = Pooling.MaxPool2d(c, kernelSize: 2L);                  // historical scalar 2d
        var maxA = Pooling.MaxPool2d(c, kernelSize: new long[] { 2L, 2L });
        var avgS = Pooling.AvgPool2d(c, kernelSize: 2L);                  // historical scalar 2d
        var avgA = Pooling.AvgPool2d(c, kernelSize: new long[] { 2L, 2L });
        // LpPool has no historical scalar *2d* overload; the scalar form is the generic
        // LpPool(c, long) which broadcasts via x.Rank()-2.
        var lpS = Pooling.LpPool(c, kernelSize: 2L);
        var lpA = Pooling.LpPool2d(c, kernelSize: new long[] { 2L, 2L });

        // Per-rank alias: MaxPool1d(x1,[2]) == MaxPool(x1,[2]) (rank inferred from kernel).
        var c1 = Tensor(new long[] { 1L, 1L, 4L }, 1f, 3f, 2f, 4f);
        var max1Alias = Pooling.MaxPool1d(c1, new long[] { 2L });
        var max1Generic = Pooling.MaxPool(c1, new long[] { 2L });

        var pen = (maxS - maxA).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (avgS - avgA).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (lpS - lpA).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                + (max1Alias - max1Generic).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return pen + touch < Scalar(1e-4f);
    }
}

/// <summary>§7-5 per-axis / asymmetric geometry vs the raw core op: MaxPool2d(x, [3,2],
/// stride:[2,1], padding:[1,0]) equals a hand-built NN.MaxPool with the SAME geometry
/// (dilations [1,1], kernel [3,2], symmetric pads [1,0,1,0], strides [2,1]). Relative-L1.
/// Pins the per-axis stride/padding plumbing.</summary>
[Module]
public partial class NNPoolPerAxisGeometryMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // runtime input [1,2,6,5]
    {
        var y = Pooling.MaxPool2d(x, kernelSize: new long[] { 3L, 2L },
            stride: new long[] { 2L, 1L }, padding: new long[] { 1L, 0L });

        // padding [1,0] (length spatialRank, symmetric) → ONNX begin..end [1,0,1,0].
        var yRef = NN.MaxPool(x, ceilMode: false,
            dilations: new long[] { 1L, 1L },
            kernelShape: new long[] { 3L, 2L },
            pads: new long[] { 1L, 0L, 1L, 0L },
            storageOrder: 0L,
            strides: new long[] { 2L, 1L });

        var diff = (y - yRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff < Scalar(1e-3f) * (Scalar(1f) + yRef.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar());
    }
}

/// <summary>§7-6 MaxUnpool round-trip: (vals, idx) = MaxPoolWithIndices(x, [2,2]); then
/// MaxUnpool(vals, idx, [2,2], outputShape: x.ShapeTensor()) → a tensor of x's shape. Assert
/// (a) the unpooled shape matches x (sum of |shape−x.shape| == 0); (b) GlobalMaxPool(u) ==
/// GlobalMaxPool(x) — the kept maxima land back (the global max of x is the max of the pooled
/// maxima, and is reinstated in u); (c) sum(u) == sum(vals) — every pooled value is scattered
/// once and the rest are zero. All on the ORT backend (MaxUnpool has no QEE values).</summary>
[Module]
public partial class NNMaxUnpoolRoundTrip
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // runtime input [1,2,4,4]
    {
        var (vals, idx) = Pooling.MaxPoolWithIndices(x, new long[] { 2L, 2L });
        var u = Pooling.MaxUnpool(vals, idx, new long[] { 2L, 2L }, outputShape: x.ShapeTensor());

        // (a) shape match.
        var shapePen = (u.ShapeTensor() - x.ShapeTensor()).Abs()
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar().Cast<float32>();

        // (b) kept global maximum reinstated.
        var gMaxU = Pooling.GlobalMaxPool2d(u).Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var gMaxX = Pooling.GlobalMaxPool2d(x).Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var maxPen = (gMaxU - gMaxX).Abs();

        // (c) sum of scattered values equals the sum of the pooled maxima.
        var sumU = u.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var sumVals = vals.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var sumPen = (sumU - sumVals).Abs();

        var ok = Within(shapePen, 1e-4f)
               + Within(maxPen, 1e-3f)
               + Within(sumPen, 1e-3f);
        return ok > Scalar(2L);   // all 3 ok-bits; > (3-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>§7-7 count_include_pad toggle: AvgPool2d([2,2], padding:[1,1]) with
/// countIncludePad:true vs false DIFFER at the padded border, each matching a hand-computed
/// 2x2 output. Input is an in-module constant [1,1,2,2] = [[1,2],[3,4]]; pad 1 → the 2x2 sits
/// at the center of a 4x4 zero-padded grid, kernel 2 stride 2 → every window covers exactly
/// ONE real cell (the four corners 1/2/3/4) with 3 padded zeros. countIncludePad:false divides
/// by the 1 real cell ⇒ [[1,2],[3,4]]; countIncludePad:true divides by the full 4 ⇒
/// [[0.25,0.5],[0.75,1]]. The two outputs must NOT be equal.</summary>
[Module]
public partial class NNAvgPoolCountIncludePadToggle
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var c = Tensor(new long[] { 1L, 1L, 2L, 2L }, 1f, 2f, 3f, 4f);

        var apFalse = Pooling.AvgPool2d(c, kernelSize: new long[] { 2L, 2L },
            padding: new long[] { 1L, 1L }, countIncludePad: false);
        var apTrue = Pooling.AvgPool2d(c, kernelSize: new long[] { 2L, 2L },
            padding: new long[] { 1L, 1L }, countIncludePad: true);

        var falseRef = Tensor(new long[] { 1L, 1L, 2L, 2L }, 1f, 2f, 3f, 4f);          // /1 real cell
        var trueRef = Tensor(new long[] { 1L, 1L, 2L, 2L }, 0.25f, 0.5f, 0.75f, 1f);   // /4 full window

        var falsePen = (apFalse - falseRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var truePen = (apTrue - trueRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var diff = (apFalse - apTrue).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();   // must differ

        var ok = Within(falsePen, 1e-4f)
               + Within(truePen, 1e-4f)
               + (diff > Scalar(1e-3f)).IfElse(Scalar(1L), Scalar(0L));

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(3L);   // 3 ok-bits + touch (4); > (4-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// Loss closed-form checks on tiny fixed values. The runtime input p = [1, 3]
/// drives L1/Huber/SmoothL1; the classification/binary losses use constant
/// inputs with hand-computed expected values:
///   L1([1,3],[0,1]) = 1.5; Huber(1,[1,3],[1,1]) = 0.75 (= SmoothL1);
///   CE([[0,0]],[0]) = ln 2; NLL([[-1,-2]],[1]) = 2;
///   BCE([.5,.5],[1,0]) = ln 2; BCEWithLogits([0,0],[1,0]) = ln 2.
/// </summary>
[Module]
public partial class NNLossClosedFormChecks
{
    public static Scalar<bit> Inline(Tensor<float32> p)
    {
        var ln2 = Scalar(0.69314718f);

        var t1 = Tensor(new long[] { 2L }, 0f, 1f);
        var pen = (L1Loss.Inline(p, t1) - Scalar(1.5f)).Abs();

        var t2 = Tensor(new long[] { 2L }, 1f, 1f);
        pen = pen + (HuberLoss.Inline(p, t2, Scalar(1f)) - Scalar(0.75f)).Abs();
        pen = pen + (SmoothL1Loss.Inline(p, t2) - Scalar(0.75f)).Abs();

        var logits = Tensor(new long[] { 1L, 2L }, 0f, 0f);
        pen = pen + (CrossEntropyLoss.Inline(logits, Vector(0L)) - ln2).Abs();

        var logProbs = Tensor(new long[] { 1L, 2L }, -1f, -2f);
        pen = pen + (NLLLoss.Inline(logProbs, Vector(1L)) - Scalar(2f)).Abs();

        var probs = Tensor(new long[] { 2L }, 0.5f, 0.5f);
        var tb = Tensor(new long[] { 2L }, 1f, 0f);
        pen = pen + (BCELoss.Inline(probs, tb) - ln2).Abs();

        var rawLogits = Tensor(new long[] { 2L }, 0f, 0f);
        pen = pen + (BCEWithLogitsLoss.Inline(rawLogits, tb) - ln2).Abs();

        // L2 over rank-2 predictions reduces over ALL elements:
        // mean([[1,2],[3,4]]²) = (1+4+9+16)/4 = 7.5. (Pins the former axis-0-only
        // reduction, which made rank-2+ predictions unusable.)
        var p22 = Tensor(new long[] { 2L, 2L }, 1f, 2f, 3f, 4f);
        var z22 = Tensor(new long[] { 2L, 2L }, 0f, 0f, 0f, 0f);
        pen = pen + (L2Loss.Inline(p22, z22) - Scalar(7.5f)).Abs();

        return pen < Scalar(1e-4f);
    }
}

// ---------------------------------------------------------------------------
// Loss configurability-knob closed-form checks (loss-knobs design §7). One
// self-checking [Module] per knob family, calling the new Reduced(...) /
// PerElement(...) overloads with CONSTANT inputs (constant weight/posWeight
// Tensors, a long? ignoreIndex, a LossReduction) and comparing to the
// hand-computed constant. Each returns a Scalar<bit> via the same
// ok-counting Within(...)/AtLeastZero(...) NaN-safety idiom as
// NNLossEdgeCaseChecks (a NaN fails every comparison, so it can never slip
// through). The runtime input `t` is only folded in as a zero-scaled touch so
// AutoTest.AdvancedTestGraph has a graph input to drive — the asserted math is
// entirely on the in-module constants. ln2 = 0.69314718.
// ---------------------------------------------------------------------------

/// <summary>
/// CrossEntropyLoss reduction + weight + ignore_index knobs (design §7).
/// All logits are uniform <c>[0,0]</c> so each non-ignored sample's CE is
/// exactly ln2:
///   reduction Mean = ln2; Sum = 2·ln2; PerElement = [ln2, ln2] (both elements);
///   weighted mean (w=[2,1], targets [0,1]) = (2·ln2+1·ln2)/(2+1) = ln2;
///   weighted mean (w=[2,1], targets [0,0]) = (2·ln2+2·ln2)/(2+2) = ln2;
///   weighted SUM (w=[2,1], targets [0,1]) = (2+1)·ln2 (pins the Σw denominator,
///     not /N — sum is unweighted-denominator);
///   ignore_index (logits [[0,0]]×3, targets [0,7,1], ignore 7) mean over the
///     2 non-ignored = ln2 (NOT (2·ln2)/3), and PerElement is 0 at the ignored slot.
/// </summary>
[Module]
public partial class NNCrossEntropyReductionWeightIgnoreChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var ln2 = Scalar(0.69314718f);
        var logits2 = Tensor(new long[] { 2L, 2L }, 0f, 0f, 0f, 0f);   // [[0,0],[0,0]]
        var tgt01 = Vector(0L, 1L);
        var tgt00 = Vector(0L, 0L);
        var weight = Tensor(new long[] { 2L }, 2f, 1f);                // [2,1]

        // reduction Mean / Sum on uniform logits.
        var ceMean = CrossEntropyLoss.Reduced(logits2, tgt01, reduction: LossReduction.Mean);
        var ceSum = CrossEntropyLoss.Reduced(logits2, tgt01, reduction: LossReduction.Sum);

        // PerElement → [ln2, ln2]; check each element via Slice.
        var cePer = CrossEntropyLoss.PerElement(logits2, tgt01);   // [2]
        var cePer0 = ((Tensor<float32>)OnnxOp.Slice(cePer, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var cePer1 = ((Tensor<float32>)OnnxOp.Slice(cePer, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Weighted mean: denominator is Σ w[target] over non-ignored samples.
        var wMean01 = CrossEntropyLoss.Reduced(logits2, tgt01, weight: weight, reduction: LossReduction.Mean);
        var wMean00 = CrossEntropyLoss.Reduced(logits2, tgt00, weight: weight, reduction: LossReduction.Mean);
        // Weighted SUM: (2+1)·ln2 = 3·ln2 — pins that sum does NOT divide by Σw.
        var wSum01 = CrossEntropyLoss.Reduced(logits2, tgt01, weight: weight, reduction: LossReduction.Sum);

        // ignore_index: mean over the 2 non-ignored = ln2 (denominator excludes ignored).
        var logits3 = Tensor(new long[] { 3L, 2L }, 0f, 0f, 0f, 0f, 0f, 0f);
        var tgtIg = Vector(0L, 7L, 1L);
        var ceIgMean = CrossEntropyLoss.Reduced(logits3, tgtIg, ignoreIndex: 7L, reduction: LossReduction.Mean);
        // PerElement returns 0 at the ignored position (index 1).
        var ceIgPer = CrossEntropyLoss.PerElement(logits3, tgtIg, ignoreIndex: 7L);   // [3]
        var ceIgPer1 = ((Tensor<float32>)OnnxOp.Slice(ceIgPer, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((ceMean - ln2).Abs(), 1e-4f)
               + Within((ceSum - Scalar(2f) * ln2).Abs(), 1e-4f)
               + Within((cePer0 - ln2).Abs(), 1e-4f)
               + Within((cePer1 - ln2).Abs(), 1e-4f)
               + Within((wMean01 - ln2).Abs(), 1e-4f)
               + Within((wMean00 - ln2).Abs(), 1e-4f)
               + Within((wSum01 - Scalar(3f) * ln2).Abs(), 1e-4f)
               + Within((ceIgMean - ln2).Abs(), 1e-4f)
               + Within(ceIgPer1.Abs(), 1e-5f) + AtLeastZero(ceIgPer1.Abs());

        // Fold a zero-scaled touch of the runtime input as an extra ok-bit so the
        // graph has a real input to drive (the math above is on constants).
        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(10L);   // all 10 closed-form ok-bits + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// CrossEntropyLoss label_smoothing knob (design §7), built in-graph from
/// LogSoftmax+NLL+uniform:
///   nontrivial: logits [[2,0]], target [0], α=0.2, K=2 → 0.32694
///     (logp=[−0.126928,−2.126928]; nll=0.126928; smooth=1.126928;
///      loss = 0.8·0.126928 + 0.2·1.126928);
///   K=2 uniform-logit collapse: logits [[0,0]], target [0], α=0.1 → ln2.
/// Stretch (design O2): the 3-way labelSmoothing+weight+ignoreIndex interaction
/// is pinned by NNCrossEntropyLabelSmoothWeightIgnoreChecks below.
/// </summary>
[Module]
public partial class NNCrossEntropyLabelSmoothingChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var ln2 = Scalar(0.69314718f);

        // Nontrivial blend: logits [[2,0]], target 0, α=0.2.
        var logitsA = Tensor(new long[] { 1L, 2L }, 2f, 0f);
        var lsA = CrossEntropyLoss.Reduced(logitsA, Vector(0L), labelSmoothing: 0.2f, reduction: LossReduction.Mean);

        // Uniform-logit collapse: logits [[0,0]], target 0, α=0.1 → ln2.
        var logitsU = Tensor(new long[] { 1L, 2L }, 0f, 0f);
        var lsU = CrossEntropyLoss.Reduced(logitsU, Vector(0L), labelSmoothing: 0.1f, reduction: LossReduction.Mean);

        var ok = Within((lsA - Scalar(0.3269280f)).Abs(), 1e-4f)
               + Within((lsU - ln2).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(2L);   // both closed-form ok-bits + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// Stretch (design O2): the 3-way labelSmoothing + weight + ignoreIndex
/// interaction. The smoothing term must thread <c>weight[target]</c> AND drop
/// ignored samples from its denominator too, exactly like the NLL term.
/// <para>
/// 3 samples, K=2, weight=[2,1], ignore 7, α=0.2:
///   sample0 logits [2,0] target 0 (w=2); sample1 logits [0,0] target 7 (IGNORED);
///   sample2 logits [1,3] target 1 (w=1).
/// Weighted-mean denominator = Σ w[target] over non-ignored = 2+1 = 3.
///   NLL term: (2·(−logp0[0]) + 1·(−logp2[1]))/3 = (2·0.126928 + 1·0.126928)/3 = 0.126928
///   smooth term (uniform per-sample mean −log-prob, same weight/ignore masking):
///     (2·u0 + 1·u2)/3 where u_n = −(1/2)Σ_k logp_n[k]; = (2·1.126928 + 1·1.126928)/3 = 1.126928
///   loss = 0.8·0.126928 + 0.2·1.126928 = 0.32694.
/// (The clean value falls out because both non-ignored samples have logit gap 2,
/// so their per-sample nll/smooth split matches the single-sample K=2 case; the
/// weighted denominator 3 cancels the weighted numerators. This simultaneously
/// pins: weight threaded through BOTH terms, ignore excluded from BOTH
/// denominators, and the (1−α)/α blend.)
/// </para>
/// </summary>
[Module]
public partial class NNCrossEntropyLabelSmoothWeightIgnoreChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var logits = Tensor(new long[] { 3L, 2L }, 2f, 0f, 0f, 0f, 1f, 3f);
        var targets = Vector(0L, 7L, 1L);
        var weight = Tensor(new long[] { 2L }, 2f, 1f);

        var loss = CrossEntropyLoss.Reduced(
            logits, targets, weight: weight, ignoreIndex: 7L,
            labelSmoothing: 0.2f, reduction: LossReduction.Mean);

        var ok = Within((loss - Scalar(0.3269280f)).Abs(), 1e-4f);
        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(1L);   // closed-form ok-bit + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// NLLLoss weight + ignore_index knobs (predictions are log-probs), mirroring
/// the CE closed forms (design §7). Log-probs are uniform <c>[−ln2, −ln2]</c> so
/// each non-ignored sample's NLL is exactly ln2:
///   weighted mean (w=[2,1], targets [0,1]) = (2·ln2+1·ln2)/3 = ln2;
///   weighted SUM (w=[2,1], targets [0,1]) = 3·ln2;
///   ignore_index (targets [0,7,1], ignore 7) mean over 2 non-ignored = ln2;
///   PerElement is 0 at the ignored slot.
/// </summary>
[Module]
public partial class NNNLLLossWeightIgnoreChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var ln2 = Scalar(0.69314718f);
        var nl2 = Scalar(-0.69314718f);
        var logp2 = Tensor(new long[] { 2L, 2L }, -0.69314718f, -0.69314718f, -0.69314718f, -0.69314718f);
        var tgt01 = Vector(0L, 1L);
        var weight = Tensor(new long[] { 2L }, 2f, 1f);

        var wMean = NLLLoss.Reduced(logp2, tgt01, weight: weight, reduction: LossReduction.Mean);
        var wSum = NLLLoss.Reduced(logp2, tgt01, weight: weight, reduction: LossReduction.Sum);

        var logp3 = Tensor(new long[] { 3L, 2L },
            -0.69314718f, -0.69314718f, -0.69314718f, -0.69314718f, -0.69314718f, -0.69314718f);
        var tgtIg = Vector(0L, 7L, 1L);
        var igMean = NLLLoss.Reduced(logp3, tgtIg, ignoreIndex: 7L, reduction: LossReduction.Mean);
        var igPer = NLLLoss.PerElement(logp3, tgtIg, ignoreIndex: 7L);   // [3]
        var igPer1 = ((Tensor<float32>)OnnxOp.Slice(igPer, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Zero-scaled touch of the runtime input (and a no-op use of nl2) so the
        // graph has a real input to drive; folded in as an extra ok-bit.
        var touch = ((t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar() + nl2 - nl2).Abs();

        var ok = Within((wMean - ln2).Abs(), 1e-4f)
               + Within((wSum - Scalar(3f) * ln2).Abs(), 1e-4f)
               + Within((igMean - ln2).Abs(), 1e-4f)
               + Within(igPer1.Abs(), 1e-5f) + AtLeastZero(igPer1.Abs());

        return ok + Within(touch, 1e-6f) > Scalar(5L);   // all 5 closed-form ok-bits + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// BCEWithLogitsLoss pos_weight knob (design §7). With logit x=0, σ(0)=0.5:
///   t=1, posWeight=2 → −2·log(0.5) = 2·ln2 (per-element, via PerElement);
///   t=0, posWeight=2 → −log(1−σ(0)) = ln2 (pos_weight has NO effect on negatives);
///   mean over [t=1, t=0], posWeight=2 = (2·ln2 + ln2)/2 = 1.5·ln2 = 1.0397208.
/// </summary>
[Module]
public partial class NNBCEWithLogitsPosWeightChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var ln2 = Scalar(0.69314718f);
        var posW = Tensor(new long[] { 1L }, 2f);

        // Per-element at t=1 (= 2·ln2) and t=0 (= ln2), posWeight=2.
        var perPos = BCEWithLogitsLoss.PerElement(Tensor(new long[] { 1L }, 0f), Tensor(new long[] { 1L }, 1f), posWeight: posW)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var perNeg = BCEWithLogitsLoss.PerElement(Tensor(new long[] { 1L }, 0f), Tensor(new long[] { 1L }, 0f), posWeight: posW)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Mean over [t=1, t=0] with posWeight=2 → 1.5·ln2.
        var posW2 = Tensor(new long[] { 2L }, 2f, 2f);
        var meanMix = BCEWithLogitsLoss.Reduced(
            Tensor(new long[] { 2L }, 0f, 0f), Tensor(new long[] { 2L }, 1f, 0f),
            posWeight: posW2, reduction: LossReduction.Mean);

        var ok = Within((perPos - Scalar(2f) * ln2).Abs(), 1e-4f)
               + Within((perNeg - ln2).Abs(), 1e-4f)
               + Within((meanMix - Scalar(1.5f) * ln2).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(3L);   // all 3 closed-form ok-bits + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// SmoothL1Loss beta knob and the Huber(δ=β)/β bridge (design §7). With error
/// e=2 (predictions [2], targets [0]):
///   beta=4 (quadratic region |e|&lt;β): 0.5·e²/β = 0.5;
///   beta=1 (linear): |e| − 0.5·β = 1.5;
///   beta=0.5 (linear): 2 − 0.25 = 1.75.
/// Each is cross-checked against HuberLoss.Reduced(δ=β)/β computed directly.
/// </summary>
[Module]
public partial class NNSmoothL1BetaChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var pred = Tensor(new long[] { 1L }, 2f);
        var tgt = Tensor(new long[] { 1L }, 0f);

        Scalar<int64> CheckBeta(float beta, float expected)
        {
            var s = SmoothL1Loss.Reduced(beta, pred, tgt, reduction: LossReduction.Mean);
            var bridge = HuberLoss.Reduced(Scalar(beta), pred, tgt, reduction: LossReduction.Mean) * Scalar(1f / beta);
            return Within((s - Scalar(expected)).Abs(), 1e-4f)
                 + Within((s - bridge).Abs(), 1e-4f);   // SmoothL1(β) == Huber(δ=β)/β
        }

        var ok = CheckBeta(4f, 0.5f) + CheckBeta(1f, 1.5f) + CheckBeta(0.5f, 1.75f);
        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(6L);   // 6 closed-form ok-bits (3 values + 3 bridges) + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// Regression-loss reductions (L1/L2/Huber, design §7):
///   L1([1,3],[0,1]): mean=1.5, sum=3, PerElement=[1,2] (both elements);
///   L2([[1,2],[3,4]],0): mean=7.5, sum=30;
///   HuberLoss(δ=1) Reduced sum vs mean on the same [1,3]/[0,1] (both small-error
///     quadratic: per-element 0.5·e² = [0.5, 2] → sum 2.5, mean 1.25).
/// </summary>
[Module]
public partial class NNRegressionReductionChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var p1 = Tensor(new long[] { 2L }, 1f, 3f);
        var t1 = Tensor(new long[] { 2L }, 0f, 1f);

        var l1Mean = L1Loss.Reduced(p1, t1, reduction: LossReduction.Mean);
        var l1Sum = L1Loss.Reduced(p1, t1, reduction: LossReduction.Sum);
        var l1Per = L1Loss.PerElement(p1, t1);   // [1, 2]
        var l1Per0 = ((Tensor<float32>)OnnxOp.Slice(l1Per, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var l1Per1 = ((Tensor<float32>)OnnxOp.Slice(l1Per, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var p22 = Tensor(new long[] { 2L, 2L }, 1f, 2f, 3f, 4f);
        var z22 = Tensor(new long[] { 2L, 2L }, 0f, 0f, 0f, 0f);
        var l2Mean = L2Loss.Reduced(p22, z22, reduction: LossReduction.Mean);
        var l2Sum = L2Loss.Reduced(p22, z22, reduction: LossReduction.Sum);

        // Huber δ=1 on [1,3]/[0,1]: errors [1,2] → quad 0.5 / linear 1·(2−0.5)=1.5 → sum 2.0, mean 1.0.
        var hSum = HuberLoss.Reduced(Scalar(1f), p1, t1, reduction: LossReduction.Sum);
        var hMean = HuberLoss.Reduced(Scalar(1f), p1, t1, reduction: LossReduction.Mean);

        var ok = Within((l1Mean - Scalar(1.5f)).Abs(), 1e-5f)
               + Within((l1Sum - Scalar(3f)).Abs(), 1e-5f)
               + Within((l1Per0 - Scalar(1f)).Abs(), 1e-5f)
               + Within((l1Per1 - Scalar(2f)).Abs(), 1e-5f)
               + Within((l2Mean - Scalar(7.5f)).Abs(), 1e-4f)
               + Within((l2Sum - Scalar(30f)).Abs(), 1e-4f)
               + Within((hSum - Scalar(2.0f)).Abs(), 1e-5f)
               + Within((hMean - Scalar(1.0f)).Abs(), 1e-5f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(8L);   // all 8 closed-form ok-bits + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

// ---------------------------------------------------------------------------
// Rig-wrapper [Module]s for the loss-knobs (design §7 "Rig-path tests"). These
// are tiny 2-input (predictions, targets) wrappers that BAKE the build-time
// knobs / a constant weight tensor, so they satisfy the TrainingRig 2-input
// scalar-loss contract. Driven by NNLibraryTrainingCoverageTests via
// TrainingRig.FromScratch + a TrainStep.
// ---------------------------------------------------------------------------

/// <summary>
/// CrossEntropyLoss with baked <c>ignoreIndex:7</c> + <c>reduction:Sum</c>, as a
/// 2-input rig loss (no extra input, output stays scalar). Confirms the
/// attribute/sum knobs are rig-safe.
/// </summary>
[Module]
public partial class NNCrossEntropyIgnoreSumLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<int64> targets)
        => CrossEntropyLoss.Reduced(predictions, targets, ignoreIndex: 7L, reduction: LossReduction.Sum);
}

/// <summary>
/// CrossEntropyLoss with a baked-constant class <c>weight=[2,1]</c> (the
/// documented weight-via-rig recipe). The weight tensor is fixed as a graph
/// constant inside the wrapper, so the loss stays a 2-input rig loss even
/// though weight is normally a third graph input.
/// </summary>
[Module]
public partial class NNCrossEntropyBakedWeightLoss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<int64> targets)
        => CrossEntropyLoss.Reduced(predictions, targets,
            weight: Tensor(new long[] { 2L }, 2f, 1f), reduction: LossReduction.Mean);
}

// ---------------------------------------------------------------------------
// Training-rig models (no hypers; layer hypers fixed via Model(...) so the
// model graphs satisfy the rig's inputs-only contract).
// ---------------------------------------------------------------------------

/// <summary>
/// Wide linear regression model: <c>[N, 2] → [N, 400]</c> (800 weights + 400 biases).
/// The 400 independent output rows make the random-init <em>starting</em> loss
/// concentrate (law of large numbers) into a narrow, platform-independent band, so the
/// convergence tests can bound it absolutely instead of relative to a single noisy draw.
/// </summary>
[Module]
public partial class NNWideRegressionModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => Linear.Model(Scalar(400L), Scalar(true)).Call(input);
}

/// <summary>Tiny conv net: Conv2d(2, k3, s1, p1) → ReLU → GlobalAvgPool → [N, 2] logits.</summary>
[Module]
public partial class NNTinyConvClassifier
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var x = Conv2d.Model(Scalar(2L), Scalar(3L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(input);
        x = x.Relu();
        x = Pooling.GlobalAvgPool2d(x);
        return x.Reshape([input.DimTensor(0), Scalar(2L)]);
    }
}

/// <summary>BatchNorm2d in eval mode followed by per-channel mean: [N, C, H, W] → [C].</summary>
[Module]
public partial class NNBatchNormEvalGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm2d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false)).Call(input);
        Vector<int64> axes = [Scalar(0L), Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>BatchNorm2d in training mode followed by per-channel mean: [N, C, H, W] → [C].</summary>
[Module]
public partial class NNBatchNormTrainGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm2d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true)).Call(input);
        Vector<int64> axes = [Scalar(0L), Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

// ---------------------------------------------------------------------------
// Generalized rank-generic BatchNorm coverage models (design §7 groups A–G).
// Every BatchNorm graph carries Globals.StateUpdate links, so ALL of these run
// through the rig (NNLibraryTrainingCoverageTests), not AutoTest — even the
// "pure" eval-path closed-form checks (the plain inference executor has no
// STATE_UPDATE_LINK op). Hypers are fixed via BatchNorm.Model(...) so the model
// graphs are inputs-only. Closed-form checks output (y − reference) so a zero
// target makes the L2 loss the mean squared elementwise deviation (≈0 ⇒ match).
// ---------------------------------------------------------------------------

// --- Group A: rank generality, eval path == init closed form x / sqrt(1 + eps) ---
// Eval at init (running 0/1, gamma=1, beta=0) must equal x / sqrt(1 + eps) for
// EVERY element at ranks 2/3/4/5. Output = y − x/sqrt(1+eps); zero target ⇒ loss≈0.

/// <summary>Group A rank-2 [N, C]: BatchNorm eval init closed-form residual y − x/sqrt(1+eps).</summary>
[Module]
public partial class NNBatchNormEvalRank2ClosedForm
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        var invStd = Scalar(1f) / (Scalar(1f) + Scalar(1e-5f)).Sqrt();
        return y - input * invStd;
    }
}

/// <summary>Group A rank-3 [N, C, L] (the form the old BatchNorm1d rejected): eval init closed-form residual.</summary>
[Module]
public partial class NNBatchNormEvalRank3ClosedForm
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        var invStd = Scalar(1f) / (Scalar(1f) + Scalar(1e-5f)).Sqrt();
        return y - input * invStd;
    }
}

/// <summary>Group A rank-4 [N, C, H, W]: eval init closed-form residual.</summary>
[Module]
public partial class NNBatchNormEvalRank4ClosedForm
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        var invStd = Scalar(1f) / (Scalar(1f) + Scalar(1e-5f)).Sqrt();
        return y - input * invStd;
    }
}

/// <summary>Group A rank-5 [N, C, D, H, W] (the new rank-5 path): eval init closed-form residual.</summary>
[Module]
public partial class NNBatchNormEvalRank5ClosedForm
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        var invStd = Scalar(1f) / (Scalar(1f) + Scalar(1e-5f)).Sqrt();
        return y - input * invStd;
    }
}

// --- Group G: alias equivalence — alias.Call(...) == BatchNorm.Call(..., affine:true, track:true) ---
// Output = alias − generic; zero target ⇒ loss≈0 proves bit-for-bit equality.

/// <summary>Group G: BatchNorm2d.Call == BatchNorm.Call(.., affine:true, track:true) on [N,C,H,W] (eval).</summary>
[Module]
public partial class NNBatchNorm2dAliasEquiv
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var alias = BatchNorm2d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false)).Call(input);
        var generic = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        return alias - generic;
    }
}

/// <summary>Group G: BatchNorm1d.Call == generic on rank-2 [N, C] (eval).</summary>
[Module]
public partial class NNBatchNorm1dAliasEquivRank2
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var alias = BatchNorm1d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false)).Call(input);
        var generic = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        return alias - generic;
    }
}

/// <summary>Group G: BatchNorm1d.Call == generic on rank-3 [N, C, L] (eval) — the formerly-rejected form.</summary>
[Module]
public partial class NNBatchNorm1dAliasEquivRank3
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var alias = BatchNorm1d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false)).Call(input);
        var generic = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        return alias - generic;
    }
}

/// <summary>Group G: BatchNorm3d.Call == generic on rank-5 [N, C, D, H, W] (eval).</summary>
[Module]
public partial class NNBatchNorm3dAliasEquiv
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var alias = BatchNorm3d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false)).Call(input);
        var generic = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
        return alias - generic;
    }
}

// --- Group D: affine on/off ---

/// <summary>Group D: affine:false eval output equals the bare normalizer x/sqrt(1+eps) (no gamma/beta).
/// A live scalar pre-weight w (init 1) gives the rig a trainable param to optimize — with affine:false
/// the BN's gamma/beta are on the dead branch and contribute none. w scales both terms identically, so
/// the residual is 0 regardless of w; loss ≈ 0 confirms the closed form.</summary>
[Module]
public partial class NNBatchNormAffineFalseClosedForm
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var xw = input * w;
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(false), Scalar(true)).Call(xw);
        var invStd = Scalar(1f) / (Scalar(1f) + Scalar(1e-5f)).Sqrt();
        return y - xw * invStd;
    }
}

/// <summary>Group D: affine:false eval model with a live scalar pre-weight, per-channel mean → [C].
/// Because affine:false routes gamma/beta to the dead branch, they receive NO gradient and are pruned
/// from the trainable-param struct — so this model's ONLY trainable param is the scalar weight (the
/// affine:true NNBatchNormEvalGradModel has 2: gamma + beta).</summary>
[Module]
public partial class NNBatchNormAffineFalseEvalGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(false), Scalar(true)).Call(input * w);
        Vector<int64> axes = [Scalar(0L), Scalar(2L), Scalar(3L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

// --- Group B: train-path normalization + EMA at rank 2, rank 3 and rank 5 ---

/// <summary>Group B rank-2 [N, C] train mode (reduceAxes={0}, empty spatial range —
/// the original BatchNorm1d MLP/tabular use case), per-channel mean → [C] (≈0 ⇒ loss≈0 vs zero target).</summary>
[Module]
public partial class NNBatchNormTrainRank2Model
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(input);
        Vector<int64> axes = [Scalar(0L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>Group B rank-3 [N, C, L] train mode, per-channel mean → [C] (≈0 ⇒ loss≈0 vs zero target).</summary>
[Module]
public partial class NNBatchNormTrainRank3Model
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(input);
        Vector<int64> axes = [Scalar(0L), Scalar(2L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

/// <summary>Group B rank-5 [N, C, D, H, W] train mode, per-channel mean → [C].</summary>
[Module]
public partial class NNBatchNormTrainRank5Model
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var y = BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(input);
        Vector<int64> axes = [Scalar(0L), Scalar(2L), Scalar(3L), Scalar(4L)];
        return y.Reduce(ReduceKind.Mean, axes, keepDims: false);
    }
}

// --- Group C: EMA correctness + momentum + biased variance (analytic, generic module) ---

/// <summary>Group C: generic BatchNorm train mode, momentum 0.9 — analytic running-stats EMA
/// through ModelState (lr=0 isolates state). Input [1,2,3,4] ⇒ ModelState [0.25, 1.025].</summary>
[Module]
public partial class NNBatchNormAnalyticMomentum09Model
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(x);
}

/// <summary>Group C: generic BatchNorm train mode, momentum 0.5 — input [1,2,3,4] ⇒ ModelState [1.25, 1.125].</summary>
[Module]
public partial class NNBatchNormAnalyticMomentum05Model
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => BatchNorm.Model(Scalar(0.5f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(x);
}

/// <summary>Group C: generic BatchNorm train mode at rank 2 [N, C] (reduceAxes={0}, paramShape=[1,C])
/// — momentum 0.9. Input [2,1] values [1,3] (N=2, C=1): batch mean 2, BIASED var 1; lr=0 isolates the
/// state ⇒ ModelState [0·0.9+2·0.1, 1·0.9+1·0.1] = [0.2, 1.0]. Pins the rank-2 batch-axis-only reduction.</summary>
[Module]
public partial class NNBatchNormAnalyticRank2Model
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(x);
}

// --- Groups E & F: track_running_stats on/off and train vs eval stat selection ---
// These need an eval pass that reads running stats moved by a prior train step,
// so the eval model's ModelState is injected with the moved values across rigs
// (both use the generic BatchNorm with the same channel count ⇒ identical
// ModelState field layout). The eval models output the full [N,C,...] tensor so
// the test can compare against each path's closed form per element.

/// <summary>Groups E/F: generic BatchNorm train mode over [N,C,H,W], full output (no reduction).</summary>
[Module]
public partial class NNBatchNormTrainFullModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true), Scalar(true), Scalar(true)).Call(input);
}

/// <summary>Groups E/F: generic BatchNorm eval mode, track:true (eval uses running stats), full output.</summary>
[Module]
public partial class NNBatchNormEvalTrackTrueFullModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(true)).Call(input);
}

/// <summary>Group E: generic BatchNorm eval mode, track:false (eval uses eval-batch stats), full output.</summary>
[Module]
public partial class NNBatchNormEvalTrackFalseFullModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => BatchNorm.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(false), Scalar(true), Scalar(false)).Call(input);
}

/// <summary>Scalar-weight model wrapped in eval-mode Dropout (the gradient must pass through unchanged).</summary>
[Module]
public partial class NNDropoutEvalGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return Dropout.Model(Scalar(0.5f), Scalar(false)).Call(input * weight);
    }
}

/// <summary>Scalar-weight model wrapped in EVAL-mode SpatialDropout. Eval is the exact
/// identity, so this is value-checkable: w·SpatialDropout_eval(x) == w·x, the loss equals
/// the no-dropout closed form and the upstream scalar weight gets the full gradient — the
/// SpatialDropout parallel to <see cref="NNDropoutEvalGradModel"/>.</summary>
[Module]
public partial class NNSpatialDropoutEvalGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return SpatialDropout.Model(Scalar(0.5f), Scalar(false)).Call(input * weight);
    }
}

/// <summary>Scalar-weight model wrapped in TRAIN-mode SpatialDropout (channel-broadcast Mul +
/// forward mask). Used for the train-step smoke: finite loss + ≥1 trainable param moves (a
/// gradient flowed through the channel mask). No exact value check (RNG mask).</summary>
[Module]
public partial class NNSpatialDropoutTrainGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return SpatialDropout.Model(Scalar(0.5f), Scalar(true)).Call(input * weight);
    }
}

/// <summary>Scalar-weight model wrapped in EVAL-mode AlphaDropout. Eval is the exact identity,
/// so this is value-checkable: w·AlphaDropout_eval(x) == w·x, the loss equals the no-dropout
/// closed form and the upstream scalar weight gets the full gradient — the AlphaDropout parallel
/// to <see cref="NNDropoutEvalGradModel"/>.</summary>
[Module]
public partial class NNAlphaDropoutEvalGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return AlphaDropout.Model(Scalar(0.5f), Scalar(false)).Call(input * weight);
    }
}

/// <summary>Scalar-weight model wrapped in TRAIN-mode AlphaDropout (SELU saturation-value +
/// affine renorm + elementwise forward mask). Used for the train-step smoke: finite loss + ≥1
/// trainable param moves (a gradient flowed through the affine + mask). No exact value (RNG mask).</summary>
[Module]
public partial class NNAlphaDropoutTrainGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return AlphaDropout.Model(Scalar(0.5f), Scalar(true)).Call(input * weight);
    }
}

/// <summary>Scalar-weight model wrapped in EVAL-mode FeatureAlphaDropout (channel-wise). Eval is
/// the exact identity, so w·FeatureAlphaDropout_eval(x) == w·x is a value-checkable anchor — the
/// FeatureAlphaDropout parallel to <see cref="NNSpatialDropoutEvalGradModel"/>.</summary>
[Module]
public partial class NNFeatureAlphaDropoutEvalGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return FeatureAlphaDropout.Model(Scalar(0.5f), Scalar(false)).Call(input * weight);
    }
}

/// <summary>Scalar-weight model wrapped in TRAIN-mode FeatureAlphaDropout (channel-broadcast
/// saturation-value + affine + forward mask). Used for the train-step smoke: finite loss + ≥1
/// trainable param moves (a gradient flowed through the affine + channel mask). No exact value (RNG).</summary>
[Module]
public partial class NNFeatureAlphaDropoutTrainGradModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return FeatureAlphaDropout.Model(Scalar(0.5f), Scalar(true)).Call(input * weight);
    }
}

/// <summary>
/// The static <c>NN</c> wrapper class (Modules/Ops/OnnxOps.cs) — window generators
/// (Blackman/Hamming/Hann, both Scalar&lt;int64&gt; and Scalar&lt;int32&gt; size overloads),
/// DeterminantMatrix, both EyeLike overloads (same-dtype and dtype-switching), and Concat.
/// Input is a 2×2 float matrix.
/// </summary>
[Module]
public partial class NNStaticWrapperWindowEyeDetCheck
{
    public static (Vector<float32>, Vector<float32>, Vector<float32>, Vector<float32>, Tensor<float32>, Tensor<float32>, Tensor<int64>, Tensor<float32>) Inline(Tensor<float32> sq)
    {
        var blackman = NN.BlackmanWindow<float32>(Scalar(5L));
        var blackman32 = NN.BlackmanWindow<float32>(Scalar(5L).Cast<int32>().Scalar());
        var hamming = NN.HammingWindow<float32>(Scalar(5L));
        var hann = NN.HannWindow<float32>(Scalar(5L));
        var det = NN.DeterminantMatrix(sq);
        var eye = NN.EyeLike<float32>(sq);
        var eyeShifted = NN.EyeLike<int64>((Variable)sq, k: 1);
        var cat = NN.Concat(new[] { sq, sq }, axis: 0);
        return (blackman, blackman32, hamming, hann, det, eye, eyeShifted, cat);
    }
}

/// <summary>
/// More <c>NN</c> wrappers: the tensor-shaped global pools (Average/Lp/Max), Max/Min,
/// Identity, integer Mod, and the Scalar-k TopK overload (which unsqueezes k internally).
/// Inputs: a [1,2,4,4] float NCHW image and two int64 vectors.
/// </summary>
[Module]
public partial class NNStaticWrapperPoolMathCheck
{
    public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<int64>, Tensor<int64>, Tensor<float32>, Tensor<int64>) Inline(
        Tensor<float32> img, Tensor<int64> ia, Tensor<int64> ib)
    {
        var gap = NN.GlobalAveragePool(img);
        var glp = NN.GlobalLpPool(img);
        var gmp = NN.GlobalMaxPool(img);
        var maxMin = NN.Max(img, NN.Min(img, img));
        var mod = NN.Mod(ia, ib);
        var ident = NN.Identity(ia);
        var (topVals, topIdx) = NN.TopK(img, Scalar(2L), axis: -1, largest: true, sorted: true);
        return (gap, glp, gmp, maxMin, mod, ident, topVals, topIdx);
    }
}

// ---------------------------------------------------------------------------
// Analytic-check fixtures promoted from the 2026-06-12 framework behavior
// test campaign.
// The constant initializers make every gradient and 1–2-step optimizer update
// hand-derivable, so NNLibraryTrainingCoverageTests can assert exact post-step
// parameter/state values through real TrainStep execution.
// ---------------------------------------------------------------------------

/// <summary>Every element 0.5 (shape-driven fill).</summary>
[TrainableParamInitializer]
public static partial class AnalyticInitHalf
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Tensor<float32>.Fill(shape, Globals.TensorData(1, 0.5f));
}

/// <summary>Every element 1.0 (shape-driven fill).</summary>
[TrainableParamInitializer]
public static partial class AnalyticInitOne
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f));
}

/// <summary>Constant [1,2,3,4] — per-element-distinct so permutation/slice
/// gradient-routing errors change the asserted values.</summary>
[TrainableParamInitializer]
public static partial class AnalyticInitRange4
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f)) * Tensor(new long[] { 4L }, 1f, 2f, 3f, 4f);
}

/// <summary>Constant [[1,2],[3,4]].</summary>
[TrainableParamInitializer]
public static partial class AnalyticInitRange22
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f)) * Tensor(new long[] { 2L, 2L }, 1f, 2f, 3f, 4f);
}

/// <summary>y = w[1] · x (broadcast): grad_w must sum-reduce over the broadcast axis.</summary>
[Module]
public partial class AnalyticBroadcastMulModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => x * AnalyticInitHalf.Init(Vector(1L));
}

/// <summary>y = x + b[1] (broadcast add).</summary>
[Module]
public partial class AnalyticBroadcastAddModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => x + AnalyticInitHalf.Init(Vector(1L));
}

/// <summary>y = relu(w ⊙ x): grads must be masked where the pre-activation ≤ 0.</summary>
[Module]
public partial class AnalyticReluModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => (x * AnalyticInitRange4.Init(Vector(4L))).Relu();
}

/// <summary>y = x · W ([2,2]·[2,2]): grad_W must equal xᵀ·gUp exactly.</summary>
[Module]
public partial class AnalyticMatMulModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => x.MatMul(AnalyticInitRange22.Init(Vector(2L, 2L)));
}

/// <summary>y = w·x + w — w consumed twice; its gradient must ACCUMULATE across both uses.</summary>
[Module]
public partial class AnalyticDoubleUseModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = AnalyticInitHalf.Init(Vector(1L));
        return x * w + w;
    }
}

/// <summary>w[4] → Reshape[2,2] → Transpose → Reshape[4] → ⊙ x: gradients must route
/// back through the index permutation [0,2,1,3].</summary>
[Module]
public partial class AnalyticPermuteModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = AnalyticInitRange4.Init(Vector(4L));
        var wp = w.Reshape([Scalar(2L), Scalar(2L)]).Transpose([1L, 0L]).Reshape([Scalar(4L)]);
        return wp * x;
    }
}

/// <summary>y = w[0:2] ⊙ x: only the sliced elements of w may receive gradient.</summary>
[Module]
public partial class AnalyticSliceParamModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = AnalyticInitRange4.Init(Vector(4L));
        return (Tensor<float32>)OnnxOp.Slice(w, Vector(0L), Vector(2L)) * x;
    }
}

/// <summary>y = w[1] · x with w₀ = 1: the minimal model for exact multi-step optimizer math
/// (with x=[1], t per test: L = mean((w·x − t)²), grad = 2(w·x − t)·x).</summary>
[Module]
public partial class AnalyticScalarWModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => x * AnalyticInitOne.Init(Vector(1L));
}

/// <summary>BatchNorm2d in training mode (momentum 0.9, eps 1e-5) — for the exact
/// running-stats EMA check through checkpoint.ModelState (lr = 0 isolates the state).</summary>
[Module]
public partial class AnalyticBatchNormModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => BatchNorm2d.Model(Scalar(0.9f), Scalar(1e-5f), Scalar(true)).Call(x);
}

/// <summary>Library Linear(out=2, bias) — for the bind-known-weights check
/// (W layout [out,in]; y = x·Wᵀ + b).</summary>
[Module]
public partial class AnalyticBindLinearModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Linear.Model(Scalar(2L), Scalar(true)).Call(x);
}

/// <summary>Loss edge semantics (campaign §3): SmoothL1's two regions (0.5d² for |d|&lt;1,
/// |d|−0.5 beyond), BCE's clamp at p∈{0,1} (≈1e-7 when correct, −ln(1e-7)=16.118 when
/// maximally wrong — finite either way), BCEWithLogits' stability at logits ±100
/// (≈0 / ≈100, no overflow or NaN), and CrossEntropy's log-softmax at non-uniform
/// logits ([1,2,3], class 0 → 2.4076059). Ok-counting via IfElse(1,0) so a NaN
/// (which fails every comparison) can never slip through.</summary>
[Module]
public partial class NNLossEdgeCaseChecks
{
    public static Scalar<bit> Inline(Tensor<float32> d)   // d = [0.5, 2]
    {
        var z1 = Tensor(new long[] { 1L }, 0f);
        var quad = SmoothL1Loss.Inline((Tensor<float32>)OnnxOp.Slice(d, Vector(0L), Vector(1L)), z1);
        var lin = SmoothL1Loss.Inline((Tensor<float32>)OnnxOp.Slice(d, Vector(1L), Vector(2L)), z1);
        var bceClamped = BCELoss.Inline(Tensor(new long[] { 2L }, 0f, 1f), Tensor(new long[] { 2L }, 0f, 1f));
        var bceWrong = BCELoss.Inline(Tensor(new long[] { 1L }, 0f), Tensor(new long[] { 1L }, 1f));
        var bwlHi = BCEWithLogitsLoss.Inline(Tensor(new long[] { 1L }, 100f), Tensor(new long[] { 1L }, 1f));
        var bwlLo = BCEWithLogitsLoss.Inline(Tensor(new long[] { 1L }, -100f), Tensor(new long[] { 1L }, 1f));
        var ce = CrossEntropyLoss.Inline(Tensor(new long[] { 1L, 3L }, 1f, 2f, 3f), Vector(0L));

        var ok = Within((quad - Scalar(0.125f)).Abs(), 1e-5f)
               + Within((lin - Scalar(1.5f)).Abs(), 1e-5f)
               + Within(bceClamped, 1e-3f) + AtLeastZero(bceClamped)
               + Within((bceWrong - Scalar(16.1181f)).Abs(), 0.01f)
               + Within(bwlHi, 1e-3f) + AtLeastZero(bwlHi)
               + Within((bwlLo - Scalar(100f)).Abs(), 0.01f)
               + Within((ce - Scalar(2.4076059f)).Abs(), 1e-4f);
        return ok > Scalar(8L);   // all 9 ok-bits required
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// LogCoshLoss closed-form + numerical-stability checks (mirrors
/// NNLossClosedFormChecks / NNLossEdgeCaseChecks). All inputs are in-module
/// constants; the runtime input <paramref name="t"/> is only folded in as a
/// zero-scaled touch so AutoTest.AdvancedTestGraph has a graph input to drive.
/// Per-element loss is <c>log(cosh(pred − target))</c>. Re-derived (double):
///   d=[0,1] (pred=[0,1], target=[0,0]): per-elem [logcosh0, logcosh1] = [0, 0.43378083];
///     Inline (mean) = 0.21689042; Reduced(..,Sum) = 0.43378083;
///   mid d=2 (pred=[2], target=[0]): logcosh2 = log(cosh 2) = 1.3250027 (pins the curve);
///   stability d=100 (pred=[100], target=[0]): logcosh100 ≈ 100 − log2 = 99.30685 and
///     MUST be finite (the naive log((e^d+e^−d)/2) overflows for |d|≳89 in float32).
/// The stability/PerElement element values use the NaN-safe Within/IfElse(1,0)
/// ok-counting idiom (a NaN fails every comparison, so it can never slip through).
/// </summary>
[Module]
public partial class NNLogCoshLossChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        // d = [0, 1]: pred=[0,1], target=[0,0].
        var pred01 = Tensor(new long[] { 2L }, 0f, 1f);
        var zero2 = Tensor(new long[] { 2L }, 0f, 0f);

        var mean = LogCoshLoss.Inline(pred01, zero2);                                  // 0.21689042
        var sum = LogCoshLoss.Reduced(pred01, zero2, reduction: LossReduction.Sum);    // 0.43378083

        // PerElement → [0, 0.43378083]; check each element via Slice.
        var per = LogCoshLoss.PerElement(pred01, zero2);   // [2]
        var per0 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per1 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Mid d=2 (pred=[2], target=[0]): log(cosh 2) = 1.3250027.
        var mid = LogCoshLoss.Inline(Tensor(new long[] { 1L }, 2f), Tensor(new long[] { 1L }, 0f));

        // Stability d=100 (pred=[100], target=[0]): ≈ 100 − log2 = 99.30685, finite (no overflow/NaN).
        var stab = LogCoshLoss.Inline(Tensor(new long[] { 1L }, 100f), Tensor(new long[] { 1L }, 0f));

        var ok = Within((mean - Scalar(0.21689042f)).Abs(), 1e-5f)
               + Within((sum - Scalar(0.43378083f)).Abs(), 1e-5f)
               + Within(per0.Abs(), 1e-5f) + AtLeastZero(per0.Abs())
               + Within((per1 - Scalar(0.43378083f)).Abs(), 1e-5f)
               + Within((mid - Scalar(1.3250027f)).Abs(), 1e-4f)
               + Within((stab - Scalar(99.30685f)).Abs(), 1e-2f) + AtLeastZero(stab);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(8L);   // all 8 closed-form ok-bits + touch (9 total)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// PoissonNLLLoss closed-form + knob + NaN-safety checks (mirrors
/// NNLossClosedFormChecks / NNLossEdgeCaseChecks). All inputs are in-module
/// constants; the runtime input <paramref name="t"/> is only folded in as a
/// zero-scaled touch. Re-derived (double):
///   logInput=true (default), per-elem <c>exp(pred) − target·pred</c>:
///     pred=[0,1], target=[1,2] → [exp0−1·0, exp1−2·1] = [1, e−2 = 0.71828183];
///     Inline (mean) = 0.85914091; Reduced(..,Sum) = 1.71828183;
///   logInput=false, per-elem <c>pred − target·log(pred+eps)</c>:
///     pred=[1,2], target=[1,2], eps=1e-8 → [1−log(1+eps), 2−2·log(2+eps)]
///     = [0.99999999, 0.61370562]; mean = 0.80685281;
///   full=true Stirling on target=[0,1,3] with pred=[0,0,0] (base = exp0−t·0 = 1 each):
///     Stirling adds <c>t·ln t − t + 0.5·ln(2π·t)</c> for t>1, else 0 →
///     per-elem [1+0, 1+0, 1+1.7640816] = [1, 1, 2.7640816].
///     CRITICAL NaN-safety gate: the target=0 element must be FINITE (no NaN from
///     the clamped 0·log lane — the impl clamps tSafe=max(target,1) inside the logs).
/// The full/stability element values use the NaN-safe Within/IfElse(1,0) idiom.
/// </summary>
[Module]
public partial class NNPoissonNLLLossChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        // logInput=true (default): pred=[0,1], target=[1,2].
        var predLog = Tensor(new long[] { 2L }, 0f, 1f);
        var tgtLog = Tensor(new long[] { 2L }, 1f, 2f);

        var mean = PoissonNLLLoss.Inline(predLog, tgtLog);                                 // 0.85914091
        var sum = PoissonNLLLoss.Reduced(predLog, tgtLog, reduction: LossReduction.Sum);   // 1.71828183

        // PerElement → [1, 0.71828183]; check each element via Slice.
        var per = PoissonNLLLoss.PerElement(predLog, tgtLog);   // [2]
        var per0 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per1 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // logInput=false: pred=[1,2] (>0), target=[1,2], default eps=1e-8. mean = 0.80685281.
        var predRate = Tensor(new long[] { 2L }, 1f, 2f);
        var noLogMean = PoissonNLLLoss.Reduced(predRate, tgtLog, logInput: false, reduction: LossReduction.Mean);

        // full=true Stirling on target=[0,1,3], pred=[0,0,0] (base = exp0 − t·0 = 1 each).
        var predZ = Tensor(new long[] { 3L }, 0f, 0f, 0f);
        var tgtF = Tensor(new long[] { 3L }, 0f, 1f, 3f);
        var perFull = PoissonNLLLoss.PerElement(predZ, tgtF, logInput: true, full: true);   // [1, 1, 2.7640816]
        var full0 = ((Tensor<float32>)OnnxOp.Slice(perFull, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var full1 = ((Tensor<float32>)OnnxOp.Slice(perFull, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var full2 = ((Tensor<float32>)OnnxOp.Slice(perFull, Vector(2L), Vector(3L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((mean - Scalar(0.85914091f)).Abs(), 1e-5f)
               + Within((sum - Scalar(1.71828183f)).Abs(), 1e-5f)
               + Within((per0 - Scalar(1f)).Abs(), 1e-5f)
               + Within((per1 - Scalar(0.71828183f)).Abs(), 1e-5f)
               + Within((noLogMean - Scalar(0.80685281f)).Abs(), 1e-5f)
               // full=true: target=0 → finite 1 (the NaN-safety gate); target=1 → 1; target=3 → 2.7640816.
               + Within((full0 - Scalar(1f)).Abs(), 1e-4f) + AtLeastZero(full0)
               + Within((full1 - Scalar(1f)).Abs(), 1e-4f)
               + Within((full2 - Scalar(2.7640816f)).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(9L);   // all 9 ok-bits + touch
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// HingeLoss closed-form / mode coverage (mirrors NNLogCoshLossChecks /
/// NNPoissonNLLLossChecks). All inputs are in-module constants; the runtime
/// input <paramref name="t"/> is only folded in as a zero-scaled touch so
/// AutoTest.AdvancedTestGraph has a graph input to drive. Per-element loss is
/// <c>relu(1 − targets·predictions)</c> with targets ∈ {−1,+1}. Re-derived
/// (double):
///   pred=[0.5,−0.5,2], target=[1,1,1]: 1−t·p=[0.5,1.5,−1] → relu → [0.5,1.5,0];
///     Inline (mean) = 0.66666667; Reduced(..,Sum) = 2.0; PerElement = [0.5,1.5,0];
///   negative-class pred=[−2], target=[−1]: relu(1−(−1)(−2)) = relu(−1) = 0;
///   exact-margin pred=[1], target=[1]: relu(1−1) = relu(0) = 0.
/// The per-element / zero-valued checks use the NaN-safe Within/IfElse(1,0)
/// ok-counting idiom (a NaN fails every comparison, so it can never slip through).
/// </summary>
[Module]
public partial class NNHingeLossChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        // pred=[0.5,−0.5,2], target=[1,1,1] → per-elem [0.5,1.5,0].
        var pred = Tensor(new long[] { 3L }, 0.5f, -0.5f, 2f);
        var tgt = Tensor(new long[] { 3L }, 1f, 1f, 1f);

        var mean = HingeLoss.Inline(pred, tgt);                                  // 0.66666667
        var sum = HingeLoss.Reduced(pred, tgt, reduction: LossReduction.Sum);    // 2.0

        // PerElement → [0.5, 1.5, 0]; check each element via Slice.
        var per = HingeLoss.PerElement(pred, tgt);   // [3]
        var per0 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per1 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per2 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(2L), Vector(3L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Negative-class: pred=[−2], target=[−1] → relu(1−2) = 0.
        var neg = HingeLoss.Inline(Tensor(new long[] { 1L }, -2f), Tensor(new long[] { 1L }, -1f));

        // Exact-margin edge: pred=[1], target=[1] → relu(0) = 0.
        var edge = HingeLoss.Inline(Tensor(new long[] { 1L }, 1f), Tensor(new long[] { 1L }, 1f));

        var ok = Within((mean - Scalar(0.66666667f)).Abs(), 1e-5f)
               + Within((sum - Scalar(2f)).Abs(), 1e-5f)
               + Within((per0 - Scalar(0.5f)).Abs(), 1e-5f)
               + Within((per1 - Scalar(1.5f)).Abs(), 1e-5f)
               + Within(per2.Abs(), 1e-5f) + AtLeastZero(per2)
               + Within(neg.Abs(), 1e-5f) + AtLeastZero(neg)
               + Within(edge.Abs(), 1e-5f) + AtLeastZero(edge);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(10L);   // all 10 ok-bits + touch (11 total)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// SquaredHingeLoss closed-form / mode coverage (mirrors NNHingeLossChecks).
/// All inputs are in-module constants; the runtime input <paramref name="t"/>
/// is a zero-scaled touch. Per-element loss is
/// <c>relu(1 − targets·predictions)²</c> with targets ∈ {−1,+1}. Re-derived
/// (double):
///   pred=[0.5,−0.5,2], target=[1,1,1]: relu(1−t·p)=[0.5,1.5,0] → square → [0.25,2.25,0];
///     Inline (mean) = 0.83333333; Reduced(..,Sum) = 2.5; PerElement = [0.25,2.25,0];
///   Keras cross-check target=[−1,1,1], pred=[0.6,−0.7,−0.5]: t·p=[−0.6,−0.7,−0.5],
///     relu(1−t·p)=[1.6,1.7,1.5] → square → [2.56,2.89,2.25], mean = 2.56666667.
/// The per-element / zero-valued checks use the NaN-safe Within/IfElse(1,0) idiom.
/// </summary>
[Module]
public partial class NNSquaredHingeLossChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        // pred=[0.5,−0.5,2], target=[1,1,1] → per-elem [0.25,2.25,0].
        var pred = Tensor(new long[] { 3L }, 0.5f, -0.5f, 2f);
        var tgt = Tensor(new long[] { 3L }, 1f, 1f, 1f);

        var mean = SquaredHingeLoss.Inline(pred, tgt);                                  // 0.83333333
        var sum = SquaredHingeLoss.Reduced(pred, tgt, reduction: LossReduction.Sum);    // 2.5

        // PerElement → [0.25, 2.25, 0]; check each element via Slice.
        var per = SquaredHingeLoss.PerElement(pred, tgt);   // [3]
        var per0 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per1 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per2 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(2L), Vector(3L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // Keras cross-check: target=[−1,1,1], pred=[0.6,−0.7,−0.5] → mean 2.56666667.
        var kerasPred = Tensor(new long[] { 3L }, 0.6f, -0.7f, -0.5f);
        var kerasTgt = Tensor(new long[] { 3L }, -1f, 1f, 1f);
        var keras = SquaredHingeLoss.Inline(kerasPred, kerasTgt);   // 2.56666667

        var ok = Within((mean - Scalar(0.83333333f)).Abs(), 1e-5f)
               + Within((sum - Scalar(2.5f)).Abs(), 1e-5f)
               + Within((per0 - Scalar(0.25f)).Abs(), 1e-5f)
               + Within((per1 - Scalar(2.25f)).Abs(), 1e-5f)
               + Within(per2.Abs(), 1e-5f) + AtLeastZero(per2)
               + Within((keras - Scalar(2.56666667f)).Abs(), 1e-5f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(7L);   // all 7 ok-bits + touch (8 total)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// BinaryFocalLoss closed-form / knob coverage (mirrors NNHingeLossChecks).
/// All inputs are in-module constants; the runtime input <paramref name="t"/>
/// is a zero-scaled touch. Per-element loss is <c>α_t·(1−p_t)^γ·ce</c> over
/// logits with binary targets ∈ {0,1}; <c>ce</c> is BCE-with-logits,
/// <c>p = σ(x)</c>, <c>p_t = p·t + (1−p)·(1−t)</c>, <c>α_t = α·t + (1−α)·(1−t)</c>.
/// Re-derived (double, logit = 0 ⇒ ce = ln2 = 0.69314718, p = p_t = 0.5):
///   Inline defaults (α=0.25,γ=2), logit=0,t=1: ln2·(1−0.5)²·0.25 = ln2·0.25·0.25
///     = 0.04332170;
///   γ=0 (collapses to α-weighted BCE), logit=0,t=1,α=0.25: ln2·1·0.25 = 0.17328680;
///   t=0, logit=0,α=0.25,γ=2: p_t=1−0.5=0.5, (1−0.5)²=0.25, α_t=1−0.25=0.75
///     ⇒ ln2·0.25·0.75 = 0.12997010;
///   α=−1 sentinel (no α-weighting), logit=0,t=1,γ=2: ln2·0.25·1 = 0.17328680.
/// Reduced/PerElement carry the α/γ knobs. The closed-form checks use the
/// NaN-safe Within/IfElse(1,0) idiom.
/// </summary>
[Module]
public partial class NNBinaryFocalLossChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var logit0 = Tensor(new long[] { 1L }, 0f);
        var tgt1 = Tensor(new long[] { 1L }, 1f);
        var tgt0 = Tensor(new long[] { 1L }, 0f);

        // Inline defaults (α=0.25, γ=2), logit=0, t=1 → 0.04332170.
        var inline = BinaryFocalLoss.Inline(logit0, tgt1);

        // γ=0 collapses to α-weighted BCE: logit=0, t=1, α=0.25 → 0.17328680.
        var gamma0 = BinaryFocalLoss.Reduced(logit0, tgt1, alpha: 0.25f, gamma: 0f, reduction: LossReduction.Mean);

        // t=0, α=0.25, γ=2: α_t=0.75 → 0.12997010. Use PerElement (single element).
        var perT0 = BinaryFocalLoss.PerElement(logit0, tgt0, alpha: 0.25f, gamma: 2f);   // [1]
        var t0 = perT0.Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // α=−1 sentinel disables α-weighting: logit=0, t=1, γ=2 → 0.17328680.
        var alphaOff = BinaryFocalLoss.Reduced(logit0, tgt1, alpha: -1f, gamma: 2f, reduction: LossReduction.Mean);

        var ok = Within((inline - Scalar(0.04332170f)).Abs(), 1e-5f) + AtLeastZero(inline)
               + Within((gamma0 - Scalar(0.17328680f)).Abs(), 1e-5f)
               + Within((t0 - Scalar(0.12996510f)).Abs(), 1e-5f) + AtLeastZero(t0)
               + Within((alphaOff - Scalar(0.17328680f)).Abs(), 1e-5f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(6L);   // all 6 ok-bits + touch (7 total)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

// ---------------------------------------------------------------------------
// Recurrent.RNN (vanilla/Elman) helper coverage — design §7 (rnn/design.md).
// Each self-checking [Module] returns a Scalar<bit> that AutoTest.AdvancedTestGraph
// requires to be true: it computes (y, hN) = Recurrent.RNN(x, …) and a hand-built
// reference from a SEEDED OnnxOp.Rnn with the SAME-shape RecurrentUniform.Init
// W/R/bias (so the seeded inits materialize identically — the same idiom as
// NNConv2dMatchesStaticConv / the Convolution helper checks above), then asserts
// a relative-L1 match. The reference reproduces the helper's internal op call:
// W [D,H,in], R [D,H,H], B = concat(bias[D,H], zeros[D,H]) on axis 1, and the
// Y [L,D,N,H] -> [L,N,D,H] (transpose 0,2,1,3) -> [L,N,D*H] reshape. RNN has no
// QEE step values, so value correctness comes from the ORT backend inside
// AdvancedTestGraph (note [2] of the design). The relu / bidirectional BPTT-throws
// guards live as [Fact]s in NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------

internal static class RnnRefHelpers
{
    /// <summary>The owned bias packed exactly as the helper does: B = concat(bias[D,H], zeros[D,H]) on axis 1.</summary>
    public static Tensor<float32> PackedBias(Scalar<int64> d, Scalar<int64> h)
    {
        var biasParam = RecurrentUniform.Init([d, h], h);         // [D, H]
        var rbZeros = TensorFill((Vector<int64>)[d, h], 0.0f);    // [D, H]
        return biasParam.Concat(1L, rbZeros);                     // [D, 2H]
    }

    /// <summary>Hand-built single-layer reference op output, reshaped to PyTorch [L, N, D*H], with the
    /// same SAME-shape seeded W/R/(optional B) as the helper. Mirrors Recurrent.RNN's internal op call.</summary>
    public static (Tensor<float32> y, Tensor<float32> hN) RefOp(
        Tensor<float32> x, long hiddenSize, RNNDirection onnxDir, string[]? activations, bool bias)
    {
        long dl = onnxDir == RNNDirection.Bidirectional ? 2L : 1L;
        var d = Scalar(dl);
        var h = Scalar(hiddenSize);
        Scalar<int64> inSize = x.DimTensor(-1);

        var w = RecurrentUniform.Init([d, h, inSize], h);   // [D, H, in]
        var r = RecurrentUniform.Init([d, h, h], h);        // [D, H, H]
        Tensor<float32>? b = bias ? PackedBias(d, h) : (Tensor<float32>?)null;

        var (yVar, yhVar) = OnnxOp.Rnn(x, w, r, b, null, null,
            null, null, activations, null, onnxDir, hiddenSize, false);
        var yLayer = (Tensor<float32>)yVar;     // [L, D, N, H]
        var yh = (Tensor<float32>)yhVar;        // [D, N, H]

        var l = yLayer.DimTensor(0);
        var n = yLayer.DimTensor(2);
        var yLNDH = yLayer.Transpose(0L, 2L, 1L, 3L);                      // [L, N, D, H]
        var y = yLNDH.Reshape((Vector<int64>)[l, n, d * h]);              // [L, N, D*H]
        return (y, yh);
    }

    /// <summary>Relative-L1 penalty (diff / (1 + |ref|)) for one tensor pair; ~0 when matching.</summary>
    public static Scalar<float32> RelL1(Tensor<float32> a, Tensor<float32> b)
    {
        var diff = (a - b).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var scale = Scalar(1f) + b.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return diff / scale;
    }
}

/// <summary>§7-2 Matches the core op (forward, tanh): Recurrent.RNN(x, H) equals a hand-built
/// OnnxOp.Rnn(x, W, R, B=concat(bias,zeros), …, Forward, H, layout:false) with the same seeded
/// W/R/B — both y AND hN, relative-L1. Pins the bias packing and the [L,D,N,H]→[L,N,D*H] reshape.</summary>
[Module]
public partial class RnnMatchesCoreOpForwardTanh
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L);
        var (yRef, hRef) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Forward, null, bias: true);
        return RnnRefHelpers.RelL1(y, yRef) + RnnRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-1/§7-2 (batchFirst) Matches the core op with batchFirst input: Recurrent.RNN(batchFirst:true)
/// on [N, L, in] equals the op fed the transposed [L, N, in], with the final Y transposed back to [N, L, D*H].
/// Pins the in-graph transpose around the layout=0 op.</summary>
[Module]
public partial class RnnMatchesCoreOpBatchFirst
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, batchFirst: true);
        var xT = x.Transpose(1L, 0L, 2L);                                  // [L, N, in]
        var (yRefLN, hRef) = RnnRefHelpers.RefOp(xT, 3L, RNNDirection.Forward, null, bias: true);
        var yRef = yRefLN.Transpose(1L, 0L, 2L);                           // [N, L, D*H]
        return RnnRefHelpers.RelL1(y, yRef) + RnnRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-1 Single-step recurrence anchor: L=1, h_0=0 ⇒ y[0] == tanh(W·x_0 + bias) in closed form
/// (R is unused at step 0), and hN == y[0]. The manual closed form uses the SAME seeded W/bias as the
/// helper (the op's [D,H,in] W contracts the input axis: tanh over W·x_0 with bias broadcast).</summary>
[Module]
public partial class RnnSingleStepAnchorTanh
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 2L);   // y [1, N, H], hN [1, N, H]

        // Manual: h_1 = tanh(W·x_0 + bias). W is [1, H, in]; squeeze D=1.
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), Scalar(2L), inSize], Scalar(2L));    // [1, H, in]
        var biasParam = RecurrentUniform.Init([Scalar(1L), Scalar(2L)], Scalar(2L));    // [1, H]
        var wHI = w.Reshape([Scalar(2L), inSize]);                          // [H, in]
        var biasH = biasParam.Reshape([Scalar(2L)]);                        // [H]

        var x0 = x.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([x.DimTensor(1), inSize]); // [N, in]
        var preact = x0.MatMul(wHI.Transpose(1L, 0L));                      // [N, H]
        var manualH1 = (preact + biasH).Tanh();                            // [N, H] (bias broadcasts)

        var yStep0 = y.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([x.DimTensor(1), Scalar(2L)]);
        var hNFlat = hN.Reshape([x.DimTensor(1), Scalar(2L)]);

        var pen = RnnRefHelpers.RelL1(yStep0, manualH1)        // y[0] == tanh(W·x_0 + b)
                + RnnRefHelpers.RelL1(hNFlat, yStep0);          // hN == y[0]
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-3 relu nonlinearity (forward only): Recurrent.RNN(Relu) equals the op with
/// activations:["Relu"] and the same seeded W/R/B. Forward-value check only (relu RNN BPTT throws
/// AD003 — pinned separately in NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class RnnReluMatchesCoreOpForward
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, nonlinearity: RnnNonlinearity.Relu);
        var (yRef, hRef) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Forward, new[] { "Relu" }, bias: true);
        return RnnRefHelpers.RelL1(y, yRef) + RnnRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-4 bias on/off: bias:false equals the no-B op; bias:true equals the concat(bias,zeros) op.
/// Both compared against the matching seeded reference (same W/R; B present iff bias).</summary>
[Module]
public partial class RnnBiasOnOffMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yNoB, hNoB) = Recurrent.RNN(x, hiddenSize: 3L, bias: false);
        var (yNoBRef, hNoBRef) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Forward, null, bias: false);

        var (yB, hB) = Recurrent.RNN(x, hiddenSize: 3L, bias: true);
        var (yBRef, hBRef) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Forward, null, bias: true);

        var pen = RnnRefHelpers.RelL1(yNoB, yNoBRef) + RnnRefHelpers.RelL1(hNoB, hNoBRef)
                + RnnRefHelpers.RelL1(yB, yBRef) + RnnRefHelpers.RelL1(hB, hBRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-5 numLayers stacking: numLayers:2 (forward, tanh) equals a hand-built two-op stack
/// feeding layer-0 Y → layer-1 X. Asserts y and the stacked [2,N,H] hN both match. Layer-1's input
/// size is D·H = H (D=1), which the helper passes as Scalar(d·H); the reference reads it from the
/// reshaped layer-0 Y's last axis (same value), so the seeded inits coincide.</summary>
[Module]
public partial class RnnNumLayersStackMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, numLayers: 2);

        // Layer 0 over x.
        var (y0, hN0) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Forward, null, bias: true);  // y0 [L, N, H]
        // Layer 1 over layer-0's Y (its in == D·H == H).
        var (y1, hN1) = RnnRefHelpers.RefOp(y0, 3L, RNNDirection.Forward, null, bias: true);
        var hRef = hN0.Concat(0L, hN1);   // [2, N, H]

        var pen = RnnRefHelpers.RelL1(y, y1) + RnnRefHelpers.RelL1(hN, hRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-6 direction Reverse (trainable): Recurrent.RNN(Reverse) equals the op direction:Reverse
/// with the same seeded W/R/B (both y and hN), relative-L1.</summary>
[Module]
public partial class RnnReverseMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, direction: RnnDirection.Reverse);
        var (yRef, hRef) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Reverse, null, bias: true);
        return RnnRefHelpers.RelL1(y, yRef) + RnnRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-6 direction Bidirectional (forward inference only): Recurrent.RNN(Bidirectional) equals
/// the op direction:Bidirectional (same seeded [2,…] W/R/B); AND y's last axis is 2H, hN is [2, N, H].
/// Forward-value only (bidirectional BPTT throws AD003 — pinned in NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class RnnBidirectionalMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        long hVal = 3L;
        var (y, hN) = Recurrent.RNN(x, hiddenSize: hVal, direction: RnnDirection.Bidirectional);
        var (yRef, hRef) = RnnRefHelpers.RefOp(x, hVal, RNNDirection.Bidirectional,
            new[] { "Tanh", "Tanh" }, bias: true);

        // Shape contract: y last axis == 2H, hN leading axis == 2 (D=2, numLayers=1).
        var lastAxisOk = (y.DimTensor(-1) - Scalar(2L * hVal)).Abs().Cast<float32>();
        var hLeadingOk = (hN.DimTensor(0) - Scalar(2L)).Abs().Cast<float32>();

        var pen = RnnRefHelpers.RelL1(y, yRef) + RnnRefHelpers.RelL1(hN, hRef)
                + lastAxisOk + hLeadingOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-7 batchFirst equivalence: RNN(batchFirst:true, [N,L,in]) equals
/// RNN(batchFirst:false, transpose([L,N,in])) with the result transposed back. Pins the in-graph
/// transpose + layout=0-always choice independently of the op reference. hN is batch-second in both
/// (PyTorch keeps hN [D·numLayers, N, H] regardless of batch_first), so it matches directly.</summary>
[Module]
public partial class RnnBatchFirstEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (yBF, hBF) = Recurrent.RNN(x, hiddenSize: 3L, batchFirst: true);     // y [N, L, D*H]

        var xT = x.Transpose(1L, 0L, 2L);                                        // [L, N, in]
        var (ySF, hSF) = Recurrent.RNN(xT, hiddenSize: 3L, batchFirst: false);   // y [L, N, D*H]
        var ySFasBF = ySF.Transpose(1L, 0L, 2L);                                 // [N, L, D*H]

        var pen = RnnRefHelpers.RelL1(yBF, ySFasBF) + RnnRefHelpers.RelL1(hBF, hSF);
        return pen < Scalar(1e-5f);
    }
}

/// <summary>§7-8 state contract: for a forward single-layer RNN, hN == y[-1] (the last step's hidden
/// state) and equals the op's Y_h; hN shape is [D·numLayers, N, H] == [1, N, H]. Pins the (y, hN)
/// return relationship.</summary>
[Module]
public partial class RnnStateContractForwardSingleLayer
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L);   // y [L, N, H], hN [1, N, H]
        var (_, hRef) = RnnRefHelpers.RefOp(x, 3L, RNNDirection.Forward, null, bias: true);

        // y[-1]: last step along axis 0, flattened to [N, H]; hN flattened to [N, H].
        var lastStep = y.Slice(Vector(-1L), Vector(System.Int64.MaxValue), Vector(0L))
            .Reshape([x.DimTensor(1), Scalar(3L)]);                  // [N, H]
        var hNFlat = hN.Reshape([x.DimTensor(1), Scalar(3L)]);      // [N, H] (D·numLayers == 1)

        // Shape contract: hN leading axis == D·numLayers == 1.
        var hLeadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();

        var pen = RnnRefHelpers.RelL1(hNFlat, lastStep)    // hN == y[-1]
                + RnnRefHelpers.RelL1(hN, hRef)             // hN == op's Y_h
                + hLeadingOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-9 trainable-corner FD grad check: a forward, tanh, single-layer Recurrent.RNN (the
/// ONLY autodiff-supported corner — design note [3]) on the loss ΣY + Σh_n, with the analytic
/// gradient (via AutoGrad) FD-checked against a two-sided directional derivative on ORT's own
/// forward. Mirrors AutoGradRnnReverseCheck — the proven AutoGrad-graph gradient path, distinct
/// from the TrainingRig memory-aware-scheduler path (the LSTM/GRU rig train-step tests exercise
/// that). x[0,0,0] is the probed scalar over a length-3 sequence.</summary>
[Module]
public partial class RnnForwardTanhGradCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        Func<Scalar<float32>, Scalar<float32>> f = v =>
        {
            var x = RecurrentTestData.BuildX(v, 3);   // [3, 1, 2]
            var (y, hN) = Recurrent.RNN(x, hiddenSize: 2L);   // forward, tanh, single-layer
            return y.Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        };
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, f(xv));
        return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(xv, grad, f);
    }
}

/// <summary>§7-3 (BPTT throw) relu RNN gradient: a loss through Recurrent.RNN(Relu) must throw AD003 at
/// lowering (relu is a non-default activation; BPTT unsupported). Mirrors AutoGradRnnBidirectionalThrowCheck.
/// Never reached past AutoGrad — the AdvancedTestGraph call throws.</summary>
[Module]
public partial class RnnReluBpttThrowCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 2);   // [2, 1, 2]
        var (_, hN) = Recurrent.RNN(x, hiddenSize: 2L, nonlinearity: RnnNonlinearity.Relu);
        var loss = hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
        return grad.Abs() < Scalar(1e9f);   // never reached: AD003 at lowering
    }
}

/// <summary>§7-6 (BPTT throw) bidirectional RNN gradient: a loss through Recurrent.RNN(Bidirectional)
/// must throw AD003 at lowering (bidirectional BPTT unimplemented). Mirrors AutoGradRnnBidirectionalThrowCheck.</summary>
[Module]
public partial class RnnBidirectionalBpttThrowCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 2);   // [2, 1, 2]
        var (_, hN) = Recurrent.RNN(x, hiddenSize: 2L, direction: RnnDirection.Bidirectional);
        var loss = hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
        return grad.Abs() < Scalar(1e9f);   // never reached: AD003 at lowering
    }
}

// ---------------------------------------------------------------------------
// Recurrent.LSTM helper coverage — design §7 (lstm/design.md). Mirrors the
// Recurrent.RNN set above EXACTLY: each self-checking [Module] returns a
// Scalar<bit> that AutoTest.AdvancedTestGraph requires to be true. It computes
// (y, hN, cN) = Recurrent.LSTM(x, …) and a hand-built reference from a SEEDED
// OnnxOp.Lstm with the SAME-shape RecurrentUniform.Init W/R/bias (so the seeded
// inits materialize identically — the same idiom as RnnRefHelpers), then asserts
// a relative-L1 match on y AND hN AND cN. The reference reproduces the helper's
// internal op call: W [D,4H,in], R [D,4H,H], B = concat(bias[D,4H], zeros[D,4H])
// on axis 1 (→ [D,8H]), gate order ONNX-native i,o,f,c, default activations,
// peephole P = null, zeroed h_0/c_0, layout=0, and the Y [L,D,N,H] -> [L,N,D,H]
// (transpose 0,2,1,3) -> [L,N,D*H] reshape. LSTM has no QEE step values, so value
// correctness comes from the ORT backend inside AdvancedTestGraph. The
// bidirectional BPTT-throws guard lives as a [Fact] in NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------

internal static class LstmRefHelpers
{
    /// <summary>The owned bias packed exactly as the helper does: B = concat(bias[D,4H], zeros[D,4H]) on axis 1 → [D,8H].</summary>
    public static Tensor<float32> PackedBias(Scalar<int64> d, Scalar<int64> fourH, Scalar<int64> h)
    {
        var biasParam = RecurrentUniform.Init([d, fourH], h);          // [D, 4H]; bound keyed on H
        var rbZeros = TensorFill((Vector<int64>)[d, fourH], 0.0f);    // [D, 4H]
        return biasParam.Concat(1L, rbZeros);                          // [D, 8H]
    }

    /// <summary>Hand-built single-layer reference op output, reshaped to PyTorch [L, N, D*H], with the
    /// same SAME-shape seeded W/R/(optional B) as the helper. Mirrors Recurrent.LSTM's internal op call.</summary>
    public static (Tensor<float32> y, Tensor<float32> hN, Tensor<float32> cN) RefOp(
        Tensor<float32> x, long hiddenSize, LSTMDirection onnxDir, bool bias)
    {
        long dl = onnxDir == LSTMDirection.Bidirectional ? 2L : 1L;
        var d = Scalar(dl);
        var h = Scalar(hiddenSize);
        var fourH = Scalar(4L * hiddenSize);
        Scalar<int64> inSize = x.DimTensor(-1);

        var w = RecurrentUniform.Init([d, fourH, inSize], h);   // [D, 4H, in]
        var r = RecurrentUniform.Init([d, fourH, h], h);        // [D, 4H, H]
        Tensor<float32>? b = bias ? PackedBias(d, fourH, h) : (Tensor<float32>?)null;

        var (yVar, yhVar, ycVar) = OnnxOp.Lstm(x, w, r, b, null, null, null, null,
            null, null, null, null, onnxDir, hiddenSize, false, false);
        var yLayer = (Tensor<float32>)yVar;     // [L, D, N, H]
        var yh = (Tensor<float32>)yhVar;        // [D, N, H]
        var yc = (Tensor<float32>)ycVar;        // [D, N, H]

        var l = yLayer.DimTensor(0);
        var n = yLayer.DimTensor(2);
        var yLNDH = yLayer.Transpose(0L, 2L, 1L, 3L);                      // [L, N, D, H]
        var y = yLNDH.Reshape((Vector<int64>)[l, n, d * h]);              // [L, N, D*H]
        return (y, yh, yc);
    }

    /// <summary>Relative-L1 penalty (diff / (1 + |ref|)) for one tensor pair; ~0 when matching.</summary>
    public static Scalar<float32> RelL1(Tensor<float32> a, Tensor<float32> b)
        => RnnRefHelpers.RelL1(a, b);
}

/// <summary>§7-1 Matches the core op (forward): Recurrent.LSTM(x, H) equals a hand-built
/// OnnxOp.Lstm(x, W, R, B=concat(bias,zeros), …, Forward, H, layout:false) with the same seeded
/// W/R/B — y AND hN AND cN, relative-L1. Pins the i,o,f,c gate packing, bias packing, and the
/// [L,D,N,H]→[L,N,D*H] reshape.</summary>
[Module]
public partial class LstmMatchesCoreOpForward
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L);
        var (yRef, hRef, cRef) = LstmRefHelpers.RefOp(x, 3L, LSTMDirection.Forward, bias: true);
        return LstmRefHelpers.RelL1(y, yRef) + LstmRefHelpers.RelL1(hN, hRef)
             + LstmRefHelpers.RelL1(cN, cRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-1 (batchFirst) Matches the core op with batchFirst input: Recurrent.LSTM(batchFirst:true)
/// on [N, L, in] equals the op fed the transposed [L, N, in], with the final Y transposed back to
/// [N, L, D*H]. hN/cN stay [D·numLayers, N, H]. Pins the in-graph transpose around the layout=0 op.</summary>
[Module]
public partial class LstmMatchesCoreOpBatchFirst
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L, batchFirst: true);
        var xT = x.Transpose(1L, 0L, 2L);                                  // [L, N, in]
        var (yRefLN, hRef, cRef) = LstmRefHelpers.RefOp(xT, 3L, LSTMDirection.Forward, bias: true);
        var yRef = yRefLN.Transpose(1L, 0L, 2L);                           // [N, L, D*H]
        return LstmRefHelpers.RelL1(y, yRef) + LstmRefHelpers.RelL1(hN, hRef)
             + LstmRefHelpers.RelL1(cN, cRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-2 Single-step gate anchor: L=1, h_0=c_0=0 ⇒ closed-form i=σ(W_i·x_0+b_i),
/// c̃=tanh(W_c·x_0+b_c), C_1=i⊙c̃, H_1=o⊙tanh(C_1) with o=σ(W_o·x_0+b_o), from the SAME seeded
/// W/bias sliced into the i,o,f,c gate blocks (the op's [D,4H,in] W contracts the input axis; R is
/// unused at step 0). Pins the gate packing ORDER and the equation — a wrong i,o,f,c↔i,f,g,o swap
/// fails this. H=2.</summary>
[Module]
public partial class LstmSingleStepGateAnchor
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 2L);   // y [1,N,H], hN/cN [1,N,H]

        long hv = 2L;
        Scalar<int64> inSize = x.DimTensor(-1);
        var h = Scalar(hv);
        var fourH = Scalar(4L * hv);

        // Same seeded W/bias as the helper (bound keyed on H).
        var w = RecurrentUniform.Init([Scalar(1L), fourH, inSize], h);   // [1, 4H, in]
        var biasParam = RecurrentUniform.Init([Scalar(1L), fourH], h);   // [1, 4H]
        var w4HI = w.Reshape([fourH, inSize]);                           // [4H, in]
        var bias4H = biasParam.Reshape([fourH]);                         // [4H]

        // Pre-activation z = W·x_0 + b, shape [N, 4H] (gate blocks i,o,f,c stacked along axis 1).
        var x0 = x.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([x.DimTensor(1), inSize]); // [N, in]
        var z = x0.MatMul(w4HI.Transpose(1L, 0L)) + bias4H;              // [N, 4H]

        // Slice the four H-blocks in ONNX i,o,f,c order along axis 1.
        Tensor<float32> Block(long idx) => z.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L)); // [N, H]
        var i = Block(0L).Sigmoid();   // input gate
        var o = Block(1L).Sigmoid();   // output gate
        // f = Block(2) (forget) — unused at step 0 since C_0 = 0.
        var cTilde = Block(3L).Tanh(); // cell candidate (g)

        var c1 = i * cTilde;           // C_1 = f⊙C_0 + i⊙c̃ = i⊙c̃  (C_0 = 0)
        var h1 = o * c1.Tanh();        // H_1 = o⊙tanh(C_1)

        var hNFlat = hN.Reshape([x.DimTensor(1), h]);   // [N, H]
        var cNFlat = cN.Reshape([x.DimTensor(1), h]);   // [N, H]
        var yStep0 = y.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([x.DimTensor(1), h]);

        var pen = LstmRefHelpers.RelL1(cNFlat, c1)        // cN == C_1
                + LstmRefHelpers.RelL1(hNFlat, h1)        // hN == H_1
                + LstmRefHelpers.RelL1(yStep0, h1);       // y[0] == H_1 == hN
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-3 bias on/off: bias:false equals the no-B op; bias:true equals the concat(bias,zeros) op.
/// Both compared against the matching seeded reference (same W/R; B present iff bias) on y, hN, cN.</summary>
[Module]
public partial class LstmBiasOnOffMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yNoB, hNoB, cNoB) = Recurrent.LSTM(x, hiddenSize: 3L, bias: false);
        var (yNoBRef, hNoBRef, cNoBRef) = LstmRefHelpers.RefOp(x, 3L, LSTMDirection.Forward, bias: false);

        var (yB, hB, cB) = Recurrent.LSTM(x, hiddenSize: 3L, bias: true);
        var (yBRef, hBRef, cBRef) = LstmRefHelpers.RefOp(x, 3L, LSTMDirection.Forward, bias: true);

        var pen = LstmRefHelpers.RelL1(yNoB, yNoBRef) + LstmRefHelpers.RelL1(hNoB, hNoBRef)
                + LstmRefHelpers.RelL1(cNoB, cNoBRef)
                + LstmRefHelpers.RelL1(yB, yBRef) + LstmRefHelpers.RelL1(hB, hBRef)
                + LstmRefHelpers.RelL1(cB, cBRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-4 numLayers stacking: numLayers:2 (forward) equals a hand-built two-op stack feeding
/// layer-0 Y → layer-1 X. Asserts y, the stacked [2,N,H] hN, and the stacked [2,N,H] cN all match.
/// Layer-1's input size is D·H = H (D=1), which the helper passes as Scalar(d·H); the reference reads
/// it from the reshaped layer-0 Y's last axis (same value), so the seeded inits coincide.</summary>
[Module]
public partial class LstmNumLayersStackMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L, numLayers: 2);

        // Layer 0 over x.
        var (y0, hN0, cN0) = LstmRefHelpers.RefOp(x, 3L, LSTMDirection.Forward, bias: true);  // y0 [L, N, H]
        // Layer 1 over layer-0's Y (its in == D·H == H).
        var (y1, hN1, cN1) = LstmRefHelpers.RefOp(y0, 3L, LSTMDirection.Forward, bias: true);
        var hRef = hN0.Concat(0L, hN1);   // [2, N, H]
        var cRef = cN0.Concat(0L, cN1);   // [2, N, H]

        var pen = LstmRefHelpers.RelL1(y, y1) + LstmRefHelpers.RelL1(hN, hRef)
                + LstmRefHelpers.RelL1(cN, cRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-5 direction Reverse (trainable): Recurrent.LSTM(Reverse) equals the op direction:Reverse
/// with the same seeded W/R/B (y, hN and cN), relative-L1.</summary>
[Module]
public partial class LstmReverseMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L, direction: RnnDirection.Reverse);
        var (yRef, hRef, cRef) = LstmRefHelpers.RefOp(x, 3L, LSTMDirection.Reverse, bias: true);
        return LstmRefHelpers.RelL1(y, yRef) + LstmRefHelpers.RelL1(hN, hRef)
             + LstmRefHelpers.RelL1(cN, cRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-5 direction Bidirectional (forward inference only): Recurrent.LSTM(Bidirectional) equals
/// the op direction:Bidirectional (same seeded [2,…] W/R/B); AND y's last axis is 2H, hN/cN are
/// [2, N, H]. Forward-value only (bidirectional BPTT throws AD003 — pinned in NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class LstmBidirectionalMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        long hVal = 3L;
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: hVal, direction: RnnDirection.Bidirectional);
        var (yRef, hRef, cRef) = LstmRefHelpers.RefOp(x, hVal, LSTMDirection.Bidirectional, bias: true);

        // Shape contract: y last axis == 2H, hN/cN leading axis == 2 (D=2, numLayers=1).
        var lastAxisOk = (y.DimTensor(-1) - Scalar(2L * hVal)).Abs().Cast<float32>();
        var hLeadingOk = (hN.DimTensor(0) - Scalar(2L)).Abs().Cast<float32>();
        var cLeadingOk = (cN.DimTensor(0) - Scalar(2L)).Abs().Cast<float32>();

        var pen = LstmRefHelpers.RelL1(y, yRef) + LstmRefHelpers.RelL1(hN, hRef)
                + LstmRefHelpers.RelL1(cN, cRef) + lastAxisOk + hLeadingOk + cLeadingOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-6 batchFirst equivalence: LSTM(batchFirst:true, [N,L,in]) equals
/// LSTM(batchFirst:false, transpose([L,N,in])) with the result transposed back. Pins the in-graph
/// transpose + layout=0-always choice independently of the op reference. hN/cN are batch-second in
/// both (PyTorch keeps them [D·numLayers, N, H] regardless of batch_first), so they match directly.</summary>
[Module]
public partial class LstmBatchFirstEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (yBF, hBF, cBF) = Recurrent.LSTM(x, hiddenSize: 3L, batchFirst: true);     // y [N, L, D*H]

        var xT = x.Transpose(1L, 0L, 2L);                                              // [L, N, in]
        var (ySF, hSF, cSF) = Recurrent.LSTM(xT, hiddenSize: 3L, batchFirst: false);   // y [L, N, D*H]
        var ySFasBF = ySF.Transpose(1L, 0L, 2L);                                       // [N, L, D*H]

        var pen = LstmRefHelpers.RelL1(yBF, ySFasBF) + LstmRefHelpers.RelL1(hBF, hSF)
                + LstmRefHelpers.RelL1(cBF, cSF);
        return pen < Scalar(1e-5f);
    }
}

/// <summary>§7-7 state contract: for a forward single-layer LSTM, hN == y[-1] (the last step's hidden
/// state) and equals the op's Y_h; cN equals the op's Y_c; hN/cN shape is [D·numLayers, N, H] ==
/// [1, N, H]. Pins the (y, hN, cN) return relationship.</summary>
[Module]
public partial class LstmStateContractForwardSingleLayer
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L);   // y [L,N,H], hN/cN [1,N,H]
        var (_, hRef, cRef) = LstmRefHelpers.RefOp(x, 3L, LSTMDirection.Forward, bias: true);

        // y[-1]: last step along axis 0, flattened to [N, H]; hN/cN flattened to [N, H].
        var lastStep = y.Slice(Vector(-1L), Vector(System.Int64.MaxValue), Vector(0L))
            .Reshape([x.DimTensor(1), Scalar(3L)]);                  // [N, H]
        var hNFlat = hN.Reshape([x.DimTensor(1), Scalar(3L)]);      // [N, H] (D·numLayers == 1)

        // Shape contract: hN/cN leading axis == D·numLayers == 1.
        var hLeadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();
        var cLeadingOk = (cN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();

        var pen = LstmRefHelpers.RelL1(hNFlat, lastStep)    // hN == y[-1]
                + LstmRefHelpers.RelL1(hN, hRef)            // hN == op's Y_h
                + LstmRefHelpers.RelL1(cN, cRef)            // cN == op's Y_c
                + hLeadingOk + cLeadingOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-8 trainable-corner FD grad check: a forward, single-layer Recurrent.LSTM (a supported
/// autodiff corner — design §5.2) on the loss ΣY + Σh_n + Σc_n, with the analytic gradient (via
/// AutoGrad) FD-checked against a two-sided directional derivative on ORT's own forward. Mirrors
/// AutoGradLstmReverseCheck / RnnForwardTanhGradCheck. x[0,0,0] is the probed scalar over a length-3
/// sequence.</summary>
[Module]
public partial class LstmForwardGradCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        Func<Scalar<float32>, Scalar<float32>> f = v =>
        {
            var x = RecurrentTestData.BuildX(v, 3);   // [3, 1, 2]
            var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 2L);   // forward, single-layer
            return y.Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + cN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        };
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, f(xv));
        return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(xv, grad, f);
    }
}

/// <summary>§7-8 trainability rig model: a forward, single-layer Recurrent.LSTM reducing (y, hN, cN)
/// to a per-batch logit pair, for a TrainingRig FromScratch / TrainStep. The owned W/R/bias
/// differentiate end-to-end here (the trainable corner, design §5.2 — and the scheduler fix that
/// landed unblocks the recurrent rig path). Output [N, 2] so CrossEntropy has two logits.</summary>
[Module]
public partial class LstmForwardTrainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // input is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(input, hiddenSize: 4L);   // y [L,N,H], hN/cN [1,N,H]
        var n = input.DimTensor(1);

        // Per-batch features: sum y over time+hidden, sum (hN+cN) over hidden → [N], stack to [N, 2].
        Vector<int64> timeHidden = [Scalar(0L), Scalar(2L)];
        var yFeat = y.Reduce(ReduceKind.Sum, timeHidden, keepDims: false).Reshape([n, Scalar(1L)]);  // [N, 1]
        Vector<int64> layerHidden = [Scalar(0L), Scalar(2L)];
        var stateFeat = (hN + cN).Reduce(ReduceKind.Sum, layerHidden, keepDims: false)
            .Reshape([n, Scalar(1L)]);                                                                 // [N, 1]
        return yFeat.Concat(1L, stateFeat);   // [N, 2]
    }
}

/// <summary>§7-5 (BPTT throw) bidirectional LSTM gradient: a loss through Recurrent.LSTM(Bidirectional)
/// must throw AD003 at lowering (bidirectional BPTT unimplemented). Mirrors AutoGradRnnBidirectionalThrowCheck /
/// RnnBidirectionalBpttThrowCheck. Never reached past AutoGrad — the AdvancedTestGraph call throws.</summary>
[Module]
public partial class LstmBidirectionalBpttThrowCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 2);   // [2, 1, 2]
        var (_, hN, _) = Recurrent.LSTM(x, hiddenSize: 2L, direction: RnnDirection.Bidirectional);
        var loss = hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
        return grad.Abs() < Scalar(1e9f);   // never reached: AD003 at lowering
    }
}

// ---------------------------------------------------------------------------
// Recurrent.GRU helper coverage — design §7 (gru/design.md). Mirrors the
// Recurrent.LSTM set above EXACTLY (which itself mirrors the RNN set): each
// self-checking [Module] returns a Scalar<bit> that AutoTest.AdvancedTestGraph
// requires to be true. It computes (y, hN) = Recurrent.GRU(x, …) and a
// hand-built reference from a SEEDED OnnxOp.Gru with the SAME-shape
// RecurrentUniform.Init W/R/bias (so the seeded inits materialize identically —
// the same idiom as RnnRefHelpers / LstmRefHelpers), then asserts a relative-L1
// match on y AND hN. The reference reproduces the helper's internal op call:
// W [D,3H,in], R [D,3H,H], B = concat(bias[D,3H], zeros[D,3H]) on axis 1 (→
// [D,6H]), gate order ONNX-native z,r,h, default activations, zeroed h_0 (no
// cell state), layout=0, the SAME linearBeforeReset value, and the
// Y [L,D,N,H] -> [L,N,D,H] (transpose 0,2,1,3) -> [L,N,D*H] reshape. GRU has no
// QEE step values, so value correctness comes from the ORT backend inside
// AdvancedTestGraph. The GRU-specific addition over the LSTM/RNN sets is the
// linearBeforeReset both-forms check (GruLinearBeforeResetBothForms). The
// bidirectional BPTT-throws guard lives as a [Fact] in NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------

internal static class GruRefHelpers
{
    /// <summary>The owned bias packed exactly as the helper does: B = concat(bias[D,3H], zeros[D,3H]) on axis 1 → [D,6H].</summary>
    public static Tensor<float32> PackedBias(Scalar<int64> d, Scalar<int64> threeH, Scalar<int64> h)
    {
        var biasParam = RecurrentUniform.Init([d, threeH], h);         // [D, 3H]; bound keyed on H
        var rbZeros = TensorFill((Vector<int64>)[d, threeH], 0.0f);    // [D, 3H]
        return biasParam.Concat(1L, rbZeros);                          // [D, 6H]
    }

    /// <summary>Hand-built single-layer reference op output, reshaped to PyTorch [L, N, D*H], with the
    /// same SAME-shape seeded W/R/(optional B) as the helper and the SAME linearBeforeReset value.
    /// Mirrors Recurrent.GRU's internal op call.</summary>
    public static (Tensor<float32> y, Tensor<float32> hN) RefOp(
        Tensor<float32> x, long hiddenSize, GRUDirection onnxDir, bool bias, bool linearBeforeReset)
    {
        long dl = onnxDir == GRUDirection.Bidirectional ? 2L : 1L;
        var d = Scalar(dl);
        var h = Scalar(hiddenSize);
        var threeH = Scalar(3L * hiddenSize);
        Scalar<int64> inSize = x.DimTensor(-1);

        var w = RecurrentUniform.Init([d, threeH, inSize], h);   // [D, 3H, in]
        var r = RecurrentUniform.Init([d, threeH, h], h);        // [D, 3H, H]
        Tensor<float32>? b = bias ? PackedBias(d, threeH, h) : (Tensor<float32>?)null;

        var (yVar, yhVar) = OnnxOp.Gru(x, w, r, b, null, null,
            null, null, null, null, onnxDir, hiddenSize, false, linearBeforeReset);
        var yLayer = (Tensor<float32>)yVar;     // [L, D, N, H]
        var yh = (Tensor<float32>)yhVar;        // [D, N, H]

        var l = yLayer.DimTensor(0);
        var n = yLayer.DimTensor(2);
        var yLNDH = yLayer.Transpose(0L, 2L, 1L, 3L);                      // [L, N, D, H]
        var y = yLNDH.Reshape((Vector<int64>)[l, n, d * h]);              // [L, N, D*H]
        return (y, yh);
    }

    /// <summary>Relative-L1 penalty (diff / (1 + |ref|)) for one tensor pair; ~0 when matching.</summary>
    public static Scalar<float32> RelL1(Tensor<float32> a, Tensor<float32> b)
        => RnnRefHelpers.RelL1(a, b);
}

/// <summary>§7-1 Matches the core op (forward): Recurrent.GRU(x, H) equals a hand-built
/// OnnxOp.Gru(x, W, R, B=concat(bias,zeros), …, Forward, H, layout:false, linearBeforeReset:true)
/// with the same seeded W/R/B — y AND hN, relative-L1. Pins the z,r,h gate packing, bias packing,
/// and the [L,D,N,H]→[L,N,D*H] reshape.</summary>
[Module]
public partial class GruMatchesCoreOpForward
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L);
        var (yRef, hRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);
        return GruRefHelpers.RelL1(y, yRef) + GruRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-1 (batchFirst) Matches the core op with batchFirst input: Recurrent.GRU(batchFirst:true)
/// on [N, L, in] equals the op fed the transposed [L, N, in], with the final Y transposed back to
/// [N, L, D*H]. hN stays [D·numLayers, N, H]. Pins the in-graph transpose around the layout=0 op.</summary>
[Module]
public partial class GruMatchesCoreOpBatchFirst
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L, batchFirst: true);
        var xT = x.Transpose(1L, 0L, 2L);                                  // [L, N, in]
        var (yRefLN, hRef) = GruRefHelpers.RefOp(xT, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);
        var yRef = yRefLN.Transpose(1L, 0L, 2L);                           // [N, L, D*H]
        return GruRefHelpers.RelL1(y, yRef) + GruRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-2 linearBeforeReset BOTH forms (the GRU numeric crux): the reset-after default
/// (linearBeforeReset:true) and the reset-before form (linearBeforeReset:false), on the same x and the
/// same seeded W/R/B, must (i) DIFFER (the knob is non-vacuous), and (ii) each equal the corresponding
/// OnnxOp.Gru(…, linearBeforeReset:true/false). Pins the reset-after default AND that the bit is honored.</summary>
[Module]
public partial class GruLinearBeforeResetBothForms
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yLbr, hLbr) = Recurrent.GRU(x, hiddenSize: 3L, linearBeforeReset: true);
        var (yLbrRef, hLbrRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);

        var (yNoLbr, hNoLbr) = Recurrent.GRU(x, hiddenSize: 3L, linearBeforeReset: false);
        var (yNoLbrRef, hNoLbrRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: false);

        // (ii) each GRU form matches its own-form op reference.
        var matchPen = GruRefHelpers.RelL1(yLbr, yLbrRef) + GruRefHelpers.RelL1(hLbr, hLbrRef)
                     + GruRefHelpers.RelL1(yNoLbr, yNoLbrRef) + GruRefHelpers.RelL1(hNoLbr, hNoLbrRef);

        // (i) the two forms must differ — assert the relative-L1 between them is non-trivial (> 1e-3).
        var formsDiff = GruRefHelpers.RelL1(yLbr, yNoLbr);
        var differOk = (Scalar(1e-3f) - formsDiff).Relu();   // 0 when they differ enough, positive when too close

        return matchPen + differOk < Scalar(1e-4f);
    }
}

/// <summary>§7-3 Single-step gate anchor: L=1, h_0=0 ⇒ closed-form z=σ(W_z·x_0+b_z),
/// ĥ=tanh(W_h·x_0+b_h) (the reset term r⊙(R_h·h_0) vanishes at h_0=0 — and with linearBeforeReset:true
/// the recurrent-bias Rb_h is gated by r and so also drops since Rb=0), H_1=(1−z)⊙ĥ, from the SAME
/// seeded W/bias sliced into the z,r,h gate blocks (the op's [D,3H,in] W contracts the input axis; R is
/// unused at step 0). Pins the gate packing ORDER and the equation — a wrong z,r,h↔r,z,n swap fails
/// this. H=2.</summary>
[Module]
public partial class GruSingleStepGateAnchor
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 2L);   // y [1,N,H], hN [1,N,H]

        long hv = 2L;
        Scalar<int64> inSize = x.DimTensor(-1);
        var h = Scalar(hv);
        var threeH = Scalar(3L * hv);

        // Same seeded W/bias as the helper (bound keyed on H).
        var w = RecurrentUniform.Init([Scalar(1L), threeH, inSize], h);   // [1, 3H, in]
        var biasParam = RecurrentUniform.Init([Scalar(1L), threeH], h);   // [1, 3H]
        var w3HI = w.Reshape([threeH, inSize]);                           // [3H, in]
        var bias3H = biasParam.Reshape([threeH]);                         // [3H]

        // Pre-activation zPre = W·x_0 + b, shape [N, 3H] (gate blocks z,r,h stacked along axis 1).
        var x0 = x.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([x.DimTensor(1), inSize]); // [N, in]
        var zPre = x0.MatMul(w3HI.Transpose(1L, 0L)) + bias3H;            // [N, 3H]

        // Slice the three H-blocks in ONNX z,r,h order along axis 1.
        Tensor<float32> Block(long idx) => zPre.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L)); // [N, H]
        var z = Block(0L).Sigmoid();   // update gate
        // r = Block(1) (reset) — its product r⊙(R_h·h_0 + Rb_h) vanishes at h_0=0, Rb=0.
        var hTilde = Block(2L).Tanh(); // candidate ĥ = tanh(W_h·x_0 + b_h)

        var h1 = (Scalar(1f) - z) * hTilde;   // H_1 = (1−z)⊙ĥ + z⊙H_0 = (1−z)⊙ĥ  (H_0 = 0)

        var hNFlat = hN.Reshape([x.DimTensor(1), h]);   // [N, H]
        var yStep0 = y.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([x.DimTensor(1), h]);

        var pen = GruRefHelpers.RelL1(hNFlat, h1)        // hN == H_1
                + GruRefHelpers.RelL1(yStep0, h1);       // y[0] == H_1 == hN
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-4 bias on/off: bias:false equals the no-B op; bias:true equals the concat(bias,zeros) op.
/// Both compared against the matching seeded reference (same W/R; B present iff bias) on y and hN.</summary>
[Module]
public partial class GruBiasOnOffMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yNoB, hNoB) = Recurrent.GRU(x, hiddenSize: 3L, bias: false);
        var (yNoBRef, hNoBRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: false, linearBeforeReset: true);

        var (yB, hB) = Recurrent.GRU(x, hiddenSize: 3L, bias: true);
        var (yBRef, hBRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);

        var pen = GruRefHelpers.RelL1(yNoB, yNoBRef) + GruRefHelpers.RelL1(hNoB, hNoBRef)
                + GruRefHelpers.RelL1(yB, yBRef) + GruRefHelpers.RelL1(hB, hBRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-5 numLayers stacking: numLayers:2 (forward) equals a hand-built two-op stack feeding
/// layer-0 Y → layer-1 X. Asserts y and the stacked [2,N,H] hN both match. Layer-1's input size is
/// D·H = H (D=1), which the helper passes as Scalar(d·H); the reference reads it from the reshaped
/// layer-0 Y's last axis (same value), so the seeded inits coincide.</summary>
[Module]
public partial class GruNumLayersStackMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L, numLayers: 2);

        // Layer 0 over x.
        var (y0, hN0) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);  // y0 [L, N, H]
        // Layer 1 over layer-0's Y (its in == D·H == H).
        var (y1, hN1) = GruRefHelpers.RefOp(y0, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);
        var hRef = hN0.Concat(0L, hN1);   // [2, N, H]

        var pen = GruRefHelpers.RelL1(y, y1) + GruRefHelpers.RelL1(hN, hRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-6 direction Reverse (trainable): Recurrent.GRU(Reverse) equals the op direction:Reverse
/// with the same seeded W/R/B (y and hN), relative-L1.</summary>
[Module]
public partial class GruReverseMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L, direction: RnnDirection.Reverse);
        var (yRef, hRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Reverse, bias: true, linearBeforeReset: true);
        return GruRefHelpers.RelL1(y, yRef) + GruRefHelpers.RelL1(hN, hRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-6 direction Bidirectional (forward inference only): Recurrent.GRU(Bidirectional) equals
/// the op direction:Bidirectional (same seeded [2,…] W/R/B); AND y's last axis is 2H, hN is [2, N, H].
/// Forward-value only (bidirectional BPTT throws AD003 — pinned in NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class GruBidirectionalMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        long hVal = 3L;
        var (y, hN) = Recurrent.GRU(x, hiddenSize: hVal, direction: RnnDirection.Bidirectional);
        var (yRef, hRef) = GruRefHelpers.RefOp(x, hVal, GRUDirection.Bidirectional, bias: true, linearBeforeReset: true);

        // Shape contract: y last axis == 2H, hN leading axis == 2 (D=2, numLayers=1).
        var lastAxisOk = (y.DimTensor(-1) - Scalar(2L * hVal)).Abs().Cast<float32>();
        var hLeadingOk = (hN.DimTensor(0) - Scalar(2L)).Abs().Cast<float32>();

        var pen = GruRefHelpers.RelL1(y, yRef) + GruRefHelpers.RelL1(hN, hRef)
                + lastAxisOk + hLeadingOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-7 batchFirst equivalence: GRU(batchFirst:true, [N,L,in]) equals
/// GRU(batchFirst:false, transpose([L,N,in])) with the result transposed back. Pins the in-graph
/// transpose + layout=0-always choice independently of the op reference. hN is batch-second in both
/// (PyTorch keeps it [D·numLayers, N, H] regardless of batch_first), so it matches directly.</summary>
[Module]
public partial class GruBatchFirstEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (yBF, hBF) = Recurrent.GRU(x, hiddenSize: 3L, batchFirst: true);     // y [N, L, D*H]

        var xT = x.Transpose(1L, 0L, 2L);                                        // [L, N, in]
        var (ySF, hSF) = Recurrent.GRU(xT, hiddenSize: 3L, batchFirst: false);   // y [L, N, D*H]
        var ySFasBF = ySF.Transpose(1L, 0L, 2L);                                 // [N, L, D*H]

        var pen = GruRefHelpers.RelL1(yBF, ySFasBF) + GruRefHelpers.RelL1(hBF, hSF);
        return pen < Scalar(1e-5f);
    }
}

/// <summary>§7-8 state contract: for a forward single-layer GRU, hN == y[-1] (the last step's hidden
/// state) and equals the op's Y_h; hN shape is [D·numLayers, N, H] == [1, N, H]. Pins the (y, hN)
/// return relationship.</summary>
[Module]
public partial class GruStateContractForwardSingleLayer
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L);   // y [L,N,H], hN [1,N,H]
        var (_, hRef) = GruRefHelpers.RefOp(x, 3L, GRUDirection.Forward, bias: true, linearBeforeReset: true);

        // y[-1]: last step along axis 0, flattened to [N, H]; hN flattened to [N, H].
        var lastStep = y.Slice(Vector(-1L), Vector(System.Int64.MaxValue), Vector(0L))
            .Reshape([x.DimTensor(1), Scalar(3L)]);                  // [N, H]
        var hNFlat = hN.Reshape([x.DimTensor(1), Scalar(3L)]);      // [N, H] (D·numLayers == 1)

        // Shape contract: hN leading axis == D·numLayers == 1.
        var hLeadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();

        var pen = GruRefHelpers.RelL1(hNFlat, lastStep)    // hN == y[-1]
                + GruRefHelpers.RelL1(hN, hRef)            // hN == op's Y_h
                + hLeadingOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-9 trainable-corner FD grad check: a forward, single-layer, linearBeforeReset:true
/// Recurrent.GRU (a supported autodiff corner — design §5) on the loss ΣY + Σh_n, with the analytic
/// gradient (via AutoGrad) FD-checked against a two-sided directional derivative on ORT's own forward.
/// Mirrors LstmForwardGradCheck / AutoGradGruReverseCheck. x[0,0,0] is the probed scalar over a
/// length-3 sequence.</summary>
[Module]
public partial class GruForwardGradCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        Func<Scalar<float32>, Scalar<float32>> f = v =>
        {
            var x = RecurrentTestData.BuildX(v, 3);   // [3, 1, 2]
            var (y, hN) = Recurrent.GRU(x, hiddenSize: 2L);   // forward, single-layer, linearBeforeReset:true
            return y.Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        };
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, f(xv));
        return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(xv, grad, f);
    }
}

/// <summary>§7-9 trainability rig model: a forward, single-layer, linearBeforeReset:true Recurrent.GRU
/// reducing (y, hN) to a per-batch logit pair, for a TrainingRig FromScratch / TrainStep. The owned
/// W/R/bias differentiate end-to-end here (the trainable corner — and the scheduler fix that landed
/// unblocks the recurrent rig path). Output [N, 2] so CrossEntropy has two logits.</summary>
[Module]
public partial class GruForwardTrainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // input is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(input, hiddenSize: 4L);   // y [L,N,H], hN [1,N,H]
        var n = input.DimTensor(1);

        // Per-batch features: sum y over time+hidden, sum hN over hidden → [N], stack to [N, 2].
        Vector<int64> timeHidden = [Scalar(0L), Scalar(2L)];
        var yFeat = y.Reduce(ReduceKind.Sum, timeHidden, keepDims: false).Reshape([n, Scalar(1L)]);  // [N, 1]
        Vector<int64> layerHidden = [Scalar(0L), Scalar(2L)];
        var hFeat = hN.Reduce(ReduceKind.Sum, layerHidden, keepDims: false)
            .Reshape([n, Scalar(1L)]);                                                                 // [N, 1]
        return yFeat.Concat(1L, hFeat);   // [N, 2]
    }
}

/// <summary>§7-6 (BPTT throw) bidirectional GRU gradient: a loss through Recurrent.GRU(Bidirectional)
/// must throw AD003 at lowering (bidirectional BPTT unimplemented). Mirrors LstmBidirectionalBpttThrowCheck.
/// Never reached past AutoGrad — the AdvancedTestGraph call throws.</summary>
[Module]
public partial class GruBidirectionalBpttThrowCheck
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 2);   // [2, 1, 2]
        var (_, hN) = Recurrent.GRU(x, hiddenSize: 2L, direction: RnnDirection.Bidirectional);
        var loss = hN.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(xv, loss);
        return grad.Abs() < Scalar(1e9f);   // never reached: AD003 at lowering
    }
}

// ---------------------------------------------------------------------------
// Recurrent single-step CELL coverage — recurrent-cells/design.md §7. Mirrors
// the Recurrent.RNN/LSTM/GRU helper sets above EXACTLY: each self-checking
// [Module] returns a Scalar<bit> that AutoTest.AdvancedTestGraph requires true.
// It computes a cell (Recurrent.RNNCell/LSTMCell/GRUCell) and a hand-built
// reference from a SEEDED seq=1 OnnxOp.Rnn/Lstm/Gru with the SAME-shape
// RecurrentUniform.Init W/R/bias (so the seeded inits materialize identically —
// the same idiom as RnnRefHelpers / LstmRefHelpers / GruRefHelpers), then
// asserts a relative-L1 match (reuse RnnRefHelpers.RelL1). A cell is one step of
// the layer: x [N,in] is Unsqueeze(0)'d to [seq=1,N,in], the previous state(s)
// h(/c) [N,H] are Unsqueeze(0)'d to [num_dir=1,N,H] and passed as initial_h
// (/initial_c), the op runs forward at layout=0, and Y_h(/Y_c) [num_dir=1,N,H]
// is squeezed back to [N,H]. Cells have NO QEE step values, so value
// correctness comes from the ORT backend inside AdvancedTestGraph. The AD003
// relu-cell BPTT-throws guard lives as a [Fact] in NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------

internal static class CellRefHelpers
{
    /// <summary>The owned bias packed exactly as a cell does: B = concat(bias[1,k·H], zeros[1,k·H]) on axis 1 → [1,2·k·H].</summary>
    public static Tensor<float32> PackedBias(Scalar<int64> kH, Scalar<int64> h)
    {
        var biasParam = RecurrentUniform.Init([Scalar(1L), kH], h);          // [1, k·H]; bound keyed on H
        var rbZeros = TensorFill((Vector<int64>)[Scalar(1L), kH], 0.0f);     // [1, k·H]
        return biasParam.Concat(1L, rbZeros);                                // [1, 2·k·H]
    }

    /// <summary>Hand-built seq=1 RNN reference: Unsqueeze x/h, run the op forward with the same SAME-shape
    /// seeded W/R/(optional B), squeeze Y_h back to [N,H]. Mirrors Recurrent.RNNCell's internal op call.</summary>
    public static Tensor<float32> RnnRefStep(
        Tensor<float32> x, Tensor<float32> h, long hiddenSize, string[]? activations, bool bias)
    {
        var hs = Scalar(hiddenSize);
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), hs, inSize], hs);   // [1, H, in]
        var r = RecurrentUniform.Init([Scalar(1L), hs, hs], hs);       // [1, H, H]
        Tensor<float32>? b = bias ? PackedBias(hs, hs) : (Tensor<float32>?)null;

        var (_, yhVar) = OnnxOp.Rnn(x.Unsqueeze(0L), w, r, b, null, h.Unsqueeze(0L),
            null, null, activations, null, RNNDirection.Forward, hiddenSize, false);
        return ((Tensor<float32>)yhVar).Squeeze(Vector(0L));           // [N, H]
    }

    /// <summary>Hand-built seq=1 LSTM reference: Unsqueeze x/h/c, run the op forward with the same SAME-shape
    /// seeded W/R/(optional B), squeeze Y_h/Y_c back to [N,H]. Mirrors Recurrent.LSTMCell's internal op call.</summary>
    public static (Tensor<float32> h, Tensor<float32> c) LstmRefStep(
        Tensor<float32> x, Tensor<float32> h, Tensor<float32> c, long hiddenSize, bool bias)
    {
        var hs = Scalar(hiddenSize);
        var fourH = Scalar(4L * hiddenSize);
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), fourH, inSize], hs);   // [1, 4H, in]
        var r = RecurrentUniform.Init([Scalar(1L), fourH, hs], hs);       // [1, 4H, H]
        Tensor<float32>? b = bias ? PackedBias(fourH, hs) : (Tensor<float32>?)null;

        var (_, yhVar, ycVar) = OnnxOp.Lstm(x.Unsqueeze(0L), w, r, b, null, h.Unsqueeze(0L), c.Unsqueeze(0L),
            null, null, null, null, null, LSTMDirection.Forward, hiddenSize, false, false);
        return (((Tensor<float32>)yhVar).Squeeze(Vector(0L)),
                ((Tensor<float32>)ycVar).Squeeze(Vector(0L)));            // each [N, H]
    }

    /// <summary>Hand-built seq=1 GRU reference: Unsqueeze x/h, run the op forward with the same SAME-shape
    /// seeded W/R/(optional B) and the SAME linearBeforeReset, squeeze Y_h back to [N,H]. Mirrors
    /// Recurrent.GRUCell's internal op call.</summary>
    public static Tensor<float32> GruRefStep(
        Tensor<float32> x, Tensor<float32> h, long hiddenSize, bool bias, bool linearBeforeReset)
    {
        var hs = Scalar(hiddenSize);
        var threeH = Scalar(3L * hiddenSize);
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), threeH, inSize], hs);   // [1, 3H, in]
        var r = RecurrentUniform.Init([Scalar(1L), threeH, hs], hs);       // [1, 3H, H]
        Tensor<float32>? b = bias ? PackedBias(threeH, hs) : (Tensor<float32>?)null;

        var (_, yhVar) = OnnxOp.Gru(x.Unsqueeze(0L), w, r, b, null, h.Unsqueeze(0L),
            null, null, null, null, GRUDirection.Forward, hiddenSize, false, linearBeforeReset);
        return ((Tensor<float32>)yhVar).Squeeze(Vector(0L));              // [N, H]
    }
}

// ===========================  RNNCell  =====================================

/// <summary>§7-1 (RNNCell) Closed-form single-step anchor: tanh. H=2, N=1, NONZERO h (so R is exercised,
/// unlike the layer's h_0=0 anchor). Asserts h' == tanh(W·x + R·h + bias) hand-computed from the SAME
/// seeded W/R/bias. x is [1, in] and h is the in-module nonzero constant [1, 2].</summary>
[Module]
public partial class RnnCellClosedFormTanh
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var h = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero previous state [N, H]
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: hv);   // [N, H]

        var hs = Scalar(hv);
        Scalar<int64> inSize = x.DimTensor(-1);
        // Same seeded W/R/bias as the cell (bound keyed on H); squeeze the D=1 axis.
        var w = RecurrentUniform.Init([Scalar(1L), hs, inSize], hs).Reshape([hs, inSize]);   // [H, in]
        var r = RecurrentUniform.Init([Scalar(1L), hs, hs], hs).Reshape([hs, hs]);           // [H, H]
        var biasH = RecurrentUniform.Init([Scalar(1L), hs], hs).Reshape([hs]);               // [H]

        // h' = tanh(W·x + R·h + bias), all [N, H] with bias broadcast.
        var preact = x.MatMul(w.Transpose(1L, 0L)) + h.MatMul(r.Transpose(1L, 0L)) + biasH;
        var manual = preact.Tanh();
        return RnnRefHelpers.RelL1(hOut, manual) < Scalar(1e-4f);
    }
}

/// <summary>§7-1 (RNNCell) Closed-form single-step anchor: relu forward. As RnnCellClosedFormTanh but
/// nonlinearity:Relu ⇒ h' == relu(W·x + R·h + bias). Forward-value only (relu cell BPTT throws AD003 —
/// pinned by a [Fact]).</summary>
[Module]
public partial class RnnCellClosedFormRelu
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var h = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero previous state
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: hv, nonlinearity: RnnNonlinearity.Relu);

        var hs = Scalar(hv);
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), hs, inSize], hs).Reshape([hs, inSize]);   // [H, in]
        var r = RecurrentUniform.Init([Scalar(1L), hs, hs], hs).Reshape([hs, hs]);           // [H, H]
        var biasH = RecurrentUniform.Init([Scalar(1L), hs], hs).Reshape([hs]);               // [H]

        var preact = x.MatMul(w.Transpose(1L, 0L)) + h.MatMul(r.Transpose(1L, 0L)) + biasH;
        var manual = preact.Relu();
        return RnnRefHelpers.RelL1(hOut, manual) < Scalar(1e-4f);
    }
}

/// <summary>§7-2/§7-3 (RNNCell) Cell ≡ seq=1 reference op AND shape contract: Recurrent.RNNCell(x, h, H)
/// equals CellRefHelpers.RnnRefStep (hand-built Unsqueeze→OnnxOp.Rnn(initial_h)→Squeeze) with the same
/// seeded W/R/B, relative-L1; AND the output is [N, H] (rank 2, last axis == H, num_dir stripped). h is
/// nonzero so initial_h is genuinely threaded. x [N, in], h [N, H].</summary>
[Module]
public partial class RnnCellMatchesSeq1Op
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);   // nonzero [N, H]
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: hv);
        var hRef = CellRefHelpers.RnnRefStep(x, h, hv, null, bias: true);

        // Shape contract: the [N, H] output has leading axis N and last axis H (the num_dir axis is
        // stripped). A wrong shape would also break the relative-L1 match below.
        var nOk = (hOut.DimTensor(0) - n).Abs().Cast<float32>();
        var lastAxisOk = (hOut.DimTensor(-1) - Scalar(hv)).Abs().Cast<float32>();
        return RnnRefHelpers.RelL1(hOut, hRef) + nOk + lastAxisOk < Scalar(1e-4f);
    }
}

/// <summary>§7-4 (RNNCell) bias on/off: bias:false equals the no-B seq=1 op; bias:true equals the
/// concat(bias,zeros) seq=1 op. Both compared against the matching seeded reference (same W/R; B iff bias).</summary>
[Module]
public partial class RnnCellBiasOnOff
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);   // nonzero [N, H]

        var hNoB = Recurrent.RNNCell(x, h, hiddenSize: hv, bias: false);
        var hNoBRef = CellRefHelpers.RnnRefStep(x, h, hv, null, bias: false);

        var hB = Recurrent.RNNCell(x, h, hiddenSize: hv, bias: true);
        var hBRef = CellRefHelpers.RnnRefStep(x, h, hv, null, bias: true);

        return RnnRefHelpers.RelL1(hNoB, hNoBRef) + RnnRefHelpers.RelL1(hB, hBRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-5 (RNNCell) State threading — THE DEFINING TEST. Two manual cell steps: feed h' from the
/// first RNNCell call as the h of the second (same seeded weights via shared shape+seed), starting from
/// h_0 = 0. Assert the two-step result equals Recurrent.RNN over the length-2 sequence [x0, x1] with the
/// matching seeded W/R/bias and h_0 = 0 — proving the cell is exactly one step of the scan. x is [2, N, in]
/// (the two steps); the layer's hN [1, N, H] is the second step's output.</summary>
[Module]
public partial class RnnCellStateThreading
{
    public static Scalar<bit> Inline(Tensor<float32> xSeq)   // xSeq is [L=2, N, in]
    {
        long hv = 3L;
        var n = xSeq.DimTensor(1);
        Scalar<int64> inSize = xSeq.DimTensor(-1);

        // x0, x1 as [N, in] step inputs.
        var x0 = xSeq.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([n, inSize]);   // [N, in]
        var x1 = xSeq.Slice(Vector(1L), Vector(2L), Vector(0L)).Reshape([n, inSize]);   // [N, in]

        // Hand-unrolled cell loop from h_0 = 0.
        var h0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);   // [N, H]
        var h1 = Recurrent.RNNCell(x0, h0, hiddenSize: hv);          // step 1
        var h2 = Recurrent.RNNCell(x1, h1, hiddenSize: hv);          // step 2 — h1 threaded back in

        // Full layer over [x0, x1]; its hN [1, N, H] is the last (= second) step's hidden state.
        var (_, hN) = Recurrent.RNN(xSeq, hiddenSize: hv);
        var hNFlat = hN.Reshape([n, Scalar(hv)]);                    // [N, H]

        return RnnRefHelpers.RelL1(h2, hNFlat) < Scalar(1e-4f);
    }
}

/// <summary>§7-6 (RNNCell) Trainable-corner FD grad check (tanh, the autodiff-supported config). The loss
/// Σh' over a single RNNCell step (nonzero h built from the probed scalar so the gradient flows through
/// BOTH x and the h-input, the cell's distinguishing input) is FD-checked via a two-sided directional
/// derivative against ORT's own forward. Mirrors RnnForwardTanhGradCheck.</summary>
[Module]
public partial class RnnCellForwardTanhGradCheck
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        Func<Scalar<float32>, Scalar<float32>> f = z =>
        {
            // x [1, 2] and h [1, 2] both depend on z, so the grad threads through x AND the h-input.
            var zv = (Tensor<float32>)OnnxOp.Unsqueeze(z, Vector(0L));
            var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var h = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(-0.2f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var hOut = Recurrent.RNNCell(x, h, hiddenSize: 2L);
            return hOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        };
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(v, f(v));
        return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(v, grad, f);
    }
}

/// <summary>§7-7 (RNNCell) trainability rig model: a hand-unrolled 2-step RNNCell (tanh) loop reducing
/// the final hidden state to a per-batch logit pair, for a TrainingRig FromScratch / TrainStep + L2Loss +
/// SGD. The owned W/R/bias differentiate end-to-end through the user loop. h_0 is a zero seed derived from
/// x's batch dim. Output [N, 2]. Input [L=2, N, in].</summary>
[Module]
public partial class RnnCellTrainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // input is [L=2, N, in]
    {
        long hv = 4L;
        var n = input.DimTensor(1);
        Scalar<int64> inSize = input.DimTensor(-1);

        var x0 = input.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([n, inSize]);   // [N, in]
        var x1 = input.Slice(Vector(1L), Vector(2L), Vector(0L)).Reshape([n, inSize]);   // [N, in]

        var h0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);   // [N, H] zero seed
        var h1 = Recurrent.RNNCell(x0, h0, hiddenSize: hv);
        var h2 = Recurrent.RNNCell(x1, h1, hiddenSize: hv);          // [N, H]

        // Per-batch features: two reductions of the final hidden state → [N, 2].
        var f0 = h2.Reduce(ReduceKind.Sum, (Vector<int64>)[Scalar(1L)], keepDims: false).Reshape([n, Scalar(1L)]);
        var f1 = h2.Reduce(ReduceKind.Mean, (Vector<int64>)[Scalar(1L)], keepDims: false).Reshape([n, Scalar(1L)]);
        return f0.Concat(1L, f1);   // [N, 2]
    }
}

/// <summary>§7-8 (RNNCell, BPTT throw) relu cell gradient: a loss through Recurrent.RNNCell(Relu) must
/// throw AD003 at lowering (relu is a non-default activation; BPTT unsupported). Mirrors
/// RnnReluBpttThrowCheck. Never reached past AutoGrad — the AdvancedTestGraph call throws.</summary>
[Module]
public partial class RnnCellReluBpttThrowCheck
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        var zv = (Tensor<float32>)OnnxOp.Unsqueeze(v, Vector(0L));
        var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var h = Tensor(new long[] { 1L, 2L }, 0.3f, -0.2f);
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: 2L, nonlinearity: RnnNonlinearity.Relu);
        var loss = hOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(v, loss);
        return grad.Abs() < Scalar(1e9f);   // never reached: AD003 at lowering
    }
}

// ===========================  LSTMCell  ====================================

/// <summary>§7-1 (LSTMCell) Closed-form single-step gate anchor — pins the i,o,f,c gate PACKING. H=2, N=1,
/// NONZERO h and c (so R and the forget gate are exercised). Hand-compute, in ONNX i,o,f,c order from the
/// SAME seeded W/R/bias: pre-act = W·x + R·h + bias; i=σ(blk0), o=σ(blk1), f=σ(blk2), g=tanh(blk3);
/// c' = f⊙c + i⊙g; h' = o⊙tanh(c'). Assert (h', c') match. A wrong i,o,f,c↔i,f,g,o swap fails this.</summary>
[Module]
public partial class LstmCellClosedFormGateAnchor
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var prevH = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero [N, H]
        var prevC = Tensor(new long[] { 1L, 2L }, -0.1f, 0.5f);   // nonzero [N, H]
        var (hOut, cOut) = Recurrent.LSTMCell(x, prevH, prevC, hiddenSize: hv);

        var hs = Scalar(hv);
        var fourH = Scalar(4L * hv);
        Scalar<int64> inSize = x.DimTensor(-1);
        // Same seeded W/R/bias as the cell (bound keyed on H); squeeze the D=1 axis.
        var w = RecurrentUniform.Init([Scalar(1L), fourH, inSize], hs).Reshape([fourH, inSize]);   // [4H, in]
        var r = RecurrentUniform.Init([Scalar(1L), fourH, hs], hs).Reshape([fourH, hs]);           // [4H, H]
        var bias4H = RecurrentUniform.Init([Scalar(1L), fourH], hs).Reshape([fourH]);              // [4H]

        // Pre-activation z = W·x + R·h + bias, [N, 4H], gate blocks i,o,f,c along axis 1.
        var z = x.MatMul(w.Transpose(1L, 0L)) + prevH.MatMul(r.Transpose(1L, 0L)) + bias4H;   // [N, 4H]
        Tensor<float32> Block(long idx) => z.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L)); // [N, H]
        var i = Block(0L).Sigmoid();   // input gate
        var o = Block(1L).Sigmoid();   // output gate
        var fg = Block(2L).Sigmoid();  // forget gate (exercised — prevC nonzero)
        var g = Block(3L).Tanh();      // cell candidate

        var cManual = fg * prevC + i * g;   // c' = f⊙c + i⊙g
        var hManual = o * cManual.Tanh();   // h' = o⊙tanh(c')

        var pen = RnnRefHelpers.RelL1(cOut, cManual) + RnnRefHelpers.RelL1(hOut, hManual);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-2/§7-3 (LSTMCell) Cell ≡ seq=1 reference op AND shape contract: Recurrent.LSTMCell(x, h, c, H)
/// equals CellRefHelpers.LstmRefStep (Unsqueeze x/h/c → OnnxOp.Lstm(initial_h, initial_c) → Squeeze) with
/// the same seeded W/R/B on BOTH h' and c', relative-L1; AND both outputs are [N, H] (rank 2, last axis ==
/// H). h, c nonzero so initial_h/initial_c are genuinely threaded.</summary>
[Module]
public partial class LstmCellMatchesSeq1Op
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);    // nonzero [N, H]
        var c = TensorFill((Vector<int64>)[n, Scalar(hv)], -0.1f);   // nonzero [N, H]
        var (hOut, cOut) = Recurrent.LSTMCell(x, h, c, hiddenSize: hv);
        var (hRef, cRef) = CellRefHelpers.LstmRefStep(x, h, c, hv, bias: true);

        // Shape contract: both outputs are [N, H] (leading axis N, last axis H, num_dir stripped).
        var nOk = (hOut.DimTensor(0) - n).Abs().Cast<float32>()
                + (cOut.DimTensor(0) - n).Abs().Cast<float32>();
        var lastAxisOk = (hOut.DimTensor(-1) - Scalar(hv)).Abs().Cast<float32>()
                       + (cOut.DimTensor(-1) - Scalar(hv)).Abs().Cast<float32>();
        return RnnRefHelpers.RelL1(hOut, hRef) + RnnRefHelpers.RelL1(cOut, cRef)
             + nOk + lastAxisOk < Scalar(1e-4f);
    }
}

/// <summary>§7-4 (LSTMCell) bias on/off: bias:false equals the no-B seq=1 op; bias:true equals the
/// concat(bias,zeros) seq=1 op, on both h' and c'.</summary>
[Module]
public partial class LstmCellBiasOnOff
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);
        var c = TensorFill((Vector<int64>)[n, Scalar(hv)], -0.1f);

        var (hNoB, cNoB) = Recurrent.LSTMCell(x, h, c, hiddenSize: hv, bias: false);
        var (hNoBRef, cNoBRef) = CellRefHelpers.LstmRefStep(x, h, c, hv, bias: false);

        var (hB, cB) = Recurrent.LSTMCell(x, h, c, hiddenSize: hv, bias: true);
        var (hBRef, cBRef) = CellRefHelpers.LstmRefStep(x, h, c, hv, bias: true);

        var pen = RnnRefHelpers.RelL1(hNoB, hNoBRef) + RnnRefHelpers.RelL1(cNoB, cNoBRef)
                + RnnRefHelpers.RelL1(hB, hBRef) + RnnRefHelpers.RelL1(cB, cBRef);
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-5 (LSTMCell) State threading — the defining test. Two manual cell steps: feed (h', c') from
/// the first LSTMCell call as the (h, c) of the second (same seeded weights), starting from h_0 = c_0 = 0.
/// Assert the two-step result equals Recurrent.LSTM over the length-2 sequence [x0, x1] (its hN/cN [1,N,H]
/// are the second step's states). Proves the cell is one step of the scan, for BOTH carried states.</summary>
[Module]
public partial class LstmCellStateThreading
{
    public static Scalar<bit> Inline(Tensor<float32> xSeq)   // xSeq is [L=2, N, in]
    {
        long hv = 3L;
        var n = xSeq.DimTensor(1);
        Scalar<int64> inSize = xSeq.DimTensor(-1);

        var x0 = xSeq.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([n, inSize]);   // [N, in]
        var x1 = xSeq.Slice(Vector(1L), Vector(2L), Vector(0L)).Reshape([n, inSize]);   // [N, in]

        var h0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);   // [N, H]
        var c0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);   // [N, H]
        var (h1, c1) = Recurrent.LSTMCell(x0, h0, c0, hiddenSize: hv);   // step 1
        var (h2, c2) = Recurrent.LSTMCell(x1, h1, c1, hiddenSize: hv);   // step 2 — (h1,c1) threaded back

        var (_, hN, cN) = Recurrent.LSTM(xSeq, hiddenSize: hv);
        var hNFlat = hN.Reshape([n, Scalar(hv)]);   // [N, H]
        var cNFlat = cN.Reshape([n, Scalar(hv)]);   // [N, H]

        return RnnRefHelpers.RelL1(h2, hNFlat) + RnnRefHelpers.RelL1(c2, cNFlat) < Scalar(1e-4f);
    }
}

/// <summary>§7-6 (LSTMCell) Trainable-corner FD grad check. The loss Σh' + Σc' over a single LSTMCell step
/// (x, h AND c built from the probed scalar so the gradient flows through all three inputs, including the
/// distinguishing h/c states) is FD-checked against ORT's own forward. Mirrors LstmForwardGradCheck.</summary>
[Module]
public partial class LstmCellForwardGradCheck
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        Func<Scalar<float32>, Scalar<float32>> f = z =>
        {
            var zv = (Tensor<float32>)OnnxOp.Unsqueeze(z, Vector(0L));
            var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var h = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(-0.2f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var c = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.1f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var (hOut, cOut) = Recurrent.LSTMCell(x, h, c, hiddenSize: 2L);
            return hOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar()
                 + cOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        };
        var grad = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(v, f(v));
        return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(v, grad, f);
    }
}

/// <summary>§7-7 (LSTMCell) trainability rig model: a hand-unrolled 2-step LSTMCell loop reducing the
/// final (h, c) to a per-batch logit pair, for a TrainingRig FromScratch / TrainStep + L2Loss + SGD. The
/// owned W/R/bias differentiate end-to-end through the user loop. h_0/c_0 are zero seeds. Output [N, 2].</summary>
[Module]
public partial class LstmCellTrainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // input is [L=2, N, in]
    {
        long hv = 4L;
        var n = input.DimTensor(1);
        Scalar<int64> inSize = input.DimTensor(-1);

        var x0 = input.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([n, inSize]);
        var x1 = input.Slice(Vector(1L), Vector(2L), Vector(0L)).Reshape([n, inSize]);

        var h0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);
        var c0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);
        var (h1, c1) = Recurrent.LSTMCell(x0, h0, c0, hiddenSize: hv);
        var (h2, c2) = Recurrent.LSTMCell(x1, h1, c1, hiddenSize: hv);

        var f0 = h2.Reduce(ReduceKind.Sum, (Vector<int64>)[Scalar(1L)], keepDims: false).Reshape([n, Scalar(1L)]);
        var f1 = c2.Reduce(ReduceKind.Sum, (Vector<int64>)[Scalar(1L)], keepDims: false).Reshape([n, Scalar(1L)]);
        return f0.Concat(1L, f1);   // [N, 2]
    }
}

// ===========================  GRUCell  =====================================

/// <summary>§7-1 (GRUCell) Closed-form single-step anchor, linearBeforeReset:true (reset-after). H=2, N=1,
/// NONZERO h. Hand-compute, in ONNX z,r,h order from the SAME seeded W/R/bias: zPre = W·x + bias (input
/// part); the recurrent term R·h enters z and r directly but the candidate uses the reset-AFTER form. With
/// the single owned bias (Rb=0): z=σ(Wz·x + Rz·h + bz), r=σ(Wr·x + Rr·h + br),
/// n=tanh(Wh·x + r⊙(Rh·h) + bh), h'=(1−z)⊙n + z⊙h. A wrong z,r,h↔r,z,n swap fails this.</summary>
[Module]
public partial class GruCellClosedFormLbrTrue
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var prevH = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero [N, H]
        var hOut = Recurrent.GRUCell(x, prevH, hiddenSize: hv, linearBeforeReset: true);

        var hs = Scalar(hv);
        var threeH = Scalar(3L * hv);
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), threeH, inSize], hs).Reshape([threeH, inSize]);   // [3H, in]
        var r = RecurrentUniform.Init([Scalar(1L), threeH, hs], hs).Reshape([threeH, hs]);           // [3H, H]
        var bias3H = RecurrentUniform.Init([Scalar(1L), threeH], hs).Reshape([threeH]);              // [3H]

        // Input contribution Wgate·x + bgate, [N, 3H]; recurrent contribution Rgate·h, [N, 3H].
        var wx = x.MatMul(w.Transpose(1L, 0L)) + bias3H;       // [N, 3H]
        var rh = prevH.MatMul(r.Transpose(1L, 0L));            // [N, 3H]
        Tensor<float32> WxBlk(long idx) => wx.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L));
        Tensor<float32> RhBlk(long idx) => rh.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L));

        // ONNX z,r,h order.
        var z = (WxBlk(0L) + RhBlk(0L)).Sigmoid();             // update gate z
        var rg = (WxBlk(1L) + RhBlk(1L)).Sigmoid();            // reset gate r
        // linearBeforeReset=true: n = tanh(Wh·x + bh + r⊙(Rh·h + Rb_h)), Rb_h = 0.
        var n = (WxBlk(2L) + rg * RhBlk(2L)).Tanh();           // candidate
        var hManual = (Scalar(1f) - z) * n + z * prevH;        // h' = (1−z)⊙n + z⊙h

        return RnnRefHelpers.RelL1(hOut, hManual) < Scalar(1e-4f);
    }
}

/// <summary>§7-1 (GRUCell) linearBeforeReset:false (reset-before, the ONNX op default) — hand-computed
/// closed form n = tanh(Wh·x + bh + (r⊙h)·Rhᵀ), AND it must (i) DIFFER from the lbr:true result and
/// (ii) match the lbr:false seq=1 reference op. Pins the lbr bit. H=2, N=1, nonzero h.</summary>
[Module]
public partial class GruCellClosedFormLbrFalse
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var prevH = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero [N, H]
        var hLbrFalse = Recurrent.GRUCell(x, prevH, hiddenSize: hv, linearBeforeReset: false);
        var hLbrTrue = Recurrent.GRUCell(x, prevH, hiddenSize: hv, linearBeforeReset: true);

        var hs = Scalar(hv);
        var threeH = Scalar(3L * hv);
        Scalar<int64> inSize = x.DimTensor(-1);
        var w = RecurrentUniform.Init([Scalar(1L), threeH, inSize], hs).Reshape([threeH, inSize]);   // [3H, in]
        var r = RecurrentUniform.Init([Scalar(1L), threeH, hs], hs).Reshape([threeH, hs]);           // [3H, H]
        var bias3H = RecurrentUniform.Init([Scalar(1L), threeH], hs).Reshape([threeH]);              // [3H]

        var wx = x.MatMul(w.Transpose(1L, 0L)) + bias3H;       // [N, 3H]
        var rh = prevH.MatMul(r.Transpose(1L, 0L));            // [N, 3H]
        Tensor<float32> WxBlk(long idx) => wx.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L));
        Tensor<float32> RhBlk(long idx) => rh.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(1L));
        Tensor<float32> RBlock(long idx) => r.Slice(Vector(idx * hv), Vector((idx + 1) * hv), Vector(0L)); // [H, H]

        var z = (WxBlk(0L) + RhBlk(0L)).Sigmoid();             // update gate
        var rg = (WxBlk(1L) + RhBlk(1L)).Sigmoid();            // reset gate
        // linearBeforeReset=false: n = tanh(Wh·x + bh + (r⊙h)·Rh_blockᵀ).
        var rhProd = (rg * prevH).MatMul(RBlock(2L).Transpose(1L, 0L));   // [N, H]
        var n = (WxBlk(2L) + rhProd).Tanh();
        var hManual = (Scalar(1f) - z) * n + z * prevH;        // [N, H]

        // (i) the two forms must differ; (ii) lbr:false matches the manual closed form AND the op reference.
        var formsDiff = RnnRefHelpers.RelL1(hLbrFalse, hLbrTrue);
        var differOk = (Scalar(1e-3f) - formsDiff).Relu();     // 0 when they differ enough
        var hRef = CellRefHelpers.GruRefStep(x, prevH, hv, bias: true, linearBeforeReset: false);

        var pen = RnnRefHelpers.RelL1(hLbrFalse, hManual) + RnnRefHelpers.RelL1(hLbrFalse, hRef) + differOk;
        return pen < Scalar(1e-4f);
    }
}

/// <summary>§7-2/§7-3 (GRUCell) Cell ≡ seq=1 reference op AND shape contract: Recurrent.GRUCell(x, h, H)
/// equals CellRefHelpers.GruRefStep (Unsqueeze x/h → OnnxOp.Gru(initial_h) → Squeeze) with the same seeded
/// W/R/B and linearBeforeReset:true, relative-L1; AND the output is [N, H] (rank 2, last axis == H). h
/// nonzero so initial_h is genuinely threaded.</summary>
[Module]
public partial class GruCellMatchesSeq1Op
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);   // nonzero [N, H]
        var hOut = Recurrent.GRUCell(x, h, hiddenSize: hv);
        var hRef = CellRefHelpers.GruRefStep(x, h, hv, bias: true, linearBeforeReset: true);

        // Shape contract: the [N, H] output has leading axis N and last axis H (num_dir stripped).
        var nOk = (hOut.DimTensor(0) - n).Abs().Cast<float32>();
        var lastAxisOk = (hOut.DimTensor(-1) - Scalar(hv)).Abs().Cast<float32>();
        return RnnRefHelpers.RelL1(hOut, hRef) + nOk + lastAxisOk < Scalar(1e-4f);
    }
}

/// <summary>§7-4 (GRUCell) bias on/off: bias:false equals the no-B seq=1 op; bias:true equals the
/// concat(bias,zeros) seq=1 op (both linearBeforeReset:true).</summary>
[Module]
public partial class GruCellBiasOnOff
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);

        var hNoB = Recurrent.GRUCell(x, h, hiddenSize: hv, bias: false);
        var hNoBRef = CellRefHelpers.GruRefStep(x, h, hv, bias: false, linearBeforeReset: true);

        var hB = Recurrent.GRUCell(x, h, hiddenSize: hv, bias: true);
        var hBRef = CellRefHelpers.GruRefStep(x, h, hv, bias: true, linearBeforeReset: true);

        return RnnRefHelpers.RelL1(hNoB, hNoBRef) + RnnRefHelpers.RelL1(hB, hBRef) < Scalar(1e-4f);
    }
}

/// <summary>§7-5 (GRUCell) State threading — the defining test. Two manual cell steps (linearBeforeReset:
/// true): feed h' from the first GRUCell call as the h of the second (same seeded weights), starting from
/// h_0 = 0. Assert the two-step result equals Recurrent.GRU over the length-2 sequence [x0, x1] (its hN
/// [1,N,H] is the second step's hidden state). Proves the cell is one step of the GRU scan.</summary>
[Module]
public partial class GruCellStateThreading
{
    public static Scalar<bit> Inline(Tensor<float32> xSeq)   // xSeq is [L=2, N, in]
    {
        long hv = 3L;
        var n = xSeq.DimTensor(1);
        Scalar<int64> inSize = xSeq.DimTensor(-1);

        var x0 = xSeq.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([n, inSize]);   // [N, in]
        var x1 = xSeq.Slice(Vector(1L), Vector(2L), Vector(0L)).Reshape([n, inSize]);   // [N, in]

        var h0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);   // [N, H]
        var h1 = Recurrent.GRUCell(x0, h0, hiddenSize: hv);          // step 1
        var h2 = Recurrent.GRUCell(x1, h1, hiddenSize: hv);          // step 2 — h1 threaded back in

        var (_, hN) = Recurrent.GRU(xSeq, hiddenSize: hv);
        var hNFlat = hN.Reshape([n, Scalar(hv)]);                    // [N, H]

        return RnnRefHelpers.RelL1(h2, hNFlat) < Scalar(1e-4f);
    }
}

/// <summary>§7-6 (GRUCell) Trainable-corner FD grad check, BOTH lbr forms. The loss Σh' over a single
/// GRUCell step (x AND h built from the probed scalar so the gradient threads through the h-input too) is
/// FD-checked for linearBeforeReset:true and :false against ORT's own forward. Mirrors GruForwardGradCheck.</summary>
[Module]
public partial class GruCellForwardGradCheckBothLbr
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        Func<Scalar<float32>, bool, Scalar<float32>> step = (z, lbr) =>
        {
            var zv = (Tensor<float32>)OnnxOp.Unsqueeze(z, Vector(0L));
            var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var h = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(-0.2f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
            var hOut = Recurrent.GRUCell(x, h, hiddenSize: 2L, linearBeforeReset: lbr);
            return hOut.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        };
        Func<Scalar<float32>, Scalar<float32>> fTrue = z => step(z, true);
        Func<Scalar<float32>, Scalar<float32>> fFalse = z => step(z, false);
        var gradTrue = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(v, fTrue(v));
        var gradFalse = Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(v, fFalse(v));
        return AutoGradCheckHelpers.ScalarDirectionalDerivCheck(v, gradTrue, fTrue)
             & AutoGradCheckHelpers.ScalarDirectionalDerivCheck(v, gradFalse, fFalse);
    }
}

/// <summary>§7-7 (GRUCell) trainability rig model: a hand-unrolled 2-step GRUCell (linearBeforeReset:true)
/// loop reducing the final hidden state to a per-batch logit pair, for a TrainingRig FromScratch /
/// TrainStep + L2Loss + SGD. The owned W/R/bias differentiate end-to-end through the user loop. h_0 is a
/// zero seed. Output [N, 2].</summary>
[Module]
public partial class GruCellTrainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // input is [L=2, N, in]
    {
        long hv = 4L;
        var n = input.DimTensor(1);
        Scalar<int64> inSize = input.DimTensor(-1);

        var x0 = input.Slice(Vector(0L), Vector(1L), Vector(0L)).Reshape([n, inSize]);
        var x1 = input.Slice(Vector(1L), Vector(2L), Vector(0L)).Reshape([n, inSize]);

        var h0 = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.0f);
        var h1 = Recurrent.GRUCell(x0, h0, hiddenSize: hv);
        var h2 = Recurrent.GRUCell(x1, h1, hiddenSize: hv);

        var f0 = h2.Reduce(ReduceKind.Sum, (Vector<int64>)[Scalar(1L)], keepDims: false).Reshape([n, Scalar(1L)]);
        var f1 = h2.Reduce(ReduceKind.Mean, (Vector<int64>)[Scalar(1L)], keepDims: false).Reshape([n, Scalar(1L)]);
        return f0.Concat(1L, f1);   // [N, 2]
    }
}

// ---------------------------------------------------------------------------
// Constant + Orthogonal initializer coverage (constant-init / orthogonal-init
// design §7). Each self-checking [Module] materializes the seeded init graph and
// asserts on the produced constant, exactly like NNLinearMatchesManualMatMul
// exercises KaimingUniform.Init. A trivial 0*x touch folds the runtime input in
// (the params are input-independent), and the multi-clause value checks use the
// same NaN-safe ok-counting Within(...) idiom as the loss checks above; the bit
// threshold is `> (N-1)` so EVERY ok-bit must be set.
// ---------------------------------------------------------------------------

/// <summary>Constant.Init must fill every element of the parameter with the supplied scalar value:
/// a [2,3] param filled with 7 has max|w − 7| ≈ 0.</summary>
[Module]
public partial class NNConstantInitFillsValue
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var w = Constant.Init([Scalar(2L), Scalar(3L)], Scalar(7f));   // [2,3] all == 7
        var maxDev = (w - Scalar(7f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return (maxDev + touch) < Scalar(1e-6f);
    }
}

/// <summary>Constant.Init is shape/rank-agnostic (no rank≥2 requirement) and reproduces negative
/// values: a rank-1 [4] bias filled with −2.5 has every element == −2.5, so max|w + 2.5| ≈ 0 AND
/// Reduce(Sum) == 4·(−2.5) == −10.</summary>
[Module]
public partial class NNConstantInitRank1Negative
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var w = Constant.Init([Scalar(4L)], Scalar(-2.5f));   // rank-1 [4] all == -2.5
        var maxDev = (w - Scalar(-2.5f)).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var sum = w.Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within(maxDev, 1e-6f)
               + Within((sum - Scalar(-10f)).Abs(), 1e-5f);
        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(2L);   // both ok-bits + touch (3 total); > (3-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>Constant specializes Zeros and Ones: Constant.Init([2,3], Scalar(0f)) == Zeros.Init([2,3])
/// and Constant.Init([2,3], Scalar(1f)) == Ones.Init([2,3]), each by relative-L1 ≈ 0.</summary>
[Module]
public partial class NNConstantInitMatchesZerosOnes
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        Vector<int64> shape = [Scalar(2L), Scalar(3L)];

        var c0 = Constant.Init(shape, Scalar(0f));
        var zeros = Zeros.Init(shape);
        var c1 = Constant.Init(shape, Scalar(1f));
        var ones = Ones.Init(shape);

        var zeroPen = (c0 - zeros).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var onesDiff = (c1 - ones).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var onesScale = Scalar(1f) + ones.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within(zeroPen, 1e-6f)
               + Within(onesDiff - Scalar(1e-6f) * onesScale, 0f);
        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(2L);   // both ok-bits + touch (3 total); > (3-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

// --- Orthogonal: the Björck approximation's convergence quality. Materialize
// Q = Orthogonal.Init([...]), form the Gram matrix, and assert it ≈ I (built via
// NN.EyeLike, as NNStaticWrapperWindowEyeDetCheck does). Tolerances are the
// EMPIRICALLY OBSERVED converged max|G − I| for the seed-19 matrices (probed at
// implementation time): [4,4] square 6.46e-3, [4,2] tall 1.19e-7, [2,4] wide
// 5.96e-8 — i.e. tall/wide converge to ~machine-eps, the square case to ~6.5e-3
// (15 Björck steps from a Frobenius-normalized seed). The asserted bounds are
// true (not loosened) bounds just above the observed errors.

/// <summary>[4,4] square Orthogonal: G = Qᵀ·Q ([4,4]) ≈ I_4 — every singular value driven to 1.
/// Observed converged max|G − I_4| = 6.46e-3 (seed 19); asserted &lt; 1e-2.</summary>
[Module]
public partial class NNOrthogonalSquareGramIsIdentity
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var q = Orthogonal.Init([Scalar(4L), Scalar(4L)]);   // [4,4]
        var g = q.Transpose(1L, 0L).MatMul(q);               // QᵀQ [4,4]
        var eye = NN.EyeLike<float32>(g);
        var maxDev = (g - eye).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return (maxDev + touch) < Scalar(1e-2f);
    }
}

/// <summary>[4,2] tall Orthogonal (r&gt;c): G = Qᵀ·Q ([2,2]) ≈ I_2 — columns orthonormal.
/// Observed converged max|G − I_2| = 1.19e-7 (seed 19); asserted &lt; 1e-5.</summary>
[Module]
public partial class NNOrthogonalTallGramIsIdentity
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var q = Orthogonal.Init([Scalar(4L), Scalar(2L)]);   // [4,2]
        var g = q.Transpose(1L, 0L).MatMul(q);               // QᵀQ [2,2]
        var eye = NN.EyeLike<float32>(g);
        var maxDev = (g - eye).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return (maxDev + touch) < Scalar(1e-5f);
    }
}

/// <summary>[2,4] wide Orthogonal (r&lt;c): G = Q·Qᵀ ([2,2]) ≈ I_2 — rows orthonormal (exercises the
/// r&lt;c branch). Observed converged max|G − I_2| = 5.96e-8 (seed 19); asserted &lt; 1e-5.</summary>
[Module]
public partial class NNOrthogonalWideGramIsIdentity
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var q = Orthogonal.Init([Scalar(2L), Scalar(4L)]);   // [2,4]
        var g = q.MatMul(q.Transpose(1L, 0L));               // QQᵀ [2,2]
        var eye = NN.EyeLike<float32>(g);
        var maxDev = (g - eye).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return (maxDev + touch) < Scalar(1e-5f);
    }
}

// ---------------------------------------------------------------------------
// Configurable UniformRange + NormalDist initializer coverage (configurable-
// uniform-normal design §7). Each self-checking [Module] materializes a large
// seeded sample via <Init>.Init([...], Scalar(...), Scalar(...)) and asserts on
// its empirical statistics, exactly like NNConstantInitFillsValue exercises
// Constant.Init. The multi-clause checks use the same NaN-safe ok-counting
// Within(...) idiom; the bit threshold is `> (N-1)` so EVERY ok-bit must hold.
// Observed seeded statistics (probed at test-authoring time, seeds 20/21):
//   UniformRange[1000](2,5):   min 2.00047, max 4.99732, mean 3.48868
//   UniformRange[1000](-1,1):  min -0.99969, max 0.99821, mean -0.00754
//   NormalDist[10000](10,0.5): mean 10.00530, std 0.49979
//   NormalDist[10000](0,2):    mean 0.02117,  std 1.99934
// The asserted tolerances bracket these observed values tightly (the affine
// shift and scale are both non-trivial, so a dropped +low/+mean or a wrong
// width/std factor fails the check).
// ---------------------------------------------------------------------------

/// <summary>UniformRange.Init over a large sample lies wholly in [low, high], spans the range, and
/// has the midpoint mean. Two ranges: a shifted (2,5) — all in [2,5], min&lt;2.2, max&gt;4.8, mean≈3.5
/// (observed 2.00047 / 4.99732 / 3.48868); and a symmetric (−1,1) — all in [−1,1], mean≈0
/// (observed −0.99969 / 0.99821 / −0.00754). Bounds and span discriminate a constant / wrong
/// sub-range; the midpoint mean discriminates a dropped shift or wrong width.</summary>
[Module]
public partial class NNUniformRangeInRange
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // (2, 5): shifted, non-(0,1) range.
        var w = UniformRange.Init([Scalar(1000L)], Scalar(2f), Scalar(5f));
        var lo = w.Reduce(ReduceKind.Min, keepDims: false).Scalar();
        var hi = w.Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var mean = w.Reduce(ReduceKind.Mean, keepDims: false).Scalar();

        // Bound violations are the positive part of crossing the boundary, required ≈ 0:
        //   lo >= low  ⇒  Relu(low − lo) == 0;   hi <= high  ⇒  Relu(hi − high) == 0.
        var ok = Within((Scalar(2f) - lo).Relu(), 1e-4f)           // lo >= 2  (every draw >= low)
               + Within((hi - Scalar(5f)).Relu(), 1e-4f)           // hi <= 5  (every draw <= high)
               + AtMost(lo, 2.2f)                                  // spans low side: min < 2.2
               + AtLeast(hi, 4.8f)                                 // spans high side: max > 4.8
               + Within((mean - Scalar(3.5f)).Abs(), 0.1f);        // midpoint mean ≈ 3.5

        // (-1, 1): symmetric range, mean ≈ 0.
        var w2 = UniformRange.Init([Scalar(1000L)], Scalar(-1f), Scalar(1f));
        var lo2 = w2.Reduce(ReduceKind.Min, keepDims: false).Scalar();
        var hi2 = w2.Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var mean2 = w2.Reduce(ReduceKind.Mean, keepDims: false).Scalar();

        ok += Within((Scalar(-1f) - lo2).Relu(), 1e-4f)            // lo2 >= -1
            + Within((hi2 - Scalar(1f)).Relu(), 1e-4f)             // hi2 <= 1
            + Within(mean2.Abs(), 0.1f);                           // mean ≈ 0

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(8L);   // all 8 ok-bits + touch (9 total); > (9-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtMost(Scalar<float32> v, float bound)
        => (v <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeast(Scalar<float32> v, float bound)
        => (v >= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>NormalDist.Init over a large sample matches the requested mean and std (std =
/// sqrt(mean(w²) − mean(w)²)). Two cases: a shifted, non-unit (10, 0.5) — mean≈10 within 0.05,
/// std≈0.5 within 0.05 (observed 10.00530 / 0.49979); and (0, 2) — mean≈0 within 0.05, std≈2
/// within 0.05 (observed 0.02117 / 1.99934). The non-zero mean discriminates a dropped +mean;
/// the non-unit std discriminates a wrong scale factor.</summary>
[Module]
public partial class NNNormalDistMoments
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // (10, 0.5): shifted, non-unit.
        var w = NormalDist.Init([Scalar(10000L)], Scalar(10f), Scalar(0.5f));
        var mean = w.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        var std = ((w * w).Reduce(ReduceKind.Mean, keepDims: false).Scalar() - mean * mean).Sqrt();

        // (0, 2): zero mean, larger std.
        var w2 = NormalDist.Init([Scalar(10000L)], Scalar(0f), Scalar(2f));
        var mean2 = w2.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        var std2 = ((w2 * w2).Reduce(ReduceKind.Mean, keepDims: false).Scalar() - mean2 * mean2).Sqrt();

        var ok = Within((mean - Scalar(10f)).Abs(), 0.05f)
               + Within((std - Scalar(0.5f)).Abs(), 0.05f)
               + Within(mean2.Abs(), 0.05f)
               + Within((std2 - Scalar(2f)).Abs(), 0.05f);

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(4L);   // all 4 ok-bits + touch (5 total); > (5-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

// ---------------------------------------------------------------------------
// Configurable-gain Xavier/Kaiming initializer coverage (configurable-gain
// design §7). Each *Gain class is materialized via <Init>.Init([64,64],
// Scalar(gain)) and checked through its empirical sample std (= sqrt(mean(w²) −
// mean(w)²)), exactly like NNNormalDistMoments exercises NormalDist. On a SQUARE
// [64,64] shape (fanIn = fanOut = 64) all four collapse to the SAME closed form,
// so at gain=2 every one has std ≈ 0.25. The module uses the same NaN-safe
// ok-counting Within(...) idiom; the bit threshold is `> (N-1)` so EVERY ok-bit
// must hold. Bands set from the observed seeded values (probed at authoring time
// on the seed-22..25 draws; 4096-sample std-estimator error ≈ 0.003):
//   gain=2:  XavierUniformGain 0.252211, XavierNormalGain 0.248208,
//            KaimingUniformGain 0.249631, KaimingNormalGain 0.252713
//   gain=1:  XavierUniformGain 0.126106  (== XavierUniform baked 0.124789)
//   gain=√2: KaimingUniformGain 0.176516 (== KaimingUniform baked 0.177119)
// The gain=2 band ±0.015 is tight around 0.25 yet brackets all four observed
// values; crucially it EXCLUDES the √6-double-bake value 0.354 (a buggy Kaiming
// using √(6/fanIn) would give 2·√(6/64)/√3 = 0.354), so the check discriminates
// the §4.1 double-bake trap.
// ---------------------------------------------------------------------------

/// <summary>All four configurable-gain inits (XavierUniformGain / XavierNormalGain /
/// KaimingUniformGain / KaimingNormalGain) on a SQUARE [64,64] (fanIn = fanOut = 64) have the SAME
/// sample std ≈ 0.25 at gain=2 (Xavier-uniform 2·√(6/128)/√3, Xavier-normal 2·√(2/128),
/// Kaiming-uniform 2·√(3/64)/√3, Kaiming-normal 2·√(1/64) — all 0.25). The ±0.015 band around 0.25
/// brackets the observed 0.2482–0.2527 yet EXCLUDES the √6-double-bake std 0.354, so a Kaiming using
/// √6/√2 instead of the bare √3/√1 base fails. Plus the gain-specialization equivalences: gain=1
/// XavierUniformGain std ≈ 0.125 ≈ the baked XavierUniform std, and gain=√2 KaimingUniformGain
/// std ≈ 0.1767 ≈ the baked KaimingUniform std (asserted both as an absolute ≈ AND as ≈ the
/// materialized baked default's std — pinning the √2-equivalence).</summary>
[Module]
public partial class NNXavierKaimingGainStd
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        Vector<int64> sq = [Scalar(64L), Scalar(64L)];

        // gain=2 on the square shape: all four collapse to std ≈ 0.25. Tight ±0.015 band
        // (observed 0.2482–0.2527); excludes the √6-double-bake value 0.354.
        var ok = Within((Std(XavierUniformGain.Init(sq, Scalar(2f))) - Scalar(0.25f)).Abs(), 0.015f)
               + Within((Std(XavierNormalGain.Init(sq, Scalar(2f))) - Scalar(0.25f)).Abs(), 0.015f)
               + Within((Std(KaimingUniformGain.Init(sq, Scalar(2f))) - Scalar(0.25f)).Abs(), 0.015f)
               + Within((Std(KaimingNormalGain.Init(sq, Scalar(2f))) - Scalar(0.25f)).Abs(), 0.015f);

        // gain=1 XavierUniformGain reproduces the baked XavierUniform default: std ≈ √(6/128)/√3 = 0.125.
        var xu1 = Std(XavierUniformGain.Init(sq, Scalar(1f)));
        var xuBaked = Std(XavierUniform.Init(sq));
        ok += Within((xu1 - Scalar(0.125f)).Abs(), 0.01f)          // gain-1 ≈ 0.125
            + Within((xu1 - xuBaked).Abs(), 0.01f);                // ≈ the baked default's std

        // gain=√2 KaimingUniformGain reproduces the baked KaimingUniform default:
        // √2·√(3/64)/√3 = √(6/64)/√3 = 0.17678.
        var ksqrt2 = Std(KaimingUniformGain.Init(sq, Scalar(1.4142135f)));
        var kuBaked = Std(KaimingUniform.Init(sq));
        ok += Within((ksqrt2 - Scalar(0.17678f)).Abs(), 0.01f)     // gain-√2 ≈ 0.1768
            + Within((ksqrt2 - kuBaked).Abs(), 0.01f);             // ≈ the baked default's std

        var touch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(8L);   // all 8 ok-bits + touch (9 total); > (9-1)
    }

    /// <summary>sample std = sqrt(mean(w²) − mean(w)²).</summary>
    private static Scalar<float32> Std(Tensor<float32> w)
    {
        var mean = w.Reduce(ReduceKind.Mean, keepDims: false).Scalar();
        return ((w * w).Reduce(ReduceKind.Mean, keepDims: false).Scalar() - mean * mean).Sqrt();
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

// ---------------------------------------------------------------------------
// TripletMarginLoss / TripletMarginWithDistance coverage (triplet-margin-loss
// design §9). Self-checking [Module]s in the established loss style: in-module
// constant anchor/positive/negative, a zero-scaled runtime touch so AutoTest has
// an input to drive, and the NaN-safe Within(...) / AtLeastZero(...) ok-counting
// idiom (a NaN fails every comparison so it can never slip a check). The
// closed-form reference is hand-computed in double precision (see the per-module
// docs) AND, for the load-bearing closed form, cross-checked against an
// INDEPENDENT graph reference distance d(x,y)=((x−y)²).sum(-1).sqrt() built from
// raw tensor ops (NOT by re-calling TripletMarginLoss).
// ---------------------------------------------------------------------------

/// <summary>
/// TripletMarginLoss load-bearing closed form (design §9.1) on a fixed
/// <c>[N=3, D=3]</c> batch, default <c>p=2</c> (Euclidean), <c>margin=1</c>,
/// <c>eps=1e-6</c>, <c>swap=false</c>. Anchors all <c>[0,0,0]</c>:
/// <list type="bullet">
///   <item>s0: p=[1,0,0], n=[0,2,0] → dAp=1, dAn=2, L=relu(1+1−2)=0;</item>
///   <item>s1: p=[2,0,0], n=[2.5,0,0] → dAp=2, dAn=2.5, L=relu(1+2−2.5)=0.5;</item>
///   <item>s2: p=[2,0,0], n=[1,0,0] → dAp=2, dAn=1, L=relu(1+2−1)=2.</item>
/// </list>
/// PerElement=[0,0.5,2]; Inline(mean)=0.8333333; Reduced(Sum)=2.5. The PerElement
/// vector is ALSO asserted equal to an independently-built reference
/// <c>relu(margin + dApRef − dAnRef)</c> where dApRef/dAnRef come from a raw-op
/// Euclidean distance, so the closed form does not lean on the module under test.
/// </summary>
[Module]
public partial class NNTripletMarginClosedFormChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var anchor = Tensor(new long[] { 3L, 3L },
            0f, 0f, 0f,  0f, 0f, 0f,  0f, 0f, 0f);
        var positive = Tensor(new long[] { 3L, 3L },
            1f, 0f, 0f,  2f, 0f, 0f,  2f, 0f, 0f);
        var negative = Tensor(new long[] { 3L, 3L },
            0f, 2f, 0f,  2.5f, 0f, 0f,  1f, 0f, 0f);

        var margin = Scalar(1f);
        var p = Scalar(2f);
        var eps = Scalar(1e-6f);

        var mean = TripletMarginLoss.Call(margin, p, eps, Scalar(false), anchor, positive, negative); // 0.8333333
        var sum = TripletMarginLoss.Reduced(margin, p, eps, Scalar(false), anchor, positive, negative,
            reduction: LossReduction.Sum);                                                            // 2.5

        // PerElement → [0, 0.5, 2]; check each element via Slice.
        var per = TripletMarginLoss.PerElement(margin, p, eps, Scalar(false), anchor, positive, negative); // [3]
        var per0 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(0L), Vector(1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per1 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(1L), Vector(2L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var per2 = ((Tensor<float32>)OnnxOp.Slice(per, Vector(2L), Vector(3L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        // INDEPENDENT reference: relu(margin + d(a,p) − d(a,n)) with a raw-op
        // Euclidean distance (no eps, no module re-call). Tolerance absorbs the
        // module's eps=1e-6 inside-the-root term.
        var perRef = (Scalar(1f) + Euclid(anchor, positive) - Euclid(anchor, negative)).Relu(); // [3]
        var refMismatch = (per - perRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((mean - Scalar(0.8333333f)).Abs(), 1e-4f)
               + Within((sum - Scalar(2.5f)).Abs(), 1e-4f)
               + Within(per0.Abs(), 1e-4f) + AtLeastZero(per0)
               + Within((per1 - Scalar(0.5f)).Abs(), 1e-4f)
               + Within((per2 - Scalar(2f)).Abs(), 1e-4f)
               + Within(refMismatch, 1e-3f);   // PerElement == independent raw-op reference

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(7L);   // all 7 ok-bits + touch (8 total); > (8-1)
    }

    /// <summary>Independent Euclidean distance d(x,y)=sqrt(Σ_feat (x−y)²) over the last axis [N].</summary>
    private static Tensor<float32> Euclid(Tensor<float32> x, Tensor<float32> y)
    {
        var diff = x - y;
        return (diff * diff).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false).Sqrt();
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// TripletMarginLoss <c>swap</c> (Balntas anchor swap), <c>margin</c> and <c>p</c>
/// knobs (design §9.2/§9.3), each via a hand-computed single-triplet case.
/// <list type="bullet">
///   <item><b>swap</b>: a=[0,0,0], p=[2,0,0], n=[3,0,0], margin=1 → dAp=2, dAn=3,
///     d(p,n)=1. swap=false: dNeg=3, L=relu(1+2−3)=0. swap=true: dNeg=min(3,1)=1,
///     L=relu(1+2−1)=2. swap RAISES the loss 0→2 (pins the min + the bit-gate).</item>
///   <item><b>margin</b> sweep on the fixed violating triplet a=[0,0,0],p=[2,0,0],
///     n=[2.5,0,0] (dAp=2, dAn=2.5): margin=0 → relu(−0.5)=0; margin=1 → 0.5;
///     margin=2 → relu(1.5)=1.5.</item>
///   <item><b>p</b> variation a=[0,0], p=[1,1], n=[1,2], margin=1: p=2 →
///     dAp=√2≈1.4142136, dAn=√5≈2.2360680, L=relu(1+1.4142136−2.2360680)=0.1781456;
///     p=1 → dAp=2, dAn=3, L=relu(1+2−3)=0. Different p ⇒ different loss (0.178 vs 0).</item>
/// </list>
/// </summary>
[Module]
public partial class NNTripletMarginSwapMarginPChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var eps = Scalar(1e-6f);

        // --- swap on/off: d(p,n) < d(a,n) so swap shrinks dNeg and grows the loss.
        var a = Tensor(new long[] { 1L, 3L }, 0f, 0f, 0f);
        var pos = Tensor(new long[] { 1L, 3L }, 2f, 0f, 0f);
        var neg = Tensor(new long[] { 1L, 3L }, 3f, 0f, 0f);
        var swapOff = TripletMarginLoss.Call(Scalar(1f), Scalar(2f), eps, Scalar(false), a, pos, neg); // 0
        var swapOn = TripletMarginLoss.Call(Scalar(1f), Scalar(2f), eps, Scalar(true), a, pos, neg);   // 2

        // --- margin sweep on a fixed violating triplet (dAp=2, dAn=2.5).
        var ma = Tensor(new long[] { 1L, 3L }, 0f, 0f, 0f);
        var mp = Tensor(new long[] { 1L, 3L }, 2f, 0f, 0f);
        var mn = Tensor(new long[] { 1L, 3L }, 2.5f, 0f, 0f);
        var m0 = TripletMarginLoss.Call(Scalar(0f), Scalar(2f), eps, Scalar(false), ma, mp, mn); // 0
        var m1 = TripletMarginLoss.Call(Scalar(1f), Scalar(2f), eps, Scalar(false), ma, mp, mn); // 0.5
        var m2 = TripletMarginLoss.Call(Scalar(2f), Scalar(2f), eps, Scalar(false), ma, mp, mn); // 1.5

        // --- p variation where L1 ≠ L2 (a=[0,0], p=[1,1], n=[1,2]).
        var pa = Tensor(new long[] { 1L, 2L }, 0f, 0f);
        var pp = Tensor(new long[] { 1L, 2L }, 1f, 1f);
        var pn = Tensor(new long[] { 1L, 2L }, 1f, 2f);
        var lP2 = TripletMarginLoss.Call(Scalar(1f), Scalar(2f), eps, Scalar(false), pa, pp, pn); // 0.1781456
        var lP1 = TripletMarginLoss.Call(Scalar(1f), Scalar(1f), eps, Scalar(false), pa, pp, pn); // 0

        var ok = Within(swapOff.Abs(), 1e-4f) + AtLeastZero(swapOff)
               + Within((swapOn - Scalar(2f)).Abs(), 1e-4f)
               + Within(m0.Abs(), 1e-4f) + AtLeastZero(m0)
               + Within((m1 - Scalar(0.5f)).Abs(), 1e-4f)
               + Within((m2 - Scalar(1.5f)).Abs(), 1e-4f)
               + Within((lP2 - Scalar(0.1781456f)).Abs(), 1e-4f)
               + Within(lP1.Abs(), 1e-4f) + AtLeastZero(lP1);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(10L);   // all 10 ok-bits + touch (11 total); > (11-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// TripletMarginLoss reduction-mode equivalences (design §9.4): on the §9.1
/// fixed batch (per-triplet losses [0, 0.5, 2]),
/// <c>Reduced(Mean) == Inline</c> (mean 0.8333333) and
/// <c>Reduced(Sum)</c> equals the sum of the PerElement vector (2.5). The
/// <c>Reduced(None)-throws</c> case is a C#-level [Fact]
/// (TestTripletMarginReducedNoneThrows), not a graph check.
/// </summary>
[Module]
public partial class NNTripletMarginReductionChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var anchor = Tensor(new long[] { 3L, 3L },
            0f, 0f, 0f,  0f, 0f, 0f,  0f, 0f, 0f);
        var positive = Tensor(new long[] { 3L, 3L },
            1f, 0f, 0f,  2f, 0f, 0f,  2f, 0f, 0f);
        var negative = Tensor(new long[] { 3L, 3L },
            0f, 2f, 0f,  2.5f, 0f, 0f,  1f, 0f, 0f);

        var margin = Scalar(1f);
        var p = Scalar(2f);
        var eps = Scalar(1e-6f);

        var inline = TripletMarginLoss.Call(margin, p, eps, Scalar(false), anchor, positive, negative);
        var redMean = TripletMarginLoss.Reduced(margin, p, eps, Scalar(false), anchor, positive, negative,
            reduction: LossReduction.Mean);
        var redSum = TripletMarginLoss.Reduced(margin, p, eps, Scalar(false), anchor, positive, negative,
            reduction: LossReduction.Sum);

        // Sum of the PerElement vector, built independently of Reduced(Sum).
        var perSum = TripletMarginLoss.PerElement(margin, p, eps, Scalar(false), anchor, positive, negative)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((redMean - inline).Abs(), 1e-5f)     // Reduced(Mean) == Inline
               + Within((redMean - Scalar(0.8333333f)).Abs(), 1e-4f)
               + Within((redSum - perSum).Abs(), 1e-5f)      // Reduced(Sum) == Σ PerElement
               + Within((redSum - Scalar(2.5f)).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(4L);   // all 4 ok-bits + touch (5 total); > (5-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// TripletMarginWithDistance (caller-supplied distance Func) coverage (design
/// §9.7). Two checks pin the Func plumbing + the build-time-bool swap:
/// <list type="bullet">
///   <item><b>Custom squared-L2</b> distance <c>d(x,y)=Σ(x−y)²</c> (no root):
///     a=[0], p=[2], n=[2.2], margin=1 → d_ap=4, d_an=4.84, L=relu(1+4−4.84)=0.16.</item>
///   <item><b>Equivalence pin</b>: passing the p=2 Euclidean Func
///     <c>d(x,y)=sqrt(Σ(x−y)²)</c> reproduces the built-in TripletMarginLoss(p=2)
///     value on the §9.1 fixed batch (mean 0.8333333), confirming the skeleton is
///     identical and only the distance is pluggable.</item>
///   <item><b>swap on the custom distance</b>: a=[0,0,0], p=[2,0,0], n=[3,0,0]
///     with the squared-L2 Func → d_ap=4, d_an=9, d_pn=1. swap=false:
///     relu(1+4−9)=0; swap=true: dNeg=min(9,1)=1, relu(1+4−1)=4.</item>
/// </list>
/// </summary>
[Module]
public partial class NNTripletMarginWithDistanceChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        // Custom distance: squared-L2 over the last axis (NO sqrt). [N]
        Func<Tensor<float32>, Tensor<float32>, Tensor<float32>> sqL2 =
            (x, y) => { var d = x - y; return (d * d).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false); };
        // p=2 Euclidean Func: sqrt of the above — should reproduce the built-in (p=2).
        Func<Tensor<float32>, Tensor<float32>, Tensor<float32>> euclid =
            (x, y) => { var d = x - y; return (d * d).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false).Sqrt(); };

        // --- custom squared-L2 closed form: a=[0], p=[2], n=[2.2] → L=0.16.
        var a1 = Tensor(new long[] { 1L, 1L }, 0f);
        var p1 = Tensor(new long[] { 1L, 1L }, 2f);
        var n1 = Tensor(new long[] { 1L, 1L }, 2.2f);
        var custom = TripletMarginWithDistance.PerElement(sqL2, 1f, false, a1, p1, n1)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();   // 0.16

        // --- equivalence: euclid Func == built-in TripletMarginLoss(p=2) on §9.1 batch.
        var anchor = Tensor(new long[] { 3L, 3L },
            0f, 0f, 0f,  0f, 0f, 0f,  0f, 0f, 0f);
        var positive = Tensor(new long[] { 3L, 3L },
            1f, 0f, 0f,  2f, 0f, 0f,  2f, 0f, 0f);
        var negative = Tensor(new long[] { 3L, 3L },
            0f, 2f, 0f,  2.5f, 0f, 0f,  1f, 0f, 0f);
        var withDist = TripletMarginWithDistance.Reduced(euclid, 1f, false, anchor, positive, negative,
            reduction: LossReduction.Mean);                                                          // 0.8333333
        var builtin = TripletMarginLoss.Call(Scalar(1f), Scalar(2f), Scalar(1e-6f), Scalar(false),
            anchor, positive, negative);

        // --- swap on the custom (squared-L2) distance: a=[0,0,0],p=[2,0,0],n=[3,0,0].
        var sa = Tensor(new long[] { 1L, 3L }, 0f, 0f, 0f);
        var sp = Tensor(new long[] { 1L, 3L }, 2f, 0f, 0f);
        var sn = Tensor(new long[] { 1L, 3L }, 3f, 0f, 0f);
        var sOff = TripletMarginWithDistance.PerElement(sqL2, 1f, false, sa, sp, sn)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();   // 0
        var sOn = TripletMarginWithDistance.PerElement(sqL2, 1f, true, sa, sp, sn)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();   // 4

        var ok = Within((custom - Scalar(0.16f)).Abs(), 1e-4f)
               + Within((withDist - builtin).Abs(), 1e-4f)        // Func(euclid) == built-in p=2
               + Within((withDist - Scalar(0.8333333f)).Abs(), 1e-4f)
               + Within(sOff.Abs(), 1e-4f) + AtLeastZero(sOff)
               + Within((sOn - Scalar(4f)).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(6L);   // all 6 ok-bits + touch (7 total); > (7-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// Rig-trainability model for TripletMarginLoss (triplet-margin-loss design §5/§9.5
/// "loss-is-the-model-tail" recipe). The model takes a single <c>[3N, D]</c> input
/// batch, splits it into anchor / positive / negative blocks of <c>N</c> rows each,
/// runs a SHARED trainable Linear embedding on each block, and RETURNS THE SCALAR
/// triplet loss as the model output. Paired with the IdentityScalarLoss adapter in
/// the rig's 2-input loss slot, this trains the embedding end-to-end — exercising
/// autodiff through Pow/Sum/Min(via swap=true)/Relu/IfElse. With N=2, D=2 the input
/// is [6, 2]: rows 0-1 anchor, 2-3 positive, 4-5 negative.
/// </summary>
[Module]
public partial class NNTripletEmbeddingRigModel
{
    public static Scalar<float32> Inline(Tensor<float32> apn)   // [3N, D] = [6, 2]
    {
        var embed = Linear.Model(Scalar(2L), Scalar(true));     // shared trainable embedding [D]→[2]
        var ea = embed.Call((Tensor<float32>)OnnxOp.Slice(apn, Vector(0L), Vector(2L)));   // anchor   [2,2]
        var ep = embed.Call((Tensor<float32>)OnnxOp.Slice(apn, Vector(2L), Vector(4L)));   // positive [2,2]
        var en = embed.Call((Tensor<float32>)OnnxOp.Slice(apn, Vector(4L), Vector(6L)));   // negative [2,2]
        // margin=10 keeps the hinge ACTIVE regardless of the random init (the
        // embedding distances are O(few)), so the triplet loss is nonzero and the
        // gradient flows; swap=true so the Min arm is on the autodiff path too.
        return TripletMarginLoss.Call(Scalar(10f), Scalar(2f), Scalar(1e-6f), Scalar(true), ea, ep, en);
    }
}

/// <summary>
/// The generic "my loss is already computed in the model" rig adapter
/// (triplet-margin-loss design §5): a 2-input <c>(prediction, target)</c> loss
/// that just returns the scalar <c>prediction</c> (the model already produced the
/// loss), with a zero-scaled touch of <c>target</c> to honour the rig's 2-input
/// contract. Lets any non-(pred,target) objective (here TripletMarginLoss) train
/// through the rig.
/// </summary>
[Module]
public partial class NNIdentityScalarLoss
{
    public static Scalar<float32> Inline(Scalar<float32> prediction, Tensor<float32> target)
        => prediction + (target * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
}

// ---------------------------------------------------------------------------
// CosineEmbeddingLoss (+ CosineSimilarity helper) coverage
// (cosine-embedding-loss design §9). Self-checking [Module]s in the established
// loss style: in-module constant x1/x2/y, a zero-scaled runtime touch so AutoTest
// has an input to drive, and the NaN-safe Within(...) / AtLeastZero(...)
// ok-counting idiom (a NaN fails every comparison so it can never slip a check).
// The load-bearing closed form (§9.1) cross-checks the per-sample [N] loss against
// an INDEPENDENT raw-op reference: cos built by hand from (x1·x2).sum(-1) /
// (‖x1‖·‖x2‖), then the where(y==1, 1−cos, relu(cos−margin)) split — NOT by
// re-calling the module. Cosines are exact (identical ⇒ 1, orthogonal ⇒ 0,
// anti-parallel ⇒ −1, 45° ⇒ 1/√2 ≈ 0.70710678; eps=1e-8 inert at every
// non-degenerate pair).
// ---------------------------------------------------------------------------

/// <summary>
/// CosineEmbeddingLoss load-bearing closed form (design §9.1) covering BOTH y
/// branches on a fixed <c>[N=6, D=2]</c> batch, <c>margin=0</c>, <c>eps=1e-8</c>.
/// All <c>x1=[1,0]</c>; the per-sample cosine and loss are hand-computed:
/// <list type="bullet">
///   <item>s0: x2=[2,0], y=+1, cos=1 → L=1−1=0;</item>
///   <item>s1: x2=[0,1], y=+1, cos=0 → L=1−0=1;</item>
///   <item>s2: x2=[-1,0], y=+1, cos=−1 → L=1−(−1)=2;</item>
///   <item>s3: x2=[1,1], y=+1, cos=1/√2≈0.70710678 → L=1−0.70710678=0.29289322;</item>
///   <item>s4: x2=[2,0], y=−1, cos=1 → L=relu(1−0)=1;</item>
///   <item>s5: x2=[1,1], y=−1, cos=0.70710678 → L=relu(0.70710678−0)=0.70710678.</item>
/// </list>
/// PerElement = [0, 1, 2, 0.29289322, 1, 0.70710678] (each checked via Slice);
/// the WHOLE vector is ALSO asserted equal to an independently-built reference
/// <c>where(y==1, 1−cosRef, relu(cosRef−margin))</c> where <c>cosRef</c> comes from
/// a raw-op <c>(x1·x2).sum(-1) / (‖x1‖·‖x2‖)</c> (no eps, no module re-call), so the
/// closed form does not lean on the module under test. <c>Inline</c>(mean) and
/// <c>Reduced(Sum)</c> are checked against the hand-summed reductions.
/// </summary>
[Module]
public partial class NNCosineEmbeddingClosedFormChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var x1 = Tensor(new long[] { 6L, 2L },
            1f, 0f,  1f, 0f,  1f, 0f,  1f, 0f,  1f, 0f,  1f, 0f);
        var x2 = Tensor(new long[] { 6L, 2L },
            2f, 0f,  0f, 1f,  -1f, 0f,  1f, 1f,  2f, 0f,  1f, 1f);
        var y = Tensor(new long[] { 6L },
            1f, 1f, 1f, 1f, -1f, -1f);

        var margin = Scalar(0f);
        var eps = Scalar(1e-8f);

        // cos = [1, 0, -1, 0.70710678, 1, 0.70710678]
        // L   = [0, 1,  2, 0.29289322, 1, 0.70710678]; sum = 4.99999998..., mean = 0.83333333.
        var mean = CosineEmbeddingLoss.Call(margin, eps, x1, x2, y);                       // 0.83333334
        var sum = CosineEmbeddingLoss.Reduced(margin, eps, x1, x2, y, reduction: LossReduction.Sum); // 5

        var per = CosineEmbeddingLoss.PerElement(margin, eps, x1, x2, y);                  // [6]
        var per0 = Slice1(per, 0L);
        var per1 = Slice1(per, 1L);
        var per2 = Slice1(per, 2L);
        var per3 = Slice1(per, 3L);
        var per4 = Slice1(per, 4L);
        var per5 = Slice1(per, 5L);

        // INDEPENDENT reference: where(y==1, 1−cosRef, relu(cosRef−margin)) with a
        // raw-op cosine (no eps clamp, no module re-call).
        var cosRef = Cos(x1, x2);
        var perRef = (y == Scalar(1f)).Where(Scalar(1f) - cosRef, (cosRef - margin).Relu()); // [6]
        var refMismatch = (per - perRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((mean - Scalar(0.83333334f)).Abs(), 1e-4f)
               + Within((sum - Scalar(5f)).Abs(), 1e-4f)
               + Within(per0.Abs(), 1e-4f) + AtLeastZero(per0)
               + Within((per1 - Scalar(1f)).Abs(), 1e-4f)
               + Within((per2 - Scalar(2f)).Abs(), 1e-4f)
               + Within((per3 - Scalar(0.29289322f)).Abs(), 1e-4f)
               + Within((per4 - Scalar(1f)).Abs(), 1e-4f)
               + Within((per5 - Scalar(0.70710678f)).Abs(), 1e-4f)
               + Within(refMismatch, 1e-3f);   // PerElement == independent raw-op reference

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(10L);   // all 10 ok-bits + touch (11 total); > (11-1)
    }

    /// <summary>Raw-op cosine cos(x1,x2)=(Σ_feat x1·x2)/(‖x1‖·‖x2‖) over the last axis [N] (no eps).</summary>
    private static Tensor<float32> Cos(Tensor<float32> a, Tensor<float32> b)
    {
        var dot = (a * b).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false);
        var na = (a * a).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false).Sqrt();
        var nb = (b * b).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false).Sqrt();
        return dot / (na * nb);
    }

    private static Scalar<float32> Slice1(Tensor<float32> v, long i)
        => ((Tensor<float32>)OnnxOp.Slice(v, Vector(i), Vector(i + 1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// CosineEmbeddingLoss <c>margin</c> gating (design §9.2): on the SAME single-pair
/// input <c>x1=[1,0]</c>, <c>x2=[1,1]</c> (cos = 1/√2 ≈ 0.70710678), two margins
/// (0 and 0.5):
/// <list type="bullet">
///   <item><b>y=−1 arm changes by the margin</b>: margin=0 ⇒ relu(0.70710678−0)=0.70710678;
///     margin=0.5 ⇒ relu(0.70710678−0.5)=0.20710678 (difference = 0.5 = the margin,
///     since the hinge is active for both).</item>
///   <item><b>y=+1 arm is UNCHANGED by the margin</b>: 1−cos=0.29289322 for both
///     margin=0 and margin=0.5 — a sharp check that margin is gated to the y=−1
///     branch only.</item>
/// </list>
/// </summary>
[Module]
public partial class NNCosineEmbeddingMarginGatingChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var x1 = Tensor(new long[] { 1L, 2L }, 1f, 0f);
        var x2 = Tensor(new long[] { 1L, 2L }, 1f, 1f);   // cos = 0.70710678
        var eps = Scalar(1e-8f);
        var yNeg = Tensor(new long[] { 1L }, -1f);
        var yPos = Tensor(new long[] { 1L }, 1f);

        // y=−1 arm: margin shifts the hinge by exactly the margin (active for both).
        var neg0 = CosineEmbeddingLoss.Call(Scalar(0f), eps, x1, x2, yNeg);     // 0.70710678
        var neg5 = CosineEmbeddingLoss.Call(Scalar(0.5f), eps, x1, x2, yNeg);   // 0.20710678
        var marginShift = neg0 - neg5;                                          // == 0.5

        // y=+1 arm: 1−cos, independent of margin.
        var pos0 = CosineEmbeddingLoss.Call(Scalar(0f), eps, x1, x2, yPos);     // 0.29289322
        var pos5 = CosineEmbeddingLoss.Call(Scalar(0.5f), eps, x1, x2, yPos);   // 0.29289322
        var posUnchanged = (pos0 - pos5).Abs();                                 // == 0

        var ok = Within((neg0 - Scalar(0.70710678f)).Abs(), 1e-4f)
               + Within((neg5 - Scalar(0.20710678f)).Abs(), 1e-4f)
               + Within((marginShift - Scalar(0.5f)).Abs(), 1e-4f)
               + Within((pos0 - Scalar(0.29289322f)).Abs(), 1e-4f)
               + Within(posUnchanged, 1e-5f);   // y=+1 arm independent of margin

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(5L);   // 5 ok-bits + touch (6 total); > (6-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// CosineEmbeddingLoss the <c>where(y==1, …, …)</c> data-dependent split (design
/// §9.3). A batch <c>N=2</c>, <c>x1=[[1,0],[1,0]]</c>, <c>x2=[[2,0],[0,1]]</c>
/// (cos = [1, 0]), <c>margin=0</c>. Same x1/x2, only y flips:
/// <list type="bullet">
///   <item>y=[+1, −1] ⇒ PerElement = [1−1, relu(0−0)] = [0, 0];</item>
///   <item>y=[−1, +1] ⇒ PerElement = [relu(1−0), 1−0] = [1, 1].</item>
/// </list>
/// The per-sample vector changes from [0,0] to [1,1] under NOTHING but the y flip,
/// pinning that <c>.Where(y==1, …)</c> selects the branch per element. Each
/// element checked via Slice.
/// </summary>
[Module]
public partial class NNCosineEmbeddingWhereSplitChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var x1 = Tensor(new long[] { 2L, 2L }, 1f, 0f,  1f, 0f);
        var x2 = Tensor(new long[] { 2L, 2L }, 2f, 0f,  0f, 1f);   // cos = [1, 0]
        var eps = Scalar(1e-8f);
        var margin = Scalar(0f);

        var yA = Tensor(new long[] { 2L }, 1f, -1f);    // → [0, 0]
        var yB = Tensor(new long[] { 2L }, -1f, 1f);    // → [1, 1]

        var perA = CosineEmbeddingLoss.PerElement(margin, eps, x1, x2, yA);
        var perB = CosineEmbeddingLoss.PerElement(margin, eps, x1, x2, yB);

        var a0 = Slice1(perA, 0L);
        var a1 = Slice1(perA, 1L);
        var b0 = Slice1(perB, 0L);
        var b1 = Slice1(perB, 1L);

        var ok = Within(a0.Abs(), 1e-4f) + AtLeastZero(a0)
               + Within(a1.Abs(), 1e-4f) + AtLeastZero(a1)
               + Within((b0 - Scalar(1f)).Abs(), 1e-4f)
               + Within((b1 - Scalar(1f)).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(6L);   // 6 ok-bits + touch (7 total); > (7-1)
    }

    private static Scalar<float32> Slice1(Tensor<float32> v, long i)
        => ((Tensor<float32>)OnnxOp.Slice(v, Vector(i), Vector(i + 1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// CosineEmbeddingLoss reduction-mode equivalences (design §9.4) on an <c>N=3</c>
/// batch with per-sample losses [0, 1, 0.29289322] (all y=+1: x1=[1,0] paired with
/// x2=[2,0] (cos=1, L=0), [0,1] (cos=0, L=1), [1,1] (cos=0.70710678, L=0.29289322)):
/// <c>Reduced(Mean) == Inline</c> (mean 0.43096441) and <c>Reduced(Sum)</c> equals
/// the sum of the PerElement vector (1.29289322). The <c>Reduced(None)-throws</c>
/// case is the separate <see cref="NNLibraryTests"/> C#-level [Fact].
/// </summary>
[Module]
public partial class NNCosineEmbeddingReductionChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var x1 = Tensor(new long[] { 3L, 2L }, 1f, 0f,  1f, 0f,  1f, 0f);
        var x2 = Tensor(new long[] { 3L, 2L }, 2f, 0f,  0f, 1f,  1f, 1f);   // cos = [1, 0, 0.70710678]
        var y = Tensor(new long[] { 3L }, 1f, 1f, 1f);                      // all pull → [0, 1, 0.29289322]
        var margin = Scalar(0f);
        var eps = Scalar(1e-8f);

        var inline = CosineEmbeddingLoss.Call(margin, eps, x1, x2, y);
        var redMean = CosineEmbeddingLoss.Reduced(margin, eps, x1, x2, y, reduction: LossReduction.Mean);
        var redSum = CosineEmbeddingLoss.Reduced(margin, eps, x1, x2, y, reduction: LossReduction.Sum);

        // Sum of the PerElement vector, built independently of Reduced(Sum).
        var perSum = CosineEmbeddingLoss.PerElement(margin, eps, x1, x2, y)
            .Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var ok = Within((redMean - inline).Abs(), 1e-5f)        // Reduced(Mean) == Inline
               + Within((redMean - Scalar(0.43096441f)).Abs(), 1e-4f)
               + Within((redSum - perSum).Abs(), 1e-5f)         // Reduced(Sum) == Σ PerElement
               + Within((redSum - Scalar(1.29289322f)).Abs(), 1e-4f);

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(4L);   // all 4 ok-bits + touch (5 total); > (5-1)
    }

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// CosineSimilarity helper (design §9.5), independent of the loss. Hand-computed
/// per-row cosines over a <c>[N=5, D=2]</c> batch, asserted == a raw-op reference
/// AND the exact constants:
/// <list type="bullet">
///   <item>[1,0]·[1,0] ⇒ 1; [1,0]·[0,1] ⇒ 0; [1,0]·[-1,0] ⇒ −1; [1,0]·[1,1] ⇒ 0.70710678.</item>
///   <item><b>SCALE-INVARIANCE</b>: cos([3,4],[3,4]) == cos([6,8],[3,4]) == 1 — a
///     positive scalar multiple of an input leaves cosine unchanged (the point of
///     cosine). Both checked == 1 AND checked equal to each other.</item>
///   <item><b>EPS guard</b>: cos([0,0],[1,0]) ⇒ dot=0, denom=max(0·1, 1e-8)=1e-8,
///     cos=0 — finite, NOT NaN (asserted via AtLeastZero + Within on |cos|).</item>
/// </list>
/// </summary>
[Module]
public partial class NNCosineSimilarityHelperChecks
{
    public static Scalar<bit> Inline(Tensor<float32> t)
    {
        var eps = Scalar(1e-8f);

        // Base batch: cos = [1, 0, -1, 0.70710678, 1]
        var a = Tensor(new long[] { 5L, 2L },
            1f, 0f,  1f, 0f,  1f, 0f,  1f, 0f,  3f, 4f);
        var b = Tensor(new long[] { 5L, 2L },
            1f, 0f,  0f, 1f,  -1f, 0f,  1f, 1f,  3f, 4f);
        var cos = CosineEmbeddingLoss.CosineSimilarity(a, b, eps);   // [5]
        var cosRef = Cos(a, b);                                      // raw-op reference [5]
        var refMismatch = (cos - cosRef).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var c0 = Slice1(cos, 0L);
        var c1 = Slice1(cos, 1L);
        var c2 = Slice1(cos, 2L);
        var c3 = Slice1(cos, 3L);

        // SCALE-INVARIANCE: cos([3,4],[3,4]) == cos([6,8],[3,4]) (positive scale).
        var s1 = Tensor(new long[] { 1L, 2L }, 3f, 4f);
        var s2 = Tensor(new long[] { 1L, 2L }, 3f, 4f);
        var s1Scaled = Tensor(new long[] { 1L, 2L }, 6f, 8f);   // 2 * [3,4]
        var cosPlain = CosineEmbeddingLoss.CosineSimilarity(s1, s2, eps).Reduce(ReduceKind.Sum, keepDims: false).Scalar();    // 1
        var cosScaled = CosineEmbeddingLoss.CosineSimilarity(s1Scaled, s2, eps).Reduce(ReduceKind.Sum, keepDims: false).Scalar(); // 1
        var scaleInvariant = (cosPlain - cosScaled).Abs();   // == 0

        // EPS guard: a zero row does not NaN — cos = 0, finite.
        var z1 = Tensor(new long[] { 1L, 2L }, 0f, 0f);
        var z2 = Tensor(new long[] { 1L, 2L }, 1f, 0f);
        var cosZero = CosineEmbeddingLoss.CosineSimilarity(z1, z2, eps).Reduce(ReduceKind.Sum, keepDims: false).Scalar(); // 0, finite

        var ok = Within(refMismatch, 1e-4f)                 // helper == raw-op cosine
               + Within((c0 - Scalar(1f)).Abs(), 1e-4f)
               + Within(c1.Abs(), 1e-4f) + AtLeastZero(c1)
               + Within((c2 - Scalar(-1f)).Abs(), 1e-4f)
               + Within((c3 - Scalar(0.70710678f)).Abs(), 1e-4f)
               + Within((cosPlain - Scalar(1f)).Abs(), 1e-4f)
               + Within((cosScaled - Scalar(1f)).Abs(), 1e-4f)
               + Within(scaleInvariant, 1e-5f)              // scale-invariance
               + Within(cosZero.Abs(), 1e-4f) + AtLeastZero(cosZero);   // eps guard: finite 0, not NaN

        var touch = (t * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar().Abs();
        return ok + Within(touch, 1e-6f) > Scalar(10L);   // 10 ok-bits + touch (11 total); > (11-1)
    }

    /// <summary>Raw-op cosine cos(a,b)=(Σ_feat a·b)/(‖a‖·‖b‖) over the last axis [N] (no eps).</summary>
    private static Tensor<float32> Cos(Tensor<float32> a, Tensor<float32> b)
    {
        var dot = (a * b).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false);
        var na = (a * a).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false).Sqrt();
        var nb = (b * b).Reduce(ReduceKind.Sum, [Scalar(-1L)], keepDims: false).Sqrt();
        return dot / (na * nb);
    }

    private static Scalar<float32> Slice1(Tensor<float32> v, long i)
        => ((Tensor<float32>)OnnxOp.Slice(v, Vector(i), Vector(i + 1L))).Reduce(ReduceKind.Sum, keepDims: false).Scalar();

    private static Scalar<int64> Within(Scalar<float32> dist, float bound)
        => (dist <= Scalar(bound)).IfElse(Scalar(1L), Scalar(0L));

    private static Scalar<int64> AtLeastZero(Scalar<float32> v)
        => (v >= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));
}

/// <summary>
/// Rig-trainability model for CosineEmbeddingLoss (cosine-embedding-loss design §5
/// "loss-is-the-model-tail" recipe, Recipe A fallback). The model takes a single
/// <c>[2N, D]</c> input batch, splits it into x1 / x2 blocks of <c>N</c> rows each,
/// runs a SHARED trainable Linear embedding on each block, and RETURNS THE SCALAR
/// cosine loss as the model output. Paired with the NNIdentityScalarLoss adapter in
/// the rig's 2-input loss slot, this trains the embedding end-to-end — exercising
/// autodiff through Reduce(Sum/L2)/Clip/Where/Relu/division. With N=2, D=2 the input
/// is [4, 2]: rows 0-1 are x1, rows 2-3 are x2. <c>y=−1</c> for both samples and
/// <c>margin=−1</c> keeps the hinge ACTIVE regardless of init (cos ≥ −1 > margin),
/// so the loss is nonzero and the gradient flows.
/// </summary>
[Module]
public partial class NNCosineEmbeddingRigModel
{
    public static Scalar<float32> Inline(Tensor<float32> pair)   // [2N, D] = [4, 2]
    {
        var embed = Linear.Model(Scalar(2L), Scalar(true));     // shared trainable embedding [D]→[2]
        var e1 = embed.Call((Tensor<float32>)OnnxOp.Slice(pair, Vector(0L), Vector(2L)));   // x1 [2,2]
        var e2 = embed.Call((Tensor<float32>)OnnxOp.Slice(pair, Vector(2L), Vector(4L)));   // x2 [2,2]
        // y=−1 (dissimilar arm) with margin=−1 keeps the hinge relu(cos−margin)
        // active for ANY cos ∈ [−1,1] (cos − (−1) = cos + 1 ≥ 0, strictly > 0 unless
        // cos=−1), so the loss is a real, nonzero objective and the gradient flows.
        var y = Tensor(new long[] { 2L }, -1f, -1f);
        return CosineEmbeddingLoss.Call(Scalar(-1f), Scalar(1e-8f), e1, e2, y);
    }
}
