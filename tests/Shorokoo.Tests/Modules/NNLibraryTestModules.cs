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

/// <summary>Linear forward output on RangeTensor([2,3],0.5,-1) at MasterSeed=0 must match the
/// frozen reference. The old check re-ran the same flatten+MatMul by hand (a tautology); the
/// reference is now an external PyTorch value (tests/pytorch-reference/linear.py).</summary>
[Module]
public partial class NNLinearMatchesManualMatMul
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = Linear.Model(Scalar(4L), Scalar(true)).Call(x);   // [2,4] = 8

        // REFERENCE: PyTorch — F.linear(x, W, b) on the seeded weights (tests/pytorch-reference/linear.py).
        var reference = Vector(-0.74641520f, 0.44564119f, -0.35047954f, -0.89765519f, -0.16976941f, -1.35970581f, -1.95182049f, 0.29875028f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>§7.1 Bilinear(useBias:true) forward output on x1 [2,3], x2 [2,4] at MasterSeed=0 must
/// match the frozen reference. The old check re-ran the bilinear form by hand (a tautology); the
/// reference is now the layer's own frozen forward output.</summary>
[Module]
public partial class NNBilinearMatchesManualForm
{
    public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2)   // x1 [N,in1], x2 [N,in2]
    {
        var y = Bilinear.Model(Scalar(3L), Scalar(4L), Scalar(2L), Scalar(true)).Call(x1, x2);   // [2,2] = 4

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.33768559f, -0.53191787f, -0.9681939f, -0.67190534f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7.2 useBias gating: the Bilinear(useBias:true) and Bilinear(useBias:false) forward
/// outputs on x1 [2,3], x2 [2,4] at MasterSeed=0 must each match their frozen reference. The old
/// check re-ran both bilinear forms by hand; the references are now the layer's own frozen outputs
/// (distinct call-sites get distinct seeded weights under per-parameter init).</summary>
[Module]
public partial class NNBilinearUseBiasFalseAndDiff
{
    public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2)   // x1 [N,in1], x2 [N,in2]
    {
        var yTrue = Bilinear.Model(Scalar(3L), Scalar(4L), Scalar(2L), Scalar(true)).Call(x1, x2);    // [2,2]
        var yFalse = Bilinear.Model(Scalar(3L), Scalar(4L), Scalar(2L), Scalar(false)).Call(x1, x2);  // [2,2]

        // REFERENCE: golden — Shorokoo's own forward outputs (useBias true, then false).
        var refTrue = Vector(-0.33768559f, -0.53191787f, -0.9681939f, -0.67190534f);
        var refFalse = Vector(-0.24297486f, 0.121671826f, -0.64173853f, 0.6667637f);

        var dTrue = (yTrue.Reshape([Scalar(-1L)]) - refTrue).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var dFalse = (yFalse.Reshape([Scalar(-1L)]) - refFalse).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return (dTrue < Scalar(1e-3f)) & (dFalse < Scalar(1e-3f));
    }
}

/// <summary>§7.3 Batch broadcasting: with x1 [2,2,3], x2 [2,2,4] the output shape is [2,2,2]
/// (asserted via ShapeTensor) AND its forward output at MasterSeed=0 must match the frozen
/// reference. The old check re-ran the per-row bilinear form by hand; the reference is now the
/// layer's own frozen forward output.</summary>
[Module]
public partial class NNBilinearBatchBroadcasts
{
    public static Scalar<bit> Inline(Tensor<float32> x1, Tensor<float32> x2)   // x1 [B,T,in1], x2 [B,T,in2]
    {
        var y = Bilinear.Model(Scalar(3L), Scalar(4L), Scalar(2L), Scalar(true)).Call(x1, x2);   // [2,2,2] = 8

        // Shape assertion: y.shape == [2, 2, 2].
        var shape = y.ShapeTensor();
        var shapeOk = (shape[0] == Scalar(2L)) & (shape[1] == Scalar(2L)) & (shape[2] == Scalar(2L));

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.33768559f, -0.53191787f, -0.9681939f, -0.67190534f, -3.199333f, -5.0979924f, -7.031103f, -13.810181f);
        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return shapeOk & (diff < Scalar(1e-3f));
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

/// <summary>Conv2d forward output on RangeTensor([1,2,5,5],0.1,-2) at MasterSeed=0 must match the
/// frozen reference. The old check re-ran Conv against a hand-built static NN.Conv (a tautology);
/// the reference is now the layer's own frozen forward output.</summary>
[Module]
public partial class NNConv2dMatchesStaticConv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = Conv2d.Model(Scalar(3L), Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(x);   // [1,3,3,3] = 27

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.5830505f, -0.2175299f, -1.1851754f, -1.8541672f, -2.8027544f, -2.7721534f, -2.8040576f, -2.450688f, -1.224263f, 0.52548397f, -0.6617862f, 0.3270598f, 2.56868f, 2.4922302f, 3.0805902f, 1.9886885f, 3.389592f, 3.977339f, 1.4277205f, 2.192732f, 1.6577607f, -1.1305982f, -0.34348002f, 0.6678303f, -2.3624249f, -1.2884641f, -1.0658423f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>Conv1d forward output on RangeTensor([1,2,7],0.25,-1.5) at MasterSeed=0 must match the
/// frozen reference. The old check re-ran Conv against a hand-built static NN.Conv (a tautology);
/// the reference is now the layer's own frozen forward output.</summary>
[Module]
public partial class NNConv1dMatchesStaticConv
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = Conv1d.Model(Scalar(3L), Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(x);   // [1,3,4] = 12

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.41373608f, -1.1612636f, -1.4508712f, -0.57331306f, 1.2845447f, 0.6888682f, 0.59342396f, 0.22833997f, 1.9253628f, 0.9222772f, -0.60086983f, -1.8340389f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>ConvTranspose2d forward output on RangeTensor([1,2,3,3],0.3,-2) at MasterSeed=0 must
/// match the frozen reference. The old check re-ran ConvTranspose against a hand-built static op (a
/// tautology); the reference is now the layer's own frozen forward output. Output [1,3,4,4]=48 is
/// collapsed to 19 via SelfCheck.Collapse.</summary>
[Module]
public partial class NNConvTranspose2dMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = ConvTranspose2d.Model(Scalar(3L), Scalar(2L), Scalar(true)).Call(x);   // [1,3,4,4] = 48

        // REFERENCE: golden — Shorokoo's own forward output, collapsed to 19 (self-generated).
        var reference = Vector(-0.019935742f, 0.28736883f, -0.25346282f, -1.7633023f, -0.2896873f, 0.5244714f, -0.41189831f, -0.9679297f, -0.27114862f, 0.92171484f, 0.41921818f, -0.305805f, -0.8045916f, 0.8351264f, 0.7873403f, 0.10005285f, -0.06093122f, -0.6823753f, -0.4446751f);

        var diff = (SelfCheck.Collapse(y, 48) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

// ---------------------------------------------------------------------------
// Generalized Convolution helper coverage (Convolution.Conv / ConvTranspose,
// src/Shorokoo.Modules/Layers/Convolution.cs) — design §7 groups 1–9. Each
// self-checking [Module] returns a Scalar<bit> that AutoTest.AdvancedTestGraph
// requires to be true: it runs the configured geometry and compares the (collapsed)
// forward output against an inlined frozen golden reference (self-generated at the
// fixed master-seed-0 per-parameter init; see SelfCheck.Collapse). The former
// hand-built NN.Conv references relied on same-shape inits materializing
// identically and were retired with per-parameter init. Conv has no QEE values,
// so value correctness comes from the ORT backend inside AdvancedTestGraph.
// ---------------------------------------------------------------------------

/// <summary>§7-1 Non-square kernel: Convolution.Conv(kernelSize:[3,5], pads:[1,2,1,2]) — frozen
/// forward-value golden (self-generated).</summary>
[Module]
public partial class ConvNonSquareKernelMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 3L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 5L], padding: [1L, 2L]);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-1.9221866f, -2.97814489f, 0.478535731f, 5.24790588f, -0.688873533f, -2.46253498f, -0.319425726f, -0.979006569f, 4.37657484f, 1.75232301f, -0.261024953f, -1.99853805f, 3.43129576f, 5.64818053f, -1.5080043f, -2.35810049f, -0.79259287f, 7.69494641f, 4.45884863f);
        var diff = (SelfCheck.Collapse(y, 189) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2 Per-axis stride &amp; dilation: stride:[1,2], dilation:[2,1] with explicit pads —
/// frozen forward-value golden (self-generated).</summary>
[Module]
public partial class ConvPerAxisStrideDilationMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L],
            stride: [1L, 2L], padding: [1L, 1L], dilation: [2L, 1L]);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.226712452f, -0.0406049096f, -0.142521508f, 0.183028158f, 0.188516234f, 0.228655454f, -0.0463592699f, -0.126701291f, 0.0221168635f, 0.492151048f, 0.213052561f, -0.145765283f, -1.27074795f, 0.857545932f, 0.651703689f, -0.0363895128f, -0.590573005f, 0.494001426f, -0.181385751f);
        var diff = (SelfCheck.Collapse(y, 40) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-3 Asymmetric pad: padding:[1,2,0,1] (ONNX begin..end order, applied verbatim) —
/// frozen forward-value golden (self-generated).</summary>
[Module]
public partial class ConvAsymmetricPadMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 2L, 0L, 1L]);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.325015575f, -0.160096992f, -0.514238701f, -0.32993945f, 0.0845054114f, -0.0510165875f, -0.386297352f, -0.0252350369f, 0.217532583f, 0.158120648f, 0.023319188f, -0.00376949423f, -0.374418345f, -0.131863433f, 0.149000732f, 0.013944535f, -0.247886784f, 0.00666655256f, 0.475331819f);
        var diff = (SelfCheck.Collapse(y, 48) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 auto_pad SAME_UPPER and VALID: both variants run and both outputs fold into one
/// frozen forward-value golden (self-generated). Forward value only — SAME has no Conv backward.</summary>
[Module]
public partial class ConvAutoPadMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var ySame = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], autoPad: AutoPad.SameUpper);
        var yValid = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], autoPad: AutoPad.Valid);
        var flat = ySame.Reshape([Scalar(-1L)]).Concat(0L, yValid.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.348030643f, 1.29787397f, 2.00380692f, -1.50932515f, 0.530210213f, -1.65958951f, 0.478100029f, 0.122887341f, 1.5420118f, 1.02560474f, -0.418286645f, -1.12823754f, 1.7738596f, 0.52667398f, -0.478433563f, -1.20338849f, 1.11250294f, 0.805302486f, 0.562707136f);
        var diff = (SelfCheck.Collapse(flat, 104) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 Groups / depthwise: groups:inC (depthwise, inC=4 → outC=4) AND a mid groups:2 —
/// both variants run and fold into one frozen forward-value golden (self-generated).</summary>
[Module]
public partial class ConvGroupsMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, 4, H, W]
    {
        // Depthwise: groups == inC (4), outChannels == inC.
        var outDw = 4L;
        var yDw = Convolution.Conv(x, outDw, kernelSize: [3L, 3L], padding: [1L, 1L], groups: 4L);
        // Mid groups: groups:2 with inC=4 (weight second axis inC/groups = 2).
        var y2 = Convolution.Conv(x, 4L, kernelSize: [3L, 3L], padding: [1L, 1L], groups: 2L);
        var flat = yDw.Reshape([Scalar(-1L)]).Concat(0L, y2.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-2.4769039f, -0.005357327f, 0.256900865f, 2.47853286f, -0.461652536f, -1.78766537f, -0.325482123f, 1.93791631f, 0.718129099f, -0.298873421f, -0.930203249f, 0.214313571f, 1.33020983f, -0.7051071f, 0.459618472f, -2.66033391f, 0.417266306f, 1.69517409f, 0.667809444f);
        var diff = (SelfCheck.Collapse(flat, 200) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 padding_mode Reflect, Replicate and Circular: all three modes run and fold into one
/// frozen forward-value golden (self-generated). Forward only — reflect/edge/wrap Pad is
/// non-differentiable and has no QEE values, so QEE / CS-roundtrip are disabled for this check in the
/// driving [Fact].</summary>
[Module]
public partial class ConvPaddingModeMatchesHandPad
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var y = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L],
            paddingMode: PaddingMode.Reflect);
        var yRep = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L],
            paddingMode: PaddingMode.Replicate);
        var yCir = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L],
            paddingMode: PaddingMode.Circular);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, yRep.Reshape([Scalar(-1L)])).Concat(0L, yCir.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-2.09115282f, 3.38504581f, 1.09607383f, 0.0994339679f, -2.58002548f, 2.213805f, 0.698460997f, -0.558512335f, 0.112954178f, -1.232869f, -1.28926419f, 1.86452973f, 0.34222955f, -1.63718008f, -2.44950396f, 2.65511615f, 1.31376371f, -0.127762417f, -0.848646684f);
        var diff = (SelfCheck.Collapse(flat, 150) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 Causal (1D): Convolution.Conv1d(paddingMode:Causal) — frozen forward-value golden
/// (self-generated): the configured layer's output must match the inlined reference. Forward only
/// (the Pad here is constant-mode, so it is differentiable, but kept inference-grade for symmetry
/// with the other padding-mode check).</summary>
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
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.37800658f, 0.3419249f, 1.5251406f, 1.1072305f, -0.20009777f, -0.18928039f, -0.1784629f, 0.41130042f, 0.59661496f, 2.2824254f, 2.3296556f, 3.8941483f, 3.7998872f, 3.7056253f);
        var diff = (SelfCheck.Collapse(y, 14) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 ConvTranspose output_padding: ConvTranspose(kernelSize:[2,2], stride:[2,2],
/// outputPadding:[1,1]) — frozen forward-value golden (self-generated).</summary>
[Module]
public partial class ConvTransposeOutputPaddingMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 3L;
        var y = Convolution.ConvTranspose(x, outChannels, kernelSize: [2L, 2L],
            stride: [2L, 2L], outputPadding: [1L, 1L]);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.343243602f, -0.129276719f, 0.0576216165f, -0.0386626929f, 0.281775445f, -0.998314627f, -0.6926795f, 0.262189563f, -0.619852588f, -0.232157749f, -0.27606478f, 0.668260588f, -0.360607299f, -0.772155445f, -0.100191667f, 0.832471463f, -0.471380406f, -0.159984422f, 0.00666867711f);
        var diff = (SelfCheck.Collapse(y, 147) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 ConvTranspose output_shape: ConvTranspose(stride:[2,2], outputShape:[…]) — frozen
/// forward-value golden (self-generated).</summary>
[Module]
public partial class ConvTransposeOutputShapeMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, C, 3, 3] → stride 2 base out 6, target 7
    {
        var outChannels = 2L;
        long[] outShape = { 7L, 7L };
        var y = Convolution.ConvTranspose(x, outChannels, kernelSize: [2L, 2L],
            stride: [2L, 2L], outputShape: outShape);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.207715835f, 0.424565871f, 0.466434119f, 0.61853318f, 0.121246895f, -0.320214402f, 0.293463427f, 0.677486508f, -0.527935168f, -0.134921777f, -0.460861299f, 0.389261009f, 0.905120224f, -0.195043558f, 0.217689877f, -0.0448414105f, -0.227881697f, -0.0719468552f, -0.208333778f);
        var diff = (SelfCheck.Collapse(y, 98) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 ConvTranspose1d rank alias smoke: rank 3, one spatial dim — frozen forward-value
/// golden (self-generated).</summary>
[Module]
public partial class ConvTranspose1dMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, L]
    {
        var outChannels = 2L;
        var y = Convolution.ConvTranspose1d(x, outChannels, kernelSize: [2L], stride: [2L]);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-1.0087571f, 1.6243131f, -0.9656336f, 1.372434f, -0.9225099f, 1.1205548f, -0.8793863f, 0.8686756f, 1.0151453f, -1.1933439f, 0.65247333f, -1.1855811f, 0.28980142f, -1.1778183f, -0.07287051f, -1.1700555f);
        var diff = (SelfCheck.Collapse(y, 16) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 ConvTranspose3d rank alias smoke: rank 5, three spatial dims — frozen forward-value
/// golden (self-generated).</summary>
[Module]
public partial class ConvTranspose3dMatchesStatic
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, C, D, H, W]
    {
        var outChannels = 2L;
        var y = Convolution.ConvTranspose3d(x, outChannels, kernelSize: [2L, 2L, 2L], stride: [2L, 2L, 2L]);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0404135142f, 0.195917169f, 0.117576553f, 0.853081528f, -0.237471911f, -0.0085286995f, 0.720186332f, -0.325168611f, -0.420569111f, -0.628446415f, 0.575789116f, 0.0487579632f, 0.84777831f, 0.780389902f, -0.458723911f, -0.522503935f, -0.363272664f, -0.539074359f, 0.0205327569f);
        var diff = (SelfCheck.Collapse(y, 128) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-8 Alias coverage: the generic Conv plus the Conv2d and ConvTranspose2d aliases all run
/// and fold into one frozen forward-value golden (self-generated). The scalar→per-axis broadcast
/// equivalence is pinned separately in TestConvScalarBroadcastEquivalence (the scalar overload needs a
/// build-time-known rank, which a symbolic [Module] input lacks).</summary>
[Module]
public partial class ConvAliasAndScalarEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var perAxis = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L]);
        var y2d = Convolution.Conv2d(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L]);
        var yT2d = Convolution.ConvTranspose2d(x, outChannels, kernelSize: [2L, 2L]);
        var flat = perAxis.Reshape([Scalar(-1L)]).Concat(0L, y2d.Reshape([Scalar(-1L)])).Concat(0L, yT2d.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-2.11651284f, 1.39266936f, 1.19503994f, -0.440624448f, 0.453784753f, 1.818783f, 1.47904183f, 0.575740795f, -0.988287516f, 0.380869842f, -0.0556775921f, 0.678857506f, -0.794682859f, -2.15588213f, -1.25938367f, 0.952404815f, 0.714105923f, -0.399691187f, 0.189130301f);
        var diff = (SelfCheck.Collapse(flat, 172) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-8 Scalar-overload broadcast: the scalar Conv(c, outC, 3, padding:1) — frozen
/// forward-value golden (self-generated): the configured layer's output must match the inlined
/// reference. The scalar overload reads c.Rank()-2 at build time, which is statically known only
/// for a concretely-shaped tensor (a symbolic graph input has no build-time rank), so the conv
/// input is a literal constant [1,1,3,3] tensor built in-module. The runtime input <paramref
/// name="x"/> only gates the result so AutoTest has a graph input to drive.</summary>
[Module]
public partial class ConvScalarBroadcastMatchesPerAxis
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        // Literal constant input → build-time-known rank 4, so the scalar overload can read Rank()-2.
        var c = Tensor(new long[] { 1L, 1L, 3L, 3L }, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f);
        var outChannels = 2L;
        var scalar = Convolution.Conv(c, outChannels, kernelSize: 3L, padding: 1L);
        // Fold a trivial dependence on x so AutoTest has a runtime input to feed.
        var xTouch = (x * Scalar(0f)).Reduce(ReduceKind.Sum, keepDims: true);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-3.934529f, -7.5396147f, -8.386295f, -8.951771f, -12.848224f, -13.157093f, -6.8513536f, -0.2597307f, 0.3744189f, -5.568619f, -7.2014875f, -1.737776f, -8.615043f, -8.6678295f, -0.9370452f, -0.6191263f, 1.814758f, 3.7070842f);
        var diff = (SelfCheck.Collapse(scalar + xTouch, 18) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-9 bias on/off: a bias:false conv — frozen forward-value golden (self-generated): the
/// configured layer's output must match the inlined reference. Both compared against the same
/// KaimingUniform-weight, zero-bias NN.Conv.</summary>
[Module]
public partial class ConvBiasOnOffMatchesZeroBias
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var outChannels = 2L;
        var yNoBias = Convolution.Conv(x, outChannels, kernelSize: [3L, 3L], padding: [1L, 1L], bias: false);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.106164664f, -0.154563161f, 0.207394125f, 0.610819924f, -0.434361581f, -0.369176532f, -0.670023677f, -0.193005095f, 0.155514932f, -0.127404392f, -0.157208162f, -0.435303484f, -0.211678814f, 0.400296261f, -0.0333141729f, -0.00299395343f, -0.264332402f, -0.386756078f, 0.656837735f);
        var diff = (SelfCheck.Collapse(yNoBias, 50) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>§7-2(a) InstanceNorm(affine:false) forward output on CurvedTensor([2,3,4,4],0.4,-3,0.05) at
/// MasterSeed=0 must match the frozen reference. The old check re-ran (x−mean)/sqrt(var+eps) by hand
/// (a tautology); the reference is now the layer's own frozen output. The input is a DISTINCT-valued
/// (quadratic) ramp, not a linear one: a linear ramp standardizes every (sample,channel) slice to the
/// identical pattern, so the frozen reference would be blind to an internal N/C transpose; the quadratic
/// gives each slice a distinct pattern so a transpose moves the output. [2,3,4,4]=96 collapsed to 19.</summary>
[Module]
public partial class NNInstanceNormAffineFalseMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = InstanceNorm.Call(Scalar(false), Scalar(1e-5f), x);   // [2,3,4,4] = 96

        // REFERENCE: golden — Shorokoo's own forward output, collapsed to 19 (self-generated).
        var reference = Vector(1.2490876f, 0.64456594f, -1.2203515f, -1.2086581f, 0.14979805f, 0.0051061511f, 0.31833237f, 1.6107239f, -0.4773435f, -0.52412701f, -0.58257258f, -0.36030608f, 1.7623465f, 0.21758696f, -0.10315616f, 0.074702829f, -1.1945477f, -0.3857736f, 0.68630159f);

        var diff = (SelfCheck.Collapse(y, 96) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2(a) GroupNorm(G=2, affine:false) forward output on CurvedTensor([2,4,3,3],0.7,-10,0.05) at
/// MasterSeed=0 must match the frozen reference. The old check re-ran the per-group
/// (x−mean)/sqrt(var+eps) by hand (a tautology); the reference is now the layer's own frozen output.
/// The input is a DISTINCT-valued (quadratic) ramp, not a linear one: a linear ramp standardizes every
/// (sample,group) region to the identical pattern, so the frozen reference would be blind to an internal
/// N/C transpose; the quadratic gives each region a distinct pattern. [2,4,3,3]=72 collapsed to 19.</summary>
[Module]
public partial class NNGroupNormAffineFalseMatchesManual
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);   // [2,4,3,3] = 72

        // REFERENCE: golden — Shorokoo's own forward output, collapsed to 19 (self-generated).
        var reference = Vector(0.59265316f, -0.40483442f, -0.44441506f, -0.38220644f, 0.70379376f, 0.16393512f, -0.073275648f, -0.24669781f, -0.29130843f, -0.055236947f, 0.26341963f, 0.93995595f, -0.34839317f, -1.2193623f, -0.28585726f, 0.43276089f, 0.16860867f, 2.0212572f, 1.1266559f);

        var diff = (SelfCheck.Collapse(y, 72) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>Embedding forward output on indices [0,1,0] at MasterSeed=0 must match the frozen
/// reference. The old check re-ran a manual Gather over the layer's own table (a tautology); the
/// reference is now the layer's own frozen forward output (rows 0 and 1, with row 0 repeated).</summary>
[Module]
public partial class NNEmbeddingMatchesGather
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var y = Embedding.Model(Scalar(5L), Scalar(4L), Scalar(-1L), Scalar(0f), Scalar(2f)).Call(indices);   // [3,4] = 12

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.14280432f, -1.404502f, -0.43904895f, 0.5944438f, 0.066021085f, 1.1457329f, -1.3943781f, 0.3395884f, 0.14280432f, -1.404502f, -0.43904895f, 0.5944438f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
        // Each distinct call-site is its own model, so reference each against its OWN realized
        // table (param [1]); under per-parameter init RNG two call-sites no longer share a weight.
        var paddedModel = Embedding.Model(numEmbeddings, dim, Scalar(2L), Scalar(0f), Scalar(2f));
        var padded = paddedModel.Call(indices);
        var gatherPadded = paddedModel.GetTrainableParam<float32>([1], rank: 2).Gather(indices, axis: 0); // [5,4]

        // off-sentinel: paddingIdx:-1 must be a no-op == plain Gather (of its own table).
        var offModel = Embedding.Model(numEmbeddings, dim, Scalar(-1L), Scalar(0f), Scalar(2f));
        var offPad = offModel.Call(indices);
        var gatherOff = offModel.GetTrainableParam<float32>([1], rank: 2).Gather(indices, axis: 0);        // [5,4]

        // Build the expected pad mask in the reference: rows where indices == 2 -> 0.
        var isPad = (indices == Scalar(2L)).Unsqueeze(-1);             // [5, 1] bit
        var zeros = gatherPadded * Scalar(0f);
        var expected = isPad.Where(zeros, gatherPadded);               // pad rows -> 0, else gather

        // (a) padded output matches the hand-masked reference exactly (covers
        //     both the all-zero pad rows AND the unchanged non-pad rows).
        var padDiff = (padded - expected).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // (b) explicit check that the pad positions carry zero L2 mass (independent of (a)).
        var padMass = (isPad.Where(padded, zeros)).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        // (c) off-sentinel is the plain Gather (no masking).
        var offDiff = (offPad - gatherOff).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

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

        // Each distinct-cap call-site is its own model and references its OWN realized table
        // (param [1]); under per-parameter init RNG the three call-sites no longer share a weight,
        // so every sub-check compares an output against the gather of that same model's table.
        var capModel = Embedding.Model(numEmbeddings, dim, Scalar(-1L), maxNorm, Scalar(2f));
        var y = capModel.Call(indices);
        var gatherRef = capModel.GetTrainableParam<float32>([1], rank: 2).Gather(indices, axis: 0); // [n, 4]

        var bigModel = Embedding.Model(numEmbeddings, dim, Scalar(-1L), Scalar(1000f), Scalar(2f));
        var bigCap = bigModel.Call(indices);
        var gatherBig = bigModel.GetTrainableParam<float32>([1], rank: 2).Gather(indices, axis: 0);

        var offModel = Embedding.Model(numEmbeddings, dim, Scalar(-1L), Scalar(0f), Scalar(2f));
        var off = offModel.Call(indices);
        var gatherOff = offModel.GetTrainableParam<float32>([1], rank: 2).Gather(indices, axis: 0);

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

        // (c) under-cap is a no-op: huge cap AND off-sentinel both == plain Gather of their own table.
        var bigDiff = (bigCap - gatherBig).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        var offDiff = (off - gatherOff).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

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

/// <summary>§8-7 init choice (static EmbeddingHelpers.Embed): the XavierUniform-selector form AND the
/// default (Normal) form both run and fold into one frozen forward-value golden (self-generated) — a
/// selector that stops being wired changes the output and fails the reference comparison.</summary>
[Module]
public partial class NNEmbeddingInitChoice
{
    public static Scalar<bit> Inline(Tensor<int64> indices)
    {
        var xavier = EmbeddingHelpers.Embed(indices, 5L, 4L, shape => XavierUniform.Init(shape));
        var normal = EmbeddingHelpers.Embed(indices, 5L, 4L);   // default init selector (Normal)
        var flat = xavier.Reshape([Scalar(-1L)]).Concat(0L, normal.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.6150066f, -0.7470366f, -0.19320095f, 0.7852122f, -0.105438106f, 0.3118135f, 0.31353027f, 0.710673f, -0.25749266f, -0.6481527f, 0.4695184f, -0.67243695f, 0.98079455f, -0.020025833f, -1.5019299f, 0.6489639f, -0.9045778f, -0.28363064f, -1.7336785f, 1.7301084f, -0.54763764f, -1.3845996f, -0.31040692f, 0.25047535f);
        var diff = (SelfCheck.Collapse(flat, 24) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>§8-1 (Sum): EmbeddingBag.Bag(BagMode.Sum) on indices [[0,1,2],[1,3,0]] at MasterSeed=0
/// must match the frozen reference. Was a finiteness-only Sanity.Reasonable check; now a frozen
/// forward-value golden (self-generated) that pins the per-feature sum over the bag axis.</summary>
[Module]
public partial class NNEmbeddingBagSumMatchesGatherReduce
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);   // [2,4] = 8

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.9563482f, -0.64395595f, 2.6079757f, -0.6798403f, 0.09591007f, 0.74071383f, 0.68464065f, 0.00798434f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§8-1 (Mean): EmbeddingBag.Bag(BagMode.Mean) on indices [[0,1,2],[1,3,0]] at MasterSeed=0
/// must match the frozen reference (full-L denominator). Was a finiteness-only Sanity.Reasonable
/// check; now a frozen forward-value golden (self-generated).</summary>
[Module]
public partial class NNEmbeddingBagMeanMatchesGatherReduce
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Mean);   // [2,4] = 8

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.31878272f, -0.21465199f, 0.8693252f, -0.22661345f, 0.031970024f, 0.24690461f, 0.22821355f, 0.0026614468f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§8-1 (Max): EmbeddingBag.Bag(BagMode.Max) on indices [[0,1,2],[1,3,0]] at MasterSeed=0
/// must match the frozen reference (per-feature max over the bag). Was a finiteness-only
/// Sanity.Reasonable check; now a frozen forward-value golden (self-generated).</summary>
[Module]
public partial class NNEmbeddingBagMaxMatchesGatherReduce
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var y = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Max);   // [2,4] = 8

        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(1.9730682f, 0.5402425f, 1.4092264f, 0.045669284f, 1.9730682f, 0.5402425f, 0.9734798f, 0.0800632f);

        var diff = (y.Reshape([Scalar(-1L)]) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>§8-3 paddingIdx zeroes pad rows for Sum (the EXACT case): Bag(..., Sum, paddingIdx:2) on a
/// bag containing the pad id, plus the UNMASKED Sum of the same bag — both fold into one frozen
/// forward-value golden (self-generated), so ignoring paddingIdx produces the unmasked numbers in the
/// masked segment and fails. Per the design, only the Sum-exact case is pinned (Mean/Max + paddingIdx
/// have documented caveats).</summary>
[Module]
public partial class NNEmbeddingBagPaddingIdxSumExact
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L], contains the pad id 2
    {
        var padded = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum, paddingIdx: 2L);   // [B, D]
        var unmasked = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);   // no paddingIdx — pad rows count
        var flat = padded.Reshape([Scalar(-1L)]).Concat(0L, unmasked.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(1.3440652f, 0.3525714f, 1.1987493f, -0.072078854f, 0.7249131f, 0.92838496f, 0.4593711f, -0.03768494f, 0.68589f, -1.3499765f, -3.0228398f, -1.3550098f, -1.0461789f, 0.24091457f, -3.0575676f, 3.6953902f);
        var diff = (SelfCheck.Collapse(flat, 16) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§8-4 init choice: the XavierUniform-selector Bag AND the default (Normal) Bag both run and
/// fold into one frozen forward-value golden (self-generated) — the embeddingInit selector staying wired
/// is what keeps the two segments distinct.</summary>
[Module]
public partial class NNEmbeddingBagInitChoice
{
    public static Scalar<bit> Inline(Tensor<int64> indices)   // indices [B, L]
    {
        var xavier = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum, shape => XavierUniform.Init(shape));
        var normal = EmbeddingBag.Bag(indices, 5L, 4L, BagMode.Sum);   // default init selector (Normal)
        var flat = xavier.Reshape([Scalar(-1L)]).Concat(0L, normal.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.009558208f, -0.35986438f, -0.65359485f, 0.7313738f, 0.23650318f, -0.75003564f, -1.5768887f, -0.01945132f, 0.68589f, -1.3499765f, -3.0228395f, -1.3550097f, 0.46807218f, -0.5217749f, -1.1111203f, -1.7688f);
        var diff = (SelfCheck.Collapse(flat, 16) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
// requires to be true: it computes (y, hN) = Recurrent.RNN(x, …) and compares the
// (collapsed) concatenation of BOTH outputs against an inlined frozen golden
// reference (self-generated at the fixed master-seed-0 per-parameter init), so the
// bias packing, direction/batchFirst plumbing, and the Y [L,D,N,H] -> [L,N,D*H]
// reshape all reach the verdict. The former hand-built OnnxOp.Rnn references relied
// on same-shape inits materializing identically and were retired with per-parameter
// init; op-level gradient coverage lives in AutoGradOpsTests. RNN has no QEE step
// values, so value correctness comes from the ORT backend inside AdvancedTestGraph
// (note [2] of the design). The relu / bidirectional BPTT-throws guards live as
// [Fact]s in NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------


/// <summary>§7-2 Forward tanh baseline: Recurrent.RNN(x, H) — frozen forward-value golden
/// (self-generated) over BOTH y and hN, covering the bias packing and the [L,D,N,H]→[L,N,D*H]
/// reshape.</summary>
[Module]
public partial class RnnMatchesCoreOpForwardTanh
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.17373544f, -0.27466792f, 0.8597976f, -0.23974895f, -0.20257777f, 0.73698604f, -0.2082305f, 0.20641482f, 0.2690121f, -0.25773376f, 0.24033892f, -0.06970912f, -0.22683054f, 0.14699924f, -0.200679f, -0.35103303f, 0.098100066f, -0.47503126f, -0.48486203f, 0.11939013f, -0.6934142f, -0.59459865f, 0.09852862f, -0.84707177f, -0.48486203f, 0.11939013f, -0.6934142f, -0.59459865f, 0.09852862f, -0.84707177f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-1/§7-2 (batchFirst) Matches the core op with batchFirst input:
/// Recurrent.RNN(batchFirst:true) on [N, L, in] — frozen forward-value golden (self-generated): the
/// configured layer's output must match the inlined reference. Pins the in-graph transpose around
/// the layout=0 op.</summary>
[Module]
public partial class RnnMatchesCoreOpBatchFirst
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, batchFirst: true);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.17373544f, -0.27466792f, 0.8597976f, -0.14137572f, 0.13215971f, 0.554265f, -0.06615907f, 0.098089814f, 0.4340241f, -0.19550431f, 0.12150943f, 0.16687167f, -0.42311764f, 0.024035573f, -0.10194969f, -0.44383717f, 0.09320021f, -0.5859147f, -0.56241554f, -0.013551831f, -0.7325691f, -0.6587949f, 0.012843847f, -0.8763608f, -0.19550431f, 0.12150943f, 0.16687167f, -0.6587949f, 0.012843847f, -0.8763608f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-1 Single-step recurrence anchor: L=1, h_0=0, so the layer computes y[0] = tanh(W·x_0 + bias)
/// (R is unused at step 0) and hN == y[0]. Frozen forward-value golden (self-generated) on y, PLUS the
/// state contract asserted relationally on the layer's own outputs (hN equals y, which at L=1 is y[0],
/// and hN's leading dim is D·numLayers == 1) — output-vs-output, valid under any initialization.</summary>
[Module]
public partial class RnnSingleStepAnchorTanh
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 2L);   // y [1, N, H], hN [1, N, H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.42514038f, -0.2159316f, -0.5531751f, -0.032021046f);
        var goldenDiff = (SelfCheck.Collapse(y, 4) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        // State contract: at L=1, hN == y[0] == y (both [1, N, H]); leading dim == 1.
        var stateDiff = (hN - y).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var hLeadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();
        return (goldenDiff + stateDiff + hLeadingOk) < Scalar(1e-3f);
    }
}

/// <summary>§7-3 relu nonlinearity (forward only): Recurrent.RNN(Relu) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined reference.
/// Forward-value check only (relu RNN BPTT throws AD003 — pinned separately in
/// NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class RnnReluMatchesCoreOpForward
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, nonlinearity: RnnNonlinearity.Relu);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0f, 0f, 1.2925681f, 0f, 0f, 0.94384974f, 0.04135573f, 0.36952493f, 0.3891021f, 0f, 0.3115339f, 0.095967904f, 0f, 0.17937636f, 0f, 0f, 0.14460433f, 0f, 0f, 0.18108352f, 0f, 0f, 0.25678098f, 0f, 0f, 0.18108352f, 0f, 0f, 0.25678098f, 0f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 bias on/off: bias:false — frozen forward-value golden (self-generated): the
/// configured layer's output must match the inlined reference. Both compared against the matching
/// seeded reference (same W/R; B present iff bias).</summary>
[Module]
public partial class RnnBiasOnOffMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yNoB, hNoB) = Recurrent.RNN(x, hiddenSize: 3L, bias: false);
        var flat = yNoB.Reshape([Scalar(-1L)]).Concat(0L, hNoB.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.14871562f, -0.2725845f, 0.7981243f, 0.08065927f, -0.20041716f, 0.6319798f, 0.054276228f, 0.16232193f, 0.26635277f, -0.009602129f, 0.18091142f, -0.06683439f, 0.032679558f, 0.12791717f, -0.2580979f, -0.107979f, 0.08164787f, -0.5305649f, -0.2599578f, 0.08001578f, -0.71661353f, -0.39458805f, 0.061018467f, -0.8619387f, -0.2599578f, 0.08001578f, -0.71661353f, -0.39458805f, 0.061018467f, -0.8619387f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 numLayers stacking: numLayers:2 (forward, tanh) — frozen forward-value golden
/// (self-generated): the configured layer's output must match the inlined reference. Asserts y and
/// the stacked [2,N,H] hN both match. Layer-1's input size is D·H = H (D=1), which the helper
/// passes as Scalar(d·H); the reference reads it from the reshaped layer-0 Y's last axis (same
/// value), so the seeded inits coincide.</summary>
[Module]
public partial class RnnNumLayersStackMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, numLayers: 2);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0366460964f, -0.00511288639f, -0.0214970655f, 0.140893645f, -0.195169352f, 0.0747870151f, 0.0171349019f, -0.208426835f, 0.267483859f, -0.036309023f, -0.0205600123f, 0.0206003018f, 0.157957204f, -0.219931483f, -0.0241384705f, 0.0286805562f, 0.145337769f, 0.0613698166f, -0.0130298134f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 direction Reverse (trainable): Recurrent.RNN(Reverse) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined
/// reference.</summary>
[Module]
public partial class RnnReverseMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L, direction: RnnDirection.Reverse);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.1424712f, -0.13236505f, 0.74037075f, -0.29117948f, -0.16957521f, 0.5282415f, -0.40410942f, -0.19258863f, 0.31853974f, -0.5038934f, -0.21793741f, 0.0035425425f, -0.43096125f, -0.18611938f, -0.24419701f, -0.48060155f, -0.16422284f, -0.52210253f, -0.5295123f, 0.17518723f, -0.6638923f, -0.5773369f, 0.24820554f, -0.8172432f, -0.1424712f, -0.13236505f, 0.74037075f, -0.29117948f, -0.16957521f, 0.5282415f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 direction Bidirectional (forward inference only): Recurrent.RNN(Bidirectional) —
/// frozen forward-value golden (self-generated): the configured layer's output must match the
/// inlined reference. Forward-value only (bidirectional BPTT throws AD003 — pinned in
/// NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class RnnBidirectionalMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        long hVal = 3L;
        var (y, hN) = Recurrent.RNN(x, hiddenSize: hVal, direction: RnnDirection.Bidirectional);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.0427910927f, 0.192823459f, 0.0277463273f, 0.0539857198f, 0.147218542f, -0.0639532388f, -0.197392916f, 0.31880921f, 0.235884195f, 0.114734327f, -0.0700399968f, -0.0252688431f, 0.28742491f, 0.207199052f, -0.167942984f, -0.207546721f, -0.341014151f, 0.0782811821f, 0.00314437023f);
        var diff = (SelfCheck.Collapse(flat, 60) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 batchFirst equivalence: RNN(batchFirst:true, [N,L,in]) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined reference. Pins
/// the in-graph transpose + layout=0-always choice independently of the op reference. hN is
/// batch-second in both (PyTorch keeps hN [D·numLayers, N, H] regardless of batch_first), so it
/// matches directly.</summary>
[Module]
public partial class RnnBatchFirstEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (yBF, hBF) = Recurrent.RNN(x, hiddenSize: 3L, batchFirst: true);     // y [N, L, D*H]
        var flat = yBF.Reshape([Scalar(-1L)]).Concat(0L, hBF.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.17373544f, -0.27466792f, 0.8597976f, -0.14137572f, 0.13215971f, 0.554265f, -0.06615907f, 0.098089814f, 0.4340241f, -0.19550431f, 0.12150943f, 0.16687167f, -0.42311764f, 0.024035573f, -0.10194969f, -0.44383717f, 0.09320021f, -0.5859147f, -0.56241554f, -0.013551831f, -0.7325691f, -0.6587949f, 0.012843847f, -0.8763608f, -0.19550431f, 0.12150943f, 0.16687167f, -0.6587949f, 0.012843847f, -0.8763608f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-8 state contract: for a forward single-layer RNN, hN == y[-1] (the last step's
/// hidden state), asserted relationally on the layer's own outputs — output-vs-output, valid under
/// any initialization — together with hN's leading dim (D·numLayers == 1) and the frozen
/// forward-value golden (self-generated) on y. Pins the (y, hN) return relationship.</summary>
[Module]
public partial class RnnStateContractForwardSingleLayer
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 3L);   // y [L, N, H], hN [1, N, H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.17373544f, -0.27466792f, 0.8597976f, -0.23974895f, -0.20257777f, 0.73698604f, -0.2082305f, 0.20641482f, 0.2690121f, -0.25773376f, 0.24033892f, -0.06970912f, -0.22683054f, 0.14699924f, -0.200679f, -0.35103303f, 0.098100066f, -0.47503126f, -0.48486203f, 0.11939013f, -0.6934142f, -0.59459865f, 0.09852862f, -0.84707177f);
        var goldenDiff = (SelfCheck.Collapse(y, 24) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        // State contract: hN == y[-1] (last step along axis 0; both [1, N, H]), leading dim == 1.
        var lastStep = y.Slice(Vector(-1L), Vector(System.Int64.MaxValue), Vector(0L));
        var stateDiff = (hN - lastStep).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var hLeadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();
        return (goldenDiff + stateDiff + hLeadingOk) < Scalar(1e-3f);
    }
}

/// <summary>§7-9 forward frozen golden: a forward, tanh, single-layer Recurrent.RNN over a length-3
/// sequence built from the probed scalar — frozen forward-value golden (self-generated). Gradient
/// correctness for the recurrent ops is pinned per input slot in AutoGradOpsTests; layer-level FD grad
/// checks were retired with per-parameter init.</summary>
[Module]
public partial class RnnForwardTanhGolden
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 3);   // [3, 1, 2]
        var (y, hN) = Recurrent.RNN(x, hiddenSize: 2L);   // forward, tanh, single-layer
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.50957435f, 0.27942073f, -0.28417426f, 0.14040172f, -0.39996016f, -0.009731531f, -0.39996016f, -0.009731531f);
        var diff = (SelfCheck.Collapse(flat, 8) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
// Recurrent.RNN set above EXACTLY: each self-checking [Module] computes
// (y, hN, cN) = Recurrent.LSTM(x, …) and compares the (collapsed) concatenation of
// ALL outputs against an inlined frozen golden reference (self-generated at the
// fixed master-seed-0 per-parameter init), so the i,o,f,c gate packing, bias
// packing, and the Y [L,D,N,H] -> [L,N,D*H] reshape all reach the verdict. The
// former hand-built OnnxOp.Lstm references were retired with per-parameter init;
// op-level gradient coverage lives in AutoGradOpsTests. LSTM has no QEE step
// values, so value correctness comes from the ORT backend inside AdvancedTestGraph.
// The bidirectional BPTT-throws guard lives as a [Fact] in
// NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------


/// <summary>§7-1 Forward baseline: Recurrent.LSTM(x, H) — frozen forward-value golden (self-generated)
/// over y, hN AND cN, covering the i,o,f,c gate packing, bias packing, and the [L,D,N,H]→[L,N,D*H]
/// reshape.</summary>
[Module]
public partial class LstmMatchesCoreOpForward
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)])).Concat(0L, cN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0280176006f, 0.0148549157f, -0.0854059747f, 0.040779366f, 0.0534180825f, 0.000506923886f, -0.00231920167f, -0.0252692755f, 0.248211911f, 0.0146093446f, 0.01809622f, -0.0843883543f, -0.0710203857f, 0.0046217465f, 0.0452203682f, 0.0173241417f, 0.0226722005f, 0.0297508294f, 0.124974755f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-1 (batchFirst) Matches the core op with batchFirst input:
/// Recurrent.LSTM(batchFirst:true) on [N, L, in] — frozen forward-value golden (self-generated):
/// the configured layer's output must match the inlined reference. hN/cN stay [D·numLayers, N, H].
/// Pins the in-graph transpose around the layout=0 op.</summary>
[Module]
public partial class LstmMatchesCoreOpBatchFirst
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L, batchFirst: true);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)])).Concat(0L, cN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0352879444f, 0.0136820285f, -0.111469804f, 0.0822292427f, 0.0866030488f, -0.0314609057f, -0.0233364428f, -0.00323860235f, 0.295652268f, 0.0081531109f, 0.0210374884f, -0.0135555522f, -0.231829053f, -0.0727247134f, 0.0796338831f, 0.0297410555f, 0.0140637151f, 0.0200658506f, 0.166782024f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2 Single-step gate anchor: L=1, h_0=c_0=0, so the layer computes i=σ(W_i·x_0+b_i),
/// c̃=tanh(W_c·x_0+b_c), C_1=i⊙c̃, H_1=o⊙tanh(C_1) with o=σ(W_o·x_0+b_o) over the ONNX i,o,f,c gate
/// blocks of the packed [D,4H,in] W (R unused at step 0). H=2. Frozen forward-value golden
/// (self-generated): a wrong gate packing or equation changes the output and fails the reference
/// comparison.</summary>
[Module]
public partial class LstmSingleStepGateAnchor
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 2L);   // y [1,N,H], hN/cN [1,N,H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.11654792f, -0.11149972f, 0.051526453f, -0.1166437f);
        var diff = (SelfCheck.Collapse(y, 4) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-3 bias on/off: bias:false — frozen forward-value golden (self-generated): the
/// configured layer's output must match the inlined reference. Both compared against the matching
/// seeded reference (same W/R; B present iff bias) on y, hN, cN.</summary>
[Module]
public partial class LstmBiasOnOffMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yNoB, hNoB, cNoB) = Recurrent.LSTM(x, hiddenSize: 3L, bias: false);
        var flat = yNoB.Reshape([Scalar(-1L)]).Concat(0L, hNoB.Reshape([Scalar(-1L)])).Concat(0L, cNoB.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0539497955f, 0.00886780428f, -0.086620491f, 0.119741064f, 0.0356586562f, 0.0132102065f, 0.017189863f, -0.0203293715f, 0.174370424f, -0.0577061549f, 0.0046504567f, -0.076768852f, 0.119967297f, -0.00842105511f, 0.0590125428f, 0.043314729f, 0.0137322597f, 0.00594937787f, 0.0947569409f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 numLayers stacking: numLayers:2 (forward) — frozen forward-value golden
/// (self-generated): the configured layer's output must match the inlined reference. Asserts y, the
/// stacked [2,N,H] hN, and the stacked [2,N,H] cN all match. Layer-1's input size is D·H = H (D=1),
/// which the helper passes as Scalar(d·H); the reference reads it from the reshaped layer-0 Y's
/// last axis (same value), so the seeded inits coincide.</summary>
[Module]
public partial class LstmNumLayersStackMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L, numLayers: 2);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)])).Concat(0L, cN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.0585897258f, -0.0901772076f, 0.0307368469f, -0.237003166f, 0.100184695f, 0.0201188779f, -0.0801968088f, -0.0235858843f, 0.334827477f, 0.0376728655f, 0.0249635131f, -0.0128003496f, -0.154156339f, 0.17062034f, 0.0281458905f, -0.0296835913f, -0.099065379f, 0.16186443f, 0.0216668767f);
        var diff = (SelfCheck.Collapse(flat, 48) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 direction Reverse (trainable): Recurrent.LSTM(Reverse) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined
/// reference.</summary>
[Module]
public partial class LstmReverseMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L, direction: RnnDirection.Reverse);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)])).Concat(0L, cN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0110210366f, 0.0169940624f, -0.0450582385f, -0.00232494097f, 0.0750565864f, -0.0277827289f, -0.0348846472f, 0.0288584201f, 0.0837512101f, 0.138900113f, 0.0094067667f, 0.0192604071f, -0.294137098f, -0.0797829464f, 0.0114426774f, -0.0199438675f, 0.068415691f, 0.0140771565f, 0.0981605737f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 direction Bidirectional (forward inference only): Recurrent.LSTM(Bidirectional) —
/// frozen forward-value golden (self-generated): the configured layer's output must match the
/// inlined reference. Forward-value only (bidirectional BPTT throws AD003 — pinned in
/// NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class LstmBidirectionalMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        long hVal = 3L;
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: hVal, direction: RnnDirection.Bidirectional);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)])).Concat(0L, cN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0171791123f, -0.0254894614f, 0.0635323668f, 0.157808322f, 0.210832445f, 0.047314367f, -0.206614433f, -0.113424866f, 0.112038186f, 0.0594917558f, -0.0908273893f, -0.113273598f, 0.042337909f, 0.180483476f, -0.0415701944f, 0.0117359051f, -0.138804063f, 0.10611206f, 0.0394607332f);
        var diff = (SelfCheck.Collapse(flat, 72) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 batchFirst equivalence: LSTM(batchFirst:true, [N,L,in]) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined reference. Pins
/// the in-graph transpose + layout=0-always choice independently of the op reference. hN/cN are
/// batch-second in both (PyTorch keeps them [D·numLayers, N, H] regardless of batch_first), so they
/// match directly.</summary>
[Module]
public partial class LstmBatchFirstEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (yBF, hBF, cBF) = Recurrent.LSTM(x, hiddenSize: 3L, batchFirst: true);     // y [N, L, D*H]
        var flat = yBF.Reshape([Scalar(-1L)]).Concat(0L, hBF.Reshape([Scalar(-1L)])).Concat(0L, cBF.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0352879444f, 0.0136820285f, -0.111469804f, 0.0822292427f, 0.0866030488f, -0.0314609057f, -0.0233364428f, -0.00323860235f, 0.295652268f, 0.0081531109f, 0.0210374884f, -0.0135555522f, -0.231829053f, -0.0727247134f, 0.0796338831f, 0.0297410555f, 0.0140637151f, 0.0200658506f, 0.166782024f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 state contract: for a forward single-layer LSTM, hN == y[-1] (the last step's
/// hidden state), asserted relationally on the layer's own outputs — output-vs-output, valid under
/// any initialization — together with hN's and cN's leading dims (D·numLayers == 1) and the frozen
/// forward-value golden (self-generated) on y. Pins the (y, hN, cN) return relationship (cN has no
/// within-layer relational anchor — it is not derivable from y — so it gets the shape contract here
/// and value coverage via the goldens that concatenate it, e.g. LstmForwardGolden).</summary>
[Module]
public partial class LstmStateContractForwardSingleLayer
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 3L);   // y [L,N,H], hN/cN [1,N,H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.11317409f, -0.17832823f, 0.16449918f, 0.06303461f, -0.15622467f, 0.13228355f, 0.056307796f, -0.19639853f, 0.15588094f, -0.06749498f, -0.15526424f, 0.11855087f, -0.13697268f, -0.13327843f, 0.10542938f, -0.26116505f, -0.074378565f, 0.07426234f, -0.32657033f, -0.02437039f, 0.058734268f, -0.4117053f, 0.036474817f, 0.03893798f);
        var goldenDiff = (SelfCheck.Collapse(y, 24) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        // State contract: hN == y[-1] (last step along axis 0; both [1, N, H]); hN and cN
        // carry the [D·numLayers == 1, N, H] leading dim.
        var lastStep = y.Slice(Vector(-1L), Vector(System.Int64.MaxValue), Vector(0L));
        var stateDiff = (hN - lastStep).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var leadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>()
                      + (cN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();
        return (goldenDiff + stateDiff + leadingOk) < Scalar(1e-3f);
    }
}

/// <summary>§7-8 forward frozen golden: a forward, single-layer Recurrent.LSTM over a length-3 sequence
/// built from the probed scalar — frozen forward-value golden (self-generated). Op-level LSTM gradient
/// coverage lives in AutoGradOpsTests.</summary>
[Module]
public partial class LstmForwardGolden
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 3);   // [3, 1, 2]
        var (y, hN, cN) = Recurrent.LSTM(x, hiddenSize: 2L);   // forward, single-layer
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)])).Concat(0L, cN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.087859236f, -0.18765318f, 0.118634604f, -0.27065408f, 0.14037783f, -0.32592142f, 0.14037783f, -0.32592142f, 0.25955802f, -0.53694844f);
        var diff = (SelfCheck.Collapse(flat, 10) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
// self-checking [Module] computes (y, hN) = Recurrent.GRU(x, …) and compares the
// (collapsed) concatenation of BOTH outputs against an inlined frozen golden
// reference (self-generated at the fixed master-seed-0 per-parameter init), so
// the z,r,h gate packing, bias packing, linearBeforeReset plumbing, and the
// Y [L,D,N,H] -> [L,N,D*H] reshape all reach the verdict. The former hand-built
// OnnxOp.Gru references were retired with per-parameter init; op-level gradient
// coverage lives in AutoGradOpsTests. GRU has no QEE step values, so value
// correctness comes from the ORT backend inside AdvancedTestGraph. The
// GRU-specific addition over the LSTM/RNN sets is the linearBeforeReset
// both-forms check (GruLinearBeforeResetBothForms). The
// bidirectional BPTT-throws guard lives as a [Fact] in NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------


/// <summary>§7-1 Forward baseline: Recurrent.GRU(x, H) (linearBeforeReset:true default) — frozen
/// forward-value golden (self-generated) over BOTH y and hN, covering the z,r,h gate packing, bias
/// packing, and the [L,D,N,H]→[L,N,D*H] reshape.</summary>
[Module]
public partial class GruMatchesCoreOpForward
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.27399692f, -0.29238102f, -0.032711826f, 0.2549979f, -0.26112053f, -0.032672692f, 0.32737115f, -0.37541848f, -0.07276946f, 0.282864f, -0.33484834f, -0.062413827f, 0.25657204f, -0.3630614f, -0.07313037f, 0.20271942f, -0.3191216f, -0.0397749f, 0.15450838f, -0.3079069f, -0.010286821f, 0.10164662f, -0.2647475f, 0.0383931f, 0.15450838f, -0.3079069f, -0.010286821f, 0.10164662f, -0.2647475f, 0.0383931f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-1 (batchFirst) Matches the core op with batchFirst input:
/// Recurrent.GRU(batchFirst:true) on [N, L, in] — frozen forward-value golden (self-generated): the
/// configured layer's output must match the inlined reference. hN stays [D·numLayers, N, H]. Pins
/// the in-graph transpose around the layout=0 op.</summary>
[Module]
public partial class GruMatchesCoreOpBatchFirst
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L, batchFirst: true);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.27399692f, -0.29238102f, -0.032711826f, 0.36274162f, -0.4014722f, -0.07511998f, 0.35258955f, -0.42404234f, -0.10562572f, 0.30863762f, -0.40577865f, -0.11112566f, 0.17734714f, -0.17030106f, -0.0050303503f, 0.18215668f, -0.23758808f, -0.006929796f, 0.14269911f, -0.2460983f, 0.015938781f, 0.095182545f, -0.22712389f, 0.054580387f, 0.30863762f, -0.40577865f, -0.11112566f, 0.095182545f, -0.22712389f, 0.054580387f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2 linearBeforeReset BOTH forms (the GRU numeric crux): the reset-after default
/// (linearBeforeReset:true) AND the reset-before form (:false) both run on the same x, and y/hN of both
/// fold into one frozen forward-value golden (self-generated) — the two segments differ, so a knob that
/// stops being honored produces the other form's numbers and fails.</summary>
[Module]
public partial class GruLinearBeforeResetBothForms
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yLbr, hLbr) = Recurrent.GRU(x, hiddenSize: 3L, linearBeforeReset: true);
        var (yLbrF, hLbrF) = Recurrent.GRU(x, hiddenSize: 3L, linearBeforeReset: false);
        var flat = yLbr.Reshape([Scalar(-1L)]).Concat(0L, hLbr.Reshape([Scalar(-1L)])).Concat(0L, yLbrF.Reshape([Scalar(-1L)])).Concat(0L, hLbrF.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.0323575415f, -0.0300120219f, 0.473940777f, 0.0864825058f, 0.119011116f, 0.00316737661f, -0.139908961f, -0.165614712f, -0.120256247f, 0.0140325018f, 0.01636663f, 0.0132588371f, 0.46535151f, 0.238572942f, 0.0229435138f, -0.130355362f, -0.171485763f, -0.0109659706f, -0.0580852107f);
        var diff = (SelfCheck.Collapse(flat, 60) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-3 Single-step gate anchor: L=1, h_0=0, so the layer computes z=σ(W_z·x_0+b_z),
/// ĥ=tanh(W_h·x_0+b_h) (the reset term r⊙(R_h·h_0) vanishes at h_0=0; with linearBeforeReset:true the
/// r-gated recurrent bias also drops since Rb=0), H_1=(1−z)⊙ĥ, over the ONNX z,r,h gate blocks of the
/// packed [D,3H,in] W. H=2. Frozen forward-value golden (self-generated): a wrong gate packing or
/// equation changes the output and fails the reference comparison.</summary>
[Module]
public partial class GruSingleStepGateAnchor
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [1, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 2L);   // y [1,N,H], hN [1,N,H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.042853884f, 0.09562365f, -0.22733301f, -0.15599082f);
        var diff = (SelfCheck.Collapse(y, 4) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 bias on/off: bias:false — frozen forward-value golden (self-generated): the
/// configured layer's output must match the inlined reference. Both compared against the matching
/// seeded reference (same W/R; B present iff bias) on y and hN.</summary>
[Module]
public partial class GruBiasOnOffMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (yNoB, hNoB) = Recurrent.GRU(x, hiddenSize: 3L, bias: false);
        var flat = yNoB.Reshape([Scalar(-1L)]).Concat(0L, hNoB.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.069591865f, -0.03954286f, -0.007924555f, 0.04195672f, -0.00976763f, 0.0013781594f, 0.04012483f, -0.005158553f, 0.0053702244f, -0.0055433866f, 0.03493568f, 0.03379327f, -0.042403914f, 0.060642436f, 0.062078618f, -0.093667366f, 0.1024601f, 0.10726606f, -0.14075814f, 0.13556382f, 0.15077028f, -0.19056305f, 0.17537698f, 0.19723761f, -0.14075814f, 0.13556382f, 0.15077028f, -0.19056305f, 0.17537698f, 0.19723761f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 numLayers stacking: numLayers:2 (forward) — frozen forward-value golden
/// (self-generated): the configured layer's output must match the inlined reference. Asserts y and
/// the stacked [2,N,H] hN both match. Layer-1's input size is D·H = H (D=1), which the helper
/// passes as Scalar(d·H); the reference reads it from the reshaped layer-0 Y's last axis (same
/// value), so the seeded inits coincide.</summary>
[Module]
public partial class GruNumLayersStackMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L, numLayers: 2);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.0130116618f, -0.0336175347f, -0.0944020733f, 0.0446957713f, 0.146289364f, 0.041906493f, -0.0455070985f, 0.0204948539f, -0.240100003f, 0.136650568f, -0.0134949288f, -0.0927946105f, 0.00541197479f, 0.132505696f, 0.158232644f, 0.0262713755f, -0.143642092f, -0.209774517f, 0.0871628254f);
        var diff = (SelfCheck.Collapse(flat, 36) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 direction Reverse (trainable): Recurrent.GRU(Reverse) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined
/// reference.</summary>
[Module]
public partial class GruReverseMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L, direction: RnnDirection.Reverse);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.4077679f, -0.44982994f, -0.0885507f, 0.3587873f, -0.40374234f, -0.0699605f, 0.3072619f, -0.3476713f, -0.049650684f, 0.25851214f, -0.3012228f, -0.024963295f, 0.20649475f, -0.23482873f, -0.001577802f, 0.16043337f, -0.19375801f, 0.028224275f, 0.10895101f, -0.114151865f, 0.042830423f, 0.07030969f, -0.087940454f, 0.07418704f, 0.4077679f, -0.44982994f, -0.0885507f, 0.3587873f, -0.40374234f, -0.0699605f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 direction Bidirectional (forward inference only): Recurrent.GRU(Bidirectional) —
/// frozen forward-value golden (self-generated): the configured layer's output must match the
/// inlined reference. Forward-value only (bidirectional BPTT throws AD003 — pinned in
/// NNLibraryTrainingCoverageTests).</summary>
[Module]
public partial class GruBidirectionalMatchesCoreOp
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        long hVal = 3L;
        var (y, hN) = Recurrent.GRU(x, hiddenSize: hVal, direction: RnnDirection.Bidirectional);
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.00368851859f, 0.0334700003f, -0.216265408f, -0.125441415f, -0.00293130303f, -0.128228769f, -0.0347601329f, 0.243910034f, -0.131938476f, 0.0313053942f, 0.0530267273f, -0.0195986298f, -0.030214189f, 0.165358456f, -0.0580716027f, -0.0444605758f, -0.00146851796f, 0.103767109f, -0.0990497375f);
        var diff = (SelfCheck.Collapse(flat, 60) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-7 batchFirst equivalence: GRU(batchFirst:true, [N,L,in]) — frozen forward-value
/// golden (self-generated): the configured layer's output must match the inlined reference. Pins
/// the in-graph transpose + layout=0-always choice independently of the op reference. hN is
/// batch-second in both (PyTorch keeps it [D·numLayers, N, H] regardless of batch_first), so it
/// matches directly.</summary>
[Module]
public partial class GruBatchFirstEquivalence
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, L, in]
    {
        var (yBF, hBF) = Recurrent.GRU(x, hiddenSize: 3L, batchFirst: true);     // y [N, L, D*H]
        var flat = yBF.Reshape([Scalar(-1L)]).Concat(0L, hBF.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.27399692f, -0.29238102f, -0.032711826f, 0.36274162f, -0.4014722f, -0.07511998f, 0.35258955f, -0.42404234f, -0.10562572f, 0.30863762f, -0.40577865f, -0.11112566f, 0.17734714f, -0.17030106f, -0.0050303503f, 0.18215668f, -0.23758808f, -0.006929796f, 0.14269911f, -0.2460983f, 0.015938781f, 0.095182545f, -0.22712389f, 0.054580387f, 0.30863762f, -0.40577865f, -0.11112566f, 0.095182545f, -0.22712389f, 0.054580387f);
        var diff = (SelfCheck.Collapse(flat, 30) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-8 state contract: for a forward single-layer GRU, hN == y[-1] (the last step's
/// hidden state), asserted relationally on the layer's own outputs — output-vs-output, valid under
/// any initialization — together with hN's leading dim (D·numLayers == 1) and the frozen
/// forward-value golden (self-generated) on y. Pins the (y, hN) return relationship.</summary>
[Module]
public partial class GruStateContractForwardSingleLayer
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [L, N, in]
    {
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 3L);   // y [L,N,H], hN [1,N,H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.27399692f, -0.29238102f, -0.032711826f, 0.2549979f, -0.26112053f, -0.032672692f, 0.32737115f, -0.37541848f, -0.07276946f, 0.282864f, -0.33484834f, -0.062413827f, 0.25657204f, -0.3630614f, -0.07313037f, 0.20271942f, -0.3191216f, -0.0397749f, 0.15450838f, -0.3079069f, -0.010286821f, 0.10164662f, -0.2647475f, 0.0383931f);
        var goldenDiff = (SelfCheck.Collapse(y, 24) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        // State contract: hN == y[-1] (last step along axis 0; both [1, N, H]), leading dim == 1.
        var lastStep = y.Slice(Vector(-1L), Vector(System.Int64.MaxValue), Vector(0L));
        var stateDiff = (hN - lastStep).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var hLeadingOk = (hN.DimTensor(0) - Scalar(1L)).Abs().Cast<float32>();
        return (goldenDiff + stateDiff + hLeadingOk) < Scalar(1e-3f);
    }
}

/// <summary>§7-9 forward frozen golden: a forward, single-layer, linearBeforeReset:true Recurrent.GRU
/// over a length-3 sequence built from the probed scalar — frozen forward-value golden (self-generated).
/// Op-level GRU gradient coverage lives in AutoGradOpsTests.</summary>
[Module]
public partial class GruForwardGolden
{
    public static Scalar<bit> Inline(Scalar<float32> xv)
    {
        var x = RecurrentTestData.BuildX(xv, 3);   // [3, 1, 2]
        var (y, hN) = Recurrent.GRU(x, hiddenSize: 2L);   // forward, single-layer, linearBeforeReset:true
        var flat = y.Reshape([Scalar(-1L)]).Concat(0L, hN.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.010756155f, 0.17122807f, 0.028650515f, 0.29306695f, -0.009885404f, 0.16097677f, -0.009885404f, 0.16097677f);
        var diff = (SelfCheck.Collapse(flat, 8) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
// [Module] runs a cell (Recurrent.RNNCell/LSTMCell/GRUCell) — with nonzero
// previous state(s) so initial_h(/initial_c) are genuinely threaded — and
// compares the (collapsed) outputs against an inlined frozen golden reference
// (self-generated at the fixed master-seed-0 per-parameter init). The former
// hand-built seq=1 OnnxOp references were retired with per-parameter init;
// op-level gradient coverage lives in AutoGradOpsTests. Cells have NO QEE step
// values, so value correctness comes from the ORT backend inside
// AdvancedTestGraph. The AD003 relu-cell BPTT-throws guard lives as a [Fact] in
// NNLibraryTrainingCoverageTests.
// ---------------------------------------------------------------------------


// ===========================  RNNCell  =====================================

/// <summary>§7-1 (RNNCell) Single-step anchor: tanh. H=2, N=1, NONZERO h (so R is exercised, unlike the
/// layer's h_0=0 anchor); the cell computes h' = tanh(W·x + R·h + bias). x is [1, in] and h is the
/// in-module nonzero constant [1, 2]. Frozen forward-value golden (self-generated).</summary>
[Module]
public partial class RnnCellClosedFormTanh
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var h = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero previous state [N, H]
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: hv);   // [N, H]
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.68653524f, -0.019340754f);
        var diff = (SelfCheck.Collapse(hOut, 2) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0f, 0.13971166f);
        var diff = (SelfCheck.Collapse(hOut, 2) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2/§7-3 (RNNCell) single step: Recurrent.RNNCell(x, h, H) with nonzero h (so initial_h is
/// genuinely threaded) — frozen forward-value golden (self-generated); the [N, H] output shape is pinned
/// by the golden's element count. x [N, in], h [N, H].</summary>
[Module]
public partial class RnnCellMatchesSeq1Op
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);   // nonzero [N, H]
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: hv);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.1470648f, -0.08752948f, 0.6934531f, -0.21378177f, -0.011267126f, 0.46671247f);
        var diff = (SelfCheck.Collapse(hOut, 6) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 (RNNCell) bias on/off: bias:false — frozen forward-value golden (self-generated):
/// the configured layer's output must match the inlined reference. Both compared against the
/// matching seeded reference (same W/R; B iff bias).</summary>
[Module]
public partial class RnnCellBiasOnOff
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);   // nonzero [N, H]
        var hNoB = Recurrent.RNNCell(x, h, hiddenSize: hv, bias: false);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.17537177f, -0.08529431f, 0.5753161f, 0.107791305f, -0.009015322f, 0.29744554f);
        var diff = (SelfCheck.Collapse(hNoB, 6) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 (RNNCell) State threading — THE DEFINING TEST. Two hand-unrolled cell steps from
/// h_0 = 0: step 2 consumes step 1's h', so the golden (self-generated, over both steps' outputs) breaks
/// if the cell stops threading state. x is [2, N, in] (the two step inputs).</summary>
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
        var h2 = Recurrent.RNNCell(x1, h1, hiddenSize: hv);          // step 2 — threads h1
        var flat = h1.Reshape([Scalar(-1L)]).Concat(0L, h2.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.2613017f, -0.178007f, 0.67919075f, -0.32434636f, -0.1030699f, 0.44535577f, 0.02198875f, 0.3140962f, -0.19885433f, 0.14249063f, 0.1580782f, -0.23733258f);
        var diff = (SelfCheck.Collapse(flat, 12) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 (RNNCell) forward frozen golden (tanh): one RNNCell step where x AND the nonzero h are
/// built from the probed scalar (the h-input is the cell's distinguishing input) — frozen forward-value
/// golden (self-generated). Op-level RNN gradient coverage lives in AutoGradOpsTests.</summary>
[Module]
public partial class RnnCellForwardTanhGolden
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        // x [1, 2] and h [1, 2] both built from the probed scalar.
        var zv = (Tensor<float32>)OnnxOp.Unsqueeze(v, Vector(0L));
        var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var h = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(-0.2f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var hOut = Recurrent.RNNCell(x, h, hiddenSize: 2L);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.5967264f, 0.3259437f);
        var diff = (SelfCheck.Collapse(hOut, 2) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>§7-1 (LSTMCell) Single-step gate anchor. H=2, N=1, NONZERO h and c (so R and the forget gate
/// are exercised); in ONNX i,o,f,c order the cell computes pre-act = W·x + R·h + bias; i=σ(blk0),
/// o=σ(blk1), f=σ(blk2), g=tanh(blk3); c' = f⊙c + i⊙g; h' = o⊙tanh(c'). Frozen forward-value golden
/// (self-generated) over (h', c'): a wrong i,o,f,c↔i,f,g,o packing changes the output and fails the
/// reference comparison.</summary>
[Module]
public partial class LstmCellClosedFormGateAnchor
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var prevH = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero [N, H]
        var prevC = Tensor(new long[] { 1L, 2L }, -0.1f, 0.5f);   // nonzero [N, H]
        var (hOut, cOut) = Recurrent.LSTMCell(x, prevH, prevC, hiddenSize: hv);
        var flat = hOut.Reshape([Scalar(-1L)]).Concat(0L, cOut.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.052346908f, 0.111870185f, 0.09054956f, 0.17204428f);
        var diff = (SelfCheck.Collapse(flat, 4) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2/§7-3 (LSTMCell) single step: Recurrent.LSTMCell(x, h, c, H) with nonzero h and c (so
/// initial_h/initial_c are genuinely threaded) — frozen forward-value golden (self-generated) over BOTH
/// h' and c'; the [N, H] output shapes are pinned by the golden's element count.</summary>
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
        var flat = hOut.Reshape([Scalar(-1L)]).Concat(0L, cOut.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.060518965f, -0.18187805f, 0.112477764f, -0.16194735f, -0.15423499f, 0.08238829f, -0.10802947f, -0.37158445f, 0.20077951f, -0.26274914f, -0.33182037f, 0.16042358f);
        var diff = (SelfCheck.Collapse(flat, 12) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 (LSTMCell) bias on/off: bias:false — frozen forward-value golden (self-generated):
/// the configured layer's output must match the inlined reference.</summary>
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
        var flat = hNoB.Reshape([Scalar(-1L)]).Concat(0L, cNoB.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.060757663f, -0.124516405f, 0.021437975f, -0.03151314f, -0.08066856f, 0.007132032f, 0.14074452f, -0.23927873f, 0.038236484f, -0.06248232f, -0.16267192f, 0.013955582f);
        var diff = (SelfCheck.Collapse(flat, 12) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 (LSTMCell) State threading — the defining test. Two hand-unrolled cell steps from
/// h_0 = c_0 = 0: step 2 consumes step 1's (h', c'), and all four step outputs fold into one frozen
/// forward-value golden (self-generated), so the cell failing to thread EITHER carried state breaks the
/// reference comparison.</summary>
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
        var (h2, c2) = Recurrent.LSTMCell(x1, h1, c1, hiddenSize: hv);   // step 2 — threads (h1, c1)
        var flat = h1.Reshape([Scalar(-1L)]).Concat(0L, c1.Reshape([Scalar(-1L)])).Concat(0L, h2.Reshape([Scalar(-1L)])).Concat(0L, c2.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.037751045f, -0.1474912f, 0.12196061f, -0.053347927f, -0.11705802f, 0.09291129f, 0.073077075f, -0.29245093f, 0.21381588f, -0.0910616f, -0.24433406f, 0.1772903f, -0.01266461f, -0.1933409f, -0.034353413f, -0.022122065f, -0.10973765f, -0.0161384f, -0.024254601f, -0.34029967f, -0.08269314f, -0.041972794f, -0.18600497f, -0.04080532f);
        var diff = (SelfCheck.Collapse(flat, 24) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 (LSTMCell) forward frozen golden: one LSTMCell step where x, h AND c are built from the
/// probed scalar — frozen forward-value golden (self-generated) over (h', c'). Op-level LSTM gradient
/// coverage lives in AutoGradOpsTests.</summary>
[Module]
public partial class LstmCellForwardGolden
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        var zv = (Tensor<float32>)OnnxOp.Unsqueeze(v, Vector(0L));
        var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var h = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(-0.2f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var c = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.1f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var (hOut, cOut) = Recurrent.LSTMCell(x, h, c, hiddenSize: 2L);
        var flat = hOut.Reshape([Scalar(-1L)]).Concat(0L, cOut.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.1531399f, -0.15501392f, 0.30757904f, -0.27641153f);
        var diff = (SelfCheck.Collapse(flat, 4) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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

/// <summary>§7-1 (GRUCell) Single-step anchor, linearBeforeReset:true (reset-after). H=2, N=1, NONZERO h.
/// With the single owned bias (Rb=0) the cell computes, in ONNX z,r,h order: z=σ(Wz·x + Rz·h + bz),
/// r=σ(Wr·x + Rr·h + br), n=tanh(Wh·x + r⊙(Rh·h) + bh), h'=(1−z)⊙n + z⊙h. Frozen forward-value golden
/// (self-generated): a wrong z,r,h↔r,z,n packing changes the output and fails the reference
/// comparison.</summary>
[Module]
public partial class GruCellClosedFormLbrTrue
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var prevH = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero [N, H]
        var hOut = Recurrent.GRUCell(x, prevH, hiddenSize: hv, linearBeforeReset: true);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.059675388f, -0.10712758f);
        var diff = (SelfCheck.Collapse(hOut, 2) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-1 (GRUCell) linearBeforeReset:false (reset-before, the ONNX op default): the cell computes
/// n = tanh(Wh·x + bh + (r⊙h)·Rhᵀ). H=2, N=1, nonzero h. Frozen forward-value golden (self-generated),
/// generated with the lbr bit honored — a regression that ignores lbr:false produces the reset-after
/// numbers and fails the reference comparison.</summary>
[Module]
public partial class GruCellClosedFormLbrFalse
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N=1, in]
    {
        long hv = 2L;
        var prevH = Tensor(new long[] { 1L, 2L }, 0.3f, -0.4f);   // nonzero [N, H]
        var hLbrFalse = Recurrent.GRUCell(x, prevH, hiddenSize: hv, linearBeforeReset: false);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(-0.061078034f, -0.1143262f);
        var diff = (SelfCheck.Collapse(hLbrFalse, 2) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-2/§7-3 (GRUCell) single step: Recurrent.GRUCell(x, h, H) (linearBeforeReset:true) with
/// nonzero h (so initial_h is genuinely threaded) — frozen forward-value golden (self-generated); the
/// [N, H] output shape is pinned by the golden's element count.</summary>
[Module]
public partial class GruCellMatchesSeq1Op
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);   // nonzero [N, H]
        var hOut = Recurrent.GRUCell(x, h, hiddenSize: hv);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.28458244f, -0.18446526f, 0.09827934f, 0.25118813f, -0.14867939f, 0.08666855f);
        var diff = (SelfCheck.Collapse(hOut, 6) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-4 (GRUCell) bias on/off: bias:false — frozen forward-value golden (self-generated):
/// the configured layer's output must match the inlined reference.</summary>
[Module]
public partial class GruCellBiasOnOff
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // x is [N, in]
    {
        long hv = 3L;
        var n = x.DimTensor(0);
        var h = TensorFill((Vector<int64>)[n, Scalar(hv)], 0.2f);

        var hNoB = Recurrent.GRUCell(x, h, hiddenSize: hv, bias: false);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.10333493f, 0.04038326f, 0.1249094f, 0.06518152f, 0.07557812f, 0.123866774f);
        var diff = (SelfCheck.Collapse(hNoB, 6) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-5 (GRUCell) State threading — the defining test. Two hand-unrolled cell steps
/// (linearBeforeReset:true) from h_0 = 0: step 2 consumes step 1's h', and both steps' outputs fold into
/// one frozen forward-value golden (self-generated), so the cell failing to thread state breaks the
/// reference comparison.</summary>
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
        var h2 = Recurrent.GRUCell(x1, h1, hiddenSize: hv);          // step 2 — threads h1
        var flat = h1.Reshape([Scalar(-1L)]).Concat(0L, h2.Reshape([Scalar(-1L)]));
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.24790405f, -0.25077158f, -0.03188353f, 0.22431527f, -0.22003904f, -0.026484352f, 0.03697352f, -0.046876684f, -0.2740378f, 0.011591375f, -0.09246402f, -0.34831437f);
        var diff = (SelfCheck.Collapse(flat, 12) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
    }
}

/// <summary>§7-6 (GRUCell) forward frozen golden (linearBeforeReset:true): one GRUCell step where x AND
/// the nonzero h are built from the probed scalar — frozen forward-value golden (self-generated).
/// Op-level GRU gradient coverage (both lbr forms) lives in AutoGradOpsTests.</summary>
[Module]
public partial class GruCellForwardGolden
{
    public static Scalar<bit> Inline(Scalar<float32> v)
    {
        var zv = (Tensor<float32>)OnnxOp.Unsqueeze(v, Vector(0L));
        var x = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(0.4f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var h = ((Tensor<float32>)OnnxOp.Concat([zv, Vector(-0.2f)], axis: 0)).Reshape([Scalar(1L), Scalar(2L)]);
        var hOut = Recurrent.GRUCell(x, h, hiddenSize: 2L, linearBeforeReset: true);
        // REFERENCE: golden — Shorokoo's own forward output, frozen (self-generated).
        var reference = Vector(0.068942785f, 0.07279945f);
        var diff = (SelfCheck.Collapse(hOut, 2) - reference).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(1e-3f);
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
