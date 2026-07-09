using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Initializers;

// Additional trainable-parameter initializers beyond the baseline set in
// Initializers.cs. Shape-only and stream-keyed/deterministic, following the same
// conventions; they reuse the InitializerMath fan-in arithmetic defined there.

/// <summary>
/// Truncated-normal initializer: standard normal clamped to [-2, 2], stream-keyed
/// (deterministic). This is an approximation — a true truncated normal resamples
/// values outside the range, but in-graph rejection sampling is not possible, so
/// out-of-range draws are clamped to the boundary instead. (Keras/JAX use a
/// truncated normal as the default dense-kernel initializer.)
/// </summary>
[TrainableParamInitializer]
public static partial class TruncatedNormal
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f).Clip(-2.0f, 2.0f);
}

/// <summary>
/// Configurable uniform initializer: fills the parameter with i.i.d. draws from
/// U(low, high), stream-keyed (deterministic). The parameterized generalization of
/// <see cref="Uniform"/> (== U(0,1) at the default bounds; <see cref="Uniform"/>
/// is retained as the fixed default form). Mirrors PyTorch's
/// <c>nn.init.uniform_(t, a, b)</c> and Keras's <c>RandomUniform(minval, maxval)</c>.
/// Any rank (no fan-in/out, no rank requirement) — works for biases and weights alike.
/// <c>low</c>/<c>high</c> are extra Inline parameters (the
/// <see cref="Constant"/>/<see cref="RecurrentUniform"/> extra-param precedent),
/// generating <c>UniformRange.Init(shape, low, high)</c>.
/// <see cref="Globals.RandomUniform(Vector{int64}, float, float)"/> takes
/// LITERAL float bounds, so the range is built in-graph as the affine transform of a
/// standard U(0,1) draw: u·(high − low) + low — the same fill-times-scalar shape
/// <see cref="XavierUniform"/> uses, plus a shift. Expects <c>low ≤ high</c>.
/// </summary>
[TrainableParamInitializer]
public static partial class UniformRange
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> low, Scalar<float32> high)
        // standard U(0,1) draw, affine-mapped to [low, high): u·(high − low) + low
        => Globals.RandomUniform(shape, low: 0.0f, high: 1.0f) * (high - low) + low;
}

/// <summary>
/// Configurable normal initializer: fills the parameter with i.i.d. draws from
/// N(mean, std) (std = standard deviation), stream-keyed (deterministic). The parameterized
/// generalization of <see cref="Normal"/> (== N(0,1) at the defaults; <see cref="Normal"/>
/// is retained as the fixed default form). Mirrors PyTorch's
/// <c>nn.init.normal_(t, mean, std)</c> and Keras's <c>RandomNormal(mean, stddev)</c>.
/// Any rank. <c>mean</c>/<c>std</c> are extra Inline parameters,
/// generating <c>NormalDist.Init(shape, mean, std)</c>.
/// <see cref="Globals.RandomNormal(Vector{int64}, float, float)"/> takes LITERAL
/// float mean/scale, so the distribution is built in-graph as the affine transform of a
/// standard N(0,1) draw: z·std + mean — the same random-times-scalar shape
/// <see cref="KaimingNormal"/> uses, plus a shift. Expects <c>std ≥ 0</c>.
/// </summary>
[TrainableParamInitializer]
public static partial class NormalDist
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> mean, Scalar<float32> std)
        // standard N(0,1) draw, affine-mapped to N(mean, std): z·std + mean
        => Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f) * std + mean;
}

/// <summary>
/// Configurable-gain Xavier/Glorot uniform: U(-a, a) with a = gain·sqrt(6/(fanIn+fanOut)),
/// stream-keyed (deterministic). The gain-parameterized generalization of <see cref="XavierUniform"/>
/// — at <c>gain = 1</c> it reproduces <see cref="XavierUniform"/>'s bound exactly (Xavier
/// bakes gain 1, so there is no √-rebasing here: the base factor is the same
/// sqrt(6/(fanIn+fanOut)) the default uses). Mirrors PyTorch's
/// <c>nn.init.xavier_uniform_(t, gain)</c>, where <c>gain</c> is a standard-deviation
/// multiplier (the value <c>nn.init.calculate_gain(nonlinearity)</c> returns — 1 for
/// linear/sigmoid, 5/3 for tanh, √2 for ReLU; the caller computes it). <c>gain</c>
/// is an extra Inline <see cref="Scalar{float32}"/> parameter (the
/// <see cref="UniformRange"/>/<see cref="NormalDist"/> precedent), generating
/// <c>XavierUniformGain.Init(shape, gain)</c>. Requires a rank &gt;= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class XavierUniformGain
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> gain)
    {
        // Xavier bakes gain 1, so the base factor is the same sqrt(6/(fanIn+fanOut)) the
        // default XavierUniform uses; the runtime gain simply scales it.
        var baseFactor = (6.0f / (InitializerMath.FanIn(shape) + InitializerMath.FanOut(shape))).Sqrt();
        return Globals.RandomUniform(shape, low: -1.0f, high: 1.0f) * (gain * baseFactor);
    }
}

/// <summary>
/// Configurable-gain Xavier/Glorot normal: N(0, std) with std = gain·sqrt(2/(fanIn+fanOut)),
/// stream-keyed (deterministic). The gain-parameterized generalization of <see cref="XavierNormal"/>
/// — at <c>gain = 1</c> it reproduces <see cref="XavierNormal"/>'s std exactly (Xavier bakes
/// gain 1, so the base factor is the same sqrt(2/(fanIn+fanOut)) the default uses). Mirrors
/// PyTorch's <c>nn.init.xavier_normal_(t, gain)</c>, where <c>gain</c> is a standard-deviation
/// multiplier (the <c>nn.init.calculate_gain</c> value; the caller computes it).
/// <c>gain</c> is an extra Inline <see cref="Scalar{float32}"/> parameter,
/// generating <c>XavierNormalGain.Init(shape, gain)</c>. Requires a rank &gt;= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class XavierNormalGain
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> gain)
    {
        var baseFactor = (2.0f / (InitializerMath.FanIn(shape) + InitializerMath.FanOut(shape))).Sqrt();
        return Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f) * (gain * baseFactor);
    }
}

/// <summary>
/// Configurable-gain Kaiming/He uniform (fan-in mode): U(-b, b) with b = gain·sqrt(3/fanIn),
/// stream-keyed (deterministic). NOTE the base factor is the bare <b>sqrt(3/fanIn)</b>, NOT the
/// sqrt(6/fanIn) of the default <see cref="KaimingUniform"/>: that default bakes the ReLU
/// gain √2 into its bound (<c>sqrt(6/fanIn) = √2·sqrt(3/fanIn)</c>), so once the caller
/// supplies the gain the √2 must NOT be double-baked. This class therefore uses the bare
/// sqrt(3/fanIn) and scales by the runtime <c>gain</c>, reproducing
/// <see cref="KaimingUniform"/> exactly at <c>gain = √2</c>
/// (<c>√2·sqrt(3/fanIn) = sqrt(6/fanIn)</c>). Mirrors PyTorch's
/// <c>nn.init.kaiming_uniform_(t, nonlinearity=...)</c>, whose gain is derived via
/// <c>calculate_gain</c> (the caller passes the raw value — e.g. <c>Scalar(MathF.Sqrt(2f))</c>
/// for ReLU). <c>gain</c> is an extra Inline <see cref="Scalar{float32}"/>
/// parameter, generating <c>KaimingUniformGain.Init(shape, gain)</c>. Requires a rank &gt;= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class KaimingUniformGain
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> gain)
    {
        // sqrt(3/fanIn) — the BARE base factor, NOT the default's sqrt(6/fanIn): the existing
        // KaimingUniform bakes gain √2 (sqrt(6/fanIn) = √2·sqrt(3/fanIn)); the user-supplied
        // gain replaces that, so the √2 must not be double-baked.
        var baseFactor = (3.0f / InitializerMath.FanIn(shape)).Sqrt();
        return Globals.RandomUniform(shape, low: -1.0f, high: 1.0f) * (gain * baseFactor);
    }
}

/// <summary>
/// Configurable-gain Kaiming/He normal (fan-in mode): N(0, std) with std = gain·sqrt(1/fanIn)
/// (= gain/sqrt(fanIn)), stream-keyed (deterministic). NOTE the base factor is the bare
/// <b>sqrt(1/fanIn)</b>, NOT the sqrt(2/fanIn) of the default <see cref="KaimingNormal"/>:
/// that default bakes the ReLU gain √2 into its std (<c>sqrt(2/fanIn) = √2·sqrt(1/fanIn)</c>),
/// so once the caller supplies the gain the √2 must NOT be double-baked. This class uses the
/// bare sqrt(1/fanIn) and scales by the runtime <c>gain</c>, reproducing
/// <see cref="KaimingNormal"/> exactly at <c>gain = √2</c>
/// (<c>√2·sqrt(1/fanIn) = sqrt(2/fanIn)</c>). Mirrors PyTorch's <c>nn.init.kaiming_normal_</c>,
/// whose gain is derived via <c>calculate_gain</c> (the caller passes the raw value).
/// <c>gain</c> is an extra Inline <see cref="Scalar{float32}"/> parameter,
/// generating <c>KaimingNormalGain.Init(shape, gain)</c>. Requires a rank &gt;= 2 shape.
/// </summary>
[TrainableParamInitializer]
public static partial class KaimingNormalGain
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> gain)
    {
        // sqrt(1/fanIn) = 1/sqrt(fanIn) — the BARE base factor, NOT the default's sqrt(2/fanIn):
        // KaimingNormal bakes gain √2 (sqrt(2/fanIn) = √2·sqrt(1/fanIn)); the user-supplied gain
        // replaces that, so the √2 must not be double-baked.
        var baseFactor = (1.0f / InitializerMath.FanIn(shape)).Sqrt();
        return Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f) * (gain * baseFactor);
    }
}

/// <summary>
/// LeCun normal: N(0, std) with std = sqrt(1 / fanIn), stream-keyed (deterministic).
/// Requires a rank &gt;= 2 shape. The default variance-scaling initializer in
/// JAX/Flax (<c>lecun_normal</c>), suited to SELU/self-normalizing networks.
/// </summary>
[TrainableParamInitializer]
public static partial class LeCunNormal
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        var std = (1.0f / InitializerMath.FanIn(shape)).Sqrt();
        return Globals.RandomNormal(shape, mean: 0.0f, scale: 1.0f) * std;
    }
}

/// <summary>
/// Approximate (semi-)orthogonal initializer (Saxe, McClelland &amp; Ganguli 2013,
/// "Exact solutions to the nonlinear dynamics of learning in deep linear neural
/// networks"), motivated by dynamical isometry: orthogonal weights preserve vector
/// norms (<c>‖Qx‖ = ‖x‖</c>), holding the input–output Jacobian's singular values
/// near 1 across depth and so avoiding the gradient explosion/vanishing that random
/// Gaussian init suffers — prized for RNN recurrent matrices and deep stacks.
///
/// <para>This is an <b>approximation by design</b> (cf. <see cref="TruncatedNormal"/>):
/// an <i>exact</i> orthogonal init needs a QR (or SVD) decomposition, and neither ONNX
/// nor Shorokoo's op set can express a QR/SVD in-graph (the only linear-algebra op is a
/// shape-only <c>Det</c>), nor is there a host-side hook to run one off-graph — an
/// initializer body is built into a computation graph and materialized by graph
/// evaluation. Instead this runs a fixed number of Björck / Newton–Schulz <i>cubic</i>
/// iterations <c>Y ← 1.5·Y − 0.5·Y·(Yᵀ·Y)</c> from a stream-keyed Gaussian, using only
/// matmul + transpose + elementwise ops (all materialize to values). Via the SVD the
/// map acts on each singular value as <c>σ ↦ 1.5σ − 0.5σ³</c>, whose fixed point is 1
/// and which converges quadratically for <c>σ ∈ (0, √3)</c>; it therefore drives every
/// singular value to 1 — in the natural (non-transposed) direction for both tall and
/// wide matrices — yielding a semi-orthogonal matrix that matches PyTorch's
/// <c>orthogonal_</c> semantics (<c>QᵀQ ≈ I</c> when rows ≥ cols, <c>QQᵀ ≈ I</c> when
/// rows &lt; cols).</para>
///
/// <para>The seed Gaussian is Frobenius-normalized first, since <c>σ_max ≤ ‖Y‖_F</c>,
/// so after dividing by <c>‖Y‖_F = 1</c> every singular value sits in <c>(0, √3)</c>
/// (the convergence region). 15 cubic steps is ample given quadratic convergence. For
/// rank &gt; 2 shapes the trailing dims are flattened to a 2-D <c>[shape[0],
/// prod(shape[1:])]</c> matrix, orthogonalized, then reshaped back (PyTorch convention).
/// Stream-keyed (deterministic); gain fixed at 1 (the baseline, like <see cref="Globals"/>'
/// Xavier/Kaiming inits). Exact QR-orthogonal init isn't expressible in Shorokoo's op set.</para>
/// </summary>
[TrainableParamInitializer]
public static partial class Orthogonal
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        // Flatten the requested shape to 2-D [r, c]: r = shape[0], c = prod(shape) / r.
        Scalar<int64> r = shape[0];
        Scalar<int64> total = shape.Reduce(ReduceKind.Prod);
        Scalar<int64> c = total / r;
        Vector<int64> flatShape = [r, c];

        // Starting point: a standard Gaussian [r, c] from the initializer's keyed stream.
        var y0 = Globals.RandomNormal(flatShape, mean: 0.0f, scale: 1.0f);

        // Frobenius-normalize so σ_max ≤ ‖Y‖_F = 1 < √3 (cubic-iteration convergence region).
        var frob = (y0 * y0).Reduce(ReduceKind.Sum, axes: null, keepDims: false).Sqrt();
        var y = y0 / frob;

        // Björck / Newton–Schulz cubic iteration: Y ← 1.5·Y − 0.5·Y·(Yᵀ·Y).
        // 15 steps — quadratic convergence drives every singular value to 1.
        for (int k = 0; k < 15; k++)
        {
            var g = y.Transpose(1L, 0L).MatMul(y);              // YᵀY  ([c, c])
            y = Scalar(1.5f) * y - 0.5f * y.MatMul(g);
        }

        return y.Reshape(shape);                                // gain fixed 1.0 (baseline)
    }
}
