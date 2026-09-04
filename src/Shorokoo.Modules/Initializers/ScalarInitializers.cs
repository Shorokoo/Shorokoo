using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Initializers;

// ---------------------------------------------------------------------------
// Rank-0 (scalar) trainable-parameter initializers.
//
// Every initializer in Initializers.cs / ExtraInitializers.cs takes a shape
// vector, and a learned scalar was written as `Ones.Init([Scalar(1L)])` — a
// length-1 rank-1 tensor that broadcasts correctly but persists as a
// `[1]`-shaped parameter where a scalar was meant. (An empty shape vector,
// `Ones.Init(EmptyVector<int64>())`, does give rank 0, but that is neither an
// obvious nor a self-describing way to ask for one.) The initializers below
// take NO shape argument and return a true rank-0 Scalar<float32>: the
// trainable-parameter counterparts of the rank-0 state initializers
// OptimizerScalarZeros / OptimizerScalarOnes.
//
// They are deterministic (no RNG, no fan-in/out — neither is meaningful for a
// lone value), and mirror the Zeros / Ones / Constant trio of the baseline set:
// ScalarZeros == ScalarConstant(0), ScalarOnes == ScalarConstant(1).
// ---------------------------------------------------------------------------

/// <summary>
/// Rank-0 zeros initializer for a <b>trainable scalar</b> parameter — a single learned
/// value stored as a true rank-0 tensor rather than the <c>[1]</c>-shaped rank-1 tensor
/// <c>Zeros.Init([Scalar(1L)])</c> would leave in the checkpoint. Seeded at the additive
/// identity <c>0.0</c>, so a residual scale or a gate bias starts as a no-op. The
/// trainable counterpart of <see cref="Optimizers.OptimizerScalarZeros"/>; the rank-0
/// case of <see cref="Zeros"/>. Create it with no shape argument:
/// <code>
/// var beta = ScalarZeros.Init();
/// </code>
/// The lone value broadcasts against tensors of any shape, so it composes with the rest
/// of a module without a reshape.
/// </summary>
[TrainableParamInitializer]
public static partial class ScalarZeros
{
    public static Scalar<float32> Inline()
        => Globals.Scalar(0.0f);
}

/// <summary>
/// Rank-0 ones initializer for a <b>trainable scalar</b> parameter — the rank-0
/// counterpart of <see cref="Ones"/>, seeded at the multiplicative identity <c>1.0</c>
/// so a learned gain (a per-layer temperature, a residual scale, a gated architecture's
/// <c>gamma</c>) starts as a no-op. Stored as a true rank-0 tensor rather than the
/// <c>[1]</c>-shaped parameter <c>Ones.Init([Scalar(1L)])</c> would leave in the
/// checkpoint. The trainable counterpart of <see cref="Optimizers.OptimizerScalarOnes"/>.
/// Create it with no shape argument:
/// <code>
/// var gamma = ScalarOnes.Init();
/// </code>
/// </summary>
[TrainableParamInitializer]
public static partial class ScalarOnes
{
    public static Scalar<float32> Inline()
        => Globals.Scalar(1.0f);
}

/// <summary>
/// Rank-0 constant initializer for a <b>trainable scalar</b> parameter: a single learned
/// value seeded at a caller-supplied <c>value</c>. The parameterized generalization of
/// <see cref="ScalarZeros"/> (== <c>ScalarConstant(0)</c>) and <see cref="ScalarOnes"/>
/// (== <c>ScalarConstant(1)</c>), exactly as <see cref="Constant"/> generalizes
/// <see cref="Zeros"/>/<see cref="Ones"/> for shaped parameters. Deterministic (no RNG,
/// no seed). Use it for a learned scalar whose starting point is neither 0 nor 1 — an
/// attention temperature seeded at <c>1/√d</c>, a LayerScale seeded at <c>1e-4</c>.
/// <c>value</c> is an extra Inline parameter (the <see cref="Constant"/> precedent),
/// generating <c>ScalarConstant.Init(value)</c>:
/// <code>
/// var temperature = ScalarConstant.Init(Scalar(0.125f));
/// </code>
/// The body scales a rank-0 <c>1.0</c> by the runtime <c>value</c> — the same fill-times-scalar
/// shape <see cref="Constant"/> uses, minus the fill — rather than handing the input straight
/// back: an initializer whose output IS its input exports as a nameless ONNX function output.
/// </summary>
[TrainableParamInitializer]
public static partial class ScalarConstant
{
    public static Scalar<float32> Inline(Scalar<float32> value)
        => Globals.Scalar(1.0f) * value;
}
