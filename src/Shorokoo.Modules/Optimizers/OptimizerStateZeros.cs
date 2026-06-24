using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Optimizers;

/// <summary>
/// Shape-only zeros initializer for optimizer state (moment buffers, velocity, accumulators,
/// step counters). Optimizer-owned: the TrainingRig replicates state created from this
/// initializer once per trainable parameter and threads it through the optimizer-state struct.
/// Inside an optimizer's <c>Inline</c>, create state at the parameter's shape:
/// <code>
/// var m = OptimizerStateZeros.Init(currentParam.ShapeTensor());
/// ...
/// Globals.StateUpdate(m, newM);
/// </code>
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class OptimizerStateZeros
{
    public static Tensor<float32> Inline(Vector<int64> shape)
        => Globals.TensorFill(shape, 0.0f);
}

/// <summary>
/// Optimizer-owned zeros initializer for <b>scalar</b> optimizer state — a single value per
/// trainable parameter, such as Adam's timestep counter — stored as a true rank-0 scalar
/// rather than a parameter-shaped buffer. The lone value broadcasts against the param-shaped
/// tensors in the update, so a step counter costs one float per parameter instead of a full
/// copy of it. Create it with no shape argument:
/// <code>
/// var step = OptimizerScalarZeros.Init();
/// Globals.StateUpdate(step, step + Scalar(1f));
/// </code>
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class OptimizerScalarZeros
{
    public static Scalar<float32> Inline()
        => Globals.Scalar(0.0f);
}

/// <summary>
/// Optimizer-owned <b>ones</b> initializer for <b>scalar</b> optimizer state — the rank-0
/// counterpart to <see cref="OptimizerScalarZeros"/>, seeded at the multiplicative identity
/// <c>1.0</c> rather than <c>0.0</c>. Used for running-product state that must start at 1 so
/// that the first step yields the first factor rather than 0 (e.g. NAdam's running momentum
/// product <c>∏ μ_i</c>, where seeding at 0 would pin the product at 0 forever). The lone
/// value broadcasts against the param-shaped tensors, so it costs one float per parameter
/// instead of a full param-shaped copy. Create it with no shape argument:
/// <code>
/// var muProduct = OptimizerScalarOnes.Init();
/// Globals.StateUpdate(muProduct, muProduct * muT);
/// </code>
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class OptimizerScalarOnes
{
    public static Scalar<float32> Inline()
        => Globals.Scalar(1.0f);
}
