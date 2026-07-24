namespace Shorokoo.Tests.Modules;

[TrainableParamInitializer]
public static partial class InitScalarWeight
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitBnRunningMean
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitBnRunningVar
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

/// <summary>
/// Optimizer-owned ones-fill state initializer. The ones value is deliberately different from
/// <see cref="Shorokoo.Modules.Optimizers.OptimizerStateZeros"/> so tests can prove optimizer
/// state really is initialized by its [StateInitializer] (not blanket zero-filled by the rig).
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class InitOptStateOnes
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

/// <summary>
/// Plain SGD plus a step-counter state created from <see cref="InitOptStateOnes"/>: the counter
/// starts at 1 and increments by 1 each step, while the parameter update ignores it. Lets tests
/// assert both the initializer-driven initial value and the per-step state round-trip.
/// </summary>
[Module]
public partial class StepCountingSgdOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate)
    {
        var stepCounter = InitOptStateOnes.Init(currentParam.ShapeTensor());
        Globals.StateUpdate(stepCounter, stepCounter + Scalar(1f));
        return currentParam - learningRate * grad;
    }
}

/// <summary>
/// Optimizer-owned state initializer that fills the parameter's shape with a supplied scalar value
/// — deliberately <b>reads a hyperparameter</b> (see <see cref="InitFromHyperOptimizer"/>) so its
/// state-init graph consumes a hyper input. Used to exercise the §2.5 value route (state init sees
/// the hyper's real value at the initial counters, not 0f) and the D5 fail-loud path.
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class InitToScalarFill
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> value)
    {
        // A param-shaped tensor filled with the (graph-valued) scalar: ones * value broadcasts it,
        // and — crucially — reads `value`, so the split-off state-init graph consumes that hyper input.
        return Globals.TensorFill(shape, 1.0f) * value;
    }
}

/// <summary>
/// SGD whose optimizer state is initialized to the learning-rate hyperparameter's value (via
/// <see cref="InitToScalarFill"/>), then carried unchanged. Because its state initializer reads the
/// LR hyper, the fresh optimizer state equals the LR at the initial counters — so a test can read
/// that state back and prove the value route feeds the real scheduled value (not the old hardcoded
/// 0f), and that a runtime LR triggers the D5 fail-loud unless supplied explicitly.
/// </summary>
[Module]
public partial class InitFromHyperOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate)
    {
        var s = InitToScalarFill.Init(currentParam.ShapeTensor(), learningRate);   // state init reads LR
        Globals.StateUpdate(s, s);                                                 // carried unchanged
        return currentParam - learningRate * grad;
    }
}

/// <summary>Impure scheduler module — carries a trainable parameter; rig build must reject it (D4).</summary>
[Module]
public partial class ParamScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var wScalar = w.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return step.Cast<float32>() * Scalar(0f) + wScalar;
    }
}

/// <summary>Impure scheduler module — carries module state (a StateUpdate); rig build must reject it (D4).</summary>
[Module]
public partial class StateScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step)
    {
        var s = InitBnRunningMean.Init(Vector(1L));
        Globals.StateUpdate(s, s + Scalar(1f));
        var sScalar = s.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return step.Cast<float32>() * Scalar(0f) + sScalar;
    }
}

/// <summary>Impure scheduler module — draws RNG; rig build must reject it (D4).</summary>
[Module]
public partial class RngScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step)
    {
        var r = Globals.RandomUniform(Vector(1L));
        var rScalar = r.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return step.Cast<float32>() * Scalar(0f) + rScalar;
    }
}

/// <summary>
/// Optimizer that misuses a module-owned state initializer for its state; the TrainingRig must
/// reject the graph with guidance towards StateOwnership.OptimizerOwned.
/// </summary>
[Module]
public partial class ModuleOwnedStateOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate)
    {
        var state = InitBnRunningMean.Init(currentParam.ShapeTensor());
        Globals.StateUpdate(state, state + Scalar(1f));
        return currentParam - learningRate * grad;
    }
}

/// <summary>
/// Model that misuses an optimizer-owned state initializer for module state; the TrainingRig
/// must reject the graph with guidance towards StateOwnership.ModuleOwned.
/// </summary>
[Module]
public partial class OptimizerOwnedStateModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var state = InitOptStateOnes.Init(Vector(1L));
        Globals.StateUpdate(state, state + Scalar(1f));
        return input * weight;
    }
}

[Module]
public partial class ScalarMultiplyModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        Vector<int64> weightShape = Vector(1L);
        var weight = InitScalarWeight.Init(weightShape);
        return input * weight;
    }
}

[Module]
public partial class ScalarMultiplyWithBatchNormModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var scalarShape = Vector(1L);

        var runningMean = InitBnRunningMean.Init(scalarShape);
        var runningVar = InitBnRunningVar.Init(scalarShape);

        Vector<int64> weightShape = Vector(1L);
        var weight = InitScalarWeight.Init(weightShape);

        Vector<int64> batchAxis = [Scalar(0L)];
        var batchMean = input.Reduce(ReduceKind.Mean, batchAxis, keepDims: false);
        var diff = input - batchMean;
        var batchVar = (diff * diff).Reduce(ReduceKind.Mean, batchAxis, keepDims: false);

        var epsilon = Scalar(1e-5f);
        var normalized = diff / (batchVar + epsilon).Sqrt();

        var momentum = Scalar(0.1f);
        var batchMeanVec = batchMean.Reshape(scalarShape);
        var batchVarVec = batchVar.Reshape(scalarShape);
        var updatedMean = runningMean * (Scalar(1f) - momentum) + batchMeanVec * momentum;
        var updatedVar = runningVar * (Scalar(1f) - momentum) + batchVarVec * momentum;
        Globals.StateUpdate(runningMean, updatedMean);
        Globals.StateUpdate(runningVar, updatedVar);

        return normalized * weight;
    }
}
