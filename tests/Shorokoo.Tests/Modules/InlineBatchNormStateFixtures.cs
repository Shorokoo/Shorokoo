namespace Shorokoo.Tests.Modules;

[TrainableParamInitializer]
public static partial class InitSimpleNew
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class TestBnRunningMeanInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class TestBnRunningVarInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitRunningMean
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitRunningVariance
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

[Module]
public partial class BatchNormWithStateUpdate
{
    public static Tensor<float32> Inline(Tensor<float32> input, Scalar<float32> momentum)
    {
        var shape = input.ShapeTensor();
        var runningMean = InitRunningMean.Init(shape);
        var runningVariance = InitRunningVariance.Init(shape);
        var batchMean = input * Scalar(0.5f);
        var batchVar = input * Scalar(0.1f);
        var updatedMean = runningMean * (Scalar(1f) - momentum) + batchMean * momentum;
        var updatedVar = runningVariance * (Scalar(1f) - momentum) + batchVar * momentum;
        Globals.StateUpdate(runningMean, updatedMean);
        Globals.StateUpdate(runningVariance, updatedVar);
        return input - batchMean;
    }
}

[Module]
public partial class InlineBatchNormWithState
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        Scalar<int64> numChannels = x.ShapeTensor()[1];

        var runningMean = TestBnRunningMeanInit.Init([numChannels]).Vec();
        var runningVar = TestBnRunningVarInit.Init([numChannels]).Vec();

        var weight = InitSimpleNew.Init([numChannels]).Vec();
        var bias = InitSimpleNew.Init([numChannels]).Vec();

        Vector<int64> spatialAxes = [Scalar(0L), Scalar(2L), Scalar(3L)];
        var batchMean = x.Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);
        var diff = x - batchMean;
        var batchVar = (diff * diff).Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);

        var rW = weight.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
        var rB = bias.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
        var invStd = (batchVar + Scalar(1e-05f)).Sqrt();
        var y = rW * diff / invStd + rB;

        var batchMeanVec = batchMean.Reshape([numChannels]).Vec();
        var batchVarVec = batchVar.Reshape([numChannels]).Vec();

        var momentum = Scalar(0.9f);
        var updatedMean = runningMean * momentum + batchMeanVec * (Scalar(1f) - momentum);
        var updatedVar = runningVar * momentum + batchVarVec * (Scalar(1f) - momentum);

        Globals.StateUpdate(runningMean, updatedMean);
        Globals.StateUpdate(runningVar, updatedVar);

        return y;
    }
}

/// <summary>Initializer used ONLY by <see cref="StateUpdateSurvivesNestedFirstUseBuild"/>, so its
/// Function is guaranteed uncached when that module's body traces — forcing a nested graph build
/// mid-trace.</summary>
[TrainableParamInitializer]
public static partial class StateWipeFreshInit
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.25f);
    }
}

/// <summary>
/// State updates registered BEFORE a nested first-use build must survive it. Building a
/// not-yet-cached sub-module/initializer mid-trace re-enters the graph builder on the same
/// thread; its entry-time state-update clearing used to wipe the outer body's already-registered
/// updates, silently dropping the WITH_STATE_DEPS wrap (and cache-order-dependently: a warm
/// Function cache hid the loss). StateUpdate(counter, ...) then first-use a fresh initializer:
/// the built graph must still carry the wrap.
/// </summary>
[Module]
public partial class StateUpdateSurvivesNestedFirstUseBuild
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var counter = InitRunningMean.Init(x.ShapeTensor());       // [StateInitializer] state
        Globals.StateUpdate(counter, counter + Scalar(1f));        // registered now — before the nested build
        var w = StateWipeFreshInit.Init([Scalar(4L)]);             // FIRST use: nested initializer body build
        return x + counter + w.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
    }
}
