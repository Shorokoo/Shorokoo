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
