namespace Shorokoo.Tests.Modules;

[TrainableParamInitializer]
public static partial class InitXavier
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.RandomNormal(shape, mean: 0.0f, scale: 0.1f);
    }
}

[TrainableParamInitializer]
public static partial class InitZeroBias
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.0f);
    }
}

[Module]
public partial class DigitClassifier
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var x = input;
        var inFeatures = Scalar(64L);
        var hiddenSize = Scalar(32L);
        var numClasses = Scalar(10L);

        var w1 = InitXavier.Init([hiddenSize, inFeatures]);
        var b1 = InitZeroBias.Init([hiddenSize]).Vec();
        x = x.MatMul(w1.Transpose(1, 0)) + b1;
        x = (Tensor<float32>)OnnxOp.Relu(x);

        var w2 = InitXavier.Init([numClasses, hiddenSize]);
        var b2 = InitZeroBias.Init([numClasses]).Vec();
        x = x.MatMul(w2.Transpose(1, 0)) + b2;
        x = (Tensor<float32>)OnnxOp.Softmax(x, axis: 1);

        return x;
    }
}

[Module]
public partial class SoftmaxL2Loss
{
    public static Scalar<float32> Inline(Tensor<float32> predictions, Tensor<float32> targets)
    {
        var diff = predictions - targets;
        var squared = diff * diff;
        var perSample = (Tensor<float32>)OnnxOp.ReduceSum((Variable)squared, Vector(1L), keepdims: false, noopWithEmptyAxes: null);
        var batchMean = (Tensor<float32>)OnnxOp.ReduceMean((Variable)perSample, Vector(0L), keepdims: false);
        return batchMean.Scalar();
    }
}
