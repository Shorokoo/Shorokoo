using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;
using static Shorokoo.NN;

namespace RetinaNet.Models
{
    // Helper functions (non-module)
    public static class ResNetHelpers
    {
        public static Scalar<int64> ChannelsNCHW(Tensor<float32> x) => x.ShapeTensor()[1];

        public static Tensor<float32> SimpleMaxPool(Tensor<float32> x, long[] kernelShape, long[] strides)
        {
            return MaxPool(
                x,
                ceilMode: false,
                dilations: new long[] { 1L, 1L },
                kernelShape: kernelShape,
                pads: new long[] { 1L, 1L, 1L, 1L },
                storageOrder: 0L,
                strides: strides,
                autoPad: AutoPad.NotSet
            );
        }
    }

    #region Trainable Parameter Initializers (New Pattern)

    /// <summary>
    /// Trainable parameter initializer that fills with 1.0.
    /// Used for batch norm weight (gamma) and running variance.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class ResNetInitSimple
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f));
        }
    }

    /// <summary>
    /// Trainable parameter initializer using random normal distribution.
    /// Used for conv and dense layer weights.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class ResNetInitWeight
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Globals.RandomNormal(shape, mean: 0.0f, scale: 0.02f, seed: 42.0f);
        }
    }

    /// <summary>
    /// Trainable parameter initializer that fills with 0.0.
    /// Used for biases, batch norm bias (beta), and running mean.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class ResNetInitZeros
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Globals.TensorFill(shape, 0.0f);
        }
    }

    #endregion

    #region State Initializers

    /// <summary>
    /// State initializer for batch norm running mean (initialized to zeros).
    /// Running mean is module-owned state that gets updated during forward passes.
    /// </summary>
    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class BnRunningMeanInit
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Globals.TensorFill(shape, 0.0f);
        }
    }

    /// <summary>
    /// State initializer for batch norm running variance (initialized to ones).
    /// Running variance is module-owned state that gets updated during forward passes.
    /// </summary>
    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class BnRunningVarInit
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Globals.TensorFill(shape, 1.0f);
        }
    }

    #endregion

    [Module]
    public partial class DenseBasic
    {
        public static Tensor<float32> Inline(
                    Tensor<float32> x,
                    [Hyper] Scalar<int64> outFeatures,
                    [Hyper] Scalar<bit> useBias)
        {
            var batchSize = x.DimTensor(0);
            var inFeatures = x.TShape[1..^0].Reduce(ReduceKind.Prod).Scalar();

            var xFlat = x.Reshape([batchSize, inFeatures]);

            var w = ResNetInitWeight.Init([outFeatures, inFeatures]);

            var wT = w.Transpose([1L, 0L]);
            var y = xFlat.MatMul(wT);

            var b = ResNetInitZeros.Init([outFeatures]).Vec();
            var biasedY = y + b;
            y = useBias.IfElse(biasedY, y);

            return y;
        }
    }

    [Module]
    public partial class BatchNorm
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<float32> momentum, [Hyper] Scalar<float32> epsilon)
        {
            var numChannels = ResNetHelpers.ChannelsNCHW(x);

            // Running statistics use state initializers (not trainable - updated via StateUpdate)
            var runningMean = BnRunningMeanInit.Init([numChannels]).Vec();
            var runningVar = BnRunningVarInit.Init([numChannels]).Vec();

            // Scale (weight) and bias are trainable parameters
            var weight = ResNetInitSimple.Init([numChannels]).Vec();
            var bias = ResNetInitZeros.Init([numChannels]).Vec();

            // Compute batch mean and variance over spatial dimensions (N, H, W) for NCHW input
            // ReduceMean over axes [0, 2, 3] keeps the channel dimension
            Vector<int64> spatialAxes = [Scalar(0L), Scalar(2L), Scalar(3L)];
            var batchMean = x.Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);
            var diff = x - batchMean;
            var batchVar = (diff * diff).Reduce(ReduceKind.Mean, spatialAxes, keepDims: true);

            // Normalize: y = weight * (x - batchMean) / sqrt(batchVar + epsilon) + bias
            // Reshape weight and bias to [1, C, 1, 1] for broadcasting
            var rW = weight.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            var rB = bias.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            var invStd = (batchVar + epsilon).Sqrt();
            var y = rW * diff / invStd + rB;

            // Squeeze batch mean and variance to [C] for running statistics update
            var batchMeanVec = batchMean.Reshape([numChannels]).Vec();
            var batchVarVec = batchVar.Reshape([numChannels]).Vec();

            // Update running statistics with exponential moving average
            // ONNX convention: momentum=0.9 means retain 90% of running stats
            var updatedMean = runningMean * momentum + batchMeanVec * (Scalar(1f) - momentum);
            var updatedVar = runningVar * momentum + batchVarVec * (Scalar(1f) - momentum);

            // Register state updates so running statistics are tracked across forward passes
            Globals.StateUpdate(runningMean, updatedMean);
            Globals.StateUpdate(runningVar, updatedVar);

            return y;
        }
    }

    // FrozenBatchNorm2d - matches PyTorch's FrozenBatchNorm2d used in detection models
    // Uses optimized formula with eps=0.0: out = x * scale + bias
    // where scale = weight * rsqrt(running_var + eps) and bias = bias - running_mean * scale
    [Module]
    public partial class FrozenBatchNorm
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<float32> epsilon)
        {
            var numChannels = ResNetHelpers.ChannelsNCHW(x);
            var runningMean = ResNetInitZeros.Init([numChannels]).Vec();
            var runningVar = ResNetInitSimple.Init([numChannels]).Vec();
            var weight = ResNetInitSimple.Init([numChannels]).Vec();
            var bias = ResNetInitZeros.Init([numChannels]).Vec();

            // Reshape to [1, C, 1, 1] for broadcasting
            var rMean = runningMean.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            var rVar = runningVar.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            var rW = weight.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            var rB = bias.Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);

            // Optimized formula matching PyTorch FrozenBatchNorm2d:
            // scale = weight * rsqrt(running_var + eps) = weight / sqrt(running_var + eps)
            // bias = bias - running_mean * scale
            // output = x * scale + bias
            var scale = rW / (rVar + epsilon).Sqrt();
            var biasAdjusted = rB - rMean * scale;
            return x * scale + biasAdjusted;
        }
    }

    [Module]
    public partial class Conv2Dk11s11
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> filters)
        {
            var inC = ResNetHelpers.ChannelsNCHW(x);
            Vector<int64> wShape = [filters, inC, Scalar(1L), Scalar(1L)];
            var w = ResNetInitWeight.Init(wShape);
            var b = VectorFill(filters, 0f);
            return Conv(x, w, b, AutoPad.NotSet, dilations: [1L, 1L], group: 1L, kernelShape: [1L, 1L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
        }
    }

    [Module]
    public partial class Conv2Dk33s11
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> filters)
        {
            var inC = ResNetHelpers.ChannelsNCHW(x);
            Vector<int64> wShape = [filters, inC, Scalar(3L), Scalar(3L)];
            var w = ResNetInitWeight.Init(wShape);
            var b = VectorFill(filters, 0f);
            return Conv(x, w, b, AutoPad.NotSet, dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: [1L,1L,1L,1L], strides: [1L, 1L]);
        }
    }

    [Module]
    public partial class Conv2Dk11s22
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> filters)
        {
            var inC = ResNetHelpers.ChannelsNCHW(x);
            Vector<int64> wShape = [filters, inC, Scalar(1L), Scalar(1L)];
            var w = ResNetInitWeight.Init(wShape);
            var b = VectorFill(filters, 0f);
            return Conv(x, w, b, AutoPad.NotSet, dilations: [1L, 1L], group: 1L, kernelShape: [1L, 1L], pads: [0L,0L,0L,0L], strides: [2L, 2L]);
        }
    }

    [Module]
    public partial class Conv2Dk33s22
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> filters)
        {
            var inC = ResNetHelpers.ChannelsNCHW(x);
            Vector<int64> wShape = [filters, inC, Scalar(3L), Scalar(3L)];
            var w = ResNetInitWeight.Init(wShape);
            var b = VectorFill(filters, 0f);
            return Conv(x, w, b, AutoPad.NotSet, dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: [1L,1L,1L,1L], strides: [2L, 2L]);
        }
    }

    [Module]
    public partial class Conv2Dk77s22
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> filters)
        {
            var inC = ResNetHelpers.ChannelsNCHW(x);
            Vector<int64> wShape = [filters, inC, Scalar(7L), Scalar(7L)];
            var w = ResNetInitWeight.Init(wShape);
            var b = VectorFill(filters, 0f);
            var y = Conv(x, w, b, AutoPad.NotSet, dilations: [1L, 1L], group: 1L, kernelShape: [7L, 7L], pads: [0L, 0L, 0L, 0L], strides: [2L, 2L]);
            return y;
        }
    }

    [Module]
    public partial class BottleneckS11
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<bit> downsample, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            var c1 = Conv2Dk11s11.Model(filters).Call(x);
            var numChannels = ResNetHelpers.ChannelsNCHW(c1);

            c1 = FrozenBatchNorm.Call(bnEps, c1);
            c1 = c1.Relu();

            var c2 = Conv2Dk33s11.Call(filters, c1);
            c2 = FrozenBatchNorm.Call(bnEps, c2);
            c2 = c2.Relu();

            var c3 = Conv2Dk11s11.Model(Scalar(4L) * filters).Call(c2);
            c3 = FrozenBatchNorm.Model(bnEps).Call(c3);

            var shortcut = Conv2Dk11s11.Model(Scalar(4L) * filters).Call(x);
            shortcut = FrozenBatchNorm.Model(bnEps).Call(shortcut);
            shortcut = downsample.IfElse(shortcut, x);

            var outSum = c3 + shortcut;
            return outSum.Relu();
        }
    }

    [Module]
    public partial class BottleneckS22
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<bit> downsample, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            var c1 = Conv2Dk11s11.Model(filters).Call(x);
            c1 = FrozenBatchNorm.Model(bnEps).Call(c1);
            c1 = c1.Relu();

            var c2 = Conv2Dk33s22.Model(filters).Call(c1);
            c2 = FrozenBatchNorm.Model(bnEps).Call(c2);
            c2 = c2.Relu();

            var c3 = Conv2Dk11s11.Model(Scalar(4L) * filters).Call(c2);
            c3 = FrozenBatchNorm.Model(bnEps).Call(c3);

            var shortcut = Conv2Dk11s22.Model(Scalar(4L) * filters).Call(x);
            shortcut = FrozenBatchNorm.Model(bnEps).Call(shortcut);
            shortcut = downsample.IfElse(shortcut, x);

            var outSum = c3 + shortcut;
            return outSum.Relu();
        }
    }

    [Module]
    public partial class BottleneckStackS11
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<int64> blocks, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            foreach (var ctx in LoopAPI.Iterate(blocks))
                x = BottleneckS11.Model(filters, ctx.IterationIndex == Scalar(0L), bnMomentum, bnEps).Call(x);

            return x;
        }
    }

    [Module]
    public partial class BottleneckStackS22
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<int64> blocks, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            foreach (var ctx in LoopAPI.Iterate(blocks))
            {
                var isFirstIteration = ctx.IterationIndex == Scalar(0L);
                var s22Output = BottleneckS22.Model(filters, downsample: Scalar(true), bnMomentum, bnEps).Call(x);
                var s11Output = BottleneckS11.Model(filters, downsample: Scalar(false), bnMomentum, bnEps).Call(x);
                x = isFirstIteration.IfElse(s22Output, s11Output);
            }
            return x;
        }
    }

    [Module]
    public partial class ResNetStem
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<float32> bnEps)
        {
            x = x.Pad(PadMode.Constant, Vector(new long[] { 0, 0, 3, 3, 0, 0, 3, 3 }), val: Scalar(0f));
            x = Conv2Dk77s22.Model(Scalar(64L)).Call(x);
            x = FrozenBatchNorm.Model(bnEps).Call(x);
            x = x.Relu();
            x = ResNetHelpers.SimpleMaxPool(x, [3L, 3L], [2L, 2L]);
            return x;
        }
    }

    [Module]
    public partial class ClassificationHead
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            x = GlobalAveragePool(x);

            var top = DenseBasic.Model(numClasses, Scalar(true)).Call(x);
            var softmaxedTop = top.Softmax();

            top = applySoftmax.IfElse(softmaxedTop, top);
            x = includeTop.IfElse(top, x);

            return x;
        }
    }

    [Module]
    public partial class ResNet50
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            var x = ResNetStem.Model(bnEps).Call(inputs);
            x = BottleneckStackS11.Model(Scalar( 64L), Scalar(3L), bnMomentum, bnEps).Call(x);
            x = BottleneckStackS22.Model(Scalar(128L), Scalar(4L), bnMomentum, bnEps).Call(x);
            x = BottleneckStackS22.Model(Scalar(256L), Scalar(6L), bnMomentum, bnEps).Call(x);
            x = BottleneckStackS22.Model(Scalar(512L), Scalar(3L), bnMomentum, bnEps).Call(x);
            x = ClassificationHead.Model(numClasses, includeTop, applySoftmax).Call(x);
            return x;
        }
    }

    [Module]
    public partial class ResNet50Debug
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            var x = ResNetStem.Model(bnEps).Call(inputs);
            var x1 = x;
            x = BottleneckStackS11.Model(Scalar(64L), Scalar(3L), bnMomentum, bnEps).Call(x);
            var x2 = x;
            x = BottleneckStackS22.Model(Scalar(128L), Scalar(4L), bnMomentum, bnEps).Call(x);
            var x3 = x;
            x = BottleneckStackS22.Model(Scalar(256L), Scalar(6L), bnMomentum, bnEps).Call(x);
            var x4 = x;
            x = BottleneckStackS22.Model(Scalar(512L), Scalar(3L), bnMomentum, bnEps).Call(x);
            var x5 = x;
            x = ClassificationHead.Model(numClasses, includeTop, applySoftmax).Call(x);
            var x6 = x;
            return (x, x1, x2, x3, x4, x5, x6);
        }
    }

    [Module]
    public partial class ResNet101
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            var x = ResNetStem.Model(bnEps).Call(inputs);
            x = BottleneckStackS11.Model(Scalar( 64L), Scalar( 3L), bnMomentum, bnEps).Call(x);
            x = BottleneckStackS22.Model(Scalar(128L), Scalar( 4L), bnMomentum, bnEps).Call(x);
            x = BottleneckStackS22.Model(Scalar(256L), Scalar(23L), bnMomentum, bnEps).Call(x);
            x = BottleneckStackS22.Model(Scalar(512L), Scalar( 3L), bnMomentum, bnEps).Call(x);
            x = ClassificationHead.Model(numClasses, includeTop, applySoftmax).Call(x);
            return x;
        }
    }

    [Module]
    public partial class BasicBlockS11
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<bit> downsample, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            var c1 = Conv2Dk33s11.Model(filters).Call(x);
            c1 = FrozenBatchNorm.Model(bnEps).Call(c1);
            c1 = c1.Relu();

            var c2 = Conv2Dk33s11.Model(filters).Call(c1);
            c2 = FrozenBatchNorm.Model(bnEps).Call(c2);

            var shortcut = Conv2Dk11s11.Model(filters).Call(x);
            shortcut = FrozenBatchNorm.Model(bnEps).Call(shortcut);
            shortcut = downsample.IfElse(shortcut, x);

            var outSum = c2 + shortcut;
            return outSum.Relu();
        }
    }

    [Module]
    public partial class BasicBlockS22
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<bit> downsample, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            var c1 = Conv2Dk33s22.Model(filters).Call(x);
            c1 = FrozenBatchNorm.Model(bnEps).Call(c1);
            c1 = c1.Relu();

            var c2 = Conv2Dk33s11.Model(filters).Call(c1);
            c2 = FrozenBatchNorm.Model(bnEps).Call(c2);

            var shortcut = Conv2Dk11s22.Model(filters).Call(x);
            shortcut = FrozenBatchNorm.Model(bnEps).Call(shortcut);
            shortcut = downsample.IfElse(shortcut, x);

            var outSum = c2 + shortcut;
            return outSum.Relu();
        }
    }

    [Module]
    public partial class BasicStackS11
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<int64> blocks, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            foreach (var ctx in LoopAPI.Iterate(blocks))
                x = BasicBlockS11.Model(filters, downsample: Scalar(false), bnMomentum, bnEps).Call(x);

            return x;
        }
    }

    [Module]
    public partial class BasicStackS22
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters, [Hyper] Scalar<int64> blocks, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps)
        {
            var x0 = BasicBlockS22.Model(filters, downsample: Scalar(true), bnMomentum, bnEps).Call(x);
            foreach (var ctx in LoopAPI.Iterate(blocks-1))
                x0 = BasicBlockS11.Model(filters, downsample: Scalar(false), bnMomentum, bnEps).Call(x0);
            return x0;
        }
    }

    [Module]
    public partial class ResNet18
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            var x = ResNetStem.Model(bnEps).Call(inputs);
            x = BasicStackS11.Model(filters: Scalar(64L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            x = BasicStackS22.Model(filters: Scalar(128L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            x = BasicStackS22.Model(filters: Scalar(256L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            x = BasicStackS22.Model(filters: Scalar(512L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            x = ClassificationHead.Model(numClasses, includeTop, applySoftmax).Call(x);
            return x;
        }
    }

    [Module]
    public partial class ResNet18Debug
    {
        public static (Tensor<float32>, Tensor<float32>, Tensor<float32>, Tensor<float32> , Tensor<float32> , Tensor<float32> , Tensor<float32>) Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            var x = ResNetStem.Model(bnEps).Call(inputs);
            var x1 = x;
            x = BasicStackS11.Model(filters: Scalar( 64L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            var x2 = x;
            x = BasicStackS22.Model(filters: Scalar(128L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            var x3 = x;
            x = BasicStackS22.Model(filters: Scalar(256L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            var x4 = x;
            x = BasicStackS22.Model(filters: Scalar(512L), blocks: Scalar(2L), bnMomentum, bnEps).Call(x);
            var x5 = x;
            x = ClassificationHead.Model(numClasses, includeTop, applySoftmax).Call(x);
            var x6 = x;
            return (x, x1, x2, x3, x4, x5, x6);
        }
    }

    [Module]
    public partial class ResNet34
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numClasses, [Hyper] Scalar<float32> bnMomentum, [Hyper] Scalar<float32> bnEps, [Hyper] Scalar<bit> includeTop, [Hyper] Scalar<bit> applySoftmax)
        {
            var x = ResNetStem.Model(bnEps).Call(inputs);
            x = BasicStackS11.Model(filters: Scalar( 64L), blocks: Scalar(3L), bnMomentum, bnEps).Call(x);
            x = BasicStackS22.Model(filters: Scalar(128L), blocks: Scalar(4L), bnMomentum, bnEps).Call(x);
            x = BasicStackS22.Model(filters: Scalar(256L), blocks: Scalar(6L), bnMomentum, bnEps).Call(x);
            x = BasicStackS22.Model(filters: Scalar(512L), blocks: Scalar(3L), bnMomentum, bnEps).Call(x);
            x = ClassificationHead.Model(numClasses, includeTop, applySoftmax).Call(x);
            return x;
        }
    }
}
