using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;
using static Shorokoo.NN;

namespace RetinaNet.Models
{
    /// <summary>
    /// RetinaNet Object Detector
    /// 
    /// A complete implementation of the RetinaNet architecture based on the PyTorch torchvision implementation.
    /// RetinaNet is a one-stage object detector that uses:
    /// - A backbone network (ResNet or ViT) for feature extraction
    /// - A Feature Pyramid Network (FPN) for multi-scale feature representation (P3-P7)
    /// - Classification and regression heads for object detection
    /// 
    /// Key features matching PyTorch implementation:
    /// - Uses P3-P7 pyramid levels (P6 and P7 are generated from C5 and P6 respectively)
    /// - Classification head uses 4 conv layers with GroupNorm + ReLU
    /// - Regression head uses 4 conv layers with GroupNorm + ReLU
    /// - Sigmoid activation on classification outputs (for focal loss compatibility)
    /// - 9 anchors per location (3 scales × 3 aspect ratios)
    /// 
    /// Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    /// PyTorch implementation: torchvision.models.detection.retinanet_resnet50_fpn
    /// </summary>
    
    // Helper functions (non-module)
    public static class RetinaNetHelpers
    {
        public static Scalar<int64> GetDim(Tensor<float32> x, Scalar<int64> axis) => x.ShapeTensor()[axis];
        public static Scalar<int64> ChannelsNCHW(Tensor<float32> x) => GetDim(x, Scalar(1L));
    }
    
    #region Trainable Parameter Initializers (New Pattern)

    /// <summary>
    /// Simple initialization using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class RetInitSimple
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f));
        }
    }

    /// <summary>
    /// Xavier/Glorot initialization using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class RetInitXavier
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            // Xavier/Glorot initialization - using 0.02f which is a common standard
            // This approximates Xavier init for common layer sizes (similar to PyTorch's default)
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 0.02f));
        }
    }

    /// <summary>
    /// Zero initialization using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class RetInitZeros
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 0.0f));
        }
    }

    /// <summary>
    /// Classification bias initialization for focal loss using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class RetInitClsBias
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            // Initialize classification head bias with prior probability
            // This helps with class imbalance during training (focal loss paper)
            // Default prior probability of 0.01 for rare positive examples
            // Computed as: -log((1 - prior) / prior) where prior = 0.01
            // = -log(0.99 / 0.01) = -log(99) ≈ -4.595
            const float priorProbability = 0.01f;
            var bias = -(float)Math.Log((1.0 - priorProbability) / priorProbability);
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, bias));
        }
    }

    #endregion

    /// <summary>
    /// ResNet18 backbone wrapper that extracts multi-scale features for RetinaNet
    /// Returns feature maps from different stages: C3, C4, C5 (stride 8, 16, 32)
    /// Uses BasicBlock architecture (2 conv layers per block) instead of Bottleneck
    /// </summary>
    [Module]
    public partial class RetResNet18Backbone
    {
        public static (Tensor<float32> c3, Tensor<float32> c4, Tensor<float32> c5) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            // Capture intermediate outputs while keeping Module.Call() sequence identical
            var stem = ResNetStem.Model(bnEps).Call(inputs);
            var c1 = BasicStackS11.Model(filters: Scalar(64L), blocks: Scalar(2L), bnMomentum, bnEps).Call(stem);
            var c2 = BasicStackS22.Model(filters: Scalar(128L), blocks: Scalar(2L), bnMomentum, bnEps).Call(c1);
            var c3 = c2;  // c3 is same as c2 (layer2 output)
            var c4 = BasicStackS22.Model(filters: Scalar(256L), blocks: Scalar(2L), bnMomentum, bnEps).Call(c3);
            var c5 = BasicStackS22.Model(filters: Scalar(512L), blocks: Scalar(2L), bnMomentum, bnEps).Call(c4);
            
            return (c3, c4, c5);
        }
    }

    /// <summary>
    /// ResNet50 backbone wrapper that extracts multi-scale features for RetinaNet
    /// Returns feature maps from different stages: C3, C4, C5 (stride 8, 16, 32)
    /// Note: c3=layer2_out (stride 8), c4=layer3_out (stride 16), c5=layer4_out (stride 32)
    /// </summary>
    [Module]
    public partial class RetResNet50Backbone
    {
        public static (Tensor<float32> c3, Tensor<float32> c4, Tensor<float32> c5) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            // Capture intermediate outputs while keeping Module.Call() sequence identical
            var c1 = ResNetStem.Model(bnEps).Call(inputs);
            var c2 = BottleneckStackS11.Model(Scalar(64L), Scalar(3L), bnMomentum, bnEps).Call(c1);
            var c3 = BottleneckStackS22.Model(Scalar(128L), Scalar(4L), bnMomentum, bnEps).Call(c2);
            var c4 = BottleneckStackS22.Model(Scalar(256L), Scalar(6L), bnMomentum, bnEps).Call(c3);
            var c5 = BottleneckStackS22.Model(Scalar(512L), Scalar(3L), bnMomentum, bnEps).Call(c4);
            
            return (c3, c4, c5);
        }
    }

    /// <summary>
    /// ResNet101 backbone wrapper that extracts multi-scale features for RetinaNet
    /// Returns feature maps from different stages: C3, C4, C5 (stride 8, 16, 32)
    /// </summary>
    [Module]
    public partial class ResNet101Backbone
    {
        public static (Tensor<float32> c3, Tensor<float32> c4, Tensor<float32> c5) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            // Capture intermediate outputs while keeping Module.Call() sequence identical
            var stem = ResNetStem.Model(bnEps).Call(inputs);
            var c1 = BottleneckStackS11.Model(Scalar(64L), Scalar(3L), bnMomentum, bnEps).Call(stem);
            var c2 = BottleneckStackS22.Model(Scalar(128L), Scalar(4L), bnMomentum, bnEps).Call(c1);
            var c3 = c2;  // c3 is same as c2 (layer2 output)
            var c4 = BottleneckStackS22.Model(Scalar(256L), Scalar(23L), bnMomentum, bnEps).Call(c3);
            var c5 = BottleneckStackS22.Model(Scalar(512L), Scalar(3L), bnMomentum, bnEps).Call(c4);

            return (c3, c4, c5);
        }
    }

    /// <summary>
    /// ViT backbone wrapper that extracts multi-scale features for RetinaNet
    /// Simulates multi-scale features by processing patches at different transformer depths
    /// </summary>
    [Module]
    public partial class RetViTBackbone
    {
        public static (Tensor<float32> c3, Tensor<float32> c4, Tensor<float32> c5) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<bit> useViTBase)
        {
            // Use ViTBase or ViTLarge configuration
            var embedDim = useViTBase.IfElse(Scalar(768L), Scalar(1024L));
            var numHeads = useViTBase.IfElse(Scalar(12L), Scalar(16L));
            var numLayers = useViTBase.IfElse(Scalar(12L), Scalar(24L));
            var hiddenDim = useViTBase.IfElse(Scalar(3072L), Scalar(4096L));
            var dropoutRate = Scalar(0.0f);
            var layerNormEps = Scalar(1e-6f);
            var maxSeqLen = Scalar(197L);
            var patchSize = Scalar(16L);
            
            // Patch embedding - use Model/Call pattern
            var x = TfPatchEmbedding.Model(patchSize, embedDim).Call(inputs);
            
            // Add CLS token and positional encoding
            x = RetAddClsTokenLocal(embedDim, x);
            x = TfPositionalEncoding.Model(maxSeqLen, embedDim).Call(x);
            
            // Extract features at different transformer depths to simulate multi-scale
            // Create transformer block models - each layer should have its own parameters
            var earlyLayers = numLayers / 3;
            var midLayers = numLayers / 3;
            var lateLayers = numLayers - earlyLayers - midLayers;
            
            // Early layers (C3 equivalent) - each layer has its own parameters
            var x_c3 = x;
            foreach (var ctx in LoopAPI.Iterate(earlyLayers))
                x_c3 = TfTransformerBlock.Model(embedDim, numHeads, hiddenDim, dropoutRate, layerNormEps).Call(x_c3);
            
            // Mid layers (C4 equivalent) - each layer has its own parameters
            var x_c4 = x_c3;
            foreach (var ctx in LoopAPI.Iterate(midLayers))
                x_c4 = TfTransformerBlock.Model(embedDim, numHeads, hiddenDim, dropoutRate, layerNormEps).Call(x_c4);
            
            // Late layers (C5 equivalent) - each layer has its own parameters
            var x_c5 = x_c4;
            foreach (var ctx in LoopAPI.Iterate(lateLayers))
                x_c5 = TfTransformerBlock.Model(embedDim, numHeads, hiddenDim, dropoutRate, layerNormEps).Call(x_c5);
            
            // Apply layer norm to all outputs - each output level gets its own layer norm parameters
            var c3 = RetLayerNormLocal(embedDim, layerNormEps, x_c3);
            var c4 = RetLayerNormLocal(embedDim, layerNormEps, x_c4);
            var c5 = RetLayerNormLocal(embedDim, layerNormEps, x_c5);


            // Convert ViT sequence outputs to spatial feature maps
            var inputHeight = Scalar(224L);
            var inputWidth = Scalar(224L);
            var featureHeight = inputHeight / patchSize;
            var featureWidth = inputWidth / patchSize;

            // Remove CLS token and reshape to spatial format
            c3 = RetRemoveCLSAndReshapeToSpatial(featureHeight, featureWidth, c3);
            c4 = RetRemoveCLSAndReshapeToSpatial(featureHeight, featureWidth, c4);
            c5 = RetRemoveCLSAndReshapeToSpatial(featureHeight, featureWidth, c5);

            return (c3, c4, c5);
        }
        
        private static Tensor<float32> RetAddClsTokenLocal(Scalar<int64> embedDim, Tensor<float32> x)
        {
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var clsToken = RetInitSimple.Init([Scalar(1L), Scalar(1L), embedDim]);
            var clsTokens = clsToken.Tile([batchSize, Scalar(1L), Scalar(1L)]);
            return clsTokens.Concat(axis: 1L, x);
        }
        
        private static Tensor<float32> RetLayerNormLocal(
            Scalar<int64> normalizedShape,
            Scalar<float32> eps,
            Tensor<float32> x)
        {
            // Simple layer normalization using mean and variance
            var mean = x.Reduce(ReduceKind.Mean, axes: Vector(new long[] {-1L}), keepDims: true);
            var variance = ((x - mean) * (x - mean)).Reduce(ReduceKind.Mean, axes: Vector(new long[] {-1L}), keepDims: true);
            var normalized = (x - mean) / (variance + eps).Sqrt();
            
            // Learnable parameters
            var gamma = RetInitSimple.Init([normalizedShape]);
            var beta = RetInitSimple.Init([normalizedShape]);
            
            return normalized * gamma + beta;
        }
        
        private static Tensor<float32> RetRemoveCLSAndReshapeToSpatial(
            Scalar<int64> height, 
            Scalar<int64> width,
            Tensor<float32> x)
        {
            // x shape: [batch, seq_len, embed_dim] where seq_len = 1 + height * width (CLS + patches)
            // Remove CLS token (first token)
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var embedDim = RetinaNetHelpers.GetDim(x, Scalar(2L));
            var seqLenWithoutCls = height * width;
            var xWithoutCLS = x.Slice(start: Vector(new long[] {0L, 1L, 0L}), end: Vector(new long[] {-1L, -1L, -1L}));
            
            // Reshape to spatial format: [batch, embed_dim, height, width]
            return xWithoutCLS.Reshape([batchSize, height, width, embedDim]).Transpose([0L, 3L, 1L, 2L]);
        }
    }

    // Helper convolution modules
    [Module]
    public partial class RetConv1x1
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters)
        {
            var inC = RetinaNetHelpers.ChannelsNCHW(x);
            Vector<int64> wShape = [filters, inC, Scalar(1L), Scalar(1L)];
            var w = RetInitXavier.Init(wShape);
            var b = RetInitZeros.Init([filters]).Vec();  // Trainable bias (matches PyTorch)
            return Conv(x, w, b, AutoPad.NotSet, dilations: [1L, 1L], group: 1L, kernelShape: [1L, 1L], pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);
        }
    }
    
    [Module]
    public partial class RetConv3x3
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters)
        {
            var inC = RetinaNetHelpers.ChannelsNCHW(x);
            var groups = Scalar(1L);
            Vector<int64> wShape = [filters, (inC / groups), Scalar(3L), Scalar(3L)];
            var w = RetInitXavier.Init(wShape);
            var b = RetInitZeros.Init([filters]).Vec();  // Trainable bias (matches PyTorch)
            return Conv(x, w, b, AutoPad.SameUpper, dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: null, strides: [1L, 1L]);
        }
    }
    
    [Module]
    public partial class RetConv3x3Stride2
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters)
        {
            var inC = RetinaNetHelpers.ChannelsNCHW(x);
            var groups = Scalar(1L);
            Vector<int64> wShape = [filters, (inC / groups), Scalar(3L), Scalar(3L)];
            var w = RetInitXavier.Init(wShape);
            var b = RetInitZeros.Init([filters]).Vec();  // Trainable bias (matches PyTorch)
            return Conv(x, w, b, AutoPad.SameUpper, dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: null, strides: [2L, 2L]);
        }
    }
    
    [Module]
    public partial class RetConv3x3WithClsBias
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> filters)
        {
            var inC = RetinaNetHelpers.ChannelsNCHW(x);
            var groups = Scalar(1L);
            Vector<int64> wShape = [filters, (inC / groups), Scalar(3L), Scalar(3L)];
            var w = RetInitXavier.Init(wShape);
            // Use focal loss prior probability bias initialization
            var b = RetInitClsBias.Init([filters]).Vec();
            return Conv(x, w, b, AutoPad.SameUpper, dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L], pads: null, strides: [1L, 1L]);
        }
    }
    
    [Module]
    public partial class RetGroupNorm
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<int64> numChannels)
        {
            var eps = Scalar(1e-5f);
            
            // x shape: [batch, channels, height, width]
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var height = RetinaNetHelpers.GetDim(x, Scalar(2L));
            var width = RetinaNetHelpers.GetDim(x, Scalar(3L));
            var channelsPerGroup = numChannels / numGroups;
            
            // Reshape to [batch, numGroups, channelsPerGroup, height, width]
            var xReshaped = x.Reshape([batchSize, numGroups, channelsPerGroup, height, width]);
            
            // Compute mean and variance over channelsPerGroup, height, width (axes 2, 3, 4)
            var mean = xReshaped.Reduce(ReduceKind.Mean, axes: Vector(new long[] {2L, 3L, 4L}), keepDims: true);
            var variance = ((xReshaped - mean) * (xReshaped - mean)).Reduce(ReduceKind.Mean, axes: Vector(new long[] {2L, 3L, 4L}), keepDims: true);
            
            // Normalize
            var normalized = (xReshaped - mean) / (variance + eps).Sqrt();
            
            // Reshape back to [batch, channels, height, width]
            normalized = normalized.Reshape([batchSize, numChannels, height, width]);
            
            // Learnable parameters (gamma and beta)
            var gamma = RetInitSimple.Init([numChannels]).Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            var beta = RetInitZeros.Init([numChannels]).Reshape([Scalar(1L), numChannels, Scalar(1L), Scalar(1L)]);
            
            return normalized * gamma + beta;
        }
    }
    
    [Module]
    public partial class RetUpsampleNearest
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] Scalar<int64> scaleFactor)
        {
            // Use rescale with nearest neighbor interpolation
            // Note: scaleFactor parameter is reserved for future use; currently using fixed 2x upsampling
            // which is the standard for FPN top-down pathway in RetinaNet
            var scales = Vector(new float[] {1.0f, 1.0f, 2.0f, 2.0f}); // 2x upsampling for H and W
            return x.Rescale(scales, mode: ResizeMode.Nearest);
        }
    }

    /// <summary>
    /// Feature Pyramid Network (FPN) for combining multi-scale features
    /// Creates P3, P4, P5, P6, P7 feature pyramids from backbone C3, C4, C5 features
    /// Following PyTorch RetinaNet implementation:
    /// - P3-P5 are built from C3-C5 using top-down pathway with lateral connections
    /// - P6 is built from P5 using 3x3 conv with stride 2
    /// - P7 is built from P6 using 3x3 conv with stride 2 + ReLU
    /// 
    /// NOTE: Temporarily modified to return FPN intermediate layers for debugging.
    /// Returns: p3, p4, p5, p6, p7, p4_1x1, p5_1x1 (7 outputs - respecting tuple limit)
    /// Testing all lateral 1x1 convolutions to see if issue is specific to one layer or systematic.
    /// After debugging, revert to returning only (p3, p4, p5, p6, p7).
    /// </summary>
    [Module]
    public partial class RetFeaturePyramidNetworkFull
    {
        public static (
            Tensor<float32> p3,
            Tensor<float32> p4,
            Tensor<float32> p5,
            Tensor<float32> p6,
            Tensor<float32> p7
        ) Inline(
            Tensor<float32> c3,
            Tensor<float32> c4,
            Tensor<float32> c5,
            [Hyper] Scalar<int64> fpnChannels)
        {
            // 1x1 convolutions to reduce channels to fpnChannels (typically 256)
            var p5_1x1 = RetConv1x1.Model(fpnChannels).Call(c5);
            var p4_1x1 = RetConv1x1.Model(fpnChannels).Call(c4);
            var p3_1x1_orig = RetConv1x1.Model(fpnChannels).Call(c3);
            
            // Top-down pathway with lateral connections
            var p5 = p5_1x1;
            
            // P4 = upsampled P5 + lateral P4
            var p5_upsampled = RetUpsampleNearest.Model(Scalar(2L)).Call(p5);
            var p4 = p4_1x1 + p5_upsampled;
            
            // P3 = upsampled P4 + lateral P3  
            var p4_upsampled = RetUpsampleNearest.Model(Scalar(2L)).Call(p4);
            var p3_1x1 = p3_1x1_orig + p4_upsampled;
            
            // Apply 3x3 convolutions to reduce aliasing (smooth)
            // Order must be p5, p4, p3 to match parameter indexing: RetConv3x3#0→P5, #1→P4, #2→P3
            p5 = RetConv3x3.Model(fpnChannels).Call(p5);
            p4 = RetConv3x3.Model(fpnChannels).Call(p4);
            var p3 = RetConv3x3.Model(fpnChannels).Call(p3_1x1);
            
            // P6 is generated from P5 using 3x3 conv with stride 2 (PyTorch: extra_blocks)
            var p6 = RetConv3x3Stride2.Model(fpnChannels).Call(p5);
            
            // P7 is generated from P6 using ReLU + 3x3 conv with stride 2
            var p7 = RetConv3x3Stride2.Model(fpnChannels).Call(p6.Relu());
            
            // Return intermediate layers for debugging (test all lateral 1x1 convs)
            return (p3, p4, p5, p6, p7);
        }
    }
    
    /// <summary>
    /// Simplified Feature Pyramid Network (P3-P5 only)
    /// For backward compatibility and simpler use cases
    /// </summary>
    [Module]
    public partial class RetFeaturePyramidNetwork
    {
        public static (Tensor<float32> p3, Tensor<float32> p4, Tensor<float32> p5) Inline(
            Tensor<float32> c3,
            Tensor<float32> c4,
            Tensor<float32> c5,
            [Hyper] Scalar<int64> fpnChannels)
        {
            // 1x1 convolutions to reduce channels to fpnChannels (typically 256)
            var p5_1x1 = RetConv1x1.Model(fpnChannels).Call(c5);
            var p4_1x1 = RetConv1x1.Model(fpnChannels).Call(c4);
            var p3_1x1 = RetConv1x1.Model(fpnChannels).Call(c3);
            
            // Top-down pathway with lateral connections
            var p5 = p5_1x1;
            
            // P4 = upsampled P5 + lateral P4
            var p5_upsampled = RetUpsampleNearest.Model(Scalar(2L)).Call(p5);
            var p4 = p4_1x1 + p5_upsampled;
            
            // P3 = upsampled P4 + lateral P3  
            var p4_upsampled = RetUpsampleNearest.Model(Scalar(2L)).Call(p4);
            var p3 = p3_1x1 + p4_upsampled;
            
            // Apply 3x3 convolutions to reduce aliasing
            p3 = RetConv3x3.Model(fpnChannels).Call(p3);
            p4 = RetConv3x3.Model(fpnChannels).Call(p4);
            p5 = RetConv3x3.Model(fpnChannels).Call(p5);
            
            return (p3, p4, p5);
        }
    }

    /// <summary>
    /// RetinaNet classification head with GroupNorm (matches PyTorch implementation)
    /// Predicts object classes for each anchor box
    /// Architecture: 4 × (Conv3x3 + GroupNorm + ReLU) + Conv3x3 + Sigmoid
    /// </summary>
    [Module]
    public partial class RetClassificationHeadWithNorm
    {
        public static Tensor<float32> Inline(
            Tensor<float32> features,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid)
        {
            var x = features;
            
            // 4 conv layers with GroupNorm and ReLU activations (PyTorch style)
            foreach (var ctx in LoopAPI.Iterate(Scalar(4L)))
            {
                x = RetConv3x3.Model(fpnChannels).Call(x);
                x = RetGroupNorm.Model(numGroups, fpnChannels).Call(x);
                x = x.Relu();
            }
            
            // Final classification layer: num_anchors * num_classes outputs
            // Use special bias initialization for focal loss (prior probability = 0.01)
            var numOutputs = numAnchors * numClasses;
            x = RetConv3x3WithClsBias.Model(numOutputs).Call(x);
            
            // Apply sigmoid for focal loss compatibility (PyTorch: classification output is sigmoid)
            var xSigmoid = x.Sigmoid();
            x = applySigmoid.IfElse(xSigmoid, x);
            
            // Reshape and transpose to [batch, height, width, num_anchors, num_classes]
            // PyTorch: [B, num_anchors*num_classes, H, W] -> [B, num_anchors, num_classes, H, W] -> [B, H, W, num_anchors, num_classes]
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var height = RetinaNetHelpers.GetDim(x, Scalar(2L));
            var width = RetinaNetHelpers.GetDim(x, Scalar(3L));
            x = x.Reshape([batchSize, numAnchors, numClasses, height, width]);
            x = x.Transpose(0, 3, 4, 1, 2);  // [B, H, W, num_anchors, num_classes]
            
            return x;
        }
    }
    
    /// <summary>
    /// RetinaNet classification head without normalization (matches torchvision default)
    /// Predicts object classes for each anchor box
    /// Uses shared weights across the 4 intermediate conv layers (via loop)
    /// </summary>
    [Module]
    public partial class RetClassificationHead
    {
        public static Tensor<float32> Inline(
            Tensor<float32> features,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels)
        {
            var x = features;
            
            // 4 conv layers with ReLU activations
            foreach (var ctx in LoopAPI.Iterate(Scalar(4L)))
            {
                x = RetConv3x3.Model(fpnChannels).Call(x);
                x = x.Relu();
            }
            
            // Final classification layer: num_anchors * num_classes outputs
            var numOutputs = numAnchors * numClasses;
            x = RetConv3x3.Model(numOutputs).Call(x);
            
            // Reshape and transpose to [batch, height, width, num_anchors, num_classes]
            // PyTorch: [B, num_anchors*num_classes, H, W] -> [B, num_anchors, num_classes, H, W] -> [B, H, W, num_anchors, num_classes]
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var height = RetinaNetHelpers.GetDim(x, Scalar(2L));
            var width = RetinaNetHelpers.GetDim(x, Scalar(3L));
            x = x.Reshape([batchSize, numAnchors, numClasses, height, width]);
            x = x.Transpose(0, 3, 4, 1, 2);  // [B, H, W, num_anchors, num_classes]
            
            return x;
        }
    }

    /// <summary>
    /// RetinaNet regression head with GroupNorm (matches PyTorch implementation)
    /// Predicts bounding box offsets for each anchor box
    /// Architecture: 4 × (Conv3x3 + GroupNorm + ReLU) + Conv3x3
    /// </summary>
    [Module]
    public partial class RetRegressionHeadWithNorm
    {
        public static Tensor<float32> Inline(
            Tensor<float32> features,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups)
        {

            // var x = features;
            // 
            // // 4 conv layers with ReLU activations
            // foreach (var ctx in LoopAPI.Iterate(Scalar(4L)))
            // {
            //     x = RetConv3x3.Model(fpnChannels).Call(x);
            //     x = x.Relu();
            // }
            // 
            // // Final regression layer: num_anchors * 4 outputs (dx, dy, dw, dh)
            // var numOutputs = numAnchors * 4;
            // x = RetConv3x3.Model(numOutputs).Call(x);
            // 
            // // Reshape and transpose to [batch, height, width, num_anchors, 4]
            // // PyTorch: [B, num_anchors*4, H, W] -> [B, num_anchors, 4, H, W] -> [B, H, W, num_anchors, 4]
            // var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            // var height = RetinaNetHelpers.GetDim(x, Scalar(2L));
            // var width = RetinaNetHelpers.GetDim(x, Scalar(3L));
            // x = x.Reshape([batchSize, numAnchors, Scalar(4L), height, width]);
            // x = x.Transpose(0, 3, 4, 1, 2);  // [B, H, W, num_anchors, 4]
            // 
            // return x;



            var x = features;
            
            // 4 conv layers with GroupNorm and ReLU activations (PyTorch style)
            foreach (var ctx in LoopAPI.Iterate(Scalar(4L)))
            {
                x = RetConv3x3.Model(fpnChannels).Call(x);
                x = RetGroupNorm.Model(numGroups, fpnChannels).Call(x);
                x = x.Relu();
            }
            
            // Final regression layer: num_anchors * 4 outputs (dx, dy, dw, dh)
            var numOutputs = numAnchors * 4;
            x = RetConv3x3.Model(numOutputs).Call(x);
            
            // Reshape and transpose to [batch, height, width, num_anchors, 4]
            // PyTorch: [B, num_anchors*4, H, W] -> [B, num_anchors, 4, H, W] -> [B, H, W, num_anchors, 4]
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var height = RetinaNetHelpers.GetDim(x, Scalar(2L)); 
            var width = RetinaNetHelpers.GetDim(x, Scalar(3L));
            x = x.Reshape([batchSize, numAnchors, Scalar(4L), height, width]);
            x = x.Transpose(0, 3, 4, 1, 2);  // [B, H, W, num_anchors, 4]
            
            return x;
        }
    }
    
    /// <summary>
    /// RetinaNet regression head without normalization (matches torchvision default)
    /// Predicts bounding box offsets for each anchor box
    /// Uses shared weights across the 4 intermediate conv layers (via loop)
    /// </summary>
    [Module]
    public partial class RetRegressionHead
    {
        public static Tensor<float32> Inline(
            Tensor<float32> features,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels)
        {
            var x = features;
            
            // 4 conv layers with ReLU activations
            foreach (var ctx in LoopAPI.Iterate(Scalar(4L)))
            {
                x = RetConv3x3.Model(fpnChannels).Call(x);
                x = x.Relu();
            }
            
            // Final regression layer: num_anchors * 4 outputs (dx, dy, dw, dh)
            var numOutputs = numAnchors * 4;
            x = RetConv3x3.Model(numOutputs).Call(x);
            
            // Reshape and transpose to [batch, height, width, num_anchors, 4]
            // PyTorch: [B, num_anchors*4, H, W] -> [B, num_anchors, 4, H, W] -> [B, H, W, num_anchors, 4]
            var batchSize = RetinaNetHelpers.GetDim(x, Scalar(0L));
            var height = RetinaNetHelpers.GetDim(x, Scalar(2L)); 
            var width = RetinaNetHelpers.GetDim(x, Scalar(3L));
            x = x.Reshape([batchSize, numAnchors, Scalar(4L), height, width]);
            x = x.Transpose(0, 3, 4, 1, 2);  // [B, H, W, num_anchors, 4]
            
            return x;
        }
    }

    /// <summary>
    /// RetinaNet detector with backbone model as input parameter (full version with P3-P7)
    /// 
    /// This is the recommended way to use RetinaNet with Shorokoo backbone models.
    /// The backbone is passed as an input parameter, allowing flexible backbone selection.
    /// 
    /// Backbone model signature: Model&lt;Tensor&lt;float32&gt;, (Tensor&lt;float32&gt;, Tensor&lt;float32&gt;, Tensor&lt;float32&gt;)&gt;
    /// - Input: Image tensor [batch, channels, height, width]
    /// - Output: Tuple of (C3, C4, C5) feature maps
    /// </summary>
    [Module]
    public partial class RetinaNet
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid,
            [Hyper] Model<Tensor<float32>, (Tensor<float32> c3, Tensor<float32> c4, Tensor<float32> c5)> backbone)
        {
            // Extract multi-scale features from the backbone model (ignore intermediate layers for regular inference)
            var (c3, c4, c5) = backbone.Call(inputs);

            // Build full feature pyramid with P6 and P7 (FPN now returns intermediates, but we ignore them here)
            var (p3, p4, p5, p6, p7) = RetFeaturePyramidNetworkFull.Model(fpnChannels).Call(c3, c4, c5);

            // Instantiate shared classification and regression heads
            // Weights are shared across all pyramid levels (P3-P7) as per PyTorch torchvision
            var classificationHead = RetClassificationHeadWithNorm.Model(numClasses, numAnchors, fpnChannels, numGroups, applySigmoid);
            var regressionHead = RetRegressionHeadWithNorm.Model(numAnchors, fpnChannels, numGroups);

            // Apply the same head instance to each pyramid level
            var cls_p3 = classificationHead.Call(p3);
            var cls_p4 = classificationHead.Call(p4);
            var cls_p5 = classificationHead.Call(p5);
            var cls_p6 = classificationHead.Call(p6);
            var cls_p7 = classificationHead.Call(p7);

            var reg_p3 = regressionHead.Call(p3);
            var reg_p4 = regressionHead.Call(p4);
            var reg_p5 = regressionHead.Call(p5);
            var reg_p6 = regressionHead.Call(p6);
            var reg_p7 = regressionHead.Call(p7);

            // Flatten outputs from each pyramid level before concatenation
            // Shape: [batch, height, width, num_anchors, num_classes/4] -> [batch, height*width*num_anchors, num_classes/4]
            var batchSize = RetinaNetHelpers.GetDim(cls_p3, Scalar(0L));
            cls_p3 = cls_p3.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p4 = cls_p4.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p5 = cls_p5.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p6 = cls_p6.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p7 = cls_p7.Reshape([batchSize, Scalar(-1L), numClasses]);
            
            reg_p3 = reg_p3.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p4 = reg_p4.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p5 = reg_p5.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p6 = reg_p6.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p7 = reg_p7.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);

            // Concatenate predictions from all pyramid levels (P3-P7)
            var classifications = cls_p3.Concat(axis: 1L, cls_p4, cls_p5, cls_p6, cls_p7);
            var regressions = reg_p3.Concat(axis: 1L, reg_p4, reg_p5, reg_p6, reg_p7);

            return (classifications, regressions);
        }
    }

    /// <summary>
    /// RetinaNet without normalization in heads (matches torchvision default)
    /// 
    /// This variant uses simple Conv3x3 + ReLU in heads without GroupNorm,
    /// matching torchvision's default RetinaNet architecture exactly.
    /// Used for validation against PyTorch/torchvision implementations.
    /// 
    /// Architecture: Backbone -> FPN (P3-P7) -> Classification/Regression heads
    /// Heads: 4 × (Conv3x3 + ReLU) + Conv3x3 (no normalization)
    /// </summary>
    /// <summary>
    /// NOTE: Temporarily modified to return early backbone intermediate layers for debugging.
    /// Returns: classifications, regressions, stem, layer1_out, layer1_block0, layer2_block0, c3 (7 outputs max)
    /// Testing early backbone layers to trace error amplification origin.
    /// After debugging, revert to returning only (classifications, regressions).
    /// </summary>
    [Module]
    public partial class RetinaNetNoNorm
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Model<Tensor<float32>, (Tensor<float32> c3, Tensor<float32> c4, Tensor<float32> c5)> backbone)
        {
            // Extract multi-scale features from the backbone model (ignore intermediate layers for regular inference)
            var (c3, c4, c5) = backbone.Call(inputs);

            // Build full feature pyramid with P6 and P7
            var (p3, p4, p5, p6, p7) = RetFeaturePyramidNetworkFull.Model(fpnChannels).Call(c3, c4, c5);

            // Instantiate shared classification and regression heads
            // Weights are shared across all pyramid levels (P3-P7) as per PyTorch torchvision
            var classificationHead = RetClassificationHead.Model(numClasses, numAnchors, fpnChannels);
            var regressionHead = RetRegressionHead.Model(numAnchors, fpnChannels);

            // Apply the same head instance to each pyramid level
            var cls_p3 = classificationHead.Call(p3);
            var cls_p4 = classificationHead.Call(p4);
            var cls_p5 = classificationHead.Call(p5);
            var cls_p6 = classificationHead.Call(p6);
            var cls_p7 = classificationHead.Call(p7);

            var reg_p3 = regressionHead.Call(p3);
            var reg_p4 = regressionHead.Call(p4);
            var reg_p5 = regressionHead.Call(p5);
            var reg_p6 = regressionHead.Call(p6);
            var reg_p7 = regressionHead.Call(p7);

            // Flatten outputs from each pyramid level before concatenation
            // Shape: [batch, height, width, num_anchors, num_classes/4] -> [batch, height*width*num_anchors, num_classes/4]
            var batchSize = RetinaNetHelpers.GetDim(cls_p3, Scalar(0L));
            cls_p3 = cls_p3.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p4 = cls_p4.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p5 = cls_p5.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p6 = cls_p6.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p7 = cls_p7.Reshape([batchSize, Scalar(-1L), numClasses]);
            
            reg_p3 = reg_p3.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p4 = reg_p4.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p5 = reg_p5.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p6 = reg_p6.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p7 = reg_p7.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);

            // Concatenate predictions from all pyramid levels (P3-P7)
            var classifications = cls_p3.Concat(axis: 1L, cls_p4, cls_p5, cls_p6, cls_p7);
            var regressions = reg_p3.Concat(axis: 1L, reg_p4, reg_p5, reg_p6, reg_p7);

            return (classifications, regressions);
        }
    }

    /// <summary>
    /// RetinaNet with debug outputs for investigating error amplification
    /// Returns 7 outputs to stay within tuple limitations: 
    /// - classifications, regressions (main outputs)
    /// - stem, layer1_block0, layer1_out, layer2_block0, c3 (early backbone layers)
    /// 
    /// Uses RetResNet50BackboneDebug which unrolls layer1 and layer2 to extract intermediate outputs.
    /// This helps identify where error first appears (between stem at 98% match and c3 at 53% match).
    /// </summary>
    [Module]
    public partial class RetinaNetResNet50NoNorm
    {
        public static (
            Tensor<float32> classifications,
            Tensor<float32> regressions
        ) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            // Extract multi-scale features from the backbone model including early intermediate layers
            // Uses RetResNet50BackboneDebug which unrolls layer1 and layer2
            var (c3, c4, c5) = RetResNet50Backbone.Model(bnMomentum, bnEps).Call(inputs);

            // Build full feature pyramid with P6 and P7
            var (p3, p4, p5, p6, p7) = RetFeaturePyramidNetworkFull.Model(fpnChannels).Call(c3, c4, c5);

            // Instantiate shared classification and regression heads
            var classificationHead = RetClassificationHead.Model(numClasses, numAnchors, fpnChannels);
            var regressionHead = RetRegressionHead.Model(numAnchors, fpnChannels);

            // Apply the same head instance to each pyramid level
            var cls_p3 = classificationHead.Call(p3);
            var cls_p4 = classificationHead.Call(p4);
            var cls_p5 = classificationHead.Call(p5);
            var cls_p6 = classificationHead.Call(p6);
            var cls_p7 = classificationHead.Call(p7);

            var reg_p3 = regressionHead.Call(p3);
            var reg_p4 = regressionHead.Call(p4);
            var reg_p5 = regressionHead.Call(p5);
            var reg_p6 = regressionHead.Call(p6);
            var reg_p7 = regressionHead.Call(p7);

            // Flatten outputs from each pyramid level before concatenation
            var batchSize = RetinaNetHelpers.GetDim(cls_p3, Scalar(0L));
            cls_p3 = cls_p3.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p4 = cls_p4.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p5 = cls_p5.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p6 = cls_p6.Reshape([batchSize, Scalar(-1L), numClasses]);
            cls_p7 = cls_p7.Reshape([batchSize, Scalar(-1L), numClasses]);

            reg_p3 = reg_p3.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p4 = reg_p4.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p5 = reg_p5.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p6 = reg_p6.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);
            reg_p7 = reg_p7.Reshape([batchSize, Scalar(-1L), Scalar(4L)]);

            // Concatenate predictions from all pyramid levels (P3-P7)
            var classifications = cls_p3.Concat(axis: 1L, cls_p4, cls_p5, cls_p6, cls_p7);
            var regressions = reg_p3.Concat(axis: 1L, reg_p4, reg_p5, reg_p6, reg_p7);

            // Return 8 outputs: c3 (last 100% match layer) + next 6 layers + classifications for reference
            return (classifications, regressions);
        }
    }


    /// <summary>
    /// RetinaNet FPN - Debug version
    /// Exposes FPN intermediate outputs: P3, P4, P5, P6, P7
    /// Takes backbone outputs (C3, C4, C5) as input
    /// </summary>
    [Module]
    public partial class RetinaNetFPNDebug
    {
        public static (Tensor<float32> p3, Tensor<float32> p4, Tensor<float32> p5, Tensor<float32> p6, Tensor<float32> p7) Inline(
            Tensor<float32> c3,
            Tensor<float32> c4,
            Tensor<float32> c5,
            [Hyper] Scalar<int64> fpnChannels)
        {
            var (p3, p4, p5, p6, p7) = RetFeaturePyramidNetworkFull.Model(fpnChannels).Call(c3, c4, c5);
            return (p3, p4, p5, p6, p7);
        }
    }

    /// <summary>
    /// RetinaNet FPN and Head Debug - P3-P5 only
    /// Takes C3-C5 inputs, returns FPN outputs (P3-P5) and per-level head raw outputs
    /// This matches PyTorch FPN which only returns 3 levels
    /// </summary>
    [Module]
    public partial class RetinaNetFPNHeadsDebug
    {
        public static (
            Tensor<float32> p3,
            Tensor<float32> p4,
            Tensor<float32> p5,
            Tensor<float32> cls_p3_raw,
            Tensor<float32> cls_p4_raw,
            Tensor<float32> cls_p5_raw,
            Tensor<float32> reg_p3_raw
        ) Inline(
            Tensor<float32> c3,
            Tensor<float32> c4,
            Tensor<float32> c5,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels)
        {
            // Build FPN P3-P5 (matching PyTorch FPN output)
            var (p3, p4, p5) = RetFeaturePyramidNetwork.Model(fpnChannels).Call(c3, c4, c5);

            // Instantiate shared classification and regression heads
            var classificationHead = RetClassificationHead.Model(numClasses, numAnchors, fpnChannels);
            var regressionHead = RetRegressionHead.Model(numAnchors, fpnChannels);

            // Apply heads to each pyramid level - keep raw output before reshape
            var cls_p3_raw = classificationHead.Call(p3);
            var cls_p4_raw = classificationHead.Call(p4);
            var cls_p5_raw = classificationHead.Call(p5);

            var reg_p3_raw = regressionHead.Call(p3);
            var reg_p4_raw = regressionHead.Call(p4);
            var reg_p5_raw = regressionHead.Call(p5);

            // Return max 7 outputs (C# tuple limitation)
            // Return P3-P5, cls_p3-p5_raw, and reg_p3_raw only
            return (p3, p4, p5, cls_p3_raw, cls_p4_raw, cls_p5_raw, reg_p3_raw);
        }
    }

    /// Main RetinaNet detector with ResNet50 backbone
    /// </summary>
    [Module]
    public partial class RetinaNetResNet50
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            var resNetBackboneModel = RetResNet50Backbone.Model(bnMomentum, bnEps);
            return RetinaNet.Model(numClasses, numAnchors, fpnChannels, numGroups, applySigmoid, resNetBackboneModel).Call(inputs);
        }
    }

    /// <summary>
    /// Main RetinaNet detector with ResNet18 backbone
    /// Uses full FPN with P3-P7 levels and GroupNorm-based classification/regression heads
    /// </summary>
    [Module]
    public partial class RetinaNetResNet18
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            var resNetBackboneModel = RetResNet18Backbone.Model(bnMomentum, bnEps);
            return RetinaNet.Model(numClasses, numAnchors, fpnChannels, numGroups, applySigmoid, resNetBackboneModel).Call(inputs);
        }
    }

    /// <summary>
    /// Main RetinaNet detector with ResNet101 backbone
    /// </summary>
    [Module]
    public partial class RetinaNetResNet101
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid,
            [Hyper] Scalar<float32> bnMomentum,
            [Hyper] Scalar<float32> bnEps)
        {
            var resNetBackboneModel = ResNet101Backbone.Model(bnMomentum, bnEps);
            return RetinaNet.Model(numClasses, numAnchors, fpnChannels, numGroups, applySigmoid, resNetBackboneModel).Call(inputs);
        }
    }

    /// <summary>
    /// Main RetinaNet detector with ViT backbone (PyTorch compatible - full version with P3-P7)
    /// Uses GroupNorm in heads and sigmoid activation for focal loss compatibility
    /// </summary>
    [Module]
    public partial class RetinaNetViTSmallFull
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid)
        {
            var vitBackboneModel = RetViTBackbone.Model(Scalar(true));
            return RetinaNet.Model(numClasses, numAnchors, fpnChannels, numGroups, applySigmoid, vitBackboneModel).Call(inputs);
        }
    }

    /// <summary>
    /// Main RetinaNet detector with ViT backbone (Large variant)
    /// </summary>
    [Module]
    public partial class RetinaNetViT
    {
        public static (Tensor<float32> classifications, Tensor<float32> regressions) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<int64> numAnchors,
            [Hyper] Scalar<int64> fpnChannels,
            [Hyper] Scalar<int64> numGroups,
            [Hyper] Scalar<bit> applySigmoid)
        {
            var vitBackboneModel = RetViTBackbone.Model(Scalar(false));
            return RetinaNet.Model(numClasses, numAnchors, fpnChannels, numGroups, applySigmoid, vitBackboneModel).Call(inputs);
        }
    }
}
