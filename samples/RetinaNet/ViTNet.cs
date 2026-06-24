using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;
using static Shorokoo.NN;

namespace RetinaNet.Models
{
    /// <summary>
    /// Vision Transformer (ViT) Image Classifier
    /// 
    /// A standard ViT implementation for image classification following the
    /// original "An Image is Worth 16x16 Words" paper architecture.
    /// 
    /// Supported configurations:
    /// - ViT-Tiny: embed_dim=192, heads=3, layers=12, mlp_dim=768 (fastest for CreateConcreteArchitecture)
    /// - ViT-Small: embed_dim=384, heads=6, layers=12, mlp_dim=1536
    /// - ViT-Base: embed_dim=768, heads=12, layers=12, mlp_dim=3072
    /// 
    /// Default input: 224x224 RGB images with 16x16 patches (196 patches + 1 CLS token)
    /// 
    /// For pretrained weights, use PyTorch torchvision:
    ///   models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    /// </summary>
    
    // Helper functions (non-module)
    public static class ViTNetHelpers
    {
        public static Scalar<int64> GetDim(Tensor<float32> x, long dimIndex) => x.DimTensor(dimIndex);
    }

    #region Trainable Parameter Initializers (New Pattern)

    /// <summary>
    /// Trainable parameter initializer using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class ViTInitSimple
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 1.0f));
        }
    }

    #endregion

    /// <summary>
    /// Layer normalization for transformers
    /// </summary>
    [Module]
    public partial class ViTLayerNorm
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> normalizedShape,
            [Hyper] Scalar<float32> eps)
        {
            // Layer norm parameters: scale (gamma) and bias (beta)
            var gamma = ViTInitSimple.Init([normalizedShape]).Vec();
            var beta = ViTInitSimple.Init([normalizedShape]).Vec();

            // Compute mean and variance over the last dimension
            var mean = x.Reduce(ReduceKind.Mean, axes: Vector(-1L), keepDims: true);
            var variance = ((x - mean).Pow(Scalar(2.0f))).Reduce(ReduceKind.Mean, axes: Vector(-1L), keepDims: true);

            // Normalize: (x - mean) / sqrt(var + eps)
            var normalized = (x - mean) / (variance + eps).Sqrt();

            // Apply scale and shift
            return normalized * gamma + beta;
        }
    }

    /// <summary>
    /// Patch embedding: Projects image patches to embedding dimension using convolution
    /// Input: [batch_size, 3, 224, 224]
    /// Output: [batch_size, num_patches, embed_dim] where num_patches = (224/16)^2 = 196
    /// </summary>
    [Module]
    public partial class ViTPatchEmbed
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> patchSize)
        {
            var batchSize = ViTNetHelpers.GetDim(x, 0);
            var inChannels = ViTNetHelpers.GetDim(x, 1);
            var height = ViTNetHelpers.GetDim(x, 2);
            var width = ViTNetHelpers.GetDim(x, 3);

            // Convolution weight: [embed_dim, in_channels, patch_size, patch_size]
            Vector<int64> wShape = [embedDim, inChannels, patchSize, patchSize];
            var w = ViTInitSimple.Init(wShape);

            // Bias for projection
            var b = ViTInitSimple.Init([embedDim]).Vec();

            // Apply convolution with stride = patch_size (non-overlapping patches)
            // Output shape: [batch_size, embed_dim, H/patch_size, W/patch_size]
            var patches = Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L],
                group: 1L,
                kernelShape: [16L, 16L],
                pads: [0L, 0L, 0L, 0L],
                strides: [16L, 16L]);

            // Compute number of patches
            var numPatchesH = height / patchSize;
            var numPatchesW = width / patchSize;
            var numPatches = numPatchesH * numPatchesW;

            // Reshape from [batch, embed_dim, H', W'] to [batch, embed_dim, num_patches]
            var flattened = patches.Reshape([batchSize, embedDim, numPatches]);

            // Transpose to [batch, num_patches, embed_dim]
            return flattened.Transpose([0L, 2L, 1L]);
        }
    }

    /// <summary>
    /// Linear projection layer
    /// </summary>
    [Module]
    public partial class ViTLinear
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> inFeatures,
            [Hyper] Scalar<int64> outFeatures,
            [Hyper] Scalar<bit> useBias)
        {
            // Weight: [in_features, out_features]
            var w = ViTInitSimple.Init([inFeatures, outFeatures]);

            // MatMul: [batch, seq, in_features] @ [in_features, out_features] -> [batch, seq, out_features]
            var y = x.MatMul(w);

            // Optional bias
            var b = ViTInitSimple.Init([outFeatures]).Vec();
            var biasedY = y + b;
            y = useBias.IfElse(biasedY, y);

            return y;
        }
    }

    /// <summary>
    /// Multi-head self-attention
    /// Input: [batch_size, seq_len, embed_dim]
    /// Output: [batch_size, seq_len, embed_dim]
    /// Uses direct weight initialization to avoid nested module calls
    /// </summary>
    [Module]
    public partial class ViTAttention
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numHeads)
        {
            var batchSize = ViTNetHelpers.GetDim(x, 0);
            var seqLen = ViTNetHelpers.GetDim(x, 1);
            var headDim = embedDim / numHeads;

            // QKV weights: project to 3 * embed_dim for Q, K, V combined
            var qkvWeight = ViTInitSimple.Init([embedDim, Scalar(3L) * embedDim]);
            var qkvBias = ViTInitSimple.Init([Scalar(3L) * embedDim]).Vec();

            // QKV projection
            var qkv = x.MatMul(qkvWeight) + qkvBias;

            // Reshape to [batch, seq, 3, num_heads, head_dim]
            qkv = qkv.Reshape([batchSize, seqLen, Scalar(3L), numHeads, headDim]);

            // Transpose to [3, batch, num_heads, seq, head_dim]
            qkv = qkv.Transpose([2L, 0L, 3L, 1L, 4L]);

            // Split Q, K, V - using slice operations
            var q = qkv.Slice(start: Vector(0L, 0L, 0L, 0L, 0L), end: Vector(1L, (long)int.MaxValue, (long)int.MaxValue, (long)int.MaxValue, (long)int.MaxValue)).Squeeze(Vector(0L));
            var k = qkv.Slice(start: Vector(1L, 0L, 0L, 0L, 0L), end: Vector(2L, (long)int.MaxValue, (long)int.MaxValue, (long)int.MaxValue, (long)int.MaxValue)).Squeeze(Vector(0L));
            var v = qkv.Slice(start: Vector(2L, 0L, 0L, 0L, 0L), end: Vector(3L, (long)int.MaxValue, (long)int.MaxValue, (long)int.MaxValue, (long)int.MaxValue)).Squeeze(Vector(0L));

            // Scaled dot-product attention
            // Q: [batch, heads, seq, head_dim]
            // K^T: [batch, heads, head_dim, seq]
            var kT = k.Transpose([0L, 1L, 3L, 2L]);
            var scale = Scalar(1.0f) / headDim.Cast<float32>().Sqrt();
            var scores = q.MatMul(kT) * scale;

            // Softmax over last dimension (keys)
            var attnWeights = scores.Softmax(axis: -1L);

            // Apply attention to values: [batch, heads, seq, head_dim]
            var attnOutput = attnWeights.MatMul(v);

            // Transpose back: [batch, seq, heads, head_dim]
            attnOutput = attnOutput.Transpose([0L, 2L, 1L, 3L]);

            // Reshape to [batch, seq, embed_dim]
            attnOutput = attnOutput.Reshape([batchSize, seqLen, embedDim]);

            // Output projection (using direct weights instead of LinearModel)
            var projWeight = ViTInitSimple.Init([embedDim, embedDim]);
            var projBias = ViTInitSimple.Init([embedDim]).Vec();
            return attnOutput.MatMul(projWeight) + projBias;
        }
    }

    /// <summary>
    /// MLP block with GELU activation
    /// Input: [batch_size, seq_len, embed_dim]
    /// Output: [batch_size, seq_len, embed_dim]
    /// Uses direct weight initialization to avoid nested module calls
    /// </summary>
    [Module]
    public partial class ViTMlp
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> hiddenDim)
        {
            // First linear: embed_dim -> hidden_dim
            var w1 = ViTInitSimple.Init([embedDim, hiddenDim]);
            var b1 = ViTInitSimple.Init([hiddenDim]).Vec();
            x = x.MatMul(w1) + b1;

            // GELU activation - use tanh approximation which is commonly used in ViT
            // PyTorch nn.GELU uses the tanh approximation by default in newer versions
            x = x.Gelu(GeluApproximate.Tanh);

            // Second linear: hidden_dim -> embed_dim
            var w2 = ViTInitSimple.Init([hiddenDim, embedDim]);
            var b2 = ViTInitSimple.Init([embedDim]).Vec();
            x = x.MatMul(w2) + b2;

            return x;
        }
    }

    /// <summary>
    /// Transformer encoder block
    /// Pre-norm architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    /// </summary>
    [Module]
    public partial class ViTTransformerBlock
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numHeads,
            [Hyper] Scalar<int64> mlpDim,
            [Hyper] Scalar<float32> layerNormEps)
        {
            // Self-attention with pre-norm and residual
            var norm1 = ViTLayerNorm.Model(embedDim, layerNormEps).Call(x);
            var attn = ViTAttention.Model(embedDim, numHeads).Call(norm1);
            x = x + attn;

            // MLP with pre-norm and residual
            var norm2 = ViTLayerNorm.Model(embedDim, layerNormEps).Call(x);
            var mlpOut = ViTMlp.Model(embedDim, mlpDim).Call(norm2);
            x = x + mlpOut;

            return x;
        }
    }

    /// <summary>
    /// Classification head: Extract CLS token and project to num_classes
    /// Uses direct weight initialization to avoid nested module calls
    /// </summary>
    [Module]
    public partial class ViTClassHead
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            // x shape: [batch, seq_len, embed_dim]
            // Extract CLS token (first token): [batch, embed_dim]
            var batchSize = ViTNetHelpers.GetDim(x, 0);
            var cls = x.Slice(start: Vector(0L, 0L, 0L), end: Vector(int.MaxValue, 1L, int.MaxValue));
            cls = cls.Reshape([batchSize, embedDim]);

            // Linear classifier using direct weights
            var w = ViTInitSimple.Init([embedDim, numClasses]);
            var b = ViTInitSimple.Init([numClasses]).Vec();
            var logits = cls.MatMul(w) + b;

            // Optional softmax
            var softmaxed = logits.Softmax();
            return applySoftmax.IfElse(softmaxed, logits);
        }
    }

    /// <summary>
    /// Vision Transformer Tiny (ViT-Ti/16)
    /// Configuration: embed_dim=192, heads=3, layers=12, mlp_dim=768
    /// Input: [batch, 3, 224, 224]
    /// Output: [batch, num_classes]
    /// 
    /// This is the smallest standard ViT with pretrained ImageNet weights available.
    /// PyTorch equivalent: timm.create_model('vit_tiny_patch16_224', pretrained=True)
    /// </summary>
    [Module]
    public partial class ViTTiny
    {
        public static Tensor<float32> Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            // ViT-Tiny configuration
            var embedDim = Scalar(192L);
            var numHeads = Scalar(3L);
            var numLayers = Scalar(12L);
            var mlpDim = Scalar(768L);  // 4 * embed_dim
            var patchSize = Scalar(16L);
            var layerNormEps = Scalar(1e-6f);
            
            // Fixed sequence length for 224x224 images with 16x16 patches
            // num_patches = (224/16)^2 = 196, plus 1 CLS token = 197
            var fixedSeqLen = Scalar(197L);

            // Patch embedding: [batch, 3, 224, 224] -> [batch, 196, 192]
            var x = ViTPatchEmbed.Model(embedDim, patchSize).Call(inputs);

            // Add CLS token: prepend learnable token
            // CLS token shape: [1, 1, embed_dim]
            var clsToken = ViTInitSimple.Init([Scalar(1L), Scalar(1L), embedDim]);
            // Reshape for proper broadcasting - will work with batch size 1
            x = (Tensor<float32>)NN.Concat([clsToken, x], axis: 1L);

            // Add positional embedding: [1, 197, 192] - using fixed sequence length
            var posEmbed = ViTInitSimple.Init([Scalar(1L), fixedSeqLen, embedDim]);
            x = x + posEmbed;

            // Transformer encoder blocks
            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = ViTTransformerBlock.Model(embedDim, numHeads, mlpDim, layerNormEps).Call(x);
            }

            // Final layer norm
            x = ViTLayerNorm.Model(embedDim, layerNormEps).Call(x);

            // Classification head
            return ViTClassHead.Model(embedDim, numClasses, applySoftmax).Call(x);
        }
    }

    /// <summary>
    /// Vision Transformer Small (ViT-S/16)
    /// Configuration: embed_dim=384, heads=6, layers=12, mlp_dim=1536
    /// </summary>
    [Module]
    public partial class ViTSmall
    {
        public static Tensor<float32> Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            var embedDim = Scalar(384L);
            var numHeads = Scalar(6L);
            var numLayers = Scalar(12L);
            var mlpDim = Scalar(1536L);
            var patchSize = Scalar(16L);
            var layerNormEps = Scalar(1e-6f);

            var x = ViTPatchEmbed.Model(embedDim, patchSize).Call(inputs);

            var clsToken = ViTInitSimple.Init([Scalar(1L), Scalar(1L), embedDim]);
            var clsTiled = clsToken.Tile(Vector(new long[] { 1L, 1L, 1L }));
            x = (Tensor<float32>)NN.Concat([clsTiled, x], axis: 1L);

            var seqLen = ViTNetHelpers.GetDim(x, 1);
            var posEmbed = ViTInitSimple.Init([Scalar(1L), seqLen, embedDim]);
            x = x + posEmbed;

            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = ViTTransformerBlock.Model(embedDim, numHeads, mlpDim, layerNormEps).Call(x);
            }

            x = ViTLayerNorm.Model(embedDim, layerNormEps).Call(x);
            return ViTClassHead.Model(embedDim, numClasses, applySoftmax).Call(x);
        }
    }

    /// <summary>
    /// Vision Transformer Base (ViT-B/16)
    /// Configuration: embed_dim=768, heads=12, layers=12, mlp_dim=3072
    /// 
    /// This is the standard ViT-B/16 from the original paper.
    /// PyTorch equivalent: torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    /// </summary>
    [Module]
    public partial class ViTBase
    {
        public static Tensor<float32> Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            var embedDim = Scalar(768L);
            var numHeads = Scalar(12L);
            var numLayers = Scalar(12L);
            var mlpDim = Scalar(3072L);
            var patchSize = Scalar(16L);
            var layerNormEps = Scalar(1e-6f);

            var x = ViTPatchEmbed.Model(embedDim, patchSize).Call(inputs);

            var clsToken = ViTInitSimple.Init([Scalar(1L), Scalar(1L), embedDim]);
            var clsTiled = clsToken.Tile(Vector(new long[] { 1L, 1L, 1L }));
            x = (Tensor<float32>)NN.Concat([clsTiled, x], axis: 1L);

            var seqLen = ViTNetHelpers.GetDim(x, 1);
            var posEmbed = ViTInitSimple.Init([Scalar(1L), seqLen, embedDim]);
            x = x + posEmbed;

            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = ViTTransformerBlock.Model(embedDim, numHeads, mlpDim, layerNormEps).Call(x);
            }

            x = ViTLayerNorm.Model(embedDim, layerNormEps).Call(x);
            return ViTClassHead.Model(embedDim, numClasses, applySoftmax).Call(x);
        }
    }

    /// <summary>
    /// Debug version of ViT-Tiny that outputs intermediate tensors
    /// Useful for comparing with PyTorch intermediate layer outputs
    /// </summary>
    [Module]
    public partial class ViTTinyDebug
    {
        public static (Tensor<float32> output, Tensor<float32> afterPatchEmbed, Tensor<float32> afterPosEmbed, Tensor<float32> afterBlocks, Tensor<float32> afterNorm) Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            var embedDim = Scalar(192L);
            var numHeads = Scalar(3L);
            var numLayers = Scalar(12L);
            var mlpDim = Scalar(768L);
            var patchSize = Scalar(16L);
            var layerNormEps = Scalar(1e-6f);

            var x = ViTPatchEmbed.Model(embedDim, patchSize).Call(inputs);
            var afterPatchEmbed = x;

            var clsToken = ViTInitSimple.Init([Scalar(1L), Scalar(1L), embedDim]);
            var clsTiled = clsToken.Tile(Vector(new long[] { 1L, 1L, 1L }));
            x = (Tensor<float32>)NN.Concat([clsTiled, x], axis: 1L);

            var seqLen = ViTNetHelpers.GetDim(x, 1);
            var posEmbed = ViTInitSimple.Init([Scalar(1L), seqLen, embedDim]);
            x = x + posEmbed;
            var afterPosEmbed = x;

            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = ViTTransformerBlock.Model(embedDim, numHeads, mlpDim, layerNormEps).Call(x);
            }
            var afterBlocks = x;

            x = ViTLayerNorm.Model(embedDim, layerNormEps).Call(x);
            var afterNorm = x;

            var output = ViTClassHead.Model(embedDim, numClasses, applySoftmax).Call(x);
            return (output, afterPatchEmbed, afterPosEmbed, afterBlocks, afterNorm);
        }
    }
}
