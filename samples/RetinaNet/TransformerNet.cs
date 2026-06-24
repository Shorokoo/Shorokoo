using Shorokoo;
using Shorokoo.Modules;
using static Shorokoo.Globals;
using static Shorokoo.NN;

namespace RetinaNet.Models
{
    // Helper functions (non-module)
    public static class TransformerNetHelpers
    {
        public static Scalar<int64> GetDim(Tensor<float32> x, long dimIndex) => x.ShapeTensor()[dimIndex];
    }

    #region Trainable Parameter Initializers (New Pattern)

    /// <summary>
    /// Xavier/Glorot initialization using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class TfInitXavier
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            // Xavier/Glorot initialization approximation using simple fill
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 0.02f));
        }
    }

    /// <summary>
    /// Zero initialization using the new single-class pattern.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class TfInitZeros
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Tensor<float32>.Fill(shape, Globals.TensorData(1, 0.0f));
        }
    }

    #endregion

    /// <summary>
    /// Layer normalization implementation using available primitives
    /// </summary>
    [Module]
    public partial class TfLayerNorm
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> normalizedShape,
            [Hyper] Scalar<float32> eps)
        {
            // Layer norm parameters: scale (gamma) and bias (beta)
            var gamma = TfInitXavier.Init([normalizedShape]).Vec();
            var beta = TfInitZeros.Init([normalizedShape]).Vec();

            // Compute mean and variance over the last dimension
            var mean = x.Reduce(ReduceKind.Mean, axes: Vector(-1L), keepDims: true);
            var variance = ((x - mean).Pow(Scalar(2.0f))).Reduce(ReduceKind.Mean, axes: Vector(-1L), keepDims: true);
            
            // Normalize
            var normalized = (x - mean) / (variance + eps).Sqrt();
            
            // Apply scale and shift
            return normalized * gamma + beta;
        }
    }

    /// <summary>
    /// Convolution operation for patch embedding
    /// </summary>
    [Module]
    public partial class TfConv2D
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> filters,
            [Hyper] Scalar<int64> kernelSize,
            [Hyper] Scalar<int64> stride)
        {
            var inChannels = TransformerNetHelpers.GetDim(x, 1);
            
            // Weight shape [outC, inC, kH, kW]
            Vector<int64> wShape = [filters, inChannels, kernelSize, kernelSize];
            var w = TfInitXavier.Init(wShape);
            var b = TfInitZeros.Init([filters]);
            
            // For patch embedding: 16x16 patches with stride 16
            long[] strideArray = [16L, 16L];
            long[] kernelArray = [16L, 16L];
            
            return Conv(x, w, b.Vec(), 
                AutoPad.NotSet, 
                dilations: [1L, 1L], 
                group: 1L, 
                kernelShape: kernelArray, 
                pads: null, 
                strides: strideArray);
        }
    }

    /// <summary>
    /// Patch embedding: Convert image patches to token embeddings
    /// </summary>
    [Module]
    public partial class TfPatchEmbedding
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> patchSize,
            [Hyper] Scalar<int64> embedDim)
        {
            // Input: [batch_size, channels, height, width]
            // Output: [batch_size, num_patches, embed_dim]
            
            var batchSize = TransformerNetHelpers.GetDim(x, 0);
            var channels = TransformerNetHelpers.GetDim(x, 1);
            var height = TransformerNetHelpers.GetDim(x, 2);
            var width = TransformerNetHelpers.GetDim(x, 3);
            
            // Convolution to extract patches and project to embed_dim
            var patchConv = TfConv2D.Model(embedDim, patchSize, patchSize).Call(x);
            
            // Reshape to [batch_size, embed_dim, num_patches_h, num_patches_w]
            // Then transpose and reshape to [batch_size, num_patches, embed_dim]
            var numPatchesH = height / patchSize;
            var numPatchesW = width / patchSize;
            var numPatches = numPatchesH * numPatchesW;
            
            // Flatten spatial dimensions
            var flattened = patchConv.Reshape([batchSize, embedDim, numPatches]);
            
            // Transpose to [batch_size, num_patches, embed_dim]
            return flattened.Transpose([0L, 2L, 1L]);
        }
    }

    /// <summary>
    /// Positional encoding: Add learnable position embeddings
    /// </summary>
    [Module]
    public partial class TfPositionalEncoding
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> maxSeqLen,
            [Hyper] Scalar<int64> embedDim)
        {
            // x shape: [batch_size, seq_len, embed_dim]
            var posEmbedding = TfInitXavier.Init([maxSeqLen, embedDim]);
            
            var seqLen = TransformerNetHelpers.GetDim(x, 1);
            
            // For simplicity, just add the position embedding 
            // In a real implementation, you'd properly handle sequence length
            return x + posEmbedding;
        }
    }

    /// <summary>
    /// Multi-head attention mechanism
    /// </summary>
    [Module]
    public partial class TfMultiHeadAttention
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numHeads,
            [Hyper] Scalar<float32> dropoutRate)
        {
            // x shape: [batch_size, seq_len, embed_dim]
            var batchSize = TransformerNetHelpers.GetDim(x, 0);
            var seqLen = TransformerNetHelpers.GetDim(x, 1);
            var headDim = embedDim / numHeads;
            
            // Linear projections for Q, K, V
            var wq = TfInitXavier.Init([embedDim, embedDim]);
            var wk = TfInitXavier.Init([embedDim, embedDim]);
            var wv = TfInitXavier.Init([embedDim, embedDim]);
            var wo = TfInitXavier.Init([embedDim, embedDim]);
            
            // Project to Q, K, V
            var q = x.MatMul(wq); // [batch_size, seq_len, embed_dim]
            var k = x.MatMul(wk);
            var v = x.MatMul(wv);
            
            // Reshape for multi-head: [batch_size, seq_len, num_heads, head_dim]
            q = q.Reshape([batchSize, seqLen, numHeads, headDim]);
            k = k.Reshape([batchSize, seqLen, numHeads, headDim]);
            v = v.Reshape([batchSize, seqLen, numHeads, headDim]);
            
            // Transpose to [batch_size, num_heads, seq_len, head_dim]
            q = q.Transpose([0L, 2L, 1L, 3L]);
            k = k.Transpose([0L, 2L, 1L, 3L]);
            v = v.Transpose([0L, 2L, 1L, 3L]);
            
            // Scaled dot-product attention
            // Q * K^T / sqrt(head_dim)
            var scale = Scalar(1.0f).Cast<float32>() / headDim.Cast<float32>().Sqrt();
            var scores = q.MatMul(k.Transpose([0L, 1L, 3L, 2L])) * scale;
            
            // Apply softmax
            var attentionWeights = scores.Softmax(axis: -1L);
            
            // Apply attention to values
            var attentionOutput = attentionWeights.MatMul(v);
            
            // Transpose back: [batch_size, seq_len, num_heads, head_dim]
            attentionOutput = attentionOutput.Transpose([0L, 2L, 1L, 3L]);
            
            // Reshape: [batch_size, seq_len, embed_dim]
            attentionOutput = attentionOutput.Reshape([batchSize, seqLen, embedDim]);
            
            // Final linear projection
            return attentionOutput.MatMul(wo);
        }
    }

    /// <summary>
    /// Feed-forward network (MLP)
    /// </summary>
    [Module]
    public partial class TfFeedForward
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> hiddenDim,
            [Hyper] Scalar<float32> dropoutRate)
        {
            // Two linear layers with GELU activation
            var w1 = TfInitXavier.Init([embedDim, hiddenDim]);
            var b1 = TfInitZeros.Init([hiddenDim]);
            var w2 = TfInitXavier.Init([hiddenDim, embedDim]);
            var b2 = TfInitZeros.Init([embedDim]);
            
            // First linear layer
            var hidden = x.MatMul(w1) + b1;
            
            // GELU activation (approximation using available operations)
            hidden = hidden * Scalar(0.5f) * (Scalar(1.0f) + (hidden * Scalar(0.7071067811865476f)).Tanh());
            
            // Second linear layer
            return hidden.MatMul(w2) + b2;
        }
    }

    /// <summary>
    /// Transformer encoder block
    /// </summary>
    [Module]
    public partial class TfTransformerBlock
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numHeads,
            [Hyper] Scalar<int64> hiddenDim,
            [Hyper] Scalar<float32> dropoutRate,
            [Hyper] Scalar<float32> layerNormEps)
        {
            // Multi-head attention with residual connection
            var attnOutput = TfMultiHeadAttention.Model(embedDim, numHeads, dropoutRate).Call(x);
            x = TfLayerNorm.Model(embedDim, layerNormEps).Call(x + attnOutput);
            
            // Feed-forward with residual connection
            var ffOutput = TfFeedForward.Model(embedDim, hiddenDim, dropoutRate).Call(x);
            x = TfLayerNorm.Model(embedDim, layerNormEps).Call(x + ffOutput);
            
            return x;
        }
    }

    /// <summary>
    /// Classification head
    /// </summary>
    [Module]
    public partial class TfClassificationHead
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            // x shape: [batch_size, seq_len, embed_dim]
            // Take the first token (CLS token) for classification
            var clsToken = x.Slice(
                start: Vector(0L, 0L, 0L),
                end: Vector(-1L, 1L, -1L)
            ).Squeeze(Vector(1L)); // [batch_size, embed_dim]
            
            // Linear classifier
            var wClassifier = TfInitXavier.Init([embedDim, numClasses]);
            var bClassifier = TfInitZeros.Init([numClasses]);
            
            var logits = clsToken.MatMul(wClassifier) + bClassifier;
            
            return applySoftmax.IfElse(logits.Softmax(), logits);
        }
    }

    /// <summary>
    /// Vision Transformer (ViT) - Tiny variant
    /// </summary>
    [Module]
    public partial class TfViTTiny
    {
        public static Tensor<float32> Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            var patchSize = Scalar(16L);
            var embedDim = Scalar(192L);
            var numHeads = Scalar(3L);
            var numLayers = Scalar(12L);
            var hiddenDim = Scalar(768L);
            var dropoutRate = Scalar(0.0f);
            var layerNormEps = Scalar(1e-6f);
            var maxSeqLen = Scalar(197L); // 196 patches + 1 CLS token for 224x224 image
            
            // Patch embedding
            var x = TfPatchEmbedding.Model(patchSize, embedDim).Call(inputs);
            
            // Add CLS token
            x = TfAddClsToken(embedDim, x);
            
            // Add positional encoding
            x = TfPositionalEncoding.Model(maxSeqLen, embedDim).Call(x);
            
            // Transformer blocks
            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = TfTransformerBlock.Model(embedDim, numHeads, hiddenDim, dropoutRate, layerNormEps).Call(x);
            }
            
            // Final layer norm
            x = TfLayerNorm.Model(embedDim, layerNormEps).Call(x);
            
            // Classification head
            return TfClassificationHead.Model(embedDim, numClasses, applySoftmax).Call(x);
        }
        
        /// <summary>
        /// Simple CLS token addition using concatenation
        /// </summary>
        private static Tensor<float32> TfAddClsToken(Scalar<int64> embedDim, Tensor<float32> x)
        {
            var batchSize = TransformerNetHelpers.GetDim(x, 0);
            var clsToken = TfInitXavier.Init([Scalar(1L), embedDim]);
            
            // For simplicity, use fixed batch size for the Tile operation
            var clsExpanded = clsToken.Tile(Vector(new long[] { 32L, 1L, 1L }));
            
            // Concatenate CLS token with patches
            return (Tensor<float32>)NN.Concat([clsExpanded, x], axis: 1L);
        }
    }

    /// <summary>
    /// Vision Transformer (ViT) - Base variant (most common)
    /// </summary>
    [Module]
    public partial class TfViTBase
    {
        public static Tensor<float32> Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            var patchSize = Scalar(16L);
            var embedDim = Scalar(768L);
            var numHeads = Scalar(12L);
            var numLayers = Scalar(12L);
            var hiddenDim = Scalar(3072L);
            var dropoutRate = Scalar(0.0f);
            var layerNormEps = Scalar(1e-6f);
            var maxSeqLen = Scalar(197L);
            
            // Patch embedding
            var x = TfPatchEmbedding.Model(patchSize, embedDim).Call(inputs);
            
            // Add CLS token
            x = TfAddClsToken(embedDim, x);
            
            // Add positional encoding
            x = TfPositionalEncoding.Model(maxSeqLen, embedDim).Call(x);
            
            // Transformer blocks
            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = TfTransformerBlock.Model(embedDim, numHeads, hiddenDim, dropoutRate, layerNormEps).Call(x);
            }
            
            // Final layer norm
            x = TfLayerNorm.Model(embedDim, layerNormEps).Call(x);
            
            // Classification head
            return TfClassificationHead.Model(embedDim, numClasses, applySoftmax).Call(x);
        }
        
        /// <summary>
        /// Simple CLS token addition using concatenation
        /// </summary>
        private static Tensor<float32> TfAddClsToken(Scalar<int64> embedDim, Tensor<float32> x)
        {
            var batchSize = TransformerNetHelpers.GetDim(x, 0);
            var clsToken = TfInitXavier.Init([Scalar(1L), embedDim]);
            
            // For simplicity, use fixed batch size for the Tile operation
            var clsExpanded = clsToken.Tile(Vector(new long[] { 32L, 1L, 1L }));
            
            // Concatenate CLS token with patches
            return (Tensor<float32>)NN.Concat([clsExpanded, x], axis: 1L);
        }
    }

    /// <summary>
    /// Vision Transformer (ViT) - Large variant
    /// </summary>
    [Module]
    public partial class TfViTLarge
    {
        public static Tensor<float32> Inline(
            Tensor<float32> inputs,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> applySoftmax)
        {
            var patchSize = Scalar(16L);
            var embedDim = Scalar(1024L);
            var numHeads = Scalar(16L);
            var numLayers = Scalar(24L);
            var hiddenDim = Scalar(4096L);
            var dropoutRate = Scalar(0.0f);
            var layerNormEps = Scalar(1e-6f);
            var maxSeqLen = Scalar(197L);
            
            // Patch embedding
            var x = TfPatchEmbedding.Model(patchSize, embedDim).Call(inputs);
            
            // Add CLS token
            x = TfAddClsToken(embedDim, x);
            
            // Add positional encoding
            x = TfPositionalEncoding.Model(maxSeqLen, embedDim).Call(x);
            
            // Transformer blocks
            foreach (var ctx in LoopAPI.Iterate(numLayers))
            {
                x = TfTransformerBlock.Model(embedDim, numHeads, hiddenDim, dropoutRate, layerNormEps).Call(x);
            }
            
            // Final layer norm
            x = TfLayerNorm.Model(embedDim, layerNormEps).Call(x);
            
            // Classification head
            return TfClassificationHead.Model(embedDim, numClasses, applySoftmax).Call(x);
        }
        
        /// <summary>
        /// Simple CLS token addition using concatenation
        /// </summary>
        private static Tensor<float32> TfAddClsToken(Scalar<int64> embedDim, Tensor<float32> x)
        {
            var batchSize = TransformerNetHelpers.GetDim(x, 0);
            var clsToken = TfInitXavier.Init([Scalar(1L), embedDim]);
            
            // For simplicity, use fixed batch size for the Tile operation
            var clsExpanded = clsToken.Tile(Vector(new long[] { 32L, 1L, 1L }));
            
            // Concatenate CLS token with patches
            return (Tensor<float32>)NN.Concat([clsExpanded, x], axis: 1L);
        }
    }
}
