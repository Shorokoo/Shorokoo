using System;

namespace Shorokoo
{
    /// <summary>
    /// Metadata-only stand-in for a stripped weight tensor: it carries the weight's
    /// dtype and shape but allocates no element storage at all. The .skpt save path
    /// (<see cref="CheckpointBuilder.Save"/>) swaps each weight parameter's tensor for
    /// one of these before serializing the model definition — so stripping never
    /// materializes a full-size zero buffer per weight — and the ONNX reader
    /// reconstructs one when an initializer carries the values-elided marker
    /// (<see cref="Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkMetaValuesElided"/>).
    /// <see cref="Checkpoint.Load"/> replaces every placeholder with the checkpoint's
    /// real tensor before the model is returned; reading a placeholder's values is a
    /// bug and fails loudly.
    /// </summary>
    internal sealed class WeightPlaceholderTensorData : TensorData
    {
        internal WeightPlaceholderTensorData(Shape shape, DType dtype) : base(shape, dtype)
        {
        }

        /// <inheritdoc/>
        public override Span<byte> AccessModifiableRawMemory() => throw ValuesElided();

        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory() => throw ValuesElided();

        /// <inheritdoc/>
        public override void Dispose()
        {
        }

        private InvalidOperationException ValuesElided() => new(
            $"Tensor {this} is a weights-stripped placeholder carrying dtype/shape metadata only — " +
            "its values were elided when the model definition was saved without its weights. " +
            "Bind the checkpoint's weights (Checkpoint.Load) before accessing parameter values.");
    }
}
