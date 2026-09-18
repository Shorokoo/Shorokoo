using System;

namespace Shorokoo
{
    /// <summary>
    /// Metadata-only stand-in for a tensor nothing will read: it carries a dtype and a shape and
    /// allocates no element storage at all, so a shape-driven pass can be handed one parameter's
    /// worth of "this shape, this type" without a buffer per parameter.
    ///
    /// <para>What it is for is shape inference —
    /// <see cref="Shorokoo.TrainingRig.RepresentativeInputFor"/> hands these to the
    /// <see cref="Core.Inference.QuickExecutionEngine"/> for every input too large to be worth
    /// materializing. Reading its values is a bug and fails loudly.</para>
    ///
    /// <para>A stripped weight in a model <i>description</i> is not one of these: it is a
    /// values-elided <see cref="TensorAttribute"/> (<see cref="TensorAttribute.WithoutValues"/>),
    /// which is what a description with no values in it is. This type is the runtime-value
    /// counterpart, and moving one into an attribute gives that.</para>
    /// </summary>
    internal sealed class WeightPlaceholderTensorData : TensorData
    {
        internal WeightPlaceholderTensorData(Shape shape, DType dtype) : base(shape, dtype)
        {
        }

        /// <inheritdoc/>
        internal override bool HasValues => false;

        /// <inheritdoc/>
        public override Span<byte> AccessModifiableRawMemory()
        {
            ThrowIfDisposed();
            throw ValuesElided();
        }

        /// <inheritdoc/>
        public override ReadOnlySpan<byte> AccessRawMemory()
        {
            ThrowIfDisposed();
            throw ValuesElided();
        }

        /// <inheritdoc/>
        /// <summary>A placeholder holds no bytes, so there is nothing to share a name for.</summary>
        internal override TensorData CloneSharing(Shorokoo.Runtime.ComputeContext context, bool ownsMemory)
            => throw new InvalidOperationException(
                "A weight placeholder carries shape and dtype but no data, so it cannot be "
                + "transferred to a compute context.");

        public override void Dispose() => IsDisposed = true;

        private InvalidOperationException ValuesElided() => new(
            $"Tensor {this} is a weights-stripped placeholder carrying dtype/shape metadata only — " +
            "its values were elided when the model definition was saved without its weights. " +
            "Bind the checkpoint's weights (Persistence.Load) before accessing parameter values.");
    }
}
