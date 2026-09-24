using Shorokoo.Onnx;
using Shorokoo.Core.Utils;

namespace Shorokoo
{
    /// <summary>
    /// A concrete value for an <c>OptionalTensor</c> input/output: either <b>present</b> (wrapping
    /// a <see cref="TensorData"/>) or <b>absent</b>. This is the <see cref="IData"/> counterpart of
    /// a nullable module parameter (<c>Tensor&lt;T&gt;?</c>) — pass
    /// <see cref="Some(TensorData)"/> to supply a value and <see cref="None(DType)"/> to drive the
    /// parameter's default branch.
    /// </summary>
    public sealed class OptionalTensorData : IData
    {
        /// <summary>The optional's element data type (independent of presence).</summary>
        public DType DType { get; }

        /// <summary>True when a value is present.</summary>
        public bool HasValue { get; }

        /// <summary>The held tensor, or <c>null</c> when absent.</summary>
        public TensorData? Value { get; }

        private OptionalTensorData(DType elementDType, bool hasValue, TensorData? value)
        {
            DType = elementDType;
            HasValue = hasValue;
            Value = value;
        }

        /// <summary>A present optional wrapping <paramref name="value"/>.</summary>
        public static OptionalTensorData Some(TensorData value)
            => new OptionalTensorData(
                (value ?? throw new System.ArgumentNullException(nameof(value))).DType, true, value);

        /// <summary>An absent optional with the given element type.</summary>
        public static OptionalTensorData None(DType elementDType)
            => new OptionalTensorData(elementDType, false, null);

        /// <summary>A present optional from a typed tensor.</summary>
        public static OptionalTensorData Some<T>(TensorData<T> value) where T : IVarType
            => Some((TensorData)value);

        /// <summary>An absent optional whose element type is inferred from <typeparamref name="T"/>.</summary>
        public static OptionalTensorData None<T>() where T : IVarType
            => None(OnnxUtils.GetDType<T>());

        /// <summary>
        /// This optional's tensor, if it has one, to be <b>read</b> by the run it is fed to rather
        /// than consumed; see <see cref="TensorData.Shared"/>. An absent optional holds nothing to
        /// consume, and feeds the same either way.
        /// </summary>
        public SharedInput Shared() => new(this, SharedInputMode.Shared);

        /// <summary>
        /// This optional's tensor, if it has one, to be consumed by the run it is fed to where
        /// nothing else is reading it when that run starts, and read otherwise; see
        /// <see cref="TensorData.TryConsume"/>.
        /// </summary>
        public SharedInput TryConsume() => new(this, SharedInputMode.TryConsume);

        public override string ToString()
            => HasValue ? $"optional:some[{Value}]" : $"optional:none[{DType}]";
    }
}
