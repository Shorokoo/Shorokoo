using System.Collections.Generic;
using Shorokoo.Core.Utils;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Represents a single tensor from a SafeTensor file with its associated metadata
    /// </summary>
    public class SafeTensor
    {
        /// <summary>
        /// Name of the tensor
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Tensor data
        /// </summary>
        public TensorData Data { get; }

        /// <summary>
        /// Additional metadata from the SafeTensor file (if any)
        /// </summary>
        public IReadOnlyDictionary<string, object> Metadata { get; }

        /// <summary>
        /// Data type as specified in the SafeTensor file
        /// </summary>
        public string DataType { get; }

        /// <summary>
        /// Shape of the tensor
        /// </summary>
        public long[] Shape { get; }

        public SafeTensor(string name, TensorData data, string dataType, long[] shape, IReadOnlyDictionary<string, object>? metadata = null)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Data = data ?? throw new ArgumentNullException(nameof(data));
            DataType = dataType ?? throw new ArgumentNullException(nameof(dataType));
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Metadata = metadata ?? new Dictionary<string, object>();
        }
    }
}