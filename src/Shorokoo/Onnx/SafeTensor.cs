using System;
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

        private readonly TensorData? _data;
        private readonly TensorAttribute? _attribute;

        /// <summary>
        /// Tensor data. A record parsed out of a file holds a tensor and hands it over; one built
        /// from a graph's own parameter holds that graph's <see cref="TensorAttribute"/>, which is
        /// immutable and shared, so reading it as a tensor takes a copy — which is why the writer
        /// reads <see cref="RawBytes"/> instead and a checkpoint save still costs no second copy
        /// of the model.
        /// </summary>
        public TensorData Data => _data ?? _attribute!.CopyToTensorData();

        /// <summary>The elements as they will be written, out of whichever form this record holds
        /// and copying neither.</summary>
        internal ReadOnlySpan<byte> RawBytes
        {
            get
            {
                if (_attribute is not null) return _attribute.Bytes;
                return _data!.AccessRawMemory();
            }
        }

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
            : this(name, data ?? throw new ArgumentNullException(nameof(data)), null, dataType, shape, metadata)
        {
        }

        /// <summary>A record over a graph literal — a model parameter on its way into a
        /// checkpoint, written straight out of the graph's own attribute.</summary>
        public SafeTensor(string name, TensorAttribute data, string dataType, long[] shape, IReadOnlyDictionary<string, object>? metadata = null)
            : this(name, null, data ?? throw new ArgumentNullException(nameof(data)), dataType, shape, metadata)
        {
        }

        private SafeTensor(
            string name, TensorData? data, TensorAttribute? attribute, string dataType, long[] shape,
            IReadOnlyDictionary<string, object>? metadata)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            _data = data;
            _attribute = attribute;
            DataType = dataType ?? throw new ArgumentNullException(nameof(dataType));
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Metadata = metadata ?? new Dictionary<string, object>();
        }
    }
}