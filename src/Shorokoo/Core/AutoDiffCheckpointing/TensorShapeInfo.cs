namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Holds shape and optional value information for a single tensor in the computation graph.
/// For tensors with fewer than <see cref="ShapeInferenceInterpreter.MaxSmallTensorElements"/> elements,
/// the actual computed values are retained. For larger tensors, only the shape is stored.
/// </summary>
internal class TensorShapeInfo
{
    /// <summary>
    /// The shape of the tensor (dimensions).
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    /// The data type of the tensor.
    /// </summary>
    public DType DType { get; }

    /// <summary>
    /// The total number of elements in the tensor.
    /// </summary>
    public long ElementCount => Shape.Count;

    /// <summary>
    /// The memory usage of this tensor in bytes.
    /// Computed as element count × bytes per element based on the data type.
    /// </summary>
    public long MemoryBytes { get; }

    /// <summary>
    /// The actual tensor data, retained only for small tensors (fewer than 1024 elements).
    /// Null for large tensors where values were discarded to save memory.
    /// </summary>
    public TensorData? Data { get; }

    /// <summary>
    /// Whether the actual tensor values are available.
    /// </summary>
    public bool HasData => Data is not null;

    public TensorShapeInfo(Shape shape, DType dtype, TensorData? data)
    {
        Shape = shape;
        DType = dtype;
        Data = data;

        // Compute memory usage: element count × bytes per element
        var bitsPerElement = dtype.EncodingBitCount;
        MemoryBytes = bitsPerElement >= 8
            ? shape.Count * (bitsPerElement / 8)
            : shape.Count; // Fallback for sub-byte types
    }

    public override string ToString()
        => $"{DType} {Shape}{(HasData ? " [values retained]" : "")}";
}
