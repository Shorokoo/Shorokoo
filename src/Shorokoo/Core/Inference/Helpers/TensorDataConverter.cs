using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Utils;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Converts raw <see cref="TensorData"/> buffers (e.g., from MODEL_PARAM_DATA or CONSTANT
/// attributes) into the ImmutableArray payloads used by <see cref="RuntimeTensor"/>.
/// </summary>
internal static class TensorDataConverter
{
    /// <summary>
    /// Returns a new runtime tensor populated from the given TensorData. Shape/dtype are
    /// always filled in; element data is filled in only when the element count is at most
    /// <paramref name="maxElements"/>.
    /// </summary>
    public static RuntimeTensor ToRuntimeTensor(TensorData data, int maxElements, Variable? reference = null)
    {
        var dtype = data.DType;
        var shape = data.Shape;
        var count = (int)shape.Count;

        ImmutableArray<float>? fData = null;
        ImmutableArray<long>? iData = null;
        ImmutableArray<bool>? bData = null;

        // DType.String is variable-length UTF-8; the underlying ORT tensor has no flat
        // byte buffer to span over, so AccessRawMemory would throw. QEE shape inference
        // for the string ops only needs dtype + shape — leave the data fields unset.
        if (shape.Count <= maxElements && dtype != DType.String)
        {
            var bytes = data.AccessRawMemory();
            if (dtype == DType.Float32)
            {
                var buf = new float[count];
                MemoryMarshal.Cast<byte, float>(bytes).CopyTo(buf);
                fData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.Float64)
            {
                var buf = new float[count];
                var src = MemoryMarshal.Cast<byte, double>(bytes);
                for (int i = 0; i < count; i++) buf[i] = (float)src[i];
                fData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.Int64)
            {
                var buf = new long[count];
                MemoryMarshal.Cast<byte, long>(bytes).CopyTo(buf);
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.Int32)
            {
                var buf = new long[count];
                var src = MemoryMarshal.Cast<byte, int>(bytes);
                for (int i = 0; i < count; i++) buf[i] = src[i];
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.Int16)
            {
                var buf = new long[count];
                var src = MemoryMarshal.Cast<byte, short>(bytes);
                for (int i = 0; i < count; i++) buf[i] = src[i];
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.Int8)
            {
                var buf = new long[count];
                var src = MemoryMarshal.Cast<byte, sbyte>(bytes);
                for (int i = 0; i < count; i++) buf[i] = src[i];
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.UInt8)
            {
                var buf = new long[count];
                for (int i = 0; i < count; i++) buf[i] = bytes[i];
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.UInt16)
            {
                var buf = new long[count];
                var src = MemoryMarshal.Cast<byte, ushort>(bytes);
                for (int i = 0; i < count; i++) buf[i] = src[i];
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.UInt32)
            {
                var buf = new long[count];
                var src = MemoryMarshal.Cast<byte, uint>(bytes);
                for (int i = 0; i < count; i++) buf[i] = src[i];
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.UInt64)
            {
                var buf = new long[count];
                var src = MemoryMarshal.Cast<byte, ulong>(bytes);
                for (int i = 0; i < count; i++) buf[i] = unchecked((long)src[i]);
                iData = ImmutableArray.Create(buf);
            }
            else if (dtype == DType.Bool)
            {
                var buf = new bool[count];
                for (int i = 0; i < count; i++) buf[i] = bytes[i] != 0;
                bData = ImmutableArray.Create(buf);
            }
        }

        return new RuntimeTensor
        {
            DType = dtype,
            Shape = shape,
            MaxShape = shape,
            Rank = shape.Dims.Length,
            MaxRank = shape.Dims.Length,
            ReferenceTensor = reference,
            FloatData = fData,
            IntData = iData,
            BoolData = bData,
        };
    }

    /// <summary>
    /// Converts an <see cref="OptionalTensorData"/> input into a <see cref="RuntimeOptionalTensor"/>
    /// — a present optional carries its value as a <see cref="RuntimeTensor"/>; an absent one carries
    /// just the element dtype with <c>HasValue == false</c>.
    /// </summary>
    public static RuntimeOptionalTensor ToRuntimeOptional(OptionalTensorData data, int maxElements, Variable? reference = null)
        => new RuntimeOptionalTensor
        {
            DType = data.DType,
            ReferenceTensor = reference,
            HasValue = data.HasValue,
            ValueTensor = data.HasValue && data.Value is not null
                ? ToRuntimeTensor(data.Value, maxElements, reference)
                : null,
        };

    /// <summary>
    /// Converts a <see cref="RuntimeOptionalTensor"/> back to an <see cref="OptionalTensorData"/>.
    /// Returns an absent value when presence is unknown/false or the held tensor has no data.
    /// </summary>
    public static OptionalTensorData ToOptionalTensorData(RuntimeOptionalTensor rt)
    {
        if (rt.HasValue == true && rt.ValueTensor is { } v && ToTensorData(v) is { } td)
            return OptionalTensorData.Some(td);
        return OptionalTensorData.None(rt.DType);
    }

    /// <summary>
    /// Converts an input <see cref="IData"/> (plain tensor or optional) into the matching
    /// <see cref="IRuntimeTensor"/> for the QuickExecutionEngine input store.
    /// </summary>
    public static IRuntimeTensor ToRuntimeInput(IData data, int maxElements, Variable? reference = null)
        => data switch
        {
            OptionalTensorData opt => ToRuntimeOptional(opt, maxElements, reference),
            TensorData td => ToRuntimeTensor(td, maxElements, reference),
            _ => throw new InvalidTensorOperationException(ErrorCodes.FW008, data.GetType().Name, "ToRuntimeInput",
                $"Unsupported input IData type for the QuickExecutionEngine: {data.GetType().Name}"),
        };

    /// <summary>
    /// Converts a runtime tensor produced by execution back into the matching output
    /// <see cref="IData"/> (plain tensor or optional). Returns null only for a plain tensor with no
    /// concrete data.
    /// </summary>
    public static IData? ToOutputData(IRuntimeTensor rt)
        => rt switch
        {
            RuntimeOptionalTensor opt => ToOptionalTensorData(opt),
            RuntimeTensor plain => ToTensorData(plain),
            _ => null,
        };

    /// <summary>
    /// Converts a <see cref="RuntimeTensor"/> back to <see cref="TensorData"/>. Returns null
    /// when the tensor has no concrete data (shape-only) or no known shape.
    /// </summary>
    public static TensorData? ToTensorData(RuntimeTensor rt)
    {
        if (rt.Shape is null) return null;
        var dims = rt.Shape.Dims;

        if (rt.IntData is { } idata)
            return TensorData(dims, idata.ToArray());
        if (rt.FloatData is { } fdata)
            return TensorData(dims, fdata.ToArray());
        if (rt.BoolData is { } bdata)
            return TensorData(dims, bdata.ToArray());

        return null;
    }
}
