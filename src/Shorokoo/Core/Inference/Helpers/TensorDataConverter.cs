using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Abstractions;
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
    ///
    /// <para>QEE holds every integer width in one <c>long</c> buffer, so an integer tensor's
    /// actual width lives only in <see cref="RuntimeTensor.DType"/> — materializing from the
    /// buffer alone would retype every non-<c>int64</c> integer tensor as <c>int64</c>. That
    /// silently corrupts host constant folding, whose folded values re-enter the graph as
    /// CONSTANT nodes and must still satisfy their consumers' type constraints.</para>
    /// </summary>
    public static TensorData? ToTensorData(RuntimeTensor rt)
    {
        if (rt.Shape is null) return null;
        var dims = rt.Shape.Dims;

        if (rt.IntData is { } idata)
            return FromIntData(dims, idata, rt.DType);
        if (rt.FloatData is { } fdata)
            return FromFloatData(dims, fdata, rt.DType);
        if (rt.BoolData is { } bdata)
            return TensorData(dims, bdata.ToArray());

        return null;
    }

    /// <summary>
    /// Materializes QEE's float buffer at <paramref name="dtype"/>'s own type. QEE keeps every
    /// float width in one <c>float</c> buffer (<see cref="ToRuntimeTensor"/> narrows Float64 on
    /// the way in), so the dtype lives only in <see cref="RuntimeTensor.DType"/> — the same trap
    /// the integer side had: emitting Float32 for a Float64/Float16/BFloat16 tensor retypes it,
    /// and a host-folded constant then violates its consumer's type constraint. Precision already
    /// lost on load is not recovered here; only the type is kept honest.
    /// </summary>
    private static TensorData FromFloatData(long[] dims, ImmutableArray<float> fdata, DType dtype)
    {
        var n = fdata.Length;
        if (dtype == DType.Float64) { var b = new double[n]; for (int i = 0; i < n; i++) b[i] = fdata[i];              return TensorData(dims, b); }
        if (dtype == DType.Float16) { var b = new Float16[n]; for (int i = 0; i < n; i++) b[i] = (Float16)fdata[i];    return TensorData(dims, b); }
        if (dtype == DType.BFloat16){ var b = new BFloat16[n]; for (int i = 0; i < n; i++) b[i] = (BFloat16)fdata[i];  return TensorData(dims, b); }
        return TensorData(dims, fdata.ToArray());
    }

    /// <summary>
    /// Materializes QEE's 64-bit integer buffer at <paramref name="dtype"/>'s own width.
    /// Values wider than the target wrap (unchecked), which is the only thing a narrower
    /// buffer can represent and matches what the ONNX runtimes produce. A dtype that is not
    /// an integer width (a Bool or Float tensor whose data landed in the int buffer) keeps
    /// the historical <c>int64</c> materialization.
    /// </summary>
    private static TensorData FromIntData(long[] dims, ImmutableArray<long> idata, DType dtype)
    {
        var n = idata.Length;
        if (dtype == DType.Int64) return TensorData(dims, idata.ToArray());
        if (dtype == DType.Int32) { var b = new int[n];    for (int i = 0; i < n; i++) b[i] = unchecked((int)idata[i]);    return TensorData(dims, b); }
        if (dtype == DType.Int16) { var b = new short[n];  for (int i = 0; i < n; i++) b[i] = unchecked((short)idata[i]);  return TensorData(dims, b); }
        if (dtype == DType.Int8)  { var b = new sbyte[n];  for (int i = 0; i < n; i++) b[i] = unchecked((sbyte)idata[i]);  return TensorData(dims, b); }
        if (dtype == DType.UInt64){ var b = new ulong[n];  for (int i = 0; i < n; i++) b[i] = unchecked((ulong)idata[i]);  return TensorData(dims, b); }
        if (dtype == DType.UInt32){ var b = new uint[n];   for (int i = 0; i < n; i++) b[i] = unchecked((uint)idata[i]);   return TensorData(dims, b); }
        if (dtype == DType.UInt16){ var b = new ushort[n]; for (int i = 0; i < n; i++) b[i] = unchecked((ushort)idata[i]); return TensorData(dims, b); }
        if (dtype == DType.UInt8) { var b = new byte[n];   for (int i = 0; i < n; i++) b[i] = unchecked((byte)idata[i]);   return TensorData(dims, b); }
        return TensorData(dims, idata.ToArray());
    }
}
