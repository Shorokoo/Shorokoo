using System;
using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

internal sealed class ConstantOfShapeOp : QuickOp
{
    public override string OpCode => OpCodes.CONSTANT_OF_SHAPE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapeInput = inputs[0];
        var valueTensor = attrs.GetTensorVal(OnnxOpAttributeNames.AttrValue);
        var dtype = valueTensor?.DType ?? DType.Float32;

        Shape? outShape = null;
        // All shape entries must be >= 0 per spec; a negative entry is invalid and degrades
        // to an unknown shape rather than a negative dim.
        if (shapeInput?.IntData is { } shapeData && shapeData.All(d => d >= 0))
            outShape = new Shape(shapeData.ToArray());

        var rt = RuntimeTensorFactory.Create(dtype, outShape);
        if (outShape is not null && RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements))
        {
            var cnt = (int)outShape.Count;
            if (DTypeHelpers.IsFloat(dtype))
            {
                var data = new float[cnt];
                var fill = ReadFloatFill(valueTensor, dtype);
                if (fill != 0f) Array.Fill(data, fill);
                return [rt with { FloatData = ImmutableArray.Create(data) }];
            }
            if (DTypeHelpers.IsInt(dtype))
            {
                var data = new long[cnt];
                var fill = ReadLongFill(valueTensor, dtype);
                if (fill != 0L) Array.Fill(data, fill);
                return [rt with { IntData = ImmutableArray.Create(data) }];
            }
            if (DTypeHelpers.IsBool(dtype))
            {
                var data = new bool[cnt];
                if (ReadBoolFill(valueTensor)) Array.Fill(data, true);
                return [rt with { BoolData = ImmutableArray.Create(data) }];
            }
        }
        return [rt];
    }

    private static float ReadFloatFill(TensorData? valueTensor, DType dtype)
    {
        if (valueTensor is null) return 0f;
        var bytes = valueTensor.AccessRawMemory();
        if (dtype == DType.Float32) return bytes.Length >= 4 ? MemoryMarshal.Cast<byte, float>(bytes)[0] : 0f;
        if (dtype == DType.Float64) return bytes.Length >= 8 ? (float)MemoryMarshal.Cast<byte, double>(bytes)[0] : 0f;
        if (dtype == DType.Float16) return bytes.Length >= 2 ? (float)MemoryMarshal.Cast<byte, Float16>(bytes)[0] : 0f;
        if (dtype == DType.BFloat16) return bytes.Length >= 2 ? (float)MemoryMarshal.Cast<byte, BFloat16>(bytes)[0] : 0f;
        return 0f;
    }

    private static long ReadLongFill(TensorData? valueTensor, DType dtype)
    {
        if (valueTensor is null) return 0L;
        var bytes = valueTensor.AccessRawMemory();
        if (dtype == DType.Int8)   return bytes.Length >= 1 ? MemoryMarshal.Cast<byte, sbyte>(bytes)[0] : 0L;
        if (dtype == DType.Int16)  return bytes.Length >= 2 ? MemoryMarshal.Cast<byte, short>(bytes)[0] : 0L;
        if (dtype == DType.Int32)  return bytes.Length >= 4 ? MemoryMarshal.Cast<byte, int>(bytes)[0] : 0L;
        if (dtype == DType.Int64)  return bytes.Length >= 8 ? MemoryMarshal.Cast<byte, long>(bytes)[0] : 0L;
        if (dtype == DType.UInt8)  return bytes.Length >= 1 ? bytes[0] : 0L;
        if (dtype == DType.UInt16) return bytes.Length >= 2 ? MemoryMarshal.Cast<byte, ushort>(bytes)[0] : 0L;
        if (dtype == DType.UInt32) return bytes.Length >= 4 ? MemoryMarshal.Cast<byte, uint>(bytes)[0] : 0L;
        if (dtype == DType.UInt64) return bytes.Length >= 8 ? (long)MemoryMarshal.Cast<byte, ulong>(bytes)[0] : 0L;
        return 0L;
    }

    private static bool ReadBoolFill(TensorData? valueTensor)
    {
        if (valueTensor is null) return false;
        var bytes = valueTensor.AccessRawMemory();
        return bytes.Length >= 1 && bytes[0] != 0;
    }
}
