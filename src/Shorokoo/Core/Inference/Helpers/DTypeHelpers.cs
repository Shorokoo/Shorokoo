using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Classifies DTypes into categories used by the QuickExecutionEngine runtime tensor.
/// Float-like dtypes use float storage, integer-like dtypes use long storage.
/// </summary>
internal enum DTypeCategory
{
    Float,
    Int,
    Bool,
    Other,
}

internal static class DTypeHelpers
{
    public static DTypeCategory Categorize(DType dtype)
    {
        if (dtype == DType.Float16 || dtype == DType.BFloat16 ||
            dtype == DType.Float32 || dtype == DType.Float64)
            return DTypeCategory.Float;

        if (dtype == DType.Int8 || dtype == DType.Int16 || dtype == DType.Int32 || dtype == DType.Int64 ||
            dtype == DType.UInt8 || dtype == DType.UInt16 || dtype == DType.UInt32 || dtype == DType.UInt64)
            return DTypeCategory.Int;

        if (dtype == DType.Bool)
            return DTypeCategory.Bool;

        return DTypeCategory.Other;
    }

    public static bool IsFloat(DType dtype) => Categorize(dtype) == DTypeCategory.Float;
    public static bool IsInt(DType dtype) => Categorize(dtype) == DTypeCategory.Int;
    public static bool IsBool(DType dtype) => dtype == DType.Bool;
}
