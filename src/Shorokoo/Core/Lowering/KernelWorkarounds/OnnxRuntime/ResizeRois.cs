using Shorokoo.Core.Nodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

/// <summary>What a constant <c>Resize</c> roi tells the workarounds when the model is built.</summary>
internal static class ResizeRois
{
    /// <summary>The values of a float32 or float64 constant <paramref name="roi"/>, starts then
    /// ends; null for any other.</summary>
    public static double[]? Values(TensorAttribute? roi)
    {
        if (roi is null) return null;
        if (roi.DType == DType.Float32) return [.. roi.Elements<float>().ToArray().Select(v => (double)v)];
        if (roi.DType == DType.Float64) return roi.Elements<double>().ToArray();
        return null;
    }

    /// <summary>Whether <paramref name="roi"/> is a constant starting every axis at 0 and ending
    /// it at 1.</summary>
    public static bool IsIdentity(TensorAttribute? roi)
    {
        if (Values(roi) is not { } values || values.Length % 2 != 0) return false;
        int half = values.Length / 2;
        return values.Take(half).All(v => v == 0) && values.Skip(half).All(v => v == 1);
    }
}
