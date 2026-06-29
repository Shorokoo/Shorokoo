using System.Collections.Immutable;
using System.Linq;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Helpers for creating <see cref="RuntimeTensor"/> instances. Centralizes the &gt;256
/// elements rule: data is only retained when the element count is small.
/// </summary>
internal static class RuntimeTensorFactory
{
    /// <summary>
    /// Creates a runtime tensor with known shape and dtype, attaching the reference variable
    /// when provided. Rank and MaxRank are derived from the shape.
    /// </summary>
    public static RuntimeTensor Create(DType dtype, Shape? shape, Variable? reference = null)
    {
        return new RuntimeTensor
        {
            DType = dtype,
            Shape = shape,
            MaxShape = shape,
            Rank = shape?.Dims.Length,
            MaxRank = shape?.Dims.Length,
            ReferenceTensor = reference,
        };
    }

    /// <summary>
    /// Creates a runtime tensor whose dtype (and optionally exact rank) are known but whose
    /// shape is not. Used by ops whose output dims are data-dependent or whose inputs are not
    /// concrete enough at QEE time — per the audit contract the shape must degrade to unknown
    /// rather than carry guessed / negative placeholder dims.
    /// </summary>
    public static RuntimeTensor CreateRankOnly(DType dtype, int? rank)
    {
        return new RuntimeTensor
        {
            DType = dtype,
            Shape = null,
            MaxShape = null,
            Rank = rank,
            MaxRank = rank,
        };
    }

    /// <summary>
    /// Decides whether the given shape is small enough for data to be retained. Returns true if
    /// the total element count is &lt;= <paramref name="maxElements"/>.
    /// </summary>
    public static bool ShouldStoreData(Shape? shape, int maxElements)
    {
        if (shape is null) return false;
        var count = shape.Count;
        return count >= 0 && count <= maxElements;
    }

    /// <summary>
    /// Applies the "data only for small tensors" rule: returns a copy with all data fields
    /// nulled out if the shape is larger than <paramref name="maxElements"/>. Otherwise returns
    /// the input unchanged.
    /// </summary>
    public static RuntimeTensor EnforceDataSizeLimit(RuntimeTensor rt, int maxElements)
    {
        if (ShouldStoreData(rt.Shape, maxElements)) return rt;
        if (rt.FloatData is null && rt.IntData is null && rt.StringData is null && rt.BoolData is null)
            return rt;
        return rt with { FloatData = null, IntData = null, StringData = null, BoolData = null };
    }

    /// <summary>
    /// <see cref="IRuntimeTensor"/>-aware dispatch that returns a copy of <paramref name="rt"/>
    /// with data-size limit enforced on the plain tensor, the optional's value tensor, or every
    /// element / template tensor of a sequence.
    /// </summary>
    public static IRuntimeTensor EnforceDataSizeLimit(IRuntimeTensor rt, int maxElements)
    {
        return rt switch
        {
            RuntimeTensor plain => EnforceDataSizeLimit(plain, maxElements),
            RuntimeOptionalTensor opt => opt.ValueTensor is null
                ? opt
                : opt with { ValueTensor = EnforceDataSizeLimit(opt.ValueTensor, maxElements) },
            RuntimeSequenceTensor seq => seq with
            {
                Tensors = seq.Tensors is { } ts
                    ? ts.Select(t => EnforceDataSizeLimit(t, maxElements)).ToImmutableArray()
                    : null,
                TemplateTensor = seq.TemplateTensor is null
                    ? null
                    : EnforceDataSizeLimit(seq.TemplateTensor, maxElements),
            },
            _ => rt,
        };
    }
}
