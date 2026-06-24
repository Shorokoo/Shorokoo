using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Contains the complete shape inference results for a <see cref="FastComputationGraph"/>.
/// Provides per-tensor shape information and summary statistics about
/// maximum tensor sizes encountered during inference.
/// </summary>
internal class ShapeInferenceResult
{
    private readonly ImmutableDictionary<FastTensorKey, TensorShapeInfo> _tensorInfos;

    /// <summary>
    /// Shape information for each tensor in the graph, keyed by <see cref="FastTensorKey"/>.
    /// </summary>
    public ImmutableDictionary<FastTensorKey, TensorShapeInfo> TensorInfos => _tensorInfos;

    /// <summary>
    /// The maximum number of elements in any single tensor in the graph.
    /// </summary>
    public long MaxElementCount { get; }

    /// <summary>
    /// The maximum rank (number of dimensions) of any tensor in the graph.
    /// </summary>
    public int MaxRank { get; }

    /// <summary>
    /// The maximum size of any single dimension across all tensors in the graph.
    /// </summary>
    public long MaxDimensionSize { get; }

    /// <summary>
    /// The total number of tensors in the graph.
    /// </summary>
    public int TensorCount => _tensorInfos.Count;

    internal ShapeInferenceResult(ImmutableDictionary<FastTensorKey, TensorShapeInfo> tensorInfos)
    {
        _tensorInfos = tensorInfos;

        long maxElements = 0;
        int maxRank = 0;
        long maxDim = 0;

        foreach (var info in tensorInfos.Values)
        {
            if (info.ElementCount > maxElements)
                maxElements = info.ElementCount;

            var rank = info.Shape.Dims.Length;
            if (rank > maxRank)
                maxRank = rank;

            foreach (var dim in info.Shape.Dims)
            {
                if (dim > maxDim)
                    maxDim = dim;
            }
        }

        MaxElementCount = maxElements;
        MaxRank = maxRank;
        MaxDimensionSize = maxDim;
    }

    /// <summary>
    /// Gets the shape info for a specific tensor by its key.
    /// </summary>
    public TensorShapeInfo? GetTensorInfo(FastTensorKey key)
        => _tensorInfos.TryGetValue(key, out var info) ? info : null;
}
