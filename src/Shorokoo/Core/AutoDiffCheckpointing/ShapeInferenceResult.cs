using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using System.Collections.Immutable;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// Contains the complete shape inference results for a <see cref="InternalComputationGraph"/>.
/// Provides per-tensor shape information.
/// </summary>
internal class ShapeInferenceResult
{
    private readonly ImmutableDictionary<FastTensorKey, TensorShapeInfo> _tensorInfos;

    /// <summary>
    /// Shape information for each tensor in the graph, keyed by <see cref="FastTensorKey"/>.
    /// </summary>
    public ImmutableDictionary<FastTensorKey, TensorShapeInfo> TensorInfos => _tensorInfos;

    /// <summary>
    /// The total number of tensors in the graph.
    /// </summary>
    public int TensorCount => _tensorInfos.Count;

    internal ShapeInferenceResult(ImmutableDictionary<FastTensorKey, TensorShapeInfo> tensorInfos)
    {
        _tensorInfos = tensorInfos;
    }

    /// <summary>
    /// Gets the shape info for a specific tensor by its key.
    /// </summary>
    public TensorShapeInfo? GetTensorInfo(FastTensorKey key)
        => _tensorInfos.TryGetValue(key, out var info) ? info : null;
}
