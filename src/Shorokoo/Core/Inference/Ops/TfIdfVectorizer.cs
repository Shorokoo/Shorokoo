using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>TfIdfVectorizer</c>. Per spec: for input shape [C] the
/// output is [C']; for [N, C] the output is [N, C'], where C' = max(ngram_indexes) + 1
/// (ngram_indexes maps each pool n-gram to its output coordinate, so the output extent
/// is the largest coordinate + 1 — NOT the pool length). Output dtype is float32.
/// </summary>
internal sealed class TfIdfVectorizerOp : QuickOp
{
    public override string OpCode => OpCodes.TFIDF_VECTORIZER;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = DType.Float32;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var indexes = attrs.GetLongsVal(OnnxOpAttributeNames.AttrNgramIndexes);
        long? cprime = indexes is { Length: > 0 } ? indexes.Max() + 1 : null;
        var dims = x.Shape.Dims;
        if (cprime is not { } c || dims.Length is not (1 or 2))
            return [RuntimeTensorFactory.Create(dtype, null)];

        long[] outDims = dims.Length == 1 ? new[] { c } : new[] { dims[0], c };
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}
