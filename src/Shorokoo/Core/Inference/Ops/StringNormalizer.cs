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
/// Shape inference for ONNX <c>StringNormalizer</c>. Input is 1-D <c>[C]</c> or 2-D
/// <c>[1, C]</c>. Without stopwords the op is a pure case change and the shape passes
/// through exactly; with stopwords the data-dependent filtering can shrink the trailing
/// dim, so the exact shape degrades to rank + a per-dim upper bound (string VALUES never
/// reach QEE, so the filtered extent is unknowable).
/// </summary>
internal sealed class StringNormalizerOp : QuickOp
{
    public override string OpCode => OpCodes.STRING_NORMALIZER;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.String;
        if (x?.Shape is null) return [RuntimeTensorFactory.Create(dtype, null)];

        var stopwords = attrs.GetStringsVal(OnnxOpAttributeNames.AttrStopwords);
        if (stopwords is null || stopwords.Length == 0)
            return [RuntimeTensorFactory.Create(dtype, x.Shape)];

        // Stopword filtering: output extent along the last dim is data-dependent (and the
        // spec pads an all-filtered result to a single empty string, so 1..C). Keep rank +
        // the input shape as the upper bound — never an exact shape with a guessed dim.
        var rank = x.Shape.Dims.Length;
        return [RuntimeTensorFactory.Create(dtype, null) with
        {
            MaxShape = x.Shape,
            Rank = rank,
            MaxRank = rank,
        }];
    }
}
