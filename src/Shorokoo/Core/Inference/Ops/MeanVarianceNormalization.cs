using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for <c>MeanVarianceNormalization</c>: passthrough. Concrete values are
/// not computed because normalization depends on per-axis stats over the runtime data.
/// </summary>
internal sealed class MeanVarianceNormalizationOp : QuickOp
{
    public override string OpCode => OpCodes.MEAN_VARIANCE_NORMALIZATION;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;
        return [RuntimeTensorFactory.Create(dtype, x?.Shape)];
    }
}
