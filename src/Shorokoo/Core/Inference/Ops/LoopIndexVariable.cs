using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Placeholder op used by Shorokoo's Looper to reference the current iteration index inside
/// the body. Produces a scalar int64 with no known concrete value (iteration is dynamic at
/// shape-inference time).
/// </summary>
internal sealed class LoopIndexVariableOp : QuickOp
{
    public override string OpCode => OpCodes.LOOP_INDEX_VARIABLE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        return [RuntimeTensorFactory.Create(DType.Int64, new Shape(Array.Empty<long>()))];
    }
}
