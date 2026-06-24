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
/// Per Shorokoo's Definitions, LOOP_SCAN_VARIABLE has input rank R and output rank R+1: the
/// output is the input with a leading "iteration" dimension prepended. The iteration-dim size
/// is not known at inference time (it would be the loop's iteration count) so we leave the
/// exact leading dim unknown and record rank-plus-one.
/// </summary>
internal sealed class LoopScanVariableOp : QuickOp
{
    public override string OpCode => OpCodes.LOOP_SCAN_VARIABLE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var dtype = x?.DType ?? DType.Float32;

        // The leading iteration dim is dynamic, so the exact shape is never known here.
        // (An exact Shape must not carry -1 placeholder dims — Shape/Size consumers would
        // surface them as concrete values.) Keep rank-plus-one metadata only.
        var rt = RuntimeTensorFactory.Create(dtype, null);
        var rank = x?.Shape?.Dims.Length ?? x?.Rank;
        if (rank is int r)
            rt = rt with { Rank = r + 1, MaxRank = r + 1 };
        return [rt];
    }
}
