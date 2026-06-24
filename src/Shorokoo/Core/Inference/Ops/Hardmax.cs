using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Hardmax</c> (opset 13+ single-axis semantics, axis default
/// −1): 1 for the FIRST maximum along the axis, 0 elsewhere (ties resolve to the first
/// occurrence per spec). Shape/dtype passthrough; values via
/// <see cref="SoftmaxFamilyOpBase"/>.
/// </summary>
internal sealed class HardmaxOp : SoftmaxFamilyOpBase
{
    public override string OpCode => OpCodes.HARDMAX;

    protected override void TransformSlice(float[] slice)
    {
        int best = 0;
        for (int i = 1; i < slice.Length; i++)
            if (slice[i] > slice[best]) best = i;
        for (int i = 0; i < slice.Length; i++) slice[i] = i == best ? 1f : 0f;
    }
}
