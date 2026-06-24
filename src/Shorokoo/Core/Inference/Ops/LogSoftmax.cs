using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>LogSoftmax</c> (opset 13+ single-axis semantics, axis default
/// −1): x − max − log(Σ exp(x − max)) along the axis. Shape/dtype passthrough; values
/// via <see cref="SoftmaxFamilyOpBase"/>.
/// </summary>
internal sealed class LogSoftmaxOp : SoftmaxFamilyOpBase
{
    public override string OpCode => OpCodes.LOG_SOFTMAX;

    protected override void TransformSlice(float[] slice)
    {
        float max = float.NegativeInfinity;
        foreach (var v in slice) if (v > max) max = v;
        float sum = 0;
        for (int i = 0; i < slice.Length; i++) sum += MathF.Exp(slice[i] - max);
        float logSum = MathF.Log(sum);
        for (int i = 0; i < slice.Length; i++) slice[i] = slice[i] - max - logSum;
    }
}
