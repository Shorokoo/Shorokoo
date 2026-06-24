using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Dropout</c> (opset 13+: <c>ratio</c> and <c>training_mode</c>
/// are optional INPUTS, <c>seed</c> is the only attribute). Up to 2 outputs (y, mask);
/// both are always emitted and the engine drops the mask when undeclared. In inference
/// mode (training_mode absent or known false) the op is an identity — y passes the
/// input data through and the mask is all-true, regardless of ratio. A connected
/// training_mode whose value is unknown at QEE time blocks value computation (the
/// concrete values are unknowable); training mode is random, so no values either.
/// </summary>
internal sealed class DropoutOp : QuickOp
{
    public override string OpCode => OpCodes.DROPOUT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var trainingMode = inputs.Length > 2 ? inputs[2] : null;
        var dtype = x?.DType ?? DType.Float32;

        var y = RuntimeTensorFactory.Create(dtype, x?.Shape);
        var mask = RuntimeTensorFactory.Create(DType.Bool, x?.Shape);

        // training_mode: absent → false; connected with known value → that value;
        // connected but unknown → null (blocks values).
        bool? training = trainingMode is null
            ? false
            : trainingMode.BoolData is { Length: > 0 } tb ? tb[0]
            : trainingMode.IntData is { Length: > 0 } ti ? ti[0] != 0
            : (bool?)null;

        if (training == false && x?.Shape is not null
            && RuntimeTensorFactory.ShouldStoreData(x.Shape, maxDataElements))
        {
            y = y with { FloatData = x.FloatData, IntData = x.IntData, BoolData = x.BoolData };
            if (x.HasAnyData)
            {
                var allTrue = new bool[x.Shape.Count];
                Array.Fill(allTrue, true);
                mask = mask with { BoolData = ImmutableArray.Create(allTrue) };
            }
        }

        return [y, mask];
    }
}
