using System.Collections.Immutable;
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
/// ONNX <c>SequenceLength</c>: produces a scalar int64 holding the sequence's count when known.
/// </summary>
internal sealed class SequenceLengthOp : QuickOp
{
    public override string OpCode => OpCodes.SEQUENCE_LENGTH;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seq = inputs.Length > 0 ? inputs[0] as RuntimeSequenceTensor : null;
        var rt = RuntimeTensorFactory.Create(DType.Int64, new Shape(Array.Empty<long>()));
        if (seq?.Count is long count) rt = rt with { IntData = ImmutableArray.Create(count) };
        return [rt];
    }
}
