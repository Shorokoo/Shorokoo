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
/// ONNX <c>SequenceEmpty</c>: produces an empty sequence. Element dtype comes from the
/// <c>dtype</c> attribute (default float32 per ONNX spec).
/// </summary>
internal sealed class SequenceEmptyOp : QuickOp
{
    public override string OpCode => OpCodes.SEQUENCE_EMPTY;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? DType.Float32;
        var seq = new RuntimeSequenceTensor
        {
            DType = dtype,
            Count = 0,
            Tensors = ImmutableArray<RuntimeTensor>.Empty,
        };
        return [seq];
    }
}
