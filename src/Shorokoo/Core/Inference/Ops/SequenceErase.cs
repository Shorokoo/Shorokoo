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
/// ONNX <c>SequenceErase</c>: returns a new sequence with the element at the given position
/// removed (or the last element when position is absent). When the sequence is concrete and
/// the position is statically known we return a concrete shorter sequence; otherwise template
/// mode with decremented count.
/// </summary>
internal sealed class SequenceEraseOp : QuickOp
{
    public override string OpCode => OpCodes.SEQUENCE_ERASE;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seqIn = inputs.Length > 0 ? inputs[0] as RuntimeSequenceTensor : null;
        var posTensor = inputs.Length > 1 ? inputs[1] as RuntimeTensor : null;

        if (seqIn is null)
            return [new RuntimeSequenceTensor { DType = DType.Invalid }];

        long? pos = null;
        if (posTensor?.IntData is { Length: > 0 } idx) pos = idx[0];

        var newCount = seqIn.Count is long c ? Math.Max(0, c - 1) : (long?)null;
        var dtype = seqIn.DType;

        if (seqIn.Tensors is { } srcTensors)
        {
            var list = new List<RuntimeTensor>(srcTensors);
            int eraseAt;
            if (pos is long p) eraseAt = (int)(p < 0 ? p + list.Count : p);
            else eraseAt = list.Count - 1;
            if (eraseAt >= 0 && eraseAt < list.Count) list.RemoveAt(eraseAt);
            return [new RuntimeSequenceTensor { DType = dtype, Count = list.Count, Tensors = list.ToImmutableArray() }];
        }

        return [new RuntimeSequenceTensor
        {
            DType = dtype,
            Count = newCount,
            TemplateTensor = SequenceHelpers.EffectiveTemplate(seqIn),
        }];
    }
}
