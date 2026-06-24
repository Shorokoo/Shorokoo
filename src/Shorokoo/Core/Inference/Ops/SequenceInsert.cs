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
/// ONNX <c>SequenceInsert</c>: returns a new sequence with a tensor inserted at the given
/// position (or appended when position is absent). When the sequence is concrete and the
/// position is statically known we produce a concrete result; otherwise we fall back to a
/// template-mode sequence.
/// </summary>
internal sealed class SequenceInsertOp : QuickOp
{
    public override string OpCode => OpCodes.SEQUENCE_INSERT;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seqIn = inputs.Length > 0 ? inputs[0] as RuntimeSequenceTensor : null;
        var tensor = inputs.Length > 1 ? inputs[1] as RuntimeTensor : null;
        var posTensor = inputs.Length > 2 ? inputs[2] as RuntimeTensor : null;

        if (seqIn is null)
            return [new RuntimeSequenceTensor { DType = tensor?.DType ?? DType.Invalid, TemplateTensor = tensor }];

        long? pos = null;
        if (posTensor?.IntData is { Length: > 0 } idx) pos = idx[0];

        var newCount = seqIn.Count is long c ? c + 1 : (long?)null;
        var dtype = seqIn.DType.IsValid ? seqIn.DType : (tensor?.DType ?? DType.Invalid);

        if (seqIn.Tensors is { } srcTensors && tensor is not null)
        {
            var list = new List<RuntimeTensor>(srcTensors);
            var insertAt = pos is long p
                ? (int)(p < 0 ? p + list.Count : Math.Min(p, list.Count))
                : list.Count;
            insertAt = Math.Clamp(insertAt, 0, list.Count);
            list.Insert(insertAt, tensor);
            return [new RuntimeSequenceTensor { DType = dtype, Count = list.Count, Tensors = list.ToImmutableArray() }];
        }

        var template = SequenceHelpers.MergeTemplates(SequenceHelpers.EffectiveTemplate(seqIn), tensor);
        return [new RuntimeSequenceTensor { DType = dtype, Count = newCount, TemplateTensor = template }];
    }
}
