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
/// ONNX <c>SequenceAt</c>: retrieves the tensor at the given position from a sequence. When
/// the sequence is concrete and the position is a statically known scalar we return the
/// specific element; otherwise we fall back to the sequence's template tensor.
/// </summary>
internal sealed class SequenceAtOp : QuickOp
{
    public override string OpCode => OpCodes.SEQUENCE_AT;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seqIn = inputs.Length > 0 ? inputs[0] as RuntimeSequenceTensor : null;
        var posTensor = inputs.Length > 1 ? inputs[1] as RuntimeTensor : null;

        if (seqIn is null)
            return [RuntimeTensorFactory.Create(DType.Invalid, null)];

        long? pos = null;
        if (posTensor?.IntData is { Length: > 0 } idx) pos = idx[0];

        if (seqIn.Tensors is { } tensors && pos is long p)
        {
            var idxInt = (int)(p < 0 ? p + tensors.Length : p);
            if (idxInt >= 0 && idxInt < tensors.Length)
                return [tensors[idxInt]];
        }

        var template = SequenceHelpers.EffectiveTemplate(seqIn)
            ?? RuntimeTensorFactory.Create(seqIn.DType, null);
        return [template];
    }
}
