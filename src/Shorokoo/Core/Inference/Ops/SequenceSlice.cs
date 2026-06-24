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
/// Shorokoo-internal <c>#SequenceSlice#</c>: slices a sub-sequence from <c>start</c>
/// (inclusive) to <c>end</c> (exclusive). When the input is concrete and both bounds are known
/// scalars we return the exact sub-sequence; otherwise template mode with the input's element
/// template and an unknown count.
/// </summary>
internal sealed class SequenceSliceOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SEQUENCE_SLICE;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seqIn = inputs.Length > 0 ? inputs[0] as RuntimeSequenceTensor : null;
        var startT = inputs.Length > 1 ? inputs[1] as RuntimeTensor : null;
        var endT = inputs.Length > 2 ? inputs[2] as RuntimeTensor : null;

        if (seqIn is null)
            return [new RuntimeSequenceTensor { DType = DType.Invalid }];

        var dtype = seqIn.DType;
        long? startV = startT?.IntData is { Length: > 0 } s ? s[0] : null;
        long? endV = endT?.IntData is { Length: > 0 } e ? e[0] : null;

        if (seqIn.Tensors is { } tensors && startV is long sv && endV is long ev)
        {
            var len = tensors.Length;
            var lo = (int)(sv < 0 ? sv + len : sv);
            var hi = (int)(ev < 0 ? ev + len : ev);
            lo = Math.Clamp(lo, 0, len);
            hi = Math.Clamp(hi, lo, len);
            var sub = System.Collections.Immutable.ImmutableArray.Create(tensors, lo, hi - lo);
            return [new RuntimeSequenceTensor { DType = dtype, Count = sub.Length, Tensors = sub }];
        }

        return [new RuntimeSequenceTensor
        {
            DType = dtype,
            Count = null,
            TemplateTensor = SequenceHelpers.EffectiveTemplate(seqIn),
        }];
    }
}
