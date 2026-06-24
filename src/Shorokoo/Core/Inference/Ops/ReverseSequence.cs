using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>ReverseSequence</c>. Output has the same shape and dtype as the
/// input. Values: for each batch slice b (along <c>batch_axis</c>, spec default 1) the first
/// <c>sequence_lens[b]</c> entries along <c>time_axis</c> (spec default 0) are reversed and
/// the rest pass through. batch_axis/time_axis must be 0 or 1 and distinct (rank ≥ 2 per
/// spec); anything else drops the value computation but keeps the pass-through shape.
/// </summary>
internal sealed class ReverseSequenceOp : QuickOp
{
    public override string OpCode => OpCodes.REVERSE_SEQUENCE;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var lens = inputs.Length > 1 ? inputs[1] : null;
        var dtype = x?.DType ?? DType.Float32;
        var rt = RuntimeTensorFactory.Create(dtype, x?.Shape);
        if (x?.Shape is null) return [rt];

        // Spec defaults: batch_axis = 1, time_axis = 0; both must be 0 or 1 and distinct.
        var batchAxis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrBatchAxis, 1);
        var timeAxis = (int)AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrTimeAxis, 0);
        var dims = x.Shape.Dims;
        if (dims.Length < 2
            || batchAxis is not (0 or 1) || timeAxis is not (0 or 1) || batchAxis == timeAxis
            || lens?.IntData is not { } lenData || lenData.Length != dims[batchAxis]
            || (x.FloatData is null && x.IntData is null && x.BoolData is null)
            || !RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            return [rt];

        var rank = dims.Length;
        var strides = new long[rank];
        long ts = 1;
        for (int d = rank - 1; d >= 0; d--) { strides[d] = ts; ts *= dims[d]; }
        long total = ts;

        // Map every output flat index to its source flat index.
        var srcOf = new int[total];
        var idx = new long[rank];
        for (long flat = 0; flat < total; flat++)
        {
            long rem = flat;
            for (int d = 0; d < rank; d++) { idx[d] = rem / strides[d]; rem -= idx[d] * strides[d]; }
            var len = lenData[(int)idx[batchAxis]];
            if (len < 1 || len > dims[timeAxis]) return [rt]; // invalid lens — don't guess
            var t = idx[timeAxis];
            var srcT = t < len ? len - 1 - t : t;
            srcOf[flat] = (int)(flat + (srcT - t) * strides[timeAxis]);
        }

        if (x.FloatData is { } fd)
        {
            var buf = new float[total];
            for (long i = 0; i < total; i++) buf[i] = fd[srcOf[i]];
            return [rt with { FloatData = ImmutableArray.Create(buf) }];
        }
        if (x.IntData is { } id)
        {
            var buf = new long[total];
            for (long i = 0; i < total; i++) buf[i] = id[srcOf[i]];
            return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        if (x.BoolData is { } bd)
        {
            var buf = new bool[total];
            for (long i = 0; i < total; i++) buf[i] = bd[srcOf[i]];
            return [rt with { BoolData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }
}
