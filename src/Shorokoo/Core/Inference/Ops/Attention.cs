using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>Attention</c> (opset 23+). Shape/dtype inference only
/// (value computation is intentionally not implemented — values come from the ORT
/// execution path). Two node variants, distinguished by the internal
/// has-optional-outputs flag (see Definitions.AC.cs):
/// <list type="bullet">
///   <item>basic form (flag 0) — inputs Q, K, V, attn_mask?, nonpad_kv_seqlen?;
///         single output Y;</item>
///   <item>KV-cache form (flag 1) — inputs Q, K, V, attn_mask?, past_key?,
///         past_value?; outputs Y, present_key, present_value.</item>
/// </list>
/// Y shapes per spec:
///   4-D: (batch, q_num_heads, q_seq, v_head_size) — Q's shape with V's last dim;
///   3-D: (batch, q_seq, q_num_heads * v_hidden / kv_num_heads) — needs the
///        q_num_heads / kv_num_heads attributes.
/// present_key = (batch, kv_heads, past_seq + kv_seq, head); present_value the same
/// with V's last dim. Any missing piece degrades to rank-only (never guessed dims).
/// </summary>
internal sealed class AttentionOp : QuickOp
{
    public override string OpCode => OpCodes.ATTENTION;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var q = inputs.Length > 0 ? inputs[0] : null;
        var k = inputs.Length > 1 ? inputs[1] : null;
        var v = inputs.Length > 2 ? inputs[2] : null;
        var hasKvCacheOutputs = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.InternalAttrHasOptionalOutputs);
        var pastKey = hasKvCacheOutputs && inputs.Length > 4 ? inputs[4] : null;
        var pastValue = hasKvCacheOutputs && inputs.Length > 5 ? inputs[5] : null;

        var yDType = q?.DType ?? DType.Float32;
        var y = InferY(q, k, v, attrs, yDType);

        if (!hasKvCacheOutputs)
            return [y];

        var keyDType = k?.DType ?? yDType;
        var valueDType = v?.DType ?? yDType;
        return [
            y,
            InferPresent(k, pastKey, keyDType),
            InferPresent(v, pastValue, valueDType),
        ];
    }

    private static RuntimeTensor InferY(RuntimeTensor? q, RuntimeTensor? k, RuntimeTensor? v,
        OnnxCSharpAttributes attrs, DType dtype)
    {
        if (q?.Shape is null)
            return RuntimeTensorFactory.CreateRankOnly(dtype, q?.Rank);

        var qDims = q.Shape.Dims;
        if (qDims.Length == 4)
        {
            // Y = (batch, q_num_heads, q_seq, v_head_size) — V's last dim.
            if (v?.Shape is { } vShape && vShape.Dims.Length == 4)
                return RuntimeTensorFactory.Create(dtype, new Shape([qDims[0], qDims[1], qDims[2], vShape.Dims[3]]));
            return RuntimeTensorFactory.CreateRankOnly(dtype, 4);
        }
        if (qDims.Length == 3)
        {
            // Y = (batch, q_seq, q_num_heads * v_head_size) with
            // v_head_size = v_hidden / kv_num_heads — needs both head-count attributes.
            var qHeads = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrQNumHeads, 0);
            var kvHeads = AttrAccess.GetLong(attrs, OnnxOpAttributeNames.AttrKvNumHeads, 0);
            if (qHeads > 0 && kvHeads > 0 && v?.Shape is { } vShape && vShape.Dims.Length == 3
                && vShape.Dims[2] % kvHeads == 0)
                return RuntimeTensorFactory.Create(dtype, new Shape([qDims[0], qDims[1], qHeads * (vShape.Dims[2] / kvHeads)]));
            return RuntimeTensorFactory.CreateRankOnly(dtype, 3);
        }
        return RuntimeTensorFactory.CreateRankOnly(dtype, qDims.Length);
    }

    /// <summary>present = source-cache shape with the sequence dim grown by the incoming
    /// K/V length: (batch, kv_heads, past_seq + kv_seq, head). Exact only for the 4-D
    /// form with both shapes known; otherwise rank-only rank 4 (the spec fixes the
    /// present outputs to 4-D).</summary>
    private static RuntimeTensor InferPresent(RuntimeTensor? incoming, RuntimeTensor? past, DType dtype)
    {
        if (incoming?.Shape is { } inc && inc.Dims.Length == 4)
        {
            if (past is null)
                return RuntimeTensorFactory.Create(dtype, inc);
            if (past.Shape is { } p && p.Dims.Length == 4)
                return RuntimeTensorFactory.Create(dtype,
                    new Shape([inc.Dims[0], inc.Dims[1], p.Dims[2] + inc.Dims[2], inc.Dims[3]]));
        }
        return RuntimeTensorFactory.CreateRankOnly(dtype, 4);
    }
}
