using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Rng;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE implementation of SHRK_RNG_SPLIT: computes the child key host-side via the same
/// Threefry bijection the in-graph function uses, so key chains resolve to REAL values in
/// QEE (bit-exact with every other execution path — a counter-based RNG is ordinary
/// integer math, so QEE finally sees production randomness keys).
/// </summary>
internal sealed class ShrkRngSplitOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_SPLIT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var key = inputs[0];
        var index = inputs[1];
        var outShape = new Shape([2]);

        long[]? keyWords = key?.IntData is { Length: 2 } k ? [k[0], k[1]] : null;

        if (keyWords is not null && index?.IntData is { Length: 1 } i)
        {
            var (x0, x1) = Threefry2x32.Bijection(
                (uint)(ulong)i[0], 0u, (uint)(ulong)keyWords[0], (uint)(ulong)keyWords[1]);
            return [RuntimeTensorFactory.Create(DType.Int64, outShape)
                with { IntData = [x0, x1] }];
        }

        // Key or index value unavailable: shape-only result.
        return [RuntimeTensorFactory.Create(DType.Int64, outShape)];
    }
}

/// <summary>
/// QEE implementation of SHRK_RNG_UNIFORM / SHRK_RNG_NORMAL: shape/rank propagation only
/// (like the unkeyed random ops). Values are deliberately not computed in QEE: the normal
/// transform uses float transcendentals whose ULP behavior may differ from the execution
/// backend, and QEE-vs-backend comparisons must not flake. The split chain (integer-exact)
/// does carry real values — see <see cref="ShrkRngSplitOp"/>.
/// </summary>
internal abstract class ShrkRngDrawOpBase : QuickOp
{
    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var shapeInput = inputs[2];
        Shape? shape = shapeInput?.IntData is { } s && s.All(d => d >= 0) ? new Shape(s.ToArray()) : null;
        var rt = RuntimeTensorFactory.Create(DType.Float32, shape);
        // Shape values unknown but the shape input's own 1-D extent gives the output rank.
        if (shape is null && shapeInput?.Shape?.Dims is { Length: 1 } sd)
            rt = rt with { Rank = (int)sd[0], MaxRank = (int)sd[0] };
        return [rt];
    }
}

/// <summary>
/// QEE implementation of SHRK_RNG_KEY_PARAM (a feed site's key entity): shape/rank propagation
/// only — [N, 2] from the site's iteration counts. Deliberately NOT the value even when
/// materialized: exposing it would let constant folding replace the entity with a plain
/// CONSTANT mid-pipeline, stripping the structural metadata (site id, realized ids, counts)
/// the export lowering and re-binding read. The entity lowers to a CONSTANT exactly once,
/// at ONNX prep (see FastLowerRandomOps).
/// </summary>
internal sealed class ShrkRngKeyOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_KEY_PARAM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var counts = attrs.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngIterCounts);
        long total = 1;
        if (counts is not null) foreach (var c in counts) total *= c;
        return [RuntimeTensorFactory.Create(DType.Int64, new Shape(total, 2))];
    }
}

internal sealed class ShrkRngUniformOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_UNIFORM;
}

internal sealed class ShrkRngNormalOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_NORMAL;
}
