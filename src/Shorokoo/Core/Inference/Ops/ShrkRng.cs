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

internal sealed class ShrkRngUniformOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_UNIFORM;
}

internal sealed class ShrkRngNormalOp : ShrkRngDrawOpBase
{
    public override string OpCode => InternalOpCodes.SHRK_RNG_NORMAL;
}
