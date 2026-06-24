using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// QEE kernel for ONNX <c>MatMulInteger</c>. Shares the numpy-style shape semantics of
/// <see cref="MatMulOp"/> (incl. the 1-D edge cases); output is int32 per spec. Concrete
/// values are computed when both inputs carry data and the zero points (optional inputs
/// 3 and 4) are absent or known scalars; 1-D per-row/per-column zero points block value
/// computation rather than guessing.
/// </summary>
internal sealed class MatMulIntegerOp : QuickOp
{
    public override string OpCode => OpCodes.MATMUL_INTEGER;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var a = inputs.Length > 0 ? inputs[0] : null;
        var b = inputs.Length > 1 ? inputs[1] : null;
        var aZpIn = inputs.Length > 2 ? inputs[2] : null;
        var bZpIn = inputs.Length > 3 ? inputs[3] : null;

        var shape = MatMulHelpers.InferShape(a?.Shape, b?.Shape);
        var rt = RuntimeTensorFactory.Create(DType.Int32, shape);
        if (shape is null || !RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
            return [rt];

        if (a!.IntData is { } ai && b!.IntData is { } bi
            && TryGetScalarZeroPoint(aZpIn, out var aZp)
            && TryGetScalarZeroPoint(bZpIn, out var bZp))
        {
            var buf = new long[shape.Count];
            if (MatMulHelpers.Accumulate(a.Shape!.Dims, b.Shape!.Dims,
                    (iA, iB, io) => buf[io] += (ai[iA] - aZp) * (bi[iB] - bZp)))
                return [rt with { IntData = ImmutableArray.Create(buf) }];
        }
        return [rt];
    }

    /// <summary>
    /// Resolves an optional zero-point input to a scalar value. Absent input → 0. A
    /// connected input must have a known single-element value; per-row/per-column 1-D
    /// zero points (or unknown values) return false so no values are emitted.
    /// </summary>
    private static bool TryGetScalarZeroPoint(RuntimeTensor? zp, out long value)
    {
        value = 0;
        if (zp is null) return true;
        if (zp.IntData is { Length: 1 } data) { value = data[0]; return true; }
        return false;
    }
}
