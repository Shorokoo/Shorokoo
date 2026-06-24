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
/// ONNX <c>ConcatFromSequence</c>: concatenates all tensors in a sequence along the given
/// axis. With <c>new_axis</c> true, a new leading dim of size = element count is inserted
/// instead.
/// </summary>
internal sealed class ConcatFromSequenceOp : QuickOp
{
    public override string OpCode => OpCodes.CONCAT_FROM_SEQUENCE;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seq = inputs.Length > 0 ? inputs[0] as RuntimeSequenceTensor : null;
        var axis = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? 0);
        var newAxis = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrNewAxis, false);

        if (seq is null)
            return [RuntimeTensorFactory.Create(DType.Invalid, null)];

        var dtype = seq.DType;
        var template = SequenceHelpers.EffectiveTemplate(seq);
        var innerShape = template?.Shape;

        // new_axis: output rank = inner rank + 1; the new dim = sequence count (when known).
        if (newAxis)
        {
            if (innerShape is null)
            {
                var unknown = RuntimeTensorFactory.Create(dtype, null);
                if (template?.Rank is int tr)
                    unknown = unknown with { Rank = tr + 1, MaxRank = tr + 1 };
                return [unknown];
            }
            var dims = new long[innerShape.Dims.Length + 1];
            var normalizedAxis = axis < 0 ? axis + dims.Length : axis;
            normalizedAxis = Math.Clamp(normalizedAxis, 0, dims.Length - 1);
            int di = 0;
            for (int d = 0; d < dims.Length; d++)
            {
                if (d == normalizedAxis) dims[d] = seq.Count ?? -1;
                else { dims[d] = innerShape.Dims[di]; di++; }
            }
            var shape = dims.Any(v => v < 0) ? null : new Shape(dims);
            var rt = RuntimeTensorFactory.Create(dtype, shape);
            if (shape is null)
                return [rt with { Rank = dims.Length, MaxRank = dims.Length }];

            // Value path: stacking concrete elements == concatenating them after unsqueezing
            // the new axis into each element's shape (same flat data, one chunk per element).
            if (seq.Tensors is { } stackSrc && stackSrc.Length > 0
                && stackSrc.All(t => t.Shape is not null && t.Shape.Equals(innerShape)))
            {
                var unsqueezed = new RuntimeTensor?[stackSrc.Length];
                var elemDims = new long[dims.Length];
                Array.Copy(dims, elemDims, dims.Length);
                elemDims[normalizedAxis] = 1;
                for (int i = 0; i < stackSrc.Length; i++)
                    unsqueezed[i] = stackSrc[i] with { Shape = new Shape(elemDims) };
                rt = ConcatOp.WithConcatValues(rt, unsqueezed, normalizedAxis, maxDataElements);
            }
            return [rt];
        }

        // Concat along existing axis: output rank = inner rank; the axis dim is the sum of
        // per-element sizes along that axis. Concrete elements may legally disagree on the
        // axis dim (only the non-axis dims must match), so the concrete path derives the
        // output shape from the per-element shapes rather than the (weaker) shared template.
        if (seq.Tensors is { } tensors && tensors.Length > 0
            && tensors.All(t => t.Shape is not null))
        {
            var rank = tensors[0].Shape!.Dims.Length;
            if (rank == 0) // rank-0 elements can't be concatenated along an axis (invalid per spec)
                return [RuntimeTensorFactory.Create(dtype, null)];
            var cAxis = axis < 0 ? axis + rank : axis;
            cAxis = Math.Clamp(cAxis, 0, rank - 1);
            var cDims = tensors[0].Shape!.Dims.ToArray();
            long sum = 0;
            bool consistent = true;
            foreach (var t in tensors)
            {
                var td = t.Shape!.Dims;
                if (td.Length != rank) { consistent = false; break; }
                for (int d = 0; d < rank; d++)
                    if (d != cAxis && td[d] != cDims[d]) { consistent = false; break; }
                if (!consistent) break;
                sum += td[cAxis];
            }
            if (!consistent)
                return [RuntimeTensorFactory.Create(dtype, null)];
            cDims[cAxis] = sum;
            var concatRt = RuntimeTensorFactory.Create(dtype, new Shape(cDims));
            var srcs = new RuntimeTensor?[tensors.Length];
            for (int i = 0; i < tensors.Length; i++) srcs[i] = tensors[i];
            return [ConcatOp.WithConcatValues(concatRt, srcs, cAxis, maxDataElements)];
        }

        if (innerShape is null)
        {
            var unknown = RuntimeTensorFactory.Create(dtype, null);
            if (template?.Rank is int tr)
                unknown = unknown with { Rank = tr, MaxRank = tr };
            return [unknown];
        }

        var outDims = innerShape.Dims.ToArray();
        if (outDims.Length == 0)
            return [RuntimeTensorFactory.Create(dtype, null)];
        var normAxis = axis < 0 ? axis + outDims.Length : axis;
        normAxis = Math.Clamp(normAxis, 0, outDims.Length - 1);
        outDims[normAxis] = seq.Count is long c ? c * outDims[normAxis] : -1;

        if (outDims.Any(v => v < 0))
            return [RuntimeTensorFactory.Create(dtype, null) with
            {
                Rank = outDims.Length,
                MaxRank = outDims.Length,
            }];
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}
