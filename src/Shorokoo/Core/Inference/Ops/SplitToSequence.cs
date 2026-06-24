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
/// QEE kernel for ONNX <c>SplitToSequence</c>. Per the opset-21 spec:
/// <list type="bullet">
///   <item>Without the <c>split</c> input, the tensor is split into size-1 chunks along
///     <c>axis</c>; <c>keepdims</c> (default 1) decides whether each chunk keeps the size-1
///     axis or has it squeezed. The sequence length equals the axis dim.</item>
///   <item>With a scalar <c>split</c> = S, chunks of size S (the LAST chunk may be smaller);
///     <c>keepdims</c> is IGNORED — elements keep the input rank.</item>
///   <item>With a 1-D <c>split</c>, per-chunk sizes that must sum to the axis dim.</item>
/// </list>
/// When the chunk sizes are statically known the concrete per-element tensors (with sliced
/// values when the input data is known) are produced, so downstream <c>SequenceAt</c> /
/// <c>ConcatFromSequence</c> / <c>SequenceLength</c> keep concrete values. A <c>split</c>
/// input that is connected but value-unknown degrades to a template whose shape is unknown
/// (rank + per-dim upper bound only — never a fabricated even split).
/// </summary>
internal sealed class SplitToSequenceOp : QuickOp
{
    public override string OpCode => OpCodes.SPLIT_TO_SEQUENCE;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs.Length > 0 ? inputs[0] as RuntimeTensor : null;
        var splitIn = inputs.Length > 1 ? inputs[1] as RuntimeTensor : null;
        bool splitConnected = inputs.Length > 1 && inputs[1] is not null;
        var dtype = x?.DType ?? DType.Float32;

        if (x?.Shape is null)
            return [new RuntimeSequenceTensor { DType = dtype, TemplateTensor = RuntimeTensorFactory.Create(dtype, null) }];

        var dims = x.Shape.Dims;
        var rank = dims.Length;
        var axisAttr = attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? 0;
        var axis = (int)(axisAttr < 0 ? axisAttr + rank : axisAttr);
        if (axis < 0 || axis >= rank)
            return [new RuntimeSequenceTensor { DType = dtype, TemplateTensor = RuntimeTensorFactory.Create(dtype, null) }];
        var axisDim = dims[axis];

        long[]? sizes = null;        // per-chunk sizes along `axis`, when statically known
        bool squeezeAxis = false;    // only possible on the no-split path with keepdims=0

        if (!splitConnected)
        {
            // No split input: size-1 chunks; keepdims decides squeeze vs keep.
            var keepDims = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrKeepdims, true);
            squeezeAxis = !keepDims;
            sizes = new long[axisDim];
            for (int i = 0; i < sizes.Length; i++) sizes[i] = 1;
        }
        else if (splitIn?.IntData is { } splitData)
        {
            // keepdims is IGNORED when split is provided (spec) — elements keep the rank.
            var splitRank = splitIn.Shape?.Dims.Length ?? splitIn.Rank;
            if (splitRank == 0 || (splitRank is null && splitData.Length == 1))
            {
                var s = splitData[0];
                if (s > 0)
                {
                    var n = (axisDim + s - 1) / s;
                    sizes = new long[n];
                    long remaining = axisDim;
                    for (int i = 0; i < n; i++) { sizes[i] = Math.Min(s, remaining); remaining -= sizes[i]; }
                }
            }
            else if (splitRank == 1)
            {
                sizes = splitData.ToArray();
                if (sizes.Any(sz => sz < 0) || sizes.Sum() != axisDim) sizes = null;
            }
        }

        if (sizes is null)
        {
            // split connected but value-unknown (or invalid): per-element axis dim is
            // unknowable. Template keeps rank + a per-dim upper bound (each chunk's axis
            // extent is at most the full axis dim); the count is unknown.
            var template = RuntimeTensorFactory.Create(dtype, null) with
            {
                MaxShape = new Shape(dims.ToArray()),
                Rank = rank,
                MaxRank = rank,
            };
            return [new RuntimeSequenceTensor { DType = dtype, TemplateTensor = template }];
        }

        // Too many chunks to materialize per-element summaries: keep the count plus the
        // strongest template (exact shape when chunks are uniform, upper bound otherwise).
        const int MaxConcreteElements = 256;
        if (sizes.Length > MaxConcreteElements)
        {
            var uniform = sizes.All(sz => sz == sizes[0]);
            long[] tDims;
            if (squeezeAxis)
            {
                tDims = new long[rank - 1];
                int ti = 0;
                for (int d = 0; d < rank; d++) if (d != axis) tDims[ti++] = dims[d];
            }
            else
            {
                tDims = dims.ToArray();
                tDims[axis] = uniform ? sizes[0] : sizes.Max();
            }
            var capTemplate = RuntimeTensorFactory.Create(dtype, uniform || squeezeAxis ? new Shape(tDims) : null) with
            {
                MaxShape = new Shape(tDims),
                Rank = tDims.Length,
                MaxRank = tDims.Length,
            };
            return [new RuntimeSequenceTensor { DType = dtype, Count = sizes.Length, TemplateTensor = capTemplate }];
        }

        // Concrete chunk sizes: produce the per-element tensors (with values when known).
        var elements = new RuntimeTensor[sizes.Length];

        long outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        long innerCount = 1;
        for (int d = axis + 1; d < rank; d++) innerCount *= dims[d];
        long inAxisStride = axisDim * innerCount;

        long offset = 0;
        for (int i = 0; i < sizes.Length; i++)
        {
            long[] elemDims;
            if (squeezeAxis)
            {
                elemDims = new long[rank - 1];
                int di = 0;
                for (int d = 0; d < rank; d++) if (d != axis) elemDims[di++] = dims[d];
            }
            else
            {
                elemDims = dims.ToArray();
                elemDims[axis] = sizes[i];
            }
            var rt = RuntimeTensorFactory.Create(dtype, new Shape(elemDims));

            if (RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements))
            {
                long chunkLen = sizes[i] * innerCount;
                long total = outerCount * chunkLen;
                if (x.IntData is { } id)
                {
                    var buf = new long[total];
                    for (long outer = 0; outer < outerCount; outer++)
                        for (long e = 0; e < chunkLen; e++)
                            buf[outer * chunkLen + e] = id[(int)(outer * inAxisStride + offset * innerCount + e)];
                    rt = rt with { IntData = ImmutableArray.Create(buf) };
                }
                else if (x.FloatData is { } fd)
                {
                    var buf = new float[total];
                    for (long outer = 0; outer < outerCount; outer++)
                        for (long e = 0; e < chunkLen; e++)
                            buf[outer * chunkLen + e] = fd[(int)(outer * inAxisStride + offset * innerCount + e)];
                    rt = rt with { FloatData = ImmutableArray.Create(buf) };
                }
                else if (x.BoolData is { } bd)
                {
                    var buf = new bool[total];
                    for (long outer = 0; outer < outerCount; outer++)
                        for (long e = 0; e < chunkLen; e++)
                            buf[outer * chunkLen + e] = bd[(int)(outer * inAxisStride + offset * innerCount + e)];
                    rt = rt with { BoolData = ImmutableArray.Create(buf) };
                }
            }
            elements[i] = rt;
            offset += sizes[i];
        }

        return [new RuntimeSequenceTensor
        {
            DType = dtype,
            Count = elements.Length,
            Tensors = ImmutableArray.Create(elements),
        }];
    }

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => System.Array.Empty<RuntimeTensor>();
}
