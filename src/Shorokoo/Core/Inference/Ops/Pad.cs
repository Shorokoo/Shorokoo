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
/// Implements the ONNX <c>Pad</c> op. Computes both the output shape and concrete data
/// for all four modes (<c>constant</c>, <c>edge</c>, <c>reflect</c>, <c>wrap</c>) on
/// <see cref="DType.Int64"/>, <see cref="DType.Float32"/>, and <see cref="DType.Bool"/>
/// inputs, subject to the usual <see cref="RuntimeTensorFactory.ShouldStoreData"/>
/// element-count limit.
///
/// Producing concrete data lets <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastFoldConstants"/>
/// chase the trainable-param-index lookup chain
/// (<c>Shape → ReduceProd → Sub → Unsqueeze → Concat → Pad → Mul → ReduceSum</c>) emitted
/// by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastConvertTrainableParamIdRefToTrainableParam"/>
/// all the way down to a single <c>CONSTANT</c>, which in turn lets
/// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastFoldSequences"/> short-circuit
/// the per-DType <c>SEQUENCE_AT(SEQUENCE_CONSTRUCT, idx)</c> trainable-param lookup so
/// unused params (e.g. the eliminated branch of a folded IF) become unreachable and are
/// swept by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastProcessorHelper.RemoveUnreachableNodes"/>.
/// </summary>
internal sealed class PadOp : QuickOp
{
    public override string OpCode => OpCodes.PAD;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dtype = x?.DType ?? DType.Float32;
        if (x?.Shape is null || inputs.Length <= 1 || inputs[1]?.IntData is not { } pads)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var rank = x.Shape.Dims.Length;
        var inDims = x.Shape.Dims;
        var outDims = inDims.ToArray();

        // axes connected but with unknown values: which dims the pads apply to is
        // unknowable, so degrade to an unknown shape rather than assume all dims.
        if (inputs.Length > 3 && inputs[3] is not null && inputs[3]!.IntData is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        // Per-dim begin pad amount, in input coordinate space.
        // (The output element at out-coord oc maps to input coord ic = oc - beginByDim[d].)
        var beginByDim = new long[rank];
        var axes = inputs.Length > 3 ? inputs[3]?.IntData : null;
        if (axes is null)
        {
            if (pads.Length < 2 * rank) return [RuntimeTensorFactory.Create(dtype, null)];
            for (int d = 0; d < rank; d++)
            {
                beginByDim[d] = pads[d];
                outDims[d] += pads[d] + pads[d + rank];
            }
        }
        else
        {
            var axesArr = axes.Value;
            if (pads.Length < 2 * axesArr.Length) return [RuntimeTensorFactory.Create(dtype, null)];
            for (int i = 0; i < axesArr.Length; i++)
            {
                var a = (int)(axesArr[i] < 0 ? axesArr[i] + rank : axesArr[i]);
                if (a < 0 || a >= rank) return [RuntimeTensorFactory.Create(dtype, null)];
                beginByDim[a] = pads[i];
                outDims[a] += pads[i] + pads[i + axesArr.Length];
            }
        }

        // Reject negative output dims (negative pads exceed input extent).
        for (int d = 0; d < rank; d++)
            if (outDims[d] < 0) return [RuntimeTensorFactory.Create(dtype, null)];

        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (!RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements))
            return [rt];

        var mode = ResolvePadMode(attrs);

        // Constant pad value (used only when mode == Constant). When the optional
        // `constant_value` input is missing, ONNX defaults to 0 / 0.0 / false. When it is
        // CONNECTED but its value is unknown at QEE time, the padded values are unknowable —
        // block value computation instead of silently padding with 0.
        long padInt = 0;
        float padFloat = 0f;
        bool padBool = false;
        if (mode == PadMode.Constant && inputs.Length > 2 && inputs[2] is { } pv)
        {
            if (pv.IntData is { Length: > 0 } iv) padInt = iv[0];
            else if (pv.FloatData is { Length: > 0 } fv) padFloat = fv[0];
            else if (pv.BoolData is { Length: > 0 } bv) padBool = bv[0];
            else return [rt];
        }

        // Strides for input and output (row-major).
        long inCount = 1;
        long outCount = 1;
        var inStrides = new long[rank];
        var outStrides = new long[rank];
        long sIn = 1, sOut = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            inStrides[i] = sIn; sIn *= inDims[i];
            outStrides[i] = sOut; sOut *= outDims[i];
            inCount *= inDims[i];
            outCount *= outDims[i];
        }

        if (x.IntData is { } xInt && xInt.Length == inCount)
        {
            var outBuf = new long[outCount];
            FillPad(xInt, outBuf, inDims, outDims, inStrides, outStrides, beginByDim, rank, mode, padInt);
            return [rt with { IntData = ImmutableArray.Create(outBuf) }];
        }
        if (x.FloatData is { } xFloat && xFloat.Length == inCount)
        {
            var outBuf = new float[outCount];
            FillPad(xFloat, outBuf, inDims, outDims, inStrides, outStrides, beginByDim, rank, mode, padFloat);
            return [rt with { FloatData = ImmutableArray.Create(outBuf) }];
        }
        if (x.BoolData is { } xBool && xBool.Length == inCount)
        {
            var outBuf = new bool[outCount];
            FillPad(xBool, outBuf, inDims, outDims, inStrides, outStrides, beginByDim, rank, mode, padBool);
            return [rt with { BoolData = ImmutableArray.Create(outBuf) }];
        }

        return [rt];
    }

    /// <summary>
    /// Walks every output element, decoding its multi-dim coord, mapping each per-dim
    /// coord back to an input coord according to <paramref name="mode"/>, and either
    /// copying the input element or writing <paramref name="padValue"/> when the coord
    /// falls outside the input under <see cref="PadMode.Constant"/>.
    /// </summary>
    private static void FillPad<T>(
        ImmutableArray<T> src, T[] dst,
        long[] inDims, long[] outDims, long[] inStrides, long[] outStrides,
        long[] beginByDim, int rank, PadMode mode, T padValue)
    {
        long outCount = dst.LongLength;
        var outIdx = new long[rank];
        for (long o = 0; o < outCount; o++)
        {
            // Decode flat output index → per-dim coords.
            long rem = o;
            for (int d = 0; d < rank; d++)
            {
                outIdx[d] = rem / outStrides[d];
                rem -= outIdx[d] * outStrides[d];
            }
            // Map each output coord back to an input coord per the chosen mode.
            bool useInput = true;
            long inFlat = 0;
            for (int d = 0; d < rank; d++)
            {
                long ic = outIdx[d] - beginByDim[d];
                long inDim = inDims[d];
                switch (mode)
                {
                    case PadMode.Constant:
                        if (ic < 0 || ic >= inDim) useInput = false;
                        break;
                    case PadMode.Edge:
                        if (inDim <= 0) { useInput = false; break; }
                        if (ic < 0) ic = 0;
                        else if (ic >= inDim) ic = inDim - 1;
                        break;
                    case PadMode.Reflect:
                        // Reflect about the edges without repeating the boundary element.
                        // Period = 2 * (inDim - 1); coord cycles 0,1,…,N-1,N-2,…,1 then repeats.
                        if (inDim <= 0) { useInput = false; break; }
                        if (inDim == 1) { ic = 0; break; }
                        long period = 2 * (inDim - 1);
                        ic %= period;
                        if (ic < 0) ic += period;
                        if (ic >= inDim) ic = period - ic;
                        break;
                    case PadMode.Wrap:
                        if (inDim <= 0) { useInput = false; break; }
                        ic %= inDim;
                        if (ic < 0) ic += inDim;
                        break;
                }
                if (!useInput) break;
                inFlat += ic * inStrides[d];
            }
            dst[o] = useInput ? src[(int)inFlat] : padValue;
        }
    }

    /// <summary>
    /// Tolerantly resolves the mode attribute. <see cref="OnnxCSharpAttributes"/>
    /// stores enum-typed attributes as the original .NET enum value, while wire-form
    /// graphs round-trip through the proto representation as a lowercase string —
    /// accept either, defaulting to <see cref="PadMode.Constant"/>.
    /// </summary>
    private static PadMode ResolvePadMode(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrMode)) return PadMode.Constant;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrMode);
        if (obj is null) return PadMode.Constant;
        if (obj is PadMode pm) return pm;
        if (obj is string s)
        {
            if (s.Equals("constant", System.StringComparison.OrdinalIgnoreCase)) return PadMode.Constant;
            if (s.Equals("reflect", System.StringComparison.OrdinalIgnoreCase)) return PadMode.Reflect;
            if (s.Equals("edge", System.StringComparison.OrdinalIgnoreCase)) return PadMode.Edge;
            if (s.Equals("wrap", System.StringComparison.OrdinalIgnoreCase)) return PadMode.Wrap;
        }
        return PadMode.Constant;
    }
}
