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
/// QEE kernel for ONNX DFT. Implemented to provide a fallback path for graphs where ORT
/// crashes — e.g. when DFT receives null optional inputs. Shape inference is always done;
/// numerical computation runs by direct O(N^2) summation when the inputs are concrete and
/// the result fits the engine's data-size limit.
///
/// Input X shape: [..., signal_dim, last] where last is 1 (real) or 2 (complex
/// [real, imag]). Output shape: input shape with axis dim possibly resized and last dim
/// forced to 2.
/// </summary>
internal sealed class DftOp : QuickOp
{
    public override string OpCode => OpCodes.DFT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var x = inputs[0];
        var dftLengthIn = inputs.Length > 1 ? inputs[1] : null;
        var axisIn = inputs.Length > 2 ? inputs[2] : null;

        bool inverse = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrInverse);
        bool onesided = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrOnesided);

        var dtype = x?.DType ?? DType.Float32;

        if (x?.Shape is null)
            return [RuntimeTensorFactory.Create(dtype, null)];

        var dims = x.Shape.Dims;
        var rank = dims.Length;

        // A connected axis/dft_length input whose value is unknown at QEE time makes the
        // output dims unknowable — degrade to rank-only rather than silently assuming the
        // spec defaults (which would produce a wrong concrete shape).
        if ((axisIn is not null && axisIn.IntData is null)
            || (dftLengthIn is not null && dftLengthIn.IntData is null))
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = rank, MaxRank = rank }];

        // axis defaults to -2 (penultimate, per ONNX spec).
        long rawAxis = -2;
        if (axisIn?.IntData is { } axData && axData.Length > 0)
            rawAxis = axData[0];
        int axis = (int)(rawAxis < 0 ? rawAxis + rank : rawAxis);
        // axis must address a real dim and may not be the trailing complex dim.
        if (axis < 0 || axis >= rank - 1)
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = rank, MaxRank = rank }];

        // N defaults to size along axis; explicit dft_length overrides (pads/truncates).
        long n = dims[axis];
        if (dftLengthIn?.IntData is { } lenData && lenData.Length > 0)
            n = lenData[0];
        if (n <= 0)
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = rank, MaxRank = rank }];

        var outDims = dims.ToArray();
        outDims[axis] = onesided ? (n / 2 + 1) : n;
        outDims[rank - 1] = 2;
        var outShape = new Shape(outDims);
        var rt = RuntimeTensorFactory.Create(dtype, outShape);

        if (x.FloatData is not { } xData
            || !RuntimeTensorFactory.ShouldStoreData(outShape, maxDataElements))
            return [rt];

        long inAxisSize = dims[axis];
        long inComplexDim = dims[rank - 1];
        long outAxisSize = outDims[axis];

        var inStrides = new long[rank];
        long ts = 1;
        for (int d = rank - 1; d >= 0; d--) { inStrides[d] = ts; ts *= dims[d]; }
        var outStrides = new long[rank];
        ts = 1;
        for (int d = rank - 1; d >= 0; d--) { outStrides[d] = ts; ts *= outDims[d]; }

        long totalOut = outStrides[0] * outDims[0];
        var output = new float[totalOut];

        var outIdx = new long[rank];
        // Iterate over output positions in steps of 2 (real,imag pair) along the last axis.
        for (long outFlatBase = 0; outFlatBase < totalOut; outFlatBase += 2)
        {
            long rem = outFlatBase;
            for (int d = 0; d < rank; d++)
            {
                outIdx[d] = rem / outStrides[d];
                rem -= outIdx[d] * outStrides[d];
            }
            long k = outIdx[axis];

            double sumR = 0, sumI = 0;
            for (long nn = 0; nn < n; nn++)
            {
                double xr = 0, xi = 0;
                if (nn < inAxisSize)
                {
                    long inFlat = 0;
                    for (int d = 0; d < rank; d++)
                    {
                        long ix = d == axis ? nn : (d == rank - 1 ? 0 : outIdx[d]);
                        inFlat += ix * inStrides[d];
                    }
                    xr = xData[(int)inFlat];
                    if (inComplexDim == 2)
                        xi = xData[(int)(inFlat + 1)];
                }
                double sign = inverse ? 1.0 : -1.0;
                double theta = sign * 2.0 * System.Math.PI * nn * k / (double)n;
                double cs = System.Math.Cos(theta), sn = System.Math.Sin(theta);
                sumR += xr * cs - xi * sn;
                sumI += xr * sn + xi * cs;
            }
            if (inverse) { sumR /= n; sumI /= n; }

            output[outFlatBase] = (float)sumR;
            output[outFlatBase + 1] = (float)sumI;
        }

        return [rt with { FloatData = ImmutableArray.Create(output) }];
    }
}
