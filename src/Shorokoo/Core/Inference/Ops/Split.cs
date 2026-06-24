using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
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
/// QEE kernel for ONNX <c>Split</c>. Chunk sizes come from the optional <c>split</c> input
/// when connected, else from <c>num_outputs</c> (last chunk SMALLER when the axis dim isn't
/// evenly divisible, per spec: chunk = ceil(dim / num_outputs)). Overrides
/// <see cref="QuickOp.Execute"/> to read the node's declared output count, which is the
/// source of truth when neither the split values nor num_outputs are available.
/// </summary>
internal sealed class SplitOp : QuickOp
{
    public override string OpCode => OpCodes.SPLIT;

    public override (IRuntimeTensor[] results, bool loopBack) Execute(
        FastNode node,
        FastComputationGraph graph,
        Dictionary<FastNodeKey, FastNode> nodeByKey,
        Dictionary<FastTensorKey, IRuntimeTensor> store,
        int maxDataElements)
    {
        var inputs = GatherInputs(node.Inputs, store);
        var rtInputs = new RuntimeTensor?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) rtInputs[i] = inputs[i] as RuntimeTensor;

        var results = ComputeSplit(rtInputs, node.Attributes, maxDataElements, node.Outputs.Count);
        var asInterface = new IRuntimeTensor[results.Length];
        for (int i = 0; i < results.Length; i++)
            asInterface[i] = RuntimeTensorFactory.EnforceDataSizeLimit(results[i], maxDataElements);
        return (asInterface, false);
    }

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => ComputeSplit(inputs, attrs, maxDataElements, declaredOutputCount: 0);

    private static RuntimeTensor[] ComputeSplit(
        RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements, int declaredOutputCount)
    {
        var x = inputs[0];
        var axis = (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrAxis) ?? 0);
        var dtype = x?.DType ?? DType.Float32;

        var splitInput = inputs.Length > 1 ? inputs[1] : null;
        long[]? splitSizes = splitInput?.IntData?.ToArray();

        var numOutputs = splitSizes?.Length
            ?? (int)(attrs.GetLongVal(OnnxOpAttributeNames.AttrNumOutputs) ?? 0);
        if (numOutputs <= 0) numOutputs = declaredOutputCount;
        if (numOutputs <= 0) numOutputs = 1;

        var results = new RuntimeTensor[numOutputs];
        // No input shape — or split connected but with unknown values (the per-chunk dims
        // are then unknowable): every output shape is unknown.
        if (x?.Shape is null || (splitInput is not null && splitSizes is null))
        {
            for (int i = 0; i < numOutputs; i++)
                results[i] = RuntimeTensorFactory.Create(dtype, null);
            return results;
        }

        var inDims = x.Shape.Dims;
        if (axis < 0) axis += inDims.Length;
        if (axis < 0 || axis >= inDims.Length)
        {
            for (int i = 0; i < numOutputs; i++)
                results[i] = RuntimeTensorFactory.Create(dtype, null);
            return results;
        }
        var dimSize = inDims[axis];

        if (splitSizes is not null && splitSizes.Any(sz => sz < 0))
        {
            for (int i = 0; i < numOutputs; i++)
                results[i] = RuntimeTensorFactory.Create(dtype, null);
            return results;
        }

        if (splitSizes is null)
        {
            // num_outputs path: chunk = ceil(dim / num); the LAST chunk is smaller when the
            // axis dim isn't evenly divisible (e.g. dim 7 into 3 → [3, 3, 1]).
            var chunk = (dimSize + numOutputs - 1) / numOutputs;
            splitSizes = new long[numOutputs];
            long remaining = dimSize;
            for (int i = 0; i < numOutputs; i++)
            {
                splitSizes[i] = Math.Min(chunk, remaining);
                remaining -= splitSizes[i];
            }
            if (splitSizes.Any(sz => sz < 0) || remaining != 0)
            {
                for (int i = 0; i < numOutputs; i++)
                    results[i] = RuntimeTensorFactory.Create(dtype, null);
                return results;
            }
        }

        for (int i = 0; i < numOutputs; i++)
        {
            var dims = inDims.ToArray();
            dims[axis] = i < splitSizes.Length ? splitSizes[i] : 0;
            results[i] = RuntimeTensorFactory.Create(dtype, new Shape(dims));
        }

        // Value path: each output is a contiguous slice of the input along `axis`.
        if (x.FloatData is null && x.IntData is null && x.BoolData is null) return results;
        if (splitSizes.Sum() > dimSize) return results; // invalid split — don't guess values

        long outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= inDims[d];
        long innerCount = 1;
        for (int d = axis + 1; d < inDims.Length; d++) innerCount *= inDims[d];
        long inAxisStride = dimSize * innerCount;

        long offset = 0;
        for (int i = 0; i < numOutputs; i++)
        {
            var rt = results[i];
            if (!RuntimeTensorFactory.ShouldStoreData(rt.Shape, maxDataElements)) { offset += splitSizes[i]; continue; }
            long chunkLen = splitSizes[i] * innerCount;
            long total = outerCount * chunkLen;
            if (x.IntData is { } id)
            {
                var buf = new long[total];
                for (long outer = 0; outer < outerCount; outer++)
                    for (long e = 0; e < chunkLen; e++)
                        buf[outer * chunkLen + e] = id[(int)(outer * inAxisStride + offset * innerCount + e)];
                results[i] = rt with { IntData = ImmutableArray.Create(buf) };
            }
            else if (x.FloatData is { } fdata)
            {
                var buf = new float[total];
                for (long outer = 0; outer < outerCount; outer++)
                    for (long e = 0; e < chunkLen; e++)
                        buf[outer * chunkLen + e] = fdata[(int)(outer * inAxisStride + offset * innerCount + e)];
                results[i] = rt with { FloatData = ImmutableArray.Create(buf) };
            }
            else if (x.BoolData is { } bdata)
            {
                var buf = new bool[total];
                for (long outer = 0; outer < outerCount; outer++)
                    for (long e = 0; e < chunkLen; e++)
                        buf[outer * chunkLen + e] = bdata[(int)(outer * inAxisStride + offset * innerCount + e)];
                results[i] = rt with { BoolData = ImmutableArray.Create(buf) };
            }
            offset += splitSizes[i];
        }
        return results;
    }
}
