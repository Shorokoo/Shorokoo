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
/// Shape inference for ONNX <c>MelWeightMatrix</c>. Output is
/// <c>[dft_length / 2 + 1, num_mel_bins]</c>; both come from scalar tensor inputs.
/// </summary>
internal sealed class MelWeightMatrixOp : QuickOp
{
    public override string OpCode => OpCodes.MEL_WEIGHT_MATRIX;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        // Inputs: num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz
        var numMelBins = inputs.Length > 0 ? inputs[0] : null;
        var dftLength = inputs.Length > 1 ? inputs[1] : null;
        var dtype = attrs.GetDTypeVal(OnnxOpAttributeNames.AttrOutputDatatype) ?? DType.Float32;

        long? bins = null, dft = null;
        if (numMelBins?.IntData is { Length: > 0 } nb) bins = nb[0];
        if (dftLength?.IntData is { Length: > 0 } dl) dft = dl[0];

        if (bins is null || dft is null || bins <= 0 || dft <= 0)
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = 2, MaxRank = 2 }];
        return [RuntimeTensorFactory.Create(dtype, new Shape(new[] { dft.Value / 2 + 1, bins.Value }))];
    }
}
