using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shape inference for ONNX <c>STFT</c>.
///
/// <para>Signal shape is <c>[batch, signal_length, 1|2]</c>. With <c>frame_step</c> = S
/// and effective frame length L (from <c>frame_length</c> input or <c>window</c> length),
/// the number of frames is <c>floor((signal_length - L) / S) + 1</c>. The output is
/// <c>[batch, n_frames, n_dft, 2]</c> where <c>n_dft = L</c> (two-sided) or
/// <c>L / 2 + 1</c> (one-sided).</para>
/// </summary>
internal sealed class STFTOp : QuickOp
{
    public override string OpCode => OpCodes.STFT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var signal = inputs.Length > 0 ? inputs[0] : null;
        var frameStep = inputs.Length > 1 ? inputs[1] : null;
        var window = inputs.Length > 2 ? inputs[2] : null;
        var frameLengthIn = inputs.Length > 3 ? inputs[3] : null;
        var dtype = signal?.DType ?? DType.Float32;

        if (signal?.Shape is null || signal.Shape.Dims.Length < 2)
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = 4, MaxRank = 4 }];

        var onesided = attrs.GetBoolVal(OnnxOpAttributeNames.AttrOnesided) ?? true;

        long? frameLength = frameLengthIn?.IntData is { Length: > 0 } fl ? fl[0]
                          : window?.Shape?.Dims is { Length: > 0 } wd ? wd[0]
                          : null;
        long? step = frameStep?.IntData is { Length: > 0 } fs ? fs[0] : null;

        var dims = signal.Shape.Dims;
        long batch = dims[0];
        long signalLength = dims[1];

        // Frame geometry unknown (e.g. frame_step connected but value-unknown): the output is
        // always rank 4 ([batch, frames, dft_unique_bins, 2]) but the inner dims are
        // unknowable — degrade to rank-only rather than emitting fabricated dims.
        if (frameLength is not { } L || step is not { } S || S <= 0 || L <= 0 || L > signalLength)
            return [RuntimeTensorFactory.Create(dtype, null) with { Rank = 4, MaxRank = 4 }];

        long nFrames = (signalLength - L) / S + 1;
        long nDft = onesided ? L / 2 + 1 : L;

        return [RuntimeTensorFactory.Create(dtype, new Shape(new[] { batch, nFrames, nDft, 2L }))];
    }
}
