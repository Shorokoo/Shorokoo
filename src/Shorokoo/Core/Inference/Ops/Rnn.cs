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
/// Shared shape inference for the ONNX recurrent family (RNN / GRU / LSTM), opset 21:
///   - Y:   layout=0 → [seq_length, num_directions, batch_size, hidden_size];
///          layout=1 → [batch_size, seq_length, num_directions, hidden_size].
///   - Y_h (and LSTM's Y_c):
///          layout=0 → [num_directions, batch_size, hidden_size];
///          layout=1 → [batch_size, num_directions, hidden_size].
/// num_directions comes from the <c>direction</c> attribute (bidirectional → 2, else 1);
/// hidden_size from the attribute, else <c>W.shape[1] / gates</c> (gates = 1/3/4 for
/// RNN/GRU/LSTM), else <c>R.shape[2]</c>. The activation attrs, clip,
/// GRU's linear_before_reset and LSTM's input_forget only affect values; the optional
/// sequence_lens / initial_h / initial_c / P inputs don't change the output shapes. Values
/// are not computed (full recurrent evaluation; shape-only is enough for propagation). The
/// <c>layout</c> attribute is declared as a bool in the node definition, so it must be read
/// with <see cref="AttrAccess.GetBool"/> — a raw <c>GetLongVal</c> threw on the boxed bool
/// and silently killed the whole inference (the engine swallows op exceptions).
/// </summary>
internal static class RecurrentShapeHelpers
{
    public static RuntimeTensor[] Infer(
        RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, long gates, bool hasCellState)
    {
        var x = inputs.Length > 0 ? inputs[0] : null;
        var w = inputs.Length > 1 ? inputs[1] : null;
        var r = inputs.Length > 2 ? inputs[2] : null;
        var dtype = x?.DType ?? DType.Float32;

        var layout = AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrLayout, false);
        var numDir = IsBidirectional(attrs) ? 2L : 1L;

        long seqLen = -1, batch = -1;
        if (x?.Shape?.Dims is { Length: 3 } xDims)
        {
            seqLen = layout ? xDims[1] : xDims[0];
            batch = layout ? xDims[0] : xDims[1];
        }

        var hidden = attrs.GetLongVal(OnnxOpAttributeNames.AttrHiddenSize) ?? -1;
        if (hidden <= 0 && w?.Shape?.Dims is { Length: 3 } wDims && wDims[1] > 0 && wDims[1] % gates == 0)
            hidden = wDims[1] / gates;
        if (hidden <= 0 && r?.Shape?.Dims is { Length: 3 } rDims && rDims[2] > 0)
            hidden = rDims[2];

        if (seqLen < 0 || batch < 0 || hidden <= 0)
        {
            var unknownY = RuntimeTensorFactory.CreateRankOnly(dtype, 4);
            var unknownH = RuntimeTensorFactory.CreateRankOnly(dtype, 3);
            return hasCellState
                ? [unknownY, unknownH, RuntimeTensorFactory.CreateRankOnly(dtype, 3)]
                : [unknownY, unknownH];
        }

        var yShape = layout
            ? new Shape(new[] { batch, seqLen, numDir, hidden })
            : new Shape(new[] { seqLen, numDir, batch, hidden });
        var stateShape = layout
            ? new Shape(new[] { batch, numDir, hidden })
            : new Shape(new[] { numDir, batch, hidden });

        return hasCellState
            ? [
                RuntimeTensorFactory.Create(dtype, yShape),
                RuntimeTensorFactory.Create(dtype, stateShape),
                RuntimeTensorFactory.Create(dtype, stateShape),
            ]
            : [
                RuntimeTensorFactory.Create(dtype, yShape),
                RuntimeTensorFactory.Create(dtype, stateShape),
            ];
    }

    /// <summary>
    /// The direction attribute carries the C# enum (RNNDirection / GRUDirection /
    /// LSTMDirection) in-framework or the wire string after a roundtrip; both spell
    /// "bidirectional" the same way. Default is forward (num_directions = 1).
    /// </summary>
    private static bool IsBidirectional(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.AttrDirection)) return false;
        var obj = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrDirection);
        return obj?.ToString()?.Equals("bidirectional", StringComparison.OrdinalIgnoreCase) == true;
    }
}

/// <summary>
/// Shape inference for ONNX <c>RNN</c> (gates = 1). See <see cref="RecurrentShapeHelpers"/>.
/// </summary>
internal sealed class RnnOp : QuickOp
{
    public override string OpCode => OpCodes.RNN;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => RecurrentShapeHelpers.Infer(inputs, attrs, gates: 1, hasCellState: false);
}
