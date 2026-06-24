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
/// Shape inference for ONNX <c>LSTM</c> (gates = 4 — W is [num_directions, 4*hidden_size,
/// input_size]). Three outputs: (Y, Y_h, Y_c); Y_c shares Y_h's shape. The peephole P input
/// and input_forget only affect values. See <see cref="RecurrentShapeHelpers"/>.
/// </summary>
internal sealed class LstmOp : QuickOp
{
    public override string OpCode => OpCodes.LSTM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => RecurrentShapeHelpers.Infer(inputs, attrs, gates: 4, hasCellState: true);
}
