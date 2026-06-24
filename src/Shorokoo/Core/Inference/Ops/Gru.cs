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
/// Shape inference for ONNX <c>GRU</c> (gates = 3 — W is [num_directions, 3*hidden_size,
/// input_size]). linear_before_reset only affects values. See
/// <see cref="RecurrentShapeHelpers"/>.
/// </summary>
internal sealed class GruOp : QuickOp
{
    public override string OpCode => OpCodes.GRU;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => RecurrentShapeHelpers.Infer(inputs, attrs, gates: 3, hasCellState: false);
}
