using System.Collections.Generic;
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
/// Shape inference for ONNX <c>Einsum</c>. Parses the equation (e.g. <c>"ij,jk->ik"</c>)
/// to resolve each label's dim from the input shapes, then assembles the output dims
/// from the right-hand side. Falls back to a shape-unknown tensor when the equation
/// uses ellipsis or any input lacks shape info.
/// </summary>
internal sealed class EinsumOp : QuickOp
{
    public override string OpCode => OpCodes.EINSUM;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var dtype = inputs.Length > 0 && inputs[0] is not null ? inputs[0]!.DType : DType.Float32;
        var equation = attrs.GetStringVal(OnnxOpAttributeNames.AttrEquation);
        if (equation is null) return [RuntimeTensorFactory.Create(dtype, null)];

        // Strip whitespace and split on "->" for explicit-output form.
        equation = equation.Replace(" ", "");
        string lhs, rhs;
        var arrow = equation.IndexOf("->", System.StringComparison.Ordinal);
        if (arrow >= 0)
        {
            lhs = equation.Substring(0, arrow);
            rhs = equation.Substring(arrow + 2);
        }
        else
        {
            lhs = equation;
            rhs = null!;
        }

        // Ellipsis support would require resolving the rank of "...": we punt to
        // shape-unknown when the equation contains one rather than risk emitting a
        // misleading shape.
        if (lhs.Contains("...") || (rhs is not null && rhs.Contains("...")))
            return [RuntimeTensorFactory.Create(dtype, null)];

        var operandLabels = lhs.Split(',');
        if (operandLabels.Length != inputs.Length) return [RuntimeTensorFactory.Create(dtype, null)];

        var dimByLabel = new Dictionary<char, long>();
        for (int i = 0; i < operandLabels.Length; i++)
        {
            var labels = operandLabels[i];
            if (inputs[i]?.Shape is not { } sh) return [RuntimeTensorFactory.Create(dtype, null)];
            var dims = sh.Dims;
            if (labels.Length != dims.Length) return [RuntimeTensorFactory.Create(dtype, null)];
            for (int j = 0; j < labels.Length; j++)
            {
                // A repeated label's dims must agree across (and within) operands, with
                // size-1 dims broadcasting; a known mismatch (or an unknown −1 dim where
                // the resolution is ambiguous) degrades to an unknown shape — never guess.
                var d = dims[j];
                if (!dimByLabel.TryGetValue(labels[j], out var existing)) dimByLabel[labels[j]] = d;
                else if (existing == d) { }
                else if (existing == 1) dimByLabel[labels[j]] = d;
                else if (d == 1) { }
                else if (existing < 0) dimByLabel[labels[j]] = d;
                else if (d < 0) { }
                else return [RuntimeTensorFactory.Create(dtype, null)];
            }
        }

        // Implicit output: labels that appear exactly once across operands, in alphabetical order.
        if (rhs is null)
        {
            var counts = new Dictionary<char, int>();
            foreach (var op in operandLabels)
                foreach (var c in op)
                    counts[c] = counts.GetValueOrDefault(c, 0) + 1;
            var unique = new List<char>();
            foreach (var kv in counts) if (kv.Value == 1) unique.Add(kv.Key);
            unique.Sort();
            rhs = new string(unique.ToArray());
        }

        var outDims = new long[rhs.Length];
        for (int i = 0; i < rhs.Length; i++)
        {
            if (!dimByLabel.TryGetValue(rhs[i], out var d))
                return [RuntimeTensorFactory.Create(dtype, null)];
            outDims[i] = d;
        }
        return [RuntimeTensorFactory.Create(dtype, new Shape(outDims))];
    }
}
