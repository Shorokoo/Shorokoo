using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Einsum (Einstein summation) =====
        //
        // Forward: y = einsum(equation, x0, x1, ..., xn)
        //   equation has the form "s0,s1,...,sn->out" (explicit) or "s0,s1,...,sn" (implicit)
        //
        // Gradient: For each input k, the gradient is computed via another einsum:
        //   grad_xk = einsum(grad_eq_k, x0,...,x(k-1), grad_y, x(k+1),...,xn)
        //   where grad_eq_k places the output grad with output subscripts and produces
        //   input k's original subscripts as the new output.
        //
        // For indices that appear only in input k ("free indices" — summed over in forward,
        // broadcast-back in backward), a ones tensor shaped like input k is included as an
        // extra operand to supply the missing dimensions.
        //
        // Supports ellipsis notation ("...") for batch dimensions.
        //
        // Limitations:
        //   - Repeated subscripts within a single operand (e.g., "ii->i") are not supported

        internal static Variable?[] EinsumGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var equation = (string)attributes.GetAttributeObj("equation")!;
            var grad = outputGrads[0]!;

            // Replace "..." with single placeholder char for easier parsing
            const char ELP = '\u2026';
            var eq = equation.Replace("...", ELP.ToString());

            // Parse: "s0,s1,...,sn->out" or "s0,s1,...,sn" (implicit)
            var arrowPos = eq.IndexOf("->");
            string inputPart, outputSubs;
            if (arrowPos >= 0)
            {
                inputPart = eq.Substring(0, arrowPos);
                outputSubs = eq.Substring(arrowPos + 2);
            }
            else
            {
                inputPart = eq;
                // Implicit mode: sorted unique chars appearing exactly once, plus ellipsis prefix
                var raw = inputPart.Replace(",", "");
                var chars = raw.Where(c => c != ELP).ToList();
                var uniqueChars = chars.GroupBy(c => c)
                    .Where(g => g.Count() == 1)
                    .Select(g => g.Key)
                    .OrderBy(c => c);
                outputSubs = (raw.Contains(ELP) ? ELP.ToString() : "")
                           + new string(uniqueChars.ToArray());
            }

            var inputSubs = inputPart.Split(',');
            var result = new Variable?[inputs.Length];

            for (int k = 0; k < inputs.Length; k++)
            {
                if (inputs[k] is null) continue;

                var targetSubs = inputSubs[k];

                // Reject repeated subscripts in a single operand (e.g., "ii")
                var targetChars = targetSubs.Where(c => c != ELP).ToList();
                System.Diagnostics.Debug.Assert(targetChars.Count == targetChars.Distinct().Count(),
                    $"Einsum gradient not supported for repeated subscripts in operand: " +
                    $"'{targetSubs.Replace(ELP.ToString(), "...")}'");

                // Collect indices from other inputs and output to detect free indices
                var otherIndices = new HashSet<char>();
                for (int i = 0; i < inputSubs.Length; i++)
                {
                    if (i == k) continue;
                    foreach (var c in inputSubs[i])
                        if (c != ELP) otherIndices.Add(c);
                }
                foreach (var c in outputSubs)
                    if (c != ELP) otherIndices.Add(c);

                bool hasFreeIndices = targetChars.Any(c => !otherIndices.Contains(c));

                // Build gradient einsum operands
                var gradSubs = new List<string>();
                var gradVars = new List<Variable>();

                for (int i = 0; i < inputs.Length; i++)
                {
                    if (i == k) continue;
                    if (inputs[i] is null) continue;
                    gradSubs.Add(inputSubs[i]);
                    gradVars.Add(inputs[i]!);
                }

                // Add output gradient with the forward output's subscripts
                gradSubs.Add(outputSubs);
                gradVars.Add(grad);

                if (hasFreeIndices)
                {
                    // Add ones tensor matching input k's shape to supply free-index dimensions
                    var one = OnnxOp.Cast(Scalar(1.0f), saturate: null, to: inputs[k]!.Type);
                    var ones = OnnxOp.Expand(one, OnnxOp.Shape(inputs[k]!));
                    gradSubs.Add(targetSubs);
                    gradVars.Add(ones);
                }

                // Assemble gradient equation and restore "..." notation
                var gradEq = string.Join(",", gradSubs) + "->" + targetSubs;
                gradEq = gradEq.Replace(ELP.ToString(), "...");

                result[k] = OnnxOp.Einsum(gradVars.ToArray(), equation: gradEq);
            }

            return result;
        }
    }
}
