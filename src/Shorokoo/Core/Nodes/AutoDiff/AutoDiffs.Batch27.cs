using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== If (IF_CLOSE) =====
        //
        // Forward: outputs = If(condition, then_branch_outputs, else_branch_outputs)
        //   Selects outputs from the then_branch when condition is true,
        //   or from the else_branch when condition is false.
        //
        // Gradient: The output gradients must be routed to the correct branch.
        //   For the then_branch inputs: gradient = IfElse(cond, dY, 0)
        //     → receives dY when condition is true (then branch was taken), 0 otherwise
        //   For the else_branch inputs: gradient = IfElse(cond, 0, dY)
        //     → receives dY when condition is false (else branch was taken), 0 otherwise
        //   Condition gradient = null (boolean, not differentiable)
        //
        //   The dispatch wrapper splits the flat inputs[] coming from the autograd
        //   engine into (cond, branchInputs):
        //     cond                          = the IF_OPEN condition (Variable)
        //     branchInputs[0..n]            = else_branch tensors  (alphabetical: "else" < "then")
        //     branchInputs[n..2n]           = then_branch tensors
        //   where n = outputGrads.Length. The returned array has shape 1 + 2n,
        //   matching the engine's expected (cond, else..., then...) layout.

        [AutoDiff(IF_CLOSE)]
        public static Variable?[] IfCloseGradient(
            Scalar<bit> cond,
            Variable?[] branchInputs,
            Variable?[] outputGrads,
            OnnxCSharpAttributes attributes)
        {
            var numOutputs = outputGrads.Length;

            var result = new Variable?[1 + branchInputs.Length]; // 1 + 2*numOutputs
            result[0] = null; // condition is boolean, not differentiable

            for (int i = 0; i < numOutputs; i++)
            {
                var dY = outputGrads[i];
                if (dY is null)
                {
                    result[1 + i] = null;              // else_branch_i
                    result[1 + numOutputs + i] = null;  // then_branch_i
                    continue;
                }

                var zeros = OnnxOp.Sub(dY, dY); // zeros_like(dY)

                // Then branch gradient: dY when condition is true, zeros when false
                var ifOpenT = OnnxOp.IfOpen(cond);
                result[1 + numOutputs + i] = OnnxOp.IfClose(new[] { dY }, new[] { zeros }, ifOpenT)[0];

                // Else branch gradient: zeros when condition is true, dY when false
                var ifOpenE = OnnxOp.IfOpen(cond);
                result[1 + i] = OnnxOp.IfClose(new[] { zeros }, new[] { dY }, ifOpenE)[0];
            }

            return result;
        }
    }
}
