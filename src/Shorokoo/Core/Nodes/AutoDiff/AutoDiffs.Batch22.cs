using System.Diagnostics;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Optional =====
        //
        // Forward: optional = Optional(tensor?)
        //   Wraps a tensor (or null) into an Optional container.
        //
        // Gradient: dTensor = OptionalGetElement(dOptional).
        //
        // Three guard branches were removed after coverage analysis confirmed they
        // are unreachable through the current dispatch path:
        //   - empty forward Optional (inputs.Length == 0 || inputs[0] is null):
        //     would require a forward Optional(null, ...) whose output reaches the
        //     loss with a non-null gradient. The natural construction (IfElse over
        //     Optional branches) is blocked elsewhere by IfClose's structure
        //     inference, so the empty-Optional path never arrives here.
        //   - dOptional is null: FastProcessAutoGrad.ProcessNode skips single-output
        //     nodes whose only outputGrad is null before invoking the gradient method.
        //   - dOptional is not Variable: every gradient producing an Optional
        //     output (OptionalGetElementGradient, AccumulateGradients for
        //     Variable) preserves the Optional wrap.

        internal static Variable?[] OptionalGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            Debug.Assert(inputs.Length > 0 && inputs[0] is not null);
            var dOptional = outputGrads[0];
            Debug.Assert(dOptional?.Structure() == DataStructure.Optional);

            return [OnnxOp.OptionalGetElement(dOptional!)];
        }

        // ===== OptionalGetElement =====
        //
        // Forward: tensor = OptionalGetElement(optional)
        // Gradient: dOptional = Optional(dTensor). The dispatchers skip
        // single-output nodes with a null outputGrad, so dTensor is always
        // non-null here.

        internal static Variable?[] OptionalGetElementGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var dTensor = outputGrads[0];
            Debug.Assert(dTensor is not null);

            return [OnnxOp.Optional(dTensor, DataStructure.Tensor, dTensor.Type)];
        }
    }
}
