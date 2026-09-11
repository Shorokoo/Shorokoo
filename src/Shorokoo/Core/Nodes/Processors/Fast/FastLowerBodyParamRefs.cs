using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Lowers the parameter definitions left in a function body that is emitted on its own — a
    /// <c>FunctionProto</c> built from a flattened body — to a <c>FUNCTION_INVOKE</c> of the
    /// parameter's own initializer function. Both spellings count: <c>MODEL_PARAM_REF</c>, and the
    /// <c>MODEL_PARAM_MODEL_REF</c> that inlining produces when the callee arrived through a model
    /// variable (out of a <c>ModelSequence</c>, say) rather than a direct call.
    ///
    /// <para>The whole-graph parameter chain (<see cref="FastExtractIdentifierTemplates"/> →
    /// <see cref="FastConvertToIdRefModelParams"/> → <see cref="FastConvertModelParamIdRefToModelParam"/>)
    /// turns a reference into a <c>MODEL_PARAM</c> that the model carries as a weight, and it only
    /// ever runs over a whole graph. A body emitted on its own is not one: it belongs to a function
    /// the concretized graph still calls (an initializer, most often), so a parameter a callee of
    /// that body owns was never registered in the model's parameter inventory and no weight will
    /// ever be fed for it. Its value is therefore exactly what its initializer computes, which is
    /// what this pass emits — the same rewrite <see cref="FastInitializeModelParams"/> applies to a
    /// <c>MODEL_PARAM</c>, minus the iteration-index operand a reference carries and a parameter
    /// does not.</para>
    ///
    /// <para>Each definition gets its own invoke, so two of them for one parameter within a single
    /// body evaluate that parameter's initializer twice — the same value for a deterministic
    /// initializer. A drawing one does not reach here under a bound <c>RngConfig</c>:
    /// <see cref="FastInitKeyedDraws.BuildKeyedDraws"/> refuses a parameter reference whose
    /// initializer draws, ahead of emission, rather than let it draw unkeyed.</para>
    ///
    /// <para>A node flagged <see cref="OnnxOpAttributeNames.ShrkAttrIsParamReference"/> is left
    /// alone. That flag marks a bare <c>IModel.GetTrainableParam</c> reference to a parameter
    /// defined elsewhere: it carries no initializer inputs and only borrows an initializer function
    /// as metadata, so invoking that function would not recompute the parameter — it would
    /// fabricate a different value and hand it over silently. Leaving the node is the honest
    /// outcome; it fails loudly at session creation instead.</para>
    /// </summary>
    internal static class FastLowerBodyParamRefs
    {
        /// <summary>
        /// Rewrites every lowerable parameter definition in <paramref name="graph"/> in place.
        /// Returns true when at least one was rewritten, so the caller can re-run
        /// <see cref="FastInlineModulesAndFunctions"/> over the invokes this produced.
        /// </summary>
        public static bool Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var invokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;
            bool any = false;

            foreach (var node in graph.Nodes)
            {
                // The operands ahead of the initializer's own: a model-relative definition names the
                // model variable it is addressed against before its iteration indices.
                int leadingOperands = node.OpCode switch
                {
                    InternalOpCodes.MODEL_PARAM_REF => 1,
                    InternalOpCodes.MODEL_PARAM_MODEL_REF => 2,
                    _ => 0,
                };
                if (leadingOperands == 0) continue;
                // No initializer to call: leave the node alone rather than emit an invoke with no
                // target. The op check downstream still reports it.
                if (node.TargetFunction is null) continue;
                // A bare reference borrows its initializer function as metadata and carries no
                // initializer inputs, so calling it would invent a value rather than recompute the
                // parameter. Leave it for the loud failure.
                if (node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsParamReference) ?? false) continue;

                DataStructure[] structure = [DataStructure.Tensor];
                DType[] dtype = [node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull()];
                long[] rank = [node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank) ?? -1];

                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [OnnxOpAttributeNames.ShrkAttrStructure] = structure,
                        [OnnxOpAttributeNames.ShrkAttrDtype] = dtype,
                        [OnnxOpAttributeNames.ShrkAttrRank] = rank,
                        [OnnxOpAttributeNames.ShrkAttrGenericTypeArgs] =
                            node.Attributes.GetDTypesVal(OnnxOpAttributeNames.ShrkAttrGenericTypeArgs),
                    },
                    invokeAttrDefs);

                // Inputs are [model?, iterationIndices, ...initializerParams]; the invoke takes the
                // initializer's parameters alone. TargetFunction carries over unchanged.
                var initializerParams = node.Inputs.Skip(leadingOperands).ToList();
                node.OpCode = InternalOpCodes.FUNCTION_INVOKE;
                node.FullInputs = new Dictionary<string, List<FastTensorKey?>> { [""] = initializerParams };
                node.IdentifierTemplate = null;
                any = true;
            }

            return any;
        }
    }
}
