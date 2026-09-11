using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Core.Factory;

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
    /// <para>A node flagged <see cref="OnnxOpAttributeNames.ShrkAttrIsParamReference"/> takes the
    /// other route. That flag marks a bare <c>IModel.GetTrainableParam</c> reference to a parameter
    /// defined elsewhere: it carries no initializer inputs and only borrows an initializer function
    /// as metadata, so invoking that function would not recompute the parameter — it would
    /// fabricate a different value and hand it over silently. It is instead pointed at the
    /// definition of the same parameter beside it in the body, whose lowered value is the one thing
    /// that <i>is</i> the parameter. The two identities are not comparable as written — a
    /// definition's is absolute within the body, a reference's relative to the model variable it is
    /// addressed against — so both are resolved to a body-absolute <see cref="ModelId"/> first, the
    /// same composition <c>FastConvertToIdRefModelParams</c> performs over a whole graph
    /// (Shorokoo/Shorokoo#318).</para>
    ///
    /// <para>A reference is rewritten only against a definition the body <i>proves</i> is the right
    /// one to read: a matching id is necessary and not sufficient. The definition must also already
    /// be in hand where the reference stands — earlier in <see cref="InternalComputationGraph.Nodes"/>,
    /// and inside no scope that does not also enclose the reference — which is the visibility rule
    /// <see cref="InternalComputationGraph.IsLinearOrderValid"/> enforces on the body as a whole.
    /// Matching on the id alone would wire an edge backwards when the reference is built before the
    /// call that defines the parameter, out of a <c>Loop</c> when only the call is inside one, and
    /// round a cycle when the defining call consumes the reference. Likewise a reference whose model
    /// variable comes from a node carrying no identifier template of its own (a model taken out of a
    /// <c>ModelSequence</c>) composes to no absolute id at all, and is not matched against one.
    /// Every reference left unresolved keeps the old behaviour and fails loudly at session creation:
    /// its model variable is removed with the rest of the model struct, so the operand it names is
    /// produced by nothing. That is the honest outcome — silently reading a parameter the body never
    /// proved is the right one would hand back plausible wrong numbers instead.</para>
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

            // Ahead of the rewrite below, which replaces every definition's node: a reference is
            // resolved against the definitions as the body still spells them, and keeps working
            // afterwards because the invoke that replaces one keeps its output key, and because
            // inlining that invoke rewires every consumer of that key — the reference among them.
            ResolveBareParamReferences(graph);

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

        /// <summary>
        /// Points every bare parameter reference in <paramref name="graph"/> at the definition of
        /// the same parameter in the same body, as an <see cref="OpCodes.IDENTITY"/> of the
        /// definition's value. Mutating in place keeps the reference's own output key, so its
        /// consumers need no rewriting. A reference with no definition beside it is left as it is.
        /// </summary>
        private static void ResolveBareParamReferences(InternalComputationGraph graph)
        {
            var nodes = graph.Nodes;
            if (!nodes.Any(FastExtractIdentifierTemplates.IsParamReference)) return;

            var nodeByKey = FastProcessorHelper.BuildNodeByKey(graph);
            var scopes = ScopeRanges(nodes);

            // Definitions by id, in node order, each with the position it stands at: which one a
            // reference may read depends on where the reference stands, so the choice cannot be
            // made here.
            var definitionsByModelId = new Dictionary<string, List<(int Index, FastTensorKey Output)>>();
            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                if (FastExtractIdentifierTemplates.IsParamReference(node)) continue;
                if (AbsoluteModelId(node, nodeByKey) is not { } modelId) continue;
                if (node.Outputs.FirstOrDefault() is not { } output) continue;
                if (!definitionsByModelId.TryGetValue(modelId, out var forId))
                    definitionsByModelId[modelId] = forId = [];
                forId.Add((i, output));
            }

            var identityAttrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(),
                Definitions.NodeDefinitions[OpCodes.IDENTITY].AttributeDefs);

            for (int i = 0; i < nodes.Count; i++)
            {
                var reference = nodes[i];
                if (!FastExtractIdentifierTemplates.IsParamReference(reference)) continue;
                if (AbsoluteModelId(reference, nodeByKey) is not { } modelId) continue;
                if (!definitionsByModelId.TryGetValue(modelId, out var candidates)) continue;

                // The first definition already in hand here. Two definitions of one parameter —
                // one model called twice — compute the same value, so the earliest visible one
                // serves; one that is not visible is not a candidate at all.
                FastTensorKey? definition = null;
                foreach (var candidate in candidates)
                {
                    if (candidate.Index >= i) break;
                    if (!IsVisibleFrom(candidate.Index, i, scopes)) continue;
                    definition = candidate.Output;
                    break;
                }
                if (definition is null) continue;

                reference.OpCode = OpCodes.IDENTITY;
                reference.Attributes = identityAttrs;
                reference.TargetFunction = null;
                reference.IdentifierTemplate = null;
                reference.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                {
                    [""] = [definition],
                };
            }
        }

        /// <summary>The (open, close) index pair of every control-flow scope in
        /// <paramref name="nodes"/>, which the body's linear-order invariant guarantees nest.</summary>
        private static List<(int Open, int Close)> ScopeRanges(IReadOnlyList<FastNode> nodes)
        {
            var scopes = new List<(int Open, int Close)>();
            var open = new Stack<int>();
            for (int i = 0; i < nodes.Count; i++)
            {
                if (FastOpsetResolver.IsOpenOpCode(nodes[i].OpCode)) open.Push(i);
                else if (FastOpsetResolver.IsCloseOpCode(nodes[i].OpCode) && open.Count != 0)
                    scopes.Add((open.Pop(), i));
            }
            return scopes;
        }

        /// <summary>Whether the node at <paramref name="producerIndex"/> is readable from
        /// <paramref name="consumerIndex"/>: no scope contains the producer without also containing
        /// the consumer. The same rule <see cref="InternalComputationGraph.IsLinearOrderValid"/>
        /// applies to the finished body, asked ahead of adding the edge rather than after.</summary>
        private static bool IsVisibleFrom(int producerIndex, int consumerIndex, List<(int Open, int Close)> scopes)
        {
            foreach (var scope in scopes)
            {
                if (!(scope.Open < producerIndex && producerIndex < scope.Close)) continue;
                if (scope.Open < consumerIndex && consumerIndex < scope.Close) continue;
                return false;
            }
            return true;
        }

        /// <summary>
        /// The <see cref="ModelId"/> <paramref name="node"/> addresses within its own body, or
        /// <c>null</c> when it addresses no parameter. A <c>MODEL_PARAM_REF</c> already carries a
        /// body-absolute template; a <c>MODEL_PARAM_MODEL_REF</c> is relative to the model variable
        /// it takes as its first operand, so its template is composed onto that model's own.
        /// </summary>
        private static string? AbsoluteModelId(FastNode node, Dictionary<FastNodeKey, FastNode> nodeByKey)
        {
            if (node.OpCode is not (InternalOpCodes.MODEL_PARAM_REF or InternalOpCodes.MODEL_PARAM_MODEL_REF))
                return null;
            if (node.IdentifierTemplate is not { } template) return null;
            var relative = new ModelParamIdentifierTemplate(template);

            if (node.OpCode == InternalOpCodes.MODEL_PARAM_REF)
                return relative.ModelIdTemplate.ToString();

            if (node.Inputs.FirstOrDefault() is not { } modelKey) return null;
            if (!nodeByKey.TryGetValue(modelKey.FastNodeKey, out var modelOwner)) return null;

            // No template on the model variable's own node — a model taken out of a ModelSequence,
            // whose SEQUENCE_AT is never stamped with one — leaves the relative id relative to
            // nothing. FastConvertToIdRefModelParams carries the same shape through as a bare
            // relative template, but there the template only NAMES the parameter and a GET_MODEL_ID
            // resolves it at run time; here the string IS the resolution, so handing back a relative
            // id would match it against absolute ones and read whichever parameter happens to sit
            // at that id. There is nothing to compose, so there is no id.
            if (modelOwner.IdentifierTemplate is not { } ownerTemplate) return null;

            return new ModelParamIdentifierTemplate(
                new ModelParamIdentifierTemplate(ownerTemplate), relative).ModelIdTemplate.ToString();
        }
    }
}
