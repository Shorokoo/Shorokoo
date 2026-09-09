using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Shared helpers for the Fast-native trainable / state param discovery processors.
    /// Walks <see cref="InternalComputationGraph.Nodes"/> in stored (topological) order, filters
    /// to param-producer ops (<c>MODEL_PARAM</c>, <c>MODEL_PARAM_DATA</c>,
    /// <c>MODEL_PARAM_ID_REF</c>), and reads dtype / rank / sanitized field name straight
    /// off each node's attributes — no round-trip to <c>ComputationGraph</c>. Sites that name
    /// the same parameter collapse into one entry, so the result is one record per parameter.
    /// </summary>
    internal static class FastDiscoverParamsHelpers
    {
        public static ImmutableArray<FastDiscoveredParamInfo> Discover(InternalComputationGraph graph, bool wantTrainable)
        {
            // One entry per PARAMETER, not per param-producer node. A single parameter is reachable
            // from several sites: one model handle called twice leaves one definition site per call
            // (Shorokoo/Shorokoo#284), and IModel.GetTrainableParam adds a bare reference site
            // (Shorokoo/Shorokoo#263). Every one of them is the same weight, so the first site seen
            // owns the entry and the rest join it as aliases — the caller rewires their consumers to
            // this entry's field and drops their nodes. Without that collapse one weight gets two
            // identically-named struct fields, which splits its gradient and its checkpoint entry.
            var definitions = new List<Definition>();
            var definitionByModelId = new Dictionary<ModelId, Definition>();
            // Bare references seen before the definition they point at, keyed by that definition's id.
            var pendingReferencesByModelId = new Dictionary<ModelId, List<(FastTensorKey OutputKey, FastNode Node)>>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM &&
                    node.OpCode != InternalOpCodes.MODEL_PARAM_DATA &&
                    node.OpCode != InternalOpCodes.MODEL_PARAM_ID_REF)
                    continue;

                // The RngSeed parameter at reserved ModelId [0] is the model's RNG identity —
                // neither a trainable weight nor per-step state. It stays embedded model data
                // (the key chains read it in place); ApplyRngConfig is its only writer.
                if (node.IdentifierTemplate ==
                        Fast.FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    continue;

                // Missing attribute → skip (mirrors the CG processor's GetIsTrainable() catch).
                var isTrainable = node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable);
                if (isTrainable is null) continue;
                if (isTrainable.Value != wantTrainable) continue;

                FastTensorKey? outputKey = null;
                foreach (var slot in node.FullOutputs.Values)
                    foreach (var k in slot)
                        if (k is FastTensorKey tk && !tk.IsEmpty) { outputKey = tk; break; }
                if (outputKey is null) continue;

                var (dtype, rank) = ExtractDTypeAndRank(node);
                if (dtype is null) continue;

                var modelId = SpecificModelIdOf(node);

                // A bare reference reads a parameter the model already owns — it declares none of
                // its own, and its ParamRef_<id path> name is a placeholder, not the parameter's.
                // Counting it as a parameter gives one weight two struct fields, and the reference
                // then reads its own fed tensor instead of the model's (Shorokoo/Shorokoo#263).
                if (IsParamReference(node))
                {
                    var referencedId = modelId
                        ?? throw new InvalidOperationException(
                            "Trainable-param discovery: a parameter reference (e.g. via "
                            + "IModel.GetTrainableParam) carries no identifier template, so the "
                            + "parameter it points at cannot be determined.");
                    if (definitionByModelId.TryGetValue(referencedId, out var referenced))
                        referenced.Aliases.Add((outputKey.Value, node));
                    else
                    {
                        if (!pendingReferencesByModelId.TryGetValue(referencedId, out var pending))
                            pendingReferencesByModelId[referencedId] = pending = [];
                        pending.Add((outputKey.Value, node));
                    }
                    continue;
                }

                // A second definition of a parameter already defined is the same weight reached a
                // second time, not a second weight (Shorokoo/Shorokoo#284).
                if (modelId is { } id && definitionByModelId.TryGetValue(id, out var existing))
                {
                    existing.Aliases.Add((outputKey.Value, node));
                    continue;
                }

                var definition = new Definition(node, outputKey.Value, isTrainable.Value, dtype, rank);
                definitions.Add(definition);
                // A node with no identifier template carries no id to be reached by a second site,
                // so it stands alone rather than joining or claiming an entry.
                if (modelId is { } newId)
                {
                    definitionByModelId[newId] = definition;
                    if (pendingReferencesByModelId.Remove(newId, out var earlier))
                        definition.Aliases.AddRange(earlier);
                }
            }

            // Mirrors the concretized path's rule (FastConvertModelParamIdRefToModelParam's
            // ExtractModelIdInfosFromStore): a reference is not a definition and cannot stand in
            // for one, so a model id that only ever appears referenced is an error, not a
            // parameter to invent.
            if (pendingReferencesByModelId.Count > 0)
                throw new InvalidOperationException(
                    $"Trainable-param discovery: model id {pendingReferencesByModelId.Keys.First()} is "
                    + "referenced (e.g. via IModel.GetTrainableParam) but has no parameter "
                    + "definition in the graph. A bare reference cannot stand in for the definition.");

            var results = ImmutableArray.CreateBuilder<FastDiscoveredParamInfo>(definitions.Count);
            foreach (var d in definitions)
                results.Add(new FastDiscoveredParamInfo(
                    ResolveParamName(d.Node, results.Count), d.OutputKey, d.IsTrainable, d.DType,
                    d.Rank, DataStructure.Tensor, d.Node, [.. d.Aliases]));

            return results.MoveToImmutable();
        }

        /// <summary>One parameter under construction: the site that first defined it plus the
        /// further sites — later definitions, bare references — that turned out to be the same
        /// parameter.</summary>
        private sealed class Definition(FastNode node, FastTensorKey outputKey, bool isTrainable, DType dtype, int? rank)
        {
            public FastNode Node { get; } = node;
            public FastTensorKey OutputKey { get; } = outputKey;
            public bool IsTrainable { get; } = isTrainable;
            public DType DType { get; } = dtype;
            public int? Rank { get; } = rank;
            public List<(FastTensorKey OutputKey, FastNode Node)> Aliases { get; } = [];
        }

        /// <summary>
        /// Whether this node is a bare REFERENCE to a parameter (<c>IModel.GetTrainableParam</c>)
        /// rather than the parameter's own definition. Only <c>MODEL_PARAM_ID_REF</c> can be one,
        /// and only it declares the attribute — <c>GetBoolVal</c> throws rather than returning
        /// null for an attribute the node's op does not declare, so the op check is load-bearing,
        /// not an optimization. <c>MODEL_PARAM</c> and <c>MODEL_PARAM_DATA</c> are definitions.
        /// </summary>
        private static bool IsParamReference(FastNode node)
            => node.OpCode == InternalOpCodes.MODEL_PARAM_ID_REF
               && (node.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsParamReference) ?? false);

        /// <summary>
        /// The id of the parameter this node's identifier template names, if it carries one.
        /// <b>Specific</b>, not generalized: the concretized path realizes a loop body's
        /// per-iteration parameters as separate nodes that share one generalized ModelIdTemplate
        /// and differ only in their template's loop indices, so keying on the template would fuse
        /// a loop's iterations into a single parameter. Before concretization no loop is unrolled
        /// and every iteration index is still -1, which the specific id leaves untouched — so this
        /// is the id that identifies a parameter on both paths.
        /// </summary>
        private static ModelId? SpecificModelIdOf(FastNode node)
            => string.IsNullOrEmpty(node.IdentifierTemplate)
                ? null
                : new ModelParamIdentifierTemplate(node.IdentifierTemplate).SpecificModelId;

        /// <summary>
        /// Reads dtype and rank from the producing node's attributes. All three handled op
        /// codes produce a single tensor whose dtype/rank are recoverable without rich CG
        /// metadata.
        /// </summary>
        private static (DType? dtype, int? rank) ExtractDTypeAndRank(FastNode node)
        {
            if (node.OpCode == InternalOpCodes.MODEL_PARAM_DATA)
            {
                var data = node.Attributes.GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData);
                if (data is null) return (null, null);
                return (data.DType, data.Shape.Dims.Length);
            }

            var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype);
            var rank = (int?)node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank);
            return (dtype, rank);
        }

        private static string ResolveParamName(FastNode node, int index)
        {
            if (!string.IsNullOrEmpty(node.IdentifierTemplate))
            {
                // FastNode.IdentifierTemplate is the serialized "[ModelId]:Cat.Mod.Param" form.
                // Take only the template-string portion (after "]:") and keep it VERBATIM: this is
                // the inference model's canonical parameter name — the same dotted string that
                // ModuleParamSetNamingScheme.ToName produces for this param's ModelId. Preserving
                // it (instead of sanitizing '.'→'_') lets a trained checkpoint's TrainableParams
                // round-trip straight back through graph.ToConcreteModel(...) by name. Field names
                // are only used as struct lookup keys; consumers that need a valid C# identifier
                // (e.g. CSharpModelBuilder) re-sanitize on their own, so dots are safe here.
                var templateString = ExtractTemplateString(node.IdentifierTemplate);
                if (!string.IsNullOrEmpty(templateString))
                    return templateString;
            }

            // No identifier template (rare): fall back to a sanitized friendly name / ordinal.
            if (!string.IsNullOrEmpty(node.FriendlyName))
                return SanitizeFieldName(node.FriendlyName);

            return $"Param{index}";
        }

        /// <summary>
        /// Pulls the dot-joined parts segment out of an <c>"[ModelId]:parts..."</c> string,
        /// matching <see cref="ModelParamIdentifierTemplate.ToTemplateString"/> on the CG side —
        /// the parameter's canonical Shorokoo name (what
        /// <see cref="ModuleParamSetNamingScheme.ToName(ModelId)"/> produces under the canonical
        /// scheme). Shared with <c>Persistence.ExportSafeTensors</c>, which names exported
        /// tensors canonically. Returns the input unchanged if it isn't in the bracketed form.
        /// </summary>
        public static string ExtractTemplateString(string identifierTemplate)
        {
            var idx = identifierTemplate.IndexOf("]:", StringComparison.Ordinal);
            if (idx < 0) return identifierTemplate;
            return identifierTemplate.Substring(idx + 2);
        }

        private static string SanitizeFieldName(string name)
        {
            var sanitized = name.Replace('.', '_').Replace('/', '_').Replace('-', '_');
            if (sanitized.Length > 0 && !char.IsLetter(sanitized[0]) && sanitized[0] != '_')
                sanitized = "_" + sanitized;
            return sanitized;
        }
    }
}
