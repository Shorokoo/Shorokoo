using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Immutable;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Shared helpers for the Fast-native trainable / state param discovery processors.
    /// Walks <see cref="FastComputationGraph.Nodes"/> in stored (topological) order, filters
    /// to param-producer ops (<c>MODEL_PARAM</c>, <c>MODEL_PARAM_DATA</c>,
    /// <c>MODEL_PARAM_ID_REF</c>), and reads dtype / rank / sanitized field name straight
    /// off each node's attributes — no round-trip to <c>ComputationGraph</c>.
    /// </summary>
    internal static class FastDiscoverParamsHelpers
    {
        public static ImmutableArray<FastDiscoveredParamInfo> Discover(FastComputationGraph graph, bool wantTrainable)
        {
            var results = ImmutableArray.CreateBuilder<FastDiscoveredParamInfo>();

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

                var name = ResolveParamName(node, results.Count);
                results.Add(new FastDiscoveredParamInfo(
                    name, outputKey.Value, isTrainable.Value, dtype, rank, DataStructure.Tensor, node));
            }

            return results.ToImmutable();
        }

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
        /// matching <see cref="ModelParamIdentifierTemplate.ToTemplateString"/> on the CG side.
        /// Returns the input unchanged if it isn't in the bracketed form.
        /// </summary>
        private static string ExtractTemplateString(string identifierTemplate)
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
