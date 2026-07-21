using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast equivalent of <c>ChangeGenericTypeSpecialization</c>.
    /// Walks every node in <c>graph</c> and rewrites its
    /// DType / DTypes / Tensor attributes so that any DType whose
    /// <see cref="DType.GenericTypeParamName"/> appears in
    /// <c>typeSpecializations</c> is replaced with the corresponding
    /// specialized DType (still tagged with the same param name).
    /// Mutates the graph in place.
    ///
    /// Because <see cref="InternalComputationGraph"/> does not store per-tensor types,
    /// this is purely an attribute rewrite — there is no per-tensor type re-inference
    /// step like the CG version's <c>reinferTypes</c>. Downstream type information
    /// is reconstructed when (or if) the FastCG is later converted back to a
    /// <c>ComputationGraph</c>.
    /// </summary>
    internal static class FastChangeGenericTypeSpecialization
    {
        public static void Process(
            InternalComputationGraph graph,
            Dictionary<string, DType> typeSpecializations)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (typeSpecializations is null || typeSpecializations.Count == 0)
                throw new ArgumentException("typeSpecializations cannot be null or empty");

            var specializedDTypesWithParam = new Dictionary<string, DType>();
            foreach (var kvp in typeSpecializations)
                specializedDTypesWithParam[kvp.Key] = DType.CreateWithGenericParam(kvp.Value, kvp.Key);

            foreach (var node in graph.Nodes)
                RewriteNodeAttributes(node, specializedDTypesWithParam);
        }

        private static void RewriteNodeAttributes(FastNode node, Dictionary<string, DType> specializedDTypesWithParam)
        {
            var attrs = node.Attributes;
            var rebuilt = attrs.GetAttributeVals().ToDictionary();
            bool anyUpdated = false;

            foreach (var def in attrs.AttributeDefs)
            {
                switch (def.Type)
                {
                    case AttributeType.DType:
                    {
                        var dt = attrs.GetDTypeVal(def.AttributeName);
                        if (dt?.GenericTypeParamName is null) continue;
                        if (specializedDTypesWithParam.TryGetValue(dt.GenericTypeParamName, out var specialized))
                        {
                            rebuilt[def.AttributeName] = specialized;
                            anyUpdated = true;
                        }
                        break;
                    }
                    case AttributeType.DTypes:
                    {
                        var dtypes = attrs.GetDTypesVal(def.AttributeName);
                        if (dtypes is null || dtypes.Length == 0) continue;
                        if (dtypes.All(x => x.GenericTypeParamName is null)) continue;

                        var updated = dtypes.Select(dt =>
                            dt.GenericTypeParamName is { } name &&
                            specializedDTypesWithParam.TryGetValue(name, out var specialized)
                                ? specialized : dt).ToArray();
                        rebuilt[def.AttributeName] = updated;
                        anyUpdated = true;
                        break;
                    }
                    case AttributeType.Tensor:
                    {
                        var tensorData = attrs.GetTensorVal(def.AttributeName);
                        var paramName = tensorData?.DType.GenericTypeParamName;
                        if (paramName is null) continue;

                        if (specializedDTypesWithParam.TryGetValue(paramName, out var targetDType))
                        {
                            rebuilt[def.AttributeName] =
                                TensorDataConversion.ConvertTensorDataType(tensorData!, targetDType);
                            anyUpdated = true;
                        }
                        break;
                    }
                }
            }

            if (anyUpdated)
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(rebuilt, attrs.AttributeDefs);
        }
    }
}
