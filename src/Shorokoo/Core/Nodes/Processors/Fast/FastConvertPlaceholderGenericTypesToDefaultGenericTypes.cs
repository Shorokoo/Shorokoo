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
using System.Diagnostics;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast equivalent of
    /// <c>ConvertPlaceholderGenericTypesToDefaultGenericTypes</c>.
    /// In an initial graph build, generic-type DTypes are placeholder stand-ins
    /// (<c>DType.GenericType1</c> .. <c>DType.GenericType8</c>) without metadata. This
    /// processor replaces them with default DTypes that carry both a base concrete type
    /// (e.g. float32) and the source-level generic parameter name (e.g. "T") via
    /// <see cref="DType.CreateWithGenericParam"/>.
    ///
    /// Operates purely as an attribute rewrite on <see cref="FastNode"/>s — Fast tensors
    /// carry no per-tensor type info, so no re-inference step is needed.
    /// </summary>
    internal static class FastConvertPlaceholderGenericTypesToDefaultGenericTypes
    {
        public static void Process(InternalComputationGraph graph, Dictionary<int, string> genericIndexToParamName)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (genericIndexToParamName.Count == 0) return;

            var genericInputNodes = graph.Nodes
                .Where(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                .ToList();

            var defaultTypes = DetermineDefaultTypes(genericInputNodes, genericIndexToParamName);

            foreach (var node in graph.Nodes)
                RewriteNodeAttributes(node, defaultTypes);
        }

        private static Dictionary<int, DType> DetermineDefaultTypes(
            List<FastNode> genericInputNodes,
            Dictionary<int, string> genericParamNames)
        {
            var defaultTypes = new Dictionary<int, DType>();

            foreach (var node in genericInputNodes)
            {
                var placeholderDType = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype).AssertNotNull();
                var genericIndex = placeholderDType.GenericTypeIndex.AssertNotNull();

                if (defaultTypes.ContainsKey(genericIndex))
                    continue;

                string[]? constraintsStr = null;
                if (node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrGenericTypeConstraints))
                    constraintsStr = node.Attributes.GetStringsVal(OnnxOpAttributeNames.ShrkAttrGenericTypeConstraints);

                var baseDType = DetermineStandInTypeFromConstraints(constraintsStr);
                var paramName = genericParamNames[genericIndex];
                defaultTypes[genericIndex] = DType.CreateWithGenericParam(baseDType, paramName);
            }

            Debug.Assert(genericParamNames.All(x => defaultTypes.ContainsKey(x.Key)));
            return defaultTypes;
        }

        private static void RewriteNodeAttributes(FastNode node, Dictionary<int, DType> defaultTypes)
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
                        if (dt is null || !dt.IsGenericType) continue;

                        rebuilt[def.AttributeName] = defaultTypes[dt.GenericTypeIndex.AssertNotNull()];
                        anyUpdated = true;
                        break;
                    }
                    case AttributeType.DTypes:
                    {
                        var dtypes = attrs.GetDTypesVal(def.AttributeName);
                        if (dtypes is null || dtypes.Length == 0) continue;
                        if (!dtypes.Any(x => x.IsGenericType)) continue;

                        var updated = dtypes.Select(dt => dt.IsGenericType
                            ? defaultTypes[dt.GenericTypeIndex.AssertNotNull()]
                            : dt).ToArray();
                        rebuilt[def.AttributeName] = updated;
                        anyUpdated = true;
                        break;
                    }
                    case AttributeType.Tensor:
                    {
                        var tensorData = attrs.GetTensorVal(def.AttributeName);
                        if (tensorData is null) continue;

                        // Detect TensorData<IGenericTypeN> via the runtime element type, mirroring
                        // the CG-side ProcessTensorDataWithGenericTypes path.
                        var tensorDataType = tensorData.GetType();
                        if (!tensorDataType.IsGenericType) continue;
                        var genericArgs = tensorDataType.GetGenericArguments();
                        if (genericArgs.Length == 0) continue;
                        var typeParam = genericArgs[0];
                        if (!typeof(IGenericType).IsAssignableFrom(typeParam)) continue;

                        int genericIndex;
                        if (typeParam == typeof(IGenericType1)) genericIndex = 1;
                        else if (typeParam == typeof(IGenericType2)) genericIndex = 2;
                        else if (typeParam == typeof(IGenericType3)) genericIndex = 3;
                        else if (typeParam == typeof(IGenericType4)) genericIndex = 4;
                        else if (typeParam == typeof(IGenericType5)) genericIndex = 5;
                        else if (typeParam == typeof(IGenericType6)) genericIndex = 6;
                        else if (typeParam == typeof(IGenericType7)) genericIndex = 7;
                        else if (typeParam == typeof(IGenericType8)) genericIndex = 8;
                        else continue;

                        var defaultDType = defaultTypes[genericIndex];
                        rebuilt[def.AttributeName] =
                            TensorDataConversion.ConvertTensorDataType(tensorData, defaultDType);
                        anyUpdated = true;
                        break;
                    }
                }
            }

            if (anyUpdated)
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(rebuilt, attrs.AttributeDefs);
        }

        private static DType DetermineStandInTypeFromConstraints(string[]? constraintNames)
        {
            if (constraintNames == null || constraintNames.Length == 0)
                return DType.Float32;

            var constraintTypes = new List<Type>();
            foreach (var name in constraintNames)
            {
                Type? constraintType = name switch
                {
                    "FloatLike" => typeof(FloatLike),
                    "IntLike" => typeof(IntLike),
                    "SignedIntLike" => typeof(SignedIntLike),
                    "UnsignedIntLike" => typeof(UnsignedIntLike),
                    "NumLike" => typeof(NumLike),
                    "IndexLike" => typeof(IndexLike),
                    "SimpleFloatLike" => typeof(SimpleFloatLike),
                    "Int8Like" => typeof(Int8Like),
                    _ => null,
                };
                if (constraintType != null)
                    constraintTypes.Add(constraintType);
            }

            if (constraintTypes.Count == 0)
                return DType.Float32;

            foreach (var candidateType in GraphBuilder.OrderedMLDataTypes)
            {
                if (constraintTypes.All(c => c.IsAssignableFrom(candidateType)))
                {
                    var dtype = OnnxUtils.GetDType(candidateType);
                    return dtype ?? DType.Float32;
                }
            }

            return DType.Float32;
        }
    }
}
