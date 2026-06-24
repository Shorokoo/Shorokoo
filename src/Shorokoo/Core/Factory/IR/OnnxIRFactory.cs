using Shorokoo;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Factory.OpsFactories;
using Shorokoo.Core.Factory;

namespace Shorokoo.Core.Factory.IR
{
    internal class OnnxIRFactory
    {
        public static ValueInfoProto CreateTensorInfo(TensorShapeProto? dims, string name, DType type, DataStructure structure, string? targetFunctionName, string? inputTypeName, float? defaultValue = null)
        {
            var valueInfo = new ValueInfoProto();
            valueInfo.Name = name;
            valueInfo.Type = new TypeProto();

            // Handle TensorStruct separately since it has a different serialization approach
            if (structure == DataStructure.TensorStruct)
            {
                // TensorStruct is serialized as a tensor type with a custom ElemType in the 2000+ range.
                // The shape is dimensionless since TensorStruct is a composite type, not a tensor with dimensions.
                // The TensorStructDef metadata will be stored separately in model metadata when writing the full model.
                var structTensorType = new TypeProto.Tensor();
                structTensorType.ElemType = type.ProtoTypeNum; // 2000+ range for TensorStruct DTypes
                // No shape = dimensionless (TensorStruct is a composite type)
                valueInfo.Type.TensorType = structTensorType;
            }
            else
            {
                // For Tensor, Sequence, Optional - use standard toElemType conversion
                var tensorType = new TypeProto.Tensor();
                if (dims is not null) tensorType.Shape = (TensorShapeProto)dims;
                tensorType.ElemType = type.ProtoTypeNum;
                
                // Store generic type parameter name in Denotation field if present
                if (type.GenericTypeParamName != null)
                {
                    valueInfo.Type.Denotation = type.GenericTypeParamName;
                }
                
                if (structure == DataStructure.Tensor)
                    valueInfo.Type.TensorType = tensorType;
                else if (structure == DataStructure.Sequence)
                {
                    var elemType = new TypeProto();
                    elemType.TensorType = tensorType;
                    
                    // Also set denotation on sequence element type
                    if (type.GenericTypeParamName != null)
                    {
                        elemType.Denotation = type.GenericTypeParamName;
                    }

                    var sequenceType = new TypeProto.Sequence();
                    sequenceType.ElemType = elemType;
                    valueInfo.Type.SequenceType = sequenceType;
                }
                else if (structure == DataStructure.Optional)
                {
                    var elemType = new TypeProto();
                    elemType.TensorType = tensorType;
                    
                    // Also set denotation on optional element type
                    if (type.GenericTypeParamName != null)
                    {
                        elemType.Denotation = type.GenericTypeParamName;
                    }

                    var optionalType = new TypeProto.Optional();
                    optionalType.ElemType = elemType;
                    valueInfo.Type.OptionalType = optionalType;
                }
            }

            if (targetFunctionName is not null)
            {
                var targetFunctionEntry = new StringStringEntryProto();
                targetFunctionEntry.Key = Function.IRFunctionSignatureParamName;
                targetFunctionEntry.Value = targetFunctionName;
                valueInfo.MetadataProps.Add(targetFunctionEntry);
            }

            if (inputTypeName is not null)
            {
                var inputTypeEntry = new StringStringEntryProto();
                inputTypeEntry.Key = Function.IRInputTypeName;
                inputTypeEntry.Value = inputTypeName;
                valueInfo.MetadataProps.Add(inputTypeEntry);
            }

            if (defaultValue is float dv)
            {
                var defaultValueEntry = new StringStringEntryProto();
                defaultValueEntry.Key = Function.IRDefaultValue;
                defaultValueEntry.Value = dv.ToString(System.Globalization.CultureInfo.InvariantCulture);
                valueInfo.MetadataProps.Add(defaultValueEntry);
            }

            return valueInfo;
        }

        public static TensorShapeProto CreateDims(TensorDim[] dims, string tensorName)
        {
            var shape = new TensorShapeProto();
            foreach (var dim in dims)
            {
                var newDim = new TensorShapeProto.Dimension();
                if (dim.Size is not null)
                    newDim.DimValue = dim.Size.Value;
                else if (dim.Symbol is not null)
                    newDim.DimParam = dim.Symbol;
                else
                    newDim.DimParam = $"{tensorName}_dim{shape.Dims.Count}";

                shape.Dims.Add(newDim);
            }

            return shape;
        }

        public static ModelProto CreateModel(GraphProto graph, FunctionProto[] functions, OpSetVersion opSetVersion)
        {
            var model = new ModelProto();
            model.IrVersion = (int)IR.Version.IrVersion;
            model.ProducerName = ShorokooVersion.Name;
            model.ProducerVersion = ShorokooVersion.VersionString;
            model.Functions.AddAll(functions);

            var opset = new OperatorSetIdProto();
            opset.Version = (int)opSetVersion;
            opset.Domain = "";
            model.OpsetImports.Add(opset);

            var funcSet = new OperatorSetIdProto();
            funcSet.Version = 1;
            funcSet.Domain = "Functions";
            model.OpsetImports.Add(funcSet);

            model.Graph = (GraphProto)graph;

            return model;
        }

        public static GraphProto CreateGraph(string graphName, TensorProto[] initializers, ValueInfoProto[] inputs, ValueInfoProto[] outputs, NodeProto[] nodes)
        {
            var graph = new GraphProto();
            graph.Name = graphName;
            graph.Initializers.AddAll(initializers.Cast<TensorProto>());
            graph.Inputs.AddAll(inputs.Cast<ValueInfoProto>());
            graph.Outputs.AddAll(outputs.Cast<ValueInfoProto>());

            Debug.Assert(Visitors.IsTopologicallyOrdered(nodes.Cast<NodeProto>()),
                $"OnnxIRFactory.CreateGraph: nodes for graph '{graphName}' are not in topological order");
            graph.Nodes.AddAll(nodes.Cast<NodeProto>());
            return graph;
        }

        public static NodeProto CreateNode(string name, string opCode, string domain, OpSetVersion version, string[] inputTensors, string[] outputTensors, OnnxProtoAttributes attributes, Dictionary<string, GraphProto> graphAttributes, string? identifierTemplate, string? stackTrace, string? nodeKeyGuid = null, bool isEtherealIdentity = false)
        {
            var node = new NodeProto();
            node.Name = name;
            node.OpType = opCode;

            if (domain is not null && domain != "")
                node.Domain = domain;

            foreach (var inputTensor in inputTensors) 
                node.Inputs.Add(inputTensor);

            foreach (var outputTensor in outputTensors)
                node.Outputs.Add(outputTensor);

            foreach (var attributeDef in attributes.attributeDefs.Where(x => !x.IsInternalAttribute))
            {
                var attribute = new AttributeProto();
                attribute.Name = attributeDef.AttributeName;
                attribute.Type = attributeDef.Type.ToProto();
                if (!attributes.IsDefaultValue(attribute.Name))
                {
                    switch (attribute.Type)
                    {
                        case AttributeProto.AttributeType.String:
                            attribute.S = Encoding.UTF8.GetBytes(attributes.GetStringVal(attribute.Name).AssertNotNull());
                            break;
                        case AttributeProto.AttributeType.Strings:
                            attribute.Strings.AddAll(attributes.GetStringsVal(attribute.Name).AssertNotNull().Select(Encoding.UTF8.GetBytes));
                            break;
                        case AttributeProto.AttributeType.Int:
                            // Check if this is a DType attribute by looking at the Shorokoo attribute type
                            if (attributeDef.Type == AttributeType.DType)
                            {
                                var dtype = attributes.GetDTypeVal(attribute.Name).AssertNotNull();
                                attribute.I = dtype.ProtoTypeNum;
                                
                                // Store generic param name in RefAttrName if present
                                if (dtype.GenericTypeParamName != null)
                                {
                                    attribute.RefAttrName = $"GenericParam:{dtype.GenericTypeParamName}";
                                }
                            }
                            else
                            {
                                // Regular integer attribute
                                attribute.I = attributes.GetLongVal(attribute.Name).AssertNotNull();
                            }
                            break;
                        case AttributeProto.AttributeType.Ints:
                            // Check if this is a DType array attribute
                            if (attributeDef.Type == AttributeType.DTypes)
                            {
                                var dtypes = attributes.GetDTypesVal(attribute.Name).AssertNotNull();
                                attribute.Ints = dtypes.Select(d => (long)d.ProtoTypeNum).ToArray();
                                
                                // Store generic param names in RefAttrName if any present
                                var paramNames = dtypes.Select(d => d.GenericTypeParamName ?? "").ToArray();
                                if (paramNames.Any(p => !string.IsNullOrEmpty(p)))
                                {
                                    attribute.RefAttrName = $"GenericParams:{string.Join(",", paramNames)}";
                                }
                            }
                            else
                            {
                                // Regular integer array attribute
                                attribute.Ints = attributes.GetLongsVal(attribute.Name).AssertNotNull();
                            }
                            break;
                        case AttributeProto.AttributeType.Float:
                            attribute.F = attributes.GetFloatVal(attribute.Name).AssertNotNull();
                            break;
                        case AttributeProto.AttributeType.Floats:
                            attribute.Floats = attributes.GetFloatsVal(attribute.Name).AssertNotNull();
                            break;
                        case AttributeProto.AttributeType.Tensor:
                            var tensor = attributes.GetTensorVal(attribute.Name).AssertNotNull();
                            attribute.T = (TensorProto)OnnxIRFactory.CreateTensor(
                                tensor.Shape.Dims,
                                name: null, // $"{name}_Tensor",
                                tensor.DType,
                                identifierTemplate: null,
                                isTrainable: true,  // Tensor attributes are not state params
                                tensor.AccessRawMemory().ToArray());
                            attribute.Type = AttributeProto.AttributeType.Tensor;
                            break;
                        case AttributeProto.AttributeType.Tensors:
                            throw new UnsupportedDTypeException(ErrorCodes.FW034, "Tensors", "type conversion", 
                                "Tensors type conversion method not implemented");
                            // var tensors = attributes.GetTensorsVal(attribute.Name).AssertNotNull();
                            // for (int i = 0; i < tensors.Length; i++)
                            // {
                            //     var t = tensors[i];
                            //     attribute.Tensors.Add((TensorProto)this.CreateTensor(
                            //         t.IsNullDim ? null : t.TensorData.Shape.Dims,
                            //         null, // $"{name}_Tensor_{i}",
                            //         t.TensorData.Type,
                            //         t.TensorData.AccessRawMemory().ToArray()));
                            // }
                            // attribute.Type = AttributeProto.AttributeType.Tensors;
                            // break;
                        case AttributeProto.AttributeType.Graph:
                            attribute.G = (GraphProto)graphAttributes[attribute.Name];
                            break;
                        case AttributeProto.AttributeType.Graphs:
                            throw new UnsupportedDTypeException(ErrorCodes.FW035, "Graphs", "array element type conversion", 
                                "Array element type conversion not implemented for Graphs");
                            // attribute.Graphs.AddAll(graphInfos.Select(g => (GraphProto)graphAttributes[attribute.Name]));
                            // attribute.Type = AttributeProto.AttributeType.Graphs;
                            // break;
                        case AttributeProto.AttributeType.TypeProto:
                            attribute.Tp = attributes.GetTypeProtoVal(attribute.Name).AssertNotNull();
                            break;
                        default:
                            throw new UnsupportedDTypeException(ErrorCodes.FW036, attribute.Type.ToString(), "optional type conversion", 
                                $"Optional type conversion not implemented for attribute type '{attribute.Type}'");
                    }

                    node.Attributes.Add(attribute);
                }
            }

            if (node.Attributes.Count(x => x.Type == AttributeProto.AttributeType.Graph) != graphAttributes.Count)
            {
                node.Attributes.AddAll(graphAttributes.OrderBy(x => x.Key).Select(x =>
                    new AttributeProto
                    {
                        Name = x.Key,
                        G = x.Value,
                        Type = AttributeProto.AttributeType.Graph
                    }));
            }

            if (stackTrace is not null)
            {
                var metataProp = new StringStringEntryProto();
                metataProp.Key = "StackTrace";
                metataProp.Value = stackTrace;
                node.MetadataProps.Add(metataProp);
            }

            if (isEtherealIdentity)
            {
                Debug.Assert(node.OpType == OpCodes.IDENTITY);
                var metataProp = new StringStringEntryProto();
                metataProp.Key = OnnxOpAttributeNames.ShrkMetaIdentityNodeEthereal;
                metataProp.Value = "true";
                node.MetadataProps.Add(metataProp);
            }

            if (identifierTemplate is not null)
            {
                var metataProp = new StringStringEntryProto();
                metataProp.Key = OnnxOpAttributeNames.ShrkMetaNodeIdentifierTemplate;
                metataProp.Value = identifierTemplate;
                node.MetadataProps.Add(metataProp);
            }

            if (nodeKeyGuid is not null)
            {
                var metataProp = new StringStringEntryProto();
                metataProp.Key = OnnxOpAttributeNames.ShrkMetaNodeKey;
                metataProp.Value = nodeKeyGuid;
                node.MetadataProps.Add(metataProp);
            }

            return node;
        }

        public static TensorProto CreateTensor(long[]? dims, string? name, DType type, string? identifierTemplate, bool isTrainable, byte[] data)
        {
            var tensor = new TensorProto();
            
            if (dims is not null)
                tensor.Dims = dims;

            if (name is not null)
                tensor.Name = name;

            tensor.RawData = data;
            tensor.data_type = type.ProtoTypeNum;
            
            if (identifierTemplate is not null)
            {
                var idTemplateEntry = new StringStringEntryProto();
                idTemplateEntry.Key = OnnxOpAttributeNames.ShrkMetaNodeIdentifierTemplate;
                idTemplateEntry.Value = identifierTemplate;
                tensor.MetadataProps.Add(idTemplateEntry);
            }
            
            // Serialize isTrainable as metadata
            var isTrainableEntry = new StringStringEntryProto();
            isTrainableEntry.Key = OnnxOpAttributeNames.ShrkMetaIsTrainable;
            isTrainableEntry.Value = isTrainable.ToString().ToLowerInvariant();
            tensor.MetadataProps.Add(isTrainableEntry);
            
            return tensor;
        }
    }
}
