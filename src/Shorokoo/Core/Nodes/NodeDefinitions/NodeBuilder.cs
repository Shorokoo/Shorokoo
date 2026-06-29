
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Text;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{

    internal struct VariableTypeInfo
    {
        public readonly DType DType { get; }
        public readonly Function? ModuleFn { get; }

        public VariableTypeInfo(DType dtype, Function? moduleFn)
        {
            DType = dtype;
            ModuleFn = moduleFn;
        }

        public static implicit operator VariableTypeInfo(DType dtype) => new VariableTypeInfo(dtype, null);
        public static implicit operator VariableTypeInfo((DType dtype, Function? moduleFn) info) => new VariableTypeInfo(info.dtype, info.moduleFn);
    }

    public class OnnxProtoAttributes
    {
        private ImmutableDictionary<string, object?> attributeVals { get; }
        public ImmutableList<NodeDefAttributeDef> attributeDefs { get; }

        public static OnnxProtoAttributes FromProtoVals(Dictionary<string, object?> attrs, ImmutableList<NodeDefAttributeDef> defs)
        {
            return new OnnxProtoAttributes(attrs, defs);
        }

        public static OnnxProtoAttributes FromCSharpVals(Dictionary<string, object?> attrs, ImmutableList<NodeDefAttributeDef> defs)
        {
            return OnnxCSharpAttributes.FromCSharpVals(attrs, defs).ToProto();
        }

        private OnnxProtoAttributes(Dictionary<string, object?> attrs, ImmutableList<NodeDefAttributeDef> defs)
        {
            var convertedAttrs = new Dictionary<string, object?>();
            foreach (var kvp in attrs)
            {
                if (kvp.Value is float floatVal)
                    convertedAttrs[kvp.Key] = (float?)floatVal;
                else if (kvp.Value is long longVal)
                    convertedAttrs[kvp.Key] = (long?)longVal;
                else if (kvp.Value is string strVal)
                    convertedAttrs[kvp.Key] = (string?)strVal;
                else
                    convertedAttrs[kvp.Key] = kvp.Value;
            }

            foreach (var def in defs)
            {
                if (!convertedAttrs.ContainsKey(def.AttributeName))
                    convertedAttrs[def.AttributeName] = null;
            }

            this.attributeVals = convertedAttrs.ToImmutableDictionary();
            this.attributeDefs = defs;
        }

        public OnnxCSharpAttributes ToCSharp()
        {
            var protoVals = new Dictionary<string, object?>();
            foreach (var def in this.attributeDefs)
            {
                if (this.attributeVals.TryGetValue(def.AttributeName, out var attrVal))
                {
                    if (attrVal is null)
                    {
                        protoVals[def.AttributeName] = attrVal;
                        continue;
                    }

                    switch (def.Type)
                    {
                        case AttributeType.Float:
                        case AttributeType.Floats:
                        case AttributeType.Long:
                        case AttributeType.Longs:
                        case AttributeType.String:
                        case AttributeType.Strings:
                        case AttributeType.Tensor:
                        case AttributeType.Graph:
                            protoVals[def.AttributeName] = attrVal;
                            break;
                        case AttributeType.Bool:
                            protoVals[def.AttributeName] = ((long)attrVal) == 1;
                            break;
                        case AttributeType.Bools:
                            protoVals[def.AttributeName] = ((long[])attrVal).Select(x => x == 1).ToArray();
                            break;
                        case AttributeType.Enum:
                            Debug.Assert(def.EnumDef is not null);
                            protoVals[def.AttributeName] = def.EnumDef.ToCSharpVal((string)attrVal);
                            break;
                        case AttributeType.Enums:
                            Debug.Assert(def.EnumDef is not null);
                            protoVals[def.AttributeName] = ((string[])attrVal).Select(x => def.EnumDef.ToCSharpVal(x)).ToArray();
                            break;
                        case AttributeType.DType:
                            // attrVal may be a long (ProtoTypeNum) when coming from AutoTest, or a DType object
                            protoVals[def.AttributeName] = attrVal is long dtypeLong ? DType.FromProtoTypeNum((int)dtypeLong) : (DType)attrVal;
                            break;
                        case AttributeType.DTypes:
                            // attrVal may be a long[] (ProtoTypeNums) when coming from AutoTest, or a DType[] object
                            protoVals[def.AttributeName] = attrVal is long[] dtypeLongs ? dtypeLongs.Select(l => DType.FromProtoTypeNum((int)l)).ToArray() : (DType[])attrVal;
                            break;
                        case AttributeType.TypeProto:
                            protoVals[def.AttributeName] = fromTypeProto((TypeProto)attrVal);
                            break;
                    }
                }
            }

            return OnnxCSharpAttributes.FromCSharpVals(protoVals, this.attributeDefs);
        }

        private static (DataStructure structure, DType type) fromTypeProto(TypeProto tproto)
        {
            DataStructure dataStructure;
            DType type;

            if (tproto.SequenceType is not null)
            {
                dataStructure = DataStructure.Sequence;
                type = tproto.SequenceType.ElemType.TensorType.ElemType;
            }
            else if (tproto.TensorType is not null)
            {
                dataStructure = DataStructure.Tensor;
                type = tproto.TensorType.ElemType;
            }
            else
            {
                Debug.Assert(tproto.OptionalType is not null,
                    "Unsupported TypeProto structure - Map and SparseTensor are not yet implemented");
                dataStructure = DataStructure.Optional;
                type = tproto.OptionalType!.ElemType.TensorType.ElemType;
            }

            return (dataStructure, type);
        }

        internal static TypeProto toTypeProto((DataStructure structure, DType type) cstproto)
        {
            var tproto = new TypeProto();
            switch (cstproto.structure)
            {
                case DataStructure.Sequence:
                    tproto.SequenceType = new TypeProto.Sequence()
                    {
                        ElemType = new TypeProto()
                        {
                            TensorType = new TypeProto.Tensor()
                            {
                                ElemType = cstproto.type.ProtoTypeNum
                            }
                        }
                    };
                    break;
                case DataStructure.Tensor:
                    tproto.TensorType = new TypeProto.Tensor()
                    {
                        ElemType = cstproto.type.ProtoTypeNum
                    };
                    break;
                case DataStructure.Optional:
                    tproto.OptionalType = new TypeProto.Optional()
                    {
                        ElemType = new TypeProto()
                        {
                            TensorType = new TypeProto.Tensor()
                            {
                                ElemType = cstproto.type.ProtoTypeNum
                            }
                        }
                    };
                    break;
                //case DataStructure.Map:
                //    tproto.MapType = new TypeProto.Map();
                //    break;
                //case DataStructure.SparseTensor:
                //    tproto.SparseTensorType = new TypeProto.SparseTensor();
                //    break;
                default:
                    Debug.Fail($"Unsupported DataStructure {cstproto.structure} - Map and SparseTensor are not yet implemented");
                    break;
            }

            return tproto;
        }

        public float? GetFloatVal(string name) => (float?)this.attributeVals[name];
        public float[]? GetFloatsVal(string name) => (float[]?)this.attributeVals[name];
        public long? GetLongVal(string name) => (long?)this.attributeVals[name];
        public long[]? GetLongsVal(string name) => (long[]?)this.attributeVals[name];
        public string? GetStringVal(string name) => (string?)this.attributeVals[name];
        public string[]? GetStringsVal(string name) => (string[]?)this.attributeVals[name];
        public TensorData? GetTensorVal(string name) => (TensorData?)this.attributeVals[name];
        public TypeProto? GetTypeProtoVal(string name) => (TypeProto?)this.attributeVals[name];
        public DType? GetDTypeVal(string name) => (DType?)this.attributeVals[name];
        public DType[]? GetDTypesVal(string name) => (DType[]?)this.attributeVals[name];
        public bool IsDefaultValue(string name) => this.attributeVals[name] is null;
    }

    public class OnnxCSharpAttributes
    {
        private ImmutableDictionary<string, object?> attributeVals { get; }
        public ImmutableList<NodeDefAttributeDef> AttributeDefs { get; }

        public static OnnxCSharpAttributes FromCSharpVals(Dictionary<string, object?> attrs, ImmutableList<NodeDefAttributeDef> defs)
        {
            return new OnnxCSharpAttributes(attrs, defs);
        }

        public OnnxCSharpAttributes SetAttributes(params (string AttributeName, object? AttributeValue)[] newAttributeVals)
        {
            var newVals = attributeVals.ToBuilder();
            foreach (var newVal in newAttributeVals)
                newVals[newVal.AttributeName] = newVal.AttributeValue;

            return new OnnxCSharpAttributes(newVals.ToDictionary(), this.AttributeDefs);
        }

        private OnnxCSharpAttributes(Dictionary<string, object?> attrs, ImmutableList<NodeDefAttributeDef> defs)
        {
            var convertedAttrs = new Dictionary<string, object?>();
            foreach (var kvp in attrs)
            {
                var def = defs.Single(x => x.AttributeName == kvp.Key);
                if (kvp.Value is null)
                    convertedAttrs[kvp.Key] = null;
                else if (def.Type == AttributeType.Enum)
                    convertedAttrs[kvp.Key] = kvp.Value;
                else if (def.Type == AttributeType.Enums)
                    convertedAttrs[kvp.Key] = kvp.Value;
                else if (kvp.Value is float floatVal)
                    convertedAttrs[kvp.Key] = (float?)floatVal;
                else if (kvp.Value is long longVal)
                    convertedAttrs[kvp.Key] = (long?)longVal;
                else if (kvp.Value is int intVal)
                    convertedAttrs[kvp.Key] = (long?)(int?)intVal;
                else if (kvp.Value is int[] intVals)
                    convertedAttrs[kvp.Key] = (long[]?)intVals.Select(x => (long)x).ToArray();
                else if (kvp.Value is int?[] intqVals)
                    convertedAttrs[kvp.Key] = (long[]?)intqVals.Select(x => (long)x.AssertNotNull()).ToArray();
                else if (kvp.Value is string strVal)
                    convertedAttrs[kvp.Key] = (string?)strVal;
                else if (kvp.Value is bool boolVal)
                    convertedAttrs[kvp.Key] = (bool?)boolVal;
                else
                    convertedAttrs[kvp.Key] = kvp.Value;

                if (kvp.Value is not null)
                {
                    if (convertedAttrs[kvp.Key] is bool?)
                        Debug.Assert(def.Type == AttributeType.Bool,
                            $"bool? attribute '{def.AttributeName}': expected AttributeType.Bool but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is bool[])
                        Debug.Assert(def.Type == AttributeType.Bools,
                            $"bool[] attribute '{def.AttributeName}': expected AttributeType.Bools but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is long?)
                        Debug.Assert(def.Type == AttributeType.Long,
                            $"long? attribute '{def.AttributeName}': expected AttributeType.Long but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is long[])
                        Debug.Assert(def.Type == AttributeType.Longs,
                            $"long[] attribute '{def.AttributeName}': expected AttributeType.Longs but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is float?)
                        Debug.Assert(def.Type == AttributeType.Float,
                            $"float? attribute '{def.AttributeName}': expected AttributeType.Float but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is float[])
                        Debug.Assert(def.Type == AttributeType.Floats,
                            $"float[] attribute '{def.AttributeName}': expected AttributeType.Floats but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is string)
                        Debug.Assert(def.Type == AttributeType.String,
                            $"string attribute '{def.AttributeName}': expected AttributeType.String but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is string[])
                        Debug.Assert(def.Type == AttributeType.Strings,
                            $"string[] attribute '{def.AttributeName}': expected AttributeType.Strings but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is TensorData)
                        Debug.Assert(def.Type == AttributeType.Tensor,
                            $"TensorData attribute '{def.AttributeName}': expected AttributeType.Tensor but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is DType)
                        Debug.Assert(def.Type == AttributeType.DType,
                            $"DType attribute '{def.AttributeName}': expected AttributeType.DType but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is DType[])
                        Debug.Assert(def.Type == AttributeType.DTypes,
                            $"DType[] attribute '{def.AttributeName}': expected AttributeType.DTypes but got {def.Type}");
                    else if (convertedAttrs[kvp.Key] is (DataStructure structure, DType type))
                        Debug.Assert(def.Type == AttributeType.TypeProto,
                            $"(DataStructure, DType) attribute '{def.AttributeName}': expected AttributeType.TypeProto but got {def.Type}");
                    else
                        Debug.Assert(
                            def.Type == AttributeType.Enum || def.Type == AttributeType.Enums || def.Type == AttributeType.Graph,
                            $"unknown type attribute '{def.AttributeName}': expected Enum, Enums, or Graph but got {def.Type}");
                }
            }

            foreach (var def in defs)
            {
                if (!convertedAttrs.ContainsKey(def.AttributeName))
                    convertedAttrs[def.AttributeName] = null;
            }

            this.attributeVals = convertedAttrs.ToImmutableDictionary();
            this.AttributeDefs = defs;
        }

        public OnnxProtoAttributes ToProto()
        {
            var protoVals = new Dictionary<string, object?>();
            foreach (var def in this.AttributeDefs)
            {
                if (this.attributeVals.TryGetValue(def.AttributeName, out var attrVal))
                {
                    if (attrVal is null)
                    {
                        protoVals[def.AttributeName] = attrVal;
                        continue;
                    }

                    switch (def.Type)
                    {
                        case AttributeType.Float:
                        case AttributeType.Floats:
                        case AttributeType.Long:
                        case AttributeType.Longs:
                        case AttributeType.String:
                        case AttributeType.Strings:
                        case AttributeType.Tensor:
                        case AttributeType.Graph:
                            protoVals[def.AttributeName] = attrVal;
                            break;
                        case AttributeType.Bool:
                            protoVals[def.AttributeName] = ((bool)attrVal) ? 1L : 0L;
                            break;
                        case AttributeType.Bools:
                            protoVals[def.AttributeName] = ((bool[])attrVal).Select(x => x ? 0L : 1L).ToArray();
                            break;
                        case AttributeType.Enum:
                            Debug.Assert(def.EnumDef is not null);
                            protoVals[def.AttributeName] = def.EnumDef.ToOnnxName(attrVal);
                            break;
                        case AttributeType.Enums:
                            Debug.Assert(def.EnumDef is not null);
                            protoVals[def.AttributeName] = ((System.Collections.IEnumerable)attrVal).Cast<object>().Select(x => def.EnumDef.ToOnnxName(x)).ToArray();
                            break;
                        case AttributeType.DType:
                            // Keep as DType object, don't convert to long
                            protoVals[def.AttributeName] = (DType)attrVal;
                            break;
                        case AttributeType.DTypes:
                            // Keep as DType array, don't convert to long array
                            protoVals[def.AttributeName] = (DType[])attrVal;
                            break;
                        case AttributeType.TypeProto:
                            protoVals[def.AttributeName] = OnnxProtoAttributes.toTypeProto(((DataStructure, DType))attrVal);
                            break;
                    }
                }
            }

            return OnnxProtoAttributes.FromProtoVals(protoVals, this.AttributeDefs);
        }

        public ImmutableDictionary<string, object?> GetAttributeVals() => this.attributeVals;

        public bool IsAttributeDefined(string name) => this.AttributeDefs.Any(x => x.AttributeName == name);

        public object? GetEnumVal(string name) => this.attributeVals[name];
        public object[]? GetEnumsVal(string name) => (object[]?)this.attributeVals[name];
        public T? GetEnumVal<T>(string name) where T : struct => (T?)this.attributeVals[name];
        public T[]? GetEnumsVal<T>(string name) where T : struct => ((object[]?)this.attributeVals[name])?.Cast<T>().ToArray();
        public bool? GetBoolVal(string name) => (bool?)this.attributeVals[name];
        public DType? GetDTypeVal(string name) => (DType?)this.attributeVals[name];
        public DType[]? GetDTypesVal(string name) => (DType[]?)this.attributeVals[name];
        public float? GetFloatVal(string name) => (float?)this.attributeVals[name];
        public float[]? GetFloatsVal(string name) => (float[]?)this.attributeVals[name];
        public long? GetLongVal(string name) => (long?)this.attributeVals[name];
        public int[]? GetIntsVal(string name) => GetLongsVal(name)?.Select(x => (int)x).ToArray();
        public long[]? GetLongsVal(string name) => (long[]?)this.attributeVals[name];
        public string? GetStringVal(string name) => (string?)this.attributeVals[name];
        public string[]? GetStringsVal(string name) => (string[]?)this.attributeVals[name];
        public TensorData? GetTensorVal(string name) => (TensorData?)this.attributeVals[name];
        public (DataStructure structure, DType dtype)? GetTypeProtoVal(string name) => ((DataStructure, DType)?)this.attributeVals[name];
        public BestGraphAttribute? GetGraphVal(string name) => (BestGraphAttribute?)this.attributeVals[name];
        public object? GetAttributeObj(string name) => this.attributeVals[name]; 
        public bool IsDefaultValue(string name) => this.attributeVals[name] is null;
    }
    

    public class OutputTensorInfo
    {
        public required DType DType { get; init; }
        public required Function? ModuleFn { get; init; }
        public required DataStructure Structure { get; init; }
        public required int? Rank { get; init; }

        /// <summary>
        /// A null name means that the output's name is unknown.
        /// An empty string as name means that the output is null (as in an optional output that is to be omitted).
        /// </summary>
        public required string? Name { get; init; }
    }

    public class BestGraphAttribute
    {
        public required string GraphAttributeName { get; init; }
        public string? DefaultGraphName { get; init; }
        public string[]? DefautGraphInputNames { get; init; }
    }

    internal class InputNodeDataProcessor
    {
        public required string? DefaultName { get; init; }
        public required NodeDefinition NodeDef { get; init; }
        public required OnnxProtoAttributes ProtoAttributes { get; init; }

        private OnnxCSharpAttributes? onnxCSharpAttributes;
        public OnnxCSharpAttributes CSharpAttributes
        {
            get
            {
                if (onnxCSharpAttributes is null)
                    this.onnxCSharpAttributes = ProtoAttributes.ToCSharp();

                return this.onnxCSharpAttributes;
            }
        }

        public required ImmutableDictionary<string, Variable?[]> FullInputs { get; init; }

        public string? StackTrace { get; init; }



        private static Variable?[] getInputsForInputDef(NodeDefInputDef inputDef, Variable?[] allInputs, NodeDefInputDef[] allInputDefs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var curInputIndex = 0;
            if (allInputDefs.Length == 2 && allInputDefs.All(x => x.VariadicCountDef is not null))
            {
                Debug.Assert(allInputs.Length % 2 == 0);
                var numItems = allInputs.Length / 2;
                if (allInputDefs[0] == inputDef)
                    return allInputs.Take(numItems).ToArray();
                else
                {
                    Debug.Assert(allInputDefs[1] == inputDef);
                    return allInputs.Skip(numItems).ToArray();
                }
            }

            foreach (var curInputDef in allInputDefs)
            {
                if (curInputDef == inputDef)
                {
                    if (curInputDef.VariadicCountDef is not null)
                    {
                        if (inferredVariadicCounts.TryGetValue(curInputDef.VariadicCountDef.CountDefName, out var count) && count is not null)
                            return allInputs.Skip(curInputIndex).Take(count.Value).ToArray();

                        Debug.Assert(allInputDefs.Last() == curInputDef);
                        return allInputs.Skip(curInputIndex).ToArray();
                    }
                    else if (curInputIndex >= allInputs.Length)
                        return [null];
                    else
                        return [allInputs[curInputIndex]];
                }

                if (curInputDef.VariadicCountDef is not null)
                {
                    var ok = inferredVariadicCounts.TryGetValue(curInputDef.VariadicCountDef.CountDefName, out int? numInputs) && numInputs is not null;
                    Debug.Assert(ok, $"Unable to infer variadic count for input definition '{curInputDef.VariadicCountDef.CountDefName}'");
                    curInputIndex += numInputs!.Value;
                }
                else
                    curInputIndex++;
            }

            Debug.Fail("Failed to calculate proper input index - no matching input definition found");
            return [];
        }

        #region Infer variadic counts

        private static int? identifyInferredVariadicCounts(NodeDefVariadicCountDef variadicDef, NodeDefAttributeDef attributeDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs)
        {
            if (attributeDef.VariadicCount == variadicDef)
            {
                Debug.Assert(attributeDef.Type == AttributeType.Long,
                    $"Expected AttributeType.Long for variadic count attribute '{attributeDef.AttributeName}'");

                var count = attributes.GetLongVal(attributeDef.AttributeName);
                Debug.Assert(count is null || count <= int.MaxValue,
                    $"Count value {count} exceeds int.MaxValue limit");

                return (int?)count;
            }
            else if (attributeDef.Structure?.StructureDefName == variadicDef.CountDefName && attributeDef.Type == AttributeType.Enums)
            {
                var count = attributes.GetEnumsVal(attributeDef.AttributeName)?.Length;
                Debug.Assert(count is null || count <= int.MaxValue,
                    $"Enums count {count} exceeds int.MaxValue limit");

                return count;
            }
            else if (attributeDef.TensorType?.TypeDefName == variadicDef.CountDefName && attributeDef.Type == AttributeType.DTypes)
            {
                var count = attributes.GetDTypesVal(attributeDef.AttributeName)?.Length;
                Debug.Assert(count is null || count <= int.MaxValue,
                    $"DTypes count {count} exceeds int.MaxValue limit");

                return count;
            }

            return null;
        }

        private static int? identifyInferredVariadicCounts(NodeDefVariadicCountDef variadicDef, NodeDefInputDef inputDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            if (inputDef.VariadicCountDef == variadicDef)
            {
                var inputsForDef = getInputsForInputDef(inputDef, inputs, nodeDef.InputDefs.ToArray(), inferredVariadicCounts);
                return inputsForDef.Length;
            }

            return null;
        }

        private static int[] identifyInferredVariadicCounts(NodeDefVariadicCountDef variadicDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts, int? knownNumOutputs)
        {
            var identifiedCounts = new List<int?>();
            if (nodeDef.OutputDefs.Any(x => x.VariadicCountDef == variadicDef) && 
                nodeDef.OutputDefs.Count == 1 && 
                knownNumOutputs is not null)
                identifiedCounts.Add(knownNumOutputs);


            foreach (var attributeDef in nodeDef.AttributeDefs)
                identifiedCounts.Add(identifyInferredVariadicCounts(variadicDef, attributeDef, nodeDef, attributes, inputs));

            foreach (var inputDef in nodeDef.InputDefs)
                identifiedCounts.Add(identifyInferredVariadicCounts(variadicDef, inputDef, nodeDef, attributes, inputs, inferredVariadicCounts));

            return identifiedCounts.NotNulls().ToArray();
        }

        private static ImmutableDictionary<string, int?> identifyInferredVariadicCounts(NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts, int? knownNumOutputs)
        {
            foreach (var variadicDef in nodeDef.VariadicDefs.Values)
            {
                if (inferredVariadicCounts.ContainsKey(variadicDef.CountDefName) && inferredVariadicCounts[variadicDef.CountDefName] is not null)
                    continue;

                var identifiedCounts = identifyInferredVariadicCounts(variadicDef, nodeDef, attributes, inputs, inferredVariadicCounts, knownNumOutputs);
                Debug.Assert(identifiedCounts.Length == 0 || identifiedCounts.All(x => x == identifiedCounts[0]),
                    $"Inconsistent variadic counts identified for variadic def '{variadicDef.CountDefName}' - all counts must be equal");

                if (identifiedCounts.Length != 0)
                    inferredVariadicCounts = inferredVariadicCounts.SetItem(variadicDef.CountDefName, identifiedCounts.First());
            }

            return inferredVariadicCounts;
        }

        #endregion

        #region Infer types

        private static VariableTypeInfo[]? identifyInferredTypes(NodeDefTypeDef variadicDef, NodeDefAttributeDef attributeDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs)
        {
            var attributeName = attributeDef.AttributeName;

            if (attributeDef.TensorType == variadicDef)
            {
                if (attributeDef.Type == AttributeType.DType)
                {
                    var dtype = attributes.GetDTypeVal(attributeName);
                    if (dtype is null) return null;

                    return [dtype];
                }
                else if (attributeDef.Type == AttributeType.DTypes)
                {
                    var dtypes = attributes.GetDTypesVal(attributeDef.AttributeName);
                    if (dtypes is null) return null;

                    return [.. dtypes];
                }
                else if (attributeDef.Type == AttributeType.Tensor)
                {
                    var dtype = attributes.GetTensorVal(attributeDef.AttributeName)?.DType;
                    if (dtype is null) return null;

                    return [dtype];
                }
                else if (attributeDef.Type == AttributeType.TypeProto)
                {
                    var typeProto = attributes.GetTypeProtoVal(attributeDef.AttributeName);
                    if (typeProto is null) return null;
                    return [typeProto.Value.dtype];
                }

                Debug.Fail($"Unknown attribute type {attributeDef.Type} for variadic type definition inference in NodeBuilder");
                return null;
            }

            return null;
        }

        private static VariableTypeInfo[]? identifyInferredTypes(NodeDefTypeDef typeDef, NodeDefInputDef inputDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            if (inputDef.TypeDef == typeDef)
            {
                var inputsForDef = getInputsForInputDef(inputDef, inputs, nodeDef.InputDefs.ToArray(), inferredVariadicCounts);
                if (inputsForDef.Length == 1 && inputsForDef[0] is null)
                    return null;

                Debug.Assert(inputsForDef.All(x => x is not null),
                    $"One or more inputs are null when processing input definition '{inputDef.ParamName}'");

                return [..inputsForDef.AssertNotNulls().Select(x => (x.Type, typeDef.TracksModuleFn ? x.ModuleFn : null))];
            }

            return null;
        }

        private static VariableTypeInfo[][] identifyInferredTypes(NodeDefTypeDef typeDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var identifiedTypes = new List<VariableTypeInfo[]?>();
            foreach (var attributeDef in nodeDef.AttributeDefs)
                identifiedTypes.Add(identifyInferredTypes(typeDef, attributeDef, nodeDef, attributes, inputs));

            foreach (var inputDef in nodeDef.InputDefs)
                identifiedTypes.Add(identifyInferredTypes(typeDef, inputDef, nodeDef, attributes, inputs, inferredVariadicCounts));

            return identifiedTypes.NotNulls().ToArray();
        }

        private static ImmutableDictionary<string, VariableTypeInfo[]?> identifyInferredTypes(NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var inferredTypes = ImmutableDictionary<string, VariableTypeInfo[]?>.Empty;

            foreach (var typeDef in nodeDef.TypeDefs.Values)
            {
                Debug.Assert(!inferredTypes.ContainsKey(typeDef.TypeDefName),
                    $"Duplicate type definition name '{typeDef.TypeDefName}' detected during node definition constraint processing");

                var hardCodedDType = typeDef.GetHardcodedDType();
                if (hardCodedDType is not null)
                {
                    inferredTypes = inferredTypes.SetItem(typeDef.TypeDefName, [hardCodedDType]);
                    continue;
                }

                var identifiedDTypes = identifyInferredTypes(typeDef, nodeDef, attributes, inputs, inferredVariadicCounts);
                Debug.Assert(identifiedDTypes.Length == 0 || identifiedDTypes.All(x => x.Select(y => y.DType).SequenceEqual(identifiedDTypes[0].Select(y => y.DType))),
                    $"Type constraint violation for '{typeDef.TypeDefName}' - identified DTypes do not match expected sequence pattern");

                var dtypesToUse = identifiedDTypes.FirstOrDefault(x => x.FirstOrDefault().ModuleFn is not null) ?? identifiedDTypes.FirstOrDefault();

                inferredTypes = inferredTypes.SetItem(typeDef.TypeDefName, dtypesToUse);
            }

            return inferredTypes;
        }

        #endregion

        #region Infer ranks

        private int?[]? identifyInferredRank(NodeDefRankDef rankDef, NodeDefAttributeDef attributeDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs)
        {
            if (attributeDef.TensorRank == rankDef)
            {
                if (attributeDef.Type == AttributeType.Long)
                {
                    var rank = attributes.GetLongVal(attributeDef.AttributeName);
                    if (rank is null) return null;

                    Debug.Assert(rank <= int.MaxValue,
                        $"Rank value {rank} exceeds int.MaxValue limit for tensor rank processing");

                    // -1 is the legacy "unknown rank" sentinel emitted by Function.Call
                    // (Function.cs L160 packs int? into long[] via `x ?? -1`). Decode it
                    // back to null here so downstream consumers see the int? null they
                    // expect for unknown-rank tensors.
                    return [rank.Value == -1 ? null : (int?)rank.Value];
                }
                else if (attributeDef.Type == AttributeType.Longs)
                {
                    var longRanks = attributes.GetLongsVal(attributeDef.AttributeName);

                    if (longRanks is null) return null;

                    Debug.Assert(longRanks.All(x => x <= int.MaxValue),
                        $"One or more rank values exceed int.MaxValue limit for tensor rank processing: [{string.Join(",", longRanks)}]");

                    // -1 elements are the legacy "unknown rank" sentinel emitted by
                    // Function.Call (Function.cs L160 packs int? into long[] via `x ?? -1`).
                    return longRanks.Select(x => x == -1 ? null : (int?)x).ToArray();
                }
                else if (attributeDef.Type == AttributeType.Tensor)
                {
                    var rank = attributes.GetTensorVal(attributeDef.AttributeName)?.Shape.Dims.Length;
                    if (rank is null) return null;

                    Debug.Assert(rank <= int.MaxValue,
                        $"Tensor rank {rank} exceeds int.MaxValue limit for tensor rank processing");

                    return [(int)rank.Value];
                }

                Debug.Fail($"Unsupported attribute type {attributeDef.Type} for rank inference in NodeBuilder");
                return null;
            }

            return null;
        }

        private int?[]? identifyInferredRank(NodeDefRankDef rankDef, NodeDefInputDef inputDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            if (inputDef.RankDef == rankDef)
            {
                var inputsForDef = getInputsForInputDef(inputDef, inputs, nodeDef.InputDefs.ToArray(), inferredVariadicCounts);
                if (inputsForDef.Length == 1 && inputsForDef[0] is null)
                    return null;

                Debug.Assert(inputsForDef.All(x => x is not null),
                    $"Some inputs are null when they should not be for input definition '{inputDef.ParamName}'");

                return inputsForDef.Select(x => x.AssertNotNull().Rank).ToArray();
            }

            return null;
        }

        private int?[][] identifyInferredRank(NodeDefRankDef rankDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var identifiedCounts = new List<int?[]?>();
            foreach (var attributeDef in nodeDef.AttributeDefs)
                identifiedCounts.Add(identifyInferredRank(rankDef, attributeDef, nodeDef, attributes, inputs));

            foreach (var inputDef in nodeDef.InputDefs)
                identifiedCounts.Add(identifyInferredRank(rankDef, inputDef, nodeDef, attributes, inputs, inferredVariadicCounts));

            return identifiedCounts.NotNulls().ToArray();
        }

        private ImmutableDictionary<string, int?[]?> identifyInferredRanks(NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var inferredRanks = ImmutableDictionary<string, int?[]?>.Empty;

            foreach (var rankDef in nodeDef.RankDefs.Values)
            {
                Debug.Assert(!inferredRanks.ContainsKey(rankDef.RankDefName),
                    $"Duplicate rank definition name '{rankDef.RankDefName}' detected during node definition constraint processing");

                var hardCodedRank = rankDef.HardCodedValue;
                if (hardCodedRank is not null)
                {
                    inferredRanks = inferredRanks.SetItem(rankDef.RankDefName, [hardCodedRank]);
                    continue;
                }

                var identifiedRanks = identifyInferredRank(rankDef, nodeDef, attributes, inputs, inferredVariadicCounts);
                if (identifiedRanks.Length > 0)
                {
                    Debug.Assert(identifiedRanks.All(x => x.Length == identifiedRanks[0].Length),
                        $"Identified ranks have inconsistent lengths for '{rankDef.RankDefName}'");

                    var bestOfRanks = new List<int?>();
                    if (!identifiedRanks.All(x => x.SequenceEqual(identifiedRanks[0])))
                    {
                        for (int i = 0; i < identifiedRanks[0].Length; i++)
                        {
                            var rankToUse = identifiedRanks.Select(x => x[i])
                                                    .Distinct().Count() == 1 ? 
                                                        identifiedRanks[0][i] : 
                                                        null;

                            bestOfRanks.Add(rankToUse);
                        }

                        identifiedRanks = [[.. bestOfRanks]];
                    }
                }

                inferredRanks = inferredRanks.SetItem(rankDef.RankDefName, identifiedRanks.FirstOrDefault());
            }

            return inferredRanks;
        }

        #endregion

        #region Infer structures

        private DataStructure[]? identifyInferredStructure(NodeDefStructureDef rankDef, NodeDefAttributeDef attributeDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs)
        {
            if (attributeDef.Structure == rankDef)
            {
                if (attributeDef.Type == AttributeType.Enum)
                {
                    var structure = attributes.GetEnumVal<DataStructure>(attributeDef.AttributeName);
                    if (structure is null) return null;

                    return [structure.Value];
                }
                else if (attributeDef.Type == AttributeType.Enums)
                    return attributes.GetEnumsVal<DataStructure>(attributeDef.AttributeName);

                else if (attributeDef.Type == AttributeType.TypeProto)
                {
                    var typeProto = attributes.GetTypeProtoVal(attributeDef.AttributeName);
                    if (typeProto is null) return null;
                    return [typeProto.Value.structure];
                }

                Debug.Fail($"Unsupported attribute type {attributeDef.Type} for structure inference");
                return null;
            }

            return null;
        }

        private DataStructure[]? identifyInferredStructure(NodeDefStructureDef rankDef, NodeDefInputDef inputDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            if (inputDef.StructureDef == rankDef)
            {
                var inputsForDef = getInputsForInputDef(inputDef, inputs, nodeDef.InputDefs.ToArray(), inferredVariadicCounts);
                if (inputsForDef.Length == 1 && inputsForDef[0] is null)
                    return null;

                Debug.Assert(inputsForDef.All(x => x is not null),
                    $"Some inputs are null when they should not be for input definition '{inputDef.ParamName}'");

                return inputsForDef.Select(x => x.AssertNotNull().Structure()).ToArray();
            }

            return null;
        }

        private DataStructure[][] identifyInferredStructure(NodeDefStructureDef structureDef, NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var identifiedCounts = new List<DataStructure[]?>();
            foreach (var attributeDef in nodeDef.AttributeDefs)
                identifiedCounts.Add(identifyInferredStructure(structureDef, attributeDef, nodeDef, attributes, inputs));

            foreach (var inputDef in nodeDef.InputDefs)
                identifiedCounts.Add(identifyInferredStructure(structureDef, inputDef, nodeDef, attributes, inputs, inferredVariadicCounts));

            return identifiedCounts.NotNulls().ToArray();
        }

        private ImmutableDictionary<string, DataStructure[]?> identifyInferredStructures(NodeDefinition nodeDef, OnnxCSharpAttributes attributes, Variable?[] inputs, ImmutableDictionary<string, int?> inferredVariadicCounts)
        {
            var inferredStructures = ImmutableDictionary<string, DataStructure[]?>.Empty;

            foreach (var structureDef in nodeDef.StructureDefs.Values)
            {
                Debug.Assert(!inferredStructures.ContainsKey(structureDef.StructureDefName),
                    $"Duplicate structure definition name '{structureDef.StructureDefName}' detected during node definition building");

                var hardCodedStructure = structureDef.HardCodedValue;
                if (hardCodedStructure is not null)
                {
                    inferredStructures = inferredStructures.SetItem(structureDef.StructureDefName, [hardCodedStructure.Value]);
                    continue;
                }

                var identifiedStructure = identifyInferredStructure(structureDef, nodeDef, attributes, inputs, inferredVariadicCounts);
                Debug.Assert(identifiedStructure.Length == 0 || identifiedStructure.All(x => x.SequenceEqual(identifiedStructure[0])),
                    $"Identified structures do not match expected pattern for '{structureDef.StructureDefName}'");

                inferredStructures = inferredStructures.SetItem(structureDef.StructureDefName, identifiedStructure.FirstOrDefault());
            }

            return inferredStructures;
        }

        #endregion


        /// <summary>
        /// 
        /// </summary>
        /// <param name="originalKnownVariadicCounts"></param>
        /// <param name="outputNames">
        /// An output name given here as an empty string means that the optional output at that position is not requested.
        /// An output name provided as null means that it is requested and its name will be automatically generated.
        /// If the output names array itself is null, then it is treated as if output name items are null.
        /// </param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        protected OutputTensorInfo[] DeduceOutputDefinitons(ImmutableDictionary<string, int?>? originalKnownVariadicCounts, string?[]? outputNames)
        {
            var allInputsToUse = this.FullInputs.OrderBy(x => x.Key).SelectMany(x => x.Value).ToArray();
            originalKnownVariadicCounts ??= ImmutableDictionary<string, int?>.Empty;

            var knownVariadicCounts = identifyInferredVariadicCounts(this.NodeDef, this.CSharpAttributes, allInputsToUse, originalKnownVariadicCounts, outputNames?.Length);
            var knownTypes = identifyInferredTypes(this.NodeDef, this.CSharpAttributes, allInputsToUse, knownVariadicCounts);
            var knownRanks = identifyInferredRanks(this.NodeDef, this.CSharpAttributes, allInputsToUse, knownVariadicCounts);
            var knownStructures = identifyInferredStructures(this.NodeDef, this.CSharpAttributes, allInputsToUse, knownVariadicCounts);

            var numProcessedOutputs = 0;

            var namelessOutputsDefs = new List<OutputTensorInfo>();

            foreach (var outputDef in this.NodeDef.OutputDefs)
            {
                Debug.Assert(outputNames is null || numProcessedOutputs <= outputNames.Length,
                    $"Processed outputs ({numProcessedOutputs}) exceed provided output names count ({outputNames?.Length})");

                var variadicName = outputDef.VariadicCountDef?.CountDefName;
                var variadicCount = variadicName is null ? 1 : knownVariadicCounts[variadicName] ?? -1;
                Debug.Assert(variadicCount >= 0, $"Variadic count cannot be negative (got {variadicCount})");

                var dtypes = knownTypes[outputDef.TypeDef.TypeDefName];
                Debug.Assert(dtypes is not null,
                    $"Output type definition '{outputDef.TypeDef.TypeDefName}' not found in known types");

                var structures = knownStructures[outputDef.StructureDef.StructureDefName];
                Debug.Assert(structures is not null,
                    $"Output structure definition '{outputDef.StructureDef.StructureDefName}' not found in known structures");

                int?[] ranks;
                var rankDefs = outputDef.RankDefs;
                if (outputDef.IsBroadcasted)
                {
                    Debug.Assert(variadicCount == 1,
                        $"Broadcasted output requires variadic count of 1 (got {variadicCount})");

                    // Can broadcast from inputs using multiple input defs with respective multiple rank defs or
                    // from inputs using a single variadic input def with a respective single rank def.
                    // This line handles both situations.
                    var inferredRanks = knownRanks.Values.NotNulls().SelectMany(x => x).ToArray();
                    ranks = inferredRanks.Any(x => x is null) ? [null] : [(inferredRanks.Length == 0) ? null : inferredRanks.Max()];
                }
                else if (rankDefs is null || rankDefs.Length == 0)
                    ranks = [null];
                else
                {
                    Debug.Assert(rankDefs.Length == 1,
                        $"Multiple rank definitions not supported for non-broadcasted output (got {rankDefs.Length})");
                    ranks = knownRanks[rankDefs[0].RankDefName] ?? [null];
                }

                ranks = ranks.Select(x => x is null ? null : x + outputDef.RankRefAdjustment).ToArray();

                if (variadicCount == 1)
                {
                    ranks = [ranks[0]];
                    dtypes = [dtypes[0]];
                    structures = [structures[0]];
                }

                if (variadicCount != 1)
                {
                    if (ranks.Length == 1) ranks = Enumerable.Repeat(ranks[0], variadicCount).ToArray();
                    if (dtypes.Length == 1) dtypes = Enumerable.Repeat(dtypes[0], variadicCount).ToArray();
                    if (structures.Length == 1) structures = Enumerable.Repeat(structures[0], variadicCount).ToArray();
                }

                Debug.Assert(ranks.Length == variadicCount && dtypes.Length == variadicCount && structures.Length == variadicCount,
                    $"Array lengths must match variadic count: ranks.Length={ranks.Length}, dtypes.Length={dtypes.Length}, structures.Length={structures.Length}, variadicCount={variadicCount}");

                for (int i = 0; i < variadicCount; i++)
                {
                    Debug.Assert(outputNames is null || numProcessedOutputs < outputNames.Length,
                        $"Processed outputs ({numProcessedOutputs}) exceed or equal provided output names count ({outputNames?.Length})");

                    var name = outputNames is null ? null : outputNames[numProcessedOutputs];
                    namelessOutputsDefs.Add(new OutputTensorInfo
                    {
                        DType = dtypes[i].DType,
                        ModuleFn = dtypes[i].ModuleFn,
                        Structure = structures[i],
                        Rank = ranks[i],
                        Name = name
                    });

                    numProcessedOutputs++;
                }
            }

            return namelessOutputsDefs.ToArray();
        }

        public virtual ImmutableDictionary<string, OutputTensorInfo[]> DeduceMultiOutputDefinitions(ImmutableDictionary<string, int?>? knownVariadicCounts, string?[]? outputNames)
        {
            var retval = new Dictionary<string, OutputTensorInfo[]>();
            var graphAttributeNames = new string[] { "" };
            if (this.NodeDef.IsOpenNode)
                graphAttributeNames = this.CSharpAttributes.AttributeDefs
                                                    .Where(x => x.Type == AttributeType.Graph)
                                                    .Select(x => x.AttributeName).ToArray();

            foreach (var graphName in graphAttributeNames)
                retval[graphName] = DeduceOutputDefinitons(knownVariadicCounts, outputNames);

            return retval.ToImmutableDictionary();
        }
        public virtual Variable?[] DeduceInputsWithNulls(Variable?[] currentInputs)
        {
            var inputDefs = this.NodeDef.InputDefs;
            var minNumInputs = inputDefs.TakeWhile(x => x.VariadicCountDef is null).Count();
            var numNewTrailingNulls = minNumInputs - currentInputs.Length;

            if (numNewTrailingNulls > 0)
                return [..currentInputs, ..Enumerable.Repeat<Variable?>(null, numNewTrailingNulls)];

            return currentInputs;
        }

        public virtual ImmutableDictionary<string, Variable?[]> DeduceFullInputsWithNulls()
        {
            var retval = new Dictionary<string, Variable?[]>();
            var graphAttributeNames = new string[] { "" };
            if (this.NodeDef.IsCloseNode)
                graphAttributeNames = this.CSharpAttributes.AttributeDefs
                                                    .Where(x => x.Type == AttributeType.Graph)
                                                    .Select(x => x.AttributeName).ToArray();

            foreach (var graphName in graphAttributeNames)
                retval[graphName] = DeduceInputsWithNulls(this.FullInputs[graphName]);

            return retval.ToImmutableDictionary();
        }

        public ImmutableDictionary<string, int?> InferVariadicCounts(int? knownNumOutputs)
        {
            var allInputsToUse = this.FullInputs.OrderBy(x => x.Key).SelectMany(x => x.Value).ToArray();
            return identifyInferredVariadicCounts(this.NodeDef, this.CSharpAttributes, allInputsToUse, ImmutableDictionary<string, int?>.Empty, knownNumOutputs);
        }
    }

    internal class InputOutputNodeDataProcessor : InputNodeDataProcessor
    {
        public required ImmutableDictionary<string, string?[]> FullOutputNames { get; init; }
    }

    public static class NodeBuilder
    {
        public static Variable BuildNodeSingleOut(string opCode, Variable?[] inputs, (string attributeName, object? attributeValue)[] attrs, string? identifierTemplateString = null, string[]? outputNames = null, Function? targetFunction = null, Node? openNode = null)
             => BuildNodeMultiOut(opCode, inputs, attrs, identifierTemplateString, outputNames, targetFunction, openNode)[0].AssertNotNull();

        public static Variable[] BuildNodeMultiOut(string opCode, Variable?[] inputs, (string attributeName, object? attributeValue)[] attrs, string? identifierTemplateString = null, string?[]? outputNames = null, Function? targetFunction = null, Node? openNode = null)
            => BuildNodeFullOut(opCode, inputs, attrs, identifierTemplateString, outputNames, targetFunction, openNode)[""].NotNulls().ToArray();

        public static T CallCustomOperator<T>(string opCode, Variable?[] inputs, object?[] attributeNameAndValues) where T : IValue
        {
            List<(string attributeName, object? attributeValue)> attrs = new List<(string attributeName, object? attributeValue)>();
            for (int i = 0; i < attributeNameAndValues.Length; i += 2)
                attrs.Add(((string)attributeNameAndValues[i].NotNull(), attributeNameAndValues[i + 1]));

            return BuildNodeSingleOut(opCode, inputs, attrs.ToArray()).ToValue<T>();
        }

        public static (T1, T2) CallCustomOperator<T1, T2>(string opCode, Variable?[] inputs, object?[] attributeNameAndValues)
            where T1 : IValue
            where T2 : IValue
        {
            List<(string attributeName, object? attributeValue)> attrs = new List<(string attributeName, object? attributeValue)>();
            for (int i = 0; i < attributeNameAndValues.Length; i += 2)
                attrs.Add(((string)attributeNameAndValues[i].NotNull(), attributeNameAndValues[i + 1]));

            var retvals = BuildNodeMultiOut(opCode, inputs, attrs.ToArray());
            return (retvals[0].AssertNotNull().ToValue<T1>(), retvals[1].AssertNotNull().ToValue<T2>());
        }

        public static (T1, T2, T3) CallCustomOperator<T1, T2, T3>(string opCode, Variable?[] inputs, object?[] attributeNameAndValues)
            where T1 : IValue
            where T2 : IValue
            where T3 : IValue
        {
            List<(string attributeName, object? attributeValue)> attrs = new List<(string attributeName, object? attributeValue)>();
            for (int i = 0; i < attributeNameAndValues.Length; i += 2)
                attrs.Add(((string)attributeNameAndValues[i].NotNull(), attributeNameAndValues[i + 1]));

            var retvals = BuildNodeMultiOut(opCode, inputs, attrs.ToArray());
            return (retvals[0].AssertNotNull().ToValue<T1>(), retvals[1].AssertNotNull().ToValue<T2>(), retvals[2].AssertNotNull().ToValue<T3>());
        }

        public static (T1, T2, T3, T4) CallCustomOperator<T1, T2, T3, T4>(string opCode, Variable?[] inputs, object?[] attributeNameAndValues)
            where T1 : IValue
            where T2 : IValue
            where T3 : IValue
            where T4 : IValue
        {
            List<(string attributeName, object? attributeValue)> attrs = new List<(string attributeName, object? attributeValue)>();
            for (int i = 0; i < attributeNameAndValues.Length; i += 2)
                attrs.Add(((string)attributeNameAndValues[i].NotNull(), attributeNameAndValues[i + 1]));

            var retvals = BuildNodeMultiOut(opCode, inputs, attrs.ToArray());
            return (retvals[0].AssertNotNull().ToValue<T1>(), retvals[1].AssertNotNull().ToValue<T2>(), retvals[2].AssertNotNull().ToValue<T3>(), retvals[3].AssertNotNull().ToValue<T4>());
        }

        public static T[] CallCustomOperatorArrayOut<T>(string opCode, Variable?[] inputs, object?[] attributeNameAndValues)
            where T : IValue
        {
            List<(string attributeName, object? attributeValue)> attrs = new List<(string attributeName, object? attributeValue)>();
            for (int i = 0; i < attributeNameAndValues.Length; i += 2)
                attrs.Add(((string)attributeNameAndValues[i].NotNull(), attributeNameAndValues[i + 1]));

            var retvals = BuildNodeMultiOut(opCode, inputs, attrs.ToArray());
            return retvals.Select(v => v.ToValue<T>()).ToArray();
        }

        public static ImmutableDictionary<string, Variable?[]> BuildNodeFullOut(string opCode, Variable?[] inputs, (string attributeName, object? attributeValue)[] attrs, string? identifierTemplateString = null, string?[]? outputNames = null, Function? targetFunction = null, Node? openNode = null)
            => BuildNode(opCode, inputs, attrs, identifierTemplateString, outputNames, targetFunction, openNode).FullOutputs;

        public static Node BuildNode(string opCode, Variable?[] inputs, (string attributeName, object? attributeValue)[] attrs, string? identifierTemplateString = null, string?[]? outputNames = null, Function? targetFunction = null, Node? openNode = null)
        {
            var nodeDefResolver = Definitions.NodeDefinitions[opCode];
            var csharpAttributeVals = attrs.ToDictionary(x => x.attributeName, x => x.attributeValue);

            var fullInputs = new Dictionary<string, Variable?[]>();
            // Inputs are graph nodes (Variable?[]); the graph identifies tensors by node reference.
            fullInputs[""] = inputs;

            if (nodeDefResolver.IsCloseNode)
            {
                foreach (var graphDef in nodeDefResolver.AttributeDefs.Where(x => x.Type == AttributeType.Graph && csharpAttributeVals.ContainsKey(x.AttributeName)))
                {
                    Debug.Assert(csharpAttributeVals[graphDef.AttributeName] is not null,
                        $"Graph attribute '{graphDef.AttributeName}' is null when it should contain variables");

                    fullInputs[graphDef.AttributeName] = (Variable?[])csharpAttributeVals[graphDef.AttributeName]!;
                    csharpAttributeVals[graphDef.AttributeName] = new BestGraphAttribute { GraphAttributeName = graphDef.AttributeName };
                }
            }

            var attributes = OnnxProtoAttributes.FromCSharpVals(csharpAttributeVals, nodeDefResolver.AttributeDefs);

            return BuildNode(opCode, fullInputs, attributes, identifierTemplateString, outputNames, targetFunction, openNode);
        }


        public static Node BuildNode(string opCode, Dictionary<string, Variable?[]> fullInputs, OnnxProtoAttributes attributes, string? identifierTemplateString = null, string?[]? outputNames = null, Function? function = null, Node? openNode = null)
        {
            var nodeDefResolver = Definitions.NodeDefinitions[opCode];
            var nodeDef = nodeDefResolver.Resolve(attributes);

            var data = outputNames is null ?
                            new InputNodeDataProcessor
                            {
                                DefaultName = null,
                                NodeDef = nodeDef,
                                FullInputs = fullInputs.ToImmutableDictionary(),
                                ProtoAttributes = attributes,
                                StackTrace = new StackTrace(fNeedFileInfo: true).ToString()
                            } :
                            new InputOutputNodeDataProcessor
                            {
                                DefaultName = null,
                                NodeDef = nodeDef,
                                FullInputs = fullInputs.ToImmutableDictionary(),
                                ProtoAttributes = attributes,
                                StackTrace = new StackTrace(fNeedFileInfo: true).ToString(),
                                FullOutputNames = ImmutableDictionary.Create<string, string?[]>().Add("", outputNames)
                            };

            ImmutableDictionary<string, int?>? knownVariadicCounts = null;
            if (openNode is not null)
            {
                knownVariadicCounts = new InputNodeDataProcessor
                {
                    DefaultName = openNode.DefaultName,
                    NodeDef = openNode.NodeDef,
                    FullInputs = openNode.FullInputs,
                    ProtoAttributes = openNode.Attributes.ToProto(),
                    StackTrace = openNode.StackTrace
                }.InferVariadicCounts(openNode.FullOutputs.Values.Select(x => x.Length).Max());
            }

            var newNode = new Node(
                            nodeDef,
                            data.CSharpAttributes,
                            data.DeduceFullInputsWithNulls(),
                            data.DeduceMultiOutputDefinitions(knownVariadicCounts, outputNames),
                            data.StackTrace,
                            data.DefaultName,
                            identifierTemplateString,
                            function,
                            openNode);

            return newNode;
        }
    }
}
