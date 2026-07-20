
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using System.Diagnostics;
using System.Net.NetworkInformation;
using static Shorokoo.Core.InternalGlobals;
using static Shorokoo.Globals;
using System.Collections.Immutable;
using System.IO.Compression;
using System.Xml.Linq;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    internal class NodeDefinitionResolver
    {
        public string OpName { get; }
        public ImmutableList<NodeDefAttributeDef> AttributeDefs { get; }
        public ImmutableList<NodeDefinition> VariantDefinitions { get; }

        public bool IsOpenNode => VariantDefinitions[0].IsOpenNode;
        public bool IsCloseNode => VariantDefinitions[0].IsCloseNode;
        public bool IsGraphNode => VariantDefinitions[0].IsGraphNode;

        public bool HasDefaultVariant { get; } = true;

        public NodeDefinitionResolver(string definitionName, string opName, ImmutableList<NodeDefAttributeDef> attributeDefs, ImmutableList<NodeDefinition> variants)
        {
            this.OpName = opName;
            this.AttributeDefs = attributeDefs;
            this.VariantDefinitions = variants;
        }

        public NodeDefinition Resolve(OnnxProtoAttributes attributes)
        {
            var toUse = this.VariantDefinitions.FirstOrDefault(x =>
                                x.VariantConstraints.All(x => x.Match(attributes)));

            Debug.Assert(toUse is not null || this.HasDefaultVariant);
            return toUse ?? this.VariantDefinitions[0];
        }


        public NodeDefinitionResolver GetOpenNodeDefinition()
        {
            if (this.VariantDefinitions[0].IsCloseNode)
            {
                var openNodeOpCode = OpName.Substring(0, OpName.Length - "#CLOSE".Length) + "#OPEN";
                return Definitions.NodeDefinitions[openNodeOpCode];
            }

            throw new InvalidTensorOperationException(ErrorCodes.FW039, OpName, "node definition creation", 
                "Failed to get open node definition for close node operation");
        }

    }

    public class NodeDefinition
    {
        public string OpName { get; }
        public string FullNodeOpName => IsGraphNode ? OpName.Substring(0, OpName.IndexOf('#')) : OpName;
        public ImmutableDictionary<string, NodeDefTypeDef> TypeDefs { get; }
        public ImmutableDictionary<string, NodeDefStructureDef> StructureDefs { get; }
        public ImmutableDictionary<string, NodeDefVariadicCountDef> VariadicDefs { get; }
        public ImmutableDictionary<string, NodeDefRankDef> RankDefs { get; }
        public ImmutableList<NodeDefInputDef> InputDefs { get; }
        public ImmutableList<NodeDefOutputDef> OutputDefs { get; }
        public ImmutableList<NodeDefAttributeDef> AttributeDefs { get; }
        public ImmutableList<NodeDefVariantSpec> VariantConstraints { get; }
        public ImmutableDictionary<string, long[][]> InputTestShapes { get; } = ImmutableDictionary<string, long[][]>.Empty;
        public ImmutableList<long[]?[]> VariadicInputTestShapes { get; } = ImmutableList<long[]?[]>.Empty;
        public ImmutableDictionary<string, TensorData?[]> InputTestValues { get; } = ImmutableDictionary<string, TensorData?[]>.Empty;
        public ImmutableDictionary<string, object?[]> AttributeTestValues { get; } = ImmutableDictionary<string, object?[]>.Empty;
        public ImmutableDictionary<string, long[]> VariadicTestCounts { get; } = ImmutableDictionary<string, long[]>.Empty;
        public Func<ImmutableList<InternalComputationGraph>>? FnTestGraphMaker { get; } = null;
        public string? CodeTemplate { get; }
        public bool IsOpenNode { get; }
        public bool IsCloseNode { get; }
        public bool IsGraphNode => IsOpenNode || IsCloseNode;

        public NodeDefinition(string name, 
            Dictionary<string, NodeDefTypeDef> typeDefs, Dictionary<string, NodeDefStructureDef> structureDefs,
            Dictionary<string, NodeDefVariadicCountDef> variadicDefs, Dictionary<string, NodeDefRankDef> rankDefs,
            List<NodeDefInputDef> inputDefs, List<NodeDefOutputDef> outputDefs,
            List<NodeDefAttributeDef> attributeDefs, List<NodeDefVariantSpec> variantConstraints,
            bool? isGraphOpenNode,
            string? codeTemplate,
            Dictionary<string, long[][]>? inputTestShapes = null,
            List<long[]?[]>? variadicInputTestShapes = null,
            Dictionary<string, TensorData?[]>? inputTestValues = null,
            Dictionary<string, object?[]>? attributeTestValues = null,
            Dictionary<string, long[]>? variadicTestCounts = null,
            Func<ImmutableList<InternalComputationGraph>>? fnTestGraphMaker = null)
        {
            OpName = name;
            TypeDefs = typeDefs.ToImmutableDictionary();
            StructureDefs = structureDefs.ToImmutableDictionary();
            VariadicDefs = variadicDefs.ToImmutableDictionary();
            RankDefs = rankDefs.ToImmutableDictionary();
            InputDefs = inputDefs.ToImmutableList();
            OutputDefs = outputDefs.ToImmutableList();
            AttributeDefs = attributeDefs.ToImmutableList();
            VariantConstraints = variantConstraints.ToImmutableList();
            IsOpenNode = isGraphOpenNode == true;
            IsCloseNode = isGraphOpenNode == false;
            CodeTemplate = codeTemplate;
            InputTestShapes = inputTestShapes?.ToImmutableDictionary() ?? InputTestShapes;
            VariadicInputTestShapes = variadicInputTestShapes?.ToImmutableList() ?? VariadicInputTestShapes;
            InputTestValues = inputTestValues?.ToImmutableDictionary() ?? InputTestValues;
            AttributeTestValues = attributeTestValues?.ToImmutableDictionary() ?? AttributeTestValues;
            VariadicTestCounts = variadicTestCounts?.ToImmutableDictionary() ?? VariadicTestCounts;
            FnTestGraphMaker = fnTestGraphMaker;
        }
    }

    public enum DataStructure
    {
        Tensor,
        Optional,
        Sequence,
        TensorStruct
    }

    public enum InputType
    {
        // A hyperparam is used to specify an architecture variant but is not needed as input to the concrete model.
        Hyperparam,

        // A ready input is requied to convert a module to a concrete model but is also needed as input to the concrete model.
        ReadyInput,

        // Model inputs are not needed to convert a module to a concrete model but are needed as input to the concrete model.
        ModelInput,

        // Generic Types inputs provide type specification  
        GenericType,
    }

    public enum AttributeType
    {
        Long,
        Longs,
        Bool,
        Bools,
        DType,
        DTypes, // Not used in any official Onnx operation, but used for shorokou modules
        Float,
        Floats,
        String,
        Strings,
        Graph,
        // Graphs, // Not used in any official Onnx operation
        Tensor,
        // Tensors, // Not used in any official Onnx operation
        TypeProto,
        Enum,
        Enums,
    }

    public class NodeDefTypeDef
    {
        public string TypeDefName { get; }
        public ImmutableList<Type> Types { get; }
        public bool TracksModuleFn { get; }

        public NodeDefTypeDef(string name, Type type, bool tracksModuleFn)
        {
            this.TypeDefName = name;
            this.Types = [type];
            TracksModuleFn = tracksModuleFn;
        }

        public NodeDefTypeDef(params NodeDefTypeDef[] defs)
        {
            this.TypeDefName = defs[0].TypeDefName;
            this.Types = [.. defs.SelectMany(x => x.Types)];
        }

        public DType? GetHardcodedDType()
        {
            if (Types.Count != 1) return null;

            var dtype = OnnxUtils.GetDType(Types[0]);
            Debug.Assert(dtype != DType.Invalid);

            return dtype;
        }
    }

    public class NodeDefVariadicCountDef
    {
        public string CountDefName { get; }
        public int MinNumItems { get; }
        public int MaxNumItems { get; }

        public NodeDefVariadicCountDef(string name, int minNumIterms=0, int maxNumItems = int.MaxValue)
        {
            this.CountDefName = name;
            this.MinNumItems = minNumIterms;
            this.MaxNumItems = maxNumItems;
        }
    }

    public class NodeDefStructureDef
    {
        public string StructureDefName { get; }

        public ImmutableList<DataStructure> Structures { get; }
        public DataStructure? HardCodedValue => Structures.Count == 1 ? Structures[0] : null;

        public NodeDefStructureDef(string name, DataStructure structure)
        {
            this.StructureDefName = name;
            this.Structures = [structure];
        }

        public NodeDefStructureDef(string name, DataStructure[] structures)
        {
            this.StructureDefName = name;
            this.Structures = [.. structures];
        }

        public bool IsHardcodedSequenc()
            => Structures.Count == 1 && Structures[0] == DataStructure.Sequence;
    }

    public class NodeDefRankDef
    {
        public string RankDefName { get; }
        public int MinRank { get; }
        public int MaxRank { get; }

        public NodeDefRankDef(string name, int minRank = 0, int maxRank = int.MaxValue)
        {
            this.RankDefName = name;
            this.MinRank = minRank;
            this.MaxRank = maxRank;
        }

        public int? HardCodedValue => MinRank == MaxRank ? MinRank : null;
    }

    public class NodeDefInputDef
    {
        public required NodeDefTypeDef TypeDef { get; init; }
        public required NodeDefStructureDef StructureDef { get; init; }
        public NodeDefVariadicCountDef? VariadicCountDef { get; init; }
        public NodeDefRankDef? RankDef { get; init; }
        public bool IsOptional { get; init; }
        public required string ParamName { get; init; }
    }

    public class NodeDefOutputDef
    {
        public required NodeDefTypeDef TypeDef { get; init; }
        public required NodeDefStructureDef StructureDef { get; init; }
        public required NodeDefVariadicCountDef? VariadicCountDef { get; init; }
        public required NodeDefRankDef[]? RankDefs { get; init; }
        public int RankRefAdjustment { get; init; } = 0;
        public bool IsBroadcasted { get; init; } = false;
        public bool IsOptional { get; init; } = false;
        public required string ParamName { get; init; }
    }

    public class NodeDefAttributeDef
    {
        public required string AttributeName { get; init; }
        public AttributeType Type { get; init; }
        public NodeDefEnumDef? EnumDef { get; init; }
        public NodeDefTypeDef? TensorType { get; init; }
        public NodeDefStructureDef? Structure { get; init; }
        public NodeDefVariadicCountDef? VariadicCount { get; init; }
        public NodeDefRankDef? TensorRank { get; init; }
        public bool IsInternalAttribute => AttributeName.Contains('#');
        public bool IsGreaterThanZero { get; init; } = false;
        public bool IsGreaterOrEqualToZero {  get; init; } = false;
        public required object? DefaultValue { get; init; }
    }
    public class NodeDefEnumDef
    {
        private string[] names;
        public Type EnumType { get; private set; }

        public NodeDefEnumDef(Type enumType, params string[] names)
        {
            this.names = names;
            this.EnumType = enumType;
        }

        public int ToCSharpNum(string onnxName)
        {
            int index = Array.IndexOf(names, onnxName);
            if (index == -1) throw new InvalidTensorOperationException(ErrorCodes.FW010, onnxName, "Onnx name validation", 
                $"The provided Onnx name '{onnxName}' is not valid");
            return index;
        }

        public ImmutableList<string> OnnxNames => names.ToImmutableList();

        public string ToOnnxName(int csharpNum) => names[csharpNum];
        public string ToOnnxName(object csharpEnumVal) => ToOnnxName(Convert.ToInt32(csharpEnumVal));
        public string ToCSharpName(string onnxName) => ToCSharpName(ToCSharpNum(onnxName));
        public string ToCSharpName(int csharpNum) => Enum.GetName(EnumType, csharpNum)!;
        public string ToCSharpFullName(object csharpEnumVal) => EnumType.Name + "." + ToCSharpName(Convert.ToInt32(csharpEnumVal));
        public object ToCSharpVal(int attrVal) => Enum.ToObject(EnumType, attrVal);
        public object ToCSharpVal(string onnxName) => Enum.ToObject(EnumType, ToCSharpNum(onnxName));
        public T ToCSharpVal<T>(int csharpNum) where T : Enum => (T)ToCSharpVal(csharpNum);
        public T ToCSharpVal<T>(string onnxName) where T : Enum => (T)ToCSharpVal(onnxName);
    }



    public class NodeDefVariantSpec
    {
        public required string AttributeName { get; init; }
        public long? AttributeLongValue { get; init; }
        public string? AttributeStringValue { get; init; }

        public bool? IsSet { get; init; }

        public bool Match(OnnxProtoAttributes attributes)
        {
            if (IsSet is not null)
                return attributes.IsDefaultValue(AttributeName) != IsSet;
            if (AttributeLongValue is not null)
                return attributes.GetLongVal(AttributeName) == AttributeLongValue;
            if (AttributeStringValue is not null)
                return attributes.GetStringVal(AttributeName) == AttributeStringValue;

            return attributes.GetLongVal(AttributeName) is null;
        }
    }

    internal class NodeDefinitionMaker
    {
        private string? definitionName;
        private string? opName;
        private Dictionary<string, NodeDefTypeDef> typeDefs = new();
        private Dictionary<string, NodeDefStructureDef> structureDefs = new();
        private Dictionary<string, NodeDefVariadicCountDef> variadicDefs = new();
        private Dictionary<string, NodeDefRankDef> rankDefs = new();
        private List<NodeDefInputDef> inputDefs = new();
        private List<NodeDefOutputDef> outputDefs = new();
        private List<NodeDefAttributeDef> attributeDefs = new();
        private List<NodeDefVariantSpec> variantConstraints = new();
        private List<NodeDefinition> variantDefinitions = new();
        private Dictionary<string, long[][]> inputTestShapes = new();
        private List<long[]?[]> variadicInputTestShapes = new();
        private Dictionary<string, TensorData?[]> inputTestValues = new();
        private Dictionary<string, object?[]> attributeTestValues = new();
        private Dictionary<string, long[]> variadicTestCounts = new();
        private Func<ImmutableList<InternalComputationGraph>>? fnTestGraphMaker = null;
        private string? codeTemplate;
        private bool? isGraphOpen = null;

        public NodeDefinitionMaker Op(string opName)
        {
            this.opName = opName;
            return this;
        }

        public NodeDefinitionMaker Tensor<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType
        {
            VarType<T>(typeName, tracksModuleFn);
            Structure(typeName, DataStructure.Tensor);

            if (minVariadicCount != -1)
                return Variadic(typeName, minVariadicCount);

            return this;
        }

        public NodeDefinitionMaker Optional<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType
        {
            VarType<T>(typeName, tracksModuleFn);
            Structure(typeName, DataStructure.Optional);

            if (minVariadicCount != -1)
                return Variadic(typeName, minVariadicCount);

            return this;
        }


        public NodeDefinitionMaker Sequence<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType
        {
            VarType<T>(typeName, tracksModuleFn);
            Structure(typeName, DataStructure.Sequence);

            if (minVariadicCount != -1)
                return Variadic(typeName, minVariadicCount);

            return this;
        }

        public NodeDefinitionMaker Any<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType
        {
            VarType<T>(typeName, tracksModuleFn);
            Structure(typeName, [DataStructure.Tensor, DataStructure.Sequence, DataStructure.Optional, DataStructure.TensorStruct]);

            if (minVariadicCount != -1)
                return Variadic(typeName, minVariadicCount);

            return this;
        }

        public NodeDefinitionMaker TensorStruct<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IStruct
        {
            VarType<T>(typeName, tracksModuleFn);
            Structure(typeName, DataStructure.TensorStruct);

            if (minVariadicCount != -1)
                return Variadic(typeName, minVariadicCount);

            return this;
        }

        public NodeDefinitionMaker VarType<T>(string typeName, bool tracksModuleFn = false)
        {
            Debug.Assert(!typeDefs.ContainsKey(typeName));
            typeDefs[typeName] = new NodeDefTypeDef(typeName, typeof(T), tracksModuleFn);
            return this;
        }

        public NodeDefinitionMaker Structure(string typeName, DataStructure structure)
        {
            Debug.Assert(!structureDefs.ContainsKey(typeName));
            structureDefs[typeName] = new NodeDefStructureDef(typeName, structure);
            return this;
        }

        public NodeDefinitionMaker Structure(string typeName, DataStructure[] structures)
        {
            if(!structureDefs.ContainsKey(typeName))
                structureDefs[typeName] = new NodeDefStructureDef(typeName, structures);

            return this;
        }

        public NodeDefinitionMaker Variadic(string name, int minCount = 0, int maxCount = int.MaxValue)
        {
            if (!variadicDefs.ContainsKey(name))
                variadicDefs[name] = new NodeDefVariadicCountDef(name, minCount, maxCount);

            return this;
        }

        public NodeDefinitionMaker Rank(string name, int minRank = 0, int maxRank = int.MaxValue)
        {
            if(!rankDefs.ContainsKey(name))
                rankDefs[name] = new NodeDefRankDef(name, minRank, maxRank);

            return this;
        }

        public NodeDefinitionMaker SingleGraphOpen(string graphAttributeName)
        {
            this.isGraphOpen = true;
            return AttributeGraph(graphAttributeName);
        }

        public NodeDefinitionMaker SingleGraphClose(string graphAttributeName)
        {
            this.isGraphOpen = false;
            return AttributeGraph(graphAttributeName);
        }

        public NodeDefinitionMaker PairGraphOpen(string graphAttributeName1, string graphAttributeName2)
        {
            this.isGraphOpen = true;
            AttributeGraph(graphAttributeName1);
            return AttributeGraph(graphAttributeName2);
        }

        public NodeDefinitionMaker PairGraphClose(string graphAttributeName1, string graphAttributeName2)
        {
            this.isGraphOpen = false;
            AttributeGraph(graphAttributeName1);
            return AttributeGraph(graphAttributeName2);
        }

        private static bool TryGetAny<T>(Dictionary<string, T> dict, string[] keys, out T? retval) where T : class
        {
            foreach (var key in keys)
            {
                if (dict.TryGetValue(key.Trim('?'), out retval))
                    return true;
            }

            retval = null;
            return false;
        }

        private static bool TryGetMany<T>(Dictionary<string, T> dict, string[] keys, out T[] retval) where T : class
        {
            var list = new List<T>();
            T? item = null;
            foreach (var key in keys)
            {
                if (dict.TryGetValue(key.Trim('?'), out item))
                    list.Add(item);
            }

            retval = list.ToArray();
            return retval.Length > 0;
        }

        public NodeDefinitionMaker Input(string paramName, string defsRef, int? rank = null, DataStructure? structure = null, bool isOptional = false)
          => Input(paramName, [defsRef], rank, structure, isOptional);

        public NodeDefinitionMaker Input(string paramName, string defsRef, string rank, DataStructure? structure = null, bool isOptional = false)
          => Input(paramName, [defsRef], rank, structure, isOptional);

        public NodeDefinitionMaker Input(string paramName, string[] defsRef, string rank, DataStructure? structure = null, bool isOptional = false)
            => Rank(rank).Input(paramName, [.. defsRef, rank], (int?)null, structure, isOptional);

        private NodeDefRankDef MakeNewRank(int rank)
        {
            var newRankName = Guid.NewGuid().ToString();
            Rank(newRankName, rank, rank);
            return rankDefs[newRankName];
        }

        private NodeDefStructureDef MakeNewStructure(DataStructure structure)
        {
            var newStructureName = Guid.NewGuid().ToString();
            Structure(newStructureName, structure);
            return structureDefs[newStructureName];
        }

        public NodeDefinitionMaker Input(string paramName, string[] defsRefs, int? rank = null, DataStructure? structure = null, bool isOptional = false)
        {
            TryGetAny(this.typeDefs, defsRefs, out var typeDef);
            TryGetAny(this.variadicDefs, defsRefs, out var variadicDef);
            TryGetAny(this.structureDefs, defsRefs, out var structureDef);
            TryGetAny(this.rankDefs, defsRefs, out var rankDef);

            if (rank is not null)
                rankDef = MakeNewRank(rank.Value);

            if (structure is not null)
                structureDef = MakeNewStructure(structure.Value);

            this.inputDefs.Add(new NodeDefInputDef
            {
                ParamName = paramName,
                TypeDef = typeDef.AssertNotNull(),
                VariadicCountDef = variadicDef,
                StructureDef = structureDef.AssertNotNull(),
                RankDef = rankDef,
                IsOptional = isOptional || defsRefs.Any(x => x.EndsWith('?'))
            });

            return this;
        }

        public NodeDefinitionMaker WithBroadcastTestShapes()
        {
            Debug.Assert(this.inputDefs.Count == 2);

            var first = this.inputDefs[0].ParamName;
            var second = this.inputDefs[1].ParamName;

            InputTestShapes(first, [[], [2], [1, 3, 2, 3], [3,4]]);
            InputTestShapes(second, [[1, 2, 3], [], [2,1,1,3], [3,4]]);

            return this;
        }

        public NodeDefinitionMaker InputTestShapes(string paramName, long[][] shapes)
        {
            inputTestShapes[paramName] = shapes;
            return this;
        }

        public NodeDefinitionMaker VariadicInputTestShapes(List<long[][]> shapes)
        {
            // shapes[i][j] is the j'th input for input set i.
            // variadicShapes[j][i] is the j'th input for input set i.
            var maxNumInputs = shapes.Select(x => x.Length).Max();
            var numTestSets = shapes.Count;
            var variadicShapes = Enumerable.Repeat((object?)null, maxNumInputs).Select(x => Enumerable.Repeat((long[]?)null, numTestSets).ToArray()).ToList();

            for (int i = 0; i < numTestSets; i++)
                for (int j = 0; j < shapes[i].Length; j++)
                    variadicShapes[j][i] = shapes[i][j];


            this.variadicInputTestShapes = variadicShapes;
            return this;
        }

        public NodeDefinitionMaker InputTestValues(string paramName, TensorData?[] values)
        {
            this.inputTestValues[paramName] = values;
            return this;
        }

        public NodeDefinitionMaker AttributeTestValues(string attributeName, object?[] values)
        {
            this.attributeTestValues[attributeName] = values;
            return this;
        }

        public NodeDefinitionMaker VariadicTestCounts(string variadicDefName, long[] count)
        {
            Debug.Assert(variadicDefs.ContainsKey(variadicDefName));
            variadicTestCounts[variadicDefName] = count;
            return this;
        }
        public NodeDefinitionMaker TestGraph(Func<ImmutableList<InternalComputationGraph>> fnTestGraphMaker)
        {
            this.fnTestGraphMaker = fnTestGraphMaker;
            return this;
        }

        public NodeDefinitionMaker Output(string paramName, string defsRef, string rank, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false)
            => Output(paramName, [defsRef], rank, rankPlusOne, rankMinusOne, rankMinusTwo, rankBroadcast, structure, isOptional);
        
        public NodeDefinitionMaker Output(string paramName, string[] defsRefs, string rank, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false)
            => Output(paramName, [..defsRefs, rank], (int?)null, rankPlusOne, rankMinusOne, rankMinusTwo, rankBroadcast, structure, isOptional);


        public NodeDefinitionMaker Output(string paramName, string defsRef, int? rank = null, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false)
            => Output(paramName, [defsRef], rank, rankPlusOne, rankMinusOne, rankMinusTwo, rankBroadcast, structure, isOptional);
        public NodeDefinitionMaker Output(string paramName, string[] defsRefs, int? rank = null, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false)
        {
            defsRefs = ((string?[])[.. defsRefs, rankPlusOne, rankMinusOne, rankMinusTwo, rankBroadcast]).NotNulls().ToArray();

            TryGetAny(this.typeDefs, defsRefs, out var typeDef);
            TryGetAny(this.variadicDefs, defsRefs, out var variadicDef);
            TryGetAny(this.structureDefs, defsRefs, out var structureDef);
            TryGetMany(this.rankDefs, defsRefs, out var rankDefsToUse);

            if (rank is not null)
                rankDefsToUse = [MakeNewRank(rank.Value)];

            if (structure is not null)
                structureDef = MakeNewStructure(structure.Value);

            isOptional = isOptional || defsRefs.Any(x => x.EndsWith('?'));

            this.outputDefs.Add(new NodeDefOutputDef
            {
                ParamName = paramName,
                TypeDef = typeDef.AssertNotNull(),
                VariadicCountDef = variadicDef,
                StructureDef = structureDef.AssertNotNull(),
                RankDefs = rankDefsToUse,
                IsBroadcasted = rankBroadcast is not null,
                RankRefAdjustment = rank is not null ? 0 :
                                    rankPlusOne is not null ? 1 :
                                    rankMinusOne is not null ? -1 :
                                    rankMinusTwo is not null ? -2 :
                                    0,
                IsOptional = isOptional
            });

            return this;
        }

        public NodeDefinitionMaker AttributeLong(string attributeName, string? rank = null, object? defaultValue = null, string? variadicCount = null)
        {
            if (rank is not null) Rank(rank);
            var rankDef = rank is null ? null : this.rankDefs[rank];

            if (variadicCount is not null) Variadic(variadicCount);
            var variadicDef = variadicCount is null ? null : this.variadicDefs[variadicCount];

            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Long, TensorRank = rankDef, VariadicCount = variadicDef, DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeLongs(string attributeName, string? rank = null, object? defaultValue = null)
        {
            if (rank is not null) Rank(rank);
            var rankDef = rank is null ? null : this.rankDefs[rank];
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Longs, TensorRank = rankDef, DefaultValue = defaultValue });
            return this;
        }


        public NodeDefinitionMaker AttributeFloat(string attributeName, object? defaultValue = null, bool gtz = false, bool gtez = false)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Float, DefaultValue = defaultValue, IsGreaterOrEqualToZero = gtez, IsGreaterThanZero = gtz });
            return this;
        }

        public NodeDefinitionMaker AttributeFloats(string attributeName, object? defaultValue = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Floats, DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeString(string attributeName, object? defaultValue = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.String, DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeStrings(string attributeName, object? defaultValue = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Strings, DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeGraph(string attributeName)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Graph, DefaultValue = null });
            return this;
        }

        public NodeDefinitionMaker AttributeDType(string attributeName, string? tensorTypeName = null, DType? defaultValue = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.DType, TensorType = tensorTypeName is null ? null : this.typeDefs[tensorTypeName], DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeDTypes(string attributeName, string? tensorTypeName = null, DType[]? defaultValue = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.DTypes, TensorType = tensorTypeName is null ? null : this.typeDefs[tensorTypeName], DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeTypeProto(string attributeName, string? tensorTypeName= null, string? structureDefName = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.TypeProto, TensorType = tensorTypeName is null ? null : this.typeDefs[tensorTypeName], Structure = structureDefName is null ? null : this.structureDefs[structureDefName], DefaultValue=null });
            return this;
        }

        public NodeDefinitionMaker AttributeBool(string attributeName, object? defaultValue = null)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Bool, DefaultValue = defaultValue });
            return this;
        }

        public NodeDefinitionMaker AttributeBools(string attributeName)
        {
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Bools, DefaultValue = new long[0] });
            return this;
        }


        public NodeDefinitionMaker AttributeEnum<T>(string attributeName, string[] enumValues, string? defaultValue = null, string? structureDefName = null)
        {
            if (structureDefName is not null) Structure(structureDefName, [DataStructure.Tensor, DataStructure.Optional, DataStructure.Sequence]);
            var structureDef = structureDefName is null ? null : this.structureDefs[structureDefName];

            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Enum, EnumDef = new NodeDefEnumDef(typeof(T), enumValues), DefaultValue = defaultValue, Structure = structureDef });
            return this;
        }

        public NodeDefinitionMaker AttributeEnums<T>(string attributeName, string[] enumValues, string[]? defaultValue = null, string? structureDefName = null)
        {
            if (structureDefName is not null) Structure(structureDefName, [DataStructure.Tensor, DataStructure.Optional, DataStructure.Sequence]);
            var structureDef = structureDefName is null ? null : this.structureDefs[structureDefName];

            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Enums, EnumDef = new NodeDefEnumDef(typeof(T), enumValues), DefaultValue = defaultValue, Structure = structureDefName is null ? null : this.structureDefs[structureDefName] });
            return this;
        }

        public NodeDefinitionMaker AttributeTensor(string attributeName, string tensorTypeName, string rank)
        {
            Rank(rank);
            var rankDef = this.rankDefs[rank];
            attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Tensor, TensorType = this.typeDefs[tensorTypeName], TensorRank = rankDef, DefaultValue = null });
            return this;
        }

        // public NodeDefinitionMaker AttributeTensors(string attributeName, string tensorTypeName)
        // {
        //     attributeDefs.Add(new NodeDefAttributeDef { AttributeName = attributeName, Type = AttributeType.Tensors, TensorType = this.typeDefs[tensorTypeName] });
        //     return this;
        // }

        private void createVariant()
        {

            Debug.Assert(this.opName is not null);

            var variantDef = new NodeDefinition(
                this.opName,
                typeDefs,
                structureDefs,
                variadicDefs,
                rankDefs,
                inputDefs,
                outputDefs,
                attributeDefs,
                variantConstraints,
                isGraphOpen,
                codeTemplate!,
                inputTestShapes,
                variadicInputTestShapes,
                inputTestValues,
                attributeTestValues,
                variadicTestCounts,
                fnTestGraphMaker);

            this.variantDefinitions.Add(variantDef);
            this.variantConstraints = new List<NodeDefVariantSpec>();
            this.inputDefs = new List<NodeDefInputDef>();
            this.outputDefs = new List<NodeDefOutputDef>();
            this.inputTestShapes = new Dictionary<string, long[][]>();
            this.variadicInputTestShapes = new List<long[]?[]>();
            this.inputTestValues = new Dictionary<string, TensorData?[]>();
            this.attributeTestValues = new Dictionary<string, object?[]>();
            this.variadicTestCounts = new Dictionary<string, long[]>();
            this.fnTestGraphMaker = null;
        }

        public NodeDefinitionMaker Constraint(string attributeName, string attributeValue)
        {
            if (this.inputDefs.Count != 0 || this.outputDefs.Count != 0)
                createVariant();

            this.variantConstraints.Add(new NodeDefVariantSpec { AttributeName = attributeName, AttributeStringValue = attributeValue });
            return this;
        }

        public NodeDefinitionMaker Constraint(string attributeName, int? attributeValue)
        {
            if (this.inputDefs.Count != 0 || this.outputDefs.Count != 0)
                createVariant();

            this.variantConstraints.Add(new NodeDefVariantSpec { AttributeName = attributeName, AttributeLongValue = attributeValue });
            return this;
        }

        public NodeDefinitionMaker ConstraintIsSet(string attributeName, bool isSet)
        {
            if (this.inputDefs.Count != 0 || this.outputDefs.Count != 0)
                createVariant();

            this.variantConstraints.Add(new NodeDefVariantSpec { AttributeName = attributeName, IsSet = isSet });
            return this;
        }

        public NodeDefinitionMaker Code(string codeTemplate, bool inline = false)
        {
            this.codeTemplate = codeTemplate;
            return this;
        }

        public NodeDefinitionMaker AlternateDefinitionName(string alternateName)
        {
            this.definitionName = alternateName;
            return this;
        }

        public NodeDefinitionResolver Finish()
        {
            createVariant();
            Debug.Assert(opName is not null);
            return new NodeDefinitionResolver(definitionName ?? opName, opName, attributeDefs.ToImmutableList(), variantDefinitions.ToImmutableList());
        }
    }
}
