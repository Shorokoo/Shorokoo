using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using static Shorokoo.Globals;
using static Shorokoo.Core.InternalGlobals;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using Shorokoo.Core.Factory;

namespace Shorokoo.Core.Factory.IR
{
    internal class TempNode
    {
        public bool IsGraphOpen { get; }
        public bool IsGraphClose { get; }
        public NodeProto Node { get; }

        public TempNode(NodeProto node, bool isGraphOpen = false, bool isGraphClose = false)
        {
            this.Node = node;
            this.IsGraphOpen = isGraphOpen;
            this.IsGraphClose = isGraphClose;
        }
    }

    /// <summary>
    /// ONNX IR loader. The public entry point <see cref="BuildFastComputationGraph"/>
    /// runs two passes:
    /// <list type="number">
    ///   <item>Each <see cref="FunctionProto"/> is reified as a <see cref="Function"/>
    ///         via <see cref="internalBuildFunctions"/> →
    ///         <see cref="internalInitFunction"/>, which walks the proto's nodes through
    ///         <see cref="EnumerateNodesInProtoOrder"/> + <see cref="CreateFastNodes"/>
    ///         to materialize FastNodes directly.</item>
    ///   <item>The top-level <see cref="GraphProto"/> goes through the same FastCG
    ///         pipeline via <see cref="internalBuildFastComputationGraph"/>.</item>
    /// </list>
    /// A post-pass <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastUnPrepFromOnnx"/>
    /// undoes the close-input identity wrapping that the saver inserted.
    /// </summary>
    internal class OnnxModelReader
    {
        private ModelProto model;
        private OpSetVersion OpSetVersion;

        public OnnxModelReader(ModelProto model)
        {
            this.model = model;
            this.OpSetVersion = (OpSetVersion)model.OpsetImports[0].Version;
        }

        public FastComputationGraph BuildFastComputationGraph()
        {
            // Lower control-flow ops Shorokoo doesn't execute natively (Scan → Loop)
            // and reject unsupported ones (SequenceMap) with an actionable error,
            // before any FastNode is materialized.
            OnnxControlFlowLowering.Process(this.model);

            var tensorStructDefs = ParseTensorStructMetadata(this.model);

            var (functions, onnxNameToFunction) = internalBuildFunctions(this.model.Functions, this.OpSetVersion, tensorStructDefs);
            var fastGraph = internalBuildFastComputationGraph(this.model.Graph, functions, onnxNameToFunction, this.OpSetVersion, tensorStructDefs);
            Shorokoo.Core.Nodes.Processors.Fast.FastUnPrepFromOnnx.Process(fastGraph);
            return fastGraph;
        }

        /// <summary>
        /// Parses TensorStructDef metadata from ONNX model metadata.
        /// Metadata keys follow the format: "shrk_tensorstruct_{ProtoTypeNum}"
        /// Values are JSON representations of TensorStructDef.
        /// </summary>
        private static ImmutableDictionary<int, TensorStructDef> ParseTensorStructMetadata(ModelProto model)
        {
            var result = new Dictionary<int, TensorStructDef>();

            foreach (var prop in model.MetadataProps)
            {
                if (prop.Key.StartsWith(ShrkMetaTensorStructDefPrefix))
                {
                    var protoTypeNumStr = prop.Key.Substring(ShrkMetaTensorStructDefPrefix.Length);
                    if (int.TryParse(protoTypeNumStr, out var protoTypeNum))
                    {
                        try
                        {
                            var def = TensorStructDef.FromJson(prop.Value);
                            result[protoTypeNum] = def;
                            // Eagerly register the struct DType at its ORIGINAL protoTypeNum
                            // so DType attributes carrying that protoTypeNum (e.g. ShrkAttrDtype
                            // on a TENSOR_STRUCT_CREATE node) resolve to the same DType through
                            // the (int)→DType implicit conversion. Using the plain
                            // GetOrCreateForTensorStruct would allocate a fresh iType in a clean
                            // process — that new iType won't match the saved attribute's value
                            // and op_Implicit trips Debug.Fail.
                            DType.GetOrCreateForTensorStructAtProtoTypeNum(def, protoTypeNum);
                        }
                        catch (Exception ex)
                        {
                            // Skip invalid TensorStructDef metadata - continue loading the model
                            // Include exception details and truncated JSON content for debugging
                            var truncatedJson = prop.Value.Length > 100 ? prop.Value.Substring(0, 100) + "..." : prop.Value;
                            Debug.Assert(false, $"Failed to parse TensorStructDef metadata for protoTypeNum {protoTypeNum}. Error: {ex.Message}. JSON: {truncatedJson}");
                        }
                    }
                }
            }

            return result.ToImmutableDictionary();
        }

        private static IEnumerable<FunctionProto> SortByReferenceHierarchy(List<FunctionProto> functions)
        {
            var toVisitFunctions = functions.ToHashSet();
            HashSet<string> visitedFunctionNames = new();
            for (int i = 0; i < functions.Count; i++)
            {
                foreach (var function in toVisitFunctions.ToList())
                {
                    if (visitedFunctionNames.Contains(function.Name))
                        continue;

                    var hasUnvisitedReference = function.Nodes
                        .Any(x => (x.Domain == "Functions" && !visitedFunctionNames.Contains(x.OpType)) ||
                        x.Attributes.Any(y =>
                                y.Name == ShrkAttrFunctionName &&
                                !visitedFunctionNames.Contains(Encoding.UTF8.GetString(y.S))));

                    var hasUnvisitedSignatureReference = function.ValueInfoes
                                .SelectMany(x => x.MetadataProps)
                                .Where(x => x.Key == Function.IRFunctionSignatureParamName)
                                .Any(x => !visitedFunctionNames.Contains(x.Value));

                    if (!hasUnvisitedReference && !hasUnvisitedSignatureReference)
                    {
                        visitedFunctionNames.Add(function.Name);
                        toVisitFunctions.Remove(function);

                        yield return function;
                    }
                }

                if (toVisitFunctions.Count == 0)
                    break;
            }

            if (toVisitFunctions.Count > 0)
                throw new OnnxNodeException(ErrorCodes.FW029, "functions", "circular reference detection",
                    "Circular reference detected in function dependencies");
        }

        private static (Function[] functions, Dictionary<string, Function> onnxNameToFunction) internalBuildFunctions(
            List<FunctionProto> functionProtos,
            OpSetVersion opset,
            ImmutableDictionary<int, TensorStructDef>? tensorStructDefs = null)
        {
            var orderedFunctionProtos = SortByReferenceHierarchy(functionProtos).ToList();
            Dictionary<string, Function> functions = new();
            List<Function>? retvals = new();

            foreach (var functionProto in orderedFunctionProtos)
            {
                var result = internalInitFunction(functionProto, functions.ToImmutableDictionary(), opset, tensorStructDefs);

                Debug.Assert(result is not null);
                functions[functionProto.Name] = result;

                retvals.Add(result);
            }

            return (retvals.ToArray(), functions);
        }

        private static Function? internalInitFunction(
            FunctionProto functionProto,
            ImmutableDictionary<string, Function> functionsMap,
            OpSetVersion opset,
            ImmutableDictionary<int, TensorStructDef>? tensorStructDefs = null)
        {
            try
            {
                var functionTypeName = functionProto.MetadataProps.FirstOrDefault(x => x.Key == Function.IRFunctionTypeParamName)?.Value;
                var functionType = Function.FromComponentTypeName(functionTypeName);

                // Restore the original module name from metadata if available
                // This preserves names like "ResNet18Debug" even when the ONNX file uses "fn_0"
                var friendlyName = functionProto.MetadataProps.FirstOrDefault(x => x.Key == Function.IRFunctionFriendlyName)?.Value;
                // Strip any built-in-op collision-avoidance prefix (OnnxFunctionName) so the
                // reconstructed DefaultName equals the original name; functionsMap stays keyed
                // by the on-disk (encoded) name so op_type / shrk_function_name lookups still match.
                var defaultName = OnnxFunctionName.Decode(functionProto.Name);

                // Walk the FunctionProto's nodes in proto order (descending into each
                // sub-graph between an emitted OPEN/CLOSE pair). Proto order is the
                // canonical execution order — a topological-sort-based reconstruction
                // would hoist scope-invariant body nodes (e.g. a shape Concat whose only
                // inputs are function-input hyperparams) above their enclosing LOOP_OPEN,
                // breaking downstream OPEN/CLOSE-band passes like
                // FastFoldConstantIterationLoops.
                var fastGraph = new FastComputationGraph();
                var tensorKeys = new Dictionary<string, FastTensorKey>();

                var infosByName = functionProto.ValueInfoes.ToDictionary(x => x.Name, x => x);
                var inputInfos = functionProto.Inputs.Select(name => infosByName[name]).ToList();
                foreach (var (info, key, fastNode) in CreateFastInputTensors(inputInfos, functionsMap, tensorStructDefs))
                {
                    fastGraph.Nodes.Add(fastNode);
                    tensorKeys[info.Name] = key;
                    fastGraph.Inputs.Add(key);
                    fastGraph.InputUniqueNames.Add(info.Name);
                }

                CreateFastNodes(fastGraph, EnumerateNodesInProtoOrder(functionProto.Nodes), tensorKeys, functionsMap, opset);

                foreach (var outputName in functionProto.Outputs)
                {
                    fastGraph.Outputs.Add(tensorKeys[outputName]);
                    fastGraph.OutputUniqueNames.Add(outputName);
                }

                Function onnxFunction = new Function(fastGraph, functionType, defaultName: defaultName, friendlyName: friendlyName);
                return onnxFunction;
            }
            catch (Function.FunctionNotFoundException)
            {
                return null;
            }
        }

        /// <summary>
        /// Parses the IsTrainable metadata value from ONNX. Shorokoo's own exports stamp
        /// it on every initializer; foreign ONNX models carry no Shorokoo metadata, and
        /// their initializers import as non-trainable constants (the standard ONNX
        /// meaning). An explicitly present but unparseable value still throws.
        /// </summary>
        private static bool ParseIsTrainableMetadata(string? value)
        {
            if (string.IsNullOrEmpty(value))
                return false;
            if (bool.TryParse(value, out var result))
                return result;
            throw new InvalidOperationException($"Invalid value '{value}' for '{ShrkMetaIsTrainable}' metadata. Expected 'true' or 'false'.");
        }

        private static bool isEtherealIdentityNode(NodeProto nodeProto)
        {
            return nodeProto.OpType == OpCodes.IDENTITY &&
                   nodeProto.Inputs.Count == 1 &&
                   nodeProto.Outputs.Count == 1 &&
                   nodeProto.MetadataProps.Where(x => x.Key == ShrkMetaIdentityNodeEthereal && x.Value == "true").Any();
        }

        public static object? getAttributeValue(AttributeProto attribute, NodeDefAttributeDef? attributeDef = null)
        {
            switch (attribute.Type)
            {
                case AttributeProto.AttributeType.Int:
                    var intVal = attribute.I;

                    // Check if this is a DType attribute
                    if (attributeDef?.Type == AttributeType.DType)
                    {
                        var dtype = (DType)(int)intVal;

                        // Check for generic parameter name in RefAttrName
                        if (!string.IsNullOrEmpty(attribute.RefAttrName) &&
                            attribute.RefAttrName.StartsWith("GenericParam:"))
                        {
                            var paramName = attribute.RefAttrName.Substring("GenericParam:".Length);
                            return DType.CreateWithGenericParam(dtype, paramName);
                        }

                        return dtype;
                    }

                    return intVal;
                case AttributeProto.AttributeType.Ints:
                    // Check if this is a DType array attribute
                    if (attributeDef?.Type == AttributeType.DTypes)
                    {
                        var dtypes = attribute.Ints.Select(i => (DType)(int)i).ToArray();

                        // Check for generic parameter names in RefAttrName
                        if (!string.IsNullOrEmpty(attribute.RefAttrName) &&
                            attribute.RefAttrName.StartsWith("GenericParams:"))
                        {
                            var paramNamesStr = attribute.RefAttrName.Substring("GenericParams:".Length);
                            var paramNames = paramNamesStr.Split(',');

                            // Reconstruct DTypes with param names
                            for (int i = 0; i < dtypes.Length && i < paramNames.Length; i++)
                            {
                                if (!string.IsNullOrEmpty(paramNames[i]))
                                {
                                    dtypes[i] = DType.CreateWithGenericParam(dtypes[i], paramNames[i]);
                                }
                            }
                        }

                        return dtypes;
                    }

                    return attribute.Ints;
                case AttributeProto.AttributeType.Float:
                    return attribute.F;
                case AttributeProto.AttributeType.Floats:
                    return attribute.Floats;
                case AttributeProto.AttributeType.String:
                    return Encoding.UTF8.GetString(attribute.S);
                case AttributeProto.AttributeType.Strings:
                    return attribute.Strings.Select(Encoding.UTF8.GetString).ToArray();
                case AttributeProto.AttributeType.Tensor:
                    return CreateTensorData(attribute.T);
                case AttributeProto.AttributeType.Tensors:
                    return attribute.Tensors.Select(t => CreateTensorData(t)).ToArray();
                case AttributeProto.AttributeType.Graph:
                    return
                        new BestGraphAttribute
                        {
                            DefaultGraphName = attribute.G.Name,
                            GraphAttributeName = attribute.Name,
                            DefautGraphInputNames = attribute.G.Inputs.Select(x => x.Name).ToArray()
                        };
                case AttributeProto.AttributeType.Graphs:
                    throw new UnsupportedDTypeException(ErrorCodes.FW031, "Graphs", "attribute conversion",
                        "Sequence element type conversion not implemented for Graphs attribute type");
                case AttributeProto.AttributeType.TypeProto:
                    return attribute.Tp;

                default:
                    throw new UnsupportedDTypeException(ErrorCodes.FW032, attribute.Type.ToString(), "attribute conversion",
                        $"Attribute type '{attribute.Type}' is not supported for conversion");
            }
        }

        private static TensorData CreateTensorData(TensorProto tensorProto)
        {
            byte[] rawDataBytes;

            var type = (DType)tensorProto.data_type;
            Shape shape;
            if (tensorProto.Dims == null || tensorProto.Dims.Length == 0)
                shape = (long[])[];
            else
                shape = tensorProto.Dims;

            if (tensorProto.RawData != null && tensorProto.RawData.Length > 0)
                rawDataBytes = tensorProto.RawData;
            else if (tensorProto.Int32Datas != null && tensorProto.Int32Datas.Length > 0)
                rawDataBytes = ConvertInt32PackedData(type, tensorProto.Int32Datas);
            else if (tensorProto.FloatDatas != null && tensorProto.FloatDatas.Length > 0)
                rawDataBytes = OnnxUtils.ConvertToByteArray(tensorProto.FloatDatas);
            else if (tensorProto.Int64Datas != null && tensorProto.Int64Datas.Length > 0)
                rawDataBytes = OnnxUtils.ConvertToByteArray(tensorProto.Int64Datas);
            else if (tensorProto.Uint64Datas != null && tensorProto.Uint64Datas.Length > 0)
                rawDataBytes = OnnxUtils.ConvertToByteArray(tensorProto.Uint64Datas);
            else if (shape == 1 || shape == 0) // If there's no data and we infer that the shape is a scalar, we'll assume 0.
                rawDataBytes = OnnxUtils.GetRawBytesZero(type);
            else
                throw new UnsupportedDTypeException(ErrorCodes.FW033, type.ToString(), "TensorProto conversion",
                    $"Data type '{type}' is not supported in TensorProto conversion");

            var tensorData = TensorData.CreateFromRawBytes(shape, type, rawDataBytes);
            return tensorData;
        }

        /// <summary>
        /// Per the ONNX spec, TensorProto.int32_data packs not just INT32 but every
        /// narrower element type — FLOAT16/BFLOAT16 (uint16 bit patterns), (U)INT8,
        /// (U)INT16 and BOOL — one element per int32 entry. Narrow each entry back to
        /// the element's true storage width before handing the bytes to
        /// <see cref="TensorData.CreateFromRawBytes"/>.
        /// </summary>
        internal static byte[] ConvertInt32PackedData(DType type, int[] int32Data)
        {
            if (type == DType.Float16 || type == DType.BFloat16 || type == DType.UInt16)
                return OnnxUtils.ConvertToByteArray(int32Data.Select(v => (ushort)v).ToArray());
            if (type == DType.Int16)
                return OnnxUtils.ConvertToByteArray(int32Data.Select(v => (short)v).ToArray());
            if (type == DType.Int8)
                return OnnxUtils.ConvertToByteArray(int32Data.Select(v => (sbyte)v).ToArray());
            if (type == DType.UInt8 || type == DType.Bool)
                return int32Data.Select(v => (byte)v).ToArray();
            return OnnxUtils.ConvertToByteArray(int32Data);
        }

        private static TensorDim[]? GetTensorDims(TensorShapeProto? shapeProto)
        {
            return shapeProto is null ? null :
                        shapeProto.Dims.Select(x =>
                        (x.DimParam == null || x.DimParam == "") ?
                            new TensorDim(x.DimValue) :
                            new TensorDim(x.DimParam)).ToArray();
        }


        /// <summary>
        /// Reconstructs a TensorStruct DType from the ProtoTypeNum and parsed TensorStructDef metadata.
        /// If the TensorStructDef was previously parsed from model metadata, uses that definition.
        /// Otherwise, creates a new DType with just the ProtoTypeNum (definition may be incomplete).
        /// </summary>
        private static DType ReconstructTensorStructDType(int protoTypeNum, ImmutableDictionary<int, TensorStructDef>? tensorStructDefs)
        {
            if (tensorStructDefs != null && tensorStructDefs.TryGetValue(protoTypeNum, out var def))
            {
                // We have the full TensorStructDef - use it to get/create the DType
                return DType.GetOrCreateForTensorStruct(def);
            }

            // No metadata found - this could happen if:
            // 1. Model was created by an older version that didn't serialize metadata
            // 2. Metadata parsing failed
            // 3. External model without Shorokoo-specific metadata
            // Log a warning (Debug.Assert) and create a minimal placeholder
            Debug.Assert(false, $"TensorStructDef metadata not found for protoTypeNum {protoTypeNum}. " +
                "The TensorStruct will have an empty field definition which may cause issues if field access is attempted.");

            var placeholderDef = new TensorStructDef(
                Array.Empty<TensorStructFieldDef>(),
                $"UnknownTensorStruct_{protoTypeNum}");
            return DType.GetOrCreateForTensorStruct(placeholderDef);
        }

        // ---- FastCG-native loading path ----
        //
        // The main graph is materialized directly as FastNodes (vs the legacy CG-then-wrap
        // path used for FunctionProtos above). Walks the proto's GraphProto.Nodes in declared
        // order — sub-graph attributes are recursed into positionally between the parent op's
        // OPEN and CLOSE TempNodes — and emits one FastNode per visited TempNode.

        /// <summary>
        /// Sets up the tensor-name → <see cref="FastTensorKey"/> map by walking model
        /// inputs and initializers, then drives <see cref="CreateFastNodes"/> over the
        /// proto nodes in declared order, and finally wires up graph outputs. Functions
        /// are pre-built by the caller via <see cref="internalBuildFunctions"/> and shared
        /// by reference (each <see cref="Function"/> internally maintains both CG and
        /// FastCG views).
        /// </summary>
        private static FastComputationGraph internalBuildFastComputationGraph(
            GraphProto graphProto,
            Function[] functions,
            Dictionary<string, Function> onnxNameToFunction,
            OpSetVersion opset,
            ImmutableDictionary<int, TensorStructDef>? tensorStructDefs = null)
        {
            var fastGraph = new FastComputationGraph();
            var functionsMap = onnxNameToFunction.ToImmutableDictionary();

            var tensorKeys = new Dictionary<string, FastTensorKey>();

            // Model inputs first. CG order: inputs precede initializers in the build, but
            // the CG ComputationGraph constructor only marks IsModelInput nodes as inputs
            // — the FastCG equivalent uses the explicit Inputs list, so we record them here
            // and skip initializer producers from that list.
            foreach (var (info, key, fastNode) in CreateFastInputTensors(graphProto.Inputs, functionsMap, tensorStructDefs))
            {
                fastGraph.Nodes.Add(fastNode);
                tensorKeys[info.Name] = key;
                fastGraph.Inputs.Add(key);
                fastGraph.InputUniqueNames.Add(info.Name);
            }

            // Initializers (MODEL_PARAM_DATA producers). These are reachable from the
            // graph via tensorKeys lookups but are not graph inputs.
            foreach (var (proto, key, fastNode) in CreateFastInitializers(graphProto.Initializers))
            {
                fastGraph.Nodes.Add(fastNode);
                tensorKeys[proto.Name] = key;
            }

            // Walk the proto's nodes in declared order — the canonical order. Sub-graph
            // attributes are recursed into positionally between the parent op's OPEN and
            // CLOSE TempNodes. Materialize one FastNode per visited TempNode.
            CreateFastNodes(fastGraph, EnumerateNodesInProtoOrder(graphProto.Nodes), tensorKeys, functionsMap, opset);

            // Outputs (in proto declaration order).
            foreach (var output in graphProto.Outputs)
            {
                fastGraph.Outputs.Add(tensorKeys[output.Name]);
                fastGraph.OutputUniqueNames.Add(output.Name);
            }

            return fastGraph;
        }

        /// <summary>
        /// Builds one input <see cref="FastNode"/> (MODEL_TENSOR_INPUT,
        /// MODEL_OPTIONAL_INPUT, MODEL_SEQUENCE_INPUT, GENERIC_TYPE_INPUT, or
        /// MODEL_TENSORSTRUCT_INPUT) per <see cref="ValueInfoProto"/>, attaching
        /// dtype / rank / input-type / target-function metadata and returning the
        /// freshly-allocated <see cref="FastTensorKey"/> for each.
        /// </summary>
        private static List<(ValueInfoProto info, FastTensorKey key, FastNode node)> CreateFastInputTensors(
            List<ValueInfoProto> inputs,
            ImmutableDictionary<string, Function> knownFunctions,
            ImmutableDictionary<int, TensorStructDef>? tensorStructDefs)
        {
            var results = new List<(ValueInfoProto, FastTensorKey, FastNode)>();

            foreach (var inputProto in inputs)
            {
                // Type is a discriminated union (TensorType / OptionalType / SequenceType …);
                // pull the inner tensor info from the appropriate branch and remember the
                // outer DataStructure so non-TensorType inputs route to the matching
                // MODEL_*_INPUT op below.
                DataStructure structure;
                TypeProto.Tensor tensorType;
                if (inputProto.Type.ShouldSerializeOptionalType())
                {
                    structure = DataStructure.Optional;
                    tensorType = inputProto.Type.OptionalType.ElemType.TensorType;
                }
                else if (inputProto.Type.ShouldSerializeSequenceType())
                {
                    structure = DataStructure.Sequence;
                    tensorType = inputProto.Type.SequenceType.ElemType.TensorType;
                }
                else
                {
                    structure = DataStructure.Tensor;
                    tensorType = inputProto.Type.TensorType;
                }
                var elemType = tensorType.ElemType;
                var isTensorStruct = structure == DataStructure.Tensor
                                  && (elemType >= 2000 && elemType <= 2999);

                var nodeKey = FastNodeKey.New();
                var key = new FastTensorKey(nodeKey, 0);

                FastNode fastNode;
                if (isTensorStruct)
                {
                    var structDType = ReconstructTensorStructDType(elemType, tensorStructDefs);

                    var inputTypeName = inputProto.MetadataProps
                        .Where(x => x.Key == Function.IRInputTypeName)
                        .SingleOrDefault()?.Value;
                    var inputType = Function.FromInputTypeName(inputTypeName);

                    var targetFunctionName = inputProto.MetadataProps
                        .Where(x => x.Key == Function.IRFunctionSignatureParamName)
                        .SingleOrDefault()?.Value;
                    var targetFunction = targetFunctionName is null ? null : knownFunctions[targetFunctionName];

                    var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_TENSORSTRUCT_INPUT].AttributeDefs;
                    var attrs = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?>
                        {
                            [AttrDtype] = structDType,
                            [ShrkAttrInputType] = inputType,
                        },
                        attrDefs);

                    fastNode = new FastNode
                    {
                        Key = nodeKey,
                        OpCode = InternalOpCodes.MODEL_TENSORSTRUCT_INPUT,
                        Attributes = attrs,
                        FullOutputs = { [""] = new List<FastTensorKey?> { key } },
                        TargetFunction = targetFunction,
                        FriendlyName = inputProto.Name,
                    };
                }
                else
                {
                    var dtype = (DType)elemType;
                    string[]? constraints = null;
                    if (!string.IsNullOrEmpty(inputProto.Type.Denotation))
                    {
                        var split = inputProto.Type.Denotation.Split(':');
                        if (split.Length == 0 || split.Length > 2)
                            throw new InvalidOperationException();

                        var genericTypeParamName = split[0].Trim();
                        var constraint = split.Length == 2 ? split[1].Trim() : null;
                        constraints = constraint is null ? null : constraint.Split(',').Select(x => x.Trim()).ToArray();
                        dtype = DType.CreateWithGenericParam(dtype, genericTypeParamName);
                    }

                    var tensorShape = GetTensorDims(tensorType.Shape);
                    long? rank = tensorShape?.Length;

                    var inputTypeName = inputProto.MetadataProps
                        .Where(x => x.Key == Function.IRInputTypeName)
                        .SingleOrDefault()?.Value;
                    var inputType = Function.FromInputTypeName(inputTypeName);

                    var targetFunctionName = inputProto.MetadataProps
                        .Where(x => x.Key == Function.IRFunctionSignatureParamName)
                        .SingleOrDefault()?.Value;
                    var targetFunction = targetFunctionName is null ? null : knownFunctions[targetFunctionName];

                    if (inputType.Equals(InputType.GenericType))
                    {
                        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.GENERIC_TYPE_INPUT].AttributeDefs;
                        var attrVals = new Dictionary<string, object?>
                        {
                            [AttrDtype] = dtype,
                            [ShrkAttrInputType] = InputType.GenericType,
                            [ShrkAttrRank] = rank,
                        };
                        if (constraints != null)
                            attrVals[ShrkAttrGenericTypeConstraints] = constraints;
                        var attrs = OnnxCSharpAttributes.FromCSharpVals(attrVals, attrDefs);

                        fastNode = new FastNode
                        {
                            Key = nodeKey,
                            OpCode = InternalOpCodes.GENERIC_TYPE_INPUT,
                            Attributes = attrs,
                            FullOutputs = { [""] = new List<FastTensorKey?> { key } },
                            FriendlyName = inputProto.Name,
                        };
                    }
                    else if (structure == DataStructure.Optional)
                    {
                        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_OPTIONAL_INPUT].AttributeDefs;
                        var attrs = OnnxCSharpAttributes.FromCSharpVals(
                            new Dictionary<string, object?>
                            {
                                [AttrDtype] = dtype,
                                [ShrkAttrInputType] = inputType,
                            },
                            attrDefs);

                        fastNode = new FastNode
                        {
                            Key = nodeKey,
                            OpCode = InternalOpCodes.MODEL_OPTIONAL_INPUT,
                            Attributes = attrs,
                            FullOutputs = { [""] = new List<FastTensorKey?> { key } },
                            TargetFunction = targetFunction,
                            FriendlyName = inputProto.Name,
                        };
                    }
                    else if (structure == DataStructure.Sequence)
                    {
                        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_SEQUENCE_INPUT].AttributeDefs;
                        var attrs = OnnxCSharpAttributes.FromCSharpVals(
                            new Dictionary<string, object?>
                            {
                                [AttrDtype] = dtype,
                                [ShrkAttrInputType] = inputType,
                            },
                            attrDefs);

                        fastNode = new FastNode
                        {
                            Key = nodeKey,
                            OpCode = InternalOpCodes.MODEL_SEQUENCE_INPUT,
                            Attributes = attrs,
                            FullOutputs = { [""] = new List<FastTensorKey?> { key } },
                            TargetFunction = targetFunction,
                            FriendlyName = inputProto.Name,
                        };
                    }
                    else
                    {
                        var defaultValueStr = inputProto.MetadataProps
                            .Where(x => x.Key == Function.IRDefaultValue)
                            .SingleOrDefault()?.Value;
                        float? defaultValue = defaultValueStr is null
                            ? null
                            : float.Parse(defaultValueStr, System.Globalization.CultureInfo.InvariantCulture);

                        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_TENSOR_INPUT].AttributeDefs;
                        var attrVals = new Dictionary<string, object?>
                        {
                            [AttrDtype] = dtype,
                            [ShrkAttrInputType] = inputType,
                            [ShrkAttrRank] = rank,
                        };
                        if (defaultValue is float dv)
                            attrVals[ShrkAttrDefaultValue] = dv;
                        var attrs = OnnxCSharpAttributes.FromCSharpVals(attrVals, attrDefs);

                        fastNode = new FastNode
                        {
                            Key = nodeKey,
                            OpCode = InternalOpCodes.MODEL_TENSOR_INPUT,
                            Attributes = attrs,
                            FullOutputs = { [""] = new List<FastTensorKey?> { key } },
                            TargetFunction = targetFunction,
                            FriendlyName = inputProto.Name,
                        };
                    }
                }

                results.Add((inputProto, key, fastNode));
            }

            return results;
        }

        /// <summary>
        /// FastCG initializer loader: one MODEL_PARAM_DATA <see cref="FastNode"/> per
        /// initializer tensor, carrying the trainability flag and identifier-template
        /// metadata that the ONNX serializer stamped onto the <see cref="TensorProto"/>.
        /// </summary>
        private static List<(TensorProto proto, FastTensorKey key, FastNode node)> CreateFastInitializers(List<TensorProto> initializers)
        {
            var results = new List<(TensorProto, FastTensorKey, FastNode)>();
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM_DATA].AttributeDefs;

            foreach (var initializer in initializers)
            {
                var identifierTemplate = initializer.MetadataProps
                    .FirstOrDefault(x => x.Key == ShrkMetaNodeIdentifierTemplate)?.Value;
                var isTrainableStr = initializer.MetadataProps
                    .FirstOrDefault(x => x.Key == ShrkMetaIsTrainable)?.Value;
                var isTrainable = ParseIsTrainableMetadata(isTrainableStr);
                var tensorData = CreateTensorData(initializer);

                var attrs = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [ShrkAttrTensorData] = tensorData,
                        [ShrkAttrIsTrainable] = isTrainable,
                    },
                    attrDefs);

                var nodeKey = FastNodeKey.New();
                var key = new FastTensorKey(nodeKey, 0);
                var fastNode = new FastNode
                {
                    Key = nodeKey,
                    OpCode = InternalOpCodes.MODEL_PARAM_DATA,
                    Attributes = attrs,
                    FullOutputs = { [""] = new List<FastTensorKey?> { key } },
                    IdentifierTemplate = identifierTemplate,
                    FriendlyName = initializer.Name,
                };
                results.Add((initializer, key, fastNode));
            }

            return results;
        }

        /// <summary>
        /// Walks the proto's <c>GraphProto.Nodes</c> recursively (sub-graph attributes
        /// descended into between the parent op's OPEN and CLOSE) emitting one
        /// <see cref="TempNode"/> per visited proto node in canonical declared order.
        /// The proto's positional order is the canonical execution order on the load
        /// side: keeping body nodes between their LOOP/IF OPEN and CLOSE is what lets
        /// downstream passes (e.g. <c>FastFoldConstantIterationLoops</c>) identify the
        /// loop body correctly.
        /// </summary>
        private static IEnumerable<TempNode> EnumerateNodesInProtoOrder(List<NodeProto> nodes)
        {
            foreach (var node in nodes)
            {
                bool hasSubgraph = false;
                foreach (var attr in node.Attributes)
                {
                    if (attr.Type == AttributeProto.AttributeType.Graph
                        || attr.Type == AttributeProto.AttributeType.Graphs)
                    {
                        hasSubgraph = true;
                        break;
                    }
                }

                if (!hasSubgraph)
                {
                    yield return new TempNode(node);
                    continue;
                }

                yield return new TempNode(node, isGraphOpen: true);

                // Sub-graphs in canonical attribute order: single Graph attrs in their
                // declared order first, then Graphs (plural / variadic) sorted by attr
                // name. Matches the ordering the IR_10 saver writes.
                foreach (var attr in node.Attributes)
                {
                    if (attr.Type == AttributeProto.AttributeType.Graph)
                    {
                        foreach (var tn in EnumerateNodesInProtoOrder(attr.G.Nodes))
                            yield return tn;
                    }
                }
                foreach (var attr in node.Attributes
                    .Where(x => x.Type == AttributeProto.AttributeType.Graphs)
                    .OrderBy(x => x.Name))
                {
                    foreach (var subgraph in attr.Graphs)
                        foreach (var tn in EnumerateNodesInProtoOrder(subgraph.Nodes))
                            yield return tn;
                }

                yield return new TempNode(node, isGraphClose: true);
            }
        }

        /// <summary>
        /// Walks <paramref name="tempNodes"/> in the canonical proto order produced by
        /// <see cref="EnumerateNodesInProtoOrder"/> and materializes one
        /// <see cref="FastNode"/> per visited node. Three-way branch:
        /// <list type="bullet">
        ///   <item>Ethereal IDENTITY: aliases the output tensor name to its input — no node emitted.</item>
        ///   <item>Function call: routes to <see cref="BuildFastTrainableParamNodeFromProto"/>
        ///         for initializer functions (TRAINABLE_PARAM) or to
        ///         <see cref="BuildFastFunctionInvokeNodeFromProto"/> for ordinary function calls.</item>
        ///   <item>Built-in op: <see cref="BuildFastBuiltinNodeFromProto"/>, with OPEN/CLOSE
        ///         handling that walks the same per-graph-attribute fan-out as the CG path.</item>
        /// </list>
        /// </summary>
        private static void CreateFastNodes(
            FastComputationGraph fastGraph,
            IEnumerable<TempNode> tempNodes,
            Dictionary<string, FastTensorKey> tensorKeys,
            IReadOnlyDictionary<string, Function> functionsMap,
            OpSetVersion opset)
        {
            var graphOpenNodeKeys = new Dictionary<NodeProto, FastNodeKey>();
            var definitions = Definitions.NodeDefinitions;

            foreach (var tempNode in tempNodes)
            {
                var opCode = tempNode.Node.OpType;
                opCode = tempNode.IsGraphOpen ? $"{opCode}#OPEN" :
                         tempNode.IsGraphClose ? $"{opCode}#CLOSE" :
                         opCode;

                var nodeProto = tempNode.Node;

                Debug.Assert(functionsMap.ContainsKey(opCode) || definitions.ContainsKey(opCode) || Definitions.ModuleOps.Contains(opCode), opCode);

                if (isEtherealIdentityNode(nodeProto))
                {
                    tensorKeys[nodeProto.Outputs.Single()] = tensorKeys[nodeProto.Inputs.Single()];
                    continue;
                }

                FastNode fastNode;
                if (functionsMap.TryGetValue(opCode, out var function))
                {
                    if ((function.FunctionType == FunctionType.TrainableParamInitializer ||
                         function.FunctionType == FunctionType.StateParamInitializer) &&
                        definitions.ContainsKey(InternalOpCodes.TRAINABLE_PARAM))
                    {
                        fastNode = BuildFastTrainableParamNodeFromProto(nodeProto, function, tensorKeys, definitions);
                    }
                    else
                    {
                        fastNode = BuildFastFunctionInvokeNodeFromProto(nodeProto, function, tensorKeys);
                    }
                }
                else if (definitions.ContainsKey(opCode))
                {
                    fastNode = BuildFastBuiltinNodeFromProto(opCode, nodeProto, tensorKeys, functionsMap, definitions, graphOpenNodeKeys);
                }
                else
                {
                    throw new UnsupportedDTypeException(ErrorCodes.FW030, nodeProto.OpType, "attribute type conversion",
                        $"Attribute type conversion not implemented for node type '{nodeProto.OpType}'");
                }

                fastGraph.Nodes.Add(fastNode);
            }
        }

        /// <summary>
        /// Builds a TRAINABLE_PARAM <see cref="FastNode"/> for a serialized initializer
        /// function call. Mirrors the CG path which re-routes initializer-typed function
        /// invocations to the TRAINABLE_PARAM op code so downstream model-param discovery
        /// still finds them.
        /// </summary>
        private static FastNode BuildFastTrainableParamNodeFromProto(
            NodeProto nodeProto,
            Function function,
            Dictionary<string, FastTensorKey> tensorKeys,
            IReadOnlyDictionary<string, NodeDefinitionResolver> definitions)
        {
            var resolver = definitions[InternalOpCodes.TRAINABLE_PARAM];
            var (attrs, nodeDef) = ParseAttributes(nodeProto, resolver);
            var identifierTemplate = nodeProto.MetadataProps.FirstOrDefault(x => x.Key == ShrkMetaNodeIdentifierTemplate)?.Value;
            var stackTrace = nodeProto.MetadataProps.FirstOrDefault(x => x.Key == "StackTrace")?.Value;
            var nodeKey = ParseFastNodeKey(nodeProto);

            var inputKeys = LookupInputKeys(nodeProto.Inputs, tensorKeys);
            inputKeys = PadInputsWithNulls(inputKeys, nodeDef);

            var outputKeys = AllocateAndRecordOutputs(nodeKey, nodeProto.Outputs, tensorKeys, baseIndex: 0);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.TRAINABLE_PARAM,
                Attributes = attrs,
                FullInputs = { [""] = inputKeys.ToList() },
                FullOutputs = { [""] = outputKeys.Select(k => (FastTensorKey?)k).ToList() },
                IdentifierTemplate = identifierTemplate,
                StackTrace = stackTrace,
                TargetFunction = function,
                FriendlyName = nodeProto.Name,
            };
        }

        /// <summary>
        /// Builds a FUNCTION_INVOKE <see cref="FastNode"/> for an ordinary function-call
        /// proto node. Output structure / dtype / rank are read from the target Function's
        /// declared outputs (matching <see cref="Function.Call"/>).
        /// </summary>
        private static FastNode BuildFastFunctionInvokeNodeFromProto(
            NodeProto nodeProto,
            Function function,
            Dictionary<string, FastTensorKey> tensorKeys)
        {
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;
            var fastFnGraph = function.OriginalFastGraph;
            var fnOutputs = FastComputationGraphConverter.BuildNodes(fastFnGraph).outputs;
            var fnRankOverrides = fastFnGraph.OutputRankOverrides is null
                ? fnOutputs.Select(x => (int?)x.Rank).ToImmutableArray()
                : fastFnGraph.OutputRankOverrides.ToImmutableArray();

            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrStructure] = fnOutputs.Select(x => x.Structure()).ToArray(),
                    [ShrkAttrDtype] = fnOutputs.Select(x => x.DType).ToArray(),
                    [ShrkAttrRank] = fnRankOverrides.Select(x => (long)(x ?? -1)).ToArray(),
                    [ShrkAttrGenericTypeArgs] = (DType[]?)null,
                },
                attrDefs);

            var stackTrace = nodeProto.MetadataProps.FirstOrDefault(x => x.Key == "StackTrace")?.Value;
            var nodeKey = ParseFastNodeKey(nodeProto);

            var inputKeys = LookupInputKeys(nodeProto.Inputs, tensorKeys);
            var outputKeys = AllocateAndRecordOutputs(nodeKey, nodeProto.Outputs, tensorKeys, baseIndex: 0);

            return new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.FUNCTION_INVOKE,
                Attributes = attrs,
                FullInputs = { [""] = inputKeys.ToList() },
                FullOutputs = { [""] = outputKeys.Select(k => (FastTensorKey?)k).ToList() },
                StackTrace = stackTrace,
                TargetFunction = function,
                FriendlyName = nodeProto.Name,
            };
        }

        /// <summary>
        /// Builds a built-in op <see cref="FastNode"/> from a proto node. Handles
        /// OPEN/CLOSE wiring: an OPEN node's outputs are the body subgraph's input
        /// names (one slot per graph attribute, fanned out alphabetically), and a CLOSE
        /// node's inputs are the body subgraph's output names (one slot per graph
        /// attribute). For CLOSE nodes, also resolves and stores the
        /// <see cref="FastNode.GraphOpenNodeKey"/> back-reference.
        /// </summary>
        private static FastNode BuildFastBuiltinNodeFromProto(
            string opCode,
            NodeProto nodeProto,
            Dictionary<string, FastTensorKey> tensorKeys,
            IReadOnlyDictionary<string, Function> functionsMap,
            IReadOnlyDictionary<string, NodeDefinitionResolver> definitions,
            Dictionary<NodeProto, FastNodeKey> graphOpenNodeKeys)
        {
            var resolver = definitions[opCode];
            var (attrs, nodeDef) = ParseAttributes(nodeProto, resolver);
            var identifierTemplate = nodeProto.MetadataProps.FirstOrDefault(x => x.Key == ShrkMetaNodeIdentifierTemplate)?.Value;
            var stackTrace = nodeProto.MetadataProps.FirstOrDefault(x => x.Key == "StackTrace")?.Value;
            var nodeKey = ParseFastNodeKey(nodeProto);

            // Determine FullInputs/FullOutputs layout based on OPEN/CLOSE/regular.
            var fullInputs = new Dictionary<string, List<FastTensorKey?>>();
            var fullOutputs = new Dictionary<string, List<FastTensorKey?>>();
            FastNodeKey? graphOpenNodeKey = null;
            string? friendlyName = nodeProto.Name;

            if (nodeDef.IsOpenNode)
            {
                var inputKeys = LookupInputKeys(nodeProto.Inputs, tensorKeys);
                inputKeys = PadInputsWithNulls(inputKeys, nodeDef);
                fullInputs[""] = inputKeys.ToList();

                // Body subgraph input names become the OPEN's outputs, one slot per graph
                // attribute (e.g. "body" for LOOP, "then_branch"/"else_branch" for IF),
                // fanned out alphabetically so the FastTensorKey OutputIndex matches the
                // flattened position in FastNode.Outputs.
                int outputCounter = 0;
                var graphAttrs = nodeProto.Attributes
                    .Where(x => x.Type == AttributeProto.AttributeType.Graph)
                    .OrderBy(x => x.Name, StringComparer.Ordinal);
                foreach (var graphAttr in graphAttrs)
                {
                    var slot = new List<FastTensorKey?>();
                    foreach (var bodyInput in graphAttr.G.Inputs)
                    {
                        var key = new FastTensorKey(nodeKey, outputCounter++);
                        slot.Add(key);
                        tensorKeys[bodyInput.Name] = key;
                    }
                    fullOutputs[graphAttr.Name] = slot;
                }
            }
            else if (nodeDef.IsCloseNode)
            {
                friendlyName = nodeProto.Name + "_CLOSE";

                // Body subgraph output names become the CLOSE's inputs, one slot per
                // graph attribute. Look them up from tensorKeys.
                var graphAttrs = nodeProto.Attributes
                    .Where(x => x.Type == AttributeProto.AttributeType.Graph);
                foreach (var graphAttr in graphAttrs)
                {
                    var slot = new List<FastTensorKey?>();
                    foreach (var bodyOutput in graphAttr.G.Outputs)
                    {
                        slot.Add(tensorKeys.TryGetValue(bodyOutput.Name, out var k) ? (FastTensorKey?)k : null);
                    }
                    fullInputs[graphAttr.Name] = slot;
                }

                var outputKeys = AllocateAndRecordOutputs(nodeKey, nodeProto.Outputs, tensorKeys, baseIndex: 0);
                fullOutputs[""] = outputKeys.Select(k => (FastTensorKey?)k).ToList();

                if (!graphOpenNodeKeys.TryGetValue(nodeProto, out var openKey))
                    throw new InvalidOperationException(
                        $"OnnxModelReader: CLOSE node '{nodeProto.Name}' has no matching OPEN node recorded.");
                graphOpenNodeKey = openKey;
            }
            else
            {
                var inputKeys = LookupInputKeys(nodeProto.Inputs, tensorKeys);
                inputKeys = PadInputsWithNulls(inputKeys, nodeDef);
                fullInputs[""] = inputKeys.ToList();

                var outputKeys = AllocateAndRecordOutputs(nodeKey, nodeProto.Outputs, tensorKeys, baseIndex: 0);
                fullOutputs[""] = outputKeys.Select(k => (FastTensorKey?)k).ToList();
            }

            // Target function from attribute, if any.
            Function? targetFunction = null;
            if (attrs.IsAttributeDefined(ShrkAttrFunctionName) && !attrs.IsDefaultValue(ShrkAttrFunctionName))
            {
                var functionName = attrs.GetStringVal(ShrkAttrFunctionName).AssertNotNull();
                targetFunction = functionsMap[functionName];
            }

            var fastNode = new FastNode
            {
                Key = nodeKey,
                OpCode = opCode,
                Attributes = attrs,
                FullInputs = fullInputs,
                FullOutputs = fullOutputs,
                IdentifierTemplate = identifierTemplate,
                StackTrace = stackTrace,
                TargetFunction = targetFunction,
                GraphOpenNodeKey = graphOpenNodeKey,
                FriendlyName = friendlyName,
            };

            if (nodeDef.IsOpenNode)
                graphOpenNodeKeys[nodeProto] = nodeKey;

            return fastNode;
        }

        /// <summary>
        /// Parses <see cref="NodeProto.Attributes"/> via <see cref="getAttributeValue"/>
        /// and resolves the matching specialized <see cref="NodeDefinition"/> from the
        /// <see cref="NodeDefinitionResolver"/>. Returns both because callers need the
        /// CSharp-form attributes for the FastNode and the resolved nodeDef for input
        /// padding / OPEN-CLOSE classification.
        /// </summary>
        private static (OnnxCSharpAttributes attrs, NodeDefinition nodeDef) ParseAttributes(
            NodeProto nodeProto, NodeDefinitionResolver resolver)
        {
            var attrDefsByName = resolver.AttributeDefs.ToDictionary(x => x.AttributeName, x => x);
            var attrsDct = nodeProto.Attributes.ToDictionary(
                x => x.Name,
                x => getAttributeValue(x, attrDefsByName.GetValueOrDefault(x.Name)));

            // Internal attributes are never serialized to ONNX (OnnxIRFactory skips
            // IsInternalAttribute defs), so the variant-selecting #has_optional_outputs#
            // flag (MAX_POOL: Y[, Indices]; ATTENTION: Y[, present_key, present_value])
            // is reconstructed here from the NodeProto's non-empty output count.
            // Without it, Resolve falls back to the FIRST variant — wrong output arity
            // for a single-output Attention node (and a silently wrong variant for
            // MaxPool, whose two variants merely happen to be structurally identical).
            if (attrDefsByName.ContainsKey(InternalAttrHasOptionalOutputs)
                && !attrsDct.ContainsKey(InternalAttrHasOptionalOutputs))
            {
                var nonEmptyOutputs = nodeProto.Outputs.Count(n => !string.IsNullOrEmpty(n));
                attrsDct[InternalAttrHasOptionalOutputs] = nonEmptyOutputs > 1 ? 1L : 0L;
            }

            // QuantizeLinear's output dtype group is bound by y_zero_point OR the
            // output_dtype attribute; when an imported node carries neither, inject the
            // spec default (uint8) so resolution has a binding source — same synthesis
            // idea as the #has_optional_outputs# reconstruction above.
            if (nodeProto.OpType == "QuantizeLinear"
                && !attrsDct.ContainsKey(OnnxOpAttributeNames.AttrOutputDtype)
                && (nodeProto.Inputs.Count < 3 || string.IsNullOrEmpty(nodeProto.Inputs[2])))
            {
                attrsDct[OnnxOpAttributeNames.AttrOutputDtype] = (long)DType.UInt8.ProtoTypeNum;
            }

            var protoAttrs = OnnxProtoAttributes.FromProtoVals(attrsDct, resolver.AttributeDefs);
            var nodeDef = resolver.Resolve(protoAttrs);
            return (protoAttrs.ToCSharp(), nodeDef);
        }

        /// <summary>
        /// Reads the optional <see cref="OnnxOpAttributeNames.ShrkMetaNodeKey"/> metadata
        /// (a serialized CG <see cref="NodeKey"/> Guid) and lifts it to a
        /// <see cref="FastNodeKey"/> so round-trips that originated as CG retain identity.
        /// Falls back to a fresh key if the metadata is absent or unparseable.
        /// </summary>
        private static FastNodeKey ParseFastNodeKey(NodeProto nodeProto)
        {
            var nodeKeyString = nodeProto.MetadataProps.FirstOrDefault(x => x.Key == ShrkMetaNodeKey)?.Value;
            if (NodeKey.TryParse(nodeKeyString, out var parsedKey))
                return FastNodeKey.FromCgKey(parsedKey);
            return FastNodeKey.New();
        }

        /// <summary>
        /// Maps a sequence of proto input names to <see cref="FastTensorKey"/>s via the
        /// caller-maintained tensor-name table. Empty / null names map to <c>null</c>
        /// (an explicit missing optional input slot).
        /// </summary>
        private static FastTensorKey?[] LookupInputKeys(
            IEnumerable<string> names, IReadOnlyDictionary<string, FastTensorKey> tensorKeys)
            => names.Select(x => string.IsNullOrEmpty(x) ? (FastTensorKey?)null : tensorKeys[x]).ToArray();

        /// <summary>
        /// Pads <paramref name="currentInputs"/> with trailing nulls up to the node-def's
        /// minimum non-variadic input count, mirroring
        /// <c>InputNodeDataProcessor.DeduceInputsWithNulls</c> on the CG side.
        /// </summary>
        private static FastTensorKey?[] PadInputsWithNulls(FastTensorKey?[] currentInputs, NodeDefinition nodeDef)
        {
            var minNumInputs = nodeDef.InputDefs.TakeWhile(x => x.VariadicCountDef is null).Count();
            var numNewTrailingNulls = minNumInputs - currentInputs.Length;
            if (numNewTrailingNulls <= 0) return currentInputs;
            var padded = new FastTensorKey?[currentInputs.Length + numNewTrailingNulls];
            Array.Copy(currentInputs, padded, currentInputs.Length);
            return padded;
        }

        /// <summary>
        /// Allocates a fresh <see cref="FastTensorKey"/> per output name, records each
        /// in the tensor-name → key table, and returns the allocated keys. Used by every
        /// non-OPEN producer to wire its outputs into the global lookup.
        /// </summary>
        private static FastTensorKey[] AllocateAndRecordOutputs(
            FastNodeKey nodeKey,
            IReadOnlyList<string> outputNames,
            Dictionary<string, FastTensorKey> tensorKeys,
            int baseIndex)
        {
            var keys = new FastTensorKey[outputNames.Count];
            for (int i = 0; i < outputNames.Count; i++)
            {
                var key = new FastTensorKey(nodeKey, baseIndex + i);
                keys[i] = key;
                if (!string.IsNullOrEmpty(outputNames[i]))
                    tensorKeys[outputNames[i]] = key;
            }
            return keys;
        }
    }
}
