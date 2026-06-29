using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Factory.CSharpFactory
{
    internal class CSharpModelBuilder
    {

        /// <summary>
        /// Sanitizes a variable name to be a valid C# identifier by replacing invalid characters.
        /// </summary>
        private static string SanitizeVariableName(string name)
        {
            // Replace colons and hyphens with underscores to make valid C# identifiers
            name = name.Replace(':', '_').Replace('-', '_');
            
            // If the name starts with a digit, prefix it with an underscore
            if (name.Length > 0 && char.IsDigit(name[0]))
            {
                name = "_" + name;
            }
            
            return name;
        }

        /// <summary>
        /// Gets a sanitized variable name from a tensor's UniqueName.
        /// </summary>
        private static string GetSanitizedVariableName(Variable tensor)
        {
            return SanitizeVariableName(tensor.UniqueName);
        }

        private struct CodeLine
        {
            public int IndentCount { get; }
            public string Code { get; }

            public CodeLine(int indentCount, string code)
            {
                IndentCount = indentCount;
                Code = code;
            }
        }

        private class NodeGenerationInfo
        {
            public NodeDefinition NodeDef { get; }
            public Node Node { get; }
            public string? InlineExpression { get; }
            public ImmutableList<CodeLine> FullCode { get; }
            public ImmutableList<Node> InlinedNodes { get; }

            public NodeGenerationInfo(NodeDefinition nodeDef, Node node, string? inlineExpression, ImmutableList<CodeLine> fullCode, ImmutableList<Node> inlinedNodes)
            {
                NodeDef = nodeDef;
                Node = node;
                InlineExpression = inlineExpression;
                FullCode = fullCode;
                InlinedNodes = inlinedNodes;
            }
        }

        public static string GetTypeDefString(Variable Variable)
            => GetTypeDefString(Variable, Variable.Rank);

        public static string GetTypeDefString(Variable variable, int? rank)
        {
            // Handle TensorStruct types
            if (variable.Structure() == DataStructure.TensorStruct)
            {
                var def = variable.Definition;
                var structTypeName = def?.TypeName;
                if (structTypeName != null && !structTypeName.Contains('.'))
                {
                    // Simple type name - use as-is
                    return $"TensorStruct<{structTypeName}>";
                }
                else
                {
                    // For dynamic TensorStruct (DTypeStruct) or fully qualified names, use DTypeStruct
                    return $"TensorStruct<DTypeStruct>";
                }
            }

            var typeName = variable.Type.ToIVarType().Name;
            if (rank == 0)
                return $"Scalar<{typeName}>";
            else if (rank == 1)
                return $"Vector<{typeName}>";
            else if (variable.Structure() == DataStructure.Tensor)
                return $"Tensor<{typeName}>";
            else
                return $"IValue<{typeName}>";
        }

        public static string GetModuleAwareTypeDefString(Function targetFunction, bool asModel)
        {
            if (asModel)
            {
                var paramList = string.Join(", ", targetFunction.NonHyperparamInputs.Select(x => GetModuleAwareTypeDefString(x, x.Rank)));
                if (targetFunction.NonHyperparamInputs.Length > 0)
                    paramList += ", ";

                if (targetFunction.Outputs.Length == 1)
                    paramList += GetModuleAwareTypeDefString(targetFunction.Outputs[0], targetFunction.OutputRankOverrides[0]);
                else
                    paramList += $"({string.Join(", ", targetFunction.Outputs.Zip(targetFunction.OutputRankOverrides).Select(x => GetModuleAwareTypeDefString(x.First, x.Second)))})";

                return $"Model<{paramList}>";
            }
            else
            {
                var hyperParamList = string.Join(", ", targetFunction.HyperparamInputs.Select(x => GetModuleAwareTypeDefString(x, x.Rank)));
                var nonHyperParamList = string.Join(", ", targetFunction.NonHyperparamInputs.Select(x => GetModuleAwareTypeDefString(x, x.Rank)));
                var outputsList = string.Join(", ", targetFunction.Outputs.Zip(targetFunction.OutputRankOverrides).Select(x => GetModuleAwareTypeDefString(x.First, x.Second)));

                if (targetFunction.HyperparamInputs.Length > 1)
                    hyperParamList = $"({hyperParamList})";

                if (targetFunction.NonHyperparamInputs.Length > 1)
                    nonHyperParamList = $"({nonHyperParamList})";

                if (targetFunction.Outputs.Length > 1)
                    outputsList = $"({outputsList})";

                var moduleTypeName = targetFunction.NonHyperparamInputs.Length > 0 ? "Module" : "CallbackModule";
                var fullParamList = string.Join(", ", ((string[])[hyperParamList, nonHyperParamList, outputsList]).Where(x => x != ""));

                return $"{moduleTypeName}<{fullParamList}>";
            }
        }

        public static string GetModuleAwareTypeDefString(Variable variable, int? rankOverride)
        {
            // Model/module params are scalar nodes distinguished by runtime DType (formerly the generic
            // Variable<IModelVarType> / Variable<IModuleVarType>).
            if (variable is Variable modelVariable && modelVariable.Type == DType.Model)
                return GetModuleAwareTypeDefString(modelVariable.ModuleFn.AssertNotNull(), asModel: true);
            else if(variable is Variable moduleVariable && moduleVariable.Type == DType.Module)
                return GetModuleAwareTypeDefString(moduleVariable.ModuleFn.AssertNotNull(), asModel: false);

            return GetTypeDefString(variable, rankOverride);
        }

        public static string GetModuleAwareTypeDefString(Variable tensor)
        {
            if (tensor.DType == DType.Model)
                return GetModuleAwareTypeDefString(tensor.ModuleFn.AssertNotNull(), asModel: true);
            else if (tensor.DType == DType.Module)
                return GetModuleAwareTypeDefString(tensor.ModuleFn.AssertNotNull(), asModel: false);

            return GetTypeDefString(tensor);
        }

        public MethodInfo BuildMethod(FastComputationGraph fastGraph, string modelName)
        {
            var code = BuildFullGraph(fastGraph, modelName);
            // Create a syntax tree from the generated code
            var syntaxTree = CSharpSyntaxTree.ParseText(code);

            // Define references to necessary assemblies
            var references = new List<MetadataReference>
            {
                MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Enumerable).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Shorokoo.Core.Variable).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Float16).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(System.Collections.Immutable.ImmutableArray).Assembly.Location),
                MetadataReference.CreateFromFile(Path.Combine(Path.GetDirectoryName(typeof(object).Assembly.Location)!, "System.Runtime.dll")), // Add System.Runtime
                // Add other necessary references here
            };

            // Compile the syntax tree into an assembly
            var compilation = CSharpCompilation.Create(
                modelName,
                new[] { syntaxTree },
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

            using (var ms = new MemoryStream())
            {
                var result = compilation.Emit(ms);
                Debug.Assert(result.Success,
                    "C# model compilation failed: " + string.Join("; ",
                        result.Diagnostics
                            .Where(d => d.IsWarningAsError || d.Severity == DiagnosticSeverity.Error)
                            .Select(d => d.ToString())));

                ms.Seek(0, SeekOrigin.Begin);
                var assembly = Assembly.Load(ms.ToArray());

                // Get the generated method and create a delegate
                var type = assembly.GetType($"GeneratedFromOnnx.{modelName}").NotNull();
                var method = type.GetMethod("BuildComputationGraph").NotNull();
                return method;
            }
        }

        public Func<TResult> BuildLambda<TResult>(FastComputationGraph fastGraph, string modelName)
        {
            var method = BuildMethod(fastGraph, modelName);
            try
            {
                return (Func<TResult>)Delegate.CreateDelegate(typeof(Func<TResult>), method);
            }
            catch
            {
                return () => (TResult)method.Invoke(null, []).NotNull();
            }
        }

        public Func<T1, TResult> BuildLambda<T1, TResult>(FastComputationGraph fastGraph, string modelName)
        {
            var method = BuildMethod(fastGraph, modelName);
            return (Func<T1, TResult>)Delegate.CreateDelegate(typeof(Func<T1, TResult>), method);
        }

        public Func<T1, T2, TResult> BuildLambda<T1, T2, TResult>(FastComputationGraph fastGraph, string modelName)
        {
            var method = BuildMethod(fastGraph, modelName);
            return (Func<T1, T2, TResult>)Delegate.CreateDelegate(typeof(Func<T1, T2, TResult>), method);
        }

        public string BuildFullGraph(FastComputationGraph fastGraph, string modelName)
        {
            var functions = FastComputationGraphConverter.FunctionsPostOrder(fastGraph);
            var functionNames = functions.ToImmutableDictionary(x => x, x => x.FriendlyName);
            var mainMethod = BuildMethodCode(fastGraph, "BuildComputationGraph", functionNames, targetFunction: null);
            var referencedMethods = functions.Select(x => BuildMethodCode(x.OriginalFastGraph, x.FriendlyName, functionNames, x)).ToList();
            var methodImplsCode = string.Join("\r\n", referencedMethods);
            methodImplsCode = mainMethod + "\r\n" + methodImplsCode;
            var fullScript = @"
using System;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using static Shorokoo.Core.Nodes.NodeDefinitions.NodeBuilder;
using static Shorokoo.Core.Nodes.NodeDefinitions.NodeBuilder;
using static Shorokoo.Core.Nodes.NodeDefinitions.NodeBuilder;
using static Shorokoo.Globals;
using static Shorokoo.Globals;
using static Shorokoo.Globals;
using static Shorokoo.Globals;
using static Shorokoo.Globals;
using static Shorokoo.Globals;
using static Shorokoo.Globals;
using Shorokoo.Core.Inference.Abstractions;

namespace GeneratedFromOnnx;

public static class " + modelName + @"
{
" + methodImplsCode + @"
}";

            return fullScript;
        }



        public string BuildMethodCode(FastComputationGraph fastGraph, string methodName, ImmutableDictionary<Function, string> functionNames, Function? targetFunction)
        {
            // Rebuild the Node/Variable view of the graph without paying for a full
            // ComputationGraph wrapper (validation, topo-sort, function discovery, etc.).
            // The codegen below is structured around Node identity and Variable metadata,
            // which the converter produces directly.
            var (topologicalOrderNodes, graphInputs, graphOutputs, _) = FastComputationGraphConverter.BuildNodes(fastGraph);

            var variableNames = new Dictionary<Variable, string>();
            var nodeCodeGenerators = new Dictionary<Node, NodeGenerationInfo>();
            var inlinedNodes = new List<Node>();

            // Build a mapping from each tensor to its consuming (child) nodes. This is expensive, so we compute it once.
            var tensorChildNodes = new Dictionary<Variable, List<Node>>();
            foreach (var n in topologicalOrderNodes)
            {
                foreach (var inputTensor in n.Inputs.Append(n.IsCloseNode ? n.ConnectingTensor : null).NotNulls())
                {
                    if (!tensorChildNodes.ContainsKey(inputTensor))
                        tensorChildNodes[inputTensor] = new List<Node>();
                    tensorChildNodes[inputTensor].Add(n);
                }
            }

            foreach (var node in topologicalOrderNodes)
            {
                (var newNodeGenerator, var newVariableNames) = this.MakeNode(node, nodeCodeGenerators.ToImmutableDictionary(), variableNames.ToImmutableDictionary(), functionNames, tensorChildNodes);
                var newInlinedVariables = newNodeGenerator.InlinedNodes;
                variableNames.AddAll(newVariableNames);
                nodeCodeGenerators[node] = newNodeGenerator;
                inlinedNodes.AddAll(newInlinedVariables);
            }

            foreach (var inlinedNode in inlinedNodes)
                nodeCodeGenerators.Remove(inlinedNode);

            var inputParams = new List<string>();
            var includeHyperparamAttribute = (targetFunction?.FunctionType == FunctionType.Module);
            foreach (var input in graphInputs)
            {
                var attribute = "";
                if (includeHyperparamAttribute && input.InputType == InputType.Hyperparam)
                {
                    // Re-emit the [Hyper(defaultValue)] default when one was recorded, so a defaulted
                    // hyperparameter round-trips through C# emission as a defaulted hyperparameter.
                    attribute = input.HyperDefaultValue is float dv
                        ? $"[Hyper({dv.ToString(System.Globalization.CultureInfo.InvariantCulture)}f)] "
                        : "[Hyper] ";
                }
                inputParams.Add($"{attribute}{GetModuleAwareTypeDefString(input)} {GetSanitizedVariableName(input)}");
            }

            var inputParamList = string.Join(", ", inputParams);
            var outputParamList = "";
            var returnLine = "";

            if (graphOutputs.Length == 1)
            {
                outputParamList = GetTypeDefString(graphOutputs[0].AssertNotNull(), graphOutputs[0].TensorDims?.Length);
                returnLine = $"return {GetSanitizedVariableName(graphOutputs[0])};";

            }
            else
            {
                outputParamList = "(" +
                    string.Join(", ", graphOutputs.Select(x => $"{GetTypeDefString(x.AssertNotNull(), x.TensorDims?.Length)} {GetSanitizedVariableName(x)}")) +
                    ")";

                returnLine = "return (" + string.Join(", ", graphOutputs.Select(x => $"{GetSanitizedVariableName(x)}")) + ");";
            }

            // Build reverse mapping from variable name to Variable for type resolution
            // Collect all code lines with scope depths. Variables that the
            // standard pipeline produces are always either pre-declared at the
            // outer scope (loop carry-overs, scan outputs) or used only within
            // their own scope, so there's no need for the duplicate-declaration
            // dedup or cross-scope hoisting passes that earlier versions of
            // this codegen carried.
            var allLines = new List<(string code, int indent)>();
            int defaultIndent = 2;

            foreach (var node in topologicalOrderNodes)
            {
                if (nodeCodeGenerators.TryGetValue(node, out var nodeCodeGenerator))
                {
                    if (node.IsCloseNode)
                        defaultIndent -= 1;

                    foreach (var codeLine in nodeCodeGenerator.AssertNotNull().FullCode)
                    {
                        var effectiveIndent = defaultIndent + codeLine.IndentCount;
                        allLines.Add((codeLine.Code, effectiveIndent));
                    }

                    if (node.IsGraphOpenNode)
                        defaultIndent += 1;
                }
            }

            var mainScript = "";
            foreach (var (code, indent) in allLines)
            {
                mainScript += new string('\t', indent) + code + "\r\n";
            }

            var methodSignature = targetFunction?.FunctionType == FunctionType.TrainableParamInitializer ||
                                   targetFunction?.FunctionType == FunctionType.StateParamInitializer ?
                $"private static {outputParamList} _{methodName}({inputParamList})" :
                $"public static {outputParamList} {methodName}({inputParamList})";

            var fullScript = methodSignature + @"
    {
" + mainScript + @"
        " + returnLine + @"
    }";
            if (targetFunction is not null)
            {
                if (targetFunction.FunctionType == FunctionType.Module)
                {
                    var moduleTypeString = GetModuleAwareTypeDefString(targetFunction, asModel: false);
                    var modelTypeString = GetModuleAwareTypeDefString(targetFunction, asModel: true);

                    var hyperParamTypes = targetFunction.HyperparamInputs.Select(x => GetModuleAwareTypeDefString(x, x.Rank)).ToList();
                    var hyperParamNames = hyperParamTypes.Select((x, i) => $"h{i}").ToList();

                    var hyperparamsDeclarationString =string.Join(", ", hyperParamTypes.Zip(hyperParamNames).Select(x => $"{x.First} {x.Second}"));
                    var hyperparamRefString = string.Join(", ", hyperParamNames);
                    if (hyperParamNames.Count > 1)
                        hyperparamRefString = $"({hyperparamRefString})";

                    var createModuleCode = $@"
        public static {moduleTypeString} {methodName}Module
            => new {moduleTypeString}(null, {methodName});";

                    var createModelCode = $@"
        public static {modelTypeString} {methodName}Model({hyperparamsDeclarationString})
            => {methodName}Module.SetHyperparams<{modelTypeString}>({hyperparamRefString});";

                    fullScript = "[Module]\r\n" + fullScript;
                    fullScript = createModuleCode + "\r\n" + createModelCode + "\r\n" + fullScript;
                }
                else if (targetFunction.FunctionType == FunctionType.TrainableParamInitializer)
                {
                    //         public static Shorokoo.float32> alpha)
                    //             => (Shorokoo.Tensor<Shorokoo.float32>)Globals.CallTrainableParamInitializer(customTrainableParamInitializer, shape, alpha);
                    var paramTypes = targetFunction.Inputs.Select(x => GetModuleAwareTypeDefString(x, x.Rank));
                    var paramNames = paramTypes.Select((x, i) => $"p{i}");
                    var paramsDeclarationString = string.Join(", ", paramTypes.Zip(paramNames).Select(x => $"{x.First} {x.Second}"));
                    var paramsRefString = string.Join(", ", paramNames);

                    var createTrainableParamInitializerCode = $@"
        public static {outputParamList} {methodName}({paramsDeclarationString})
            => ({outputParamList})Globals.CallTrainableParamInitializer(_{methodName}, defaultName: {'"' + methodName + '"'}, isTrainable: true, {paramsRefString});";

                    fullScript = "[TrainableParamInitializer]\r\n" + fullScript;

                    fullScript = createTrainableParamInitializerCode + "\r\n" + fullScript;
                }
                else if (targetFunction.FunctionType == FunctionType.StateParamInitializer)
                {
                    var paramTypes = targetFunction.Inputs.Select(x => GetModuleAwareTypeDefString(x, x.Rank));
                    var paramNames = paramTypes.Select((x, i) => $"p{i}");
                    var paramsDeclarationString = string.Join(", ", paramTypes.Zip(paramNames).Select(x => $"{x.First} {x.Second}"));
                    var paramsRefString = string.Join(", ", paramNames);

                    var createStateParamInitializerCode = $@"
        public static {outputParamList} {methodName}({paramsDeclarationString})
            => ({outputParamList})Globals.CallTrainableParamInitializer(_{methodName}, defaultName: {'"' + methodName + '"'}, isTrainable: false, {paramsRefString});";

                    fullScript = "[StateInitializer]\r\n" + fullScript;

                    fullScript = createStateParamInitializerCode + "\r\n" + fullScript;
                }
            }

            return fullScript;
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames, ImmutableDictionary<Function, string> functionNames, Dictionary<Variable, List<Node>> tensorChildNodes)
        {
            if (node.IsGraphNode)
            {
                if (node.FullNodeOpName == OpCodes.LOOP)
                    return MakeLoopNode(node, nodeCodeGenerators, currentNames, tensorChildNodes);

                if (node.FullNodeOpName == OpCodes.IF)
                    return MakeIfNode(node, nodeCodeGenerators, currentNames, tensorChildNodes);
            }

            if (node.OpName == OpCodes.CONSTANT)
            {
                var retVal = MakeConstantNode(node, nodeCodeGenerators, currentNames);
                if (retVal is not null)
                    return retVal.Value;
            }

            if (node.OpName == InternalOpCodes.CREATE_MODULE)
                return MakeCreateModuleNode(node, nodeCodeGenerators, currentNames, functionNames);

            if (node.OpName == InternalOpCodes.MODEL_TENSOR_INPUT ||
                node.OpName == InternalOpCodes.MODEL_OPTIONAL_INPUT ||
                node.OpName == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
                node.OpName == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT)
                return MakeInputNode(node, nodeCodeGenerators, currentNames);

            // Handle TensorStruct operations
            if (node.OpName == InternalOpCodes.TENSOR_STRUCT_GETFIELD)
                return MakeTensorStructGetFieldNode(node, nodeCodeGenerators, currentNames);

            if (node.OpName == InternalOpCodes.TENSOR_STRUCT_CREATE)
                return MakeTensorStructCreateNode(node, nodeCodeGenerators, currentNames);

            string? codeTemplateOverride = null;
            var useTuplesForMultiOutputs = false;
            if (node.OpName == InternalOpCodes.MODULE_SET_HYPERPARAMS)
            {
                codeTemplateOverride = MakeSetHyperparamsNodeCodeTemplate(node, nodeCodeGenerators, currentNames, functionNames);
                useTuplesForMultiOutputs = true;
            }
            else if (node.OpName == InternalOpCodes.MODEL_INVOKE)
            {
                codeTemplateOverride = MakeModelInvokeNodeCodeTemplate(node, nodeCodeGenerators, currentNames, functionNames);
                useTuplesForMultiOutputs = true;
            }
            else if (node.IsFunction || node.IsModelParamInitializer)
            {
                codeTemplateOverride = MakeCallFunctionCodeTemplate(node, nodeCodeGenerators, currentNames, functionNames);
                useTuplesForMultiOutputs = true;
            }


            var nodeDef = node.NodeDef;

            // Per-node code generation. The per-input inlining mechanism that this
            // function used to support is currently disabled (the standard MakeNode
            // path always nulls out InlineExpression below, so no upstream node ever
            // exposes an inline form). Build the code template with no inlines.
            var inlines = ImmutableDictionary<Node, string>.Empty;
            string mostInLinesCode = MakeNodeWithInlines(node, inlines, currentNames, codeTemplateOverride);
            ImmutableList<Node> inlineNodes = node.IsModelInput ? [node] : [];

            var inlineExpressionForNode = mostInLinesCode;
            var newVariables = new Dictionary<Variable, string>();
            NodeGenerationInfo nodeGenerator;
            if (nodeDef.OutputDefs.Count == 1)
            {
                if (nodeDef.OutputDefs[0].VariadicCountDef is null)
                {
                    var variableName = GetSanitizedVariableName(node.Outputs[0]!);
                    var assignmentPart = new CodeLine(0, "var " + variableName + " = " + inlineExpressionForNode + ";");

                    // Don't inline nodes for simplicity
                    inlineExpressionForNode = null;
                    
                    nodeGenerator = new NodeGenerationInfo(nodeDef, node, inlineExpressionForNode, [assignmentPart], inlineNodes);
                    newVariables[node.Outputs[0]!] = GetSanitizedVariableName(node.Outputs[0]!);
                }
                else if (useTuplesForMultiOutputs)
                {
                    var variableNames = node.Outputs.Select(x => GetSanitizedVariableName(x!)).ToList();

                    var codeLine = variableNames.Count == 1 ?
                        new CodeLine(0, $"var {variableNames[0]} = {inlineExpressionForNode};") :
                        new CodeLine(0, $"(var {string.Join(", var ", variableNames)}) = {inlineExpressionForNode};");

                    for (int outputIndex = 0; outputIndex < node.Outputs.Length; outputIndex++)
                        newVariables[node.Outputs[outputIndex]!] = GetSanitizedVariableName(node.Outputs[outputIndex]!);

                    // We don't support using a tuple as an inline expression, so we set it to null.
                    nodeGenerator = new NodeGenerationInfo(nodeDef, node, null, [codeLine], inlineNodes);
                }
                else
                {
                    Debug.Assert(node.Inputs.Length > 0 && node.Inputs[0] is not null);
                    var arrayVariableName = SanitizeVariableName(node.Inputs[0]!.UniqueName + "_split");
                    var codeLines = new List<CodeLine>();

                    codeLines.Add(new CodeLine(0, $"var {arrayVariableName} = {inlineExpressionForNode};"));

                    // Possible optimization: for long arrays, or when an array item is only used once,
                    // we may want to preffer to use indexing into the array as the variable name.

                    for (var i = 0; i < node.Outputs.Length; i++)
                        codeLines.Add(new CodeLine(0, $"var {GetSanitizedVariableName(node.Outputs[i]!)} = {arrayVariableName}[{i}];"));

                    for (var i = 0; i < node.Outputs.Length; i++)
                        newVariables[node.Outputs[i].AssertNotNull()] = GetSanitizedVariableName(node.Outputs[i]!);

                    // We don't support using an array as an inline expression, so we set it to null.
                    nodeGenerator = new NodeGenerationInfo(nodeDef, node, inlineExpression: null, codeLines.ToImmutableList(), inlineNodes);
                }
            }
            else
            {
                var variableNames = new List<string>();
                var lastOuputIsArray = nodeDef.OutputDefs.Count < node.Outputs.Length;
                Debug.Assert(!lastOuputIsArray);
                for (int outputIndex = 0; outputIndex < nodeDef.OutputDefs.Count; outputIndex++)
                    variableNames.Add(GetSanitizedVariableName(node.Outputs[outputIndex]!));

                Debug.Assert(variableNames.Count > 1);
                var codeLine = new CodeLine(0, $"(var {string.Join(", var ", variableNames)}) = {inlineExpressionForNode};");

                for (int outputIndex = 0; outputIndex < nodeDef.OutputDefs.Count; outputIndex++)
                    newVariables[node.Outputs[outputIndex]!] = GetSanitizedVariableName(node.Outputs[outputIndex]!);

                // We don't support using a tuple as an inline expression, so we set it to null.
                nodeGenerator = new NodeGenerationInfo(nodeDef, node, null, [codeLine], inlineNodes);
            }

            return (nodeGenerator, newVariables);
        }

        private
            (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables)?
            MakeConstantNode
            (Node node, ImmutableDictionary<Node, NodeGenerationInfo> currentNodeGenerators, ImmutableDictionary<Variable, string> currentNames)
        {
            var attributes = node.Attributes;
            if (attributes.IsDefaultValue(AttrValue))
                return null;

            var tensorDataAttribute = attributes.GetTensorVal(AttrValue).AssertNotNull();
            if (tensorDataAttribute.AccessRawMemory().Length > 500)
                return null;

            string dataParams;
            var dtype = tensorDataAttribute.DType;
            if (tensorDataAttribute.DType == DType.Float32)
            {
                var paramList = tensorDataAttribute.As<float32>().AccessMemory().ToArray().Select(x => $"{x}f");
                dataParams = string.Join(", ", paramList);
            }
            else if (tensorDataAttribute.DType == DType.Float64)
            {
                var paramList = tensorDataAttribute.As<float64>().AccessMemory().ToArray().Select(x => $"{x}d");
                dataParams = string.Join(", ", paramList);
            }
            else if (tensorDataAttribute.DType == DType.Int16)
            {
                var paramList = tensorDataAttribute.As<int16>().AccessMemory().ToArray().Select(x => $"{x}").ToList();
                var useCollectionExpression = (paramList.Count >= 4);
                if (!useCollectionExpression)
                    paramList = paramList.Select(x => $"(short){x}").ToList();

                dataParams = string.Join(", ", paramList);
                if (useCollectionExpression)
                    dataParams = "(short[])[" + dataParams + "]";
            }
            else if (tensorDataAttribute.DType == DType.Int32)
            {
                var paramList = tensorDataAttribute.As<int32>().AccessMemory().ToArray().Select(x => $"{x}");
                dataParams = string.Join(", ", paramList);
            }
            else if (tensorDataAttribute.DType == DType.Int64)
            {
                var paramList = tensorDataAttribute.As<int64>().AccessMemory().ToArray().Select(x => $"{x}L");
                dataParams = string.Join(", ", paramList);
            }
            else if (tensorDataAttribute.DType == DType.UInt16)
            {
                var paramList = tensorDataAttribute.As<uint16>().AccessMemory().ToArray().Select(x => $"{x}").ToList();
                var useCollectionExpression = (paramList.Count >= 4);
                if (!useCollectionExpression)
                    paramList = paramList.Select(x => $"(ushort){x}").ToList();

                dataParams = string.Join(", ", paramList);
                if (useCollectionExpression)
                    dataParams = "(ushort[])[" + dataParams + "]";
            }
            else if (tensorDataAttribute.DType == DType.UInt32)
            {
                var paramList = tensorDataAttribute.As<uint32>().AccessMemory().ToArray().Select(x => $"{x}").ToList();
                var useCollectionExpression = (paramList.Count >= 4);
                if (!useCollectionExpression)
                    paramList = paramList.Select(x => $"(uint){x}").ToList();

                dataParams = string.Join(", ", paramList);
                if (useCollectionExpression)
                    dataParams = "(uint[])[" + dataParams + "]";
            }
            else if (tensorDataAttribute.DType == DType.UInt64)
            {
                var paramList = tensorDataAttribute.As<uint64>().AccessMemory().ToArray().Select(x => $"{x}UL");
                dataParams = string.Join(", ", paramList);
            }
            else if (tensorDataAttribute.DType == DType.Bool)
            {
                var paramList = tensorDataAttribute.As<bit>().AccessMemory().ToArray().Select(x => $"{x.ToString().ToLower()}");
                dataParams = string.Join(", ", paramList);
            }
            else
            {
                return null;
            }

            string csharpExpression;
            var shape = tensorDataAttribute.Shape;
            if (shape.Dims.Length == 0)
                csharpExpression = $"Scalar({dataParams})";
            else if (shape.Dims.Length == 1)
                csharpExpression = dataParams == string.Empty ? $"EmptyVector<{dtype.ToIVarType().Name}>()" : $"Vector({dataParams})";
            else
                csharpExpression = $"Tensor([{string.Join(", ", shape.Dims)}], {dataParams})";

            var outputTensor = node.Outputs[0]!;
            var outputTensorName = GetSanitizedVariableName(outputTensor);
            var codeLine = new CodeLine(0, $"var {outputTensorName} = {csharpExpression};");

            // Don't inline constant nodes to keep the generated code simple
            var nodeGenerator = new NodeGenerationInfo(node.NodeDef, node, null, [codeLine], []);
            var newVariables = new Dictionary<Variable, string> { [outputTensor] = outputTensorName };
            return (nodeGenerator, newVariables);
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeLoopNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> currentNodeGenerators, ImmutableDictionary<Variable, string> currentNames, Dictionary<Variable, List<Node>> tensorChildNodes)
        {
            if (node.IsGraphOpenNode)
            {
                var openNode = node;
                Debug.Assert(openNode.OpName == OpCodes.LOOP_OPEN);
                var connectingTensor = openNode.ConnectingTensor.AssertNotNull();

                var closeNode = tensorChildNodes[connectingTensor][0];

                List<CodeLine> lines = new List<CodeLine>();
                List<Node> inlinedVariables = new();
                var newVariableNames = new Dictionary<Variable, string>();

                var allNames = currentNames.Values.ToHashSet();
                var ctxName = "ctx";
                var ctxIndex = 1;
                while (allNames.Contains(ctxName)) ctxName = "ctx" + ctxIndex++;

                var maxIteration = openNode.Inputs[0]!;
                var maxIterationCode = currentNames[maxIteration];

                // Let's hackily use the connecting tensor to carry the name of the ctx variable.
                newVariableNames[connectingTensor] = ctxName;

                // The iteration index comes from the ctx variable.
                newVariableNames[openNode.Outputs[0]!] = $"{ctxName}.IterationIndex";

                var numLoopVariables = openNode.Inputs.Length == 1 ? 0 : openNode.Inputs.Length - 2;
                List<string> toInitVariableNames = new();
                for (int loopVariableIndex = 0; loopVariableIndex < numLoopVariables; loopVariableIndex++)
                {
                    var outputLoopVariable = closeNode.Outputs[loopVariableIndex]!;
                    var loopVariableName = GetSanitizedVariableName(outputLoopVariable);
                    var initializerVariable = openNode.Inputs[loopVariableIndex + 2]!;

                    var initializerVariableCodeGenerator = currentNodeGenerators[initializerVariable.ParentNode.AssertNotNull()];
                    var initializerVariableCode = currentNames[initializerVariable];

                    if (outputLoopVariable.Rank is not null &&
                        initializerVariableCodeGenerator.Node.Outputs.Length == 1 && initializerVariableCodeGenerator.Node.Outputs[0]!.Structure() == DataStructure.Tensor &&
                        initializerVariableCodeGenerator.Node.Outputs[0]!.Rank is null)
                    {
                        if (outputLoopVariable.Rank == 0)
                            initializerVariableCode = $"({initializerVariableCode}).Scalar()";
                        else if (outputLoopVariable.Rank == 1)
                            initializerVariableCode = $"({initializerVariableCode}).Vec()";
                    }
                    else if (outputLoopVariable.Rank is null &&
                        initializerVariableCodeGenerator.Node.Outputs.Length == 1 && initializerVariableCodeGenerator.Node.Outputs[0]!.Structure() == DataStructure.Tensor &&
                        initializerVariableCodeGenerator.Node.Outputs[0]!.Rank is not null)
                    {
                        initializerVariableCode = $"((Tensor<{outputLoopVariable.DType.AssertNotNull().ToIVarType().Name}>)({initializerVariableCode}))";
                    }

                    lines.Add(new CodeLine(0, $"var {loopVariableName} = {initializerVariableCode};"));

                    var insideScopeLoopVariable = openNode.Outputs[loopVariableIndex + 2]!;
                    var afterLoopScopeLoopVariable = closeNode.Outputs[loopVariableIndex]!;
                    newVariableNames[insideScopeLoopVariable] = loopVariableName;
                    newVariableNames[afterLoopScopeLoopVariable] = loopVariableName;

                    var insideScopeChildNodes = tensorChildNodes.TryGetValue(insideScopeLoopVariable, out var cn) ? cn : [];
                    if (insideScopeChildNodes.Count == 0)
                        toInitVariableNames.Add(loopVariableName);
                }

                var numScanVariables = closeNode.Outputs.Length - numLoopVariables;
                for (int scanVariableIndex = 0; scanVariableIndex < numScanVariables; scanVariableIndex++)
                {
                    var scannedVariable = closeNode.Outputs[scanVariableIndex + numLoopVariables]!;
                    var scannedVariableName = GetSanitizedVariableName(scannedVariable);

                    var ivartypeName = scannedVariable.DType.AssertNotNull().ToIVarType().Name;
                    var rank = scannedVariable.Rank;
                    var rankName = rank == 0 ? "Scalar" : rank == 1 ? "Vector" : "Tensor";

                    lines.Add(new CodeLine(0, $"{rankName}<{ivartypeName}> {scannedVariableName} = null!;"));

                    newVariableNames[scannedVariable] = scannedVariableName;
                }

                lines.Add(new CodeLine(0, $"foreach(var {ctxName} in LoopAPI.Iterate({maxIterationCode}))"));
                lines.Add(new CodeLine(0, "{"));

                if (toInitVariableNames.Count > 0)
                    lines.Add(new CodeLine(1, $"LoopAPI.Init({string.Join(", ", toInitVariableNames)});"));

                var nodeCodeGenerator = new NodeGenerationInfo(openNode.NodeDef, openNode, null, lines.ToImmutableList(), inlinedVariables.ToImmutableList());

                return (nodeCodeGenerator, newVariableNames);
            }
            else
            {
                var closeNode = node;
                Debug.Assert(closeNode.OpName == OpCodes.LOOP_CLOSE);
                var connectingTensor = closeNode.ConnectingTensor.AssertNotNull();
                var openNode = connectingTensor.ParentNode.AssertNotNull();

                var ctxName = currentNames[connectingTensor];

                List<CodeLine> lines = new List<CodeLine>();
                List<Node> inlinedVariables = new();

                var numLoopVariables = openNode.Inputs.Length == 1 ? 0 : openNode.Inputs.Length - 2;
                var numScanVariables = closeNode.Outputs.Length - numLoopVariables;

                for (var loopVariableIndex = 0; loopVariableIndex < numLoopVariables; loopVariableIndex++)
                {
                    var openNodeOutputVariable = openNode.Outputs[loopVariableIndex + 2]!;
                    var loopVariableName = currentNames[openNodeOutputVariable];

                    var closeLoopInputVariable = closeNode.Inputs[loopVariableIndex + 1]!;
                    var closeLoopInputVariableCode = currentNames[closeLoopInputVariable];

                    lines.Add(new CodeLine(1, $"{loopVariableName} = {closeLoopInputVariableCode};"));
                }

                for (var scanVariableIndex = 0; scanVariableIndex < numScanVariables; scanVariableIndex++)
                {
                    var scannedVariable = closeNode.Outputs[numLoopVariables + scanVariableIndex]!;
                    var scannnedVariableName = currentNames[scannedVariable];

                    var scanVariable = closeNode.Inputs[scanVariableIndex + numLoopVariables + 1]!;
                    var scanVariableName = currentNames[scanVariable];

                    lines.Add(new CodeLine(1, $"{scannnedVariableName} = {ctxName}.Scan({scanVariableName});"));
                }

                if (closeNode.Inputs[0] is not null)
                {
                    var isConstantTrue = false;
                    var continueWhileVariable = closeNode.Inputs[0]!;
                    var parentNode = continueWhileVariable.ParentNode.AssertNotNull();
                    var continueWhileVariableCodeGenerator = currentNodeGenerators[parentNode];

                    if (parentNode.OpName == OpCodes.CONSTANT)
                    {
                        var constantAttributes = parentNode.Attributes;
                        if (!constantAttributes.IsDefaultValue(AttrValue))
                        {
                            var tensorDataAttribute = constantAttributes.GetTensorVal(AttrValue).AssertNotNull();
                            if (tensorDataAttribute.DType == DType.Bool && tensorDataAttribute.As<bit>().AccessMemory()[0] == true)
                            {
                                // LoopAPI.Iterate will automatically add the ctx.Break(Scalar(true)) when ctx.Break is not explicitly called.
                                // So we neither need the Break call, nor the creation of the Scalar(true) constant.
                                isConstantTrue = true;
                                inlinedVariables.Add(continueWhileVariableCodeGenerator.Node);
                            }
                        }
                    }

                    if (!isConstantTrue)
                    {
                        var continueWhileVariableCode = currentNames[continueWhileVariable];
                        lines.Add(new CodeLine(1, $"{ctxName}.ContinueWhile({continueWhileVariableCode});"));
                    }
                }

                lines.Add(new CodeLine(0, "}"));

                var nodeCodeGenerator = new NodeGenerationInfo(closeNode.NodeDef, closeNode, null, lines.ToImmutableList(), inlinedVariables.ToImmutableList());
                return (nodeCodeGenerator, []);
            }
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeIfNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames, Dictionary<Variable, List<Node>> tensorChildNodes)
        {
            var newVariables = new Dictionary<Variable, string>();
            if (node.IsGraphOpenNode)
            {
                // The graph open node produces no code.
                var openNode = node;
                Debug.Assert(openNode.OpName == OpCodes.IF_OPEN);
                var connectingTensor = openNode.ConnectingTensor.AssertNotNull();

                var closeNode = tensorChildNodes[connectingTensor][0];
                var nodeCodeGenerator = new NodeGenerationInfo(openNode.NodeDef, openNode, null, [], []);
                return (nodeCodeGenerator, []);
            }
            else
            {
                var closeNode = node;
                Debug.Assert(closeNode.OpName == OpCodes.IF_CLOSE);
                var connectingTensor = closeNode.ConnectingTensor.AssertNotNull();
                var openNode = connectingTensor.ParentNode.AssertNotNull();

                var numItems = closeNode.Outputs.Length;

                if (closeNode.Inputs.Length != numItems * 2)
                    throw new InvalidTensorOperationException(ErrorCodes.FW024, $"inputs:{closeNode.Inputs.Length}", "IfNode input validation",
                        $"Expected {numItems * 2} inputs but got {closeNode.Inputs.Length}");

                var condName = currentNames[openNode.Inputs[0]!];
                var whenTrueNames = closeNode.FullInputs[AttrThenBranch].Select(x => currentNames[x!]).ToArray();
                var whenFalseNames = closeNode.FullInputs[AttrElseBranch].Select(x => currentNames[x!]).ToArray();
                var outVariables = closeNode.Outputs;
                var outNames = closeNode.Outputs.Select(x => GetSanitizedVariableName(x!)).ToArray();

                List<CodeLine> lines = new List<CodeLine>();
                if (numItems > 8)
                {
                    var arrayResultName = SanitizeVariableName(closeNode.DefaultName) + "_arr";
                    lines.Add(new CodeLine(0, $"var {arrayResultName} = Ops.IfElse({condName}, [{string.Join(", ", whenTrueNames)}], [{string.Join(", ", whenFalseNames)}]);"));

                    for (var i = 0; i < numItems; i++)
                    {
                        var outTypeDef = GetTypeDefString(outVariables[i]!, outVariables[i]!.Rank);
                        lines.Add(new CodeLine(0, $"var {outNames[i]} = ({outTypeDef}){arrayResultName}[{i}];"));
                        newVariables[outVariables[i]!] = outNames[i];
                    }
                }
                else
                {
                    if (numItems == 1)
                        lines.Add(new CodeLine(0, $"var {outNames[0]} = Ops.IfElse({condName}, {whenTrueNames[0]}, {whenFalseNames[0]});"));
                    else // if (numItems <= 8)
                        lines.Add(new CodeLine(0, $"var ({string.Join(", ", outNames)}) = Ops.IfElse({condName}, ({string.Join(", ", whenTrueNames)}), ({string.Join(", ", whenFalseNames)}));"));

                    for (var i = 0; i < numItems; i++)
                        newVariables[outVariables[i]!] = outNames[i];
                }


                var nodeCodeGenerator = new NodeGenerationInfo(closeNode.NodeDef, closeNode, null, lines.ToImmutableList(), []);
                return (nodeCodeGenerator, newVariables);
            }
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeCreateModuleNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames, ImmutableDictionary<Function, string> functionNames)
        {
            Debug.Assert(node.OpName == InternalOpCodes.CREATE_MODULE);
            // Node outputs are Immutable* graph values; convert to the value-struct handle (an
            // `as Scalar<…>?` would be null since the runtime object is the immutable, not the struct).
            var moduleVariable = node.Outputs[0].AssertNotNull().ToValue<Scalar<IModuleVarType>>();

            var targetFunction = moduleVariable.ModuleFn.AssertNotNull();

            var functionName = functionNames[targetFunction];

            // No code lines required, the module is a globally available static property.
            var nodeCodeGenerator = new NodeGenerationInfo(node.NodeDef, node, null, [], []);
            return (nodeCodeGenerator, new Dictionary<Variable, string> { [node.Outputs[0].AssertNotNull()] = $"{functionName}Module" });
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeInputNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames)
        {
            Debug.Assert(node.OpName == InternalOpCodes.MODEL_TENSOR_INPUT ||
                         node.OpName == InternalOpCodes.MODEL_OPTIONAL_INPUT ||
                         node.OpName == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
                         node.OpName == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT);
            var nodeCodeGenerator = new NodeGenerationInfo(node.NodeDef, node, null, [], []);
            return (nodeCodeGenerator, new Dictionary<Variable, string> { [node.Outputs[0].AssertNotNull()] = GetSanitizedVariableName(node.Outputs[0]!) });
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeTensorStructGetFieldNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames)
        {
            Debug.Assert(node.OpName == InternalOpCodes.TENSOR_STRUCT_GETFIELD);

            var structInput = node.Inputs[0]!;
            var structInputName = currentNames[structInput];
            var fieldName = node.Attributes.GetStringVal(OnnxOpAttributeNames.ShrkAttrFieldName).AssertNotNull();

            // Emit a TENSOR_STRUCT_GETFIELD graph node directly. This mirrors the
            // canonical Module-level pattern (see e.g. TensorStructLoopCarry.Inline)
            // and works regardless of how the input struct variable was built — the
            // TensorStruct<T> instances returned by InternalOp.TensorStructCreate
            // don't have their in-memory _fields dictionary populated, so calling
            // the runtime `.GetField<T>(name)` method on them would throw.
            var outputVariable = node.Outputs[0].AssertNotNull();
            var fieldDType = outputVariable.Type;
            var fieldRank = outputVariable.Rank ?? 0;
            var fieldStructure = outputVariable.Structure();
            var fieldCastTypeDef = GetTypeDefString(outputVariable, outputVariable.Rank);
            var inlineExpression =
                $"({fieldCastTypeDef})Shorokoo.Core.Nodes.NodeDefinitions.InternalOp.TensorStructGetField(" +
                $"{structInputName}, \"{fieldName}\", " +
                $"Shorokoo.DType.{fieldDType}, {fieldRank}, " +
                $"Shorokoo.Core.Nodes.NodeDefinitions.DataStructure.{fieldStructure})";

            var outputTensor = node.Outputs[0]!;
            var outputTensorName = GetSanitizedVariableName(outputTensor);
            var codeLine = new CodeLine(0, $"var {outputTensorName} = {inlineExpression};");

            // Don't inline for simplicity
            var nodeCodeGenerator = new NodeGenerationInfo(node.NodeDef, node, null, [codeLine], []);
            return (nodeCodeGenerator, new Dictionary<Variable, string> { [outputTensor] = outputTensorName });
        }

        private (NodeGenerationInfo nodeGenerator, Dictionary<Variable, string> newVariables) MakeTensorStructCreateNode(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames)
        {
            Debug.Assert(node.OpName == InternalOpCodes.TENSOR_STRUCT_CREATE);

            var structDType = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype).AssertNotNull();
            var structDef = structDType.TensorStructDef.AssertNotNull();
            var structTypeName = structDef.TypeName;

            // Positional field references in TensorStructDef order — InternalOp.TensorStructCreate
            // takes an ordered Variable[] (no field-name labels) and the inputs to the graph node
            // are already in that order.
            var fieldRefs = node.Inputs.Select(input => currentNames[input!]).ToList();
            var fieldsArrayLiteral = $"new Shorokoo.Core.Variable[] {{ {string.Join(", ", fieldRefs)} }}";

            // Build the dtype expression. Prefer the static struct-type path
            // (StructDefExtractor.ExtractFromType<T>) when we have a simple, unqualified IStruct
            // type name — that matches the canonical Module-level pattern. For DTypeStruct or
            // fully-qualified names we can't safely emit a reflectable type literal, so fall
            // back to looking the def up by name on the global registry.
            string dtypeExpression;
            if (structTypeName != null && !structTypeName.Contains('.'))
            {
                dtypeExpression =
                    $"Shorokoo.DType.GetOrCreateForTensorStruct(" +
                    $"Shorokoo.Core.StructDefExtractor.ExtractFromType<{structTypeName}>())";
            }
            else
            {
                // Codegen for DTypeStruct / fully-qualified-name structs is not supported —
                // we don't have a reflectable IStruct type to emit. Generate a throw so the
                // compiled lambda fails loudly if anyone ever reaches this path.
                dtypeExpression =
                    $"throw new System.NotSupportedException(" +
                    $"\"TensorStructCreate codegen for dynamic/dotted TypeName '{structDef.TypeName}' is not supported\")";
            }

            var createExpression =
                $"Shorokoo.Core.Nodes.NodeDefinitions.InternalOp.TensorStructCreate({dtypeExpression}, {fieldsArrayLiteral})";

            var outputTensor = node.Outputs[0]!;
            var outputTensorName = GetSanitizedVariableName(outputTensor);
            var codeLine = new CodeLine(0, $"var {outputTensorName} = {createExpression};");

            // TensorStruct creation is typically not inlined due to complexity
            var nodeCodeGenerator = new NodeGenerationInfo(node.NodeDef, node, null, [codeLine], []);
            return (nodeCodeGenerator, new Dictionary<Variable, string> { [outputTensor] = outputTensorName });
        }

        private string MakeSetHyperparamsNodeCodeTemplate(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames, ImmutableDictionary<Function, string> functionNames)
        {
            Debug.Assert(node.OpName == InternalOpCodes.MODULE_SET_HYPERPARAMS);
            var moduleVariable = node.Inputs[0].AssertNotNull().ToValue<Scalar<IModuleVarType>>();
            var targetFunction = moduleVariable.ModuleFn.AssertNotNull();
            var hyperparamInputs = targetFunction.HyperparamInputs;

            // The param list start at number 3 (1-based, +1 for module variable, +1 for iteration indices vector)
            var paramsList = string.Join("", hyperparamInputs.Select((x, i) => $"{{{i + 3}:param}}"));


            if (targetFunction.FunctionType == FunctionType.ModuleSignature)
            {
                // Can't call the globally availabe model creation method on a module signature because it is effectively
                // a callback method. We do not know the actual function it refers to.

                // So we have to call SetHyperparams by hand and specify the Model type.
                var modelTypeDeclaration = GetModuleAwareTypeDefString(targetFunction, asModel: true);
                var codeTemplate = $"{{1:this}}.SetHyperparams<{modelTypeDeclaration}>({paramsList})";
                return codeTemplate;
            }
            else
            {
                // node.Inputs layout is [inputModule, iterationIndices, ...hyperparams],
                // so Length = 2 + hyperparams.Length. The codegen template below uses
                // {i + 3}:param for the variadic, which already accounts for both prefix
                // slots; this assertion just guards the layout invariant.
                Debug.Assert(node.Inputs.Length == hyperparamInputs.Length + 2);

                var functionName = functionNames[targetFunction];

                var codeTemplate = $"{functionName}Model({paramsList})";
                return codeTemplate;
            }
        }

        private string MakeModelInvokeNodeCodeTemplate(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames, ImmutableDictionary<Function, string> functionNames)
        {
            Debug.Assert(node.OpName == InternalOpCodes.MODEL_INVOKE);
            var modelVariable = node.Inputs[0].AssertNotNull().ToValue<Scalar<IModelVarType>>();
            var targetFunction = modelVariable.ModuleFn.AssertNotNull();
            var modelInputs = targetFunction.NonHyperparamInputs;
            var paramsList = string.Join("", modelInputs.Select((x, i) => $"{{{i + 2}:param}}"));
            var functionName = functionNames[targetFunction];

            // Model<TInput1, ..., TInput8, TOutputs> generic tops out at 8 non-hyperparam
            // inputs, so a Module with more than 8 inputs can't be invoked through the
            // standard pipeline. The bare `Call(p1, p2, ...)` form is always valid here.
            return $"{{1:this}}.Call({paramsList})";
        }

        private string MakeCustomCodeTemplate(Node node, ImmutableDictionary<Node, string> inlineInputs, ImmutableDictionary<Variable, string> currentNames)
        {
            var outputTensors = node.Outputs;
            var nodeDef = node.NodeDef;
            var methodCore = "";
            if (nodeDef.OutputDefs.Count != node.Outputs.Length)
            {
                if (nodeDef.OutputDefs.Count != 1 && node.Outputs.Length > 0)
                    throw new UnsupportedDTypeException(ErrorCodes.FW025, "variadic outputs", "custom node", 
                        "Custom nodes with variadic outputs that also has other outputs is not supported");

                if (node.Outputs.Length == 0)
                    throw new UnsupportedDTypeException(ErrorCodes.FW026, "no outputs", "custom node", 
                        "Custom nodes with no outputs is not supported");


                var typeDefString = GetTypeDefString(node.Outputs[0].NotNull(), node.Outputs[0]!.TensorDims?.Length);
                methodCore = $"CallCustomOperatorArrayOut<{typeDefString}>";
            }
            else
            {
                var typeDefs = string.Join(", ", node.Outputs.Select(x => GetTypeDefString(x.NotNull(), x!.TensorDims?.Length)));
                methodCore = $"CallCustomOperator<{typeDefs}>";
            }

            var variableInputPlaceHolders = string.Join(", ", Enumerable.Range(1, node.Inputs.Length).Select(x => "{" + x + ":}"));
            var attributeInputPlaceHolders = string.Join(", ", Enumerable.Range(0, nodeDef.AttributeDefs.Count)
                    .Select(x => '"' + nodeDef.AttributeDefs[x].AttributeName + '"' + ", " + "{" + (char)('a' + (char)x) + ":}"));

            var methodParams = $"(\"{nodeDef.OpName}\", [{variableInputPlaceHolders}], [{attributeInputPlaceHolders}])";
            return methodCore + methodParams;
        }

        private string MakeCallFunctionCodeTemplate(Node node, ImmutableDictionary<Node, NodeGenerationInfo> nodeCodeGenerators, ImmutableDictionary<Variable, string> currentNames, ImmutableDictionary<Function, string> functionNames)
        {
            var methodName = functionNames[(node.TargetFunction).AssertNotNull()];
            var offset =  node.OpName == InternalOpCodes.TRAINABLE_PARAM_REF ? 2
                        : node.OpName == InternalOpCodes.TRAINABLE_PARAM_ID_REF ? 3
                        : node.OpName == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF ? 3
                        : 1;

            var paramsList = string.Join("", node.Inputs.Skip(offset-1).Select((x, i) => $"{{{i + offset}:param}}"));
            return $"{methodName}({paramsList})";
        }
        

        private static string EscapeString(string input)
        {
            return input.Replace("\\", "\\\\")
                        .Replace("\"", "\\\"")
                        .Replace("\n", "\\n")
                        .Replace("\r", "\\r")
                        .Replace("\t", "\\t");
        }

        private static DType inferDTypeForDef(NodeDefTypeDef typeDef, Node node)
        {
            var nodeDef = node.NodeDef;
            Debug.Assert(nodeDef.TypeDefs.ContainsValue(typeDef));
            var inputs = node.Inputs;
            for (int i = 0; i < nodeDef.InputDefs.Count; i++)
            {
                if (nodeDef.InputDefs[i].TypeDef == typeDef && i < inputs.Length && inputs[i] is not null)
                    return inputs[i].AssertNotNull().Type;
            }

            var outputs = node.Outputs;
            for (int i = 0; i < nodeDef.OutputDefs.Count; i++)
            {
                if (nodeDef.OutputDefs[i].TypeDef == typeDef && i < outputs.Length && outputs[i] is not null)
                    return outputs[i].AssertNotNull().Type;
            }

            throw new InvalidTensorOperationException(ErrorCodes.FW027, typeDef.TypeDefName, "DType inference", 
                "Failed to infer DType for node definition type definition");
        }

        private string MakeNodeWithInlines(Node node, ImmutableDictionary<Node, string> inlineInputs, ImmutableDictionary<Variable, string> currentNames, string? codeTemplateOverride)
        {
            var nodeDef = node.NodeDef;
            var codeTemplate = codeTemplateOverride ?? node.NodeDef.CodeTemplate ?? MakeCustomCodeTemplate(node, inlineInputs, currentNames);
            var attributeDefs = node.NodeDef.AttributeDefs;
            var attributes = node.Attributes;
            var result = codeTemplate;

            foreach (var typeDef in nodeDef.TypeDefs.Values)
            {
                while (true)
                {
                    var placeholder = $"{{{typeDef.TypeDefName}:";
                    var startIdx = result.IndexOf(placeholder);
                    if (startIdx == -1) break;

                    var endIdx = result.IndexOf("}", startIdx);
                    var fullPlaceholder = result[startIdx..(endIdx + 1)];
                    var keyword = result[(startIdx + placeholder.Length)..endIdx];

                    string replacement;
                    if (keyword == "ivartype")
                        replacement = inferDTypeForDef(typeDef, node).ToIVarType().Name;
                    else
                        throw new UnsupportedDTypeException(ErrorCodes.FW028, keyword, "code template",
                            $"Code template keyword '{keyword}' is not implemented");

                    result = result.Replace(fullPlaceholder, replacement);
                }
            }

            var isVariadic = false;
            if (nodeDef.InputDefs.Any(x => x.VariadicCountDef is not null))
            {
                isVariadic = true;
                int placeholderInputNumber = 1;
                for (; placeholderInputNumber <= node.Inputs.Length; placeholderInputNumber++)
                {
                    if (!result.Contains($"{{{placeholderInputNumber + 1}:"))
                        break;
                }
                // Also skip inputs already referenced by {N:this} patterns
                while (placeholderInputNumber <= node.Inputs.Length && result.Contains($"{{{placeholderInputNumber}:this}}"))
                    placeholderInputNumber++;

                var placeholder = "{#:";
                var startIdx = result.IndexOf(placeholder);
                if (startIdx != -1)
                {
                    var endIdx = result.IndexOf("}", startIdx);
                    var fullPlaceholder = result[startIdx..(endIdx + 1)];
                    var keyword = result[(startIdx + placeholder.Length)..endIdx];

                    var generatedPlaceholders = (placeholderInputNumber > node.Inputs.Length) ? "" :
                                        string.Join("", Enumerable.Range(
                                            placeholderInputNumber, node.Inputs.Length - placeholderInputNumber + 1)
                                            .Select(x => $"{{{x}:{keyword}}}"));

                    result = result.Replace(fullPlaceholder, generatedPlaceholders);
                }
            }

            // Replace input placeholders ({1:keyword}, {2:keyword}, etc.)
            var inputs = isVariadic ? node.Inputs :
                                nodeDef.InputDefs.Select((x, i) => i < node.Inputs.Length ? node.Inputs[i] : null).ToImmutableArray();
            for (int i = 0; i < inputs.Length; i++)
            {
                while (true)
                {
                    var input = inputs[i];
                    var placeholder = $"{{{i + 1}:";
                    var startIdx = result.IndexOf(placeholder);
                    if (startIdx == -1) break;

                    var endIdx = result.IndexOf("}", startIdx);
                    var fullPlaceholder = result[startIdx..(endIdx + 1)];
                    var keyword = result[(startIdx + placeholder.Length)..endIdx];

                    string replacement;
                    if (input is null)
                        replacement = "null, ";
                    else
                    {
                        // Per-input inlining is disabled (see MakeNode), so the
                        // low_op / high_op precedence-paren wrap that fires only
                        // for inlined inputs never fires either.
                        var expression = currentNames[input];
                        replacement = expression;
                        if (keyword.StartsWith("param"))
                            replacement += ", ";
                    }

                    result = result.Replace(fullPlaceholder, replacement);
                }
            }


            // Replace attribute placeholders ({a:keyword}, {b:keyword}, etc.)
            for (int i = 0; i < attributeDefs.Count; i++)
            {
                while (true)
                {
                    var attrLetter = (char)('a' + i);
                    var placeholder = $"{{{attrLetter}:";
                    var startIdx = result.IndexOf(placeholder);
                    if (startIdx == -1) break;

                    var endIdx = result.IndexOf("}", startIdx);
                    var fullPlaceholder = result[startIdx..(endIdx + 1)];
                    var keyword = result[(startIdx + placeholder.Length)..endIdx];

                    var attrDef = attributeDefs[i];
                    var attrName = attributeDefs[i].AttributeName;
                    var attrType = attributeDefs[i].Type;
                    string attrValue = "";
                    if (attributes.IsDefaultValue(attrName))
                        attrValue = "null";
                    else if (attrType is AttributeType.Long)
                        attrValue = attributes.GetLongVal(attrName).ToString().AssertNotNull() + "L";
                    else if (attrType is AttributeType.Float)
                        attrValue = attributes.GetFloatVal(attrName).ToString().AssertNotNull() + "f";
                    else if (attrType is AttributeType.Bool)
                        attrValue = attributes.GetBoolVal(attrName).AssertNotNull() ? "true" : "false";
                    else if (attrType is AttributeType.String)
                        attrValue = '"' + EscapeString(attributes.GetStringVal(attrName).AssertNotNull()) + '"';
                    else if (attrType is AttributeType.Tensor)
                    {
                        var tensor = attributes.GetTensorVal(attrName).AssertNotNull();
                        var base64Data = Convert.ToBase64String(tensor.AccessRawMemory());
                        var dimsCSharp = $"[{String.Join(", ", tensor.Shape.Dims.Select(x => $"{x}L"))}]";
                        if (keyword == "base64string")
                            attrValue = base64Data;
                        else if (keyword == "dims")
                            attrValue = dimsCSharp;
                        else
                        {
                            attrValue = $"Shorokoo.Globals.TensorData(DType.{tensor.DType}, {dimsCSharp}, \"{base64Data}\")";
                        }
                    }
                    else if (attrType is AttributeType.Enum)
                    {
                        var enumDef = attrDef.EnumDef.AssertNotNull();
                        var enumVal = attributes.GetEnumVal(attrName).AssertNotNull();
                        attrValue = enumDef.ToCSharpFullName(enumVal);
                    }
                    else if (attrType is AttributeType.Enums)
                    {
                        var enumDef = attrDef.EnumDef.AssertNotNull();
                        var enumsVal = attributes.GetEnumsVal(attrName).AssertNotNull();
                        var attrValues = enumsVal.Select(enumVal => enumDef.ToCSharpFullName(enumVal)).ToArray();
                        attrValue = $"{string.Join(", ", attrValues)}";
                    }
                    else if (attrType is AttributeType.Longs)
                    {
                        var attrValues = attributes.GetLongsVal(attrName).AssertNotNull();
                        attrValue = $"{string.Join(", ", attrValues.Select(x => $"{x}L"))}";
                        if (keyword != "params")
                            attrValue = $"[{attrValue}]";
                    }
                    else if (attrType is AttributeType.Floats)
                    {
                        var attrValues = attributes.GetFloatsVal(attrName).AssertNotNull();
                        attrValue = $"{string.Join(", ", attrValues.Select(x => $"{x}f"))}";
                        if (keyword != "params")
                            attrValue = $"[{attrValue}]";
                    }

                    if (keyword.StartsWith("param"))
                        attrValue += ", ";

                    result = result.Replace(fullPlaceholder, attrValue);
                }
            }

            // Replace output placeholders ({o1:keyword}, {o2:keyword}, etc.)
            for (int i = 0; i < node.Outputs.Length; i++)
            {
                while (true)
                {
                    var output = node.Outputs[i];
                    var placeholder = $"{{o{i + 1}:";
                    var startIdx = result.IndexOf(placeholder);
                    if (startIdx == -1) break;

                    var endIdx = result.IndexOf("}", startIdx);
                    var fullPlaceholder = result[startIdx..(endIdx + 1)];
                    var keyword = result[(startIdx + placeholder.Length)..endIdx];

                    if (keyword == "torank")
                    {
                        if (output is not { Kind: DataStructure.Tensor } tensor)
                        {
                            result = result.Replace(fullPlaceholder, "");
                        }
                        else
                        {
                            // tensor.Rank is the statically known rank; fall back to the node's
                            // InternalAttrRank attribute for Identity nodes that don't carry one.
                            var rank = tensor.Rank;
                            if (rank is null && attributes.IsAttributeDefined(InternalAttrRank) && !attributes.IsDefaultValue(InternalAttrRank))
                            {
                                rank = (int?)attributes.GetLongVal(InternalAttrRank);
                            }
                            
                            // For SEQUENCE_AT nodes, try to infer rank from the sequence's element rank
                            if (rank is null && node.OpName == OpCodes.SEQUENCE_AT)
                            {
                                rank = InferSequenceElementRank(node);
                            }
                            
                            if (rank == 0)
                                result = result.Replace(fullPlaceholder, ".Scalar()");
                            else if (rank == 1)
                                result = result.Replace(fullPlaceholder, ".Vec()");
                            else
                                result = result.Replace(fullPlaceholder, "");
                        }
                    }
                    else if (keyword == "fromvar")
                    {
                        if (output is not { Kind: DataStructure.Tensor } tensor)
                            result = result.Replace(fullPlaceholder, "");
                        else
                        {
                            var dtypename = tensor.Type.ToIVarType().Name;

                            if (tensor.Rank == 0)
                                result = result.Replace(fullPlaceholder, $".{dtypename}().Scalar()");
                            else if (tensor.Rank == 1)
                                result = result.Replace(fullPlaceholder, $".{dtypename}().Vec()");
                            else
                                result = result.Replace(fullPlaceholder, $".{dtypename}().Tensor()");
                        }
                    }
                    // Note: an "objtype" keyword arm was here too, but no
                    // node-definition CodeTemplate in the codebase uses
                    // {oN:objtype}, so it was unreachable and removed.
                }
            }

            // replace num outputs placeholders
            while (true)
            {
                var numOutputs = node.Outputs.Length;
                var placeholder = $"{{numoutputs:";
                var startIdx = result.IndexOf(placeholder);
                if (startIdx == -1) break;

                var endIdx = result.IndexOf("}", startIdx);
                var fullPlaceholder = result[startIdx..(endIdx + 1)];
                var keyword = result[(startIdx + placeholder.Length)..endIdx];

                var attrValue = numOutputs.ToString().AssertNotNull() + "L";
                if (keyword.StartsWith("param"))
                    attrValue += ", ";

                result = result.Replace(fullPlaceholder, attrValue);
            }


            var trailingNulls = ")";
            var maxNullTrimCount = codeTemplate.Split(":param?}").Length - 1;
            for (var trailingNullCount = 0; trailingNullCount < maxNullTrimCount; trailingNullCount++)
            {
                if (!result.Contains("null, " + trailingNulls))
                    break;

                trailingNulls = "null, " + trailingNulls;
            }

            result = result.Replace(trailingNulls, ")");
            if (result.Contains(", )"))
                result = result.Replace(", )", ")");

            if (result.Contains(", ]"))
                result = result.Replace(", ]", "]");

            return result;

            // Code template can look like one of these:
            // "{1:low_op} & {2:low_op}"
            // "{1:this}.AveragePool({e:param}{b:param?}{c:param?}{d:param?}{f:param?}{g:param?})"
            // "NN.AffineGrid({1:param}{2:param}{A:param?})"

            // The various template kinds are:
            // low_op: low priority operator, always surround in brackets when inlining inside a high_op operator. e.g. (a + b) * c.
            // high_op: high priority operator, only surround in brackets when inlining as the right hand side of any other operator. e.g. a * (b / c)
            // param: a parameter in a function:
            //   - when null (as determined from Variable.IsNull property), use "null, ", or just "null" if last parameter.
            //   - when not inlined and not null, use the Name from it's VariableGenerationInfo
            //   - when inlined and not null use, use the CSharpExpression from it's VariableGenerationInfo
            //
            // param?: Same as param, however if it an all subsequent param? are null, then use "" (omit the parameter entirely).
            // this: The "this" object in an object member invocation. It should never be null, it should never be inlined.
            //   - Use the Name from it's VariableGenerationInfo
            //
            // {#:keyword}: The # is a one based index that specifies which input from node.Inputs to use.
            //              So, e.g. {2:param} means to use node.Inputs[2-1], and the info can be found in currentNames[node.Inputs[2-1]]
            //
            // {x:keyword}: The x is a and alphabetic index that specifies which attribute value from attributeDefs to use.
            //              So, e.g. {e:param} can be obtained using:
            //              attributes.GetLongVal(attributeDefs[(int)('e' - 'a')].AttributeName)
            //              Note that just as with inputs, attributes can be null. GetLongVal, returns a int?
        }
        
        /// <summary>
        /// Infer the element rank from a SEQUENCE_AT node by tracing back to how the sequence was constructed.
        /// </summary>
        private static int? InferSequenceElementRank(Node sequenceAtNode)
        {
            Debug.Assert(sequenceAtNode.OpName == OpCodes.SEQUENCE_AT);
            
            // Get the input sequence tensor
            var sequenceInput = sequenceAtNode.Inputs[0];
            if (sequenceInput is null)
                return null;
                
            // Trace back through the sequence construction to find element ranks
            return InferSequenceElementRankFromTensor(sequenceInput);
        }
        
        /// <summary>
        /// Recursively traces back through sequence construction operations to infer element rank.
        /// Examines SEQUENCE_INSERT, SEQUENCE_CONSTRUCT, SEQUENCE_ERASE, and IDENTITY nodes
        /// to find the rank of elements that were originally added to the sequence.
        /// </summary>
        /// <param name="sequenceTensor">The sequence tensor to trace back from.</param>
        /// <returns>The inferred element rank, or null if it cannot be determined.</returns>
        private static int? InferSequenceElementRankFromTensor(Variable sequenceTensor)
        {
            if (sequenceTensor is null)
                return null;
                
            var parentNode = sequenceTensor.ParentNode;
            if (parentNode is null)
                return null;
                
            if (parentNode.OpName == OpCodes.SEQUENCE_EMPTY)
            {
                // Empty sequence - no elements to infer from
                return null;
            }
            else if (parentNode.OpName == OpCodes.SEQUENCE_INSERT)
            {
                // Look at the element being inserted
                var elementInput = parentNode.Inputs[1];
                if (elementInput is not null)
                {
                    var elementRank = elementInput.Rank;
                    if (elementRank is not null)
                        return elementRank;
                }
                
                // Also check the input sequence (for chain of inserts)
                var inputSequence = parentNode.Inputs[0];
                if (inputSequence is not null)
                {
                    var chainedRank = InferSequenceElementRankFromTensor(inputSequence);
                    if (chainedRank is not null)
                        return chainedRank;
                }
                
                return null;
            }
            else if (parentNode.OpName == OpCodes.SEQUENCE_CONSTRUCT)
            {
                // Look at the first element
                foreach (var input in parentNode.Inputs)
                {
                    if (input is not null)
                    {
                        var elementRank = input.Rank;
                        if (elementRank is not null)
                            return elementRank;
                    }
                }
                return null;
            }
            else if (parentNode.OpName == OpCodes.SEQUENCE_ERASE)
            {
                // Look at the input sequence
                var inputSequence = parentNode.Inputs[0];
                if (inputSequence is not null)
                    return InferSequenceElementRankFromTensor(inputSequence);
                return null;
            }
            else if (parentNode.OpName == OpCodes.IDENTITY)
            {
                // Pass through identity
                var inputSequence = parentNode.Inputs[0];
                if (inputSequence is not null)
                    return InferSequenceElementRankFromTensor(inputSequence);
                return null;
            }
            
            return null;
        }
    }
}
