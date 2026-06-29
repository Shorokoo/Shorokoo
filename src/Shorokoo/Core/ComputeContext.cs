using System;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;

namespace Shorokoo.Runtime
{

    /// <summary>
    /// A compiled computation graph backed by a Shorokoo inference session.
    /// Created once via <see cref="ComputeContext.Compile"/>, then invoked repeatedly
    /// via <see cref="Execute"/> — each call only feeds new data, with zero graph
    /// rebuilding or session creation overhead.
    /// </summary>
    public class CompiledGraph
    {
        private readonly IShorokooInferenceSession _session;
        private readonly Dictionary<string, string> _onnxInputNameByOriginal;
        private readonly string[] _originalInputNames;

        internal CompiledGraph(
            IShorokooInferenceSession session,
            Dictionary<string, string> onnxInputNameByOriginal,
            string[] originalInputNames)
        {
            _session = session;
            _onnxInputNameByOriginal = onnxInputNameByOriginal;
            _originalInputNames = originalInputNames;
        }

        /// <summary>
        /// Executes the compiled graph with the given inputs.
        /// TensorDataStruct inputs are automatically expanded into individual fields.
        /// </summary>
        public NamedModelParam[] Execute(params IData[] inputs)
        {
            var expandedInputs = ComputeContext.ExpandStructInputs(inputs);

            if (expandedInputs.Length != _originalInputNames.Length)
            {
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "CompiledGraph.Execute",
                    $"inputs.Length={expandedInputs.Length}, graph.Inputs.Length={_originalInputNames.Length}",
                    "Input length mismatch: number of provided inputs does not match the graph's expected input tensor count");
            }

            var namedInputs = expandedInputs.Zip(_originalInputNames)
                .Select(zip => NamedModelParam.FromIData(zip.Second, ModelParamType.InputParam, zip.First))
                .ToArray();

            return Run(namedInputs);
        }

        /// <summary>
        /// Executes the compiled graph with pre-built named inputs.
        /// </summary>
        public NamedModelParam[] Run(params NamedModelParam[] inputs)
        {
            var sessionInputs = new Dictionary<string, IShorokooTensorValue>();
            foreach (var input in inputs)
            {
                var onnxName = _onnxInputNameByOriginal.TryGetValue(input.ParamName, out var mapped)
                    ? mapped : input.ParamName;
                sessionInputs[onnxName] = input.ToTensorValue();
            }

            var results = _session.Run(sessionInputs, _session.OutputNames);

            return results.Zip(_session.OutputNames)
                .Select(x => OnnxUtils.CreateNamedModelParam(x.First, ModelParamType.OutputParam, x.Second))
                .ToArray();
        }
    }

    /// <summary>
    /// Provides a runtime to execute the operations described by a VirtualGraph.
    ///
    /// Reusing TensorData Outputs from one ExecutationContext to another ComputeContext
    /// will work but may incur a performance penalty as the data is shifted between the two contexts.
    /// </summary>
    public class ComputeContext
    {
        private static ComputeContext? _defaultComputeContext;

        /// <summary>
        /// Process-wide default context, created lazily on first access and used wherever no
        /// explicit context is supplied. Settable to swap in a custom context.
        /// </summary>
        public static ComputeContext Default
        {
            get
            {
                if (_defaultComputeContext == null)
                    _defaultComputeContext = new ComputeContext();

                return _defaultComputeContext;
            }

            set { _defaultComputeContext = value; }
        }

        /// <summary>Creates a compute context backed by the default inference-session factory.</summary>
        public ComputeContext()
        {
        }

        /// <summary>
        /// Compiles the graph into a reusable <see cref="CompiledGraph"/>: the ONNX model and
        /// inference session are built once, so repeated executions only feed new data.
        /// </summary>
        public CompiledGraph Compile(FastComputationGraph graph)
        {
            var originalInputNames = ResolveOriginalInputNames(graph);
            return CompileFromModel(
                () => FastOnnxModelBuilder.BuildOnnxModel(graph, prepForOnnx: true),
                originalInputNames);
        }

        private CompiledGraph CompileFromModel(Func<ModelProto> buildModel, string[] originalInputNames)
        {
            var model = buildModel();

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            var session = CreateSession(modelData,
                HasOptionalOps(model.Graph) || HasTrainingModeDropout(model.Graph));

            var onnxInputNameByOriginal = new Dictionary<string, string>();
            for (int i = 0; i < originalInputNames.Length && i < session.InputNames.Count; i++)
                onnxInputNameByOriginal[originalInputNames[i]] = session.InputNames[i];

            return new CompiledGraph(session, onnxInputNameByOriginal, originalInputNames);
        }

        private static string[] ResolveOriginalInputNames(FastComputationGraph graph)
        {
            var names = new string[graph.Inputs.Count];
            for (int i = 0; i < graph.Inputs.Count; i++)
                names[i] = graph.InputUniqueNames.Count > i && graph.InputUniqueNames[i] is string n
                    ? n
                    : graph.Inputs[i].ToString();
            return names;
        }

        /// <summary>
        /// Evaluates the given output variables by building and executing a zero-input graph,
        /// returning their concrete tensor data.
        /// </summary>
        public TensorData[] Eval(Variable[] outputs)
        {
            var graph = new FastComputationGraph([], [.. outputs]);
            var results = this.Execute(graph).Select(x => x.ToTensorData()).ToArray();

            return results;
        }

        /// <summary>Params convenience over <see cref="Eval(Variable[])"/> for two or more outputs.</summary>
        public TensorData[] Eval(Variable output1, Variable output2, params Variable[] outputs)
        {
            var allOutputs = new[] { output1, output2 }.Concat(outputs).ToArray();
            return Eval(allOutputs);
        }

        /// <summary>Evaluates a single output variable.</summary>
        public TensorData Eval(Variable output)
        {
            var allOutputs = new[] { output };
            return Eval(allOutputs)[0];
        }

        /// <summary>Evaluates a single typed tensor, returning element-typed <see cref="TensorData{T}"/>.</summary>
        public TensorData<T> Eval<T>(Tensor<T> output)
            where T : IVarType
        {
            return (TensorData<T>)Eval((Variable)output);
        }

        /// <summary>Executes a graph that takes no inputs.</summary>
        public NamedModelParam[] Execute(FastComputationGraph graph) => this.Execute(graph, []);

        /// <summary>
        /// Executes the graph, pairing the inputs positionally with the graph's inputs.
        /// TensorDataStruct inputs are automatically expanded into individual fields.
        /// </summary>
        public NamedModelParam[] Execute(FastComputationGraph graph, params IData[] inputs)
        {
            var expandedInputs = ExpandStructInputs(inputs);

            if (expandedInputs.Length != graph.Inputs.Count)
            {
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "Execute", $"inputs.Length={expandedInputs.Length}, graph.InputTensors.Count={graph.Inputs.Count}",
                    "Input length mismatch: number of provided inputs does not match the graph's expected input tensor count");
            }

            var originalInputNames = ResolveOriginalInputNames(graph);

            var namedInputs = expandedInputs.Zip(originalInputNames)
                .Select((zip) => NamedModelParam.FromIData(zip.Second, ModelParamType.InputParam, zip.First))
                .ToArray();

            return Run(graph, namedInputs);
        }

        /// <summary>
        /// Expands TensorDataStruct inputs into individual field data entries.
        /// </summary>
        internal static IData[] ExpandStructInputs(IData[] inputs)
        {
            var expandedInputs = new List<IData>();
            foreach (var input in inputs)
            {
                if (input is TensorDataStruct structData)
                {
                    foreach (var field in structData.Definition.Fields)
                    {
                        if (!structData.Fields.TryGetValue(field.Name, out var fieldData))
                        {
                            throw new InvalidTensorOperationException(ErrorCodes.CR006, "Execute",
                                $"field={field.Name}, struct={structData.Definition.TypeName ?? "anonymous"}",
                                $"TensorDataStruct is missing data for field '{field.Name}'");
                        }
                        expandedInputs.Add(fieldData);
                    }
                }
                else
                {
                    expandedInputs.Add(input);
                }
            }
            return expandedInputs.ToArray();
        }

        /// <summary>
        /// Executes the graph with pre-built named inputs. Builds the ONNX model and a fresh
        /// inference session per call (disposed afterwards); use <see cref="Compile"/> for repeated runs.
        /// </summary>
        public NamedModelParam[] Run(FastComputationGraph graph, params NamedModelParam[] inputs)
        {
            var originalInputNames = ResolveOriginalInputNames(graph);
            return RunFromModel(
                () => FastOnnxModelBuilder.BuildOnnxModel(graph, prepForOnnx: true),
                originalInputNames,
                inputs);
        }

        private NamedModelParam[] RunFromModel(Func<ModelProto> buildModel, string[] originalInputNames, NamedModelParam[] inputs)
        {
            var model = buildModel();

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            var session = CreateSession(modelData,
                HasOptionalOps(model.Graph) || HasTrainingModeDropout(model.Graph));

            var onnxInputNameByOriginal = new Dictionary<string, string>();
            for (int i = 0; i < originalInputNames.Length && i < session.InputNames.Count; i++)
                onnxInputNameByOriginal[originalInputNames[i]] = session.InputNames[i];

            var sessionInputs = new Dictionary<string, IShorokooTensorValue>();
            foreach (var input in inputs)
            {
                var onnxName = onnxInputNameByOriginal.TryGetValue(input.ParamName, out var mapped)
                    ? mapped : input.ParamName;
                sessionInputs[onnxName] = input.ToTensorValue();
            }
            var results = session.Run(sessionInputs, session.OutputNames);

            var retVal = results.Zip(session.OutputNames).Select(x =>
                        OnnxUtils.CreateNamedModelParam(x.First, ModelParamType.OutputParam, x.Second))
                        .ToArray();

            // Dispose the session to free native memory. The returned tensor values
            // own their memory independently of the session.
            session.Dispose();

            return retVal;
        }

        private IShorokooInferenceSession CreateSession(byte[] modelData, bool disableOptimizations = false)
        {
            // ORT's constant-folding pass calls GetDeleteFunc on Optional values,
            // which OptionalTypeBase doesn't implement -- session init throws
            // "GetDeleteFunc is not implemented". Disabling optimizations skips the
            // fold pass; Optional ops then go through the normal execution path,
            // which ORT handles correctly.
            var optLevel = disableOptimizations
                ? ShorokooGraphOptimization.DisableAll
                : ShorokooGraphOptimization.EnableAll;
            return InferenceBackend.Factory.CreateSession(
                modelData,
                optLevel,
                ShorokooLogSeverity.Fatal);
        }

        private static bool HasOptionalOps(GraphProto graph)
        {
            foreach (var node in graph.Nodes)
            {
                if (node.OpType.StartsWith("Optional", StringComparison.Ordinal)) return true;
                foreach (var attr in node.Attributes)
                {
                    if (attr.G is not null && HasOptionalOps(attr.G)) return true;
                    foreach (var sub in attr.Graphs)
                        if (HasOptionalOps(sub)) return true;
                }
            }
            return false;
        }

        /// <summary>
        /// True if the graph (or any subgraph) contains a Dropout node with a connected
        /// training_mode input (the 3rd input, non-empty). ONNX Runtime's constant-folding
        /// pass evaluates a Dropout whose data input is provably constant at session-load
        /// time as the *inference-mode identity* — it does NOT honor the random training-mode
        /// draw — and bakes an all-true (no-drop) mask. That silently turns any module whose
        /// mask is built on a constant ones tensor (SpatialDropout's `x * Dropout(ones,…)`,
        /// the AlphaDropout family) into a no-op. Disabling optimizations skips the fold pass
        /// so the real (seeded, random) Dropout kernel runs — exactly the
        /// <see cref="HasOptionalOps"/> precedent above. The check is an over-approximation:
        /// it also disables opts for *eval*-mode Dropout (which wires a constant-false
        /// training flag), which is merely a rare perf cost, never a correctness change.
        /// </summary>
        private static bool HasTrainingModeDropout(GraphProto graph)
        {
            foreach (var node in graph.Nodes)
            {
                if (node.OpType == OpCodes.DROPOUT
                    && node.Inputs.Count > 2
                    && !string.IsNullOrEmpty(node.Inputs[2]))
                    return true;
                foreach (var attr in node.Attributes)
                {
                    if (attr.G is not null && HasTrainingModeDropout(attr.G)) return true;
                    foreach (var sub in attr.Graphs)
                        if (HasTrainingModeDropout(sub)) return true;
                }
            }
            return false;
        }

        /// <summary>Lifts concrete tensor data into graph variables.</summary>
        public class ArgsProcessor
        {
            /// <summary>Lifts each element of the data sequence into a tensor variable.</summary>
            public TensorSequence<T> Get<T>(TensorDataSequence<T> sequence) where T : IVarType
            {
                return Globals.TensorSequence<T>(sequence.AsList.Select(x => Get(x)).ToArray());
            }

            /// <summary>Lifts the tensor data into a tensor variable.</summary>
            public Tensor<T> Get<T>(TensorData<T> tensorData) where T : IVarType
            {
                return (Variable)Globals.Tensor(tensorData);
            }
        }

        /// <summary>
        /// Executes a graph containing StateUpdate nodes: state-update nodes are lowered to extra
        /// outputs, and the resulting state values are folded back into a copy of the graph.
        /// Returns the regular outputs plus the state-updated graph for the next call.
        /// </summary>
        public (NamedModelParam[] regularOutputs, FastComputationGraph updatedGraph) ExecuteWithState(FastComputationGraph graph, params TensorData[] inputs)
        {
            var loweredGraph = LowerStateUpdateNodesOnFast(graph);
            var allOutputs = this.Execute(loweredGraph, inputs);
            return ProcessExecuteWithStateResults(graph, allOutputs);
        }

        /// <summary>
        /// Named-input overload of
        /// <see cref="ExecuteWithState(FastComputationGraph, TensorData[])"/>.
        /// </summary>
        public (NamedModelParam[] regularOutputs, FastComputationGraph updatedGraph) ExecuteWithState(FastComputationGraph graph, params NamedModelParam[] inputs)
        {
            var loweredGraph = LowerStateUpdateNodesOnFast(graph);
            var allOutputs = this.Run(loweredGraph, inputs);
            return ProcessExecuteWithStateResults(graph, allOutputs);
        }

        private static FastComputationGraph LowerStateUpdateNodesOnFast(FastComputationGraph graph)
        {
            var hasStateNodes = graph.Nodes.Any(n =>
                n.OpCode == InternalOpCodes.WITH_STATE_DEPS ||
                n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
            if (!hasStateNodes) return graph;

            var clone = graph.Clone();
            FastLowerStateUpdateNodes.Process(clone);
            return clone;
        }

        private (NamedModelParam[] regularOutputs, FastComputationGraph updatedGraph) ProcessExecuteWithStateResults(FastComputationGraph graph, NamedModelParam[] allOutputs)
        {
            var stateUpdateOutputCount = graph.GetStateUpdateOutputCount();


            var regularOutputCount = allOutputs.Length - stateUpdateOutputCount;

            var regularOutputs = allOutputs.Take(regularOutputCount).ToArray();
            var stateUpdateOutputs = allOutputs.Skip(regularOutputCount).Select(x => x.ToTensorData()).ToArray();


            var updatedGraph = graph.WithUpdatedStates(stateUpdateOutputs);

            return (regularOutputs, updatedGraph);
        }

    }

    /// <summary>
    /// First half of the fluent eager-evaluation helper: holds the input tensors of an
    /// <c>inputs.Eval(outputs).With(data)</c> chain. See <see cref="ComputeContextExtensions.Eval"/>.
    /// </summary>
    public class EvalFrom
    {
        private Variable[] inputs;

        /// <summary>Captures the graph inputs to evaluate from.</summary>
        public EvalFrom(Variable[] inputs)
        {
            this.inputs = inputs;
        }

        /// <summary>Selects the output tensors to evaluate.</summary>
        public EvalTo To(Variable[] outputs)
        {
            return new EvalTo(this.inputs, outputs);
        }
    }

    /// <summary>
    /// Second half of the fluent eager-evaluation helper: executes the captured
    /// inputs → outputs subgraph on <see cref="ComputeContext.Default"/> via <see cref="With"/>.
    /// </summary>
    public class EvalTo
    {
        private Variable[] inputs;
        private Variable[] outputs;

        /// <summary>Captures the inputs and outputs of the subgraph to execute.</summary>
        public EvalTo(Variable[] inputs, Variable[] outputs)
        {
            this.inputs = inputs;
            this.outputs = outputs;
        }

        /// <summary>Executes the subgraph with <paramref name="inputData"/> and returns the output values.</summary>
        public TensorData[] With(TensorData[] inputData)
        {
            var graph = new FastComputationGraph([..this.inputs], [..this.outputs]);
            return ComputeContext.Default.Execute(graph, inputData).Select(x => x.ToTensorData()).ToArray();
        }
    }

    /// <summary>Extension entry points for eager evaluation and data conversion.</summary>
    public static class ComputeContextExtensions
    {
        /// <summary>Unwraps the backend tensor value carried by <paramref name="data"/>.</summary>
        public static IShorokooTensorValue ToTensorValue(this IData data)
        {
            if (data is IOnnxData onnxData)
                return onnxData.Value;

            throw new UnsupportedDTypeException(ErrorCodes.CR006, data?.GetType()?.Name ?? "null", "ToTensorValue",
                "Data type is not supported for tensor value conversion");
        }

        /// <summary>Starts a fluent eager evaluation: <c>inputs.Eval(outputs).With(data)</c>.</summary>
        public static EvalTo Eval(this IEnumerable<Variable> inputTensors, params Variable[] outputTensors)
        {
            return new EvalFrom(inputTensors.ToArray()).To(outputTensors);
        }
    }
}
