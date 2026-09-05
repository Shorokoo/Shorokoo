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
    /// Created once via <see cref="ComputeContext.Compile(ComputationGraph)"/>, then invoked repeatedly
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
    /// The runtime that turns a <see cref="ComputationGraph"/> into an inference session and runs it —
    /// once via <see cref="Execute(ComputationGraph, IData[])"/>, or repeatedly via a
    /// <see cref="CompiledGraph"/> from <see cref="Compile(ComputationGraph)"/>.
    ///
    /// A context carries no per-instance <i>compute</i> configuration — no device, execution provider,
    /// thread count or session options — and every session it creates is built by the one process-wide
    /// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend.Factory"/>. Two distinct
    /// instances therefore name a <i>phase</i> of the work, never a device. Its one per-instance
    /// setting observes rather than configures: <see cref="Progress"/>, the sink a long build reports
    /// its stages to.
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

        /// <summary>
        /// Optional sink for build progress. When set, every build that runs on this context — the
        /// <c>ToConcreteArchitecture</c> lowering pipeline, and every build that takes this context as
        /// its <c>mergeContext</c>: <c>TrainingRig.FromScratch</c>, each <c>With…</c> derivation, and
        /// <c>TrainingRig.Load</c> — reports each stage as it enters it, so a build that runs for
        /// minutes is visibly alive and its last report names the stage it is in. <c>null</c> (the
        /// default) reports nothing and costs nothing. Only a full <c>FromScratch</c> concretizes, so
        /// only its report stream opens with <see cref="BuildPhase.Concretize"/>; a derivation or a
        /// load starts at <see cref="BuildPhase.TrainingStep"/>, on a clock of its own.
        ///
        /// <para>Reports are raised synchronously on the building thread, so use
        /// <see cref="SynchronousBuildProgress"/> (which calls its handler inline) rather than
        /// <see cref="System.Progress{T}"/> when the order of reports matters — and note that an
        /// exception thrown by the sink propagates out of the build. A context shared by concurrent
        /// builds delivers their reports interleaved on their own threads; the sink must tolerate
        /// that.</para>
        /// </summary>
        public IProgress<BuildProgress>? Progress { get; set; }

        /// <summary>Creates a compute context. Nothing about the <i>compute</i> is per-context — its
        /// sessions come from the process-wide
        /// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend.Factory"/> — so the only
        /// thing to set on the instance is the observational <see cref="Progress"/> sink.</summary>
        public ComputeContext()
        {
        }

        /// <summary>
        /// Compiles the graph into a reusable <see cref="CompiledGraph"/>: the ONNX model and
        /// inference session are built once, so repeated executions only feed new data.
        /// </summary>
        public CompiledGraph Compile(ComputationGraph graph)
        {
            graph.RequireConcretized("ComputeContext.Compile");
            return Compile(graph.ToInternal());
        }

        /// <summary>Executes a graph that takes no inputs.</summary>
        public NamedModelParam[] Execute(ComputationGraph graph) => this.Execute(graph, []);

        /// <summary>
        /// Executes the graph, pairing the inputs positionally with the graph's inputs.
        /// TensorDataStruct inputs are automatically expanded into individual fields.
        /// Requires a concretized graph — a module graph fails fast with the lowering
        /// hint instead of dying deep inside session creation.
        /// </summary>
        public NamedModelParam[] Execute(ComputationGraph graph, params IData[] inputs)
        {
            graph.RequireConcretized("ComputeContext.Execute");
            return this.Execute(graph.ToInternal(), inputs);
        }

        /// <summary>
        /// Executes the graph with pre-built named inputs. Builds the ONNX model and a fresh
        /// inference session per call (disposed afterwards); use <see cref="Compile(ComputationGraph)"/>
        /// for repeated runs.
        /// </summary>
        public NamedModelParam[] Run(ComputationGraph graph, params NamedModelParam[] inputs)
        {
            graph.RequireConcretized("ComputeContext.Run");
            return this.Run(graph.ToInternal(), inputs);
        }

        /// <summary>
        /// Executes a graph containing StateUpdate nodes: state-update nodes are lowered to extra
        /// outputs, and the resulting state values are folded back into a copy of the graph.
        /// Returns the regular outputs plus the state-updated graph (same
        /// <see cref="ComputationGraph.Kind"/>) for the next call. The input graph is unchanged.
        /// </summary>
        public (NamedModelParam[] regularOutputs, ComputationGraph updatedGraph) ExecuteWithState(
            ComputationGraph graph, params TensorData[] inputs)
        {
            graph.RequireConcretized("ComputeContext.ExecuteWithState");
            var (regularOutputs, updatedGraph) = ExecuteWithState(graph.ToInternal(), inputs);
            // updatedGraph is either the private copy itself (no state params) or a fresh
            // clone with the new state values — exclusively owned either way.
            return (regularOutputs, new ComputationGraph(updatedGraph, graph.Kind));
        }

        /// <summary>
        /// Named-input overload of
        /// <see cref="ExecuteWithState(ComputationGraph, TensorData[])"/>.
        /// </summary>
        public (NamedModelParam[] regularOutputs, ComputationGraph updatedGraph) ExecuteWithState(
            ComputationGraph graph, params NamedModelParam[] inputs)
        {
            graph.RequireConcretized("ComputeContext.ExecuteWithState");
            var (regularOutputs, updatedGraph) = ExecuteWithState(graph.ToInternal(), inputs);
            return (regularOutputs, new ComputationGraph(updatedGraph, graph.Kind));
        }

        internal CompiledGraph Compile(InternalComputationGraph graph)
        {
            var originalInputNames = ResolveOriginalInputNames(graph);
            return CompileFromModel(
                () => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true),
                originalInputNames);
        }

        private CompiledGraph CompileFromModel(Func<ModelProto> buildModel, string[] originalInputNames)
        {
            var model = buildModel();

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            var session = CreateSession(modelData, HasOptionalOps(model.Graph));

            var onnxInputNameByOriginal = new Dictionary<string, string>();
            for (int i = 0; i < originalInputNames.Length && i < session.InputNames.Count; i++)
                onnxInputNameByOriginal[originalInputNames[i]] = session.InputNames[i];

            return new CompiledGraph(session, onnxInputNameByOriginal, originalInputNames);
        }

        private static string[] ResolveOriginalInputNames(InternalComputationGraph graph)
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
            var graph = new InternalComputationGraph([], [.. outputs]);
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
        internal NamedModelParam[] Execute(InternalComputationGraph graph) => this.Execute(graph, []);

        /// <summary>
        /// Executes the graph, pairing the inputs positionally with the graph's inputs.
        /// TensorDataStruct inputs are automatically expanded into individual fields.
        /// </summary>
        internal NamedModelParam[] Execute(InternalComputationGraph graph, params IData[] inputs)
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
        /// inference session per call (disposed afterwards); use <see cref="Compile(ComputationGraph)"/> for repeated runs.
        /// </summary>
        internal NamedModelParam[] Run(InternalComputationGraph graph, params NamedModelParam[] inputs)
        {
            var originalInputNames = ResolveOriginalInputNames(graph);
            return RunFromModel(
                () => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true),
                originalInputNames,
                inputs);
        }

        private NamedModelParam[] RunFromModel(Func<ModelProto> buildModel, string[] originalInputNames, NamedModelParam[] inputs)
        {
            var model = buildModel();

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            var session = CreateSession(
                modelData, HasOptionalOps(model.Graph) || IsFullyConstant(model.Graph));
            try
            {
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

                return results.Zip(session.OutputNames).Select(x =>
                            OnnxUtils.CreateNamedModelParam(x.First, ModelParamType.OutputParam, x.Second))
                            .ToArray();
            }
            finally
            {
                // Dispose the session to free native memory — on the throwing path too, where
                // the memory it holds is the memory the caller has just been told it lacks. The
                // returned tensor values stay valid across it, and the finally also keeps the
                // session rooted across the native calls above. They are not, however, free of
                // it: a result keeps its session's ALLOCATOR alive, so a caller that retains one
                // retains that session's arena — see the `ort-values-are-never-disposed`
                // finding, and `FastInitializeModelParams.Rehost` for a caller that must not.
                session.Dispose();
            }
        }

        private IShorokooInferenceSession CreateSession(byte[] modelData, bool disableOptimizations = false)
        {
            // Both conditions that pass true here are avoiding ORT's constant-folding pass:
            // it calls GetDeleteFunc on Optional values, which OptionalTypeBase doesn't
            // implement -- session init throws "GetDeleteFunc is not implemented" -- and on
            // an input-less graph it folds the whole graph at build time, at a cost that
            // scales with the data (see IsFullyConstant). Disabling optimizations skips the
            // fold pass; the nodes then go through the normal execution path, which ORT
            // handles correctly and which reuses buffers.
            var optLevel = disableOptimizations
                ? ShorokooGraphOptimization.DisableAll
                : ShorokooGraphOptimization.EnableAll;
            return InferenceBackend.Factory.CreateSession(
                modelData,
                optLevel,
                ShorokooLogSeverity.Fatal);
        }

        /// <summary>
        /// Whether the model takes no runtime input, so every node's value is already
        /// determined when the session is built.
        ///
        /// <para>Such a graph is the one case where ORT's constant-folding pass computes the
        /// WHOLE graph at session build: it walks the nodes in order, evaluating each into a
        /// freshly allocated initializer, and the chain's intermediates pile up instead of
        /// flowing through an execution plan that reuses buffers. Parameter initialization is
        /// exactly this shape — <c>FastInitializeModelParams</c>
        /// hands over an input-less graph of every parameter's keyed Threefry draw — so the fold
        /// materialized every int64 intermediate of every draw at once. Rig construction then
        /// cost kilobytes of host memory per parameter ELEMENT — a thousand times the 4 bytes the
        /// fp32 parameter itself occupies — so a few-million-parameter model wanted tens of GB
        /// and minutes just to build, and a GPT-sized embedding died with ORT's bare
        /// "bad allocation" (Shorokoo/Shorokoo#194, #195).</para>
        ///
        /// <para>Folding buys nothing here in any case. The session is built, run once and
        /// disposed (see <see cref="RunFromModel"/>), so the work happens exactly once either
        /// way — the only question is whether it happens in the fold pass or in the execution
        /// plan, and only the latter reuses buffers. Running the graph unoptimized is therefore
        /// both faster and dramatically smaller.</para>
        ///
        /// <para>It is also value-identical in practice, which is worth spelling out because
        /// "disable the optimizer" usually is not. Folding runs before the fusions that rearrange
        /// arithmetic, and it evaluates each node with the same CPU kernel the execution plan
        /// would — so on a graph this predicate accepts, folding leaves literals and those
        /// fusions find nothing to work on. The caveat is that ORT's folding skips what it cannot
        /// evaluate (a node with no CPU kernel, a non-deterministic op), and an unfolded tail
        /// COULD have been fused before and is not now; no such difference has been observed.
        /// <c>RngInitFrozenDerivationTests</c> asserts exact initial weights through this path
        /// for a uniform, a raw-bits and a dense-normal initializer — which pins the values, not
        /// the optimization level, since they are identical either way.</para>
        ///
        /// <para>The predicate is a property of the GRAPH, not of the caller, so it also catches
        /// every other input-less one-shot: the RNG key resolver, optimizer-state seeding (which
        /// bakes its inputs to constants and then clears them, so it is always input-less), and
        /// <c>Eval</c>, which builds a zero-input graph unconditionally — so every eager
        /// evaluation now takes this path. That breadth is intended: each is a constant computed
        /// once and discarded, and the paragraph above applies to each unchanged. The
        /// order-of-magnitude figures are measured on parameter initialization, which is the
        /// shape that made it matter.</para>
        ///
        /// <para>It is deliberately scoped to <see cref="RunFromModel"/>. A
        /// <see cref="CompileFromModel"/> session is kept and re-run, so there optimization is
        /// amortized and stays on — which is why a keyed feed inside a training-step or exported
        /// model still gets its constant key chain folded, as
        /// <c>Documentation/rng-configuration.md</c> says it does.</para>
        /// </summary>
        private static bool IsFullyConstant(GraphProto graph) => graph.Inputs.Count == 0;

        private static bool HasOptionalOps(GraphProto graph)
        {
            var found = false;
            FastOnnxModelBuilder.ForEachGraphRecursive(graph, g =>
            {
                found = found || g.Nodes.Any(node =>
                    node.OpType.StartsWith("Optional", StringComparison.Ordinal));
            });
            return found;
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
        internal (NamedModelParam[] regularOutputs, InternalComputationGraph updatedGraph) ExecuteWithState(InternalComputationGraph graph, params TensorData[] inputs)
        {
            var loweredGraph = LowerStateUpdateNodesOnFast(graph);
            var allOutputs = this.Execute(loweredGraph, inputs);
            return ProcessExecuteWithStateResults(graph, allOutputs);
        }

        /// <summary>
        /// Named-input overload of
        /// <see cref="ExecuteWithState(InternalComputationGraph, TensorData[])"/>.
        /// </summary>
        internal (NamedModelParam[] regularOutputs, InternalComputationGraph updatedGraph) ExecuteWithState(InternalComputationGraph graph, params NamedModelParam[] inputs)
        {
            var loweredGraph = LowerStateUpdateNodesOnFast(graph);
            var allOutputs = this.Run(loweredGraph, inputs);
            return ProcessExecuteWithStateResults(graph, allOutputs);
        }

        private static InternalComputationGraph LowerStateUpdateNodesOnFast(InternalComputationGraph graph)
        {
            var hasStateNodes = graph.Nodes.Any(n =>
                n.OpCode == InternalOpCodes.WITH_STATE_DEPS ||
                n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
            if (!hasStateNodes) return graph;

            var clone = graph.Clone();
            FastLowerStateUpdateNodes.Process(clone);
            return clone;
        }

        private (NamedModelParam[] regularOutputs, InternalComputationGraph updatedGraph) ProcessExecuteWithStateResults(InternalComputationGraph graph, NamedModelParam[] allOutputs)
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
            var graph = new InternalComputationGraph([..this.inputs], [..this.outputs]);
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
