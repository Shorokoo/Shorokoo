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
    /// via <see cref="Execute(IData[])"/> — each call only feeds new data, with zero graph
    /// rebuilding or session creation overhead.
    /// </summary>
    public class CompiledGraph
    {
        private readonly IShorokooInferenceSession _session;
        private readonly IShorokooInferenceSessionFactory _backend;
        // The context that compiled this graph: outputs belong to it, and it decides whether they
        // leave it. A compiled graph runs on the backend it was built with whatever happens to the
        // context afterwards, so this is about the results and not about where the work runs.
        private readonly ComputeContext _owner;
        private readonly Dictionary<string, string> _onnxInputNameByOriginal;
        private readonly string[] _originalInputNames;

        internal CompiledGraph(
            IShorokooInferenceSession session,
            IShorokooInferenceSessionFactory backend,
            Dictionary<string, string> onnxInputNameByOriginal,
            string[] originalInputNames,
            ShorokooGraphOptimization optimization,
            DeviceMemorySettings deviceMemory,
            RunSettings defaultRunSettings,
            ComputeContext owner)
        {
            _owner = owner;
            _session = session;
            _backend = backend;
            _onnxInputNameByOriginal = onnxInputNameByOriginal;
            _originalInputNames = originalInputNames;
            Optimization = optimization;
            DeviceMemory = deviceMemory;
            DefaultRunSettings = defaultRunSettings;
        }

        /// <summary>
        /// What a run of this graph uses when the call names nothing: the
        /// <see cref="ComputeContext.RunSettings"/> of the context that compiled it, taken when
        /// it was compiled. Each <c>Execute</c> / <c>Run</c> overload has a sibling taking a
        /// <see cref="Shorokoo.Core.Inference.Abstractions.RunSettings"/> that overrides it for
        /// one call, so this is a default and never a ceiling.
        /// </summary>
        public RunSettings DefaultRunSettings { get; }
        /// The backend this graph was compiled on and runs on — fixed when it was compiled, since
        /// the session belongs to that backend and cannot move. Feeding it data another backend
        /// built is allowed: the session rebuilds what it must, provided the data is host-resident.
        /// </summary>
        public BackendDescription Backend => _backend.Description;

        /// <summary>The graph-optimization profile the session was built with (test hook).</summary>
        internal ShorokooGraphOptimization Optimization { get; }

        /// <summary>
        /// The arena settings this graph's session was built with: the compiling context's
        /// <see cref="ComputeContext.DeviceMemory"/> as it stood then, with
        /// <see cref="ArenaExtendStrategy.Auto"/> already settled to the strategy this session
        /// got. A session keeps what it was built with, so this is what the session actually has —
        /// not what its context says now, and not <c>Auto</c>.
        /// </summary>
        public DeviceMemorySettings DeviceMemory { get; }

        /// <summary>
        /// Executes the compiled graph with the given inputs.
        /// TensorDataStruct inputs are automatically expanded into individual fields.
        /// </summary>
        public NamedModelParam[] Execute(params IData[] inputs)
            => Run(NameInputs(inputs), retainedOutputNames: null, DefaultRunSettings);

        /// <summary>
        /// <see cref="Execute(IData[])"/> under <paramref name="runSettings"/> instead of
        /// <see cref="DefaultRunSettings"/>. ONNX Runtime reads these off the run, so this
        /// changes nothing about the session and applies to this call alone.
        /// </summary>
        public NamedModelParam[] Execute(IData[] inputs, RunSettings runSettings)
            => Run(NameInputs(inputs), retainedOutputNames: null, runSettings);

        /// <summary>
        /// Executes the compiled graph, leaving the outputs whose index is <c>true</c> in
        /// <paramref name="retainOnDevice"/> in the execution provider's own memory instead of
        /// fetching them back to the host — so a value produced by one call can be fed straight
        /// into the next without crossing the bus. A retained output is not host-readable
        /// (<see cref="TensorData.IsHostResident"/>); every other output comes back exactly as
        /// <see cref="Execute(IData[])"/>'s do, and on a session with no device memory
        /// (<see cref="HasDeviceMemory"/>) nothing is retained and this <i>is</i>
        /// <see cref="Execute(IData[])"/>.
        /// </summary>
        /// <param name="inputs">The graph inputs, struct inputs expanded as in <see cref="Execute(IData[])"/>.
        /// They may themselves be values a previous call retained.</param>
        /// <param name="retainOnDevice">One flag per graph output, in output order.</param>
        public NamedModelParam[] Execute(IData[] inputs, bool[] retainOnDevice)
            => Execute(inputs, retainOnDevice, DefaultRunSettings);

        /// <summary>
        /// <see cref="Execute(IData[], bool[])"/> under <paramref name="runSettings"/> instead of
        /// <see cref="DefaultRunSettings"/>, for this call alone.
        /// </summary>
        public NamedModelParam[] Execute(IData[] inputs, bool[] retainOnDevice, RunSettings runSettings)
        {
            if (retainOnDevice is null) throw new ArgumentNullException(nameof(retainOnDevice));
            if (retainOnDevice.Length != _session.OutputNames.Count)
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "CompiledGraph.Execute",
                    $"retainOnDevice.Length={retainOnDevice.Length}, graph.Outputs.Count={_session.OutputNames.Count}",
                    "Retention flag count does not match the graph's output count");

            var retained = new HashSet<string>();
            for (int i = 0; i < retainOnDevice.Length; i++)
                if (retainOnDevice[i]) retained.Add(_session.OutputNames[i]);

            return Run(NameInputs(inputs), retained, runSettings);
        }

        /// <summary>
        /// Executes the compiled graph with pre-built named inputs.
        /// </summary>
        public NamedModelParam[] Run(params NamedModelParam[] inputs)
            => Run(inputs, retainedOutputNames: null, DefaultRunSettings);

        /// <summary>
        /// <see cref="Run(NamedModelParam[])"/> under <paramref name="runSettings"/> instead of
        /// <see cref="DefaultRunSettings"/>, for this call alone.
        /// </summary>
        public NamedModelParam[] Run(NamedModelParam[] inputs, RunSettings runSettings)
            => Run(inputs, retainedOutputNames: null, runSettings);

        // Every Execute and Run overload funnels here, so one guard covers the lot -- and covers it
        // before anything is fed, which a per-overload one would not for the retaining path.
        private NamedModelParam[] Run(
            NamedModelParam[] inputs, IReadOnlySet<string>? retainedOutputNames, RunSettings runSettings)
        {
            ArgumentNullException.ThrowIfNull(runSettings);
            var sessionInputs = new Dictionary<string, IShorokooTensorValue>();
            foreach (var input in inputs)
            {
                var onnxName = _onnxInputNameByOriginal.TryGetValue(input.ParamName, out var mapped)
                    ? mapped : input.ParamName;
                sessionInputs[onnxName] = input.ToTensorValue();
            }

            var results = retainedOutputNames is null
                ? _session.Run(sessionInputs, _session.OutputNames, runSettings)
                : _session.RunRetainingOutputs(
                    sessionInputs, _session.OutputNames, retainedOutputNames, runSettings);

            return _owner.Deliver(results.Zip(_session.OutputNames)
                .Select(x => OnnxUtils.CreateNamedModelParam(
                    x.First, ModelParamType.OutputParam, x.Second, _owner))
                .ToArray());
        }

        /// <summary>Pairs the expanded inputs with the graph's input names, positionally.</summary>
        private NamedModelParam[] NameInputs(IData[] inputs)
        {
            var expandedInputs = ComputeContext.ExpandStructInputs(inputs);

            if (expandedInputs.Length != _originalInputNames.Length)
            {
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "CompiledGraph.Execute",
                    $"inputs.Length={expandedInputs.Length}, graph.Inputs.Length={_originalInputNames.Length}",
                    "Input length mismatch: number of provided inputs does not match the graph's expected input tensor count");
            }

            return expandedInputs.Zip(_originalInputNames)
                .Select(zip => NamedModelParam.FromIData(zip.Second, ModelParamType.InputParam, zip.First))
                .ToArray();
        }

        /// <summary>
        /// Whether this graph's session produces its outputs somewhere other than host memory, so
        /// <see cref="Execute(IData[], bool[])"/> has somewhere to retain them.
        /// </summary>
        public bool HasDeviceMemory => _session.HasDeviceMemory;

        /// <summary>How many outputs this graph's session produces — the length
        /// <see cref="Execute(IData[], bool[])"/> requires of a retention array, so a caller can
        /// size one without deriving the count a second way and disagreeing.</summary>
        public int OutputCount => _session.OutputNames.Count;
    }

    /// <summary>
    /// The runtime that turns a <see cref="ComputationGraph"/> into an inference session and runs it —
    /// once via <see cref="Execute(ComputationGraph, IData[])"/>, or repeatedly via a
    /// <see cref="CompiledGraph"/> from <see cref="Compile(ComputationGraph)"/>.
    ///
    /// A context may name the backend it runs on, and two contexts may name different ones — a CPU
    /// context and a CUDA context in one process, each compiling and running on its own device:
    /// <code>
    /// var cpu  = new ComputeContext(new LinuxCpuInferenceFactory());
    /// var cuda = new ComputeContext(new LinuxGpuInferenceFactory());
    /// cpu.Execute(graph, input);   // on the host
    /// cuda.Execute(graph, input);  // the same graph, on the card
    /// </code>
    /// A context constructed without one runs on the process default
    /// (<see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend.Factory"/>), which is what
    /// every context did before backends could differ. Which device a context will use is read off
    /// <see cref="Backend"/>.
    ///
    /// <para>It also carries how its sessions and runs are configured — <see cref="DeviceMemory"/>
    /// for the arena each session it compiles is built with, and <see cref="RunSettings"/> for what
    /// its runs do by default. Both are per instance, so two contexts may differ and neither
    /// reaches the other's sessions.</para>
    ///
    /// <para>A context names where the <i>work</i> runs; it does not change where tensors are
    /// built. Every <c>TensorData</c> in the program is built by the default backend
    /// (<see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend.Factory"/>) wherever it is
    /// built, and a session on another backend converts what it is fed, handing its own outputs
    /// back. So the same data feeds either context and the same model runs on both, with nothing to
    /// say at the call site. The conversion costs a host copy per feed, and is possible only for
    /// data the host can read — see
    /// <see cref="Shorokoo.Core.Inference.Abstractions.BackendTransfer"/>.</para>
    /// </summary>
    public class ComputeContext : IDisposable
    {
        private static ComputeContext? _defaultComputeContext;

        // Null means the default backend, read when the work runs rather than at construction: a
        // context built before InferenceBackend.Factory was assigned must still honour it.
        private readonly IShorokooInferenceSessionFactory? _backend;

        /// <summary>
        /// Process-wide default context, created lazily on first access and used wherever no
        /// explicit context is supplied. Settable to swap in a custom context.
        ///
        /// <para>This names <i>which</i> context is the fallback; it is not a way to reconfigure
        /// one. A context's <see cref="DeviceMemory"/> and <see cref="RunSettings"/> are
        /// initialize-only, so assigning here cannot alter a context anything else already holds,
        /// and cannot reach a session that has already been compiled — including those compiled by
        /// the context being replaced. Code that wants a configuration of its own should hold its
        /// own context rather than assign this one.</para>
        /// </summary>
        public static ComputeContext Default
        {
            get
            {
                if (_defaultComputeContext is not null) return _defaultComputeContext;

                // The backend a process loaded, under the rule that a CPU one wins: the unnamed
                // default should not be the card. Reading InferenceBackend.Factory is what
                // discovers and records one when nothing has been loaded yet, and what refuses --
                // naming the packages to deploy -- when there is nothing to discover.
                var backend = InferenceBackend.Remembered ?? InferenceBackend.Factory;

                // Its outputs leave it. The default context is the one nobody named and nobody
                // disposes, so a result that belonged to it would be tied to a lifetime the caller
                // never sees; detached, a result is the caller's and outlives everything here.
                return _defaultComputeContext = new ComputeContext(backend, detachesOutputs: true);
            }

            set { _defaultComputeContext = value; }
        }

        /// <summary>Creates a compute context that runs on the process-wide
        /// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend.Factory"/>, on the
        /// shipped defaults. Set <see cref="DeviceMemory"/> or <see cref="RunSettings"/> in an
        /// object initializer to compile and run under something else.</summary>
        public ComputeContext() : this(detachesOutputs: false)
        {
        }

        /// <summary>Creates a compute context on the process-wide backend, detaching its outputs
        /// or not — see <see cref="DetachesOutputs"/>.</summary>
        public ComputeContext(bool detachesOutputs)
        {
            DetachesOutputs = detachesOutputs;
        }

        private readonly DeviceMemorySettings _deviceMemory = DeviceMemorySettings.Default;

        /// <summary>
        /// The arena settings every session this context compiles from now on is built with.
        /// ONNX Runtime reads them while a session is being created and that session keeps them
        /// for life, so this configures the sessions to come and never the ones already built —
        /// to run a graph under a different budget, compile it on a context that carries one.
        ///
        /// <para>Its default <see cref="ArenaExtendStrategy.Auto"/> resolves per session, so one
        /// context can still give a session it knows is reused across shapes a different arena
        /// strategy from the rest; <see cref="CompiledGraph.DeviceMemory"/> reports which one a
        /// graph got.</para>
        ///
        /// <para>Ignored by the CPU backends, which have no device arena.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException">A null settings object.</exception>
        public DeviceMemorySettings DeviceMemory
        {
            get => _deviceMemory;
            init => _deviceMemory = value ?? throw new ArgumentNullException(nameof(value));
        }

        private readonly RunSettings _runSettings = RunSettings.Default;

        /// <summary>
        /// What runs on this context do when the call names nothing of its own. A
        /// <see cref="CompiledGraph"/> takes a copy of this when it is compiled
        /// (<see cref="CompiledGraph.DefaultRunSettings"/>), and every <see cref="CompiledGraph"/>
        /// run entry point also takes a
        /// <see cref="Shorokoo.Core.Inference.Abstractions.RunSettings"/> to override it for one
        /// call, because ORT reads these off the run rather than the session. This context's own
        /// one-shot entry points — <see cref="Execute(ComputationGraph, IData[])"/>,
        /// <see cref="Run(ComputationGraph, NamedModelParam[])"/>, <c>Eval</c> and
        /// <c>ExecuteWithState</c> — build and dispose a session per call and offer no such
        /// override; they run on this.
        /// </summary>
        /// <exception cref="ArgumentNullException">A null settings object.</exception>
        public RunSettings RunSettings
        {
            get => _runSettings;
            init => _runSettings = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>
        /// Creates a compute context that compiles and runs on <paramref name="backend"/>, whatever
        /// the process default is. This is how one program drives two devices: a context per
        /// backend, each with sessions of its own. The data they run on is shared — see the note on
        /// the class.
        /// </summary>
        /// <param name="backend">The backend its sessions are built by — a platform factory
        /// (<c>new LinuxGpuInferenceFactory()</c>) where one native ONNX Runtime serves both, or one
        /// from <see cref="Shorokoo.Core.Inference.Abstractions.IsolatedBackend.Load"/> where each
        /// backend needs a native of its own.</param>
        /// <exception cref="ArgumentNullException"><paramref name="backend"/> is null.</exception>
        public ComputeContext(IShorokooInferenceSessionFactory backend)
            : this(backend, detachesOutputs: false)
        {
        }

        /// <summary>
        /// Creates a compute context on <paramref name="backend"/>, detaching its outputs or not —
        /// see <see cref="DetachesOutputs"/>.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="backend"/> is null.</exception>
        public ComputeContext(IShorokooInferenceSessionFactory backend, bool detachesOutputs)
        {
            ArgumentNullException.ThrowIfNull(backend);
            _backend = backend;
            DetachesOutputs = detachesOutputs;
        }

        /// <summary>
        /// Whether a run's output tensors leave this context behind.
        ///
        /// <para>With it set, every tensor a run produces is transferred to the null context — the
        /// framework's own host memory — and the tensor the session handed back is disposed. That
        /// disposal frees nothing, by construction: a transfer within one memory space moves the
        /// ownership off the original, and one across spaces has already released it. What the
        /// caller gets back is a tensor that outlives this context, which is what a context that
        /// is disposed per run needs its results to do.</para>
        ///
        /// <para>Without it, outputs belong to this context and disposing it takes them with it.</para>
        /// </summary>
        public bool DetachesOutputs { get; }

        private readonly ConditionalWeakTable<TensorStorage, object> _ownedStorage = new();
        private static readonly object OwnedMarker = new();
        private bool _disposed;

        /// <summary>Puts a storage on this context's books; its disposal will release it.</summary>
        internal void TakeOwnership(TensorStorage storage)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _ownedStorage.AddOrUpdate(storage, OwnedMarker);
        }

        /// <summary>Takes a storage off this context's books, because something else owns it now.</summary>
        internal void ReleaseOwnership(TensorStorage storage) => _ownedStorage.Remove(storage);

        /// <summary>
        /// Releases everything this context still owns, and the backend with it.
        ///
        /// <para>Every tensor whose bytes were on this context's books is invalidated: reading one
        /// afterwards throws rather than reading freed memory, whether or not that tensor was
        /// itself disposed. Bytes that were transferred away are not touched — they belong to the
        /// context that took them, and the whole point of a same-space transfer is that disposing
        /// the source leaves them standing.</para>
        ///
        /// <para>The tracking is weak, so a tensor the program has already dropped does not keep
        /// its storage alive waiting for this.</para>
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            foreach (var (storage, _) in _ownedStorage) storage.Release();
            _ownedStorage.Clear();

            // Only a backend this context was given, and only one that has something to release.
            // The process-wide default belongs to the process, not to whichever context read it.
            if (_backend is IDisposable disposable) disposable.Dispose();

            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Hands a run's outputs to the caller the way this context was asked to: as they are, or
        /// detached from it.
        /// </summary>
        internal NamedModelParam[] Deliver(NamedModelParam[] outputs)
        {
            if (!DetachesOutputs) return outputs;

            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] is TensorDataSequenceModelParam sequenceParam)
                {
                    // A sequence is detached by forgetting this context rather than by being moved:
                    // it holds its runtime value outright, so it already outlives the context --
                    // this context's books never had it -- and its elements are copied out of that
                    // value one at a time, so there is nothing yet to move. Clearing the context is
                    // what stops each of those elements being handed one that may be disposed
                    // before it is read.
                    sequenceParam.ToTensorDataSequence().Context = null;
                    continue;
                }
                if (outputs[i] is not TensorDataModelParam tensorParam) continue;
                var original = tensorParam.ToTensorData();
                var detached = original.TransferTo(null);
                original.Dispose();
                outputs[i] = new TensorDataModelParam(
                    tensorParam.ParamName, tensorParam.ParamType, detached);
            }
            return outputs;
        }

        /// <summary>The backend this context's work runs on: the one it was constructed with, or
        /// the default when it names none.</summary>
        internal IShorokooInferenceSessionFactory Factory => _backend ?? InferenceBackend.Factory;

        /// <summary>
        /// The backend this context compiles and runs on — its name, its device, and the CUDA device
        /// it allocates on. A context constructed with a backend reports that one; a context without
        /// reports the process default, which is what every context reported when only one could be
        /// live. Read it to log the device a run used, or call
        /// <see cref="Shorokoo.Core.Inference.Abstractions.InferenceBackend.RequireDevice"/> to
        /// refuse to start on the wrong one.
        ///
        /// <para>Reading this resolves the process default if this context names no backend and none
        /// is live yet, exactly as compiling would.</para>
        /// </summary>
        public BackendDescription Backend => Factory.Description;

        /// <summary>Where this context's tensors live. Two contexts reporting the same space can
        /// pass a tensor between them without copying it.</summary>
        public MemorySpace MemorySpace => Factory.MemorySpace;

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

        internal CompiledGraph Compile(InternalComputationGraph graph) => Compile(graph, inputDims: null, trainingStep: false);

        /// <summary>
        /// Compiles the graph for a session that will only ever be fed inputs of exactly
        /// <paramref name="inputDims"/> (one entry per graph input, in input order; a null entry keeps
        /// that input's shape symbolic). The dims are stamped on the model's graph inputs, which lets
        /// ONNX Runtime resolve every intermediate shape at session build and fold the graph's shape
        /// arithmetic away — a large share of a training step's kernels. The caller owns the contract:
        /// ORT rejects a differently-shaped feed at <c>Run</c>, so the caller must compile another
        /// <see cref="CompiledGraph"/> for another shape (see <c>TrainingRig</c>'s shape-keyed cache).
        /// </summary>
        /// <param name="graph">The graph to compile.</param>
        /// <param name="inputDims">Concrete dims per graph input, or null to keep them symbolic.</param>
        /// <param name="trainingStep">True for a lowered training step the memory-aware pass has
        /// already scheduled: what it duplicates, it duplicates on purpose, so the session must not
        /// merge it back (<see cref="ShorokooGraphOptimization.TrainingStep"/>). Every other graph
        /// — a user's <see cref="Compile(ComputationGraph)"/> included — runs the ordinary profile.</param>
        /// <param name="reusedAcrossShapes">True only where this session is <i>known</i> to be fed
        /// differing input shapes — the shapes have already differed, not merely could. It is the
        /// one thing that moves <see cref="ArenaExtendStrategy.Auto"/> off exact-size extension
        /// (<see cref="DeviceMemorySettings.Resolve"/>); a symbolic graph is not by itself
        /// evidence, since an ordinary compiled graph fed one shape for its whole life is symbolic
        /// too.</param>
        internal CompiledGraph Compile(
            InternalComputationGraph graph,
            IReadOnlyList<long[]?>? inputDims,
            bool trainingStep,
            bool reusedAcrossShapes = false)
        {
            graph.RequireRunnableOps("ComputeContext.Compile");
            var originalInputNames = ResolveOriginalInputNames(graph);
            return CompileFromModel(
                () => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true, inputDims: inputDims),
                originalInputNames,
                trainingStep,
                reusedAcrossShapes);
        }

        private CompiledGraph CompileFromModel(
            Func<ModelProto> buildModel,
            string[] originalInputNames,
            bool trainingStep,
            bool reusedAcrossShapes)
        {
            var model = buildModel();

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            var optimization = SessionOptimization(HasOptionalOps(model.Graph), trainingStep);
            // Settled here, not inside the session: CompiledGraph then reports the strategy this
            // session actually got rather than the Auto that asked for it.
            var deviceMemory = DeviceMemory.Resolve(reusedAcrossShapes);
            var session = CreateSession(modelData, optimization, deviceMemory);

            var onnxInputNameByOriginal = new Dictionary<string, string>();
            for (int i = 0; i < originalInputNames.Length && i < session.InputNames.Count; i++)
                onnxInputNameByOriginal[originalInputNames[i]] = session.InputNames[i];

            return new CompiledGraph(
                session, Factory, onnxInputNameByOriginal, originalInputNames, optimization,
                deviceMemory, RunSettings, this);
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
        /// returning their concrete tensor data. Requires concretized outputs — a
        /// <c>[Module]</c>'s output fails fast with the lowering hint.
        /// </summary>
        public TensorData[] Eval(Variable[] outputs)
        {
            var graph = new InternalComputationGraph([], [.. outputs]);
            graph.RequireRunnableOps("ComputeContext.Eval");
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
            // Before the arity check below: a module graph's inputs routinely disagree with what the
            // caller passed (its [Hyper] parameters are inputs too), and CR006 would report that
            // instead of the machinery that is the real problem.
            graph.RequireRunnableOps("ComputeContext.Execute");

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
            graph.RequireRunnableOps("ComputeContext.Run");
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
                var results = session.Run(sessionInputs, session.OutputNames, RunSettings);

                return Deliver(results.Zip(session.OutputNames).Select(x =>
                            OnnxUtils.CreateNamedModelParam(x.First, ModelParamType.OutputParam, x.Second, this))
                            .ToArray());
            }
            finally
            {
                // Dispose the session to free native memory — on the throwing path too, where
                // the memory it holds is the memory the caller has just been told it lacks. The
                // returned tensor values stay valid across it, and the finally also keeps the
                // session rooted across the native calls above. They are not, however, free of
                // it: a result keeps its session's ALLOCATOR alive, so a caller that retains one
                // retains that session's arena — see `FastProcessorHelper.RehostOffSession` for
                // what a caller that must not does, and Shorokoo/Shorokoo#180 for the general
                // question.
                session.Dispose();
            }
        }

        private static ShorokooGraphOptimization SessionOptimization(bool disableOptimizations, bool trainingStep)
        {
            // Both conditions that pass true here are avoiding ORT's constant-folding pass:
            // it calls GetDeleteFunc on Optional values, which OptionalTypeBase doesn't
            // implement -- session init throws "GetDeleteFunc is not implemented" -- and on
            // an input-less graph it folds the whole graph at build time, at a cost that
            // scales with the data (see IsFullyConstant). Disabling optimizations skips the
            // fold pass; the nodes then go through the normal execution path, which ORT
            // handles correctly and which reuses buffers.
            return disableOptimizations ? ShorokooGraphOptimization.DisableAll
                : trainingStep ? ShorokooGraphOptimization.TrainingStep
                : ShorokooGraphOptimization.EnableAll;
        }

        // A one-shot session: built, fed once, and disposed, so no differing shapes can reach it.
        private IShorokooInferenceSession CreateSession(byte[] modelData, bool disableOptimizations = false)
            => CreateSession(
                modelData,
                SessionOptimization(disableOptimizations, trainingStep: false),
                DeviceMemory.Resolve(reusedAcrossShapes: false));

        private IShorokooInferenceSession CreateSession(
            byte[] modelData, ShorokooGraphOptimization optimization, DeviceMemorySettings deviceMemory)
            => Factory.CreateSession(
                modelData, optimization, ShorokooLogSeverity.Fatal, deviceMemory);

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
            graph.RequireRunnableOps("Eval(...).With");
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
