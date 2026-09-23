using System;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
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
using Shorokoo.Core.Backends;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;

namespace Shorokoo.Runtime
{

    /// <summary>
    /// A compiled computation graph backed by a Shorokoo session.
    /// Created once via <see cref="ComputeContext.Compile(ComputationGraph)"/>, then invoked repeatedly
    /// via <see cref="Execute(IData[])"/> — each call only feeds new data, with zero graph
    /// rebuilding or session creation overhead.
    /// </summary>
    public class CompiledGraph : IDisposable
    {
        private readonly IShorokooSession _session;
        private readonly IShorokooBackend _backend;
        // The context that compiled this graph: its runs are this context's runs, so it takes the
        // locks on what they read and its outputs are attached to it.
        private readonly ComputeContext _owner;
        private readonly Dictionary<string, string> _onnxInputNameByOriginal;
        private readonly string[] _originalInputNames;

        internal CompiledGraph(
            IShorokooSession session,
            IShorokooBackend backend,
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
        /// <see cref="Shorokoo.Core.Backends.RunSettings"/> that overrides it for
        /// one call, so this is a default and never a ceiling.
        /// </summary>
        public RunSettings DefaultRunSettings { get; }

        /// <summary>
        /// The backend this graph was compiled on and runs on — fixed when it was compiled, since
        /// the session belongs to that backend and cannot move. Feeding it data another backend
        /// built is allowed: the session rebuilds what it must, provided the data is host-resident.
        /// </summary>
        public BackendDescription Backend => _backend.Description;

        /// <summary>True once this graph's session has been released.</summary>
        public bool IsDisposed { get; private set; }

        /// <summary>
        /// Releases the session behind this graph. A session is the expensive thing a
        /// compile produces — on a card it owns the execution provider's whole per-session state
        /// and its arena, which dwarfs any tensor the run produces — so it is released with the
        /// context that compiled it rather than left to a finalizer. Disposing twice is harmless,
        /// and running a disposed graph is refused rather than answered.
        /// </summary>
        public void Dispose()
        {
            if (IsDisposed) return;
            IsDisposed = true;
            _session.Dispose();
            GC.SuppressFinalize(this);
        }

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
            // Before the work, not after it. The outputs are attached to the compiling context as
            // they are wrapped, so a disposed one threw from inside the wrap of output 0 -- with the
            // native run already paid for, on a card a whole step's allocation, and outputs 1..n
            // never wrapped and so left to their finalizers. The exception also named the context
            // rather than the graph the caller had actually invoked.
            ObjectDisposedException.ThrowIf(_owner.IsDisposed, this);
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            ArgumentNullException.ThrowIfNull(runSettings);
            // Before the feeds, which is the only place it can be checked and still mean anything.
            // The backend refuses an already-cancelled run too, but by then every feed has been
            // locked, every value built and every donated tensor consumed -- so the caller who
            // caught the cancellation and meant to retry had nothing left to retry with.
            runSettings.CancellationToken.ThrowIfCancellationRequested();
            var sessionInputs = new Dictionary<string, IShorokooTensorValue>();
            var feeds = new RunFeeds(_owner, inputs.Length,
                () => TensorDeath.ConsumedBy(
                    ComputeContext.DescribeGraph(_originalInputNames, _session.OutputNames), _backend.Description));
            _owner.EnterRun();
            try
            {
                foreach (var input in inputs)
                {
                    var onnxName = _onnxInputNameByOriginal.TryGetValue(input.ParamName, out var mapped)
                        ? mapped : input.ParamName;
                    // On this graph's own backend, because that is the runtime about to read the
                    // value. An input that already holds one hands it over whatever is passed; one
                    // held in plain managed memory -- every literal in the program -- builds it
                    // here, and this is what decides which runtime builds it. The input is held
                    // first -- read-locked, or consumed if it was donated -- and the value built
                    // after.
                    sessionInputs[onnxName] = feeds.Feed(input, _backend);
                }
                ComputeContext.RefuseUnleasedFeed(feeds.Held, inputs.Length);

                using var eviction = ComputeContext.LinkEvictions(feeds.Leases, runSettings.CancellationToken);
                var settings = eviction is null
                    ? runSettings : runSettings with { CancellationToken = eviction.Token };

                IReadOnlyList<IShorokooTensorValue> results;
                // Either side of the native call and nothing else: the arena figures are about
                // what the run allocates, and the wrapping above allocates nowhere near it.
                var arenaBefore = _owner.StartRunStats(_session);
                try
                {
                    results = retainedOutputNames is null
                        ? _session.Run(sessionInputs, _session.OutputNames, settings)
                        : _session.RunRetainingOutputs(
                            sessionInputs, _session.OutputNames, retainedOutputNames, settings);
                }
                catch (OperationCanceledException stopped)
                    when (ComputeContext.StoppedByCaller(stopped, runSettings.CancellationToken))
                {
                    throw new OperationCanceledException(
                        stopped.Message, stopped.InnerException, runSettings.CancellationToken);
                }
                finally
                {
                    // However the run ended. A run that failed for want of memory is the one whose
                    // figures are worth most, so it is recorded like any other.
                    _owner.FinishRunStats(_session, arenaBefore);
                }

                return _owner.AdoptOutputs(results, _session.OutputNames, _backend);
            }
            finally
            {
                // However the run ends. What the run holds outlives the native call by
                // construction: it is given up here and nowhere else, so a terminated run's feeds
                // are still held right up to the moment it gives up, and a consumed one's memory
                // is released only once the run can no longer read it.
                feeds.Dispose();
                _owner.ExitRun();
            }
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

        /// <summary>
        /// Where this graph's session produces its outputs, which on a GPU backend is the one
        /// signal for "did part of this graph run on the host" that costs nothing: the session
        /// already knows, so there is no profiling and no extra run behind this.
        ///
        /// <para><see cref="SessionOutputPlacement.Mixed"/> on a GPU backend says outright that
        /// some of this graph ran on the host and its results crossed the bus to get back;
        /// <see cref="SessionOutputPlacement.Host"/> on one says all of it did. For <i>which</i>
        /// nodes, and what they cost, see <see cref="ReadNodePlacement"/> — which is not free.</para>
        /// </summary>
        public SessionOutputPlacement OutputPlacement => _session.OutputPlacement;

        /// <summary>
        /// This session's own memory arena, as its runtime reports it, or <c>null</c> on a backend
        /// that reports none. Unlike <see cref="Shorokoo.Core.Backends.DeviceMemory"/>,
        /// which reads the whole device across every process on it, this is this session's
        /// allocator and nobody else's.
        ///
        /// <para>It is a reading, so it costs a call into the backend and nothing is remembered.
        /// For a record per run, folded as the runs happen, set
        /// <see cref="DiagnosticSettings.CollectRunStatistics"/> on the compiling context and read
        /// <see cref="ComputeContext.RunStats"/>.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This graph's session has been released.</exception>
        public ArenaStatistics? ReadArenaStatistics()
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _session.ReadArenaStatistics();
        }

        /// <summary>
        /// The pinned host arena this session's execution provider stages its host-device crossings
        /// through, or <c>null</c> on a backend that has no such arena — every CPU one, which
        /// stages nothing.
        ///
        /// <para>Separate from <see cref="ReadArenaStatistics"/> rather than added into it: these
        /// are bytes of host memory the provider pinned, not bytes of the device, and a graph ORT
        /// gave partly to the host pays here for every output that crosses back. A CUDA session
        /// whose graph never crosses answers with a record of zeros, which is the arena saying it
        /// was never asked for anything.</para>
        ///
        /// <para>A reading, like its sibling: a call into the backend, nothing remembered.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This graph's session has been released.</exception>
        public ArenaStatistics? ReadPinnedArenaStatistics()
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _session.ReadPinnedArenaStatistics();
        }

        /// <summary>
        /// Which execution provider ran each node of this graph, with the bytes each moved — or
        /// <c>null</c> unless the context that compiled this graph carried
        /// <see cref="DiagnosticSettings.TraceNodePlacement"/>, which is off by default because
        /// recording costs every run the session makes.
        ///
        /// <para><b>Reading it stops the recording.</b> The trace covers every run made up to this
        /// call, runs after it are not recorded, and a second read hands back the same trace. So
        /// call it once, after the runs you are asking about.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This graph's session has been released.</exception>
        public NodePlacement? ReadNodePlacement()
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _session.ReadNodePlacement();
        }

        /// <summary>How many outputs this graph's session produces — the length
        /// <see cref="Execute(IData[], bool[])"/> requires of a retention array, so a caller can
        /// size one without deriving the count a second way and disagreeing.</summary>
        public int OutputCount => _session.OutputNames.Count;
    }

    /// <summary>
    /// The runtime that turns a <see cref="ComputationGraph"/> into a session and runs it —
    /// once via <see cref="Execute(ComputationGraph, IData[])"/>, or repeatedly via a
    /// <see cref="CompiledGraph"/> from <see cref="Compile(ComputationGraph)"/>.
    ///
    /// A context may name the backend it runs on, and two contexts may name different ones — a CPU
    /// context and a CUDA context in one process, each compiling and running on its own device:
    /// <code>
    /// var cpu  = new ComputeContext(new LinuxCpuBackend());
    /// var cuda = new ComputeContext(new LinuxGpuBackend());
    /// cpu.Execute(graph, input);   // on the host
    /// cuda.Execute(graph, input);  // the same graph, on the card
    /// </code>
    /// A context constructed without one runs on the process default
    /// (<see cref="Shorokoo.Core.Backends.DefaultBackend.Instance"/>), which is what
    /// every context did before backends could differ. Which device a context will use is read off
    /// <see cref="Backend"/>.
    ///
    /// <para>It also carries how its sessions and runs are configured — <see cref="DeviceMemory"/>
    /// for the arena each session it compiles is built with, and <see cref="RunSettings"/> for what
    /// its runs do by default. Both are per instance, so two contexts may differ and neither
    /// reaches the other's sessions.</para>
    ///
    /// <para>The same data feeds either context and the same model runs on both, with nothing to
    /// say at the call site. A literal costs nothing to share: it is managed bytes until something
    /// runs (<see cref="Shorokoo.HostTensorData{T}"/>), and the context that feeds it to a session
    /// is the one that materialises it, on its own backend. What a context cannot do is change
    /// where an existing runtime value lives — a tensor another backend produced is that backend's,
    /// so a session here converts it as it is fed and hands its own outputs back. That conversion
    /// costs a host copy per feed and is possible only for data the host can read; see
    /// <see cref="Shorokoo.Core.Backends.BackendTransfer"/>. <see cref="TensorData.To"/> puts a
    /// tensor where this context can read it once, instead.</para>
    ///
    /// <para><b>A context does not own tensors.</b> It keeps a weak list of the tensors attached to
    /// it — its runs' outputs and inputs, and what <see cref="TensorData.To"/> and
    /// <see cref="TensorData.CopyTo"/> placed for it — for its own accounting, and
    /// <see cref="Detach"/> takes one off. Attachment never keeps a tensor alive and never ends one's
    /// life: disposing a context releases what the context itself holds, its compiled sessions,
    /// and leaves every tensor as it was.</para>
    /// </summary>
    public class ComputeContext : IDisposable
    {
        private static volatile ComputeContext? _defaultComputeContext;
        private static readonly object _defaultGate = new();

        // Null means the default backend, read when the work runs rather than at construction: a
        // context built before DefaultBackend.Instance was assigned must still honour it.
        private readonly IShorokooBackend? _backend;

        // Counts reads of Default made inside a CountInstanceReads call, and nothing else. It is an
        // AsyncLocal rather than a static counter because the callers that care run in parallel
        // with the rest of a test suite: a process-wide count would also see everything else
        // running at the time. An AsyncLocal flows into whatever the measured call does — a thread
        // it starts included — and no further; the box is what makes an increment made down there
        // visible back up here.
        private static readonly AsyncLocal<StrongBox<int>?> _instanceReads = new();

        /// <summary>
        /// Runs <paramref name="work"/> and returns how many times it read <see cref="Default"/>.
        ///
        /// <para>The seam a test needs to hold the graph-building path to "requires no inference
        /// backend". It counts <i>asks</i> rather than backends resolved, because a resolution is
        /// unobservable in the process that can observe anything: a host with a backend loaded —
        /// every test host — answers a wrongly eager read in silence, and the failure shows up only
        /// where no backend was deployed, which is exactly the program that describes a model and
        /// exports it as ONNX rather than running it. Reads that hit the cached default count too:
        /// the question is whether the path reached for the default at all, not whether this
        /// particular process had already paid for one.</para>
        /// </summary>
        internal static int CountInstanceReads(Action work)
        {
            var outer = _instanceReads.Value;
            var counter = new StrongBox<int>(0);
            _instanceReads.Value = counter;
            try { work(); }
            finally { _instanceReads.Value = outer; }
            return Volatile.Read(ref counter.Value);
        }

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
        /// <para>Reading this resolves a backend, and refuses — naming the packages to
        /// deploy — when there is none. So it belongs at the point work actually runs: a
        /// <c>compute ??= ComputeContext.Default</c> at the top of a graph pass turns that whole
        /// pass into a backend requirement, including for the graphs it has nothing to execute
        /// for. Thread the nullable context through and resolve it where the execution is.</para>
        /// </summary>
        public static ComputeContext Default
        {
            get
            {
                if (_instanceReads.Value is { } counter) Interlocked.Increment(ref counter.Value);
                // A disposed one is not handed back. Default is a cached singleton and is now
                // disposable, so `using var ctx = ComputeContext.Default;` would otherwise poison
                // the process: every later read returns the same dead object and every run through
                // it throws, with no way back short of assigning the setter.
                // Under a lock, and the field is volatile: the getter now resolves a backend and
                // constructs a context, so two threads racing it each handed their caller a
                // different default -- process-wide lazily initialized state, in a suite that runs
                // four tests at once.
                if (_defaultComputeContext is { IsDisposed: false } live) return live;
                lock (_defaultGate)
                {
                    if (_defaultComputeContext is { IsDisposed: false } bound) return bound;

                    // The backend a process loaded, under the rule that a CPU one wins: the unnamed
                    // default should not be the card. Reading DefaultBackend.Instance is what
                    // discovers and records one when nothing has been loaded yet, and what refuses
                    // -- naming the packages to deploy -- when there is nothing to discover.
                    var backend = DefaultBackend.Remembered ?? DefaultBackend.Instance;
                    return _defaultComputeContext = new ComputeContext(backend);
                }
            }

            set { lock (_defaultGate) _defaultComputeContext = value; }
        }

        /// <summary>
        /// The framework's own host memory, as a context: a name for host memory, to be a target of
        /// <see cref="TensorData.To"/> and <see cref="TensorData.CopyTo"/> like any other context.
        ///
        /// <para>It holds nothing and runs nothing. No tensor is attached to it — its list is always
        /// empty, having no budget to keep — and <see cref="Compile(ComputationGraph)"/>,
        /// <see cref="Execute(ComputationGraph, IData[])"/>,
        /// <see cref="Run(ComputationGraph, NamedModelParam[])"/> and <c>Eval</c> all throw,
        /// naming a real context as the fix. It deliberately does <i>not</i> fall back to the
        /// process-wide backend: that would put back the implicit resolution that made merely
        /// describing a graph require a deployed backend.</para>
        ///
        /// <para>Its backend can read any host memory, whichever runtime allocated it, so
        /// <c>To(ComputeContext.Host)</c> hands a host-readable tensor back as it is and copies
        /// anything else into the framework's own managed memory — which is what
        /// <see cref="TensorData.ToHost"/> does. It cannot be disposed: <see cref="Dispose"/> does
        /// nothing and <see cref="IsDisposed"/> is always false.</para>
        /// </summary>
        public static ComputeContext Host { get; } = new(HostBackend.Instance, isHost: true);

        /// <summary>Creates a compute context that runs on the process-wide
        /// <see cref="Shorokoo.Core.Backends.DefaultBackend.Instance"/>, on the
        /// shipped defaults. Set <see cref="DeviceMemory"/> or <see cref="RunSettings"/> in an
        /// object initializer to compile and run under something else.</summary>
        public ComputeContext()
        {
        }

        private readonly DeviceMemorySettings _deviceMemory = DeviceMemorySettings.Default;

        /// <summary>
        /// The arena settings every session this context compiles from now on is built with.
        /// ONNX Runtime reads them while a session is being created and that session keeps them
        /// for life, so this configures the sessions to come and never the ones already built —
        /// to run a graph under a different budget, compile it on a context that carries one.
        ///
        /// <para>It also bounds what can be <i>placed</i> in this context's memory. A tensor copied
        /// onto its card — <see cref="TensorData.CopyTo"/>, <see cref="TensorData.To"/>,
        /// <see cref="AllocateUninitialized(Shape, DType)"/> — is allocated out of an arena built
        /// with these same settings, so a <see cref="DeviceMemorySettings.LimitBytes"/> here is a
        /// ceiling on the tensors as well as on the sessions, and what those tensors are actually
        /// holding is reported by <see cref="ReadTransferArenaStatistics"/>. A tensor is charged to
        /// the arena it was allocated from, once: handing it to a second context that can read it
        /// where it is copies nothing, so there is nothing there to re-charge.</para>
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
        /// <see cref="Shorokoo.Core.Backends.RunSettings"/> to override it for one
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

        private readonly DiagnosticSettings _diagnostics = DiagnosticSettings.Default;

        // Non-null exactly when this context was asked to collect run statistics. Built here
        // rather than in the constructor because the setting arrives through an object
        // initializer, which runs after it -- an init accessor is still construction, so the
        // field stays readonly and no later assignment can turn collection on or off under a run.
        private readonly RunStatisticsCollector? _runStatistics;

        /// <summary>
        /// What this context records about the sessions it compiles and the runs they make.
        /// Everything in it is off by default, and a context that leaves it alone pays nothing:
        /// no figures are read, no session is built differently, and <see cref="RunStats"/> stays
        /// empty.
        ///
        /// <para><see cref="DiagnosticSettings.TraceNodePlacement"/> is read when a session is
        /// built, exactly as <see cref="DeviceMemory"/> is, so a graph already compiled keeps what
        /// its context carried then — to trace a graph's node placement, compile it on a context
        /// that asks for it. <see cref="DiagnosticSettings.CollectRunStatistics"/> settles with the
        /// context itself, so <see cref="RunStats"/> covers every run this context ever makes,
        /// sessions compiled later included.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException">A null settings object.</exception>
        public DiagnosticSettings Diagnostics
        {
            get => _diagnostics;
            init
            {
                _diagnostics = value ?? throw new ArgumentNullException(nameof(value));
                _runStatistics = value.CollectRunStatistics
                    ? new RunStatisticsCollector(value.RecentRunCapacity)
                    : null;
            }
        }

        /// <summary>
        /// What every run this context has made did to the memory of the sessions it ran in.
        /// Empty unless <see cref="Diagnostics"/> carries
        /// <see cref="DiagnosticSettings.CollectRunStatistics"/>, and empty too on a backend that
        /// reports no arena figures.
        ///
        /// <para>A snapshot: the aggregates and the retained window as they stand at this read,
        /// unaffected by runs that come after it.</para>
        /// </summary>
        public RunStatistics RunStats => _runStatistics?.Snapshot() ?? RunStatistics.Empty;

        /// <summary>
        /// The arena the tensors this context has placed in its backend's own memory came out of —
        /// the other half of its device footprint, beside the sessions
        /// <see cref="CompiledGraph.ReadArenaStatistics"/> reports.
        ///
        /// <para><c>null</c> when there is nothing to report: a backend with no device memory, or
        /// a device nothing has yet been placed on <i>under these settings</i> — by this context
        /// or by any other. It fills as soon as one of them places a tensor, because the arena is
        /// keyed on the device and the settings rather than on the context: it is built with
        /// <see cref="DeviceMemory"/>, so <see cref="ArenaStatistics.LimitBytes"/> is this
        /// context's own budget, and every context naming the same budget on the same card reads
        /// this same arena and shares that one ceiling between them. Settings differing only by
        /// <see cref="ArenaExtendStrategy.Auto"/> against what it resolves to are the same
        /// settings here.</para>
        ///
        /// <para>A backend written before budgets existed reports <c>null</c> too, having no
        /// budgeted arena to answer for — so <c>null</c> does not on its own distinguish "nothing
        /// placed yet" from "this backend does not honour a budget".</para>
        ///
        /// <para>It is a reading, so it costs a call into the backend and nothing is
        /// remembered.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        public ArenaStatistics? ReadTransferArenaStatistics()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return ResolvedBackend.ReadTransferArenaStatistics(DeviceMemory);
        }

        /// <summary>
        /// The arena figures <paramref name="session"/> reports before a run, or <c>null</c> when
        /// nothing is collecting or the backend reports none. Paired with
        /// <see cref="FinishRunStats"/> in a <c>finally</c>, either side of the call into the
        /// backend and nowhere else.
        /// </summary>
        internal ArenaStatistics? StartRunStats(IShorokooSession session)
            => _runStatistics is null ? null : session.ReadArenaStatistics();

        /// <summary>
        /// Folds what the run did into this context's aggregates and its recent-run ring.
        ///
        /// <para>Folded as the run finishes rather than gathered on read, because the sessions
        /// cannot be relied on to still be there: this context tracks its compiled graphs weakly,
        /// so one the program has dropped is collected and everything it did would vanish from a
        /// figure computed by walking the live ones — silently, and downwards.</para>
        /// </summary>
        internal void FinishRunStats(IShorokooSession session, ArenaStatistics? before)
        {
            if (_runStatistics is null || before is not { } start) return;
            if (session.ReadArenaStatistics() is { } end) _runStatistics.Record(start, end);
        }

        /// <summary>
        /// Creates a compute context that compiles and runs on <paramref name="backend"/>, whatever
        /// the process default is. This is how one program drives two devices: a context per
        /// backend, each with sessions of its own. The data they run on is shared — see the note on
        /// the class.
        /// </summary>
        /// <param name="backend">The backend its sessions are built by — a platform backend
        /// (<c>new LinuxGpuBackend()</c>) where one native ONNX Runtime serves both, or one
        /// from <see cref="Shorokoo.Core.Backends.IsolatedBackend.Load"/> where each
        /// backend needs a native of its own.</param>
        /// <exception cref="ArgumentNullException"><paramref name="backend"/> is null.</exception>
        public ComputeContext(IShorokooBackend backend)
            : this(backend, isHost: false)
        {
        }

        private ComputeContext(IShorokooBackend backend, bool isHost)
        {
            ArgumentNullException.ThrowIfNull(backend);
            _backend = backend;
            _isHost = isHost;
            // Eagerly, because the backend is already in hand: a context that names one is on that
            // backend's books from the moment it exists. One that names none registers when
            // something first resolves the default for it -- see ResolvedBackend -- rather than
            // resolving a backend here just to be listed.
            BackendRegistry.Attach(backend, this);
            _registeredOn = backend;
        }

        // Whether this is the host context: the one that holds tensors, compiles nothing and
        // cannot be disposed.
        private readonly bool _isHost;

        // The backend this context was last recorded against, so ResolvedBackend can enrol a
        // default-backend context without a table write per call.
        private IShorokooBackend? _registeredOn;

        // The graphs this context compiled, so its disposal releases their sessions. Weak, like
        // the tensor list and for the same reason: a graph the program has dropped must not be
        // kept alive waiting for this. A dropped one is the finalizer's, which is what it was
        // before; what this fixes is the graph the program still holds when the context goes.
        private readonly ConditionalWeakTable<CompiledGraph, object> _compiled = new();
        private static readonly object OwnedMarker = new();
        // Guards the disposal flag, the run and lock counts and the attachment list against the
        // writers that matter: a second Dispose, an attachment racing one, a lock taken as one
        // starts, and a detach racing a run's lock. All are ordinary in a design whose premise is
        // several live contexts driven at once, and the flag alone settled none of them.
        //
        // Ordering: this gate is never held while a tensor's own gate is taken, and a tensor never
        // takes this one. That is what keeps a delete -- which waits for readers holding nothing
        // at all -- from waiting on a run that is waiting on a gate the deleter holds.
        private readonly object _gate = new();
        // Volatile: read by IsDisposed from threads that never took _gate -- the Default getter's
        // liveness check among them.
        private volatile bool _disposed;

        /// <summary>Whether this context has been disposed, and so has released its sessions.
        /// Always false for <see cref="Host"/>, which cannot be disposed.</summary>
        public bool IsDisposed => _disposed;

        // The tensors attached to this context. Weak: attachment is not ownership, and a tensor the
        // program has dropped must not be kept alive by a context's accounting.
        private readonly WeakSet<TensorData> _attached = new();

        // The reader locks this context's runs hold, per tensor or sequence, by reference. What
        // Detach refuses on: a context may not let go of a tensor one of its own runs is reading.
        // Strong, and harmlessly so -- a run holds everything it reads strongly anyway, for exactly
        // as long as its lock.
        private readonly Dictionary<object, int> _locksHeld = new(ReferenceEqualityComparer.Instance);

        // How many reader locks this context holds in all. Disposing a context while it is
        // processing is invalid, and this is what makes that a refusal rather than an assumption.
        private int _leases;

        // How many runs of this context are inside a call into the backend. Separate from the lock
        // count, which answers a different question and does not cover this one: a run fed nothing
        // takes no lock, and its session is mid-call all the same -- and disposing this context is
        // what releases it.
        private int _runs;

        /// <summary>
        /// The tensors attached to this context: alive, and not collected. A snapshot: one attached
        /// after this returns is not in the list.
        ///
        /// <para>A tensor becomes attached by being an output of one of this context's runs, by
        /// being read by one, and by <see cref="TensorData.To"/> or <see cref="TensorData.CopyTo"/>
        /// with this context as the target; <see cref="Detach"/> takes one off. The list is weak and
        /// it is not ownership: it never keeps a tensor alive, never ends one's life, and a tensor
        /// that dies or is collected drops out of it. <see cref="Host"/>'s is always empty.</para>
        /// </summary>
        public IReadOnlyList<TensorData> Tensors
        {
            get
            {
                var live = new List<TensorData>();
                foreach (var tensor in _attached.Snapshot())
                    if (!tensor.IsDisposed) live.Add(tensor);
                return live;
            }
        }

        /// <summary>
        /// Attaches <paramref name="tensor"/> to this context, for its accounting. Idempotent. The
        /// host context attaches nothing: it keeps no accounts.
        /// </summary>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        internal void Attach(TensorData tensor)
        {
            if (_isHost) return;
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _attached.Add(tensor);
            }
        }

        /// <summary>
        /// Takes <paramref name="tensor"/> off this context's list. It never ends the tensor's life,
        /// never releases its memory and never affects another context's list; a tensor that is not
        /// attached is left as it is.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="InvalidOperationException">A run of this context is reading
        /// <paramref name="tensor"/>. A context may not let go of what it is in the middle of
        /// reading; wait for the run to return.</exception>
        public void Detach(TensorData tensor)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            lock (_gate)
            {
                if (_locksHeld.ContainsKey(tensor))
                    throw new InvalidOperationException(
                        $"A run of this compute context is reading tensor {tensor}, so the context "
                        + "cannot detach it until that run returns.");
                _attached.Remove(tensor);
            }
        }

        /// <summary>Whether this context's backend can read <paramref name="tensor"/>'s memory as
        /// it stands — the question <see cref="TensorData.To"/> asks, answered by the
        /// backend.</summary>
        internal bool CanAddress(TensorData tensor) => ResolvedBackend.CanAddress(tensor.Location);

        /// <summary>The memory this context's tensors live in, shared with every other context
        /// whose backend allocates in the same place.</summary>
        public MemoryDevice Device => MemoryDevice.Of(ResolvedBackend);

        /// <summary>
        /// A tensor of <paramref name="shape"/> and <paramref name="dtype"/> in this context's
        /// memory with nothing written into it — the buffer holds whatever was last there, and the
        /// caller fills it in place through <c>AccessModifiableMemory</c>.
        ///
        /// <para>This is the tensor a producer wants. Building one from a managed array copies it
        /// into the runtime's buffer, so the tensor exists twice for as long as the caller holds
        /// the array it was built from — and for a feed built fresh per step that array is the
        /// whole input. Filling the runtime's buffer directly never has the second copy at all
        /// (Shorokoo/Shorokoo#359). It is not a way to wrap a managed array you already have:
        /// the buffer stays the runtime's, which is what lets it be released like every other
        /// tensor the runtime hands back.</para>
        ///
        /// <para>On <see cref="Host"/> the buffer is a managed array, since that is what the
        /// framework's own host memory is; on a real backend it is the memory that backend
        /// allocates in, which on a CUDA one is the card's and so is not writable through a span
        /// at all. <see cref="TensorData.IsHostResident"/> says which. Either way the tensor is
        /// attached to this context, as a <see cref="TensorData.CopyTo"/> result is.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is
        /// <see cref="DType.Utf8"/>, whose elements are variable-length, or has no whole-byte
        /// element stride, or <paramref name="shape"/> has no known element count.</exception>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        public TensorData AllocateUninitialized(Shape shape, DType dtype)
        {
            ArgumentNullException.ThrowIfNull(dtype);
            ObjectDisposedException.ThrowIf(_disposed, this);
            // Refused here rather than left to the backend, so the same dtype is refused in the
            // same words wherever it is asked for -- and so that a shape with no element count is
            // caught while it can still be said what is wrong with it.
            if (dtype == DType.Utf8)
                throw new NotSupportedException(
                    "String tensors are variable-length and not byte-stride, so there is no buffer "
                    + "of a fixed size to allocate. Build one from its elements with "
                    + "TensorData(dims, string[]).");
            var bits = dtype.EncodingBitCount;
            if (bits < 8 || shape.Count < 0)
                throw new NotSupportedException(
                    $"A tensor of {shape}:{dtype} cannot be allocated as a flat buffer: its "
                    + "elements have no whole-byte stride, or its shape has no known element count.");

            if (_isHost)
                return TensorData.NewHostTensor(shape, dtype, new byte[checked(shape.Count * (bits / 8))]);

            var backend = ResolvedBackend;
            var value = backend.CreateUninitializedTensorInBackendMemory(
                (ShorokooTensorElementType)(int)dtype, (long[])shape, DeviceMemory);
            TensorData allocated;
            try
            {
                allocated = TensorData.Create(shape, dtype, value, backend);
            }
            catch
            {
                // Nothing else names it yet, and on a card it is a device allocation that would
                // otherwise sit on the finalizer queue.
                backend.Release(value);
                throw;
            }
            try
            {
                Attach(allocated);
            }
            catch
            {
                allocated.Delete();
                throw;
            }
            return allocated;
        }

        /// <summary>
        /// <see cref="AllocateUninitialized(Shape, DType)"/> typed, so the result can be filled
        /// without a cast:
        /// <c>context.AllocateUninitialized&lt;float32&gt;(new Shape(64L, 768L)).WriteMemory&lt;float&gt;(dst =&gt; …)</c>.
        /// <see cref="Shape"/> is not a collection type, so a bare <c>[64L, 768L]</c> literal does
        /// not convert to it; pass <c>new Shape(...)</c> or a <c>long[]</c>.
        ///
        /// <para>Fill it through <see cref="TensorData{T}.WriteMemory{V}"/> rather than by taking
        /// a bare <c>AccessModifiableMemory</c> span. On a real backend the buffer is the
        /// runtime's, and the tensor is the only thing keeping it alive: taking the span is the
        /// tensor's last read, so a fill written as one expression has no reachable tensor for its
        /// whole duration and writes into a block the finalizer may already have handed back.</para>
        /// </summary>
        /// <exception cref="NotSupportedException">The element type has no flat byte
        /// buffer — see the overload above.</exception>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        public TensorData<T> AllocateUninitialized<T>(Shape shape) where T : IVarType
            => (TensorData<T>)AllocateUninitialized(shape, OnnxUtils.GetDType<T>());

        /// <summary>
        /// Takes a reader lock on <paramref name="tensor"/> for a run of this context, attaches the
        /// tensor to this context, and hands back the lease to drop when the run is done. While a
        /// lease is outstanding the tensor cannot be deleted — a delete is refused or declined, and
        /// a deliberate one signals <see cref="TensorLease.Eviction"/> and waits — and this context
        /// cannot detach it. The lease holds the tensor itself, so nothing a run reads can be
        /// collected under it.
        ///
        /// <para>Any context may lock any tensor: the lock is the reading context's, and lives on
        /// the tensor. It is a count, not a flag — the same tensor fed twice under two names in one
        /// run is locked twice and released twice, and two runs may read one tensor at once.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This context has been disposed, or the tensor
        /// is dead.</exception>
        internal TensorLease Lock(TensorData tensor)
        {
            ArgumentNullException.ThrowIfNull(tensor);
            // The two gates are taken one after the other rather than nested, in either direction:
            // the count is this context's business and the lock is the tensor's, and a delete
            // waiting for readers must never find itself behind a gate a run is queued on.
            // A read attaches the reader, as a run's output does: a tensor this context's runs
            // read is one its accounting has to see. Attached with the count, under the one gate;
            // a lock then refused leaves at worst a dead tensor on the list, which the list skips.
            CountLock(tensor, attach: true);
            try
            {
                return new TensorLease(this, tensor, tensor.AcquireReadLock());
            }
            catch
            {
                ReleaseLease(tensor);
                throw;
            }
        }

        /// <summary><see cref="Lock(TensorData)"/> for a sequence a run of this context is
        /// reading.</summary>
        /// <exception cref="ArgumentNullException"><paramref name="sequence"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This context, or the sequence, has been
        /// disposed.</exception>
        internal TensorLease Lock(TensorDataSequence sequence)
        {
            ArgumentNullException.ThrowIfNull(sequence);
            CountLock(sequence, attach: false);
            try
            {
                sequence.AcquireReadLock();
                return new TensorLease(this, sequence);
            }
            catch
            {
                ReleaseLease(sequence);
                throw;
            }
        }

        /// <summary>Counts one more lock of this context's on <paramref name="target"/>, and
        /// attaches it if it is a tensor being read.</summary>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        private void CountLock(object target, bool attach)
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _leases++;
                _locksHeld[target] = _locksHeld.TryGetValue(target, out var held) ? held + 1 : 1;
                if (attach && !_isHost) _attached.Add((TensorData)target);
            }
        }

        /// <summary>
        /// Refuses a run that did not hold every input it was given — by a reader lock, or by
        /// consuming it.
        ///
        /// <para>Checked here rather than only in a test, because there is nothing behind a lock:
        /// past asking the backend to stop there is no further escalation, so a feed path that
        /// forgets one is Shorokoo/Shorokoo#366 again with the machinery sitting unused beside it.
        /// The locker is this repository's own run path rather than a caller, which is exactly what
        /// makes the count checkable at all.</para>
        ///
        /// <para>Against the inputs the run was handed, and deliberately not against a counter the
        /// feed loop keeps beside the lock with no branch in between, which would agree with the
        /// lock count by construction and could not fail. What actually enforces the rule for a
        /// kind of input nothing knows how to hold is <see cref="RunFeeds.Feed"/>'s own refusal;
        /// this catches the other shape, a feed path that learns to skip an input and its lock
        /// with it.</para>
        /// </summary>
        internal static void RefuseUnleasedFeed(int leases, int inputs)
        {
            if (leases == inputs) return;
            throw new InvalidOperationException(
                $"This run was given {inputs} input(s) and locked {leases} of them. Every fed input "
                + "is held for the length of the run; one that is not can have its memory freed "
                + "under the run by another thread.");
        }

        /// <summary>
        /// One signal for every tensor this run has locked, plus whatever the caller asked to
        /// stop the run with. Null when there is nothing to link — the caller's settings then go
        /// to the backend untouched.
        ///
        /// <para>This is how a locker discharges its one obligation: the backend is handed the
        /// linked token as <c>RunSettings.CancellationToken</c>, so a deliberate delete of
        /// anything this run is reading asks the run to stop. The correctness of the delete does
        /// not rest on it — the memory waits for the lock either way — so a backend that ignores
        /// the token makes a delete slow and never unsafe.</para>
        /// </summary>
        internal static CancellationTokenSource? LinkEvictions(
            IReadOnlyList<TensorLease> leases, CancellationToken caller)
        {
            var signals = new List<CancellationToken>(leases.Count + 1);
            foreach (var lease in leases)
                if (lease.Eviction.CanBeCanceled) signals.Add(lease.Eviction);
            if (signals.Count == 0) return null;
            if (caller.CanBeCanceled) signals.Add(caller);
            return CancellationTokenSource.CreateLinkedTokenSource([.. signals]);
        }

        /// <summary>
        /// Whether a stopped run was stopped by what its caller asked with, rather than by an
        /// eviction the run linked in. The backend is handed the linked signal, so what it throws
        /// carries a token the caller has never seen — and
        /// <see cref="Shorokoo.Core.Backends.RunSettings.CancellationToken"/>
        /// promises the caller's own back, which is what lets a program racing several runs tell
        /// which cancellation stopped this one.
        /// </summary>
        internal static bool StoppedByCaller(OperationCanceledException stopped, CancellationToken caller)
            => caller.IsCancellationRequested && stopped.CancellationToken != caller;

        /// <summary>Records that one of this context's locks on <paramref name="target"/> has been
        /// dropped. Called by <see cref="TensorLease.Dispose"/>, and by a lock refused after it was
        /// counted.</summary>
        internal void ReleaseLease(object target)
        {
            lock (_gate)
            {
                _leases--;
                if (_locksHeld.TryGetValue(target, out var held) && held > 1) _locksHeld[target] = held - 1;
                else _locksHeld.Remove(target);
            }
        }

        /// <summary>
        /// Records that a run of this context has started, so that disposing it is refused until
        /// the run returns. Paired with <see cref="ExitRun"/> in a <c>finally</c>, in both run
        /// paths and nowhere else.
        /// </summary>
        /// <exception cref="ObjectDisposedException">This context has been disposed, so its
        /// sessions are already gone and there is nothing left to run on.</exception>
        internal void EnterRun()
        {
            // The host context runs nothing -- Compile, Execute and Run all refuse there -- and
            // cannot be disposed, so there is no question here for a count to answer.
            if (_isHost) return;
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _runs++;
            }
        }

        /// <summary>Records that a run of this context has returned, however it ended. Called from
        /// the <c>finally</c> that pairs with <see cref="EnterRun"/> and nowhere else.</summary>
        internal void ExitRun()
        {
            if (_isHost) return;
            lock (_gate) _runs--;
        }

        /// <summary>
        /// Releases what this context itself holds — the sessions it compiled — and nothing else.
        /// The backend is left alone, and so is every tensor: a context is not a tool for deleting
        /// tensors, and one attached to it lives on exactly as it would have, released through its
        /// own backend when it is deleted or collected. Its list of attached tensors is cleared.
        /// </summary>
        /// <exception cref="InvalidOperationException">A run of this context is in flight, or a
        /// lock it took is still held. Disposing a context while it is processing is invalid; wait
        /// for the run to return.</exception>
        public void Dispose()
        {
            // The host context is not disposable, and this is the whole of it: it holds nothing and
            // compiles nothing, so there is nothing here whose release anyone could be waiting for,
            // and `using var c = ComputeContext.Host;` must not turn it into a dead name.
            if (_isHost) return;

            List<CompiledGraph> compiled;
            lock (_gate)
            {
                if (_disposed) return;
                // Before the flag, so a refusal leaves a context that still works. Two refusals,
                // because a run reading nothing locks nothing and its session is mid-call all the
                // same -- a use-after-free of the session rather than of any tensor, and it says so.
                if (_runs > 0)
                    throw new InvalidOperationException(
                        $"This compute context has {_runs} run(s) in flight: disposing it would "
                        + "release the session they are inside, under a live call into "
                        + "the backend. Wait for the run to return.");
                if (_leases > 0)
                    throw new InvalidOperationException(
                        $"This compute context still holds {_leases} lock(s): a run of it is "
                        + "reading tensors, and disposing it under that read is not something it "
                        + "can answer for. Wait for the run to return.");
                _disposed = true;

                compiled = [.. _compiled.Select(entry => entry.Key)];
                _compiled.Clear();
                _attached.Clear();
            }

            // Outside the gate: a session's disposal is a native call into the backend, and has no
            // business running under a gate every attachment to this context takes.
            foreach (var graph in compiled) graph.Dispose();

            // The backend is deliberately left alone. It was handed in, so it may be shared with
            // another context or be the process-wide one -- disposing a backend two contexts were
            // given would kill the second, which is the arrangement this whole design is for.

            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// A run's outputs as the tensors and sequences the caller gets back, each allocated by
        /// <paramref name="backend"/> — the backend the run ran on, and so the one that releases
        /// it — and every tensor among them attached to this context.
        /// </summary>
        internal NamedModelParam[] AdoptOutputs(
            IReadOnlyList<IShorokooTensorValue> results, IReadOnlyList<string> names,
            IShorokooBackend backend)
        {
            var outputs = new NamedModelParam[results.Count];
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = OnnxUtils.CreateNamedModelParam(
                    results[i], ModelParamType.OutputParam, names[i], backend);
                if (outputs[i] is TensorDataModelParam tensor) Attach(tensor.ToTensorData());
            }
            return outputs;
        }

        /// <summary>
        /// Names a graph for a message about a run of it: by its inputs and outputs, which is what a
        /// caller can recognise it by — graphs carry no name of their own.
        /// </summary>
        internal static string DescribeGraph(IReadOnlyList<string> inputs, IReadOnlyList<string> outputs)
            => $"the graph ({string.Join(", ", inputs)}) -> ({string.Join(", ", outputs)})";

        /// <summary>The backend this context's work runs on: the one it was constructed with, or
        /// the default when it names none.</summary>
        internal IShorokooBackend ResolvedBackend
        {
            get
            {
                if (_backend is { } named) return named;
                var backend = DefaultBackend.Instance;
                // A context that named no backend is on the default one's books from the first
                // time anything resolves it. Guarded by the last backend seen rather than written
                // every time: this is read once per feed, and a table write per feed would be a
                // process-wide lock on the hot path. The default can be reassigned, which is why
                // the guard compares rather than latching.
                if (!ReferenceEquals(_registeredOn, backend))
                {
                    BackendRegistry.Attach(backend, this);
                    _registeredOn = backend;
                }
                return backend;
            }
        }

        /// <summary>
        /// Refuses the host context, which holds tensors and runs nothing.
        ///
        /// <para>It refuses rather than forwarding to <see cref="DefaultBackend.Instance"/>. A
        /// host context that quietly resolved the process-wide backend would put back the implicit
        /// resolution that made describing a graph require a deployed runtime — the thing giving a
        /// tensor a context was for.</para>
        /// </summary>
        private void RefuseHostContext(string operation)
        {
            if (!_isHost) return;
            throw new InvalidOperationException(
                $"ComputeContext.Host holds tensors and runs nothing, so it cannot {operation} a "
                + "graph. It is the framework's own host memory, which is where a tensor that "
                + "belongs to no backend lives. Compile and run on a context over a real backend "
                + "-- new ComputeContext() takes the process-wide one, and "
                + "new ComputeContext(backend) takes the one you name.");
        }

        /// <summary>
        /// The backend this context compiles and runs on — its name, its device, and the CUDA device
        /// it allocates on. A context constructed with a backend reports that one; a context without
        /// reports the process default, which is what every context reported when only one could be
        /// live. Read it to log the device a run used, or call
        /// <see cref="Shorokoo.Core.Backends.DefaultBackend.RequireDevice"/> to
        /// refuse to start on the wrong one.
        ///
        /// <para>Reading this resolves the process default if this context names no backend and none
        /// is live yet, exactly as compiling would.</para>
        /// </summary>
        public BackendDescription Backend => ResolvedBackend.Description;

        /// <summary>Where this context's tensors live. The same space is necessary for passing a
        /// tensor between two contexts without copying it, and sufficient only on the host: a
        /// device allocation means nothing to a runtime that did not make it, so two contexts on
        /// one card share it when they share a backend's runtime and copy through the host when
        /// they do not.</summary>
        public MemorySpace MemorySpace => ResolvedBackend.MemorySpace;

        /// <summary>
        /// Compiles the graph into a reusable <see cref="CompiledGraph"/>: the ONNX model and
        /// session are built once, so repeated executions only feed new data.
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
        /// session per call (disposed afterwards); use <see cref="Compile(ComputationGraph)"/>
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
            RefuseHostContext("compile");
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

            var graph = new CompiledGraph(
                session, ResolvedBackend, onnxInputNameByOriginal, originalInputNames, optimization,
                deviceMemory, RunSettings, this);
            // Enrolled under the same gate a disposal takes, so a compile racing a disposal either
            // lands before it and is released with everything else, or finds the context gone.
            lock (_gate)
            {
                if (_disposed)
                {
                    graph.Dispose();
                    throw new ObjectDisposedException(GetType().Name);
                }
                _compiled.AddOrUpdate(graph, OwnedMarker);
            }
            return graph;
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
        ///
        /// <para>The results outlive this context, as every run's outputs do, and can go straight
        /// back into a graph as literals through <see cref="TensorData.MoveToAttribute"/>.</para>
        /// </summary>
        public TensorData[] Eval(Variable[] outputs)
        {
            var graph = new InternalComputationGraph([], [.. outputs]);
            graph.RequireRunnableOps("ComputeContext.Eval");
            return this.Execute(graph).Select(x => x.ToTensorData()).ToArray();
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
        /// session per call (disposed afterwards); use <see cref="Compile(ComputationGraph)"/> for repeated runs.
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
            // Before the work, for the reason CompiledGraph.Run refuses before its own: the outputs
            // are handed to this context as they are wrapped, so a disposed one threw from inside
            // the wrap of output 0 with the native run already paid for -- on a card a whole step's
            // allocation -- and outputs 1..n never wrapped and so left to their finalizers.
            RefuseHostContext("run");
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            // Before the model is built, the session created and the feeds held -- see
            // CompiledGraph.Run. This path pays for a whole model build and a session on top.
            RunSettings.CancellationToken.ThrowIfCancellationRequested();
            var model = buildModel();

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            IShorokooSession? session = null;
            var backend = ResolvedBackend;
            var feeds = new RunFeeds(this, inputs.Length,
                () => TensorDeath.ConsumedBy(
                    DescribeGraph(originalInputNames, session?.OutputNames ?? []), backend.Description));
            // Before the session, so that everything this context is about to build is inside the
            // window its disposal is refused in -- the session most of all, since disposing the
            // context is what would release it.
            EnterRun();
            try
            {
                session = CreateSession(
                    modelData, HasOptionalOps(model.Graph) || IsFullyConstant(model.Graph));
                var onnxInputNameByOriginal = new Dictionary<string, string>();
                for (int i = 0; i < originalInputNames.Length && i < session.InputNames.Count; i++)
                    onnxInputNameByOriginal[originalInputNames[i]] = session.InputNames[i];

                var sessionInputs = new Dictionary<string, IShorokooTensorValue>();
                foreach (var input in inputs)
                {
                    var onnxName = onnxInputNameByOriginal.TryGetValue(input.ParamName, out var mapped)
                        ? mapped : input.ParamName;
                    // Held first, then the value -- see CompiledGraph.Run -- on this context's
                    // backend: the one that just built the session above, and so the runtime that
                    // is about to read what it is fed.
                    sessionInputs[onnxName] = feeds.Feed(input, backend);
                }
                RefuseUnleasedFeed(feeds.Held, inputs.Length);

                using var eviction = LinkEvictions(feeds.Leases, RunSettings.CancellationToken);
                var settings = eviction is null
                    ? RunSettings : RunSettings with { CancellationToken = eviction.Token };
                IReadOnlyList<IShorokooTensorValue> results;
                var arenaBefore = StartRunStats(session);
                try
                {
                    results = session.Run(sessionInputs, session.OutputNames, settings);
                }
                catch (OperationCanceledException stopped)
                    when (StoppedByCaller(stopped, RunSettings.CancellationToken))
                {
                    throw new OperationCanceledException(
                        stopped.Message, stopped.InnerException, RunSettings.CancellationToken);
                }
                finally
                {
                    // Before the session goes, in the finally below: its arena is what is being
                    // read, and a disposed session has none.
                    FinishRunStats(session, arenaBefore);
                }

                // Nothing retained: this is the one-shot path, which builds a session, feeds it
                // once and disposes it, so there is no later run for a device-resident output to
                // be fed into.
                return AdoptOutputs(results, session.OutputNames, backend);
            }
            finally
            {
                // However the run ends, and before the session goes: what the run holds is given
                // up here and nowhere else, so a terminated run's feeds are still held right up to
                // the moment it gives up, and a consumed one's memory is released only once nothing
                // can read it.
                feeds.Dispose();

                // Dispose the session to free native memory — on the throwing path too, where
                // the memory it holds is the memory the caller has just been told it lacks. The
                // returned tensor values stay valid across it, and the finally also keeps the
                // session rooted across the native calls above. They are not, however, free of
                // it: a result keeps its session's ALLOCATOR alive, so a caller that retains one
                // retains that session's arena — see `FastProcessorHelper.RehostOffSession` for
                // what a caller that must not does, and Shorokoo/Shorokoo#180 for the general
                // question.
                session?.Dispose();

                // Last, so that this context is answerable for its session right up to the moment
                // the session is gone.
                ExitRun();
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
        private IShorokooSession CreateSession(byte[] modelData, bool disableOptimizations = false)
            => CreateSession(
                modelData,
                SessionOptimization(disableOptimizations, trainingStep: false),
                DeviceMemory.Resolve(reusedAcrossShapes: false));

        private IShorokooSession CreateSession(
            byte[] modelData, ShorokooGraphOptimization optimization, DeviceMemorySettings deviceMemory)
            => ResolvedBackend.CreateSession(
                modelData, optimization, ShorokooLogSeverity.Fatal, deviceMemory, Diagnostics);

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
        /// <summary>
        /// The runtime value of <paramref name="data"/>, on the process-wide default backend —
        /// <see cref="DefaultBackend.Instance"/>, resolved here if nothing has resolved one yet.
        /// Use <see cref="ToTensorValue(IData, IShorokooBackend)"/> wherever the
        /// backend that is going to read the value is known, since a value belongs to the runtime
        /// that made it.
        ///
        /// <para>The value returned belongs to <paramref name="data"/>: read it, do not dispose
        /// it.</para>
        /// </summary>
        public static IShorokooTensorValue ToTensorValue(this IData data)
            => data.ToTensorValue(DefaultBackend.Instance);

        /// <summary>
        /// The runtime value of <paramref name="data"/> as a value of
        /// <paramref name="backend"/>'s runtime, built there if it does not exist yet.
        ///
        /// <para>This used to unwrap <see cref="IOnnxData"/> and throw at everything else, which
        /// made it a hole rather than an entry point: a tensor literal is held as managed bytes
        /// (<see cref="HostTensorData{T}"/>), a string literal as managed strings
        /// (<see cref="HostStringTensorData"/>) and a copied sequence as the tensors it was
        /// copied into, and none of the three carries a runtime value until something asks for one.
        /// Asking each of them is what this does now; a value a runtime already made is still
        /// handed straight over.</para>
        ///
        /// <para>The value returned belongs to <paramref name="data"/>: read it, do not dispose
        /// it.</para>
        /// </summary>
        public static IShorokooTensorValue ToTensorValue(
            this IData data, IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            return data switch
            {
                TensorData tensor => tensor.ToTensorValue(backend),
                TensorDataSequence sequence => sequence.ToTensorValue(backend),
                // Nothing in the framework is IOnnxData without being one of the two above. A
                // caller's own IData may be, and unwrapping it is what this method promised.
                IOnnxData onnxData => onnxData.Value,
                _ => throw new UnsupportedDTypeException(
                    ErrorCodes.CR006, data?.GetType()?.Name ?? "null", "ToTensorValue",
                    "Data type is not supported for tensor value conversion"),
            };
        }

        /// <summary>Starts a fluent eager evaluation: <c>inputs.Eval(outputs).With(data)</c>.</summary>
        public static EvalTo Eval(this IEnumerable<Variable> inputTensors, params Variable[] outputTensors)
        {
            return new EvalFrom(inputTensors.ToArray()).To(outputTensors);
        }
    }
}
