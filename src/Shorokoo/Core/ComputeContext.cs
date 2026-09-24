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
        private readonly IShorokooBackend _backend;
        // The context that compiled this graph: its runs are this context's runs, so it takes the
        // locks on what they read and its outputs are attached to it.
        private readonly ComputeContext _owner;
        private readonly Dictionary<string, string> _onnxInputNameByOriginal;
        private readonly string[] _originalInputNames;

        // The session's outputs, by name, in its order -- every session this graph is ever built on
        // is the same model's, so they are taken once rather than asked of whichever is current.
        private readonly string[] _outputNames;

        // The session this graph runs on and what it was built with. Replaced only on a context whose
        // device memory is under a budget, at the start of a run -- which the budget serializes --
        // when the run finds the context holding more of its memory than the session's arena limit
        // left room for (see Within). Everything else that reaches into the session does so under
        // _sessionGate, which the replacement takes too, so nothing calls into one being disposed.
        private volatile BuiltSession _built;
        private readonly object _sessionGate = new();

        // The model the session was built from, kept only where it may have to be built again: on a
        // context under a device-memory budget. Anywhere else a session is built once and this is
        // null, so an ordinary compile keeps no second copy of its model.
        private readonly byte[]? _model;

        // The outputs the lowering proved may be written into the memory of the inputs they are
        // paired with -- empty for a graph nothing marked -- which every session this graph is built
        // on is built with, a rebuilt one included.
        private readonly IReadOnlyList<OutputAlias> _outputAliases;

        internal CompiledGraph(
            IShorokooSession session,
            IShorokooBackend backend,
            Dictionary<string, string> onnxInputNameByOriginal,
            string[] originalInputNames,
            ShorokooGraphOptimization optimization,
            DeviceMemorySettings deviceMemory,
            RunSettings defaultRunSettings,
            ComputeContext owner,
            string? description = null,
            byte[]? model = null,
            IReadOnlyList<OutputAlias>? outputAliases = null)
        {
            _owner = owner;
            _onnxInputNameByOriginal = onnxInputNameByOriginal;
            _built = new BuiltSession(session, deviceMemory, BindableOf(session));
            _outputNames = [.. session.OutputNames];
            _backend = backend;
            _originalInputNames = originalInputNames;
            Optimization = optimization;
            DefaultRunSettings = defaultRunSettings;
            _description = description;
            _model = model;
            _outputAliases = outputAliases ?? [];
        }

        /// <summary>The outputs the lowering marked as ones a run may write into the memory of an
        /// input it consumed, by position: which output, into which input (test hook).</summary>
        internal IReadOnlyList<(int Output, int Input)> MarkedPairs()
        {
            var session = _built.Session;
            return [.. _outputAliases.Select(alias => (
                session.OutputNames.ToList().IndexOf(alias.Output),
                session.InputNames.ToList().IndexOf(alias.Input)))];
        }

        /// <summary>
        /// A session and the settings it was built with, and the arena it allocates in as the
        /// tensors a run leaves there record it: a token of its own rather than the session, so an
        /// output that outlives a rebuilt session does not keep the managed wrapper of it alive.
        /// </summary>
        private sealed class BuiltSession(
            IShorokooSession session, DeviceMemorySettings deviceMemory,
            IReadOnlyList<(string Output, string Input)> bindable)
        {
            internal IShorokooSession Session { get; } = session;

            internal DeviceMemorySettings DeviceMemory { get; } = deviceMemory;

            internal object Arena { get; } = new();

            /// <summary>
            /// The inputs, by the names the graph was compiled with, whose consumed memory a run of
            /// this session keeping <paramref name="retained"/> on the device may write an output into:
            /// those paired with an output it keeps there. An output is produced where it is kept, and
            /// can be written only into memory there; one the run fetches back can be written only
            /// into host memory.
            /// </summary>
            internal IReadOnlySet<string> WrittenInto(IReadOnlySet<string> retained)
            {
                if (bindable.Count == 0 || retained.Count == 0) return System.Collections.Frozen.FrozenSet<string>.Empty;
                HashSet<string>? inputs = null;
                foreach (var (output, input) in bindable)
                    if (retained.Contains(output)) (inputs ??= new(StringComparer.Ordinal)).Add(input);
                return inputs is null ? System.Collections.Frozen.FrozenSet<string>.Empty : inputs;
            }
        }

        /// <summary>
        /// The pairs by which a run of <paramref name="session"/> may write an output into the consumed
        /// memory of an input (<see cref="IShorokooSession.BindableAliases"/>), each input named as a
        /// run's feeds name it: by the name the graph was compiled with.
        /// </summary>
        private IReadOnlyList<(string Output, string Input)> BindableOf(IShorokooSession session)
        {
            var aliases = session.BindableAliases;
            if (aliases.Count == 0) return [];
            // A feed whose name the graph did not rename is fed under that name as it is.
            var originalOf = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (original, own) in _onnxInputNameByOriginal) originalOf[own] = original;
            return [.. aliases.Select(alias => (alias.Output, originalOf.GetValueOrDefault(alias.Input, alias.Input)))];
        }

        // What a message about a run of this graph calls it, where the compiler knew better than a
        // list of input and output names -- the training rig names its step.
        private readonly string? _description;

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
        /// built is allowed: what this backend cannot address where it is — another device's, or
        /// another runtime's — is read through a copy in memory it can, made through the host.
        /// </summary>
        public BackendDescription Backend => _backend.Description;

        private volatile bool _disposed;

        // How many of this graph's runs are between their start and their return, under
        // _sessionGate: a disposal while any is would release the session under a live call.
        private int _running;

        /// <summary>True once this graph's session has been released.</summary>
        public bool IsDisposed => _disposed;

        /// <summary>
        /// Releases the session behind this graph. A session is the expensive thing a
        /// compile produces — on a card it owns the execution provider's whole per-session state
        /// and its arena, which dwarfs any tensor the run produces — so it is released with the
        /// context that compiled it rather than left to a finalizer. Disposing twice is harmless,
        /// and running a disposed graph is refused rather than answered.
        /// </summary>
        /// <exception cref="InvalidOperationException">A run of this graph is in flight. Disposing
        /// it would release the session that run is inside; wait for the run to return.</exception>
        public void Dispose()
        {
            BuiltSession built;
            lock (_sessionGate)
            {
                if (_disposed) return;
                // Before the flag, so a refusal leaves a graph that still works -- as a context in
                // the same position refuses.
                if (_running > 0)
                    throw new InvalidOperationException(
                        $"This compiled graph has {_running} run(s) in flight: disposing it would "
                        + "release the session they are inside, under a live call into the backend. "
                        + "Wait for the run to return.");
                _disposed = true;
                built = _built;
            }
            built.Session.Dispose();
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
        ///
        /// <para>On a context whose device memory is under a budget,
        /// <see cref="DeviceMemorySettings.LimitBytes"/> here is the session's own arena limit
        /// rather than the budget: the budget less what the context held in its memory outside the
        /// session when it was built, rounded up to the next sixty-fourth of the budget so that the
        /// session is kept while that grows a little. It comes down, and never goes up, when a run
        /// finds the context holding more than the session left room for and the session is built
        /// again.</para>
        /// </summary>
        public DeviceMemorySettings DeviceMemory => _built.DeviceMemory;

        /// <summary>
        /// Executes the compiled graph with the given inputs.
        /// TensorDataStruct inputs are automatically expanded into individual fields.
        ///
        /// <para>A tensor fed as it is is <b>consumed</b>: the run takes it when it starts, and it
        /// is dead from then on, however the run ends. Feed <c>t.Shared()</c> to have it read and
        /// left alive, or <c>t.TryConsume()</c> to have it consumed only when nothing else is reading
        /// it; the same goes for a sequence, a struct or an optional. Outputs are new tensors,
        /// attached to the context that compiled this graph.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">The compiling context is under a device-memory
        /// budget, and what the run would hold in the context's memory outside its session's arena
        /// leaves nothing of the budget for the arena, or its arena cannot take what it was fed
        /// beside what the context holds (<see cref="DeviceMemorySettings.LimitBytes"/>). Nothing it
        /// was fed has been taken.</exception>
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
        /// (<see cref="TensorData.IsHostResident"/>). Only a tensor is retained: a sequence output
        /// comes back to the host however it is flagged, since its elements are read from there.
        /// Every other output comes back exactly as
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
            => Run(NameInputs(inputs), Retained(retainOnDevice), runSettings);

        /// <summary>
        /// <see cref="Execute(IData[], bool[])"/>, or <see cref="Execute(IData[])"/> when
        /// <paramref name="retainOnDevice"/> is null, with each input called
        /// <paramref name="labels"/>' entry in a message about it — for a caller whose inputs the
        /// graph names by identifiers nobody would recognise, the training step's among them — and
        /// the run called <paramref name="description"/>, where the caller runs one graph as
        /// different things: a training rig's own step and a resident run's.
        /// </summary>
        internal NamedModelParam[] Execute(
            IData[] inputs, IReadOnlyList<string> labels, bool[]? retainOnDevice, string? description = null)
        {
            var named = NameInputs(inputs);
            for (int i = 0; i < named.Length && i < labels.Count; i++) named[i].Label = labels[i];
            return Run(named, retainOnDevice is null ? null : Retained(retainOnDevice), DefaultRunSettings, description);
        }

        /// <summary>The names of the outputs <paramref name="retainOnDevice"/> flags, refusing an
        /// array that is not one flag per output.</summary>
        private HashSet<string> Retained(bool[] retainOnDevice)
        {
            if (retainOnDevice is null) throw new ArgumentNullException(nameof(retainOnDevice));
            if (retainOnDevice.Length != _outputNames.Length)
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "CompiledGraph.Execute",
                    $"retainOnDevice.Length={retainOnDevice.Length}, graph.Outputs.Count={_outputNames.Length}",
                    "Retention flag count does not match the graph's output count");

            var retained = new HashSet<string>();
            for (int i = 0; i < retainOnDevice.Length; i++)
                if (retainOnDevice[i]) retained.Add(_outputNames[i]);
            return retained;
        }

        /// <summary>
        /// Executes the compiled graph with pre-built named inputs. A parameter's data is consumed
        /// by the run unless the parameter says otherwise (<see cref="NamedModelParam.FeedMode"/>),
        /// as <see cref="ComputeContext.Run(ComputationGraph, NamedModelParam[])"/> describes.
        /// </summary>
        /// <exception cref="InvalidOperationException">The compiling context is under a device-memory
        /// budget, and what the run would hold in the context's memory outside its session's arena
        /// leaves nothing of the budget for the arena, or its arena cannot take what it was fed
        /// beside what the context holds (<see cref="DeviceMemorySettings.LimitBytes"/>). Nothing it
        /// was fed has been taken.</exception>
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
            NamedModelParam[] inputs, IReadOnlySet<string>? retainedOutputNames, RunSettings runSettings,
            string? description = null)
        {
            // Before the work, not after it. The outputs are attached to the compiling context as
            // they are wrapped, so a disposed one threw from inside the wrap of output 0 -- with the
            // native run already paid for, on a card a whole step's allocation, and outputs 1..n
            // never wrapped and so left to their finalizers. The exception also named the context
            // rather than the graph the caller had actually invoked.
            ObjectDisposedException.ThrowIf(_owner.IsDisposed, this);
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            ArgumentNullException.ThrowIfNull(inputs);
            ArgumentNullException.ThrowIfNull(runSettings);
            // Before the feeds, which is the only place it can be checked and still mean anything.
            // The backend refuses an already-cancelled run too, but by then every feed has been
            // locked, every value built and every consumed tensor taken -- so the caller who caught
            // the cancellation and meant to retry had nothing left to retry with.
            runSettings.CancellationToken.ThrowIfCancellationRequested();
            var budget = _owner.BudgetIn();
            var feeds = new RunFeeds(_owner, _backend, Identity(description), budget);
            // Under a device-memory budget this waits for any run of the context already in flight:
            // two at once would each be counting the room the other's arena is taking.
            var entered = _owner.EnterRun(budget, runSettings.CancellationToken);
            Exception? failed = null;
            var counted = false;
            try
            {
                // Counted in under the session's gate, which a disposal takes too, so a disposal
                // either sees this run and refuses, or has already released the session and this run
                // refuses instead -- now that the run may have waited for another.
                lock (_sessionGate)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    _running++;
                    counted = true;
                }

                // Everything that can refuse the run over what it is fed, before anything is taken.
                feeds.Prepare(inputs);

                // Under a budget, the session this run can use -- kept, or built again with the
                // arena limit what the context now holds leaves -- decided before anything is
                // taken, so a run the budget cannot fit is refused having consumed nothing.
                feeds.WrittenInto = _built.WrittenInto(retainedOutputNames ?? ComputeContext.NoOutputsRetained);
                var built = feeds.Budget is { } limit ? Within(limit, feeds) : _built;
                var session = built.Session;

                // On this graph's own backend, because that is the runtime about to read the values:
                // what it can address it is handed as it stands, and anything else -- every literal
                // in the program, held in managed memory, and a tensor of another device or runtime
                // -- through a copy that backend builds, or, where it is consumed and no output may
                // be written into it, in host memory for the runtime to copy into its arena. Each
                // input is held first -- read-locked, or consumed -- and its value built after.
                var sessionInputs = feeds.Feed(name =>
                    _onnxInputNameByOriginal.TryGetValue(name, out var mapped) ? mapped : name);

                // Per output, the input whose consumed memory the session wrote it into, or null.
                var results = _owner.CallSession(session, feeds, sessionInputs, _outputNames,
                    retainedOutputNames ?? ComputeContext.NoOutputsRetained, runSettings, out var aliasedInputs);

                return _owner.AdoptOutputs(
                    results, _outputNames, _backend, ArenasOf(results.Count, aliasedInputs, sessionInputs, feeds, built));
            }
            catch (Exception e) when ((failed = e) is null)
            {
                // Never entered: the filter only records what the run failed with, for the
                // release below not to take its place.
                throw;
            }
            finally
            {
                // However the run ends, and each step however the one before it did -- a context
                // left counting a run, or a budget gate left held, would refuse or block everything
                // after it for good. What the run holds outlives the native call by construction: it
                // is given up here and nowhere else, so a terminated run's feeds are still held right
                // up to the moment it gives up. What it consumed went to the backend with the call,
                // and only what never got that far is released here.
                try
                {
                    feeds.Dispose(failed);
                }
                finally
                {
                    if (counted) lock (_sessionGate) _running--;
                    _owner.ExitRun(entered);
                }
            }
        }

        /// <summary>
        /// A run of this graph as a message names it, holding only the names that takes — not this
        /// graph, which a tensor the run consumes would otherwise keep alive, with its session, its
        /// kept model and its context, for as long as the tensor is referenced. It is called
        /// <paramref name="run"/> where the caller has a name for this run of its own.
        /// </summary>
        private RunIdentity Identity(string? run)
        {
            var description = run ?? _description;
            var inputs = _originalInputNames;
            var outputs = _outputNames;
            var backend = _backend.Description;
            return new RunIdentity(() => ComputeContext.DescribeRun(
                description ?? ComputeContext.DescribeGraph(inputs, outputs), backend));
        }

        /// <summary>
        /// The arena each output of a run is in, as a device-memory budget counts it: the arena of
        /// the session that ran — <paramref name="built"/>'s — for an output it allocated there, and
        /// for one it wrote into the memory of a tensor the run consumed (output aliasing), the arena
        /// that memory was in: the consumed tensor's own record, which is none where it was never an
        /// arena's. The session's arena limit covers only what the arena itself allocates, so an
        /// output living where the consumed tensor lived is counted where that tensor was — in the
        /// discount of every later run of this session, unless that memory is this session's arena
        /// already.
        /// </summary>
        private Func<int, object?> ArenasOf(
            int outputs, IReadOnlyList<string?>? aliasedInputs,
            IReadOnlyDictionary<string, IShorokooTensorValue> sessionInputs, RunFeeds feeds, BuiltSession built)
        {
            // Built only once an output turns out to have been written into consumed memory: a run
            // that aliased nothing -- which answers with no entries, or none but nulls -- allocates
            // nothing here.
            object?[]? arenas = null;
            var aliased = 0;
            for (int i = 0; aliasedInputs is not null && i < outputs && i < aliasedInputs.Count; i++)
            {
                if (aliasedInputs[i] is not { } input) continue;
                if (arenas is null)
                {
                    arenas = new object?[outputs];
                    Array.Fill(arenas, built.Arena);
                }
                aliased++;
                // A value the run did not hand over has no record here, and none is the answer that
                // never under-counts: the output is then counted outside every arena.
                arenas[i] = sessionInputs.TryGetValue(input, out var value) ? feeds.ArenaOfHanded(value) : null;
            }
            if (arenas is null) return _ => built.Arena;
            _owner.CountAliasedOutputs(aliased);
            return i => arenas[i];
        }

        /// <summary>
        /// The session a run under a device-memory budget of <paramref name="limit"/> bytes can use,
        /// with <paramref name="feeds"/> admitted against it: this graph's session while its arena
        /// limit is still within what the budget allows, and otherwise a new one built with the
        /// limit what the context now holds leaves.
        ///
        /// <para>What the budget allows a session is the budget less the <i>discount</i>: the bytes
        /// the context holds in its memory outside that session's arena for the length of the run —
        /// every tensor attached to it there, and what the run itself reads there or copies there to
        /// read; not a host tensor it consumes, which the runtime copies into the arena itself unless
        /// an output may be written into it. A tensor the session's own earlier runs left in its arena is inside the limit
        /// already, where it is, and is not discounted again — and so is one a run wrote into such a
        /// tensor's memory, where one written into memory outside the arena is discounted with the
        /// rest (see <see cref="ArenasOf"/>). ONNX Runtime fixes an arena's limit
        /// when the session is built, and building one costs about as much as the graph is large,
        /// so a session is kept for as long as its limit fits and built again only when the discount
        /// has grown past the room it left — never merely because it has fallen — or when its limit
        /// cannot take what the run would have the runtime copy into its arena, which a session
        /// built with what the budget leaves now may. See
        /// <see cref="ComputeContext.ArenaLimitWithin"/> for the limit a new one gets.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">What the context holds leaves no room for the
        /// run's arena, or less than what the run would have the runtime copy into it. Nothing has
        /// been taken.</exception>
        private BuiltSession Within(long limit, RunFeeds feeds)
        {
            var built = _built;
            var plan = feeds.Plan(built.Arena);
            if (built.DeviceMemory.LimitBytes is { } current && current <= limit - plan.Outside
                && plan.InArena <= current)
            {
                feeds.Admit(current, plan);
                return built;
            }

            // A new session's arena starts empty, so what this one's runs left in its arena is
            // outside the new one, and is discounted with everything else.
            return Rebuild(feeds.AdmitFresh(limit));
        }

        /// <summary>
        /// Builds this graph's session again with an arena limit of <paramref name="arenaLimit"/>,
        /// and releases the one it replaces. What the old session's runs left in its arena survives
        /// the release — each output keeps the arena it came from alive — so nothing a caller holds
        /// is touched.
        /// </summary>
        private BuiltSession Rebuild(long arenaLimit)
        {
            var model = _model ?? throw new InvalidOperationException(
                "This compiled graph kept no model to build its session again from, so its arena "
                + "limit cannot come down to what its context's device-memory budget now allows.");
            var deviceMemory = _built.DeviceMemory with { LimitBytes = arenaLimit };
            var session = _owner.BuildSession(_backend, model, Optimization, deviceMemory, _outputAliases);
            var fresh = new BuiltSession(session, deviceMemory, BindableOf(session));
            BuiltSession old;
            lock (_sessionGate)
            {
                if (_disposed)
                {
                    fresh.Session.Dispose();
                    throw new ObjectDisposedException(GetType().Name);
                }
                old = _built;
                _built = fresh;
            }
            // Outside the gate. Nothing is inside the old session: a rebuild happens only under a
            // budget, inside this run's turn at the context's budget gate, where every other run of
            // the context waits; the other readers of a session call into it under the session's
            // gate, which the swap was made under.
            old.Session.Dispose();
            return fresh;
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
        /// <exception cref="ObjectDisposedException">This graph has been disposed, and its session
        /// with it.</exception>
        public bool HasDeviceMemory
        {
            get
            {
                lock (_sessionGate)
                {
                    ObjectDisposedException.ThrowIf(IsDisposed, this);
                    return _built.Session.HasDeviceMemory;
                }
            }
        }

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
        /// <exception cref="ObjectDisposedException">This graph has been disposed, and its session
        /// with it.</exception>
        public SessionOutputPlacement OutputPlacement
        {
            get
            {
                lock (_sessionGate)
                {
                    ObjectDisposedException.ThrowIf(IsDisposed, this);
                    return _built.Session.OutputPlacement;
                }
            }
        }

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
        ///
        /// <para>It reads the session this graph runs on now. Under a device-memory budget that can
        /// be a later session than the one it was compiled with — see <see cref="DeviceMemory"/> —
        /// and a new session's arena starts its figures afresh.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This graph's session has been released.</exception>
        public ArenaStatistics? ReadArenaStatistics()
        {
            lock (_sessionGate)
            {
                ObjectDisposedException.ThrowIf(IsDisposed, this);
                return _built.Session.ReadArenaStatistics();
            }
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
            lock (_sessionGate)
            {
                ObjectDisposedException.ThrowIf(IsDisposed, this);
                return _built.Session.ReadPinnedArenaStatistics();
            }
        }

        /// <summary>
        /// Which execution provider ran each node of this graph, with the bytes each moved — or
        /// <c>null</c> unless the context that compiled this graph carried
        /// <see cref="DiagnosticSettings.TraceNodePlacement"/>, which is off by default because
        /// recording costs every run the session makes.
        ///
        /// <para><b>Reading it stops the recording.</b> The trace covers every run made up to this
        /// call, runs after it are not recorded, and a second read hands back the same trace. So
        /// call it once, after the runs you are asking about. Under a device-memory budget, a
        /// session built again for a lower arena limit (see <see cref="DeviceMemory"/>) starts a
        /// trace of its own, which covers the runs from then on.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This graph's session has been released.</exception>
        public NodePlacement? ReadNodePlacement()
        {
            lock (_sessionGate)
            {
                ObjectDisposedException.ThrowIf(IsDisposed, this);
                return _built.Session.ReadNodePlacement();
            }
        }

        /// <summary>How many outputs this graph's session produces — the length
        /// <see cref="Execute(IData[], bool[])"/> requires of a retention array, so a caller can
        /// size one without deriving the count a second way and disagreeing.</summary>
        public int OutputCount => _outputNames.Length;
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
    /// cpu.Execute(graph, input.Shared());   // on the host, reading input
    /// cuda.Execute(graph, input);           // the same graph, on the card, consuming it
    /// </code>
    /// A context constructed without one runs on the process default
    /// (<see cref="Shorokoo.Core.Backends.DefaultBackend.Instance"/>), which is what
    /// every context did before backends could differ. Which device a context will use is read off
    /// <see cref="Backend"/>.
    ///
    /// <para>It also carries how its sessions and runs are configured — <see cref="DeviceMemory"/>
    /// for the arena each session it compiles is built with and for the budget it keeps on its
    /// device's memory, and <see cref="RunSettings"/> for what its runs do by default. Both are per
    /// instance, so two contexts may differ and neither reaches the other's sessions.</para>
    ///
    /// <para>The same data feeds either context and the same model runs on both, with nothing to
    /// say at the call site. A literal costs nothing to share: it is managed bytes until something
    /// runs (<see cref="Shorokoo.HostTensorData{T}"/>), and the context that feeds it to a session
    /// is the one that builds its runtime value, on its own backend. What a context cannot do is
    /// read memory its backend cannot address — a card another runtime allocated on, or host memory
    /// fed to a run on a card — so a run here reads such a tensor through a copy in its own memory:
    /// made on the first shared read, held by the tensor and reused by the next, or made and
    /// consumed in the tensor's place when the tensor is consumed. <see cref="TensorData.To"/> puts
    /// a tensor where this context can read it once, instead.</para>
    ///
    /// <para><b>Feeding.</b> A tensor fed to a run as it is is <b>consumed</b> by it: taken when the
    /// run starts, dead from then on however the run ends, and its memory the run's backend's to
    /// release. <c>t.Shared()</c> is read and left alive, and <c>t.TryConsume()</c> consumed only
    /// if nothing else is reading it then — see <see cref="SharedInput"/>. A run's outputs are new
    /// tensors, attached to the context that ran it.</para>
    ///
    /// <para><b>A context does not own tensors.</b> It keeps a weak list of the tensors attached to
    /// it — its runs' outputs and inputs, and what <see cref="TensorData.To"/> and
    /// <see cref="TensorData.CopyTo"/> placed for it — for its own accounting, and
    /// <see cref="Detach"/> takes one off. Attachment never keeps a tensor alive and never ends one's
    /// life: disposing a context releases what the context itself holds, its compiled sessions,
    /// and leaves every tensor as it was. The accounting is what the device-memory budget is kept
    /// by: the bytes of what is attached in the context's memory, read with
    /// <see cref="ReadDeviceMemoryUse"/>.</para>
    /// </summary>
    public partial class ComputeContext : IDisposable
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
        /// This context's device-memory budget, and the arena settings every session it compiles is
        /// built with. Initialize-only: a context keeps what it was built with.
        ///
        /// <para><b><see cref="DeviceMemorySettings.LimitBytes"/> is a budget on this context's
        /// device memory</b>, and covers both halves of what it holds there: the tensors attached to
        /// it in its memory — what <see cref="TensorData.To"/>, <see cref="TensorData.CopyTo"/> and
        /// <see cref="AllocateUninitialized(Shape, DType)"/> placed for it, what its runs read there
        /// or copied there to read, and the outputs they left there — and, while one of its runs
        /// executes, the arena that run computes in. A transfer that would take the attached bytes
        /// past the limit is refused, naming the budget, what is attached and what was asked for; a
        /// session's arena is capped at the budget less what the context holds outside it for the
        /// run, and is built again with a lower cap when that has grown past the room it left.
        /// <see cref="ReadDeviceMemoryUse"/> reads what is attached against the limit.</para>
        ///
        /// <para>Under a budget the context's runs also go one at a time — a second waits for the
        /// first to return, as does a transfer onto the context or a compile on it — and each run
        /// hands its arena's unused blocks back as it ends, whatever
        /// <see cref="RunSettings.ShrinkArenaAfterRun"/> says. A context with no limit is none of
        /// this.</para>
        ///
        /// <para>Its default <see cref="ArenaExtendStrategy.Auto"/> resolves per session, so one
        /// context can still give a session it knows is reused across shapes a different arena
        /// strategy from the rest; <see cref="CompiledGraph.DeviceMemory"/> reports which one a
        /// graph got, and under a budget the arena limit it got.</para>
        ///
        /// <para>Ignored where the context's memory is the host's — the CPU backends have no device
        /// arena, and a device-memory budget does not govern host memory.</para>
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
        /// being read by one (fed <c>.Shared()</c>, or through <c>.TryConsume()</c> while another
        /// run held it), by being the copy one of its runs read in a tensor's place, by
        /// <see cref="TensorData.To"/> or <see cref="TensorData.CopyTo"/> with this context as the
        /// target, and by <see cref="AllocateUninitialized(Shape, DType)"/> on it;
        /// <see cref="Detach"/> takes one off. A tensor a run consumes is dead, and is on no
        /// list. The list is weak and it is not ownership: it never keeps a tensor alive, never ends
        /// one's life, and a tensor that dies or is collected drops out of it. <see cref="Host"/>'s
        /// is always empty.</para>
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

        /// <summary>Whether this context's runs can use <paramref name="tensor"/>'s memory as it
        /// stands — the question <see cref="TensorData.To"/> asks, answered by the backend: it can
        /// address the memory, or the tensor is where its runs read one of its dtype.</summary>
        internal bool CanAddress(TensorData tensor)
        {
            var backend = ResolvedBackend;
            return backend.CanAddress(tensor.Location) || tensor.IsWhereRunsRead(backend);
        }

        /// <summary>Whether <paramref name="tensor"/> is on this context's books.</summary>
        internal bool Attaches(TensorData tensor) => _attached.Contains(tensor);

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
        /// at all. <see cref="TensorData.IsHostResident"/> says which. On a real backend the tensor is
        /// attached to this context, as a <see cref="TensorData.CopyTo"/> result is — <see cref="Host"/>
        /// keeps no list, so there it is attached to nothing — and on a context under a
        /// device-memory budget it is refused, as a copy would be, when the budget cannot take
        /// it.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is
        /// <see cref="DType.Utf8"/>, whose elements are variable-length, or complex, which no memory
        /// here holds, or has no whole-byte element stride; or <paramref name="shape"/> has no known
        /// element count. Refused alike on every context, before any budget is asked.</exception>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        /// <exception cref="InvalidOperationException">This context's device-memory budget cannot
        /// take the tensor alongside what is attached to it
        /// (<see cref="DeviceMemorySettings.LimitBytes"/>).</exception>
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
            // A complex element has a width, so the check below would pass it, and each context
            // would then refuse it in words of its own, or not at all before a budget did.
            if (dtype == DType.Complex64 || dtype == DType.Complex128)
                throw new NotSupportedException(
                    $"A {dtype} tensor cannot be allocated: no memory here holds complex elements -- "
                    + "neither the framework's host memory nor an ONNX Runtime backend's.");
            var bits = TensorData.StorageBits(dtype);
            if (bits < 8 || shape.Count < 0)
                throw new NotSupportedException(
                    $"A tensor of {shape}:{dtype} cannot be allocated as a flat buffer: its "
                    + "elements have no whole-byte stride, or its shape has no known element count.");

            var bytes = checked(shape.Count * (bits / 8));
            if (_isHost)
                return TensorData.NewHostTensor(shape, dtype, new byte[bytes]);

            var backend = ResolvedBackend;
            return Placed(bytes, () => $"AllocateUninitialized of {shape}:{dtype}", () => TensorData.Create(
                shape, dtype, backend.CreateUninitializedTensorInBackendMemory(
                    (ShorokooTensorElementType)(int)dtype, (long[])shape), backend));
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
        /// <exception cref="InvalidOperationException">This context's device-memory budget cannot
        /// take the tensor alongside what is attached to it
        /// (<see cref="DeviceMemorySettings.LimitBytes"/>).</exception>
        public TensorData<T> AllocateUninitialized<T>(Shape shape) where T : IVarType
            => (TensorData<T>)AllocateUninitialized(shape, OnnxUtils.GetDType<T>());

        /// <summary>
        /// Takes a reader lock on <paramref name="target"/> — a tensor or a sequence — for a run of
        /// this context, attaches a tensor to this context, and hands back the lease to drop when
        /// the run is done. While a lease is outstanding the target cannot be deleted — a delete is
        /// refused or declined, and a deliberate one signals <see cref="TensorLease.Eviction"/> and
        /// waits — nor consumed by another run, and this context cannot detach it. The lease holds
        /// the target itself, so nothing a run reads can be collected under it.
        ///
        /// <para>Any context may lock any tensor: the lock is the reading context's, and lives on
        /// the tensor. It is a count, not a flag — two runs may read one tensor at once, and each
        /// holds a lock of its own; one run holds each tensor it is fed once, however many inputs
        /// it feeds.</para>
        /// </summary>
        /// <param name="target">What is being read.</param>
        /// <param name="reader">Who is reading it — a run — for a run the tensor has to refuse
        /// meanwhile to name; null for a holder with nothing to say.</param>
        /// <param name="attach">Whether a tensor locked is attached to this context: false for one
        /// the reader never reads in this context's memory — an element a run holds only for the
        /// sequence it is fed, whose value is built in host memory.</param>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This context has been disposed, or the target
        /// is dead.</exception>
        internal TensorLease Lock(ILifetimeOwner target, object? reader = null, bool attach = true)
        {
            ArgumentNullException.ThrowIfNull(target);
            // The two gates are taken one after the other rather than nested, in either direction:
            // the count is this context's business and the lock is the target's, and a delete
            // waiting for readers must never find itself behind a gate a run is queued on.
            // A read attaches the tensor read, as a run's output does: a tensor this context's runs
            // read is one its accounting has to see. Attached with the count, under the one gate;
            // a lock then refused leaves at worst a dead tensor on the list, which the list skips.
            CountLock(target, attach);
            try
            {
                return new TensorLease(this, target, target.Life.AcquireReadLock(reader), reader);
            }
            catch
            {
                ReleaseLease(target);
                throw;
            }
        }

        /// <summary>Counts one more lock of this context's on <paramref name="target"/>, and
        /// attaches it if it is a tensor to be attached.</summary>
        /// <exception cref="ObjectDisposedException">This context has been disposed.</exception>
        private void CountLock(ILifetimeOwner target, bool attach)
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _leases++;
                _locksHeld[target] = _locksHeld.TryGetValue(target, out var held) ? held + 1 : 1;
                if (attach && target is TensorData tensor && !_isHost) _attached.Add(tensor);
            }
        }

        /// <summary>What a run asks to retain on the device when it asks for nothing.</summary>
        internal static IReadOnlySet<string> NoOutputsRetained { get; } = new HashSet<string>();

        /// <summary>
        /// One signal for every tensor this run has locked, plus whatever the caller asked to
        /// stop the run with. Null when there is nothing to link — the caller's settings then go
        /// to the backend with no token of this run's linked into them.
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
        ///
        /// <para>Where this context's memory is under its device-memory budget —
        /// <paramref name="budget"/>, the reading the run's plan is made against too, so the two
        /// cannot disagree — it first waits for the budget gate, so the context's runs go one at a
        /// time; what it entered is what <see cref="ExitRun"/> is handed back.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This context has been disposed, so its
        /// sessions are already gone and there is nothing left to run on.</exception>
        /// <exception cref="OperationCanceledException"><paramref name="cancellation"/> was cancelled
        /// while the run waited for the one before it. Nothing was taken.</exception>
        internal BudgetGate? EnterRun(long? budget, CancellationToken cancellation)
        {
            // The host context runs nothing -- Compile, Execute and Run all refuse there -- and
            // cannot be disposed, so there is no question here for a count to answer.
            if (_isHost) return null;
            var gate = EnterBudget(budget, cancellation);
            try
            {
                lock (_gate)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    _runs++;
                }
            }
            catch
            {
                gate?.Exit();
                throw;
            }
            return gate;
        }

        /// <summary>Records that a run of this context has returned, however it ended, and lets the
        /// next one in where <paramref name="entered"/> is the budget gate its
        /// <see cref="EnterRun"/> entered. Called from the <c>finally</c> that pairs with it and
        /// nowhere else.</summary>
        internal void ExitRun(BudgetGate? entered)
        {
            if (_isHost) return;
            lock (_gate) _runs--;
            entered?.Exit();
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
        /// it — and every tensor among them attached to this context. For a session run once and
        /// released, whose arena goes with it, so no output records one.
        /// </summary>
        internal NamedModelParam[] AdoptOutputs(
            IReadOnlyList<IShorokooTensorValue> results, IReadOnlyList<string> names, IShorokooBackend backend)
            => AdoptOutputs(results, names, backend, static _ => null);

        /// <summary>
        /// <see cref="AdoptOutputs(IReadOnlyList{IShorokooTensorValue}, IReadOnlyList{string}, IShorokooBackend)"/>
        /// with each output's arena answered by <paramref name="arenaOf"/>, given its position: an
        /// output a run wrote into consumed memory is in whatever arena that memory was, if any, not
        /// in the running session's.
        /// </summary>
        internal NamedModelParam[] AdoptOutputs(
            IReadOnlyList<IShorokooTensorValue> results, IReadOnlyList<string> names,
            IShorokooBackend backend, Func<int, object?> arenaOf)
        {
            var outputs = new NamedModelParam[results.Count];
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = OnnxUtils.CreateNamedModelParam(
                    results[i], ModelParamType.OutputParam, names[i], backend);
                if (outputs[i] is not TensorDataModelParam named) continue;
                var tensor = named.ToTensorData();
                if (arenaOf(i) is { } arena && !tensor.Space.IsHost) tensor.RecordArena(arena);
                Attach(tensor);
            }
            return outputs;
        }

        /// <summary>
        /// Whether this context's compiles mark outputs a run may write into the memory of an input
        /// it consumed — output aliasing, which the training rig's steps use for the state they
        /// replace (<see cref="OutputAlias"/>). On unless turned off, and turned off only by a test
        /// comparing a run with aliasing to one without.
        /// </summary>
        internal bool OutputAliasing { get; init; } = true;

        // How many outputs this context's runs have written into consumed memory, over its life.
        private long _aliasedOutputs;

        /// <summary>How many outputs this context's runs have written into the memory of an input
        /// they consumed, over its life (test hook).</summary>
        internal long AliasedOutputs => Interlocked.Read(ref _aliasedOutputs);

        /// <summary>Records that a run wrote <paramref name="count"/> outputs into consumed
        /// memory.</summary>
        internal void CountAliasedOutputs(int count)
        {
            if (count > 0) Interlocked.Add(ref _aliasedOutputs, count);
        }

        /// <summary>
        /// The pairs of <paramref name="candidates"/> <paramref name="graph"/> proves, by name — none
        /// where there are no candidates or this context aliases nothing
        /// (<see cref="OutputAliasing"/>). A candidate naming a position the graph does not have is
        /// no candidate.
        /// </summary>
        private IReadOnlyList<OutputAlias> MarkedAliases(
            GraphProto graph, IReadOnlyList<(int Output, int Input)>? candidates)
        {
            if (!OutputAliasing || candidates is not { Count: > 0 }) return [];
            var named = new List<OutputAlias>(candidates.Count);
            foreach (var (output, input) in candidates)
                if (output >= 0 && output < graph.Outputs.Count && input >= 0 && input < graph.Inputs.Count)
                    named.Add(new OutputAlias(graph.Outputs[output].Name, graph.Inputs[input].Name));
            return OutputAliasProof.Prove(graph, named);
        }

        /// <summary>
        /// Names a graph for a message about a run of it: by its inputs and outputs, which is what a
        /// caller can recognise it by — graphs carry no name of their own. A long list is cut short,
        /// saying how much was left out: a message is read, and a training step has an input per
        /// parameter.
        /// </summary>
        internal static string DescribeGraph(IReadOnlyList<string> inputs, IReadOnlyList<string> outputs)
            => $"the graph ({Names(inputs)}) -> ({Names(outputs)})";

        private static string Names(IReadOnlyList<string> names)
        {
            const int Shown = 6;
            if (names.Count <= Shown) return string.Join(", ", names);
            return $"{string.Join(", ", names.Take(Shown))}, and {names.Count - Shown} more";
        }

        /// <summary>Names a run for a message about it: the graph, and the context it ran on — by
        /// its backend, contexts having no name of their own.</summary>
        internal static string DescribeRun(string graph, BackendDescription backend)
            => $"a run of {graph} on the compute context over {backend}";

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
        /// <exception cref="InvalidOperationException">This context is under a device-memory budget,
        /// and what is attached to it in its memory leaves nothing of the budget for the session's
        /// arena (<see cref="DeviceMemorySettings.LimitBytes"/>).</exception>
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
        ///
        /// <para>A tensor fed as it is is consumed by the run; <c>t.Shared()</c> is read and left
        /// alive, and <c>t.TryConsume()</c> is consumed only when nothing else is reading it — see
        /// <see cref="CompiledGraph.Execute(IData[])"/>.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">This context is under a device-memory
        /// budget that cannot take the run, as <see cref="Compile(ComputationGraph)"/> and
        /// <see cref="CompiledGraph.Execute(IData[])"/> refuse one. Nothing it was fed has been
        /// taken.</exception>
        public NamedModelParam[] Execute(ComputationGraph graph, params IData[] inputs)
        {
            graph.RequireConcretized("ComputeContext.Execute");
            return this.Execute(graph.ToInternal(), inputs);
        }

        /// <summary>
        /// Executes the graph with pre-built named inputs. Builds the ONNX model and a fresh
        /// session per call (disposed afterwards); use <see cref="Compile(ComputationGraph)"/>
        /// for repeated runs.
        ///
        /// <para>A parameter's data is consumed by the run unless the parameter says otherwise
        /// (<see cref="NamedModelParam.FeedMode"/>): passed <c>p.Shared()</c> or
        /// <c>p.TryConsume()</c>, or made from a <see cref="SharedInput"/> —
        /// <c>NamedModelParam.FromIData(name, type, t.Shared())</c>.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">This context is under a device-memory
        /// budget that cannot take the run, as <see cref="Compile(ComputationGraph)"/> and
        /// <see cref="CompiledGraph.Execute(IData[])"/> refuse one. Nothing it was fed has been
        /// taken.</exception>
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
        /// Its inputs are fed as <see cref="Execute(ComputationGraph, IData[])"/>'s are: consumed
        /// as they are, read through <c>.Shared()</c>.
        /// </summary>
        public (NamedModelParam[] regularOutputs, ComputationGraph updatedGraph) ExecuteWithState(
            ComputationGraph graph, params IData[] inputs)
        {
            graph.RequireConcretized("ComputeContext.ExecuteWithState");
            var (regularOutputs, updatedGraph) = ExecuteWithState(graph.ToInternal(), inputs);
            // updatedGraph is either the private copy itself (no state params) or a fresh
            // clone with the new state values — exclusively owned either way.
            return (regularOutputs, new ComputationGraph(updatedGraph, graph.Kind));
        }

        /// <summary>
        /// Named-input overload of
        /// <see cref="ExecuteWithState(ComputationGraph, IData[])"/>.
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
        /// <param name="description">What a message about a run of the compiled graph calls it, in
        /// place of the list of its input and output names — "a TrainingRig's training step" for the
        /// rig's own, whose inputs are one per parameter.</param>
        /// <param name="aliasCandidates">Outputs the caller would have a run write into the memory of
        /// an input, by position — output <c>Output</c> into input <c>Input</c>, the training rig's
        /// updated state into the state it replaces. Each is kept only where
        /// <see cref="OutputAliasProof"/> proves it over the model as built, and the session is built
        /// with those (<see cref="OutputAlias"/>); none on a context that aliases nothing
        /// (<see cref="OutputAliasing"/>).</param>
        internal CompiledGraph Compile(
            InternalComputationGraph graph,
            IReadOnlyList<long[]?>? inputDims,
            bool trainingStep,
            bool reusedAcrossShapes = false,
            string? description = null,
            IReadOnlyList<(int Output, int Input)>? aliasCandidates = null)
        {
            graph.RequireRunnableOps("ComputeContext.Compile");
            var originalInputNames = ResolveOriginalInputNames(graph);
            return CompileFromModel(
                () => FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true, inputDims: inputDims),
                originalInputNames,
                trainingStep,
                reusedAcrossShapes,
                description,
                aliasCandidates);
        }

        private CompiledGraph CompileFromModel(
            Func<ModelProto> buildModel,
            string[] originalInputNames,
            bool trainingStep,
            bool reusedAcrossShapes,
            string? description,
            IReadOnlyList<(int Output, int Input)>? aliasCandidates = null)
        {
            RefuseHostContext("compile");
            var model = buildModel();
            var outputAliases = MarkedAliases(model.Graph, aliasCandidates);

            var memoryStream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(memoryStream, model);
            var modelData = memoryStream.ToArray();

            var optimization = SessionOptimization(HasOptionalOps(model.Graph), trainingStep);
            // Settled here, not inside the session: CompiledGraph then reports the strategy this
            // session actually got rather than the Auto that asked for it.
            var deviceMemory = DeviceMemory.Resolve(reusedAcrossShapes);
            var backend = ResolvedBackend;
            var space = backend.MemorySpace;
            byte[]? kept = null;
            IShorokooSession session;
            // Under a budget the session is built one at a time with runs of this context and what
            // is placed in its memory -- its weights go into its arena as it is built -- and with an
            // arena limit of what the context holds there now leaves. A run finding the context
            // holding more builds it again with less, from the model this keeps for the purpose.
            var gate = EnterBudget(CancellationToken.None);
            try
            {
                if (BudgetIn() is { } limit)
                {
                    var (attached, tensors) = AttachedIn();
                    var arena = ArenaLimitWithin(limit, attached)
                        ?? throw NoRoomToCompile(space, limit, attached, tensors);
                    deviceMemory = deviceMemory with { LimitBytes = arena };
                    kept = modelData;
                }
                session = BuildSession(backend, modelData, optimization, deviceMemory, outputAliases);
            }
            finally
            {
                gate?.Exit();
            }

            var onnxInputNameByOriginal = SessionNamesOf(originalInputNames, session);

            var graph = new CompiledGraph(
                session, backend, onnxInputNameByOriginal, originalInputNames, optimization,
                deviceMemory, RunSettings, this, description, kept, outputAliases);
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
        /// Expands TensorDataStruct inputs into individual field data entries. A struct fed through
        /// <c>.Shared()</c> or <c>.TryConsume()</c> expands into fields fed the same way, which is
        /// what a mode on a composite means: it applies to every member. A field the struct was
        /// built with through one of them keeps its own unless the struct is fed <c>.Shared()</c>,
        /// and a shared one is read even where the struct is fed as it is
        /// (<see cref="TensorDataStruct.FieldFeedMode"/>).
        /// </summary>
        internal static IData[] ExpandStructInputs(IData[] inputs)
        {
            var expandedInputs = new List<IData>();
            foreach (var input in inputs)
            {
                var (value, sharing) = input is SharedInput shared
                    ? (shared.Value, (SharedInputMode?)shared.Mode)
                    : (input, null);
                if (value is TensorDataStruct structData)
                {
                    foreach (var field in structData.Definition.Fields)
                    {
                        if (!structData.Fields.TryGetValue(field.Name, out var fieldData))
                        {
                            throw new InvalidTensorOperationException(ErrorCodes.CR006, "Execute",
                                $"field={field.Name}, struct={structData.Definition.TypeName ?? "anonymous"}",
                                $"TensorDataStruct is missing data for field '{field.Name}'");
                        }
                        expandedInputs.Add(structData.FieldFeedMode(field.Name, sharing) is { } mode
                            ? new SharedInput(fieldData, mode)
                            : fieldData);
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
            // The outputs are the session's, named once it is built; a refusal before then names
            // the graph by its inputs.
            var outputNames = new StrongBox<IReadOnlyList<string>>([]);
            var budget = BudgetIn();
            var feeds = new RunFeeds(this, backend, OneShotRun(originalInputNames, outputNames, backend.Description), budget);
            Exception? failed = null;
            // Before the session, so that everything this context is about to build is inside the
            // window its disposal is refused in -- the session most of all, since disposing the
            // context is what would release it. Under a device-memory budget it also waits for the
            // context's run in flight, if any.
            var entered = EnterRun(budget, RunSettings.CancellationToken);
            try
            {
                // Everything that can refuse the run over what it is fed, before a session is built
                // for it and before anything is taken.
                feeds.Prepare(inputs);

                // A one-shot session: built, fed once, and disposed, so no differing shapes can
                // reach it -- and, under a budget, built with the arena limit a kept session would get
                // for what this run holds on the device, since its arena starts empty and all of that
                // is outside it: what that leaves of the budget, rounded as for a session kept while
                // the context's holdings grow a little, though this one is never kept.
                var deviceMemory = DeviceMemory.Resolve(reusedAcrossShapes: false);
                if (feeds.Budget is { } limit)
                {
                    deviceMemory = deviceMemory with { LimitBytes = feeds.AdmitFresh(limit) };
                }
                session = BuildSession(
                    backend, modelData,
                    SessionOptimization(
                        HasOptionalOps(model.Graph) || IsFullyConstant(model.Graph), trainingStep: false),
                    deviceMemory);
                outputNames.Value = [.. session.OutputNames];
                var onnxInputNameByOriginal = SessionNamesOf(originalInputNames, session);

                // Held first, then the values -- see CompiledGraph.Run -- on this context's
                // backend: the one that just built the session above, and so the runtime that is
                // about to read what it is fed.
                var sessionInputs = feeds.Feed(name =>
                    onnxInputNameByOriginal.TryGetValue(name, out var mapped) ? mapped : name);

                // Nothing retained: this is the one-shot path, which builds a session, feeds it
                // once and disposes it, so there is no later run for a device-resident output to
                // be fed into -- and a session built for one run was marked to alias nothing.
                var results = CallSession(
                    session, feeds, sessionInputs, session.OutputNames, NoOutputsRetained, RunSettings, out _);
                return AdoptOutputs(results, session.OutputNames, backend);
            }
            catch (Exception e) when ((failed = e) is null)
            {
                // Never entered: the filter only records what the run failed with, for the
                // release below not to take its place.
                throw;
            }
            finally
            {
                // However the run ends, and each step however the one before it did. Before the
                // session goes: what the run holds is given up here and nowhere else, so a
                // terminated run's feeds are still held right up to the moment it gives up. What it
                // consumed went to the backend with the call; only what never got that far is
                // released here.
                try
                {
                    feeds.Dispose(failed);
                }
                finally
                {
                    try
                    {
                        // Dispose the session to free native memory — on the throwing path too,
                        // where the memory it holds is the memory the caller has just been told it
                        // lacks. The returned tensor values stay valid across it, and the finally
                        // also keeps the session rooted across the native calls above. They are
                        // not, however, free of it: a result keeps its session's ALLOCATOR alive, so
                        // a caller that retains one retains that session's arena — see
                        // `FastProcessorHelper.RehostOffSession` for what a caller that must not
                        // does, and Shorokoo/Shorokoo#180 for the general question.
                        session?.Dispose();
                    }
                    finally
                    {
                        // Last, so that this context is answerable for its session right up to the
                        // moment the session is gone.
                        ExitRun(entered);
                    }
                }
            }
        }

        /// <summary>
        /// The native call both run paths make, made the same way: every lock's eviction signal
        /// linked with the caller's; under a budget, the arena shrunk as the run ends whatever the
        /// caller asked, since the budget counted it at its limit only for this run; the arena's
        /// figures read either side of the call and nothing else; what the run consumed handed over
        /// in the call, the backend's alone from then on whatever the call does; and a stop the caller
        /// asked for rethrown with the caller's own token. Gives back, per output, the input whose
        /// consumed memory the session wrote it into, or null — or nothing, where it wrote none.
        /// </summary>
        internal IReadOnlyList<IShorokooTensorValue> CallSession(
            IShorokooSession session, RunFeeds feeds, IReadOnlyDictionary<string, IShorokooTensorValue> sessionInputs,
            IReadOnlyList<string> outputNames, IReadOnlySet<string> retainedOutputNames, RunSettings runSettings,
            out IReadOnlyList<string?> aliasedInputs)
        {
            using var eviction = LinkEvictions(feeds.Leases, runSettings.CancellationToken);
            var settings = feeds.Budget is null ? runSettings : runSettings with { ShrinkArenaAfterRun = true };
            if (eviction is not null) settings = settings with { CancellationToken = eviction.Token };

            IReadOnlyList<string?> aliased = [];
            var arenaBefore = StartRunStats(session);
            try
            {
                var results = feeds.HandOver(consumed => session.RunConsuming(
                    sessionInputs, consumed, outputNames, retainedOutputNames, settings, out aliased));
                aliasedInputs = aliased;
                return results;
            }
            catch (OperationCanceledException stopped) when (StoppedByCaller(stopped, runSettings.CancellationToken))
            {
                throw new OperationCanceledException(
                    stopped.Message, stopped.InnerException, runSettings.CancellationToken);
            }
            finally
            {
                // However the run ended, and before a one-shot session goes: a run that failed for
                // want of memory is the one whose figures are worth most, and a disposed session has
                // no arena left to read.
                FinishRunStats(session, arenaBefore);
            }
        }

        /// <summary>The session's own name for each of <paramref name="originals"/>, by position: what
        /// a run of it is fed under.</summary>
        private static Dictionary<string, string> SessionNamesOf(string[] originals, IShorokooSession session)
        {
            var names = new Dictionary<string, string>();
            for (int i = 0; i < originals.Length && i < session.InputNames.Count; i++)
                names[originals[i]] = session.InputNames[i];
            return names;
        }

        /// <summary>
        /// A one-shot run as a message names it, holding only the names that takes: its outputs are
        /// the session's, which <paramref name="outputs"/> is given once the session is built.
        /// Nothing of the run itself, which a tensor it consumes would otherwise keep alive.
        /// </summary>
        private static RunIdentity OneShotRun(
            string[] inputs, StrongBox<IReadOnlyList<string>> outputs, BackendDescription backend)
            => new(() => DescribeRun(DescribeGraph(inputs, outputs.Value ?? []), backend));

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

        /// <summary>A session of <paramref name="backend"/> over <paramref name="modelData"/>, built
        /// with <paramref name="deviceMemory"/>, with what this context records about its sessions,
        /// and with the outputs the lowering proved it may write into consumed inputs'
        /// memory.</summary>
        internal IShorokooSession BuildSession(
            IShorokooBackend backend, byte[] modelData, ShorokooGraphOptimization optimization,
            DeviceMemorySettings deviceMemory, IReadOnlyList<OutputAlias>? outputAliases = null)
            => backend.CreateSession(
                modelData, optimization, ShorokooLogSeverity.Fatal, deviceMemory, Diagnostics,
                outputAliases ?? []);

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
        internal (NamedModelParam[] regularOutputs, InternalComputationGraph updatedGraph) ExecuteWithState(InternalComputationGraph graph, params IData[] inputs)
        {
            var loweredGraph = LowerStateUpdateNodesOnFast(graph);
            var allOutputs = this.Execute(loweredGraph, inputs);
            return ProcessExecuteWithStateResults(graph, allOutputs);
        }

        /// <summary>
        /// Named-input overload of
        /// <see cref="ExecuteWithState(InternalComputationGraph, IData[])"/>.
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

        /// <summary>
        /// Executes the subgraph with <paramref name="inputData"/> and returns the output values.
        /// Each input is fed as <see cref="ComputeContext.Execute(ComputationGraph, IData[])"/>
        /// feeds it: a tensor given as it is is consumed, and <c>t.Shared()</c> is read and left
        /// alive.
        /// </summary>
        public TensorData[] With(params IData[] inputData)
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
                // What a shared feed wraps, which is where the value is: the wrapper holds none.
                SharedInput shared => shared.Value.ToTensorValue(backend),
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
