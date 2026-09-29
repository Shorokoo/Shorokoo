using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Backends;
using Shorokoo.Onnx;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast-native port of <c>InitializeModelParams</c>.
    /// Walks <c>graph</c> for every <c>MODEL_PARAM</c> node, rewrites it
    /// to a <c>FUNCTION_INVOKE</c> of its initializer <see cref="Function"/> (preserving
    /// the original initializer-param inputs, the output <see cref="FastTensorKey"/>, and
    /// the target function, keyed to take the parameter's stream key as one input more), then
    /// runs the resulting graph through
    /// a session compiled for the slice of the graph feeding one initializer's output — one run
    /// per parameter, one session per distinct slice (see <see cref="ChunkFor"/>) — each result
    /// copied off its session (see <see cref="FastProcessorHelper.RehostOffSession"/>).
    /// The decoded results are returned as a
    /// <see cref="ModelId"/> → <see cref="TensorData"/> dictionary.
    /// </summary>
    internal static class FastInitializeModelParams
    {
        /// <param name="graph">The concrete architecture whose initializers to run.</param>
        /// <param name="computeContext">The context the initializers run on, or null for
        /// <see cref="ComputeContext.Default"/>. Resolved only once there is an initializer to
        /// run — resolving the default context resolves a backend, and a graph with no
        /// trainable parameter returns empty below without executing anything. Building and
        /// exporting such a model must not require a backend be deployed.</param>
        /// <param name="rngConfig">The RNG configuration the per-parameter init streams derive
        /// from, or null to initialize outside the keyed scheme.</param>
        /// <param name="paramInfos">The parameter inventory keyed initialization needs; required
        /// whenever <paramref name="rngConfig"/> is supplied.</param>
        public static ImmutableDictionary<ModelId, TensorData> Process(
            InternalComputationGraph graph,
            ComputeContext? computeContext,
            RngConfig? rngConfig = null,
            ConcreteModelParamInfos? paramInfos = null)
        {
            // Keyed per-parameter initialization needs BOTH the config and the inventory:
            // with a config but no inventory, every parameter would skip the keyed draw
            // substitution and initialize through its un-keyed initializer function — values
            // not derived from the config at all, while the config looks engaged (its override
            // validation below still runs).
            if (rngConfig is not null && paramInfos is null)
                throw new System.ArgumentNullException(nameof(paramInfos),
                    "FastInitializeModelParams: an RngConfig was supplied without the parameter " +
                    "inventory, but keyed per-parameter initialization needs both — without the " +
                    "inventory every parameter would initialize outside the keyed scheme, from " +
                    "values not derived from the config. Pass GetConcreteModelParamInfos() " +
                    "of the same concrete architecture.");

            var workGraph = graph.Clone();

            var functionInvokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;

            // Per-parameter initialization RNG: map each parameter's ModelId to its
            // canonical name + shape so a random initializer draws in-graph keyed noise on
            // that parameter's own stream (see FastInitKeyedDraws). Null config disables it.
            var infoById = rngConfig is null
                ? null
                : paramInfos!.ParamInfos.ToDictionary(x => x.ModelId);

            // Resolve every parameter's init key ONCE, up front, by executing one small graph of
            // split chains (RngKeyResolver) — the host still computes no RNG itself (#136). Each
            // parameter's initializer is then fed its resolved key as an input.
            //
            // The alternative — emitting each parameter's split chain inside its own initializer
            // body — is what a naive "move the fold in-graph" does, and it is materially worse:
            // it multiplies the initialization graph by the ModelId depth of every parameter, it
            // makes every parameter's slice a different graph (so no two share a session — see
            // ChunkFor), and it makes the shared chain's placement dependent on which control-flow
            // scope the first draw happens to sit in.
            // Only parameters whose initializer actually draws need a key; a constant-filled
            // initializer (zeros/ones bias, etc.) would otherwise pay for a key nothing reads.
            var initKeys = infoById is null
                ? null
                : ResolveInitKeys(
                    workGraph.Nodes
                        .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM &&
                                    n.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId) is not [0] &&
                                    n.TargetFunction is { } f && FastInitKeyedDraws.DrawsRandomness(f))
                        .Select(n => new ModelId(
                            n.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull()))
                        .Distinct(),
                    rngConfig!, computeContext);

            // One keyed body per initializer, shared by every parameter that uses it: each call
            // site passes its own parameter's stream key, through a graph input of its own, so
            // same-shaped parameters slice to the same session (see ChunkFor).
            var keyedByInitializer = new Dictionary<Function, Function?>(ReferenceEqualityComparer.Instance);
            // Settled at the first initializer to key, which is the first point that needs the
            // context's device: a graph with nothing to draw resolves no backend.
            long? chunkPositions = null;
            var keyInputNodes = new List<FastNode>();
            var keyByInput = new Dictionary<FastTensorKey, ulong>();

            var collectedModelIds = new List<ModelId>();
            var collectedOutputKeys = new List<FastTensorKey>();
            // What this call is initializing, kept for the failure message below: a native
            // allocation failure aborts the whole session and names nothing on its own.
            var collectedInventory =
                new List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)>();

            foreach (var node in workGraph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM) continue;

                // The RngSeed parameter at reserved ModelId [0] carries the runtime RNG
                // identity, not a weight: it has no initializer function to run —
                // ApplyRngConfig is its initialization.
                if (node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId) is [0])
                    continue;

                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype).AssertNotNull();
                var rank = node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank) ?? -1;
                var modelIdVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull();
                var modelId = new ModelId(modelIdVals);
                var shape = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrShape);
                // Both are cleared by the rewrite below; read them while they are still there.
                var identifierTemplate = node.IdentifierTemplate;
                ConcreteModelParamInfo? paramInfo = null;

                // Replace the (shared) initializer with a per-parameter keyed-draw clone
                // before the node is rewritten to FUNCTION_INVOKE (which preserves TargetFunction).
                if (infoById is not null)
                {
                    // The mirror of the unmatched-override check below: a parameter the bound
                    // config cannot key must fail loudly — skipping just this one would leave
                    // its initializer un-keyed (backend randomness) while its siblings stay
                    // keyed, with nothing reporting the mix.
                    if (!infoById.TryGetValue(modelId, out var info))
                        throw new System.InvalidOperationException(
                            "FastInitializeModelParams: the trainable parameter " +
                            $"'{node.IdentifierTemplate}' at ModelId [{string.Join(", ", modelId.Vals)}] " +
                            "is missing from the supplied parameter inventory, so it would silently " +
                            "initialize un-keyed (backend randomness not derived from the RngConfig) " +
                            "while the other parameters stay keyed. The inventory must be " +
                            "GetConcreteModelParamInfos() of this same graph.");

                    paramInfo = info;

                    if (node.TargetFunction is { } initFn)
                    {
                        // Init draws under the configured algorithm's registry name (the key
                        // tree itself is algorithm-independent — the split is always the default
                        // algorithm), so a param's init values switch with the algorithm just
                        // like runtime feeds. The ordinal in the name keeps two initializers that
                        // share a name apart.
                        if (!keyedByInitializer.TryGetValue(initFn, out var keyed))
                            keyedByInitializer[initFn] = keyed = FastInitKeyedDraws.BuildKeyedDraws(
                                initFn, $"{initFn.DefaultName}__rng{keyedByInitializer.Count}",
                                info.ToShorokooIdString(), Core.Rng.RngAlgorithms.NameOf(rngConfig!.Algorithm),
                                chunkPositions ??= ChunkPositionsOn(computeContext ?? ComputeContext.Default));
                        if (keyed is not null)
                        {
                            // Stream key = init master folded along the parameter's ModelId path —
                            // the RNG key tree IS the ModelId tree — resolved above by executing
                            // the derivation, so a param's init stream stays reconstructible
                            // offline from its ModelId, plus the trip number of each enclosing loop
                            // for a draw inside one: the keyed body folds those onto this key
                            // in-graph, and without them every trip of such a draw collapses to one
                            // sample (#343). Only a drawing initializer is keyed, and every drawing
                            // one had its key resolved above.
                            var keyInput = FastInternalOp.RuntimeInput(DType.UInt64, rank: 0);
                            var keyTensor = InternalComputationGraph.InputKeyOf(keyInput);
                            keyInputNodes.Add(keyInput);
                            keyByInput[keyTensor] = initKeys![modelId];
                            node.TargetFunction = keyed;
                            node.FullInputs[""] = [.. node.FullInputs.TryGetValue("", out var args) ? args : [], keyTensor];
                        }
                    }
                }

                var newAttributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [OnnxOpAttributeNames.ShrkAttrStructure] = new[] { DataStructure.Tensor },
                        [OnnxOpAttributeNames.ShrkAttrDtype] = new[] { dtype },
                        [OnnxOpAttributeNames.ShrkAttrRank] = new[] { rank },
                        [OnnxOpAttributeNames.ShrkAttrGenericTypeArgs] = null,
                    },
                    functionInvokeAttrDefs);

                node.OpCode = InternalOpCodes.FUNCTION_INVOKE;
                node.Attributes = newAttributes;
                node.IdentifierTemplate = null;
                // FullInputs and TargetFunction (the initializer fn) are preserved
                // unchanged: FUNCTION_INVOKE expects the same variadic input list and
                // a TargetFunction reference, matching what MODEL_PARAM stored.

                var outputKey = node.FullOutputs[""][0]!.Value;
                collectedModelIds.Add(modelId);
                collectedOutputKeys.Add(outputKey);
                collectedInventory.Add((identifierTemplate, paramInfo, modelId, dtype, shape));
            }

            // Fail-loud override validation, mirroring the Runtime-side check at bind
            // (FastBindRngConfig): a Params override that matches no parameter of this graph
            // would otherwise be a silent no-op — the exact re-keying hazard explicit seeding
            // exists to prevent.
            if (rngConfig is not null)
            {
                var paramPaths = collectedModelIds
                    .Select(id => string.Join(",", id.Vals))
                    .ToHashSet();
                var unmatched = rngConfig.OverrideKeys
                    .Where(k => k.collection == RngCollection.Params && !paramPaths.Contains(k.pathKey))
                    .Select(k => $"[{k.pathKey}]")
                    .ToArray();
                if (unmatched.Length > 0)
                    throw new System.InvalidOperationException(
                        "RngConfig.Override(Params, ...) matches no trainable parameter of this " +
                        "graph: " + string.Join(", ", unmatched) +
                        ". Parameter stream paths are listed by GetRngStreamReport(); overrides " +
                        "must use a reported path exactly.");
            }

            if (collectedOutputKeys.Count == 0)
                return ImmutableDictionary<ModelId, TensorData>.Empty;

            // Past that return there IS an initializer to execute, so this is the first point that
            // genuinely needs a backend — and therefore the first point that may resolve the
            // default context, which is what resolves one. A graph with no trainable parameter has
            // already returned above having asked for nothing.
            var compute = computeContext ?? ComputeContext.Default;

            foreach (var keyInput in keyInputNodes)
                workGraph.AddInput(keyInput);

            // Each parameter's slice, grouped with the slices that are the same graph (see
            // SameSlice): parameters whose slices match — same initializer, same shape — differ only
            // in the stream keys they are fed, so they run on one session. Each chunk drops the
            // inputs its initializer does not read, so a slice takes its own parameters' keys and
            // nothing else, and its one output is its parameter's initial value.
            var groups = new List<InitGroup>();
            var groupsBySize = new Dictionary<int, List<InitGroup>>();
            var seenIds = new HashSet<ModelId>();
            for (int i = 0; i < collectedOutputKeys.Count; i++)
            {
                // Two parameters at one ModelId would silently collapse to one entry below — the
                // caller indexes the result by a per-parameter id and would hand two struct fields
                // the same value — so fail rather than return one of them.
                if (!seenIds.Add(collectedModelIds[i]))
                    throw new System.InvalidOperationException(
                        "FastInitializeModelParams: two model parameters share ModelId " +
                        $"[{string.Join(", ", collectedModelIds[i].Vals)}]. Parameter ids must be " +
                        "unique within a concrete architecture.");

                var chunk = ChunkFor(workGraph, collectedOutputKeys[i]);
                if (!groupsBySize.TryGetValue(chunk.Nodes.Count, out var sameSize))
                    groupsBySize[chunk.Nodes.Count] = sameSize = [];
                var group = sameSize.FirstOrDefault(g => SameSlice(g.Chunk, chunk));
                if (group is null)
                {
                    group = new InitGroup(chunk);
                    sameSize.Add(group);
                    groups.Add(group);
                }
                // In the slice's own input order, which SameSlice has matched to the group's.
                group.Members.Add((i, [.. chunk.Inputs.Select(k => keyByInput[k])]));
            }

            var elements = collectedInventory.Select(x => ElementCount(x.Shape)).ToArray();
            var results = RunsSideBySide(compute, elements)
                ? RunConcurrently(compute, groups, elements, collectedInventory)
                : RunInTurn(compute, groups, collectedInventory);

            var builder = ImmutableDictionary.CreateBuilder<ModelId, TensorData>();
            for (int i = 0; i < results.Length; i++)
                builder[collectedModelIds[i]] = results[i];
            return builder.ToImmutable();
        }

        /// <summary>
        /// Whether the runs go side by side, each on a single-threaded session, rather than one
        /// after another on sessions that spread each operator over every core. A keyed draw is
        /// hundreds of integer passes, and a backend's thread pool spreads them poorly: measured on
        /// four cores, four single-threaded runs side by side drew 2.2x what the same four drew one
        /// after another multi-threaded. Side by side is only a gain while there is work to put
        /// beside the largest run, so it is taken where no parameter holds more than half of the
        /// elements — a lone large parameter drawn single-threaded would run slower than it does
        /// alone on every core. On the CPU only: a card's session is its own parallelism, and there
        /// the runs keep their order.
        /// </summary>
        private static bool RunsSideBySide(ComputeContext compute, long[] elements)
        {
            if (elements.Length < 2 || compute.Backend.Device != ComputeDevice.Cpu) return false;
            if (_sideBySide is { } decided) return decided;
            if (System.Environment.ProcessorCount < 2) return false;
            if (elements.Any(e => e < 0)) return false;
            double total = elements.Sum(e => (double)e);
            return elements.Max() * 2.0 <= total;
        }

        /// <summary>
        /// The stream positions a keyed draw computes per chunk on <paramref name="compute"/>'s
        /// device: a cache-sized chunk on the CPU, and a far larger one on a card, where each trip
        /// of the chunk loop costs a launch per pass (see <c>RuntimeRng.DeviceChunkPositions</c>).
        /// The values are the same whichever it is.
        /// </summary>
        private static long ChunkPositionsOn(ComputeContext compute)
            => compute.Backend.Device == ComputeDevice.Cpu
                ? Core.Rng.RuntimeRng.CpuChunkPositions
                : Core.Rng.RuntimeRng.DeviceChunkPositions;

        [System.ThreadStatic] private static bool? _sideBySide;

        /// <summary>
        /// Decides <see cref="RunsSideBySide"/> for the initializations this thread starts until
        /// the returned scope is disposed, where a CPU context has two or more parameters to run,
        /// whatever the core count —
        /// for a test to hold the two schedules against each other, and to drive the side-by-side
        /// one over parameters its own rule would run in turn. Thread-scoped, like the other
        /// fault-injection facilities, so tests running in parallel do not see each other's.
        /// </summary>
        internal static System.IDisposable DecideSideBySide(bool sideBySide)
        {
            var previous = _sideBySide;
            _sideBySide = sideBySide;
            return new SideBySideScope(previous);
        }

        private sealed class SideBySideScope(bool? previous) : System.IDisposable
        {
            public void Dispose() => _sideBySide = previous;
        }

        /// <summary>The elements of a shape, or -1 where it is not known; saturates rather than
        /// wrapping.</summary>
        private static long ElementCount(long[]? shape)
        {
            if (shape is null) return -1;
            long n = 1;
            foreach (var d in shape)
            {
                if (d < 0) return -1;
                if (d != 0 && n > long.MaxValue / d) return long.MaxValue;
                n *= d;
            }
            return n;
        }

        /// <summary>A run's own values, fed to its group's session.</summary>
        private static IData[] KeyFeeds(ulong[] keys) => [.. keys.Select(k => (IData)Shorokoo.Globals.TensorData([], k))];

        /// <summary>
        /// Every parameter's initial value, one group at a time: a group's session is disposed
        /// before the next one's is built, so only one session's arena is ever alive; each result
        /// is copied off it as it is produced, and each run hands the arena's unused blocks back as
        /// it ends (see ChunkFor).
        /// </summary>
        private static TensorData[] RunInTurn(
            ComputeContext compute, List<InitGroup> groups,
            List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)> inventory)
        {
            var results = new TensorData[inventory.Count];
            foreach (var group in groups)
            {
                int current = group.Members[0].Param;
                try
                {
                    // Unoptimized, as the one-shot run of an input-less graph is: the session runs
                    // each draw once, so ORT's folding would buy nothing and cost the data's size
                    // in memory (see ComputeContext.IsFullyConstant).
                    using var compiled = compute.Compile(group.Chunk, ShorokooGraphOptimization.DisableAll);
                    var shrinking = compiled.DefaultRunSettings with { ShrinkArenaAfterRun = true };
                    foreach (var (param, keys) in group.Members)
                    {
                        current = param;
                        results[param] = FastProcessorHelper.RehostOffSession(
                            compiled.Execute(KeyFeeds(keys), shrinking)[0].ToTensorData());
                    }
                }
                catch (System.Exception ex) when (IsAllocationFailure(ex))
                {
                    throw Labelled(ex, inventory, current);
                }
            }
            return results;
        }

        /// <summary>
        /// What one parameter's run is modelled to hold while it is in flight: its value a few times
        /// over — the drawn tensor, what the initializer computes from it, the copy taken off the
        /// session — and one chunk's working memory of the draw.
        /// </summary>
        private static long InFlightBytes(long elements)
            => elements > (long.MaxValue - ChunkWorkingBytes) / InFlightBytesPerElement
                ? long.MaxValue
                : elements * InFlightBytesPerElement + ChunkWorkingBytes;

        private const long InFlightBytesPerElement = 32;
        private const long ChunkWorkingBytes = 64L << 20;

        /// <summary>
        /// Every parameter's initial value, the runs side by side (see <see cref="RunsSideBySide"/>).
        ///
        /// <para>Every group's session is built first, one after another on this thread, each with a
        /// single intra-op thread; the runs then go to up to <see cref="System.Environment.ProcessorCount"/>
        /// workers, taken in parameter order, a run starting only while the runs in flight are
        /// modelled to hold no more than a quarter of the memory available to the process
        /// (<see cref="InFlightBytes"/>) — or alone, whatever it holds. The values are the same bit
        /// for bit whichever thread draws them, and each lands in its parameter's own slot.</para>
        ///
        /// <para>A failure stops new runs from starting, and those in flight finish. Runs start in
        /// parameter order, so every parameter before a failed one has been run by then, and the
        /// failure reported is the first parameter's in order that failed, named as CR008. A run
        /// in turn (<see cref="RunInTurn"/>) walks group by group instead, the groups in the order
        /// of their first parameters; it reports the same parameter wherever a failure is the
        /// slice's rather than the moment's — the parameters of one group run one graph at one
        /// shape and fail alike, so the first group to fail is the one holding the first
        /// parameter in order that fails. A failure that comes and goes with the moment, such as
        /// memory another process takes, can name a different parameter under the two. Every
        /// session is disposed however the runs end.</para>
        /// </summary>
        private static TensorData[] RunConcurrently(
            ComputeContext compute, List<InitGroup> groups, long[] elements,
            List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)> inventory)
        {
            int count = inventory.Count;
            var groupOf = new int[count];
            var keysOf = new ulong[count][];
            for (int g = 0; g < groups.Count; g++)
                foreach (var (param, keys) in groups[g].Members)
                    (groupOf[param], keysOf[param]) = (g, keys);

            var costs = elements.Select(InFlightBytes).ToArray();
            long budget = System.Math.Max(costs.Max(),
                System.GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 4);

            var sessions = new CompiledGraph?[groups.Count];
            try
            {
                for (int g = 0; g < groups.Count; g++)
                {
                    try
                    {
                        sessions[g] = compute.Compile(
                            groups[g].Chunk, ShorokooGraphOptimization.DisableAll, intraOpThreads: 1);
                    }
                    catch (System.Exception ex) when (IsAllocationFailure(ex))
                    {
                        throw Labelled(ex, inventory, groups[g].Members[0].Param);
                    }
                }

                var results = new TensorData[count];
                var failures = new System.Exception?[count];
                var gate = new object();
                int next = 0;
                long inFlight = 0;
                bool stopped = false;

                void Work()
                {
                    while (true)
                    {
                        int i;
                        lock (gate)
                        {
                            while (true)
                            {
                                if (next == count || stopped) return;
                                if (inFlight == 0 || inFlight <= budget - costs[next]) break;
                                System.Threading.Monitor.Wait(gate);
                            }
                            i = next++;
                            inFlight += costs[i];
                        }
                        try
                        {
                            var compiled = sessions[groupOf[i]]!;
                            results[i] = FastProcessorHelper.RehostOffSession(compiled.Execute(
                                KeyFeeds(keysOf[i]),
                                compiled.DefaultRunSettings with { ShrinkArenaAfterRun = true })[0].ToTensorData());
                        }
                        catch (System.Exception ex)
                        {
                            failures[i] = ex;
                            lock (gate) stopped = true;
                        }
                        finally
                        {
                            lock (gate)
                            {
                                inFlight -= costs[i];
                                System.Threading.Monitor.PulseAll(gate);
                            }
                        }
                    }
                }

                // Joined before the sessions go, however this thread leaves: a worker still running
                // would be running a disposed session.
                int workers = System.Math.Min(System.Environment.ProcessorCount, count);
                var threads = new List<System.Threading.Thread>(workers - 1);
                try
                {
                    for (int t = 1; t < workers; t++)
                    {
                        var thread = new System.Threading.Thread(Work) { IsBackground = true, Name = "Shorokoo initialization" };
                        thread.Start();
                        threads.Add(thread);
                    }
                    Work();
                }
                finally
                {
                    foreach (var thread in threads) thread.Join();
                }

                for (int i = 0; i < count; i++)
                {
                    if (failures[i] is not { } failure) continue;
                    if (IsAllocationFailure(failure)) throw Labelled(failure, inventory, i);
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
                }
                return results;
            }
            finally
            {
                foreach (var compiled in sessions) compiled?.Dispose();
            }
        }

        /// <summary>
        /// An allocation failure relabelled with the parameter it happened to. The backend reports
        /// an out-of-memory abort as a bare "bad allocation", with no parameter, shape or size —
        /// nothing to separate "this parameter is too large" from "this graph is malformed" (#208).
        /// Each run initializes one parameter, and a group's session is built for parameters of one
        /// shape, so the one being initialized — the group's first while its session is built — IS
        /// the one that failed: name it, with the rest of the model as context; the inner exception
        /// keeps the original diagnosis. Only allocation failures are relabelled: everything else
        /// this can raise (a missing backend package, an unsupported op, a malformed graph) already
        /// says what it is, and keeping its type keeps the catch clauses around this API working.
        /// </summary>
        private static ComputeContextException Labelled(
            System.Exception failure,
            List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)> inventory,
            int failing)
            => new(ErrorCodes.CR008, "FastInitializeModelParams",
                DescribeInventory(inventory, failing) + " Underlying failure: " + failure.Message, failure);

        /// <summary>
        /// The single-parameter slice of the rewritten initialization graph: a copy whose only
        /// output is <paramref name="outputKey"/>, swept down to the nodes that actually feed it.
        ///
        /// <para>Initialization runs one parameter per RUN, on a session built for that
        /// parameter's slice rather than for the whole model. Slicing changes no value: each keyed
        /// draw hangs off its own stream key, and the one way one parameter's initializer reads
        /// another's — being handed that parameter (Shorokoo/Shorokoo#324) — leaves the source's
        /// whole initializer in the slice, where it recomputes the same deterministic value. What
        /// slicing changes is the size of the graph any one session has to build (a shared source
        /// is built once per dependent, not once). That matters because the backend's session
        /// build is SUPERLINEAR in graph size. Measured on an N-layer stack of [384, 384] normal
        /// draws, with constant folding already off (see <c>ComputeContext.IsFullyConstant</c>),
        /// building the whole-model init graph cost 0.23 s at N=1, 0.88 s at N=2, 3.4 s at N=4,
        /// 13.4 s at N=8 and 32 s at N=12 — a clean 4x per doubling, session build alone. Left
        /// whole, a GPT-sized model spends minutes of pure build before the first gradient
        /// (Shorokoo/Shorokoo#195).</para>
        ///
        /// <para>A slice's session build is itself a near-constant cost — a few tenths of a second
        /// on a CPU, whatever the parameter's size, since it is the keyed draw's lowering the
        /// backend builds — so a session per parameter makes rig construction cost that much per
        /// parameter (Shorokoo/Shorokoo#404). It is paid once per distinct slice instead: the
        /// stream key is an input of the slice, not a literal in it (see
        /// <see cref="FastInitKeyedDraws.BuildKeyedDraws"/>), so the slices of parameters with the
        /// same initializer and shape are one graph (<see cref="SameSlice"/>), compiled once and run
        /// once per parameter with that parameter's key. A model whose layers repeat builds a
        /// handful of sessions however deep it is.</para>
        ///
        /// <para>Memory is bounded by three things. Each keyed draw is computed a chunk of stream
        /// positions at a time (see <see cref="FastInitKeyedDraws"/>), so a run's working memory is
        /// a chunk's — hundreds of integer passes over it — rather than hundreds of bytes per element
        /// of the parameter. Each result is copied off its session
        /// (<see cref="FastProcessorHelper.RehostOffSession"/>): a result keeps its session's arena
        /// alive, so N retained results would otherwise hold N arenas. And each run hands its
        /// arena's unused blocks back as it ends (<see cref="RunSettings.ShrinkArenaAfterRun"/>): a
        /// session run a second time on the same shapes lays its intermediates out in one block
        /// sized to the first run's peak, which an arena still holding the first run's blocks
        /// cannot supply, so without the shrink a shared session's arena doubles. What stays is the
        /// parameters' values themselves: a 57.9 M-element model peaks at 2.2 GB of working set
        /// through its whole <c>FromScratch</c> drawn in turn, and 2.8 GB drawn three side by side
        /// (see <see cref="RunConcurrently"/>, which holds what is in flight at once to a quarter of
        /// the memory available).</para>
        ///
        /// <para>Sharing one function BODY across the parameters does not substitute for slicing:
        /// the backend inlines every call site, so the graph it builds is the same size either
        /// way. Cloning and sweeping the graph once per parameter is quadratic in the parameter
        /// count, but it is host-side pointer work against a session build measured in tenths of
        /// a second.</para>
        /// </summary>
        private static InternalComputationGraph ChunkFor(
            InternalComputationGraph workGraph, FastTensorKey outputKey)
        {
            var chunk = workGraph.Clone();
            chunk.SetOutputs([outputKey]);
            FastProcessorHelper.RemoveUnreachableNodes(chunk, keepUnreadInputs: false);
            return chunk;
        }

        /// <summary>The parameters whose slices are one graph: the first one's slice, to compile,
        /// and each member's stream keys in its slice's input order.</summary>
        private sealed class InitGroup(InternalComputationGraph chunk)
        {
            public InternalComputationGraph Chunk { get; } = chunk;
            public List<(int Param, ulong[] Keys)> Members { get; } = [];
        }

        /// <summary>
        /// Whether two slices of the initialization graph compute the same thing up to the identity
        /// of their tensors: node for node the same op, attributes, target function and wiring. Such
        /// slices build backend models that differ at most in names, so either one's session
        /// computes the other's value from the other's inputs. Only a stream key differs between
        /// the slices of same-shaped parameters, and a key is an input, not part of the graph.
        ///
        /// <para>What a node carries for a reader rather than for the computation is not compared:
        /// its stack trace, friendly name and parameter-name template. They differ between
        /// parameters declared on different source lines, or read back from a file, and at most
        /// name things in the model the backend is handed; comparing them would give each such
        /// parameter a session of its own for nothing.</para>
        ///
        /// <para>Conservative by construction: anything it cannot show equal — an attribute of a
        /// kind it does not know, a read of a tensor not yet produced — counts as different, which
        /// costs a session and never a wrong value. A function is the same only when it is the same
        /// object, which is what one keyed body per initializer gives every parameter using it.</para>
        /// </summary>
        private static bool SameSlice(InternalComputationGraph a, InternalComputationGraph b)
        {
            if (a.Nodes.Count != b.Nodes.Count) return false;
            var tensors = new Dictionary<FastTensorKey, FastTensorKey>();
            var nodes = new Dictionary<FastNodeKey, FastNodeKey>();
            for (int i = 0; i < a.Nodes.Count; i++)
            {
                var x = a.Nodes[i];
                var y = b.Nodes[i];
                if (x.OpCode != y.OpCode || !ReferenceEquals(x.TargetFunction, y.TargetFunction))
                    return false;
                if (x.GraphOpenNodeKey is { } openX
                        ? y.GraphOpenNodeKey is not { } openY || !nodes.TryGetValue(openX, out var mapped) || !mapped.Equals(openY)
                        : y.GraphOpenNodeKey is not null)
                    return false;
                if (!SameAttributes(x.Attributes.GetAttributeVals(), y.Attributes.GetAttributeVals()))
                    return false;
                if (x.FullInputs.Count != y.FullInputs.Count) return false;
                foreach (var (group, keysX) in x.FullInputs)
                {
                    if (!y.FullInputs.TryGetValue(group, out var keysY) || keysX.Count != keysY.Count) return false;
                    for (int k = 0; k < keysX.Count; k++)
                        if (keysX[k] is { } kx
                                ? keysY[k] is not { } ky || !tensors.TryGetValue(kx, out var mk) || !mk.Equals(ky)
                                : keysY[k] is not null)
                            return false;
                }
                if (x.FullOutputs.Count != y.FullOutputs.Count) return false;
                foreach (var (group, keysX) in x.FullOutputs)
                {
                    if (!y.FullOutputs.TryGetValue(group, out var keysY) || keysX.Count != keysY.Count) return false;
                    for (int k = 0; k < keysX.Count; k++)
                    {
                        if (keysX[k] is { } kx)
                        {
                            if (keysY[k] is not { } ky) return false;
                            tensors[kx] = ky;
                        }
                        else if (keysY[k] is not null) return false;
                    }
                }
                nodes[x.Key] = y.Key;
            }
            return true;
        }

        private static bool SameAttributes(
            ImmutableDictionary<string, object?> a, ImmutableDictionary<string, object?> b)
        {
            if (a.Count != b.Count) return false;
            foreach (var (name, value) in a)
                if (!b.TryGetValue(name, out var other) || !SameValue(value, other))
                    return false;
            return true;
        }

        private static bool SameValue(object? a, object? b)
        {
            if (ReferenceEquals(a, b)) return true;
            if (a is null || b is null || a.GetType() != b.GetType()) return false;
            switch (a)
            {
                case float f:
                    return System.BitConverter.SingleToInt32Bits(f) == System.BitConverter.SingleToInt32Bits((float)b);
                case double d:
                    return System.BitConverter.DoubleToInt64Bits(d) == System.BitConverter.DoubleToInt64Bits((double)b);
                case string or long or int or bool or DType or System.Enum:
                    return a.Equals(b);
                case System.Array xs:
                {
                    var ys = (System.Array)b;
                    if (xs.Length != ys.Length) return false;
                    for (int i = 0; i < xs.Length; i++)
                        if (!SameValue(xs.GetValue(i), ys.GetValue(i))) return false;
                    return true;
                }
                case TensorAttribute x:
                {
                    var y = (TensorAttribute)b;
                    if (!x.Shape.Equals(y.Shape) || !x.DType.Equals(y.DType) || !x.StorageDType.Equals(y.StorageDType) ||
                        x.HasValues != y.HasValues)
                        return false;
                    if (!x.HasValues) return true;
                    return x.DType == DType.Utf8
                        ? x.Values.SequenceEqual(y.Values)
                        : x.Bytes.SequenceEqual(y.Bytes);
                }
                default:
                    return false;
            }
        }

        /// <summary>
        /// True for the failures #208 is about: a managed out-of-memory, or a native allocation
        /// abort the backend reports only as text ("bad allocation", "std::bad_alloc", ...).
        /// Shares one marker table with the training step's own allocation reporting, so the two
        /// cannot drift into recognizing different sets of the same backend's failures.
        /// </summary>
        private static bool IsAllocationFailure(System.Exception ex)
            => Core.Utils.AllocationFailureReport.IsAllocationFailure(ex);

        /// <summary>
        /// Renders the parameter at <paramref name="failing"/> — the one whose own initialization
        /// session just aborted — and then the model it belongs to as context: the parameter
        /// count, the total element count and byte size, and the largest parameters by size.
        /// Sizes saturate rather than wrap — the parameter that blew its session up is exactly
        /// the one whose element count can overflow Int64, and it has to stay at the top of the
        /// list.
        /// </summary>
        private static string DescribeInventory(
            List<(string? Template, ConcreteModelParamInfo? Info, ModelId Id, DType DType, long[]? Shape)> inventory,
            int failing)
        {
            const long Unknown = -1;

            static long AddSat(long a, long b) => a > long.MaxValue - b ? long.MaxValue : a + b;

            static long ElementCount(long[]? shape)
            {
                if (shape is null) return Unknown;
                long n = 1;
                foreach (var d in shape)
                {
                    if (d < 0) return Unknown;
                    if (d != 0 && n > long.MaxValue / d) return long.MaxValue;
                    n *= d;
                }
                return n;
            }

            static long ByteCount(DType dtype, long elements)
            {
                if (elements < 0) return Unknown;
                int bits;
                try { bits = dtype.EncodingBitCount; }
                catch (UnsupportedDTypeException) { return Unknown; }
                if (bits <= 0) return Unknown;
                return elements > long.MaxValue / bits ? long.MaxValue : elements * bits / 8;
            }

            static string Bytes(long bytes) => bytes < 0
                ? "unknown size"
                : bytes == long.MaxValue ? "more than 8 EiB"
                : bytes >= 1L << 60 ? Fmt(bytes / (double)(1L << 60), "EiB")
                : bytes >= 1L << 50 ? Fmt(bytes / (double)(1L << 50), "PiB")
                : bytes >= 1L << 40 ? Fmt(bytes / (double)(1L << 40), "TiB")
                : bytes >= 1L << 30 ? Fmt(bytes / (double)(1L << 30), "GiB")
                : bytes >= 1L << 20 ? Fmt(bytes / (double)(1L << 20), "MiB")
                : bytes.ToString("N0", CultureInfo.InvariantCulture) + " bytes";

            static string Fmt(double v, string unit) => v.ToString("F2", CultureInfo.InvariantCulture) + " " + unit;

            var sized = inventory
                .Select(x =>
                {
                    var elements = ElementCount(x.Shape);
                    return (x.Template, x.Info, x.Id, x.Shape, DType: x.DType, Elements: elements,
                        Bytes: ByteCount(x.DType, elements));
                })
                .ToArray();

            static string Describe(
                (string? Template, ConcreteModelParamInfo? Info, ModelId Id, long[]? Shape,
                 DType DType, long Elements, long Bytes) p)
                => $"'{p.Info?.ToShorokooIdString() ?? p.Template ?? "<unnamed>"}' " +
                   $"at ModelId [{string.Join(", ", p.Id.Vals)}] " +
                   $"{p.DType} [{(p.Shape is null ? "unknown shape" : string.Join(", ", p.Shape))}] " +
                   $"= {Bytes(p.Bytes)}";

            var totalBytes = sized.Any(x => x.Bytes < 0)
                ? Unknown : sized.Aggregate(0L, (t, x) => AddSat(t, x.Bytes));
            var totalElements = sized.Any(x => x.Elements < 0)
                ? Unknown : sized.Aggregate(0L, (t, x) => AddSat(t, x.Elements));
            var largest = sized.OrderByDescending(x => x.Bytes).Take(5).Select(Describe);

            return $"initializing the model parameter {Describe(sized[failing])} failed. " +
                   $"It is 1 of {sized.Length} " +
                   $"({(totalElements < 0 ? "unknown"
                        : totalElements == long.MaxValue ? "more than 9.2e18"
                        : totalElements.ToString("N0", CultureInfo.InvariantCulture))} " +
                   $"elements, {Bytes(totalBytes)} in total); the largest of them, in order: " +
                   string.Join("; ", largest) + ".";
        }

        /// <summary>
        /// Resolves each parameter's init stream key by EXECUTING the in-graph derivation once
        /// for the whole model (#136: the host runs no RNG itself), in bounded chunks — instead of
        /// embedding a split chain per parameter in the much larger initialization graph.
        /// </summary>
        private static Dictionary<ModelId, ulong> ResolveInitKeys(
            IEnumerable<ModelId> modelIds, RngConfig rngConfig, ComputeContext? computeContext)
        {
            var ids = modelIds.ToArray();
            var resolved = Core.Rng.RngKeyResolver.Resolve(
                [.. ids.Select(id => rngConfig.InitKeySpec(id.Vals))], computeContext);
            var keys = new Dictionary<ModelId, ulong>(ids.Length);
            for (int i = 0; i < ids.Length; i++)
                keys[ids[i]] = resolved[i];
            return keys;
        }

    }
}
