using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using Shorokoo.Core.Factory;
using Shorokoo.Graph;
using Shorokoo.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

// ---------------------------------------------------------------------------
// Model families for the memory-pass benchmark. Each is a training graph the
// memory-aware pass runs over inside TrainingRig; together they cover the shapes
// the pass has to handle: a dense MLP, a conv stack, one- and two-layer
// transformer encoders, a recurrent model (whose backward the scheduler cannot
// currently linearize), and dense / chunked attention (SdpaMeanPoolModel and
// ChunkedSdpaMeanPoolModel, shared with the attention coverage tests).
// ---------------------------------------------------------------------------

[Module]
public partial class MemoryPassMlp
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, F]
    {
        var h = Linear.Model(Scalar(512L), Scalar(true)).Call(x).Relu();
        h = Linear.Model(Scalar(512L), Scalar(true)).Call(h).Relu();
        return Linear.Model(Scalar(64L), Scalar(true)).Call(h);
    }
}

[Module]
public partial class MemoryPassConv
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, C, H, W]
    {
        var h = Conv2d.Model(Scalar(32L), Scalar(3L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(x).Relu();
        h = Conv2d.Model(Scalar(32L), Scalar(3L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(h).Relu();
        Vector<int64> spatial = [Scalar(2L), Scalar(3L)];
        return h.Reduce(ReduceKind.Mean, spatial, keepDims: false);
    }
}

[Module]
public partial class MemoryPassEncoder1
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, L, E]
    {
        var y = TransformerEncoderLayer.Model(Scalar(128L), Scalar(4L), Scalar(512L), Scalar(true)).Call(x);
        Vector<int64> seq = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, seq, keepDims: false);
    }
}

[Module]
public partial class MemoryPassEncoder2
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, L, E]
    {
        var y = TransformerEncoderLayer.Model(Scalar(128L), Scalar(4L), Scalar(512L), Scalar(true)).Call(x);
        y = TransformerEncoderLayer.Model(Scalar(128L), Scalar(4L), Scalar(512L), Scalar(true)).Call(y);
        Vector<int64> seq = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, seq, keepDims: false);
    }
}

[Module]
public partial class MemoryPassLstm
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, L, F]
    {
        var y = Recurrent.LSTM(x, 128).y;
        Vector<int64> seq = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, seq, keepDims: false);
    }
}

/// <summary>
/// Regression gate for the memory-aware pass that <see cref="TrainingRig"/> runs over every
/// lowered training step (<c>MemoryAwareGraphOptimizer</c>): per model family, the pass's own
/// modelled peak activation bytes and modelled compute must not regress against the recorded
/// baseline (<c>Benchmarks/memory-pass-baseline.json</c>), and neither may the <b>real</b> peak
/// memory of the graph it emits.
///
/// <para>Two kinds of figure sit side by side per family, each for the pre-optimization graph
/// (<c>Unoptimized…</c>) and for the pass's output. The modelled ones (<c>PeakBytes</c>,
/// <c>ComputeTime</c>) are the pass's own <b>model</b> of memory and compute — the numbers it
/// optimizes, so they guard the optimizer's output rather than the runtime; they are what the
/// user documentation's "Sizing an attention run" quotes, and every change to the scheduler,
/// rematerializer, objective or evaluator moves them. Twice in this pass's history a change
/// silently invalidated documented figures with no test failing; this is the test that should
/// have failed. The real ones (<c>RealPeakBytes</c>, <c>KernelTimeMs</c>) come from running the
/// same graph in an ONNX Runtime session built exactly as the rig builds its own — the same
/// ONNX model (<see cref="FastOnnxModelBuilder.BuildInternalOnnxModel"/>, ONNX-prepped), the
/// same optimization level and log severity — on a feed of the shapes the pass was judged on
/// (<see cref="TrainingRig.OptimizationInputShapes"/>). The model has been seen to diverge from
/// reality by tens of percent in both directions, so the pass is judged by both.</para>
///
/// <para><c>RealPeakBytes</c> is the process's resident high-water mark over one training step:
/// <c>/proc/self/clear_refs</c> resets the mark, the step runs with ORT's CPU arena disabled so
/// every activation is a real allocation, and the <c>VmHWM</c> delta is the step's peak. The
/// figure is only meaningful when glibc's dynamic mmap threshold is off — otherwise freed
/// tensors stay in the heap, the mark never moves, and every delta reads ~0 — so the process
/// must start with <c>MALLOC_MMAP_THRESHOLD_=16384 MALLOC_TRIM_THRESHOLD_=0 MALLOC_TOP_PAD_=0</c>
/// in its environment (tensors under that threshold are heap-served; a <c>malloc_trim</c> before
/// each reading keeps them visible once the heap has holes, but a family whose activations are all
/// that small still reads near its page-granularity floor). Nothing in-process can set that after the fact, and spawning a child test
/// host with it would put a build-sized process inside a measurement class, so the test reads the
/// environment instead: with it, the column is measured in-process (repeat runs agree to about
/// 0.2 MB; the largest of five is kept, since the process can only mask a peak, never inflate
/// one); without it, <c>RealPeakBytes</c> is recorded
/// as <c>null</c>, the JSON carries a note saying why, and only the modelled columns are gated.
/// The gate invocation that runs this class sets the variables. <c>KernelTimeMs</c> — the summed
/// kernel time of one step from ORT's profiler, nested subgraph kernels not double counted — and
/// the execution-order figures need no environment and are always recorded; they use a second
/// session with the rig's default arena, run after both memory readings, since the profiler's
/// event buffer and the parse of its output would otherwise pollute them.</para>
///
/// <para><c>OrtOrderVsGraph</c> and <c>OrtOrderVsReverseDfs</c> record how faithfully ORT ran the
/// optimized graph in the order the pass emitted it: the longest common subsequence between the
/// profiled kernel sequence and, respectively, the model's node order and the order ORT's
/// topological sort is expected to produce (a DFS from the leaves, higher-index siblings first),
/// as a fraction of the kernels matched by name. They are diagnostics — a low first figure means
/// the schedule the pass computed is not the one ORT executes.</para>
///
/// <para>Regression is judged per family and in one direction only: modelled peak may not exceed
/// the baseline by more than <see cref="PeakRegressionFactor"/>, modelled compute by more than
/// <see cref="ComputeRegressionFactor"/> — a pass that buys memory with unbounded recompute is a
/// regression too — and real peak by more than <see cref="RealPeakRegressionFactor"/> plus a
/// <see cref="RealPeakNoiseFloorBytes"/> allowance for the page-granularity noise of a family
/// whose tensors are heap-served (three recordings of the LSTM spread over ±0.4 MB), checked
/// only when both the baseline and this run measured it. Kernel time is recorded, not gated: it
/// is wall clock and the perf baseline already budgets that. Improvements do not fail the gate;
/// re-record the baseline to lock them in: <c>SHOROKOO_UPDATE_MEMORY_PASS_BASELINE=1 dotnet test
/// --filter "FullyQualifiedName~MemoryPassBenchmarkTests"</c> (with the malloc variables set)
/// rewrites the JSON in place and skips the assertions for that run.</para>
///
/// <para>Modelled figures are deterministic for a given graph, so their factors are tight
/// compared with the wall-clock gates; the slack covers only benign lowering changes that shift
/// a few small tensors. Real memory carries allocator and GC noise, hence the looser factor.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class MemoryPassBenchmarkTests
{
    private const double PeakRegressionFactor = 1.05;
    private const double ComputeRegressionFactor = 1.25;
    private const double RealPeakRegressionFactor = 1.10;
    private const long RealPeakNoiseFloorBytes = 1L << 20;
    private const string BaselineStrategy = "Baseline";
    private const double ReliefRetentionFactor = 0.85;

    /// <summary>The share of the unoptimized peak the pass removed.</summary>
    private static double Relief(FamilyMeasurement m)
        => m.UnoptimizedPeakBytes == 0 ? 0 : 1.0 - (double)m.PeakBytes / m.UnoptimizedPeakBytes;

    private const string MallocEnvironment =
        "MALLOC_MMAP_THRESHOLD_=16384 MALLOC_TRIM_THRESHOLD_=0 MALLOC_TOP_PAD_=0";

    private static readonly bool RealMemoryMeasurable =
        OperatingSystem.IsLinux()
        && File.Exists("/proc/self/clear_refs")
        && MallocEnvironment.Split(' ').All(kv =>
            Environment.GetEnvironmentVariable(kv[..kv.IndexOf('=')]) == kv[(kv.IndexOf('=') + 1)..]);

    private static readonly (string Family, Func<ComputationGraph> Model, long[] Shape)[] Suite =
    [
        ("mlp",         () => MemoryPassMlp.ComputationGraph,          [64L, 256L]),
        ("conv",        () => MemoryPassConv.ComputationGraph,         [8L, 3L, 64L, 64L]),
        ("encoder1",    () => MemoryPassEncoder1.ComputationGraph,     [8L, 128L, 128L]),
        ("encoder2",    () => MemoryPassEncoder2.ComputationGraph,     [8L, 128L, 128L]),
        ("lstm",        () => MemoryPassLstm.ComputationGraph,         [8L, 32L, 64L]),
        ("attn-dense",  () => SdpaMeanPoolModel.ComputationGraph,      [2L, 4L, 256L, 32L]),
        ("attn-d64",    () => SdpaMeanPoolModel.ComputationGraph,      [2L, 4L, 256L, 64L]),
        ("attn-chunk4", () => ChunkedSdpaMeanPoolModel.ComputationGraph, [2L, 4L, 256L, 32L]),
    ];

    [Fact]
    public void MemoryPassStaysWithinBaseline()
    {
        var measured = MeasureSuite();

        if (Environment.GetEnvironmentVariable("SHOROKOO_UPDATE_MEMORY_PASS_BASELINE") is "1" or "true")
        {
            WriteBaselineToSource(measured);
            return;
        }

        var baseline = LoadBaseline();
        foreach (var (family, _, _) in Suite)
        {
            var now = measured.Families[family];
            var was = baseline.Families[family];
            Assert.True(now.PeakBytes <= was.PeakBytes * PeakRegressionFactor);
            Assert.True(now.ComputeTime <= was.ComputeTime * ComputeRegressionFactor);

            // A family the pass used to act on must still get most of the relief it got. The
            // factors above are one-sided with slack, so a change that switches the pass off on
            // one family reads as a couple of percent and slips through; that is how a premature
            // stop in the rematerializer's candidate walk once disabled the pass on the one-layer
            // encoder, and the same stop quietly cost chunked attention part of its relief while
            // leaving the strategy name intact.
            if (was.Strategy != BaselineStrategy)
            {
                Assert.NotEqual(BaselineStrategy, now.Strategy);
                Assert.True(Relief(now) >= Relief(was) * ReliefRetentionFactor);
            }
            if (was.RealPeakBytes is long wasReal && now.RealPeakBytes is long nowReal)
                Assert.True(nowReal <= wasReal * RealPeakRegressionFactor + RealPeakNoiseFloorBytes);
        }
    }

    // ----- measurement --------------------------------------------------------


    /// <summary>The ONNX model exactly as the rig compiles it: ONNX-prepped, with the concrete
    /// input dims the pass was judged on stamped on every input, so ORT folds the shape
    /// arithmetic the same way and plans the same buffers.</summary>
    internal static ModelProto RigModel(ComputationGraph graph, (Shape Shape, DType DType)[] inputShapes)
        => FastOnnxModelBuilder.BuildInternalOnnxModel(graph.ToInternal(), prepForOnnx: true,
            inputDims: inputShapes.Select(s => (long[]?)s.Shape.Dims.Select(d => (long)d).ToArray()).ToList());

    internal static byte[] RigModelBytes(ComputationGraph graph, (Shape Shape, DType DType)[] inputShapes)
    {
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, RigModel(graph, inputShapes));
        return stream.ToArray();
    }

    private static MemoryPassMeasurement MeasureSuite()
    {
        var families = new Dictionary<string, FamilyMeasurement>();
        foreach (var (family, model, shape) in Suite)
            families[family] = Measure(model(), shape);
        return new MemoryPassMeasurement
        {
            RealMemoryNote = RealMemoryMeasurable
                ? $"RealPeakBytes = VmHWM delta of one train step, ORT CPU arena off, under {MallocEnvironment}"
                : $"RealPeakBytes not measured: the test host must start with {MallocEnvironment} in its environment",
            Families = families,
        };
    }

    private static FamilyMeasurement Measure(ComputationGraph model, long[] shape)
    {
        var count = 1L;
        foreach (var d in shape) count *= d;

        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(shape, FloatPattern(count)))];
        var rig = TrainingRig.FromScratch(
            model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.01f);

        var before = new RealRun(rig.PreOptimizationGraph, rig.OptimizationInputShapes);
        var after = new RealRun(rig.TrainingStepPureGraph, rig.OptimizationInputShapes);
        RealRun.MeasurePeaks(before, after);
        before.Profile();
        after.Profile();

        return new FamilyMeasurement
        {
            UnoptimizedPeakBytes = rig.PreOptimizationEval.PeakMemoryBytes,
            PeakBytes = rig.OptimizationResult.Evaluation.PeakMemoryBytes,
            UnoptimizedRealPeakBytes = before.PeakBytes,
            RealPeakBytes = after.PeakBytes,
            UnoptimizedComputeTime = rig.PreOptimizationEval.TotalComputeTime,
            ComputeTime = rig.OptimizationResult.Evaluation.TotalComputeTime,
            UnoptimizedKernelTimeMs = before.KernelTimeMs,
            KernelTimeMs = after.KernelTimeMs,
            OrtOrderVsGraph = after.OrderVsGraph,
            OrtOrderVsReverseDfs = after.OrderVsReverseDfs,
            Nodes = rig.TrainingStepPureGraph.ToInternal().GetAllNodes().Length,
            Strategy = rig.OptimizationResult.StrategyName,
        };
    }

    private static float[] FloatPattern(long count)
    {
        var values = new float[count];
        for (var i = 0; i < values.Length; i++)
            values[i] = ((i * 37) % 101) * 0.01f - 0.5f;
        return values;
    }

    // ----- real session -----------------------------------------------------------

    /// <summary>
    /// One training-step graph run in ORT sessions built as the rig builds its own. Memory is read
    /// before any profile is parsed: the parse leaves managed garbage whose gradual decommit lowers
    /// RSS for seconds afterwards and masks the small peaks.
    /// </summary>
    private sealed class RealRun
    {
        private const int MeasuredRuns = 5;

        private readonly byte[] _model;
        private readonly (Shape Shape, DType DType)[] _inputShapes;
        private readonly string[] _graphOrder;
        private readonly string[] _reverseDfsOrder;

        public long? PeakBytes { get; private set; }
        public double KernelTimeMs { get; private set; }
        public double OrderVsGraph { get; private set; }
        public double OrderVsReverseDfs { get; private set; }

        public RealRun(ComputationGraph graph, (Shape Shape, DType DType)[] inputShapes)
        {
            var proto = RigModel(graph, inputShapes);
            var stream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(stream, proto);
            _model = stream.ToArray();
            _inputShapes = inputShapes;
            _graphOrder = proto.Graph.Nodes.Where(n => n.OpType != "Constant").Select(n => n.Name).ToArray();
            _reverseDfsOrder = ReverseDfsOrder(proto.Graph);
        }

        /// <summary>
        /// Reads both graphs' peaks in one interleaved sequence: a second session measured after a
        /// first one reads higher on a family whose tensors are heap-served (the LSTM's 4 KB
        /// per-iteration values), because the heap the first run left behind is what the second
        /// allocates from. Alternating the two puts them in the same heap state, so the pair is
        /// comparable even where the absolute reading is not.
        /// </summary>
        public static void MeasurePeaks(RealRun first, RealRun second)
        {
            if (!RealMemoryMeasurable) return;
            using var firstOptions = RigSessionOptions();
            using var secondOptions = RigSessionOptions();
            firstOptions.EnableCpuMemArena = false;
            secondOptions.EnableCpuMemArena = false;
            using var firstSession = new InferenceSession(first._model, firstOptions);
            using var secondSession = new InferenceSession(second._model, secondOptions);
            var firstFeeds = Feeds(firstSession, first._inputShapes);
            var secondFeeds = Feeds(secondSession, second._inputShapes);
            using var runOptions = new RunOptions();

            long firstPeak = 0, secondPeak = 0;
            for (var run = 0; run <= MeasuredRuns; run++)
            {
                var a = ReadPeak(firstSession, firstFeeds, runOptions);
                var b = ReadPeak(secondSession, secondFeeds, runOptions);
                if (run == 0) continue;
                firstPeak = Math.Max(firstPeak, a);
                secondPeak = Math.Max(secondPeak, b);
            }
            GC.KeepAlive(firstFeeds);
            GC.KeepAlive(secondFeeds);
            first.PeakBytes = firstPeak;
            second.PeakBytes = secondPeak;
        }

        // Tensors under the mmap threshold come from the heap, and a heap full of resident free
        // holes serves them without a page fault, so trim the holes away first: reuse then faults
        // them back in and shows in RSS. Anything lowering RSS during a run masks part of the
        // peak, and nothing else in the process allocates during one, so wait for RSS to settle
        // and keep the largest reading.
        private static long ReadPeak(InferenceSession session, Dictionary<string, OrtValue> feeds, RunOptions runOptions)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            malloc_trim(0);
            WaitForStableRss();
            File.WriteAllText("/proc/self/clear_refs", "5");
            var before = ProcStatusBytes("VmHWM:");
            var outputs = session.Run(runOptions, feeds, session.OutputNames);
            var after = ProcStatusBytes("VmHWM:");
            foreach (var o in outputs) o.Dispose();
            return after - before;
        }

        public void Profile()
        {
            var dir = Path.Combine(Path.GetTempPath(), "shorokoo-memory-pass-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                using var options = RigSessionOptions();
                options.EnableProfiling = true;
                options.ProfileOutputPathPrefix = Path.Combine(dir, "profile");
                using var session = new InferenceSession(_model, options);
                var feeds = Feeds(session, _inputShapes);
                using var runOptions = new RunOptions();
                for (var run = 0; run <= MeasuredRuns; run++)
                    foreach (var o in session.Run(runOptions, feeds, session.OutputNames)) o.Dispose();
                GC.KeepAlive(feeds);

                var events = ReadEvents(session.EndProfiling());
                var kernels = events.Where(e => e.Cat == "Node" && e.Name.EndsWith("_kernel_time")).OrderBy(e => e.Ts).ToList();
                var runs = events.Where(e => e.Cat == "Session" && e.Name == "model_run").Skip(1)
                    .Select(run => TopLevel(kernels.Where(k => k.Ts >= run.Ts && k.Ts <= run.Ts + run.Dur)).ToList())
                    .ToList();

                KernelTimeMs = runs.Min(run => run.Sum(k => k.Dur)) / 1000.0;
                var observed = runs[^1].Select(k => k.Name[..^"_kernel_time".Length]).Where(_graphOrder.Contains).Distinct().ToArray();
                OrderVsGraph = Fidelity(observed, _graphOrder);
                OrderVsReverseDfs = Fidelity(observed, _reverseDfsOrder);
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        private readonly record struct ProfileEvent(string Cat, string Name, long Ts, long Dur);

        // Streamed: a whole-file JArray of a few thousand kernels x six runs is tens of MB of
        // garbage, and its decommit would shadow the next family's memory readings.
        private static List<ProfileEvent> ReadEvents(string path)
        {
            var events = new List<ProfileEvent>();
            using var reader = new JsonTextReader(new StreamReader(path));
            while (reader.Read())
            {
                if (reader.TokenType != JsonToken.StartObject) continue;
                string cat = "", name = "";
                long ts = 0, dur = 0;
                while (reader.Read() && reader.TokenType != JsonToken.EndObject)
                {
                    var key = (string)reader.Value!;
                    reader.Read();
                    switch (key)
                    {
                        case "cat": cat = (string)reader.Value!; break;
                        case "name": name = (string)reader.Value!; break;
                        case "ts": ts = Convert.ToInt64(reader.Value); break;
                        case "dur": dur = Convert.ToInt64(reader.Value); break;
                        default: reader.Skip(); break;
                    }
                }
                events.Add(new ProfileEvent(cat, name, ts, dur));
            }
            return events;
        }

        // Kernels of a subgraph (a Loop body, say) are profiled inside their parent's interval;
        // only intervals that start after the previous top-level kernel ended are top level.
        private static IEnumerable<ProfileEvent> TopLevel(IEnumerable<ProfileEvent> kernels)
        {
            var end = long.MinValue;
            foreach (var k in kernels)
            {
                if (k.Ts < end) continue;
                end = k.Ts + k.Dur;
                yield return k;
            }
        }

        /// <summary>
        /// The session exactly as the rig builds its own: the <see cref="ShorokooGraphOptimization.TrainingStep"/>
        /// profile. Plain ORT_ENABLE_ALL is not it — its CommonSubexpressionElimination merges every
        /// rematerialization clone back, so the optimized graph measures the same as the unoptimized one.
        /// </summary>
        private static SessionOptions RigSessionOptions()
        {
            var options = new SessionOptions();
            OrtSessionFactory.Configure(options, ShorokooGraphOptimization.TrainingStep, ShorokooLogSeverity.Fatal);
            return options;
        }

        private static Dictionary<string, OrtValue> Feeds(InferenceSession session, (Shape Shape, DType DType)[] inputShapes)
        {
            var feeds = new Dictionary<string, OrtValue>();
            for (var i = 0; i < session.InputNames.Count; i++)
                feeds[session.InputNames[i]] = Synthesize(inputShapes[i].Shape, inputShapes[i].DType);
            return feeds;
        }

        private static OrtValue Synthesize(Shape shape, DType dtype)
        {
            var n = shape.Count;
            if (dtype == DType.Float32) return OrtValue.CreateTensorValueFromMemory(FloatPattern(n), shape.Dims);
            if (dtype == DType.Int64) return OrtValue.CreateTensorValueFromMemory(new long[n], shape.Dims);
            if (dtype == DType.Int32) return OrtValue.CreateTensorValueFromMemory(new int[n], shape.Dims);
            if (dtype == DType.Bool) return OrtValue.CreateTensorValueFromMemory(new bool[n], shape.Dims);
            throw new NotSupportedException(dtype.ToString());
        }

        [System.Runtime.InteropServices.DllImport("libc")]
        private static extern int malloc_trim(nuint pad);

        private static void WaitForStableRss()
        {
            var last = ProcStatusBytes("VmRSS:");
            var stable = 0;
            for (var i = 0; i < 800 && stable < 10; i++)
            {
                Thread.Sleep(25);
                var now = ProcStatusBytes("VmRSS:");
                stable = now == last ? stable + 1 : 0;
                last = now;
            }
        }

        private static long ProcStatusBytes(string key)
        {
            foreach (var line in File.ReadLines("/proc/self/status"))
                if (line.StartsWith(key, StringComparison.Ordinal))
                    return long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) * 1024;
            throw new InvalidOperationException(key + " not found in /proc/self/status");
        }

        // ----- execution order ---------------------------------------------------------

        private static double Fidelity(string[] observed, string[] expected)
        {
            var index = new Dictionary<string, int>();
            for (var i = 0; i < expected.Length; i++) index[expected[i]] = i;
            var ranks = observed.Where(index.ContainsKey).Select(n => index[n]).ToArray();
            return ranks.Length == 0 ? 0 : (double)LongestIncreasingRun(ranks) / ranks.Length;
        }

        private static int LongestIncreasingRun(int[] ranks)
        {
            var tails = new List<int>();
            foreach (var r in ranks)
            {
                var at = tails.BinarySearch(r);
                if (at < 0) at = ~at;
                if (at == tails.Count) tails.Add(r); else tails[at] = r;
            }
            return tails.Count;
        }

        // ORT's Graph::PerformTopologicalSortAndCheckIsAcyclic, on the node set it sees (Constant
        // nodes become initializers): nodes with no producer first in file order, then a DFS from
        // the leaves, each node's producers pushed in ascending index so the highest pops first.
        private static string[] ReverseDfsOrder(GraphProto graph)
        {
            var nodes = graph.Nodes.Where(n => n.OpType != "Constant").ToArray();
            var producer = new Dictionary<string, int>();
            for (var i = 0; i < nodes.Length; i++)
                foreach (var o in nodes[i].Outputs) producer[o] = i;
            var inputs = nodes.Select(n => n.Inputs.Where(producer.ContainsKey).Select(x => producer[x]).Distinct().OrderBy(x => x).ToArray()).ToArray();
            var hasConsumer = new bool[nodes.Length];
            foreach (var ins in inputs) foreach (var p in ins) hasConsumer[p] = true;

            var order = new List<int>();
            var seen = new HashSet<int>();
            var added = new HashSet<int>();
            for (var i = 0; i < nodes.Length; i++)
                if (inputs[i].Length == 0) { order.Add(i); seen.Add(i); added.Add(i); }

            var stack = new Stack<int>();
            for (var i = 0; i < nodes.Length; i++)
                if (!hasConsumer[i]) stack.Push(i);
            while (stack.Count > 0)
            {
                var current = stack.Pop();
                if (added.Contains(current)) continue;
                if (seen.Contains(current)) { order.Add(current); added.Add(current); continue; }
                seen.Add(current);
                stack.Push(current);
                foreach (var p in inputs[current])
                    if (!seen.Contains(p)) stack.Push(p);
            }
            return order.Select(i => nodes[i].Name).ToArray();
        }
    }

    // ----- baseline I/O --------------------------------------------------------

    private const string BaselineFileName = "memory-pass-baseline.json";

    private static string BaselineOutputPath =>
        Path.Combine(AppContext.BaseDirectory, "Benchmarks", BaselineFileName);

    private static MemoryPassMeasurement LoadBaseline()
    {
        var path = BaselineOutputPath;
        // Absent baseline: record it with SHOROKOO_UPDATE_MEMORY_PASS_BASELINE=1.
        Assert.True(File.Exists(path));
        var baseline = JsonConvert.DeserializeObject<MemoryPassMeasurement>(File.ReadAllText(path));
        Assert.NotNull(baseline);
        return baseline!;
    }

    private static void WriteBaselineToSource(MemoryPassMeasurement measured)
    {
        var json = JsonConvert.SerializeObject(measured, Formatting.Indented);

        var outPath = BaselineOutputPath;
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        File.WriteAllText(outPath, json);

        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "Benchmarks", BaselineFileName);
            if (File.Exists(candidate) && !string.Equals(candidate, outPath, StringComparison.Ordinal))
            {
                File.WriteAllText(candidate, json);
                break;
            }
            if (File.Exists(Path.Combine(dir.FullName, "Shorokoo.Tests.csproj")))
            {
                var src = Path.Combine(dir.FullName, "Benchmarks", BaselineFileName);
                Directory.CreateDirectory(Path.GetDirectoryName(src)!);
                File.WriteAllText(src, json);
                break;
            }
            dir = dir.Parent;
        }
    }

    private sealed class MemoryPassMeasurement
    {
        public string Scenario { get; set; } = "L2Loss, SGD, LinuxCPU, net10.0; modelled peak/compute from the memory-aware pass, real peak/kernel time from ORT sessions built as the rig builds its own";
        public string RealMemoryNote { get; set; } = "";
        public double PeakRegressionFactor { get; set; } = MemoryPassBenchmarkTests.PeakRegressionFactor;
        public double ComputeRegressionFactor { get; set; } = MemoryPassBenchmarkTests.ComputeRegressionFactor;
        public double RealPeakRegressionFactor { get; set; } = MemoryPassBenchmarkTests.RealPeakRegressionFactor;
        public long RealPeakNoiseFloorBytes { get; set; } = MemoryPassBenchmarkTests.RealPeakNoiseFloorBytes;
        public required Dictionary<string, FamilyMeasurement> Families { get; set; }
    }

    private sealed class FamilyMeasurement
    {
        public long UnoptimizedPeakBytes { get; set; }
        public long PeakBytes { get; set; }
        public long? UnoptimizedRealPeakBytes { get; set; }
        public long? RealPeakBytes { get; set; }
        public double UnoptimizedComputeTime { get; set; }
        public double ComputeTime { get; set; }
        public double UnoptimizedKernelTimeMs { get; set; }
        public double KernelTimeMs { get; set; }
        public double OrtOrderVsGraph { get; set; }
        public double OrtOrderVsReverseDfs { get; set; }
        public int Nodes { get; set; }
        public string Strategy { get; set; } = "";
    }
}
