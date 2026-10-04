using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using Shorokoo.Core.Factory;
using Shorokoo.Graph;
using Shorokoo.OnnxRuntime;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Backends;
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
/// Regression gate for the memory-aware pass that <see cref="TrainingRig"/> runs over every lowered
/// training step (<c>MemoryAwareGraphOptimizer</c>), per model family and per backend: ONNX Runtime
/// and PyTorch, each on the host. The pass chooses among its strategies' steps by the peak the
/// backend's own model of a run puts on each (<see cref="GraphOptimizationResult.BackendPeakBytes"/>),
/// so that is the figure gated against the recorded baseline (<c>Benchmarks/memory-pass-baseline.json</c>):
/// the peak of the step the pass hands back may not exceed the baseline's by more than
/// <see cref="PeakRegressionFactor"/>, and a family the baseline records the pass acting on must keep
/// most of the relief recorded for it (<see cref="ReliefRetentionFactor"/>), since a change switching
/// the pass off on one family moves the peak alone by only a few percent. Where a backend's model
/// answers for no step of a family, the pass chose by its own evaluation, and that is the peak
/// recorded and gated instead (<c>PeakModel</c> says which).
///
/// <para>The pass's modelled compute is the other term of the objective it weighs those peaks
/// against, so it is gated too, by <see cref="ComputeRegressionFactor"/>: a pass that buys memory
/// with unbounded recompute is a regression as well.</para>
///
/// <para>On ONNX Runtime each step is also run, in sessions over the model the rig compiles
/// (<see cref="RigModel"/>) at the optimization level and log severity the rig builds its own with,
/// on a feed of the shapes the pass was judged on. <c>RealPeakBytes</c> is the process's resident
/// high-water mark over one run with ONNX Runtime's CPU arena off, each value a real allocation and
/// no state written in place: <c>/proc/self/clear_refs</c> resets the mark and the <c>VmHWM</c> delta
/// is the run's peak, the largest of five. It is measured only on Linux, in a process started with
/// <c>MALLOC_MMAP_THRESHOLD_=16384 MALLOC_TRIM_THRESHOLD_=0 MALLOC_TOP_PAD_=0</c> — without them glibc
/// keeps freed tensors and the mark never moves — and gated, where the baseline has it too, by
/// <see cref="RealPeakRegressionFactor"/> plus <see cref="RealPeakNoiseFloorBytes"/>, since one step
/// whose peak is a few MiB reads anywhere from 1 to 5 MiB in separate processes. Where the pass
/// hands back the step it was handed, the step is read once for both columns. <c>KernelTimeMs</c>
/// (ONNX Runtime's profiler, a subgraph's kernels counted once) and <c>OrtOrderVsGraph</c> /
/// <c>OrtOrderVsReverseDfs</c> (how much of the profiled kernel order follows the model's node order,
/// and the depth-first order ONNX Runtime sorts a graph into) are recorded, not gated.</para>
///
/// <para>Improvements do not fail the gate. Re-record the baseline to lock them in, on Linux under
/// the malloc variables: <c>SHOROKOO_UPDATE_MEMORY_PASS_BASELINE=1 dotnet test --filter
/// "FullyQualifiedName~MemoryPassBenchmarkTests"</c> rewrites the JSON in place and skips the
/// assertions for that run.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class MemoryPassBenchmarkTests
{
    private const double PeakRegressionFactor = 1.05;
    private const double ComputeRegressionFactor = 1.25;
    private const double RealPeakRegressionFactor = 1.10;
    private const long RealPeakNoiseFloorBytes = 4L << 20;
    private const double ReliefRetentionFactor = 0.85;
    private const string BaselineStrategy = "Baseline";
    private const string OnnxRuntime = "onnxruntime";

    private static double Relief(FamilyMeasurement m)
        => m.HandedPeakBytes == 0 ? 0 : 1.0 - (double)m.PeakBytes / m.HandedPeakBytes;

    private const string MallocEnvironment =
        "MALLOC_MMAP_THRESHOLD_=16384 MALLOC_TRIM_THRESHOLD_=0 MALLOC_TOP_PAD_=0";

    private static readonly bool RealMemoryMeasurable =
        OperatingSystem.IsLinux()
        && File.Exists("/proc/self/clear_refs")
        && MallocEnvironment.Split(' ').All(kv =>
            Environment.GetEnvironmentVariable(kv[..kv.IndexOf('=')]) == kv[(kv.IndexOf('=') + 1)..]);

    internal static readonly (string Family, Func<ComputationGraph> Model, long[] Shape)[] Suite =
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

    private static readonly (string Name, Func<ComputeContext> Context)[] Backends =
    [
        (OnnxRuntime, () => new ComputeContext()),
        ("torch",     () => new ComputeContext(new Shorokoo.PyTorch.Cpu.TorchCpuBackend())),
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
        foreach (var (backend, _) in Backends)
        {
            var now = measured.Families[family][backend];
            var was = baseline.Families[family][backend];
            Assert.Equal(was.PeakModel, now.PeakModel);
            Assert.True(now.PeakBytes <= was.PeakBytes * PeakRegressionFactor);
            Assert.True(now.ComputeTime <= was.ComputeTime * ComputeRegressionFactor);
            if (was.Strategy != BaselineStrategy)
            {
                Assert.NotEqual(BaselineStrategy, now.Strategy);
                Assert.True(Relief(now) >= Relief(was) - Math.Abs(Relief(was)) * (1 - ReliefRetentionFactor));
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
        var families = new Dictionary<string, Dictionary<string, FamilyMeasurement>>();
        foreach (var (family, model, shape) in Suite)
        {
            families[family] = [];
            foreach (var (backend, context) in Backends)
                families[family][backend] = Measure(model(), shape, backend, context);
        }
        return new MemoryPassMeasurement
        {
            RealMemoryNote = RealMemoryMeasurable
                ? $"RealPeakBytes = VmHWM delta of one train step, ORT CPU arena off, under {MallocEnvironment}"
                : $"RealPeakBytes not measured: the test host must start with {MallocEnvironment} in its environment",
            Families = families,
        };
    }

    private static FamilyMeasurement Measure(ComputationGraph model, long[] shape, string backend, Func<ComputeContext> newContext)
    {
        IData[] sample = [TensorData(shape, new float[shape.Aggregate(1L, (a, d) => a * d)])];
        using var context = newContext();
        var rig = TrainingRig.FromScratch(model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample,
            new SGDOptimizerHyperparameters { LearningRate = 0.01f }, runtimeContext: context);
        var result = rig.OptimizationResult;
        var chosen = result.AllStrategies.Select(s => s.Graph).ToList().FindIndex(g => ReferenceEquals(g, result.OptimizedGraph));
        var measurement = new FamilyMeasurement
        {
            Strategy = result.StrategyName,
            PeakModel = result.BackendPeakBytes is null ? "pass" : "backend",
            HandedPeakBytes = result.BackendPeakBytes?[0] ?? rig.PreOptimizationEval.PeakMemoryBytes,
            PeakBytes = result.BackendPeakBytes?[chosen] ?? result.Evaluation.PeakMemoryBytes,
            UnoptimizedComputeTime = rig.PreOptimizationEval.TotalComputeTime,
            ComputeTime = result.Evaluation.TotalComputeTime,
            Nodes = rig.TrainingStepPureGraph.ToInternal().GetAllNodes().Length,
        };
        if (backend != OnnxRuntime) return measurement;

        var before = new RealRun(rig.PreOptimizationGraph, rig.OptimizationInputShapes);
        var after = chosen == 0 ? before : new RealRun(rig.TrainingStepPureGraph, rig.OptimizationInputShapes);
        RealRun.MeasurePeaks(before, after);
        before.Profile();
        if (after != before) after.Profile();
        measurement.UnoptimizedRealPeakBytes = before.PeakBytes;
        measurement.RealPeakBytes = after.PeakBytes;
        measurement.UnoptimizedKernelTimeMs = before.KernelTimeMs;
        measurement.KernelTimeMs = after.KernelTimeMs;
        measurement.OrtOrderVsGraph = after.OrderVsGraph;
        measurement.OrtOrderVsReverseDfs = after.OrderVsReverseDfs;
        return measurement;
    }

    // ----- real session -----------------------------------------------------------

    // One training step's model, run in ONNX Runtime sessions at the rig's session profile. Memory is
    // read before any profile is parsed: the parse leaves managed garbage whose gradual decommit
    // lowers RSS for seconds afterwards and masks the small peaks.
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

        // Both steps' peaks in one interleaved sequence, so that each is read in the heap state the
        // other left; a step the pass handed back as it came is read once.
        public static void MeasurePeaks(RealRun first, RealRun second)
        {
            if (!RealMemoryMeasurable) return;
            RealRun[] runs = ReferenceEquals(first, second) ? [first] : [first, second];
            var options = runs.Select(_ => RigSessionOptions()).ToArray();
            var sessions = new List<InferenceSession>();
            try
            {
                foreach (var o in options) o.EnableCpuMemArena = false;
                for (var i = 0; i < runs.Length; i++) sessions.Add(new InferenceSession(runs[i]._model, options[i]));
                var feeds = runs.Select((run, i) => Feeds(sessions[i], run._inputShapes)).ToArray();
                using var runOptions = new RunOptions();
                var peaks = new long[runs.Length];
                for (var pass = 0; pass <= MeasuredRuns; pass++)
                    for (var i = 0; i < runs.Length; i++)
                    {
                        var peak = ReadPeak(sessions[i], feeds[i], runOptions);
                        if (pass > 0) peaks[i] = Math.Max(peaks[i], peak);
                    }
                GC.KeepAlive(feeds);
                for (var i = 0; i < runs.Length; i++) runs[i].PeakBytes = peaks[i];
            }
            finally
            {
                foreach (var session in sessions) session.Dispose();
                foreach (var o in options) o.Dispose();
            }
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
                // The prefix before the switch: ORT reads it when profiling is enabled and
                // ignores a later change, so the other order writes the profile into the process's
                // working directory under ORT's own name and the cleanup below deletes an empty
                // folder.
                options.ProfileOutputPathPrefix = Path.Combine(dir, "profile");
                options.EnableProfiling = true;
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
            OrtBackend.Configure(options, ShorokooGraphOptimization.TrainingStep, ShorokooLogSeverity.Fatal);
            return options;
        }

        private static Dictionary<string, OrtValue> Feeds(InferenceSession session, (Shape Shape, DType DType)[] inputShapes)
        {
            var feeds = new Dictionary<string, OrtValue>();
            for (var i = 0; i < session.InputNames.Count; i++)
                feeds[session.InputNames[i]] = SyntheticFeed.Tensor(inputShapes[i].Shape, inputShapes[i].DType, i);
            return feeds;
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
        var json = JsonConvert.SerializeObject(measured, Formatting.Indented,
            new JsonSerializerSettings { NullValueHandling = NullValueHandling.Ignore });

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
        public string Scenario { get; set; } = "L2Loss, SGD, net10.0, each family on each backend on the host: "
            + "HandedPeakBytes / PeakBytes = the peak the backend's model of a run puts on the step as handed to the pass / as the pass hands it back "
            + "(PeakModel 'pass': that model answered for no step, and these are the pass's own figures); compute = the pass's model; "
            + "on onnxruntime, real peak and kernel time of each step in ONNX Runtime sessions over the model the rig compiles";
        public string RealMemoryNote { get; set; } = "";
        public double PeakRegressionFactor { get; set; } = MemoryPassBenchmarkTests.PeakRegressionFactor;
        public double ComputeRegressionFactor { get; set; } = MemoryPassBenchmarkTests.ComputeRegressionFactor;
        public double ReliefRetentionFactor { get; set; } = MemoryPassBenchmarkTests.ReliefRetentionFactor;
        public double RealPeakRegressionFactor { get; set; } = MemoryPassBenchmarkTests.RealPeakRegressionFactor;
        public long RealPeakNoiseFloorBytes { get; set; } = MemoryPassBenchmarkTests.RealPeakNoiseFloorBytes;
        public required Dictionary<string, Dictionary<string, FamilyMeasurement>> Families { get; set; }
    }

    private sealed class FamilyMeasurement
    {
        public string Strategy { get; set; } = "";
        public string PeakModel { get; set; } = "";
        public long HandedPeakBytes { get; set; }
        public long PeakBytes { get; set; }
        public double UnoptimizedComputeTime { get; set; }
        public double ComputeTime { get; set; }
        public int Nodes { get; set; }
        public long? UnoptimizedRealPeakBytes { get; set; }
        public long? RealPeakBytes { get; set; }
        public double? UnoptimizedKernelTimeMs { get; set; }
        public double? KernelTimeMs { get; set; }
        public double? OrtOrderVsGraph { get; set; }
        public double? OrtOrderVsReverseDfs { get; set; }
    }
}
