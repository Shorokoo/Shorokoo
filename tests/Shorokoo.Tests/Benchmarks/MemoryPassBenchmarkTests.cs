using Newtonsoft.Json;
using Shorokoo.Graph;
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
/// baseline (<c>Benchmarks/memory-pass-baseline.json</c>).
///
/// <para>The figures are the pass's <b>model</b> of memory, not a reading of allocated bytes —
/// the same numbers the pass optimizes, so this guards the optimizer's output rather than the
/// runtime. That is deliberate: it is what the user documentation's "Sizing an attention run"
/// quotes, and every change to the scheduler, rematerializer, objective or evaluator moves it.
/// Twice in this pass's history a change silently invalidated documented figures with no test
/// failing; this is the test that should have failed.</para>
///
/// <para>Regression is judged per family and in one direction only: measured peak may not exceed
/// the baseline by more than <see cref="PeakRegressionFactor"/>, and modelled compute by more
/// than <see cref="ComputeRegressionFactor"/> — a pass that buys memory with unbounded recompute
/// is a regression too. Improvements do not fail the gate; re-record the baseline to lock them
/// in: <c>SHOROKOO_UPDATE_MEMORY_PASS_BASELINE=1 dotnet test --filter
/// "FullyQualifiedName~MemoryPassBenchmarkTests"</c> rewrites the JSON in place and skips the
/// assertions for that run.</para>
///
/// <para>Modelled figures are deterministic for a given graph, so the factors are tight compared
/// with the wall-clock gates; the slack covers only benign lowering changes that shift a few
/// small tensors.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class MemoryPassBenchmarkTests
{
    private const double PeakRegressionFactor = 1.05;
    private const double ComputeRegressionFactor = 1.25;

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
        }
    }

    // ----- measurement --------------------------------------------------------

    private static MemoryPassMeasurement MeasureSuite()
    {
        var families = new Dictionary<string, FamilyMeasurement>();
        foreach (var (family, model, shape) in Suite)
            families[family] = Measure(model(), shape);
        return new MemoryPassMeasurement { Families = families };
    }

    private static FamilyMeasurement Measure(ComputationGraph model, long[] shape)
    {
        var count = 1L;
        foreach (var d in shape) count *= d;
        var values = new float[count];
        for (var i = 0; i < values.Length; i++)
            values[i] = ((i * 37) % 101) * 0.01f - 0.5f;

        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(shape, values))];
        var rig = TrainingRig.FromScratch(
            model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.01f);

        return new FamilyMeasurement
        {
            UnoptimizedPeakBytes = rig.PreOptimizationEval.PeakMemoryBytes,
            PeakBytes = rig.OptimizationResult.Evaluation.PeakMemoryBytes,
            UnoptimizedComputeTime = rig.PreOptimizationEval.TotalComputeTime,
            ComputeTime = rig.OptimizationResult.Evaluation.TotalComputeTime,
            Nodes = rig.TrainingStepPureGraph.ToInternal().GetAllNodes().Length,
            Strategy = rig.OptimizationResult.StrategyName,
        };
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
        public string Scenario { get; set; } = "L2Loss, SGD, LinuxCPU, net10.0; modelled peak/compute from the memory-aware pass";
        public double PeakRegressionFactor { get; set; } = MemoryPassBenchmarkTests.PeakRegressionFactor;
        public double ComputeRegressionFactor { get; set; } = MemoryPassBenchmarkTests.ComputeRegressionFactor;
        public required Dictionary<string, FamilyMeasurement> Families { get; set; }
    }

    private sealed class FamilyMeasurement
    {
        public long UnoptimizedPeakBytes { get; set; }
        public long PeakBytes { get; set; }
        public double UnoptimizedComputeTime { get; set; }
        public double ComputeTime { get; set; }
        public int Nodes { get; set; }
        public string Strategy { get; set; } = "";
    }
}
