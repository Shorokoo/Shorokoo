using System.Diagnostics;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;
using Newtonsoft.Json;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// The fixed linear model behind the perf baseline: a single trainable affine
/// map <c>[4,2] → [4,1]</c> (weight <c>[1,2]</c> + bias <c>[1]</c>), no hypers.
/// This is the pinned "R-1" release-validation scenario, frozen here as code so
/// the benchmark measures the same shape every run.
/// </summary>
[Module]
public partial class PerfBaselineLinearModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var w = Shorokoo.Modules.Initializers.XavierUniform.Init([Scalar(1L), Scalar(2L)]); // [outFeatures, inFeatures]
        var b = Shorokoo.Modules.Initializers.Zeros.Init([Scalar(1L)]).Vec();               // [outFeatures]
        return x.MatMul(w.Transpose([1L, 0L])) + b;                         // [4,2]@[2,1] + [1] -> [4,1]
    }
}

/// <summary>
/// Code-pinned performance gate for the training hot path — the structural fix
/// called for by <c>release-test-plan</c> <c>R-1</c> and <c>test-suite-gaps #4</c>
/// ("No performance baselines"). It measures the four phases R-1 tracks for the
/// pinned linear scenario — graph-build (concretize), <c>TrainingRig.FromScratch</c>,
/// <c>Compile</c>, and steady-state <c>TrainStep</c> throughput — and compares each
/// against a baseline checked in beside this test
/// (<c>Benchmarks/perf-baseline.json</c>).
///
/// <para>
/// The gate is deliberately loose: it fails only when the current run is more than
/// <see cref="SlowdownFactor"/>× slower than the baseline (for throughput, less than
/// 1/<see cref="SlowdownFactor"/>× as fast). A 2× band absorbs the ordinary
/// cross-machine / fresh-container variance that left R-1 "inconclusive" while still
/// catching a real order-of-changes regression. Each phase is measured best-of-N to
/// further suppress noise.
/// </para>
///
/// <para>
/// To re-record the baseline (e.g. after an intentional perf shift or a hardware
/// change), run this test with the environment variable
/// <c>SHOROKOO_UPDATE_PERF_BASELINE=1</c> set: it rewrites the source
/// <c>perf-baseline.json</c> in place with the freshly measured numbers and then
/// passes. Commit the updated file.
/// </para>
/// </summary>
[Trait("Domain", "Training")]
[Trait("Purpose", "Coverage")]
public class TrainingPerfBaselineTests
{
    /// <summary>Current must not be slower than this multiple of the baseline.</summary>
    private const double SlowdownFactor = 2.0;

    // Pinned scenario geometry: 4 samples, 2 features in, 1 feature out.
    private static readonly long[] InputShape = [4L, 2L];
    private static readonly long[] TargetShape = [4L, 1L];

    // Measurement budgets. Setup phases (one-shot) are taken best-of over a handful
    // of full rebuilds; throughput is a warmed steady-state rate.
    private const int SetupRebuilds = 5;
    private const int ThroughputWarmupSteps = 25;
    private const int ThroughputMeasuredSteps = 300;

    [Fact]
    public void TrainingHotPathStaysWithinBaseline()
    {
        var measured = MeasureScenario();

        if (Environment.GetEnvironmentVariable("SHOROKOO_UPDATE_PERF_BASELINE") is "1" or "true")
        {
            measured.SlowdownFactor = SlowdownFactor;
            WriteBaselineToSource(measured);
            return; // recording mode: don't assert against the very numbers we just wrote
        }

        var baseline = LoadBaseline();

        // Times: current must be <= factor × baseline. Throughput: current must be >= baseline / factor.
        AssertNotSlower("concretize (graph build)", measured.ConcretizeMs, baseline.ConcretizeMs, baseline.SlowdownFactor);
        AssertNotSlower("FromScratch", measured.FromScratchMs, baseline.FromScratchMs, baseline.SlowdownFactor);
        AssertNotSlower("Compile", measured.CompileMs, baseline.CompileMs, baseline.SlowdownFactor);
        AssertNotLessThroughput("steady-state TrainStep throughput",
            measured.TrainStepsPerSecond, baseline.TrainStepsPerSecond, baseline.SlowdownFactor);
    }

    private static void AssertNotSlower(string phase, double measuredMs, double baselineMs, double factor)
    {
        double budget = baselineMs * factor;
        Assert.True(measuredMs <= budget,
            $"{phase} regressed: {measuredMs:F1} ms > {factor:F1}× baseline {baselineMs:F1} ms (budget {budget:F1} ms). " +
            $"If this is an accepted shift, re-record with SHOROKOO_UPDATE_PERF_BASELINE=1.");
    }

    private static void AssertNotLessThroughput(string phase, double measured, double baseline, double factor)
    {
        double floor = baseline / factor;
        Assert.True(measured >= floor,
            $"{phase} regressed: {measured:F0}/s < baseline {baseline:F0}/s ÷ {factor:F1} (floor {floor:F0}/s). " +
            $"If this is an accepted shift, re-record with SHOROKOO_UPDATE_PERF_BASELINE=1.");
    }

    // ----- measurement -------------------------------------------------------

    private static PerfMeasurement MeasureScenario()
    {
        var baseGraph = PerfBaselineLinearModel.ComputationGraph;
        var exampleInput = TensorData(InputShape, new float[8]);

        double concretizeMs = double.MaxValue, fromScratchMs = double.MaxValue, compileMs = double.MaxValue;

        // Best-of-N over full rebuilds — the minimum is the least noise-contaminated estimate.
        for (int i = 0; i < SetupRebuilds; i++)
        {
            var ctx = new ComputeContext();

            var sw = Stopwatch.StartNew();
            _ = baseGraph.ToConcreteArchitecture(baseGraph.FromOrderedInputs([exampleInput]), ctx);
            sw.Stop();
            concretizeMs = Math.Min(concretizeMs, sw.Elapsed.TotalMilliseconds);

            sw.Restart();
            var rig = TrainingRig.FromScratch(
                baseGraph, Losses.L2Loss, Optimizers.Adam,
                baseGraph.FromOrderedInputs([exampleInput]),
                new AdamOptimizerHyperparameters { LearningRate = 1e-3f });
            sw.Stop();
            fromScratchMs = Math.Min(fromScratchMs, sw.Elapsed.TotalMilliseconds);

            sw.Restart();
            _ = ctx.Compile(rig.TrainingStepPureGraph);
            sw.Stop();
            compileMs = Math.Min(compileMs, sw.Elapsed.TotalMilliseconds);
        }

        double stepsPerSecond = MeasureThroughput(baseGraph, exampleInput);

        return new PerfMeasurement
        {
            Scenario = "linear [4,2]->[4,1], Adam, L2Loss, LinuxCPU, net10.0",
            ConcretizeMs = concretizeMs,
            FromScratchMs = fromScratchMs,
            CompileMs = compileMs,
            TrainStepsPerSecond = stepsPerSecond,
        };
    }

    private static double MeasureThroughput(InternalComputationGraph baseGraph, TensorData exampleInput)
    {
        var ctx = new ComputeContext();
        var rig = TrainingRig.FromScratch(
            baseGraph, Losses.L2Loss, Optimizers.Adam,
            baseGraph.FromOrderedInputs([exampleInput]),
            new AdamOptimizerHyperparameters { LearningRate = 1e-3f });
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);

        var inputBatch = rig.InputDef.FromOrderedData(TensorData(InputShape, new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }));
        var targetBatch = rig.TargetDef.FromOrderedData(TensorData(TargetShape, new float[] { 1f, 0f, 1f, 0f }));

        var ckpt = rig.CreateDefaultCheckpoint();
        for (int i = 0; i < ThroughputWarmupSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled).Checkpoint;

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < ThroughputMeasuredSteps; i++)
            ckpt = rig.TrainStep(ckpt, inputBatch, targetBatch, compiled).Checkpoint;
        sw.Stop();

        return ThroughputMeasuredSteps / sw.Elapsed.TotalSeconds;
    }

    // ----- baseline I/O ------------------------------------------------------

    private static readonly string BaselineFileName = "perf-baseline.json";

    private static string BaselineOutputPath =>
        Path.Combine(AppContext.BaseDirectory, "Benchmarks", BaselineFileName);

    private static PerfMeasurement LoadBaseline()
    {
        var path = BaselineOutputPath;
        Assert.True(File.Exists(path), $"perf baseline not found at {path} — record it with SHOROKOO_UPDATE_PERF_BASELINE=1.");
        var json = File.ReadAllText(path);
        var baseline = JsonConvert.DeserializeObject<PerfMeasurement>(json);
        Assert.NotNull(baseline);
        return baseline!;
    }

    /// <summary>
    /// Rewrites the baseline in the source tree (not just the copied output) so the
    /// recorded numbers can be committed. Walks up from the test output directory to
    /// the project's <c>Benchmarks/</c> folder.
    /// </summary>
    private static void WriteBaselineToSource(PerfMeasurement measured)
    {
        var json = JsonConvert.SerializeObject(measured, Formatting.Indented);

        // Always refresh the copy next to the running assembly so an immediate re-run sees it.
        var outPath = BaselineOutputPath;
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        File.WriteAllText(outPath, json);

        // Best-effort: also update the source-tree copy if we can locate it.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "Benchmarks", BaselineFileName);
            if (File.Exists(candidate) && !string.Equals(candidate, outPath, StringComparison.Ordinal))
            {
                File.WriteAllText(candidate, json);
                break;
            }
            // Recognize the project root by its csproj and write the source baseline there.
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

    /// <summary>One recorded or measured set of timings for the pinned scenario.</summary>
    private sealed class PerfMeasurement
    {
        public string Scenario { get; set; } = "";
        public double SlowdownFactor { get; set; } = TrainingPerfBaselineTests.SlowdownFactor;
        public double ConcretizeMs { get; set; }
        public double FromScratchMs { get; set; }
        public double CompileMs { get; set; }
        public double TrainStepsPerSecond { get; set; }
    }
}
