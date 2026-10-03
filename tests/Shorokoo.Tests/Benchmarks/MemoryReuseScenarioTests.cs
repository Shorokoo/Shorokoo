using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Jax;
using Shorokoo.OnnxRuntime;
using Shorokoo.PythonHost;
using Shorokoo.PythonTranslation;
using Shorokoo.PyTorch;
using Python.Runtime;
using Shorokoo.Runtime;
using OrtElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// The memory-reuse scenario, run under every configuration that bears on it, its figures written
/// to <c>$SHOROKOO_MEMORY_REUSE_DIR</c> (default: a folder under the temp path), one Markdown report
/// and one JSON file per fact.
///
/// <para><b>The scenario.</b> Inputs A and B, each <c>[2N, M]</c> float, both consumed by the run.
/// <c>A_half = A[0:N]</c> and <c>B_half = B[N:2N]</c>, each a <c>Slice</c> whose bounds the run
/// computes from A's shape. C is a <c>[N, M]</c> tensor of 2s (<c>ConstantOfShape</c> of A_half's
/// shape) through <c>Neg</c>, <c>Abs</c>, <c>Sigmoid</c>; <c>L0 = Concat(C, B_half)</c> on axis 0,
/// then <c>Sigmoid</c>, <c>Neg</c>, <c>Abs</c> give L. Outputs L, A_half, B_half. Every dimension
/// is symbolic, so ONNX Runtime cannot fold the fill and its chain into a constant.</para>
///
/// <para><b>The ideal</b> is no memory beyond A and B: A_half stays where it is in A, B_half is
/// copied into A's second half, C is made in B's first half — so that B holds the concatenation
/// with no copy, B_half's rows being in place already — and every unary op runs in place.</para>
///
/// <para><b>Configurations</b>, on the host and on the card: ONNX Runtime's own arena or
/// Shorokoo's allocator; graph optimizations on or off; the memory pattern on or off; and five
/// approaches. <i>Plain</i>: ONNX Runtime allocates every output. <i>BindFinal</i>: A_half, B_half
/// and L bound into the consumed inputs. <i>BindAll</i>: every intermediate made a graph output and
/// bound to its final place. <i>Split</i>: the same over two sessions, with no concatenation.
/// <i>Shipped</i>: the backend's own session and <c>RunConsuming</c>. Every request the allocator serves and every block it takes back is
/// logged in order; an arena session is read through its arena's figures.</para>
///
/// <para>Run with <c>dotnet test -p:ShorokooGpuTests=true --filter
/// "FullyQualifiedName~MemoryReuseScenarioTests"</c>; the card's fact needs that build, the host's
/// and the Python one run in either.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
[Collection(SerialMeasurement.Name)]
public class MemoryReuseScenarioTests
{
    private const long N = 4096;
    private const long M = 2048;
    private const long HalfCount = N * M;
    private const long WholeCount = 2 * N * M;
    private const long HalfBytes = HalfCount * sizeof(float);
    private const long WholeBytes = WholeCount * sizeof(float);
    private const long Big = 1L << 20;
    private const int Runs = 3;
    private const int TimedRuns = 30;

    private static readonly string[] Outputs = ["L", "A_half", "B_half"];
    private static readonly string[] Intermediates = ["C0", "C1", "C2", "C3", "L0", "L1", "L2"];

    /// <summary>The precision the harness's contexts compute in: TensorFloat-32 allowed where
    /// <c>$SHOROKOO_MEMORY_REUSE_TF32</c> is <c>1</c>, full precision otherwise.</summary>
    private static PrecisionSettings Precision()
        => new() { AllowTensorFloat32 = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_TF32") is "1" };

    private static string OutputDirectory()
    {
        var dir = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_DIR")
            ?? Path.Combine(Path.GetTempPath(), "shorokoo-memory-reuse");
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void RecordTheScenarioOnTheHost() => Record(card: false, "host");

    [CudaFact]
    public void RecordTheScenarioOnTheCard() => Record(card: true, "card");

    /// <summary>
    /// The scenario through Shorokoo's public API — a compute context compiling the scenario's graph
    /// and executing it — on the backend <c>$SHOROKOO_MEMORY_REUSE_BACKEND</c> names: <c>ort</c> (the
    /// default; the host, or the card in a <c>-p:ShorokooGpuTests=true</c> build), <c>torch-cpu</c> or
    /// <c>torch-cuda</c>, each in a process of its own, since a process has one Python environment.
    /// Every run is measured twice over: fed inputs it may not consume (shared), which runs unplaced,
    /// and fed inputs it consumes, which places. What is measured beyond the inputs is
    /// what the run asked of the backend's allocator: the ONNX Runtime session's allocator account,
    /// torch's CPU allocations as its profiler records them, or torch's CUDA allocator's peak.
    /// With <c>$SHOROKOO_MEMORY_REUSE_LOG</c> set, every block of a mebibyte or more Shorokoo's
    /// allocator hands out or takes back during a measured run is logged, in order.
    /// </summary>
    [Fact]
    public void RecordTheScenarioThroughTheComputeContext()
    {
        var backend = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_BACKEND") ?? "ort";
        File.Delete(Path.Combine(OutputDirectory(), $"allocations-{backend}.txt"));
        File.Delete(Path.Combine(OutputDirectory(), "torch-profile.txt"));
        var rows = (int)(2 * N);
        var columns = (int)M;
        var a = new float[rows * columns];
        var b = new float[rows * columns];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (i % 1000) * 0.001f;
            b[i] = -((i % 777) * 0.001f);
        }
        using var context = backend switch
        {
            "torch-cpu" => new ComputeContext(new Shorokoo.PyTorch.Cpu.TorchCpuBackend()) { Precision = Precision() },
            "torch-cuda" => new ComputeContext(new Shorokoo.PyTorch.Cuda.TorchCudaBackend()) { Precision = Precision() },
            _ => new ComputeContext { Precision = Precision() },
        };
        var compiled = context.Compile(ComputeContextLifetimeCoverageTests.TwoHalves());
        var lines = new List<string> { $"# The scenario through the compute context: {backend}, A and B [{rows}, {columns}] float", "",
            "| run | inputs | beyond the inputs | allocated | outputs on blocks |", "|---|---|---|---|---|" };
        (NamedModelParam[] Outputs, double Milliseconds, long Peak, long Allocated) Run(bool consume, bool measure)
        {
            var x = TensorData([(long)rows, columns], a).To(context);
            var y = TensorData([(long)rows, columns], b).To(context);
            NamedModelParam[] Execute() => consume ? compiled.Execute(x, y) : compiled.Execute(x.Shared(), y.Shared());
            var watch = Stopwatch.StartNew();
            var log = new List<CachingAllocator.Event>();
            if (measure && Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_LOG") is not null)
                CachingAllocator.Observer = e => { lock (log) log.Add(e); };
            NamedModelParam[] outputs;
            long peak, allocated;
            try
            {
                (outputs, peak, allocated) = measure ? Measured(backend, compiled, Execute) : (Execute(), 0L, 0L);
            }
            finally
            {
                CachingAllocator.Observer = null;
            }
            if (log.Count > 0)
                File.AppendAllText(Path.Combine(OutputDirectory(), $"allocations-{backend}.txt"),
                    $"{(consume ? "consumed" : "shared")}: " + string.Join(" ", log.Where(e => e.Size >= Big).Select(e => $"{(e.Allocation ? "+" : "-")}{Mib(e.Size)}{(e.Allocation && !e.Fresh ? "(kept)" : "")}")) + Environment.NewLine);
            if (backend == "torch-cuda") Synchronize();
            watch.Stop();
            if (!consume)
            {
                x.Delete();
                y.Delete();
            }
            return (outputs, watch.Elapsed.TotalMilliseconds, peak, allocated);
        }
        for (int run = 0; run < 2 * Runs; run++)
        {
            var consume = run % 2 == 1;
            var (outputs, _, peak, allocated) = Run(consume, measure: true);
            lines.Add($"| {run / 2 + 1} | {(consume ? "consumed" : "shared")} | {Mib(peak)} | {(allocated < 0 ? "-" : Mib(allocated))} | {outputs.Count(o => o.ToTensorData().Block is not null)}/{outputs.Length} |");
            foreach (var output in outputs) output.ToTensorData().Delete();
        }
        var times = new Dictionary<bool, List<double>> { [false] = [], [true] = [] };
        for (int run = 0; run < 2 * TimedRuns; run++)
        {
            var consume = run % 2 == 1;
            var (outputs, milliseconds, _, _) = Run(consume, measure: false);
            times[consume].Add(milliseconds);
            foreach (var output in outputs) output.ToTensorData().Delete();
        }
        foreach (var (consume, each) in times)
            lines.Add($"{(consume ? "consumed" : "shared")}: median {Median(each):0.00} ms, tenth percentile {each.Order().ElementAt(each.Count / 10):0.00} ms, least {each.Min():0.00} ms over {each.Count} runs");
        if (compiled.Session is OrtSession { Placements: { } placements })
            foreach (var entry in placements.Entries)
                lines.Add($"ONNX Runtime signature: {entry.Stage}, modelled {Mib(entry.PredictedPlainPeak)} plain and {Mib(entry.PredictedPlacedPeak)} placed, placed run measured {Mib(entry.PlacedPeak)}, {entry.Plan.Count} placed{(entry.Refusal is null ? "" : ", " + entry.Refusal)}");
        if (compiled.Session is TorchSession { Placements: { } torch })
            foreach (var entry in torch.Entries)
                lines.Add($"torch signature: {entry.Plan.Count} placed{(entry.Refusal is null ? "" : ", " + entry.Refusal)}");
        File.WriteAllText(Path.Combine(OutputDirectory(), $"through-the-compute-context-{backend}.md"), string.Join("\n", lines) + "\n");
    }

    /// <summary>
    /// Placement across the memory-pass benchmark's model families (<see cref="MemoryPassBenchmarkTests.Suite"/>)
    /// on ONNX Runtime, through the public API, with placement off — every session running unplaced —
    /// and on: a resident training step (AdamW, the rig's state written over the state
    /// it consumes), and inference of the same model on an input each run consumes. A step's or a
    /// run's peak is what Shorokoo's allocator had handed out at most beyond what it had as the
    /// step began, on the card in a <c>-p:ShorokooGpuTests=true</c> build and on the host otherwise,
    /// read off its observer; time is the median of steps run with nothing observing.
    /// <c>$SHOROKOO_MEMORY_REUSE_FAMILIES</c>, a comma-separated list, picks families, and
    /// <c>$SHOROKOO_MEMORY_REUSE_SCALE</c> multiplies each family's batch, and
    /// <c>$SHOROKOO_MEMORY_REUSE_ROUNDS</c> repeats each family's pair of measurements, off then on.
    /// <c>$SHOROKOO_MEMORY_REUSE_UNPAIRED</c> times the consuming runs alone, with no shared run
    /// between them.
    /// </summary>
    [Fact]
    public void RecordPlacementAcrossTheBenchmarkFamilies()
    {
        var only = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_FAMILIES")?.Split(',');
        var backend = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_BACKEND") ?? "ort";
        File.Delete(Path.Combine(OutputDirectory(), "refusals.md"));
        var onCard = backend == "torch-cuda"
                     || (backend == "ort" && DefaultBackend.Instance.GetType().Assembly.GetName().Name?.EndsWith("GPU", StringComparison.Ordinal) == true);
        var where = backend == "ort" ? $"ONNX Runtime on the {(onCard ? "card" : "host")}" : backend;
        var lines = new List<string>
        {
            $"# Placement across the benchmark families, {where}, batch x{(long.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SCALE"), out var x) ? x : 1)}", "",
            "| family | placement | step peak | step ms | step plans | inference peak | inference ms | ms, input shared | inference plans |",
            "|---|---|---|---|---|---|---|---|---|",
        };
        var scale = long.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SCALE"), out var s) ? s : 1;
        foreach (var (family, model, benchmarkShape) in MemoryPassBenchmarkTests.Suite)
        {
            if (only is not null && !only.Contains(family)) continue;
            long[] shape = [benchmarkShape[0] * scale, .. benchmarkShape[1..]];
            var rounds = int.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_ROUNDS"), out var r) ? r : 1;
            foreach (var placing in Enumerable.Range(0, 2 * rounds).Select(k => k % 2 == 1))
            {
                var m = MeasureFamily(model(), shape, placing, onCard, backend);
                lines.Add($"| {family} | {(placing ? "on" : "off")} | {Mib(m.StepPeak)} | {m.StepMs:0.00} | {m.StepPlans} | {Mib(m.InferencePeak)} | {m.InferenceMs:0.00} | {m.SharedInferenceMs:0.00} | {m.InferencePlans} |");
            }
        }
        File.WriteAllText(Path.Combine(OutputDirectory(), $"placement-across-the-families-{(backend == "ort" ? onCard ? "card" : "host" : backend)}-x{scale}.md"), string.Join("\n", lines) + "\n");
    }

    /// <summary>
    /// What the memory-aware pass buys each benchmark family's training step on the backend
    /// <c>$SHOROKOO_MEMORY_REUSE_BACKEND</c> names, measured against what it models: the step it
    /// chose and the step it was handed (its graph swapped in for the chosen one), each run as a
    /// resident step (AdamW, L2 loss, the batch fed <c>.Shared()</c>) in turn —
    /// <c>$SHOROKOO_MEMORY_REUSE_ROUNDS</c> times — for its peak (as
    /// <see cref="RecordWhatBoundsATrainingStepsPeak"/> reads it) and its median time; beside the
    /// pass's own figures for both (its evaluator's peak and compute) and, where the backend models a
    /// run, the backend's peak; and how long the rig took to build. <c>$SHOROKOO_MEMORY_REUSE_SCALE</c> multiplies each family's batch;
    /// <c>$SHOROKOO_MEMORY_REUSE_MEMORY_WEIGHT</c> sets the pass's memory weight.
    /// </summary>
    [Fact]
    public void RecordWhatThePassBuysOnEachBackend()
    {
        TrainingRig.PassMemoryWeight.Value = double.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_MEMORY_WEIGHT"),
            NumberStyles.Float, CultureInfo.InvariantCulture, out var weight) ? weight : null;
        var weightName = TrainingRig.PassMemoryWeight.Value?.ToString(CultureInfo.InvariantCulture) ?? "default";
        var backend = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_BACKEND") ?? "ort";
        var onCard = backend == "torch-cuda"
                     || (backend == "ort" && DefaultBackend.Instance.GetType().Assembly.GetName().Name?.EndsWith("GPU", StringComparison.Ordinal) == true);
        var scale = long.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SCALE"), out var s) ? s : 1;
        var rounds = int.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_ROUNDS"), out var r) ? r : 2;
        var only = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_FAMILIES")?.Split(',');
        var where = backend == "ort" ? $"ONNX Runtime on the {(onCard ? "card" : "host")}" : backend;
        var lines = new List<string>
        {
            $"# What the memory-aware pass buys a training step, {where}, batch x{scale}, memory weight {weightName}{(Precision().AllowTensorFloat32 ? ", TensorFloat-32 allowed" : "")}", "",
            "| family | chosen | real peak, handed -> chosen | step ms, handed -> chosen | pass's model, handed -> chosen | backend's model, handed -> chosen | modelled compute, chosen / handed | rig built, s |",
            "|---|---|---|---|---|---|---|---|",
        };
        const int Warm = 3, Timed = 9;
        // Beside the benchmark's families, its encoders and dense attention with each layer, or the
        // whole attention, a [Module(Checkpoint = true)] segment: what recomputation buys.
        (string, Func<ComputationGraph>, long[])[] checkpointed =
        [
            ("encoder1-ckpt", () => Shorokoo.Tests.Modules.CheckpointedEncoder1.ComputationGraph, [8L, 128L, 128L]),
            ("encoder2-ckpt", () => Shorokoo.Tests.Modules.CheckpointedEncoder2.ComputationGraph, [8L, 128L, 128L]),
            ("attn-dense-ckpt", () => Shorokoo.Tests.Modules.CheckpointedMeanPooledAttention.ComputationGraph, [2L, 4L, 256L, 32L]),
        ];
        foreach (var (family, model, benchmarkShape) in MemoryPassBenchmarkTests.Suite.Concat(checkpointed))
        {
            if (only is not null && !only.Contains(family)) continue;
            long[] shape = [benchmarkShape[0] * scale, .. benchmarkShape[1..]];
            var count = (int)shape.Aggregate(1L, (a, d) => a * d);
            float[] Values(int seed) => [.. Enumerable.Range(0, count).Select(i => ((i * 7 + seed) % 101) / 101f - 0.5f)];
            using var context = backend switch
            {
                "torch-cpu" => new ComputeContext(new Shorokoo.PyTorch.Cpu.TorchCpuBackend()) { Precision = Precision() },
                "torch-cuda" => new ComputeContext(new Shorokoo.PyTorch.Cuda.TorchCudaBackend()) { Precision = Precision() },
                _ => new ComputeContext { Precision = Precision() },
            };
            var sample = TensorData(shape, Values(0));
            var concrete = model().ToConcreteArchitecture([sample]).ToConcreteModel();
            var predicted = context.Execute(concrete, sample.Shared())[0].ToTensorData();
            long[] dims = [.. predicted.Shape.Dims.Select(d => (long)d)];
            predicted.Delete();
            var building = Stopwatch.StartNew();
            var rig = TrainingRig.FromScratch(
                model(), Shorokoo.Modules.Losses.L2Loss.ComputationGraph, Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph,
                [sample.CopyTo(ComputeContext.Host)], new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f },
                runtimeContext: context);
            var built = building.Elapsed.TotalSeconds;
            var result = rig.OptimizationResult;
            var chosenAt = result.AllStrategies.Select((x, i) => (x, i)).First(p => ReferenceEquals(p.x.Graph, result.OptimizedGraph)).i;
            var chosen = rig.TrainingStepPureGraph;
            var handed = rig.PreOptimizationGraph;
            var input = rig.InputDef.FromOrderedData(sample.CopyTo(context));
            var targets = rig.TargetDef.FromOrderedData(TensorData(dims, new float[dims.Aggregate(1L, (a, d) => a * d)]).CopyTo(context));
            var steps = typeof(TrainingRig).GetField("_compiledTrainSteps", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            var graphOf = typeof(TrainingRig).GetProperty(nameof(TrainingRig.TrainingStepPureGraph))!;
            var peaks = new Dictionary<bool, List<long>> { [false] = [], [true] = [] };
            var times = new Dictionary<bool, List<double>> { [false] = [], [true] = [] };
            foreach (var asChosen in Enumerable.Range(0, 2 * rounds).Select(k => k % 2 == 1))
            {
                graphOf.SetValue(rig, asChosen ? chosen : handed);
                var compiled = (System.Collections.IDictionary)steps.GetValue(rig)!;
                lock (compiled) compiled.Clear();
                using var run = rig.BeginResidentRun(rig.CreateInitialCheckpoint());
                for (int i = 0; i < Warm; i++) run.Step(input.Shared(), targets.Shared());
                peaks[asChosen].Add(backend switch
                {
                    "ort" => Observed(onCard, () => { run.Step(input.Shared(), targets.Shared()); return 0; }).Peak,
                    _ => TorchObserved(backend, () => { run.Step(input.Shared(), targets.Shared()); return 0; }).Peak,
                });
                var t = new List<double>();
                for (int i = 0; i < Timed; i++)
                {
                    var watch = Stopwatch.StartNew();
                    run.Step(input.Shared(), targets.Shared());
                    t.Add(watch.Elapsed.TotalMilliseconds);
                }
                t.Sort();
                times[asChosen].Add(t[t.Count / 2]);
            }
            graphOf.SetValue(rig, chosen);
            string Pair(Func<bool, string> of) => $"{of(false)} -> {of(true)}";
            static string Ms(double ms) => ms.ToString("0.0", CultureInfo.InvariantCulture);
            var backendPeaks = result.BackendPeakBytes;
            lines.Add($"| {family} | {result.StrategyName} | {Pair(c => string.Join("/", peaks[c].Select(Mib)))} | {Pair(c => string.Join("/", times[c].Select(Ms)))} "
                      + $"| {Mib(result.AllStrategies[0].Evaluation.PeakMemoryBytes)} -> {Mib(result.Evaluation.PeakMemoryBytes)} "
                      + $"| {(backendPeaks is null ? "-" : $"{Mib(backendPeaks[0])} -> {Mib(backendPeaks[chosenAt])}")} "
                      + $"| {result.Evaluation.TotalComputeTime / result.AllStrategies[0].Evaluation.TotalComputeTime:0.000} | {built:0.0} |");
            File.WriteAllText(Path.Combine(OutputDirectory(), $"what-the-pass-buys-{(backend == "ort" ? onCard ? "card" : "host" : backend)}-x{scale}-w{weightName}.md"), string.Join("\n", lines) + "\n");
        }
    }

    /// <summary>
    /// What bounds each benchmark family's training step's peak: the step (AdamW, L2 loss, state
    /// written over itself, batch fed <c>.Shared()</c>) measured on the backend
    /// <c>$SHOROKOO_MEMORY_REUSE_BACKEND</c> names — the most its allocator handed out beyond what it
    /// had as the step began, read off Shorokoo's allocator's observer on ONNX Runtime and off torch's
    /// CUDA allocator on a card (torch keeps no such figure on the CPU) — beside the step's
    /// <see cref="StepAnatomy"/>: the modelled peak in the order the backend runs the step (ONNX
    /// Runtime's depth-first order, or the graph's own for a translation), and what is held there,
    /// by part. <c>$SHOROKOO_MEMORY_REUSE_SCALE</c> multiplies each family's batch.
    /// </summary>
    [Fact]
    public void RecordWhatBoundsATrainingStepsPeak()
    {
        var backend = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_BACKEND") ?? "ort";
        var onCard = backend == "torch-cuda"
                     || (backend == "ort" && DefaultBackend.Instance.GetType().Assembly.GetName().Name?.EndsWith("GPU", StringComparison.Ordinal) == true);
        var scale = long.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SCALE"), out var s) ? s : 1;
        var only = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_FAMILIES")?.Split(',');
        var where = backend == "ort" ? $"ONNX Runtime on the {(onCard ? "card" : "host")}" : backend;
        var lines = new List<string>
        {
            $"# What bounds a training step's peak, {where}, batch x{scale}", "",
            "| family | measured | modelled, graph handed over | modelled, graph run | forward | backward | parameter gradients | update | every element-wise op over a dying operand | batch freed after its last read | a shape read holds nothing | all three | modelled before the memory-aware pass (its strategy) | modelled in the graph's own order, after the pass and before | batch | parameters | state | peak at |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        };
        // Beside the benchmark's families, one whose parameters outweigh its activations: a 4096-wide
        // linear layer, 64 MiB of weights, where the optimizer's state and arithmetic are the step.
        (string, Func<ComputationGraph>, long[])[] wide = [("wide", () => WideLinearModel.ComputationGraph, [2L, 4096L])];
        foreach (var (family, model, benchmarkShape) in MemoryPassBenchmarkTests.Suite.Concat(wide))
        {
            if (only is not null && !only.Contains(family)) continue;
            long[] shape = [benchmarkShape[0] * scale, .. benchmarkShape[1..]];
            var count = (int)shape.Aggregate(1L, (a, d) => a * d);
            float[] Values(int seed) => [.. Enumerable.Range(0, count).Select(i => ((i * 7 + seed) % 101) / 101f - 0.5f)];
            using var context = backend switch
            {
                "torch-cpu" => new ComputeContext(new Shorokoo.PyTorch.Cpu.TorchCpuBackend()) { Precision = Precision() },
                "torch-cuda" => new ComputeContext(new Shorokoo.PyTorch.Cuda.TorchCudaBackend()) { Precision = Precision() },
                _ => new ComputeContext { Precision = Precision() },
            };
            var sample = TensorData(shape, Values(0));
            var concrete = model().ToConcreteArchitecture([sample]).ToConcreteModel();
            var predicted = context.Execute(concrete, sample.Shared())[0].ToTensorData();
            long[] dims = [.. predicted.Shape.Dims.Select(d => (long)d)];
            predicted.Delete();
            var rig = TrainingRig.FromScratch(
                model(), Shorokoo.Modules.Losses.L2Loss.ComputationGraph, Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph,
                [sample.CopyTo(ComputeContext.Host)], new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f },
                runtimeContext: context);
            var input = rig.InputDef.FromOrderedData(sample.CopyTo(context));
            var target = TensorData(dims, new float[dims.Aggregate(1L, (a, d) => a * d)]);
            var targets = rig.TargetDef.FromOrderedData(target.CopyTo(context));
            long measured;
            var settled = new List<string>();
            OrtPlacements.Settled = e => { lock (settled) settled.Add(e.Stage == OrtPlacements.Stage.Adopted ? $"ONNX Runtime placed {e.Plan.Count} ({Mib(e.Plan.Sum(p => p.Bytes))})" : $"ONNX Runtime refused: {e.Refusal} (blocks {string.Join(",", e.BlockBytes.Select(b => $"{b.Key}={Mib(b.Value)}"))})"); };
            TorchPlacements.Settled = e => { lock (settled) settled.Add(e.Plan.Count > 0 ? $"torch placed {e.Plan.Count} ({Mib(e.Plan.Sum(p => p.Bytes))})" : $"torch refused: {e.Refusal}"); };
            using (var run = rig.BeginResidentRun(rig.CreateInitialCheckpoint()))
            {
                // A batch handed over as it is, as a loader hands one, is the step's to consume; one
                // passed .Shared() is only read.
                var consumed = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_BATCH") == "consumed";
                (IData Input, IData Target) Batch() => consumed
                    ? (rig.InputDef.FromOrderedData(sample.CopyTo(context)), rig.TargetDef.FromOrderedData(target.CopyTo(context)))
                    : (input.Shared(), targets.Shared());
                for (int i = 0; i < 3; i++)
                {
                    var (x, t) = Batch();
                    run.Step(x, t);
                }
                var (bx, bt) = Batch();
                measured = backend switch
                {
                    "ort" => Observed(onCard, () => { run.Step(bx, bt); return 0; }).Peak,
                    _ => TorchObserved(backend, () => { run.Step(bx, bt); return 0; }).Peak,
                };
            }
            OrtPlacements.Settled = null;
            TorchPlacements.Settled = null;
            var stepModel = MemoryPassBenchmarkTests.RigModel(rig.TrainingStepPureGraph, rig.OptimizationInputShapes);
            if (backend.StartsWith("torch", StringComparison.Ordinal) && Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_LOG") is not null)
                File.WriteAllText(Path.Combine(OutputDirectory(), $"step-{family}.py"), OnnxToPythonTranslator.Translate(stepModel, [], Shorokoo.PyTorch.TorchDialect.Instance, null, TorchInPlace.Plan(stepModel.Graph)).Source);
            var state = rig.UpdatedParamFieldCount + rig.UpdatedStateFieldCount + rig.UpdatedOptimizerStateFieldCount;
            StepAnatomy Anatomy(GraphProto graph) => new(graph, rig.UpdatedParamFieldCount, state, rig.UpdatedOptimizerStateFieldCount);
            var handed = Anatomy(stepModel.Graph);
            // ONNX Runtime runs the graph it rewrites, in its own order, writing an activation over
            // its input where the input dies there; a translation runs the graph as handed over, in
            // its own order.
            var ortRun = backend == "ort" ? OrtRun(stepModel, onCard, rig.OptimizationInputShapes) : default;
            var anatomy = backend == "ort" ? Anatomy(ortRun.Graph) : handed;
            int[] OrderOf(StepAnatomy a, GraphProto g) => backend == "ort" ? a.OrtOrder() : TranslationOrder(g);
            var order = backend == "ort" ? ortRun.Order : TranslationOrder(stepModel.Graph);
            Func<NodeProto, int, int, bool> shares = backend == "ort" ? OutputAliasProof.Shares : PlacementMemory.PyTorch.SharesUnlessPlaced;
            Func<NodeProto, int, bool>? own = backend == "ort" ? static (node, slot) => slot == 0 && OrtWritesOver.Contains(node.OpType) : null;
            var peak = anatomy.Peak(order, shares, own);
            var handedPeak = handed.Peak(OrderOf(handed, stepModel.Graph), shares, own);
            var inPlace = anatomy.Peak(order, shares, (node, slot) => (own?.Invoke(node, slot) ?? false) || (OutputAliasProof.IsStandard(node)
                && ((PlacementProof.InPlaceUnary.Contains(node.OpType) && slot == 0) || (PlacementProof.InPlaceBinary.Contains(node.OpType) && node.Inputs.Count == 2))));
            var batchFreed = anatomy.Peak(order, shares, own, batchFreed: true);
            var shapesFree = anatomy.Peak(order, shares, own, shapeReadsHold: false);
            // The step before the memory-aware pass rematerialized or reordered anything, as the
            // backend runs it.
            var preModel = MemoryPassBenchmarkTests.RigModel(rig.PreOptimizationGraph, rig.OptimizationInputShapes);
            var preRun = backend == "ort" ? OrtRun(preModel, onCard, rig.OptimizationInputShapes) : default;
            var preAnatomy = backend == "ort" ? Anatomy(preRun.Graph) : Anatomy(preModel.Graph);
            var pre = preAnatomy.Peak(backend == "ort" ? preRun.Order : TranslationOrder(preModel.Graph), shares, own);
            // A translation run in the graph's own order rather than the one the memory-aware pass
            // schedules for, ONNX Runtime's.
            var listed = backend == "ort" ? peak : anatomy.Peak(GraphOrder(stepModel.Graph), shares, own);
            var listedPre = backend == "ort" ? pre : preAnatomy.Peak(GraphOrder(preModel.Graph), shares, own);
            var everything = anatomy.Peak(order, shares, (node, slot) => (own?.Invoke(node, slot) ?? false) || (OutputAliasProof.IsStandard(node)
                && ((PlacementProof.InPlaceUnary.Contains(node.OpType) && slot == 0) || (PlacementProof.InPlaceBinary.Contains(node.OpType) && node.Inputs.Count == 2))),
                batchFreed: true, shapeReadsHold: false);
            lines.Add($"| {family} | {(measured < 0 ? "-" : Mib(measured))} | {Mib(handedPeak.Bytes)} | {Mib(peak.Bytes)} | {Mib(peak.ByPart[StepAnatomy.Part.Forward])} | {Mib(peak.ByPart[StepAnatomy.Part.Backward])} "
                      + $"| {Mib(peak.ByPart[StepAnatomy.Part.ParameterGradient])} | {Mib(peak.ByPart[StepAnatomy.Part.Update])} | {Mib(inPlace.Bytes)} | {Mib(batchFreed.Bytes)} "
                      + $"| {Mib(shapesFree.Bytes)} | {Mib(everything.Bytes)} | {Mib(pre.Bytes)} ({rig.OptimizationResult.StrategyName}) | {Mib(listed.Bytes)}, {Mib(listedPre.Bytes)} "
                      + $"| {Mib(anatomy.BatchBytes)} | {Mib(anatomy.ParameterBytes)} | {Mib(anatomy.StateBytes)} | {peak.Position}/{order.Length}: {anatomy.At(order, peak.Position)} |");
            File.AppendAllText(Path.Combine(OutputDirectory(), $"step-anatomy-{backend}-{(onCard ? "card" : "host")}-x{scale}-detail.md"),
                $"\n## {family}\n\nplacements: {string.Join(", ", settled.GroupBy(x => x).Select(g => g.Count() == 1 ? g.Key : $"{g.Count()}x {g.Key}"))}\n\n"
                + $"nodes by part: {string.Join(", ", anatomy.NodesByPart.Select(p => $"{p.Key} {p.Value}"))}; values of unknown shape: {anatomy.UnknownValues}\n\nunknown from: {anatomy.UnknownRoots}\n\n"
                + "largest held at the peak: " + string.Join(", ", peak.Largest.Select(l => $"{l.Op} {l.Part} {Mib(l.Bytes)} (read until {l.Until})")) + "\n"
                + $"\nplanned in the batch, as a translation lays values out: {BatchPlan(stepModel.Graph, state, PlacementMemory.PyTorch)}; as ONNX Runtime does: {BatchPlan(stepModel.Graph, state, PlacementMemory.OnnxRuntime)}"
                + (SessionModel(rig) is { } handedOver ? $"; in the graph the step's session was handed, as ONNX Runtime lays values out: {BatchPlan(handedOver.Graph, state, PlacementMemory.OnnxRuntime, rig.OptimizationInputShapes)} ({handedOver.Graph.Nodes.Count} nodes against {stepModel.Graph.Nodes.Count}; "
                    + $"unknown shapes {UnknownIn(handedOver.Graph, rig.OptimizationInputShapes)}; " + $"the batch read by {BatchReaders(handedOver.Graph, state)} there, by {BatchReaders(stepModel.Graph, state)} in the other)" : "") + "\n"
                + (backend == "ort" ? $"\n{ortRun.Held}\n" : "")
                + "\nbefore the memory-aware pass, largest held at the peak: " + string.Join(", ", pre.Largest.Select(l => $"{l.Op} {l.Part} {Mib(l.Bytes)} (read until {l.Until})")) + "\n");
        }
        File.WriteAllText(Path.Combine(OutputDirectory(), $"step-anatomy-{backend}-{(onCard ? "card" : "host")}-x{scale}.md"), string.Join("\n", lines) + "\n");
    }

    /// <summary>The model the rig's shape-specialized training-step session on ONNX Runtime was
    /// handed, where it keeps one.</summary>
    private static ModelProto? SessionModel(TrainingRig rig)
    {
        var steps = (System.Collections.IDictionary?)typeof(TrainingRig)
            .GetField("_compiledTrainSteps", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.GetValue(rig);
        return steps?.Values.Cast<CompiledGraph>().Select(c => c.Session).OfType<OrtSession>().FirstOrDefault()?.Placements?.OriginalModel;
    }

    /// <summary>How many values of <paramref name="graph"/> fed <paramref name="fed"/> have no shape the
    /// placement evaluates, and the first operators where evaluation stops.</summary>
    private static string UnknownIn(GraphProto graph, (Shape Shape, DType DType)[] fed)
    {
        var given = graph.Inputs.Select((i, k) => (i.Name, Dims: k < fed.Length ? fed[k].Shape.Dims.Select(d => (long)d).ToArray() : [], i.Type.TensorType.ElemType))
            .ToDictionary(i => i.Name, i => (i.Dims, i.ElemType), StringComparer.Ordinal);
        var shapes = PlacementShapes.Evaluate(graph, given);
        var unknown = graph.Nodes.Where(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o))).ToList();
        return $"{unknown.Count} nodes, from " + string.Join(", ", unknown.Where(n => n.Inputs.All(i => i.Length == 0 || shapes.ContainsKey(i))).Select(n => $"{n.Domain}:{n.OpType}").Distinct().Take(6));
    }

    /// <summary>The operators reading a training step's batch — the inputs after its
    /// <paramref name="state"/> — or a view of it, each at its place in the graph's order.</summary>
    private static string BatchReaders(GraphProto graph, int state)
    {
        var memory = graph.Inputs.Skip(state).Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        var readers = new List<string>();
        for (int n = 0; n < graph.Nodes.Count; n++)
        {
            var node = graph.Nodes[n];
            if (!node.Inputs.Any(memory.Contains) || OutputAliasProof.ReadsOnlyAShape(node)) continue;
            readers.Add($"{node.OpType}@{n}");
            for (int o = 0; o < node.Outputs.Count; o++)
                if (Enumerable.Range(0, node.Inputs.Count).Any(i => memory.Contains(node.Inputs[i]) && OutputAliasProof.Shares(node, i, o))) memory.Add(node.Outputs[o]);
        }
        return $"{string.Join(" ", readers)} of {graph.Nodes.Count}";
    }

    /// <summary>What the planner places in a training step's batch — the inputs after its
    /// <paramref name="state"/> — were the step to consume it, under <paramref name="memory"/>: each
    /// value's operator and size.</summary>
    private static string BatchPlan(GraphProto graph, int state, PlacementMemory memory, (Shape Shape, DType DType)[]? fed = null)
    {
        var given = graph.Inputs.Select((i, k) => (i.Name, Dims: fed is null || k >= fed.Length
                ? i.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray() : fed[k].Shape.Dims.Select(d => (long)d).ToArray(), i.Type.TensorType.ElemType))
            .ToDictionary(i => i.Name, i => (i.Dims, i.ElemType), StringComparer.Ordinal);
        var shapes = PlacementShapes.Evaluate(graph, given);
        var blocks = graph.Inputs.Skip(state).Where(i => shapes.TryGetValue(i.Name, out var v) && v.Bytes > 0)
            .ToDictionary(i => i.Name, i => shapes[i.Name].Bytes, StringComparer.Ordinal);
        var plan = new PlacementProof(graph, blocks, shapes, graph.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal), memory)
            .Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes);
        return plan.Count == 0 ? "nothing" : string.Join(", ", plan.Select(p => $"{graph.Nodes.First(n => n.Outputs.Contains(p.Value)).OpType} {Mib(p.Bytes)}"));
    }

    /// <summary>The order a translation runs <paramref name="graph"/>'s nodes in, by index.</summary>
    private static int[] TranslationOrder(GraphProto graph)
    {
        var index = new Dictionary<NodeProto, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < graph.Nodes.Count; i++) index[graph.Nodes[i]] = i;
        return [.. Shorokoo.PythonTranslation.OnnxToPythonTranslator.RunOrder(graph.Nodes).Select(node => index[node])];
    }

    /// <summary>The order a translation would run <paramref name="graph"/>'s nodes in were it to
    /// take them in the graph's own order, shapes read first, by index.</summary>
    private static int[] GraphOrder(GraphProto graph)
    {
        var index = new Dictionary<NodeProto, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < graph.Nodes.Count; i++) index[graph.Nodes[i]] = i;
        return [.. Shorokoo.PythonTranslation.OnnxToPythonTranslator.ShapesFirst(graph.Nodes).Select(node => index[node])];
    }

    /// <summary>The activations ONNX Runtime's kernels write over their input where it dies there
    /// (they register to run in place).</summary>
    private static readonly HashSet<string> OrtWritesOver = new(StringComparer.Ordinal)
    {
        "Relu", "Sigmoid", "Tanh", "Elu", "LeakyRelu", "HardSigmoid", "Selu", "Softplus", "Softsign", "ThresholdedRelu",
    };

    /// <summary>
    /// Where ONNX Runtime's run of each benchmark family's training step holds other than the
    /// memory-aware pass's model of it, kernel by kernel: the step the pass chose for a rig on the
    /// default context (host or card, by the build), at <c>$SHOROKOO_MEMORY_REUSE_SCALE</c> times the
    /// batch (16 by default), built with the pass's own value names and run once as a training
    /// step's session runs it, its allocations through Shorokoo's allocator each put to the kernel
    /// running as it was made. Per kernel, beside the model's figure at the node the kernel's output
    /// comes from: what the run held as the kernel began, the most it held during it, what it held
    /// after, and the blocks it took. The inputs are left out of both sides, and the run's outputs
    /// that the step writes into its inputs out of the run's.
    /// </summary>
    [Fact]
    public void RecordWhereOnnxRuntimeHoldsOtherThanThePassModels()
    {
        var onCard = DefaultBackend.Instance.GetType().Assembly.GetName().Name?.EndsWith("GPU", StringComparison.Ordinal) == true;
        var scale = long.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SCALE"), out var sc) ? sc : 16;
        var only = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_FAMILIES")?.Split(',');
        var where = onCard ? "card" : "host";
        var summary = new List<string>
        {
            $"# Where ONNX Runtime holds other than the pass models, on the {where}, batch x{scale}", "",
            "| family | real peak (resident run) | run peak (one run, outputs into inputs left out) | its graph at that kernel, freed at last read | its graph planned as the pass plans, in the run's order | the same, in the order the pass predicts | the pass's modelled peak | the pass's model at that kernel's node | the backend's model of the chosen step | scratch over 1 MiB, by operator |",
            "|---|---|---|---|---|---|---|---|---|---|",
        };
        var detail = new StringBuilder();
        foreach (var (family, model, benchmarkShape) in MemoryPassBenchmarkTests.Suite)
        {
            if (only is not null && !only.Contains(family)) continue;
            long[] shape = [benchmarkShape[0] * scale, .. benchmarkShape[1..]];
            var count = (int)shape.Aggregate(1L, (a, d) => a * d);
            using var context = new ComputeContext();
            var sample = TensorData(shape, [.. Enumerable.Range(0, count).Select(i => ((i * 7) % 101) / 101f - 0.5f)]);
            var rig = TrainingRig.FromScratch(
                model(), Shorokoo.Modules.Losses.L2Loss.ComputationGraph, Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph,
                [sample.CopyTo(ComputeContext.Host)], new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f },
                runtimeContext: context);

            // The real peak of a resident step, as the pass's figures are compared with elsewhere.
            var shapes = rig.OptimizationInputShapes;
            var concrete = model().ToConcreteArchitecture([sample]).ToConcreteModel();
            var predicted = context.Execute(concrete, sample.Shared())[0].ToTensorData();
            long[] outDims = [.. predicted.Shape.Dims.Select(d => (long)d)];
            predicted.Delete();
            var input = rig.InputDef.FromOrderedData(sample.CopyTo(context));
            var targets = rig.TargetDef.FromOrderedData(TensorData(outDims, new float[outDims.Aggregate(1L, (a, d) => a * d)]).CopyTo(context));
            long real;
            using (var run = rig.BeginResidentRun(rig.CreateInitialCheckpoint()))
            {
                for (int i = 0; i < 3; i++) run.Step(input.Shared(), targets.Shared());
                real = Observed(onCard, () => { run.Step(input.Shared(), targets.Shared()); return 0; }).Peak;
            }

            // The pass's model, node by node, in the order it walks the step it chose; each value by
            // the name a session's model gives it, which the preparation of that model renumbers.
            var result = rig.OptimizationResult;
            var workarounds = Shorokoo.Core.Lowering.KernelWorkarounds.KernelWorkaroundRegistry.For(context.ResolvedBackend.KernelWorkaroundSet);
            var graph = result.OptimizedGraph;
            var prepared = graph.Clone();
            var keyOf = new Dictionary<object, Shorokoo.Core.Graph.FastNodeKey>(ReferenceEqualityComparer.Instance);
            foreach (var node in prepared.Nodes) keyOf[node] = node.Key;
            Shorokoo.Core.Nodes.Processors.Fast.FastIdentityWrapping.WrapAliasedOutputs(prepared);
            typeof(Shorokoo.Core.Factory.FastOnnxModelBuilder).GetMethod("RunPrePasses", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!
                .Invoke(null, [prepared, true, true, workarounds, true, (Func<InternalComputationGraph, bool>)(_ => true), false]);
            var renamed = prepared.Nodes.Where(keyOf.ContainsKey).ToDictionary(n => keyOf[n], n => n.Key);
            var details = result.Evaluation.NodeDetails;
            long inputBytes = 0;
            var positionOf = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int p = 0; p < details.Count; p++)
                foreach (var output in graph.Nodes[details[p].NodeIndex].Outputs)
                    if (output is { } key && renamed.TryGetValue(key.FastNodeKey, out var newKey))
                        positionOf[new Shorokoo.Core.Graph.FastTensorKey(newKey, key.OutputIndex).ToString()] = p;
            long ModelAt(int p) => p < 0 ? 0 : details[p].CurrentMemoryBytes + details[p].ExtraMemoryBytes - inputBytes;
            var modelPeakAt = Enumerable.Range(0, details.Count).MaxBy(ModelAt);

            // The run, kernel by kernel.
            var stepModel = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true,
                inputDims: [.. shapes.Select(s => (long[]?)s.Shape.Dims.Select(d => (long)d).ToArray())], workarounds: workarounds);
            var timeline = OrtTimeline(stepModel, onCard, shapes);
            var ran = timeline.Graph;
            var given = new Dictionary<string, (long[], int)>(StringComparer.Ordinal);
            for (int i = 0; i < ran.Inputs.Count && i < shapes.Length; i++)
                given[ran.Inputs[i].Name] = ([.. shapes[i].Shape.Dims.Select(d => (long)d)], ran.Inputs[i].Type?.TensorType?.ElemType ?? 1);
            var valueShapes = PlacementShapes.Evaluate(ran, given);
            long BytesOf(string v) => valueShapes.TryGetValue(v, out var x) ? Math.Max(x.Bytes, 0) : 0;
            inputBytes = ran.Inputs.Sum(i => BytesOf(i.Name));
            var stateOutputs = ran.Outputs.Take(rig.UpdatedParamFieldCount + rig.UpdatedStateFieldCount + rig.UpdatedOptimizerStateFieldCount)
                .Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
            detail.Append($"\n## {family} {string.Join("x", shape)}: {ran.Nodes.Count} nodes run ({graph.Nodes.Count} modelled); inputs {Mib(inputBytes)}; "
                          + $"real {Mib(real)}; modelled peak {Mib(ModelAt(modelPeakAt))} at {details[modelPeakAt].OpCode} #{modelPeakAt}\n\n");
            detail.Append("operators run, against modelled: "
                          + string.Join(", ", ran.Nodes.GroupBy(n => n.OpType).Select(g => (g.Key, Run: g.Count(), Model: details.Count(d => d.OpCode == g.Key)))
                              .Where(c => c.Run != c.Model).OrderBy(c => c.Key).Select(c => $"{c.Key} {c.Run}/{c.Model}"))
                          + "; modelled, not run: " + string.Join(", ", details.GroupBy(d => d.OpCode).Where(g => ran.Nodes.All(n => n.OpType != g.Key)).Select(g => $"{g.Key} {g.Count()}")) + "\n\n");
            detail.Append("| # | kernel | outputs | run before / most / after | its graph, freed at last read | its graph, as the pass plans | run - planned | pass's model at its node | blocks taken |\n|---|---|---|---|---|---|---|---|---|\n");

            // The graph the run ran, its values freed after their last reader in the order its
            // kernels ran: a view (ONNX Runtime's reshapes) as its input's memory, an activation
            // over its input where that dies there, the inputs and the outputs the step writes
            // into its inputs taking nothing, every other output held to the end.
            var order = timeline.Kernels.Select(k => k.Node).ToList();
            var kernelAt = new Dictionary<int, int>();
            for (int k = 0; k < order.Count; k++) kernelAt[order[k]] = k;
            var root = new Dictionary<string, string>(StringComparer.Ordinal);
            string RootOf(string v) => root.TryGetValue(v, out var r) && r != v ? root[v] = RootOf(r) : v;
            foreach (var n in order)
            {
                var node = ran.Nodes[n];
                for (int o = 0; o < node.Outputs.Count; o++)
                    for (int i = 0; i < node.Inputs.Count; i++)
                        if (node.Outputs[o].Length > 0 && node.Inputs[i].Length > 0 && OutputAliasProof.Shares(node, i, o))
                        {
                            root[node.Outputs[o]] = RootOf(node.Inputs[i]);
                            break;
                        }
            }
            var lastRead = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var output in ran.Outputs) lastRead[RootOf(output.Name)] = int.MaxValue;
            for (int k = 0; k < order.Count; k++)
                foreach (var v in ran.Nodes[order[k]].Inputs.Where(v => v.Length > 0))
                    lastRead[RootOf(v)] = Math.Max(lastRead.GetValueOrDefault(RootOf(v), -1), k);
            var notCounted = ran.Inputs.Select(i => i.Name).Concat(ran.Initializers.Select(i => i.Name)).Concat(stateOutputs).ToHashSet(StringComparer.Ordinal);
            var alive = new Dictionary<string, long>(StringComparer.Ordinal);
            long liveBytes = 0;
            var planTrace = new Dictionary<int, string>();
            var planned = PlannedOccupancy(ran, order, valueShapes, notCounted, planTrace);
            var plannedInItsOrder = PlannedOccupancy(ran, new StepAnatomy(ran, 0, 0, 0).OrtOrder(), valueShapes, notCounted);

            long stateSoFar = 0, previousDelta = 0, runPeak = 0, modelAtRunPeak = 0, graphAtRunPeak = 0;
            int position = -1, runPeakAt = 0;
            var heldAtRunPeak = new List<string>();
            var scratch = new Dictionary<string, (int Count, long Bytes)>(StringComparer.Ordinal);
            for (int k = 0; k < timeline.Kernels.Count; k++)
            {
                var kernel = timeline.Kernels[k];
                var node = ran.Nodes[kernel.Node];
                var outputs = node.Outputs.Where(o => o.Length > 0).ToList();
                foreach (var o in outputs)
                {
                    if (positionOf.TryGetValue(o, out var p)) position = p;
                    if (RootOf(o) != o || notCounted.Contains(o)) continue;
                    var bytes = BytesOf(o);
                    var over = slotOver(node);
                    if (over is { } slot && slot < node.Inputs.Count && node.Inputs[slot] is { Length: > 0 } operand
                        && alive.TryGetValue(RootOf(operand), out var operandBytes) && operandBytes == bytes && lastRead.GetValueOrDefault(RootOf(operand), -1) == k)
                    {
                        alive.Remove(RootOf(operand));
                        root[RootOf(operand)] = o;
                        alive[o] = bytes;
                        continue;
                    }
                    alive[o] = bytes;
                    liveBytes += bytes;
                }
                var liveDuring = liveBytes;
                foreach (var v in node.Inputs.Concat(node.Outputs).Where(v => v.Length > 0).Distinct())
                {
                    var r = RootOf(v);
                    if (alive.TryGetValue(r, out var size) && lastRead.GetValueOrDefault(r, k) <= k)
                    {
                        alive.Remove(r);
                        liveBytes -= size;
                    }
                }
                var outputBytes = outputs.Sum(BytesOf);
                var most = kernel.Most - stateSoFar;
                var before = kernel.Before - stateSoFar;
                stateSoFar += outputs.Where(stateOutputs.Contains).Sum(BytesOf);
                var after = kernel.After - stateSoFar;
                var transient = most - Math.Max(before, after);
                var op = $"{node.Domain}:{node.OpType}".TrimStart(':');
                if (transient >= Big)
                {
                    var was = scratch.GetValueOrDefault(op);
                    scratch[op] = (was.Count + 1, Math.Max(was.Bytes, transient));
                }
                var modelled = ModelAt(position);
                if (most > runPeak)
                {
                    runPeakAt = k;
                    (runPeak, modelAtRunPeak, graphAtRunPeak) = (most, modelled, liveDuring);
                    heldAtRunPeak = [.. alive.Where(a => a.Value >= Big).OrderByDescending(a => a.Value)
                        .Select(a => $"{Mib(a.Value)} {ProducerOf(a.Key)}")];
                }
                var delta = most - planned[k];
                if (Math.Abs(delta - previousDelta) >= Big / 2 || transient >= Big)
                    detail.Append($"| {k} | {op} | {Mib(outputBytes)} | {Mib(before)} / {Mib(most)} / {Mib(after)} | {Mib(liveDuring)} | {Mib(planned[k])} | {Mib(delta)} | {Mib(modelled)} "
                                  + $"| {string.Join(" ", kernel.Taken.Where(b => b >= Big / 4).Select(Mib))} | {string.Join(" ", node.Inputs.Where(v => v.Length > 0).Select(v => $"{v}:{Mib(BytesOf(v))}"))} -> {string.Join(" ", outputs)} | {planTrace.GetValueOrDefault(k, "")} |\n");
                previousDelta = delta;
            }
            string ProducerOf(string v) => ran.Nodes.FirstOrDefault(n => n.Outputs.Contains(v)) is { } p ? $"{p.OpType}#{kernelAt.GetValueOrDefault(ran.Nodes.IndexOf(p), -1)}" : "input";
            int? slotOver(NodeProto n) => OrtWritesOver.Contains(n.OpType) ? 0 : null;
            detail.Append($"\nsession built in {SessionBuildMs:0} ms; before the first kernel: {Mib(timeline.BeforeFirst)} taken\n\nheld at the run's peak ({Mib(runPeak)}; its graph {Mib(graphAtRunPeak)}): {string.Join(", ", heldAtRunPeak)}\n");
            detail.Append($"\nblocks of a mebibyte or more the run held at its peak, kernel {runPeakAt}: "
                + string.Join(", ", timeline.Blocks.Where(b => b.Size >= Big && b.Taken <= runPeakAt && (b.Freed < 0 || b.Freed > runPeakAt))
                    .Select(b => $"{Mib(b.Size)} taken in {OpAt(b.Taken)}, freed in {(b.Freed < 0 ? "-" : OpAt(b.Freed))}")) + "\n");
            string OpAt(int k) => k < 0 ? "-" : $"{ran.Nodes[order[k]].OpType}#{k}";
            summary.Add($"| {family} | {Mib(real)} | {Mib(runPeak)} | {Mib(graphAtRunPeak)} | {Mib(planned.Max())} | {Mib(plannedInItsOrder.Max())} | {Mib(ModelAt(modelPeakAt))} | {Mib(modelAtRunPeak)} | {(result.BackendPeakBytes is { } judged ? Mib(judged[result.AllStrategies.Select((x, i) => (x, i)).First(p => ReferenceEquals(p.x.Graph, result.OptimizedGraph)).i]) : "-")} "
                        + $"| {string.Join(", ", scratch.OrderByDescending(x => x.Value.Bytes).Select(x => $"{x.Key} x{x.Value.Count} up to {Mib(x.Value.Bytes)}"))} |");
            File.WriteAllText(Path.Combine(OutputDirectory(), $"ort-against-model-{where}-x{scale}.md"), string.Join("\n", summary) + "\n");
            File.WriteAllText(Path.Combine(OutputDirectory(), $"ort-against-model-{where}-x{scale}-detail.md"), detail.ToString());
        }
    }

    /// <summary>
    /// What a run of <paramref name="graph"/> in <paramref name="order"/> (node indices) holds after
    /// each node, laid out as ONNX Runtime's allocation plan lays it out (<see cref="OrtRunMemory"/>,
    /// without its kernels' scratch): a view as its input's memory, an activation over its input
    /// where that dies there, and a dead value's buffer kept for the next value of its exact shape
    /// and type — occupied from its first value's birth to its last one's death — or given back
    /// where none takes it, or where the graph does not fix the shape of either. The values named in
    /// <paramref name="notCounted"/> take nothing; every other graph output is held to the end.
    /// </summary>
    private static long[] PlannedOccupancy(GraphProto graph, IReadOnlyList<int> order, Dictionary<string, PlacementShapes.Value> shapes, HashSet<string> notCounted, Dictionary<int, string>? trace = null)
    {
        var root = new Dictionary<string, string>(StringComparer.Ordinal);
        string RootOf(string v) => root.TryGetValue(v, out var r) && r != v ? root[v] = RootOf(r) : v;
        foreach (var n in order)
        {
            var node = graph.Nodes[n];
            for (int o = 0; o < node.Outputs.Count; o++)
                for (int i = 0; i < node.Inputs.Count; i++)
                    if (node.Outputs[o].Length > 0 && node.Inputs[i].Length > 0 && OutputAliasProof.Shares(node, i, o))
                    {
                        root[node.Outputs[o]] = RootOf(node.Inputs[i]);
                        break;
                    }
        }
        var outputs = graph.Outputs.Select(o => RootOf(o.Name)).ToHashSet(StringComparer.Ordinal);
        var lastRead = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int k = 0; k < order.Count; k++)
            foreach (var v in graph.Nodes[order[k]].Inputs.Where(v => v.Length > 0))
                lastRead[RootOf(v)] = k;
        var buffers = new List<(long Bytes, string Shape, int Start, int End)>();
        var bufferOf = new Dictionary<string, int>(StringComparer.Ordinal);
        var users = new Dictionary<int, int>();
        var free = new List<int>();
        // ONNX Runtime reuses no buffer for, and hands none on from, a value whose shape the graph
        // does not fix.
        var unfixed = OrtRunMemory.ShapesNotFixed(graph);
        for (int k = 0; k < order.Count; k++)
        {
            var node = graph.Nodes[order[k]];
            foreach (var o in node.Outputs.Where(o => o.Length > 0))
            {
                if (RootOf(o) != o || notCounted.Contains(o) || !shapes.TryGetValue(o, out var value) || value.Bytes <= 0) continue;
                var shape = $"{value.ElementType}[{string.Join(",", value.Shape)}]";
                if (OrtWritesOver.Contains(node.OpType) && node.Inputs.Count > 0 && node.Inputs[0] is { Length: > 0 } operand
                    && bufferOf.TryGetValue(RootOf(operand), out var over) && users[over] == 1 && lastRead.GetValueOrDefault(RootOf(operand), -1) == k
                    && buffers[over].Shape == shape)
                {
                    bufferOf[o] = over;
                    users[over]++;
                    continue;
                }
                var taken = unfixed.Contains(o) ? -1 : free.FindLastIndex(b => buffers[b].Shape == shape);
                if (taken >= 0)
                {
                    var id = free[taken];
                    free.RemoveAt(taken);
                    if (trace is not null) trace[k] = trace.GetValueOrDefault(k, "") + $" {o} into the buffer freed at {buffers[id].End};";
                    buffers[id] = buffers[id] with { End = int.MaxValue };
                    bufferOf[o] = id;
                    users[id] = 1;
                    continue;
                }
                buffers.Add((value.Bytes, shape, k, int.MaxValue));
                bufferOf[o] = buffers.Count - 1;
                users[buffers.Count - 1] = 1;
                if (trace is not null && value.Bytes >= (1 << 20)) trace[k] = trace.GetValueOrDefault(k, "") + $" {o} new;";
            }
            foreach (var v in node.Inputs.Concat(node.Outputs).Where(v => v.Length > 0).Select(RootOf).Distinct())
            {
                if (!bufferOf.TryGetValue(v, out var id) || outputs.Contains(v) || lastRead.GetValueOrDefault(v, k) > k) continue;
                bufferOf.Remove(v);
                if (--users[id] > 0) continue;
                buffers[id] = buffers[id] with { End = k };
                if (!unfixed.Contains(v)) free.Add(id);
                if (trace is not null && buffers[id].Bytes >= (1 << 20)) trace[k] = trace.GetValueOrDefault(k, "") + $" frees {v};";
            }
        }
        var occupancy = new long[order.Count];
        foreach (var (bytes, _, start, end) in buffers)
            for (int k = start; k <= Math.Min(end, order.Count - 1); k++)
                occupancy[k] += bytes;
        return occupancy;
    }

    /// <summary>One kernel of a profiled run: its node in the graph run, and what the run held as
    /// it began, at most during it, and after it, with the blocks it took.</summary>
    /// <summary>How long the last session <see cref="OrtTimeline"/> built took to build.</summary>
    private static double SessionBuildMs;

    private sealed record OrtKernel(int Node, long Before, long Most, long After, List<long> Taken);

    /// <summary>
    /// The graph ONNX Runtime runs for <paramref name="model"/>, built as a training step's session
    /// is — on the card's provider where <paramref name="onCard"/> — and one run of it fed
    /// <paramref name="shapes"/>, kernel by kernel: every block the run took or gave back through
    /// Shorokoo's allocator, put to the kernel running at the time on the run's own clock, and
    /// what was taken before the first kernel (the feeds' copies on a card).
    /// </summary>
    private static (GraphProto Graph, List<OrtKernel> Kernels, long BeforeFirst, List<(long Size, int Taken, int Freed)> Blocks) OrtTimeline(ModelProto model, bool onCard, (Shape Shape, DType DType)[] shapes)
    {
        var directory = Path.Combine(Path.GetTempPath(), $"shorokoo-ort-timeline-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            using var stream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(stream, model);
            string profile;
            var host = RuntimeAllocator.ForHost().Shared.Open("timeline");
            var card = onCard ? RuntimeAllocator.ForCard(0).Shared.Open("timeline") : null;
            var events = new List<(long Ticks, bool Allocation, IntPtr Address, long Size)>();
            long runStart;
            using (var options = new SessionOptions())
            {
                OrtBackend.Configure(options, ShorokooGraphOptimization.TrainingStep, ShorokooLogSeverity.Fatal);
                options.AddSessionConfigEntry("session.use_env_allocators", "1");
                options.OptimizedModelFilePath = Path.Combine(directory, "optimized.onnx");
                options.ProfileOutputPathPrefix = Path.Combine(directory, "profile");
                options.EnableProfiling = true;
                if (onCard) OrtBackend.AppendCuda(options, 0, DeviceMemorySettings.Default, Precision());
                InferenceSession session;
                var built = Stopwatch.StartNew();
                using (CachingAllocator.Charge(host, card))
                    session = new InferenceSession(stream.ToArray(), options);
                SessionBuildMs = built.Elapsed.TotalMilliseconds;
                using (session)
                {
                    var feeds = new Dictionary<string, OrtValue>();
                    for (var i = 0; i < session.InputNames.Count; i++)
                        feeds[session.InputNames[i]] = Shorokoo.Tests.Utils.SyntheticFeed.Tensor(shapes[i].Shape, shapes[i].DType, i);
                    using var runOptions = new RunOptions();
                    using (CachingAllocator.Charge(host, card))
                        foreach (var output in session.Run(runOptions, feeds, session.OutputNames)) output.Dispose();
                    CachingAllocator.Observer = e =>
                    {
                        if (e.OnCard != onCard) return;
                        lock (events) events.Add((Stopwatch.GetTimestamp(), e.Allocation, e.Address, e.Size));
                    };
                    runStart = Stopwatch.GetTimestamp();
                    try
                    {
                        using (CachingAllocator.Charge(host, card))
                            foreach (var output in session.Run(runOptions, feeds, session.OutputNames)) output.Dispose();
                    }
                    finally
                    {
                        CachingAllocator.Observer = null;
                    }
                    foreach (var feed in feeds.Values) feed.Dispose();
                    profile = session.EndProfiling();
                }
            }
            GraphProto graph;
            using (var written = File.OpenRead(Path.Combine(directory, "optimized.onnx")))
                graph = ProtoBuf.Serializer.Deserialize<ModelProto>(written).Graph;
            var index = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int n = 0; n < graph.Nodes.Count; n++) index.TryAdd(graph.Nodes[n].Name, n);
            var profiled = JArray.Parse(File.ReadAllText(profile));
            var runEvent = profiled.Where(e => (string?)e["cat"] == "Session" && (string?)e["name"] == "model_run").OrderBy(e => (long)e["ts"]!).Last();
            var kernels = profiled
                .Where(e => (string?)e["cat"] == "Node" && ((string?)e["name"])?.EndsWith("_kernel_time", StringComparison.Ordinal) == true
                            && (long)e["ts"]! >= (long)runEvent["ts"]!)
                .Select(e => (Name: ((string)e["name"]!)[..^"_kernel_time".Length], Ts: (long)e["ts"]!))
                .Where(k => index.ContainsKey(k.Name))
                .OrderBy(k => k.Ts).ToList();
            long Us(long ticks) => (long)runEvent["ts"]! + (ticks - runStart) * 1_000_000 / Stopwatch.Frequency;
            var live = new Dictionary<IntPtr, long>();
            long held = 0, beforeFirst = 0;
            var result = new List<OrtKernel>();
            var blocks = new List<(long Size, int Taken, int Freed)>();
            var blockAt = new Dictionary<IntPtr, int>();
            var next = 0;
            var current = -1;
            void Apply((long Ticks, bool Allocation, IntPtr Address, long Size) ev, List<long>? taken)
            {
                if (ev.Allocation)
                {
                    live[ev.Address] = ev.Size; held += ev.Size; taken?.Add(ev.Size);
                    blockAt[ev.Address] = blocks.Count;
                    blocks.Add((ev.Size, current, -1));
                }
                else if (live.Remove(ev.Address, out var was))
                {
                    held -= was;
                    if (blockAt.Remove(ev.Address, out var b)) blocks[b] = blocks[b] with { Freed = current };
                }
            }
            while (next < events.Count && (kernels.Count == 0 || Us(events[next].Ticks) < kernels[0].Ts))
            {
                if (events[next].Allocation) beforeFirst += events[next].Size;
                Apply(events[next++], null);
            }
            for (int k = 0; k < kernels.Count; k++)
            {
                current = k;
                var before = held;
                var most = held;
                var taken = new List<long>();
                var end = k + 1 < kernels.Count ? kernels[k + 1].Ts : long.MaxValue;
                while (next < events.Count && Us(events[next].Ticks) < end)
                {
                    Apply(events[next++], taken);
                    most = Math.Max(most, held);
                }
                result.Add(new OrtKernel(index[kernels[k].Name], before, most, held, taken));
            }
            return (graph, result, beforeFirst, blocks);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>The graph ONNX Runtime runs for <paramref name="model"/>, built as a training step's
    /// session is — on the card's provider where <paramref name="onCard"/> — and written out, and
    /// the order its kernels ran in on a run fed <paramref name="shapes"/>, read off its profile, by
    /// node index: the nodes it ran no kernel for last, in its depth-first order.</summary>
    private static (GraphProto Graph, int[] Order, string Held) OrtRun(ModelProto model, bool onCard, (Shape Shape, DType DType)[] shapes)
    {
        var directory = Path.Combine(Path.GetTempPath(), $"shorokoo-step-anatomy-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            using var stream = new MemoryStream();
            ProtoBuf.Serializer.Serialize(stream, model);
            string profile;
            // The session allocates through Shorokoo's allocator, as a backend's does, so that every
            // block it takes is seen, with when: the blocks held at the run's peak are put to the
            // kernel that ran as each was taken.
            var host = RuntimeAllocator.ForHost().Shared.Open("anatomy");
            var card = onCard ? RuntimeAllocator.ForCard(0).Shared.Open("anatomy") : null;
            var events = new List<(long Ticks, bool Allocation, IntPtr Address, long Size)>();
            long runStart;
            using (var options = new SessionOptions())
            {
                OrtBackend.Configure(options, ShorokooGraphOptimization.TrainingStep, ShorokooLogSeverity.Fatal);
                options.AddSessionConfigEntry("session.use_env_allocators", "1");
                options.OptimizedModelFilePath = Path.Combine(directory, "optimized.onnx");
                options.ProfileOutputPathPrefix = Path.Combine(directory, "profile");
                options.EnableProfiling = true;
                if (onCard) OrtBackend.AppendCuda(options, 0, DeviceMemorySettings.Default, Precision());
                InferenceSession session;
                using (CachingAllocator.Charge(host, card))
                    session = new InferenceSession(stream.ToArray(), options);
                using (session)
                {
                    var feeds = new Dictionary<string, OrtValue>();
                    for (var i = 0; i < session.InputNames.Count; i++)
                        feeds[session.InputNames[i]] = Shorokoo.Tests.Utils.SyntheticFeed.Tensor(shapes[i].Shape, shapes[i].DType, i);
                    using var runOptions = new RunOptions();
                    using (CachingAllocator.Charge(host, card))
                        foreach (var output in session.Run(runOptions, feeds, session.OutputNames)) output.Dispose();
                    CachingAllocator.Observer = e =>
                    {
                        if (e.OnCard != onCard) return;
                        lock (events) events.Add((System.Diagnostics.Stopwatch.GetTimestamp(), e.Allocation, e.Address, e.Size));
                    };
                    runStart = System.Diagnostics.Stopwatch.GetTimestamp();
                    try
                    {
                        using (CachingAllocator.Charge(host, card))
                            foreach (var output in session.Run(runOptions, feeds, session.OutputNames)) output.Dispose();
                    }
                    finally
                    {
                        CachingAllocator.Observer = null;
                    }
                    foreach (var feed in feeds.Values) feed.Dispose();
                    profile = session.EndProfiling();
                }
            }
            GraphProto graph;
            using (var written = File.OpenRead(Path.Combine(directory, "optimized.onnx")))
                graph = ProtoBuf.Serializer.Deserialize<ModelProto>(written).Graph;
            var index = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int n = 0; n < graph.Nodes.Count; n++) index.TryAdd(graph.Nodes[n].Name, n);
            var profiled = JArray.Parse(File.ReadAllText(profile));
            var runEvent = profiled.Where(e => (string?)e["cat"] == "Session" && (string?)e["name"] == "model_run").OrderBy(e => (long)e["ts"]!).Last();
            var kernels = profiled
                .Where(e => (string?)e["cat"] == "Node" && ((string?)e["name"])?.EndsWith("_kernel_time", StringComparison.Ordinal) == true
                            && (long)e["ts"]! >= (long)runEvent["ts"]!)
                .Select(e => (Name: ((string)e["name"]!)[..^"_kernel_time".Length], Ts: (long)e["ts"]!, Dur: (long)e["dur"]!))
                .OrderBy(k => k.Ts).ToList();
            var ran = kernels.Select(k => k.Name).Where(index.ContainsKey).Select(name => index[name]).Distinct().ToList();
            var seen = ran.ToHashSet();
            var anatomy = new StepAnatomy(graph, 0, 0, 0);

            // The blocks held at the peak of the observed run, each put to the kernel running as it
            // was taken -- the last to start by then -- on the run's own clock.
            var live = new Dictionary<IntPtr, (long Size, long Ticks)>();
            Dictionary<IntPtr, (long Size, long Ticks)> atPeak = [];
            long held = 0, most = 0, mostAt = 0;
            foreach (var e in events)
            {
                if (e.Allocation) { live[e.Address] = (e.Size, e.Ticks); held += e.Size; }
                else if (live.Remove(e.Address, out var was)) held -= was.Size;
                if (held > most) { most = held; atPeak = new(live); mostAt = e.Ticks; }
            }
            string KernelAt(long ticks)
            {
                var at = (long)runEvent["ts"]! + (ticks - runStart) * 1_000_000 / System.Diagnostics.Stopwatch.Frequency;
                var kernel = kernels.LastOrDefault(k => k.Ts <= at);
                if (kernel.Name is null) return "before the first kernel";
                var op = index.TryGetValue(kernel.Name, out var n) ? $"{graph.Nodes[n].OpType} #{ran.IndexOf(n)}" : "?";
                return at <= kernel.Ts + kernel.Dur ? op : $"after {op}";
            }
            var heldText = $"observed peak {Mib(most)} reached in {KernelAt(mostAt)}: " + string.Join(", ", atPeak.Values.Where(v => v.Size >= Big).OrderByDescending(v => v.Size)
                .Select(v => $"{Mib(v.Size)} by {KernelAt(v.Ticks)}"));
            return (graph, [.. ran, .. anatomy.OrtOrder().Where(n => !seen.Contains(n))], heldText);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Why placement does or does not reach each benchmark family's inference graph: per family, at
    /// <c>$SHOROKOO_MEMORY_REUSE_SCALE</c> times its batch, the values of a mebibyte or more, each
    /// with its operator and why it cannot be placed (<see cref="PlacementProof.Unplaceable"/>) or
    /// where the planner put it, the operators of the values whose shape is not known, and the plan.
    /// </summary>
    [Fact]
    public void RecordWhyPlacementReachesEachFamilyOrNot()
    {
        var scale = long.TryParse(Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SCALE"), out var s) ? s : 1;
        var lines = new List<string> { $"# Why placement reaches each family's inference graph, batch x{scale}" };
        foreach (var (family, model, benchmarkShape) in MemoryPassBenchmarkTests.Suite)
        {
            long[] shape = [benchmarkShape[0] * scale, .. benchmarkShape[1..]];
            var count = (int)shape.Aggregate(1L, (a, d) => a * d);
            var sample = TensorData(shape, new float[count]);
            var concrete = model().ToConcreteArchitecture([sample]).ToConcreteModel();
            using var context = new ComputeContext();
            var compiled = context.Compile(concrete);
            var graph = ((OrtSession)compiled.Session).Placements!.OriginalModel.Graph!;
            var input = graph.Inputs.First(i => graph.Initializers.All(t => t.Name != i.Name)).Name;
            var shapes = PlacementShapes.Evaluate(graph, new Dictionary<string, (long[], int)> { [input] = (shape, 1) });
            var proof = new PlacementProof(graph, new Dictionary<string, long> { [input] = count * 4L }, shapes, new HashSet<string>(graph.Outputs.Select(o => o.Name)));
            var plan = proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes);
            lines.Add($"\n## {family} {string.Join("x", shape)}: {graph.Nodes.Count} nodes, input {Mib(count * 4L)}, {plan.Count} placed\n");
            var unknown = graph.Nodes.Where(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o))).GroupBy(n => n.OpType).Select(g => $"{g.Key} x{g.Count()}");
            lines.Add($"unknown shapes: {string.Join(", ", unknown)}");
            var roots = graph.Nodes.Where(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o))
                                               && n.Inputs.All(i => i.Length == 0 || shapes.ContainsKey(i)))
                .Select(n => $"{n.OpType}({string.Join("; ", n.Inputs.Select(i => i.Length == 0 ? "-" : $"{Producer(graph, i)} {string.Join("x", shapes[i].Shape)}:{shapes[i].ElementType}{(shapes[i].Ints is { } v ? "=" + string.Join(",", v) : "")}"))})")
                .Distinct().Take(12);
            lines.Add($"first unknown: {string.Join(" | ", roots)}");
            var range = graph.Nodes.FirstOrDefault(n => n.OpType == "Range" && !shapes.ContainsKey(n.Outputs[0]));
            if (range is not null) lines.Add($"range limit: {Explain(graph, shapes, range.Inputs[1], 5)}");
            foreach (var node in graph.Nodes)
                foreach (var value in node.Outputs.Where(o => o.Length > 0 && shapes.TryGetValue(o, out var v) && v.Bytes >= PlacementProof.Smallest))
                {
                    var placed = plan.FirstOrDefault(p => p.Value == value);
                    lines.Add($"- {node.OpType} {value} {Mib(shapes[value].Bytes)}: {(placed.Value is not null ? $"placed at {placed.Block}+{placed.Offset}" : proof.Unplaceable(value) ?? "placeable, not placed")}");
                }
        }
        File.WriteAllText(Path.Combine(OutputDirectory(), $"why-placement-reaches-each-family-x{scale}.md"), string.Join("\n", lines) + "\n");
    }

    /// <summary>
    /// Why placement reaches a resident training step or not: <c>WideLinearModel</c> under AdamW,
    /// its batch fed <c>.Shared()</c>, with output aliasing on and off and placement on. Per
    /// signature the session settled: how it settled, its blocks, what was placed, and for every
    /// value of a mebibyte or more the step hands out, where it went or why it could not
    /// (<see cref="PlacementProof.WhyNot"/>).
    /// </summary>
    [Fact]
    public void RecordWhyPlacementReachesATrainingStep()
    {
        var settled = new List<OrtPlacements.Entry>();
        var lines = new List<string> { "# Why placement reaches a resident training step" };
        OrtPlacements.Settled = entry => { lock (settled) settled.Add(entry); };
        try
        {
            foreach (var aliasing in (bool[])[true, false])
            {
                using var context = new ComputeContext { OutputAliasing = aliasing, ValuePlacement = true, Precision = Precision() };
                var sample = TensorData([2L, 4096L], [.. Enumerable.Range(0, 8192).Select(i => (i % 13) / 13f)]);
                var rig = TrainingRig.FromScratch(
                    WideLinearModel.ComputationGraph, Shorokoo.Modules.Losses.L2Loss.ComputationGraph,
                    Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph, [sample.CopyTo(ComputeContext.Host)],
                    new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f }, runtimeContext: context);
                var input = rig.InputDef.FromOrderedData(sample);
                var target = rig.TargetDef.FromOrderedData(TensorData([2L, 4096L], new float[8192]));
                lock (settled) settled.Clear();
                using (var run = rig.BeginResidentRun(rig.CreateInitialCheckpoint()))
                    for (int i = 0; i < 3; i++) run.Step(input.Shared(), target.Shared());
                lines.Add($"\n## Output aliasing {(aliasing ? "on" : "off")}");
                foreach (var entry in settled.Where(e => e.Graph is not null))
                {
                    var graph = entry.Graph!;
                    lines.Add($"\n{entry.Stage}{(entry.Refusal is null ? "" : ": " + entry.Refusal)}; modelled {Mib(entry.PredictedPlainPeak)} plain, {Mib(entry.PredictedPlacedPeak)} placed; placed run measured {Mib(entry.PlacedPeak)}; "
                              + $"{entry.BlockBytes.Count} blocks of {Mib(entry.BlockBytes.Values.Sum())}; {entry.Plan.Count} placed, {Mib(entry.Plan.Sum(p => p.Bytes))}\n");
                    var shapes = PlacementShapes.Evaluate(graph, entry.Given);
                    if (entry.VariantGraph is { } variantGraph)
                    {
                        var variantProof = new PlacementProof(variantGraph, entry.BlockBytes, PlacementShapes.Evaluate(variantGraph, entry.Given), runsInOrder: true);
                        lines.Add($"variant graph, in its own order: modelled {Mib(variantProof.ModelledPeak([]))} plain, {Mib(variantProof.ModelledPeak(entry.Plan))} placed; "
                                  + $"its order {(variantGraph.Nodes.Select(n => n.Name).SequenceEqual(graph.Nodes.Select(n => n.Name)) ? "is" : "is not")} the plain graph's");
                    }
                    var outputs = graph.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
                    var proof = new PlacementProof(graph, entry.BlockBytes, shapes, outputs, runsInOrder: true);
                    var plan = proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes);
                    foreach (var output in graph.Outputs.Select(o => o.Name).Where(o => shapes.TryGetValue(o, out var v) && v.Bytes >= PlacementProof.Smallest))
                    {
                        var at = plan.FirstOrDefault(p => p.Value == output);
                        lines.Add($"- {output} {Mib(shapes[output].Bytes)}: {(at.Value is not null ? $"placed at {at.Block}+{at.Offset}" : proof.WhyNot(output, plan))}");
                    }
                }
            }
        }
        finally
        {
            OrtPlacements.Settled = null;
        }
        File.WriteAllText(Path.Combine(OutputDirectory(), "why-placement-reaches-a-training-step.md"), string.Join("\n", lines) + "\n");
    }

    /// <summary>Why a settled signature's graph places what it does: the operators of its values
    /// whose shape is unknown, and each value of a mebibyte or more, where the planner put it or why
    /// it could not.</summary>
    private static string Why(OrtPlacements.Entry entry)
    {
        var graph = entry.Graph!;
        var shapes = PlacementShapes.Evaluate(graph, entry.Given);
        var proof = new PlacementProof(graph, entry.BlockBytes, shapes, runsInOrder: true);
        var plan = proof.Plan(PlacementProof.Smallest, PlacementProof.IdleOutputBytes);
        var lines = new List<string>
        {
            $"{graph.Nodes.Count} nodes; unknown shapes: {string.Join(", ", graph.Nodes.Where(n => n.Outputs.Any(o => o.Length > 0 && !shapes.ContainsKey(o))).GroupBy(n => $"{n.Domain}:{n.OpType}").Select(g => $"{g.Key} x{g.Count()}"))}",
        };
        foreach (var node in graph.Nodes)
            foreach (var value in node.Outputs.Where(o => o.Length > 0 && shapes.TryGetValue(o, out var v) && v.Bytes >= PlacementProof.Smallest))
            {
                var at = plan.FirstOrDefault(p => p.Value == value);
                lines.Add($"- {node.OpType} {value} {Mib(shapes[value].Bytes)}: {(at.Value is not null ? $"placed at {at.Block}+{at.Offset}" : proof.WhyNot(value, plan))}");
            }
        return string.Join("\n", lines);
    }

    /// <summary>What makes <paramref name="value"/> in <paramref name="graph"/>, and what that reads,
    /// two levels down.</summary>
    private static string Producer(GraphProto graph, string value, int depth = 2)
    {
        if (graph.Initializers.Any(t => t.Name == value)) return "init";
        if (graph.Inputs.Any(i => i.Name == value)) return "input";
        var node = graph.Nodes.FirstOrDefault(n => n.Outputs.Contains(value));
        if (node is null) return "?";
        return depth == 0 ? node.OpType : $"{node.OpType}[{string.Join(",", node.Inputs.Where(i => i.Length > 0).Select(i => Producer(graph, i, depth - 1)))}]";
    }

    /// <summary>How <paramref name="value"/> is made, with each value's known contents, a few
    /// levels down.</summary>
    private static string Explain(GraphProto graph, Dictionary<string, PlacementShapes.Value> shapes, string value, int depth)
    {
        var known = shapes.TryGetValue(value, out var v) ? $"{string.Join("x", v.Shape)}:{v.ElementType}{(v.Ints is { } i ? "=" + string.Join(",", i) : "")}" : "unknown";
        var node = graph.Nodes.FirstOrDefault(n => n.Outputs.Contains(value));
        if (node is null || depth == 0) return known;
        var attrs = string.Join(",", node.Attributes.Select(a => $"{a.Name}={a.I}"));
        return $"{node.OpType}<{attrs}>({string.Join("; ", node.Inputs.Where(x => x.Length > 0).Select(x => Explain(graph, shapes, x, depth - 1)))})->{known}";
    }

    private sealed record FamilyFigures(
        long StepPeak, double StepMs, string StepPlans, long InferencePeak, double InferenceMs, double SharedInferenceMs, string InferencePlans);

    /// <summary>A backend of the default's type whose sessions each keep thread pools of their own
    /// (<see cref="OrtBackend.SessionsShareThreadPools"/> off), for <c>$SHOROKOO_MEMORY_REUSE_SESSION_POOLS</c>.</summary>
    private static OrtBackend SessionPools()
    {
        var backend = (OrtBackend)Activator.CreateInstance(DefaultBackend.Instance.GetType())!;
        typeof(OrtBackend).GetProperty(nameof(OrtBackend.SessionsShareThreadPools))!.SetValue(backend, false);
        return backend;
    }

    private static FamilyFigures MeasureFamily(ComputationGraph model, long[] shape, bool placing, bool onCard, string backend = "ort")
    {
        const int Warm = 4, Timed = 15;
        var count = (int)shape.Aggregate(1L, (a, d) => a * d);
        float[] Values(int seed) => [.. Enumerable.Range(0, count).Select(i => ((i * 7 + seed) % 101) / 101f - 0.5f)];
        using var context = backend switch
        {
            "torch-cpu" => new ComputeContext(new Shorokoo.PyTorch.Cpu.TorchCpuBackend()) { ValuePlacement = placing, Precision = Precision() },
            "torch-cuda" => new ComputeContext(new Shorokoo.PyTorch.Cuda.TorchCudaBackend()) { ValuePlacement = placing, Precision = Precision() },
            _ when Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_SESSION_POOLS") is "1" => new ComputeContext(SessionPools()) { ValuePlacement = placing, Precision = Precision() },
            _ => new ComputeContext { ValuePlacement = placing, Precision = Precision() },
        };
        (T, long) Observe<T>(Func<T> run) => backend == "ort" ? Observed(onCard, run) : TorchObserved(backend, run);
        var settled = new List<OrtPlacements.Entry>();
        var torchSettled = new List<TorchPlacements.Entry>();
        OrtPlacements.Settled = entry => { lock (settled) settled.Add(entry); };
        TorchPlacements.Settled = entry => { lock (torchSettled) torchSettled.Add(entry); };
        using var unhook = new Unhook();
        string Settled()
        {
            lock (torchSettled)
                if (torchSettled.Count > 0)
                {
                    var torchLine = string.Join(", ", torchSettled.Select(e => e.Plan.Count > 0 ? $"placed {e.Plan.Count}" : $"refused: {e.Refusal}")
                        .GroupBy(x => x).Select(g => g.Count() == 1 ? g.Key : $"{g.Count()}x {g.Key}"));
                    torchSettled.Clear();
                    return torchLine;
                }
            lock (settled)
            {
                foreach (var e in settled.Where(e => e.Graph is not null && e.Stage == OrtPlacements.Stage.Refused))
                    File.AppendAllText(Path.Combine(OutputDirectory(), "refusals.md"), $"\n## {e.Refusal}\n\n{Why(e)}\n");
                var line = string.Join(", ", settled.Select(e => e.Stage == OrtPlacements.Stage.Adopted
                    ? $"adopted {e.Plan.Count} (modelled {Mib(e.PredictedPlainPeak)} to {Mib(e.PredictedPlacedPeak)}, placed run {Mib(e.PlacedPeak)})"
                    : $"refused: {e.Refusal} ({string.Join(",", e.BlockBytes.Select(b => $"{b.Key}={b.Value}"))}; {string.Join(",", e.Given.Select(g => $"{g.Key}:{string.Join("x", g.Value.Shape)}:{g.Value.ElementType}"))})").GroupBy(x => x).Select(g => g.Count() == 1 ? g.Key : $"{g.Count()}x {g.Key}"));
                settled.Clear();
                return line.Length == 0 ? "-" : line;
            }
        }

        var sample = TensorData(shape, Values(0));
        var concrete = model.ToConcreteArchitecture([sample]).ToConcreteModel();
        var predicted = context.Execute(concrete, sample.Shared())[0].ToTensorData();
        long[] dims = [.. predicted.Shape.Dims.Select(d => (long)d)];
        var target = TensorData(dims, new float[dims.Aggregate(1L, (a, d) => a * d)]);
        predicted.Delete();

        var rig = TrainingRig.FromScratch(
            model, Shorokoo.Modules.Losses.L2Loss.ComputationGraph, Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph,
            [sample.CopyTo(ComputeContext.Host)], new Shorokoo.Modules.Optimizers.AdamWOptimizerHyperparameters { LearningRate = 0.001f },
            runtimeContext: context);
        var input = rig.InputDef.FromOrderedData(sample);
        var targets = rig.TargetDef.FromOrderedData(target);
        long stepPeak;
        var stepTimes = new List<double>();
        using (var run = rig.BeginResidentRun(rig.CreateInitialCheckpoint()))
        {
            for (int i = 0; i < Warm; i++) run.Step(input.Shared(), targets.Shared());
            (_, stepPeak) = Observe(() => run.Step(input.Shared(), targets.Shared()));
            for (int i = 0; i < Timed; i++)
            {
                var watch = Stopwatch.StartNew();
                run.Step(input.Shared(), targets.Shared());
                stepTimes.Add(watch.Elapsed.TotalMilliseconds);
            }
        }
        var stepPlans = Settled();

        var compiled = context.Compile(concrete);
        NamedModelParam[] Infer() => compiled.Execute(TensorData(shape, Values(1)));
        for (int i = 0; i < Warm; i++) Release(Infer());
        var (outputs, inferencePeak) = Observe(Infer);
        Release(outputs);
        // Each timed run is paired with one on a shared input, which no run writes into: the same
        // session's plain run beside its placed one, in the same moment of a loaded machine.
        var inferenceTimes = new List<double>();
        var sharedTimes = new List<double>();
        // With $SHOROKOO_MEMORY_REUSE_TWIN set, the shared runs go to a second compile of the model:
        // two sessions taking turns, which is what a plain session and its variant do.
        var twin = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_TWIN") is null ? null : context.Compile(concrete);
        for (int i = 0; i < Timed; i++)
        {
            var input1 = TensorData(shape, Values(1));
            var watch = Stopwatch.StartNew();
            var each = compiled.Execute(input1);
            inferenceTimes.Add(watch.Elapsed.TotalMilliseconds);
            Release(each);
            if (Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_UNPAIRED") is not null) continue;
            var kept = TensorData(shape, Values(1));
            watch.Restart();
            each = (twin ?? compiled).Execute(kept.Shared());
            sharedTimes.Add(watch.Elapsed.TotalMilliseconds);
            Release(each);
            kept.Delete();
        }
        return new FamilyFigures(stepPeak, Median(stepTimes), stepPlans, inferencePeak, Median(inferenceTimes), sharedTimes.Count == 0 ? 0 : Median(sharedTimes), Settled());
    }

    private sealed class Unhook : IDisposable
    {
        public void Dispose()
        {
            OrtPlacements.Settled = null;
            TorchPlacements.Settled = null;
        }
    }

    private static void Release(NamedModelParam[] outputs)
    {
        foreach (var output in outputs) output.ToTensorData().Delete();
    }

    /// <summary>What <paramref name="run"/> answers, and the most torch held at once beyond what it
    /// held as the run began: on CUDA off its allocator, on the CPU, where torch keeps no such figure,
    /// off its profiler's memory timeline. With <c>$SHOROKOO_MEMORY_REUSE_LOG</c> set, what is held at
    /// that peak is appended to <c>torch-cuda-peak.txt</c> or <c>torch-cpu-peak.txt</c>: each block,
    /// with the line of the translation that made it on CUDA, and on the CPU the operator, the
    /// support package's function and the number of the translation's call (its calls, in order, in
    /// <c>torch-cpu-calls.txt</c>).</summary>
    private static (T Result, long Peak) TorchObserved<T>(string backend, Func<T> run)
    {
        using (PythonRuntime.Gil())
        {
            using var scope = Py.CreateScope();
            scope.Exec(backend == "torch-cuda"
                ? """
                  import os, torch
                  torch.cuda.synchronize()
                  torch.cuda.reset_peak_memory_stats()
                  before = torch.cuda.memory_allocated()
                  if os.environ.get("SHOROKOO_MEMORY_REUSE_LOG"):
                      torch.cuda.memory._record_memory_history(max_entries=500000)
                  """
                : """
                  import torch
                  from torch.profiler import profile, ProfilerActivity
                  prof = profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True)
                  prof.__enter__()
                  """);
            var result = run();
            scope.Exec(backend == "torch-cuda"
                ? """
                  torch.cuda.synchronize()
                  peak = torch.cuda.max_memory_allocated() - before
                  if os.environ.get("SHOROKOO_MEMORY_REUSE_LOG"):
                      # Every block held at the run's peak, each put to the line of the translation
                      # or the support package that asked for it.
                      snapshot = torch.cuda.memory._snapshot()
                      torch.cuda.memory._record_memory_history(enabled=None)
                      live, held, most, at_most = {}, 0, -1, {}
                      for event in snapshot["device_traces"][0]:
                          if event["action"] == "alloc":
                              live[event["addr"]] = event
                              held += event["size"]
                              if held > most:
                                  most, at_most = held, dict(live)
                          elif event["action"] in ("free_requested", "free_completed") and event["addr"] in live:
                              held -= live.pop(event["addr"])["size"]
                      def where(event):
                          frames = [f for f in event.get("frames", []) if "shorokoo" in f["filename"]]
                          main = [f for f in frames if f["name"] == "main"]
                          return " < ".join(f"{f['name']}:{f['line']}" for f in frames[:2] + main[:1]) or "?"
                      lines = sorted(((e["size"], where(e)) for e in at_most.values()), reverse=True)[:16]
                      with open(os.path.join(os.environ["SHOROKOO_MEMORY_REUSE_DIR"], "torch-cuda-peak.txt"), "a") as f:
                          f.write(f"peak {most / 2**20:.2f} MiB: " + "; ".join(f"{s / 2**20:.2f} {w}" for s, w in lines) + chr(10))
                  """
                : """
                  prof.__exit__(None, None, None)
                  def transfer(e):
                      while e is not None:
                          if "from_host" in e.name:
                              return True
                          e = e.cpu_parent
                      return False
                  # The most the run held at once: its memory profile's timeline of every tensor
                  # storage made and let go of, in order, what was there before the run left out.
                  held = peak = 0
                  for _, action, _, size in prof._memory_profile().timeline:
                      if action.name == "CREATE":
                          held += size
                          peak = max(peak, held)
                      elif action.name == "DESTROY":
                          held -= size
                  import os
                  if os.environ.get("SHOROKOO_MEMORY_REUSE_LOG"):
                      import tempfile
                      directory = os.environ.get("SHOROKOO_MEMORY_REUSE_DIR") or os.path.join(tempfile.gettempdir(), "shorokoo-memory-reuse")
                      # Every storage held at the run's peak, each put to the operator and the line of
                      # the translation or the support package that made it.
                      memory = prof._memory_profile()
                      live, held, most, at_most = {}, 0, -1, {}
                      for t, action, key, size in memory.timeline:
                          if action.name == "CREATE":
                              live[key[0]] = (t, size)
                              held += size
                              if held > most:
                                  most, at_most = held, dict(live)
                          elif action.name == "DESTROY":
                              held -= size
                              live.pop(key[0], None)
                      made = {}
                      for node in memory._op_tree.sorted_nodes:
                          if node.tag == torch._C._profiler._EventType.Allocation:
                              made.setdefault(node.start_time_ns, node)
                      written = []
                      def where(t):
                          # The operator, the support package's function and the translation's call
                          # that made the storage: its number among the run's own calls, in order.
                          node = made.get(t)
                          names, call = [], "?"
                          child, parent = node, node.parent if node is not None else None
                          while parent is not None:
                              if parent.name.endswith(": main") and call == "?":
                                  calls = [c for c in parent.children if ".py(" in c.name and "stop_point" not in c.name]
                                  if not written:
                                      written.append(1)
                                      with open(os.path.join(directory, "torch-cpu-calls.txt"), "w") as f:
                                          f.write(chr(10).join(c.name.split("/")[-1].split(chr(92))[-1] for c in calls) + chr(10))
                                  call = next((str(i) for i, c in enumerate(calls) if c.start_time_ns == child.start_time_ns and c.name == child.name), "?")
                              if (parent.name.startswith("aten::") and not any(n.startswith("aten::") for n in names)) or (".py(" in parent.name and "shorokoo" in parent.name):
                                  names.append(parent.name.split("/")[-1].split(chr(92))[-1])
                              child, parent = parent, parent.parent
                          return f"#{call} " + (" < ".join(names[:3]) or "?")
                      peak_at = max((t for t, _ in at_most.values()), default=0)
                      blocks = sorted(((size, where(t)) for t, size in at_most.values() if size >= 2**20), reverse=True)
                      with open(os.path.join(directory, "torch-cpu-peak.txt"), "a") as f:
                          f.write(f"peak {most / 2**20:.2f} MiB at {where(peak_at)}: " + "; ".join(f"{s / 2**20:.2f} {w}" for s, w in blocks) + chr(10))
                  """);
            return (result, scope.Get<long>("peak"));
        }
    }

    /// <summary>What <paramref name="run"/> answers, and the most Shorokoo's allocator had handed
    /// out — on the card or the host — beyond what it had as the run began.</summary>
    private static (T Result, long Peak) Observed<T>(bool onCard, Func<T> run)
    {
        var gate = new object();
        long current = 0, peak = 0;
        var live = new Dictionary<IntPtr, long>();
        Dictionary<IntPtr, long> atPeak = [];
        var log = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_LOG") is not null;
        CachingAllocator.Observer = e =>
        {
            if (e.OnCard != onCard) return;
            lock (gate)
            {
                current += e.Allocation ? e.Size : -e.Size;
                if (log)
                {
                    if (e.Allocation) live[e.Address] = e.Size;
                    else live.Remove(e.Address);
                }
                if (current > peak && log) atPeak = new Dictionary<IntPtr, long>(live);
                peak = Math.Max(peak, current);
            }
        };
        try
        {
            return (run(), peak);
        }
        finally
        {
            CachingAllocator.Observer = null;
            // The blocks held at the peak, largest first: what an observed run's peak is made of.
            if (log)
                File.AppendAllText(Path.Combine(OutputDirectory(), $"observed-peak-{(onCard ? "card" : "host")}.txt"),
                    $"peak {Mib(peak)}: {string.Join(" ", atPeak.Values.Where(v => v >= Big).OrderByDescending(v => v).Select(Mib))}, and {Mib(atPeak.Values.Where(v => v < Big).Sum())} in {atPeak.Values.Count(v => v < Big)} blocks under a mebibyte\n");
        }
    }

    private static double Median(List<double> values)
    {
        var sorted = values.Order().ToList();
        return sorted[sorted.Count / 2];
    }

    private static void Synchronize()
    {
        using (PythonRuntime.Gil())
        {
            using var torch = Py.Import("torch");
            using var cuda = torch.GetAttr("cuda");
            cuda.InvokeMethod("synchronize").Dispose();
        }
    }

    /// <summary>What <paramref name="run"/> returned, the most it held at once beyond what was
    /// allocated as it began, and everything it allocated, as the backend's allocator records them.</summary>
    private static (NamedModelParam[] Outputs, long Peak, long Allocated) Measured(string backend, CompiledGraph compiled, Func<NamedModelParam[]> run)
    {
        if (compiled.Session is OrtSession session)
        {
            var outputs = session.Measured(run, out var peak);
            return (outputs, peak, -1L);
        }
        using (PythonRuntime.Gil())
        {
            using var scope = Py.CreateScope();
            scope.Exec(backend == "torch-cuda"
                ? """
                  import torch
                  torch.cuda.synchronize()
                  torch.cuda.reset_peak_memory_stats()
                  before = torch.cuda.memory_allocated()
                  allocated_before = torch.cuda.memory_stats().get("allocated_bytes.all.allocated", 0)
                  """
                : """
                  import torch
                  from torch.profiler import profile, ProfilerActivity
                  prof = profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True)
                  prof.__enter__()
                  """);
            var outputs = run();
            scope.Exec(backend == "torch-cuda"
                ? """
                  torch.cuda.synchronize()
                  peak = torch.cuda.max_memory_allocated() - before
                  allocated = torch.cuda.memory_stats().get("allocated_bytes.all.allocated", 0) - allocated_before
                  """
                : """
                  prof.__exit__(None, None, None)
                  def transfer(e):
                      while e is not None:
                          if "from_host" in e.name:
                              return True
                          e = e.cpu_parent
                      return False
                  events = sorted((e for e in prof.events() if not transfer(e)), key=lambda e: e.time_range.start)
                  allocated = sum(e.self_cpu_memory_usage for e in events if e.self_cpu_memory_usage > 0)
                  running = peak = 0
                  for e in events:
                      running += e.self_cpu_memory_usage
                      peak = max(peak, running)
                  """);
            return (outputs, scope.Get<long>("peak"), scope.Get<long>("allocated"));
        }
    }

    /// <summary>
    /// The scenario on PyTorch, eagerly on the card and the CPU, and on JAX/XLA on the CPU — each way
    /// it can be written, and through Shorokoo's translation of its model — run by
    /// <c>memory_reuse_scenario.py</c> beside this file in the PyTorch backend's CUDA environment
    /// (which carries JAX's CPU build too), or its CPU environment where that cannot be had.
    /// </summary>
    [Fact]
    public void RecordTheScenarioOnPyTorchAndJax()
    {
        var dir = OutputDirectory();
        var root = AppContext.BaseDirectory;
        while (!File.Exists(Path.Combine(root, "Shorokoo.sln")))
            root = Path.GetDirectoryName(root) ?? throw new DirectoryNotFoundException("No folder above the tests holds Shorokoo.sln.");
        PythonEnvironment environment;
        try
        {
            environment = PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cu13);
        }
        catch (PythonEnvironmentException)
        {
            environment = PythonEnvironmentResolver.Resolve(PythonEnvironmentLock.Cpu);
        }

        ModelProto model;
        using (var stream = new MemoryStream(Scenario(Shapes.Computed, exposeIntermediates: false)))
            model = ProtoBuf.Serializer.Deserialize<ModelProto>(stream);
        var torch = OnnxToPythonTranslator.Translate(model, [], TorchDialect.Instance);
        var jax = OnnxToPythonTranslator.Translate(model, [], JaxDialect.Instance);
        File.WriteAllText(Path.Combine(dir, "scenario-torch.py"), torch.Source);
        File.WriteAllText(Path.Combine(dir, "scenario-jax.py"), jax.Source);
        static object Constants(TranslatedModel translated) => translated.Constants.Select(c => new
        {
            code = (int)c.ElementType,
            shape = c.Shape,
            bytes = Convert.ToBase64String(c.Bytes ?? []),
        }).ToList();
        var outFile = Path.Combine(dir, "memory-reuse-python.json");
        var configFile = Path.Combine(dir, "memory-reuse-python-config.json");
        File.WriteAllText(configFile, JsonConvert.SerializeObject(new
        {
            N, M,
            torch_source = torch.Source,
            torch_constants = Constants(torch),
            jax_source = jax.Source,
            jax_constants = Constants(jax),
            torch_package = Path.Combine(root, "src", "Backend", "PyTorch", "Shorokoo.PyTorch", "Python"),
            jax_package = Path.Combine(root, "src", "Backend", "Jax", "Shorokoo.Jax", "Python"),
            @out = outFile,
            hlo_dir = dir,
        }));

        var python = Path.Combine(environment.Directory, OperatingSystem.IsWindows() ? "Scripts" : "bin",
            OperatingSystem.IsWindows() ? "python.exe" : "python");
        var script = Path.Combine(root, "tests", "Shorokoo.Tests", "Benchmarks", "memory_reuse_scenario.py");
        using var process = Process.Start(new ProcessStartInfo(python, [script, configFile])
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        })!;
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        Assert.True(process.WaitForExit(TimeSpan.FromMinutes(15)));
        File.WriteAllText(Path.Combine(dir, "memory-reuse-python.log"), stdout.Result + stderr.Result);
        Assert.Equal(0, process.ExitCode);

        var results = JObject.Parse(File.ReadAllText(outFile));
        var text = new StringBuilder();
        text.AppendLine("# Memory reuse scenario on PyTorch and JAX");
        text.AppendLine();
        text.AppendLine(CultureInfo.InvariantCulture, $"Inputs A and B are [{2 * N}, {M}] float, {Mib(WholeBytes)} each; halves {Mib(HalfBytes)}. Python: {environment}.");
        text.AppendLine();
        text.AppendLine("## PyTorch (third of three runs)");
        text.AppendLine();
        text.AppendLine("| device | variant | allocations >= 1 MiB | MiB allocated | peak beyond inputs (MiB) | held once inputs let go (MiB) | L / A_half / B_half | correct | ms |");
        text.AppendLine("|---|---|---|---|---|---|---|---|---|");
        foreach (var r in results["torch"]!)
            text.AppendLine(r["error"] is { } error
                ? $"| {r["device"]} | {r["variant"]} | failed: {error} | | | | | | |"
                : $"| {r["device"]} | {r["variant"]} | {r["allocations"]} | {r["bytes_allocated_mib"]} | {r["peak_beyond_inputs_mib"]} | {r["held_after_inputs_released_mib"]} | {r["outputs"]} | {r["correct"]} | {r["ms"]} |");
        text.AppendLine();
        text.AppendLine("## JAX/XLA on the CPU (buffer assignment of the compiled program)");
        text.AppendLine();
        text.AppendLine("| variant | donated | arguments | outputs | aliased | temporaries | beyond inputs | inputs deleted | outputs | correct | ms | aliasing |");
        text.AppendLine("|---|---|---|---|---|---|---|---|---|---|---|---|");
        foreach (var r in results["jax"]!)
            text.AppendLine(
                $"| {r["variant"]} | {r["donated"]} | {r["argument_mib"]} | {r["output_mib"]} | {r["alias_mib"]} | {r["temp_mib"]} | {r["beyond_inputs_mib"]} | {r["inputs_deleted"]} | {r["outputs"]} | {r["correct"]} | {r["ms"]} | {((string?)r["aliases"])?.Replace("|", "/")}{(r["error"] is { } e ? " failed: " + e : "")} |");
        if (results["errors"] is JArray errors)
            foreach (var error in errors) text.AppendLine().AppendLine("```").AppendLine((string?)error).AppendLine("```");
        File.WriteAllText(Path.Combine(dir, "memory-reuse-python.md"), text.ToString());
        Assert.Null(results["errors"]);
    }

    // ---- the scenario as an ONNX model ----

    /// <summary>
    /// How the model states its shapes, which decides what ONNX Runtime can know of them when it
    /// plans the run. <i>Computed</i>: A and B are <c>[rows, cols]</c> and the run computes
    /// <c>rows / 2</c> for the slices, so the planner knows no dimension of a half, the fill or the
    /// concatenation. <i>Paired</i>: A and B are <c>[2, n, m]</c> and each half is the slice
    /// <c>[0:1]</c> or <c>[1:2]</c> of the leading dimension, <c>[1, n, m]</c> — the same bytes in
    /// the same order, with every shape symbolic yet known to the planner. <i>Static</i>: A and B
    /// are <c>[2N, M]</c> with constant slice bounds, so the whole of C's chain is computable when
    /// the session is built. <i>Tracked</i>: as Paired, with the fill made as <c>A_half * 0 + 2</c>
    /// rather than by <c>ConstantOfShape</c>, whose shape the planner cannot follow from a shape the
    /// run computes — so every shape in the graph is symbolic and known to the planner.
    /// </summary>
    internal enum Shapes { Computed, Paired, Static, Tracked }

    private static readonly Shapes[] AllShapes = [Shapes.Computed, Shapes.Paired, Shapes.Tracked, Shapes.Static];

    private static bool Leading(Shapes shapes) => shapes is Shapes.Paired or Shapes.Tracked;

    private static long[] WholeShape(Shapes shapes) => Leading(shapes) ? [2, N, M] : [2 * N, M];

    private static long[] HalfShape(Shapes shapes) => Leading(shapes) ? [1, N, M] : [N, M];

    /// <summary>
    /// The scenario's model: whole, its outputs alone or every intermediate too; or, with
    /// <paramref name="front"/>, its front half alone — the halves and C's chain, every value an
    /// output — which <see cref="Back"/> finishes.
    /// </summary>
    internal static byte[] Scenario(Shapes shapes, bool exposeIntermediates, bool front = false)
    {
        var graph = new GraphProto { Name = "memory_reuse" };
        string?[] dims = shapes switch
        {
            Shapes.Computed => ["rows", "cols"],
            Shapes.Paired or Shapes.Tracked => ["2", "n", "m"],
            _ => [(2 * N).ToString(CultureInfo.InvariantCulture), M.ToString(CultureInfo.InvariantCulture)],
        };
        graph.Inputs.Add(FloatValue("A", dims));
        graph.Inputs.Add(FloatValue("B", dims));
        graph.Initializers.Add(Int64s("zero", 0));
        switch (shapes)
        {
            case Shapes.Computed:
                graph.Initializers.Add(Int64s("two", 2));
                graph.Nodes.Add(Node("shape_A", "Shape", "A", "rows", Int("start", 0), Int("end", 1)));
                graph.Nodes.Add(Node("half_rows", "Div", "rows two", "half"));
                graph.Nodes.Add(Node("slice_A", "Slice", "A zero half zero", "A_half"));
                graph.Nodes.Add(Node("slice_B", "Slice", "B half rows zero", "B_half"));
                break;
            case Shapes.Paired or Shapes.Tracked:
                graph.Initializers.Add(Int64s("one", 1));
                graph.Initializers.Add(Int64s("two", 2));
                graph.Nodes.Add(Node("slice_A", "Slice", "A zero one zero", "A_half"));
                graph.Nodes.Add(Node("slice_B", "Slice", "B one two zero", "B_half"));
                break;
            default:
                graph.Initializers.Add(Int64s("half", N));
                graph.Initializers.Add(Int64s("rows", 2 * N));
                graph.Nodes.Add(Node("slice_A", "Slice", "A zero half zero", "A_half"));
                graph.Nodes.Add(Node("slice_B", "Slice", "B half rows zero", "B_half"));
                break;
        }
        if (shapes == Shapes.Tracked)
        {
            graph.Initializers.Add(new TensorProto { Name = "zero_f", data_type = (int)TensorProto.DataType.Float, Dims = [], FloatDatas = [0f] });
            graph.Initializers.Add(new TensorProto { Name = "two_f", data_type = (int)TensorProto.DataType.Float, Dims = [], FloatDatas = [2f] });
            graph.Nodes.Add(Node("fill_mul", "Mul", "A_half zero_f", "c_zero"));
            graph.Nodes.Add(Node("fill_C", "Add", "c_zero two_f", "C0"));
        }
        else
        {
            graph.Nodes.Add(Node("shape_C", "Shape", "A_half", "c_shape"));
            graph.Nodes.Add(Node("fill_C", "ConstantOfShape", "c_shape", "C0", new AttributeProto
            {
                Name = "value", Type = AttributeProto.AttributeType.Tensor,
                T = new TensorProto { data_type = (int)TensorProto.DataType.Float, Dims = [1], FloatDatas = [2f] },
            }));
        }
        graph.Nodes.Add(Node("c_neg", "Neg", "C0", "C1"));
        graph.Nodes.Add(Node("c_abs", "Abs", "C1", "C2"));
        graph.Nodes.Add(Node("c_sigmoid", "Sigmoid", "C2", "C3"));
        if (front)
        {
            foreach (var name in (string[])["A_half", "B_half", "C0", "C1", "C2", "C3"])
                graph.Outputs.Add(FloatValue(name, null));
            return Serialized(graph);
        }
        graph.Nodes.Add(Node("concat_L", "Concat", "C3 B_half", "L0", Int("axis", 0)));
        AddLChain(graph);
        foreach (var name in exposeIntermediates ? [.. Outputs, .. Intermediates] : Outputs)
            graph.Outputs.Add(FloatValue(name, null));
        return Serialized(graph);
    }

    /// <summary>The scenario's back half: L's chain over a whole-shaped input L0 — which, run after
    /// <see cref="Scenario"/>'s front half wrote C into B's first half, is B itself, the
    /// concatenation made by placement alone — every value an output.</summary>
    internal static byte[] Back(Shapes shapes)
    {
        var graph = new GraphProto { Name = "memory_reuse_back" };
        graph.Inputs.Add(FloatValue("L0", Leading(shapes) ? ["2", "n", "m"] : shapes == Shapes.Static
            ? [(2 * N).ToString(CultureInfo.InvariantCulture), M.ToString(CultureInfo.InvariantCulture)]
            : ["rows", "cols"]));
        AddLChain(graph);
        foreach (var name in (string[])["L1", "L2", "L"])
            graph.Outputs.Add(FloatValue(name, null));
        return Serialized(graph);
    }

    private static void AddLChain(GraphProto graph)
    {
        graph.Nodes.Add(Node("l_sigmoid", "Sigmoid", "L0", "L1"));
        graph.Nodes.Add(Node("l_neg", "Neg", "L1", "L2"));
        graph.Nodes.Add(Node("l_abs", "Abs", "L2", "L"));
    }

    private static byte[] Serialized(GraphProto graph)
    {
        var model = new ModelProto { IrVersion = 10, Graph = graph, ProducerName = "memory-reuse-scenario" };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static TensorProto Int64s(string name, long value)
        => new() { Name = name, data_type = (int)TensorProto.DataType.Int64, Dims = [1], Int64Datas = [value] };

    /// <summary>A float value, of the shape <paramref name="dims"/> names — a number for a fixed
    /// dimension, a name for a symbolic one — or of no stated shape.</summary>
    private static ValueInfoProto FloatValue(string name, string?[]? dims)
    {
        var tensor = new TypeProto.Tensor { ElemType = (int)TensorProto.DataType.Float };
        if (dims is not null)
        {
            tensor.Shape = new TensorShapeProto();
            foreach (var dim in dims)
                tensor.Shape.Dims.Add(long.TryParse(dim, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                    ? new TensorShapeProto.Dimension { DimValue = value }
                    : new TensorShapeProto.Dimension { DimParam = dim! });
        }
        return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = tensor } };
    }

    private static AttributeProto Int(string name, long value)
        => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static NodeProto Node(string name, string op, string inputs, string outputs, params AttributeProto[] attributes)
    {
        var node = new NodeProto { Name = name, OpType = op, Domain = "" };
        node.Inputs.AddRange(inputs.Split(' '));
        node.Outputs.AddRange(outputs.Split(' '));
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] Pattern(bool b)
    {
        var values = new float[WholeCount];
        for (long i = 0; i < values.Length; i++)
            values[i] = b ? 0.5f - (i % 997) * 0.002f : (i % 1013) * 0.001f;
        return values;
    }

    // ---- the consumed inputs, in memory of the harness's own ----

    /// <summary>A and B, allocated once outside every allocator under test and filled again before
    /// each run, and the views a run binds over them.</summary>
    private sealed class Inputs : IDisposable
    {
        internal readonly bool OnCard;
        internal readonly IntPtr A;
        internal readonly IntPtr B;
        internal readonly float[] APattern = Pattern(b: false);
        internal readonly float[] BPattern = Pattern(b: true);
        private readonly OrtMemoryInfo? _cardInfo;

        /// <summary>The card's memory, as a binding names where ONNX Runtime is to leave an output;
        /// null on the host.</summary>
        internal OrtMemoryInfo? CardInfo => _cardInfo;

        internal Inputs(bool onCard)
        {
            OnCard = onCard;
            _cardInfo = onCard ? new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default) : null;
            A = Allocate();
            B = Allocate();
            Assert.NotEqual(IntPtr.Zero, A);
            Assert.NotEqual(IntPtr.Zero, B);
        }

        private IntPtr Allocate()
            => OnCard ? CudaRuntime.Allocate(0, WholeBytes) : Marshal.AllocHGlobal((nint)WholeBytes);

        internal void Refill()
        {
            Write(A, APattern);
            Write(B, BPattern);
        }

        private void Write(IntPtr to, float[] from)
        {
            if (OnCard) Assert.Equal(0, CudaInterop.CopyHostToDevice(MemoryMarshal.AsBytes(from.AsSpan()), to));
            else Marshal.Copy(from, 0, to, from.Length);
        }

        internal float[] Read(IntPtr from, long count)
        {
            var values = new float[count];
            if (OnCard) Assert.Equal(0, CudaInterop.CopyDeviceToHost(from, MemoryMarshal.AsBytes(values.AsSpan())));
            else Marshal.Copy(from, values, 0, values.Length);
            return values;
        }

        internal OrtValue View(IntPtr at, long[] shape)
            => OrtValue.CreateTensorValueWithData(
                _cardInfo ?? OrtMemoryInfo.DefaultInstance, OrtElementType.Float, shape, at,
                shape.Aggregate((long)sizeof(float), (a, d) => a * d));

        /// <summary>Where <paramref name="address"/> is, in the inputs' words.</summary>
        internal string Where(IntPtr address)
        {
            long offset;
            if ((offset = (long)address - (long)A) >= 0 && offset < WholeBytes) return $"A+{offset >> 20}MiB";
            if ((offset = (long)address - (long)B) >= 0 && offset < WholeBytes) return $"B+{offset >> 20}MiB";
            return "own";
        }

        public void Dispose()
        {
            if (OnCard)
            {
                CudaRuntime.Release(0, A);
                CudaRuntime.Release(0, B);
            }
            else
            {
                Marshal.FreeHGlobal(A);
                Marshal.FreeHGlobal(B);
            }
            _cardInfo?.Dispose();
        }

        /// <summary>Whether the three outputs, at those addresses, hold what the scenario computes
        /// from A and B as they were filled.</summary>
        internal bool Intact()
            => Read(A, WholeCount).AsSpan().SequenceEqual(APattern) && Read(B, WholeCount).AsSpan().SequenceEqual(BPattern);

        internal bool Correct(IntPtr l, IntPtr aHalf, IntPtr bHalf) => Mismatch(l, aHalf, bHalf) is null;

        /// <summary>The first value of the three outputs that is not what the scenario computes, in
        /// words, or null where every one is.</summary>
        internal string? Mismatch(IntPtr l, IntPtr aHalf, IntPtr bHalf)
        {
            var lValues = Read(l, WholeCount);
            var aValues = Read(aHalf, HalfCount);
            var bValues = Read(bHalf, HalfCount);
            var c = Sigmoid(Sigmoid(2f));
            for (long i = 0; i < HalfCount; i++)
            {
                if (aValues[i] != APattern[i]) return $"A_half[{i}] = {aValues[i]}, not {APattern[i]}";
                if (bValues[i] != BPattern[HalfCount + i]) return $"B_half[{i}] = {bValues[i]}, not {BPattern[HalfCount + i]}";
                if (MathF.Abs(lValues[i] - c) > 1e-4f) return $"L[{i}] = {lValues[i]}, not {c}";
                var expected = Sigmoid(BPattern[HalfCount + i]);
                if (MathF.Abs(lValues[HalfCount + i] - expected) > 1e-4f) return $"L[{HalfCount + i}] = {lValues[HalfCount + i]}, not {expected}";
            }
            return null;
        }
    }

    // ---- configurations ----

    private enum AllocatorKind { Arena, Shorokoo }

    private enum Approach { Plain, BindFinal, BindAll, Split, Shipped }

    private sealed record Config(bool Card, Shapes Shapes, AllocatorKind Allocator, bool Optimized, bool MemoryPattern, Approach Approach, bool Parallel = false)
    {
        public override string ToString()
            => $"{(Card ? "card" : "host")}/{Shapes}/{Allocator}/{(Optimized ? "opt-all" : "opt-off")}/{(MemoryPattern ? "pattern-on" : "pattern-off")}/{Approach}{(Parallel ? "/parallel" : "")}";
    }

    private static IEnumerable<Config> Configurations(bool card)
    {
        foreach (var shapes in AllShapes)
        {
            foreach (var approach in (Approach[])[Approach.Plain, Approach.BindFinal, Approach.BindAll, Approach.Split])
                foreach (var allocator in (AllocatorKind[])[AllocatorKind.Arena, AllocatorKind.Shorokoo])
                    foreach (var optimized in (bool[])[true, false])
                        foreach (var pattern in (bool[])[true, false])
                            yield return new Config(card, shapes, allocator, optimized, pattern, approach);
            yield return new Config(card, shapes, AllocatorKind.Shorokoo, true, true, Approach.Plain, Parallel: true);
            yield return new Config(card, shapes, AllocatorKind.Shorokoo, true, true, Approach.Shipped);
        }
    }

    /// <summary>Where each output a bound approach binds goes: into which input, from which byte,
    /// at what shape.</summary>
    private static (string Name, bool InA, long Offset, long[] Shape)[] Bindings(Approach approach, Shapes shapes)
    {
        var whole = WholeShape(shapes);
        var half = HalfShape(shapes);
        (string, bool, long, long[])[] finals =
        [
            ("L", false, 0, whole),
            ("A_half", true, 0, half),
            ("B_half", true, HalfBytes, half),
        ];
        if (approach == Approach.BindFinal) return finals;
        return
        [
            .. finals,
            ("C0", false, 0, half), ("C1", false, 0, half), ("C2", false, 0, half), ("C3", false, 0, half),
            ("L0", false, 0, whole), ("L1", false, 0, whole), ("L2", false, 0, whole),
        ];
    }

    // ---- what a run did ----

    /// <summary>What the allocators were asked and given back during one run, in order.</summary>
    private sealed class Trace
    {
        private readonly object _gate = new();
        private readonly List<(CachingAllocator.Event Event, long Ticks)> _events = [];
        private long _start;

        internal void Start()
        {
            _start = Stopwatch.GetTimestamp();
            CachingAllocator.Observer = e =>
            {
                var now = Stopwatch.GetTimestamp();
                lock (_gate) _events.Add((e, now));
            };
        }

        internal static void Stop() => CachingAllocator.Observer = null;

        /// <summary>Each event, with the microseconds from <see cref="Start"/> to it.</summary>
        internal List<(CachingAllocator.Event Event, double Micros)> Events
        {
            get
            {
                lock (_gate) return [.. _events.Select(e => (e.Event, (e.Ticks - _start) * 1e6 / Stopwatch.Frequency))];
            }
        }

        internal double MicrosSinceStart() => (Stopwatch.GetTimestamp() - _start) * 1e6 / Stopwatch.Frequency;
    }

    /// <summary>One run as it was observed, before its events are put to the kernels that made
    /// them.</summary>
    private sealed record RawRun(
        int Run,
        List<(CachingAllocator.Event Event, double Micros)> Events,
        double Micros,
        Dictionary<string, IntPtr> Outputs,
        bool Correct,
        Dictionary<string, string>? ArenaBefore,
        Dictionary<string, string>? ArenaAfter)
    {
        /// <summary>Whether A and B still held what they were filled with once the run was over,
        /// where that was asked.</summary>
        internal bool? InputsIntact { get; init; }

        /// <summary>The first output value that was wrong, in words.</summary>
        internal string? Mismatch { get; init; }
    }

    /// <summary>The kernels of one run as ONNX Runtime profiled them: microseconds from the run's
    /// start, and the run's length.</summary>
    private sealed record ProfiledRun(double Micros, List<(string Node, double Start, double End)> Nodes)
    {
        /// <summary>The kernel an event <paramref name="micros"/> into the harness's measure of
        /// the run happened in or after, the two clocks aligned on the middle of the slack between
        /// the harness's measure and the profile's.</summary>
        internal string? At(double micros, double runMicros, bool allocation)
        {
            if (Nodes.Count == 0) return null;
            var at = micros - Math.Max(0, (runMicros - Micros) / 2);
            if (allocation)
            {
                var within = Nodes.FirstOrDefault(n => at >= n.Start - 2 && at <= n.End + 2);
                if (within.Node is not null) return Short(within.Node);
                var nearest = Nodes.MinBy(n => Math.Min(Math.Abs(at - n.Start), Math.Abs(at - n.End)));
                return Short(nearest.Node) + "?";
            }
            var last = Nodes.LastOrDefault(n => n.Start <= at);
            return last.Node is null ? "before the kernels" : (at <= last.End ? "in " : "after ") + Short(last.Node);
        }

        private static string Short(string node) => node.Split(':')[0];
    }

    private sealed record RunRecord(
        int Run,
        int Requests,
        long RequestedBytes,
        int BigRequests,
        long PeakBytes,
        long HeldAtEndBytes,
        int FreshBlocks,
        List<string> Sequence,
        Dictionary<string, string> OutputPlaces,
        bool Correct,
        double Milliseconds,
        Dictionary<string, string>? ArenaBefore,
        Dictionary<string, string>? ArenaAfter,
        List<IntPtr> BigAddresses,
        List<long> BigSizes);

    private static RunRecord Summarize(RawRun raw, Inputs inputs, ProfiledRun? profile)
    {
        var (run, timed, micros, outputs, correct, arenaBefore, arenaAfter) = raw;
        var milliseconds = micros / 1000;
        string Kernel(double at, bool allocation) => profile?.At(at, micros, allocation) is { } node ? $" [{node}]" : "";
        var live = new Dictionary<IntPtr, (long Bytes, string Label)>();
        var sequence = new List<string>();
        List<IntPtr> bigAddresses = [];
        List<long> bigSizes = [];
        int requests = 0, big = 0, fresh = 0, small = 0;
        long requested = 0, inUse = 0, peak = 0;
        foreach (var (e, at) in timed)
        {
            if (e.Allocation)
            {
                requests++;
                requested += e.Requested;
                if (e.Fresh) fresh++;
                inUse += e.Requested;
                peak = Math.Max(peak, inUse);
                if (e.Requested >= Big)
                {
                    big++;
                    bigAddresses.Add(e.Address);
                    bigSizes.Add(e.Requested);
                    var label = $"#{big}";
                    live[e.Address] = (e.Requested, label);
                    var where = inputs.Where(e.Address);
                    sequence.Add($"+{Mib(e.Requested)} {label}{(e.Fresh ? " fresh" : " cached")}{(where == "own" ? "" : " at " + where)}{Kernel(at, true)}");
                }
                else
                {
                    small++;
                    live[e.Address] = (e.Requested, "small");
                }
            }
            else if (live.Remove(e.Address, out var block))
            {
                inUse -= block.Bytes;
                if (block.Label != "small") sequence.Add($"-{Mib(block.Bytes)} {block.Label}{Kernel(at, false)}");
            }
            else if (e.Requested >= Big)
                sequence.Add($"-{Mib(e.Requested)} block from before the run{Kernel(at, false)}");
        }
        if (small > 0) sequence.Add($"({small} requests under 1 MiB)");
        if (raw.InputsIntact is { } intact) sequence.Add($"(A and B {(intact ? "intact" : "written")} after the run)");
        if (raw.Mismatch is { } wrong) sequence.Add($"(wrong: {wrong})");
        var places = outputs.ToDictionary(o => o.Key, o =>
        {
            var where = inputs.Where(o.Value);
            if (where != "own") return where;
            var index = bigAddresses.LastIndexOf(o.Value);
            return index >= 0 ? $"#{index + 1}" : "own";
        });
        return new RunRecord(run, requests, requested, big, peak, inUse, fresh, sequence, places, correct,
            milliseconds, arenaBefore, arenaAfter, bigAddresses, bigSizes);
    }

    private static string Mib(long bytes) => (bytes / (double)(1 << 20)).ToString("0.##", CultureInfo.InvariantCulture) + "MiB";

    // ---- running a configuration ----

    private sealed record ConfigRecord(string Config, List<RunRecord> Runs, List<string> NodeOrder, string? Failure);

    private static void Record(bool card, string name)
    {
        var dir = OutputDirectory();
        _ = RuntimeAllocator.ForHost();
        if (card) _ = RuntimeAllocator.ForCard(0);
        foreach (var shapes in AllShapes)
        {
            File.WriteAllBytes(Path.Combine(dir, $"scenario-{shapes}.onnx"), Scenario(shapes, exposeIntermediates: false));
            File.WriteAllBytes(Path.Combine(dir, $"scenario-{shapes}-exposed.onnx"), Scenario(shapes, exposeIntermediates: true));
        }

        List<ConfigRecord> records = [];
        using (var inputs = new Inputs(card))
        {
            var only = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_ONLY");
            foreach (var config in Configurations(card))
            {
                if (only is not null && !config.ToString().Contains(only, StringComparison.Ordinal)) continue;
                try
                {
                    var model = Scenario(config.Shapes, exposeIntermediates: config.Approach == Approach.BindAll);
                    records.Add(config.Approach switch
                    {
                        Approach.Shipped => RunShipped(config, model),
                        Approach.Split => RunSplit(config, inputs, dir),
                        _ => RunConfig(config, model, inputs, dir),
                    });
                }
                catch (Exception failure)
                {
                    records.Add(new ConfigRecord(config.ToString(), [], [], failure.GetType().Name + ": " + failure.Message));
                }
            }
        }
        File.WriteAllText(Path.Combine(dir, $"memory-reuse-{name}.json"), JsonConvert.SerializeObject(records, Formatting.Indented));
        File.WriteAllText(Path.Combine(dir, $"memory-reuse-{name}.md"), Report(name, records));
        Assert.All(records, r => Assert.Null(r.Failure));
    }

    private static InferenceSession NewSession(byte[] model, Config config, string profilePrefix)
    {
        using var options = new SessionOptions();
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        options.GraphOptimizationLevel = config.Optimized ? GraphOptimizationLevel.ORT_ENABLE_ALL : GraphOptimizationLevel.ORT_DISABLE_ALL;
        options.EnableMemoryPattern = config.MemoryPattern;
        options.ExecutionMode = config.Parallel ? ExecutionMode.ORT_PARALLEL : ExecutionMode.ORT_SEQUENTIAL;
        if (config.Allocator == AllocatorKind.Shorokoo) options.AddSessionConfigEntry("session.use_env_allocators", "1");
        options.ProfileOutputPathPrefix = profilePrefix;
        options.EnableProfiling = true;
        if (config.Card) OrtBackend.AppendCuda(options, 0, DeviceMemorySettings.Default, PrecisionSettings.Default);
        return new InferenceSession(model, options);
    }

    private static Dictionary<string, string>? ArenaFigures(InferenceSession session, Config config)
    {
        if (config.Allocator != AllocatorKind.Arena) return null;
        using var cardInfo = config.Card ? new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default) : null;
        using var allocator = new OrtAllocator(session, cardInfo ?? OrtMemoryInfo.DefaultInstance);
        return OrtArenaStats.ReadRaw(allocator) is { } raw ? new Dictionary<string, string>(raw) : null;
    }

    private static ConfigRecord RunConfig(Config config, byte[] model, Inputs inputs, string dir)
    {
        var profilePrefix = Path.Combine(dir, "profile-" + config.ToString().Replace('/', '-'));
        CachingAllocator.Account? host = null, cardAccount = null;
        if (config.Allocator == AllocatorKind.Shorokoo)
        {
            host = CachingAllocator.ForHost().Open("scenario");
            cardAccount = config.Card ? CachingAllocator.ForCard(0).Open("scenario") : null;
        }
        InferenceSession session;
        using (CachingAllocator.Charge(host, cardAccount)) session = NewSession(model, config, profilePrefix);
        List<RawRun> raws = [];
        List<ProfiledRun> profiled;
        try
        {
            for (int run = 1; run <= Runs; run++)
            {
                raws.Add(RunOnce(session, config, inputs, run, host, cardAccount));
            }
        }
        finally
        {
            var profile = session.EndProfiling();
            session.Dispose();
            if (host is not null) host.Allocator.Close(host);
            if (cardAccount is not null) cardAccount.Allocator.Close(cardAccount);
            profiled = ProfiledRuns(profile);
            try { File.Delete(profile); } catch (IOException) { }
        }
        var runs = raws.Select((raw, i) => Summarize(raw, inputs, i < profiled.Count ? profiled[i] : null)).ToList();
        var order = profiled.Count > 0 ? profiled[^1].Nodes.Select(n => n.Node).ToList() : [];
        return new ConfigRecord(config.ToString(), runs, order, null);
    }

    private static RawRun RunOnce(
        InferenceSession session, Config config, Inputs inputs, int run,
        CachingAllocator.Account? host, CachingAllocator.Account? card)
    {
        inputs.Refill();
        using var a = inputs.View(inputs.A, WholeShape(config.Shapes));
        using var b = inputs.View(inputs.B, WholeShape(config.Shapes));
        using var runOptions = new RunOptions();
        // An output ONNX Runtime allocates is left on the card through a binding, as the backend
        // leaves one: a plain run would copy it to the host.
        var bound = config.Approach is Approach.BindFinal or Approach.BindAll;
        OrtIoBinding? binding = null;
        List<(string Name, OrtValue Value, IntPtr At)> views = [];
        if (bound || config.Card)
        {
            binding = session.CreateIoBinding();
            binding.BindInput("A", a);
            binding.BindInput("B", b);
        }
        if (bound)
        {
            foreach (var (name, inA, offset, shape) in Bindings(config.Approach, config.Shapes))
            {
                var at = (inA ? inputs.A : inputs.B) + (nint)offset;
                var view = inputs.View(at, shape);
                views.Add((name, view, at));
                binding!.BindOutput(name, view);
            }
        }
        else if (binding is not null)
            foreach (var name in Outputs) binding.BindOutputToDevice(name, inputs.CardInfo!);

        var arenaBefore = ArenaFigures(session, config);
        var trace = new Trace();
        IDisposableReadOnlyCollection<OrtValue>? fetched = null;
        double micros;
        using (CachingAllocator.Charge(host, card))
        {
            trace.Start();
            try
            {
                if (binding is null) fetched = session.Run(runOptions, ["A", "B"], [a, b], Outputs);
                else if (!bound) fetched = session.RunWithBoundResults(runOptions, binding);
                else
                {
                    session.RunWithBinding(runOptions, binding);
                    binding.SynchronizeBoundOutputs();
                }
            }
            finally
            {
                micros = trace.MicrosSinceStart();
                Trace.Stop();
            }
        }
        var arenaAfter = ArenaFigures(session, config);

        var addresses = new Dictionary<string, IntPtr>();
        if (fetched is not null)
            for (int i = 0; i < Outputs.Length; i++) addresses[Outputs[i]] = OrtBackend.AddressOf(fetched[i]);
        else
            foreach (var (name, _, at) in views.Where(v => Outputs.Contains(v.Name))) addresses[name] = at;
        var mismatch = inputs.Mismatch(addresses["L"], addresses["A_half"], addresses["B_half"]);
        var raw = new RawRun(run, trace.Events, micros, addresses, mismatch is null, arenaBefore, arenaAfter)
        {
            InputsIntact = config.Approach == Approach.Plain ? inputs.Intact() : null,
            Mismatch = mismatch,
        };

        fetched?.Dispose();
        foreach (var (_, view, _) in views) view.Dispose();
        binding?.Dispose();
        return raw;
    }

    /// <summary>
    /// The scenario split in two sessions, run one after the other: the front half writes A_half
    /// over A's first half, B_half into A's second, and C's chain into B's first half; the back half
    /// then runs L's chain over B — which holds the concatenation already, C in its first half and
    /// B_half's rows where they always were — writing each value back into B. Nothing is
    /// concatenated and nothing copied but B_half; the session boundary is what orders the back
    /// half's writes into B after the front half's reads of it.
    /// </summary>
    private static ConfigRecord RunSplit(Config config, Inputs inputs, string dir)
    {
        var prefix = Path.Combine(dir, "profile-" + config.ToString().Replace('/', '-'));
        CachingAllocator.Account? host = null, card = null;
        if (config.Allocator == AllocatorKind.Shorokoo)
        {
            host = CachingAllocator.ForHost().Open("scenario");
            card = config.Card ? CachingAllocator.ForCard(0).Open("scenario") : null;
        }
        InferenceSession front, back;
        using (CachingAllocator.Charge(host, card))
        {
            front = NewSession(Scenario(config.Shapes, exposeIntermediates: true, front: true), config, prefix + "-front");
            back = NewSession(Back(config.Shapes), config, prefix + "-back");
        }
        var whole = WholeShape(config.Shapes);
        var half = HalfShape(config.Shapes);
        List<RunRecord> runs = [];
        try
        {
            for (int run = 1; run <= Runs; run++)
            {
                inputs.Refill();
                using var a = inputs.View(inputs.A, whole);
                using var b = inputs.View(inputs.B, whole);
                using var l0 = inputs.View(inputs.B, whole);
                List<OrtValue> views = [];
                OrtValue View(IntPtr at, long[] shape)
                {
                    var view = inputs.View(at, shape);
                    views.Add(view);
                    return view;
                }
                using var first = front.CreateIoBinding();
                first.BindInput("A", a);
                first.BindInput("B", b);
                first.BindOutput("A_half", View(inputs.A, half));
                first.BindOutput("B_half", View(inputs.A + (nint)HalfBytes, half));
                foreach (var name in (string[])["C0", "C1", "C2", "C3"]) first.BindOutput(name, View(inputs.B, half));
                using var second = back.CreateIoBinding();
                second.BindInput("L0", l0);
                foreach (var name in (string[])["L1", "L2", "L"]) second.BindOutput(name, View(inputs.B, whole));

                var arenaBefore = Merged(ArenaFigures(front, config), ArenaFigures(back, config));
                using var runOptions = new RunOptions();
                var trace = new Trace();
                double micros;
                using (CachingAllocator.Charge(host, card))
                {
                    trace.Start();
                    try
                    {
                        front.RunWithBinding(runOptions, first);
                        first.SynchronizeBoundOutputs();
                        back.RunWithBinding(runOptions, second);
                        second.SynchronizeBoundOutputs();
                    }
                    finally
                    {
                        micros = trace.MicrosSinceStart();
                        Trace.Stop();
                    }
                }
                var arenaAfter = Merged(ArenaFigures(front, config), ArenaFigures(back, config));
                var addresses = new Dictionary<string, IntPtr>
                {
                    ["L"] = inputs.B, ["A_half"] = inputs.A, ["B_half"] = inputs.A + (nint)HalfBytes,
                };
                var mismatch = inputs.Mismatch(inputs.B, inputs.A, inputs.A + (nint)HalfBytes);
                runs.Add(Summarize(new RawRun(run, trace.Events, micros, addresses, mismatch is null, arenaBefore, arenaAfter) { Mismatch = mismatch }, inputs, null));
                foreach (var view in views) view.Dispose();
            }
        }
        finally
        {
            foreach (var session in (InferenceSession[])[front, back])
            {
                var profile = session.EndProfiling();
                session.Dispose();
                try { File.Delete(profile); } catch (IOException) { }
            }
            if (host is not null) host.Allocator.Close(host);
            if (card is not null) card.Allocator.Close(card);
        }
        return new ConfigRecord(config.ToString(), runs, [], null);
    }

    /// <summary>Two sessions' arena figures, added together.</summary>
    private static Dictionary<string, string>? Merged(Dictionary<string, string>? first, Dictionary<string, string>? second)
    {
        if (first is null || second is null) return first ?? second;
        return first.Keys.Intersect(second.Keys).ToDictionary(key => key,
            key => (Figure(first, key) + Figure(second, key)).ToString(CultureInfo.InvariantCulture));
    }

    /// <summary>The backend's own path: a session of the stock backend for the device, fed tensors
    /// it placed, run through <c>RunConsuming</c>, which releases them as it returns.</summary>
    private static ConfigRecord RunShipped(Config config, byte[] model)
    {
        OrtBackend backend = config.Card ? new CardBackend() : new HostBackend();
        using var session = backend.CreateSession(model, ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Error, DeviceMemorySettings.Default);
        var aBytes = MemoryMarshal.AsBytes(Pattern(b: false).AsSpan()).ToArray();
        var bBytes = MemoryMarshal.AsBytes(Pattern(b: true).AsSpan()).ToArray();
        List<RunRecord> runs = [];
        using var reader = new Inputs(config.Card);
        for (int run = 1; run <= Runs; run++)
        {
            var a = (OrtTensorValue)((IShorokooBackend)backend).CreateTensorInBackendMemory(ShorokooTensorElementType.Float, aBytes, WholeShape(config.Shapes));
            var b = (OrtTensorValue)((IShorokooBackend)backend).CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bBytes, WholeShape(config.Shapes));
            var aAt = OrtBackend.AddressOf(a.Inner);
            var bAt = OrtBackend.AddressOf(b.Inner);
            var trace = new Trace();
            trace.Start();
            IReadOnlyList<IShorokooTensorValue> results;
            double micros;
            try
            {
                results = session.RunConsuming(
                    new Dictionary<string, IShorokooTensorValue> { ["A"] = a, ["B"] = b }, [a, b], Outputs, RunSettings.Default);
            }
            finally
            {
                micros = trace.MicrosSinceStart();
                Trace.Stop();
            }
            var addresses = new Dictionary<string, IntPtr>();
            for (int i = 0; i < Outputs.Length; i++) addresses[Outputs[i]] = OrtBackend.AddressOf(((OrtTensorValue)results[i]).Inner);
            var correct = reader.Correct(addresses["L"], addresses["A_half"], addresses["B_half"]);
            var places = new ShippedPlaces(aAt, bAt);
            var record = Summarize(new RawRun(run, trace.Events, micros, addresses, correct, null, null), reader, null);
            record = record with
            {
                OutputPlaces = addresses.ToDictionary(o => o.Key, o => places.Where(o.Value) ?? record.OutputPlaces[o.Key]),
                Sequence = [.. record.Sequence, $"(consumed inputs: A at {aAt:X}, B at {bAt:X})"],
            };
            runs.Add(record);
            foreach (var value in results) ((IShorokooBackend)backend).Release(value);
        }
        return new ConfigRecord(config.ToString(), runs, [], null);
    }

    private sealed record ShippedPlaces(IntPtr A, IntPtr B)
    {
        internal string? Where(IntPtr address)
        {
            long offset;
            if ((offset = (long)address - (long)A) >= 0 && offset < WholeBytes) return $"consumed A+{offset >> 20}MiB";
            if ((offset = (long)address - (long)B) >= 0 && offset < WholeBytes) return $"consumed B+{offset >> 20}MiB";
            return null;
        }
    }

    private sealed class HostBackend : OrtBackend;

    private sealed class CardBackend() : OrtBackend(cudaDeviceId: 0);

    /// <summary>The kernels of the last run the profile at <paramref name="path"/> recorded, in the
    /// order they ran: node, operator, provider.</summary>
    private static List<ProfiledRun> ProfiledRuns(string path)
    {
        List<ProfiledRun> runs = [];
        if (!File.Exists(path)) return runs;
        var events = JArray.Parse(File.ReadAllText(path));
        var kernels = events
            .Where(e => (string?)e["cat"] == "Node" && ((string?)e["name"])?.EndsWith("_kernel_time", StringComparison.Ordinal) == true)
            .Select(e => (Ts: (long)e["ts"]!, Dur: (long)e["dur"]!,
                Name: $"{((string)e["name"]!).Replace("_kernel_time", "")}:{e["args"]?["op_name"]}@{((string?)e["args"]?["provider"])?.Replace("ExecutionProvider", "")}"))
            .OrderBy(k => k.Ts)
            .ToList();
        foreach (var run in events.Where(e => (string?)e["name"] == "model_run").Select(e => (Ts: (long)e["ts"]!, Dur: (long)e["dur"]!)).OrderBy(r => r.Ts))
            runs.Add(new ProfiledRun(run.Dur, [.. kernels
                .Where(k => k.Ts >= run.Ts && k.Ts <= run.Ts + run.Dur)
                .Select(k => (k.Name, (double)(k.Ts - run.Ts), (double)(k.Ts - run.Ts + k.Dur)))]));
        return runs;
    }

    // ---- the report ----

    private static string Report(string name, List<ConfigRecord> records)
    {
        var text = new StringBuilder();
        text.AppendLine(CultureInfo.InvariantCulture, $"# Memory reuse scenario on the {name}");
        text.AppendLine();
        text.AppendLine(CultureInfo.InvariantCulture, $"Inputs A and B are [{2 * N}, {M}] float, {Mib(WholeBytes)} each; halves {Mib(HalfBytes)}.");
        text.AppendLine("Figures are what ONNX Runtime asked of Shorokoo's allocator during the run (beyond the inputs, outputs included),");
        text.AppendLine("or for an arena session what its arena reports. Run 1 is the session's first run.");
        text.AppendLine();
        text.AppendLine("| configuration | run | requests >= 1 MiB | bytes asked | peak in use | fresh blocks | arena allocs / in use / max in use / reserved | L / A_half / B_half | correct |");
        text.AppendLine("|---|---|---|---|---|---|---|---|---|");
        foreach (var record in records)
        {
            if (record.Failure is not null)
            {
                text.AppendLine(CultureInfo.InvariantCulture, $"| {record.Config} | - | failed: {record.Failure.Replace('|', '/').Replace('\n', ' ')} | | | | | | |");
                continue;
            }
            foreach (var run in record.Runs)
            {
                var arena = run.ArenaAfter is { } after
                    ? $"{Delta(run.ArenaBefore, after, "NumAllocs")} / {Mib(Figure(after, "InUse"))} / {Mib(Figure(after, "MaxInUse"))} / {Mib(Figure(after, "TotalAllocated"))}"
                    : "";
                text.AppendLine(CultureInfo.InvariantCulture,
                    $"| {record.Config} | {run.Run} | {run.BigRequests} | {Mib(run.RequestedBytes)} | {Mib(run.PeakBytes)} | {run.FreshBlocks} | {arena} | {run.OutputPlaces["L"]} / {run.OutputPlaces["A_half"]} / {run.OutputPlaces["B_half"]} | {run.Correct} |");
            }
        }
        text.AppendLine();
        text.AppendLine("## Sequences");
        foreach (var record in records.Where(r => r.Failure is null))
        {
            text.AppendLine();
            text.AppendLine(CultureInfo.InvariantCulture, $"### {record.Config}");
            if (record.NodeOrder.Count > 0) text.AppendLine(CultureInfo.InvariantCulture, $"kernels: {string.Join(", ", record.NodeOrder)}");
            foreach (var run in record.Runs)
                text.AppendLine(CultureInfo.InvariantCulture, $"- run {run.Run} ({run.Milliseconds:0.0} ms): {string.Join(", ", run.Sequence)}");
        }
        return text.ToString();
    }

    private static long Figure(Dictionary<string, string> figures, string name)
        => figures.TryGetValue(name, out var value) ? long.Parse(value, CultureInfo.InvariantCulture) : -1;

    private static string Delta(Dictionary<string, string>? before, Dictionary<string, string> after, string name)
        => before is null ? Figure(after, name).ToString(CultureInfo.InvariantCulture)
            : (Figure(after, name) - Figure(before, name)).ToString(CultureInfo.InvariantCulture);
}
