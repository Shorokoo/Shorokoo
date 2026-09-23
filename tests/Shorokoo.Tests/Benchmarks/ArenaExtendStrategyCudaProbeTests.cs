using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory;
using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

// ---------------------------------------------------------------------------
// The model Shorokoo/Shorokoo#326 reported on, rebuilt from the specification in
// that issue: a decoder-only transformer, fp32, over a 50,257-token vocabulary
// with an untied language-model head.
//
// The issue gives a parameter count, and the count settles three things it does
// not spell out. 49,214,208 = 2·50257·384 + 6·(4·384² + 2·384·1536) exactly, so
// there are no biases, no learnable layer-norm scale or shift, and no learned
// positional table -- a parameter-free positional scheme, which this
// reconstruction leaves out rather than guessing at, since it costs a handful of
// elementwise ops on tensors the attention matrices dwarf.
// ---------------------------------------------------------------------------

internal static class DecoderOnlyTransformer
{
    internal const long Vocabulary = 50257;

    /// <summary>
    /// The step's own next-token loss over the tokens it was fed, so the rig takes a forwarding
    /// loss and needs no target of its own (Shorokoo/Shorokoo#331). A language model's targets are
    /// its inputs shifted by one; the shift wraps, so the last position of each row is scored
    /// against the first rather than against a token the batch does not carry.
    /// </summary>
    internal static Scalar<float32> Loss(Tensor<int64> tokens, long layers, long width, long heads)
    {
        var logits = Model(tokens, layers, width, heads);
        var length = tokens.DimTensor(1);
        var next = tokens.Slice(Vector(1L), length.Unsqueeze(), axes: Vector(1L))
            .Concat(1L, tokens.Slice(Vector(0L), Vector(1L), axes: Vector(1L)));
        return CrossEntropyLoss.Call(logits, next.Reshape([Scalar(-1L)]));
    }

    /// <summary>Token ids <c>[B, L]</c> to logits <c>[B·L, 50257]</c>. The flattening is on the
    /// small side of the head matmul, which is the shape #326 restructured its own head into.</summary>
    internal static Tensor<float32> Model(Tensor<int64> tokens, long layers, long width, long heads)
    {
        var h = EmbeddingHelpers.Embed(tokens, Vocabulary, width);
        for (long i = 0; i < layers; i++) h = Block(h, width, heads);
        h = LayerNorm.Call(1L, false, Scalar(1e-5f), h);
        var head = XavierUniform.Init([Scalar(width), Scalar(Vocabulary)]);
        return h.Reshape([Scalar(-1L), Scalar(width)]).MatMul(head);
    }

    /// <summary>Pre-layer-norm block, causal self-attention then a GELU FFN of width 4E, both
    /// residual -- <see cref="TransformerEncoderLayer"/>'s structure with the mask on and the
    /// affine and bias parameters off.</summary>
    private static Tensor<float32> Block(Tensor<float32> x, long width, long heads)
    {
        var attnIn = LayerNorm.Call(1L, false, Scalar(1e-5f), x);
        var h = x + MultiHeadAttention.Call(width, heads, false, true, attnIn, attnIn, attnIn);

        var ffIn = LayerNorm.Call(1L, false, Scalar(1e-5f), h);
        var w1 = XavierUniform.Init([Scalar(width), Scalar(width * 4)]);
        var w2 = XavierUniform.Init([Scalar(width * 4), Scalar(width)]);
        return h + ffIn.MatMul(w1).Gelu().MatMul(w2);
    }
}

/// <summary>#326's baseline: 6 layers, width 384, 6 heads of 64 — 49,214,208 parameters.</summary>
[Module]
public partial class ArenaProbeDecoderBaseline
{
    public static Scalar<float32> Inline(Tensor<int64> tokens)
        => DecoderOnlyTransformer.Loss(tokens, layers: 6, width: 384, heads: 6);
}

/// <summary>The variant #326 reports does not fit at batch 8: 12 layers, width 768, 12 heads of 64
/// — 162,129,408 parameters, 1.2% under the 164.1 M the issue names. Width 768 is what the issue's
/// own failure names, its allocation of 2,359,296 bytes being one [768, 768] fp32 projection.</summary>
[Module]
public partial class ArenaProbeDecoderLarge
{
    public static Scalar<float32> Inline(Tensor<int64> tokens)
        => DecoderOnlyTransformer.Loss(tokens, layers: 12, width: 768, heads: 12);
}

/// <summary>
/// What each <c>ArenaExtendStrategy</c> costs a real training step on a CUDA card
/// (<see href="https://github.com/Shorokoo/Shorokoo/issues/357">Shorokoo/Shorokoo#357</see>), which
/// <see cref="ArenaExtendStrategyProbeTests"/> could only answer for the host arena and for a
/// workload of four chained matmuls. The model is #326's own, rebuilt from its specification, and
/// the figures come from inside the framework — <see cref="ComputeContext.RunStats"/> reading the
/// session's own arena either side of every step — rather than from sampling the device, which
/// cannot tell an arena's blocks from anything else on the card.
///
/// <para>Manual: it measures the machine, takes minutes, and prints a table rather than gating
/// anything. Run with <c>dotnet test -p:ShorokooGpuTests=true --filter
/// "FullyQualifiedName~ArenaExtendStrategyCudaProbeTests" --logger
/// "console;verbosity=detailed"</c>. Expect it to hold most of the card while it runs.</para>
///
/// <para>Three figures per step, all of the training step's own session arena.
/// <c>PeakBytes</c> is the high-water mark of what the arena had handed out, which never falls;
/// <c>TotalAllocatedBytes</c> is what it has taken from the device, in use or not, and is the
/// figure #326 watched reach the whole card; the extension count is the blocks behind it. The
/// device-wide reading beside them is context, not the measurement — it carries the provider's own
/// context, the libraries, and whatever else is on the card.</para>
///
/// <para>What this prints, and what it decides, is recorded once — in
/// <c>Documentation/inference.md</c> under "Device memory (GPU backends)" — rather than also here,
/// so that a machine disagreeing with it has only one place to correct.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
[Collection(DeviceMemoryPeak.Name)]
public class ArenaExtendStrategyCudaProbeTests
{
    private const long Batch = 8;
    private const long Sequence = 1024;
    private const int Steps = 8;

    private readonly record struct StepReading(
        int Step, long PeakBytes, MemoryFigureKind PeakKind, long ArenaBytes, long Extensions, long DeviceUsedBytes);

    private static string Mib(long bytes) => $"{bytes / (1024.0 * 1024.0):N0} MiB";

    private static void Show(string leg, IReadOnlyList<StepReading> steps)
    {
        foreach (var s in steps)
            Console.WriteLine(
                $"{leg,-34} step {s.Step}  peak {Mib(s.PeakBytes),12} ({s.PeakKind,-10})  " +
                $"arena {Mib(s.ArenaBytes),12}  blocks {s.Extensions,4}  device {Mib(s.DeviceUsedBytes),12}");
    }

    /// <summary>
    /// Runs <paramref name="steps"/> training steps of <paramref name="model"/> under one arena
    /// configuration and hands back what the step's own arena did on each, or null with the reason
    /// where a step failed to allocate. The context is disposed either way, so the leg's own
    /// sessions go with it.
    ///
    /// <para>That is not the same as the card being clear. A tensor placed on a card comes out of
    /// an arena keyed on (device, settings) and held for the life of the process, so a leg that
    /// places one leaves that arena behind for every later leg, and each leg here names different
    /// settings. Nothing in this probe places one — the rig is fed host tensors and moves them
    /// itself — but the <c>device</c> column is a whole-card reading, so read it as the card
    /// during that leg rather than as the leg alone.</para>
    /// </summary>
    private static (List<StepReading> Steps, long Parameters, string? Failure) Probe(
        ComputationGraph model, ArenaExtendStrategy strategy, long? limitBytes, int steps = Steps)
    {
        var readings = new List<StepReading>();
        long parameters = 0;
        using var ctx = new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings { ArenaExtend = strategy, LimitBytes = limitBytes },
            Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
        };

        try
        {
            NamedModelParam[] sample =
            [
                new TensorDataModelParam("tokens", ModelParamType.InputParam,
                    TensorData([Batch, Sequence], new long[Batch * Sequence])),
            ];
            var rig = TrainingRig.FromScratch(
                model, ForwardingLoss.ComputationGraph,
                AdamWOptimizer.ComputationGraph, sample,
                new AdamWOptimizerHyperparameters { LearningRate = 0.0003f },
                runtimeContext: ctx);
            // Not an assertion: this method reports failures rather than throwing them, and one
            // raised here would be caught below and printed as a Failure like any other. The rig
            // is target-free because the loss forwards the model's own, which is the shape a
            // language model wants anyway, and a leg that lost that would say so in its own row.
            if (rig.HasTargets) return ([], 0, "the rig grew a target slot; the loss is not forwarding");

            var ckpt = rig.CreateInitialCheckpoint();
            foreach (var field in rig.TrainableParamStructDef.Fields)
                parameters += ((TensorData)ckpt.TrainableParams.Fields[field.Name]).Shape.Count;
            Console.WriteLine(
                $"built: {parameters:N0} parameters, {strategy}, " +
                $"cap {(limitBytes is { } cap ? Mib(cap) : "none")}, {steps} steps");

            var tokens = rig.InputDef.FromOrderedData(
                TensorData([Batch, Sequence], ArenaProbeTokens.Ids(Batch * Sequence)));

            long seen = 0;
            for (int step = 0; step < steps; step++)
            {
                ckpt = rig.TrainStep(ckpt, tokens.Shared());
                var stats = ctx.RunStats;
                var made = stats.RecentRuns.Skip((int)seen).ToArray();
                seen = stats.RunCount;
                if (made.Length == 0) continue;
                var last = made[^1];
                readings.Add(new StepReading(
                    step, stats.PeakBytes, last.PeakKind, last.Arena.TotalAllocatedBytes,
                    last.Arena.ArenaExtensionCount, DeviceMemory.Sample()?.UsedBytes ?? 0));
            }
        }
        catch (Exception ex)
        {
            return (readings, parameters, ex.ToString());
        }

        return (readings, parameters, null);
    }

    /// <summary>
    /// #326's batch-8 baseline under each strategy in turn, uncapped. The ratio between the two
    /// settled figures is what #357 asks for; the per-step columns are what step 1 does to each.
    /// </summary>
    [CudaFact]
    public void ProbeWhatEachArenaExtendStrategyCostsABatch8TrainingStepOnACard()
    {
        var model = ArenaProbeDecoderBaseline.ComputationGraph;

        var same = Probe(model, ArenaExtendStrategy.SameAsRequested, null);
        Console.WriteLine($"baseline parameters: {same.Parameters:N0}");
        Show("SameAsRequested, uncapped", same.Steps);
        Console.WriteLine($"SameAsRequested, uncapped: {same.Failure ?? "fits"}");
        Collect();

        var pow2 = Probe(model, ArenaExtendStrategy.NextPowerOfTwo, null);
        Show("NextPowerOfTwo, uncapped", pow2.Steps);
        Console.WriteLine($"NextPowerOfTwo, uncapped: {pow2.Failure ?? "fits"}");
        Collect();

        if (same.Steps.Count != 0 && pow2.Steps.Count != 0)
            Console.WriteLine(
                $"settled arena ratio NextPowerOfTwo/SameAsRequested: " +
                $"{(double)pow2.Steps[^1].ArenaBytes / same.Steps[^1].ArenaBytes:F3}");
    }

    /// <summary>
    /// The same pair with the arena capped just above what step 0 takes, which is where #326's
    /// clipping path would be: the region step 1 asks for does not fit under this cap. The cap is
    /// a constant rather than a reading, so this costs two rig builds and not three; it sits above
    /// the step-0 arena of either strategy and well below the settled one.
    /// </summary>
    [CudaFact]
    public void ProbeWhatACapNearTheStep0PeakDoesToEachStrategy()
    {
        var model = ArenaProbeDecoderBaseline.ComputationGraph;
        const long cap = 10L * 1024 * 1024 * 1024;
        Console.WriteLine($"cap: {Mib(cap)}");

        ArenaExtendStrategy[] strategies = [ArenaExtendStrategy.SameAsRequested, ArenaExtendStrategy.NextPowerOfTwo];
        foreach (var strategy in strategies)
        {
            var leg = Probe(model, strategy, cap, steps: 4);
            Show($"{strategy}, capped", leg.Steps);
            Console.WriteLine($"{strategy}, capped: {leg.Failure ?? "fits"}");
            Collect();
        }
    }

    /// <summary>
    /// The larger variant at batch 8, the configuration #326 reports it moved off. Whether either
    /// strategy fits it, and at what, is the question; the answer is in the user documentation
    /// with the rest.
    /// </summary>
    [CudaFact]
    public void ProbeWhetherExactSizeExtensionFitsTheLargerVariantAtBatch8()
    {
        var model = ArenaProbeDecoderLarge.ComputationGraph;

        var same = Probe(model, ArenaExtendStrategy.SameAsRequested, null, steps: 4);
        Console.WriteLine($"large parameters: {same.Parameters:N0}");
        Show("large SameAsRequested", same.Steps);
        Console.WriteLine($"large SameAsRequested: {same.Failure ?? "fits"}");
        Collect();

        var pow2 = Probe(model, ArenaExtendStrategy.NextPowerOfTwo, null, steps: 4);
        Show("large NextPowerOfTwo", pow2.Steps);
        Console.WriteLine($"large NextPowerOfTwo: {pow2.Failure ?? "fits"}");
        Collect();
    }

    private static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}

/// <summary>Feed data for the probe: token ids inside the vocabulary.</summary>
internal static class ArenaProbeTokens
{
    internal static long[] Ids(long count)
    {
        var ids = new long[count];
        for (long i = 0; i < count; i++) ids[i] = i * 7919 % DecoderOnlyTransformer.Vocabulary;
        return ids;
    }
}
