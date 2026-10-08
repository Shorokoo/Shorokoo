using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Modules.Layers;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.OnnxRuntime;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>One 4096-wide linear layer: under AdamW its weight and two moments are 192 MiB of
/// state, against a step whose every other tensor is a few batch rows or the weight's own
/// gradient.</summary>
[Module]
public partial class WideLinearModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => Linear.Model(Scalar(4096L), Scalar(true)).Call(x);
}

/// <summary>The gradient of the sum of a <c>[V, 4]</c> table's rows read at <c>ids</c>: each row
/// counts the ids that read it.</summary>
[Module]
public partial class GatheredTableGradientModel
{
    public static Tensor<float32> Inline(Tensor<float32> table, Tensor<int64> ids)
        => Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(table, table.Gather(ids).Reduce(ReduceKind.Sum, keepDims: false).Scalar());
}

/// <summary>One pre-LayerNorm transformer encoder layer, 64 wide with four heads: its gradients
/// reduce over every row of the batch, which is where the card's kernels add up in whatever order
/// its threads finish.</summary>
[Module]
public partial class AttentionLayerModel
{
    public static Tensor<float32> Inline(Tensor<float32> x) => TransformerEncoderLayer.Call(64L, 4L, 256L, true, x);
}

/// <summary>Sixty-four 64-by-64 weights, each its own parameter of one shape, so that initializing
/// the model runs one draw session sixty-four times over.</summary>
[Module]
public partial class SquareStackModel
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
#pragma warning disable MSG007 // separate trace-order parameters are the shape under test
        for (int i = 0; i < 64; i++)
            x = x.MatMul(Shorokoo.Modules.Initializers.XavierUniform.Init([Scalar(64L), Scalar(64L)]));
#pragma warning restore MSG007
        return x;
    }
}

/// <summary>
/// What the CUDA execution provider actually does, run rather than read: a graph on the card, the
/// device-memory budget and the shrinking arena a resident training loop needs, and the memory
/// statistics and node placement of
/// <see href="https://github.com/Shorokoo/Shorokoo/issues/377">Shorokoo/Shorokoo#377</see>, every
/// CUDA path of which had been reviewed on a machine with no card.
///
/// <para>Excluded from the coverage suite. These need the process-wide default backend to be a
/// GPU one, which the test project deploys under <c>-p:ShorokooGpuTests=true</c>; without it
/// <see cref="CudaFactAttribute"/> skips each of them and says so rather than failing on a box
/// with no card. Run with <c>--filter "Purpose=Hardware"</c>, and with that switch and nothing
/// else — it changes the backend every test in the process discovers.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
[Collection(ProcessWideMemory.Name)]
public class GpuExecutionTests
{
    /// <summary>
    /// Explicitly exercises the CUDA execution provider.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AddTwoFloat32Scalars_ReturnsCorrectSum()
    {
        // [CudaFact] already gates on a GPU backend being loaded; confirm the
        // bound factory really is a CUDA one rather than a silent CPU fallback.
        var backend = DefaultBackend.Instance.GetType().Assembly.GetName().Name ?? "";
        Assert.EndsWith("GPU", backend);

        var result = AddTwoScalars(new ComputeContext(), 2.0f, 3.0f);
        Assert.Equal(5.0f, result);
    }

    /// <summary>
    /// Two resident runs of an attention layer from one seed and one batch, on contexts that ask for
    /// deterministic compute, compute bit for bit the same losses and weights — as they do not, from
    /// the first steps on, without it.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TwoRunsFromOneSeedAreBitIdenticalUnderDeterministicCompute()
    {
        var x = TensorData([4L, 256L, 64L], [.. Enumerable.Range(0, 4 * 256 * 64).Select(i => MathF.Sin(0.37f * i))]);
        var y = TensorData([4L, 256L, 64L], [.. Enumerable.Range(0, 4 * 256 * 64).Select(i => MathF.Cos(0.11f * i))]);
        float[] Train()
        {
            var rig = TrainingRig.FromScratch(
                AttentionLayerModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph, [x],
                new AdamWOptimizerHyperparameters { LearningRate = 1e-3f },
                runtimeContext: new ComputeContext { Diagnostics = new DiagnosticSettings { DeterministicCompute = true } },
                rngConfig: new RngConfig { MasterSeed = 7 });
            using var run = rig.BeginResidentRun();
            float[] losses = [.. Enumerable.Range(0, 5).Select(_ => run.Step(rig.InputDef.FromOrderedData(x).Shared(), rig.TargetDef.FromOrderedData(y).Shared()))];
            return [.. losses, .. Weights(run.StepToCheckpoint(rig.InputDef.FromOrderedData(x).Shared(), rig.TargetDef.FromOrderedData(y).Shared()))];
        }

        Assert.Equal(Train().Select(BitConverter.SingleToInt32Bits), Train().Select(BitConverter.SingleToInt32Bits));
    }

    [CudaFact]
    public void CudaProvider_RigsFromOneSeedStartFromTheSameWeights()
    {
        var x = TensorData([2L, 64L], new float[128]);
        float[] Initial() => Weights(TrainingRig.FromScratch(
            SquareStackModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph, [x],
            new AdamWOptimizerHyperparameters { LearningRate = 1e-3f },
            runtimeContext: new ComputeContext(), rngConfig: new RngConfig { MasterSeed = 7 }).CreateInitialCheckpoint());

        var first = Initial();
        for (int i = 0; i < 4; i++) Assert.Equal(first, Initial());
    }

    /// <summary>
    /// The device-memory surface end to end, on both settings of every knob: the shipped
    /// defaults build and run a session, and so does a budgeted power-of-two arena with per-run
    /// shrinkage on, with the card reporting a reading either way. Each configuration is a
    /// context of its own, so the two legs cannot reach each other and neither leaves anything
    /// behind for the next test.
    /// </summary>
    [CudaFact]
    public void CudaProvider_RunsUnderEveryDeviceMemoryConfigurationAndReportsTheCardsUsage()
    {
        DeviceMemory.ResetPeak();
        try
        {
            Assert.Equal(5.0f, AddTwoScalars(new ComputeContext(), 2.0f, 3.0f));

            var budgeted = new ComputeContext
            {
                DeviceMemory = new DeviceMemorySettings
                {
                    LimitBytes = 2L * 1024 * 1024 * 1024,
                },
                RunSettings = new RunSettings { ShrinkArenaAfterRun = true },
            };
            Assert.Equal(5.0f, AddTwoScalars(budgeted, 2.0f, 3.0f));

            var reading = DeviceMemory.Sample();
            Assert.NotNull(reading);
            Assert.True(reading!.Value.TotalBytes > 0);
            Assert.True(reading.Value.UsedBytes > 0);
            Assert.Equal(reading.Value.UsedBytes, DeviceMemory.PeakUsedBytes);
        }
        finally
        {
            DeviceMemory.ResetPeak();
        }
    }

    /// <summary>
    /// A transfer onto the card is refused before it allocates when the context's budget cannot
    /// take it alongside what is attached there — a copy, an allocation, and a tensor already on
    /// the card handed over as it stands — naming the budget, what is attached and what was asked;
    /// and a run fed more than the arena the budget leaves it is refused having taken nothing. The
    /// same tensor fits a context with no budget, so what refused it was the budget and not the
    /// card; and what the context lets go of is room it has again.
    /// </summary>
    [CudaFact]
    public void CudaProvider_ATransferPastItsContextsBudgetIsRefusedBeforeItAllocatesOnTheCard()
    {
        const long MiB = 1024 * 1024;
        using var budgeted = new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings { LimitBytes = 64 * MiB },
        };
        using var uncapped = new ComputeContext();

        var held = TensorData([12L << 20], new float[12 << 20]).CopyTo(budgeted);
        Assert.False(held.IsHostResident);
        Assert.Equal(new DeviceMemoryUse(48 * MiB, 1, 64 * MiB), budgeted.ReadDeviceMemoryUse());

        var tooBig = TensorData([8L << 20], new float[8 << 20]);
        var refused = Assert.Throws<InvalidOperationException>(() => tooBig.CopyTo(budgeted));
        Assert.Contains("asks this compute context for 33554432 bytes of CUDA device 0 memory", refused.Message);
        Assert.Contains("is 67108864 bytes, and 50331648 bytes of it are attached", refused.Message);
        Assert.Throws<InvalidOperationException>(() => tooBig.To(budgeted));
        var onTheCard = tooBig.CopyTo(uncapped);
        Assert.False(onTheCard.IsHostResident);
        Assert.Throws<InvalidOperationException>(() => onTheCard.To(budgeted));

        var x = InputVector<float32>();
        var doubled = budgeted.Compile(new InternalComputationGraph([x], [x + x]));
        Assert.Throws<InvalidOperationException>(() => doubled.Execute(tooBig));
        Assert.False(tooBig.IsDisposed);
        Assert.Equal(new DeviceMemoryUse(48 * MiB, 1, 64 * MiB), budgeted.ReadDeviceMemoryUse());

        held.Delete();
        Assert.Same(onTheCard, onTheCard.To(budgeted));
        Assert.Equal(new DeviceMemoryUse(32 * MiB, 1, 64 * MiB), budgeted.ReadDeviceMemoryUse());
    }

    [CudaFact]
    public void CudaProvider_TheTwoHalvesScenarioRunsWithNothingAllocatedBeyondWhatItConsumesAndItsOutputsOutliveTheSession()
    {
        const int Rows = 512, Columns = 1024;
        var (a, b, l) = ComputeContextLifetimeCoverageTests.TwoHalvesValues(Rows, Columns);
        NamedModelParam[] outputs = [];
        static float[] Read(NamedModelParam p) => [.. p.ToTensorData().ToHost().As<float32>().AccessMemory<float>()];
        using (var context = new ComputeContext())
        {
            var compiled = context.Compile(ComputeContextLifetimeCoverageTests.TwoHalves());
            for (int run = 0; run < 3; run++)
            {
                outputs = compiled.Execute(TensorData([(long)Rows, Columns], a).CopyTo(context), TensorData([(long)Rows, Columns], b).CopyTo(context));
                Assert.True(l.Zip(Read(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
                Assert.Equal(a[..(Rows / 2 * Columns)], Read(outputs[1]));
                Assert.Equal(b[(Rows / 2 * Columns)..], Read(outputs[2]));
            }
            var entry = Assert.Single(((OrtSession)compiled.Session).Placements!.Entries);
            Assert.Equal(OrtPlacements.Stage.Adopted, entry.Stage);
            Assert.True(entry.PlacedPeak < 1L << 20);
            Assert.All(outputs, o => Assert.False(o.ToTensorData().IsHostResident));
            Assert.Same(outputs[1].ToTensorData().Block, outputs[2].ToTensorData().Block);
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.True(l.Zip(Read(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        Assert.Equal(b[(Rows / 2 * Columns)..], Read(outputs[2]));
    }

    [CudaFact]
    public void CudaProvider_ARunThatFitsItsBudgetOnlyWhenPlacedSucceedsOnItsFirstCall()
    {
        const int Rows = 512, Columns = 1024;
        const long MiB = 1024 * 1024;
        var (a, b, l) = ComputeContextLifetimeCoverageTests.TwoHalvesValues(Rows, Columns);
        static float[] Read(NamedModelParam p) => [.. p.ToTensorData().ToHost().As<float32>().AccessMemory<float>()];
        NamedModelParam[] Run(ComputeContext on)
            => on.Compile(ComputeContextLifetimeCoverageTests.TwoHalves())
                .Execute(TensorData([(long)Rows, Columns], a).CopyTo(on), TensorData([(long)Rows, Columns], b).CopyTo(on));
        using var unplaced = new ComputeContext { ValuePlacement = false, DeviceMemory = new DeviceMemorySettings { LimitBytes = 6 * MiB } };
        using var context = new ComputeContext { DeviceMemory = new DeviceMemorySettings { LimitBytes = 6 * MiB } };

        Assert.ThrowsAny<Exception>(() => Run(unplaced));
        var outputs = Run(context);
        Assert.True(l.Zip(Read(outputs[0]), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        Assert.Equal(a[..(Rows / 2 * Columns)], Read(outputs[1]));
        Assert.Equal(b[(Rows / 2 * Columns)..], Read(outputs[2]));
        Assert.All(outputs, o => Assert.NotNull(o.ToTensorData().Block));
    }

    [CudaFact]
    public void CudaProvider_ARunRefusedItsMemoryAfterTrainingAndLoadingModelsLeavesTheCardUsable()
    {
        for (int i = 0; i < 3; i++)
        {
            CudaProvider_AModelLoadedCompiledReadsItsWeightsOntoTheCardAndRunsAsTheLoadedGraphDoes();
            CudaProvider_AModelOverSixteenMebibytesPlacesARunsValuesWhereThatPays();
            CudaProvider_ACopyTheCudaRuntimeFailsIsAnErrorRatherThanADeclinedRange();
            CudaProvider_ASessionsLimitCapsWhatItAllocatesAndNotAnInputReadWhereItIs();
            CudaProvider_ASessionRunOverManyShapesKeepsNoMoreOnTheCardThanItsBusiestRunHadInUse();
            CudaProvider_AShrinkingRunHandsBlocksBackAndEndsBelowThePeakItReached();
            CudaProvider_TwoRunsFromOneSeedAreBitIdenticalUnderDeterministicCompute();
            CudaProvider_ARunThatFitsItsBudgetOnlyWhenPlacedSucceedsOnItsFirstCall();
        }
    }

    [CudaFact]
    public void CudaProvider_OutputsOnOneBlockOfASessionsMemoryEachFreeTheirOwnPagesAndWhatNoneStandsOnGoesWithTheRun()
    {
        const long MiB = 1024 * 1024;
        using var context = new ComputeContext { DeviceMemory = new DeviceMemorySettings { LimitBytes = 512 * MiB } };
        var (both, together) = ComputeContextLifetimeCoverageTests.OutputsOnBlocksEnding(context, ComputeContextLifetimeCoverageTests.TwoHalves(), 4096, 1024, firstPage: 0);
        var (one, _) = ComputeContextLifetimeCoverageTests.OutputsOnBlocksEnding(context, ComputeContextLifetimeCoverageTests.TwoHalves(oneHalf: true), 4096, 1024, firstPage: 0);
        Assert.Equal(2, together);
        Assert.True(one[0].OnBlocks > 0);
        Assert.All(both, stage => Assert.Equal((stage.OnBlocks, stage.OnBlocks), (stage.InUse - both[^1].InUse, stage.Books)));
        Assert.All(one, stage => Assert.Equal((stage.OnBlocks, stage.OnBlocks), (stage.InUse - one[^1].InUse, stage.Books)));
    }

    [CudaFact]
    public void CudaProvider_AModelOverSixteenMebibytesPlacesARunsValuesWhereThatPays()
    {
        using var context = new ComputeContext();
        var (output, session, expected) = ComputeContextLifetimeCoverageTests.LargeModelRun(context, 9 << 19);
        Assert.Equal(OrtPlacements.Stage.Adopted, Assert.Single(Assert.IsType<OrtPlacements>(session.Placements).Entries).Stage);
        Assert.NotNull(output.Block);
        Assert.Equal(expected, [.. output.ToHost().As<float32>().AccessMemory<float>()]);
    }

    [CudaFact]
    public void CudaProvider_TheSessionAModelOverSixteenMebibytesPlacesThroughHoldsNoCopyOfTheWeightsItCarries()
    {
        using var context = new ComputeContext();
        var (_, session, _) = ComputeContextLifetimeCoverageTests.LargeModelRun(context, 9 << 19);
        var entry = Assert.Single(Assert.IsType<OrtPlacements>(session.Placements).Entries);
        Assert.Equal(OrtPlacements.Stage.Adopted, entry.Stage);
        Assert.True(entry.VariantHeld < 1L << 20);
        Assert.True(session.HeldBytes < (18L + 1) << 20);
    }

    [CudaFact]
    public void CudaProvider_ASessionRunsEachProvidersNodesInTheOrderOfTheGraphItWritesOutBuiltFromAModelOrFromAWrittenGraph()
    {
        foreach (var orders in ((string[])["encoder2", "attn-chunk4"]).Select(ComputeContextLifetimeCoverageTests.RunOrders))
        {
            Assert.NotEmpty(orders);
            Assert.All(orders, order => Assert.Equal(order.Written, order.Ran));
        }
    }

    [CudaFact]
    public void CudaProvider_ARunPlacesAValueOverAnInputABranchItNeedNotFollowReadsWhereTheSessionRunsThatBranchFirst()
    {
        var (entry, output, expected) = ComputeContextLifetimeCoverageTests.BranchesRun(DefaultBackend.Instance);
        Assert.Equal(OrtPlacements.Stage.Adopted, entry.Stage);
        Assert.Contains("a", entry.Plan.Select(p => p.Value));
        Assert.Equal(expected, output);
    }

    /// <summary>
    /// <paramref name="graph"/> over <paramref name="inputs"/>, each <c>float[N, N]</c>, run on the
    /// card with the inputs shared and with them consumed: the output each time, and the provider
    /// each operator ran on in the shared run.
    /// </summary>
    private static (byte[] Shared, byte[] Consumed, ILookup<string, string> Providers) SharedAndConsumed(GraphProto graph, params string[] inputs)
    {
        const int N = 1024;
        var backend = DefaultBackend.Instance;
        var model = ComputeContextLifetimeCoverageTests.ModelOf(graph);
        byte[] x = [.. MemoryMarshal.AsBytes(Enumerable.Range(0, N * N).Select(i => (i % 13) * 0.25f - 1.5f).ToArray().AsSpan())];
        (byte[], ILookup<string, string>?) Run(bool consume)
        {
            using var session = backend.CreateSession(model, ShorokooGraphOptimization.EnableAll, LogSettings.None, new DeviceMemorySettings(),
                new DiagnosticSettings { TraceNodePlacement = !consume });
            var fed = inputs.ToDictionary(name => name, _ => backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, x, [N, N]));
            var feeds = fed.ToDictionary(f => f.Key, f => f.Value);
            using var output = consume
                ? session.RunConsuming(feeds, [.. fed.Values], [graph.Outputs[0].Name], RunSettings.Default, out _).Single()
                : session.Run(feeds, [graph.Outputs[0].Name], RunSettings.Default).Single();
            if (!consume) foreach (var value in fed.Values) value.Dispose();
            return (backend.CopyTensorToHost(output), consume ? null : session.ReadNodePlacement()!.Nodes.ToLookup(n => n.OpType, n => n.Provider));
        }

        var (shared, providers) = Run(consume: false);
        return (shared, Run(consume: true).Item1, providers!);
    }

    [CudaFact]
    public void CudaProvider_AGatheredTablesGradientSumsRepeatedIdsOnTheCardWithoutAWarning()
    {
        long[] ids = [5, 9, 5, 5, 0, 9, 5, 63];
        System.Collections.Concurrent.ConcurrentQueue<RuntimeLogMessage> logged = [];
        var warnings = new RunSettings { Log = new LogSettings { Sink = logged.Enqueue } };
        var rig = TrainingRig.FromScratch(NNGatheredTableModel.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [TensorData([ids.Length], ids)], new SGDOptimizerHyperparameters { LearningRate = 1f }, runtimeContext: new ComputeContext { RunSettings = warnings });
        var initial = rig.CreateInitialCheckpoint();
        var before = Weights(initial);
        var after = Weights(rig.TrainStep(initial, rig.InputDef.FromOrderedData(TensorData([ids.Length], ids)).Shared(),
            rig.TargetDef.FromOrderedData(TensorData([ids.Length, 4L], new float[ids.Length * 4])).Shared()));

        using var traced = new ComputeContext { Diagnostics = new DiagnosticSettings { TraceNodePlacement = true }, RunSettings = warnings };
        var table = TensorData([64L, 4L], new float[256]);
        var read = TensorData([ids.Length], ids);
        using var compiled = traced.Compile(GatheredTableGradientModel.ComputationGraph.ToConcreteArchitecture([table, read]).ToConcreteModel());
        var gradient = compiled.Execute(table, read)[0].ToTensorData().CopyMemory<float>();
        var placement = compiled.ReadNodePlacement()!;

        Assert.Empty(logged);
        for (int i = 0; i < before.Length; i++)
        {
            int reads = ids.Count(id => id == i / 4);
            Assert.Equal(before[i] * (1f - reads / (2f * ids.Length)), after[i], 1e-5f);
            Assert.Equal(reads, gradient[i]);
        }
        Assert.All(placement.Nodes.Where(n => n.OpType is "ScatterElements" or "ScatterND"), n => Assert.Equal("CUDAExecutionProvider", n.Provider));
        Assert.Contains(placement.Nodes, n => n.OpType == "ScatterElements");
    }

    [CudaFact]
    public void CudaProvider_ARandomDrawsBitwiseOperatorsRunOnTheHost()
    {
        using var traced = new ComputeContext { Diagnostics = new DiagnosticSettings { TraceNodePlacement = true } };
        var (t, cond) = (TensorData([8L], new float[8]), TensorData(DType.Bool, [], true));
        using var compiled = traced.Compile(DropoutSquaredInOneIfArmModel.ComputationGraph.ToConcreteArchitecture([t, cond]).ToConcreteModel());
        compiled.Execute(t.Shared(), cond.Shared());
        Assert.Contains("BitShift", compiled.ReadNodePlacement()!.NodesOn("CPUExecutionProvider").Select(n => n.OpType));
    }

    [CudaFact]
    public void CudaProvider_ARunWithANodeTheCardHasNoKernelForComputesWhatItComputesUnplaced()
    {
        var (shared, consumed, providers) = SharedAndConsumed(ComputeContextLifetimeCoverageTests.GraphOf("x:float[1024,1024]", "z:float[1024,1024]",
            ComputeContextLifetimeCoverageTests.Op("Relu", "x", "r"), ComputeContextLifetimeCoverageTests.Op("Hardmax", "r", "y"),
            ComputeContextLifetimeCoverageTests.Op("Neg", "y", "z")), "x");
        Assert.Equal(["CPUExecutionProvider"], providers["Hardmax"]);
        Assert.Equal(shared, consumed);
    }

    [CudaFact]
    public void CudaProvider_ARunWhoseHostNodeReadsAConsumedInputComputesWhatItComputesUnplaced()
    {
        var (shared, consumed, providers) = SharedAndConsumed(ComputeContextLifetimeCoverageTests.GraphOf("x:float[1024,1024] w:float[1024,1024]", "z:float[1024,1024]",
            ComputeContextLifetimeCoverageTests.Op("Relu", "w", "t"), ComputeContextLifetimeCoverageTests.Op("Hardmax", "x", "y"),
            ComputeContextLifetimeCoverageTests.Op("Neg", "y", "n"), ComputeContextLifetimeCoverageTests.Op("Add", "n t", "z")), "x", "w");
        Assert.Equal(["CPUExecutionProvider"], providers["Hardmax"]);
        Assert.Equal(shared, consumed);
    }

    [CudaFact]
    public void CudaProvider_ABudgetCountsABlockOnceForAsLongAsAnyTensorOnItIsAttached()
    {
        const long MiB = 1024 * 1024;
        using var budgeted = new ComputeContext { DeviceMemory = new DeviceMemorySettings { LimitBytes = 64 * MiB } };
        var backend = DefaultBackend.Instance;
        var owner = (OrtTensorValue)backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, new byte[16 * MiB], [4L << 20]);
        var block = new SharedBlock(16 * MiB, () => backend.Release(owner));
        var first = TensorData.Create((long[])[2L << 20], DType.Float32, OrtBackend.View(owner, 0, ShorokooTensorElementType.Float, [2L << 20], 8 * MiB, block, 0), backend).To(budgeted);
        var second = TensorData.Create((long[])[1L << 20], DType.Float32, OrtBackend.View(owner, 8 * MiB, ShorokooTensorElementType.Float, [1L << 20], 4 * MiB, block, 8 * MiB), backend).To(budgeted);
        Assert.Equal(new DeviceMemoryUse(16 * MiB, 2, 64 * MiB), budgeted.ReadDeviceMemoryUse());
        first.Delete();
        Assert.Equal(new DeviceMemoryUse(16 * MiB, 1, 64 * MiB), budgeted.ReadDeviceMemoryUse());
        second.Delete();
        Assert.Equal(new DeviceMemoryUse(0, 0, 64 * MiB), budgeted.ReadDeviceMemoryUse());
        Assert.True(block.IsReleased);
    }

    [CudaFact]
    public void CudaProvider_APlacedRunComputesInItsContextsPrecision()
    {
        var host = ComputeContextLifetimeCoverageTests.ProductIntoConsumedOnTheHost();
        using var strictContext = new ComputeContext();
        using var allowedContext = new ComputeContext { Precision = SideBySideModel.AllowingTensorFloat32 };
        var strict = ComputeContextLifetimeCoverageTests.RunProductIntoConsumed(strictContext);
        var allowed = ComputeContextLifetimeCoverageTests.RunProductIntoConsumed(allowedContext);

        Assert.Equal((true, true), (strict.Placed, allowed.Placed));
        SideBySideModel.AssertFullPrecision([host], [strict.Values]);
        SideBySideModel.AssertTensorFloat32([host], [allowed.Values]);
    }

    [CudaFact]
    public void CudaProvider_ARunFedTwoTensorsStandingOnOneBlockCountsTheBlockOnce()
    {
        const long MiB = 1024 * 1024;
        var backend = DefaultBackend.Instance;
        var owner = (OrtTensorValue)backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, new byte[16 * MiB], [4L << 20]);
        var block = new SharedBlock(16 * MiB, () => backend.Release(owner));
        TensorData On(long offset) => TensorData.Create((long[])[1L << 20], DType.Float32, OrtBackend.View(owner, offset, ShorokooTensorElementType.Float, [1L << 20], 4 * MiB, block, offset), backend);
        var (first, second) = (On(0), On(8 * MiB));
        using var budgeted = new ComputeContext { DeviceMemory = new DeviceMemorySettings { LimitBytes = 28 * MiB } };
        var a = InputTensor<float32>("A", rank: 1);
        var b = InputTensor<float32>("B", rank: 1);
        var sum = budgeted.Compile(new InternalComputationGraph([a, b], [OnnxOp.Add(a, b)])).Execute(first.Shared(), second.Shared());
        Assert.Equal(new DeviceMemoryUse(20 * MiB, 3, 28 * MiB), budgeted.ReadDeviceMemoryUse());
        ComputeContext.ReleaseOutputs(sum);
        first.Delete();
        second.Delete();
        Assert.True(block.IsReleased);
    }

    /// <summary>
    /// What a run's session may allocate is capped at exactly the budget less what the context holds
    /// on the card for the run — here the eight-byte copy of the shape it is fed: with nothing else
    /// held, a run whose session needs 160 MiB fits a 256 MiB budget; with 100 MiB held on the
    /// card the same session is limited to the room that leaves, and the same run fails with an
    /// allocation failure as the allocator refuses the block that would pass it. Once the tensor is
    /// gone the session gets the room back, and is never built again.
    /// </summary>
    [CudaFact]
    public void CudaProvider_WhatARunsSessionAllocatesIsCappedAtItsContextsBudgetLessWhatTheContextHoldsOnTheCard()
    {
        const long MiB = 1024 * 1024;
        using var ctx = new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings { LimitBytes = 256 * MiB },
        };
        var filled = ArenaProbeModels.Filled(ctx);
        float Sum()
        {
            var outputs = filled.Execute(ArenaProbeModels.FilledShape(40L << 20));
            var sum = ArenaProbeModels.Sum(outputs);
            ComputeContext.ReleaseOutputs(outputs);
            return sum;
        }
        long Limit() => Assert.IsType<ArenaStatistics>(filled.ReadArenaStatistics()).LimitBytes;

        Assert.True(Sum() > 0f);
        Assert.Equal((256 * MiB - 8, 256 * MiB - 8), (filled.DeviceMemory.LimitBytes!.Value, Limit()));

        var held = TensorData([25L << 20], new float[25 << 20]).CopyTo(ctx);
        Assert.Equal(AllocationPool.Device, AllocationFailureReport.Classify(
            Assert.ThrowsAny<Exception>(() => Sum()), gpuBackend: false));
        Assert.Equal((156 * MiB - 8, 156 * MiB - 8), (filled.DeviceMemory.LimitBytes!.Value, Limit()));

        held.Delete();
        Assert.True(Sum() > 0f);
        Assert.Equal(256 * MiB - 8, Limit());
    }

    /// <summary>
    /// A request the card cannot serve — a run's, or a tensor's placed on the card — fails as an
    /// allocation failure on the card, the way ONNX Runtime's own allocators fail one, and leaves the
    /// card as usable as it found it: the session that failed runs on.
    /// </summary>
    [CudaFact]
    public void CudaProvider_ARequestTheCardCannotServeFailsAsACardAllocationFailureAndTheCardRunsOn()
    {
        using var ctx = new ComputeContext();
        var filled = ArenaProbeModels.Filled(ctx);
        AllocationPool Refused(Action request)
            => AllocationFailureReport.Classify(Assert.ThrowsAny<Exception>(request), gpuBackend: false);
        float Sum() => ArenaProbeModels.Sum(filled.Execute(ArenaProbeModels.FilledShape(1000)));

        Assert.Equal(AllocationPool.Device, Refused(() => filled.Execute(ArenaProbeModels.FilledShape(1L << 50))));
        Assert.Equal(1000f, Sum());
        Assert.Equal(AllocationPool.Device, Refused(() => DefaultBackend.Instance.CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType.Float, [1L << 50])));
        Assert.Equal(1000f, Sum());
    }

    /// <summary>
    /// What a session's limit caps, measured on a session of its own: only what the session
    /// allocates. A 64 MiB input already on the card is read where it is by a session limited to
    /// 32 MiB, which never allocates it; the same bytes in host memory are refused, since a session
    /// takes its inputs only in its own memory. That is why a context's budget discounts what it
    /// holds on the card from the session's limit for the length of the run.
    /// </summary>
    [CudaFact]
    public void CudaProvider_ASessionsLimitCapsWhatItAllocatesAndNotAnInputReadWhereItIs()
    {
        const long MiB = 1024 * 1024;
        var backend = DefaultBackend.Instance;
        var x = InputVector<float32>("x");
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            new InternalComputationGraph([x], [OnnxOp.ReduceSum(x)]), prepForOnnx: true);
        var model = new MemoryStream();
        ProtoBuf.Serializer.Serialize(model, proto);
        using var session = backend.CreateSession(
            model.ToArray(), ShorokooGraphOptimization.EnableAll, LogSettings.None,
            new DeviceMemorySettings { LimitBytes = 32 * MiB });
        var bytes = new byte[64 * MiB];
        IReadOnlyList<IShorokooTensorValue> Run(IShorokooTensorValue input) => session.Run(
            new Dictionary<string, IShorokooTensorValue> { [session.InputNames[0]] = input },
            session.OutputNames, RunSettings.Default);

        using var onCard = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bytes, [16L << 20]);
        Assert.False(onCard.IsHostAccessible);
        foreach (var output in Run(onCard)) output.Dispose();
        var allocated = Assert.IsType<ArenaStatistics>(session.ReadArenaStatistics());
        Assert.Equal(32 * MiB, allocated.LimitBytes);
        Assert.True(allocated.MaxInUseBytes < 32 * MiB);

        using var onHost = backend.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, bytes, [16L << 20]);
        Assert.True(onHost.IsHostAccessible);
        Assert.Contains("outside the memory its runs read it in", Assert.Throws<InvalidOperationException>(() => Run(onHost)).Message);
    }

    [CudaFact]
    public void CudaProvider_ATensorOnTheCardLargerThanOneArraySavesItsBytesThroughBoundedPieces()
    {
        const int N = 640 << 20;
        using var ctx = new ComputeContext();
        var limit = InputScalar<int32>("l");
        var compiled = ctx.Compile(new InternalComputationGraph([limit], [OnnxOp.Range(Scalar(0), limit, Scalar(1))]));
        var held = compiled.Execute(TensorData(DType.Int32, [], N))[0].ToTensorData();
        Assert.False(held.IsHostResident);
        long[] sampled = [0, (1L << 29) - 1, 1L << 29, (1L << 29) + 1, (9L << 26) + 12345, N - 1];
        var probe = new SamplingStream(sampled);

        long allocated = GC.GetAllocatedBytesForCurrentThread();
        held.WriteContentTo(probe);
        Assert.True(GC.GetAllocatedBytesForCurrentThread() - allocated < 64L << 20);
        Assert.Equal(4L * N, probe.Length);
        Assert.Equal(sampled.Select(i => (int)i), probe.Values);
    }

    [CudaFact]
    public void CudaProvider_ACopyTheCudaRuntimeFailsIsAnErrorRatherThanADeclinedRange()
    {
        using var info = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        using var bogus = new OrtTensorValue(OrtValue.CreateTensorValueWithData(info, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, [4L], (IntPtr)16, 16));
        Assert.Throws<InvalidOperationException>(() => DefaultBackend.Instance.TryCopyTensorRangeToHost(bogus, 0, new byte[16]));
        using var ctx = new ComputeContext();
        Assert.Equal(5f, AddTwoScalars(ctx, 2f, 3f));
    }

    /// <summary>Keeps the <c>int32</c> element at each sampled index of what is written to it, and
    /// nothing else.</summary>
    private sealed class SamplingStream(long[] sampled) : Stream
    {
        private readonly byte[] _pending = new byte[4];
        private long _position;
        public List<int> Values { get; } = [];
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            foreach (var index in sampled)
                for (long b = 4 * index; b < 4 * index + 4; b++)
                    if (b >= _position && b < _position + buffer.Length)
                    {
                        _pending[b - 4 * index] = buffer[(int)(b - _position)];
                        if (b == 4 * index + 3) Values.Add(BitConverter.ToInt32(_pending));
                    }
            _position += buffer.Length;
        }
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => _position;
        public override long Position { get => _position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }

    /// <summary>
    /// A tensor fed to a run on the card as it is goes to that run: one in the card's memory is
    /// handed over where it is, and one in host memory as the card copy it already holds where it
    /// has one, and a fresh card copy otherwise — dead afterwards either way. Fed
    /// <c>.Shared()</c>, a host tensor is copied onto the card once, attached to the context that
    /// read it, and read there by every run after until the tensor lets its copies go.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AFeedIsConsumedOnTheCardAndAHostOneIsReadThroughOneCopyUntilItLetsItGo()
    {
        using var ctx = new ComputeContext();
        var a = InputVector<float32>();
        var b = InputVector<float32>();
        var compiled = ctx.Compile(new InternalComputationGraph([a, b], [a * b + a]));
        float[] Run(IData x, IData y)
        {
            var output = compiled.Execute(x, y)[0].ToTensorData();
            ctx.Detach(output);
            return output.As<float32>().CopyMemory<float>();
        }
        var onHost = TensorData([2L], 10f, 20f);
        var onCard = TensorData([2L], 1f, 2f).To(ctx);
        Assert.False(onCard.IsHostResident);

        Assert.Equal([11f, 42f], Run(onCard, onHost.Shared()));
        Assert.True(onCard.IsDisposed);
        var copy = Assert.Single(ctx.Tensors, t => !t.IsHostResident);
        Assert.Equal(MemorySpace.Cuda(0), copy.Space);
        Assert.Equal([11f, 42f], Run(TensorData([2L], 1f, 2f), onHost.Shared()));
        Assert.Same(copy, Assert.Single(ctx.Tensors, t => !t.IsHostResident));

        onHost.ReleaseRunCopies();
        Assert.True(copy.IsDisposed);
        Assert.Equal([11f, 42f], Run(TensorData([2L], 1f, 2f), onHost.Shared()));
        Assert.Equal([11f, 42f], Run(TensorData([2L], 1f, 2f), onHost));
        Assert.True(onHost.IsDisposed);
        Assert.DoesNotContain(ctx.Tensors, t => !t.IsHostResident);
    }

    /// <summary>
    /// Every read of a tensor on the card copies its values to the host and returns them; the
    /// tensor stays where it is, alive, and the next run reads it there.
    /// </summary>
    [CudaFact]
    public void CudaProvider_ReadingATensorOnTheCardCopiesItsValuesToTheHostAndLeavesItThere()
    {
        using var ctx = new ComputeContext();
        var a = InputVector<float32>();
        var doubled = ctx.Compile(new InternalComputationGraph([a], [a + a]));
        var onCard = doubled.Execute(TensorData([3L], 1f, 2f, 3f))[0].ToTensorData().As<float32>();
        float[] values = [2f, 4f, 6f];

        Assert.Equal((MemorySpace.Cuda(0), false), (onCard.Space, onCard.IsHostResident));
        Assert.Equal(values, onCard.AccessMemory().ToArray());
        Assert.Equal(6f, onCard.AccessMemory()[2]);
        Assert.Equal(values, onCard.CopyMemory());
        Assert.Equal(4f, onCard.ValueAt(1));
        Assert.Equal(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(values).ToArray(), onCard.CopyRawMemory());
        Assert.Equal<object>([2f, 4f, 6f], onCard.DebugData);
        Assert.Equal((MemorySpace.Cuda(0), false, false), (onCard.Space, onCard.IsHostResident, onCard.IsDisposed));
        Assert.Equal(System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>((float[])[4f, 8f, 12f]).ToArray(), doubled.Execute(onCard.Shared())[0].ToTensorData().CopyRawMemory());
        Assert.Equal([4f, 8f, 12f], doubled.Execute(onCard)[0].ToTensorData().As<float32>().CopyMemory());
    }

    [CudaFact]
    public void CudaProvider_AnInMemoryLoaderOverTheCardBatchesAsOneOverTheHostDoes()
    {
        using var ctx = new ComputeContext();
        var (inputs, targets) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f, 5f, 6f), TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f, 10f, 12f));
        var (onCardInputs, onCardTargets) = (inputs.CopyTo(ctx), targets.CopyTo(ctx));
        var onCard = new InMemoryDataLoader(onCardInputs, onCardTargets, batchSize: 3, shuffle: true, seed: 7);
        var onHost = new InMemoryDataLoader(inputs, targets, batchSize: 3, shuffle: true, seed: 7);
        float[] Rows(DataBatch batch) => ((TensorData)((TensorDataStruct)batch.Target).Fields["targets"]).As<float32>().CopyMemory();

        Assert.False(((TensorData)onCardTargets.Fields["targets"]).IsHostResident);
        Assert.Equal([.. Enumerable.Range(0, 4).SelectMany(_ => Rows(onHost.Next()))], Enumerable.Range(0, 4).SelectMany(_ => Rows(onCard.Next())));
    }

    /// <summary>
    /// The interaction the whole integration turns on, and the one no CPU test can reach: state
    /// left in the provider's own memory across steps, while the arena it lives in is budgeted,
    /// extends by request, and is handed back after every run. The trained result has to be the
    /// same as an ordinary step loop's on the shipped defaults.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AResidentRunTrainsTheSameUnderABudgetedAndShrinkingArena()
    {
        var (input, target) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f),
                               TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));
        var expected = StepLoopWeights(input, target);

        var rig = ScalarRig(new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings
            {
                LimitBytes = 2L * 1024 * 1024 * 1024,
            },
            RunSettings = new RunSettings { ShrinkArenaAfterRun = true },
        });
        using var run = rig.BeginResidentRun();
        run.Step(input.Shared(), target.Shared());
        run.Step(input.Shared(), target.Shared());
        var published = run.StepToCheckpoint(input, target);

        Assert.Equal(3, published.Step);
        var actual = Weights(published);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 4);
    }

    /// <summary>
    /// A checkpoint a resident run hands out stays on the card, where the run goes on from it: the
    /// next step reads it in place, so no copy of it is made, on the card or anywhere else, and
    /// the caller's checkpoint holds the state once.
    /// </summary>
    [CudaFact]
    public void CudaProvider_ACheckpointAResidentRunHandsOutStaysOnTheCardAndIsReadThereWithoutACopy()
    {
        var (input, target) = (TrainingRigHelpers.InBatch(1f, 2f, 3f, 4f),
                               TrainingRigHelpers.TargetBatch(2f, 4f, 6f, 8f));
        var rig = ScalarRig();
        using var run = rig.BeginResidentRun();
        run.Step(input.Shared(), target.Shared());
        var published = run.StepToCheckpoint(input.Shared(), target.Shared());
        run.Step(input.Shared(), target.Shared());

        TensorData[] state = [.. ((TensorDataStruct[])[published.TrainableParams, published.ModelState, published.OptimizerState])
            .SelectMany(fields => fields.Fields.Values.OfType<TensorData>())];
        Assert.NotEmpty(state);
        Assert.All(state, t => Assert.False(t.IsHostResident));
        Assert.All(state, t => Assert.True(t.CopiesAreEmpty));
    }

    [CudaFact]
    public void CudaProvider_AFittedCheckpointSavesFromTheCardAndLoadsOntoItInBoundedPieces()
    {
        TrainingRig Rig(ComputationGraph model) => TrainingRig.FromScratch(
            model, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [TensorData([1L], [1f])], new AdamWOptimizerHyperparameters { LearningRate = 0.1f });
        TrainingCheckpoint Fitted(TrainingRig rig, long width) => rig.Fit(
            [rig.InputDef.FromOrderedData(TensorData([1L], [2f]))], [rig.TargetDef.FromOrderedData(TensorData([width], new float[width]))],
            numEpochs: 2).FinalCheckpoint;
        var (wideRig, narrowRig) = (Rig(WideMultiplyModel.ComputationGraph), Rig(ScalarMultiplyModel.ComputationGraph));
        var (wide, narrow) = (Fitted(wideRig, 1L << 20), Fitted(narrowRig, 1L));
        var home = wide.ToHost();
        float[] State(TrainingCheckpoint c) =>
            [.. TrainingRigHelpers.FlattenStruct(c.TrainableParams), .. TrainingRigHelpers.FlattenStruct(c.OptimizerState)];
        bool OnCard(TrainingCheckpoint c) => c.TrainableParams.Fields.Values.Concat(c.OptimizerState.Fields.Values)
            .OfType<TensorData>().All(t => !t.IsHostResident);
        long stateBytes = 4L * State(home).Length;
        Assert.True(OnCard(wide));

        (Action<TrainingCheckpoint, string> Save, Func<TrainingRig, string, TrainingCheckpoint> Load, string Suffix)[] forms =
        [
            ((c, p) => c.Save(p), (r, p) => r.LoadCheckpoint(p), ".safetensors"),
            ((c, p) => Persistence.SaveTrainingCheckpointToSkpt(c, p), (r, p) => r.LoadCheckpointFromSkpt(p), ".skpt"),
            ((c, p) => Persistence.ForTrainingCheckpoint(c).SaveAsDirectory(p), (r, p) => r.LoadCheckpointFromSkpt(p), "_dir"),
        ];
        var path = TrainingRigHelpers.TempPath("device_ckpt");
        try
        {
            foreach (var (save, load, suffix) in forms)
            {
                var (widePath, narrowPath) = (path + "_wide" + suffix, path + "_narrow" + suffix);
                Assert.True(Allocation(() => save(wide, widePath)) - Allocation(() => save(narrow, narrowPath)) < stateBytes / 2);
                Assert.True(Allocation(() => load(wideRig, widePath)) - Allocation(() => load(narrowRig, narrowPath)) < stateBytes / 2);
                var loaded = load(wideRig, widePath);
                Assert.True(OnCard(loaded));
                Assert.Equal(State(home), State(loaded.ToHost()));
            }
        }
        finally
        {
            foreach (var (_, _, suffix) in forms)
                foreach (var p in (string[])[path + "_wide" + suffix, path + "_narrow" + suffix])
                    if (Directory.Exists(p)) Directory.Delete(p, recursive: true);
                    else File.Delete(p);
        }
    }

    [CudaFact]
    public void CudaProvider_AModelLoadedCompiledReadsItsWeightsOntoTheCardAndRunsAsTheLoadedGraphDoes()
    {
        const long Width = 8192;
        var numOut = TensorData(DType.Int64, [], Width);
        var input = TensorData([1L, Width], [.. Enumerable.Range(0, (int)Width).Select(i => MathF.Sin(i))]);
        var path = TrainingRigHelpers.TempPath("compiled") + ".skpt";
        var (onnx, onnxPair) = (Path.ChangeExtension(path, ".onnx"), Path.ChangeExtension(path, ".pair.onnx"));
        try
        {
            {
                var model = FCLayer.ComputationGraph.ToConcreteArchitecture([numOut, input]).ToConcreteModel();
                Persistence.From(model).WithModel().WithWeights().Save(path);
                Persistence.ExportOnnx(model, onnx);
                Persistence.ExportOnnx(model, onnxPair, new OnnxExternalDataOptions { SizeThreshold = 0 });
            }
            using var context = new ComputeContext();
            float[] Run(CompiledGraph compiled) => [.. compiled.Execute(numOut.Shared(), input.Shared())[0].ToTensorData().ToHost().As<float32>().AccessMemory<float>()];
            float[] expected;
            using (var viaGraph = context.Compile(Persistence.Load(path))) expected = Run(viaGraph);

            foreach (var load in (Func<CompiledGraph>[])[() => context.ImportCompiledOnnx(onnx), () => context.ImportCompiledOnnx(onnxPair), () => context.LoadCompiled(path)])
            {
                GC.Collect();
                long managed = GC.GetAllocatedBytesForCurrentThread();
                long Private() => System.Diagnostics.Process.GetCurrentProcess().PrivateMemorySize64;
                long Card() => DeviceMemory.Read()!.Value.UsedBytes;
                var (host, card) = (Private(), Card());
                using var loaded = load();
                var (hostGrowth, cardGrowth) = (Private() - host, Card() - card);
                Assert.True(GC.GetAllocatedBytesForCurrentThread() - managed < 64L << 20);
                Assert.True(cardGrowth < 3 * 4 * Width * Width / 2);
                Assert.True(hostGrowth - cardGrowth < 4 * Width * Width / 2);
                Assert.Equal([false, false], loaded.SuppliedTensors.Select(t => t.IsHostResident));
                Assert.True(expected.Zip(Run(loaded)).All(p => MathF.Abs(p.First - p.Second) <= 1e-5f * MathF.Max(1f, MathF.Abs(p.First))));
            }
        }
        finally
        {
            File.Delete(path);
            File.Delete(onnx);
            File.Delete(onnxPair);
            File.Delete(onnxPair + ".data");
        }
    }

    private static long Allocation(Action act)
    {
        act();
        var before = GC.GetAllocatedBytesForCurrentThread();
        act();
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    /// <summary>
    /// Whether ONNX Runtime lets a run's feed go once the last node reading it has run, measured
    /// on a session of its own: it does not. A 64 MiB feed in the session's own arena, read by the
    /// graph's first node alone and held by nothing but the run, keeps its block to the end, so an
    /// arena capped at 136 MiB — room for the run's own two 64 MiB blocks — cannot take them beside
    /// it. The same bytes fed from outside the arena fit; and with the output bound into the feed's
    /// own memory the run fits where the feed is, and computes what it should. Writing an output
    /// into what a run consumed is the one way a run reuses that memory.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AFeedIsHeldUntilItsRunEndsSoOnlyAnOutputWrittenIntoItReusesItsMemory()
    {
        const long MiB = 1024 * 1024;
        const long N = 16L << 20;
        var backend = DefaultBackend.Instance;
        var x = InputVector<float32>("x");
        var fill = InputVector<int64>("fill");
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(new InternalComputationGraph([x, fill],
            [OnnxOp.Neg(OnnxOp.Expand(OnnxOp.ReduceSum(x, keepdims: false), fill))]), prepForOnnx: true);
        var model = new MemoryStream();
        ProtoBuf.Serializer.Serialize(model, proto);
        using var options = new SessionOptions();
        OrtBackend.Configure(options, ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal);
        CudaLibraries.Prepare();
        using var cuda = new OrtCUDAProviderOptions();
        cuda.UpdateOptions(new Dictionary<string, string>
        {
            ["device_id"] = "0",
            ["gpu_mem_limit"] = (136 * MiB).ToString(System.Globalization.CultureInfo.InvariantCulture),
            ["arena_extend_strategy"] = "kSameAsRequested",
        });
        options.AppendExecutionProvider_CUDA(cuda);
        using var session = new InferenceSession(model.ToArray(), options);
        using var onDevice = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
        using var shape = OrtValue.CreateTensorValueFromMemory<long>([N], [1L]);
        float[] fed = new float[N];
        Array.Fill(fed, 1f / N);
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(fed.AsSpan()).ToArray();
        OrtTensorValue Outside() => (OrtTensorValue)backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bytes, [N]);
        OrtValue Run(OrtValue feed, OrtValue? into = null, bool releasedFirst = false)
        {
            using var binding = session.CreateIoBinding();
            binding.BindInput(session.InputNames[0], feed);
            binding.BindInput(session.InputNames[1], shape);
            if (into is null) binding.BindOutputToDevice(session.OutputNames[0], onDevice);
            else binding.BindOutput(session.OutputNames[0], into);
            if (releasedFirst) feed.Dispose();
            using var runOptions = new RunOptions();
            session.RunWithBoundResults(runOptions, binding).Dispose();
            return binding.GetOutputValues()[0];
        }

        using var outside = Outside();
        var inArena = Run(outside.Inner);
        Assert.Contains("BFCArena", Assert.ThrowsAny<OnnxRuntimeException>(
            () => Run(inArena, releasedFirst: true)).Message);

        using var sameBytesOutside = Outside();
        Run(sameBytesOutside.Inner).Dispose();

        var feed = Run(outside.Inner);
        using var written = new OrtTensorValue(Run(feed, into: feed));
        var values = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(backend.CopyTensorToHost(written));
        Assert.Equal(N, values.Length);
        Assert.Equal([(float)N, N, N], [values[0], values[(int)(N / 2)], values[(int)(N - 1)]]);
    }

    /// <summary>
    /// A resident run whose steps write their state over the state they consume trains exactly as
    /// one that writes it anywhere else — the same bits, not merely close — with and without a
    /// budget on the context, which counts that state outside the arena. Every step writes all 56
    /// of the stack's state outputs over their inputs, the first one over the copies of the initial
    /// checkpoint it took, and the checkpoint step too, since its state stays on the card.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AResidentRunWritingItsStateOverTheStateItConsumedTrainsExactlyAsWithout()
    {
        (float[] Values, long Aliased) Trained(ComputeContext context)
        {
            using (context)
            {
                var sample = TensorData([2L, 8L], [.. Enumerable.Range(0, 16).Select(i => i / 16f)]);
                var rig = TrainingRig.FromScratch(
                    Modules.PlainTinyMlpStack.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
                    [sample.CopyTo(ComputeContext.Host)],
                    new AdamWOptimizerHyperparameters { LearningRate = 0.01f }, runtimeContext: context);
                var input = rig.InputDef.FromOrderedData(sample);
                var target = rig.TargetDef.FromOrderedData(TensorData([2L, 16L], [.. Enumerable.Range(0, 32).Select(i => i / 32f)]));
                using var run = rig.BeginResidentRun();
                for (int i = 0; i < 4; i++) run.Step(input.Shared(), target.Shared());
                var final = run.StepToCheckpoint(input.Shared(), target.Shared());
                return ([.. Weights(final), .. TrainingRigHelpers.FlattenStruct(final.ToHost().OptimizerState)], context.AliasedOutputs);
            }
        }

        var (plain, none) = Trained(new ComputeContext { OutputAliasing = false });
        var (aliased, written) = Trained(new ComputeContext());
        var (budgeted, budgetWritten) = Trained(new ComputeContext
        {
            DeviceMemory = new DeviceMemorySettings { LimitBytes = 64L * 1024 * 1024 },
        });

        Assert.Equal(plain, aliased);
        Assert.Equal(plain, budgeted);
        Assert.Equal(0L, none);
        Assert.Equal(5 * 56L, written);
        Assert.Equal(5 * 56L, budgetWritten);
    }

    // A resident AdamW run of WideLinearModel, shrinking as under a budget: the plain step holds the
    // new state beside the consumed one, two of its three parameter-sized values at once on the card
    // (the third is written after a temporary goes).
    [CudaFact]
    public void CudaProvider_WritingAStepsStateOverWhatItConsumedTakesMostOfTheStateOffTheCardsPeak()
    {
        DeviceMemory.ResetPeak();
        try
        {
            StepPeaks(aliasing: true, placing: false);
            var plain = StepPeaks(aliasing: false);
            var aliased = StepPeaks(aliasing: true, placing: false);

            Assert.True(plain.Arena - aliased.Arena >= aliased.State);
            Assert.True(plain.Card - aliased.Card >= aliased.State / 2);
        }
        finally
        {
            DeviceMemory.ResetPeak();
        }
    }

    [CudaFact]
    public void CudaProvider_PlacingAStepsStateOverWhatItConsumedTakesTheStateOffTheCardsPeakAsAliasingDoes()
    {
        DeviceMemory.ResetPeak();
        try
        {
            StepPeaks(aliasing: true, placing: false);
            var plain = StepPeaks(aliasing: false);
            var aliased = StepPeaks(aliasing: true, placing: false);
            var placed = StepPeaks(aliasing: false, placing: true);

            Assert.True(plain.Arena - placed.Arena >= placed.State - (1L << 20));
            Assert.True(placed.Arena <= aliased.Arena + (1L << 20));
        }
        finally
        {
            DeviceMemory.ResetPeak();
        }
    }

    /// <summary>
    /// The resident AdamW step of <see cref="WideLinearModel"/>, four times: the most its context's
    /// arena held, the most the card held beyond what it did as the run began, and the bytes of the
    /// step's state — with output aliasing as <paramref name="aliasing"/> says, and placement as
    /// <paramref name="placing"/> does, or as aliasing does where it says nothing.
    /// </summary>
    private static (long Arena, long Card, long State) StepPeaks(bool aliasing, bool? placing = null)
    {
        using var context = new ComputeContext
        {
            OutputAliasing = aliasing,
            ValuePlacement = placing,
            Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
            RunSettings = new RunSettings { ShrinkArenaAfterRun = true },
        };
        var sample = TensorData([2L, 4096L], [.. Enumerable.Range(0, 8192).Select(i => (i % 13) / 13f)]);
        var rig = TrainingRig.FromScratch(
            WideLinearModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
            [sample.CopyTo(ComputeContext.Host)],
            new AdamWOptimizerHyperparameters { LearningRate = 0.001f }, runtimeContext: context);
        var input = rig.InputDef.FromOrderedData(sample);
        var target = rig.TargetDef.FromOrderedData(TensorData([2L, 4096L], new float[8192]));
        var initial = rig.CreateInitialCheckpoint();
        var state = ((TensorDataStruct[])[initial.TrainableParams, initial.ModelState, initial.OptimizerState])
            .SelectMany(s => s.Fields.Values.OfType<TensorData>()).Sum(t => t.ByteCount);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        var x = InputVector<float32>("x");
        context.Execute(new InternalComputationGraph([x], [x + 1f]), TensorData([1L], 0f)).Single().ToTensorData().Delete();
        DeviceMemory.ResetPeak();
        var idle = DeviceMemory.Sample()!.Value.ProcessBytes!.Value;
        using var stop = new CancellationTokenSource();
        var sampler = Task.Run(() => { while (!stop.IsCancellationRequested) DeviceMemory.Sample(); });
        try
        {
            using var run = rig.BeginResidentRun(initial);
            for (int i = 0; i < 4; i++) run.Step(input.Shared(), target.Shared());
        }
        finally
        {
            stop.Cancel();
            sampler.Wait();
        }
        return (context.RunStats.PeakBytes, DeviceMemory.PeakProcessBytes - idle, state);
    }

    /// <summary>
    /// The ten arena figures off a CUDA allocator, which is what the whole statistics surface
    /// rests on and what no CPU machine can ask for: the allocator is built by name and a box
    /// without a card refuses it, so only the failure path has ever run. And what the arena holds
    /// before the first run — the session's weights, out of this same arena, so run 1's peak is
    /// them plus what the run added and the record has to carry both.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TheDeviceArenaAnswersAndAlreadyHoldsTheWeightsBeforeTheFirstRun()
    {
        using var ctx = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
        };
        var compiled = ArenaProbeModels.Weighted(ctx);

        var built = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
        Assert.Equal(ArenaProbeModels.WeightBytes, built.MaxInUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.InUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.RequestedInUseBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.MaxAllocSizeBytes);
        Assert.Equal(ArenaProbeModels.WeightBytes, built.TotalAllocatedBytes);
        Assert.Equal(1L, built.AllocationCount);
        Assert.Equal(-1L, built.LimitBytes);
        // The card takes the weight as a block of its own where the host arena reserves it, so
        // these two counts mean different things on the two devices and do not compare.
        Assert.Equal(1L, built.ArenaExtensionCount);
        Assert.Equal(0L, built.ReserveCount);

        compiled.Execute(ArenaProbeModels.WeightedInput());
        var run = Assert.Single(ctx.RunStats.RecentRuns);
        Assert.Equal(ArenaProbeModels.WeightBytes, run.PriorPeakBytes);
        Assert.Equal(MemoryFigureKind.Measured, run.PeakKind);
        Assert.True(run.PeakBytes > run.PriorPeakBytes);
        Assert.True(run.PeakBytes - run.PriorPeakBytes < ArenaProbeModels.WeightBytes / 16);
    }

    /// <summary>
    /// A one-shot run whose intermediates fill 256 MiB of the card and whose output is four
    /// kilobytes, made after the fill is freed, leaves the card holding the output and nothing more
    /// once it returns, though the output is kept. Compiled, with the outputs kept, a session holds in
    /// use its weights and each output's own bytes once a run that hands its memory back is over,
    /// nothing beyond what is in use, and its weights alone once the outputs are let go of.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AKeptOutputHoldsOnlyItsOwnBytesOnTheCardAndNothingOfItsSessionsArena()
    {
        const long MiB = 1024 * 1024;
        using var ctx = new ComputeContext { RunSettings = new RunSettings { ShrinkArenaAfterRun = true } };

        ctx.Execute(ArenaProbeModels.Widened(), ArenaProbeModels.FilledShape(1024))[0].ToTensorData().Delete();
        var before = HeldOnTheCard();
        var widened = ctx.Execute(ArenaProbeModels.Widened(), ArenaProbeModels.FilledShape(64L << 20))[0].ToTensorData();
        Assert.True(HeldOnTheCard() - before < 16 * MiB);

        using var unshrinking = new ComputeContext();
        before = HeldOnTheCard();
        var widenedUnshrunk = unshrinking.Execute(ArenaProbeModels.Widened(), ArenaProbeModels.FilledShape(64L << 20))[0].ToTensorData();
        Assert.True(HeldOnTheCard() - before < 16 * MiB);
        widenedUnshrunk.Delete();

        var filled = ArenaProbeModels.Filled(ctx);
        var spread = ArenaProbeModels.Spread(ctx);
        (long, long, long, long) Held()
        {
            var learned = Assert.IsType<ArenaStatistics>(filled.ReadArenaStatistics());
            var settled = Assert.IsType<ArenaStatistics>(spread.ReadArenaStatistics());
            return (learned.RequestedInUseBytes, settled.RequestedInUseBytes,
                learned.TotalAllocatedBytes - learned.InUseBytes, settled.TotalAllocatedBytes - settled.InUseBytes);
        }
        var built = Held();
        var sum = filled.Execute(ArenaProbeModels.FilledShape(64L << 20))[0].ToTensorData();
        var spreadSum = spread.Execute(ArenaProbeModels.Ones(64 << 20))[0].ToTensorData();
        var kept = Held();
        Assert.Equal([64 << 20, 64 << 20, -(64 << 20)], [widened.ValueAt<float>(999), sum.ValueAt<float>(0), spreadSum.ValueAt<float>(999)]);
        var bytes = (sum.ByteCount, spreadSum.ByteCount);
        sum.Delete();
        spreadSum.Delete();

        Assert.Equal((built.Item1 + bytes.Item1, built.Item2 + bytes.Item2, 0L, 0L), kept);
        Assert.Equal((built.Item1, built.Item2), (Held().Item1, Held().Item2));
    }

    /// <summary>
    /// An output whose shape only the run learns — 256 MiB of indices of a 128 MiB tensor on the
    /// card — is the block its session's run wrote it into, on the card once: the session's
    /// allocator holds exactly its bytes more while it is kept and nothing more once it is let go,
    /// and the card holds less than the output twice over for the run's every block.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AnOutputWhoseShapeTheRunLearnsIsOnTheCardOnceAsTheBlockItsSessionWroteItInto()
    {
        using var ctx = new ComputeContext();
        var learned = ArenaProbeModels.Learned(ctx);
        long Requested() => Assert.IsType<ArenaStatistics>(learned.ReadArenaStatistics()).RequestedInUseBytes;
        var ones = ArenaProbeModels.Ones(32 << 20).CopyTo(ctx);
        var (before, requestedBefore) = (HeldOnTheCard(), Requested());

        var indices = learned.Execute(ones.Shared())[0].ToTensorData();
        var (bytes, held, requested) = (indices.ByteCount, HeldOnTheCard() - before, Requested() - requestedBefore);
        indices.Delete();

        Assert.Equal((bytes, requestedBefore), (requested, Requested()));
        Assert.True(held < 2 * bytes);
    }

    /// <summary>
    /// The card's allocator — the one tensors placed on the card and everything a session allocates
    /// there come from — keeps the blocks of tensors that are gone, through a run that keeps what it
    /// has, and hands them back to the card as a run that hands back its memory ends, or as the
    /// program asks.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TheCardsAllocatorHandsBackWhatNoTensorUsesAsARunThatHandsBackItsMemoryEndsOrTheProgramAsks()
    {
        const long MiB = 1024 * 1024;
        using var ctx = new ComputeContext();
        var product = ArenaProbeModels.MatMul(ctx);
        void Run(bool shrink) => product.Execute(
            [ArenaProbeModels.MatMulOperand(8), ArenaProbeModels.MatMulOperand(8)],
            new RunSettings { ShrinkArenaAfterRun = shrink })[0].ToTensorData().Delete();

        HeldOnTheCard();
        Run(shrink: true);
        var before = HeldOnTheCard();
        void PlaceAndDelete()
        {
            foreach (var placed in Enumerable.Range(0, 8).Select(_ => TensorData([8L << 20], new float[8 << 20]).CopyTo(ctx)).ToList())
                placed.Delete();
        }
        PlaceAndDelete();
        Run(shrink: false);
        var kept = HeldOnTheCard() - before;
        Run(shrink: true);
        var shrunk = HeldOnTheCard() - before;
        PlaceAndDelete();
        var keptAgain = HeldOnTheCard() - before;
        var released = DeviceMemory.ReleaseCached();

        Assert.True(kept >= 256 * MiB);
        Assert.True(shrunk < 16 * MiB);
        Assert.True(keptAgain >= 256 * MiB);
        Assert.True(released >= 256 * MiB);
        Assert.True(HeldOnTheCard() - before < 16 * MiB);
    }

    [CudaFact]
    public void CudaProvider_ASessionRunOverManyShapesKeepsNoMoreOnTheCardThanItsBusiestRunHadInUse()
    {
        using var ctx = new ComputeContext();
        var filled = ArenaProbeModels.Filled(ctx);
        foreach (var elements in (long[])[16L << 20, 3L << 22, 1L << 23, 3L << 21, 1L << 22])
            ComputeContext.ReleaseOutputs(filled.Execute(ArenaProbeModels.FilledShape(elements)));
        var held = Assert.IsType<ArenaStatistics>(filled.ReadArenaStatistics());
        Assert.True(held.TotalAllocatedBytes <= held.MaxInUseBytes);
    }

    [CudaFact]
    public void CudaProvider_ASessionCyclingThroughShapesTakesNothingMoreFromTheCardAfterItsFirstCycle()
    {
        using var ctx = new ComputeContext();
        var filled = ArenaProbeModels.Filled(ctx);
        (long, long) Cycle()
        {
            foreach (var elements in (long[])[2L << 20, 8L << 20, 4L << 20, 6L << 20])
                ComputeContext.ReleaseOutputs(filled.Execute(ArenaProbeModels.FilledShape(elements)));
            var held = Assert.IsType<ArenaStatistics>(filled.ReadArenaStatistics());
            return (held.ArenaShrinkageCount, held.TotalAllocatedBytes);
        }

        Cycle();
        var second = Cycle();
        Assert.Equal(second, Cycle());
    }

    [CudaFact]
    public void CudaProvider_BlocksUpToAMebibyteHoldTheirOwnBytesAndALargerOneWholeCardPagesOfItsOwn()
    {
        var card = RuntimeAllocator.ForCard(0);
        var account = card.Shared.Open("probe");
        List<OrtValue> values = [];
        long Take(long floats)
        {
            using (CachingAllocator.Charge(null, account))
                values.Add(OrtValue.CreateAllocatedTensorValue(card.Managed, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, [floats]));
            return (long)OrtBackend.AddressOf(values[^1]) % (2L << 20);
        }

        Take(1000);
        Take(1L << 18);
        var large = Take((1L << 18) + 1);
        var held = card.Shared.Statistics(account).TotalAllocatedBytes;
        values.ForEach(value => value.Dispose());
        card.Shared.Close(account);

        Assert.Equal((0L, 4096 + (1L << 20) + (2L << 20)), (large, held));
    }

    [CudaFact]
    public void CudaProvider_ABlockHandsItsFirstGranuleBackToTheCardButKeepsItsAddressUntilItGoes()
    {
        var card = RuntimeAllocator.ForCard(0);
        var account = card.Shared.Open("probe");
        OrtValue Take(long floats)
        {
            using (CachingAllocator.Charge(null, account))
                return OrtValue.CreateAllocatedTensorValue(card.Managed, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, [floats]);
        }

        var first = Take(3L << 19);
        var address = OrtBackend.AddressOf(first);
        card.Shared.ReleaseRange(address, 0, 4L << 20, toTheEnd: false);
        var held = card.Shared.Statistics(account).TotalAllocatedBytes;
        var second = Take(1L << 19);
        var aliased = OrtBackend.AddressOf(second) == address;
        second.Dispose();
        first.Dispose();
        card.Shared.Close(account);

        Assert.Equal((4L << 20, false), (held, aliased));
    }

    [CudaFact]
    public void CudaProvider_TheReuseScenarioTakesItsKnownBlocksAndNoMoreBeyondItsInputsAndWeightsOnTheCard()
    {
        Assert.Equal((10L, 6L), ArenaProbeModels.ReuseRun(ArenaProbeModels.ReuseShapes.Computed));
        Assert.Equal((10L, 6L), ArenaProbeModels.ReuseRun(ArenaProbeModels.ReuseShapes.Paired));
        Assert.Equal((5L, 5L), ArenaProbeModels.ReuseRun(ArenaProbeModels.ReuseShapes.Static));
        Assert.Equal((11L, 6L), ArenaProbeModels.ReuseRun(ArenaProbeModels.ReuseShapes.Tracked));
    }

    [CudaFact]
    public void CudaProvider_TheReuseScenarioTakesWhatTheCardsArenaIsAskedForAndHoldsNoMoreThanItAtOnce()
    {
        var arena = new OrtArenaCardBackend();
        foreach (var shapes in Enum.GetValues<ArenaProbeModels.ReuseShapes>())
        {
            var (ours, theirs) = (ArenaProbeModels.ReuseRun(shapes), ArenaProbeModels.ReuseRun(shapes, arena));
            Assert.Equal(theirs.Allocations, ours.Allocations);
            Assert.True(ours.Halves <= theirs.Halves);
        }
    }

    private sealed class OrtArenaCardBackend : OrtBackend
    {
        public OrtArenaCardBackend() : base(cudaDeviceId: 0) => SessionsUseOrtArena = true;
    }

    [CudaFact]
    public void CudaProvider_AnAdamOrAdamWStepUpdatesATableInOneFusedPassToTheBit()
    {
        NNLibraryOptimizerTrainingCoverageTests.AssertFusedToTheBit(fuses => new FusingCardBackend(fuses), AdamOptimizer.ComputationGraph, [0.001f, 0.9f, 0.999f, 1e-8f]);
        NNLibraryOptimizerTrainingCoverageTests.AssertFusedToTheBit(fuses => new FusingCardBackend(fuses), AdamWOptimizer.ComputationGraph, [0.001f, 0.9f, 0.999f, 1e-8f, 0.01f]);
        NNLibraryOptimizerTrainingCoverageTests.AssertFusedToTheBit(fuses => new FusingCardBackend(fuses), AdamWOptimizer.ComputationGraph, [0.01f, 0.8f, 0.9f, 1e-6f, 0f]);
        NNLibraryOptimizerTrainingCoverageTests.AssertFusedToTheBit(fuses => new FusingCardBackend(fuses), AdamWOptimizer.ComputationGraph,
            new AdamWOptimizerHyperparameters { WeightDecay = Hyperparameter.Runtime() }.InOptimizerOrder(), 0.1f);
        NNLibraryOptimizerTrainingCoverageTests.AssertFusedToTheBit(fuses => new FusingCardBackend(fuses), AdamWOptimizer.ComputationGraph,
            [0.001f, 0.9f, 0.999f, 1e-8f, 0.01f], null, NNWideGatheredTableProjectionModel.ComputationGraph, 8256);
    }

    private sealed class FusingCardBackend : OrtBackend
    {
        public FusingCardBackend(bool fuses) : base(cudaDeviceId: 0) => FusesOptimizerUpdates = fuses;
    }

    /// <summary>What this process holds on the card once every tensor nothing reaches any more is
    /// released: a released tensor's finalizer can leave another to the next collection.</summary>
    private static long HeldOnTheCard()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        return DeviceMemory.Read()!.Value.ProcessBytes!.Value;
    }

    /// <summary>
    /// A graph the provider cannot run whole: one output is computed on the card and one on the
    /// host, which is what <see cref="SessionOutputPlacement.Mixed"/ > is for, and the crossing
    /// is charged to the pinned host arena rather than to the device one. A graph with no node the
    /// provider can run is <see cref="SessionOutputPlacement.Host"/>, its device memory holding
    /// nothing but the output it copies onto the card, and one it runs whole is
    /// <see cref="SessionOutputPlacement.Device"/>. Every output comes back on the card.
    /// </summary>
    [CudaFact]
    public void CudaProvider_OutputPlacementSeparatesADeviceGraphAPartitionedOneAndOneThatFellBack()
    {
        using var ctx = new ComputeContext();

        var partitioned = ArenaProbeModels.Partitioned(ctx);
        Assert.Equal(SessionOutputPlacement.Mixed, partitioned.OutputPlacement);

        var pinnedBefore = Assert.IsType<ArenaStatistics>(partitioned.ReadPinnedArenaStatistics());
        Assert.Equal(0L, pinnedBefore.AllocationCount);

        Assert.All(partitioned.Execute(ArenaProbeModels.Square()), o => Assert.False(o.ToTensorData().IsHostResident));
        var pinned = Assert.IsType<ArenaStatistics>(partitioned.ReadPinnedArenaStatistics());
        var device = Assert.IsType<ArenaStatistics>(partitioned.ReadArenaStatistics());
        Assert.True(pinned.AllocationCount > 0);
        Assert.True(pinned.MaxInUseBytes > 0);
        Assert.NotEqual(device, pinned);

        var host = ArenaProbeModels.HostOnly(ctx);
        Assert.All(host.Execute(ArenaProbeModels.Square()), o => Assert.False(o.ToTensorData().IsHostResident));
        Assert.Equal(SessionOutputPlacement.Host, host.OutputPlacement);
        Assert.Equal(1L, Assert.IsType<ArenaStatistics>(host.ReadArenaStatistics()).AllocationCount);

        var onCard = ArenaProbeModels.MatMul(ctx);
        Assert.All(onCard.Execute(ArenaProbeModels.MatMulOperand(8), ArenaProbeModels.MatMulOperand(8)), o => Assert.False(o.ToTensorData().IsHostResident));
        Assert.Equal(SessionOutputPlacement.Device, onCard.OutputPlacement);
    }

    /// <summary>
    /// The node trace of a genuinely partitioned graph: two providers, the fallen-back node named,
    /// and the copy node the runtime inserts at the boundary sitting <i>at</i> the boundary. That
    /// last one is the whole point — the copy carries the highest graph index of the four, so
    /// ordering the trace by index would move exactly the node that marks the fallback to the end,
    /// past the node it feeds.
    /// </summary>
    [CudaFact]
    public void CudaProvider_APartitionedTraceNamesBothProvidersAndKeepsTheCopyAtTheBoundary()
    {
        using var ctx = new ComputeContext
        {
            Diagnostics = new DiagnosticSettings { TraceNodePlacement = true },
        };
        var compiled = ArenaProbeModels.Partitioned(ctx);
        compiled.Execute(ArenaProbeModels.Square());

        var placement = Assert.IsType<NodePlacement>(compiled.ReadNodePlacement());
        Assert.Equal(["CUDAExecutionProvider", "CPUExecutionProvider"],
            placement.Providers.Select(share => share.Provider));

        var onHost = Assert.Single(placement.NodesOn("CPUExecutionProvider"));
        Assert.Equal("Det", onHost.OpType);
        Assert.Equal(placement.Nodes.Count, placement.Providers.Sum(share => share.NodeCount));

        var ran = placement.Nodes.ToList();
        var copy = Assert.Single(ran.Where(node => node.OpType.StartsWith("Memcpy", StringComparison.Ordinal)));
        Assert.Equal("MemcpyToHost", copy.OpType);
        Assert.Equal(ran.Max(node => node.NodeIndex), copy.NodeIndex);
        Assert.True(copy.NodeIndex > onHost.NodeIndex);
        Assert.Equal(ran.IndexOf(onHost) - 1, ran.IndexOf(copy));
        Assert.NotEqual(ran.Count - 1, ran.IndexOf(copy));
    }

    /// <summary>
    /// Shrinkage, asked for and observed: the arena hands blocks back at the end of every run, and
    /// what it is holding afterwards drops below the high-water mark it reached — which is the one
    /// case where <see cref="RunStatistics.ArenaBytes"/> sits under
    /// <see cref="RunStatistics.PeakBytes"/> rather than above it. Nothing had ever seen it
    /// happen; the CPU arena never shrank in testing.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AShrinkingRunHandsBlocksBackAndEndsBelowThePeakItReached()
    {
        static RunStatistics Runs(bool shrink)
        {
            using var ctx = new ComputeContext
            {
                Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
                RunSettings = new RunSettings { ShrinkArenaAfterRun = shrink },
            };
            var compiled = ArenaProbeModels.MatMul(ctx);
            var operand = ArenaProbeModels.MatMulOperand(512);
            for (int run = 0; run < 3; run++) ComputeContext.ReleaseOutputs(compiled.Execute(operand.Shared(), operand.Shared()));
            return ctx.RunStats;
        }

        var shrinking = Runs(shrink: true);
        Assert.True(shrinking.ArenaShrinkageCount >= 3);
        Assert.True(shrinking.PeakBytes > 0);
        Assert.True(shrinking.ArenaBytes < shrinking.PeakBytes);

        var keeping = Runs(shrink: false);
        Assert.Equal(0L, keeping.ArenaShrinkageCount);
        Assert.Equal(shrinking.PeakBytes, keeping.PeakBytes);
        Assert.True(keeping.ArenaBytes >= keeping.PeakBytes);
        // The blocks figure falls when they go back, so a shrinking context undercounts them.
        Assert.True(keeping.ArenaExtensionCount > shrinking.ArenaExtensionCount);
    }

    /// <summary>
    /// The comparison #198 asked for and no CPU machine can make: what this context's own sessions
    /// took, against what the card reports for every process on it. The session figure is the
    /// smaller of the two and accounts for part of the rise — the rest is the provider's context,
    /// its libraries and whatever else holds the device.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TheContextsRunStatisticsAccountForPartOfWhatTheCardReports()
    {
        DeviceMemory.ResetPeak();
        try
        {
            using var ctx = new ComputeContext
            {
                Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
            };
            var compiled = ArenaProbeModels.MatMul(ctx);
            var operand = ArenaProbeModels.MatMulOperand(1024);

            var idle = DeviceMemory.Sample();
            Assert.NotNull(idle);
            for (int run = 0; run < 5; run++)
            {
                compiled.Execute(operand.Shared(), operand.Shared());
                DeviceMemory.Sample();
            }

            var stats = ctx.RunStats;
            Assert.Equal(5L, stats.RunCount);
            Assert.True(stats.PeakBytes > 0);
            var device = Assert.IsType<ArenaStatistics>(compiled.ReadArenaStatistics());
            Assert.Equal(stats.PeakBytes, device.MaxInUseBytes);

            var grew = DeviceMemory.PeakUsedBytes - idle!.Value.UsedBytes;
            Assert.True(grew >= stats.PeakBytes);
            Assert.True(DeviceMemory.PeakUsedBytes < idle.Value.TotalBytes);
        }
        finally
        {
            DeviceMemory.ResetPeak();
        }
    }

    /// <summary>
    /// The card's figure for this process alone, which #406 asked for: a gigabyte held on the card
    /// is in it, and it is a part of what the card reports across every process.
    /// </summary>
    [CudaFact]
    public void CudaProvider_TheProcessFigureHoldsWhatThisProcessPutOnTheCard()
    {
        using var ctx = new ComputeContext();
        Assert.True(DeviceMemory.Read()!.Value.ProcessBytes > 0);

        var shape = InputVector<int64>("shape");
        var held = ctx.Execute(new InternalComputationGraph([shape], [OnnxOp.Expand(Vector(1f), shape)]), TensorData([1L], 256L << 20))[0].ToTensorData();
        DeviceMemoryReading reading;
        try { reading = DeviceMemory.Read()!.Value; }
        finally { held.Delete(); }

        Assert.True(reading.ProcessBytes >= 1L << 30);
        Assert.True(reading.ProcessBytes <= reading.UsedBytes);
    }

    /// <summary>
    /// The placement probe hands ONNX Runtime the session as a bare handle, and a compiled graph is
    /// held weakly by its context, so this reads the placement off a graph nothing else holds while
    /// another thread collects and drains finalizers. <b>It means something only in Release</b>:
    /// unoptimized code roots a local to the end of its scope, so the hazard cannot arise in Debug
    /// at all.
    ///
    /// <para>What it pins is the path, not the <c>GC.KeepAlive</c> in it — removing that does not
    /// reproduce a fault, because the probe is reached through a <c>Lazy</c> whose factory closure
    /// holds the session while the factory runs. A change that takes the closure away is what this
    /// would catch.</para>
    /// </summary>
    [CudaFact]
    public void CudaProvider_OutputPlacementReadsFromAGraphHeldOnlyInALocalWhileAnotherThreadCollects()
    {
        using var collecting = new CancellationTokenSource();
        var churn = Task.Run(() =>
        {
            // The finalizers are the hazard, not the collection: a session the collector frees is
            // only released once ~InferenceSession runs. Throttled, because a bare collect loop
            // starves the thread compiling the sessions and buys no extra collections per probe.
            while (!collecting.IsCancellationRequested)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                Thread.Sleep(1);
            }
        });
        try
        {
            for (int round = 0; round < 20; round++)
                Assert.Equal(SessionOutputPlacement.Device, PlacementOfAGraphNothingElseHolds());
        }
        finally
        {
            collecting.Cancel();
            churn.Wait();
        }
    }

    /// <summary>The graph, its context and the session under both are unreachable from the moment
    /// the placement is asked for: every one of them is a local at its last read. Not inlined, so
    /// the caller's frame cannot keep them alive either.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static SessionOutputPlacement PlacementOfAGraphNothingElseHolds()
    {
        var a = InputVector<float32>();
        var b = InputVector<float32>();
        var context = new ComputeContext();
        var compiled = context.Compile(new InternalComputationGraph([a, b], [a * b + a]));
        return compiled.OutputPlacement;
    }

    private static TrainingRig ScalarRig(ComputeContext? runtimeContext = null) => TrainingRig.FromScratch(
        ScalarMultiplyModel.ComputationGraph, L2Loss.ComputationGraph, AdamWOptimizer.ComputationGraph,
        [TensorData([4L], [1f, 2f, 3f, 4f])],
        new AdamWOptimizerHyperparameters { LearningRate = 0.1f }, runtimeContext: runtimeContext);

    private static float[] StepLoopWeights(TensorDataStruct input, TensorDataStruct target)
    {
        var rig = ScalarRig();
        var ckpt = rig.CreateInitialCheckpoint();
        for (int i = 0; i < 3; i++) ckpt = rig.TrainStep(ckpt, input.Shared(), target.Shared());
        return Weights(ckpt);
    }

    private static float[] Weights(TrainingCheckpoint checkpoint) =>
        TrainingRigHelpers.FlattenStruct(checkpoint.ToHost().TrainableParams);

    private static float AddTwoScalars(ComputeContext ctx, float left, float right)
    {
        var a = InputScalar<float32>();
        var b = InputScalar<float32>();
        var c = a + b;

        var graph = new InternalComputationGraph([a, b], [c]);
        var results = ctx.Execute(
            (graph),
            TensorData([], left),
            TensorData([], right));

        return results[0].ToTensorData().As<float32>().ValueAt<float>(0);
    }
}
