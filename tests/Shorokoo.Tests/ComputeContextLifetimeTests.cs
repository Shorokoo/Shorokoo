using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// <c>y = -x</c>, the whole model: one node, no trainable parameter, no randomness, nothing whose
/// geometry has to be resolved by running anything. A model this small is the one that most
/// obviously needs no backend to describe, which is what makes it the subject of
/// <see cref="ComputeContextLifetimeCoverageTests.TestBuildingAndExportingAModelAsksForNoComputeContextAtAll"/>.
/// </summary>
[Module]
public partial class BackendFreeNegate
{
    public static Tensor<float32> Inline(Tensor<float32> x) => -x;
}

/// <summary>
/// A compute context owns the tensors whose memory is on its books, releases them when it is
/// disposed, and can be asked to hand its results out detached so they outlive it.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ComputeContextLifetimeCoverageTests
{
    private static TensorData Sample() => TensorData([4L], (float[])[1f, 2f, 3f, 4f]);

    private static float[] Floats(TensorData t) => [.. t.As<float32>().AccessMemory<float>()];

    private static (InternalComputationGraph Graph, TensorData A, TensorData B, float[] Expected) Model()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        float[] av = [1f, 2f, 3f, 4f];
        float[] bv = [10f, 20f, 30f, 40f];
        return (new InternalComputationGraph([a, b], [a * b + a]),
            TensorData([4L], av), TensorData([4L], bv),
            [.. av.Zip(bv, (x, y) => x * y + x)]);
    }

    [Fact]
    public void TestDisposingAContextReleasesWhatItOwnsAndInvalidatesIt()
    {
        var context = new ComputeContext();
        var owned = Sample().TransferTo(context);

        context.Dispose();

        // The tensor was never disposed itself; its memory went when the context did.
        Assert.False(owned.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => Floats(owned));
    }

    [Fact]
    public void TestDisposingTheSourceContextLeavesMemoryThatWasTransferredAway()
    {
        var source = new ComputeContext();
        var target = new ComputeContext();

        var onSource = Sample().TransferTo(source);
        var onTarget = onSource.TransferTo(target);

        // Same space, so nothing moved but the ownership -- and the ownership is what disposal
        // follows. The source has nothing left to release.
        source.Dispose();

        Assert.Equal([1f, 2f, 3f, 4f], Floats(onTarget));
        Assert.Same(target, onTarget.Context);

        // And the target still governs it.
        target.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(onTarget));
    }

    [Fact]
    public void TestDisposingAContextLeavesTensorsItOnlyHadAccessTo()
    {
        var owner = Sample();
        var borrower = new ComputeContext();
        var reader = owner.GiveAccessTo(borrower);

        borrower.Dispose();

        // The borrower never owned the bytes, so its disposal cannot have taken them.
        Assert.Equal([1f, 2f, 3f, 4f], Floats(owner));
        Assert.Equal([1f, 2f, 3f, 4f], Floats(reader));
    }

    [Fact]
    public void TestDisposingAContextTwiceIsHarmless()
    {
        var context = new ComputeContext();
        _ = Sample().TransferTo(context);
        context.Dispose();
        context.Dispose();
    }

    [Fact]
    public void TestTheHostContextRefusesToCompileOrRunAndResolvesNoBackendDoingIt()
    {
        var (graph, a, b, _) = Model();

        foreach (var refused in (Action[])[
            () => ComputeContext.Host.Compile(graph),
            () => ComputeContext.Host.Execute(graph, a, b),
            () => ComputeContext.Host.Run(graph),
            () => ComputeContext.Host.Eval(InputVector<float32>("a") * 2f),
            () => ComputeContext.Host.ExecuteWithState(graph, a, b)])
        {
            var ex = Assert.Throws<InvalidOperationException>(refused);
            Assert.Contains("new ComputeContext(", ex.Message);
        }

        var backendReads = 0;
        var contextReads = ComputeContext.CountInstanceReads(() =>
            backendReads = DefaultBackend.CountInstanceReads(
                () => Assert.Throws<InvalidOperationException>(
                    () => ComputeContext.Host.Compile(graph))));
        Assert.Equal(0, contextReads);
        Assert.Equal(0, backendReads);
    }

    [Fact]
    public void TestTheHostContextCannotBeDisposed()
    {
        var detached = Sample();

        ComputeContext.Host.Dispose();
        ComputeContext.Host.Dispose();

        Assert.False(ComputeContext.Host.IsDisposed);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(detached));
        Assert.Equal([1f, 2f, 3f, 4f], Floats(Sample().TransferTo(ComputeContext.Host)));
    }

    [Fact]
    public void TestAContextListsTheTensorsAttachedToItAndTheHostContextTheDetachedOnes()
    {
        using var first = new ComputeContext();
        using var second = new ComputeContext();
        var detached = Sample();
        var onFirst = detached.CopyTo(first);

        Assert.Contains(onFirst, first.Tensors);
        Assert.DoesNotContain(onFirst, second.Tensors);
        Assert.Contains(detached, ComputeContext.Host.Tensors);

        var onSecond = onFirst.TransferTo(second);

        Assert.Contains(onSecond, second.Tensors);
        Assert.DoesNotContain(onFirst, first.Tensors);
    }

    [Fact]
    public void TestAMemoryDeviceIsOnePerSpaceAndListsTheBackendsAndContextsOnIt()
    {
        var cpu = new StubBackend(ComputeDevice.Cpu, null);
        var alsoCpu = new StubBackend(ComputeDevice.Cpu, null);
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        var host = MemoryDevice.For(MemorySpace.Host);

        Assert.Same(host, MemoryDevice.Of(cpu));
        Assert.Same(host, MemoryDevice.Of(alsoCpu));
        Assert.Same(MemoryDevice.For(MemorySpace.Cuda(0)), MemoryDevice.Of(card));
        Assert.NotSame(host, MemoryDevice.Of(card));
        Assert.Equal(MemorySpace.Host, host.Space);

        Assert.Contains(cpu, host.Backends);
        Assert.Contains(alsoCpu, host.Backends);
        Assert.Contains(HostBackend.Instance, host.Backends);
        Assert.DoesNotContain(card, host.Backends);

        using var context = new ComputeContext(cpu);
        Assert.Contains(context, cpu.ContextsOn());
        Assert.DoesNotContain(context, alsoCpu.ContextsOn());
        Assert.Same(host, context.Device);
        Assert.Same(host, ComputeContext.Host.Device);
    }

    [Fact]
    public void TestTheHostBackendBuildsNeitherASessionNorAValue()
    {
        IShorokooBackend backend = HostBackend.Instance;
        var raw = Sample().CopyRawMemory();

        Assert.Equal(MemorySpace.Host, backend.MemorySpace);
        Assert.Equal("Shorokoo.HostMemory", ComputeContext.Host.Backend.Name);

        var session = Assert.Throws<NotSupportedException>(() => backend.CreateSession(
            default, ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal,
            DeviceMemorySettings.Default));
        Assert.Contains("Shorokoo.LinuxCPU", session.Message);

        Action[] values =
        [
            () => backend.CreateTensor<float>([1f, 2f], [2L]),
            () => backend.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, raw, [4L]),
            () => backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, raw, [4L]),
            () => backend.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Float, [4L]),
            () => backend.CreateStringTensor(["a", "b"], [2L]),
            () => backend.CreateSequence([]),
        ];
        foreach (var build in values)
            Assert.Contains("builds no runtime values", Assert.Throws<NotSupportedException>(build).Message);
    }

    [Fact]
    public void TestAContextThatDetachesOutputsHandsBackResultsThatOutliveIt()
    {
        var (graph, a, b, expected) = Model();
        var context = new ComputeContext(DefaultBackend.Instance, detachesOutputs: true);

        Assert.True(context.DetachesOutputs);
        var result = context.Execute(graph, a, b)[0].ToTensorData();

        // Detached: the result is in the framework's own host memory, which is what makes the
        // next line safe.
        Assert.Same(ComputeContext.Host, result.Context);

        context.Dispose();
        Assert.Equal(expected, Floats(result));
    }

    [Fact]
    public void TestAContextThatDoesNotDetachTakesItsOutputsWithIt()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext(DefaultBackend.Instance, detachesOutputs: false);

        Assert.False(context.DetachesOutputs);
        var result = context.Execute(graph, a, b)[0].ToTensorData();
        Assert.Same(context, result.Context);

        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(result));
    }

    [Fact]
    public void TestACompiledGraphDetachesItsOutputsWhenItsContextDoes()
    {
        var (graph, a, b, expected) = Model();
        var context = new ComputeContext(DefaultBackend.Instance, detachesOutputs: true);
        var compiled = context.Compile(graph);

        var result = compiled.Execute(a, b)[0].ToTensorData();
        Assert.Same(ComputeContext.Host, result.Context);

        context.Dispose();
        Assert.Equal(expected, Floats(result));
    }

    [Fact]
    public void TestTheDefaultContextDetachesItsOutputs()
    {
        // Whatever this process ended up with, the default context is the one nobody disposes, so
        // its results must not be tied to it.
        Assert.True(ComputeContext.Default.DetachesOutputs);
    }

    [Fact]
    public void TestADisposedContextRefusesToRunBeforeItRunsAnything()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext();
        var compiled = context.Compile(graph);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() => compiled.Execute(a, b));
        Assert.Throws<ObjectDisposedException>(() => context.Execute(graph, a, b));
        Assert.Throws<ObjectDisposedException>(() => context.Eval(InputVector<float32>("a") * 2f));
    }

    [Fact]
    public void TestDisposingAContextDropsTheRuntimeValuesItsHostTensorsWereFedAs()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext();
        var onContext = (HostTensorData<float32>)a.CopyTo(context);

        context.Execute(graph, onContext, b);
        Assert.False(onContext.MaterializationsAreEmpty);

        context.Dispose();

        Assert.True(onContext.MaterializationsAreEmpty);
    }

    [Fact]
    public void TestDisposingAContextReleasesTheSessionsItCompiled()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext();
        var compiled = context.Compile(graph);
        context.Execute(graph, a, b);

        Assert.False(compiled.IsDisposed);

        context.Dispose();

        Assert.True(compiled.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => context.Compile(graph));
    }

    [Fact]
    public void TestBuildingAndExportingAModelAsksForNoComputeContextAtAll()
    {
        var module = BackendFreeNegate.ComputationGraph;
        var sample = TensorData([8L], new float[8]);
        var onnx = Path.Combine(Path.GetTempPath(), $"shorokoo-backend-free-{Guid.NewGuid():N}.onnx");

        try
        {
            var backendReads = 0;
            var reads = ComputeContext.CountInstanceReads(() =>
                backendReads = DefaultBackend.CountInstanceReads(() =>
                {
                    var concrete = module
                        .ToConcreteArchitecture(module.FromOrderedInputs([sample]))
                        .ToConcreteModel();
                    Persistence.ExportOnnx(concrete, onnx);
                }));

            Assert.Equal(0, reads);
            // Both seams, because they are two independent ways to a backend: a graph pass reaches
            // DefaultBackend.Instance without going through any compute context -- every
            // OnnxUtils.CreateTensorValue does -- so counting only the context's reads would let a
            // regression that rebuilt a literal on the default backend through at zero.
            Assert.Equal(0, backendReads);
            Assert.True(new FileInfo(onnx).Length > 0);
        }
        finally
        {
            if (File.Exists(onnx)) File.Delete(onnx);
        }

    }

    // A graph of many cheap kernels over a big-enough buffer: long enough for another thread to
    // land inside the native run, and made of enough nodes that ORT has a boundary to stop at --
    // a graph that is one kernel cannot be stopped at all.
    private const int Wide = 1 << 20;
    private const int Deep = 48;

    private static (InternalComputationGraph Graph, float[] Expected) Chain()
    {
        var a = InputVector<float32>("a");
        var y = a;
        for (int i = 0; i < Deep; i++) y = y + a;
        return (new InternalComputationGraph([a], [y]), [.. Feed().Select(v => (Deep + 1) * v)]);
    }

    private static float[] Feed() => [.. Enumerable.Range(0, Wide).Select(i => (float)(i % 7))];

    private static HostTensorData<float32> Wide32() => (HostTensorData<float32>)TensorData([(long)Wide], Feed());

    /// <summary>
    /// Shorokoo/Shorokoo#366: feeding a tensor on one thread while another disposes it, or writes
    /// to it, used to free the buffer the execution provider was reading. The run now holds what it
    /// reads, and the bytes go when it returns rather than under it.
    /// </summary>
    [Fact]
    public void TestATensorDisposedOrWrittenOnAnotherThreadStaysValidForTheRunFeedingIt()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);

        foreach (var interfere in (Action<HostTensorData<float32>>[])[
            static t => t.Dispose(),
            static t => t.AccessModifiableMemory<float>()[0] = 99f])
        {
            for (int round = 0; round < 3; round++)
            {
                var fed = Wide32();
                var other = Task.Run(() =>
                {
                    SpinWait.SpinUntil(() => !fed.MaterializationsAreEmpty, TimeSpan.FromSeconds(10));
                    interfere(fed);
                });

                var result = Floats(compiled.Execute(fed)[0].ToTensorData());
                other.Wait();

                Assert.Equal(expected, result);
            }
        }
    }

    [Fact]
    public void TestATensorDisposedWhileARunFeedsItIsFreedWhenThatRunReturns()
    {
        using var context = new ComputeContext();
        var (graph, _) = Chain();
        var compiled = context.Compile(graph);
        var fed = Wide32();

        var disposer = Task.Run(() =>
        {
            SpinWait.SpinUntil(() => !fed.MaterializationsAreEmpty, TimeSpan.FromSeconds(10));
            fed.Dispose();
            return fed.MaterializationsAreEmpty;
        });

        compiled.Execute(fed);

        Assert.False(disposer.Result);
        Assert.True(fed.MaterializationsAreEmpty);
    }

    [Fact]
    public void TestASecondHandleSurvivesTheDisposalOfTheContextTheFirstWasAttachedTo()
    {
        var first = new ComputeContext();
        var second = new ComputeContext();
        var onFirst = Sample().TransferTo(first);
        var onSecond = onFirst.GiveAccessTo(second);

        first.Dispose();

        Assert.Equal([1f, 2f, 3f, 4f], Floats(onSecond));
        second.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(onSecond));
    }

    [Fact]
    public void TestALockIsRefusedByEveryContextButTheOneTheTensorIsAttachedTo()
    {
        using var owner = new ComputeContext();
        using var other = new ComputeContext();
        var t = Sample().TransferTo(owner);

        Assert.Throws<InvalidOperationException>(() => other.Lock(t));
        using (var lease = owner.Lock(t)) Assert.False(lease.Eviction.IsCancellationRequested);

        var detached = Sample();
        Assert.Throws<InvalidOperationException>(() => owner.Lock(detached));
        using (ComputeContext.Host.Lock(detached)) Assert.False(detached.TryDelete());

        Assert.True(t.TryDelete());
        Assert.Throws<ObjectDisposedException>(() => owner.Lock(t));
    }

    [Fact]
    public void TestDisposingAContextWithALeaseOutstandingThrowsAndLeavesItUsable()
    {
        var context = new ComputeContext();
        var t = Sample().TransferTo(context);
        var lease = context.Lock(t);

        Assert.Throws<InvalidOperationException>(context.Dispose);
        Assert.False(context.IsDisposed);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(t));

        lease.Dispose();
        context.Dispose();
        Assert.True(context.IsDisposed);
    }

    [Fact]
    public void TestTryDeleteRefusesWhileALeaseIsOutstandingAndSucceedsOnceItDrops()
    {
        using var context = new ComputeContext();
        var t = Sample().TransferTo(context);
        var lease = context.Lock(t);

        Assert.False(t.TryDelete());
        Assert.False(lease.Eviction.IsCancellationRequested);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(t));

        lease.Dispose();
        Assert.True(t.TryDelete());
        Assert.True(t.TryDelete());
        Assert.Throws<ObjectDisposedException>(() => Floats(t));
    }

    [Fact]
    public async Task TestDeleteAsyncDeletesAtOnceAndReclaimsWhenTheLockDrops()
    {
        using var context = new ComputeContext();
        var a = InputVector<float32>("a");
        var graph = new InternalComputationGraph([a], [a + a]);
        var t = (HostTensorData<float32>)Sample().CopyTo(context);
        context.Execute(graph, t);
        Assert.False(t.MaterializationsAreEmpty);

        using var lease = context.Lock(t);
        Assert.False(await t.DeleteAsync(TimeSpan.FromMilliseconds(20)));

        // Deleted already, and said so: only the reclamation was still waiting.
        Assert.Throws<ObjectDisposedException>(() => Floats(t));
        Assert.True(lease.Eviction.IsCancellationRequested);
        Assert.False(t.MaterializationsAreEmpty);
        Assert.False(await t.DeleteAsync(TimeSpan.Zero));

        lease.Dispose();
        Assert.True(t.MaterializationsAreEmpty);
        Assert.True(await t.DeleteAsync(TimeSpan.Zero));
    }

    [Fact]
    public async Task TestDeleteAsyncCompletesOnceTheRunReadingTheTensorHasStopped()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);
        var fed = Wide32();

        var run = Task.Run(() => Floats(compiled.Execute(fed)[0].ToTensorData()));
        Assert.True(SpinWait.SpinUntil(() => !fed.MaterializationsAreEmpty, TimeSpan.FromSeconds(10)));

        Assert.True(await fed.DeleteAsync(TimeSpan.FromSeconds(30)));
        Assert.True(fed.MaterializationsAreEmpty);
        Assert.Throws<ObjectDisposedException>(() => Floats(fed));

        // Stopped or finished, never a wrong answer: a run that reaches its last kernel before the
        // flag is read returns what it computed, and one stopped short says so.
        var stopped = await Record.ExceptionAsync(() => run);
        if (stopped is null) Assert.Equal(expected, run.Result);
        else Assert.IsAssignableFrom<OperationCanceledException>(stopped);
    }

    [Fact]
    public void TestDisposingAContextIsRefusedForARunInFlightAsWellAsForALease()
    {
        var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();

        var run = Task.Run(() =>
            Floats(compiled.Run(new HeldFeed(Wide32(), reached, release))[0].ToTensorData()));
        Assert.True(reached.Wait(TimeSpan.FromSeconds(10)));

        Assert.Contains("in flight", Assert.Throws<InvalidOperationException>(context.Dispose).Message);
        Assert.False(context.IsDisposed);
        release.Set();
        Assert.Equal(expected, run.Result);

        var lease = context.Lock(Sample().TransferTo(context));
        Assert.Contains("lease(s)", Assert.Throws<InvalidOperationException>(context.Dispose).Message);
        lease.Dispose();
        context.Dispose();
        Assert.True(context.IsDisposed);
    }

    [Fact]
    public void TestARunOverADonatedFeedAnswersAsAKeptOneAndGivesTheBytesBackWithTheRun()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);

        var kept = Wide32();
        Assert.Equal(expected, Floats(compiled.Execute(kept)[0].ToTensorData()));
        Assert.False(kept.MaterializationsAreEmpty);

        var donated = Wide32();
        var donation = donated.Donate();
        Assert.Equal(expected, Floats(compiled.Execute(donation)[0].ToTensorData()));
        Assert.True(donated.MaterializationsAreEmpty);
        Assert.Throws<ObjectDisposedException>(() => Floats(donated));
        Assert.Throws<ObjectDisposedException>(() => compiled.Execute(donation));
    }

    [Fact]
    public void TestAnAlreadyCancelledRunIsRefusedBeforeItTakesWhatItWasFed()
    {
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        var settings = new RunSettings { CancellationToken = cancelled.Token };
        var (graph, expected) = Chain();
        using var context = new ComputeContext();
        var compiled = context.Compile(graph);

        var fed = Wide32().Donate();
        Assert.Throws<OperationCanceledException>(() => compiled.Run(
            [NamedModelParam.FromIData("a", ModelParamType.InputParam, fed)], settings));
        Assert.Equal(expected, Floats(compiled.Execute(fed)[0].ToTensorData()));

        using var stopped = new ComputeContext { RunSettings = settings };
        var oneShot = Wide32().Donate();
        Assert.Throws<OperationCanceledException>(() => stopped.Run(
            graph, new DonatedTensorModelParam("a", ModelParamType.InputParam, oneShot)));
        Assert.Equal(expected, Floats(context.Run(
            graph, new DonatedTensorModelParam("a", ModelParamType.InputParam, oneShot))[0].ToTensorData()));
    }

    [Fact]
    public void TestAFeedNothingKnowsHowToLockIsRefusedAndSoIsAnInputThatWasNotLocked()
    {
        Assert.Throws<InvalidOperationException>(() => ComputeContext.LeaseFeed(new UnlockableParam()));
        Assert.Throws<InvalidOperationException>(() => ComputeContext.RefuseUnleasedFeed(1, 2));
        ComputeContext.RefuseUnleasedFeed(2, 2);
    }

    [Fact]
    public async Task TestAnAllocationThatHoldsNothingIsNeverDeadAndNeverDeleted()
    {
        Assert.False(TensorStorage.None.TryDelete());
        Assert.False(await TensorStorage.None.DeleteAsync(TimeSpan.Zero, default));
        Assert.True(TensorStorage.None.IsLive);
    }

    [Fact]
    public void TestFeedingADonationTwiceIsRefusedInTheTensorsOwnWords()
    {
        using var context = new ComputeContext();
        var (graph, _) = Chain();
        var compiled = context.Compile(graph);
        var fed = Wide32();
        var donation = fed.Donate();
        compiled.Execute(donation);

        var refused = Assert.Throws<ObjectDisposedException>(() => compiled.Execute(donation));

        Assert.Contains(fed.ToString(), refused.Message);
        Assert.DoesNotContain(nameof(TensorStorage), refused.Message);
    }

    /// <summary>A parameter of a kind no lock knows about, which is what <c>LeaseFeed</c>'s
    /// refusal is for.</summary>
    private sealed class UnlockableParam : NamedModelParam
    {
        public override IShorokooTensorValue ToTensorValue() => throw new NotSupportedException();
        public override TensorData ToTensorData() => throw new NotSupportedException();
        public override TensorData<T> ToTensorData<T>() => throw new NotSupportedException();
        public override TensorDataSequence ToTensorDataSequence() => throw new NotSupportedException();
        public override TensorDataSequence<T> ToTensorDataSequence<T>() => throw new NotSupportedException();
    }

    // A context lists the tensors attached to it, so a handle the caller has let go of must come
    // off. Disposal used to leave it on, and the list then answered which tensors had ever been
    // attached rather than which are -- a graph literal moved into an attribute is disposed by the
    // move, so describing a graph alone grew it.
    [Fact]
    public void TestATensorComesOffItsContextsListWhenItIsDisposedOrMovedAway()
    {
        using var context = new ComputeContext();

        var held = context.AllocateUninitialized<float32>((long[])[2L]);
        Assert.Contains(held, context.Tensors);

        var spent = (TensorData<float32>)TensorData([2L], 1f, 2f);
        Assert.Contains(spent, ComputeContext.Host.Tensors);
        _ = spent.MoveToAttribute();
        Assert.DoesNotContain(spent, ComputeContext.Host.Tensors);

        held.Dispose();
        Assert.DoesNotContain(held, context.Tensors);
    }

    [Fact]
    public void TestAnAllocatedTensorIsFilledInPlaceAndFeedsAsACopiedOneDoes()
    {
        using var context = new ComputeContext();
        var a = InputVector<float32>("a");
        var graph = new InternalComputationGraph([a], [a + a]);
        float[] values = [1f, 2f, 3f, 4f];

        var allocated = context.AllocateUninitialized<float32>((long[])[4L]);
        values.CopyTo(allocated.AccessModifiableMemory<float>());

        Assert.Same(context, allocated.Context);
        Assert.Equal(values, Floats(allocated));
        Assert.Equal(Floats(Sample().CopyTo(context)), Floats(allocated));
        Assert.Equal(
            Floats(context.Execute(graph, Sample().CopyTo(context))[0].ToTensorData()),
            Floats(context.Execute(graph, allocated)[0].ToTensorData()));

        long[] pair = [2L, 3L];
        Assert.Equal(DType.Float32, context.AllocateUninitialized(pair, DType.Float32).DType);
        Assert.Equal(new Shape(pair), context.AllocateUninitialized(pair, DType.Float32).Shape);
        Assert.Equal(24, ComputeContext.Host.AllocateUninitialized(pair, DType.Float32).CopyRawMemory().Length);
        Assert.Throws<NotSupportedException>(() => context.AllocateUninitialized(pair, DType.Utf8));
        Assert.Throws<ArgumentNullException>(() => context.AllocateUninitialized(pair, null!));
    }

    [Fact]
    public void TestADonatedFeedFreesItsBytesWhileAKeptOneStillReads()
    {
        using var context = new ComputeContext();
        var a = InputVector<float32>("a");
        var graph = new InternalComputationGraph([a], [a + a]);

        var donated = (HostTensorData<float32>)Sample();
        context.Run(graph, new DonatedTensorModelParam("a", ModelParamType.InputParam, donated.Donate()));
        Assert.True(donated.MaterializationsAreEmpty);

        var shared = (HostTensorData<float32>)Sample();
        using var reader = shared.GiveAccessTo(context);
        context.Execute(graph, shared.Donate());
        Assert.False(shared.MaterializationsAreEmpty);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(reader));
    }

    /// <summary>Holds a run open where the value is built: inside the window a disposal has to be
    /// refused in, and leased on <c>Host</c> rather than on the running context.</summary>
    private sealed class HeldFeed(
        TensorData data, ManualResetEventSlim reached, ManualResetEventSlim release)
        : TensorDataModelParam("a", ModelParamType.InputParam, data)
    {
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            reached.Set();
            release.Wait(TimeSpan.FromSeconds(30));
            return base.ToTensorValue(backend);
        }
    }

    internal sealed class StubBackend(ComputeDevice device, int? cudaDeviceId)
        : IShorokooBackend
    {
        public BackendDescription Description { get; } =
            new($"stub-{device}", device, cudaDeviceId);

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory) => throw new NotSupportedException();

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => throw new NotSupportedException();
    }
}

/// <summary>
/// Which backend the process is on, and which one it remembered. Both are process-wide, and the
/// default context caches what it resolves from them, so these run alone.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
[Collection(ProcessWideBackend.Name)]
public class ProcessWideBackendCoverageTests
{
    [Fact]
    public void TestACpuBackendWinsTheRememberedSlotAndAGpuOneDoesNotTakeItBack()
    {
        var live = DefaultBackend.Remembered;
        try
        {
            DefaultBackend.ForgetRemembered();
            var gpu = new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cuda, 0);
            var cpu = new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cpu, null);

            DefaultBackend.Remember(gpu);
            Assert.Same(gpu, DefaultBackend.Remembered);

            // The CPU one displaces it...
            DefaultBackend.Remember(cpu);
            Assert.Same(cpu, DefaultBackend.Remembered);

            // ...and is not displaced back, by that GPU backend or another.
            DefaultBackend.Remember(gpu);
            DefaultBackend.Remember(new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cuda, 1));
            Assert.Same(cpu, DefaultBackend.Remembered);

            // Nor by a second CPU one: the first backend loaded is the one that counts.
            DefaultBackend.Remember(new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cpu, null));
            Assert.Same(cpu, DefaultBackend.Remembered);
        }
        finally
        {
            DefaultBackend.ForgetRemembered();
            if (live is not null) DefaultBackend.Remember(live);
        }
    }

    /// <summary>
    /// Assigning <see cref="DefaultBackend.Instance"/> settles both slots outright — the live one
    /// and the remembered one — rather than going through the first-CPU-wins rule that governs a
    /// backend the process merely loaded. It does not say anything about
    /// <see cref="ComputeContext.Default"/>, which caches the backend it resolves on first read and
    /// keeps it: a program that means the default to follow an assignment has to make the
    /// assignment before anything reads Default.
    /// </summary>
    [Fact]
    public void TestAssigningTheDefaultBackendSettlesBothTheLiveAndTheRememberedSlot()
    {
        // Default rather than Current, which is null until something resolves one: capturing null
        // here and restoring nothing in the finally left this test's throwing stub as the process's
        // backend, and every later test that ran anything failed inside it.
        var liveDefault = DefaultBackend.Instance;
        var liveRemembered = DefaultBackend.Remembered;
        try
        {
            DefaultBackend.ForgetRemembered();
            DefaultBackend.Remember(new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cpu, null));

            var named = new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cuda, 0);
            DefaultBackend.Instance = named;

            Assert.Same(named, DefaultBackend.Remembered);
            Assert.Same(named, DefaultBackend.Current);
        }
        finally
        {
            DefaultBackend.ForgetRemembered();
            DefaultBackend.Instance = liveDefault;
            DefaultBackend.ForgetRemembered();
            if (liveRemembered is not null) DefaultBackend.Remember(liveRemembered);
        }
    }
}
