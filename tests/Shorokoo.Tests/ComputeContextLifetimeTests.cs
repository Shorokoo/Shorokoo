using System.Runtime.CompilerServices;
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
/// A compute context keeps a weak list of the tensors attached to it and owns none of them: its
/// disposal releases its sessions and nothing else. A tensor's life is its own — deleted, consumed
/// by a run it was donated to, or moved into an attribute — and a run holds what it reads.
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

    private static InternalComputationGraph Doubling()
    {
        var a = InputVector<float32>("a");
        return new InternalComputationGraph([a], [a + a]);
    }

    [Fact]
    public void TestDisposingAContextReleasesItsSessionsAndLeavesEveryTensorAttachedToItAlive()
    {
        var (graph, a, b, expected) = Model();
        var context = new ComputeContext();
        var compiled = context.Compile(graph);
        var placed = Sample().To(context);
        var copied = Sample().CopyTo(context);
        var output = compiled.Execute(a, b)[0].ToTensorData();
        var allocated = context.AllocateUninitialized<float32>(new Shape(4L));

        context.Dispose();

        Assert.True(compiled.IsDisposed);
        Assert.Empty(context.Tensors);
        Assert.All((TensorData[])[placed, copied, output, allocated, a, b], t => Assert.False(t.IsDisposed));
        Assert.Equal([1f, 2f, 3f, 4f], Floats(placed));
        Assert.Equal([1f, 2f, 3f, 4f], Floats(copied));
        Assert.Equal(expected, Floats(output));
        Assert.Throws<ObjectDisposedException>(() => Sample().To(context));
        Assert.Throws<ObjectDisposedException>(() => Sample().CopyTo(context));
    }

    [Fact]
    public void TestDisposingAContextTwiceIsHarmless()
    {
        var context = new ComputeContext();
        _ = Sample().To(context);
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
    public void TestTheHostContextCannotBeDisposedAndKeepsNoList()
    {
        var detached = Sample();

        ComputeContext.Host.Dispose();
        ComputeContext.Host.Dispose();

        Assert.False(ComputeContext.Host.IsDisposed);
        Assert.Same(detached, detached.To(ComputeContext.Host));
        Assert.NotSame(detached, detached.CopyTo(ComputeContext.Host));
        Assert.Empty(ComputeContext.Host.Tensors);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(detached));
    }

    [Fact]
    public void TestAContextListsWhatIsAttachedToItUntilItIsDetachedOrDies()
    {
        using var first = new ComputeContext();
        using var second = new ComputeContext();
        var t = Sample();

        Assert.Same(t, t.To(first));
        Assert.Contains(t, first.Tensors);
        Assert.DoesNotContain(t, second.Tensors);

        Assert.Same(t, t.To(second));
        Assert.Contains(t, first.Tensors);
        Assert.Contains(t, second.Tensors);

        first.Detach(t);
        first.Detach(t);
        Assert.DoesNotContain(t, first.Tensors);
        Assert.Contains(t, second.Tensors);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(t));

        t.Delete();
        Assert.DoesNotContain(t, second.Tensors);
        Assert.Throws<ArgumentNullException>(() => first.Detach(null!));
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
    public void TestARunsOutputsAndWhatItReadsAreAttachedToItsContextAndOutliveIt()
    {
        var (graph, a, b, expected) = Model();
        var context = new ComputeContext();
        var compiled = context.Compile(graph);

        var oneShot = context.Execute(graph, a, b)[0].ToTensorData();
        var fromCompiled = compiled.Execute(a, b)[0].ToTensorData();
        var evaluated = context.Eval(Scalar(2f) * Scalar(3f));

        Assert.All((TensorData[])[oneShot, fromCompiled, evaluated, a, b], t => Assert.Contains(t, context.Tensors));
        Assert.Same(DefaultBackend.Instance, oneShot.AllocatingBackend);
        Assert.Same(DefaultBackend.Instance, fromCompiled.AllocatingBackend);
        Assert.Same(HostBackend.Instance, a.AllocatingBackend);

        context.Dispose();

        Assert.Equal(expected, Floats(oneShot));
        Assert.Equal(expected, Floats(fromCompiled));
        Assert.Equal(6f, evaluated.As<float32>().ValueAt<float>(0));
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
    public void TestDeletingATensorReleasesTheRuntimeValuesItWasFedAsAndDisposingItsContextDoesNot()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext();
        var fed = (HostTensorData<float32>)a.CopyTo(context);

        context.Execute(graph, fed, b);
        Assert.False(fed.MaterializationsAreEmpty);

        context.Dispose();
        Assert.False(fed.MaterializationsAreEmpty);

        fed.Delete();
        Assert.True(fed.MaterializationsAreEmpty);
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
    /// Shorokoo/Shorokoo#366: feeding a tensor on one thread while another deletes it, or writes to
    /// it, used to free the buffer the execution provider was reading. The run holds what it reads,
    /// so the delete is declined and the write's retired copies wait for the run.
    /// </summary>
    [Fact]
    public void TestATensorDeletedOrWrittenOnAnotherThreadStaysValidForTheRunFeedingIt()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);

        foreach (var interfere in (Action<HostTensorData<float32>>[])[
            static t => t.TryDelete(),
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
    public void TestATensorARunIsReadingCannotBeDeletedMovedOrDetachedByThatRunsContext()
    {
        using var context = new ComputeContext();
        using var other = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);
        var fed = Wide32();
        Assert.Same(fed, fed.To(other));
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();

        var run = Task.Run(() => Floats(compiled.Run(new HeldFeed(fed, reached, release))[0].ToTensorData()));
        Assert.True(reached.Wait(TimeSpan.FromSeconds(10)));

        Assert.Throws<InvalidOperationException>(fed.Delete);
        Assert.Throws<InvalidOperationException>(fed.Dispose);
        Assert.False(fed.TryDelete());
        Assert.Throws<InvalidOperationException>(() => fed.MoveToAttribute());
        Assert.Throws<InvalidOperationException>(() => context.Detach(fed));
        other.Detach(fed);
        Assert.False(fed.IsDisposed);

        release.Set();
        Assert.Equal(expected, run.Result);
        context.Detach(fed);
        Assert.True(fed.TryDelete());
    }

    [Fact]
    public void TestAnyContextMayLockATensorAndNoneMayDetachWhatItHoldsALockOn()
    {
        using var owner = new ComputeContext();
        using var other = new ComputeContext();
        var t = Sample();

        using (owner.Lock(t))
        using (var second = other.Lock(t))
        {
            Assert.False(second.Eviction.IsCancellationRequested);
            Assert.False(t.TryDelete());
            Assert.Throws<InvalidOperationException>(() => owner.Detach(t));
            Assert.Throws<InvalidOperationException>(() => other.Detach(t));
            Assert.Contains(t, owner.Tensors);
        }

        owner.Detach(t);
        Assert.True(t.TryDelete());
        Assert.Throws<ObjectDisposedException>(() => owner.Lock(t));
    }

    [Fact]
    public void TestARunsLockHoldsTheTensorItselfAliveForAsLongAsItIsHeld()
    {
        using var context = new ComputeContext();
        var (lease, tensor) = LockedAndDropped(context);

        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();

        Assert.True(tensor.IsAlive);
        lease.Dispose();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (TensorLease Lease, WeakReference Tensor) LockedAndDropped(ComputeContext context)
    {
        var tensor = Sample();
        return (context.Lock(tensor), new WeakReference(tensor));
    }

    [Fact]
    public void TestDisposingAContextWithALeaseOutstandingThrowsAndLeavesItUsable()
    {
        var context = new ComputeContext();
        var t = Sample().To(context);
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
        var t = Sample().To(context);
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
        var t = (HostTensorData<float32>)Sample().CopyTo(context);
        context.Execute(Doubling(), t);
        Assert.False(t.MaterializationsAreEmpty);

        using var lease = context.Lock(t);
        Assert.False(await t.DeleteAsync(TimeSpan.FromMilliseconds(20)));

        // Deleted already, and said so: only the reclamation was still waiting.
        Assert.Contains("deleted", Assert.Throws<ObjectDisposedException>(() => Floats(t)).Message);
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

        var lease = context.Lock(Sample());
        Assert.Contains("lock(s)", Assert.Throws<InvalidOperationException>(context.Dispose).Message);
        lease.Dispose();
        context.Dispose();
        Assert.True(context.IsDisposed);
    }

    [Fact]
    public void TestARunConsumesADonatedFeedWhenItStartsAndGivesItsMemoryBackWhenItReturns()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);

        var kept = Wide32();
        Assert.Equal(expected, Floats(compiled.Execute(kept)[0].ToTensorData()));
        Assert.False(kept.IsDisposed);
        Assert.False(kept.MaterializationsAreEmpty);

        var donated = Wide32();
        var donation = donated.Donate();
        Assert.False(donated.IsDisposed);
        Assert.Equal(expected, Floats(compiled.Execute(donation)[0].ToTensorData()));
        Assert.True(donated.IsDisposed);
        Assert.True(donated.MaterializationsAreEmpty);
        Assert.DoesNotContain(donated, context.Tensors);
        Assert.Throws<ObjectDisposedException>(() => Floats(donated));
        Assert.Throws<ObjectDisposedException>(() => compiled.Execute(donation));
        donation.Dispose();
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
    public void TestAFeedNothingKnowsHowToHoldIsRefusedAndSoIsAnInputThatWasNotHeld()
    {
        using var context = new ComputeContext();
        using var feeds = new RunFeeds(context, 1, () => TensorDeath.Deleted);

        Assert.Throws<InvalidOperationException>(
            () => feeds.Feed(new UnlockableParam(), DefaultBackend.Instance));
        Assert.Throws<InvalidOperationException>(() => ComputeContext.RefuseUnleasedFeed(1, 2));
        ComputeContext.RefuseUnleasedFeed(2, 2);
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

        var refused = Assert.Throws<ObjectDisposedException>(() => compiled.Execute(donation)).Message;

        Assert.Contains(fed.ToString(), refused);
        Assert.Contains("consumed by a run of", refused);
        Assert.Contains(context.Backend.ToString(), refused);
        Assert.Contains("Donate()", refused);
    }

    [Fact]
    public void TestADonationIsRefusedWhileAnotherRunReadsTheTensorAndTakesNothing()
    {
        using var context = new ComputeContext();
        var shared = (HostTensorData<float32>)Sample();

        using (context.Lock(shared))
            Assert.Contains("another run is reading it", Assert.Throws<InvalidOperationException>(
                () => context.Execute(Doubling(), shared.Donate())).Message);

        Assert.False(shared.IsDisposed);
        Assert.Equal([2f, 4f, 6f, 8f], Floats(context.Execute(Doubling(), shared.Donate())[0].ToTensorData()));
        Assert.True(shared.IsDisposed);
        Assert.True(shared.MaterializationsAreEmpty);
    }

    [Fact]
    public void TestEveryAccessToADeadTensorThrowsSayingHowItDiedAndEndingItAgainIsHarmless()
    {
        using var context = new ComputeContext();
        TensorData Deleted() { var t = Sample(); t.Delete(); return t; }
        TensorData Moved() { var t = Sample(); _ = t.MoveToAttribute(); return t; }
        TensorData Consumed() { var t = Sample(); context.Execute(Doubling(), t.Donate()); return t; }
        Func<TensorData, object>[] accesses =
        [
            t => t.CopyRawMemory(), t => t.Data, t => t.IsHostResident, t => t.ToTensorValue(),
            t => t.To(context), t => t.CopyTo(context), t => t.ToHost(), t => t.Donate(),
            t => t.MoveToAttribute(), t => context.Lock(t),
        ];

        foreach (var (dead, cause) in (IEnumerable<(TensorData, string)>)[
            (Deleted(), "deleted"), (Moved(), "MoveToAttribute()"), (Consumed(), "Donate()")])
        {
            Assert.True(dead.IsDisposed);
            Assert.All(accesses, access => Assert.Contains(cause,
                Assert.Throws<ObjectDisposedException>(() => access(dead)).Message));
            dead.Dispose();
            dead.Delete();
            Assert.True(dead.TryDelete());
            Assert.Equal(new Shape(4L), dead.Shape);
            Assert.Equal(DType.Float32, dead.DType);
        }
    }

    [Fact]
    public void TestATensorComesOffItsContextsListWhenItDies()
    {
        using var context = new ComputeContext();

        var deleted = context.AllocateUninitialized<float32>((long[])[2L]);
        var moved = Sample().To(context);
        Assert.Contains(deleted, context.Tensors);
        Assert.Contains(moved, context.Tensors);

        deleted.Dispose();
        _ = moved.MoveToAttribute();

        Assert.DoesNotContain(deleted, context.Tensors);
        Assert.DoesNotContain(moved, context.Tensors);
    }

    [Fact]
    public void TestAnAllocatedTensorIsFilledInPlaceAndFeedsAsACopiedOneDoes()
    {
        using var context = new ComputeContext();
        var graph = Doubling();
        float[] values = [1f, 2f, 3f, 4f];

        var allocated = context.AllocateUninitialized<float32>((long[])[4L]);
        values.CopyTo(allocated.AccessModifiableMemory<float>());

        Assert.Contains(allocated, context.Tensors);
        Assert.Same(DefaultBackend.Instance, allocated.AllocatingBackend);
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

    /// <summary>Holds a run open where the value is built: after the run has taken its lock on the
    /// feed, and inside the window a disposal of its context has to be refused in.</summary>
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

    /// <summary>A parameter of a kind no run knows how to hold, which is what
    /// <c>RunFeeds.Feed</c>'s refusal is for.</summary>
    private sealed class UnlockableParam : NamedModelParam
    {
        public override IShorokooTensorValue ToTensorValue() => throw new NotSupportedException();
        public override TensorData ToTensorData() => throw new NotSupportedException();
        public override TensorData<T> ToTensorData<T>() => throw new NotSupportedException();
        public override TensorDataSequence ToTensorDataSequence() => throw new NotSupportedException();
        public override TensorDataSequence<T> ToTensorDataSequence<T>() => throw new NotSupportedException();
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
