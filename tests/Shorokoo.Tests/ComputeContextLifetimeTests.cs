using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.OnnxRuntime;
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
/// by a run it was fed to as it is, or moved into an attribute — and a run holds what it reads.
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
        var output = compiled.Execute(a.Shared(), b.Shared())[0].ToTensorData();

        context.Dispose();

        Assert.True(compiled.IsDisposed);
        Assert.Empty(context.Tensors);
        Assert.All((TensorData[])[placed, copied, output, a, b], t => Assert.False(t.IsDisposed));
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
    public void TestABackendOnAnUnnamedDeviceReadsItsOwnAllocationsThereAndNoOtherBackends()
    {
        IShorokooBackend backend = new StubBackend(ComputeDevice.Other, null);
        IShorokooBackend other = new StubBackend(ComputeDevice.Other, null);
        var own = new MemoryLocation(MemorySpace.UnknownDevice, backend.RuntimeIdentity);

        Assert.True(backend.CanAddress(own));
        Assert.False(other.CanAddress(own));
        Assert.False(backend.CanAddress(new MemoryLocation(MemorySpace.UnknownDevice, other.RuntimeIdentity)));
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

        var oneShot = context.Execute(graph, a.Shared(), b.Shared())[0].ToTensorData();
        var fromCompiled = compiled.Execute(a.Shared(), b.Shared())[0].ToTensorData();
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
        Assert.False(a.IsDisposed || b.IsDisposed);
    }

    [Fact]
    public void TestDeletingATensorReleasesTheCopiesItWasReadThroughAndDisposingItsContextDoesNot()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext();
        var fed = (HostTensorData<float32>)a.CopyTo(context);

        context.Execute(graph, fed.Shared(), b);
        Assert.False(fed.CopiesAreEmpty);

        context.Dispose();
        Assert.False(fed.CopiesAreEmpty);

        fed.Delete();
        Assert.True(fed.CopiesAreEmpty);
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
        Assert.Throws<ObjectDisposedException>(() => compiled.OutputPlacement);
    }

    [Fact]
    public void TestAGraphThatHandsAnInputBackAsItsOutputGivesTheOutputMemoryOfItsOwn()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        var passthrough = context.Compile(new InternalComputationGraph([x], [x]), [[4L]], trainingStep: false);
        var source = Sample();
        var output = passthrough.Execute(source.Shared())[0].ToTensorData();
        var copy = source.CopyHeldAt(TensorData.RunMemoryOf(context.ResolvedBackend, source.DType))!;

        Assert.False(SameMemory(output, copy));
        Assert.Equal([1f, 2f, 3f, 4f], Floats(output));

        var doubled = x * 2f;
        var twice = context.Compile(new InternalComputationGraph([x], [doubled, doubled]), [[4L]], trainingStep: false)
            .Execute(Sample());
        Assert.False(SameMemory(twice[0].ToTensorData(), twice[1].ToTensorData()));
        Assert.Equal([2f, 4f, 6f, 8f], Floats(twice[1].ToTensorData()));
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
                        .ToConcreteArchitecture([sample])
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

    [Fact]
    public void TestATensorARunConsumedKeepsNeitherTheGraphNorTheContextThatRanItAlive()
    {
        var (graph, context, consumed) = ConsumedByAGraphNobodyHolds();

        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);

        Assert.False(graph.IsAlive);
        Assert.False(context.IsAlive);
        Assert.Contains("consumed by a run of the graph (a) -> (",
            Assert.Throws<ObjectDisposedException>(() => Floats(consumed)).Message);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (WeakReference Graph, WeakReference Context, TensorData Consumed) ConsumedByAGraphNobodyHolds()
    {
        var context = new ComputeContext();
        var compiled = context.Compile(Doubling());
        var consumed = Sample();
        compiled.Execute(consumed);
        return (new WeakReference(compiled), new WeakReference(context), consumed);
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

        var run = Task.Run(() => Floats(compiled.Run(
            new HeldFeed(fed, reached, release) { FeedMode = SharedInputMode.Shared })[0].ToTensorData()));
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
        context.Execute(Doubling(), t.Shared());
        Assert.False(t.CopiesAreEmpty);

        using var lease = context.Lock(t);
        Assert.False(await t.DeleteAsync(TimeSpan.FromMilliseconds(20)));

        // Deleted already, and said so: only the reclamation was still waiting.
        Assert.Contains("deleted", Assert.Throws<ObjectDisposedException>(() => Floats(t)).Message);
        Assert.True(lease.Eviction.IsCancellationRequested);
        Assert.False(t.CopiesAreEmpty);
        Assert.False(await t.DeleteAsync(TimeSpan.Zero));

        lease.Dispose();
        Assert.True(t.CopiesAreEmpty);
        Assert.True(await t.DeleteAsync(TimeSpan.Zero));
    }

    [Fact]
    public async Task TestDeleteAsyncCompletesOnceTheRunReadingTheTensorHasStopped()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);
        var fed = Wide32();

        var run = Task.Run(() => Floats(compiled.Execute(fed.Shared())[0].ToTensorData()));
        Assert.True(SpinWait.SpinUntil(() => !fed.CopiesAreEmpty, TimeSpan.FromSeconds(10)));

        Assert.True(await fed.DeleteAsync(TimeSpan.FromSeconds(30)));
        Assert.True(fed.CopiesAreEmpty);
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
    public void TestABareFeedIsConsumedASharedOneReadAndATriedOneConsumedUnlessAnotherRunReadsItWhenTheRunStarts()
    {
        using var context = new ComputeContext();
        using var other = new ComputeContext();
        bool Survives(Func<TensorData, IData> feed, bool readWhenCalled = false, bool readWhenRun = false)
        {
            var t = Sample();
            IData fed;
            using (readWhenCalled ? other.Lock(t) : null) fed = feed(t);
            using (readWhenRun ? other.Lock(t) : null)
                Assert.Equal([2f, 4f, 6f, 8f], Floats(context.Execute(Doubling(), fed)[0].ToTensorData()));
            var survived = !t.IsDisposed;
            Assert.Equal(survived, context.Tensors.Contains(t));
            return survived;
        }

        Assert.False(Survives(t => t));
        Assert.True(Survives(t => t.Shared()));
        Assert.False(Survives(t => t.TryConsume()));
        Assert.True(Survives(t => t.TryConsume(), readWhenRun: true));
        Assert.False(Survives(t => t.TryConsume(), readWhenCalled: true));
        Assert.True(Survives(t => t.Shared(), readWhenRun: true));

        var named = Sample();
        context.Run(Doubling(), NamedModelParam.FromIData("a", ModelParamType.InputParam, named.Shared()));
        Assert.False(named.IsDisposed);
        context.Run(Doubling(), NamedModelParam.FromIData("a", ModelParamType.InputParam, named));
        Assert.True(named.IsDisposed);
    }

    [Fact]
    public void TestAParameterPassedSharedIsReadByEveryRunAsTheMessageOfOneConsumedAsItIsAsks()
    {
        using var context = new ComputeContext();
        using var other = new ComputeContext();
        var compiled = context.Compile(Doubling());
        var p = new TensorDataModelParam("a", ModelParamType.InputParam, Sample());
        var tried = new TensorDataModelParam("a", ModelParamType.InputParam, Sample());

        Assert.Equal([2f, 4f, 6f, 8f], Floats(compiled.Run(p.Shared())[0].ToTensorData()));
        Assert.Equal([2f, 4f, 6f, 8f], Floats(compiled.Run(p.Shared())[0].ToTensorData()));
        Assert.False(p.ToTensorData().IsDisposed);
        Assert.Null(p.FeedMode);
        compiled.Run(p);
        Assert.Contains("pass it there as .Shared()", Assert.Throws<ObjectDisposedException>(() => compiled.Run(p)).Message);

        using (other.Lock(tried.ToTensorData())) compiled.Run(tried.TryConsume());
        Assert.False(tried.ToTensorData().IsDisposed);
        compiled.Run(tried.TryConsume());
        Assert.True(tried.ToTensorData().IsDisposed);
    }

    [Fact]
    public void TestAStructsFieldIsFedAsItWasGivenAndReadWhereItOrItsStructIsShared()
    {
        using var context = new ComputeContext();
        var (graph, _, _, _) = Model();
        TensorStructFieldDef[] fields =
            [new TensorStructFieldDef("a", DataStructure.Tensor, 1, DType.Float32), new TensorStructFieldDef("b", DataStructure.Tensor, 1, DType.Float32)];
        var def = new TensorStructDef(fields, "Pair");
        (bool A, bool B) Consumed(Func<TensorDataStruct, IData> feed, Func<TensorData, IData> b)
        {
            var pair = def.FromOrderedData(Sample(), b(Sample()));
            Assert.Equal([2f, 6f, 12f, 20f], Floats(context.Execute(graph, feed(pair))[0].ToTensorData()));
            return (((TensorData)pair[0]).IsDisposed, ((TensorData)pair[1]).IsDisposed);
        }

        Assert.Equal<(bool, bool)>([(true, false), (false, false), (true, false), (true, true), (true, true)], [
            Consumed(s => s, b => b.Shared()),
            Consumed(s => s.Shared(), b => b),
            Consumed(s => s.TryConsume(), b => b.Shared()),
            Consumed(s => s, b => b.TryConsume()),
            Consumed(s => s, b => b),
        ]);
    }

    [Fact]
    public void TestTheSameTensorFedTwiceInOneCallIsReadIfAnyOccurrenceIsSharedElseFedAsABareOneIfAnyIsBare()
    {
        using var context = new ComputeContext();
        using var other = new ComputeContext();
        var graph = Model().Graph;
        bool Survives(Func<TensorData, IData> first, Func<TensorData, IData> second, bool readElsewhere = false)
        {
            var t = Sample();
            using (readElsewhere ? other.Lock(t) : null)
                Assert.Equal([2f, 6f, 12f, 20f], Floats(context.Execute(graph, first(t), second(t))[0].ToTensorData()));
            var survived = !t.IsDisposed;
            Assert.Equal(survived, context.Tensors.Contains(t));
            return survived;
        }
        string RefusedWhileReadElsewhere(Func<TensorData, IData> first, Func<TensorData, IData> second)
        {
            var t = Sample();
            using var read = other.Lock(t);
            var refusal = Assert.Throws<InvalidOperationException>(() => context.Execute(graph, first(t), second(t))).Message;
            Assert.False(t.IsDisposed);
            return refusal;
        }

        Assert.False(Survives(t => t, t => t));
        Assert.True(Survives(t => t, t => t.Shared()));
        Assert.True(Survives(t => t.Shared(), t => t));
        Assert.True(Survives(t => t.Shared(), t => t, readElsewhere: true));
        Assert.True(Survives(t => t.TryConsume(), t => t.Shared()));
        Assert.True(Survives(t => t.Shared(), t => t.TryConsume()));
        Assert.False(Survives(t => t.TryConsume(), t => t));
        Assert.False(Survives(t => t.TryConsume(), t => t.TryConsume()));
        Assert.True(Survives(t => t.TryConsume(), t => t.TryConsume(), readElsewhere: true));
        Assert.Contains("is being read by", RefusedWhileReadElsewhere(t => t, t => t));
        Assert.Equal(RefusedWhileReadElsewhere(t => t, t => t), RefusedWhileReadElsewhere(t => t.TryConsume(), t => t));
        Assert.Equal(RefusedWhileReadElsewhere(t => t, t => t), RefusedWhileReadElsewhere(t => t, t => t.TryConsume()));
    }

    /// <summary>Which of <paramref name="pairs"/> — (output, input) over inputs a and b, output 0
    /// into a where none is named — the lowered graph of <paramref name="outputs"/> proves.</summary>
    private static (int Output, int Input)[] Marked(
        Func<Tensor<float32>, Tensor<float32>, Variable[]> outputs, params (int Output, int Input)[] pairs)
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = FastOnnxModelBuilder.BuildInternalOnnxModel(
            new InternalComputationGraph([a, b], [.. outputs(a, b)]), prepForOnnx: true).Graph;
        if (pairs.Length == 0) pairs = [(0, 0)];
        OutputAlias Named((int Output, int Input) p) => new(graph.Outputs[p.Output].Name, graph.Inputs[p.Input].Name);
        var proven = OutputAliasProof.Prove(graph, pairs.Select(Named));
        return [.. pairs.Where(p => proven.Contains(Named(p)))];
    }

    [Fact]
    public void TestAnOutputIsMarkedToBeWrittenIntoAnInputOnlyWhereNothingReadsTheInputAfterItIsWritten()
    {
        Assert.Equal([(0, 0)], Marked((a, b) => [a - b]));
        Assert.Equal([(0, 0)], Marked((a, b) => [a * 0.5f + b]));
        Assert.Equal([(0, 0)], Marked((a, b) => [a - (Tensor<float32>)OnnxOp.ReduceSum(a, keepdims: false)]));
        Assert.Equal([(0, 0)], Marked((a, b) => { var twice = a * 2f; return [twice, twice + b]; }, (0, 0), (1, 0)));
        Assert.Empty(Marked((a, b) => [b - a]));
        Assert.Empty(Marked((a, b) => [a - b, a * b]));
        Assert.Empty(Marked((a, b) => [a - b, (Tensor<float32>)OnnxOp.Flatten(a) * 2f]));
        Assert.Empty(Marked((a, b) => [a - b, OnnxOp.Flatten(a)]));
        Assert.Empty(Marked((a, b) => { var t = a * 2f; return [b + (Tensor<float32>)OnnxOp.Cast(OnnxOp.Size(t), null, DType.Float32), t]; }));
        Assert.Empty(Marked((a, b) => [OnnxOp.CumSum(a, Scalar(0L), exclusive: false, reverse: false)]));
        Assert.Empty(Marked((a, b) => [OnnxOp.Identity(a, rank: 1)]));
    }

    /// <summary>A graph written as a runtime hands one back: inputs and outputs by name, typed where
    /// the name says so (<c>a:float[4]</c>).</summary>
    internal static GraphProto GraphOf(string inputs, string outputs, params NodeProto[] nodes)
    {
        var graph = new GraphProto();
        graph.Inputs.AddRange(Names(inputs).Select(Info));
        graph.Outputs.AddRange(Names(outputs).Select(Info));
        graph.Nodes.AddRange(nodes);
        return graph;
    }

    private static string[] Names(string names) => names.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    private static ValueInfoProto Info(string spec)
    {
        var (name, type) = (spec.Split(':')[0], spec.Split(':').ElementAtOrDefault(1));
        if (type is null) return new ValueInfoProto { Name = name };
        var tensor = new TypeProto.Tensor
        {
            ElemType = (int)Enum.Parse<TensorProto.DataType>(type[..type.IndexOf('[')], ignoreCase: true),
            Shape = new TensorShapeProto(),
        };
        tensor.Shape.Dims.AddRange(type[(type.IndexOf('[') + 1)..^1].Split(',')
            .Select(d => new TensorShapeProto.Dimension { DimValue = long.Parse(d) }));
        return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = tensor } };
    }

    internal static NodeProto Op(string op, string inputs, string outputs, string domain = "", GraphProto? body = null, (string Name, long Value)? attribute = null)
    {
        var node = new NodeProto { OpType = op, Domain = domain };
        node.Inputs.AddRange(Names(inputs));
        node.Outputs.AddRange(Names(outputs));
        if (body is not null)
            node.Attributes.Add(new AttributeProto { Name = "then_branch", Type = AttributeProto.AttributeType.Graph, G = body });
        if (attribute is { } a)
            node.Attributes.Add(new AttributeProto { Name = a.Name, Type = AttributeProto.AttributeType.Int, I = a.Value });
        return node;
    }

    private static GraphProto Initializing(string name, GraphProto graph)
    {
        graph.Initializers.Add(new TensorProto { Name = name });
        return graph;
    }

    /// <summary>Whether <paramref name="graph"/> proves its output O written into its input
    /// <paramref name="input"/>.</summary>
    private static bool Proves(GraphProto graph, string input = "a")
        => OutputAliasProof.Prove(graph, [new OutputAlias("O", input)]).Count == 1;

    [Fact]
    public void TestAWrittenGraphRefusesAnAliasWhereARuntimeAliasASubgraphOrADataReadingShapeStillReadsTheInputOrTheTypesDiffer()
    {
        var norm = Op("BatchNormalization", "X scale B mean var", "Y rm rv");
        Assert.False(Proves(GraphOf("X scale B mean var two zero", "O Z Y", norm, Op("Mul", "rm two", "O"), Op("Add", "rm zero", "Z")), "mean"));
        Assert.True(Proves(GraphOf("X scale B mean var two", "O Y", norm, Op("Mul", "rm two", "O")), "mean"));
        Assert.False(Proves(GraphOf("a b", "O S", Op("Shape", "a", "S", domain: "custom"), Op("Sub", "a b", "O"))));
        Assert.True(Proves(GraphOf("a b", "O S", Op("Shape", "a", "S"), Op("Sub", "a b", "O"))));
        Assert.False(Proves(GraphOf("a b c", "O Z", Op("Sub", "a b", "O"), Op("If", "c", "Z", body: GraphOf("", "t", Op("Identity", "a", "t"))))));
        Assert.False(Proves(Initializing("a", GraphOf("a b", "O", Op("Sub", "a b", "O")))));
        Assert.False(Proves(GraphOf("a b c", "O", Op("If", "c", "O", body: GraphOf("", "t", Op("Identity", "b", "t"))))));
        Assert.True(Proves(GraphOf("a b c", "O", Op("Neg", "a", "n"), Op("If", "c", "g", body: GraphOf("", "t", Op("Identity", "n", "t"))), Op("Sub", "b g", "O"))));
        Assert.False(Proves(GraphOf("a b c", "O", Op("Neg", "a", "n"), Op("If", "c", "g", body: GraphOf("", "t", Op("Shape", "n", "t"))), Op("Sub", "b g", "O"))));
        Assert.True(Proves(GraphOf("a:float[4] b", "O:float[4]", Op("Neg", "b", "O"))));
        Assert.False(Proves(GraphOf("a:float[4] b", "O:double[4]", Op("Cast", "b", "O"))));
        Assert.False(Proves(GraphOf("a:float[4] b", "O:float[2,2]", Op("Neg", "b", "O"))));
        Assert.False(Proves(GraphOf("a b", "O Z", Op("Flatten", "a", "v"), Op("ReduceSum", "v", "g"), Op("Sub", "a g", "O"), Op("Neg", "v", "Z"))));
        Assert.False(Proves(GraphOf("a b", "O Z", Op("AllReduce", "a", "v", domain: "com.microsoft"), Op("ReduceSum", "v", "g"), Op("Sub", "a g", "O"), Op("Neg", "v", "Z"))));
        Assert.False(Proves(GraphOf("X scale B mean var two zero", "O Z Y", norm, Op("Mul", "rv two", "O"), Op("Add", "rv zero", "Z")), "var"));
        Assert.False(Proves(GraphOf("a x z", "O Z", Op("SequenceConstruct", "a x", "S"), Op("SequenceErase", "S", "E"),
            Op("ConcatFromSequence", "E", "c"), Op("ReduceSum", "c", "g"), Op("Sub", "a g", "O"), Op("SequenceAt", "E z", "Z"))));
    }

    [Fact]
    public void TestAWrittenGraphRefusesAnAliasWhereTheInputIsAnOutputTheOutputIsListedTwiceOrItsWriterIsNotAStandardOperator()
    {
        Assert.True(Proves(GraphOf("a b", "O", Op("Sub", "a b", "O"))));
        Assert.False(Proves(GraphOf("a b", "O a", Op("Sub", "a b", "O"))));
        Assert.False(Proves(GraphOf("a b", "O O", Op("Sub", "a b", "O"))));
        Assert.False(Proves(GraphOf("a b", "O", Op("Sub", "a b", "O", domain: "custom"))));
    }

    private static PlacementProof PlacementsOver(GraphProto graph, string consumed, PlacementMemory? memory = null)
    {
        var inputs = graph.Inputs.Where(i => i.Type?.TensorType?.Shape is not null).ToDictionary(
            i => i.Name, i => (i.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray(), i.Type.TensorType.ElemType));
        var shapes = PlacementShapes.Evaluate(graph, inputs);
        return new PlacementProof(graph, Names(consumed).ToDictionary(n => n, n => shapes[n].Bytes), shapes, memory: memory);
    }

    private static bool Places(GraphProto graph, string consumed, params Placement[] placements)
        => PlacementsOver(graph, consumed).Prove(placements).Count == placements.Length;

    private static bool PlacesOnTorch(GraphProto graph, string consumed, params Placement[] placements)
        => PlacementsOver(graph, consumed, PlacementMemory.PyTorch).Prove(placements).Count == placements.Length;

    private static GraphProto FirstHalf(string outputs, params NodeProto[] nodes)
        => WithInts(WithInts(GraphOf("a:float[256] c:float[128]", outputs, [Op("Slice", "a zero half zero", "x"), .. nodes]), "zero", 0), "half", 128);

    private static Placement At(string value, string block, long offset, long bytes = 512) => new(value, block, offset, bytes);

    internal static GraphProto WithInts(GraphProto graph, string name, params long[] values)
    {
        graph.Initializers.Add(new TensorProto { Name = name, data_type = (int)TensorProto.DataType.Int64, Dims = [values.Length], Int64Datas = values });
        return graph;
    }

    private static GraphProto Halves()
        => WithInts(WithInts(WithInts(GraphOf("a:float[128] b:float[128] c:float[64]", "O",
            Op("Slice", "b zero two zero", "x"), Op("Neg", "c", "y"), Op("Concat", "x y", "O", attribute: ("axis", 0))),
            "zero", 0), "two", 64), "four", 128);

    [Fact]
    public void TestAPlacementIsProvedOnlyForAValueOfItsOwnInsideItsBlockAtItsSize()
    {
        Assert.True(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Neg", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Neg", "a", "O")), "a", At("O", "a", 4)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Neg", "a", "O")), "a", At("O", "a", 0, 256)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Neg", "a", "O")), "b", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "O O", Op("Neg", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "b", Op("Neg", "a", "O")), "a", At("b", "a", 0)));
        Assert.False(Places(Initializing("k", GraphOf("a:float[128]", "k")), "a", At("k", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Flatten", "b", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Foo", "b", "O", domain: "custom")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] c", "O", Op("If", "c", "O", body: GraphOf("", "t", Op("Neg", "a", "t")))), "a", At("O", "a", 0)));
    }

    [Fact]
    public void TestAPlacementIsRefusedWhereAnythingReadsTheBytesItOverwritesWithoutRunningFirst()
    {
        Assert.False(Places(GraphOf("a:float[128]", "O Z", Op("Neg", "a", "O"), Op("Exp", "a", "Z")), "a", At("O", "a", 0)));
        Assert.True(Places(GraphOf("a:float[128]", "O", Op("Exp", "a", "t"), Op("Neg", "t", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "O Z", Op("Exp", "a", "t"), Op("Shape", "t", "s"), Op("ConstantOfShape", "s", "O"), Op("Neg", "t", "Z")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "O Z", Op("Flatten", "a", "v"), Op("Neg", "a", "O"), Op("Exp", "v", "Z")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] c", "O Z", Op("Neg", "a", "O"), Op("If", "c", "Z", body: GraphOf("", "t", Op("Identity", "a", "t")))), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "O a", Op("Neg", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "O P", Op("Neg", "a", "O"), Op("Exp", "O", "P")), "a", At("O", "a", 0), At("P", "a", 0)));
        Assert.True(Places(GraphOf("a:float[128]", "P", Op("Neg", "a", "O"), Op("Exp", "O", "P")), "a", At("O", "a", 0), At("P", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "P Z", Op("Neg", "a", "O"), Op("Exp", "O", "P"), Op("Abs", "O", "Z")), "a", At("O", "a", 0), At("P", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128]", "x y", Op("Split", "b", "x y")), "a", At("x", "a", 0, 256), At("y", "a", 0, 256)));
    }

    [Fact]
    public void TestAWriterMayOverwriteWhatItReadsOnlyWhereItReadsEachByteInThePositionItWritesIt()
    {
        Assert.False(Places(GraphOf("a:float[8,16]", "O", Op("Transpose", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[8,16] b:float[8,16]", "O", Op("MatMul", "a b", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128] c:bool[128]", "O", Op("Where", "c a b", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128] b:float[128] c:float[128]", "O", Op("Sum", "a b c", "O")), "a", At("O", "a", 0)));
        Assert.True(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Sum", "a b", "O")), "a", At("O", "a", 0)));
        Assert.True(Places(GraphOf("a:float[128] b:float[128]", "O", Op("Sub", "b a", "O")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "O", Op("Neg", "a", "O", domain: "custom")), "a", At("O", "a", 0)));
        Assert.False(Places(GraphOf("a:float[128]", "O", Op("Softmax", "a", "O")), "a", At("O", "a", 0)));
        Assert.True(Places(Halves(), "b", At("x", "b", 0, 256), At("y", "b", 256, 256), At("O", "b", 0)));
        Assert.False(Places(WithInts(WithInts(GraphOf("a:float[256]", "x", Op("Slice", "a zero half zero", "x")), "zero", 0), "half", 128), "a", At("x", "a", 256)));
        Assert.True(Places(WithInts(WithInts(GraphOf("a:float[256]", "x", Op("Slice", "a zero half zero", "x")), "zero", 0), "half", 128), "a", At("x", "a", 0)));
    }

    [Fact]
    public void TestAPlacementInsideWhatASliceDoesNotReadNeedsNoOrderAndAConcatenationIsInPlaceOnlyWherePartsLieInTheirSlots()
    {
        var sliced = WithInts(WithInts(WithInts(GraphOf("a:float[128] c:float[64]", "O Z", Op("Slice", "a two four zero", "Z"), Op("Neg", "c", "O")), "zero", 0), "two", 64), "four", 128);
        Assert.True(Places(sliced, "a", At("O", "a", 0, 256)));
        Assert.False(Places(sliced, "a", At("O", "a", 256, 256)));
        var swapped = WithInts(WithInts(GraphOf("a:float[128] b:float[128] c:float[64]", "O",
            Op("Slice", "b zero two zero", "x"), Op("Neg", "c", "y"), Op("Concat", "y x", "O", attribute: ("axis", 0))), "zero", 0), "two", 64);
        Assert.False(Places(swapped, "b", At("x", "b", 0, 256), At("y", "b", 256, 256), At("O", "b", 0)));
        Assert.False(Places(swapped, "b", At("x", "b", 0, 256), At("O", "b", 0)));
        Assert.True(Places(swapped, "b", At("x", "b", 256, 256), At("O", "b", 0)));
    }

    [Fact]
    public void TestOnTorchAViewIsItsInputsMemoryUnlessPlacedAndSoIsWhatAnOperatorNotKnownToComputeAfreshHandsBack()
    {
        Assert.True(Places(FirstHalf("O x", Op("Neg", "x", "O")), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(FirstHalf("O x", Op("Neg", "x", "O")), "a", At("O", "a", 0)));
        Assert.True(PlacesOnTorch(FirstHalf("O x", Op("Neg", "c", "O")), "a", At("O", "a", 512)));
        Assert.True(PlacesOnTorch(FirstHalf("x", Op("Neg", "c", "O")), "a", At("x", "a", 0)));
        Assert.True(PlacesOnTorch(FirstHalf("x", Op("Neg", "c", "O")), "a", At("x", "a", 512)));
        Assert.False(PlacesOnTorch(FirstHalf("x", Op("Neg", "c", "O")), "a", At("x", "a", 256)));
        Assert.True(PlacesOnTorch(FirstHalf("O x", Op("Neg", "x", "O")), "a", At("x", "a", 512), At("O", "a", 0)));
        Assert.True(Places(GraphOf("a:float[128]", "O Z", Op("Cast", "a", "y", attribute: ("to", 1)), Op("Neg", "y", "O"), Op("Exp", "y", "Z")), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:float[128]", "O Z", Op("Cast", "a", "y", attribute: ("to", 1)), Op("Neg", "y", "O"), Op("Exp", "y", "Z")), "a", At("O", "a", 0)));
        Assert.True(Places(GraphOf("a:float[128] c:bool[1]", "O:float[128] Z:float[128]", Op("If", "c", "Z", body: GraphOf("", "t", Op("Identity", "a", "t"))), Op("Neg", "Z", "O")), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:float[128] c:bool[1]", "O:float[128] Z:float[128]", Op("If", "c", "Z", body: GraphOf("", "t", Op("Identity", "a", "t"))), Op("Neg", "Z", "O")), "a", At("O", "a", 0)));
    }

    [Fact]
    public void TestOnTorchOnlyWhatItWritesIntoAGivenRangeAllocatingNoMoreThanAPlainRunIsPlaced()
    {
        Assert.True(Places(GraphOf("a:int64[64]", "O", Op("Neg", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:int64[64]", "O", Op("Neg", "a", "O")), "a", At("O", "a", 0)));
        Assert.True(PlacesOnTorch(GraphOf("a:float[128]", "O", Op("Neg", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:float[128] b:float[128]", "O", Op("Clip", "a", "O")), "b", At("O", "b", 0)));
        Assert.True(PlacesOnTorch(GraphOf("a:float[128] b:float[128] c:float[1]", "O", Op("Clip", "a c", "O")), "a", At("O", "a", 0)));
        Assert.True(PlacesOnTorch(GraphOf("a:float[128] b:float[128]", "O", Op("Softmax", "a", "O")), "b", At("O", "b", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:float[128] b:float[128]", "O", Op("Softmax", "a", "O")), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:int64[128] b:float[128]", "O", Op("Softmax", "a", "O")), "b", At("O", "b", 0)));
        Assert.True(PlacesOnTorch(GraphOf("a:float[2,64] s:float[64] b:float[128]", "O", Op("LayerNormalization", "a s", "O")), "b", At("O", "b", 0)));
        Assert.False(PlacesOnTorch(GraphOf("a:float[2,64] s:float[64] b:float[128]", "O M", Op("LayerNormalization", "a s", "O M")), "b", At("O", "b", 0)));
        Assert.True(PlacesOnTorch(GraphOf("a:float[8,16] b:float[16,8]", "O", Op("Transpose", "a", "O")), "b", At("O", "b", 0)));
        Assert.True(PlacesOnTorch(GraphOf("a:float[8,16] b:float[16,8] c:float[8,8]", "O", Op("MatMul", "a b", "O")), "c", At("O", "c", 0, 256)));
        Assert.False(PlacesOnTorch(GraphOf("a:float[8,8] b:float[8,8]", "O", Op("MatMul", "a b", "O")), "a", At("O", "a", 0, 256)));
        Assert.True(PlacesOnTorch(WithInts(GraphOf("a:float[128]", "O", Op("Reshape", "a s", "O")), "s", 2, 64), "a", At("O", "a", 0)));
        Assert.False(PlacesOnTorch(WithInts(GraphOf("a:float[8,16]", "O", Op("Transpose", "a", "t"), Op("Reshape", "t s", "O")), "s", 128), "a", At("O", "a", 0)));
    }

    [Fact]
    public void TestAPlacementsShapesFollowTheShapeArithmeticConvolutionsPoolsAndRecurrencesAGraphRuns()
    {
        Assert.Equal("2:7=4,2", ShapeOf(GraphOf("a:float[2,4]", "O", Op("Shape", "a", "s"), Op("Gather", "s one", "c"), Op("Unsqueeze", "c zero", "u"),
            Op("ReduceProd", "s", "p", attribute: ("keepdims", 0)), Op("Div", "p c", "q"), Op("Unsqueeze", "q zero", "v"), Op("Concat", "u v", "t", attribute: ("axis", 0)), Op("Reshape", "a t", "O")), "t"));
        Assert.Equal("4x2", ShapeOf(GraphOf("a:float[2,4]", "O", Op("Shape", "a", "s"), Op("Gather", "s one", "c"), Op("Unsqueeze", "c zero", "u"),
            Op("ReduceProd", "s", "p", attribute: ("keepdims", 0)), Op("Div", "p c", "q"), Op("Unsqueeze", "q zero", "v"), Op("Concat", "u v", "t", attribute: ("axis", 0)), Op("Reshape", "a t", "O")), "O"));
        Assert.Equal("3:7=1,2,3", ShapeOf(GraphOf("a:float[4]", "O", Op("Size", "a", "n"), Op("Range", "one n one", "O")), "O"));
        Assert.Equal("2:7=2,4", ShapeOf(GraphOf("a:float[2,4]", "O", Op("Shape", "a", "s"), Op("MemcpyFromHost", "s", "O")), "O"));
        Assert.Equal(":7=1", ShapeOf(GraphOf("a:float[4]", "O", Op("Size", "a", "n"), Op("Equal", "n four", "e"), Op("Not", "e", "f"), Op("Cast", "f", "c", attribute: ("to", 7)),
            Op("Where", "e one zero", "w"), Op("Add", "w c", "O")), "O"));
        Assert.Equal("unknown", ShapeOf(GraphOf("a:float[4]", "O", Op("Size", "a", "n"), Op("Cast", "n", "u", attribute: ("to", 13)), Op("Sub", "u five", "d"), Op("Range", "zero d one", "O")), "O"));
        Assert.Equal("1x4x8x8", ShapeOf(GraphOf("x:float[1,3,8,8] w:float[4,3,3,3]", "O", With(Op("Conv", "x w", "O"), "pads", 1, 1, 1, 1)), "O"));
        Assert.Equal("1x4x4x4", ShapeOf(GraphOf("x:float[1,3,8,8] w:float[4,3,3,3]", "O", With(With(Op("Conv", "x w", "O"), "pads", 1, 1, 1, 1), "strides", 2, 2)), "O"));
        Assert.Equal("1x3x4x4:7", ShapeOf(GraphOf("x:float[1,3,8,8]", "O I", With(With(Op("MaxPool", "x", "O I"), "kernel_shape", 2, 2), "strides", 2, 2)), "I"));
        Assert.Equal("1x3x1x1", ShapeOf(GraphOf("x:float[1,3,8,8]", "O", Op("GlobalAveragePool", "x", "O")), "O"));
        Assert.Equal("5x1x2x4 1x2x4", ShapeOf(GraphOf("x:float[5,2,3] w:float[1,16,3] r:float[1,16,4]", "Y H C", Op("LSTM", "x w r", "Y H C", attribute: ("hidden_size", 4))), "Y", "H"));
    }

    [Fact]
    public void TestAPlacementsShapesFollowTheShapeArithmeticFusedOperatorsAndTransposedConvolutionsOfATrainingStep()
    {
        Assert.Equal("4:7=1,4,2,1", ShapeOf(WithInts(GraphOf("a:float[4,2]", "O", Op("Shape", "a", "s"), Op("Pad", "s pads one", "O")), "pads", 1, 1), "O"));
        Assert.Equal("1:7=4", ShapeOf(GraphOf("a:float[4,2]", "O", Op("Shape", "a", "s"), Op("Greater", "s two", "g"), Op("Compress", "s g", "O")), "O"));
        Assert.Equal("1:7=0", ShapeOf(GraphOf("a:float[4]", "O", Op("Shape", "a", "s"), Op("Shape", "s", "n"), Op("Expand", "zero n", "O")), "O"));
        Assert.Equal("1x3x6x6", ShapeOf(GraphOf("x:float[1,4,4,4] w:float[4,3,3,3]", "O", Op("ConvTranspose", "x w", "O")), "O"));
        Assert.Equal("1x3x8x8", ShapeOf(GraphOf("x:float[1,4,4,4] w:float[4,3,2,2]", "O", With(Op("ConvTranspose", "x w", "O"), "strides", 2, 2)), "O"));
        Assert.Equal("1x4x8x8", ShapeOf(GraphOf("x:float[1,3,8,8] w:float[4,3,3,3]", "O", With(Op("FusedConv", "x w", "O", domain: "com.microsoft"), "pads", 1, 1, 1, 1)), "O"));
        Assert.Equal("8x16", ShapeOf(GraphOf("a:float[8,4] b:float[4,16]", "O", Op("FusedGemm", "a b", "O", domain: "com.microsoft")), "O"));
        Assert.Equal("4x2x8x8", ShapeOf(GraphOf("x:float[4,2,8,8] m:float[8,8]", "O", Op("BiasSoftmax", "x m", "O", domain: "com.microsoft")), "O"));
        Assert.Equal("4x2", ShapeOf(GraphOf("a:float[4,2] c:bool[1]", "t", PyTorchBackendCoverageTests.Branch("c", GraphOf("", "t", Op("Neg", "a", "t")), GraphOf("", "t", Op("Abs", "a", "t")))), "t"));
        Assert.Equal("unknown", ShapeOf(GraphOf("a:float[4,2] c:bool[1]", "t", PyTorchBackendCoverageTests.Branch("c", GraphOf("", "t", Op("Neg", "a", "t")), GraphOf("", "t", Op("Shape", "a", "t")))), "t"));
        Assert.Equal("2:7=4,2", ShapeOf(GraphOf("a:float[4,2]", "t", Op("Greater", "two one", "c"),
            PyTorchBackendCoverageTests.Branch("c", GraphOf("", "t", Op("Shape", "a", "t")), GraphOf("", "t", Op("Neg", "a", "t")))), "t"));
    }

    /// <summary>The shapes <see cref="PlacementShapes"/> evaluates for <paramref name="values"/> of
    /// <paramref name="graph"/>, with the small integers zero to five as initializers it may read,
    /// as <c>4x2</c> for a float value, <c>4x2:7</c> for another type, <c>=4,2</c> after it for
    /// known contents, and <c>unknown</c>.</summary>
    private static string ShapeOf(GraphProto graph, params string[] values)
    {
        foreach (var (name, value) in (ReadOnlySpan<(string, long)>)[("zero", 0), ("one", 1), ("two", 2), ("four", 4), ("five", 5)])
            graph.Initializers.Add(new TensorProto { Name = name, data_type = (int)TensorProto.DataType.Int64, Dims = [], Int64Datas = [value] });
        var inputs = graph.Inputs.Where(i => i.Type?.TensorType?.Shape is not null).ToDictionary(
            i => i.Name, i => (i.Type.TensorType.Shape.Dims.Select(d => d.DimValue).ToArray(), i.Type.TensorType.ElemType));
        var shapes = PlacementShapes.Evaluate(graph, inputs);
        return string.Join(" ", values.Select(v => shapes.TryGetValue(v, out var x)
            ? string.Join("x", x.Shape) + (x.ElementType == 1 ? "" : $":{x.ElementType}") + (x.Ints is { } ints ? "=" + string.Join(",", ints) : "")
            : "unknown"));
    }

    /// <summary>A <c>Loop</c> of <paramref name="inputs"/> — its trip count and its carried values,
    /// with no condition — making <paramref name="outputs"/> by <paramref name="body"/>.</summary>
    internal static NodeProto Loop(string inputs, string outputs, GraphProto body)
    {
        var loop = Op("Loop", inputs, outputs);
        loop.Inputs.Insert(1, "");
        loop.Attributes.Add(new AttributeProto { Name = "body", Type = AttributeProto.AttributeType.Graph, G = body });
        return loop;
    }

    internal static NodeProto With(NodeProto node, string name, params long[] ints)
    {
        node.Attributes.Add(new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Ints, Ints = ints });
        return node;
    }

    [Fact]
    public void TestThePlannerPlacesTheTwoHalvesScenarioWithNothingLeftToAllocate()
    {
        var graph = ProtoBuf.Serializer.Deserialize<ModelProto>(new MemoryStream(
            Benchmarks.MemoryReuseScenarioTests.Scenario(Benchmarks.MemoryReuseScenarioTests.Shapes.Computed, exposeIntermediates: false))).Graph;
        var shapes = PlacementShapes.Evaluate(graph, new Dictionary<string, (long[], int)> { ["A"] = ([512, 64], 1), ["B"] = ([512, 64], 1) });
        var plan = new PlacementProof(graph, new Dictionary<string, long> { ["A"] = 131072, ["B"] = 131072 }, shapes).Plan(smallest: 65536, idleOutputBytes: 0);
        Assert.Equal(
            ["A_half@A+0", "B_half@A+65536", "C0@B+0", "C1@B+0", "C2@B+0", "C3@B+0", "L0@B+0", "L1@B+0", "L2@B+0", "L@B+0"],
            plan.Select(p => $"{p.Value}@{p.Block}+{p.Offset}").Order(StringComparer.Ordinal));
        Assert.Empty(new PlacementProof(graph, new Dictionary<string, long> { ["A"] = 131072 }, shapes).Plan(smallest: 65536, idleOutputBytes: 0).Where(p => p.Value == "L"));
        var torchPlan = new PlacementProof(graph, new Dictionary<string, long> { ["A"] = 131072, ["B"] = 131072 }, shapes, memory: PlacementMemory.PyTorch)
            .Plan(smallest: 65536, idleOutputBytes: 0);
        Assert.Equal(plan.Select(p => p.ToString()).Order(), torchPlan.Select(p => p.ToString()).Order());
    }

    /// <summary>Two tensors standing on one host block of eight floats, its halves, and the block.</summary>
    private static (TensorData First, TensorData Second, SharedBlock Block, OrtTensorValue Owner) Halved()
    {
        var backend = DefaultBackend.Instance;
        var owner = (OrtTensorValue)backend.CreateTensor((float[])[1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], [8L]);
        var block = new SharedBlock(32, () => backend.Release(owner));
        var first = TensorData.Create((long[])[4], DType.Float32, OrtBackend.View(owner, 0, ShorokooTensorElementType.Float, [4], 16, block, 0), backend);
        var second = TensorData.Create((long[])[4], DType.Float32, OrtBackend.View(owner, 16, ShorokooTensorElementType.Float, [4], 16, block, 16), backend);
        return (first, second, block, owner);
    }

    [Fact]
    public void TestTensorsStandingOnOneBlockEachOwnTheirRangeAndTheBlockGoesWithTheLastOfThem()
    {
        var (first, second, block, owner) = Halved();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(first));
        Assert.Equal([5f, 6f, 7f, 8f], Floats(second));
        Assert.Same(block, first.Block);
        Assert.Equal(2, block.Leases);
        first.Delete();
        Assert.False(block.IsReleased);
        Assert.Equal([5f, 6f, 7f, 8f], Floats(second));
        second.Delete();
        Assert.True(block.IsReleased);
        Assert.Throws<ObjectDisposedException>(() => owner.Inner);
    }

    [Fact]
    public void TestAnOutputWrittenIntoTheInputItIsMarkedForIsNotPlacedElsewhere()
    {
        var backend = DefaultBackend.Instance;
        using var session = Aliasing(backend, GraphOf("a:float[512,512] b:float[512,512]", "O:float[512,512]", Op("Sub", "a b", "O")));
        float[] values = [.. Enumerable.Range(0, 512 * 512).Select(i => (float)(i % 7))];
        var a = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), [512, 512]);
        var b = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, MemoryMarshal.AsBytes(values.AsSpan()).ToArray(), [512, 512]);
        var address = OrtBackend.AddressOf(((OrtTensorValue)a).Inner);
        using var output = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b }, [a, b], ["O"], RunSettings.Default, out var aliased).Single();
        Assert.Equal(["a"], aliased);
        Assert.Equal(address, OrtBackend.AddressOf(((OrtTensorValue)output).Inner));
    }

    [Fact]
    public void TestEverySessionAPlacedRunIsBuiltWithIsBuiltInItsContextsPrecision()
    {
        const int Rows = 1024, Columns = 1024;
        var (a, b, l) = TwoHalvesValues(Rows, Columns);
        foreach (var allowed in (bool[])[false, true])
        {
            var seen = new List<PrecisionSettings>();
            using var context = new ComputeContext(new PrecisionRecordingBackend(seen)) { Precision = new PrecisionSettings { AllowTensorFloat32 = allowed } };
            var compiled = context.Compile(TwoHalves());
            var outputs = compiled.Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b));
            Assert.Equal(OrtPlacements.Stage.Adopted, Assert.Single(((OrtSession)compiled.Session).Placements!.Entries).Stage);
            Assert.NotNull(outputs[0].ToTensorData().Block);
            Assert.Equal([allowed], seen.Select(p => p.AllowTensorFloat32).Distinct());
            Assert.True(l.Zip(Floats(outputs[0].ToTensorData()), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        }
    }

    internal sealed class PrecisionRecordingBackend(List<PrecisionSettings> seen)
        : OrtBackend((_, _, precision) => { lock (seen) seen.Add(precision); }, ComputeDevice.Cpu, cudaDeviceId: null, stockProvider: true);

    internal static InternalComputationGraph ProductIntoConsumed()
    {
        var a = InputTensor<float32>("A", rank: 2);
        var b = InputTensor<float32>("B", rank: 2);
        var e = InputTensor<float32>("E", rank: 2);
        return new InternalComputationGraph([a, b, e], [OnnxOp.Add(OnnxOp.MatMul(a, b), e)]);
    }

    internal static (float[] Values, bool Placed) RunProductIntoConsumed(ComputeContext context)
    {
        static TensorData Waves(long rows, long columns, float frequency)
            => TensorData([rows, columns], [.. Enumerable.Range(0, (int)(rows * columns)).Select(i => MathF.Sin(frequency * i))]);
        var output = context.Compile(ProductIntoConsumed()).Execute(
            Waves(1024, 1024, 0.37f).CopyTo(context), Waves(1024, 512, 0.11f).CopyTo(context),
            TensorData([1024L, 512L], new float[1 << 19]).CopyTo(context)).Single().ToTensorData();
        return ([.. output.ToHost().As<float32>().AccessMemory<float>()], output.Block is not null);
    }

    internal static float[] ProductIntoConsumedOnTheHost()
    {
        using var host = new ComputeContext(new PrecisionRecordingBackend([]));
        return RunProductIntoConsumed(host).Values;
    }

    [Fact]
    public void TestASessionOfAContextThatPlacesNothingKeepsNothingToPlaceWith()
    {
        const int Rows = 1024, Columns = 1024;
        var (a, b, l) = TwoHalvesValues(Rows, Columns);
        using var context = new ComputeContext { ValuePlacement = false };
        var compiled = context.Compile(TwoHalves());
        var outputs = compiled.Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b));
        Assert.Null(((OrtSession)compiled.Session).Placements);
        Assert.True(l.Zip(Floats(outputs[0].ToTensorData()), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
    }

    [Fact]
    public void TestASignatureWhosePlacedRunFailsRunsAsUsualFromThen()
    {
        const int Rows = 1024, Columns = 1024;
        var (a, b, l) = TwoHalvesValues(Rows, Columns);
        using var context = new ComputeContext();
        var compiled = context.Compile(TwoHalves());
        var failures = 1;
        OrtPlacements.PlacedRunFault = () => failures-- > 0 ? new InvalidOperationException() : null;
        try
        {
            Assert.Throws<InvalidOperationException>(() => compiled.Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b)));
            var outputs = compiled.Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b));
            Assert.Equal(OrtPlacements.Stage.Refused, Assert.Single(((OrtSession)compiled.Session).Placements!.Entries).Stage);
            Assert.True(l.Zip(Floats(outputs[0].ToTensorData()), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        }
        finally
        {
            OrtPlacements.PlacedRunFault = null;
        }
    }

    [Fact]
    public void TestACardSessionsNodesWritingHostMemoryAreTheHostOnesWhateverTheyRead()
    {
        string[] Host(GraphProto graph) => [.. OrtPlacements.HostNodes(graph).Select(n => n.OpType).Order()];
        Assert.Equal(["Hardmax"], Host(GraphOf("x w", "z", Op("Hardmax", "x", "y"), Op("MemcpyFromHost", "y", "y_card"),
            Op("Neg", "y_card", "n"), Op("Relu", "w", "t"), Op("Add", "n t", "z"))));
        Assert.Equal(["Hardmax", "Softmax"], Host(GraphOf("x", "z", Op("Hardmax", "x", "y"), Op("Softmax", "y", "s"),
            Op("MemcpyFromHost", "s", "s_card"), Op("Neg", "s_card", "z"))));
        Assert.Equal(["Hardmax"], Host(GraphOf("x w", "z", Op("Hardmax", "x", "y"), Op("MemcpyFromHost", "y", "y_card"),
            Op("MemcpyFromHost", "x", "x_card"), Op("Relu", "x_card", "r"), Op("Add", "y_card r", "s"), Op("Mul", "s w", "z"))));
        Assert.Equal(["MemcpyToHost", "Shape"], Host(GraphOf("x", "z", Op("Relu", "x", "r"), Op("Shape", "r", "s"),
            Op("MemcpyToHost", "r", "r_host"), Op("Neg", "r", "z"))));
    }

    [Fact]
    public void TestABlockWhoseLastPagePassesItsTensorCountsWhatItStillHoldsOfTheTensor()
    {
        var backend = DefaultBackend.Instance;
        const long Floats = (1 << 18) + 1;
        var owner = (OrtTensorValue)backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, new byte[Floats * 4], [Floats]);
        var block = OrtBackend.BlockOver(owner, Floats * 4, () => backend.Release(owner));
        var first = OrtBackend.View(owner, 0, ShorokooTensorElementType.Float, [1L << 17], 512L << 10, block, 0);
        var second = OrtBackend.View(owner, 512L << 10, ShorokooTensorElementType.Float, [1L << 17], 512L << 10, block, 512L << 10);
        second.Dispose();
        Assert.Equal(512L << 10, block.HeldBytes);
        first.Dispose();
    }

    [Fact]
    public void TestASequenceThatFailsToTakeACopyOfAViewLetsGoOfTheCopiesItMade()
    {
        var (first, second, _, _) = Halved();
        var backend = (OrtBackend)DefaultBackend.Instance;
        var host = RuntimeAllocator.ForHost();
        var account = host.Shared.Open("probe");
        var a = (OrtTensorValue)((IOnnxData)first).Value;
        var b = (OrtTensorValue)((IOnnxData)second).Value;
        b.Dispose();
        using (CachingAllocator.Charge(account, null))
            Assert.ThrowsAny<Exception>(() => backend.CreateSequence([a, b]));
        Assert.Equal(0L, host.Shared.Statistics(account).InUseBytes);
        host.Shared.Close(account);
    }

    [Fact]
    public void TestATensorStandingOnABlockMovesSavesLoadsAndJoinsASequenceAsItsOwnRange()
    {
        var (first, second, block, _) = Halved();
        using var context = new ComputeContext();
        Assert.Equal([5f, 6f, 7f, 8f], Floats(second.ToHost()));
        Assert.Equal([5f, 6f, 7f, 8f], Floats(second.CopyTo(context)));
        using var stream = new MemoryStream();
        SafeTensorLoader.SaveSafeTensorsToStream(stream, [new("first", first, "F32", [4L]), new("second", second, "F32", [4L])]);
        var path = Path.Combine(Path.GetTempPath(), $"halves-{Guid.NewGuid():N}.safetensors");
        try
        {
            File.WriteAllBytes(path, stream.ToArray());
            var loaded = SafeTensorLoader.LoadTensorDictionary(path);
            Assert.Equal([1f, 2f, 3f, 4f], Floats(loaded["first"]));
            Assert.Equal([5f, 6f, 7f, 8f], Floats(loaded["second"]));
        }
        finally
        {
            File.Delete(path);
        }
        var backend = DefaultBackend.Instance;
        using var sequence = backend.CreateSequence([OrtBackend.View((OrtTensorValue)((IOnnxData)first).Value, 8, ShorokooTensorElementType.Float, [2], 8, block, 8)]);
        Assert.Equal(2, block.Leases);
        Assert.Equal([3f, 4f], sequence.GetValue(0).GetTensorDataAsSpan<float>().ToArray());
        first.Delete();
        second.Delete();
        Assert.True(block.IsReleased);
        Assert.Equal([3f, 4f], sequence.GetValue(0).GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public void TestAnOutputWrittenIntoAConsumedTensorStandingOnABlockHoldsALeaseOnTheBlock()
    {
        var backend = DefaultBackend.Instance;
        var owner = (OrtTensorValue)backend.CreateTensor((float[])[1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], [8L]);
        var block = new SharedBlock(32, () => backend.Release(owner));
        var a = OrtBackend.View(owner, 0, ShorokooTensorElementType.Float, [4], 16, block, 0);
        var second = TensorData.Create((long[])[4], DType.Float32, OrtBackend.View(owner, 16, ShorokooTensorElementType.Float, [4], 16, block, 16), backend);
        using var session = Aliasing(backend, GraphOf("a:float[4] b:float[4]", "O:float[4]", Op("Sub", "a b", "O")));
        var outputs = session.RunConsuming(
            new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = DefaultBackend.Instance.CreateTensor((float[])[1f, 1f, 1f, 1f], [4L]) },
            [a], ["O"], RunSettings.Default, out var aliased);
        Assert.Equal(["a"], aliased);
        second.Delete();
        Assert.False(block.IsReleased);
        Assert.Equal([0f, 1f, 2f, 3f], outputs[0].GetTensorDataAsSpan<float>().ToArray());
        outputs[0].Dispose();
        Assert.True(block.IsReleased);
    }

    /// <summary>
    /// Two consumed inputs A and B of one shape; A's first half and B's second half sliced out by
    /// bounds the run computes; a fill of 2s shaped like a half, through Neg, Abs and Sigmoid;
    /// concatenated with B's half, through Sigmoid, Neg and Abs. Outputs that, and the two halves —
    /// or A's half alone where <paramref name="oneHalf"/>.
    /// </summary>
    internal static InternalComputationGraph TwoHalves(bool oneHalf = false)
    {
        var a = InputTensor<float32>("A", rank: 2);
        var b = InputTensor<float32>("B", rank: 2);
        var zero = OnnxOp.Constant((long[])[0L]);
        var rows = OnnxOp.Shape(a, end: 1, start: 0);
        var half = OnnxOp.Div(rows, OnnxOp.Constant((long[])[2L]));
        var aHalf = OnnxOp.Slice(a, zero, half, zero);
        var bHalf = OnnxOp.Slice(b, half, rows, zero);
        var c = OnnxOp.ConstantOfShape(OnnxOp.Shape(aHalf), TensorAttribute.Create(new Shape(1L), (float[])[2f]));
        var l0 = OnnxOp.Concat([OnnxOp.Sigmoid(OnnxOp.Abs(OnnxOp.Neg(c))), bHalf], 0);
        var l = OnnxOp.Abs(OnnxOp.Neg(OnnxOp.Sigmoid(l0)));
        return new InternalComputationGraph([a, b], oneHalf ? [l, aHalf] : [l, aHalf, bHalf]);
    }

    internal static (float[] A, float[] B, float[] L) TwoHalvesValues(int rows, int columns)
    {
        float[] a = [.. Enumerable.Range(0, rows * columns).Select(i => (i % 1013) * 0.001f)];
        float[] b = [.. Enumerable.Range(0, rows * columns).Select(i => 0.5f - (i % 997) * 0.002f)];
        static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
        var half = rows / 2 * columns;
        float[] l = [.. Enumerable.Range(0, rows * columns).Select(i => i < half ? Sigmoid(Sigmoid(2f)) : Sigmoid(b[i]))];
        return (a, b, l);
    }

    [Fact]
    public void TestTheTwoHalvesScenarioRunsWithNothingAllocatedBeyondWhatItConsumesAndItsOutputsOutliveTheSession()
    {
        const int Rows = 512, Columns = 1024;
        var (a, b, l) = TwoHalvesValues(Rows, Columns);
        NamedModelParam[] outputs = [];
        using (var context = new ComputeContext())
        {
            var compiled = context.Compile(TwoHalves());
            for (int run = 0; run < 3; run++)
            {
                outputs = compiled.Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b));
                Assert.True(l.Zip(Floats(outputs[0].ToTensorData()), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
                Assert.Equal(a[..(Rows / 2 * Columns)], Floats(outputs[1].ToTensorData()));
                Assert.Equal(b[(Rows / 2 * Columns)..], Floats(outputs[2].ToTensorData()));
            }
            var entry = Assert.Single(((OrtSession)compiled.Session).Placements!.Entries);
            Assert.Equal(OrtPlacements.Stage.Adopted, entry.Stage);
            Assert.True(entry.PlacedPeak < 1L << 20);
            Assert.All(outputs, o => Assert.NotNull(o.ToTensorData().Block));
            Assert.Same(outputs[1].ToTensorData().Block, outputs[2].ToTensorData().Block);
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.True(l.Zip(Floats(outputs[0].ToTensorData()), (x, y) => MathF.Abs(x - y) < 1e-5f).All(x => x));
        Assert.Equal(b[(Rows / 2 * Columns)..], Floats(outputs[2].ToTensorData()));
    }

    /// <summary>
    /// <paramref name="graph"/> run consuming A and B, each a value a session made in its own memory,
    /// and its outputs deleted one by one: after the run and after each deletion, what that session
    /// has in use, the bytes of the outputs still standing on a block, and what
    /// <paramref name="context"/>'s books hold; and the most outputs that stood on one block.
    /// </summary>
    internal static ((long InUse, long OnBlocks, long Books)[] Stages, int MostOnABlock) OutputsOnBlocksEnding(
        ComputeContext context, InternalComputationGraph graph, int rows, int columns)
    {
        var (a, b, _) = TwoHalvesValues(rows, columns);
        var x = InputTensor<float32>("x", rank: 2);
        var made = context.Compile(new InternalComputationGraph([x], [x + 1f]));
        TensorData Made(float[] values)
        {
            var source = TensorData([(long)rows, columns], values).CopyTo(context);
            var value = made.Execute(source.Shared()).Single().ToTensorData();
            source.Delete();
            return value;
        }
        var outputs = context.Compile(graph).Execute(Made(a), Made(b)).Select(o => o.ToTensorData()).ToList();
        var most = outputs.Where(o => o.Block is not null).GroupBy(o => o.Block).Max(g => (int?)g.Count()) ?? 0;
        List<(long, long, long)> stages = [];
        void Stage() => stages.Add((made.ReadArenaStatistics()!.Value.InUseBytes,
            outputs.Where(o => !o.IsDisposed && o.Block is not null).Sum(o => o.ByteCount), context.ReadDeviceMemoryUse().AttachedBytes));
        Stage();
        foreach (var output in outputs)
        {
            output.Delete();
            Stage();
        }
        return ([.. stages], most);
    }

    [Fact]
    public void TestOutputsOnOneBlockOfASessionsMemoryEachFreeTheirOwnPagesAndWhatNoneStandsOnGoesWithTheRun()
    {
        using var context = new ComputeContext();
        var (both, together) = OutputsOnBlocksEnding(context, TwoHalves(), 1024, 1024);
        var (one, _) = OutputsOnBlocksEnding(context, TwoHalves(oneHalf: true), 1024, 1024);
        Assert.Equal(2, together);
        Assert.True(one[0].OnBlocks > 0);
        Assert.All(both, stage => Assert.Equal(stage.OnBlocks, stage.InUse - both[^1].InUse));
        Assert.All(one, stage => Assert.Equal(stage.OnBlocks, stage.InUse - one[^1].InUse));
    }

    [Fact]
    public void TestOutputsOnOneBlockOfAHostTensorMadeFromHostDataEachFreeTheirOwnPages()
    {
        const int Rows = 1024, Columns = 1024;
        var (a, b, _) = TwoHalvesValues(Rows, Columns);
        using var context = new ComputeContext();
        var outputs = context.Compile(TwoHalves()).Execute(TensorData([(long)Rows, Columns], a), TensorData([(long)Rows, Columns], b))
            .Select(o => o.ToTensorData()).ToList();
        var block = outputs.Where(o => o.Block is not null).GroupBy(o => o.Block).Single(g => g.Count() == 2).Key!;
        List<long> held = [block.HeldBytes];
        foreach (var output in outputs.Where(o => o.Block == block))
        {
            output.Delete();
            if (!block.IsReleased) held.Add(block.HeldBytes);
        }
        Assert.Equal([4L << 20, 2L << 20], held);
    }

    /// <summary>
    /// A model carrying <paramref name="n"/> floats of its own, added across each row of its consumed
    /// input past a Relu; its output, the session that ran it, and the output it should have.
    /// </summary>
    internal static (TensorData Output, OrtSession Session, float[] Expected) LargeModelRun(ComputeContext context, int n)
    {
        float[] b = [.. Enumerable.Range(0, n).Select(i => (i % 7) * 0.5f)];
        float[] x = [.. Enumerable.Range(0, 2 * n).Select(i => (i % 5) - 2f)];
        var input = InputTensor<float32>("x", rank: 2);
        var compiled = context.Compile(new InternalComputationGraph(
            [input], [OnnxOp.Add(OnnxOp.Relu(input), OnnxOp.Constant(TensorAttribute.Create(new Shape(1L, n), b)))]));
        var output = compiled.Execute(TensorData([2L, n], x).CopyTo(context)).Single().ToTensorData();
        return (output, (OrtSession)compiled.Session, [.. x.Select((v, i) => MathF.Max(v, 0f) + b[i % n])]);
    }

    [Fact]
    public void TestAModelOverSixteenMebibytesPlacesARunsValuesWhereThatPays()
    {
        using var context = new ComputeContext();
        var (output, session, expected) = LargeModelRun(context, 9 << 19);
        var entry = Assert.Single(Assert.IsType<OrtPlacements>(session.Placements).Entries);
        Assert.Equal(OrtPlacements.Stage.Adopted, entry.Stage);
        Assert.NotNull(output.Block);
        Assert.Equal(expected, Floats(output));
    }

    /// <summary>
    /// A run of <c>y = -Relu(x) * x·W</c> consuming x, through a session of
    /// <paramref name="backend"/>: the matrix product reads x on a branch of its own, which the
    /// session runs before the other. The plan it settled on, the output, and what it should be.
    /// </summary>
    internal static (OrtPlacements.Entry Entry, float[] Output, float[] Expected) BranchesRun(IShorokooBackend backend)
    {
        const int N = 512;
        var graph = GraphOf($"x:float[{N},{N}]", $"y:float[{N},{N}]",
            Op("Relu", "x", "a"), Op("Neg", "a", "n"), Op("MatMul", "x W", "b"), Op("Mul", "n b", "y"));
        float[] w = [.. Enumerable.Range(0, N * N).Select(i => (i * 7 % 3) - 1f)];
        float[] x = [.. Enumerable.Range(0, N * N).Select(i => (i % 5) - 2f)];
        var raw = new byte[w.Length * 4];
        Buffer.BlockCopy(w, 0, raw, 0, raw.Length);
        graph.Initializers.Add(new TensorProto { Name = "W", data_type = 1, Dims = [N, N], RawData = raw });
        using var session = (OrtSession)backend.CreateSession(
            ModelOf(graph), ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal, new DeviceMemorySettings(), DiagnosticSettings.Default);
        var input = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, MemoryMarshal.AsBytes(x.AsSpan()).ToArray(), [N, N]);
        var output = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["x"] = input }, [input], ["y"], RunSettings.Default, out _).Single();
        var bytes = backend.CopyTensorToHost(output);
        output.Dispose();
        var expected = new float[N * N];
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
            {
                var xik = x[i * N + k];
                for (int j = 0; j < N; j++) expected[i * N + j] += xik * w[k * N + j];
            }
        for (int i = 0; i < expected.Length; i++) expected[i] *= -MathF.Max(x[i], 0f);
        return (Assert.Single(session.Placements!.Entries), MemoryMarshal.Cast<byte, float>(bytes).ToArray(), expected);
    }

    [Fact]
    public void TestARunPlacesAValueOverAnInputABranchItNeedNotFollowReadsWhereTheSessionRunsThatBranchFirst()
    {
        var (entry, output, expected) = BranchesRun(DefaultBackend.Instance);
        Assert.Equal(OrtPlacements.Stage.Adopted, entry.Stage);
        Assert.Contains("a", entry.Plan.Select(p => p.Value));
        Assert.Equal(expected, output);
    }

    /// <summary>
    /// For a session over <paramref name="family"/>'s model built as a plain session is, and one
    /// built as a variant is from the graph the first wrote out, each provider's nodes in the order
    /// they ran and in the order of the graph the session wrote out.
    /// </summary>
    internal static List<(string[] Ran, string[] Written)> RunOrders(string family)
    {
        var backend = (OrtBackend)DefaultBackend.Instance;
        var (_, modelOf, shape) = Benchmarks.MemoryPassBenchmarkTests.Suite.Single(s => s.Family == family);
        var count = (int)shape.Aggregate(1L, (a, d) => a * d);
        var sample = TensorData(shape, [.. Enumerable.Range(0, count).Select(i => (i % 101) / 101f - 0.5f)]);
        using var context = new ComputeContext();
        var model = ((OrtSession)context.Compile(modelOf().ToConcreteArchitecture([sample]).ToConcreteModel()).Session).Placements!.OriginalModel;
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        var raw = MemoryMarshal.AsBytes(sample.As<float32>().AccessMemory<float>()).ToArray();
        List<(string[], string[])> orders = [];
        string[] directories = [.. Enumerable.Range(0, 2).Select(_ => Path.Combine(Path.GetTempPath(), $"run-order-{Guid.NewGuid():N}"))];
        foreach (var directory in directories) Directory.CreateDirectory(directory);
        try
        {
            for (int build = 0; build < 2; build++)
            {
                var built = backend.NewSession(
                    build == 0 ? stream.ToArray() : File.ReadAllBytes(Path.Combine(directories[0], OrtBackend.OptimizedModelFile)),
                    build == 0 ? ShorokooGraphOptimization.EnableAll : ShorokooGraphOptimization.DisableAll, ShorokooLogSeverity.Fatal,
                    new DeviceMemorySettings(), new DiagnosticSettings { TraceNodePlacement = true }, directories[build], 0, [], PrecisionSettings.Default,
                    externalDataDirectory: build == 0 ? null : directories[0]);
                using var session = backend.Wrap(built, []);
                var input = backend.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, raw, shape);
                foreach (var output in ((IShorokooSession)session).Run(
                    new Dictionary<string, IShorokooTensorValue> { [model.Graph!.Inputs[0].Name] = input }, session.OutputNames, RunSettings.Default))
                    output.Dispose();
                input.Dispose();
                using var profile = System.Text.Json.JsonDocument.Parse(File.ReadAllText(built.Session.EndProfiling()));
                var ran = profile.RootElement.EnumerateArray()
                    .Where(e => e.TryGetProperty("cat", out var cat) && cat.GetString() == "Node" && e.GetProperty("name").GetString()!.EndsWith("_kernel_time"))
                    .Select(e => (Ts: e.GetProperty("ts").GetInt64(), Name: e.GetProperty("name").GetString()![..^"_kernel_time".Length],
                        Provider: e.GetProperty("args").GetProperty("provider").GetString()!))
                    .OrderBy(e => e.Ts).DistinctBy(e => e.Name).ToList();
                using var written = File.OpenRead(Path.Combine(directories[build], OrtBackend.OptimizedModelFile));
                var listed = ProtoBuf.Serializer.Deserialize<ModelProto>(written).Graph!.Nodes.Select(n => n.Name).ToList();
                foreach (var provider in ran.Select(e => e.Provider).Distinct())
                {
                    var names = ran.Where(e => e.Provider == provider).Select(e => e.Name).ToHashSet();
                    orders.Add(([.. ran.Where(e => e.Provider == provider).Select(e => e.Name)], [.. listed.Where(names.Contains)]));
                }
            }
        }
        finally
        {
            foreach (var directory in directories) Directory.Delete(directory, recursive: true);
        }
        return orders;
    }

    [Fact]
    public void TestASessionRunsEachProvidersNodesInTheOrderOfTheGraphItWritesOutBuiltFromAModelOrFromAWrittenGraph()
    {
        foreach (var orders in ((string[])["encoder2", "attn-chunk4"]).Select(RunOrders))
        {
            Assert.NotEmpty(orders);
            Assert.All(orders, order => Assert.Equal(order.Written, order.Ran));
        }
    }

    [Fact]
    public void TestASerializedModelProvesWhatItsGraphProvesAndOneWithoutAGraphProvesNothing()
    {
        OutputAlias[] pair = [new("O", "a")];
        var proved = GraphOf("a b", "O", Op("Sub", "a b", "O"));
        var refused = GraphOf("a b", "O Z", Op("Sub", "a b", "O"), Op("Neg", "a", "Z"));
        Assert.Single(OutputAliasProof.Prove(ModelOf(proved), pair));
        Assert.Equal(OutputAliasProof.Prove(proved, pair), OutputAliasProof.Prove(ModelOf(proved), pair));
        Assert.Equal(OutputAliasProof.Prove(refused, pair), OutputAliasProof.Prove(ModelOf(refused), pair));
        var graphless = new MemoryStream();
        ProtoBuf.Serializer.Serialize(graphless, new ModelProto { IrVersion = 8 });
        Assert.Empty(OutputAliasProof.Prove(graphless.ToArray(), pair));
    }

    [Fact]
    public void TestARunWritesAMarkedOutputIntoAnInputOnlyWhereItConsumedItAloneAndItsShapeIsSettled()
    {
        using var context = new ComputeContext();
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = new InternalComputationGraph([a, b], [a - b]);
        var compiled = context.Compile(graph, [[4L], [4L]], trainingStep: false, aliasCandidates: [(0, 0)]);
        long Aliased(CompiledGraph run, IData x, IData y, float[] expected)
        {
            var before = context.AliasedOutputs;
            Assert.Equal(expected, Floats(run.Execute(x, y)[0].ToTensorData()));
            return context.AliasedOutputs - before;
        }
        TensorData Tens() => TensorData([4L], (float[])[10f, 20f, 30f, 40f]);
        var kept = Tens();

        Assert.Equal([(0, 0)], compiled.MarkedPairs());
        Assert.Equal(1, Aliased(compiled, Tens(), Sample(), [9f, 18f, 27f, 36f]));
        Assert.Equal(0, Aliased(compiled, kept.Shared(), Sample(), [9f, 18f, 27f, 36f]));
        Assert.Equal([10f, 20f, 30f, 40f], Floats(kept));
        Assert.Equal(1, Aliased(compiled, kept.TryConsume(), Sample(), [9f, 18f, 27f, 36f]));
        var twice = Sample();
        Assert.Equal(0, Aliased(compiled, twice, twice, [0f, 0f, 0f, 0f]));
        var unsettled = context.Compile(graph, inputDims: null, trainingStep: false, aliasCandidates: [(0, 0)]);
        Assert.Equal(0, Aliased(unsettled, Tens(), Sample(), [9f, 18f, 27f, 36f]));
        var (s, t) = (InputScalar<float32>("s"), InputScalar<float32>("t"));
        var scalar = context.Compile(new InternalComputationGraph([s, t], [s - t]), [[], []], trainingStep: false, aliasCandidates: [(0, 0)]);
        Assert.Equal(1, Aliased(scalar, TensorData([], 10f), TensorData([], 3f), [7f]));
        var anyRank = InputTensor<float32>("w");
        var shapeless = context.Compile(new InternalComputationGraph([anyRank, b], [anyRank - b]), [null, [1L]], trainingStep: false, aliasCandidates: [(0, 0)]);
        Assert.Equal(0, Aliased(shapeless, TensorData([], 10f), TensorData([1L], 3f), [7f]));
        Assert.Empty(context.Compile(graph).MarkedPairs());
        using var unaliased = new ComputeContext { OutputAliasing = false };
        Assert.Empty(unaliased.Compile(graph, [[4L], [4L]], trainingStep: false, aliasCandidates: [(0, 0)]).MarkedPairs());
    }

    [Fact]
    public void TestAnOutputTheRuntimeFoldsToAConstantIsMemoryOfItsOwnOnEveryRun()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        float[] ThreeRuns(Variable output)
        {
            var compiled = context.Compile(new InternalComputationGraph([x], [output]), [[4L]], trainingStep: false);
            TensorData[] runs = [.. Enumerable.Range(0, 3).Select(_ => compiled.Execute(Sample())[0].ToTensorData())];
            Assert.False(SameMemory(runs[0], runs[1]) || SameMemory(runs[0], runs[2]) || SameMemory(runs[1], runs[2]));
            return [.. runs.SelectMany(run => run.As<float32>().CopyMemory<float>())];
        }

        Assert.Equal(Enumerable.Repeat(1f, 12),
            ThreeRuns(OnnxOp.ConstantOfShape(OnnxOp.Shape(x), TensorData(DType.Float32, [1L], 1f).MoveToAttribute())));
        Assert.Equal(Thrice(1f, 2f, 3f, 4f), ThreeRuns(Vector(1f, 2f, 3f, 4f)));
        Assert.Equal(Thrice(1f, 2f, 3f, 4f), ThreeRuns(OnnxOp.Identity(Vector(1f, 2f, 3f, 4f), rank: 1)));
        Assert.Equal(Thrice(1f, 2f, 3f, 4f), ThreeRuns(OnnxOp.Reshape(Vector(1f, 2f, 3f, 4f), Vector(2L, 2L), allowZero: false)));
        Assert.Equal(Thrice(11f, 22f, 33f, 44f), ThreeRuns(Vector(1f, 2f, 3f, 4f) + Vector(10f, 20f, 30f, 40f)));
    }

    private static float[] Thrice(params float[] values) => [.. values, .. values, .. values];

    private static bool SameMemory(TensorData a, TensorData b)
    {
        var same = Unsafe.AreSame(
            ref MemoryMarshal.GetReference(a.AccessRawMemory()), ref MemoryMarshal.GetReference(b.AccessRawMemory()));
        GC.KeepAlive(a);
        GC.KeepAlive(b);
        return same;
    }

    /// <summary>The model a lowering hands a backend for <paramref name="graph"/>.</summary>
    internal static byte[] ModelOf(GraphProto graph)
    {
        var model = new ModelProto { IrVersion = 8, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 17 });
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    /// <summary>A session of <paramref name="backend"/> over <paramref name="graph"/>, built to
    /// write its output O into its input a.</summary>
    internal static OrtSession Aliasing(IShorokooBackend backend, GraphProto graph)
        => (OrtSession)backend.CreateSession(
            ModelOf(graph), ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal,
            new DeviceMemorySettings(), DiagnosticSettings.Default,
            [new OutputAlias("O", "a")]);

    [Fact]
    public void TestAPairIsDroppedWhereTheGraphTheRuntimeRunsReadsTheInputAfterTheOutputIsWritten()
    {
        var graph = GraphOf("a:float[4,4] x:float[4,4] y:float[4,4]", "O:float[4,4] Z:float[4,4]",
            Op("Transpose", "a", "t"), Op("MatMul", "x t", "m"), Op("MatMul", "y t", "Z"), Op("Sub", "a m", "O"));
        using var session = Aliasing(DefaultBackend.Instance, graph);
        float[] a = [.. Enumerable.Range(1, 16).Select(v => (float)v)];
        float[] t = [.. Enumerable.Range(0, 16).Select(k => a[k % 4 * 4 + k / 4])];
        IShorokooTensorValue Square(float[] values) => DefaultBackend.Instance.CreateTensor(values, [4L, 4L]);
        float[] Diagonal(float v) => [.. Enumerable.Range(0, 16).Select(k => k % 5 == 0 ? v : 0f)];
        var consumed = Square(a);
        using var x = Square(Diagonal(1f));
        using var y = Square(Diagonal(2f));
        var outputs = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = consumed, ["x"] = x, ["y"] = y },
            [consumed], ["O", "Z"], RunSettings.Default, out var aliased);

        Assert.True(Proves(graph));
        Assert.Empty(session.BindableAliases);
        Assert.All(aliased, Assert.Null);
        Assert.Equal([.. a.Zip(t, (p, q) => p - q)], outputs[0].GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([.. t.Select(q => 2f * q)], outputs[1].GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public void TestASessionRunForACallerThatDoesNotAskWhatItAliasedStillWritesItsOutputIntoWhatItConsumed()
    {
        var backend = DefaultBackend.Instance;
        using var session = Aliasing(backend, GraphOf("a:float[4] b:float[4]", "O:float[4]", Op("Sub", "a b", "O")));
        var a = backend.CreateTensor<float>([10f, 20f, 30f, 40f], [4L]);
        using var b = backend.CreateTensor<float>([1f, 2f, 3f, 4f], [4L]);
        ref var consumedMemory = ref MemoryMarshal.GetReference(a.GetTensorDataAsSpan<float>());

        using var o = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["b"] = b }, [a], ["O"],
            RunSettings.Default)[0];

        Assert.Equal([9f, 18f, 27f, 36f], o.GetTensorDataAsSpan<float>().ToArray());
        Assert.True(Unsafe.AreSame(ref consumedMemory, ref MemoryMarshal.GetReference(o.GetTensorDataAsSpan<float>())));
    }

    [Fact]
    public void TestASessionThatCannotWriteItsGraphOutIsBuiltWithoutAliasingOnlyWhereTheRuntimeRefusesItsCompiledNodes()
    {
        var graph = GraphOf("a:float[4] b:float[4]", "O:float[4]", Op("Sub", "a b", "O"));
        var transient = new ScriptedBackend(build => { if (build == 0) throw new OutOfMemoryException(); });
        Assert.Throws<OutOfMemoryException>(() => Aliasing(transient, graph));

        var compiling = new ScriptedBackend(build =>
        {
            if (build == 0)
                throw OrtFailure("Unable to serialize model as it contains compiled nodes. Please disable any execution providers which generate compiled nodes.");
        });
        using var unaliased = Aliasing(compiling, graph);
        Assert.Equal((2, 0), (compiling.Builds, unaliased.BindableAliases.Count));
    }

    [Fact]
    public void TestASessionHoldingMoreThanSixteenMebibytesOfInitializersIsBuiltAgainWithoutWritingItsGraphOutAndStillAliases()
    {
        (int Builds, float O, string? Aliased) Built(ScriptedBackend backend, int floats)
        {
            var graph = GraphOf("a:float[1] i:int64[1]", "O:float[1]", Op("Gather", "C i", "g"), Op("Sub", "a g", "O"));
            graph.Initializers.Add(new TensorProto { Name = "C", Dims = [floats], data_type = 1, RawData = new byte[4L * floats] });
            using var session = Aliasing(backend, graph);
            var a = backend.CreateTensor<float>([5f], [1L]);
            using var i = backend.CreateTensor<long>([0L], [1L]);
            using var o = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["i"] = i }, [a], ["O"],
                RunSettings.Default, out var aliased)[0];
            return (backend.Builds, o.GetTensorDataAsSpan<float>()[0], aliased[0]);
        }

        const int SixteenMebibytes = 4 << 20;
        Assert.Equal((1, 5f, "a"), Built(new ScriptedBackend(_ => { }), SixteenMebibytes + 1));
        Assert.Equal((2, 5f, "a"), Built(ScriptedBackend.Stock(_ => { }), SixteenMebibytes + 1));
        Assert.Equal((1, 5f, "a"), Built(ScriptedBackend.Stock(_ => { }), SixteenMebibytes));
    }

    /// <summary>ONNX Runtime's own failure, which only ONNX Runtime constructs.</summary>
    private static OnnxRuntimeException OrtFailure(string message)
        => (OnnxRuntimeException)Activator.CreateInstance(
            typeof(OnnxRuntimeException), BindingFlags.NonPublic | BindingFlags.Instance, binder: null,
            [Enum.ToObject(typeof(OnnxRuntimeException).Assembly.GetType("Microsoft.ML.OnnxRuntime.ErrorCode")!, 1), message],
            culture: null)!;

    /// <summary>The CPU backend, calling <c>build</c> with the number of each session it builds,
    /// from 0, where a provider would be appended. Made with <c>new</c>, it is built through the
    /// constructor a subclass appending a provider of its own calls; made by <see cref="Stock"/>,
    /// it stands for ONNX Runtime's own CPU provider, as the CPU packages' backends do.</summary>
    private sealed class ScriptedBackend : OrtBackend
    {
        private readonly int[] _builds;

        internal ScriptedBackend(Action<int> build) : this([0], build) { }

        private ScriptedBackend(int[] builds, Action<int> build)
            : base((_, _, _) => build(builds[0]++), ComputeDevice.Cpu, cudaDeviceId: null) => _builds = builds;

        private ScriptedBackend(int[] builds, Action<int> build, bool stockProvider)
            : base((_, _, _) => build(builds[0]++), ComputeDevice.Cpu, cudaDeviceId: null, stockProvider) => _builds = builds;

        internal static ScriptedBackend Stock(Action<int> build) => new([0], build, stockProvider: true);

        internal int Builds => _builds[0];
    }

    [Fact]
    public void TestConsumingATensorAnotherRunIsReadingIsRefusedNamingThatRunAndTakesNothing()
    {
        using var context = new ComputeContext();
        var (graph, _) = Chain();
        var compiled = context.Compile(graph);
        var read = Wide32();
        var bystander = Wide32();
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var run = Task.Run(() => compiled.Run(
            new HeldFeed(read, reached, release) { FeedMode = SharedInputMode.Shared }));
        Assert.True(reached.Wait(TimeSpan.FromSeconds(10)));

        var refused = Assert.Throws<InvalidOperationException>(
            () => context.Execute(Model().Graph, bystander, read)).Message;
        Assert.Contains($"is being read by a run of the graph (a) -> (", refused);
        Assert.Contains("cannot consume it as input 'b'", refused);
        Assert.Contains(".Shared()", refused);
        Assert.False(bystander.IsDisposed);
        Assert.Equal(Floats(bystander).Select(v => 2 * v), Floats(context.Execute(Doubling(), read.TryConsume())[0].ToTensorData()));
        Assert.False(read.IsDisposed);

        release.Set();
        run.Wait();
        context.Execute(Doubling(), read.TryConsume());
        Assert.True(read.IsDisposed);
    }

    [Fact]
    public void TestARunThatFailsStillConsumesWhatItWasFedAsItIs()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        var failing = new InternalComputationGraph([x], [OnnxOp.Reshape(x, Vector(3L), allowZero: false)]);
        var fed = Sample();
        var kept = Sample();

        Assert.ThrowsAny<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => context.Execute(failing, fed));
        Assert.ThrowsAny<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => context.Execute(failing, kept.Shared()));

        Assert.True(fed.IsDisposed);
        Assert.True(fed.CopiesAreEmpty);
        Assert.False(kept.IsDisposed);
        Assert.Contains("consumed by a run of", Assert.Throws<ObjectDisposedException>(() => Floats(fed)).Message);
    }

    [Fact]
    public void TestTheBackendReleasesTheValueOfATensorARunConsumedWhetherTheRunSucceedsOrFails()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        var failing = new InternalComputationGraph([x], [OnnxOp.Reshape(x, Vector(3L), allowZero: false)]);
        bool Released(Action<TensorData> run)
        {
            var value = DefaultBackend.Instance.CreateTensor<float>([1f, 2f, 3f, 4f], [4L]);
            try { run(TensorData.Create(new Shape(4L), DType.Float32, value, DefaultBackend.Instance)); }
            catch (OnnxRuntimeException) { }
            try { _ = value.Shape; return false; }
            catch (ObjectDisposedException) { return true; }
        }

        Assert.All((bool[])[
            Released(t => context.Execute(Doubling(), t)),
            Released(t => context.Execute(failing, t)),
            Released(t => context.Compile(Doubling()).Execute(t)),
            Released(t => context.Compile(failing).Execute(t))], Assert.True);
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

        var fed = Wide32();
        Assert.Throws<OperationCanceledException>(() => compiled.Run(
            [NamedModelParam.FromIData("a", ModelParamType.InputParam, fed)], settings));
        Assert.Equal(expected, Floats(compiled.Execute(fed)[0].ToTensorData()));

        using var stopped = new ComputeContext { RunSettings = settings };
        var oneShot = Wide32();
        Assert.Throws<OperationCanceledException>(() => stopped.Run(
            graph, new TensorDataModelParam("a", ModelParamType.InputParam, oneShot)));
        Assert.Equal(expected, Floats(context.Run(
            graph, new TensorDataModelParam("a", ModelParamType.InputParam, oneShot))[0].ToTensorData()));
        Assert.True(oneShot.IsDisposed);
    }

    [Fact]
    public void TestAFeedNothingKnowsHowToHoldIsRefusedAndAStructIsToldToFeedItsFields()
    {
        using var context = new ComputeContext();
        using var feeds = new RunFeeds(context, DefaultBackend.Instance, new RunIdentity(() => "a run"), budget: null);
        TensorStructFieldDef[] fields = [new TensorStructFieldDef("x", DataStructure.Tensor, 1, DType.Float32)];
        var whole = new TensorStructModelParam("s", ModelParamType.InputParam, new TensorDataStruct(
            new TensorStructDef(fields, "S"), new Dictionary<string, IData> { { "x", Sample() } }));

        Assert.Throws<InvalidOperationException>(() => feeds.Prepare([new UnlockableParam()]));
        Assert.Contains("StructData", Assert.Throws<InvalidTensorOperationException>(() => feeds.Prepare([whole])).Message);
    }

    [Fact]
    public void TestAConsumedTensorSaysWhichRunTookItAndToPassItSharedAndARunFedItAgainTakesNothing()
    {
        using var context = new ComputeContext();
        var (graph, a, b, _) = Model();
        var compiled = context.Compile(graph);
        compiled.Execute(a, b.TryConsume());

        var bystander = Sample();
        Assert.Throws<ObjectDisposedException>(() => compiled.Execute(bystander, a));
        Assert.Throws<ObjectDisposedException>(() => context.Execute(graph, bystander, b));
        Assert.False(bystander.IsDisposed);

        var bare = Assert.Throws<ObjectDisposedException>(() => compiled.Execute(a, Sample())).Message;
        Assert.Contains(a.ToString(), bare);
        Assert.Contains("consumed by a run of the graph (a, b) -> (", bare);
        Assert.Contains($"on the compute context over {context.Backend}", bare);
        Assert.Contains("which it fed as input 'a'", bare);
        Assert.Contains("pass it there as .Shared()", bare);
        var tried = Assert.Throws<ObjectDisposedException>(() => Floats(b)).Message;
        Assert.Contains("which it fed as input 'b'", tried);
        Assert.Contains("passed as .TryConsume()", tried);
        Assert.Contains("pass it there as .Shared()", tried);
        Assert.Contains("or pass the struct, sequence or checkpoint that held it that way", tried);
        Assert.DoesNotContain("HostTensorData", tried);
    }

    [Fact]
    public void TestEveryAccessToADeadTensorThrowsSayingHowItDiedAndEndingItAgainIsHarmless()
    {
        using var context = new ComputeContext();
        TensorData Deleted() { var t = Sample(); t.Delete(); return t; }
        TensorData Moved() { var t = Sample(); _ = t.MoveToAttribute(); return t; }
        TensorData Consumed() { var t = Sample(); context.Execute(Doubling(), t); return t; }
        Func<TensorData, object>[] accesses =
        [
            t => t.CopyRawMemory(), t => t.Data, t => t.IsHostResident, t => t.ToTensorValue(),
            t => t.To(context), t => t.CopyTo(context), t => t.ToHost(), t => t.Shared(),
            t => t.TryConsume(), t => t.MoveToAttribute(), t => context.Lock(t),
        ];

        foreach (var (dead, cause) in (IEnumerable<(TensorData, string)>)[
            (Deleted(), "deleted"), (Moved(), "MoveToAttribute()"), (Consumed(), ".Shared()")])
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

        var deleted = Sample().CopyTo(context);
        var moved = Sample().To(context);
        Assert.Contains(deleted, context.Tensors);
        Assert.Contains(moved, context.Tensors);

        deleted.Dispose();
        _ = moved.MoveToAttribute();

        Assert.DoesNotContain(deleted, context.Tensors);
        Assert.DoesNotContain(moved, context.Tensors);
    }

    [Fact]
    public void TestATensorReadFromAStreamFeedsAsACopiedOneDoesAndOneWithNoFlatBufferIsRefused()
    {
        using var context = new ComputeContext();
        var graph = Doubling();
        float[] values = [1f, 2f, 3f, 4f];

        var read = context.ReadTensor(new Shape(4L), DType.Float32, new MemoryStream(MemoryMarshal.AsBytes(values.AsSpan()).ToArray()));

        Assert.Contains(read, context.Tensors);
        Assert.Same(DefaultBackend.Instance, read.AllocatingBackend);
        Assert.Equal(values, Floats(read));
        Assert.Equal(
            Floats(context.Execute(graph, Sample().CopyTo(context))[0].ToTensorData()),
            Floats(context.Execute(graph, read)[0].ToTensorData()));

        long[] pair = [2L, 3L];
        Assert.Equal(24, ComputeContext.Host.ReadTensor(pair, DType.Float32, new MemoryStream(new byte[24])).CopyRawMemory().Length);
        Assert.Throws<EndOfStreamException>(() => context.ReadTensor(pair, DType.Float32, new MemoryStream(new byte[8])));
        Assert.Throws<NotSupportedException>(() => context.ReadTensor(pair, DType.Utf8, Stream.Null));
        Assert.Throws<NotSupportedException>(() => context.ReadTensor(pair, DType.Int4, Stream.Null));
        Assert.Throws<NotSupportedException>(() => ComputeContext.Host.ReadTensor(pair, DType.UInt4, Stream.Null));
        Assert.Throws<ArgumentNullException>(() => context.ReadTensor(pair, null!, Stream.Null));

        using var budgeted = new ComputeContext(new StubBackend(ComputeDevice.Cuda, 0))
        {
            DeviceMemory = new DeviceMemorySettings { LimitBytes = 64 },
        };
        foreach (var complex in (DType[])[DType.Complex64, DType.Complex128])
            foreach (var where in (ComputeContext[])[context, ComputeContext.Host, budgeted])
                Assert.Contains("complex", Assert.Throws<NotSupportedException>(() => where.ReadTensor(pair, complex, Stream.Null)).Message);
    }

    /// <summary>Holds a run open once it has held its feeds and before it builds their values:
    /// inside the window a disposal of its context has to be refused in.</summary>
    private sealed class HeldFeed(
        TensorData data, ManualResetEventSlim reached, ManualResetEventSlim release)
        : TensorDataModelParam("a", ModelParamType.InputParam, data)
    {
        internal override void Held()
        {
            reached.Set();
            release.Wait(TimeSpan.FromSeconds(30));
        }
    }

    /// <summary>A parameter of a kind no run knows how to hold, which is what
    /// <c>RunFeeds.Prepare</c>'s refusal is for.</summary>
    private sealed class UnlockableParam : NamedModelParam
    {
        internal override IShorokooTensorValue ToTensorValue() => throw new NotSupportedException();
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

/// <summary>
/// Where the process makes its temporary files, which every test making one reads, so these run
/// alone.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
[Collection(ProcessWideTempFolder.Name)]
public class ProcessWideTempFolderCoverageTests
{
    [Fact]
    public void TestASessionWithNowhereToWriteItsGraphOutIsBuiltAliasingNothing()
    {
        var graph = ComputeContextLifetimeCoverageTests.GraphOf("a:float[4] b:float[4]", "O:float[4]",
            ComputeContextLifetimeCoverageTests.Op("Sub", "a b", "O"));
        using (var writable = ComputeContextLifetimeCoverageTests.Aliasing(DefaultBackend.Instance, graph))
            Assert.Single(writable.BindableAliases);

        var blocker = Path.GetTempFileName();
        var variable = OperatingSystem.IsWindows() ? "TMP" : "TMPDIR";
        var was = Environment.GetEnvironmentVariable(variable);
        try
        {
            // A file where a folder would have to be, so nothing can be made in the temp folder.
            Environment.SetEnvironmentVariable(variable, Path.Combine(blocker, "temp"));
            using var session = ComputeContextLifetimeCoverageTests.Aliasing(DefaultBackend.Instance, graph);
            Assert.Empty(session.BindableAliases);
        }
        finally
        {
            Environment.SetEnvironmentVariable(variable, was);
            File.Delete(blocker);
        }
    }
}
