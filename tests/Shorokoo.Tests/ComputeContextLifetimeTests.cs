using System.Reflection;
using System.Runtime.CompilerServices;
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
    /// Shorokoo/Shorokoo#366: writing to a tensor on one thread while a run on another reads it
    /// used to free the buffer the execution provider was reading. The write retires the copy the
    /// run reads through, and the retired copy waits for the run.
    /// </summary>
    [Fact]
    public void TestATensorWrittenOnAnotherThreadStaysValidForTheRunFeedingIt()
    {
        using var context = new ComputeContext();
        var (graph, expected) = Chain();
        var compiled = context.Compile(graph);
        var copyAt = TensorData.RunMemoryOf(DefaultBackend.Instance, DType.Float32);

        for (int round = 0; round < 3; round++)
        {
            var fed = Wide32();
            var ran = false;
            using var spinning = new ManualResetEventSlim();
            var other = Task.Run(() =>
            {
                spinning.Set();
                var spin = new SpinWait();
                while (fed.CopyHeldAt(copyAt) is not { IsLocked: true } && !Volatile.Read(ref ran))
                    spin.SpinOnce(sleep1Threshold: -1);
                fed.AccessModifiableMemory<float>()[0] = 99f;
            });
            Assert.True(spinning.Wait(TimeSpan.FromSeconds(10)));

            float[] result;
            try { result = Floats(compiled.Execute(fed.Shared())[0].ToTensorData()); }
            finally { Volatile.Write(ref ran, true); }
            other.Wait();

            Assert.Equal(expected, result);
        }
    }

    [Fact]
    public void TestARunReadingATensorWhileItIsWrittenLeavesNothingStaleForTheNextRun()
    {
        using var context = new ComputeContext();
        var t = Sample();

        t.As<float32>().WriteMemory<float>(span =>
        {
            span[0] = 9f;
            context.Execute(Doubling(), t.Shared());
            span[1] = 9f;
        });

        Assert.Equal([18f, 18f, 6f, 8f], Floats(context.Execute(Doubling(), t.Shared())[0].ToTensorData()));
    }

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
    private static GraphProto GraphOf(string inputs, string outputs, params NodeProto[] nodes)
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

    private static NodeProto Op(string op, string inputs, string outputs, string domain = "", GraphProto? body = null)
    {
        var node = new NodeProto { OpType = op, Domain = domain };
        node.Inputs.AddRange(Names(inputs));
        node.Outputs.AddRange(Names(outputs));
        if (body is not null)
            node.Attributes.Add(new AttributeProto { Name = "then_branch", Type = AttributeProto.AttributeType.Graph, G = body });
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
        Assert.True(Proves(GraphOf("a:float[4] b", "O:float[4]", Op("Neg", "b", "O"))));
        Assert.False(Proves(GraphOf("a:float[4] b", "O:double[4]", Op("Cast", "b", "O"))));
        Assert.False(Proves(GraphOf("a:float[4] b", "O:float[2,2]", Op("Neg", "b", "O"))));
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
    public void TestAnOutputTheRuntimeFoldsToAConstantIsMemoryOfItsOwnThatAWriteDoesNotCarryIntoAnotherRun()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        float[] AfterAWrite(Variable output)
        {
            var compiled = context.Compile(new InternalComputationGraph([x], [output]), [[4L]], trainingStep: false);
            TensorData<float32> Run() => compiled.Execute(Sample())[0].ToTensorData().As<float32>();
            var (first, second) = (Run(), Run());
            first.WriteMemory<float>(written => written.Fill(9f));
            return [.. second.CopyMemory<float>(), .. Run().CopyMemory<float>()];
        }

        Assert.Equal([1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f],
            AfterAWrite(OnnxOp.ConstantOfShape(OnnxOp.Shape(x), TensorData(DType.Float32, [1L], 1f).MoveToAttribute())));
        Assert.Equal([1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f], AfterAWrite(Vector(1f, 2f, 3f, 4f)));
        Assert.Equal([1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f], AfterAWrite(OnnxOp.Identity(Vector(1f, 2f, 3f, 4f), rank: 1)));
        Assert.Equal([1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f], AfterAWrite(OnnxOp.Reshape(Vector(1f, 2f, 3f, 4f), Vector(2L, 2L), allowZero: false)));
        Assert.Equal([11f, 22f, 33f, 44f, 11f, 22f, 33f, 44f], AfterAWrite(Vector(1f, 2f, 3f, 4f) + Vector(10f, 20f, 30f, 40f)));
    }

    /// <summary>The model a lowering hands a backend for <paramref name="graph"/>.</summary>
    private static byte[] ModelOf(GraphProto graph)
    {
        var model = new ModelProto { IrVersion = 8, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 17 });
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    /// <summary>A session of <paramref name="backend"/> over <paramref name="graph"/>, built to
    /// write its output O into its input a.</summary>
    private static OrtSession Aliasing(IShorokooBackend backend, GraphProto graph)
        => (OrtSession)backend.CreateSession(
            ModelOf(graph), ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Fatal,
            new DeviceMemorySettings().Resolve(reusedAcrossShapes: false), DiagnosticSettings.Default,
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
            [consumed], ["O", "Z"], ComputeContext.NoOutputsRetained, RunSettings.Default, out var aliased);

        Assert.True(Proves(graph));
        Assert.Empty(session.OutputAliases);
        Assert.All(aliased, Assert.Null);
        Assert.Equal([.. a.Zip(t, (p, q) => p - q)], outputs[0].GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([.. t.Select(q => 2f * q)], outputs[1].GetTensorDataAsSpan<float>().ToArray());
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
        Assert.Equal((2, 0), (compiling.Builds, unaliased.OutputAliases.Count));
    }

    [Fact]
    public void TestASessionHoldingMoreThanSixteenMebibytesOfInitializersIsBuiltAgainWithoutWritingItsGraphOutAndStillAliases()
    {
        (int Builds, float O, string? Aliased) Built(int floats, ComputeDevice device = ComputeDevice.Cpu)
        {
            var backend = new ScriptedBackend(_ => { }, device);
            var graph = GraphOf("a:float[1] i:int64[1]", "O:float[1]", Op("Gather", "C i", "g"), Op("Sub", "a g", "O"));
            graph.Initializers.Add(new TensorProto { Name = "C", Dims = [floats], data_type = 1, RawData = new byte[4L * floats] });
            using var session = Aliasing(backend, graph);
            var a = backend.CreateTensor<float>([5f], [1L]);
            using var i = backend.CreateTensor<long>([0L], [1L]);
            using var o = session.RunConsuming(new Dictionary<string, IShorokooTensorValue> { ["a"] = a, ["i"] = i }, [a], ["O"],
                ComputeContext.NoOutputsRetained, RunSettings.Default, out var aliased)[0];
            return (backend.Builds, o.GetTensorDataAsSpan<float>()[0], aliased[0]);
        }

        const int SixteenMebibytes = 4 << 20;
        Assert.Equal((2, 5f, "a"), Built(SixteenMebibytes + 1));
        Assert.Equal((1, 5f, "a"), Built(SixteenMebibytes));
        Assert.Equal((1, 5f, "a"), Built(SixteenMebibytes + 1, ComputeDevice.Other));
    }

    /// <summary>ONNX Runtime's own failure, which only ONNX Runtime constructs.</summary>
    private static OnnxRuntimeException OrtFailure(string message)
        => (OnnxRuntimeException)Activator.CreateInstance(
            typeof(OnnxRuntimeException), BindingFlags.NonPublic | BindingFlags.Instance, binder: null,
            [Enum.ToObject(typeof(OnnxRuntimeException).Assembly.GetType("Microsoft.ML.OnnxRuntime.ErrorCode")!, 1), message],
            culture: null)!;

    /// <summary>The CPU backend, calling <c>build</c> with the number of each session it builds,
    /// from 0, where a provider would be appended.</summary>
    private sealed class ScriptedBackend : OrtBackend
    {
        private readonly int[] _builds;

        internal ScriptedBackend(Action<int> build, ComputeDevice device = ComputeDevice.Cpu)
            : this([0], build, device) { }

        private ScriptedBackend(int[] builds, Action<int> build, ComputeDevice device)
            : base((_, _) => build(builds[0]++), device, cudaDeviceId: null) => _builds = builds;

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
        using var feeds = new RunFeeds(context, DefaultBackend.Instance, new RunIdentity(() => "a run"));
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
        Assert.Throws<NotSupportedException>(() => context.AllocateUninitialized(pair, DType.Int4));
        Assert.Throws<NotSupportedException>(() => ComputeContext.Host.AllocateUninitialized(pair, DType.UInt4));
        Assert.Throws<ArgumentNullException>(() => context.AllocateUninitialized(pair, null!));
    }

    /// <summary>Holds a run open where the value is built: after the run has held the feed, and
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
