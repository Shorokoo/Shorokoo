using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Whether a tensor is handed to a context as it is or copied into its memory is asked of the
/// context's backend, given where the tensor's memory is: the same device and the same runtime. Two
/// cards are two memory spaces, so a tensor reaching another card is copied through the host; two
/// backends on one card copy too unless they share a runtime. There is no direct device-to-device
/// path.
///
/// <para>Stubbed backends rather than real ones, deliberately: the decision under test is which
/// route a transfer takes, and that is made from what the backends answer. Proving it needs two
/// CUDA devices to <i>report</i>, not two to exist — and a machine with one card could not run
/// this at all otherwise. What a second card would add is confidence that the copy itself lands
/// correctly on a device this process has not been using, which no stub can stand in for.</para>
///
/// <para>The same stubs cover which backend a tensor's memory is released through, and a context's
/// device-memory budget — what it counts, what it refuses, the arena limit its sessions are built
/// with and when they are built again — for the same reason: a stub card is what makes those
/// hand-offs observable without one.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class CrossDeviceRoutingCoverageTests
{
    [Fact]
    public void TestTwoCardsAreTwoSpacesAndOneCardIsOne()
    {
        Assert.NotEqual(MemorySpace.Cuda(0), MemorySpace.Cuda(1));
        Assert.Equal(MemorySpace.Cuda(0), MemorySpace.Cuda(0));
        Assert.NotEqual(MemorySpace.Host, MemorySpace.Cuda(0));

        IShorokooBackend first = new StubBackend(ComputeDevice.Cuda, 0);
        IShorokooBackend second = new StubBackend(ComputeDevice.Cuda, 0);
        Assert.Equal(first.MemorySpace, second.MemorySpace);
        Assert.Equal(MemorySpace.Cuda(0), first.MemorySpace);

        var runtime = new ValueRuntime(1);
        Assert.Equal(new MemoryLocation(MemorySpace.Host, runtime), new MemoryLocation(MemorySpace.Host, runtime));
        Assert.NotEqual(new MemoryLocation(MemorySpace.Host, runtime), new MemoryLocation(MemorySpace.Host, new ValueRuntime(1)));
    }

    /// <summary>A runtime identity equal by value to any other of the same number.</summary>
    private sealed record ValueRuntime(int Number);

    [Fact]
    public void TestTwoBackendsOnAnUnnamedDeviceSharingARuntimeEachReadOnlyWhatItAllocated()
    {
        var shared = new object();
        var first = new StubBackend(ComputeDevice.Other, null) { Runtime = shared };
        var second = new StubBackend(ComputeDevice.Other, null) { Runtime = shared };
        using var onFirst = new ComputeContext(first);
        using var onSecond = new ComputeContext(second);
        var allocated = TensorData([2L], (float[])[1f, 2f]).CopyTo(onFirst);

        Assert.True(allocated.FeedsInPlace(first));
        Assert.False(allocated.FeedsInPlace(second));
        Assert.Same(allocated, allocated.To(onFirst));
        Assert.NotSame(allocated, allocated.To(onSecond));
    }

    [Fact]
    public void TestATensorAlreadyOnOneCardReachesAnotherOnlyByGoingThroughTheHost()
    {
        var firstCard = new StubBackend(ComputeDevice.Cuda, 0);
        var secondCard = new StubBackend(ComputeDevice.Cuda, 1);
        using var one = new ComputeContext(firstCard);
        using var two = new ComputeContext(secondCard);

        var onFirst = TensorData([2L], (float[])[3f, 4f]).To(one);
        Assert.Equal(MemorySpace.Cuda(0), onFirst.Space);
        Assert.Same(firstCard, onFirst.AllocatingBackend);
        Assert.False(onFirst.IsHostResident);

        var onSecond = onFirst.To(two);

        Assert.Equal(1, firstCard.HostCopies);
        Assert.Equal(1, secondCard.BackendMemoryBuilds);
        Assert.Equal(MemorySpace.Cuda(1), onSecond.Space);
        Assert.Same(secondCard, onSecond.AllocatingBackend);
        Assert.False(onFirst.IsDisposed);
        Assert.Contains(onFirst, one.Tensors);
        Assert.Contains(onSecond, two.Tensors);
    }

    [Fact]
    public void TestTwoContextsOnOneCardShareItsAllocationOnlyWhenTheyShareARuntime()
    {
        var backend = new StubBackend(ComputeDevice.Cuda, 0);
        var otherRuntime = new StubBackend(ComputeDevice.Cuda, 0);
        using var one = new ComputeContext(backend);
        using var alsoOne = new ComputeContext(backend);
        using var other = new ComputeContext(otherRuntime);

        var onCard = TensorData([2L], (float[])[5f, 6f]).To(one);

        Assert.Same(onCard, onCard.To(alsoOne));
        Assert.Contains(onCard, alsoOne.Tensors);
        Assert.Equal(0, backend.HostCopies);

        var copied = onCard.To(other);
        Assert.NotSame(onCard, copied);
        Assert.Equal(1, backend.HostCopies);
        Assert.Equal(1, otherRuntime.BackendMemoryBuilds);
    }

    [Fact]
    public void TestAnUnknownMemorySpaceIsReadOnlyByTheBackendThatMadeItAndComesHomeThroughIt()
    {
        var other = new StubBackend(ComputeDevice.Other, null);
        Assert.Equal(MemorySpace.UnknownDevice, ((IShorokooBackend)other).MemorySpace);
        using var context = new ComputeContext(other);
        using var elsewhere = new ComputeContext(new StubBackend(ComputeDevice.Other, null));

        var onUnknown = TensorData([2L], (float[])[3f, 4f]).CopyTo(context);
        Assert.False(onUnknown.Space.IsKnown);

        Assert.Same(onUnknown, onUnknown.To(context));
        Assert.NotSame(onUnknown, onUnknown.To(elsewhere));
        var home = onUnknown.ToHost();

        Assert.Equal(2, other.HostCopies);
        Assert.Equal([3f, 4f], (float[])[.. home.As<float32>().AccessMemory<float>()]);
        Assert.Same(HostBackend.Instance, home.AllocatingBackend);
    }

    [Fact]
    public void TestToAndCopyToAnotherCardGoThroughHostBytesAndLeaveTheSourceWhereItIs()
    {
        var target = new StubBackend(ComputeDevice.Cuda, 1);
        using var secondCard = new ComputeContext(target);
        var onHost = TensorData([2L], (float[])[7f, 8f]);

        var moved = onHost.To(secondCard);
        var copy = onHost.CopyTo(secondCard);

        Assert.Equal(2, target.BackendMemoryBuilds);
        Assert.Equal(MemorySpace.Cuda(1), moved.Space);
        Assert.Equal(MemorySpace.Cuda(1), copy.Space);
        Assert.NotSame(moved, copy);
        Assert.False(onHost.IsDisposed);
        Assert.Equal([7f, 8f], onHost.As<float32>().AccessMemory<float>().ToArray());
        Assert.Equal([7f, 8f], (float[])[.. moved.ToHost().As<float32>().AccessMemory<float>()]);
    }

    [Fact]
    public void TestWhetherATensorIsHandedOverOrCopiedIsAskedOfTheTargetsBackend()
    {
        var everything = new StubBackend(ComputeDevice.Cuda, 0) { Addresses = _ => true };
        var nothing = new StubBackend(ComputeDevice.Cpu, null) { Addresses = _ => false };
        using var generous = new ComputeContext(everything);
        using var strict = new ComputeContext(nothing);
        var onHost = TensorData([2L], (float[])[1f, 2f]);

        Assert.Same(onHost, onHost.To(generous));
        Assert.Equal([onHost.Location], everything.AskedAbout);
        Assert.Equal(0, everything.BackendMemoryBuilds);

        Assert.NotSame(onHost, onHost.To(strict));
        Assert.Equal([onHost.Location], nothing.AskedAbout);
    }

    [Fact]
    public void TestATensorsMemoryIsReleasedThroughTheBackendThatMadeItWhateverItIsAttachedTo()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        var one = new ComputeContext(card);
        var alsoOne = new ComputeContext(card);
        using var other = new ComputeContext(new StubBackend(ComputeDevice.Cuda, 1));

        var onCard = TensorData([2L], (float[])[1f, 2f]).To(one);
        Assert.Same(onCard, onCard.To(alsoOne));
        var elsewhere = onCard.To(other);

        one.Dispose();
        alsoOne.Dispose();
        Assert.Empty(card.Released);
        Assert.False(onCard.IsDisposed);

        onCard.Delete();
        Assert.Single(card.Released);
        Assert.Same(card.Built[0], card.Released[0]);
        Assert.False(elsewhere.IsDisposed);
    }

    [Fact]
    public void TestAContextCountsTheTensorsAttachedToItInItsOwnMemoryAgainstItsBudget()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(card) { DeviceMemory = Budget(64) };
        using var unbudgeted = new ComputeContext(card);
        using var onHost = new ComputeContext(new StubBackend(ComputeDevice.Cpu, null)) { DeviceMemory = Budget(64) };
        Assert.Equal(new DeviceMemoryUse(0, 0, 64), budgeted.ReadDeviceMemoryUse());

        var copied = Floats(4).CopyTo(budgeted);
        var moved = Floats(2).To(budgeted);
        var allocated = budgeted.AllocateUninitialized<float32>(new Shape(1L));
        var sharedWithIt = Floats(3).CopyTo(unbudgeted).To(budgeted);
        var hostSide = Floats(8).To(onHost);

        Assert.Equal(new DeviceMemoryUse(40, 4, 64), budgeted.ReadDeviceMemoryUse());
        Assert.Equal(24L, budgeted.ReadDeviceMemoryUse().AvailableBytes);
        Assert.Equal(new DeviceMemoryUse(12, 1, null), unbudgeted.ReadDeviceMemoryUse());
        Assert.Equal(new DeviceMemoryUse(32, 1, null), onHost.ReadDeviceMemoryUse());
        Assert.Equal(default, ComputeContext.Host.ReadDeviceMemoryUse());

        budgeted.Detach(copied);
        moved.Delete();
        Assert.Equal(new DeviceMemoryUse(16, 2, 64), budgeted.ReadDeviceMemoryUse());
        Assert.False(copied.IsDisposed);
        Assert.Same(sharedWithIt, Assert.Single(unbudgeted.Tensors));
        GC.KeepAlive((object[])[allocated, hostSide]);
    }

    [Fact]
    public void TestAPlacementOntoABudgetedCardIsRefusedBeforeItAllocatesWhenItWouldPassTheLimit()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(card) { DeviceMemory = Budget(32) };
        using var unbudgeted = new ComputeContext(card);
        var held = Floats(6).CopyTo(budgeted);
        var elsewhere = Floats(4).CopyTo(unbudgeted);
        string Refused(Func<object> place) => Assert.Throws<InvalidOperationException>(place).Message;

        var copy = Refused(() => Floats(4).CopyTo(budgeted));
        Assert.Contains("CopyTo(context) of Tensor (4,):Float32 asks this compute context for 16 bytes", copy);
        Assert.Contains("the budget (DeviceMemorySettings.LimitBytes) is 32 bytes, and 24 bytes of it", copy);
        Assert.Contains("To(context) of Tensor (4,):Float32", Refused(() => Floats(4).To(budgeted)));
        Assert.Contains("AllocateUninitialized of (4,):Float32", Refused(() => budgeted.AllocateUninitialized<float32>(new Shape(4L))));
        Assert.Contains("To(context) of Tensor (4,):Float32", Refused(() => elsewhere.To(budgeted)));
        Assert.Equal(2, card.Built.Count);
        Assert.DoesNotContain(elsewhere, budgeted.Tensors);

        var toTheLimit = Floats(2).CopyTo(budgeted);
        Assert.Equal(new DeviceMemoryUse(32, 2, 32), budgeted.ReadDeviceMemoryUse());
        held.Delete();
        Assert.Same(elsewhere, elsewhere.To(budgeted));
        Assert.Same(elsewhere, elsewhere.To(budgeted));
        Assert.Equal(new DeviceMemoryUse(24, 2, 32), budgeted.ReadDeviceMemoryUse());
        Assert.Equal(1024L, Floats(256).CopyTo(unbudgeted).ByteCount);
        GC.KeepAlive(toTheLimit);
    }

    [Fact]
    public void TestASessionsArenaIsTheBudgetLessWhatItsContextHoldsOutsideItAndOnlyEverComesDown()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Echo());
        long?[] Limits() => [.. card.Sessions.Select(s => s.LimitBytes)];

        Assert.Equal([6300L], Limits());
        compiled.Execute(Floats(10));
        var held = Floats(250).CopyTo(context);
        compiled.Execute(Floats(10));
        compiled.Execute(Floats(10));
        Assert.Equal([6300L, 5300L], Limits());
        Assert.Equal(5300L, compiled.DeviceMemory.LimitBytes);

        held.Delete();
        compiled.Execute(Floats(10));
        Assert.Equal([6300L, 5300L], Limits());

        context.Execute(Echo(), Floats(10));
        Assert.Equal([6300L, 5300L, 6300L], Limits());
        Assert.Equal(5, card.Runs.Count);
        Assert.All(card.Runs, run => Assert.True(run.ShrinkArenaAfterRun));

        var free = new StubBackend(ComputeDevice.Cuda, 0);
        using var unbudgeted = new ComputeContext(free);
        unbudgeted.Compile(Echo()).Execute(Floats(10));
        Assert.Null(Assert.Single(free.Sessions).LimitBytes);
        Assert.False(Assert.Single(free.Runs).ShrinkArenaAfterRun);
    }

    [Fact]
    public void TestWhatASessionsOwnRunsLeftInItsArenaIsInsideItsLimitUntilTheSessionIsBuiltAgain()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Echo());
        TensorData Kept() => compiled.Execute([Floats(20)], [true])[0].ToTensorData();

        var (first, second, third) = (Kept(), Kept(), Kept());
        Assert.False(first.IsHostResident);
        Assert.Single(card.Sessions);

        var placed = Floats(100).CopyTo(context);
        var fourth = Kept();
        Assert.Equal([6300L, 5700L], card.Sessions.Select(s => s.LimitBytes));
        compiled.Execute(second.Shared());
        Assert.Equal(2, card.Sessions.Count);
        GC.KeepAlive((object[])[first, third, placed, fourth]);
    }

    [Fact]
    public void TestARunItsContextsBudgetCannotFitIsRefusedBeforeItTakesWhatItWasFed()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card) { DeviceMemory = Budget(64) };
        var compiled = context.Compile(Echo());
        var fed = Floats(16);

        var consumed = Assert.Throws<InvalidOperationException>(() => compiled.Execute(fed)).Message;
        Assert.Contains("copy 64 bytes it was fed into the arena it computes in", consumed);
        Assert.Contains("leaves 63 bytes", consumed);
        var read = Assert.Throws<InvalidOperationException>(() => compiled.Execute(fed.Shared())).Message;
        Assert.Contains("would hold 64 bytes of CUDA device 0 memory outside its own arena", read);
        Assert.Contains("0 bytes of the 0 tensor(s) attached to its compute context there, and 64 bytes more", read);
        Assert.Contains("64-byte device-memory budget", read);
        Assert.Throws<InvalidOperationException>(() => context.Execute(Echo(), fed));
        Assert.False(fed.IsDisposed);
        Assert.Empty(card.Built);

        Assert.Equal([1f, 2f, 3f], Run(compiled, TensorData([3L], (float[])[1f, 2f, 3f])));
        var full = context.AllocateUninitialized<float32>(new Shape(16L));
        Assert.Contains("leave nothing of the 64 bytes it has", Assert.Throws<InvalidOperationException>(
            () => context.Compile(Echo())).Message);
        GC.KeepAlive(full);
    }

    [Fact]
    public void TestACopyARunReadsCountsOnTheBudgetOfEveryContextWhoseRunsReadIt()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var maker = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        using var reader = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        using var small = new ComputeContext(card) { DeviceMemory = Budget(40) };
        var source = Floats(10);

        Run(maker.Compile(Echo()), source.Shared());
        Run(reader.Compile(Echo()), source.Shared());
        Assert.Single(card.Built);
        Assert.Equal(40L, maker.ReadDeviceMemoryUse().AttachedBytes);
        Assert.Equal(40L, reader.ReadDeviceMemoryUse().AttachedBytes);

        Assert.Throws<InvalidOperationException>(() => small.Execute(Echo(), source.Shared()));
        Assert.Equal(0L, small.ReadDeviceMemoryUse().AttachedBytes);
        Assert.False(source.IsDisposed);
    }

    [Fact]
    public void TestACopyARunMeantToReadButFoundRetiredIsReplacedUnlessAnotherRunStillHoldsIt()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        using var reader = new ComputeContext(card);
        var compiled = context.Compile(Echo());
        var source = Floats(100);
        NamedModelParam Writing(float value) => Hooked(source, () => source.As<float32>().AccessModifiableMemory<float>()[0] = value);

        Run(compiled, source.Shared());
        Assert.Equal(9f, compiled.Run(Writing(9f))[0].ToTensorData().As<float32>().ValueAt<float>(0));

        var copy = Assert.Single(context.Tensors, t => t.Space == MemorySpace.Cuda(0));
        using var reading = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var held = Task.Run(() => reader.Compile(Echo()).Run(
            Hooked(copy, () => { reading.Set(); release.Wait(TimeSpan.FromSeconds(10)); })));
        Assert.True(reading.Wait(TimeSpan.FromSeconds(10)));
        Assert.Contains("still held by another run", Assert.Throws<InvalidOperationException>(
            () => compiled.Run(Writing(7f))).Message);
        release.Set();
        Assert.True(held.Wait(TimeSpan.FromSeconds(10)));
    }

    [Fact]
    public void TestABudgetedContextRunsOneAtATimeAndPlacesAndCompilesNothingWhileItRunsWhereAnUnbudgetedOneOverlaps()
    {
        static (int MostAtOnce, bool PlacedDuringRun, bool CompiledDuringRun) Overlap(
            ComputeContext context, StubBackend card, bool expectingOverlap)
        {
            var patience = expectingOverlap ? TimeSpan.FromSeconds(10) : TimeSpan.FromMilliseconds(200);
            var compiled = context.Compile(Echo());
            using var inside = new SemaphoreSlim(0);
            using var release = new ManualResetEventSlim();
            var now = 0;
            var most = 0;
            card.DuringRun = () =>
            {
                InterlockedMax(ref most, Interlocked.Increment(ref now));
                inside.Release();
                release.Wait(TimeSpan.FromSeconds(10));
                Interlocked.Decrement(ref now);
            };
            Task[] runs = [Task.Run(() => compiled.Execute(Floats(1))), Task.Run(() => compiled.Execute(Floats(1)))];
            Assert.True(inside.Wait(TimeSpan.FromSeconds(10)));
            inside.Wait(patience);
            var placement = Task.Run(() => Floats(1).CopyTo(context));
            var compile = Task.Run(() => context.Compile(Echo()));
            var placed = placement.Wait(patience);
            var compiledDuringRun = compile.Wait(patience);
            release.Set();
            Assert.True(Task.WaitAll([.. runs, placement, compile], TimeSpan.FromSeconds(10)));
            return (most, placed, compiledDuringRun);
        }

        var budgetedCard = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(budgetedCard) { DeviceMemory = Budget(1L << 20) };
        Assert.Equal((1, false, false), Overlap(budgeted, budgetedCard, expectingOverlap: false));

        var unbudgetedCard = new StubBackend(ComputeDevice.Cuda, 0);
        using var unbudgeted = new ComputeContext(unbudgetedCard);
        Assert.Equal((2, true, true), Overlap(unbudgeted, unbudgetedCard, expectingOverlap: true));
    }

    [Fact]
    public void TestABudgetedRunWaitingForTheOneInFlightStopsWhenCancelledHavingTakenNothing()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var cancel = new CancellationTokenSource();
        using var context = new ComputeContext(card)
        {
            DeviceMemory = Budget(1L << 20),
            RunSettings = new RunSettings { CancellationToken = cancel.Token },
        };
        var compiled = context.Compile(Echo());
        using var inside = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        card.DuringRun = () => { inside.Set(); release.Wait(TimeSpan.FromSeconds(10)); };
        var inFlight = Task.Run(() => compiled.Run(
            [NamedModelParam.FromIData("a", ModelParamType.InputParam, Floats(1))], RunSettings.Default));
        Assert.True(inside.Wait(TimeSpan.FromSeconds(10)));

        var (compiledFeed, oneShotFeed) = (Floats(1), Floats(1));
        Task[] waiting = [Task.Run(() => compiled.Execute(compiledFeed)), Task.Run(() => context.Execute(Echo(), oneShotFeed))];
        Assert.Equal(-1, Task.WaitAny(waiting, TimeSpan.FromMilliseconds(200)));
        cancel.Cancel();
        Assert.All(waiting, stopped => Assert.IsAssignableFrom<OperationCanceledException>(
            Assert.Throws<AggregateException>(() => stopped.Wait(TimeSpan.FromSeconds(10))).InnerException));
        Assert.False(inFlight.IsCompleted);
        Assert.False(compiledFeed.IsDisposed || oneShotFeed.IsDisposed);

        release.Set();
        Assert.True(inFlight.Wait(TimeSpan.FromSeconds(10)));
    }

    [Fact]
    public void TestASequencePlacedOnABudgetedCardAttachesAllItsElementsOrNone()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(card) { DeviceMemory = Budget(32) };
        using var unbudgeted = new ComputeContext(card);
        var held = Floats(4).CopyTo(budgeted);
        TensorData[] elements = [Floats(2).CopyTo(unbudgeted), Floats(3).CopyTo(unbudgeted)];
        var sequence = TensorDataSequence.OfElements([.. elements], DType.Float32);

        Assert.Contains("To(context) of a sequence of 2 tensors asks this compute context for 20 bytes",
            Assert.Throws<InvalidOperationException>(() => sequence.To(budgeted)).Message);
        Assert.All(elements, e => Assert.DoesNotContain(e, budgeted.Tensors));
        Assert.Equal(new DeviceMemoryUse(16, 1, 32), budgeted.ReadDeviceMemoryUse());

        held.Delete();
        Assert.Same(sequence, sequence.To(budgeted));
        Assert.Same(sequence, sequence.To(budgeted));
        Assert.Equal(new DeviceMemoryUse(20, 2, 32), budgeted.ReadDeviceMemoryUse());
    }

    [Fact]
    public void TestAnOutputWrittenIntoConsumedMemoryIsCountedInTheArenaThatMemoryWasInAndNoOther()
    {
        var (aliasedLimits, aliased) = KeepingTwo(aliases: true);
        Assert.Equal([6300L, 6200L], aliasedLimits);
        Assert.Equal(2L, aliased);
        var (plainLimits, plain) = KeepingTwo(aliases: false);
        Assert.Equal([6300L], plainLimits);
        Assert.Equal(0L, plain);

        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true, Aliases = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Doubled(), inputDims: null, trainingStep: false, aliasCandidates: [(0, 0)]);
        TensorData Kept(IData a) => compiled.Execute([a], [true])[0].ToTensorData();
        var onCard = Floats(20).To(context);
        var written = Kept(Kept(onCard.Shared()));
        Kept(Floats(1).Shared());
        Assert.Single(card.Sessions);
        Assert.Equal(1L, context.AliasedOutputs);
        GC.KeepAlive((object[])[onCard, written]);
    }

    /// <summary>The arena limits a budgeted context's session went through, and how many outputs
    /// were written into consumed memory, over two runs each consuming a host tensor copied onto
    /// the card and leaving its output there.</summary>
    private static (long?[] Limits, long Aliased) KeepingTwo(bool aliases)
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true, Aliases = aliases };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Doubled(), inputDims: null, trainingStep: false, aliasCandidates: [(0, 0)]);
        TensorData Kept(IData a) => compiled.Execute([a], [true])[0].ToTensorData();
        var (first, second) = (Kept(Floats(20)), Kept(Floats(20)));
        GC.KeepAlive((object[])[first, second]);
        return ([.. card.Sessions.Select(s => s.LimitBytes)], context.AliasedOutputs);
    }

    [Fact]
    public void TestASessionsArenaLimitIsTheBudgetLessTheDiscountRoundedUpToTheNextSixtyFourthOfIt()
    {
        Assert.Equal(6300L, ComputeContext.ArenaLimitWithin(6400, 0));
        Assert.Equal(6300L, ComputeContext.ArenaLimitWithin(6400, 99));
        Assert.Equal(6200L, ComputeContext.ArenaLimitWithin(6400, 100));
        Assert.Equal(5300L, ComputeContext.ArenaLimitWithin(6400, 1040));
        Assert.Equal(100L, ComputeContext.ArenaLimitWithin(6400, 6250));
        Assert.Equal(100L, ComputeContext.ArenaLimitWithin(6400, 6300));
        Assert.Equal(50L, ComputeContext.ArenaLimitWithin(6400, 6304));
        Assert.Equal(25L, ComputeContext.ArenaLimitWithin(6400, 6351));
        Assert.Equal(1L, ComputeContext.ArenaLimitWithin(6400, 6399));
        Assert.Null(ComputeContext.ArenaLimitWithin(6400, 6400));
        Assert.Null(ComputeContext.ArenaLimitWithin(6400, 7000));
        Assert.Equal(1L, ComputeContext.ArenaLimitWithin(10, 9));
        Assert.Equal(60L << 30, ComputeContext.ArenaLimitWithin(64L << 30, 3L << 30));
        Assert.Equal(163L, ComputeContext.ArenaLimitWithin(6463, 6250));
        Assert.Equal(100L, ComputeContext.ArenaLimitWithin(6401, 6300));
        Assert.Equal(100L, ComputeContext.ArenaLimitWithin(6463, 6300));
    }

    [Fact]
    public void TestARunReadingAnotherRuntimesTensorOnItsCardCountsBothThatTensorAndItsCopy()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        var otherRuntime = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        using var elsewhere = new ComputeContext(otherRuntime);
        var compiled = context.Compile(Echo());
        var source = Floats(1000).CopyTo(elsewhere);

        Assert.Throws<InvalidOperationException>(() => compiled.Execute(source.Shared()));
        Assert.Equal(new DeviceMemoryUse(0, 0, 6400), context.ReadDeviceMemoryUse());
        Assert.False(source.IsDisposed);
    }

    [Fact]
    public void TestADiscountClimbingThroughTheLastPartOfTheBudgetRebuildsTheSessionOnlyAFewTimes()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Echo());
        List<TensorData> held = [Floats(1575).CopyTo(context)];
        for (int step = 0; step < 24; step++)
        {
            Run(compiled, held[0].Shared());
            held.Add(Floats(1).CopyTo(context));
        }

        Assert.InRange(card.Sessions.Count, 2, 9);
    }

    [Fact]
    public void TestACompositePlacedOntoABudgetedCardIsRefusedWholeBeforeAnyOfItIsPlaced()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(card) { DeviceMemory = Budget(64) };
        using var free = new ComputeContext(card);
        TensorStructFieldDef[] fields =
        [
            new TensorStructFieldDef("a", DataStructure.Tensor, 1, DType.Float32),
            new TensorStructFieldDef("b", DataStructure.Tensor, 1, DType.Float32),
        ];
        var pair = new TensorDataStruct(new TensorStructDef(fields, "Pair"),
            new Dictionary<string, IData> { { "a", Floats(4).CopyTo(free) }, { "b", Floats(16) } });
        var triple = TensorDataSequence.OfElements([Floats(6), Floats(6), Floats(6)], DType.Float32);
        var built = card.Built.Count;

        Assert.Throws<InvalidOperationException>(() => pair.To(budgeted));
        Assert.Contains("sequence of 3 tensors", Assert.Throws<InvalidOperationException>(() => triple.CopyTo(budgeted)).Message);
        Assert.Equal(new DeviceMemoryUse(0, 0, 64), budgeted.ReadDeviceMemoryUse());
        Assert.Equal(built, card.Built.Count);

        var twice = Floats(10).CopyTo(free);
        var same = new TensorDataStruct(new TensorStructDef(fields, "Pair"),
            new Dictionary<string, IData> { { "a", twice }, { "b", twice } });
        Assert.Same(same, same.To(budgeted));
        Assert.Equal(new DeviceMemoryUse(40, 1, 64), budgeted.ReadDeviceMemoryUse());
    }

    [Fact]
    public void TestAStringInItsRuntimesHostMemoryReachesACardContextAsItStandsByToAsByAFeed()
    {
        using var cpu = new ComputeContext();
        var card = new StubBackend(ComputeDevice.Cuda, 0) { Runtime = cpu.ResolvedBackend.RuntimeIdentity };
        using var onCard = new ComputeContext(card);
        var s = InputVector<utf8>("s");
        var produced = cpu.Execute(
            new InternalComputationGraph([s], [OnnxOp.Identity(s, rank: 1)]), TensorData([2L], "a", "b"))[0].ToTensorData();

        Assert.True(produced.FeedsInPlace(card));
        Assert.Same(produced, produced.To(onCard));
    }

    [Fact]
    public void TestAFeedTheRunCanNeitherReadWhereItIsNorCopyIsRefusedBeforeTheRunTakesAnything()
    {
        var cpu = new StubBackend(ComputeDevice.Cpu, null);
        using var context = new ComputeContext(cpu);
        var compiled = context.Compile(Sum());
        var bystander = Floats(2);
        var unreadable = TensorData.Create(new Shape(2L), DType.Float32,
            new StubValue(ShorokooTensorElementType.Float, new byte[8], [2L], hostAccessible: false));

        Assert.Throws<InvalidOperationException>(() => compiled.Execute(bystander, unreadable.Shared()));
        Assert.False(bystander.IsDisposed);
        Assert.False(unreadable.IsDisposed);
    }

    [Fact]
    public void TestAConsumedHostFeedNoOutputIsWrittenIntoIsCopiedIntoTheArenaSoALoopKeepsItsSession()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Doubled());
        TensorData Kept(IData a) => compiled.Execute([a], [true])[0].ToTensorData();

        var state = Kept(Floats(200));
        state = Kept(state);
        state = Kept(state);

        Assert.Equal([6300L], card.Sessions.Select(s => s.LimitBytes));
        Assert.Empty(card.Built);
        GC.KeepAlive(state);
    }

    [Fact]
    public void TestAConsumedHostValueOfTheCardsOwnRuntimeIsHandedToItAsItIs()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Doubled());
        var output = compiled.Execute(Floats(200))[0].ToTensorData();
        var value = output.ToTensorValue(card);

        compiled.Execute(output);

        Assert.Same(value, card.Handed[^1]);
        Assert.Contains(value, card.Released);
        Assert.True(output.IsDisposed);
        Assert.Empty(card.Built);
    }

    [Fact]
    public void TestAConsumedHostFeedWhosePairedOutputComesHomeGoesToTheSessionFromTheHost()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true, Aliases = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Doubled(), inputDims: null, trainingStep: false, aliasCandidates: [(0, 0)]);

        var state = compiled.Execute(Floats(200))[0].ToTensorData();
        state = compiled.Execute(state)[0].ToTensorData();

        Assert.Equal([6300L], card.Sessions.Select(s => s.LimitBytes));
        Assert.Empty(card.Built);
        Assert.Equal(2L, context.AliasedOutputs);
        GC.KeepAlive(state);
    }

    [Fact]
    public void TestATriedHostFeedNothingElseReadsIsCopiedIntoTheArenaAsAConsumedOneIs()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Doubled());
        TensorData Kept(IData a) => compiled.Execute([a], [true])[0].ToTensorData();

        var state = Kept(Floats(200).TryConsume());
        state = Kept(state.TryConsume());

        Assert.Equal([6300L], card.Sessions.Select(s => s.LimitBytes));
        Assert.Empty(card.Built);
        GC.KeepAlive(state);
    }

    [Fact]
    public void TestAConsumedFeedWhoseHeldCopyAnotherRunIsReadingGoesToTheSessionFromTheHost()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        using var reader = new ComputeContext(card);
        var compiled = context.Compile(Echo());
        var source = Floats(250);
        Run(compiled, source.Shared());
        var copy = Assert.Single(context.Tensors, t => t.Space == MemorySpace.Cuda(0));

        using (reader.Lock(copy))
            Assert.Equal(new float[250], Run(compiled, source));

        Assert.True(source.IsDisposed);
    }

    [Fact]
    public void TestASessionTooSmallForWhatARunHasItsRuntimeCopyInIsBuiltAgainWhereTheBudgetNowHasRoom()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var held = Floats(1000).CopyTo(context);
        var compiled = context.Compile(Echo());
        held.Delete();

        Assert.Equal(new float[1000], Run(compiled, Floats(1000)));
        Assert.Equal([2300L, 6300L], card.Sessions.Select(s => s.LimitBytes));
    }

    [Fact]
    public void TestAHostTensorFedToTwoInputsIsCountedIntoTheArenaOncePerInput()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        var compiled = context.Compile(Sum());
        var twice = Floats(1000);

        Assert.Contains("copy 8000 bytes", Assert.Throws<InvalidOperationException>(() => compiled.Execute(twice, twice)).Message);
        Assert.False(twice.IsDisposed);
    }

    [Fact]
    public void TestARunThatFailsIsHeardOverACopyOfWhatItConsumedThatFailsToBeReleased()
    {
        var host = new StubBackend(ComputeDevice.Cpu, null) { FailingReleases = 1 };
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var reading = new ComputeContext(host);
        using var running = new ComputeContext(card);
        var onCard = OnCard(running, 1f);
        Run(reading.Compile(Echo()), onCard.Shared());
        card.FailsRuns = true;

        Assert.Equal("The stub run failed.",
            Assert.Throws<InvalidOperationException>(() => running.Compile(Echo()).Execute(onCard)).Message);
    }

    [Fact]
    public void TestAListSequenceARunReadsPutsNoneOfItsElementsOnTheRunningContextsBooks()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(card) { DeviceMemory = Budget(6400) };
        using var staging = new ComputeContext(card);
        var sequence = TensorDataSequence.OfElements(
            [Floats(500).CopyTo(staging), Floats(500).CopyTo(staging)], DType.Float32);
        var compiled = budgeted.Compile(TensorBesideSequence());
        long during = -1;
        card.DuringRun = () => during = budgeted.ReadDeviceMemoryUse().AttachedBytes;

        compiled.Execute(Floats(1), sequence.Shared());

        Assert.Equal(0L, during);
        Assert.Equal(0L, budgeted.ReadDeviceMemoryUse().AttachedBytes);
    }

    [Fact]
    public void TestASequenceBeingReadOutOfItsValueCannotBeConsumedOrDisposedUntilTheReadIsDone()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card);
        var compiled = context.Compile(TensorBesideSequence());
        var sequence = OnnxUtils.CreateTensorDataSequenceFromValue(
            DType.Float32, card.CreateSequence([HostFloats(2), HostFloats(2)]), card);
        using var reading = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        card.DuringElementRead = () => { reading.Set(); release.Wait(TimeSpan.FromSeconds(10)); };
        var read = Task.Run(() => sequence[0]);
        Assert.True(reading.Wait(TimeSpan.FromSeconds(10)));

        Assert.Throws<InvalidOperationException>(() => compiled.Execute(Floats(1), sequence));
        Assert.Throws<InvalidOperationException>(sequence.Dispose);

        release.Set();
        Assert.True(read.Wait(TimeSpan.FromSeconds(10)));
        card.DuringElementRead = null;
        Assert.False(sequence.IsDisposed);
    }

    [Fact]
    public void TestAHostValueBeingCopiedOutCannotBeConsumedOrDeletedUntilTheCopyIsDone()
    {
        var host = new StubBackend(ComputeDevice.Cpu, null);
        using var context = new ComputeContext(host);
        var compiled = context.Compile(Echo());
        foreach (var copyOut in (Func<TensorData, object>[])[
            t => t.CopyRawMemory(), t => t.As<float32>().CopyMemory<float>(), t => t.As<float32>().ValueAt<float>(0),
            t => TensorDataSequence.Create([t], null)])
        {
            using var reading = new ManualResetEventSlim();
            using var release = new ManualResetEventSlim();
            var tensor = TensorData.Create(new Shape(2L), DType.Float32, new StubValue(
                ShorokooTensorElementType.Float, new byte[8], [2L], hostAccessible: true)
            {
                DuringRead = () => { reading.Set(); release.Wait(TimeSpan.FromSeconds(10)); },
            }, host);
            var copying = Task.Run(() => copyOut(tensor));
            Assert.True(reading.Wait(TimeSpan.FromSeconds(10)));

            Assert.Throws<InvalidOperationException>(() => compiled.Execute(tensor));
            Assert.Throws<InvalidOperationException>(tensor.Delete);

            release.Set();
            Assert.True(copying.Wait(TimeSpan.FromSeconds(10)));
        }
    }

    [Fact]
    public void TestAReleaseThatThrowsLetsEveryOtherCopyOfATensorGoToo()
    {
        var first = new StubBackend(ComputeDevice.Cuda, 0) { FailingReleases = 1 };
        var second = new StubBackend(ComputeDevice.Cuda, 1);
        using var one = new ComputeContext(first);
        using var two = new ComputeContext(second);
        var source = Floats(2);
        Run(one.Compile(Echo()), source.Shared());
        Run(two.Compile(Echo()), source.Shared());

        Assert.Throws<InvalidOperationException>(source.ReleaseRunCopies);
        Assert.Equal([first.Built[0]], first.Released);
        Assert.Equal([second.Built[0]], second.Released);
    }

    private static StubValue HostFloats(int count)
        => new(ShorokooTensorElementType.Float, new byte[count * 4], [count], hostAccessible: true);

    private static InternalComputationGraph TensorBesideSequence()
    {
        var x = InputVector<float32>("x");
        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        return new InternalComputationGraph(
            [x, seq], [OnnxOp.Identity(x, rank: 1), OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)]);
    }

    private static DeviceMemorySettings Budget(long bytes) => new() { LimitBytes = bytes };

    /// <summary>A shared feed for input "a" that calls <paramref name="held"/> once the run holds
    /// it, before anything is built for it — where a write or a second run can be landed.</summary>
    private static NamedModelParam Hooked(TensorData data, Action held)
        => new HookedFeed(data, held) { FeedMode = SharedInputMode.Shared };

    private sealed class HookedFeed(TensorData data, Action held)
        : TensorDataModelParam("a", ModelParamType.InputParam, data)
    {
        internal override void Held() => held();
    }

    private static TensorData Floats(int count) => TensorData([(long)count], new float[count]);

    private static float[] Run(CompiledGraph compiled, IData feed)
        => [.. compiled.Execute(feed)[0].ToTensorData().As<float32>().AccessMemory<float>()];

    private static void InterlockedMax(ref int target, int value)
    {
        int seen;
        while (value > (seen = Volatile.Read(ref target)))
            if (Interlocked.CompareExchange(ref target, value, seen) == seen) return;
    }

    [Fact]
    public void TestAConsumedFeedIsReleasedOnceByTheBackendItWasHandedToWhetherTheRunReturnsOrFails()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card);
        var compiled = context.Compile(Sum());
        var (taken, read, failed) = (OnCard(context, 1f), OnCard(context, 3f), OnCard(context, 5f));

        compiled.Execute(taken, read.Shared());
        card.FailsRuns = true;
        Assert.Throws<InvalidOperationException>(() => compiled.Execute(failed, read.Shared()));

        Assert.Equal([card.Built[0], card.Built[2]], card.Handed);
        Assert.Equal([card.Built[0], card.Built[2]], card.Released);
        Assert.True(taken.IsDisposed && failed.IsDisposed);
        Assert.False(read.IsDisposed);
        Assert.All(card.Built, v => Assert.Equal(0, ((StubValue)v).Disposals));
    }

    [Fact]
    public void TestASessionThatLeavesConsumedFeedsToTheDefaultDisposesEachOnceAndNothingElseReleasesThem()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card);
        var compiled = context.Compile(Sum());
        var (taken, read, failed) = (OnCard(context, 1f), OnCard(context, 3f), OnCard(context, 5f));

        compiled.Execute(taken, read.Shared());
        card.FailsRuns = true;
        Assert.Throws<InvalidOperationException>(() => compiled.Execute(failed, read.Shared()));

        Assert.Equal([1, 0, 1], card.Built.Select(v => ((StubValue)v).Disposals));
        Assert.Empty(card.Released);
        Assert.True(taken.IsDisposed && failed.IsDisposed);
    }

    [Fact]
    public void TestAHostTensorACardRunReadsIsCopiedOnceUntilWrittenAndConsumedThroughTheCopyItHolds()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        var budget = new DeviceMemorySettings { LimitBytes = 1L << 20 };
        using var context = new ComputeContext(card) { DeviceMemory = budget };
        var compiled = context.Compile(Echo());
        float[] Run(IData feed) => [.. compiled.Execute(feed)[0].ToTensorData().As<float32>().AccessMemory<float>()];
        var source = TensorData([2L], (float[])[1f, 2f]);

        Assert.Equal([1f, 2f], Run(source.Shared()));
        Assert.Equal([1f, 2f], Run(source.Shared()));
        var copy = Assert.Single(context.Tensors, t => t.Space == MemorySpace.Cuda(0));
        Assert.Single(card.Built);
        Assert.Empty(card.Handed);

        source.As<float32>().AccessModifiableMemory<float>()[0] = 7f;
        Assert.True(copy.IsDisposed);
        Assert.Equal([card.Built[0]], card.Released);
        Assert.Equal([7f, 2f], Run(source.Shared()));

        Assert.Equal([7f, 2f], Run(source));
        Assert.True(source.IsDisposed);
        Assert.Equal(2, card.Built.Count);
        Assert.Equal([card.Built[1]], card.Handed);

        var fresh = TensorData([2L], (float[])[4f, 6f]);
        Assert.Equal([4f, 6f], Run(fresh));
        Assert.True(fresh.IsDisposed);
        Assert.Equal(2, card.Built.Count);
        Assert.True(card.Handed[1].IsHostAccessible);
        Assert.Equal([card.Built[0], card.Built[1], card.Handed[1]], card.Released);
    }

    [Fact]
    public void TestATensorBeingCopiedOutOfItsMemoryCannotBeConsumedOrDeletedUntilTheCopyIsDone()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { ReleasesWhatItConsumes = true };
        using var context = new ComputeContext(card);
        var compiled = context.Compile(Echo());
        foreach (var read in (Func<TensorData, object>[])[
            t => t.ToHost(), t => t.CopyTo(ComputeContext.Host), t => t.MoveToAttribute()])
        {
            var onCard = OnCard(context, 1f);
            using var copying = new ManualResetEventSlim();
            using var release = new ManualResetEventSlim();
            card.DuringHostCopy = () => { copying.Set(); release.Wait(TimeSpan.FromSeconds(10)); };
            var reading = Task.Run(() => read(onCard));
            Assert.True(copying.Wait(TimeSpan.FromSeconds(10)));

            Assert.Throws<InvalidOperationException>(() => compiled.Execute(onCard));
            Assert.Throws<InvalidOperationException>(onCard.Delete);
            Assert.DoesNotContain(card.Built[^1], card.Released);

            release.Set();
            Assert.True(reading.Wait(TimeSpan.FromSeconds(10)));
            card.DuringHostCopy = null;
        }
    }

    [Fact]
    public void TestAReleaseThatThrowsStillLetsGoOfTheRestAndOfTheBudgetedContext()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0) { FailingReleases = 1 };
        using var context = new ComputeContext(card) { DeviceMemory = Budget(1L << 20) };
        var compiled = context.Compile(Sum());
        var (taken, read) = (OnCard(context, 1f), OnCard(context, 3f));
        using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        Assert.Contains("stopped by the test", Assert.Throws<InvalidOperationException>(() => compiled.Run(
            new HookedFeed(taken, () => throw new InvalidOperationException("stopped by the test")),
            new TensorDataModelParam("b", ModelParamType.InputParam, read) { FeedMode = SharedInputMode.Shared })).Message);

        Assert.Equal([card.Built[0]], card.Released);
        Assert.True(read.TryDelete());
        Assert.Equal([1f, 2f], compiled.Execute(
            [OnCard(context, 1f), OnCard(context, 3f)], new RunSettings { CancellationToken = patience.Token })[0]
            .ToTensorData().As<float32>().CopyMemory<float>());
    }

    [Fact]
    public void TestATensorConsumedWhileItWaitedToBePlacedOntoABudgetedContextIsRefusedRatherThanHandedBackDead()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var budgeted = new ComputeContext(card) { DeviceMemory = Budget(1L << 20) };
        using var free = new ComputeContext(card);
        var placed = OnCard(free, 1f);
        using var inside = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        card.DuringRun = () => { inside.Set(); release.Wait(TimeSpan.FromSeconds(10)); };
        var holding = Task.Run(() => budgeted.Compile(Echo()).Execute(Floats(1)));
        Assert.True(inside.Wait(TimeSpan.FromSeconds(10)));
        card.DuringRun = null;

        var placing = Task.Run(() => placed.To(budgeted));
        Assert.False(placing.Wait(TimeSpan.FromMilliseconds(200)));
        free.Compile(Echo()).Execute(placed);
        release.Set();

        Assert.True(holding.Wait(TimeSpan.FromSeconds(10)));
        Assert.Throws<ObjectDisposedException>(() => placing.GetAwaiter().GetResult());
        Assert.DoesNotContain(placed, budgeted.Tensors);
    }

    [Fact]
    public void TestAGraphCannotBeDisposedWhileOneOfItsRunsIsInFlight()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        using var context = new ComputeContext(card);
        var compiled = context.Compile(Echo());
        using var inside = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        card.DuringRun = () => { inside.Set(); release.Wait(TimeSpan.FromSeconds(10)); };
        var running = Task.Run(() => Run(compiled, Floats(1)));
        Assert.True(inside.Wait(TimeSpan.FromSeconds(10)));

        Assert.Contains("in flight", Assert.Throws<InvalidOperationException>(compiled.Dispose).Message);
        Assert.False(compiled.IsDisposed);
        Assert.Equal(0, card.SessionDisposals);

        release.Set();
        Assert.True(running.Wait(TimeSpan.FromSeconds(10)));
        compiled.Dispose();
        Assert.True(compiled.IsDisposed);
        Assert.Equal(1, card.SessionDisposals);
    }

    [Fact]
    public void TestASequenceIsNotDisposedWhileACopyOfItIsBeingBuiltAndTheCopyGoesWithItAfter()
    {
        var cpu = new StubBackend(ComputeDevice.Cpu, null);
        var sequence = TensorDataSequence.OfElements([Floats(2), Floats(3)], DType.Float32);
        Exception? duringBuild = null;
        cpu.DuringSequenceBuild = () => duringBuild = Record.Exception(sequence.Dispose);

        sequence.ToTensorValue(cpu);
        Assert.IsType<InvalidOperationException>(duringBuild);
        sequence.Dispose();

        Assert.True(sequence.IsDisposed);
        Assert.Single(cpu.Sequences);
        Assert.All(cpu.Sequences, built => Assert.Contains(built, cpu.Released));
    }

    private static TensorData OnCard(ComputeContext context, float first)
        => TensorData([2L], (float[])[first, first + 1f]).To(context);

    private static InternalComputationGraph Sum()
    {
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        return new InternalComputationGraph([a, b], [a + b]);
    }

    private static InternalComputationGraph Echo()
    {
        var a = InputVector<float32>("a");
        return new InternalComputationGraph([a], [OnnxOp.Identity(a, rank: 1)]);
    }

    private static InternalComputationGraph Doubled()
    {
        var a = InputVector<float32>("a");
        return new InternalComputationGraph([a], [a * 2f]);
    }

    /// <summary>A backend that answers about itself and records what it was asked to build, what it
    /// was asked whether it could address, and what it released — so a transfer's route can be read
    /// off it without a native runtime or a card. Its sessions run nothing (see
    /// <see cref="StubSession"/>), which is all a test of who owns a feed needs of them, and it
    /// records what each was built with and what each run was given.</summary>
    private sealed class StubBackend(ComputeDevice device, int? cudaDeviceId)
        : IShorokooBackend
    {
        public int BackendMemoryBuilds => Built.Count;

        internal List<IShorokooTensorValue> Built { get; } = [];

        internal List<IShorokooTensorValue> Released { get; } = [];

        /// <summary>The device-memory settings each of its sessions was built with.</summary>
        internal List<DeviceMemorySettings> Sessions { get; } = [];

        /// <summary>What each run of its sessions was given, in order.</summary>
        internal List<RunSettings> Runs { get; } = [];

        /// <summary>Called inside every run of its sessions, where a test holds one open.</summary>
        internal Action? DuringRun { get; set; }

        internal List<MemoryLocation> AskedAbout { get; } = [];

        /// <summary>What its sessions were handed to consume, over every run.</summary>
        internal List<IShorokooTensorValue> Handed { get; } = [];

        /// <summary>Whether its sessions release what they consume through this backend, as a
        /// native one does, rather than leaving it to the interface's default.</summary>
        internal bool ReleasesWhatItConsumes { get; init; }

        /// <summary>Whether its releasing sessions write the outputs they were built to alias into
        /// the memory of the values they consumed, as a native one does.</summary>
        internal bool Aliases { get; init; }

        /// <summary>Whether its sessions' runs throw.</summary>
        internal bool FailsRuns { get; set; }

        /// <summary>How many of its next releases throw after recording what they were given.</summary>
        internal int FailingReleases { get; set; }

        /// <summary>Called inside every copy of one of its values back to the host, where a test
        /// holds one open.</summary>
        internal Action? DuringHostCopy { get; set; }

        /// <summary>How many of its sessions have been disposed.</summary>
        internal int SessionDisposals { get; set; }

        /// <summary>What <see cref="CanAddress"/> answers, where a test decides; the interface's own
        /// answer otherwise.</summary>
        internal Func<MemoryLocation, bool>? Addresses { get; init; }

        /// <summary>The runtime it says its allocations belong to, where a test shares one with
        /// another backend; itself otherwise.</summary>
        internal object? Runtime { get; init; }

        public object RuntimeIdentity => Runtime ?? this;

        public BackendDescription Description { get; } = new($"stub-{device}", device, cudaDeviceId);

        public bool CanAddress(MemoryLocation location)
        {
            AskedAbout.Add(location);
            return Addresses is { } answer
                ? answer(location)
                : location.Space == ((IShorokooBackend)this).MemorySpace
                  && (location.Space.IsKnown
                      ? location.IsManaged || ReferenceEquals(location.Runtime, RuntimeIdentity)
                      : ReferenceEquals(location.Runtime, this));
        }

        public void Release(IShorokooTensorValue value)
        {
            Released.Add(value);
            if (FailingReleases <= 0) return;
            FailingReleases--;
            throw new InvalidOperationException("The stub release failed.");
        }

        public IShorokooTensorValue CreateTensorInBackendMemory(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
        {
            var value = new StubValue(elementType, data, shape, hostAccessible: device == ComputeDevice.Cpu);
            lock (Built) Built.Add(value);
            return value;
        }

        public int HostCopies { get; private set; }

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => new StubValue(elementType, data, shape, hostAccessible: true);

        /// <summary>The sequence values it built, in order.</summary>
        internal List<IShorokooTensorValue> Sequences { get; } = [];

        /// <summary>Called inside every sequence value it builds, where a test lands something
        /// mid-build.</summary>
        internal Action? DuringSequenceBuild { get; set; }

        /// <summary>Called inside every read of an element out of a sequence value it built, where a
        /// test holds one open.</summary>
        internal Action? DuringElementRead { get; set; }

        public byte[] CopyTensorToHost(IShorokooTensorValue value)
        {
            HostCopies++;
            DuringHostCopy?.Invoke();
            return ((StubValue)value).Bytes;
        }

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory)
            => Build(modelBytes, deviceMemory, []);

        IShorokooSession IShorokooBackend.CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity, DeviceMemorySettings deviceMemory,
            DiagnosticSettings diagnostics, IReadOnlyList<OutputAlias> outputAliases)
            => Build(modelBytes, deviceMemory, Aliases ? outputAliases : []);

        private IShorokooSession Build(
            ReadOnlyMemory<byte> modelBytes, DeviceMemorySettings deviceMemory, IReadOnlyList<OutputAlias> aliases)
        {
            Sessions.Add(deviceMemory);
            var graph = ProtoBuf.Serializer
                .Deserialize<Shorokoo.Core.Factory.IR.ModelProto>(new MemoryStream(modelBytes.ToArray())).Graph;
            string[] inputs = [.. graph.Inputs.Select(i => i.Name)];
            string[] outputs = [.. graph.Outputs.Select(o => o.Name)];
            return ReleasesWhatItConsumes
                ? new ReleasingStubSession(this, inputs, outputs, aliases)
                : new StubSession(this, inputs, outputs);
        }

        /// <summary>Whether this backend's memory is its own rather than the host's, which is where
        /// its sessions leave what they are asked to retain.</summary>
        internal bool OnADevice => device != ComputeDevice.Cpu;

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
        {
            DuringSequenceBuild?.Invoke();
            var sequence = new StubSequenceValue(values, this);
            Sequences.Add(sequence);
            return sequence;
        }
    }

    /// <summary>A sequence value over the values it was built from, which it hands out copies
    /// of.</summary>
    private sealed class StubSequenceValue(IReadOnlyList<IShorokooTensorValue> values, StubBackend backend)
        : IShorokooTensorValue
    {
        public bool IsHostAccessible => true;
        public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Sequence;
        public ShorokooTensorElementType ElementType => ShorokooTensorElementType.Float;
        public long[] Shape => [];
        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => values.Count;

        public IShorokooTensorValue GetValue(int index)
        {
            backend.DuringElementRead?.Invoke();
            var value = (StubValue)values[index];
            return new StubValue(value.ElementType, [.. value.Bytes], value.Shape, hostAccessible: true);
        }

        public ShorokooTensorElementType GetSequenceElementType() => ShorokooTensorElementType.Float;
        public void Dispose() { }
    }

    /// <summary>A session that runs nothing: every output is a copy of its first input, in host
    /// memory unless the run asked for it to be retained, when it is left in the backend's own. It
    /// leaves what it consumes to the interface's default, which disposes each value.</summary>
    private class StubSession(StubBackend backend, string[] inputNames, string[] outputNames)
        : IShorokooSession
    {
        private protected StubBackend Backend => backend;

        public IReadOnlyList<string> InputNames => inputNames;

        public IReadOnlyList<string> OutputNames => outputNames;

        public bool HasDeviceMemory => backend.OnADevice;

        public IReadOnlyList<IShorokooTensorValue> Run(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames, RunSettings runSettings)
            => RunRetainingOutputs(inputs, outputNames, new HashSet<string>(), runSettings);

        public IReadOnlyList<IShorokooTensorValue> RunRetainingOutputs(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames, IReadOnlySet<string> retainedOutputNames,
            RunSettings runSettings)
        {
            lock (backend.Runs) backend.Runs.Add(runSettings);
            backend.DuringRun?.Invoke();
            if (backend.FailsRuns) throw new InvalidOperationException("The stub run failed.");
            var first = (StubValue)inputs[inputNames[0]];
            return [.. outputNames.Select(name => (IShorokooTensorValue)new StubValue(
                first.ElementType, [.. first.Bytes], first.Shape,
                hostAccessible: !(backend.OnADevice && retainedOutputNames.Contains(name))))];
        }

        public void Dispose() => backend.SessionDisposals++;
    }

    /// <summary>A session that releases what it consumes through its backend once the run is over,
    /// however it ends — as a native one does — and writes each output it was built to alias into
    /// the memory of the value its input was fed, where the run consumed that value alone and it is
    /// in the memory the output is produced in.</summary>
    private sealed class ReleasingStubSession(
        StubBackend backend, string[] inputNames, string[] outputNames, IReadOnlyList<OutputAlias> aliases)
        : StubSession(backend, inputNames, outputNames), IShorokooSession
    {
        IReadOnlyList<OutputAlias> IShorokooSession.BindableAliases => aliases;

        IReadOnlyList<IShorokooTensorValue> IShorokooSession.RunConsuming(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyCollection<IShorokooTensorValue> consumed, IReadOnlyList<string> outputNames,
            IReadOnlySet<string> retainedOutputNames, RunSettings runSettings)
            => ((IShorokooSession)this).RunConsuming(
                inputs, consumed, outputNames, retainedOutputNames, runSettings, out _);

        IReadOnlyList<IShorokooTensorValue> IShorokooSession.RunConsuming(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyCollection<IShorokooTensorValue> consumed, IReadOnlyList<string> outputNames,
            IReadOnlySet<string> retainedOutputNames, RunSettings runSettings,
            out IReadOnlyList<string?> aliasedInputs)
        {
            var written = new string?[outputNames.Count];
            aliasedInputs = written;
            Backend.Handed.AddRange(consumed);
            try
            {
                var results = RunRetainingOutputs(inputs, outputNames, retainedOutputNames, runSettings).ToArray();
                foreach (var alias in aliases)
                {
                    var output = outputNames.ToList().IndexOf(alias.Output);
                    if (output < 0 || inputs[alias.Input] is not StubValue value) continue;
                    if (!consumed.Contains(value) || inputs.Values.Count(v => ReferenceEquals(v, value)) != 1) continue;
                    if (value.IsHostAccessible != results[output].IsHostAccessible) continue;
                    results[output] = new StubValue(value.ElementType, value.Bytes, value.Shape, value.IsHostAccessible);
                    written[output] = alias.Input;
                }
                return results;
            }
            finally
            {
                foreach (var value in consumed) Backend.Release(value);
            }
        }
    }

    /// <summary>A tensor value belonging to no runtime, which says whether the host may read it and
    /// counts its disposals.</summary>
    private sealed class StubValue(
        ShorokooTensorElementType elementType, byte[] data, long[] shape, bool hostAccessible)
        : IShorokooTensorValue
    {
        internal byte[] Bytes => data;

        internal int Disposals { get; private set; }

        /// <summary>Called inside every read of its bytes, where a test holds one open.</summary>
        internal Action? DuringRead { get; init; }

        public bool IsHostAccessible => hostAccessible;

        public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Tensor;
        public ShorokooTensorElementType ElementType => elementType;
        public long[] Shape => shape;

        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
        {
            DuringRead?.Invoke();
            return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(data);
        }

        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
            => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(data.AsSpan());

        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => throw new NotSupportedException();
        public IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public ShorokooTensorElementType GetSequenceElementType() => throw new NotSupportedException();
        public void Dispose() => Disposals++;
    }
}
