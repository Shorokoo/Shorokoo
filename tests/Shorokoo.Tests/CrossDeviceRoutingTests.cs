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
/// <para>The same stubs cover which context's <see cref="DeviceMemorySettings"/> a placed tensor is
/// allocated under, and which backend a tensor's memory is released through, for the same reason:
/// a stub is what makes those hand-offs observable.</para>
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
    }

    [Fact]
    public void TestATensorOnOneCardReachesAnotherOnlyByGoingThroughTheHost()
    {
        // The case the class is named for: the source is already on a card. Starting on the host
        // makes the copy the route for the trivial reason that the source is host-resident, so a
        // regression that handed a tensor between two CUDA device ids over -- or that dropped
        // DeviceId from the space comparison -- passed.
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
    public void TestAnUnknownMemorySpaceIsNeverSharedAndComesHomeThroughTheBackendThatMadeIt()
    {
        var other = new StubBackend(ComputeDevice.Other, null);
        Assert.Equal(MemorySpace.UnknownDevice, ((IShorokooBackend)other).MemorySpace);
        using var context = new ComputeContext(other);

        var onUnknown = TensorData([2L], (float[])[3f, 4f]).CopyTo(context);
        Assert.False(onUnknown.Space.IsKnown);

        Assert.NotSame(onUnknown, onUnknown.To(context));
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
    public void TestATensorPlacedOnACardIsAllocatedUnderTheTargetContextsBudgetAndStaysThere()
    {
        var card = new StubBackend(ComputeDevice.Cuda, 0);
        var tight = new DeviceMemorySettings { LimitBytes = 1L << 20 };
        var wide = new DeviceMemorySettings { LimitBytes = 1L << 30 };
        using var small = new ComputeContext(card) { DeviceMemory = tight };
        using var large = new ComputeContext(card) { DeviceMemory = wide };
        using var unbudgeted = new ComputeContext(card);

        var onCard = TensorData([2L], (float[])[1f, 2f]).CopyTo(small);
        TensorData([2L], (float[])[3f, 4f]).To(large);
        TensorData([2L], (float[])[5f, 6f]).CopyTo(unbudgeted);
        small.AllocateUninitialized<float32>(new Shape(2L));

        Assert.Equal([tight, wide, DeviceMemorySettings.Default, tight], card.Budgets);
        Assert.Equal(tight.LimitBytes, small.ReadTransferArenaStatistics()!.Value.LimitBytes);
        Assert.Equal(wide.LimitBytes, large.ReadTransferArenaStatistics()!.Value.LimitBytes);
        Assert.Equal(-1, unbudgeted.ReadTransferArenaStatistics()!.Value.LimitBytes);
        Assert.Null(ComputeContext.Host.ReadTransferArenaStatistics());

        // Handed over rather than copied, so there is nothing to re-charge: the memory stays under
        // the budget it was allocated under however many contexts it is attached to afterwards.
        Assert.Same(onCard, onCard.To(large));
        Assert.Same(onCard, onCard.To(unbudgeted));
        Assert.Equal(4, card.Budgets.Count);
        Assert.Contains(onCard, large.Tensors);
        Assert.Contains(onCard, unbudgeted.Tensors);
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
        Assert.Equal([card.Built[1], card.Built[2]], card.Handed);
        Assert.Equal([card.Built[0], card.Built[1], card.Built[2]], card.Released);
        Assert.Equal([budget, budget, budget], card.Budgets);
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

    /// <summary>A backend that answers about itself and records what it was asked to build, what it
    /// was asked whether it could address, and what it released — so a transfer's route can be read
    /// off it without a native runtime or a card. Its sessions run nothing (see
    /// <see cref="StubSession"/>), which is all a test of who owns a feed needs of them.</summary>
    private sealed class StubBackend(ComputeDevice device, int? cudaDeviceId)
        : IShorokooBackend
    {
        public int BackendMemoryBuilds => Built.Count;

        internal List<IShorokooTensorValue> Built { get; } = [];

        internal List<IShorokooTensorValue> Released { get; } = [];

        internal List<DeviceMemorySettings> Budgets { get; } = [];

        internal List<MemoryLocation> AskedAbout { get; } = [];

        /// <summary>What its sessions were handed to consume, over every run.</summary>
        internal List<IShorokooTensorValue> Handed { get; } = [];

        /// <summary>Whether its sessions release what they consume through this backend, as a
        /// native one does, rather than leaving it to the interface's default.</summary>
        internal bool ReleasesWhatItConsumes { get; init; }

        /// <summary>Whether its sessions' runs throw.</summary>
        internal bool FailsRuns { get; set; }

        /// <summary>What <see cref="CanAddress"/> answers, where a test decides; the interface's own
        /// answer otherwise.</summary>
        internal Func<MemoryLocation, bool>? Addresses { get; init; }

        public BackendDescription Description { get; } = new($"stub-{device}", device, cudaDeviceId);

        public bool CanAddress(MemoryLocation location)
        {
            AskedAbout.Add(location);
            return Addresses is { } answer
                ? answer(location)
                : location.Space.IsKnown && location.Space == ((IShorokooBackend)this).MemorySpace
                  && (location.IsManaged || ReferenceEquals(location.Runtime, this));
        }

        public void Release(IShorokooTensorValue value) => Released.Add(value);

        public IShorokooTensorValue CreateTensorInBackendMemory(
            ShorokooTensorElementType elementType, byte[] data, long[] shape,
            DeviceMemorySettings deviceMemory)
        {
            Budgets.Add(deviceMemory);
            var value = new StubValue(elementType, data, shape, hostAccessible: device == ComputeDevice.Cpu);
            Built.Add(value);
            return value;
        }

        public IShorokooTensorValue CreateUninitializedTensorInBackendMemory(
            ShorokooTensorElementType elementType, long[] shape, DeviceMemorySettings deviceMemory)
            => CreateTensorInBackendMemory(
                elementType, new byte[TensorElementLayout.ByteCount(elementType, shape)], shape,
                deviceMemory);

        public ArenaStatistics? ReadTransferArenaStatistics(DeviceMemorySettings deviceMemory)
            => new ArenaStatistics(0, deviceMemory.LimitBytes ?? -1, 0, 0, 0, 0, 0, 0, 0);

        public int HostCopies { get; private set; }

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => throw new NotSupportedException();

        public byte[] CopyTensorToHost(IShorokooTensorValue value)
        {
            HostCopies++;
            return ((StubValue)value).Bytes;
        }

        public IShorokooSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory)
        {
            var graph = ProtoBuf.Serializer
                .Deserialize<Shorokoo.Core.Factory.IR.ModelProto>(new MemoryStream(modelBytes.ToArray())).Graph;
            string[] inputs = [.. graph.Inputs.Select(i => i.Name)];
            string[] outputs = [.. graph.Outputs.Select(o => o.Name)];
            return ReleasesWhatItConsumes
                ? new ReleasingStubSession(this, inputs, outputs)
                : new StubSession(this, inputs, outputs);
        }

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => throw new NotSupportedException();
    }

    /// <summary>A session that runs nothing: every output is a host copy of its first input. It
    /// leaves what it consumes to the interface's default, which disposes each value.</summary>
    private class StubSession(StubBackend backend, string[] inputNames, string[] outputNames)
        : IShorokooSession
    {
        private protected StubBackend Backend => backend;

        public IReadOnlyList<string> InputNames => inputNames;

        public IReadOnlyList<string> OutputNames => outputNames;

        public IReadOnlyList<IShorokooTensorValue> Run(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyList<string> outputNames, RunSettings runSettings)
        {
            if (backend.FailsRuns) throw new InvalidOperationException("The stub run failed.");
            var first = (StubValue)inputs[inputNames[0]];
            return [.. outputNames.Select(_ => (IShorokooTensorValue)new StubValue(
                first.ElementType, [.. first.Bytes], first.Shape, hostAccessible: true))];
        }

        public void Dispose() { }
    }

    /// <summary>A session that releases what it consumes through its backend once the run is over,
    /// however it ends — as a native one does.</summary>
    private sealed class ReleasingStubSession(StubBackend backend, string[] inputNames, string[] outputNames)
        : StubSession(backend, inputNames, outputNames), IShorokooSession
    {
        IReadOnlyList<IShorokooTensorValue> IShorokooSession.RunConsuming(
            IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
            IReadOnlyCollection<IShorokooTensorValue> consumed, IReadOnlyList<string> outputNames,
            IReadOnlySet<string> retainedOutputNames, RunSettings runSettings)
        {
            Backend.Handed.AddRange(consumed);
            try
            {
                return Run(inputs, outputNames, runSettings);
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

        public bool IsHostAccessible => hostAccessible;

        public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Tensor;
        public ShorokooTensorElementType ElementType => elementType;
        public long[] Shape => shape;

        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged
            => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(data);

        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged
            => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(data.AsSpan());

        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => throw new NotSupportedException();
        public IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public ShorokooTensorElementType GetSequenceElementType() => throw new NotSupportedException();
        public void Dispose() => Disposals++;
    }
}
