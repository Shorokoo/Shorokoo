using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Two different cards are two different memory spaces, so a tensor moving between them is copied
/// through the host rather than re-wrapped. There is no direct device-to-device path.
///
/// <para>Stubbed backends rather than real ones, deliberately: the decision under test is which
/// route a transfer takes, and that is made from the spaces the backends report. Proving it needs
/// two CUDA devices to <i>report</i>, not two to exist — and a machine with one card could not run
/// this at all otherwise. What a second card would add is confidence that the copy itself lands
/// correctly on a device this process has not been using, which no stub can stand in for.</para>
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

        IShorokooInferenceBackend first = new StubBackend(ComputeDevice.Cuda, 0);
        IShorokooInferenceBackend second = new StubBackend(ComputeDevice.Cuda, 0);
        Assert.Equal(first.MemorySpace, second.MemorySpace);
        Assert.Equal(MemorySpace.Cuda(0), first.MemorySpace);
    }

    [Fact]
    public void TestATensorOnOneCardReachesAnotherOnlyByGoingThroughTheHost()
    {
        // The case the class is named for, and the one every other test here skips: the source is
        // already on a card. Starting on the host makes CopyAcross the route for the trivial
        // reason that the source is host-resident, so a regression that re-wrapped between two
        // CUDA device ids -- or that dropped DeviceId from the space comparison -- passed.
        var firstCard = new StubBackend(ComputeDevice.Cuda, 0);
        var secondCard = new StubBackend(ComputeDevice.Cuda, 1);
        using var one = new ComputeContext(firstCard);
        using var two = new ComputeContext(secondCard);

        var onFirst = TensorData([2L], (float[])[3f, 4f]).TransferTo(one);
        Assert.Equal(MemorySpace.Cuda(0), onFirst.Space);
        Assert.False(onFirst.IsHostResident);

        var onSecond = onFirst.TransferTo(two);

        // Through the host: the owning backend was asked for the bytes, and the target was asked
        // to build from them. Neither happens on a re-wrap.
        Assert.Equal(1, firstCard.HostCopies);
        Assert.Equal(1, secondCard.BackendMemoryBuilds);
        Assert.Equal(MemorySpace.Cuda(1), onSecond.Space);
        Assert.True(onSecond.OwnsMemory);
        Assert.True(onFirst.IsDisposed);
    }

    [Fact]
    public void TestTwoContextsOnOneCardShareItsAllocationOnlyWhenTheyShareABackend()
    {
        // Same space is necessary and, off the host, not sufficient: an allocation means nothing
        // to a runtime that did not make it. Same backend shares; two backends reporting the same
        // card copy through the host.
        var backend = new StubBackend(ComputeDevice.Cuda, 0);
        using var one = new ComputeContext(backend);
        using var alsoOne = new ComputeContext(backend);
        using var otherRuntime = new ComputeContext(new StubBackend(ComputeDevice.Cuda, 0));

        var onCard = TensorData([2L], (float[])[5f, 6f]).TransferTo(one);
        var shared = onCard.GiveAccessTo(alsoOne);
        Assert.False(shared.OwnsMemory);
        Assert.Equal(0, backend.HostCopies);

        // The same request across runtimes cannot be served without allocating, which is what
        // GiveAccessTo promises not to do.
        Assert.Throws<InvalidOperationException>(() => onCard.GiveAccessTo(otherRuntime));
    }

    [Fact]
    public void TestAnUnknownMemorySpaceRefusesEveryTransfer()
    {
        // A backend on some other execution provider reports a space nothing can name. Two such
        // tensors compare equal as spaces without being in the same place, so every operation is
        // refused rather than guessed at.
        var other = new StubBackend(ComputeDevice.Other, null);
        Assert.Equal(MemoryKind.Unknown, ((IShorokooInferenceBackend)other).MemorySpace.Kind);

        using var context = new ComputeContext(other);
        var onUnknown = TensorData([2L], (float[])[1f, 2f]).TransferTo(context);
        Assert.Equal(MemoryKind.Unknown, onUnknown.Space.Kind);

        Assert.Throws<InvalidOperationException>(() => onUnknown.TransferTo(null));
        Assert.Throws<InvalidOperationException>(() => onUnknown.CopyTo(null));
        Assert.Throws<InvalidOperationException>(() => onUnknown.GiveAccessTo(context));
    }

    [Fact]
    public void TestATransferToAnotherCardGoesThroughHostBytes()
    {
        var target = new StubBackend(ComputeDevice.Cuda, 1);
        using var secondCard = new ComputeContext(target);

        float[] values = [1f, 2f, 3f, 4f];
        var onHost = TensorData([4L], values);

        var moved = onHost.TransferTo(secondCard);

        // The target was asked to build the tensor from raw bytes -- the host route -- rather than
        // being handed the allocation, which is the only thing that can cross a space boundary.
        Assert.Equal(1, target.BackendMemoryBuilds);
        Assert.Equal(MemorySpace.Cuda(1), moved.Space);
        Assert.True(moved.OwnsMemory);
        Assert.True(onHost.IsDisposed);
    }

    [Fact]
    public void TestACopyToAnotherCardLeavesTheSourceWhereItIs()
    {
        var target = new StubBackend(ComputeDevice.Cuda, 1);
        using var secondCard = new ComputeContext(target);
        var onHost = TensorData([2L], (float[])[7f, 8f]);

        var copy = onHost.CopyTo(secondCard);

        Assert.Equal(1, target.BackendMemoryBuilds);
        Assert.Equal(MemorySpace.Cuda(1), copy.Space);
        Assert.True(onHost.OwnsMemory);
        Assert.Equal([7f, 8f], onHost.As<float32>().AccessMemory<float>().ToArray());
    }

    [Fact]
    public void TestGiveAccessToAnotherCardIsRefusedBecauseReachingItMeansAllocating()
    {
        using var secondCard = new ComputeContext(new StubBackend(ComputeDevice.Cuda, 1));
        var onHost = TensorData([2L], (float[])[7f, 8f]);

        var ex = Assert.Throws<InvalidOperationException>(() => onHost.GiveAccessTo(secondCard));
        Assert.Contains("CopyTo", ex.Message);
    }

    [Fact]
    public void TestADetachingContextLeavesARetainedDeviceOutputOnTheCard()
    {
        using var card = new ComputeContext(new StubBackend(ComputeDevice.Cuda, 0), detachesOutputs: true);
        var onCard = TensorData([2L], (float[])[1f, 2f]).TransferTo(card);
        NamedModelParam[] outputs =
            [new TensorDataModelParam("state", ModelParamType.OutputParam, onCard)];

        var delivered = card.Deliver(outputs, new HashSet<string> { "state" });

        Assert.Same(card, delivered[0].ToTensorData().Context);
        Assert.Equal(MemorySpace.Cuda(0), delivered[0].ToTensorData().Space);
    }

    /// <summary>A backend that answers about itself and records what it was asked to build, so a
    /// transfer's route can be read off it without a session, a native runtime or a card.</summary>
    private sealed class StubBackend(ComputeDevice device, int? cudaDeviceId)
        : IShorokooInferenceBackend
    {
        public int BackendMemoryBuilds { get; private set; }

        public BackendDescription Description { get; } = new($"stub-{device}", device, cudaDeviceId);

        public IShorokooTensorValue CreateTensorInBackendMemory(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
        {
            BackendMemoryBuilds++;
            return new StubValue(elementType, data, shape, hostAccessible: device == ComputeDevice.Cpu);
        }

        public int HostCopies { get; private set; }

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => throw new NotSupportedException();

        public byte[] CopyTensorToHost(IShorokooTensorValue value)
        {
            HostCopies++;
            return ((StubValue)value).Bytes;
        }

        public IShorokooInferenceSession CreateSession(
            ReadOnlyMemory<byte> modelBytes, ShorokooGraphOptimization graphOptimization,
            ShorokooLogSeverity logSeverity,
            DeviceMemorySettings deviceMemory) => throw new NotSupportedException();

        public IShorokooTensorValue CreateTensor<T>(T[] data, long[] shape) where T : unmanaged
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateStringTensor(IReadOnlyList<string> data, long[] shape)
            => throw new NotSupportedException();

        public IShorokooTensorValue CreateSequence(IReadOnlyList<IShorokooTensorValue> values)
            => throw new NotSupportedException();
    }

    /// <summary>A tensor value belonging to no runtime, which says whether the host may read it —
    /// enough for a routing test, which never runs anything on it.</summary>
    private sealed class StubValue(
        ShorokooTensorElementType elementType, byte[] data, long[] shape, bool hostAccessible)
        : IShorokooTensorValue
    {
        internal byte[] Bytes => data;

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
        public void Dispose() { }
    }
}
