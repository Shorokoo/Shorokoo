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

        IShorokooInferenceBackend first = new StubFactory(ComputeDevice.Cuda, 0);
        IShorokooInferenceBackend second = new StubFactory(ComputeDevice.Cuda, 0);
        Assert.Equal(first.MemorySpace, second.MemorySpace);
        Assert.Equal(MemorySpace.Cuda(0), first.MemorySpace);
    }

    [Fact]
    public void TestATransferToAnotherCardGoesThroughHostBytes()
    {
        var target = new StubFactory(ComputeDevice.Cuda, 1);
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
        var target = new StubFactory(ComputeDevice.Cuda, 1);
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
        using var secondCard = new ComputeContext(new StubFactory(ComputeDevice.Cuda, 1));
        var onHost = TensorData([2L], (float[])[7f, 8f]);

        var ex = Assert.Throws<InvalidOperationException>(() => onHost.GiveAccessTo(secondCard));
        Assert.Contains("CopyTo", ex.Message);
    }

    [Fact]
    public void TestADetachingContextLeavesARetainedDeviceOutputOnTheCard()
    {
        using var card = new ComputeContext(new StubFactory(ComputeDevice.Cuda, 0), detachesOutputs: true);
        var onCard = TensorData([2L], (float[])[1f, 2f]).TransferTo(card);
        NamedModelParam[] outputs =
            [new TensorDataModelParam("state", ModelParamType.OutputParam, onCard)];

        var delivered = card.Deliver(outputs, new HashSet<string> { "state" });

        Assert.Same(card, delivered[0].ToTensorData().Context);
        Assert.Equal(MemorySpace.Cuda(0), delivered[0].ToTensorData().Space);
    }

    /// <summary>A backend that answers about itself and records what it was asked to build, so a
    /// transfer's route can be read off it without a session, a native runtime or a card.</summary>
    private sealed class StubFactory(ComputeDevice device, int? cudaDeviceId)
        : IShorokooInferenceBackend
    {
        public int BackendMemoryBuilds { get; private set; }

        public BackendDescription Description { get; } = new($"stub-{device}", device, cudaDeviceId);

        public IShorokooTensorValue CreateTensorInBackendMemory(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
        {
            BackendMemoryBuilds++;
            return new StubValue(elementType, data, shape, hostAccessible: device != ComputeDevice.Cuda);
        }

        public IShorokooTensorValue CreateTensorFromRawBytes(
            ShorokooTensorElementType elementType, byte[] data, long[] shape)
            => throw new NotSupportedException();

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
