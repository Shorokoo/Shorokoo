using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// <c>y = -x</c>, the whole model: one node, no trainable parameter, no randomness, nothing whose
/// geometry has to be resolved by running anything. A model this small is the one that most
/// obviously needs no inference backend to describe, which is what makes it the subject of
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
        Assert.True(onTarget.OwnsMemory);

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
    public void TestAContextThatDetachesOutputsHandsBackResultsThatOutliveIt()
    {
        var (graph, a, b, expected) = Model();
        var context = new ComputeContext(InferenceBackend.Factory, detachesOutputs: true);

        Assert.True(context.DetachesOutputs);
        var result = context.Execute(graph, a, b)[0].ToTensorData();

        // Detached: the result belongs to nobody, which is what makes the next line safe.
        Assert.Null(result.Context);
        Assert.True(result.OwnsMemory);

        context.Dispose();
        Assert.Equal(expected, Floats(result));
    }

    [Fact]
    public void TestAContextThatDoesNotDetachTakesItsOutputsWithIt()
    {
        var (graph, a, b, _) = Model();
        var context = new ComputeContext(InferenceBackend.Factory, detachesOutputs: false);

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
        var context = new ComputeContext(InferenceBackend.Factory, detachesOutputs: true);
        var compiled = context.Compile(graph);

        var result = compiled.Execute(a, b)[0].ToTensorData();
        Assert.Null(result.Context);

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
    public void TestBuildingAndExportingAModelAsksForNoComputeContextAtAll()
    {
        var module = BackendFreeNegate.ComputationGraph;
        var sample = TensorData([8L], new float[8]);
        var onnx = Path.Combine(Path.GetTempPath(), $"shorokoo-backend-free-{Guid.NewGuid():N}.onnx");
        var liveBackend = InferenceBackend.Current;

        try
        {
            var reads = ComputeContext.CountDefaultReads(() =>
            {
                var concrete = module
                    .ToConcreteArchitecture(module.FromOrderedInputs([sample]))
                    .ToConcreteModel();
                Persistence.ExportOnnx(concrete, onnx);
            });

            Assert.Equal(0, reads);
            Assert.True(new FileInfo(onnx).Length > 0);
        }
        finally
        {
            if (File.Exists(onnx)) File.Delete(onnx);
        }

        Assert.Same(liveBackend, InferenceBackend.Current);
    }

    internal sealed class StubFactory(ComputeDevice device, int? cudaDeviceId)
        : IShorokooInferenceSessionFactory
    {
        public BackendDescription Description { get; } =
            new($"stub-{device}", device, cudaDeviceId);

        public IShorokooInferenceSession CreateSession(
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
        var live = InferenceBackend.Remembered;
        try
        {
            InferenceBackend.ForgetRemembered();
            var gpu = new ComputeContextLifetimeCoverageTests.StubFactory(ComputeDevice.Cuda, 0);
            var cpu = new ComputeContextLifetimeCoverageTests.StubFactory(ComputeDevice.Cpu, null);

            InferenceBackend.Remember(gpu);
            Assert.Same(gpu, InferenceBackend.Remembered);

            // The CPU one displaces it...
            InferenceBackend.Remember(cpu);
            Assert.Same(cpu, InferenceBackend.Remembered);

            // ...and is not displaced back, by that GPU backend or another.
            InferenceBackend.Remember(gpu);
            InferenceBackend.Remember(new ComputeContextLifetimeCoverageTests.StubFactory(ComputeDevice.Cuda, 1));
            Assert.Same(cpu, InferenceBackend.Remembered);

            // Nor by a second CPU one: the first backend loaded is the one that counts.
            InferenceBackend.Remember(new ComputeContextLifetimeCoverageTests.StubFactory(ComputeDevice.Cpu, null));
            Assert.Same(cpu, InferenceBackend.Remembered);
        }
        finally
        {
            InferenceBackend.ForgetRemembered();
            if (live is not null) InferenceBackend.Remember(live);
        }
    }

    /// <summary>
    /// Assigning <see cref="InferenceBackend.Factory"/> settles both slots outright — the live one
    /// and the remembered one — rather than going through the first-CPU-wins rule that governs a
    /// backend the process merely loaded. It does not say anything about
    /// <see cref="ComputeContext.Default"/>, which caches the backend it resolves on first read and
    /// keeps it: a program that means the default to follow an assignment has to make the
    /// assignment before anything reads Default.
    /// </summary>
    [Fact]
    public void TestAssigningTheFactorySettlesBothTheLiveAndTheRememberedSlot()
    {
        var liveFactory = InferenceBackend.Current;
        var liveRemembered = InferenceBackend.Remembered;
        try
        {
            InferenceBackend.ForgetRemembered();
            InferenceBackend.Remember(new ComputeContextLifetimeCoverageTests.StubFactory(ComputeDevice.Cpu, null));

            var named = new ComputeContextLifetimeCoverageTests.StubFactory(ComputeDevice.Cuda, 0);
            InferenceBackend.Factory = named;

            Assert.Same(named, InferenceBackend.Remembered);
            Assert.Same(named, InferenceBackend.Current);
        }
        finally
        {
            InferenceBackend.ForgetRemembered();
            if (liveFactory is not null) InferenceBackend.Factory = liveFactory;
            InferenceBackend.ForgetRemembered();
            if (liveRemembered is not null) InferenceBackend.Remember(liveRemembered);
        }
    }
}
