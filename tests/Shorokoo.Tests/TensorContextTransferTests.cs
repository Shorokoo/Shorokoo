using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// A tensor is its memory, and never moves: <c>To</c> hands the tensor itself to a context whose
/// backend can read it where it is and copies otherwise, <c>CopyTo</c> always copies, and
/// <c>ToHost</c> is <c>To</c> for host memory. None of them touches the source.
///
/// <para>Everything here is host memory and host contexts. The device pairing is
/// <c>CrossDeviceRoutingCoverageTests</c>' on stubs, and <c>SideBySideBackendHardwareTests</c>' on a
/// card.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class TensorContextTransferCoverageTests
{
    private static TensorData Sample() => TensorData([4L], (float[])[1f, 2f, 3f, 4f]);

    private static float[] Floats(TensorData t) => [.. t.As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestATensorBuiltFromAnArrayIsTheFrameworksOwnManagedHostMemory()
    {
        var t = Sample();

        Assert.Same(HostBackend.Instance, t.AllocatingBackend);
        Assert.Equal(MemorySpace.Host, t.Space);
        Assert.Same(MemoryDevice.For(MemorySpace.Host), t.Device);
        Assert.True(t.Location.IsManaged);
        Assert.Equal(new MemoryLocation(MemorySpace.Host, HostBackend.Instance), t.Location);
        Assert.True(t.IsHostResident);
        Assert.False(t.IsDisposed);
    }

    [Fact]
    public void TestToAContextThatCanReadTheTensorIsTheSameObjectAndCopiesNothing()
    {
        using var cpu = new ComputeContext();
        var secondBackend = (IShorokooBackend)Activator.CreateInstance(DefaultBackend.Instance.GetType())!;
        using var alsoCpu = new ComputeContext(secondBackend);
        var t = Sample();

        Assert.NotSame(DefaultBackend.Instance, secondBackend);
        Assert.Same(DefaultBackend.Instance.RuntimeIdentity, secondBackend.RuntimeIdentity);

        foreach (var to in (Func<TensorData, TensorData>[])[
            x => x.To(cpu), x => x.To(alsoCpu), x => x.To(ComputeContext.Host), x => x.ToHost()])
            Assert.Same(t, to(t));

        var output = cpu.Execute(Doubling(), t)[0].ToTensorData();
        Assert.Same(DefaultBackend.Instance, output.AllocatingBackend);
        Assert.Same(output, output.To(alsoCpu));
        Assert.Same(output, output.To(ComputeContext.Host));
        Assert.Same(output, output.ToHost());
        Assert.Equal([1f, 2f, 3f, 4f], Floats(t));
    }

    [Fact]
    public void TestCopyToAlwaysMakesAnIndependentCopyAndLeavesTheSourceAlone()
    {
        using var cpu = new ComputeContext();
        var source = Sample();

        foreach (var target in (ComputeContext[])[cpu, ComputeContext.Host])
        {
            var copy = (TensorData<float32>)source.CopyTo(target);

            Assert.NotSame(source, copy);
            Assert.Same(HostBackend.Instance, copy.AllocatingBackend);
            copy.AccessModifiableMemory<float>()[0] = 9f;
            Assert.Equal([9f, 2f, 3f, 4f], Floats(copy));
            Assert.Equal([1f, 2f, 3f, 4f], Floats(source));
        }

        var kept = source.CopyTo(cpu);
        source.Delete();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(kept));
        Assert.Contains(kept, cpu.Tensors);
    }

    [Fact]
    public void TestAStringTensorCrossesContextsByItsElements()
    {
        using var context = new ComputeContext();
        var strings = TensorData([2L], "a", "b");

        var copy = strings.CopyTo(context);

        Assert.NotSame(strings, copy);
        Assert.Contains(copy, context.Tensors);
        Assert.Equal(["a", "b"], ((HostStringTensorData)copy).Strings);
        Assert.Equal(["a", "b"], ((HostStringTensorData)strings).Strings);
        Assert.Same(strings, strings.To(context));
        Assert.Same(strings, strings.ToHost());
    }

    [Fact]
    public void TestTheOperationsRefuseADeadTensorADisposedContextAndNoContextAtAll()
    {
        var dead = Sample();
        dead.Delete();
        var disposed = new ComputeContext();
        disposed.Dispose();

        Assert.Throws<ObjectDisposedException>(() => dead.To(ComputeContext.Host));
        Assert.Throws<ObjectDisposedException>(() => dead.CopyTo(ComputeContext.Host));
        Assert.Throws<ObjectDisposedException>(() => dead.ToHost());
        Assert.Throws<ObjectDisposedException>(() => Sample().To(disposed));
        Assert.Throws<ObjectDisposedException>(() => Sample().CopyTo(disposed));
        Assert.Throws<ArgumentNullException>(() => Sample().To(null!));
        Assert.Throws<ArgumentNullException>(() => Sample().CopyTo(null!));
    }

    [Fact]
    public void TestAWriteThroughATensorIsSeenByARunFedWhatToHandedOver()
    {
        using var context = new ComputeContext();
        var t = (TensorData<float32>)TensorData([2L], (float[])[1f, 2f]);
        var handed = t.To(context);

        Assert.Equal([2f, 4f], Floats(context.Execute(Doubling(), handed)[0].ToTensorData()));
        t.AccessModifiableMemory<float>()[0] = 99f;
        Assert.Equal([198f, 4f], Floats(context.Execute(Doubling(), handed)[0].ToTensorData()));
    }

    private static InternalComputationGraph Doubling()
    {
        var a = InputVector<float32>("a");
        return new InternalComputationGraph([a], [a + a]);
    }
}
