using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// A tensor is a handle on an allocation that counts what names it, and moves between compute
/// contexts by <c>TransferTo</c> / <c>CopyTo</c> / <c>GiveAccessTo</c>.
///
/// <para>Everything here is host memory and two host contexts, which is the case the rules are
/// hardest to see working: nothing crashes when a reference is miscounted, so only the assertions
/// show it. The device pairing is <c>SideBySideBackendHardwareTests</c>'s.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class TensorContextTransferCoverageTests
{
    private static TensorData Sample() => TensorData([4L], (float[])[1f, 2f, 3f, 4f]);

    private static float[] Floats(TensorData t) => [.. t.As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestATensorWithNoContextHoldsHostMemory()
    {
        var t = Sample();

        Assert.Same(ComputeContext.Host, t.Context);
        Assert.Equal(MemorySpace.Host, t.Space);
        Assert.Same(MemoryDevice.For(MemorySpace.Host), t.Device);
        Assert.True(t.IsHostResident);
    }

    [Fact]
    public void TestTheHostContextIsExactlyWhatTheNullContextWas()
    {
        Assert.Equal(MemorySpace.Host, ComputeContext.Host.MemorySpace);

        foreach (var round in (Func<TensorData, TensorData>[])[
            static t => t.TransferTo(null),
            static t => t.TransferTo(ComputeContext.Host),
            static t => t.CopyTo(null),
            static t => t.CopyTo(ComputeContext.Host),
            static t => t.Detach(),
            static t => t.TransferTo(new ComputeContext()).TransferTo(null),
            static t => t.TransferTo(new ComputeContext()).TransferTo(ComputeContext.Host),
            static t => t.GiveAccessTo(new ComputeContext()).CopyTo(null),
            static t => t.GiveAccessTo(new ComputeContext()).CopyTo(ComputeContext.Host)])
        {
            var result = round(Sample());
            Assert.Same(ComputeContext.Host, result.Context);
            Assert.Equal(MemorySpace.Host, result.Space);
            Assert.Equal([1f, 2f, 3f, 4f], Floats(result));
        }
    }

    [Fact]
    public void TestTransferWithinOneSpaceMovesTheHandleAndNotTheBytes()
    {
        var source = Sample();
        var cpu = new ComputeContext();

        var moved = source.TransferTo(cpu);

        // The whole point: the data did not go anywhere, the handle did.
        Assert.Equal(MemorySpace.Host, moved.Space);
        Assert.Same(cpu, moved.Context);
        Assert.Same(cpu, source.Context);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(moved));

        // And the source is still readable -- it handed its reference over, it did not lose the
        // bytes.
        Assert.Equal([1f, 2f, 3f, 4f], Floats(source));
    }

    [Fact]
    public void TestTransferringTwiceInOneSpaceLeavesExactlyOneReference()
    {
        var first = new ComputeContext();
        var second = new ComputeContext();
        var source = Sample();

        var a = source.TransferTo(first);
        var b = a.TransferTo(second);

        Assert.Equal([1f, 2f, 3f, 4f], Floats(b));

        // Each transfer handed its own reference over rather than adding one, so the last handle
        // alone is what the bytes are waiting on.
        b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(source));
        Assert.Throws<ObjectDisposedException>(() => Floats(a));
    }

    [Fact]
    public void TestTransferringASecondHandleOnwardLeavesTheFirstReading()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        var onward = reader.TransferTo(new ComputeContext());
        onward.Dispose();

        Assert.Equal([1f, 2f, 3f, 4f], Floats(owner));
    }

    [Fact]
    public void TestTheHostContextTakesASecondHandleOnBytesAnotherTensorHolds()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        foreach (var onHost in (TensorData[])[owner.GiveAccessTo(null), reader.TransferTo(null)])
        {
            Assert.Same(ComputeContext.Host, onHost.Context);
            Assert.Equal([1f, 2f, 3f, 4f], Floats(onHost));
        }
    }

    [Fact]
    public void TestGiveAccessToLeavesBothHandlesReadingUntilTheLastOneGoes()
    {
        var owner = Sample();
        var cpu = new ComputeContext();

        var reader = owner.GiveAccessTo(cpu);

        Assert.Same(cpu, reader.Context);
        Assert.Same(ComputeContext.Host, owner.Context);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(reader));

        // Either one may go first and the other reads on; the bytes wait for the second.
        reader.Dispose();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(owner));
        owner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(reader));
    }

    [Fact]
    public void TestDisposingOneHandleLeavesTheOtherReadingAndDeletingStopsThemBoth()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        owner.Dispose();

        // The case that was a use-after-free: a second name for a buffer whose first name was
        // disposed is now a buffer that is simply still alive.
        Assert.False(reader.IsDisposed);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(reader));

        // Deletion still invalidates every handle, which is what makes it deletion.
        Assert.True(reader.TryDelete());
        Assert.Contains("deleted", Assert.Throws<ObjectDisposedException>(() => Floats(reader)).Message);
    }

    [Fact]
    public void TestCopyToAlwaysCopiesAndLeavesTheSourceAlone()
    {
        var source = Sample();
        var cpu = new ComputeContext();

        var copy = source.CopyTo(cpu);

        Assert.Same(cpu, copy.Context);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(copy));

        // Independent storage: releasing the source leaves the copy whole, which is what makes
        // CopyTo the way to outlive a context.
        source.Dispose();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(copy));
    }

    [Fact]
    public void TestCopyToWorksFromASecondHandleAndOutlivesBoth()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        var copy = reader.CopyTo(null);
        owner.Dispose();
        reader.Dispose();

        Assert.Equal([1f, 2f, 3f, 4f], Floats(copy));
    }

    [Fact]
    public void TestTheRoundTripBetweenTwoHostContextsIsAllNoOpsOnTheData()
    {
        var first = new ComputeContext();
        var second = new ComputeContext();

        var t = Sample().TransferTo(first).TransferTo(second).TransferTo(null);

        Assert.Same(ComputeContext.Host, t.Context);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(t));
    }
    [Fact]
    public void TestAStringTensorCrossesContextsByItsElements()
    {
        using var context = new ComputeContext();
        var strings = TensorData([2L], "a", "b");

        var copy = strings.CopyTo(context);

        Assert.Same(context, copy.Context);
        Assert.Equal(["a", "b"], ((HostStringTensorData)copy).Strings);
        Assert.Equal(["a", "b"], ((HostStringTensorData)strings).Strings);
    }

    [Fact]
    public void TestATransferredFromTensorNamesTheContextThatWillFreeItsBytes()
    {
        // A same-space transfer leaves the source readable with its reference handed over -- that
        // is the documented contract. What it must not leave behind is a source whose Context names
        // a context that no longer governs its bytes: Context is the only thing on the object that
        // says whose disposal takes them away.
        var first = new ComputeContext();
        var second = new ComputeContext();
        var tensor = Sample().TransferTo(first);
        tensor.TransferTo(second);

        Assert.Same(second, tensor.Context);

        first.Dispose();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(tensor));
        second.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(tensor));
    }

    [Fact]
    public void TestATransferFromFrameworkHostMemoryLeavesTheSourceNamingItsNewOwner()
    {
        var context = new ComputeContext();
        var tensor = Sample();
        Assert.Same(ComputeContext.Host, tensor.Context);

        tensor.TransferTo(context);

        Assert.Same(context, tensor.Context);
        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(tensor));
    }
}
