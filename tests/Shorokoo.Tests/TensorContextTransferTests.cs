using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// A tensor belongs to a compute context, says whether it owns its bytes, and moves between
/// contexts by <c>TransferTo</c> / <c>CopyTo</c> / <c>GiveAccessTo</c>.
///
/// <para>Everything here is host memory and two host contexts, which is the case the rules are
/// hardest to see working: nothing crashes when ownership is wrong, so only the assertions show
/// it. The device pairing is <c>SideBySideBackendHardwareTests</c>'s.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class TensorContextTransferCoverageTests
{
    private static TensorData Sample() => TensorData([4L], (float[])[1f, 2f, 3f, 4f]);

    private static float[] Floats(TensorData t) => [.. t.As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestATensorWithNoContextOwnsHostMemory()
    {
        var t = Sample();

        Assert.Same(ComputeContext.Host, t.Context);
        Assert.True(t.OwnsMemory);
        Assert.Equal(MemorySpace.Host, t.Space);
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
            Assert.True(result.OwnsMemory);
            Assert.Equal([1f, 2f, 3f, 4f], Floats(result));
        }

        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());
        foreach (var refused in (Func<TensorData>[])[
            () => owner.GiveAccessTo(ComputeContext.Host),
            () => reader.TransferTo(ComputeContext.Host)])
            Assert.Contains(
                "CopyTo(null)", Assert.Throws<InvalidOperationException>(refused).Message);
    }

    [Fact]
    public void TestTransferWithinOneSpaceMovesOwnershipAndNotTheBytes()
    {
        var source = Sample();
        var cpu = new ComputeContext();

        var moved = source.TransferTo(cpu);

        // The whole point: the data did not go anywhere, the ownership did.
        Assert.Equal(MemorySpace.Host, moved.Space);
        Assert.Same(cpu, moved.Context);
        Assert.True(moved.OwnsMemory);
        Assert.False(source.OwnsMemory);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(moved));

        // And the source is still readable -- it stopped owning the bytes, it did not lose them.
        Assert.Equal([1f, 2f, 3f, 4f], Floats(source));
    }

    [Fact]
    public void TestTransferringTwiceInOneSpaceLeavesExactlyOneOwner()
    {
        var first = new ComputeContext();
        var second = new ComputeContext();
        var source = Sample();

        var a = source.TransferTo(first);
        var b = a.TransferTo(second);

        Assert.False(source.OwnsMemory);
        Assert.False(a.OwnsMemory);
        Assert.True(b.OwnsMemory);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(b));
    }

    [Fact]
    public void TestTransferFromANonOwnerGivesANonOwner()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        var onward = reader.TransferTo(new ComputeContext());

        // Neither of them owns: a transfer cannot invent an ownership its source never had.
        Assert.False(reader.OwnsMemory);
        Assert.False(onward.OwnsMemory);
        Assert.True(owner.OwnsMemory);
    }

    [Fact]
    public void TestTheNullContextCannotHoldMemoryItDoesNotOwn()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        foreach (var refused in (Func<TensorData>[])[
            () => owner.GiveAccessTo(null),
            () => reader.TransferTo(null)])
        {
            var ex = Assert.Throws<InvalidOperationException>(() => refused());
            Assert.Contains("CopyTo(null)", ex.Message);
        }

        // And the way through is the one the message names.
        Assert.True(reader.CopyTo(null).OwnsMemory);
    }

    [Fact]
    public void TestGiveAccessToNeverTakesOwnershipAndLeavesTheSourceOwning()
    {
        var owner = Sample();
        var cpu = new ComputeContext();

        var reader = owner.GiveAccessTo(cpu);

        Assert.False(reader.OwnsMemory);
        Assert.True(owner.OwnsMemory);
        Assert.Same(cpu, reader.Context);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(reader));

        // Disposing a reader frees nothing, so the owner reads on.
        reader.Dispose();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(owner));
    }

    [Fact]
    public void TestReleasingTheOwnerMakesEveryReaderThrowRatherThanReadFreedMemory()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        owner.Dispose();

        // The reader was never disposed. It is pointing at memory whose owner let go, which is
        // the case a raw pointer cannot detect and this is here to make impossible.
        Assert.False(reader.IsDisposed);
        var ex = Assert.Throws<ObjectDisposedException>(() => Floats(reader));
        Assert.Contains("released it", ex.Message);
    }

    [Fact]
    public void TestCopyToAlwaysCopiesAndLeavesTheSourceAlone()
    {
        var source = Sample();
        var cpu = new ComputeContext();

        var copy = source.CopyTo(cpu);

        Assert.True(copy.OwnsMemory);
        Assert.True(source.OwnsMemory);
        Assert.Same(cpu, copy.Context);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(copy));

        // Independent storage: releasing the source leaves the copy whole, which is what makes
        // CopyTo the way to outlive a context.
        source.Dispose();
        Assert.Equal([1f, 2f, 3f, 4f], Floats(copy));
    }

    [Fact]
    public void TestCopyToWorksFromATensorThatOwnsNothing()
    {
        var owner = Sample();
        var reader = owner.GiveAccessTo(new ComputeContext());

        var copy = reader.CopyTo(null);

        Assert.True(copy.OwnsMemory);
        Assert.Equal([1f, 2f, 3f, 4f], Floats(copy));
    }

    [Fact]
    public void TestTheRoundTripBetweenTwoHostContextsIsAllNoOpsOnTheData()
    {
        var first = new ComputeContext();
        var second = new ComputeContext();

        var t = Sample().TransferTo(first).TransferTo(second).TransferTo(null);

        Assert.Same(ComputeContext.Host, t.Context);
        Assert.True(t.OwnsMemory);
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
        // A same-space transfer leaves the source readable and owning nothing -- that is the
        // documented contract. What it must not leave behind is a source whose Context names a
        // context that no longer governs its bytes: Context is the only thing on the object that
        // says whose disposal takes them away.
        var first = new ComputeContext();
        var second = new ComputeContext();
        var tensor = Sample().TransferTo(first);
        tensor.TransferTo(second);

        Assert.False(tensor.OwnsMemory);
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

        Assert.False(tensor.OwnsMemory);
        Assert.Same(context, tensor.Context);
        context.Dispose();
        Assert.Throws<ObjectDisposedException>(() => Floats(tensor));
    }
}
