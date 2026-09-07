using System.Linq;

namespace Shorokoo.Tests;

/// <summary>The shape the build refuses: a carry the body assigns a value computed outside the
/// loop. Nothing in the body produces that value, so the loop has no node to re-trace and hand the
/// result back through.</summary>
[Module]
public partial class ZeroTripCarryFromOutsideTheBody
{
    public static Scalar<int64> Inline(Scalar<int64> n)
    {
        var carry = n + Scalar(5L);
        foreach (var ctx in LoopAPI.Iterate(n * Scalar(0L)))
        {
            LoopAPI.Init(carry);
#pragma warning disable MSG005 // the shape under test
            carry = n;
#pragma warning restore MSG005
        }
        return carry;
    }
}

/// <summary>The remedy the refusal names: wrapping the outside value makes the body produce it,
/// so the carry has a node the fourth pass can rebind to the loop's result.</summary>
[Module]
public partial class ZeroTripCarryWrappedFromOutside
{
    public static Scalar<int64> Inline(Scalar<int64> n)
    {
        var carry = n + Scalar(5L);
        foreach (var ctx in LoopAPI.Iterate(n * Scalar(0L)))
        {
            LoopAPI.Init(carry);
            carry = LoopAPI.Carry(n);
        }
        return carry;
    }
}

/// <summary>The same loop whose body value is a body node, which keeps the carry.</summary>
[Module]
public partial class ZeroTripCarryFromInsideTheBody
{
    public static Scalar<int64> Inline(Scalar<int64> n)
    {
        var carry = n + Scalar(5L);
        foreach (var ctx in LoopAPI.Iterate(n * Scalar(0L)))
        {
            LoopAPI.Init(carry);
            carry = n + Scalar(1L);
        }
        return carry;
    }
}

/// <summary>A nested rolled loop whose carry is assigned a shape-derived value, so the inner
/// LOOP_CLOSE's own inputs are all loop-invariant. Building a vector from that carry gives the
/// close's output consumers that look hoistable; moving them out of the outer loop would put them
/// above the nested loop that produces what they read.</summary>
[Module]
public partial class NestedLoopResultUsedInTheOuterBody
{
    public static Scalar<int64> Inline(Tensor<float32> x, Scalar<int64> trips)
    {
        var total = Scalar(0L);
        foreach (var outer in LoopAPI.Iterate(trips))
        {
            var carry = outer.IterationIndex + Scalar(1L);
            foreach (var inner in LoopAPI.Iterate(outer.IterationIndex))
            {
                LoopAPI.Init(carry);
                carry = x.ShapeTensor()[-1L] - Scalar(1L);
            }
            Vector<int64> pair = [carry, carry];
            total = total + pair.Reduce(ReduceKind.Sum).Scalar();
        }
        return total;
    }
}

/// <summary>A scan whose per-iteration value is the loop's iteration index itself — an output of
/// the LOOP_OPEN rather than of a body node.</summary>
[Module]
public partial class ScanOfTheIterationIndex
{
    public static Vector<int64> Inline(Scalar<int64> n)
    {
        Vector<int64> scanned = Vector(0L);
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            scanned = ctx.Scan(ctx.IterationIndex);
        return scanned;
    }
}

/// <summary>A rolled loop with no carries and a single scan output.</summary>
[Module]
public partial class RolledLoopWithOnlyAScanOutput
{
    public static Vector<int64> Inline(Scalar<int64> n)
    {
        Variable? scanned = null;
        foreach (var ctx in LoopAPI.Iterate(n))
            scanned = ctx.Scan(ctx.IterationIndex);
        return (Vector<int64>)scanned!;
    }
}

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class LoopSemanticsTests
{
    static bool Returns<TModule>(long n, double expected)
        => AutoTest.AdvancedTestGraph<TModule>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], n)], expected: [expected]);

    /// <summary>A zero-iteration loop returns the carry's pre-loop value, which is what
    /// <c>LoopAPI.Init</c> records. The body's assigned value must come from inside the body: one
    /// computed outside it is the same tensor the rest of the graph holds, so the loop's result has
    /// nowhere to land and the shape is refused rather than silently returning the body's value.</summary>
    [Fact]
    public void TestAZeroTripLoopReturnsTheCarrysPreLoopValue()
    {
        Assert.True(Returns<ZeroTripCarryFromInsideTheBody>(3, 8d));
        Assert.True(Returns<ZeroTripCarryWrappedFromOutside>(3, 8d));
        var ex = Assert.Throws<InvalidTensorOperationException>(
            () => _ = ZeroTripCarryFromOutsideTheBody.ComputationGraph);
        Assert.Contains("computed outside the loop", ex.Message);
    }

    [Fact]
    public void TestANestedLoopsResultStaysBelowTheLoopThatProducesIt()
        => Assert.True(AutoTest.AdvancedTestGraph<NestedLoopResultUsedInTheOuterBody>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [1L, 3L, 5L, 5L],
                    Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray()),
                TensorData(DType.Int64, [], 3L)],
            expected: [18d]));

    // Shorokoo/Shorokoo#279: with no carry to force it, the exported Loop omits its cond input, so
    // ONNX Runtime rejects the model. Pre-existing; reproduces on main.
    [Fact(Skip = "Shorokoo/Shorokoo#279: a scan-only rolled loop exports a Loop node with too few inputs")]
    public void TestARolledLoopWithOnlyAScanOutputExports()
        => Assert.True(AutoTest.AdvancedTestGraph<RolledLoopWithOnlyAScanOutput>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], 3L)],
            expected: [0d, 1d, 2d]));

    [Fact]
    public void TestAScanOfTheIterationIndexStacksItsIterations()
        => Assert.True(AutoTest.AdvancedTestGraph<ScanOfTheIterationIndex>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], 3L)],
            expected: [0d, 1d, 2d]));
}
