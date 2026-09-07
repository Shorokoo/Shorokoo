using System.Linq;

namespace Shorokoo.Tests;

/// <summary>A zero-iteration loop must return the carry's pre-loop value, which is what
/// <c>LoopAPI.Init</c> exists to record. It does when the body assigns a value computed inside the
/// body, and does not when the body assigns one computed outside it: the carry is dropped and the
/// body's value is returned even though the body never ran.</summary>
[Module]
public partial class ZeroTripCarryFromOutsideTheBody
{
    public static Scalar<int64> Inline(Scalar<int64> n)
    {
        var carry = n + Scalar(5L);
        foreach (var ctx in LoopAPI.Iterate(n * Scalar(0L)))
        {
            LoopAPI.Init(carry);
            carry = n;
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

/// <summary>A scan output whose per-iteration value is the loop's iteration index itself, rather
/// than a body node's output.</summary>
[Module]
public partial class ScanSeededBeforeTheLoop
{
    public static Vector<int64> Inline(Scalar<int64> n)
    {
        Vector<int64> scanned = Vector(0L);
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            scanned = ctx.Scan(ctx.IterationIndex);
        return scanned;
    }
}

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class LoopCarryPinTests
{
    static bool Returns<TModule>(long n, double expected)
        => AutoTest.AdvancedTestGraph<TModule>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], n)], expected: [expected]);

    // Shorokoo/Shorokoo#266: LoopAPI.Init records nothing when the body's assigned value has no
    // producer inside the body, so a zero-iteration loop returns that value instead of the
    // pre-loop one. ZeroTripCarryFromInsideTheBody is the passing control.
    /// <summary>A zero-iteration loop returns the carry's pre-loop value, which is what
    /// <c>LoopAPI.Init</c> records. The body's assigned value must come from inside the body: one
    /// computed outside it is the same tensor the rest of the graph holds, so the loop's result has
    /// nowhere to land and the shape is refused rather than silently returning the body's value.</summary>
    [Fact]
    public void TestAZeroTripLoopReturnsTheCarrysPreLoopValue()
    {
        Assert.True(Returns<ZeroTripCarryFromInsideTheBody>(3, 8d));
        var ex = Assert.Throws<InvalidTensorOperationException>(
            () => _ = ZeroTripCarryFromOutsideTheBody.ComputationGraph);
        Assert.Contains("computed outside the loop", ex.Message);
    }

    // Shorokoo/Shorokoo#268: this concretizes to a graph whose node order fails the pipeline's own
    // IsLinearOrderValid invariant — a Debug.Fail, and in Release a missing-producer error later.
    [Fact]
    public void TestANestedLoopsResultStaysBelowTheLoopThatProducesIt()
        => Assert.True(AutoTest.AdvancedTestGraph<NestedLoopResultUsedInTheOuterBody>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [1L, 3L, 5L, 5L],
                    Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray()),
                TensorData(DType.Int64, [], 3L)],
            expected: [18d]));

    [Fact]
    public void TestAScanOutputSeededBeforeTheLoopStacksItsIterations()
        => Assert.True(AutoTest.AdvancedTestGraph<ScanSeededBeforeTheLoop>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], 3L)],
            expected: [0d, 1d, 2d]));
}
