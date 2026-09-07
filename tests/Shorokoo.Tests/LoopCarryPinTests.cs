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

/// <summary>A nested rolled loop whose carry is assigned a value derived from a graph input's
/// shape. Concretizing it leaves the graph in an order the pipeline's own invariant rejects.</summary>
[Module]
public partial class NestedZeroTripShapeCarry
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
            var conv = NN.Conv(x, InitSimple.Init([Scalar(3L), Scalar(3L), Scalar(3L), Scalar(3L)]),
                InitSimple.Init([Scalar(3L)]).Vec(), AutoPad.NotSet,
                pads: [carry, carry, carry, carry], strides: Vector(1L, 1L), dilations: [carry, carry],
                kernelShape: [Scalar(3L), Scalar(3L)], group: Scalar(1L));
            total = total + carry + conv.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar().Cast<int64>() * Scalar(0L);
        }
        return total;
    }
}

/// <summary>A scan output seeded before the loop and returned as the loop's result.</summary>
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
    [Fact(Skip = "Shorokoo/Shorokoo#266: LoopAPI.Init drops a carry assigned from outside the body")]
    public void TestAZeroTripLoopReturnsTheCarrysPreLoopValue()
    {
        Assert.True(Returns<ZeroTripCarryFromInsideTheBody>(3, 8d));
        Assert.True(Returns<ZeroTripCarryFromOutsideTheBody>(3, 8d));
    }

    // Shorokoo/Shorokoo#268: this concretizes to a graph whose node order fails the pipeline's own
    // IsLinearOrderValid invariant — a Debug.Fail, and in Release a missing-producer error later.
    [Fact(Skip = "Shorokoo/Shorokoo#268: nested rolled loop with a shape-derived carry breaks linear order")]
    public void TestANestedRolledLoopCarryingAShapeDerivedValueConcretizes()
        => Assert.True(AutoTest.AdvancedTestGraph<NestedZeroTripShapeCarry>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [1L, 3L, 5L, 5L],
                    Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray()),
                TensorData(DType.Int64, [], 3L)],
            expected: [9d]));

    // Shorokoo/Shorokoo#267: a scan target seeded before the loop builds scan inputs that are not
    // body-produced, which the unroller asserts against. Only the `Variable? x = null` shape works.
    [Fact(Skip = "Shorokoo/Shorokoo#267: a scan output seeded before the loop is not body-produced")]
    public void TestAScanOutputSeededBeforeTheLoopStacksItsIterations()
        => Assert.True(AutoTest.AdvancedTestGraph<ScanSeededBeforeTheLoop>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Int64, [], 3L)],
            expected: [0d, 1d, 2d]));
}
