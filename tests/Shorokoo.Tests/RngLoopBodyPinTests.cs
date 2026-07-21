using System.Linq;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

namespace Shorokoo.Tests;

// Loop-body pin, no pin (baseline): w[2] then w2[3] -> local slots 1, 2 in creation order.
[Module]
public partial class LoopBodyNoPin
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));
        var acc = a.Call(x);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            var w = InitSimple.Init([Scalar(2L)]);
            var w2 = InitSimple.Init([Scalar(3L)]);
            acc = acc + w.Reduce(ReduceKind.Sum) + w2.Reduce(ReduceKind.Sum);
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

// Rng.Pin(w2, w) INSIDE the loop body -> w2[3] takes loop-local slot 1, w[2] slot 2.
// The loop's own top-level slot (2) and the Linear's slot (1) are untouched.
[Module]
public partial class LoopBodyPositionalPin
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));
        var acc = a.Call(x);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            var w = InitSimple.Init([Scalar(2L)]);
            var w2 = InitSimple.Init([Scalar(3L)]);
            acc = acc + w.Reduce(ReduceKind.Sum) + w2.Reduce(ReduceKind.Sum);
            Rng.Pin(w2, w);
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

// Rng.Pin(([2], w)) INSIDE the loop -> w[2] pinned to loop-local slot 2; w2[3] fills slot 1.
[Module]
public partial class LoopBodySparsePin
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var acc = x;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            var w = InitSimple.Init([Scalar(2L)]);
            var w2 = InitSimple.Init([Scalar(3L)]);
            acc = acc + w.Reduce(ReduceKind.Sum) + w2.Reduce(ReduceKind.Sum);
            Rng.Pin(([2], w));
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

// Two sibling loops, each pinned independently: loop A swaps (Pin(q,p)); loop B keeps order (Pin(r,s)).
[Module]
public partial class SiblingLoopsPin
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var acc = x;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))          // loop A -> top slot 1
        {
            var p = InitSimple.Init([Scalar(2L)]);
            var q = InitSimple.Init([Scalar(3L)]);
            acc = acc + p.Reduce(ReduceKind.Sum) + q.Reduce(ReduceKind.Sum);
            Rng.Pin(q, p);
            ctx.ContinueWhile(Scalar(true));
        }
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))          // loop B -> top slot 2
        {
            var r = InitSimple.Init([Scalar(4L)]);
            var s = InitSimple.Init([Scalar(5L)]);
            acc = acc + r.Reduce(ReduceKind.Sum) + s.Reduce(ReduceKind.Sum);
            Rng.Pin(r, s);
            ctx.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

// Nested loops, pin inside the INNER body: Pin(v,u) -> inner-local slot 1 = v[3], slot 2 = u[2].
[Module]
public partial class NestedLoopPin
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var acc = x;
        foreach (var outer in LoopAPI.Iterate(Scalar(2L)))
        {
            foreach (var inner in LoopAPI.Iterate(Scalar(2L)))
            {
                var u = InitSimple.Init([Scalar(2L)]);
                var v = InitSimple.Init([Scalar(3L)]);
                acc = acc + u.Reduce(ReduceKind.Sum) + v.Reduce(ReduceKind.Sum);
                Rng.Pin(v, u);
                inner.ContinueWhile(Scalar(true));
            }
            outer.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

// Two sibling loops, each nested 2 levels deep, with pins at several scope depths:
//   Loop A (top slot 1): inner body pins Pin(a2, a1) -> a2[3] inner-local 1, a1[2] inner-local 2.
//   Loop B (top slot 2): its OUTER body holds a direct param b0[4] AND the inner loop, and pins
//     Pin(([2], b0)) -> b0 to outer-B local slot 2, pushing the inner loop to outer-B local 1;
//     its INNER body pins Pin(b2, b1) -> b2[6] inner-local 1, b1[5] inner-local 2.
[Module]
public partial class SiblingNestedLoopsPin
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var acc = x;
        foreach (var outerA in LoopAPI.Iterate(Scalar(2L)))
        {
            foreach (var innerA in LoopAPI.Iterate(Scalar(2L)))
            {
                var a1 = InitSimple.Init([Scalar(2L)]);
                var a2 = InitSimple.Init([Scalar(3L)]);
                acc = acc + a1.Reduce(ReduceKind.Sum) + a2.Reduce(ReduceKind.Sum);
                Rng.Pin(a2, a1);
                innerA.ContinueWhile(Scalar(true));
            }
            outerA.ContinueWhile(Scalar(true));
        }
        foreach (var outerB in LoopAPI.Iterate(Scalar(2L)))
        {
            var b0 = InitSimple.Init([Scalar(4L)]);
            foreach (var innerB in LoopAPI.Iterate(Scalar(2L)))
            {
                var b1 = InitSimple.Init([Scalar(5L)]);
                var b2 = InitSimple.Init([Scalar(6L)]);
                acc = acc + b1.Reduce(ReduceKind.Sum) + b2.Reduce(ReduceKind.Sum);
                Rng.Pin(b2, b1);
                innerB.ContinueWhile(Scalar(true));
            }
            acc = acc + b0.Reduce(ReduceKind.Sum);
            Rng.Pin(([2], b0));
            outerB.ContinueWhile(Scalar(true));
        }
        return acc;
    }
}

/// <summary>
/// Pins written INSIDE loop bodies reshape only that loop's local id slots — across sibling
/// loops and any nesting depth — while leaving the loop's own (parent-scope) slot alone. Each
/// loop body is traced several times during construction; the pin records only in the
/// canonical pass, so it resolves to the surviving nodes exactly once.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngLoopBodyPinTests
{
    // Concretize and return each trainable param's (full ModelId, shape).
    private static (int[] id, long[] shape)[] Params<TModule>()
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([1L, 4L], 0.1f, 0.2f, 0.3f, 0.4f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        return arch.GetConcreteModelParamInfos().ParamInfos
            .Select(i => (i.ModelId.Vals.ToArray(), i.Shape.Dims.ToArray()))
            .ToArray();
    }

    // The loop-local slot (last id element) that the rank-1 param of a given size landed on.
    // Loop init params here are all vectors, so rank-1 + size uniquely identifies one.
    private static int LocalSlotOfShape((int[] id, long[] shape)[] ps, long size)
    {
        var hit = ps.Where(p => p.shape.Length == 1 && p.shape[0] == size).ToArray();
        Assert.NotEmpty(hit);
        var slots = hit.Select(p => p.id[^1]).Distinct().ToArray();
        Assert.Single(slots);   // same local slot in every unrolled iteration
        return slots[0];
    }

    // The top-level loop slot (first id element) of the rank-1 param of a given size.
    private static int TopSlotOfShape((int[] id, long[] shape)[] ps, long size)
    {
        var tops = ps.Where(p => p.shape.Length == 1 && p.shape[0] == size)
            .Select(p => p.id[0]).Distinct().ToArray();
        Assert.Single(tops);
        return tops[0];
    }

    [Fact]
    public void TestLoopBodyPositionalPinSwapsLocalSlotsAndKeepsLoopSlot()
    {
        var baseline = Params<LoopBodyNoPin>();
        // Baseline creation order: w[2] -> local slot 1, w2[3] -> local slot 2.
        Assert.Equal(1, LocalSlotOfShape(baseline, 2));
        Assert.Equal(2, LocalSlotOfShape(baseline, 3));

        var pinned = Params<LoopBodyPositionalPin>();
        // Pin(w2, w) -> w2[3] to local slot 1, w[2] to local slot 2.
        Assert.Equal(1, LocalSlotOfShape(pinned, 3));
        Assert.Equal(2, LocalSlotOfShape(pinned, 2));
        // The loop's own top-level slot is unchanged: the Linear ([2,4]) still holds slot 1.
        Assert.Contains(pinned, p => p.shape.SequenceEqual((long[])[2L, 4L]) && p.id.SequenceEqual((int[])[1, 1]));
    }

    [Fact]
    public void TestLoopBodySparsePinTakesNamedLocalSlot()
    {
        var ps = Params<LoopBodySparsePin>();
        // Pin(([2], w)) -> w[2] at local slot 2; w2[3] fills the free local slot 1.
        Assert.Equal(2, LocalSlotOfShape(ps, 2));
        Assert.Equal(1, LocalSlotOfShape(ps, 3));
    }

    [Fact]
    public void TestSiblingLoopsPinIndependently()
    {
        var ps = Params<SiblingLoopsPin>();
        // Loop A: Pin(q, p) -> q[3] local 1, p[2] local 2.
        Assert.Equal(1, LocalSlotOfShape(ps, 3));
        Assert.Equal(2, LocalSlotOfShape(ps, 2));
        // Loop B: Pin(r, s) -> creation order kept: r[4] local 1, s[5] local 2.
        Assert.Equal(1, LocalSlotOfShape(ps, 4));
        Assert.Equal(2, LocalSlotOfShape(ps, 5));
        // Distinct top-level loop slots (A's params start with 1, B's with 2).
        Assert.Equal(1, TopSlotOfShape(ps, 3));
        Assert.Equal(2, TopSlotOfShape(ps, 4));
    }

    [Fact]
    public void TestNestedLoopPinReshapesInnerScope()
    {
        var ps = Params<NestedLoopPin>();
        // Inner Pin(v, u) -> v[3] inner-local slot 1, u[2] inner-local slot 2.
        Assert.Equal(1, LocalSlotOfShape(ps, 3));
        Assert.Equal(2, LocalSlotOfShape(ps, 2));
    }

    // The id element at position `index` for the rank-1 param of a given size.
    private static int IdElemOfShape((int[] id, long[] shape)[] ps, long size, int index)
    {
        var vals = ps.Where(p => p.shape.Length == 1 && p.shape[0] == size)
            .Select(p => p.id[index]).Distinct().ToArray();
        Assert.Single(vals);
        return vals[0];
    }

    [Fact]
    public void TestSiblingLoopsWithTwoLevelNestingPinAtEveryDepth()
    {
        var ps = Params<SiblingNestedLoopsPin>();

        // Sibling loops occupy distinct top-level slots: A's params start with 1, B's with 2.
        Assert.Equal(1, TopSlotOfShape(ps, 2));   // a1 in loop A
        Assert.Equal(1, TopSlotOfShape(ps, 3));   // a2 in loop A
        Assert.Equal(2, TopSlotOfShape(ps, 4));   // b0 in loop B
        Assert.Equal(2, TopSlotOfShape(ps, 5));   // b1 in loop B
        Assert.Equal(2, TopSlotOfShape(ps, 6));   // b2 in loop B

        // Inner A: Pin(a2, a1) -> a2[3] inner-local slot 1, a1[2] inner-local slot 2.
        Assert.Equal(1, LocalSlotOfShape(ps, 3));
        Assert.Equal(2, LocalSlotOfShape(ps, 2));

        // Inner B: Pin(b2, b1) -> b2[6] inner-local slot 1, b1[5] inner-local slot 2.
        Assert.Equal(1, LocalSlotOfShape(ps, 6));
        Assert.Equal(2, LocalSlotOfShape(ps, 5));

        // Outer B: Pin(([2], b0)) -> b0[4] takes outer-B local slot 2, so the inner B loop is
        // pushed to outer-B local slot 1. The outer-B-local slot is id element index 2 (after
        // the first -1). b0's path is [2, -1, 2]; b1/b2's paths run through the inner loop at
        // outer-B-local slot 1, so their element at index 2 is 1.
        Assert.Equal(2, IdElemOfShape(ps, 4, 2));   // b0 -> outer-B local slot 2
        Assert.Equal(1, IdElemOfShape(ps, 5, 2));   // inner-B loop -> outer-B local slot 1
        Assert.Equal(1, IdElemOfShape(ps, 6, 2));
    }
}
