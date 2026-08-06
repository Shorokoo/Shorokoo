using System.Linq;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;

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
/// Loop between two Linears, natural source order and no pin. A <c>LoopAPI.Iterate</c> loop
/// occupies exactly one top-level id slot at its source position: a → slot 1, loop → slot 2,
/// b → slot 3. This is the invariant the codegen pin suggestion relies on to emit correct
/// sparse slots for loop-containing bodies.
/// </summary>
[Module]
public partial class LoopPinBaseline
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = Linear.Model(Scalar(2L), Scalar(false));      // slot 1
        var acc = a.Call(x);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))       // slot 2 (Init inside)
        {
            var w = InitSimple.Init(acc.ShapeTensor());
            acc = acc * w;
            ctx.ContinueWhile(Scalar(true));
        }
        var b = Linear.Model(Scalar(3L), Scalar(false));      // slot 3
        return acc.Concat(-1L, b.Call(acc));
    }
}

/// <summary>
/// The same three consumers with <c>a</c> and <c>b</c> created in the opposite order, then
/// pinned with the exact sparse statement the codegen suggestion emits for the baseline.
/// </summary>
[Module]
public partial class LoopPinReordered
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var b = Linear.Model(Scalar(3L), Scalar(false));      // created 1st — would be slot 1 unpinned
        var a = Linear.Model(Scalar(2L), Scalar(false));      // created 2nd
        var acc = a.Call(x);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            var w = InitSimple.Init(acc.ShapeTensor());
            acc = acc * w;
            ctx.ContinueWhile(Scalar(true));
        }
        Rng.Pin(([1], a), ([3], b));                           // codegen's suggested sparse pin
        return acc.Concat(-1L, b.Call(acc));
    }
}

/// <summary>
/// Pins written INSIDE loop bodies reshape only that loop's local id slots — across sibling
/// loops and any nesting depth — while leaving the loop's own (parent-scope) slot alone. Each
/// loop body is traced several times during construction; the pin records only in the canonical
/// pass, so it resolves to the surviving nodes exactly once. Also the slot model behind the
/// codegen pin suggestion: a loop is one top-level slot, and a sparse pin around it freezes the
/// nameable consumers without disturbing the loop's own streams.
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

    // The id element at position `index` for the rank-1 param of a given size. Loop init params
    // here are all vectors, so rank-1 + size uniquely identifies one; the value must agree
    // across every unrolled iteration.
    private static int IdElemOfShape((int[] id, long[] shape)[] ps, long size, int index)
    {
        var hit = ps.Where(p => p.shape.Length == 1 && p.shape[0] == size).ToArray();
        Assert.NotEmpty(hit);
        var vals = hit.Select(p => p.id[index >= 0 ? index : p.id.Length + index]).Distinct().ToArray();
        Assert.Single(vals);
        return vals[0];
    }

    // The loop-local slot (last id element) / the top-level loop slot (first id element).
    private static int LocalSlotOfShape((int[] id, long[] shape)[] ps, long size)
        => IdElemOfShape(ps, size, -1);
    private static int TopSlotOfShape((int[] id, long[] shape)[] ps, long size)
        => IdElemOfShape(ps, size, 0);

    [Fact]
    public void TestLoopBodyPinsTakeLocalSlotsAndKeepTheLoopsOwnSlot()
    {
        // Baseline creation order: w[2] -> local slot 1, w2[3] -> local slot 2.
        var baseline = Params<LoopBodyNoPin>();
        Assert.Equal(1, LocalSlotOfShape(baseline, 2));
        Assert.Equal(2, LocalSlotOfShape(baseline, 3));

        // Pin(w2, w) -> w2[3] to local slot 1, w[2] to local slot 2; the loop's own top-level
        // slot is unchanged (the Linear's [2,4] weight still holds slot 1).
        var pinned = Params<LoopBodyPositionalPin>();
        Assert.Equal(1, LocalSlotOfShape(pinned, 3));
        Assert.Equal(2, LocalSlotOfShape(pinned, 2));
        Assert.Contains(pinned, p => p.shape.SequenceEqual((long[])[2L, 4L]) && p.id.SequenceEqual((int[])[1, 1]));

        // Pin(([2], w)) -> w[2] at local slot 2; w2[3] fills the free local slot 1.
        var sparse = Params<LoopBodySparsePin>();
        Assert.Equal(2, LocalSlotOfShape(sparse, 2));
        Assert.Equal(1, LocalSlotOfShape(sparse, 3));
    }

    [Fact]
    public void TestPinsInSiblingAndNestedLoopScopesApplyIndependentlyAtEveryDepth()
    {
        // Sibling loops, one swapping (Pin(q, p)) and one keeping creation order (Pin(r, s)),
        // at distinct top-level slots.
        var siblings = Params<SiblingLoopsPin>();
        Assert.Equal(1, LocalSlotOfShape(siblings, 3));
        Assert.Equal(2, LocalSlotOfShape(siblings, 2));
        Assert.Equal(1, LocalSlotOfShape(siblings, 4));
        Assert.Equal(2, LocalSlotOfShape(siblings, 5));
        Assert.Equal(1, TopSlotOfShape(siblings, 3));
        Assert.Equal(2, TopSlotOfShape(siblings, 4));

        // Inner Pin(v, u) -> v[3] inner-local slot 1, u[2] inner-local slot 2.
        var nested = Params<NestedLoopPin>();
        Assert.Equal(1, LocalSlotOfShape(nested, 3));
        Assert.Equal(2, LocalSlotOfShape(nested, 2));

        // Two sibling loops nested two levels deep, pinned at several scope depths.
        var ps = Params<SiblingNestedLoopsPin>();
        Assert.Equal(1, TopSlotOfShape(ps, 2));   // a1 in loop A
        Assert.Equal(1, TopSlotOfShape(ps, 3));   // a2 in loop A
        Assert.Equal(2, TopSlotOfShape(ps, 4));   // b0 in loop B
        Assert.Equal(2, TopSlotOfShape(ps, 5));   // b1 in loop B
        Assert.Equal(2, TopSlotOfShape(ps, 6));   // b2 in loop B
        Assert.Equal(1, LocalSlotOfShape(ps, 3)); // inner A: Pin(a2, a1)
        Assert.Equal(2, LocalSlotOfShape(ps, 2));
        Assert.Equal(1, LocalSlotOfShape(ps, 6)); // inner B: Pin(b2, b1)
        Assert.Equal(2, LocalSlotOfShape(ps, 5));
        // Outer B: Pin(([2], b0)) -> b0 takes outer-B local slot 2 (id element index 2, after
        // the first -1), pushing the inner B loop to outer-B local slot 1.
        Assert.Equal(2, IdElemOfShape(ps, 4, 2));
        Assert.Equal(1, IdElemOfShape(ps, 5, 2));
        Assert.Equal(1, IdElemOfShape(ps, 6, 2));
    }

    [Fact]
    public void TestLoopOccupiesOneTopSlotAndASparsePinFreezesTheNamedConsumersAroundIt()
    {
        // Baseline: a (out=2) → slot 1, loop → slot 2, b (out=3) → slot 3. Despite b being
        // created first in the reordered module, the sparse pin reproduces that mapping.
        static void AssertSlots((int[] id, long[] shape)[] ps)
        {
            long OutAt(int[] path) => ps.Single(p => p.id.SequenceEqual(path)).shape[0];
            Assert.Equal(2L, OutAt([1, 1]));
            Assert.Equal(3L, OutAt([3, 1]));
            // The loop occupies slot 2; its (unrolled) interior params live under [2, *].
            Assert.Contains(ps, p => p.id.Length >= 2 && p.id[0] == 2);
        }
        AssertSlots(Params<LoopPinBaseline>());
        AssertSlots(Params<LoopPinReordered>());
    }
}
