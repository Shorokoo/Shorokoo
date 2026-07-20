using System.Linq;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

namespace Shorokoo.Tests;

/// <summary>
/// Loop between two Linears, natural source order and no pin. A <c>LoopAPI.Iterate</c> loop
/// occupies exactly one top-level id slot at its source position, so: a → slot 1, loop → slot
/// 2, b → slot 3. This is the invariant the codegen pin suggestion relies on to emit correct
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
/// The same three consumers but with <c>a</c> and <c>b</c> created in the opposite order, then
/// pinned with the exact sparse statement the codegen suggestion emits for the baseline
/// (<c>Rng.Pin(([1], a), ([3], b))</c>). The pin must reproduce the baseline slot assignment —
/// a → 1, b → 3 — with the loop keeping slot 2 (the reserved gap), proving the suggestion
/// freezes the named streams against reordering without disturbing the loop's streams.
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
/// Validates the slot model behind the codegen pin suggestion for loop-containing bodies: a
/// <c>LoopAPI.Iterate</c> loop is one top-level slot, and the sparse pin the generator emits
/// (<c>([slot], item)</c> per nameable consumer) freezes those consumers' streams while leaving
/// the loop's slot — which has no nameable handle — undisturbed.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RngLoopPinTests
{
    private static (long slot1Out, long slot3Out, bool hasLoopSlot) Slots<TModule>()
    {
        var g = ((ComputationGraph)typeof(TModule)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var input = TensorData([1L, 4L], 0.1f, 0.2f, 0.3f, 0.4f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        var infos = arch.GetConcreteModelParamInfos().ParamInfos;

        long OutAt(int[] path) => infos.Single(i => i.ModelId.Vals.SequenceEqual(path)).Shape.Dims[0];
        // The loop occupies slot 2; its (unrolled, here) interior params live under [2, *].
        bool hasLoopSlot = infos.Any(i => i.ModelId.Vals.Length >= 2 && i.ModelId.Vals[0] == 2);
        return (OutAt([1, 1]), OutAt([3, 1]), hasLoopSlot);
    }

    [Fact]
    public void TestLoopOccupiesOneSlotBetweenNamedConsumers()
    {
        // Baseline: a (out=2) → slot 1, loop → slot 2, b (out=3) → slot 3.
        var (slot1, slot3, hasLoop) = Slots<LoopPinBaseline>();
        Assert.Equal(2L, slot1);
        Assert.Equal(3L, slot3);
        Assert.True(hasLoop);
    }

    [Fact]
    public void TestSparsePinFreezesNamedConsumersAroundLoop()
    {
        // Despite b being created first, the sparse pin reproduces the baseline mapping:
        // a → slot 1 (out=2), b → slot 3 (out=3), loop untouched at slot 2.
        var (slot1, slot3, hasLoop) = Slots<LoopPinReordered>();
        Assert.Equal(2L, slot1);
        Assert.Equal(3L, slot3);
        Assert.True(hasLoop);
    }
}
