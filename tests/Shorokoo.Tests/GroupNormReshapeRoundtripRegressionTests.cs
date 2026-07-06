using System.Linq;
using Shorokoo.Modules.Layers;
using Shorokoo.Tests.Utils;
using static Shorokoo.Globals;

namespace Shorokoo.Tests;

// Minimal repro of a save/load round-trip serialization bug.
//
// GroupNorm normalizes over a [N, G, -1] reshape and then reshapes BACK to the input shape,
// so its output carries DYNAMIC shape metadata (derived through a Shape->Reshape chain rather
// than a static constant). Reshaping such a dynamic-shape function output to a STATIC shape
// makes a graph pass synthesize a reconciling reshape wired to GroupNorm's internal shape
// tensor. The graph executes correctly as-is, but after a compressed save/load round-trip that
// synthesized reshape references a tensor name that no node produces, so ONNX Runtime rejects
// the reloaded graph at init ("GetIndexFromName ... a name which does not exist: ... _new_reshape").
//
// A free-dimension flatten (Reshape([-1])) does NOT trigger it, and the same static reshape on a
// statically-shaped output (e.g. InstanceNorm) is fine — so the trigger is specifically
// (dynamic-shape function output) + (static-shape reshape) + (save/load round-trip).
[Module]
public partial class GroupNormStaticReshapeRoundtripRepro
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [2, 4, 3, 3] = 72 elements
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var flat = y.Reshape([Scalar(72L)]);              // STATIC target shape — the trigger (vs. [-1])
        return SelfCheck.Nan(flat) < Scalar(1f);          // finite output => true; self-checking
    }
}

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class GroupNormReshapeRoundtripRegressionTests
{
    private static TensorData Range(long[] dims, float scale, float offset)
    {
        long total = 1; foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims, Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    /// <summary>
    /// Reshaping GroupNorm's (dynamic-shape) output to a static shape must survive Shorokoo's
    /// compressed save/load round-trip. Today the reloaded graph has a dangling reshape input and
    /// ONNX Runtime fails to initialize it. AutoTest.AdvancedTestGraph performs the round-trip, so
    /// this fails until the serialization bug is fixed.
    ///
    /// Pinned to the open bug https://github.com/Shorokoo/Shorokoo/issues/10 (static reshape of a
    /// dynamic-shape function output dangles a reshape input after save/load round-trip). The test
    /// is self-checking, so removing the Skip flips it green the moment the round-trip is fixed.
    /// </summary>
    [Fact(Skip = "Shorokoo/Shorokoo#10: static reshape of a dynamic-shape function output dangles a reshape input after the compressed save/load round-trip")]
    public void GroupNormOutputStaticReshapeSurvivesRoundtrip()
        => Assert.True(AutoTest.AdvancedTestGraph<GroupNormStaticReshapeRoundtripRepro>(
            hyperparamInputs: [], runtimeInputs: [Range([2L, 4L, 3L, 3L], 0.7f, -10f)]));
}
