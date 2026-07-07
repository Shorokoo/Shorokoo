using System.Linq;
using Shorokoo.Modules.Layers;
using Shorokoo.Tests.Utils;
using static Shorokoo.Globals;

namespace Shorokoo.Tests;

// Regression test for Shorokoo/Shorokoo#10.
//
// GroupNorm restores its output shape via Reshape(normalized, Shape(x)) — a reshape whose
// shape input is a LIVE node. When such a reshape directly feeds a Reshape with a fully
// static constant target (e.g. [72]), ONNX Runtime's ReshapeFusion::FuseContiguousReshapes
// (present through at least ORT 1.26) fuses the pair into one node and then crashes session
// initialization while moving the live shape edge onto the fused two-input node ("Attempting
// to get index by a name which does not exist: ... for node: ..._new_reshape"). A [-1]
// flatten never triggers it (its output shape stays uninferable, so no fusion), and
// initializer-shaped reshapes are safe (initializers carry no edges) — the trigger is exactly
// (live-node shape input) + (following static-target reshape).
//
// Shorokoo works around the upstream ORT bug at ONNX prep: FastComposeContiguousReshapes
// rewires the static reshape to bypass the metadata-only producer chain, removing the
// adjacency the fusion mis-handles. This module pins that the pattern loads and runs.
[Module]
public partial class GroupNormStaticReshapeRepro
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
public class GroupNormStaticReshapeRegressionTests
{
    private static TensorData Range(long[] dims, float scale, float offset)
    {
        long total = 1; foreach (var d in dims) total *= d;
        return TensorData(DType.Float32, dims, Enumerable.Range(0, (int)total).Select(i => (object)(i * scale + offset)).ToArray());
    }

    /// <summary>
    /// Reshaping GroupNorm's (live-Shape-fed) output to a static shape must execute and survive
    /// the ONNX round-trips — i.e. the FastComposeContiguousReshapes prep pass keeps the exported
    /// model clear of the ORT ReshapeFusion crash described in Shorokoo/Shorokoo#10.
    /// </summary>
    [Fact]
    public void GroupNormOutputStaticReshapeLoadsAndRuns()
        => Assert.True(AutoTest.AdvancedTestGraph<GroupNormStaticReshapeRepro>(
            hyperparamInputs: [], runtimeInputs: [Range([2L, 4L, 3L, 3L], 0.7f, -10f)]));

    /// <summary>
    /// Same ORT ReshapeFusion crash, one Identity deeper: a STATEFUL module's output is
    /// WITH_STATE_DEPS-wrapped, which the inference lowering turns into an IDENTITY sitting
    /// between GroupNorm's dynamic restore reshape and the user's static reshape. ORT's
    /// EliminateIdentity runs in the same L1 loop as ReshapeFusion, so the crashing adjacency
    /// re-forms inside ORT unless FastComposeContiguousReshapes walks through same-scope
    /// identities too.
    /// </summary>
    [Fact]
    public void StatefulGroupNormOutputStaticReshapeLoadsAndRuns()
        => Assert.True(AutoTest.AdvancedTestGraph<StatefulGroupNormStaticReshapeRepro>(
            hyperparamInputs: [], runtimeInputs: [Range([2L, 4L, 3L, 3L], 0.7f, -10f)]));
}

// Stateful GroupNorm: the StateUpdate forces the module's output to be wrapped in
// WITH_STATE_DEPS — lowered to an IDENTITY between the dynamic reshape and its consumers.
[Module]
public partial class _StatefulGroupNormInner
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [2, 4, 3, 3]
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var counter = Shorokoo.Tests.Modules.InitRunningMean.Init(x.ShapeTensor());
        Globals.StateUpdate(counter, counter + Scalar(1f));
        return y;
    }
}

[Module]
public partial class StatefulGroupNormStaticReshapeRepro
{
    public static Scalar<bit> Inline(Tensor<float32> x)
    {
        var y = _StatefulGroupNormInner.Call(x);
        var flat = y.Reshape([Scalar(72L)]);              // STATIC target — the trigger
        return SelfCheck.Nan(flat) < Scalar(1f);          // finite output => true; self-checking
    }
}
