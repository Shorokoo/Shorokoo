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

// Companion pin for Shorokoo/Shorokoo#12: the copy-dim spelling of the same flatten. The
// shape input carries a 0 ("copy dim 0 from the input", allowzero unset), so
// FastComposeContiguousReshapes deliberately declines to compose it — this module pins that
// ORT's ReshapeFusion also declines such targets and the uncomposed pattern loads and runs.
// A future ORT that extends the fusion to 0-targets would surface here.
[Module]
public partial class GroupNormKeepDimsReshapeRepro
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [2, 4, 3, 3]
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var flat = y.Reshape([Scalar(-1L)], keepDims: [0]); // shape input [0, -1] → [2, 36]
        return SelfCheck.Nan(flat) < Scalar(1f);            // finite output => true; self-checking
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

    [Fact]
    public void GroupNormStaticStatefulAndKeepDimsReshapesLoadAndRun()
    {
        var x = Range([2L, 4L, 3L, 3L], 0.7f, -10f);
        Assert.True(AutoTest.AdvancedTestGraph<GroupNormStaticReshapeRepro>(hyperparamInputs: [], runtimeInputs: [x]));
        // One IDENTITY deeper: a STATEFUL module's WITH_STATE_DEPS wrapper lowers to an Identity
        // between the dynamic restore reshape and the static one, which ORT's EliminateIdentity
        // re-fuses unless FastComposeContiguousReshapes walks through same-scope identities.
        Assert.True(AutoTest.AdvancedTestGraph<StatefulGroupNormStaticReshapeRepro>(hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(AutoTest.AdvancedTestGraph<GroupNormKeepDimsReshapeRepro>(hyperparamInputs: [], runtimeInputs: [x]));
    }
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
