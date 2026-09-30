using System.Linq;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests;

// Shorokoo/Shorokoo#10: GroupNorm restores its output shape via Reshape(normalized, Shape(x)),
// a reshape whose shape input is a live node, and a static-target reshape directly after it
// ([72]) is composed across that chain at ONNX prep (FastComposeContiguousReshapes). These
// modules pin that the layout loads and runs.
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

// Shorokoo/Shorokoo#12: the copy-dim spelling of the same flatten. Its shape input carries a 0
// ("copy dim 0 from the input", allowzero unset), which FastComposeContiguousReshapes declines
// to compose, so the pattern loads and runs uncomposed.
[Module]
public partial class GroupNormKeepAxesReshapeRepro
{
    public static Scalar<bit> Inline(Tensor<float32> x)   // [2, 4, 3, 3]
    {
        var y = GroupNorm.Call(Scalar(2L), Scalar(false), Scalar(1e-5f), x);
        var flat = y.Reshape([Scalar(-1L)], keepAxes: [0]); // shape input [0, -1] → [2, 36]
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
    public void GroupNormStaticStatefulAndKeepAxesReshapesLoadAndRun()
    {
        var x = Range([2L, 4L, 3L, 3L], 0.7f, -10f);
        Assert.True(AutoTest.AdvancedTestGraph<GroupNormStaticReshapeRepro>(hyperparamInputs: [], runtimeInputs: [x]));
        // One IDENTITY deeper: a STATEFUL module's WITH_STATE_DEPS wrapper lowers to an Identity
        // between the dynamic restore reshape and the static one.
        Assert.True(AutoTest.AdvancedTestGraph<StatefulGroupNormStaticReshapeRepro>(hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(AutoTest.AdvancedTestGraph<GroupNormKeepAxesReshapeRepro>(hyperparamInputs: [], runtimeInputs: [x]));
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
