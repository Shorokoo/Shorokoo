using System.Reflection;
using Shorokoo.Core.Lowering;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Tests;

/// <summary>
/// Autograd ENGINE path-checking semantics in <c>FastProcessAutoGrad</c>, the AD003
/// attribute-envelope guards on the gradient implementations, and the operator-lowering fallback
/// the engine reaches for when an op has no <c>[AutoDiff]</c> rule. Each module scenario drives a
/// module from <c>Modules/AutoGradEngineModules.cs</c> through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/>; the AD003
/// <c>AutoDiffNotSupportedException</c> surfaces from the AUTO_GRAD lowering during
/// concretization, i.e. out of the <c>AdvancedTestGraph</c> call itself.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradEngineTests
{
    // An unsupported op on the loss→param path must throw AD003 at lowering — never silently cut
    // the chain and hand the parameter a zeros gradient.
    [Fact]
    public void TestAutoGradEngineThrowsAD003()
    {
        void AssertAD003<TModule>(long length, string messageFragment)
        {
            var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
                AutoTest.AdvancedTestGraph<TModule>(
                    hyperparamInputs: [],
                    runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [length])]));
            Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
            Assert.Contains(messageFragment, ex.Message);
        }

        AssertAD003<AutoGradEngineLoopOnParamPathCheck>(4L, "dynamic loops");
        AssertAD003<AutoGradEnginePadReflectThrowCheck>(4L, "constant");
        AssertAD003<AutoGradEngineScatterMulThrowCheck>(4L, "reduction");
    }

    // An unregistered op with NO parameter behind it is a legitimate gradient leaf (chain cut
    // there); Slice with steps != 1 scatters onto the exact flat offsets the forward selected.
    [Fact]
    public void TestAutoGradEngineDifferentiates()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineRandomLeafCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineSliceStepsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [6L])]));
    }

    // An op with no [AutoDiff] rule but a registered OpLowering is differentiated by a reverse
    // walk over the primitives that lowering emits; the walk never runs a second lowering, so the
    // two ways it can fail to reach a rule both refuse rather than recurse.
    [Fact]
    public void TestAutoGradEngineDifferentiatesThroughAnOperatorLowering()
    {
        Variable x = InputTensor<float32>("x", rank: 1);
        Variable dy = InputTensor<float32>("dy", rank: 1);
        var attrs = OnnxCSharpAttributes.FromCSharpVals(
            new(), Definitions.NodeDefinitions[SOFTSIGN].AttributeDefs);
        var gradOps = AutoDiffs.GetGradientOps();

        Assert.False(gradOps.ContainsKey(SOFTSIGN));
        Assert.True(OpLoweringRegistry.TryGet(SOFTSIGN, out var softsign));

        var grads = LoweredGradient.Compute(softsign, [x], [dy], attrs, gradOps);
        Assert.Single(grads);
        Assert.NotNull(grads[0]);

        Assert.Equal(ErrorCodes.AD003, Assert.Throws<AutoDiffNotSupportedException>(() =>
            LoweredGradient.Compute(softsign, [x], [dy], attrs,
                gradOps.Where(kv => kv.Key != ABS).ToDictionary())).ErrorCode);

        var selfBuilding = new OpLowering(SOFTSIGN, typeof(AutoGradEngineTests).GetMethod(
            nameof(BuildsItsOwnOpCode), BindingFlags.NonPublic | BindingFlags.Static)!);
        Assert.Throws<InvalidOperationException>(() =>
            LoweredGradient.Compute(selfBuilding, [x], [dy], attrs, gradOps));
    }

    private static Variable?[] BuildsItsOwnOpCode<T>(Tensor<T> x) where T : IVarType
        => [OnnxOp.Softsign(x)];
}
