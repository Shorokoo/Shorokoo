namespace Shorokoo.Tests;

/// <summary>
/// Autograd ENGINE path-checking semantics in <c>FastProcessAutoGrad</c> and the AD003
/// attribute-envelope guards on the gradient implementations. Each scenario drives a module from
/// <c>Modules/AutoGradEngineModules.cs</c> through <see cref="AutoTest.AdvancedTestGraph{TModule}"/>;
/// the AD003 <c>AutoDiffNotSupportedException</c> surfaces from the AUTO_GRAD lowering during
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
}
