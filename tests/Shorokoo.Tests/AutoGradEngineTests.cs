namespace Shorokoo.Tests;

/// <summary>
/// Phase 4 AD-B2 tests for the autograd ENGINE path-checking semantics in
/// <c>FastProcessAutoGrad</c> and the AD003 attribute-envelope guards added to the
/// gradient implementations. Follows the same one-liner pattern as
/// <see cref="AutoGradStructuralTests"/>: each scenario drives
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/> against a module from
/// <c>Modules/AutoGradEngineModules.cs</c>. For the throwing scenarios the
/// <c>AutoDiffNotSupportedException</c> (AD003) surfaces from the AUTO_GRAD lowering
/// during concretization — i.e. out of the <c>AdvancedTestGraph</c> call itself,
/// before any execution backend runs — so <c>Assert.Throws</c> wraps the whole call.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradEngineTests
{
    /// <summary>
    /// An unregistered op (a dynamic LOOP) on the loss→param path must throw AD003
    /// at lowering — never silently cut the chain and hand the parameter a zeros
    /// gradient (a silently frozen parameter is the worst failure mode for a
    /// training framework).
    /// </summary>
    [Fact]
    public void TestAutoGradEngineUnregisteredOpOnParamPathThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradEngineLoopOnParamPathCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("dynamic loops", ex.Message);
    }

    /// <summary>
    /// An unregistered op (RandomNormal) feeding the loss path with NO parameter
    /// behind it is a legitimate gradient leaf: the chain is cut there and
    /// differentiation succeeds with the exact gradient.
    /// </summary>
    [Fact]
    public void TestAutoGradEngineNonParamLeafStillDifferentiates()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineRandomLeafCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
    }

    /// <summary>
    /// Slice with steps != 1 used to mis-place gradients silently (the Pad-based
    /// adjoint assumes contiguous step-1 slices); the steps path now scatters onto
    /// the exact flat offsets the forward selected.
    /// </summary>
    [Fact]
    public void TestAutoGradEngineSliceStepsGradient()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineSliceStepsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [6L])]));
    }

    /// <summary>
    /// AD003 guards for unsupported attribute combinations: Pad mode='reflect' and
    /// ScatterND reduction='mul' gradients must fail loudly at lowering instead of
    /// emitting a silently-wrong gradient subgraph.
    /// </summary>
    [Fact]
    public void TestAutoGradEnginePadReflectThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradEnginePadReflectThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("constant", ex.Message);
    }

    [Fact]
    public void TestAutoGradEngineScatterMulReductionThrows()
    {
        var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
            AutoTest.AdvancedTestGraph<AutoGradEngineScatterMulThrowCheck>(
                hyperparamInputs: [],
                runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
        Assert.Contains("reduction", ex.Message);
    }
}
