using System.Linq;
using Shorokoo.Core.Nodes.Processors.Fast;

namespace Shorokoo.Tests;

/// <summary>
/// Tests for the attribute-tensorization infrastructure (SHRK_CONV → ONNX Conv lowering via
/// <c>FastLowerAttributeTensorOps</c>). Geometry that resolves to one value per Conv node is
/// lowered and checked against a standard Conv with identical geometry, driven through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/> so the whole lower → roundtrip (ONNX/CS/QEE)
/// pipeline runs; geometry that would differ per iteration of a loop the unroll left rolled cannot
/// become a static attribute and is refused instead.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ConvVariantTests
{
    static TensorData Image => TensorData(DType.Float32, [1L, 3L, 5L, 5L],
        Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray());

    static void Lowers<TModule>(params TensorData[] inputs)
        => Assert.True(AutoTest.AdvancedTestGraph<TModule>(hyperparamInputs: [], runtimeInputs: inputs));

    static void Refused<TModule>(params TensorData[] inputs)
    {
        var ex = Assert.Throws<FastPipelineUnsupportedException>(
            () => AutoTest.AdvancedTestGraph<TModule>(hyperparamInputs: [], runtimeInputs: inputs));
        Assert.Contains("FastLowerAttributeTensorOps", ex.Message);
        Assert.Contains("'pads'", ex.Message);
    }

    static TensorData[] Rolled => [Image, TensorData(DType.Int64, [], 3L)];

    [Fact]
    public void ConvVariant_LowersAndResolvesStandardShapeAndLoopIndexAttrs()
    {
        Lowers<ConvVariantMatchesStandard>(Image);
        Lowers<ConvVariantShapeDependentAttrs>(Image);
        Lowers<ConvVariantLoopShapeAndIndexAttrs>(Image);
    }

    /// <summary>An AUTO_GRAD in the loop body puts a member of <c>InternalOpCodes.ModuleStageOps</c>
    /// inside a constant-trip loop at the first FastSimplify. Unrolling it is still required: gating
    /// the unroll on that whole set instead of the four Stage-F parameter op-codes leaves the loop
    /// rolled, and FastLowerAttributeTensorOps then refuses its per-iteration dilation outright.</summary>
    [Fact]
    public void TestALoopWithAutoGradInItsBodyIsStillUnrolledSoItsConvGeometryStaysPerIteration()
        => Lowers<ConvVariantLoopWithAutoGradInBody>(Image);

    /// <summary>Geometry the unroll cannot make static — it differs on each pass of a loop left
    /// rolled by a runtime trip count — is a build error rather than one iteration's value baked
    /// into all of them, whether it comes off the index directly, through a nested loop's carry, or
    /// out of a nested loop that ran zero times and returned that carry's initializer.</summary>
    [Fact]
    public void TestGeometryThatVariesPerIterationOfARolledLoopIsRefused()
    {
        Refused<ConvVariantDynamicTripLoopGeometry>(Rolled);
        Refused<ConvVariantNestedRolledLoopGeometry>(Rolled);
        Refused<ConvVariantNestedZeroTripCarryGeometry>(Rolled);
    }

    /// <summary>The refusal is about geometry that varies, not about loops. A rolled loop with
    /// literal geometry, a conv reading a rolled loop's result, and a conv reading a nested loop's
    /// carry whose sibling carry is the only one tracking the outer index all execute with one
    /// geometry, so all three must still lower.</summary>
    [Fact]
    public void TestGeometryThatDoesNotVaryPerIterationStillLowersAroundARolledLoop()
    {
        Lowers<ConvVariantDynamicTripLoopInvariantGeometry>(Rolled);
        Lowers<ConvVariantGeometryFromARolledLoopResult>(Rolled);
        Lowers<ConvVariantNestedRolledLoopSiblingCarryGeometry>(Rolled);
    }
}
