using Shorokoo.Core.Nodes.Processors.Fast;
using System.Linq;

namespace Shorokoo.Tests;

/// <summary>
/// Tests for the attribute-tensorization infrastructure (SHRK_CONV → ONNX Conv lowering via
/// <c>FastLowerAttributeTensorOps</c>). The self-checking module compares the variant Conv against
/// the standard Conv with identical geometry; driving it through <see cref="AutoTest.AdvancedTestGraph{TModule}"/>
/// exercises the full lower → roundtrip (ONNX/CS/QEE) pipeline, which would fail if the variant
/// were not correctly lowered to a standard Conv with the geometry resolved to static attributes.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class ConvVariantTests
{
    [Fact]
    public void ConvVariant_LowersAndResolvesStandardShapeAndLoopIndexAttrs()
    {
        var x = TensorData(DType.Float32, [1L, 3L, 5L, 5L],
            Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray());

        Assert.True(AutoTest.AdvancedTestGraph<ConvVariantMatchesStandard>(hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvVariantShapeDependentAttrs>(hyperparamInputs: [], runtimeInputs: [x]));
        Assert.True(AutoTest.AdvancedTestGraph<ConvVariantLoopShapeAndIndexAttrs>(hyperparamInputs: [], runtimeInputs: [x]));
    }

    /// <summary>An AUTO_GRAD in the loop body puts a member of <c>InternalOpCodes.ModuleStageOps</c>
    /// inside a constant-trip loop at the first FastSimplify. Unrolling it is still required: gating
    /// the unroll on that whole set instead of the four Stage-F parameter op-codes leaves the loop
    /// rolled, and FastLowerAttributeTensorOps then bakes iteration 0's dilation into all three.</summary>
    [Fact]
    public void TestALoopWithAutoGradInItsBodyIsStillUnrolledSoItsConvGeometryStaysPerIteration()
        => Assert.True(AutoTest.AdvancedTestGraph<ConvVariantLoopWithAutoGradInBody>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [1L, 3L, 5L, 5L],
                    Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray())]));

    /// <summary>A loop the native unroll cannot flatten — its trip count is a graph input — leaves
    /// FastLowerAttributeTensorOps facing index-dependent SHRK_CONV geometry inside a rolled loop.
    /// A static attribute cannot vary per iteration, so the build must refuse it rather than
    /// resolve one value by the QEE/ORT fallback and bake it into every iteration (which returned
    /// 3x the dilation-1 conv instead of the d=1,2,3 sum, silently).</summary>
    [Fact]
    public void TestVariantOpGeometryInARolledLoopIsRefusedNotBakedFromOneIteration()
    {
        var ex = Assert.Throws<FastPipelineUnsupportedException>(
            () => AutoTest.AdvancedTestGraph<ConvVariantDynamicTripLoopGeometry>(
                hyperparamInputs: [],
                runtimeInputs: [
                    TensorData(DType.Float32, [1L, 3L, 5L, 5L],
                        Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray()),
                    TensorData(DType.Int64, [], 3L)]));
        Assert.Contains("varies per loop iteration", ex.Message);
    }
}
