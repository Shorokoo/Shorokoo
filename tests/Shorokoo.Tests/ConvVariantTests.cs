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
    public void ConvVariant_LowersAndMatchesStandardConv()
    {
        var x = TensorData(DType.Float32, [1L, 3L, 5L, 5L],
            Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray());

        Assert.True(AutoTest.AdvancedTestGraph<ConvVariantMatchesStandard>(
            hyperparamInputs: [], runtimeInputs: [x]));
    }

    [Fact]
    public void ConvVariant_ResolvesShapeDependentAttrs()
    {
        var x = TensorData(DType.Float32, [1L, 3L, 5L, 5L],
            Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray());

        Assert.True(AutoTest.AdvancedTestGraph<ConvVariantShapeDependentAttrs>(
            hyperparamInputs: [], runtimeInputs: [x]));
    }

    [Fact]
    public void ConvVariant_ResolvesLoopShapeAndIndexDependentAttrs()
    {
        var x = TensorData(DType.Float32, [1L, 3L, 5L, 5L],
            Enumerable.Range(0, 75).Select(i => (object)(float)i).ToArray());

        Assert.True(AutoTest.AdvancedTestGraph<ConvVariantLoopShapeAndIndexAttrs>(
            hyperparamInputs: [], runtimeInputs: [x]));
    }
}
