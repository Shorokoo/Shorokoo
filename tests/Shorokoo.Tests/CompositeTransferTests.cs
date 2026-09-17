using System.Collections.Immutable;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The composites carry a context and move like a tensor does, recursing through what they own.
/// And the one place a bound tensor is refused outright: an operator's attribute.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class CompositeTransferCoverageTests
{
    private static TensorData Sample(float first) => TensorData([2L], (float[])[first, first + 1f]);

    private static float[] Floats(TensorData t) => [.. t.As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestASequenceMovesItsOwnedElementsAndCarriesTheContext()
    {
        var context = new ComputeContext();
        var sequence = TensorDataSequence.Create([Sample(1f), Sample(3f)], DType.Float32);

        var moved = sequence.TransferTo(context);

        Assert.Same(context, moved.Context);
        Assert.Equal(2, moved.Count);
        Assert.Equal([1f, 2f], Floats(moved[0]));
        Assert.Equal([3f, 4f], Floats(moved[1]));
        Assert.All(moved, e => Assert.Same(context, e.Context));
    }

    [Fact]
    public void TestASequenceCopyLeavesTheOriginalOwningItsElements()
    {
        var sequence = TensorDataSequence.Create([Sample(1f)], DType.Float32);

        var copy = sequence.CopyTo(new ComputeContext());

        Assert.True(sequence[0].OwnsMemory);
        Assert.True(copy[0].OwnsMemory);
        Assert.Equal([1f, 2f], Floats(copy[0]));
    }

    [Fact]
    public void TestASequenceGivesAccessWithoutTakingOwnership()
    {
        var sequence = TensorDataSequence.Create([Sample(1f)], DType.Float32);

        var readers = sequence.GiveAccessTo(new ComputeContext());

        Assert.False(readers[0].OwnsMemory);
        Assert.True(sequence[0].OwnsMemory);
    }

    [Fact]
    public void TestAStructMovesItsFieldsAndNestsIntoTheComposites()
    {
        var context = new ComputeContext();
        TensorStructFieldDef[] innerFields =
            [new TensorStructFieldDef("deep", DataStructure.Tensor, 1, DType.Float32)];
        var inner = new TensorDataStruct(
            new TensorStructDef(innerFields, "Inner"),
            new Dictionary<string, IData> { { "deep", Sample(9f) } });

        TensorStructFieldDef[] outerFields =
        [
            new TensorStructFieldDef("flat", DataStructure.Tensor, 1, DType.Float32),
            new TensorStructFieldDef("nested", DataStructure.TensorStruct, null, inner.DType),
        ];
        var outer = new TensorDataStruct(
            new TensorStructDef(outerFields, "Outer"),
            new Dictionary<string, IData> { { "flat", Sample(1f) }, { "nested", inner } });

        var moved = outer.TransferTo(context);

        Assert.Same(context, moved.Context);
        Assert.Same(context, ((TensorData)moved.Fields["flat"]).Context);

        var movedInner = (TensorDataStruct)moved.Fields["nested"];
        Assert.Same(context, movedInner.Context);
        Assert.Same(context, ((TensorData)movedInner.Fields["deep"]).Context);
        Assert.Equal([9f, 10f], Floats((TensorData)movedInner.Fields["deep"]));
    }

    /// <summary>
    /// A graph's description is the same description on every machine. A tensor bound to a context
    /// is bound to one backend's memory, so a graph that captured one could only be built where
    /// that context is — which is why an operator attribute refuses one.
    /// </summary>
    [Fact]
    public void TestAnOperatorAttributeRefusesATensorBoundToAContext()
    {
        var bound = Sample(1f).TransferTo(new ComputeContext());
        ImmutableList<NodeDefAttributeDef> defs = [new NodeDefAttributeDef
            { AttributeName = "value", Type = AttributeType.Tensor, DefaultValue = null }];

        var ex = Assert.Throws<ArgumentException>(() => OnnxProtoAttributes.FromCSharpVals(
            new Dictionary<string, object?> { ["value"] = bound }, defs));
        Assert.Contains("CopyTo(null)", ex.Message);

        // And the detached form the message names goes through.
        var detached = bound.CopyTo(null);
        _ = OnnxProtoAttributes.FromCSharpVals(
            new Dictionary<string, object?> { ["value"] = detached }, defs);
    }
}
