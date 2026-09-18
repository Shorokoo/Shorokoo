using System.Collections.Immutable;
using Shorokoo.Core.Inference.Abstractions;
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

    [Fact]
    public void TestASequenceARunProducedCarriesItsContextDownToItsElements()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        var graph = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);

        var sequence = context.Execute(graph, TensorData([2L], (float[])[1f, 2f]))[0].ToTensorDataSequence();

        Assert.Same(context, sequence.Context);
        Assert.Equal(2, sequence.Count);
        Assert.All(sequence, e => Assert.Same(context, e.Context));
        Assert.All(sequence, e => Assert.Equal(MemorySpace.Host, e.Space));
        Assert.Equal([1f, 2f], Floats(sequence[0]));
        Assert.Equal([2f, 4f], Floats(sequence[1]));
    }

    [Fact]
    public void TestASequenceIsReleasedWithTheContextThatProducedIt()
    {
        var x = InputVector<float32>("x");
        var graph = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);
        var context = new ComputeContext();
        var sequence = context.Execute(graph, TensorData([2L], (float[])[1f, 2f]))[0]
            .ToTensorDataSequence();

        Assert.Equal(2, sequence.Count);

        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() => sequence.Count);
        Assert.Throws<ObjectDisposedException>(() => sequence[0]);
        Assert.Throws<ObjectDisposedException>(() => ((IOnnxData)sequence).Value);
        Assert.Throws<ObjectDisposedException>(
            () => new ComputeContext().Execute(
                new InternalComputationGraph([x], [OnnxOp.SequenceAt(x, Scalar(0L))]), sequence));
    }

    [Fact]
    public void TestADetachingContextHandsASequenceOutBelongingToNobody()
    {
        var x = InputVector<float32>("x");
        var graph = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);

        TensorDataSequence sequence;
        using (var context = new ComputeContext(detachesOutputs: true))
            sequence = context.Execute(graph, TensorData([2L], (float[])[1f, 2f]))[0].ToTensorDataSequence();

        Assert.Null(sequence.Context);
        Assert.All(sequence, e => Assert.Null(e.Context));
        Assert.Equal([2f, 4f], Floats(sequence[1]));
    }

    [Fact]
    public void TestASequenceThatHasBeenTransferredCanBeFedBackIntoASession()
    {
        using var producer = new ComputeContext();
        var x = InputVector<float32>("x");
        var construct = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);
        var produced = producer
            .Execute(construct, TensorData([2L], (float[])[1f, 2f]))[0].ToTensorDataSequence();

        using var consumer = new ComputeContext();
        var moved = produced.TransferTo(consumer);

        // There is nothing of a runtime's left in it to hand over: the transfer holds the tensors
        // themselves, which is exactly why this case needed answering.
        Assert.False(moved is IOnnxData);

        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        var join = new InternalComputationGraph(
            [seq], [OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)]);

        Assert.Equal([1f, 2f, 2f, 4f], Floats(consumer.Execute(join, moved)[0].ToTensorData()));

        // Twice, because what it built is kept rather than rebuilt: a run must not be handed a
        // sequence whose elements an earlier one has already had released under it.
        Assert.Equal([1f, 2f, 2f, 4f], Floats(consumer.Execute(join, moved)[0].ToTensorData()));
        Assert.Same(moved.ToTensorValue(), moved.ToTensorValue());

        // And the elements it was moved are untouched by any of it — the sequence value holds
        // copies, so the run cannot reach the storage each element owns.
        Assert.Equal([1f, 2f], Floats(moved[0]));
        Assert.Equal([2f, 4f], Floats(moved[1]));
        Assert.All(moved, e => Assert.Same(consumer, e.Context));
    }

    [Fact]
    public void TestAStructMovesAPresentOptionalFieldWithTheRest()
    {
        var context = new ComputeContext();
        TensorStructFieldDef[] fields =
        [
            new TensorStructFieldDef("plain", DataStructure.Tensor, 1, DType.Float32),
            new TensorStructFieldDef("maybe", DataStructure.Optional, 1, DType.Float32),
        ];
        var subject = new TensorDataStruct(
            new TensorStructDef(fields, "WithOptional"),
            new Dictionary<string, IData>
            {
                { "plain", Sample(1f) },
                { "maybe", OptionalTensorData.Some(Sample(5f)) },
            });

        var moved = subject.TransferTo(context);

        var optional = (OptionalTensorData)moved.Fields["maybe"];
        Assert.Same(context, optional.Value!.Context);
        Assert.Equal([5f, 6f], Floats(optional.Value));
    }

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

    [Fact]
    public void TestATensorAlreadyCapturedAsAnAttributeRefusesToBeBoundToAContext()
    {
        var literal = TensorData([2L], (float[])[1f, 2f]);
        _ = OnnxOp.Constant(value: literal);
        using var context = new ComputeContext();

        Assert.Throws<InvalidOperationException>(() => literal.TransferTo(context));
        Assert.Throws<InvalidOperationException>(() => literal.GiveAccessTo(context));
        Assert.Null(literal.Context);
        Assert.True(literal.OwnsMemory);
        Assert.Equal([1f, 2f], Floats(literal));
    }

    [Fact]
    public void TestAFailedCompositeTransferLeavesTheSourceIntact()
    {
        using var target = new ComputeContext();

        var good = Sample(1f);
        var doomed = Sample(7f);
        doomed.Dispose();
        var sequence = TensorDataSequence.OfElements([good, doomed], DType.Float32);
        Assert.ThrowsAny<Exception>(() => sequence.TransferTo(target));
        Assert.Equal([1f, 2f], Floats(sequence[0]));
        Assert.True(sequence[0].OwnsMemory);

        TensorStructFieldDef[] fields =
        [
            new TensorStructFieldDef("a", DataStructure.Tensor, 1, DType.Float32),
            new TensorStructFieldDef("b", DataStructure.Tensor, 1, DType.Float32),
        ];
        var first = Sample(5f);
        var second = Sample(8f);
        second.Dispose();
        var composite = new TensorDataStruct(
            new TensorStructDef(fields, "Pair"),
            new Dictionary<string, IData> { { "a", first }, { "b", second } });

        Assert.ThrowsAny<Exception>(() => composite.TransferTo(target));
        Assert.True(first.OwnsMemory);
        Assert.Null(first.Context);
    }

    [Fact]
    public void TestCopyingACompositeCopiesTheElementsItOnlyHasAccessTo()
    {
        var source = new ComputeContext();
        var sequence = TensorDataSequence.Create([Sample(1f)], DType.Float32).TransferTo(source);
        TensorStructFieldDef[] fields =
            [new TensorStructFieldDef("f", DataStructure.Tensor, 1, DType.Float32)];
        var struc = new TensorDataStruct(
            new TensorStructDef(fields, "S"),
            new Dictionary<string, IData> { { "f", Sample(5f) } }).TransferTo(source);

        // A reader owns none of its elements, which is the case the ownership gate on the transfer
        // operations is for -- and the case a copy must not be gated by, since a copy is exactly
        // what a caller holding no ownership has to reach for.
        var sequenceReader = sequence.GiveAccessTo(source);
        var structReader = struc.GiveAccessTo(source);

        var keptSequence = sequenceReader.CopyTo(null);
        var keptStruct = structReader.CopyTo(null);

        Assert.NotSame(sequenceReader[0], keptSequence[0]);
        Assert.True(keptSequence[0].OwnsMemory);
        Assert.Null(keptSequence[0].Context);
        Assert.NotSame(StructField(structReader), StructField(keptStruct));
        Assert.True(StructField(keptStruct).OwnsMemory);

        // The point of the copy: it outlives the context the elements were read from.
        source.Dispose();
        Assert.Equal([1f, 2f], Floats(keptSequence[0]));
        Assert.Equal([5f, 6f], Floats(StructField(keptStruct)));
    }

    private static TensorData StructField(TensorDataStruct s) => (TensorData)s.Fields["f"];
}
