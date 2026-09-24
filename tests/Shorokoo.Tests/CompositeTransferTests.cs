using System.Collections.Immutable;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The composites take <c>To</c>, <c>CopyTo</c> and <c>ToHost</c> as a tensor does, applying them to
/// what they hold; a sequence has a life of its own that a run holds while it reads it. And a
/// tensor's move into an operator attribute, which ends it.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class CompositeTransferCoverageTests
{
    private static TensorData Sample(float first) => TensorData([2L], (float[])[first, first + 1f]);

    private static float[] Floats(TensorData t) => [.. t.As<float32>().AccessMemory<float>()];

    [Fact]
    public void TestASequenceCopyIsIndependentOfTheOriginalAndAttachedToTheTarget()
    {
        using var context = new ComputeContext();
        var sequence = TensorDataSequence.Create([Sample(1f), Sample(3f)], DType.Float32);

        var copy = sequence.CopyTo(context);
        sequence.Dispose();

        Assert.Equal(2, copy.Count);
        Assert.Equal([1f, 2f], Floats(copy[0]));
        Assert.Equal([3f, 4f], Floats(copy[1]));
        Assert.All(copy, e => Assert.Contains(e, context.Tensors));
        Assert.All(copy, e => Assert.Same(HostBackend.Instance, e.AllocatingBackend));
    }

    [Fact]
    public void TestASequenceIsItselfWhereverItsElementsCanBeRead()
    {
        using var context = new ComputeContext();
        var held = TensorDataSequence.OfElements([Sample(1f), Sample(3f)], DType.Float32);
        var produced = TensorDataSequence.Create([Sample(5f)], DType.Float32);

        Assert.Same(held, held.To(context));
        Assert.Same(held, held.ToHost());
        Assert.All(held, e => Assert.Contains(e, context.Tensors));
        Assert.Same(produced, produced.To(context));
        Assert.Same(produced, produced.To(ComputeContext.Host));
        Assert.Same(produced, produced.ToHost());
        Assert.Equal([5f, 6f], Floats(produced.To(context)[0]));
    }

    [Fact]
    public void TestAStructIsItselfWhereNothingHadToBeCopiedAndCopiesItsFieldsThroughEveryNesting()
    {
        using var context = new ComputeContext();
        TensorStructFieldDef[] innerFields =
            [new TensorStructFieldDef("deep", DataStructure.Tensor, 1, DType.Float32)];
        var inner = new TensorDataStruct(
            new TensorStructDef(innerFields, "Inner"),
            new Dictionary<string, IData> { { "deep", Sample(9f) } });

        TensorStructFieldDef[] outerFields =
        [
            new TensorStructFieldDef("flat", DataStructure.Tensor, 1, DType.Float32),
            new TensorStructFieldDef("nested", DataStructure.TensorStruct, null, inner.DType),
            new TensorStructFieldDef("maybe", DataStructure.Optional, 1, DType.Float32),
        ];
        var outer = new TensorDataStruct(
            new TensorStructDef(outerFields, "Outer"),
            new Dictionary<string, IData>
            {
                { "flat", Sample(1f) }, { "nested", inner }, { "maybe", OptionalTensorData.Some(Sample(5f)) },
            });

        Assert.Same(outer, outer.To(context));
        Assert.Same(outer, outer.ToHost());
        Assert.Contains((TensorData)outer.Fields["flat"], context.Tensors);

        var copy = outer.CopyTo(context);
        var copiedInner = (TensorDataStruct)copy.Fields["nested"];
        var copiedDeep = (TensorData)copiedInner.Fields["deep"];
        var copiedMaybe = ((OptionalTensorData)copy.Fields["maybe"]).Value!;

        Assert.NotSame(outer.Fields["flat"], copy.Fields["flat"]);
        Assert.NotSame(inner.Fields["deep"], copiedDeep);
        Assert.NotSame(((OptionalTensorData)outer.Fields["maybe"]).Value, copiedMaybe);
        Assert.Equal([9f, 10f], Floats(copiedDeep));
        Assert.Equal([5f, 6f], Floats(copiedMaybe));
        Assert.Contains(copiedDeep, context.Tensors);
    }

    [Fact]
    public void TestASequenceARunProducedIsItsBackendsAndSoIsEveryElementReadOutOfIt()
    {
        using var context = new ComputeContext();
        var x = InputVector<float32>("x");
        var graph = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);

        var sequence = context.Execute(graph, TensorData([2L], (float[])[1f, 2f]))[0].ToTensorDataSequence();

        Assert.Same(DefaultBackend.Instance, ((OnnxTensorDataSequence<float32>)sequence).AllocatingBackend);
        Assert.Equal(2, sequence.Count);
        Assert.All(sequence, e => Assert.Same(DefaultBackend.Instance, e.AllocatingBackend));
        Assert.All(sequence, e => Assert.Equal(MemorySpace.Host, e.Space));
        Assert.Equal([1f, 2f], Floats(sequence[0]));
        Assert.Equal([2f, 4f], Floats(sequence[1]));
    }

    [Fact]
    public void TestASequenceOutlivesTheContextThatProducedItAndDiesWhenItIsDisposed()
    {
        var x = InputVector<float32>("x");
        var graph = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);
        var context = new ComputeContext();
        var sequence = context.Execute(graph, TensorData([2L], (float[])[1f, 2f]))[0]
            .ToTensorDataSequence();

        context.Dispose();
        Assert.Equal([2f, 4f], Floats(sequence[1]));

        sequence.Dispose();
        sequence.Dispose();

        Assert.True(sequence.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => sequence.Count);
        Assert.Throws<ObjectDisposedException>(() => sequence[0]);
        Assert.Throws<ObjectDisposedException>(() => ((IOnnxData)sequence).Value);
        Assert.Throws<ObjectDisposedException>(() => sequence.To(ComputeContext.Host));
        Assert.Throws<ObjectDisposedException>(
            () => new ComputeContext().Execute(
                new InternalComputationGraph([x], [OnnxOp.SequenceAt(x, Scalar(0L))]), sequence));
    }

    [Fact]
    public void TestASequenceARunIsReadingCannotBeDisposedUntilTheRunReturns()
    {
        using var context = new ComputeContext();
        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        var compiled = context.Compile(new InternalComputationGraph(
            [seq], [OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)]));
        var sequence = TensorDataSequence.Create([Sample(1f), Sample(3f)], DType.Float32);
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();

        var run = Task.Run(() => Floats(compiled.Run(
            new HeldSequence(sequence, reached, release) { FeedMode = SharedInputMode.Shared })[0].ToTensorData()));
        Assert.True(reached.Wait(TimeSpan.FromSeconds(10)));

        Assert.Throws<InvalidOperationException>(sequence.Dispose);
        Assert.False(sequence.IsDisposed);

        release.Set();
        Assert.Equal([1f, 2f, 3f, 4f], run.Result);
        sequence.Dispose();
        Assert.True(sequence.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => context.Lock(sequence));
    }

    /// <summary>Holds a run open once it has locked the sequence it is fed.</summary>
    private sealed class HeldSequence(
        TensorDataSequence data, ManualResetEventSlim reached, ManualResetEventSlim release)
        : TensorDataSequenceModelParam("seq", ModelParamType.InputParam, data)
    {
        internal override void Held()
        {
            reached.Set();
            release.Wait(TimeSpan.FromSeconds(30));
        }
    }

    [Fact]
    public void TestASequenceCopiedToAContextHoldsNoRuntimeValueAndCanStillBeFedBackIntoASession()
    {
        using var producer = new ComputeContext();
        var x = InputVector<float32>("x");
        var construct = new InternalComputationGraph([x], [OnnxOp.SequenceConstruct(x, x + x)]);
        var produced = producer
            .Execute(construct, TensorData([2L], (float[])[1f, 2f]))[0].ToTensorDataSequence();

        using var consumer = new ComputeContext();
        var copied = produced.CopyTo(consumer);

        Assert.False(copied is IOnnxData);

        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        var join = new InternalComputationGraph(
            [seq], [OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)]);

        Assert.Equal([1f, 2f, 2f, 4f], Floats(consumer.Execute(join, copied.Shared())[0].ToTensorData()));
        Assert.Equal([1f, 2f, 2f, 4f], Floats(consumer.Execute(join, copied.Shared())[0].ToTensorData()));
        Assert.Same(copied.ToTensorValue(), copied.ToTensorValue());

        Assert.Equal([1f, 2f], Floats(copied[0]));
        Assert.Equal([2f, 4f], Floats(copied[1]));
        Assert.All(copied, e => Assert.Contains(e, consumer.Tensors));
    }

    [Fact]
    public void TestASequenceFedAgainAfterOneOfItsElementsWasWrittenIsReadAsWritten()
    {
        using var context = new ComputeContext();
        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        var join = new InternalComputationGraph([seq], [OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)]);
        var sequence = TensorDataSequence.OfElements([Sample(1f), Sample(3f)], DType.Float32);

        Assert.Equal([1f, 2f, 3f, 4f], Floats(context.Execute(join, sequence.Shared())[0].ToTensorData()));
        sequence[0].As<float32>().AccessModifiableMemory<float>()[0] = 9f;
        Assert.Equal([9f, 2f, 3f, 4f], Floats(context.Execute(join, sequence.Shared())[0].ToTensorData()));
    }

    [Fact]
    public void TestASequenceFedAsItIsIsConsumedWithItsElementsAndOneFedSharedIsRead()
    {
        using var context = new ComputeContext();
        var join = context.Compile(Join());
        float[] Run(IData feed) => Floats(join.Execute(feed)[0].ToTensorData());
        TensorDataSequence Pair() => TensorDataSequence.OfElements([Sample(1f), Sample(3f)], DType.Float32);
        var (bare, shared, tried) = (Pair(), Pair(), Pair());
        var element = bare[0];

        Assert.Equal([1f, 2f, 3f, 4f], Run(shared.Shared()));
        Assert.Equal([1f, 2f, 3f, 4f], Run(bare));
        Assert.Equal([1f, 2f, 3f, 4f], Run(tried.TryConsume()));
        Assert.False(shared.IsDisposed || shared[0].IsDisposed);
        Assert.True(bare.IsDisposed && element.IsDisposed && tried.IsDisposed);
        Assert.Contains("consumed by a run of the graph (seq)", Assert.Throws<ObjectDisposedException>(() => bare.Count).Message);
        Assert.Contains("pass it there as .Shared()", Assert.Throws<ObjectDisposedException>(() => Floats(element)).Message);
        Assert.Contains("consumed by a run of the graph (seq)", Assert.Throws<ObjectDisposedException>(() => Run(bare)).Message);
    }

    [Fact]
    public void TestASequenceAnotherRunIsReadingIsReadWhenTriedAndRefusedWhenFedAsItIs()
    {
        using var context = new ComputeContext();
        var join = context.Compile(Join());
        var sequence = TensorDataSequence.OfElements([Sample(1f), Sample(3f)], DType.Float32);
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var holding = Task.Run(() => join.Run(
            new HeldSequence(sequence, reached, release) { FeedMode = SharedInputMode.Shared }));
        Assert.True(reached.Wait(TimeSpan.FromSeconds(10)));

        Assert.Equal([1f, 2f, 3f, 4f], Floats(join.Execute(sequence.TryConsume())[0].ToTensorData()));
        Assert.Contains("is being read by a run of the graph (seq)",
            Assert.Throws<InvalidOperationException>(() => join.Execute(sequence)).Message);
        release.Set();
        holding.Wait();
        Assert.False(sequence.IsDisposed);
    }

    [Fact]
    public void TestAListSequencesElementsFollowTheFeedRulesWithItWhereverElseTheyAreFed()
    {
        using var context = new ComputeContext();
        using var other = new ComputeContext();
        var join = context.Compile(Join());
        var both = context.Compile(ElementThenSequence());
        TensorDataSequence Pair() => TensorDataSequence.OfElements([Sample(1f), Sample(3f)], DType.Float32);
        (bool Element, bool Sequence) Survive(Func<TensorData, IData> element, Func<TensorDataSequence, IData> sequence)
        {
            var s = Pair();
            var e = s[0];
            Assert.Equal([1f, 2f, 1f, 2f, 3f, 4f], Floats(both.Execute(element(e), sequence(s))[0].ToTensorData()));
            return (!e.IsDisposed, !s.IsDisposed);
        }

        Assert.Equal((true, false), Survive(e => e.Shared(), s => s));
        Assert.Equal((true, true), Survive(e => e, s => s.Shared()));
        Assert.Equal((true, true), Survive(e => e.TryConsume(), s => s.Shared()));
        Assert.Equal((false, false), Survive(e => e, s => s));

        var readElsewhere = Pair();
        var read = readElsewhere[0];
        using (other.Lock(read))
        {
            Assert.Contains("element 0 of input 'seq'",
                Assert.Throws<InvalidOperationException>(() => join.Execute(readElsewhere)).Message);
            Assert.Throws<InvalidOperationException>(readElsewhere.Dispose);
            Assert.False(readElsewhere.IsDisposed || read.IsDisposed);
            Assert.Equal([1f, 2f, 3f, 4f], Floats(join.Execute(readElsewhere.TryConsume())[0].ToTensorData()));
        }
        Assert.True(readElsewhere.IsDisposed);
        Assert.Equal([1f, 2f], Floats(read));

        var sequenceRead = Pair();
        using (other.Lock(sequenceRead))
            Assert.Equal([1f, 2f, 3f, 4f], Floats(join.Execute(sequenceRead.TryConsume())[0].ToTensorData()));
        Assert.False(sequenceRead.IsDisposed || sequenceRead[0].IsDisposed || sequenceRead[1].IsDisposed);
    }

    [Fact]
    public void TestARunReadingAListSequenceHoldsEachOfItsElementsUntilItReturns()
    {
        using var context = new ComputeContext();
        var join = context.Compile(Join());
        var sequence = TensorDataSequence.OfElements([Sample(1f), Sample(3f)], DType.Float32);
        var element = sequence[0];
        using var reached = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var holding = Task.Run(() => Floats(join.Run(
            new HeldSequence(sequence, reached, release) { FeedMode = SharedInputMode.Shared })[0].ToTensorData()));
        Assert.True(reached.Wait(TimeSpan.FromSeconds(10)));

        Assert.Throws<InvalidOperationException>(element.Delete);
        Assert.False(element.TryDelete());
        Assert.Throws<InvalidOperationException>(() => element.MoveToAttribute());
        Assert.Throws<InvalidOperationException>(() => context.Execute(Doubling(), element));

        release.Set();
        Assert.Equal([1f, 2f, 3f, 4f], holding.Result);
        Assert.True(element.TryDelete());
    }

    private static InternalComputationGraph Join()
    {
        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        return new InternalComputationGraph([seq], [OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)]);
    }

    private static InternalComputationGraph ElementThenSequence()
    {
        var x = InputVector<float32>("x");
        var seq = InternalOp.ModuleSequenceInput(DType.Float32, null, null, "seq");
        return new InternalComputationGraph(
            [x, seq], [OnnxOp.Concat([x, OnnxOp.ConcatFromSequence(seq, axis: 0, newAxis: false)], axis: 0)]);
    }

    [Fact]
    public void TestATensorOnAContextMovesIntoAnAttributeAndDiesSayingSo()
    {
        using var context = new ComputeContext();
        var output = context.Execute(Doubling(), Sample(1f))[0].ToTensorData();
        var literal = TensorData([2L], (float[])[1f, 2f]);

        var fromOutput = output.MoveToAttribute();
        var fromLiteral = literal.MoveToAttribute();

        Assert.Equal([2f, 4f], fromOutput.Elements<float>().ToArray());
        Assert.Equal([1f, 2f], fromLiteral.Elements<float>().ToArray());
        foreach (var moved in (TensorData[])[output, literal])
        {
            Assert.True(moved.IsDisposed);
            Assert.DoesNotContain(moved, context.Tensors);
            Assert.Contains("MoveToAttribute()",
                Assert.Throws<ObjectDisposedException>(() => moved.To(context)).Message);
            Assert.Throws<ObjectDisposedException>(() => moved.MoveToAttribute());
        }
    }

    [Fact]
    public void TestAFailedCompositeCopyReleasesWhatItMadeAndLeavesTheSourceWhole()
    {
        using var target = new ComputeContext();

        var element = Sample(1f);
        var spentElement = Sample(7f);
        spentElement.Dispose();
        var sequence = TensorDataSequence.OfElements([element, spentElement], DType.Float32);
        Assert.Throws<ObjectDisposedException>(() => sequence.CopyTo(target));
        Assert.Equal([1f, 2f], Floats(element));
        Assert.Empty(target.Tensors);

        // Both assignments of the two keys, because a struct's fields are walked in hash order:
        // one of them reaches the field that survives before the one that throws.
        foreach (var spentIsA in (bool[])[false, true])
            foreach (var optional in (bool[])[false, true])
            {
                var kept = Sample(5f);
                var spent = Sample(8f);
                spent.Dispose();
                Assert.Throws<ObjectDisposedException>(() => Pair(kept, spent, spentIsA, optional).CopyTo(target));
                Assert.Equal([5f, 6f], Floats(kept));
                Assert.Empty(target.Tensors);
            }
    }

    private static TensorDataStruct Pair(TensorData kept, TensorData spent, bool spentIsA, bool optional)
    {
        var keptKind = optional ? DataStructure.Optional : DataStructure.Tensor;
        TensorStructFieldDef[] fields =
        [
            new TensorStructFieldDef("a", spentIsA ? DataStructure.Tensor : keptKind, 1, DType.Float32),
            new TensorStructFieldDef("b", spentIsA ? keptKind : DataStructure.Tensor, 1, DType.Float32),
        ];
        IData keptField = optional ? OptionalTensorData.Some(kept) : kept;
        return new TensorDataStruct(
            new TensorStructDef(fields, "Pair"),
            new Dictionary<string, IData>
            {
                { "a", spentIsA ? spent : keptField },
                { "b", spentIsA ? keptField : spent },
            });
    }

    [Fact]
    public void TestASequenceTheProviderKeptIsInItsBackendsMemoryAndCopiedToBeRead()
    {
        var card = new ComputeContextLifetimeCoverageTests.StubBackend(ComputeDevice.Cuda, 0);
        using var onCard = new ComputeContext(card);
        var unrecorded = new OnnxTensorDataSequence<float32>(new StubSequenceValue());
        var produced = new OnnxTensorDataSequence<float32>(new StubSequenceValue(), card);

        Assert.Equal(MemorySpace.Host, unrecorded.Space);
        Assert.Equal(MemorySpace.Cuda(0), produced.Space);
        Assert.Same(card, produced.AllocatingBackend);
        Assert.Same(produced, produced.To(onCard));
        Assert.NotSame(produced, produced.ToHost());
        Assert.Equal(0, produced.ToHost().Count);
    }

    /// <summary>A sequence value belonging to no runtime, which is enough to ask a sequence where
    /// it is.</summary>
    private sealed class StubSequenceValue : IShorokooTensorValue
    {
        public bool IsHostAccessible => false;
        public ShorokooOnnxValueType ValueType => ShorokooOnnxValueType.Sequence;
        public ShorokooTensorElementType ElementType => ShorokooTensorElementType.Float;
        public long[] Shape => [];
        public ReadOnlySpan<T> GetTensorDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public Span<T> GetTensorMutableDataAsSpan<T>() where T : unmanaged => throw new NotSupportedException();
        public IReadOnlyList<string> GetStringTensorData() => throw new NotSupportedException();
        public int GetValueCount() => 0;
        public IShorokooTensorValue GetValue(int index) => throw new NotSupportedException();
        public ShorokooTensorElementType GetSequenceElementType() => ShorokooTensorElementType.Float;
        public void Dispose() { }
    }

    private static InternalComputationGraph Doubling()
    {
        var a = InputVector<float32>("a");
        return new InternalComputationGraph([a], [a + a]);
    }
}
