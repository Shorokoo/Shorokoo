using System.Collections.Immutable;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Lowering.KernelWorkarounds;
using Shorokoo.Jax.Cpu;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Tests.OnnxProtoBuilders;

namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class KernelWorkaroundPassTests
{
    private static readonly KernelWorkaroundSet NegSet = new("neg", [new NegAsSubtraction()]);
    private static readonly KernelWorkaroundSet AbsSet = new("abs", [new AbsAsIf()]);
    private static readonly KernelWorkaroundSet TopKSet = new("topk", [new TopKThroughIdentities()]);
    private static readonly KernelWorkaroundSet UnarySet = new("unary", [new UnaryThroughIdentity()]);

    [Fact]
    public void TestAGraphNoWorkaroundAppliesToBuildsTheSameModel()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var d = InputTensor<float64>("d", rank: 1);
        Assert.Equal(Bytes(Session(Graph(x, OnnxOp.Mul(x, x)), null)), Bytes(Session(Graph(x, OnnxOp.Mul(x, x)), NegSet)));
        Assert.Equal(Bytes(Session(Graph(d, OnnxOp.Neg(d)), null)), Bytes(Session(Graph(d, OnnxOp.Neg(d)), NegSet)));
    }

    [Fact]
    public void TestAWorkaroundRewritesTheSessionModelInsideIfLoopAndFunctionBodies()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var two = TensorData(DType.Float32, [2L], 2f, -3f);
        AssertRewritten(NegSet, MUL, SUB, true, Graph(x, OnnxOp.Mul(OnnxOp.Neg(x), x)), two);
        AssertRewritten(NegSet, "If", SUB, true, Graph(x, Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(0f)), OnnxOp.Neg(x), x)), two);
        AssertRewritten(NegSet, "Loop", SUB, true, Concrete(KernelWorkaroundNegInLoop.ComputationGraph, TensorData(DType.Float32, [2L], 2f, 3f)), TensorData(DType.Float32, [2L], 2f, 3f));
        AssertRewritten(NegSet, "Functions", SUB, true, Import(NegInAFunction()), TensorData(DType.Float32, [3L], 1f, -2f, 4f));
        AssertRewritten(AbsSet, ADD, "If", true, Graph(x, OnnxOp.Add(OnnxOp.Abs(x), x)), two);
        AssertRewritten(TopKSet, ADD, IDENTITY, false, TopKGraph(), TensorData(DType.Float32, [4L], 3f, 1f, 4f, 2f));
        AssertRewritten(UnarySet, ADD, IDENTITY, false, Graph(x, OnnxOp.Add(OnnxOp.Neg(x), OnnxOp.Abs(x))), two);
    }

    [Fact]
    public void TestEveryValueOfTheModelBuiltWithoutWorkaroundsKeepsItsName()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var g = Graph(x, OnnxOp.Add(OnnxOp.Mul(OnnxOp.Neg(x), x), OnnxOp.Neg(OnnxOp.Abs(x))));
        Assert.Empty(Untouched(g, NegSet).Except(Signatures(Session(g, NegSet))));
        Assert.Empty(Untouched(g, AbsSet).Except(Signatures(Session(g, AbsSet))));
    }

    [Fact]
    public void TestOnlyTheSessionModelCarriesTheOnnxRuntimeWorkaround()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var g = Graph(x, OnnxOp.AveragePool(x, null, null, true, null, [2L], [2L, 0L], null));
        string[] asWritten = [.. g.Nodes.Select(n => n.OpCode)];
        Assert.Equal([0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.5f],
            Pooled(g));
        Assert.Equal(asWritten, g.Nodes.Select(n => n.OpCode));
        Assert.Contains(AllNodes(Session(g, KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == PAD);
        Assert.DoesNotContain(AllNodes(FastOnnxModelBuilder.BuildOnnxModel(g)), n => n.OpType == PAD);
        Assert.DoesNotContain(AllNodes(FastOnnxModelBuilder.BuildInternalOnnxModel(g, applyExecutionLowerings: false,
            emitInputsAsNodes: true, workarounds: KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == PAD);
        Assert.DoesNotContain(AllNodes(FastOnnxModelBuilder.BuildInternalOnnxModel(g,
            workarounds: KernelWorkaroundRegistry.OnnxRuntime)), n => n.OpType == PAD);
    }

    [Fact]
    public void TestAnImportedModelGetsTheOnnxRuntimeWorkaroundWhenItsSessionIsBuilt()
    {
        var g = Import(PaddedAsFarAsTheKernel());
        Assert.Equal([0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.5f],
            Pooled(g));
        Assert.DoesNotContain(g.Nodes, n => n.OpCode == PAD);
    }

    [Fact]
    public void TestOnlyTheOnnxRuntimeBackendsNameAWorkaroundSet()
    {
        Assert.Same(KernelWorkaroundRegistry.OnnxRuntime, KernelWorkaroundRegistry.For(DefaultBackend.Instance.KernelWorkaroundSet));
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)new TorchCpuBackend()).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)new JaxCpuBackend()).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For(((IShorokooBackend)HostBackend.Instance).KernelWorkaroundSet).IsEmpty);
        Assert.True(KernelWorkaroundRegistry.For("none of them").IsEmpty);
    }

    [Fact]
    public void TestBuildersBuildThePlainOperator()
    {
        var x = InputTensor<float32>("x", rank: 3);
        var y = InputTensor<float32>("y", rank: 4);
        var i = InputTensor<int64>("i", rank: 2);
        Assert.True(Plain(OnnxOp.AveragePool(x, null, null, true, null, [2L], [2L, 0L], null), AVERAGE_POOL, x));
        Assert.True(Plain(OnnxOp.MaxPool(x, AutoPad.SameUpper, null, [2L], [2L], null, null, null), MAX_POOL, x));
        Assert.True(Plain(OnnxOp.MaxPoolWithIndices(x, null, null, null, [2L], [2L, 2L], null, null).indices, MAX_POOL, x));
        Assert.True(Plain(OnnxOp.LpPool(x, AutoPad.SameLower, null, null, [2L], 2L, null, [3L]), LP_POOL, x));
        Assert.True(Plain(OnnxOp.Col2Im(x, Vector(5L), Vector(2L), [1L], [1L, 1L], [1L]), COL2IM, x));
        Assert.True(Plain(OnnxOp.Resize(y, Vector(0f, 0f, 0f, 0.5f, 1f, 1f, 1f, 1.5f), Vector(1f, 1f, 1f, 1f), null, null, null,
            CoordinateTransformationMode.Tf_crop_and_resize, null, null, -1f, null, ResizeMode.Linear, null), RESIZE, y));
        Assert.True(Plain((Variable)i.Reduce(ReduceKind.Max, Vector(1L)), REDUCE_MAX, i));
        Assert.True(Plain((Variable)i.Reduce(ReduceKind.Min), REDUCE_MIN, i));
    }

    private static InternalComputationGraph Graph(Variable input, Variable output) => new([input], [output]);

    private static InternalComputationGraph Concrete(ComputationGraph module, TensorData x)
        => module.ToInternal().ToConcreteArchitecture([x]).ToConcreteModel();

    private static InternalComputationGraph TopKGraph()
    {
        var x = InputTensor<float32>("x", rank: 1);
        var (values, indices) = OnnxOp.TopK(x, Vector(2L));
        return new([x], [OnnxOp.Add(values, values), indices]);
    }

    private static ModelProto Session(InternalComputationGraph g, KernelWorkaroundSet? set)
        => FastOnnxModelBuilder.BuildInternalOnnxModel(g, prepForOnnx: true, workarounds: set);

    private static void AssertRewritten(KernelWorkaroundSet set, string within, string introduced, bool removes, InternalComputationGraph g, params TensorData[] x)
    {
        string[] asWritten = [.. g.Nodes.Select(n => n.OpCode)];
        var plain = Session(g, null);
        var rewritten = Session(g, set);
        Assert.Contains(AllNodes(plain), n => n.OpType == within || n.Domain == within);
        Assert.True(AllNodes(rewritten).Count(n => n.OpType == introduced) > AllNodes(plain).Count(n => n.OpType == introduced));
        Assert.Equal(removes, !AllNodes(rewritten).Any(n => set.OpCodes.Contains(n.OpType)));
        Assert.Equal(Run(g, plain, x), Run(g, rewritten, x));
        Assert.Equal(asWritten, g.Nodes.Select(n => n.OpCode));
    }

    private static byte[][] Run(InternalComputationGraph g, ModelProto model, TensorData[] x)
        => [.. ComputeContext.Default.ExecuteModel(g, model, [.. x.Select(t => (IData)t.Shared())])
            .Select(p => p.ToTensorData().AccessRawMemory().ToArray())];

    private static float[] Pooled(InternalComputationGraph g)
        => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(new ComputeContext()
            .Execute(g, TensorData(DType.Float32, [1L, 1L, 5L], 1f, 2f, 3f, 4f, 5f).Shared())[0].ToTensorData().AccessRawMemory()).ToArray();

    private static byte[] Bytes(ModelProto model)
    {
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static IEnumerable<NodeProto> AllNodes(ModelProto model)
        => model.Graph.Nodes.Concat(model.Functions.SelectMany(f => f.Nodes)).SelectMany(Within);

    private static IEnumerable<NodeProto> Within(NodeProto node)
        => node.Attributes.Where(a => a.G is not null).SelectMany(a => a.G!.Nodes.SelectMany(Within)).Prepend(node);

    private static IEnumerable<string> Signatures(ModelProto model)
        => model.Graph.Nodes.Select(n => $"{n.OpType}({string.Join(",", n.Inputs)})->{string.Join(",", n.Outputs)}")
            .Concat(model.Graph.Outputs.Select(o => o.Name));

    private static IEnumerable<string> Untouched(InternalComputationGraph g, KernelWorkaroundSet set)
        => Signatures(Session(g, null)).Where(s => !set.OpCodes.Any(op => s.StartsWith(op + "(", StringComparison.Ordinal)));

    private static bool Plain(Variable output, string opCode, Variable input)
        => output.OwningNode is { } node && node.OpCode == opCode && node.Inputs[0] == input;

    private static AttributeProto Ints(string name, params long[] values)
        => new() { Name = name, Type = AttributeProto.AttributeType.Ints, Ints = values };

    private static NodeProto Node(string opType, string[] inputs, string[] outputs, params AttributeProto[] attributes)
    {
        var node = new NodeProto { OpType = opType, Name = outputs[0] + "_node" };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static ModelProto PaddedAsFarAsTheKernel()
    {
        var graph = new GraphProto { Name = "padded_pool" };
        graph.Inputs.Add(TensorInfo("x", 1, 1, 1, 5));
        graph.Nodes.Add(Node("AveragePool", ["x"], ["y"], Ints("kernel_shape", 2), Ints("pads", 2, 0),
            new AttributeProto { Name = "count_include_pad", Type = AttributeProto.AttributeType.Int, I = 1 }));
        graph.Outputs.Add(TensorInfo("y", 1, 1, 1, 6));
        return WrapModel(graph);
    }

    private static ModelProto NegInAFunction()
    {
        var fn = new FunctionProto { Name = "NegFn", Domain = "Functions" };
        fn.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        fn.Inputs.Add("fn_x");
        fn.Outputs.Add("fn_y");
        fn.ValueInfoes.Add(TensorInfo("fn_x", 1, 3));
        fn.ValueInfoes.Add(TensorInfo("fn_y", 1, 3));
        fn.Nodes.Add(Node("Neg", ["fn_x"], ["negated"]));
        fn.Nodes.Add(Node("Mul", ["negated", "fn_x"], ["fn_y"]));

        var graph = new GraphProto { Name = "neg_fn_graph" };
        graph.Inputs.Add(TensorInfo("x", 1, 3));
        var call = Node("NegFn", ["x"], ["y"]);
        call.Domain = "Functions";
        graph.Nodes.Add(call);
        graph.Outputs.Add(TensorInfo("y", 1, 3));

        var model = WrapModel(graph);
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "Functions", Version = 1 });
        model.Functions.Add(fn);
        return model;
    }

    private sealed class NegAsSubtraction : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEG]);

        public override bool Applies(WorkaroundSite site) => site.DTypeOf(0) == DType.Float32;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => [OnnxOp.Sub(OnnxOp.CastLike(Scalar(0f), inputs[0]!, null), inputs[0]!)];
    }

    private sealed class AbsAsIf : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([ABS]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        {
            var x = inputs[0]!;
            return [Shorokoo.Core.Nodes.Ops.IfElse((Scalar<bit>)OnnxOp.Less(OnnxOp.ReduceMin(x, keepdims: false), Scalar(0f)),
                OnnxOp.Mul(x, OnnxOp.Sign(x)), OnnxOp.Identity(x, null))];
        }
    }

    private sealed class UnaryThroughIdentity : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([NEG, ABS]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
            => [OnnxOp.Identity(site.OpCode == NEG ? OnnxOp.Neg(inputs[0]!) : OnnxOp.Abs(inputs[0]!), null)];
    }

    private sealed class TopKThroughIdentities : KernelWorkaround
    {
        public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([TOPK]);

        public override bool Applies(WorkaroundSite site) => true;

        public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        {
            var (values, indices) = OnnxOp.TopK(inputs[0]!, inputs[1]!, site.Attributes.GetLongVal(OnnxOpAttributeNames.AttrAxis),
                site.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrLargest), site.Attributes.GetBoolVal(OnnxOpAttributeNames.AttrSorted));
            return [OnnxOp.Identity(values, null), OnnxOp.Identity(indices, null)];
        }
    }
}

[Module]
public partial class KernelWorkaroundNegInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> x)
    {
        var a = x;
        var trips = x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>();
        foreach (var ctx in LoopAPI.Iterate(trips))
            a = (Tensor<float32>)OnnxOp.Neg(a) * Scalar(2f);
        return a;
    }
}
