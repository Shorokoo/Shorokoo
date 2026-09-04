using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCoverageTests
{
    [Fact]
    public void TestStateUpdateSurvivesNestedFirstUseModuleBuild()
    {
        var graph = ((ComputationGraph)typeof(Modules.StateUpdateSurvivesNestedFirstUseBuild)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.WITH_STATE_DEPS);
    }

    private static double[] Rep(double v, int n) => [.. Enumerable.Repeat(v, n)];

    /// <summary>Concretizes <see cref="Modules.GatedBiasHyperLayer"/> under <paramref name="hint"/>
    /// and runs it with <paramref name="atExecute"/>. Null means the contradiction did not survive
    /// to a result: either the concrete model no longer declares the gating bit (so the two values
    /// cannot differ) or the run was rejected. Written by hand rather than through
    /// <c>AutoTest.AdvancedTestGraph</c>, which feeds one value array as both the concretization
    /// hints and the execution inputs and so cannot express a contradiction at all.</summary>
    private static float[]? GatedBiasOutcome(bool hint, bool atExecute)
    {
        var g = Modules.GatedBiasHyperLayer.ComputationGraph;
        var x = TensorData([2L], 1f, 2f);
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([], hint), x]))
            .ToConcreteModel(RngConfig.Default);
        var gated = model.InputNames.Contains(GatingInput);
        if (!gated && atExecute != hint) return null;
        IData[] inputs = gated ? [TensorData([], atExecute), x] : [x];
        NamedModelParam[] outputs;
        try
        {
            outputs = new ComputeContext().Execute(model, inputs);
        }
        catch (ShorokooException)
        {
            return null;
        }
        return outputs[0].ToTensorData().As<float32>().AccessMemory<float>().ToArray();
    }

    private const string GatingInput = "useBias";

    /// <summary>
    /// Concretizing the gated hyperparameter as <c>false</c> prunes the bias but leaves the gating
    /// bit a live graph input, so <c>Execute</c> accepts the contradicting value <c>true</c> and
    /// silently returns the surviving unbiased branch's result instead of rejecting the run. The
    /// fault is one-directional: a <c>true</c> hint keeps both branches, so executing it with
    /// <c>false</c> correctly returns the unbiased result. Tracked as Shorokoo/Shorokoo#217.
    /// </summary>
    [Fact(Skip = "Shorokoo/Shorokoo#217: a false gating hint executed with true silently returns the unbiased result")]
    public void TestAFalseGatedHyperparamHintExecutedWithTrueIsSilentlyIgnored()
    {
        float[] plain = [1f, 2f];
        float[] biased = [2f, 3f];
        Assert.Equal(plain, GatedBiasOutcome(false, false));
        Assert.Equal(biased, GatedBiasOutcome(true, true));
        Assert.Null(GatedBiasOutcome(false, true));
    }

    [Fact]
    public void TestSimpleHyperparamLoopSequenceOptionalAndConditionalModulesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimplestLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<StaticAndInputShapedParamsLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<HypersLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0.5f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<SimpleWithHyperparam>(
            hyperparamInputs: [TensorData(DType.Int64, [], 7L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.7, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<BackbonerSquared>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L]), TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<CustomTrainableParamInitializer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Int64, [2L], 2L, 5L), TensorData(DType.Float32, [], 0.5f)],
            expected: Rep(1.0, 10)));

        Assert.True(AutoTest.AdvancedTestGraph<LoopLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 4L), TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(32.0, 8)));
        Assert.True(AutoTest.AdvancedTestGraph<TwoStackLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 4L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(7.0, 8)));
        Assert.True(AutoTest.AdvancedTestGraph<ModelsCreatedInLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<SimplestBackboneCalledInNestedLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(4.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<HyperparamModelSequenceSimpleLooped>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(32.3, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<NestedLoopWithSubmoduleInnerLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(1562.5, 10)));

        Assert.True(AutoTest.AdvancedTestGraph<SimpleModelSequence>(
            hyperparamInputs: [TensorData(DType.Int64, [], 5L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<SeqHypersSequenceCalled>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<OptionalHypersLayerStraight>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<OptionalHypersEmptyThenAppend>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));

        Assert.True(AutoTest.AdvancedTestGraph<ConditionalTrainableParamLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(0.1, 10)));
        Assert.True(AutoTest.AdvancedTestGraph<ConditionalTrainableParamInDynamicLoopLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(0.1, 10)));
    }

    [Fact]
    public void TestModuleGraphRoundtripSaveLoadGenericInputAndTensorStructCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<CallsSimplestModule>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<CallsHypersLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<CallsCallsHypersLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<SimplestLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<HypersLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0.5f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<LoopLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 4L), TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(32.0, 8)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<SimpleModelSequence>(
            hyperparamInputs: [TensorData(DType.Int64, [], 5L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<OptionalHypersLayerStraight>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<SeqHypersSequenceCalled>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<NestedLoopWithSubmoduleInnerLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])],
            expected: Rep(1562.5, 10)));

        AssertSaveLoadOnly<InlineBatchNormWithState>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 3L, 4L, 4L])]);
        AssertSaveLoadOnly<SimplePairSum>(
            hyperparamInputs: [],
            runtimeInputs: []);

        AssertGenericSaveLoadOnly<SimpleGenericLayer>();
        AssertGenericSaveLoadOnly<GenericComposedLayer>();

        AssertSaveLoadOnly<RealGenericTensorStructSumCaller>(
            hyperparamInputs: [],
            runtimeInputs: [],
            genericTypes: new() { ["T"] = DType.Float32 });
    }

    [Fact]
    public void TestTrainableParamsSerializeAsOnnxInitializers()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();

        var paramNodes = concrete.ToInternal().Nodes
            .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA).ToArray();
        Assert.Equal(2, paramNodes.Length);
        Assert.All(paramNodes, n =>
        {
            Assert.True(n.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable));
            Assert.False(string.IsNullOrEmpty(n.IdentifierTemplate));
        });

        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(concrete);
        var inits = proto.Graph.Initializers.ToArray();
        Assert.Equal(2, inits.Length);
        Assert.All(inits, t =>
        {
            Assert.Equal("true", t.MetadataProps
                .First(p => p.Key == OnnxOpAttributeNames.ShrkMetaIsTrainable).Value);
            Assert.NotNull(t.MetadataProps
                .FirstOrDefault(p => p.Key == OnnxOpAttributeNames.ShrkMetaNodeIdentifierTemplate));
        });
        Assert.DoesNotContain(proto.Graph.Nodes, n =>
            n.OpType == OpCodes.CONSTANT
            && n.Attributes.Any(a => a.T is { Dims.Length: 2 }));

        var direct = ComputeContext.Default.Execute(concrete, numOut, input)[0]
            .ToTensorData().AccessRawMemory().ToArray();
        using var ms = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        var reimported = OnnxModelImporter.FromOnnxModel(ms.ToArray());
        var reParams = reimported.ToInternal().Nodes
            .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA
                && (n.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false))
            .ToArray();
        Assert.Equal(2, reParams.Length);
        Assert.All(reParams, n => Assert.False(string.IsNullOrEmpty(n.IdentifierTemplate)));
        var roundtrip = ComputeContext.Default.Execute(reimported, numOut, input)[0]
            .ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(direct, roundtrip);
    }

    [Fact]
    public void TestVanillaExportSignatureIONamesKnownRankDimsAndModuleStageRefusal()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();

        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(concrete);

        string[] inputNames = proto.Graph.Inputs.Select(x => x.Name).ToArray();
        string[] outputNames = proto.Graph.Outputs.Select(x => x.Name).ToArray();
        Assert.Equal(concrete.InputNames.Count, inputNames.Length);
        Assert.Contains("input", inputNames);
        Assert.All(inputNames.Concat(outputNames), n =>
        {
            Assert.DoesNotContain(":", n);
            Assert.DoesNotMatch("^N[0-9]+(_T[0-9]+)?$", n);
        });

        using var ms = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        var bytes = ms.ToArray();

        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(bytes);
        Assert.Equal(inputNames, session.InputNames);
        Assert.Equal(outputNames, session.OutputNames);
        var tensorInMeta = session.InputMetadata["input"];
        Assert.Equal(typeof(float), tensorInMeta.ElementType);
        var hyperName = inputNames.Single(n => n != "input");
        var hyperMeta = session.InputMetadata[hyperName];
        Assert.Equal(typeof(long), hyperMeta.ElementType);
        Assert.Empty(hyperMeta.Dimensions);
        var outMeta = session.OutputMetadata[outputNames.Single()];
        Assert.Equal(typeof(float), outMeta.ElementType);

        var direct = ComputeContext.Default.Execute(concrete, numOut, input)[0]
            .ToTensorData().As<float32>().AccessMemory().ToArray();
        long[] hyperData = [4L];
        int[] scalarDims = [];
        int[] inputDims = [4, 4];
        var hyperTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(hyperData, scalarDims);
        var inputTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(
            input.As<float32>().AccessMemory().ToArray(), inputDims);
        List<Microsoft.ML.OnnxRuntime.NamedOnnxValue> feeds = [
            Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(hyperName, hyperTensor),
            Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("input", inputTensor),
        ];
        using var results = session.Run(feeds);
        var ortOut = results.Single();
        Assert.Equal(outputNames.Single(), ortOut.Name);
        Assert.Equal(direct, ortOut.AsEnumerable<float>().ToArray());

        var reimported = OnnxModelImporter.FromOnnxModel(bytes);
        Assert.Equal(inputNames, reimported.InputNames);
        Assert.Equal(outputNames, reimported.OutputNames);
        var reexported = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(reimported);
        Assert.Equal(inputNames, reexported.Graph.Inputs.Select(x => x.Name).ToArray());
        Assert.Equal(outputNames, reexported.Graph.Outputs.Select(x => x.Name).ToArray());

        var xs = TensorData([2L], 1f, 5f);
        var ys = TensorData([2L], 3f, 2f);
        var rankGraph = VectorMinMaxOthersBugPinCheck.ComputationGraph;
        var rankConcrete = rankGraph
            .ToConcreteArchitecture(rankGraph.FromOrderedInputs([xs, ys])).ToConcreteModel();
        var rankProto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(rankConcrete);

        foreach (var name in (string[])["xs", "ys"])
        {
            var info = rankProto.Graph.Inputs.Single(x => x.Name == name);
            var dim = Assert.Single(info.Type.TensorType.Shape.Dims);
            Assert.Equal($"{name}_dim0", dim.DimParam);
        }
        var rankOutInfo = rankProto.Graph.Outputs.Single();
        Assert.Equal(DType.Bool.ProtoTypeNum, rankOutInfo.Type.TensorType.ElemType);
        Assert.NotNull(rankOutInfo.Type.TensorType.Shape);
        Assert.Empty(rankOutInfo.Type.TensorType.Shape.Dims);

        using var rankMs = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(rankMs, rankProto);
        using var rankSession = new Microsoft.ML.OnnxRuntime.InferenceSession(rankMs.ToArray());
        int[] dynamicRank1 = [-1];
        Assert.Equal(dynamicRank1, rankSession.InputMetadata["xs"].Dimensions);
        Assert.Equal(dynamicRank1, rankSession.InputMetadata["ys"].Dimensions);
        Assert.Equal(typeof(float), rankSession.InputMetadata["xs"].ElementType);
        var rankOutMeta = rankSession.OutputMetadata[rankOutInfo.Name];
        Assert.Equal(typeof(bool), rankOutMeta.ElementType);
        Assert.Empty(rankOutMeta.Dimensions);

        var moduleStage = TwoStackLayer.ComputationGraph;
        var internalGraph = moduleStage.ToInternal();
        Assert.Contains(internalGraph.Nodes, n => n.OpCode == InternalOpCodes.CREATE_MODULE);

        var kindEx = Assert.Throws<ModelException>(
            () => Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(moduleStage));
        Assert.Equal(ErrorCodes.FW045, kindEx.ErrorCode);
        Assert.Contains("ToConcreteModel", kindEx.Message);

        var ex = Assert.Throws<ModelException>(
            () => Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(internalGraph));
        Assert.Equal(ErrorCodes.FW045, ex.ErrorCode);
        Assert.Contains(InternalOpCodes.CREATE_MODULE, ex.Message);
        Assert.Contains("ToConcreteModel", ex.Message);

        var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleStage, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(data);
        Assert.Equal(internalGraph.Nodes.Count, reloaded.ToInternal().Nodes.Count);
        Assert.Contains(reloaded.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.CREATE_MODULE);
    }

    private static Tensor<float32> DoubleScalar(Tensor<float32> input) => input + input;

    [Fact]
    public void TestFastFunctionInvokeNodeReload()
    {
        System.Func<Tensor<float32>, Tensor<float32>> impl = DoubleScalar;
        var fn = Shorokoo.Core.ModuleHelper.CreateTargetFunction(impl);
        Assert.Equal(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.Module, fn.FunctionType);

        var input = (Tensor<float32>)Shorokoo.Core.Nodes.NodeDefinitions.InternalOp.ModuleTensorInput(
            DType.Float32, rank: 1, Shorokoo.Core.Nodes.NodeDefinitions.InputType.ModelInput,
            targetFunction: null, defaultName: "input");
        var callResult = fn.Call(input);
        var output = (Tensor<float32>)callResult[0];

        var graph = new InternalComputationGraph(
            System.Collections.Immutable.ImmutableArray.Create<Shorokoo.Core.Variable>(input),
            System.Collections.Immutable.ImmutableArray.Create<Shorokoo.Core.Variable>(output));

        Assert.Single(graph.Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        var preInvoke = graph.Nodes.Single(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        Assert.Same(fn, preInvoke.TargetFunction);

        var data = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphCore(data, "<roundtrip>", null).Graph;

        Assert.Single(reloaded.Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        var postInvoke = reloaded.Nodes.Single(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        Assert.NotNull(postInvoke.TargetFunction);
        Assert.Equal(fn.DefaultName, postInvoke.TargetFunction!.DefaultName);
        Assert.Equal(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.Module, postInvoke.TargetFunction.FunctionType);
        DType[] expectedDTypes = [DType.Float32];
        Assert.Equal(expectedDTypes, postInvoke.Attributes.GetDTypesVal(OnnxOpAttributeNames.ShrkAttrDtype)!);
        Assert.Single(postInvoke.Outputs);
    }

    [Fact]
    public void TestGenericModulesDefaultSpecializationCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimpleGenericLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericScaleLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.2, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericAddLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L]), TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.2, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericComposedLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.4, 5)));
    }

    [Fact]
    public void TestGenericModulesUserSpecializationCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimpleGenericLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericScaleLayer>(
            hyperparamInputs: [TensorData(DType.Float64, [], 2.0)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: Rep(0.2, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<AddThree>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: Rep(3.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericConstantOfShapeLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Int64, [1L], 5L)],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: Rep(5.0, 5)));
        // Periodic Blackman, N=8: 0.42 - 0.5cos(2pi n/N) + 0.08cos(4pi n/N).
        Assert.True(AutoTest.AdvancedTestGraph<GenericBlackmanWindowLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Int64, [], 8L)],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: [0.0, 0.066446609, 0.34, 0.773553391, 1.0, 0.773553391, 0.34, 0.066446609]));
        Assert.True(AutoTest.AdvancedTestGraph<GenericCastLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["TIn"] = DType.Float64, ["TOut"] = DType.Float32 },
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericThreeTypeParamLayer>(
            hyperparamInputs: [TensorData(DType.Float64, [], 1.0), TensorData(DType.Int32, [], 2), TensorData(DType.Int32, [], 3)],
            runtimeInputs: [TensorData(DType.Float64, [], 4.0), TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 6f)],
            genericTypes: new() { ["T"] = DType.Float64, ["Q"] = DType.Int32, ["R"] = DType.Float32 },
            expected: [1.0]));
        Assert.True(AutoTest.AdvancedTestGraph<GenericAddLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L]), TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: Rep(0.2, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<GenericComposedLayer>(
            hyperparamInputs: [TensorData(DType.Float64, [], 2.0)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 },
            expected: Rep(0.4, 5)));
    }

    [Fact]
    public void TestModuleOnModuleTrainableParamRefFunctionLinkCoverage()
    {
        var moduleGraph = CallsSimplestModule.ComputationGraph;
        Assert.Contains(moduleGraph.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);

        var binary = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(binary);
        Assert.Contains(reloaded.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);

        TensorData[] sampleInputs = [TensorDataWithSmallVals(DType.Float32, [5L])];
        var concreteArch = reloaded.ToConcreteArchitecture(reloaded.FromOrderedInputs([.. sampleInputs]));

        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);
        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
    }

    [Fact]
    public void TestSpecializeFullPartialAndThenConcretizePipelineCoverage()
    {
        var factor = TensorData(DType.Float32, [], 2f);
        var bias   = TensorData(DType.Float32, [], 0.5f);
        var input  = TensorDataWithSmallVals(DType.Float32, [5L]);

        var moduleGraph  = HypersLayer.ComputationGraph;
        TensorData[] allHints = [factor, bias, input];
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allHints]));
        var model        = concreteArch.ToConcreteModel();
        int originalInputCount = model.ToInternal().Inputs.Count;

        var specializedModel = model.Specialize(model.FromOrderedInputs([factor, bias]));
        Assert.Equal(originalInputCount - 2, specializedModel.ToInternal().Inputs.Count);
        Assert.Equal(originalInputCount, model.ToInternal().Inputs.Count);

        var expected = ComputeContext.Default.Execute(model, factor, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        var actual   = ComputeContext.Default.Execute(specializedModel, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, actual);

        var partialHints = new ModelParamList([
            new TensorDataModelParam(model.InputNames[0]!, ModelParamType.InputParam, factor)
        ]);
        var partial = model.Specialize(partialHints);
        Assert.Equal(originalInputCount - 1, partial.ToInternal().Inputs.Count);
        var partialActual = ComputeContext.Default.Execute(partial, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, partialActual);

        var specialized = moduleGraph.Specialize(moduleGraph.FromOrderedInputs([factor, bias]));
        Assert.Equal(["input"], specialized.InputNames);

        var concrete = specialized
            .ToConcreteArchitecture(specialized.FromOrderedInputs([input]))
            .ToConcreteModel();
        Assert.Single(concrete.ToInternal().Inputs);
        Assert.Equal(expected,
            ComputeContext.Default.Execute(concrete, input)[0].ToTensorData().AccessRawMemory().ToArray());

        var numOut  = TensorData(DType.Int64, [], 3L);
        var fcInput = TensorDataWithSmallVals(DType.Float32, [2L, 5L]);
        var fcGraph = FCLayer.ComputationGraph;

        var fcSpecialized = fcGraph.Specialize(fcGraph.FromOrderedInputs([numOut]));
        Assert.Equal(["input"], fcSpecialized.InputNames);

        var fcConcrete = fcSpecialized
            .ToConcreteArchitecture(fcSpecialized.FromOrderedInputs([fcInput]))
            .ToConcreteModel();
        Assert.Single(fcConcrete.ToInternal().Inputs);

        var fcActual = ComputeContext.Default.Execute(fcConcrete, fcInput)[0].ToTensorData().AccessRawMemory().ToArray();
        var fcRef = fcGraph.ToConcreteArchitecture(fcGraph.FromOrderedInputs([numOut, fcInput])).ToConcreteModel();
        var fcExpected = ComputeContext.Default.Execute(fcRef, numOut, fcInput)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(fcExpected, fcActual);
    }

    private static float[] RunFloats(ComputationGraph model, params TensorData[] inputs)
        => ComputeContext.Default.Execute(model, inputs)[0].ToTensorData().As<float32>().AccessMemory<float>().ToArray();

    private static void AssertBakedHypersMatchHintedHypers(
        ComputationGraph module, TensorData[] hypers, TensorData input)
    {
        var hinted    = module.ToConcreteArchitecture(module.FromOrderedInputs([.. hypers, input]));
        var baked     = module.Specialize(module.FromOrderedInputs([.. hypers]));
        var bakedArch = baked.ToConcreteArchitecture(baked.FromOrderedInputs([input]));

        ModelId[] hintedIds = [.. hinted.GetConcreteModelParamInfos().ModelIds];
        ModelId[] bakedIds  = [.. bakedArch.GetConcreteModelParamInfos().ModelIds];
        Assert.Equal(hintedIds, bakedIds);
        Assert.Equal(RunFloats(hinted.ToConcreteModel(), [.. hypers, input]),
                     RunFloats(bakedArch.ToConcreteModel(), input));
    }

    /// <summary>The documented lowering pipeline is <c>Specialize</c> -> <c>ToConcreteArchitecture</c>
    /// -> <c>ToConcreteModel</c>, so baking a hyper must reach the same concrete model as passing it
    /// as a concretization hint. Both modules allocate trainable params inside a
    /// <c>LoopAPI.Iterate</c> body, which is what the baked route mishandles today.
    /// Tracked as Shorokoo/Shorokoo#221.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#221: Specialize mishandles trainable params allocated inside a loop body")]
    public void TestBakingHypersMatchesHintingThemWhenParamsLiveInsideALoopBody()
    {
        var input = TensorDataWithSmallVals(DType.Float32, [2L, 5L]);
        AssertBakedHypersMatchHintedHypers(LoopLayer.ComputationGraph,
            [TensorData(DType.Int64, [], 4L), TensorData(DType.Int64, [], 3L)], input);
        AssertBakedHypersMatchHintedHypers(ConditionalTrainableParamInLoopLayer.ComputationGraph,
            [TensorData(DType.Int64, [], 3L), TensorData(DType.Int64, [], 4L)], input);
    }

    private static string ConcretizeWithNoHints(ComputationGraph graph)
        => Assert.IsType<InvalidOperationException>(Record.Exception(
            () => graph.ToConcreteArchitecture(new ModelParamList([])))).Message;

    [Fact]
    public void TestConcretizingWithoutTheHintAParamShapeNeedsFailsWithACatchableExceptionNotAnAssertion()
    {
        Assert.Contains("input 'input'", ConcretizeWithNoHints(SimplestLayer.ComputationGraph));
        Assert.Contains("input 'input'", ConcretizeWithNoHints(ConditionalTrainableParamInLoopLayer.ComputationGraph));
        Assert.DoesNotContain("threshold", ConcretizeWithNoHints(ConditionalTrainableParamInLoopLayer.ComputationGraph));
        Assert.Contains("input 'input'", ConcretizeWithNoHints(StaticAndInputShapedParamsLayer.ComputationGraph));
    }

    [Fact]
    public void TestZeroTripLoopLeavesTheAccumulatorUntouchedOnBothEngines()
        => Assert.True(AutoTest.AdvancedTestGraph<AnalyticLoopZeroTripCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 0f, 5f, 7f)]));

    [Fact]
    public void TestZeroTripLoopStacksNoRowsIntoItsScanOutput()
    {
        var g = ZeroTripLoopWithScanOutput.ComputationGraph;
        var x = TensorData(DType.Float32, [3L], 0f, 5f, 7f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x])).ToConcreteModel().ToInternal();
        var scanned = Assert.IsType<RuntimeTensor>(
            new QuickExecutionEngine().Run(concrete, x)[concrete.Outputs[0]]);
        Assert.Equal(0L, scanned.Shape!.Dims[0]);
    }

    [Fact]
    public void TestOpSemanticsAndControlFlowAnalyticChecksAndLoopGraphOnnxRoundtrip()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticOpSemanticsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [3L], 10f, 20f, 30f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticNaNMismatchGuardCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticIfElseNaNIsolationCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticLoopAccumulateCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], 3f, 4f, 5f, 6f)]));

        var g = AnalyticLoopAccumulateCheck.ComputationGraph;
        var x = TensorData(DType.Float32, [4L], 3f, 4f, 5f, 6f);
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x])).ToConcreteModel();
        var direct = ComputeContext.Default.Execute(concrete, x)[0].ToTensorData().AccessRawMemory().ToArray();
        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(concrete);
        using var ms = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        var reimported = OnnxModelImporter.FromOnnxModel(ms.ToArray());
        var roundtrip = ComputeContext.Default.Execute(reimported, x)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(direct, roundtrip);
        Assert.Equal(1, direct[0]);
    }

    private static string[] NodesNotOwningTheirOutputs(InternalComputationGraph g)
        => [.. g.Nodes
            .Where(n => n.FullOutputs.Values.SelectMany(slots => slots)
                .Any(k => k is FastTensorKey t && !t.IsEmpty && !t.FastNodeKey.Equals(n.Key)))
            .Select(n => n.OpCode.ToString())];

    [Fact]
    public void TestEveryNodeOwnsTheTensorKeysItProduces()
    {
        Assert.Empty(NodesNotOwningTheirOutputs(ConstLoopWithScanOutput.ComputationGraph.ToInternal()));
        Assert.Empty(NodesNotOwningTheirOutputs(TensorStructLoopCarry.ComputationGraph.ToInternal()));

        var g = MixedTensorStructLoopRuntimeTripCount.ComputationGraph.ToInternal();
        var inputs = g.FromOrderedInputs([
            TensorData(DType.Float32, [2L], 2f, 5f),
            TensorData(DType.Float32, [], 1f),
            TensorData(DType.Float32, [], 2f)]);
        Assert.Empty(NodesNotOwningTheirOutputs(g.ToConcreteArchitecture(inputs)));
    }

    private static Tensor<float32> MachineryFreeBody(Tensor<float32> x) => x + x;

    [Fact]
    public void TestGraphKindStampingChecksCopySemanticsAndWithKindReStampCoverage()
    {
        var sample = TensorData([2L], 1.0f, 2.0f);

        var moduleGraph = ScalarMultiplyModel.ComputationGraph;
        Assert.Equal(GraphKind.Module, moduleGraph.Kind);

        var arch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([sample]));
        Assert.Equal(GraphKind.ConcreteArchitecture, arch.Kind);

        var model = arch.ToConcreteModel();
        Assert.Equal(GraphKind.ConcreteModel, model.Kind);
        Assert.Equal(GraphKind.ConcreteModel, model.Specialize(new ModelParamList([])).Kind);

        var exArch = Assert.Throws<InvalidOperationException>(
            () => arch.ToConcreteArchitecture(arch.FromOrderedInputs([sample])));
        Assert.Contains("'module'", exArch.Message);
        Assert.Contains("'concrete-architecture'", exArch.Message);

        var exModule = Assert.Throws<InvalidOperationException>(() => moduleGraph.ToConcreteModel());
        Assert.Contains("'concrete-architecture'", exModule.Message);
        Assert.Contains("'module'", exModule.Message);

        var exTwice = Assert.Throws<InvalidOperationException>(() => model.ToConcreteModel());
        Assert.Contains("'concrete-model'", exTwice.Message);

        var exInfos = Assert.Throws<InvalidOperationException>(() => moduleGraph.GetConcreteModelParamInfos());
        Assert.Contains("'concrete-architecture'", exInfos.Message);

        var exExec = Assert.Throws<InvalidOperationException>(
            () => ComputeContext.Default.Execute(moduleGraph, sample));
        Assert.Contains("concretized", exExec.Message);
        Assert.Contains("'module'", exExec.Message);

        var nodeCount = model.ToInternal().Nodes.Count;
        var copy = model.ToInternal();
        copy.Nodes.Clear();
        Assert.Equal(nodeCount, model.ToInternal().Nodes.Count);

        var source = model.ToInternal();
        var frozen = ComputationGraph.FromInternal(source, GraphKind.ConcreteModel);
        source.Nodes.Clear();
        Assert.Equal(nodeCount, frozen.ToInternal().Nodes.Count);
        Assert.Equal(GraphKind.ConcreteModel, frozen.Kind);
        Assert.Equal(GraphKind.ConcreteModel, ComputationGraph.FromInternal(model.ToInternal()).Kind);

        Assert.Single(ComputeContext.Default.Execute(model, sample));

        Assert.Same(moduleGraph, moduleGraph.WithKind(GraphKind.Module));
        Assert.Equal(GraphKind.Module, arch.WithKind(GraphKind.Module).Kind);

        var exToArch = Assert.Throws<InvalidOperationException>(
            () => moduleGraph.WithKind(GraphKind.ConcreteArchitecture));
        Assert.Contains("module-stage op", exToArch.Message);

        var exToModel = Assert.Throws<InvalidOperationException>(
            () => arch.WithKind(GraphKind.ConcreteModel));
        Assert.Contains("unmaterialized", exToModel.Message);

        var exBackToArch = Assert.Throws<InvalidOperationException>(
            () => model.WithKind(GraphKind.ConcreteArchitecture));
        Assert.Contains("initialized", exBackToArch.Message);
        var exBackToModule = Assert.Throws<InvalidOperationException>(
            () => model.WithKind(GraphKind.Module));
        Assert.Contains("initialized", exBackToModule.Message);

        var misStamped = ComputationGraph.FromInternal(
            ModuleFactory.ComputationGraph((Func<Tensor<float32>, Tensor<float32>>)MachineryFreeBody)
                .ToInternal());
        Assert.Equal(GraphKind.ConcreteModel, misStamped.Kind);
        var reStamped = misStamped.WithKind(GraphKind.Module);
        var relowered = reStamped.ToConcreteArchitecture(reStamped.FromOrderedInputs([sample]));
        Assert.Equal(GraphKind.ConcreteArchitecture, relowered.Kind);
    }

    private static void AssertGenericSaveLoadOnly<TModule>()
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var moduleGraph = (ComputationGraph)prop.GetValue(null)!;
        Assert.Contains(moduleGraph.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT);

        var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(data);
        Assert.Contains(reloaded.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT);
        Assert.Equal(moduleGraph.ToInternal().Nodes.Count, reloaded.ToInternal().Nodes.Count);
    }

    private static void AssertSaveLoadOnly<TModule>(
        TensorData[] hyperparamInputs,
        TensorData[] runtimeInputs,
        System.Collections.Generic.Dictionary<string, DType>? genericTypes = null)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();

        if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
        {
            if (genericTypes is not null && genericTypes.Count > 0)
                Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
            moduleGraph = Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType.Process(moduleGraph);
        }

        var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
        moduleGraph = CompressedFormatUtils.LoadFastGraphCore(data, "<roundtrip>", null).Graph;

        var allInputs = new System.Collections.Generic.List<TensorData>();
        allInputs.AddRange(hyperparamInputs);
        allInputs.AddRange(runtimeInputs);

        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));
        var archData = CompressedFormatUtils.SaveFastGraphToBinary(concreteArch, compressed: true);
        concreteArch = CompressedFormatUtils.LoadFastGraphCore(archData, "<roundtrip>", null).Graph;

        var concreteModel = concreteArch.ToConcreteModel();
        var modelData = CompressedFormatUtils.SaveFastGraphToBinary(concreteModel, compressed: true);
        var reloadedModel = CompressedFormatUtils.LoadFastGraphFromBinary(modelData);

        Assert.NotEmpty(reloadedModel.ToInternal().Nodes);
        Assert.NotEmpty(reloadedModel.ToInternal().Outputs);
    }
}
