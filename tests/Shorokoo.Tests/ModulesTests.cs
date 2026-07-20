using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Graph;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests that drive the AutoTester (ONNX roundtrip, CS roundtrip,
/// <see cref="Shorokoo.Core.Inference.QuickExecutionEngine"/>) over a curated
/// set of existing modules. Tests are grouped by module flavour — simple, hyperparam,
/// loop / nested-loop, sequence / submodel, conditional trainable params — so a single
/// [Fact] drives many <see cref="AutoTest.AdvancedTestGraph{TModule}"/> calls.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCoverageTests
{
    [Fact]
    public void TestStateUpdateSurvivesNestedFirstUseModuleBuild()
    {
        // Globals.StateUpdate registers, then the body first-uses StateWipeFreshInit — whose
        // Function is uncached (it is referenced nowhere else), so its body graph builds
        // mid-trace. The registered update must survive that nested build: the module graph
        // carries the STATE_UPDATE_LINK and its outputs are WITH_STATE_DEPS-wrapped. Before the
        // fix, the nested build's entry-time clear wiped the registration and the wrap was
        // silently dropped.
        var graph = ((ComputationGraph)typeof(Modules.StateUpdateSurvivesNestedFirstUseBuild)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.WITH_STATE_DEPS);
    }

    [Fact]
    public void TestSimpleAndHyperparamModulesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimplestLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<HypersLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0.5f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<SimpleWithHyperparam>(
            hyperparamInputs: [TensorData(DType.Int64, [], 7L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<BackbonerSquared>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L]), TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<CustomTrainableParamInitializer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Int64, [2L], 2L, 5L), TensorData(DType.Float32, [], 0.5f)]));
    }

    [Fact]
    public void TestLoopModulesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<LoopLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 4L), TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<TwoStackLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 4L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<ModelsCreatedInLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<SimplestBackboneCalledInNestedLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<HyperparamModelSequenceSimpleLooped>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<NestedLoopWithSubmoduleInnerLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
    }

    [Fact]
    public void TestSequenceAndOptionalModulesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimpleModelSequence>(
            hyperparamInputs: [TensorData(DType.Int64, [], 5L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<SeqHypersSequenceCalled>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<OptionalHypersLayerStraight>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<OptionalHypersEmptyThenAppend>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
    }

    [Fact]
    public void TestConditionalTrainableParamModulesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<ConditionalTrainableParamLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<ConditionalTrainableParamInDynamicLoopLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
    }

    /// <summary>
    /// Coverage for the pre-concretization moduleGraph ONNX save/load path that
    /// the standard <see cref="AutoTest.AdvancedTestGraph{TModule}"/> can't reach
    /// (it roundtrips the concrete model, by which point modules are inlined and
    /// trainable params are materialized as MODEL_PARAM_DATA). Driving these modules
    /// through the moduleGraph roundtrip exercises
    /// <c>OnnxModelReader.BuildFastFunctionInvokeNodeFromProto</c>,
    /// <c>OnnxModelReader.BuildFastTrainableParamNodeFromProto</c>, and the legacy
    /// <c>internalBuildFunctions</c> / <c>internalInitFunction</c> /
    /// <c>CreateNodes</c> FunctionProto build path — load-time paths that fire
    /// only when the saved graph still contains <c>FUNCTION_INVOKE</c> /
    /// <c>SEQUENCE_CONSTRUCT</c> with a TargetFunction / unmaterialized
    /// <c>MODEL_PARAM</c> references. <c>SimpleModelSequence</c> covers the
    /// SEQUENCE_CONSTRUCT-with-TargetFunction save path
    /// (<c>FastOpsetResolver.Resolve</c> writing <c>ShrkAttrFunctionName</c> for any
    /// node whose schema declares it), and <c>OptionalHypersLayerStraight</c> covers
    /// the optional FunctionProto input load path (<c>CreateInputTensors</c>'
    /// <c>OptionalType</c> / <c>SequenceType</c> branches of the value-info type
    /// union, routed to <c>InternalOp.ModuleOptionalInput</c> /
    /// <c>ModuleSequenceInput</c>).
    /// </summary>
    [Fact]
    public void TestModuleGraphOnnxRoundtripCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<CallsSimplestModule>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<CallsHypersLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<CallsCallsHypersLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<SimplestLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<HypersLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0.5f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<LoopLayer>(
            hyperparamInputs: [TensorData(DType.Int64, [], 4L), TensorData(DType.Int64, [], 3L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<SimpleModelSequence>(
            hyperparamInputs: [TensorData(DType.Int64, [], 5L)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<OptionalHypersLayerStraight>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        // SeqHypersLayer takes a TensorSequence<float32> hyperparam, so its
        // FunctionProto serializes with a Sequence-typed input — drives the
        // DataStructure.Sequence branches of CreateFastInputTensors at load.
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<SeqHypersSequenceCalled>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        // Outer module loops twice, calling LoopLayer (whose body has its own
        // constant-iter LOOP) each time. The inner-loop body builds the inner
        // MODEL_PARAM_REF's shape vector from function-input hyperparams
        // (independent of the loop-iter scalar), so the function-body reload must
        // keep those nodes inside the OPEN/CLOSE band for nested unrolling to
        // recover all 6 trainable params — exercising internalInitFunction's
        // proto-order walker (EnumerateNodesInProtoOrder + CreateFastNodes).
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<NestedLoopWithSubmoduleInnerLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [2L, 5L])]));
    }

    /// <summary>
    /// Save/reload coverage for module shapes whose concrete model can't actually
    /// execute via ONNX Runtime (e.g. <see cref="InlineBatchNormWithState"/> uses
    /// <c>STATE_UPDATE_LINK</c>, which isn't a registered ORT op) but whose load
    /// paths are still worth exercising. We push the graphs through the
    /// moduleGraph → save → load → ToConcreteArchitecture → save → load →
    /// ToConcreteModel pipeline and assert the final concrete model is well-formed.
    /// Skips the runtime-execution step that <see cref="AutoTest.TestGraph"/>
    /// would otherwise do.
    ///
    /// <para>
    /// The state-initializer modules drive <c>OnnxModelReader.CreateFastInitializers</c>
    /// and <c>ParseIsTrainableMetadata</c>'s <c>false</c> arm: state params
    /// materialize as non-trainable <c>MODEL_PARAM_DATA</c> nodes (per
    /// <c>FastApplyModelParamValues</c>) and serialize as ONNX
    /// <c>graphProto.Initializers</c> tensors alongside the trainable ones.
    /// </para>
    /// </summary>
    [Fact]
    public void TestModuleGraphSaveLoadOnlyCoverage()
    {
        // State-initializer module: drives CreateFastInitializers +
        // ParseIsTrainableMetadata via the MODEL_PARAM_DATA-as-initializer
        // save/load path.
        AssertSaveLoadOnly<InlineBatchNormWithState>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [1L, 3L, 4L, 4L])]);
        // TensorStruct-input module: drives CreateFastInputTensors's TensorStruct
        // branch, ReconstructTensorStructDType, and ParseTensorStructMetadata.
        // Architecture-only — execution requires a TensorDataStruct input shape
        // the AutoTester's flat TensorData[] API doesn't model.
        AssertSaveLoadOnly<SimplePairSum>(
            hyperparamInputs: [],
            runtimeInputs: []);
    }

    /// <summary>
    /// Trainable parameters follow ONNX convention on export (issue #11): a concrete
    /// model's weights are <c>MODEL_PARAM_DATA</c> nodes flagged trainable, serialized
    /// as <c>graph.initializer</c> <c>TensorProto</c>s (with trainability + parameter-name
    /// metadata) — never baked into <c>Constant</c> op-nodes. The export → import
    /// roundtrip preserves the trainability flag and the parameter names, and executes
    /// bit-identically.
    /// </summary>
    [Fact]
    public void TestTrainableParamsSerializeAsOnnxInitializers()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;   // two trainable params: weights [4,4], bias [4]
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();

        // Concrete-model form: both weights are trainable MODEL_PARAM_DATA nodes that
        // kept their parameter-name IdentifierTemplate.
        var paramNodes = concrete.ToInternal().Nodes
            .Where(n => n.OpCode == InternalOpCodes.MODEL_PARAM_DATA).ToArray();
        Assert.Equal(2, paramNodes.Length);
        Assert.All(paramNodes, n =>
        {
            Assert.True(n.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable));
            Assert.False(string.IsNullOrEmpty(n.IdentifierTemplate));
        });

        // Exported form: the weights are graph initializers carrying IsTrainable=true and
        // the parameter-name metadata; no Constant op-node holds the [4,4] weight data.
        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(concrete);
        var inits = proto.Graph.Initializers
            .Where(t => t.Name != OnnxOpAttributeNames.ShrkRngKeysTensorName).ToArray();
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

        // Roundtrip: the loaded graph rebuilds trainable MODEL_PARAM_DATA nodes (still
        // discoverable/retrainable, names intact) and executes bit-identically.
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

    /// <summary>
    /// User-facing ONNX export (issue #36, defect 1): a concrete model's exported
    /// graph inputs/outputs carry the model's signature names (never internal
    /// <c>N{k}_T{s}</c> ids), dtype always stamped, and rank-many symbolic dims where
    /// the rank is known — all discoverable by a plain (no Shorokoo) ONNX Runtime
    /// <c>InferenceSession</c>, which also executes the file to the same values as
    /// Shorokoo's own execution. <c>OnnxModelImporter</c> round-trips the names, and a
    /// re-export of the re-imported graph keeps them stable.
    /// </summary>
    [Fact]
    public void TestVanillaExportSignatureIONamesAndTypes()
    {
        var numOut = TensorData(DType.Int64, [], 4L);
        var input = TensorDataWithSmallVals(DType.Float32, [4L, 4L]);
        var g = FCLayer.ComputationGraph;
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([numOut, input])).ToConcreteModel();

        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(concrete);

        // Proto level: I/O names come from the signature, in signature order.
        string[] inputNames = proto.Graph.Inputs.Select(x => x.Name).ToArray();
        string[] outputNames = proto.Graph.Outputs.Select(x => x.Name).ToArray();
        Assert.Equal(concrete.InputNames.Count, inputNames.Length);
        Assert.Contains("input", inputNames);
        Assert.All(inputNames.Concat(outputNames), n =>
        {
            Assert.DoesNotContain(":", n);   // not a serialized CG tensor key
            Assert.DoesNotMatch("^N[0-9]+(_T[0-9]+)?$", n);   // not an internal Fast id
        });

        using var ms = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        var bytes = ms.ToArray();

        // Stock ONNX Runtime (no Shorokoo anywhere): I/O discoverable by name with
        // dtypes reported. The [Hyper] scalar is a rank-0 int64; the rank-agnostic
        // Tensor<float32> input stays fully dynamic (no dims stamped), per spec.
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

        // The exported file executes in the stock runtime, by input NAME, to the same
        // values Shorokoo computes for the same concrete model.
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

        // OnnxModelImporter round-trips the names, and re-exporting keeps them stable.
        var reimported = OnnxModelImporter.FromOnnxModel(bytes);
        Assert.Equal(inputNames, reimported.InputNames);
        Assert.Equal(outputNames, reimported.OutputNames);
        var reexported = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(reimported);
        Assert.Equal(inputNames, reexported.Graph.Inputs.Select(x => x.Name).ToArray());
        Assert.Equal(outputNames, reexported.Graph.Outputs.Select(x => x.Name).ToArray());
    }

    /// <summary>
    /// User-facing ONNX export (issue #36, defect 1): where the rank IS statically
    /// known, the exported I/O ValueInfos carry rank-many dims — symbolic (dynamic)
    /// since only the rank is fixed — named after the tensor's exported name; a
    /// known rank-0 output is stamped as a true scalar (shape present, zero dims).
    /// Verified both at the proto level and through stock ONNX Runtime metadata.
    /// </summary>
    [Fact]
    public void TestVanillaExportStampsKnownRankDims()
    {
        var xs = TensorData([2L], 1f, 5f);
        var ys = TensorData([2L], 3f, 2f);
        var g = VectorMinMaxOthersBugPinCheck.ComputationGraph;   // Vector<float32> xs, ys → Scalar<bit>
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([xs, ys])).ToConcreteModel();

        var proto = Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(concrete);

        // Proto level: each rank-1 input has one symbolic dim named {name}_dim0; the
        // scalar output is stamped with a present-but-empty shape and the bool dtype.
        foreach (var name in (string[])["xs", "ys"])
        {
            var info = proto.Graph.Inputs.Single(x => x.Name == name);
            var shape = info.Type.TensorType.Shape;
            var dim = Assert.Single(shape.Dims);
            Assert.Equal($"{name}_dim0", dim.DimParam);
        }
        var outInfo = proto.Graph.Outputs.Single();
        Assert.Equal(DType.Bool.ProtoTypeNum, outInfo.Type.TensorType.ElemType);
        Assert.NotNull(outInfo.Type.TensorType.Shape);
        Assert.Empty(outInfo.Type.TensorType.Shape.Dims);

        // Stock ONNX Runtime reports the same: rank-1 dynamic inputs, scalar output.
        using var ms = new System.IO.MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(ms.ToArray());
        int[] dynamicRank1 = [-1];
        Assert.Equal(dynamicRank1, session.InputMetadata["xs"].Dimensions);
        Assert.Equal(dynamicRank1, session.InputMetadata["ys"].Dimensions);
        Assert.Equal(typeof(float), session.InputMetadata["xs"].ElementType);
        var outMeta = session.OutputMetadata[outInfo.Name];
        Assert.Equal(typeof(bool), outMeta.ElementType);
        Assert.Empty(outMeta.Dimensions);
    }

    /// <summary>
    /// User-facing ONNX export (issue #36, defect 2): a module-stage graph — still
    /// carrying Shorokoo-internal orchestration ops like <c>ShrkCreateModule</c> /
    /// <c>ShrkModelInvoke</c> — cannot be expressed in the vanilla dialect, so the
    /// export fails AT EXPORT TIME with the offending ops named and the fix
    /// (concretize first) spelled out, instead of silently writing a file that only
    /// Shorokoo can re-import. Shorokoo's own persistence
    /// (<c>CompressedFormatUtils</c>, the internal dialect) still accepts the very
    /// same graph unchanged.
    /// </summary>
    [Fact]
    public void TestModuleStageGraphFailsVanillaExportNamingOffendingOps()
    {
        var g = TwoStackLayer.ComputationGraph;
        var internalGraph = g.ToInternal();
        Assert.Contains(internalGraph.Nodes, n => n.OpCode == InternalOpCodes.CREATE_MODULE);

        // Public export surface: the reliable-kind FW045 gate refuses the module-stage
        // graph up front, spelling out the fix (concretize first).
        var kindEx = Assert.Throws<ModelException>(
            () => Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(g));
        Assert.Equal(ErrorCodes.FW045, kindEx.ErrorCode);
        Assert.Contains("ToConcreteModel", kindEx.Message);

        // Emission-side op-scan gate (internal-graph overload): the offending ops are
        // named and the fix is spelled out.
        var ex = Assert.Throws<ModelException>(
            () => Shorokoo.Core.Factory.FastOnnxModelBuilder.BuildOnnxModel(internalGraph));
        Assert.Equal(ErrorCodes.FW045, ex.ErrorCode);
        Assert.Contains(InternalOpCodes.CREATE_MODULE, ex.Message);
        Assert.Contains("ToConcreteModel", ex.Message);

        // The internal dialect remains available for the same module-stage graph.
        var data = CompressedFormatUtils.SaveFastGraphToBinary(g, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(data);
        Assert.Equal(internalGraph.Nodes.Count, reloaded.ToInternal().Nodes.Count);
        Assert.Contains(reloaded.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.CREATE_MODULE);
    }

    /// <summary>
    /// Drives <c>OnnxModelReader.CreateFastInputTensors</c>'s
    /// <c>GENERIC_TYPE_INPUT</c> branch by saving generic modules' raw
    /// <c>ComputationGraph</c>s without first running
    /// <c>FastChangeGenericTypeSpecialization</c> /
    /// <c>FastToConcreteDataType</c>. The unspecialized form serializes the
    /// generic placeholder DType through the input's <c>Denotation</c> field
    /// (e.g. <c>"T:FloatLike"</c>), which the loader parses back into a
    /// generic-paramed DType and routes to <c>InternalOp.GenericTypeInput</c>.
    /// <c>SimpleGenericLayer</c> covers the bare <c>T</c>/no-constraint case;
    /// <c>GenericComposedLayer</c> covers a generic module with constraints and
    /// nested generic-sub-module calls, so the function-side
    /// <c>CreateFastInputTensors</c> overload (line 637 onward) also fires.
    /// </summary>
    [Fact]
    public void TestModuleGraphGenericInputSaveLoadCoverage()
    {
        AssertGenericSaveLoadOnly<SimpleGenericLayer>();
        AssertGenericSaveLoadOnly<GenericComposedLayer>();
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

    /// <summary>
    /// Coverage for save/load of a TensorStruct passed to a sub-function.
    /// <c>RealGenericTensorStructSumCaller</c> constructs a
    /// <c>RealGenericPairStruct&lt;float32, float32&gt;</c> internally and forwards it
    /// to <c>RealGenericTensorStructSum</c>. The struct DType lives in the dynamic
    /// 2000-2999 range, so the saved <c>TENSOR_STRUCT_CREATE::dtype</c> attribute
    /// carries an iType that isn't in the static <see cref="DType"/> mapping.
    ///
    /// <para>The loader registers each struct from <c>shrk_tensorstruct_{protoTypeNum}</c>
    /// metadata at the ORIGINAL protoTypeNum via
    /// <see cref="DType.GetOrCreateForTensorStructAtProtoTypeNum"/>, and
    /// <c>op_Implicit(int)</c> consults the TensorStruct registry for iTypes in the
    /// 2000-2999 range. Both pieces are needed: without registration the saved iType
    /// would be reassigned to whatever the registry's next slot is, and without the
    /// implicit-conversion lookup the saved attribute would still trip
    /// <c>Debug.Fail</c>.</para>
    /// </summary>
    [Fact]
    public void TestModuleGraphTensorStructInFunctionInputCoverage()
    {
        AssertSaveLoadOnly<RealGenericTensorStructSumCaller>(
            hyperparamInputs: [],
            runtimeInputs: [],
            genericTypes: new() { ["T"] = DType.Float32 });
    }

    // -----------------------------------------------------------------------
    // Custom save/load test for the reload-time BuildFastFunctionInvokeNodeFromProto
    // branch. Not a [Module]-driven roundtrip: we construct the graph by hand
    // because no current Shorokoo flow emits a saved FUNCTION_INVOKE node whose
    // TargetFunction is not an initializer fn (regular function/module calls are
    // inlined by FastInlineModulesAndFunctions before any save, and the only
    // post-FastInitializeModelParams path produces FUNCTION_INVOKE nodes whose
    // TargetFunction is a TrainableParam/StateParam initializer — those route to
    // BuildFastTrainableParamNodeFromProto on reload, not the function-invoke
    // branch).
    //
    // To exercise that branch, we materialize a small graph against a
    // hand-built FunctionType.Module Function, save it, and reload it.
    // -----------------------------------------------------------------------

    /// <summary>Toy module body used by <see cref="TestFastFunctionInvokeNodeReload"/>.</summary>
    private static Tensor<float32> DoubleScalar(Tensor<float32> input) => input + input;

    /// <summary>
    /// Drives <see cref="Shorokoo.Core.Factory.IR.OnnxModelReader"/>'s
    /// <c>BuildFastFunctionInvokeNodeFromProto</c> by building a one-call graph
    /// against a hand-constructed <see cref="FunctionType.Module"/> Function.
    /// The save layer rewrites <c>FUNCTION_INVOKE</c>'s opcode to the function's
    /// default name (FastOpsetResolver), and on reload the loader's dispatch
    /// finds that name in <c>functionsMap</c>, sees the FunctionType is not an
    /// initializer, and routes to <c>BuildFastFunctionInvokeNodeFromProto</c> —
    /// the only path that hits that method.
    /// </summary>
    [Fact]
    public void TestFastFunctionInvokeNodeReload()
    {
        // Build a Module-typed Function from a delegate. ModuleHelper.CreateTargetFunction
        // defaults FunctionType to Module; only the explicit initializer flags
        // produce TrainableParam/StateParam initializer types.
        System.Func<Tensor<float32>, Tensor<float32>> impl = DoubleScalar;
        var fn = Shorokoo.Core.ModuleHelper.CreateTargetFunction(impl);
        Assert.Equal(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.Module, fn.FunctionType);

        // Outer graph: 1 model-tensor input → fn.Call(input) → output.
        var input = (Tensor<float32>)Shorokoo.Core.Nodes.NodeDefinitions.InternalOp.ModuleTensorInput(
            DType.Float32, rank: 1, Shorokoo.Core.Nodes.NodeDefinitions.InputType.ModelInput,
            targetFunction: null, defaultName: "input");
        var callResult = fn.Call(input);
        var output = (Tensor<float32>)callResult[0];

        var graph = new InternalComputationGraph(
            System.Collections.Immutable.ImmutableArray.Create<Shorokoo.Core.Variable>(input),
            System.Collections.Immutable.ImmutableArray.Create<Shorokoo.Core.Variable>(output));

        // Pre-save sanity: the graph carries the FUNCTION_INVOKE node pointing at fn.
        Assert.Single(graph.Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        var preInvoke = graph.Nodes.Single(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        Assert.Same(fn, preInvoke.TargetFunction);

        // Save → load. (.ToInternal(): post-load node inspection only, no mutation.)
        var data = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(data).ToInternal();

        // Post-load: BuildFastFunctionInvokeNodeFromProto produced a fresh
        // FUNCTION_INVOKE FastNode with a rebuilt TargetFunction (same default
        // name; new Function instance per the post-pass in FastUnPrepFromOnnx).
        Assert.Single(reloaded.Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        var postInvoke = reloaded.Nodes.Single(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        Assert.NotNull(postInvoke.TargetFunction);
        Assert.Equal(fn.DefaultName, postInvoke.TargetFunction!.DefaultName);
        Assert.Equal(Shorokoo.Core.Nodes.OnnxNodes.FunctionType.Module, postInvoke.TargetFunction.FunctionType);
        // The reloaded FUNCTION_INVOKE carries its function's output signature on
        // the ShrkAttrDtype attribute — the attribute population block in
        // BuildFastFunctionInvokeNodeFromProto fills it from the function's
        // OriginalFastGraph outputs.
        var dtypes = postInvoke.Attributes.GetDTypesVal(OnnxOpAttributeNames.ShrkAttrDtype)!;
        Assert.Equal(new[] { DType.Float32 }, dtypes);
        // Single output slot — that's the canonical FUNCTION_INVOKE layout when
        // the loader walks the function's declared outputs.
        Assert.Single(postInvoke.Outputs);
    }

    private static void AssertSaveLoadOnly<TModule>(
        TensorData[] hyperparamInputs,
        TensorData[] runtimeInputs,
        System.Collections.Generic.Dictionary<string, DType>? genericTypes = null)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        // ToInternal(): the generated property returns a shared readonly ComputationGraph;
        // the specialization processors below mutate, so work on a mutable deep copy.
        var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();

        if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
        {
            if (genericTypes is not null && genericTypes.Count > 0)
                Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
            moduleGraph = Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType.Process(moduleGraph);
        }

        var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
        moduleGraph = CompressedFormatUtils.LoadFastGraphFromBinary(data).ToInternal();

        var allInputs = new System.Collections.Generic.List<TensorData>();
        allInputs.AddRange(hyperparamInputs);
        allInputs.AddRange(runtimeInputs);

        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));
        var archData = CompressedFormatUtils.SaveFastGraphToBinary(concreteArch, compressed: true);
        concreteArch = CompressedFormatUtils.LoadFastGraphFromBinary(archData).ToInternal();

        var concreteModel = concreteArch.ToConcreteModel();
        var modelData = CompressedFormatUtils.SaveFastGraphToBinary(concreteModel, compressed: true);
        var reloadedModel = CompressedFormatUtils.LoadFastGraphFromBinary(modelData);

        Assert.NotEmpty(reloadedModel.ToInternal().Nodes);
        Assert.NotEmpty(reloadedModel.ToInternal().Outputs);
    }

    /// <summary>
    /// Generic <c>[Module]</c> coverage. Building a generic-method module's
    /// <c>ComputationGraph</c> routes through
    /// <see cref="Shorokoo.Modules.GraphBuilder.BuildInternalComputationGraphFromMethod"/> →
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastConvertPlaceholderGenericTypesToDefaultGenericTypes.Process"/>,
    /// which rewrites <c>IGenericType1..8</c> placeholder DTypes on DType / DTypes / Tensor
    /// attributes to default concrete-typed DTypes carrying the source generic-param name.
    /// Driving the graph through the AutoTester then exercises
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType"/>'s
    /// generic-function specialization + GENERIC_TYPE_INPUT removal pipeline.
    /// </summary>
    [Fact]
    public void TestGenericModulesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimpleGenericLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<GenericScaleLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<GenericAddLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L]), TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<GenericComposedLayer>(
            hyperparamInputs: [TensorData(DType.Float32, [], 2f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
    }

    /// <summary>
    /// Exercises the caller-supplied generic specialization path in
    /// <see cref="AutoTest.AdvancedTestGraph"/>: when <c>genericTypes</c> is non-null
    /// the helper invokes <c>FastChangeGenericTypeSpecialization.Process</c> before
    /// <c>FastToConcreteDataType.Process</c>, replacing the default Float32 baked in
    /// at build time with the caller-chosen concrete type (Float64 here). The default
    /// generic path (above) only reaches <c>FastChangeGenericTypeSpecialization</c>
    /// transitively via nested call-site specialization in composed modules.
    /// </summary>
    [Fact]
    public void TestGenericModulesUserSpecializationCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SimpleGenericLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 }));
        Assert.True(AutoTest.AdvancedTestGraph<GenericScaleLayer>(
            hyperparamInputs: [TensorData(DType.Float64, [], 2.0)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 }));
        // AddThree builds a Scalar<T>((ushort)3) constant — exercises the Tensor
        // attribute arm of FastChangeGenericTypeSpecialization.RewriteNodeAttributes
        // (TensorData with generic-typed element type).
        Assert.True(AutoTest.AdvancedTestGraph<AddThree>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 }));
        // GenericConstantOfShapeLayer uses TensorFill<T> — also Tensor attribute.
        Assert.True(AutoTest.AdvancedTestGraph<GenericConstantOfShapeLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Int64, [1L], 5L)],
            genericTypes: new() { ["T"] = DType.Float64 }));
        // GenericBlackmanWindowLayer carries an output_datatype DType attribute.
        Assert.True(AutoTest.AdvancedTestGraph<GenericBlackmanWindowLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Int64, [], 8L)],
            genericTypes: new() { ["T"] = DType.Float64 }));
        // GenericCastLayer<TIn,TOut> — Cast op carries a 'to' DType attribute that
        // must be specialized (different in/out types).
        Assert.True(AutoTest.AdvancedTestGraph<GenericCastLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["TIn"] = DType.Float64, ["TOut"] = DType.Float32 }));
        // Multi-type-param module — exercises the Process loop with 3 keys (T, Q, R).
        Assert.True(AutoTest.AdvancedTestGraph<GenericThreeTypeParamLayer>(
            hyperparamInputs: [TensorData(DType.Float64, [], 1.0), TensorData(DType.Int32, [], 2), TensorData(DType.Int32, [], 3)],
            runtimeInputs: [TensorData(DType.Float64, [], 4.0), TensorData(DType.Float32, [], 5f), TensorData(DType.Float32, [], 6f)],
            genericTypes: new() { ["T"] = DType.Float64, ["Q"] = DType.Int32, ["R"] = DType.Float32 }));
        // GenericAddLayer with Float64.
        Assert.True(AutoTest.AdvancedTestGraph<GenericAddLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L]), TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 }));
        // GenericComposedLayer with Float64 — composed module exercises nested
        // call-site specialization (MODEL_INVOKE with ShrkAttrGenericTypeArgs).
        Assert.True(AutoTest.AdvancedTestGraph<GenericComposedLayer>(
            hyperparamInputs: [TensorData(DType.Float64, [], 2.0)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float64, [5L])],
            genericTypes: new() { ["T"] = DType.Float64 }));
    }

    /// <summary>
    /// Coverage for the REF-family attribute emission in
    /// <c>InternalOp.TrainableParamRef</c> (and its two REF-family siblings,
    /// <c>TrainableParamModelRef</c> and <c>TrainableParamIdRef</c> in
    /// <c>InternalOp.cs</c>): the <c>ShrkAttrFunctionName</c> +
    /// <c>ShrkAttrDomainName</c> attributes must be written so that
    /// <c>FastOpsetResolver.SetExternalEnvForOnnx</c> emits
    /// <c>shrk_function_name</c> on save and <c>OnnxModelReader.CreateNodes</c>
    /// rebuilds the node's <c>TargetFunction</c> on reload — required by the
    /// <c>TargetFunction?.FunctionType ∈ {TrainableParamInitializer,
    /// StateParamInitializer}</c> invariant enforced in <c>Node</c>'s constructor.
    ///
    /// <para>
    /// Test shape: a non-concrete graph from a [Module] that calls another
    /// [Module] (<see cref="CallsSimplestModule"/> →
    /// <c>SimplestLayer.Call(x)</c>) contains a <c>MODEL_INVOKE</c> whose
    /// function body references a <c>MODEL_PARAM_REF</c> via
    /// <c>InitSimple</c>. Saving and reloading round-trips the graph through
    /// the ONNX <c>FunctionProto</c> path, exercising the REF-family
    /// attribute emission.
    /// </para>
    /// </summary>
    [Fact]
    public void TestModuleOnModuleTrainableParamRefFunctionLinkCoverage()
    {
        var moduleGraph = CallsSimplestModule.ComputationGraph;
        Assert.Contains(moduleGraph.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);

        // Save the non-concrete graph to ONNX bytes and reload; the reloaded
        // MODEL_PARAM_REF inside SimplestLayer's nested function body must
        // carry a non-null TargetFunction.
        var binary = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(binary);

        // The top-level MODEL_INVOKE survives the round-trip — both saver and
        // reader keep this op-code as-is. The SimplestLayer function body
        // (with the now-correctly-tagged MODEL_PARAM_REF) is held inside
        // the reloaded MODEL_INVOKE's TargetFunction.
        Assert.Contains(reloaded.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);

        // Running ToConcreteArchitecture on the reloaded graph forces
        // FastInlineModulesAndFunctions to inline the MODEL_INVOKE — this
        // pulls in SimplestLayer's body, which contains MODEL_PARAM_REF
        // nodes with their TargetFunction wired to InitSimple.
        TensorData[] sampleInputs = [TensorDataWithSmallVals(DType.Float32, [5L])];
        var concreteArch = reloaded.ToConcreteArchitecture(reloaded.FromOrderedInputs([.. sampleInputs]));

        // Inlining + the rest of ToConcreteArchitecture's pipeline must
        // have removed every MODULE_INVOKE / FUNCTION_INVOKE.
        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);
        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
    }


    /// <summary>
    /// Coverage for <see cref="InternalComputationGraphExtensions.Specialize"/>: bakes a
    /// partial set of named input values into constants and runs
    /// <c>FastSimplify</c> to fold them through the graph. Verifies full
    /// specialization removes all named inputs, partial specialization removes only
    /// the supplied ones, the original graph is not modified, and results are
    /// numerically identical to executing the unspecialized graph with the same values.
    /// </summary>
    [Fact]
    public void TestSpecializeCoverage()
    {
        var factor = TensorData(DType.Float32, [], 2f);
        var bias   = TensorData(DType.Float32, [], 0.5f);
        var input  = TensorDataWithSmallVals(DType.Float32, [5L]);

        var moduleGraph  = HypersLayer.ComputationGraph;
        TensorData[] allHints = [factor, bias, input];
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allHints]));
        var model        = concreteArch.ToConcreteModel();
        int originalInputCount = model.ToInternal().Inputs.Count;

        // Full specialization: bake both hyperparams in.
        var specialized = model.Specialize(model.FromOrderedInputs([factor, bias]));
        Assert.Equal(originalInputCount - 2, specialized.ToInternal().Inputs.Count);
        Assert.Equal(originalInputCount, model.ToInternal().Inputs.Count); // original unchanged

        var expected = ComputeContext.Default.Execute(model, factor, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        var actual   = ComputeContext.Default.Execute(specialized, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, actual);

        // Partial specialization: bake in only the first hyperparam.
        var partialHints = new ModelParamList([
            new TensorDataModelParam(model.InputNames[0]!, ModelParamType.InputParam, factor)
        ]);
        var partial = model.Specialize(partialHints);
        Assert.Equal(originalInputCount - 1, partial.ToInternal().Inputs.Count);
        var partialActual = ComputeContext.Default.Execute(partial, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, partialActual);
    }

    /// <summary>
    /// Coverage for the documented <c>Specialize</c> → <c>ToConcreteArchitecture</c> →
    /// <c>ToConcreteModel</c> pipeline (see <c>Documentation/inference.md</c>): baking the
    /// <c>[Hyper]</c> inputs into the raw module graph up front drops them from the input
    /// list, so the concretized model runs on the bare runtime input alone — with results
    /// identical to the legacy flow that concretizes with all inputs as hints and re-supplies
    /// the hyper values at execution time. Covers both a value-only hyper (<c>HypersLayer</c>)
    /// and a shape-determining hyper that feeds a trainable-parameter shape
    /// (<c>FCLayer</c>'s <c>numOutFeatures</c>) — the case the docs' <c>Dense</c>/<c>outFeatures</c>
    /// example uses, where the baked constant must still resolve the param shape during
    /// architecture lowering.
    /// </summary>
    [Fact]
    public void TestSpecializeThenConcretizePipelineCoverage()
    {
        var factor = TensorData(DType.Float32, [], 2f);
        var bias   = TensorData(DType.Float32, [], 0.5f);
        var input  = TensorDataWithSmallVals(DType.Float32, [5L]);

        var graph = HypersLayer.ComputationGraph;     // inputs: factor, bias, input

        // Specialize first (bake the two hypers), then concretize on the remaining input.
        var specialized = graph.Specialize(graph.FromOrderedInputs([factor, bias]));
        Assert.Equal(["input"], specialized.InputNames);

        var concrete = specialized
            .ToConcreteArchitecture(specialized.FromOrderedInputs([input]))
            .ToConcreteModel();
        Assert.Single(concrete.ToInternal().Inputs);

        var actual = ComputeContext.Default.Execute(concrete, input)[0].ToTensorData().AccessRawMemory().ToArray();

        var refConcrete = graph.ToConcreteArchitecture(graph.FromOrderedInputs([factor, bias, input])).ToConcreteModel();
        var expected = ComputeContext.Default.Execute(refConcrete, factor, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, actual);

        // Shape-determining hyper: numOutFeatures feeds InitSimple.Init([numOutFeatures, numInFeatures]).
        var numOut  = TensorData(DType.Int64, [], 3L);
        var fcInput = TensorDataWithSmallVals(DType.Float32, [2L, 5L]);
        var fcGraph = FCLayer.ComputationGraph;       // inputs: numOutFeatures, input

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

    /// <summary>Analytic op-semantics checks from the 2026-06-12 campaign — int64
    /// trunc-toward-zero division, negative-step Slice, negative Gather indices,
    /// empty-tensor ReduceSum, large-logit softmax stability. The exact expected
    /// values live in <see cref="AnalyticOpSemanticsCheck"/>.</summary>
    [Fact]
    public void TestOpSemanticsAnalyticChecks()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticOpSemanticsCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [3L], 10f, 20f, 30f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticNaNMismatchGuardCheck>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)]));
    }

    /// <summary>Analytic control-flow checks from the 2026-06-12 campaign: IfElse must
    /// not leak NaN from the unselected branch; a runtime-trip-count loop computes 8·x
    /// for 3 trips and is the identity for 0 trips; and the loop-bearing concrete graph
    /// survives an ONNX export → import roundtrip with bit-identical results (the
    /// .zsrk roundtrip is already exercised inside AdvancedTestGraph itself).</summary>
    [Fact]
    public void TestControlFlowAnalyticChecks()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticIfElseNaNIsolationCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], 1f, 2f, 3f, 4f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticLoopAccumulateCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [4L], 3f, 4f, 5f, 6f)]));
        Assert.True(AutoTest.AdvancedTestGraph<AnalyticLoopZeroTripCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [3L], 0f, 5f, 7f)]));

        // ONNX proto export → import roundtrip of the runtime-trip-count loop graph.
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
        Assert.Equal(1, direct[0]);   // the self-check bit itself
    }

    /// <summary>
    /// Issue #54: the graph kind is stamped by every producing path, checked
    /// fail-fast by the lowering entry points with an error naming the actual and
    /// required kinds, and cannot be invalidated through the readonly wrapper —
    /// conversions between ComputationGraph and InternalComputationGraph copy.
    /// </summary>
    [Fact]
    public void TestGraphKindStampingChecksAndCopySemanticsCoverage()
    {
        var sample = TensorData([2L], 1.0f, 2.0f);

        // Stamping along the lowering pipeline.
        var moduleGraph = ScalarMultiplyModel.ComputationGraph;
        Assert.Equal(GraphKind.Module, moduleGraph.Kind);

        var arch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([sample]));
        Assert.Equal(GraphKind.ConcreteArchitecture, arch.Kind);

        var model = arch.ToConcreteModel();
        Assert.Equal(GraphKind.ConcreteModel, model.Kind);

        // Specialize preserves the kind (nothing to bake here — empty value set).
        Assert.Equal(GraphKind.ConcreteModel, model.Specialize(new ModelParamList([])).Kind);

        // Kind checks fail fast, naming the actual and required kinds.
        var exArch = Assert.Throws<InvalidOperationException>(
            () => arch.ToConcreteArchitecture(arch.FromOrderedInputs([sample])));
        Assert.Contains("'module'", exArch.Message);
        Assert.Contains("'concrete-architecture'", exArch.Message);

        var exModule = Assert.Throws<InvalidOperationException>(() => moduleGraph.ToConcreteModel());
        Assert.Contains("'concrete-architecture'", exModule.Message);
        Assert.Contains("'module'", exModule.Message);

        // Re-lowering an already-concrete model is refused too.
        var exTwice = Assert.Throws<InvalidOperationException>(() => model.ToConcreteModel());
        Assert.Contains("'concrete-model'", exTwice.Message);

        // The param-query surface is kind-gated the same way.
        var exInfos = Assert.Throws<InvalidOperationException>(() => moduleGraph.GetConcreteModelParamInfos());
        Assert.Contains("'concrete-architecture'", exInfos.Message);

        // Execution is kind-gated too: a module graph fails fast with the lowering
        // hint instead of dying deep inside session creation.
        var exExec = Assert.Throws<InvalidOperationException>(
            () => ComputeContext.Default.Execute(moduleGraph, sample));
        Assert.Contains("concretized", exExec.Message);
        Assert.Contains("'module'", exExec.Message);

        // Conversions copy: mutating a ToInternal() copy never affects the wrapper…
        var nodeCount = model.ToInternal().Nodes.Count;
        var copy = model.ToInternal();
        copy.Nodes.Clear();
        Assert.Equal(nodeCount, model.ToInternal().Nodes.Count);

        // …and FromInternal freezes a copy, so later source mutation cannot
        // invalidate the stamp behind the frozen wrapper.
        var source = model.ToInternal();
        var frozen = ComputationGraph.FromInternal(source, GraphKind.ConcreteModel);
        source.Nodes.Clear();
        Assert.Equal(nodeCount, frozen.ToInternal().Nodes.Count);
        Assert.Equal(GraphKind.ConcreteModel, frozen.Kind);

        // FromInternal without an explicit kind falls back to op-scan classification.
        Assert.Equal(GraphKind.ConcreteModel, ComputationGraph.FromInternal(model.ToInternal()).Kind);

        // The kind-stamped model still executes as before.
        var outputs = ComputeContext.Default.Execute(model, sample);
        Assert.Single(outputs);
    }


    private static Tensor<float32> MachineryFreeBody(Tensor<float32> x) => x + x;

    /// <summary>
    /// WithKind re-stamps a graph when the target kind is structurally valid for its
    /// content, and refuses impossible stamps naming the violated requirement. The
    /// valid path is the escape hatch for op-scan misclassification: a machinery-free
    /// module body op-scans as concrete-model, and WithKind restores the true kind.
    /// </summary>
    [Fact]
    public void TestWithKindReStampValidationCoverage()
    {
        var sample = TensorData([2L], 1.0f, 2.0f);
        var moduleGraph = ScalarMultiplyModel.ComputationGraph;
        var arch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([sample]));
        var model = arch.ToConcreteModel();

        // Same-kind re-stamp is the identity.
        Assert.Same(moduleGraph, moduleGraph.WithKind(GraphKind.Module));

        // Valid: an architecture carries no initialized parameters, so Module is legal.
        Assert.Equal(GraphKind.Module, arch.WithKind(GraphKind.Module).Kind);

        // Invalid: a module graph's parameter space is not statically known…
        var exToArch = Assert.Throws<InvalidOperationException>(
            () => moduleGraph.WithKind(GraphKind.ConcreteArchitecture));
        Assert.Contains("module-stage op", exToArch.Message);

        // …an architecture's parameters are not initialized…
        var exToModel = Assert.Throws<InvalidOperationException>(
            () => arch.WithKind(GraphKind.ConcreteModel));
        Assert.Contains("unmaterialized", exToModel.Message);

        // …and a concrete model's parameters ARE initialized, so neither earlier kind fits.
        var exBackToArch = Assert.Throws<InvalidOperationException>(
            () => model.WithKind(GraphKind.ConcreteArchitecture));
        Assert.Contains("initialized", exBackToArch.Message);
        var exBackToModule = Assert.Throws<InvalidOperationException>(
            () => model.WithKind(GraphKind.Module));
        Assert.Contains("initialized", exBackToModule.Message);

        // Misclassification escape hatch: a machinery-free module body op-scans as
        // concrete-model when no stamp is available; WithKind(Module) makes it lowerable.
        var misStamped = ComputationGraph.FromInternal(
            ModuleFactory.ComputationGraph((Func<Tensor<float32>, Tensor<float32>>)MachineryFreeBody)
                .ToInternal());
        Assert.Equal(GraphKind.ConcreteModel, misStamped.Kind);
        var reStamped = misStamped.WithKind(GraphKind.Module);
        var relowered = reStamped.ToConcreteArchitecture(reStamped.FromOrderedInputs([sample]));
        Assert.Equal(GraphKind.ConcreteArchitecture, relowered.Kind);
    }

}
