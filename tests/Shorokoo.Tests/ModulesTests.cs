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
    /// trainable params are materialized as constants). Driving these modules
    /// through the moduleGraph roundtrip exercises
    /// <c>OnnxModelReader.BuildFastFunctionInvokeNodeFromProto</c>,
    /// <c>OnnxModelReader.BuildFastTrainableParamNodeFromProto</c>, and the legacy
    /// <c>internalBuildFunctions</c> / <c>internalInitFunction</c> /
    /// <c>CreateNodes</c> FunctionProto build path — load-time paths that fire
    /// only when the saved graph still contains <c>FUNCTION_INVOKE</c> /
    /// <c>SEQUENCE_CONSTRUCT</c> with a TargetFunction / unmaterialized
    /// <c>TRAINABLE_PARAM</c> references. <c>SimpleModelSequence</c> covers the
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
        // TRAINABLE_PARAM_REF's shape vector from function-input hyperparams
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
    /// and <c>ParseIsTrainableMetadata</c>: state params materialize as
    /// <c>MODEL_PARAM_DATA</c> nodes (per <c>FastApplyModelParamValues</c> line 49)
    /// and serialize as ONNX <c>graphProto.Initializers</c> tensors, which the
    /// trainable-only modules above don't produce (those materialize as inline
    /// <c>Constant</c> ops instead).
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
        var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;
        Assert.Contains(moduleGraph.Nodes, n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT);

        var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);
        Assert.Contains(reloaded.Nodes, n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT);
        Assert.Equal(moduleGraph.Nodes.Count, reloaded.Nodes.Count);
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

        var graph = new FastComputationGraph(
            System.Collections.Immutable.ImmutableArray.Create<Shorokoo.Core.Variable>(input),
            System.Collections.Immutable.ImmutableArray.Create<Shorokoo.Core.Variable>(output));

        // Pre-save sanity: the graph carries the FUNCTION_INVOKE node pointing at fn.
        Assert.Single(graph.Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        var preInvoke = graph.Nodes.Single(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
        Assert.Same(fn, preInvoke.TargetFunction);

        // Save → load.
        var data = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);

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
        var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;

        if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
        {
            if (genericTypes is not null && genericTypes.Count > 0)
                Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
            moduleGraph = Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType.Process(moduleGraph);
        }

        var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
        moduleGraph = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);

        var allInputs = new System.Collections.Generic.List<TensorData>();
        allInputs.AddRange(hyperparamInputs);
        allInputs.AddRange(runtimeInputs);

        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));
        var archData = CompressedFormatUtils.SaveFastGraphToBinary(concreteArch, compressed: true);
        concreteArch = CompressedFormatUtils.LoadFastGraphFromBinary(archData, isCompressed: true);

        var concreteModel = concreteArch.ToConcreteModel();
        var modelData = CompressedFormatUtils.SaveFastGraphToBinary(concreteModel, compressed: true);
        var reloadedModel = CompressedFormatUtils.LoadFastGraphFromBinary(modelData, isCompressed: true);

        Assert.NotEmpty(reloadedModel.Nodes);
        Assert.NotEmpty(reloadedModel.Outputs);
    }

    /// <summary>
    /// Generic <c>[Module]</c> coverage. Building a generic-method module's
    /// <c>ComputationGraph</c> routes through
    /// <see cref="Shorokoo.Modules.GraphBuilder.BuildFastComputationGraphFromMethod"/> →
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
    /// function body references a <c>TRAINABLE_PARAM_REF</c> via
    /// <c>InitSimple</c>. Saving and reloading round-trips the graph through
    /// the ONNX <c>FunctionProto</c> path, exercising the REF-family
    /// attribute emission.
    /// </para>
    /// </summary>
    [Fact]
    public void TestModuleOnModuleTrainableParamRefFunctionLinkCoverage()
    {
        var moduleGraph = CallsSimplestModule.ComputationGraph;
        Assert.Contains(moduleGraph.Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);

        // Save the non-concrete graph to ONNX bytes and reload; the reloaded
        // TRAINABLE_PARAM_REF inside SimplestLayer's nested function body must
        // carry a non-null TargetFunction.
        var binary = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph);
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(binary);

        // The top-level MODEL_INVOKE survives the round-trip — both saver and
        // reader keep this op-code as-is. The SimplestLayer function body
        // (with the now-correctly-tagged TRAINABLE_PARAM_REF) is held inside
        // the reloaded MODEL_INVOKE's TargetFunction.
        Assert.Contains(reloaded.Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);

        // Running ToConcreteArchitecture on the reloaded graph forces
        // FastInlineModulesAndFunctions to inline the MODEL_INVOKE — this
        // pulls in SimplestLayer's body, which contains TRAINABLE_PARAM_REF
        // nodes with their TargetFunction wired to InitSimple.
        var sampleInputs = new[] { TensorDataWithSmallVals(DType.Float32, [5L]) };
        var concreteArch = reloaded.ToConcreteArchitecture(reloaded.FromOrderedInputs([.. sampleInputs]));

        // Inlining + the rest of ToConcreteArchitecture's pipeline must
        // have removed every MODULE_INVOKE / FUNCTION_INVOKE.
        Assert.DoesNotContain(concreteArch.Nodes, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);
        Assert.DoesNotContain(concreteArch.Nodes, n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE);
    }


    /// <summary>
    /// Coverage for <see cref="FastComputationGraphExtensions.Specialize"/>: bakes a
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
        var allHints     = new[] { factor, bias, input };
        var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allHints]));
        var model        = concreteArch.ToConcreteModel();
        int originalInputCount = model.Inputs.Count;

        // Full specialization: bake both hyperparams in.
        var specialized = model.Specialize(model.FromOrderedInputs([factor, bias]));
        Assert.Equal(originalInputCount - 2, specialized.Inputs.Count);
        Assert.Equal(originalInputCount, model.Inputs.Count); // original unchanged

        var expected = ComputeContext.Default.Execute(model, factor, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        var actual   = ComputeContext.Default.Execute(specialized, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, actual);

        // Partial specialization: bake in only the first hyperparam.
        var partialHints = new ModelParamList([
            new TensorDataModelParam(model.InputUniqueNames[0]!, ModelParamType.InputParam, factor)
        ]);
        var partial = model.Specialize(partialHints);
        Assert.Equal(originalInputCount - 1, partial.Inputs.Count);
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
        Assert.Equal(["input"], specialized.InputUniqueNames);

        var concrete = specialized
            .ToConcreteArchitecture(specialized.FromOrderedInputs([input]))
            .ToConcreteModel();
        Assert.Single(concrete.Inputs);

        var actual = ComputeContext.Default.Execute(concrete, input)[0].ToTensorData().AccessRawMemory().ToArray();

        var refConcrete = graph.ToConcreteArchitecture(graph.FromOrderedInputs([factor, bias, input])).ToConcreteModel();
        var expected = ComputeContext.Default.Execute(refConcrete, factor, bias, input)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(expected, actual);

        // Shape-determining hyper: numOutFeatures feeds InitSimple.Init([numOutFeatures, numInFeatures]).
        var numOut  = TensorData(DType.Int64, [], 3L);
        var fcInput = TensorDataWithSmallVals(DType.Float32, [2L, 5L]);
        var fcGraph = FCLayer.ComputationGraph;       // inputs: numOutFeatures, input

        var fcSpecialized = fcGraph.Specialize(fcGraph.FromOrderedInputs([numOut]));
        Assert.Equal(["input"], fcSpecialized.InputUniqueNames);

        var fcConcrete = fcSpecialized
            .ToConcreteArchitecture(fcSpecialized.FromOrderedInputs([fcInput]))
            .ToConcreteModel();
        Assert.Single(fcConcrete.Inputs);

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
        var reimported = OnnxModelImporter.FromOnnxModelToFastGraph(ms.ToArray());
        var roundtrip = ComputeContext.Default.Execute(reimported, x)[0].ToTensorData().AccessRawMemory().ToArray();
        Assert.Equal(direct, roundtrip);
        Assert.Equal(1, direct[0]);   // the self-check bit itself
    }
}
