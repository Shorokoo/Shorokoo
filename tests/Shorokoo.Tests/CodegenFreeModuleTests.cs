using System.Runtime.InteropServices;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose parity tests for the codegen-free module path
/// (<see cref="ModuleFactory"/> / <c>GraphBuilder.BuildFastComputationGraphFromDelegate</c>):
/// the same small models are built once from a <c>[Module]</c> class and once from a plain
/// delegate, and must execute identically; the delegate-built graphs must also train through
/// <see cref="TrainingRig"/>, export to ONNX via <c>FastOnnxModelBuilder</c>, and survive the
/// <see cref="AutoTest"/> roundtrips, with trainable-param initializers, <c>[Hyper]</c>
/// parameters, and <c>Globals.StateUpdate</c> behaving exactly as in <c>Inline</c> methods.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class CodegenFreeModuleTests
{
    // ───────────────────────── codegen-free module bodies ─────────────────────────
    // Static methods with flattened parameters — the exact shape of a [Module] Inline
    // method, minus the partial class + attribute.

    /// <summary>Mirror of <see cref="SimplestLayer"/>.Inline, written as a plain static method.</summary>
    private static Tensor<float32> SimplestBody(Tensor<float32> input)
    {
        var weights = InitSimple.Init(input.ShapeTensor());
        return input * weights;
    }

    /// <summary>Hyperparam-bearing body: y = x * (w * factor + bias) with w initialized to 1.</summary>
    private static Tensor<float32> ScaleAndShiftBody(
        Tensor<float32> input, [Hyper] Scalar<float32> factor, [Hyper] Scalar<float32> bias)
    {
        var weights = InitSimple.Init(input.ShapeTensor());
        return input * (weights * factor + bias);
    }

    /// <summary>Mirror of <see cref="ScalarMultiplyModel"/>.Inline — the canonical trainable model.</summary>
    private static Tensor<float32> ScalarMultiplyBody(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        return input * weight;
    }

    /// <summary>Two-runtime-input body for the multi-input FromFunc overloads.</summary>
    private static Tensor<float32> WeightedSumBody(Tensor<float32> a, Tensor<float32> b)
    {
        var weights = InitSimple.Init(a.ShapeTensor());
        return (a + b) * weights;
    }

    /// <summary>State-initializer + Globals.StateUpdate inside a delegate body.</summary>
    private static Tensor<float32> StatefulBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var updated = state + Scalar(1f);
        Globals.StateUpdate(state, updated);
        return input * Scalar(2f);
    }

    /// <summary>StateUpdate misuse: targets a plain runtime input instead of a state variable.</summary>
    private static Tensor<float32> StateUpdateOnInputBody(Tensor<float32> input)
    {
        Globals.StateUpdate(input, input + Scalar(1f));
        return input * Scalar(2f);
    }

    /// <summary>StateUpdate misuse: targets a trainable parameter instead of a state variable.</summary>
    private static Tensor<float32> StateUpdateOnTrainableBody(Tensor<float32> input)
    {
        var weight = InitSimple.Init(input.ShapeTensor());
        Globals.StateUpdate(weight, weight + Scalar(1f));
        return input * weight;
    }

    /// <summary>Correct StateUpdate through the Identity node a .Vec() rank-cast inserts.</summary>
    private static Tensor<float32> StateUpdateThroughVecBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L)).Vec();
        Globals.StateUpdate(state, state + Scalar(1f));
        return input * Scalar(2f);
    }

    // ───────────────────────────────── helpers ─────────────────────────────────

    /// <summary>Concretizes a module graph with the given ordered inputs and executes it.</summary>
    private static byte[][] ExecuteConcretized(FastComputationGraph moduleGraph, params TensorData[] inputs)
    {
        var concreteModel = moduleGraph
            .ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. inputs]))
            .ToConcreteModel();
        return ComputeContext.Default.Execute(concreteModel, (IData[])inputs)
            .Select(x => x.ToTensorData().AccessRawMemory().ToArray())
            .ToArray();
    }

    /// <summary>Wraps a constant-fed module output into a no-input graph, concretizes, executes.</summary>
    private static byte[][] ExecuteOutputs(params Variable[] outputs)
        => ExecuteConcretized(new FastComputationGraph([], [.. outputs]));

    // ─────────────────────────────────── tests ───────────────────────────────────

    /// <summary>
    /// Deliverable parity check: the same model built via the <c>[Module]</c> source generator
    /// (<see cref="SimplestLayer"/>.ComputationGraph), via a static method group, and via a
    /// non-capturing static lambda must produce byte-identical execution results. The lambda
    /// variant proves the delegate-target invoke path (compiler lambdas are instance methods
    /// on a display-class singleton, not static methods).
    /// </summary>
    [Fact]
    public void TestFromFuncGraphMatchesModuleClassExecution()
    {
        var input = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });

        var codegen = ExecuteConcretized(SimplestLayer.ComputationGraph, input);
        var methodGroup = ExecuteConcretized(
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)SimplestBody, "CodegenFreeSimplest"),
            input);
        var lambda = ExecuteConcretized(
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)(static x => x * InitSimple.Init(x.ShapeTensor()))),
            input);

        Assert.Equal(codegen.Length, methodGroup.Length);
        for (int i = 0; i < codegen.Length; i++)
        {
            Assert.Equal(codegen[i], methodGroup[i]);
            Assert.Equal(codegen[i], lambda[i]);
        }
    }

    /// <summary>
    /// Parity on the Model/Call invocation path: <c>ModuleFactory.FromFunc(...).SetHyperparams()
    /// .Call(x)</c> (the codegen-free spelling of the generated <c>Foo.Model().Call(x)</c>) must
    /// execute identically to the generated members. Also exercises the multi-input tuple
    /// overload via the <c>Model&lt;T1, T2, TOut&gt;</c> two-argument Call binding.
    /// </summary>
    [Fact]
    public void TestFromFuncModelCallPathParity()
    {
        // Single input: generated SimplestLayer.Model().Call vs FromFunc path.
        var viaCodegen = SimplestLayer.Model().Call(Tensor([4L], 1f, 2f, 3f, 4f));
        var viaFactory = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(SimplestBody)
            .SetHyperparams()
            .Call(Tensor([4L], 1f, 2f, 3f, 4f));

        Assert.Equal(ExecuteOutputs(viaCodegen)[0], ExecuteOutputs(viaFactory)[0]);

        // Two inputs: FromFunc<T1, T2, TOut> with a Model<T1, T2, TOut> for a 2-arg Call.
        var pairModule = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>, Tensor<float32>>(WeightedSumBody);
        var pairModel = pairModule.SetHyperparams<Model<Tensor<float32>, Tensor<float32>, Tensor<float32>>>();
        var pairOut = pairModel.Call(Tensor([3L], 1f, 2f, 3f), Tensor([3L], 10f, 20f, 30f));

        // weights init to 1 → (a + b) * 1.
        var expected = TensorData([3L], new float[] { 11f, 22f, 33f }).AccessRawMemory().ToArray();
        Assert.Equal(expected, ExecuteOutputs(pairOut)[0]);
    }

    /// <summary>
    /// FromFunc module WITH hyperparameters: model creation binds the hypers via
    /// <c>SetHyperparams((factor, bias))</c> and the bound values flow through the call —
    /// y = x * (1 * 2 + 0.5) = 2.5x. The hyper split comes from the [Hyper] attributes on the
    /// delegate's parameters, including on an explicitly-typed lambda's parameters. The hypered
    /// module graph also runs the full AutoTest ONNX/CS/QEE roundtrip.
    /// </summary>
    [Fact]
    public void TestFromFuncWithHypersModelCreationAndCall()
    {
        var module = ModuleFactory.FromFuncWithHypers<Tensor<float32>, Scalar<float32>, Scalar<float32>, Tensor<float32>>(
            ScaleAndShiftBody, "CodegenFreeScaleAndShift");
        var model = module.SetHyperparams((Scalar(2f), Scalar(0.5f)));
        var output = model.Call(Tensor([4L], 1f, 2f, 3f, 4f));

        var expected = TensorData([4L], new float[] { 2.5f, 5f, 7.5f, 10f }).AccessRawMemory().ToArray();
        Assert.Equal(expected, ExecuteOutputs(output)[0]);

        // [Hyper] on an explicitly-typed lambda parameter is honored too: y = x * k.
        var lambdaModule = ModuleFactory.FromFuncWithHypers<Tensor<float32>, Scalar<float32>, Tensor<float32>>(
            static (Tensor<float32> x, [Hyper] Scalar<float32> k) => x * k);
        var lambdaOut = lambdaModule.SetHyperparams(Scalar(3f)).Call(Tensor([2L], 1f, 2f));
        Assert.Equal(
            TensorData([2L], new float[] { 3f, 6f }).AccessRawMemory().ToArray(),
            ExecuteOutputs(lambdaOut)[0]);

        // Full AutoTester roundtrip over the hypered module graph (hyperparams stay ordinary
        // graph inputs post-concretization, mirroring the generated-module coverage tests).
        Assert.True(AutoTest.AdvancedTestGraph(
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)ScaleAndShiftBody),
            hyperparamInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0.5f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
    }

    /// <summary>
    /// A FromFunc-built model graph trains one step through <see cref="TrainingRig"/>: the rig
    /// builds, the default checkpoint carries the initializer value (1.0), and one SGD step on a
    /// non-zero-gradient batch produces a finite loss and moves the weight.
    /// </summary>
    [Fact]
    public void TestFromFuncTrainsOneStepThroughTrainingRig()
    {
        var modelGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)ScalarMultiplyBody, "CodegenFreeScalarMultiply");

        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            new NamedModelParam[]
            {
                new TensorDataModelParam("input", ModelParamType.InputParam,
                    TensorData([4L], new float[] { 1f, 2f, 3f, 4f })),
            },
            0.1f);

        var initial = rig.CreateDefaultCheckpoint();
        Assert.Single(rig.TrainableParamStructDef.Fields);
        var weightField = rig.TrainableParamStructDef.Fields[0].Name;
        var initialWeight = ((TensorData<float32>)initial.TrainableParams.Fields[weightField]).AccessMemory()[0];
        Assert.Equal(1.0f, initialWeight);

        var modelInputDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32) },
            "ModelInput");
        var targetDef = new TensorStructDef(
            new[] { new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32) },
            "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], new float[] { 1f, 2f, 3f, 4f }) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], new float[] { 0f, 0f, 0f, 0f }) } });

        var ctx = new ComputeContext();
        var compiled = ctx.Compile(rig.TrainingStepPureGraph);
        var stepResult = rig.TrainStep(initial, inputBatch, targetBatch, compiled);

        Assert.True(float.IsFinite(stepResult.Loss));
        var steppedWeight = ((TensorData<float32>)stepResult.Checkpoint.TrainableParams.Fields[weightField]).AccessMemory()[0];
        Assert.NotEqual(initialWeight, steppedWeight);
    }

    /// <summary>
    /// FromFunc graphs export to ONNX (<c>FastOnnxModelBuilder.BuildOnnxModel</c> succeeds on the
    /// concrete model) and pass the standard <see cref="AutoTest"/> pipeline (ONNX roundtrip, CS
    /// codegen, QEE) for both the single-input and the two-input module shapes.
    /// </summary>
    [Fact]
    public void TestFromFuncOnnxExportAndAutoTest()
    {
        Func<Tensor<float32>, Tensor<float32>> simplest = SimplestBody;
        var moduleGraph = ModuleFactory.ComputationGraph(simplest);
        var sampleInput = TensorDataWithSmallVals(DType.Float32, [5L]);
        var concreteModel = moduleGraph
            .ToConcreteArchitecture(moduleGraph.FromOrderedInputs([sampleInput]))
            .ToConcreteModel();

        var proto = FastOnnxModelBuilder.BuildOnnxModel(concreteModel);
        Assert.NotNull(proto);
        Assert.NotNull(proto.Graph);

        Assert.True(AutoTest.TestGraph(concreteModel, sampleInputs: [sampleInput]));

        Assert.True(AutoTest.AdvancedTestGraph(
            ModuleFactory.ComputationGraph(simplest),
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph(
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>)WeightedSumBody),
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L]), TensorDataWithSmallVals(DType.Float32, [5L])]));
    }

    /// <summary>
    /// Constraint surface of the codegen-free path: capturing lambdas are rejected (the body is
    /// reflected + cached by MethodInfo); the [Hyper] annotations on the delegate must match the
    /// factory overload's hyper split in both directions; tuple-typed parameters are rejected in
    /// favor of the flattened overloads. Also verifies <c>Globals.StateUpdate</c> inside a
    /// delegate body registers state exactly as in an Inline method (STATE_UPDATE_LINK reachable
    /// from the outputs via WITH_STATE_DEPS wrapping).
    /// </summary>
    [Fact]
    public void TestFromFuncConstraintsAndStateUpdates()
    {
        // Capturing lambda → rejected with the documented error.
        var captured = Scalar(2f);
        Assert.Throws<InvalidOperationException>(() =>
            ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(x => x * captured));

        // FromFuncWithHypers without [Hyper] annotations → rejected.
        Assert.Throws<ArgumentException>(() =>
            ModuleFactory.FromFuncWithHypers<Scalar<float32>, Tensor<float32>, Tensor<float32>>(
                static (h, x) => x * h));

        // [Hyper]-annotated body handed to the no-hyper FromFunc → rejected.
        Assert.Throws<ArgumentException>(() =>
            ModuleFactory.FromFunc<Tensor<float32>, Scalar<float32>, Scalar<float32>, Tensor<float32>>(
                ScaleAndShiftBody));

        // Tuple-typed parameter → rejected (bodies take flattened parameters).
        Assert.Throws<ArgumentException>(() =>
            ModuleFactory.FromFunc<(Tensor<float32>, Tensor<float32>), Tensor<float32>>(
                static t => t.Item1 + t.Item2));

        // Globals.StateUpdate works inside the delegate body: the built graph carries the
        // STATE_UPDATE_LINK and the outputs are wrapped with WITH_STATE_DEPS, exactly as for a
        // [Module] Inline body using state initializers.
        var statefulGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)StatefulBody);
        Assert.Contains(statefulGraph.Nodes, n => n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
        Assert.Contains(statefulGraph.Nodes, n => n.OpCode == InternalOpCodes.WITH_STATE_DEPS);
    }

    /// <summary>No-runtime-input body for the <c>CallbackModule&lt;TOut&gt;</c> path.</summary>
    private static Tensor<float32> NoInputBody()
        => Vector(1f, 2f, 3f) * Scalar(2f);

    /// <summary>Hyperparam-only body for the <c>CallbackModule&lt;THyper, TOut&gt;</c> path.</summary>
    private static Tensor<float32> HyperOnlyBody([Hyper] Scalar<float32> factor)
        => Vector(1f, 2f) * factor;

    /// <summary>
    /// Covers the no-runtime-input module classes in <c>Core.ModuleBaseTypes</c>:
    /// <c>CallbackModule&lt;TOut&gt;</c> (via <see cref="ModuleFactory.FromFunc{TOut}"/>) +
    /// <c>BaseModel&lt;TOut&gt;</c>/<c>Model&lt;TOutputs&gt;.Call</c>, the hyperparam-only
    /// <c>CallbackModule&lt;THyper, TOut&gt;</c> + <c>SetHyperparams(hyper)</c>, and the
    /// <c>InputType</c>-based constructors that source-generated nested-module signatures use.
    /// </summary>
    [Fact]
    public void TestNoInputAndHyperOnlyCallbackModulesCoverage()
    {
        // CallbackModule<TOut>: no hyperparams, no runtime inputs.
        var noInput = ModuleFactory.FromFunc<Tensor<float32>>(NoInputBody);
        var noInputOut = noInput.SetHyperparams().Call();
        var noInputBytes = ExecuteOutputs(noInputOut);
        Assert.Equal(new float[] { 2f, 4f, 6f },
            MemoryMarshal.Cast<byte, float>(noInputBytes[0]).ToArray());

        // CallbackModule<THyper, TOut>: hyperparam-only signature, no runtime inputs.
        var hyperOnly = new CallbackModule<Scalar<float32>, Tensor<float32>>(HyperOnlyBody);
        var hyperOut = hyperOnly.SetHyperparams(Scalar(3f)).Call();
        var hyperBytes = ExecuteOutputs(hyperOut);
        Assert.Equal(new float[] { 3f, 6f },
            MemoryMarshal.Cast<byte, float>(hyperBytes[0]).ToArray());

        // InputType-based constructors (the nested-module input spelling): each must
        // produce a usable module/model variable.
        Assert.NotNull(new CallbackModule<Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(new CallbackModule<Scalar<float32>, Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(new Module<Tensor<float32>, Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(new Module<Scalar<float32>, Tensor<float32>, Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(((IModel)new Model<Tensor<float32>>(InputType.ReadyInput)).ModelVariable);
        Assert.NotNull(((IModel)new Model<Tensor<float32>, Tensor<float32>>(InputType.ReadyInput)).ModelVariable);
    }

    /// <summary>
    /// Executes the stateful delegate-built graph through
    /// <see cref="ComputeContext.ExecuteWithState(FastComputationGraph, TensorData[])"/>,
    /// covering the Fast state pipeline end-to-end: <c>FastLowerStateUpdateNodes</c>
    /// (STATE_UPDATE_LINK/WITH_STATE_DEPS → IDENTITY + extra state outputs) and the
    /// <c>FastComputationGraph</c> state surface (<c>GetStateParamDataNodes</c> /
    /// <c>GetStateUpdateOutputCount</c> / <c>WithUpdatedStates</c>). The state starts at 0
    /// (InitBnRunningMean) and increments by 1 per execution while the main output stays
    /// input * 2.
    /// </summary>
    [Fact]
    public void TestStatefulGraphExecuteWithStateCoverage()
    {
        var moduleGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)StatefulBody);
        var input = TensorData([4L], new float[] { 1f, 2f, 3f, 4f });
        var concrete = moduleGraph
            .ToConcreteArchitecture(moduleGraph.FromOrderedInputs([input]))
            .ToConcreteModel();

        Assert.Equal(1, concrete.GetStateUpdateOutputCount());
        var stateNodes = concrete.GetStateParamDataNodes();
        Assert.Single(stateNodes);

        float StateValue(FastComputationGraph g) =>
            g.GetStateParamDataNodes()[0].Attributes
                .GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData)!
                .As<float32>().AccessMemory()[0];

        Assert.Equal(0f, StateValue(concrete));

        var (outputs1, updated1) = ComputeContext.Default.ExecuteWithState(concrete, input);
        Assert.Single(outputs1);
        Assert.Equal(new float[] { 2f, 4f, 6f, 8f },
            MemoryMarshal.Cast<byte, float>(outputs1[0].ToTensorData().AccessRawMemory()).ToArray());
        Assert.Equal(1f, StateValue(updated1));

        var (outputs2, updated2) = ComputeContext.Default.ExecuteWithState(updated1, input);
        Assert.Equal(new float[] { 2f, 4f, 6f, 8f },
            MemoryMarshal.Cast<byte, float>(outputs2[0].ToTensorData().AccessRawMemory()).ToArray());
        Assert.Equal(2f, StateValue(updated2));
    }

    /// <summary>
    /// <see cref="Globals.StateUpdate{T}(T, T)"/> only accepts state variables — tensors created
    /// by a [StateInitializer] class's Init method. Targeting a runtime input or a trainable
    /// parameter must throw <see cref="InvalidStateUpdateException"/> at graph-build time, with
    /// declaration instructions in the message; the correct pattern (including a state variable
    /// reaching StateUpdate through the Identity node a .Vec() rank-cast inserts) still builds.
    /// </summary>
    [Fact]
    public void TestStateUpdateRejectsNonStateVariables()
    {
        // A runtime input is not a state variable.
        var inputEx = Assert.Throws<InvalidStateUpdateException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateOnInputBody));
        Assert.Equal(ErrorCodes.SU001, inputEx.ErrorCode);
        Assert.Contains("[StateInitializer]", inputEx.Message);

        // A trainable parameter is not a state variable either.
        var trainableEx = Assert.Throws<InvalidStateUpdateException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateOnTrainableBody));
        Assert.Equal(ErrorCodes.SU002, trainableEx.ErrorCode);
        Assert.Contains("trainable parameter", trainableEx.Message);

        // The correct pattern still builds, .Vec() Identity included.
        Assert.NotNull(ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)StateUpdateThroughVecBody, "CodegenFreeVecState"));
    }
}
