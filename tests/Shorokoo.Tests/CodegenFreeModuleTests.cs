using System.Runtime.InteropServices;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Parity coverage for the codegen-free module path (<see cref="ModuleFactory"/> /
/// <c>GraphBuilder.BuildInternalComputationGraphFromDelegate</c>): the same small models built once
/// from a <c>[Module]</c> class and once from a plain delegate must execute identically, and the
/// delegate-built graphs must train through <see cref="TrainingRig"/>, export to ONNX, and survive
/// the <see cref="AutoTest"/> roundtrips — with trainable-param initializers, <c>[Hyper]</c>
/// parameters and <c>Globals.StateUpdate</c> behaving exactly as in <c>Inline</c> methods.
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

    /// <summary>
    /// In-loop StateUpdate misuse: the updated value is recomputed each iteration but never
    /// carried nor consumed after the loop, so it has no post-loop value to register.
    /// </summary>
    private static Tensor<float32> StateUpdateInsideLoopBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            acc = acc * Scalar(2f);
            Globals.StateUpdate(state, state + Scalar(1f));
        }
        return acc;
    }

    /// <summary>In-loop StateUpdate misuse: the updated value is a scanned result.</summary>
    private static Tensor<float32> StateUpdateOnScanInsideLoopBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = state + Scalar(0f);
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            acc = acc + Scalar(1f);
            Globals.StateUpdate(state, ctx.Scan(acc));
        }
        return input * Scalar(2f);
    }

    /// <summary>In-loop StateUpdate happy path: state ← state + 10 + 3 per execution.</summary>
    private static Tensor<float32> StateUpdateInLoopCarriedBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = state + Scalar(10f);
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            acc = acc + Scalar(1f);
            Globals.StateUpdate(state, acc);
        }
        return input * Scalar(2f);
    }

    /// <summary>The documented after-the-loop spelling of <see cref="StateUpdateInLoopCarriedBody"/>.</summary>
    private static Tensor<float32> StateUpdateAfterLoopCarriedBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = state + Scalar(10f);
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            acc = acc + Scalar(1f);
        }
        Globals.StateUpdate(state, acc);
        return input * Scalar(2f);
    }

    /// <summary>Two nesting levels: state ← state + 10 + 2·3 per execution.</summary>
    private static Tensor<float32> StateUpdateInNestedLoopBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = state + Scalar(10f);
        foreach (var outerCtx in LoopAPI.Iterate(Scalar(2L)))
        {
            foreach (var innerCtx in LoopAPI.Iterate(Scalar(3L)))
            {
                acc = acc + Scalar(1f);
                Globals.StateUpdate(state, acc);
            }
        }
        return input * Scalar(2f);
    }

    /// <summary>Zero iterations: the close output falls back to the initializer, so state ← state + 10.</summary>
    private static Tensor<float32> StateUpdateInZeroIterationLoopBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = state + Scalar(10f);
        foreach (var ctx in LoopAPI.Iterate(Scalar(0L)))
        {
            acc = acc + Scalar(1f);
            Globals.StateUpdate(state, acc);
        }
        return input * Scalar(2f);
    }

    /// <summary>
    /// A loop-construction failure striking on the fourth body trace — after the canonical (third)
    /// pass recorded the pending registration, before the loop completes and resolves it.
    /// </summary>
    private static Tensor<float32> StateUpdateInLoopThenFourthPassThrowBody(Tensor<float32> input)
    {
        var state = InitBnRunningMean.Init(Vector(1L));
        var acc = state + Scalar(0f);
        var bodyRuns = 0;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
        {
            acc = acc + Scalar(1f);
            Globals.StateUpdate(state, acc);
            if (++bodyRuns == 4)
                throw new FormatException("simulated fourth-pass loop-construction failure");
        }
        return input * Scalar(2f);
    }

    // ───────────────────────────────── helpers ─────────────────────────────────

    private static byte[][] ExecuteConcretized(ComputationGraph moduleGraph, params TensorData[] inputs)
    {
        var concreteModel = moduleGraph
            .ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. inputs]))
            .ToConcreteModel();
        return ComputeContext.Default.Execute(concreteModel, (IData[])inputs)
            .Select(x => x.ToTensorData().AccessRawMemory().ToArray())
            .ToArray();
    }

    private static byte[][] ExecuteOutputs(params Variable[] outputs)
        => ExecuteConcretized(new InternalComputationGraph([], [.. outputs]).ToComputationGraph(GraphKind.Module));

    private static float StateValue(ComputationGraph graph) =>
        graph.ToInternal().GetStateParamDataNodes()[0].Attributes
            .GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData)!
            .As<float32>().AccessMemory()[0];

    private static ComputationGraph Concretize(
        Func<Tensor<float32>, Tensor<float32>> body, string? name, TensorData input)
    {
        var graph = ModuleFactory.ComputationGraph(body, name);
        return graph.ToConcreteArchitecture(graph.FromOrderedInputs([input])).ToConcreteModel();
    }

    private static float[] Floats(byte[] bytes) => MemoryMarshal.Cast<byte, float>(bytes).ToArray();

    // ─────────────────────────────────── tests ───────────────────────────────────

    /// <summary>
    /// A model built via the <c>[Module]</c> source generator, via a static method group and via a
    /// non-capturing static lambda executes byte-identically (the lambda proves the delegate-target
    /// invoke path); the <c>FromFunc(...).SetHyperparams().Call(x)</c> spelling matches the
    /// generated <c>Foo.Model().Call(x)</c>; and bound <c>[Hyper]</c> values flow through the call.
    /// </summary>
    [Fact]
    public void TestFromFuncParityAndCallPaths()
    {
        var input = TensorData([4L], 1f, 2f, 3f, 4f);

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

        // Single input: generated SimplestLayer.Model().Call vs the FromFunc path.
        var viaCodegen = SimplestLayer.Model().Call(Tensor([4L], 1f, 2f, 3f, 4f));
        var viaFactory = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(SimplestBody)
            .SetHyperparams()
            .Call(Tensor([4L], 1f, 2f, 3f, 4f));
        Assert.Equal(ExecuteOutputs(viaCodegen)[0], ExecuteOutputs(viaFactory)[0]);

        // Two inputs: FromFunc<T1, T2, TOut> with a Model<T1, T2, TOut> for a 2-arg Call;
        // weights init to 1 → (a + b) * 1.
        var pairModule = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>, Tensor<float32>>(WeightedSumBody);
        var pairModel = pairModule.SetHyperparams<Model<Tensor<float32>, Tensor<float32>, Tensor<float32>>>();
        var pairOut = pairModel.Call(Tensor([3L], 1f, 2f, 3f), Tensor([3L], 10f, 20f, 30f));
        Assert.Equal(TensorData([3L], 11f, 22f, 33f).AccessRawMemory().ToArray(), ExecuteOutputs(pairOut)[0]);

        // Hypers bound via SetHyperparams((factor, bias)): y = x * (1 * 2 + 0.5) = 2.5x.
        var hypered = ModuleFactory.FromFuncWithHypers<Tensor<float32>, Scalar<float32>, Scalar<float32>, Tensor<float32>>(
            ScaleAndShiftBody, "CodegenFreeScaleAndShift");
        var hyperedOut = hypered.SetHyperparams((Scalar(2f), Scalar(0.5f))).Call(Tensor([4L], 1f, 2f, 3f, 4f));
        Assert.Equal(TensorData([4L], 2.5f, 5f, 7.5f, 10f).AccessRawMemory().ToArray(), ExecuteOutputs(hyperedOut)[0]);

        // [Hyper] on an explicitly-typed lambda parameter is honored too: y = x * k.
        var lambdaModule = ModuleFactory.FromFuncWithHypers<Tensor<float32>, Scalar<float32>, Tensor<float32>>(
            static (Tensor<float32> x, [Hyper] Scalar<float32> k) => x * k);
        var lambdaOut = lambdaModule.SetHyperparams(Scalar(3f)).Call(Tensor([2L], 1f, 2f));
        Assert.Equal(TensorData([2L], 3f, 6f).AccessRawMemory().ToArray(), ExecuteOutputs(lambdaOut)[0]);
    }

    /// <summary>
    /// A FromFunc-built model trains one step through <see cref="TrainingRig"/>, exports to ONNX and
    /// passes the <see cref="AutoTest"/> pipeline for the one- and two-input shapes; its constraint
    /// surface rejects capturing lambdas, mismatched <c>[Hyper]</c> splits and tuple parameters; and
    /// <c>Globals.StateUpdate</c> inside a delegate body registers state as in an Inline method.
    /// </summary>
    [Fact]
    public void TestFromFuncTrainingExportAndConstraints()
    {
        var modelGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)ScalarMultiplyBody, "CodegenFreeScalarMultiply");

        var rig = TrainingRig.FromScratch(
            modelGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph,
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([4L], 1f, 2f, 3f, 4f))],
            0.1f);

        var initial = rig.CreateInitialCheckpoint();
        Assert.Single(rig.TrainableParamStructDef.Fields);
        var weightField = rig.TrainableParamStructDef.Fields[0].Name;
        var initialWeight = ((TensorData<float32>)initial.TrainableParams.Fields[weightField]).AccessMemory()[0];
        Assert.Equal(1.0f, initialWeight);

        var modelInputDef = new TensorStructDef(
            [new TensorStructFieldDef("input", DataStructure.Tensor, 1, DType.Float32)], "ModelInput");
        var targetDef = new TensorStructDef(
            [new TensorStructFieldDef("targets", DataStructure.Tensor, 1, DType.Float32)], "Target");
        var inputBatch = new TensorDataStruct(modelInputDef,
            new Dictionary<string, IData> { { "input", TensorData([4L], 1f, 2f, 3f, 4f) } });
        var targetBatch = new TensorDataStruct(targetDef,
            new Dictionary<string, IData> { { "targets", TensorData([4L], 0f, 0f, 0f, 0f) } });

        var stepResult = rig.TrainStep(initial, inputBatch, targetBatch);
        Assert.True(float.IsFinite(stepResult.Loss!.Value));
        Assert.NotEqual(initialWeight,
            ((TensorData<float32>)stepResult.TrainableParams.Fields[weightField]).AccessMemory()[0]);

        // ONNX export + the standard AutoTest pipeline, single- and two-input shapes.
        Func<Tensor<float32>, Tensor<float32>> simplest = SimplestBody;
        var simplestGraph = ModuleFactory.ComputationGraph(simplest);
        var sampleInput = TensorDataWithSmallVals(DType.Float32, [5L]);
        var concreteModel = simplestGraph
            .ToConcreteArchitecture(simplestGraph.FromOrderedInputs([sampleInput]))
            .ToConcreteModel();

        var proto = FastOnnxModelBuilder.BuildOnnxModel(concreteModel);
        Assert.NotNull(proto);
        Assert.NotNull(proto.Graph);
        Assert.True(AutoTest.TestGraph(concreteModel, sampleInputs: [sampleInput]));
        Assert.True(AutoTest.AdvancedTestGraph(
            ModuleFactory.ComputationGraph(simplest),
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph(
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>, Tensor<float32>>)WeightedSumBody),
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L]), TensorDataWithSmallVals(DType.Float32, [5L])]));

        // Hyperparams stay ordinary graph inputs post-concretization.
        Assert.True(AutoTest.AdvancedTestGraph(
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)ScaleAndShiftBody),
            hyperparamInputs: [TensorData(DType.Float32, [], 2f), TensorData(DType.Float32, [], 0.5f)],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));

        // Capturing lambda → rejected (the body is reflected + cached by MethodInfo).
        var captured = Scalar(2f);
        Assert.Throws<InvalidOperationException>(() =>
            ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(x => x * captured));
        // FromFuncWithHypers without [Hyper] annotations, and a [Hyper]-annotated body handed to
        // the no-hyper FromFunc → both rejected.
        Assert.Throws<ArgumentException>(() =>
            ModuleFactory.FromFuncWithHypers<Scalar<float32>, Tensor<float32>, Tensor<float32>>(
                static (h, x) => x * h));
        Assert.Throws<ArgumentException>(() =>
            ModuleFactory.FromFunc<Tensor<float32>, Scalar<float32>, Scalar<float32>, Tensor<float32>>(
                ScaleAndShiftBody));
        // Tuple-typed parameter → rejected (bodies take flattened parameters).
        Assert.Throws<ArgumentException>(() =>
            ModuleFactory.FromFunc<(Tensor<float32>, Tensor<float32>), Tensor<float32>>(
                static t => t.Item1 + t.Item2));

        // Globals.StateUpdate inside a delegate body carries the STATE_UPDATE_LINK and wraps the
        // outputs with WITH_STATE_DEPS, exactly as for a [Module] Inline body.
        var statefulGraph = ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)StatefulBody);
        Assert.Contains(statefulGraph.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
        Assert.Contains(statefulGraph.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.WITH_STATE_DEPS);
    }

    /// <summary>No-runtime-input body for the <c>CallbackModule&lt;TOut&gt;</c> path.</summary>
    private static Tensor<float32> NoInputBody()
        => Vector(1f, 2f, 3f) * Scalar(2f);

    /// <summary>Hyperparam-only body for the <c>CallbackModule&lt;THyper, TOut&gt;</c> path.</summary>
    private static Tensor<float32> HyperOnlyBody([Hyper] Scalar<float32> factor)
        => Vector(1f, 2f) * factor;

    /// <summary>
    /// The no-runtime-input module classes in <c>Core.ModuleBaseTypes</c>:
    /// <c>CallbackModule&lt;TOut&gt;</c> + <c>Model&lt;TOutputs&gt;.Call</c>, the hyperparam-only
    /// <c>CallbackModule&lt;THyper, TOut&gt;</c> + <c>SetHyperparams(hyper)</c>, and the
    /// <c>InputType</c>-based constructors source-generated nested-module signatures use.
    /// </summary>
    [Fact]
    public void TestNoInputAndHyperOnlyCallbackModulesCoverage()
    {
        var noInput = ModuleFactory.FromFunc<Tensor<float32>>(NoInputBody);
        Assert.Equal<float>([2f, 4f, 6f], Floats(ExecuteOutputs(noInput.SetHyperparams().Call())[0]));

        var hyperOnly = new CallbackModule<Scalar<float32>, Tensor<float32>>(HyperOnlyBody);
        Assert.Equal<float>([3f, 6f], Floats(ExecuteOutputs(hyperOnly.SetHyperparams(Scalar(3f)).Call())[0]));

        Assert.NotNull(new CallbackModule<Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(new CallbackModule<Scalar<float32>, Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(new Module<Tensor<float32>, Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(new Module<Scalar<float32>, Tensor<float32>, Tensor<float32>>(InputType.ReadyInput).ModuleVariable);
        Assert.NotNull(((IModel)new Model<Tensor<float32>>(InputType.ReadyInput)).ModelVariable);
        Assert.NotNull(((IModel)new Model<Tensor<float32>, Tensor<float32>>(InputType.ReadyInput)).ModelVariable);
    }

    /// <summary>
    /// The Fast state pipeline end-to-end via
    /// <see cref="ComputeContext.ExecuteWithState(ComputationGraph, TensorData[])"/>:
    /// <c>FastLowerStateUpdateNodes</c> plus the <c>InternalComputationGraph</c> state surface. The
    /// state starts at 0 and increments by 1 per execution while the output stays input * 2.
    /// </summary>
    [Fact]
    public void TestStatefulGraphExecuteWithStateCoverage()
    {
        var input = TensorData([4L], 1f, 2f, 3f, 4f);
        var concrete = Concretize(StatefulBody, null, input);

        Assert.Equal(1, concrete.ToInternal().GetStateUpdateOutputCount());
        Assert.Single(concrete.ToInternal().GetStateParamDataNodes());
        Assert.Equal(0f, StateValue(concrete));

        var (outputs1, updated1) = ComputeContext.Default.ExecuteWithState(concrete, input);
        Assert.Single(outputs1);
        Assert.Equal<float>([2f, 4f, 6f, 8f], Floats(outputs1[0].ToTensorData().AccessRawMemory().ToArray()));
        Assert.Equal(1f, StateValue(updated1));

        var (outputs2, updated2) = ComputeContext.Default.ExecuteWithState(updated1, input);
        Assert.Equal<float>([2f, 4f, 6f, 8f], Floats(outputs2[0].ToTensorData().AccessRawMemory().ToArray()));
        Assert.Equal(2f, StateValue(updated2));
    }

    /// <summary>
    /// <see cref="Globals.StateUpdate{T}(T, T)"/> only accepts state variables — tensors created by
    /// a [StateInitializer] class's Init method — and only inside a module build in progress.
    /// Targeting a runtime input or a trainable parameter throws at graph-build time with
    /// declaration instructions; the correct pattern (including through a .Vec() Identity) builds.
    /// </summary>
    [Fact]
    public void TestStateUpdateRejectsNonStateVariables()
    {
        var inputEx = Assert.Throws<InvalidStateUpdateException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateOnInputBody));
        Assert.Equal(ErrorCodes.SU001, inputEx.ErrorCode);
        Assert.Contains("[StateInitializer]", inputEx.Message);

        var trainableEx = Assert.Throws<InvalidStateUpdateException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateOnTrainableBody));
        Assert.Equal(ErrorCodes.SU002, trainableEx.ErrorCode);
        Assert.Contains("trainable parameter", trainableEx.Message);

        Assert.NotNull(ModuleFactory.ComputationGraph(
            (Func<Tensor<float32>, Tensor<float32>>)StateUpdateThroughVecBody, "CodegenFreeVecState"));

        // With no module build active the registration could never be harvested, so it throws at
        // the call site rather than sitting in a thread-static list forever.
        var noBuildEx = Assert.Throws<InvalidOperationException>(
            () => Globals.StateUpdate(Scalar(1f), Scalar(2f)));
        Assert.Contains("inside a module body", noBuildEx.Message);
    }

    /// <summary>
    /// <c>Globals.StateUpdate</c> inside a <c>LoopAPI.Iterate</c> body registers the post-loop value
    /// of the updated tensor: a single loop's carried close output (identical to the documented
    /// after-the-loop pattern), the outer close output across two nesting levels, and — at zero
    /// iterations — the carried variable's pre-loop initializer.
    /// </summary>
    [Fact]
    public void TestStateUpdateInLoopsRegistersPostLoopValue()
    {
        var input = TensorData([4L], 1f, 2f, 3f, 4f);

        ComputationGraph[] singleLoopVariants =
        [
            Concretize(StateUpdateInLoopCarriedBody, "CodegenFreeInLoopState", input),
            Concretize(StateUpdateAfterLoopCarriedBody, "CodegenFreeAfterLoopState", input),
        ];
        foreach (var concrete in singleLoopVariants)
        {
            Assert.Equal(1, concrete.ToInternal().GetStateUpdateOutputCount());
            Assert.Equal(0f, StateValue(concrete));

            var (outputs1, updated1) = ComputeContext.Default.ExecuteWithState(concrete, input);
            Assert.Equal<float>([2f, 4f, 6f, 8f], Floats(outputs1[0].ToTensorData().AccessRawMemory().ToArray()));
            Assert.Equal(13f, StateValue(updated1));   // +10 initializer, +1 × 3 iterations

            var (_, updated2) = ComputeContext.Default.ExecuteWithState(updated1, input);
            Assert.Equal(26f, StateValue(updated2));   // re-reads the updated state: 13 + 10 + 3
        }

        var nested = Concretize(StateUpdateInNestedLoopBody, null, input);
        Assert.Equal(1, nested.ToInternal().GetStateUpdateOutputCount());
        Assert.Equal(0f, StateValue(nested));
        var (nestedOutputs, nested1) = ComputeContext.Default.ExecuteWithState(nested, input);
        Assert.Equal<float>([2f, 4f, 6f, 8f], Floats(nestedOutputs[0].ToTensorData().AccessRawMemory().ToArray()));
        Assert.Equal(16f, StateValue(nested1));        // +10, then +1 × 2·3
        var (_, nested2) = ComputeContext.Default.ExecuteWithState(nested1, input);
        Assert.Equal(32f, StateValue(nested2));

        var zeroIter = Concretize(StateUpdateInZeroIterationLoopBody, null, input);
        Assert.Equal(0f, StateValue(zeroIter));
        var (_, zero1) = ComputeContext.Default.ExecuteWithState(zeroIter, input);
        Assert.Equal(10f, StateValue(zero1));          // the body contributes nothing
        var (_, zero2) = ComputeContext.Default.ExecuteWithState(zero1, input);
        Assert.Equal(20f, StateValue(zero2));
    }

    /// <summary>
    /// In-loop StateUpdate rejections: an updated value that never surfaces as a loop output, and a
    /// scanned result whose post-loop form is the stacked per-iteration tensor, both fail the module
    /// build with an error naming the fix. A loop-construction failure that strikes after the
    /// registration recorded must surface as-is, not as the misleading "never resolved" error.
    /// </summary>
    [Fact]
    public void TestStateUpdateInLoopsRejectsUnresolvableValues()
    {
        var uncarried = Assert.Throws<InvalidOperationException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateInsideLoopBody));
        Assert.Contains("does not surface as a loop output", uncarried.Message);
        Assert.Contains("LoopAPI.Iterate", uncarried.Message);

        var scanned = Assert.Throws<InvalidOperationException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateOnScanInsideLoopBody));
        Assert.Contains("scanned", scanned.Message);
        Assert.Contains("LoopAPI.Iterate", scanned.Message);

        var masked = Assert.Throws<FormatException>(() =>
            ModuleFactory.ComputationGraph(
                (Func<Tensor<float32>, Tensor<float32>>)StateUpdateInLoopThenFourthPassThrowBody));
        Assert.Equal("simulated fourth-pass loop-construction failure", masked.Message);
    }
}
