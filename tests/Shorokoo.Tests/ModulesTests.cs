using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModulesCoverageTests
{
    [Fact]
    public void TestCalledPassThroughSwapAndRepeatedOutputModulesWireThroughToTheCallersInputs()
    {
        TensorData[] x = [TensorData(DType.Float32, [2L], 1f, 2f)];
        Assert.True(AutoTest.AdvancedTestGraph<Modules.CallerOfPassThroughSub>(
            hyperparamInputs: [], runtimeInputs: x, expected: [2.0, 4.0]));
        Assert.True(AutoTest.AdvancedTestGraph<Modules.CallerOfSwapSub>(
            hyperparamInputs: [], runtimeInputs: x, expected: [1.0, 2.0]));
        Assert.True(AutoTest.AdvancedTestGraph<Modules.CallerOfRepeatedOutputSub>(
            hyperparamInputs: [], runtimeInputs: x, expected: [6.0, 12.0]));
        Assert.True(AutoTest.AdvancedTestGraph<Modules.LoopCarriedPassThrough>(
            hyperparamInputs: [], runtimeInputs: x, expected: [1.0, 2.0]));
    }

    [Fact]
    public void TestAnInitializerCallingAModuleFlattensThatCallInItsFunctionBody()
    {
        TensorData[] x = [TensorData(DType.Float32, [2L], 1f, 2f)];
        Assert.True(AutoTest.AdvancedTestGraph<Modules.UsesInitCallingAModule>(
            hyperparamInputs: [], runtimeInputs: x, expected: [2.0, 4.0]));
        Assert.True(AutoTest.AdvancedTestGraph<Modules.UsesInitCallingHyperModule>(
            hyperparamInputs: [], runtimeInputs: x, expected: [2.0, 4.0]));
    }

    /// <summary>Flattening moves the callee's MODEL_PARAM_REF into the body, but the parameter
    /// lowering chain only ever runs over a whole graph, so the emitted body still names
    /// #ModelParamRef#. Tracked as Shorokoo/Shorokoo#287.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#287: a flattened body keeps its callee's parameter ref unlowered")]
    public void TestAnInitializerCallingAParamOwningModuleFlattensThatCallInItsFunctionBody()
        => Assert.True(AutoTest.AdvancedTestGraph<Modules.UsesInitCallingAParamOwningModule>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)], expected: [1.0, 2.0]));

    /// <summary>The same body with its call wrapped in a loop emits a Shrk-free proto that ORT
    /// still rejects for missing type information on the loop's carried input, while the identical
    /// shape at module level lowers fine. Tracked as Shorokoo/Shorokoo#287.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#287: a flattened body wrapping its call in a loop loses type information")]
    public void TestAnInitializerCallingAModuleInALoopFlattensThatCallInItsFunctionBody()
        => Assert.True(AutoTest.AdvancedTestGraph<Modules.UsesInitCallingAModuleInALoop>(
            hyperparamInputs: [], runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)], expected: [4.0, 8.0]));

    [Fact]
    public void TestTheNativeContainerKeepsASubModuleBoundaryThatOnnxExportFlattens()
    {
        var g = Modules.UsesInitCallingAModule.ComputationGraph;
        var reloaded = CompressedFormatUtils.LoadFastGraphFromBinary(
            CompressedFormatUtils.SaveFastGraphToBinary(g, compressed: false)).ToInternal();
        var bodies = reloaded.LocalFunctions.SelectMany(f => f.Body.ToInternal().Nodes);
        Assert.Contains(bodies, n => n.OpCode == InternalOpCodes.MODEL_INVOKE);
    }

    [Fact]
    public void TestFromOrderedInputsRefusesMoreValuesThanTheGraphHasDataInputs()
    {
        var g = SimpleGenericLayer.ComputationGraph;
        Assert.Throws<ModelException>(
            () => g.FromOrderedInputs([TensorData([], 0f), TensorData([2L], 1f, 2f)]));
    }

    [Fact]
    public void TestStateUpdateSurvivesNestedFirstUseModuleBuild()
    {
        var graph = ((ComputationGraph)typeof(Modules.StateUpdateSurvivesNestedFirstUseBuild)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.STATE_UPDATE_LINK);
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.WITH_STATE_DEPS);
    }

    private static double[] Rep(double v, int n) => [.. Enumerable.Repeat(v, n)];

    private const string GatingInput = "useBias";

    private static bool GatedBias(bool bake, bool viaSpecialize, int paramCount, bool gated, float[] expected)
    {
        var g = Modules.GatedBiasHyperLayer.ComputationGraph;
        var bit = TensorData([], bake);
        var x = TensorData([2L], 1f, 2f);
        if (viaSpecialize) g = g.Specialize(g.FromOrderedInputs([bit]));
        var arch = g.ToConcreteArchitecture(
            viaSpecialize ? g.FromOrderedInputs([x]) : g.FromOrderedInputs([bit, x]));
        var model = arch.ToConcreteModel(RngConfig.Default);
        var gateIsInput = model.InputNames.Contains(GatingInput);
        IData[] inputs = gateIsInput ? [bit, x] : [x];
        var result = new ComputeContext().Execute(model, inputs)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray();
        return arch.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length == paramCount
            && gateIsInput == gated
            && result.SequenceEqual(expected);
    }

    [Fact]
    public void TestAGatedHyperparamBakesTheParameterSpaceAndStaysALiveInput()
    {
        float[] plain = [1f, 2f];
        float[] biased = [2f, 3f];
        Assert.True(GatedBias(bake: false, viaSpecialize: false, paramCount: 0, gated: true, plain));
        Assert.True(GatedBias(bake: true, viaSpecialize: false, paramCount: 1, gated: true, biased));
        Assert.True(GatedBias(bake: false, viaSpecialize: true, paramCount: 0, gated: false, plain));
        Assert.True(GatedBias(bake: true, viaSpecialize: true, paramCount: 1, gated: false, biased));
    }

    private static (int Params, int Ifs, float[] Paramless, float[] Gated) TwoGatesRun(
        bool bake, bool viaSpecialize, bool atExecute)
    {
        var g = Modules.TwoGatesOneHyperLayer.ComputationGraph;
        var x = TensorData([2L], 1f, 2f);
        if (viaSpecialize) g = g.Specialize(g.FromOrderedInputs([TensorData([], bake)]));
        var arch = g.ToConcreteArchitecture(
            viaSpecialize ? g.FromOrderedInputs([x]) : g.FromOrderedInputs([TensorData([], bake), x]));
        IData[] inputs = viaSpecialize ? [x] : [TensorData([], atExecute), x];
        var o = new ComputeContext().Execute(arch.ToConcreteModel(RngConfig.Default), inputs);
        return (arch.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length,
                IfCount(arch), Floats(o[0]), Floats(o[1]));
    }

    private static bool TwoGatesMatch(
        (int Params, int Ifs, float[] Paramless, float[] Gated) r,
        int paramCount, int ifNodes, float[] paramless, float[] gated)
        => r.Params == paramCount && r.Ifs == ifNodes
           && r.Paramless.SequenceEqual(paramless) && r.Gated.SequenceEqual(gated);

    private static bool TwoGates(bool bake, bool atExecute, int paramCount, int ifNodes, float[] paramless, float[] gated)
        => TwoGatesMatch(TwoGatesRun(bake, viaSpecialize: false, atExecute), paramCount, ifNodes, paramless, gated);

    private static bool TwoGatesSpecialized(bool bake, int paramCount, float[] paramless, float[] gated)
        => TwoGatesMatch(TwoGatesRun(bake, viaSpecialize: true, atExecute: bake), paramCount, 0, paramless, gated);

    /// <summary>One <c>[Hyper]</c> bit gating two <c>IfElse</c>es — one holding a trainable
    /// parameter, one not. Concretizing the bit off prunes the parameter, so its branch can never
    /// be taken and the whole <c>IfElse</c> is folded away; the paramless one on the same bit is
    /// untouched and still switches at run time. Concretizing it on prunes nothing and folds
    /// nothing. The gated branch multiplies by the parameter, so a stand-in left in place of the
    /// fold would show up as zeros rather than hiding behind an identical value.</summary>
    [Fact]
    public void TestAPrunedParamsBranchFoldsAwayWhileAParamlessBranchOnTheSameHyperStaysLive()
    {
        float[] then = [10f, 20f], els = [100f, 200f], scaled = [1f, 2f], sevens = [7f, 14f];
        Assert.True(TwoGates(bake: false, atExecute: false, paramCount: 0, ifNodes: 1, els, sevens));
        Assert.True(TwoGates(bake: false, atExecute: true, paramCount: 0, ifNodes: 1, then, sevens));
        Assert.True(TwoGates(bake: true, atExecute: false, paramCount: 1, ifNodes: 2, els, sevens));
        Assert.True(TwoGates(bake: true, atExecute: true, paramCount: 1, ifNodes: 2, then, scaled));
        Assert.True(TwoGatesSpecialized(bake: false, paramCount: 0, els, sevens));
        Assert.True(TwoGatesSpecialized(bake: true, paramCount: 1, then, scaled));
    }

    private static long BiggestConstant(ComputationGraph arch)
    {
        long biggest = 0;
        foreach (var n in arch.ToInternal().Nodes)
        {
            if (n.OpCode != OpCodes.CONSTANT) continue;
            if (n.Attributes?.GetAttributeVals().GetValueOrDefault(OnnxOpAttributeNames.AttrValue) is not TensorData td) continue;
            long count = 1;
            foreach (var d in td.Shape.Dims) count *= d;
            biggest = Math.Max(biggest, count);
        }
        return biggest;
    }

    private static long GatedBigParamBiggestConstant(bool viaSpecialize, bool bake)
    {
        var g = Modules.BigGatedParamLayer.ComputationGraph;
        var bit = TensorData([], bake);
        var x = TensorData([2L], 1f, 2f);
        if (viaSpecialize) g = g.Specialize(g.FromOrderedInputs([bit]));
        return BiggestConstant(g.ToConcreteArchitecture(
            viaSpecialize ? g.FromOrderedInputs([x]) : g.FromOrderedInputs([bit, x])));
    }

    private static ComputationGraph NestedBigArch(bool outer, bool inner)
    {
        var g = Modules.NestedBigGatedParamLayer.ComputationGraph;
        return g.ToConcreteArchitecture(g.FromOrderedInputs(
            [TensorData([], outer), TensorData([], inner), TensorData([2L], 1f, 2f)]));
    }

    private static int IfCount(ComputationGraph arch)
        => arch.ToInternal().Nodes.Count(n => n.OpCode == OpCodes.IF_OPEN);

    private static int ExpandCount(ComputationGraph arch)
        => arch.ToInternal().Nodes.Count(n => n.OpCode == OpCodes.EXPAND);

    private static float[] Floats(NamedModelParam p)
        => p.ToTensorData().As<float32>().AccessMemory<float>().ToArray();

    /// <summary>A gated <c>[256, 256]</c> parameter switched off must not reappear as a stand-in
    /// of its own size, on either route and whether or not its branch survives. Exact counts, so
    /// that a stand-in growing dense (65536), a surviving branch being folded away by mistake (1),
    /// and the intended shape-driven stand-in (2) are all distinguishable; the gate-on row is the
    /// positive control proving the scanner sees constants at all.</summary>
    [Fact]
    public void TestPruningAGatedParamDoesNotLeaveItsBytesBehindAsAStandIn()
    {
        Assert.Equal(2, GatedBigParamBiggestConstant(viaSpecialize: false, bake: true));
        Assert.Equal(0, GatedBigParamBiggestConstant(viaSpecialize: true, bake: false));
        Assert.Equal(0, GatedBigParamBiggestConstant(viaSpecialize: false, bake: false));

        var survives = NestedBigArch(outer: false, inner: true);
        Assert.Equal(2, BiggestConstant(survives));
        Assert.Equal(1, ExpandCount(survives));

        var innerFolds = NestedBigArch(outer: false, inner: false);
        Assert.Equal(1, BiggestConstant(innerFolds));
        Assert.Equal(0, ExpandCount(innerFolds));
    }

    /// <summary>The stand-in is a hand-built <c>EXPAND</c>, so it has to actually run: concretize
    /// with the outer gate off — which keeps the enclosed branch and its stand-in — then execute
    /// with that still-live gate on, so the branch reading the stand-in is the one taken. Zeros,
    /// summed and added, leave the input untouched. The rank-0 parameter takes the emitter's
    /// no-dims path, which cannot <c>EXPAND</c> at all.</summary>
    [Fact]
    public void TestAStandInBranchThatSurvivesStillExecutes()
    {
        var x = TensorData([2L], 1f, 2f);
        var arch = NestedBigArch(outer: false, inner: true);
        var o = new ComputeContext().Execute(arch.ToConcreteModel(RngConfig.Default),
            [TensorData([], true), TensorData([], true), x]);
        Assert.Equal([1f, 2f], Floats(o[0]));

        var reloaded = CompressedFormatUtils.LoadFastGraphCore(
            CompressedFormatUtils.SaveFastGraphToBinary(arch.ToInternal(), compressed: true),
            "<roundtrip>", null).Graph;
        Assert.Equal(1, reloaded.Nodes.Count(n => n.OpCode == OpCodes.EXPAND));

        var g = Modules.NestedRank0GatedParamLayer.ComputationGraph;
        var r0 = g.ToConcreteArchitecture(g.FromOrderedInputs(
            [TensorData([], false), TensorData([], true), x]));
        Assert.Equal(0, ExpandCount(r0));
        Assert.Equal(0, r0.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length);
        var r0Out = new ComputeContext().Execute(r0.ToConcreteModel(RngConfig.Default),
            [TensorData([], true), TensorData([], true), x]);
        Assert.Equal([0f, 0f], Floats(r0Out[0]));
    }

    private static bool Exclusivity(ComputationGraph g, TensorData[] hints, IData[] exec,
        int paramCount, int ifNodes, float[] slot0, float[] slot1)
    {
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([.. hints]));
        var o = new ComputeContext().Execute(arch.ToConcreteModel(RngConfig.Default), exec);
        return arch.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length == paramCount
            && IfCount(arch) == ifNodes
            && Floats(o[0]).SequenceEqual(slot0)
            && Floats(o[1]).SequenceEqual(slot1);
    }

    /// <summary>Folding an <c>IF_CLOSE</c> takes all of its slots, so a tuple <c>IfElse</c> whose
    /// slot 0 holds a pruned parameter must not fold — slot 1 holds none and keeps switching at
    /// run time. Likewise a parameter two gates share is owned by neither, so both stay. In both
    /// the parameter is still pruned, so the branch that held it reads a zero stand-in.</summary>
    [Fact]
    public void TestAPrunedParamOnlyFoldsTheGateThatExclusivelyOwnsIt()
    {
        var x = TensorData([2L], 1f, 2f);
        TensorData Bit(bool b) => TensorData([], b);

        Assert.True(Exclusivity(Modules.TupleGateHyperLayer.ComputationGraph,
            [Bit(false), x], [Bit(true), x],
            paramCount: 0, ifNodes: 1, slot0: [1f, 2f], slot1: [10f, 20f]));

        Assert.True(Exclusivity(Modules.SharedDeadParamTwoGatesLayer.ComputationGraph,
            [Bit(false), x, Bit(false)], [Bit(false), x, Bit(true)],
            paramCount: 0, ifNodes: 2, slot0: [1f, 2f], slot1: [1f, 2f]));
    }

    /// <summary>A gate with a pruned parameter on each branch folds whichever way it points.
    /// Only the losing branch's parameter can unlock the fold, so a gate must not be written off
    /// because the winning branch's parameter was the one looked at first.</summary>
    [Fact]
    public void TestAGateWithAPrunedParamOnEachBranchFoldsEitherWay()
    {
        var g = Modules.ParamOnBothBranchesLayer.ComputationGraph;
        var x = TensorData([2L], 1f, 2f);
        (int Params, int Ifs, float[] Out) Run(bool flag)
        {
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs(
                [TensorData([], false), TensorData([], flag), x]));
            var o = new ComputeContext().Execute(arch.ToConcreteModel(RngConfig.Default),
                [TensorData([], false), TensorData([], flag), x]);
            return (arch.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length,
                    IfCount(arch), Floats(o[0]));
        }
        foreach (var flag in (bool[])[false, true])
        {
            var (parms, ifs, outs) = Run(flag);
            Assert.Equal(0, parms);
            Assert.Equal(0, ifs);
            Assert.Equal([9f, 18f], outs);
        }
    }

    /// <summary>The fold is not specific to a bit-valued gate on the then branch: a computed
    /// condition folds, and so does a parameter living on the else branch.</summary>
    [Fact]
    public void TestAComputedConditionFoldsAnElseBranchParam()
    {
        var g = Modules.ElseBranchComputedGateLayer.ComputationGraph;
        var x = TensorData([2L], 1f, 2f);
        bool Case(long mode, int paramCount, int ifNodes, float[] expected)
        {
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([], mode), x]));
            var o = new ComputeContext().Execute(arch.ToConcreteModel(RngConfig.Default),
                [TensorData([], mode), x]);
            return arch.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length == paramCount
                && IfCount(arch) == ifNodes
                && Floats(o[0]).SequenceEqual(expected);
        }
        Assert.True(Case(mode: 9L, paramCount: 0, ifNodes: 0, [5f, 10f]));
        Assert.True(Case(mode: 1L, paramCount: 1, ifNodes: 1, [2f, 3f]));
    }

    /// <summary>The same on a real library layer: <c>Linear(useBias: false)</c> keeps one
    /// parameter and no gate, and its bias branch reads the live weight without owning it.</summary>
    [Fact]
    public void TestLinearWithoutBiasFoldsItsGateAndKeepsTheWeight()
    {
        var g = Shorokoo.Modules.Layers.Linear.ComputationGraph;
        var x = TensorData([1L, 2L], 1f, 2f);
        bool Case(bool useBias, int paramCount, int ifNodes)
        {
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs(
                [TensorData([], 3L), TensorData([], useBias), x]));
            return arch.InitializeTrainableParams(rngConfig: RngConfig.Default).ModelParams.Length == paramCount
                && IfCount(arch) == ifNodes
                && arch.InputNames.Contains("useBias");
        }
        Assert.True(Case(useBias: false, paramCount: 1, ifNodes: 0));
        Assert.True(Case(useBias: true, paramCount: 2, ifNodes: 1));
    }

    /// <summary>Branching on a graph value inside a loop body concretizes for either value of the
    /// gate: the IF_OPEN stays inside the enclosing loop, so its IF_CLOSE is not stranded and the
    /// node order stays valid. No trainable parameters are involved.</summary>
    [Fact]
    public void TestAnIfElseInsideALoopBodyConcretizes()
    {
        var x = TensorData([2L], 1f, 2f);
        float[] Run(bool flag)
        {
            var g = Modules.LoopGateHyperLayer.ComputationGraph;
            TensorData[] hints = [TensorData([], 3L), TensorData([], flag), x];
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([.. hints]));
            return Floats(new ComputeContext().Execute(arch.ToConcreteModel(RngConfig.Default), hints)[0]);
        }
        Assert.Equal([4f, 5f], Run(flag: true));
        Assert.Equal([8f, 16f], Run(flag: false));
    }

    [Fact]
    public void TestLoopInvariantHoistingStaysInsideIfScopes()
    {
        static bool Ordered(ComputationGraph g, TensorData[] hints, params string[] ops)
        {
            var lowered = g.ToConcreteArchitecture(g.FromOrderedInputs([.. hints])).ToInternal();
            var at = ops.Select(op => lowered.Nodes.FindIndex(n => n.OpCode == op)).ToList();
            return lowered.IsLinearOrderValid()
                && at.All(i => i >= 0)
                && at.Zip(at.Skip(1)).All(p => p.First < p.Second);
        }

        var x = TensorData([2L], 1f, 2f);
        TensorData[] gated = [TensorData([], 3L), TensorData([], true), x];

        Assert.True(Ordered(Modules.LoopLazyOptionalLayer.ComputationGraph, [TensorData([], 3L), x, x],
            OpCodes.OPTIONAL_HAS_ELEMENT, OpCodes.LOOP_OPEN, OpCodes.IF_OPEN, OpCodes.OPTIONAL_GET_ELEMENT, OpCodes.IF_CLOSE));
        Assert.True(Ordered(Modules.LazyIfLoopBodyLayer.ComputationGraph, gated,
            OpCodes.IF_OPEN, OpCodes.MUL, OpCodes.LOOP_OPEN, OpCodes.LOOP_CLOSE, OpCodes.IF_CLOSE));
        Assert.True(Ordered(Modules.InvariantGateInLoopLayer.ComputationGraph, gated,
            OpCodes.LOOP_OPEN, OpCodes.IF_OPEN, OpCodes.IF_CLOSE, OpCodes.ADD, OpCodes.LOOP_CLOSE));

        var lazyOptional = Modules.LoopLazyOptionalLayer.ComputationGraph;
        var lowered = lazyOptional.ToConcreteArchitecture(lazyOptional.FromOrderedInputs([TensorData([], 3L), x, x]))
                                  .ToConcreteModel().ToInternal();
        IData[] absent = [TensorData([], 3L), x, OptionalTensorData.None(DType.Float32)];
        Assert.Equal([8f, 16f],
            ((TensorData<float32>)new QuickExecutionEngine().Execute(lowered, absent)[0]).AccessMemory().ToArray());
    }

    /// <summary>A draw has no inputs to be loop-dependent on, but a second execution of one is a
    /// second sample, so shrinking the loop must not lift it out either — the same wrong answer
    /// <see cref="TestAZeroInputOpInALoopBodyStaysInTheLoopBody"/> guards, reached through
    /// concretization rather than the module build.</summary>
    [Fact]
    public void TestLoopInvariantHoistingLeavesADrawInTheLoopBody()
    {
        static void BodyKeeps(ComputationGraph g, params string[] ops)
        {
            string[] lowered = [.. g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([], 3L)]))
                                     .ToInternal().Nodes.Select(n => n.OpCode)];
            foreach (var op in ops)
                Assert.InRange(
                    Array.IndexOf(lowered, op),
                    Array.IndexOf(lowered, OpCodes.LOOP_OPEN) + 1,
                    Array.IndexOf(lowered, OpCodes.LOOP_CLOSE) - 1);
        }

        BodyKeeps(ScanZeroInputOpInLoopBody.ComputationGraph, OpCodes.RANDOM_UNIFORM);
        BodyKeeps(ZeroInputOpInLoopBody.ComputationGraph, OpCodes.RANDOM_UNIFORM, OpCodes.ADD);
    }

    /// <summary>Unrolling a constant-trip loop clones the body per iteration. A draw is
    /// loop-invariant by dataflow, so it would otherwise be shared — leaving every unrolled
    /// iteration reading the one sample.</summary>
    [Fact]
    public void TestUnrollingALoopGivesEachIterationItsOwnDraw()
    {
        var g = Modules.ConstantTripDrawScanLayer.ComputationGraph;
        var lowered = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([], 1f)])).ToInternal();
        Assert.Equal(3, lowered.Nodes.Count(n => n.OpCode == OpCodes.RANDOM_UNIFORM));
    }

    /// <summary>A callee first used inside a loop body. Its Function is cached per method, so its
    /// input markers reach the caller's trace on the first call only, and whether that call is the
    /// one inside the loop depends on suite ordering — hence a callee this test alone uses.</summary>
    [Fact]
    public void TestAnInitializerFirstUsedInsideALoopBodyBuilds()
        => Assert.True(AutoTest.AdvancedTestGraph<Modules.InitializerFirstUsedInLoopBodyLayer>(
            hyperparamInputs: [TensorData([], 3L)],
            runtimeInputs: [TensorData([2L], 1f, 2f)],
            expected: [4.0, 5.0]));

    /// <summary>Nesting an IF and a loop three deep survives concretization but not the ONNX
    /// build. Tracked as Shorokoo/Shorokoo#270.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#270: FastScopeConfigurator breaks or refuses a three-deep IF/loop nesting")]
    public void TestNestedIfAndLoopScopesLowerToOnnx()
    {
        var x = TensorData([2L], 1f, 2f);
        IData[] runtime = [TensorData([], 3L), TensorData([], true), x];

        var inner = Modules.IfInLoopInIfLayer.ComputationGraph;
        var spec = inner.Specialize(inner.FromOrderedInputs([TensorData([], 3L)]));
        var specModel = spec.ToConcreteArchitecture(spec.FromOrderedInputs([TensorData([], true), x]))
                            .ToConcreteModel(RngConfig.Default);
        Assert.Equal([10f, 20f], Floats(new ComputeContext().Execute(specModel, [TensorData([], true), x])[0]));

        var outer = Modules.LoopInLazyIfInLoopLayer.ComputationGraph;
        var outerModel = outer.ToConcreteArchitecture(outer.FromOrderedInputs([.. runtime.Cast<TensorData>()]))
                              .ToConcreteModel(RngConfig.Default);
        Assert.Equal([25f, 50f], Floats(new ComputeContext().Execute(outerModel, runtime)[0]));
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
        // Round trip keeps the callee a function, so these reach FunctionProto emission,
        // which the inlined-away forms above never do.
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<Modules.CallerOfPassThroughSub>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)],
            expected: [2.0, 4.0]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<Modules.CallerOfSwapSub>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)],
            expected: [1.0, 2.0]));
        Assert.True(AutoTest.AdvancedTestGraphWithModuleGraphRoundtrip<Modules.CallerOfRepeatedOutputSub>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L], 1f, 2f)],
            expected: [6.0, 12.0]));
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
        using var ms = new MemoryStream();
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

        using var ms = new MemoryStream();
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
        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)DoubleScalar);
        Assert.Equal(FunctionType.Module, fn.FunctionType);

        var input = InvokeInput("input");
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

    private static float[] ConcretizeAndRun(ComputationGraph g, params TensorData[] inputs) =>
        RunFloats(g.ToConcreteArchitecture(g.FromOrderedInputs([.. inputs])).ToConcreteModel(), inputs);

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
    /// as a concretization hint. Every module here allocates trainable params inside a
    /// <c>LoopAPI.Iterate</c> body: the first two shape theirs from the baked hyper and from
    /// the input's shape tensor respectively, and the third sits in a loop whose trip count
    /// was already constant before the bake.</summary>
    [Fact]
    public void TestBakingHypersMatchesHintingThemWhenParamsLiveInsideALoopBody()
    {
        var input = TensorDataWithSmallVals(DType.Float32, [2L, 5L]);
        AssertBakedHypersMatchHintedHypers(LoopLayer.ComputationGraph,
            [TensorData(DType.Int64, [], 4L), TensorData(DType.Int64, [], 3L)], input);
        AssertBakedHypersMatchHintedHypers(ConditionalTrainableParamInLoopLayer.ComputationGraph,
            [TensorData(DType.Int64, [], 3L), TensorData(DType.Int64, [], 4L)], input);
        AssertBakedHypersMatchHintedHypers(TrainableWithHyperInLoop.ComputationGraph,
            [TensorData(DType.Int64, [], 5L)], input);
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
        using var ms = new MemoryStream();
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

    private static void AssertDrawInsideLoopBody(ComputationGraph graph, string drawOpCode = OpCodes.RANDOM_UNIFORM)
    {
        string[] codes = [.. graph.ToInternal().Nodes.Select(n => n.OpCode)];
        Assert.InRange(
            Array.IndexOf(codes, drawOpCode),
            Array.IndexOf(codes, OpCodes.LOOP_OPEN) + 1,
            Array.IndexOf(codes, OpCodes.LOOP_CLOSE) - 1);
    }

    /// <summary>A scan input resolves through the looper's outer-scope fallback when the scanned
    /// value comes from a node it does not track, which names the first pass's node outside the
    /// loop. Binding that would stack one draw once per iteration, so the scan takes the fallback
    /// only when it names a loop carry.</summary>
    [Fact]
    public void TestScanningAZeroInputOpKeepsTheDrawInsideTheLoopBody()
    {
        AssertDrawInsideLoopBody(ScanZeroInputOpInLoopBody.ComputationGraph);
        AssertDrawInsideLoopBody(ScanKeyedFeedInLoopBody.ComputationGraph, InternalOpCodes.SHRK_RANDOM_UNIFORM);
    }

    /// <summary>A node the loop body creates stays in the body whether or not it has inputs. The
    /// position is asserted rather than a value because the fault it guards is in the graph, so
    /// every engine would agree on the wrong answer.</summary>
    [Fact]
    public void TestAZeroInputOpInALoopBodyStaysInTheLoopBody()
        => AssertDrawInsideLoopBody(ZeroInputOpInLoopBody.ComputationGraph);

    /// <summary>A local carrying the value another held one iteration ago reads that value, not
    /// the one from before the loop — accumulated, and scanned.</summary>
    [Fact]
    public void TestALagOneLoopCarryCarriesThePreviousIterationsValue()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [], 10f), TensorData(DType.Int64, [], 3L)];
        Assert.True(AutoTest.AdvancedTestGraph<LagOneCarry>(
            hyperparamInputs: [], runtimeInputs: inputs, expected: [31.0]));
        Assert.True(AutoTest.AdvancedTestGraph<LagOneCarryScanned>(
            hyperparamInputs: [], runtimeInputs: inputs, expected: [10.0, 10.0, 11.0]));
    }

    /// <summary>Reading a lag carry after the loop needs the assignment wrapped, exactly as
    /// carrying a value computed outside the body does: a bare assignment produces no body node
    /// for the loop to point the variable at, and is refused naming the wrapper rather than
    /// asserting in Debug and exporting a broken scope in Release.</summary>
    [Fact]
    public void TestALagOneCarryIsReadAfterTheLoopThroughLoopApiCarry()
    {
        TensorData[] inputs = [TensorData(DType.Float32, [], 10f), TensorData(DType.Int64, [], 3L)];
        Assert.True(AutoTest.AdvancedTestGraph<LagOneCarryWrappedReadAfterLoop>(
            hyperparamInputs: [], runtimeInputs: inputs, expected: [43.0]));
        Assert.Contains("LoopAPI.Carry", Assert.Throws<UnsupportedLoopVariableAssignmentException>(
            () => LagOneCarryReadAfterLoop.ComputationGraph).Message);
    }

    /// <summary>An enclosing loop's ctx.Scan called from inside a nested loop's body records the
    /// value the outer body ends each of its iterations with.</summary>
    [Fact]
    public void TestTheOuterLoopsScanCanBeCalledFromAnInnerLoopBody()
        => Assert.True(AutoTest.AdvancedTestGraph<OuterScanFromInnerBody>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 10f), TensorData(DType.Int64, [], 2L), TensorData(DType.Int64, [], 3L)],
            expected: [13.0, 16.0]));

    private static Tensor<float32> MachineryFreeBody(Tensor<float32> x) => x + x;

    /// <summary>Every entry point that can hand a graph to ONNX Runtime refuses un-lowered module
    /// machinery first, naming the module to lower rather than an internal op in an "invalid
    /// model" — and names no module where none can be written.</summary>
    [Fact]
    public void TestEagerEvalRefusesAModuleOutputAndNamesTheLowering()
    {
        var x = Tensor([2L], 1.0f, 2.0f);
        var sample = TensorData([2L], 1.0f, 2.0f);
        var linearInput = Tensor([4L, 4L], [.. Enumerable.Repeat(0.5f, 16)]);
        Variable[] noInputs = [];
        InternalComputationGraph ModuleGraph() => ScalarMultiplyModel.ComputationGraph.ToInternal();

        (string Operation, string Op, string? Module, Action Run)[] cases =
        [
            ("OnnxEngine.Eval", InternalOpCodes.CREATE_MODULE, nameof(ScalarMultiplyModel),
                () => OnnxEngine.Eval(ScalarMultiplyModel.Call(x))),
            ("OnnxEngine.Eval", InternalOpCodes.CREATE_MODULE, nameof(ScalarMultiplyModel),
                () => OnnxEngine.Eval([ScalarMultiplyModel.Call(x)])),
            ("OnnxEngine.Eval", InternalOpCodes.CREATE_MODULE, "Linear",
                () => OnnxEngine.Eval(Shorokoo.Modules.Layers.Linear.Call(Scalar(4L), Scalar(false), linearInput))),
            ("ComputeContext.Eval", InternalOpCodes.CREATE_MODULE, nameof(ScalarMultiplyModel),
                () => new ComputeContext().Eval(ScalarMultiplyModel.Call(x))),
            ("Tensor.Eval", InternalOpCodes.CREATE_MODULE, nameof(ScalarMultiplyModel),
                () => ScalarMultiplyModel.Call(x).Eval()),
            ("Eval(...).With", InternalOpCodes.CREATE_MODULE, nameof(ScalarMultiplyModel),
                () => noInputs.Eval(ScalarMultiplyModel.Call(x)).With([])),
            ("ComputeContext.Execute", InternalOpCodes.MODEL_PARAM_REF, null,
                () => ComputeContext.Default.Execute(ModuleGraph(), sample)),
            ("ComputeContext.Compile", InternalOpCodes.MODEL_PARAM_REF, null,
                () => ComputeContext.Default.Compile(ModuleGraph())),
            ("ComputeContext.Run", InternalOpCodes.MODEL_PARAM_REF, null,
                () => ComputeContext.Default.Run(ModuleGraph())),
            // The stamped gate hands over to the same refusal rather than offering WithKind, which
            // on a graph that really is a module is an invitation to make it lie about itself.
            ("ComputeContext.Execute", InternalOpCodes.MODEL_PARAM_REF, null,
                () => ComputeContext.Default.Execute(ScalarMultiplyModel.ComputationGraph, sample)),
            // A module built from a delegate names its declaring type, which has no
            // ComputationGraph — so the name is a label, never spelled into code.
            ("OnnxEngine.Eval", InternalOpCodes.CREATE_MODULE, nameof(DelegateModuleBody),
                () => OnnxEngine.Eval(ModuleFactory
                    .FromFunc<Tensor<float32>, Tensor<float32>>(DelegateModuleBody.Double)
                    .SetHyperparams().Call(x))),
        ];

        foreach (var (operation, op, module, run) in cases)
        {
            var message = Assert.Throws<InvalidOperationException>(run).Message;
            Assert.StartsWith($"{operation} requires a concretized graph", message);
            Assert.Contains("'module'", message);
            Assert.Contains(op, message);
            Assert.Contains("ToConcreteArchitecture", message);
            Assert.Contains("ToConcreteModel", message);
            Assert.DoesNotContain("WithKind", message);
            Assert.DoesNotContain("var g = ", message);
            if (module is null)
                Assert.DoesNotContain("It comes from module", message);
            else
                Assert.Contains($"It comes from module '{module}'", message);
        }
    }

    /// <summary>The refusal's remedy is the one that works — for a module with <c>[Hyper]</c>
    /// parameters as much as without — and the gate leaves alone what does run: a module-typed
    /// function invoke, the initializer invokes <c>ToConcreteModel</c> executes, plain ops.</summary>
    [Fact]
    public void TestTheLoweringTheRefusalNamesRunsAndRunnableGraphsAreUntouched()
    {
        var sample = TensorData([2L], 1.0f, 2.0f);
        var g = ScalarMultiplyModel.ComputationGraph;
        var model = g.ToConcreteArchitecture(g.FromOrderedInputs([sample])).ToConcreteModel();
        Assert.Equal([1.0f, 2.0f],
            ComputeContext.Default.Execute(model, sample)[0].ToTensorData().As<float32>().AccessMemory<float>().ToArray());

        // Hypers are graph inputs, and come first — the order the message tells the reader to use.
        var lg = Shorokoo.Modules.Layers.Linear.ComputationGraph;
        IData[] linear = [TensorData([], 4L), TensorData([], false),
            TensorData([4L, 4L], [.. Enumerable.Repeat(0.5f, 16)])];
        Assert.Equal(["outFeatures", "useBias", "x"], lg.InputNames);
        var linearModel = lg.ToConcreteArchitecture(lg.FromOrderedInputs([.. linear.Cast<TensorData>()]))
            .ToConcreteModel();
        Assert.Equal(new Shape(4L, 4L), ComputeContext.Default.Execute(linearModel, linear)[0].ToTensorData().Shape);

        // A module-typed function invoke is inlined and runs; refusing it would refuse a graph that
        // executes correctly.
        Assert.Equal([2.0f, 4.0f], ComputeContext.Default.Execute(ModuleInvokeGraph(), sample)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray());


        // A generic module's graph carries #GenericTypeInput#, a module-stage op that is never
        // emitted as a node: it becomes a graph input, so the graph runs.
        Assert.Equal([1f, 2f], ComputeContext.Default
            .Execute(SimpleGenericLayer.ComputationGraph.ToInternal(), TensorData([], 0f), sample)[0]
            .ToTensorData().As<float32>().AccessMemory<float>().ToArray());

        var both = Assert.Throws<InvalidOperationException>(() => OnnxEngine.Eval(
            [ScalarMultiplyModel.Call(Tensor([2L], 1.0f, 2.0f)), SimplestLayer.Call(Tensor([2L], 1.0f, 2.0f))]))
            .Message;
        Assert.Contains($"'{nameof(ScalarMultiplyModel)}', '{nameof(SimplestLayer)}'", both);

        Assert.Equal(5f, OnnxEngine.Eval(Scalar(2f) + Scalar(3f)).As<float32>().AccessMemory()[0]);
    }

    [Fact]
    public void TestModuleTypedFunctionInvokesInlineTheirArgsHyperparamsAndParams()
    {
        var input = TensorData([2L], 1f, 2f);
        Assert.Equal([2f, 4f], RunInvoke((Func<Tensor<float32>, Tensor<float32>>)DoubleScalar, x => [x], input));
        Assert.Equal([3f, 6f], RunInvoke((Func<Tensor<float32>, Scalar<float32>, Tensor<float32>>)ScaledByHyper, x => [Scalar(3f), x], input));
        Assert.Equal([1f, 2f], RunInvoke((Func<Tensor<float32>, Tensor<float32>>)TimesOwnParam, x => [x], input));
        Assert.Equal([1f, 2f], RunInvoke((Func<Tensor<float32>, Tensor<float32>>)PassThrough, x => [x], input));
    }

    private static Tensor<float32> PassThrough(Tensor<float32> t) => t;

    [Fact]
    public void TestAFunctionCallCountingArgumentsAgainstTheBodyRefusesEveryMismatch()
    {
        string Arity(Delegate body, int argCount) => Assert.Throws<ModuleException>(
            () => ModuleFn(body).Call([.. Enumerable.Range(0, argCount).Select(i => InvokeInput($"a{i}"))])).Message;

        var oneIn = (Func<Tensor<float32>, Tensor<float32>>)DoubleScalar;
        var twoIn = (Func<Tensor<float32>, Scalar<float32>, Tensor<float32>>)ScaledByHyper;

        Assert.Contains("0 argument(s)", Arity(oneIn, 0));
        Assert.Contains("2 argument(s)", Arity(oneIn, 2));
        Assert.Contains("1 hyperparameter(s)", Arity(twoIn, 1));
        Assert.Contains("3 argument(s)", Arity(twoIn, 3));
        Assert.Contains("null argument", Assert.Throws<ModuleException>(
            () => ModuleFn(twoIn).Call(InvokeInput("a"), null)).Message);
        Assert.Contains("0 argument(s)", Assert.Throws<ModuleException>(
            () => ModuleFn(twoIn).Call(null!)).Message);
    }

    [Fact]
    public void TestImportingAModelWhoseFunctionCallMiscountsItsInputsIsRefused()
    {
        var g = ComputationGraph.FromInternal(ModuleInvokeGraph(), GraphKind.Module);
        var onnx = SrkFileFormat.Read(CompressedFormatUtils.SaveFastGraphToBinary(g, compressed: false)).OnnxBytes;

        using var read = new MemoryStream(onnx);
        var proto = ProtoBuf.Serializer.Deserialize<ModelProto>(read);
        var fnNames = proto.Functions.Select(f => f.Name).ToHashSet();
        proto.Graph.Nodes.Single(n => fnNames.Contains(n.OpType)).Inputs.Clear();

        using var ms = new MemoryStream();
        ProtoBuf.Serializer.Serialize(ms, proto);
        var ex = Assert.Throws<ModuleException>(() => OnnxModelImporter.FromOnnxModel(ms.ToArray()));
        Assert.Contains("0 input(s)", ex.Message);
    }

    [Fact]
    public void TestEachCallSiteOfAModuleTypedFunctionGetsItsOwnParameter()
    {
        var fn = ModuleFn((Func<Tensor<float32>, Scalar<int64>, Tensor<float32>>)SizedByHyper);
        var arch = ConcretizeInvokes(x =>
            [(Tensor<float32>)fn.Call(Scalar(2L), x)[0], (Tensor<float32>)fn.Call(Scalar(5L), x)[0]]);
        var infos = arch.GetConcreteModelParamInfos();
        Assert.Equal(2, infos.ModelIds.Distinct().Count());
        Assert.Equal(2, infos.ParamInfos.Select(i => i.ToShorokooIdString()).Distinct().Count());

        var run = ComputeContext.Default.Execute(arch.ToConcreteModel(), TensorData([2L], 1f, 2f));
        Assert.Equal(2, run[0].ToTensorData().As<float32>().AccessMemory<float>().Length);
        Assert.Equal(5, run[1].ToTensorData().As<float32>().AccessMemory<float>().Length);
    }

    [Fact]
    public void TestACallSiteIdNeverTakesTheSlotReservedForTheRngSeed()
    {
        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)TimesOwnParam);
        var ids = ConcretizeInvokes(x => [(Tensor<float32>)fn.Call(x)[0]])
            .GetConcreteModelParamInfos().ModelIds;
        Assert.NotEmpty(ids);
        Assert.All(ids, id => Assert.NotEqual(0, id.Vals[0]));
    }

    [Fact]
    public void TestOneModuleReachedThroughBothInvokeFormsKeepsTwoParameterNames()
    {
        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)TimesOwnParam);
        var x = InvokeInput("input");
        var viaModel = ModuleFactory
            .FromFunc<Tensor<float32>, Tensor<float32>>(TimesOwnParam, "TimesOwnParam")
            .SetHyperparams().Call(x);
        var g = ComputationGraph.FromInternal(
            new InternalComputationGraph([x], [(Tensor<float32>)fn.Call(x)[0] + viaModel]), GraphKind.Module);

        var input = TensorData([2L], 1f, 2f);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        var names = arch.GetConcreteModelParamInfos().ParamInfos
            .Select(i => i.ToShorokooIdString()).ToList();
        Assert.Equal(2, names.Count);
        Assert.Equal(2, names.Distinct().Count());
        Assert.Equal([2f, 4f], RunFloats(arch.ToConcreteModel(), input));
    }

    /// <summary>What a call site realizes per iteration is decided by where the callee is
    /// CREATED, not by which invoke form reaches it: one call site of one callee shares its
    /// parameter across the loop's iterations — weight sharing — and only creating the model
    /// inside the body makes a parameter per iteration.</summary>
    [Fact]
    public void TestAModuleInvokedInALoopSharesOneParameterAcrossIterations()
    {
        int ParamIdentities(Func<Tensor<float32>, Tensor<float32>> callOnce)
        {
            var x = InvokeInput("input");
            var acc = x;
            foreach (var _ in LoopAPI.Iterate(Scalar(3L))) acc = callOnce(acc);
            var g = ComputationGraph.FromInternal(new InternalComputationGraph([x], [acc]), GraphKind.Module);
            var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([2L], 1f, 2f)]));
            return arch.GetConcreteModelParamInfos().ModelIds.Distinct().Count();
        }
        static Model<Tensor<float32>, Tensor<float32>> NewModel() => ModuleFactory
            .FromFunc<Tensor<float32>, Tensor<float32>>(TimesOwnParam, "TimesOwnParam").SetHyperparams();

        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)TimesOwnParam);
        var createdOutside = NewModel();
        Assert.Equal(1, ParamIdentities(t => (Tensor<float32>)fn.Call(t)[0]));
        Assert.Equal(1, ParamIdentities(createdOutside.Call));
        Assert.Equal(3, ParamIdentities(t => NewModel().Call(t)));
    }

    [Fact]
    public void TestAGenericModuleConcretizesThroughAnyDepthOfNonGenericCallers()
    {
        var input = TensorData([2L], 1f, 2f);
        Assert.Equal([2f, 4f], ConcretizeAndRun(WrapsNonGenericCallerOfGenericModule.ComputationGraph, input));
        Assert.Equal([2f, 4f], ConcretizeAndRun(WrapsWrapperOfGenericModule.ComputationGraph, input));
        Assert.Equal([10f, 20f], ConcretizeAndRun(CallsGenericModulesSeveralWays.ComputationGraph, input));
        Assert.Equal([10f, 20f], ConcretizeAndRun(WrapsCallerOfGenericModulesSeveralWays.ComputationGraph, input));
    }

    [Fact]
    public void TestAGenericModuleConcretizesThroughEveryKindOfReferenceThatNamesIt()
    {
        var input = TensorData([2L], 1f, 2f);
        Assert.Equal([2f, 4f], ConcretizeAndRun(PassesAGenericCallerAsAModelParameter.ComputationGraph, input));
        Assert.Equal([2f, 4f], ConcretizeAndRun(HoldsAGenericModuleInAModelSequence.ComputationGraph, input));
        Assert.Equal([2f, 4f], ConcretizeAndRun(AppendsAGenericModuleToAnEmptyModelSequence.ComputationGraph, input));
    }

    [Fact]
    public void TestErasureLeavesAFunctionWithNoGenericMaterialBelowItAlone()
    {
        var before = CallsAGenericModuleAndAPlainNeighbour.ComputationGraph.ToInternal();
        Function Neighbour(InternalComputationGraph g) => g.Nodes.Select(n => n.TargetFunction).NotNulls()
            .Single(f => f.DefaultName.Contains(nameof(PlainNonGenericNeighbour)));

        Assert.Same(Neighbour(before), Neighbour(FastToConcreteDataType.Process(before)));
    }

    [Fact]
    public void TestAGenericModuleInASequenceConcretizesWhenTheGraphUsesTwoTypeArguments()
    {
        var input = TensorData([2L], 1f, 2f);
        Assert.Equal([4f, 8f], ConcretizeAndRun(HoldsOneOfTwoSpecializationsInAModelSequence.ComputationGraph, input));
        Assert.Equal([4f, 8f], ConcretizeAndRun(UsesTheSequenceHoldingGenericAtTwoTypeArguments.ComputationGraph, input));
    }

    /// <summary>A generic parameter initializer called with an explicit type argument from a
    /// non-generic body is spliced with its shape input wired to nothing.
    /// Tracked as Shorokoo/Shorokoo#295.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#295: a generic param initializer called from a non-generic body loses its shape input")]
    public void TestAGenericParamInitializerCalledFromANonGenericBodyConcretizes()
    {
        var g = NonGenericCallerOfGenericParamInitializer.ComputationGraph;
        Assert.Equal([1f, 2f], ConcretizeAndRun(g, TensorData([1L], 2L), TensorData([2L], 1f, 2f)));
    }

    [Fact]
    public void TestACallSiteParameterNameIsNotShiftedByAModuleWhoseNameEndsWithTheCallSites()
    {
        var input = TensorData([2L], 1f, 2f);
        string[] Names(bool withDecoy)
        {
            var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)TimesOwnParam);
            var x = InvokeInput("input");
            var body = (Tensor<float32>)fn.Call(x)[0];
            if (withDecoy)
                body += ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(
                    TimesOwnParamDecoy, "X" + nameof(ModulesCoverageTests)).SetHyperparams().Call(x);
            var g = ComputationGraph.FromInternal(new InternalComputationGraph([x], [body]), GraphKind.Module);
            return [.. g.ToConcreteArchitecture(g.FromOrderedInputs([input]))
                .GetConcreteModelParamInfos().ParamInfos.Select(i => i.ToShorokooIdString())];
        }

        var callSite = $".{nameof(ModulesCoverageTests)}#0.";
        Assert.Contains(Names(withDecoy: false), n => n.Contains(callSite));
        Assert.Contains(Names(withDecoy: true), n => n.Contains(callSite));
    }

    private static Tensor<float32> TimesOwnParamDecoy(Tensor<float32> t) => t * InitSimple.Init([Scalar(2L)]);

    private static Tensor<float32> SizedByHyper(Tensor<float32> t, [Hyper] Scalar<int64> n)
        => InitSimple.Init([n]);

    [Fact]
    public void TestAModuleNameCarryingATemplateSeparatorStillGetsOneNamePerCallSite()
    {
        // The name is registered dotted first, so both routes below share it; templates store part
        // names escaped, so a raw-name scan would find nothing and hand both sites index 0.
        var viaModel = ModuleFactory
            .FromFunc<Tensor<float32>, Tensor<float32>>(DottedNameBody, "Dotted.Layer").SetHyperparams();
        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)DottedNameBody);
        var arch = ConcretizeInvokes(x => [(Tensor<float32>)fn.Call(x)[0] + viaModel.Call(x)]);
        var names = arch.GetConcreteModelParamInfos().ParamInfos.Select(i => i.ToShorokooIdString()).ToList();
        Assert.Equal(2, names.Count);
        Assert.Equal(2, names.Distinct().Count());
    }

    [Fact]
    public void TestTwoCallSitesOfADrawingBodyGetSeparateRngStreams()
    {
        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)DrawsOnce);
        var arch = ConcretizeInvokes(x => [(Tensor<float32>)fn.Call(x)[0] - (Tensor<float32>)fn.Call(x)[0]]);
        var zero = TensorData([2L], 0f, 0f);
        Assert.All(
            RunFloats(arch.ToConcreteModel(RngConfig.Default), zero),
            v => Assert.NotEqual(0f, v));
    }

    /// <summary>A draw inside a module invoked from a loop is re-executed per iteration, and a
    /// second execution of a draw is a second sample: the invoke site's loop scope reaches the
    /// spliced feed even though the model was created outside the loop.</summary>
    [Fact]
    public void TestAModuleInvokedInALoopDrawsAFreshSamplePerIteration()
    {
        var zero = TensorData([2L], 0f, 0f);
        var fn = ModuleFn((Func<Tensor<float32>, Tensor<float32>>)DrawsOnce);
        var arch = ConcretizeInvokes(x =>
        {
            var acc = x;
            foreach (var _ in LoopAPI.Iterate(Scalar(3L))) acc = (Tensor<float32>)fn.Call(acc)[0];
            return [acc];
        }, zero);
        var three = RunFloats(arch.ToConcreteModel(RngConfig.Default), zero);

        var once = ConcretizeInvokes(x => [(Tensor<float32>)fn.Call(x)[0]], zero);
        var one = RunFloats(once.ToConcreteModel(RngConfig.Default), zero);
        Assert.All(three.Zip(one, (t, o) => Math.Abs(t - 3 * o)), d => Assert.True(d > 1e-5f));

        // The MODEL_INVOKE route, with the model created OUTSIDE the loop, took its scope from
        // that creation site and reused one sample the same way.
        var model = ModuleFactory.FromFunc<Tensor<float32>, Tensor<float32>>(DrawsOnce, "DrawsOnce")
            .SetHyperparams();
        var viaModel = ConcretizeInvokes(x =>
        {
            var acc = x;
            foreach (var _ in LoopAPI.Iterate(Scalar(3L))) acc = model.Call(acc);
            return [acc];
        }, zero);
        var modelThree = RunFloats(viaModel.ToConcreteModel(RngConfig.Default), zero);
        var modelOnce = ConcretizeInvokes(x => [model.Call(x)], zero);
        var modelOne = RunFloats(modelOnce.ToConcreteModel(RngConfig.Default), zero);
        Assert.All(modelThree.Zip(modelOne, (t, o) => Math.Abs(t - 3 * o)), d => Assert.True(d > 1e-5f));
    }

    private static Tensor<float32> DottedNameBody(Tensor<float32> t) => t * InitSimple.Init([Scalar(2L)]);

    private static Tensor<float32> DrawsOnce(Tensor<float32> t) => t + RandomUniform([Scalar(2L)], 0f, 1f);

    private static Tensor<float32> ScaledByHyper(Tensor<float32> t, [Hyper] Scalar<float32> h) => t * h;

    private static Tensor<float32> TimesOwnParam(Tensor<float32> t) => t * InitSimple.Init([Scalar(2L)]);

    private static Function ModuleFn(Delegate body) => ModuleHelper.CreateTargetFunction(body);

    private static Tensor<float32> InvokeInput(string name)
        => (Tensor<float32>)InternalOp.ModuleTensorInput(
            DType.Float32, rank: 1, InputType.ModelInput, targetFunction: null, defaultName: name);

    private static float[] RunInvoke(Delegate body, Func<Tensor<float32>, Variable[]> callArgs, TensorData input)
        => RunFloats(
            ConcretizeInvokes(x => [(Tensor<float32>)ModuleFn(body).Call(callArgs(x))[0]], input).ToConcreteModel(),
            input);

    private static ComputationGraph ConcretizeInvokes(
        Func<Tensor<float32>, Variable[]> outputs, TensorData? input = null)
    {
        var hint = input ?? TensorData([2L], 1f, 2f);
        var x = InvokeInput("input");
        var g = ComputationGraph.FromInternal(
            new InternalComputationGraph([x], [.. outputs(x)]), GraphKind.Module);
        return g.ToConcreteArchitecture(g.FromOrderedInputs([hint]));
    }

    [Fact]
    public void TestAGenericModuleCanBeConcretizedThroughPublicApi()
    {
        var input = TensorData([2L], 1f, 2f);
        Assert.Equal([1f, 2f], ConcretizeAndRun(SimpleGenericLayer.ComputationGraph, input));
        Assert.Equal([3f, 6f], ConcretizeAndRun(GenericScaleLayer.ComputationGraph, TensorData([], 3f), input));
        Assert.Equal([1f, 2f], ConcretizeAndRun(GenericLayerWithTrainableParams.ComputationGraph, TensorData([1L], 2L), input));
        Assert.Equal([2f, 4f], ConcretizeAndRun(NonGenericCallerOfGenericModule.ComputationGraph, input));
    }

    /// <summary>A module body reachable as a delegate, for a module the source generator never saw.</summary>
    private static class DelegateModuleBody
    {
        public static Tensor<float32> Double(Tensor<float32> t) => t + t;
    }

    /// <summary>A bare module-typed function invoke over a machinery-free body.</summary>
    private static InternalComputationGraph ModuleInvokeGraph()
    {
        var input = InvokeInput("input");
        return new InternalComputationGraph(
            [input], [(Tensor<float32>)ModuleFn((Func<Tensor<float32>, Tensor<float32>>)DoubleScalar).Call(input)[0]]);
    }

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

        if (genericTypes is not null && genericTypes.Count > 0
            && moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
            Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);

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

    private static InternalComputationGraph TinyArch(ComputationGraph model)
    {
        var g = model.ToInternal();
        return g.ToConcreteArchitecture(g.FromOrderedInputs([TensorData([2L, 8L], new float[16])]));
    }

    private static int CheckpointedInvokes(ComputationGraph model)
        => model.ToInternal().Nodes.Count(n => n.OpCode == InternalOpCodes.MODEL_INVOKE && CheckpointSegment.IsRequested(n));

    private static int Segments(InternalComputationGraph arch)
        => arch.Nodes.Select(CheckpointSegment.IdOf).OfType<long>().Distinct().Count();

    [Fact]
    public void TestModuleCheckpointAttributeMarksEachCallAsOneInlinedSegmentCoverage()
    {
        Assert.Equal(3, CheckpointedInvokes(Modules.CheckpointedTinyMlpStack.ComputationGraph));
        Assert.Equal(0, CheckpointedInvokes(Modules.PlainTinyMlpStack.ComputationGraph));

        var checkpointed = TinyArch(Modules.CheckpointedTinyMlpStack.ComputationGraph);
        var plain = TinyArch(Modules.PlainTinyMlpStack.ComputationGraph);
        Assert.Equal(3, Segments(checkpointed));
        Assert.Equal(3, checkpointed.Nodes.Count(CheckpointSegment.ProducesSegmentOutput));
        Assert.Equal(0, Segments(plain));
        Assert.Equal(plain.Nodes.Count, checkpointed.Nodes.Count);
    }
}
