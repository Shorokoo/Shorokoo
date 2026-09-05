using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Modules whose graph shape targets otherwise-uncovered branches of
/// <c>Shorokoo.Core.Nodes.Processors.Fast.FastProcessors</c> — TensorStruct slots in LOOP / IF
/// control flow, TensorStruct-typed sequence ops, constant-loop unrolling edge cases, module
/// reparenting and static trainable-param selection — driven through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/>.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class FastProcessorsCoverageTests
{
    private static TensorData Scalar32(float v) => TensorData(DType.Float32, [], v);
    private static double[] Rep(double v, int n) => [.. Enumerable.Repeat(v, n)];
    private static TensorData Flag(bool v) => TensorData(DType.Bool, [], v);

    /// <summary>Only a pruned parameter's shape is ever observed, so concretizing one must not
    /// cost its whole tensor in host memory: the gap between pruning a small bias and a large one
    /// stays far below the tensor itself.</summary>
    [Fact]
    public void TestPrunedParamCandidateIsNotMaterializedInFull()
    {
        static long ConcretizationBytes(InternalComputationGraph g)
        {
            var inputs = g.FromOrderedInputs([TensorData([1L, 4L], [1f, 2f, 3f, 4f])]);
            var before = GC.GetAllocatedBytesForCurrentThread();
            g.ToConcreteArchitecture(inputs);
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }

        ConcretizationBytes(PrunedBiasSmallLinear.ComputationGraph.ToInternal());
        var small = ConcretizationBytes(PrunedBiasSmallLinear.ComputationGraph.ToInternal());
        var large = ConcretizationBytes(PrunedBiasLargeLinear.ComputationGraph.ToInternal());
        Assert.True(large - small < 8L << 20);
    }

    [Fact]
    public void TestTensorStructInControlFlow()
    {
        // Bare TensorStruct loop vars / IF branches: ExpandLoopOpenStructLoopVars,
        // ExpandLoopCloseStructLoopVars, ExpandIfCloseStructBranches.
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructLoopCarry>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(1f), Scalar32(2f)],
            expected: [12.0]));
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructIfElseReturn>(
            hyperparamInputs: [], runtimeInputs: [Flag(true), Scalar32(3f), Scalar32(5f)],
            expected: [8.0]));

        // Sequence-of-struct loop vars, the plain-tensor passthrough else branch of
        // ExpandIfCloseStructBranches, and the scan-output tail of ExpandLoopCloseStructLoopVars.
        Assert.True(AutoTest.AdvancedTestGraph<SequenceOfStructLoopCarry>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(1f), Scalar32(2f)],
            expected: [1.0]));
        Assert.True(AutoTest.AdvancedTestGraph<IfElseMixedStructAndPlainSlots>(
            hyperparamInputs: [], runtimeInputs: [Flag(true), Scalar32(3f), Scalar32(5f)],
            expected: [61.0]));

        // A plain carry interleaved with a struct carry, both slot orders of the
        // expansion, unrolled (constant trip count) and not (runtime trip count).
        Assert.True(AutoTest.AdvancedTestGraph<MixedTensorStructLoop>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(1f), Scalar32(2f)],
            expected: [10.0]));
        Assert.True(AutoTest.AdvancedTestGraph<MixedTensorStructLoopRuntimeTripCount>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [2L], 2f, 5f), Scalar32(1f), Scalar32(2f)],
            expected: [10.0]));
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructLoopCarryWithScanOutput>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(1f), Scalar32(2f)],
            expected: [3.0, 6.0, 9.0]));

        // Sequence<TensorStruct> out of both IF branches, with a CONSTANT positioned after the
        // IF_CLOSE — the FastOnnxModelReader topological re-tour + IfCloseOp sequence branch.
        Assert.True(AutoTest.AdvancedTestGraph<SequenceOfStructIfElseReturn>(
            hyperparamInputs: [], runtimeInputs: [Flag(true), Scalar32(3f), Scalar32(5f)],
            expected: [3.0]));

        // TensorStruct-typed SEQUENCE_CONSTRUCT / AT / EMPTY / INSERT / ERASE / LENGTH.
        Assert.True(AutoTest.AdvancedTestGraph<SequenceOpsOnStructs>(
            hyperparamInputs: [],
            runtimeInputs: [Scalar32(1f), Scalar32(2f), Scalar32(3f), Scalar32(4f)],
            expected: [6.0]));

        // MODEL_TENSORSTRUCT_INPUT producer in the graph's input list is rewritten into one
        // MODEL_TENSOR_INPUT per struct field (architecture pipeline only — execution would need
        // a TensorDataStruct shape AdvancedTestGraph's flat TensorData[] API cannot model).
        var graph = SimplePairSum.ComputationGraph;
        Assert.Contains(graph.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT);
        var concreteArch = graph.ToConcreteArchitecture(new ModelParamList());
        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT);
        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.TENSOR_STRUCT_CREATE);
        Assert.DoesNotContain(concreteArch.ToInternal().Nodes, n => n.OpCode == InternalOpCodes.TENSOR_STRUCT_GETFIELD);
    }

    [Fact]
    public void TestModuleAndConstantLoopEdges()
    {
        // MODULE_SET_HYPERPARAMS arm of FastReparentToCallSite: three call levels are needed so
        // the middle module's body carries the inner module's node for the reparenter to rewrite.
        Assert.True(AutoTest.AdvancedTestGraph<CallsHypersLayer>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));
        Assert.True(AutoTest.AdvancedTestGraph<CallsCallsHypersLayer>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])],
            expected: Rep(0.1, 5)));

        // FastFoldConstantIterationLoops.UnrollOne non-happy paths: zero-iteration early return,
        // scan-output UNSQUEEZE+CONCAT, dynamic-break WHERE/AND gating, and the scope-pair
        // propagation arm for a nested iteration-index-dependent IF.
        Assert.True(AutoTest.AdvancedTestGraph<ZeroIterConstLoopLayer>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(42f)],
            expected: [42.0]));
        Assert.True(AutoTest.AdvancedTestGraph<ConstLoopWithScanOutput>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(0f)],
            expected: [1.0, 2.0, 3.0]));
        Assert.True(AutoTest.AdvancedTestGraph<ConstLoopWithDynamicBreak>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(0f), Flag(true)],
            expected: [3.0]));
        Assert.True(AutoTest.AdvancedTestGraph<ConstLoopWithNestedIterDependentIf>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(0f)],
            expected: [12.0]));

        // Shorokoo/Shorokoo#22: a static param site materializes through a direct MODEL_PARAM
        // reference, so no sequence param-selection machinery survives and each param keeps its
        // own canonical model id.
        var g = ((ComputationGraph)typeof(AutoGradStructConvStridePadCheck)
            .GetProperty("ComputationGraph")!.GetValue(null)!).ToInternal();
        var x = TensorData([1L, 2L, 5L, 5L], new float[1 * 2 * 5 * 5]);
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([x]));
        Assert.DoesNotContain(arch.Nodes, n => n.OpCode == OpCodes.SEQUENCE_AT);
        Assert.DoesNotContain(arch.Nodes, n => n.OpCode == OpCodes.SEQUENCE_CONSTRUCT);
        Assert.Equal(2, arch.GetConcreteModelParamInfos().ParamInfos
            .Select(p => p.ModelId).Distinct().Count());
    }

    private static bool GateBlocks(string opCode)
    {
        var graph = new InternalComputationGraph();
        foreach (var op in (string[])[OpCodes.LOOP_OPEN, opCode, OpCodes.LOOP_CLOSE])
            graph.Nodes.Add(new FastNode { Key = FastNodeKey.New(), OpCode = op });
        return FastFoldConstantIterationLoops.BodyHoldsUnresolvedParamMachinery(graph, 0, 2);
    }

    /// <summary>The unroll gate blocks only the four Stage-F param op-codes, not the whole
    /// <c>ModuleStageOps</c> set — <c>AUTO_GRAD</c> above all, which is still present at the first
    /// <c>FastSimplify</c> and whose loop must be unrolled for autograd and variant-op lowering.
    /// Drives the production predicate over every member of that set, and partitions the set so a
    /// new entry has to choose exactly one side.</summary>
    [Fact]
    public void TestTheUnrollGateBlocksTheStageFParamOpsAndLetsEveryOtherModuleStageOpThrough()
    {
        string[] blocked = [
            InternalOpCodes.MODEL_PARAM_REF, InternalOpCodes.MODEL_PARAM_ID_REF,
            InternalOpCodes.MODEL_PARAM_MODEL_REF, InternalOpCodes.MODULE_SET_HYPERPARAMS];
        string[] allowed = [
            InternalOpCodes.AUTO_GRAD, InternalOpCodes.MODEL_INVOKE, InternalOpCodes.FUNCTION_INVOKE,
            InternalOpCodes.MODEL_HYPERPARAM, InternalOpCodes.GET_MODEL_ID, InternalOpCodes.NEW_MODEL_LIKE,
            InternalOpCodes.CREATE_MODULE, InternalOpCodes.TENSOR_STRUCT_CREATE,
            InternalOpCodes.TENSOR_STRUCT_GETFIELD, InternalOpCodes.MODEL_TENSORSTRUCT_INPUT,
            InternalOpCodes.GENERIC_TYPE_INPUT];

        Assert.All(blocked, op => Assert.True(GateBlocks(op)));
        Assert.All(allowed, op => Assert.False(GateBlocks(op)));
        Assert.False(GateBlocks(OpCodes.ADD));

        Assert.Empty(blocked.Intersect(allowed));
        Assert.Equal(InternalOpCodes.ModuleStageOps.Count, blocked.Length + allowed.Length);
        Assert.Equal(InternalOpCodes.ModuleStageOps, [.. blocked, .. allowed]);
    }

    /// <summary>LoopAPI binds a scan input on the third of its four body-tracing passes, and the
    /// caller's local has by then been advanced by the two earlier passes into the outer graph, so
    /// scanning a carry before the body updates it stacks a loop-invariant outer value — x + 2,
    /// whatever the trip count. Rolled that is silent; unrolled the unroller finds no body-produced
    /// key and dies. The expected values are supplied because every engine executes the same wrong
    /// graph, so an engine comparison alone passes. Tracked as Shorokoo/Shorokoo#232.</summary>
    [Fact(Skip = "Shorokoo/Shorokoo#232: a scan input read before the body's update binds outside the loop body")]
    public void TestScanningACarryBeforeTheBodyUpdatesItStacksThePerIterationValues()
    {
        Assert.True(AutoTest.AdvancedTestGraph<ScanCarryBeforeUpdate>(
            hyperparamInputs: [],
            runtimeInputs: [Scalar32(10f), TensorData(DType.Int64, [], 3L)],
            expected: [10.0, 11.0, 12.0]));
        Assert.True(AutoTest.AdvancedTestGraph<ScanCarryBeforeUpdateConstTrip>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(10f)], expected: [10.0, 11.0, 12.0]));
    }
}
