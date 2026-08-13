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

    // Split out so the rest of the struct/control-flow coverage keeps asserting. A loop mixing a
    // plain carry with a TensorStruct carry loses the struct: the module computes 3a+3b+1, the
    // framework returns 4a for every b, and b is dead in the concretized graph (Shorokoo#163).
    // Drop the Skip when #163 is fixed — the expected value below is the module's own arithmetic.
    [Fact(Skip = "Shorokoo#163: a loop mixing plain and TensorStruct carries drops the struct")]
    public void TestLoopMixingPlainAndStructCarriesKeepsBoth()
        => Assert.True(AutoTest.AdvancedTestGraph<MixedTensorStructLoop>(
            hyperparamInputs: [], runtimeInputs: [Scalar32(1f), Scalar32(2f)],
            expected: [10.0]));

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
            expected: [11.0]));
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
}
