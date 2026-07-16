using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests that drive <see cref="AutoTest.AdvancedTestGraph{TModule}"/>
/// against modules whose graph shape targets specific uncovered branches in
/// <c>Shorokoo.Core.Nodes.Processors.Fast.FastProcessors</c>. The modules surface
/// bare TensorStruct slots in LOOP / IF control flow, exercising
/// <c>FastUnpackTensorStructs.ExpandLoopOpenStructLoopVars</c>,
/// <c>ExpandLoopCloseStructLoopVars</c>, and <c>ExpandIfCloseStructBranches</c> —
/// expansion paths that the existing sequence-of-struct B2b tests don't reach.
/// </summary>
[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class FastProcessorsCoverageTests
{
    [Fact]
    public void TestBareTensorStructInControlFlowCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructLoopCarry>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f), TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructIfElseReturn>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Bool, [], true), TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 5.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<MixedTensorStructLoop>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f), TensorData(DType.Float32, [], 2.0f)]));
    }

    /// <summary>
    /// Drives the sequence-of-struct branches of
    /// <c>FastUnpackTensorStructs.ExpandLoopOpenStructLoopVars</c>,
    /// <c>ExpandLoopCloseStructLoopVars</c>, the plain-tensor passthrough
    /// <c>else</c> branch of <c>ExpandIfCloseStructBranches</c> (hit only when
    /// an IF returns a mix of struct + plain slots), and the scan-output
    /// expansion block at the tail of <c>ExpandLoopCloseStructLoopVars</c> (hit
    /// only when a struct-loop-var LOOP also has a <c>ctx.Scan</c> output). The
    /// pure <c>Sequence&lt;TensorStruct&gt;</c>-from-IF case lives in
    /// <see cref="TestSequenceOfStructIfElseReturnCoverage"/>.
    /// </summary>
    [Fact]
    public void TestSequenceOfTensorStructInControlFlowCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SequenceOfStructLoopCarry>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f), TensorData(DType.Float32, [], 2.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<IfElseMixedStructAndPlainSlots>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Bool, [], true), TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 5.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<TensorStructLoopCarryWithScanOutput>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 1.0f), TensorData(DType.Float32, [], 2.0f)]));
    }

    /// <summary>
    /// Coverage for an IfElse returning a <c>Sequence&lt;TensorStruct&gt;</c> from each
    /// branch. Drives the sequence-of-struct branch of
    /// <c>FastUnpackTensorStructs.ExpandIfCloseStructBranches</c> and exercises the
    /// full ONNX save/reload + CS roundtrip + QEE on a graph whose top-level node
    /// order includes a <c>CONSTANT</c> (the <c>Scalar(0L)</c> index for the trailing
    /// <c>SEQUENCE_AT</c>) positionally after an <c>IF_CLOSE</c> — covering the
    /// <c>FastOnnxModelReader</c> topological re-tour
    /// (<c>BuildTempNodeGraph</c> + <c>NodeGuide.GiveTour</c>) constant-placement
    /// path and the <c>IfCloseOp</c> sequence-typed branch handling that the
    /// base-class IRuntimeTensor→RuntimeTensor cast would otherwise strip to
    /// <c>null</c>.
    /// </summary>
    [Fact]
    public void TestSequenceOfStructIfElseReturnCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SequenceOfStructIfElseReturn>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Bool, [], true), TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 5.0f)]));
    }


    /// <summary>
    /// Drives the <c>MODULE_SET_HYPERPARAMS</c> arm of
    /// <c>FastInlineModulesAndFunctions.FastReparentToCallSite</c>
    /// (~L703-727 of <c>FastProcessors.cs</c>) — the mirror of the
    /// <c>MODEL_PARAM_REF</c> arm at L676-702 hit by
    /// <c>CallsSimplestModule</c>. Needs three call levels: the outer
    /// module inlines the middle module, whose body contains the inner
    /// module's <c>MODULE_SET_HYPERPARAMS</c> node that the reparenter
    /// rewrites with the call-site's iteration indices and prepended
    /// model-id.
    /// </summary>
    [Fact]
    public void TestModuleOnHyperparamModuleCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<CallsHypersLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
        Assert.True(AutoTest.AdvancedTestGraph<CallsCallsHypersLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [5L])]));
    }

    /// <summary>
    /// Drives the TensorStruct-typed branches of <c>SEQUENCE_CONSTRUCT</c>,
    /// <c>SEQUENCE_AT</c>, <c>SEQUENCE_EMPTY</c>, <c>SEQUENCE_INSERT</c>,
    /// <c>SEQUENCE_ERASE</c>, and <c>SEQUENCE_LENGTH</c> handlers in
    /// <c>FastUnpackTensorStructs.Process</c> — the unified topological walk
    /// that rewrites each sequence op into one parallel op per struct field
    /// (or, for LENGTH, collapses to a single read from field[0]).
    /// </summary>
    [Fact]
    public void TestSequenceOfTensorStructCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<SequenceOpsOnStructs>(
            hyperparamInputs: [],
            runtimeInputs: [
                TensorData(DType.Float32, [], 1.0f), TensorData(DType.Float32, [], 2.0f),
                TensorData(DType.Float32, [], 3.0f), TensorData(DType.Float32, [], 4.0f)]));
    }

    /// <summary>
    /// Drives the four "non-happy-path" shapes in
    /// <c>FastFoldConstantIterationLoops.UnrollOne</c>: zero-iteration early return,
    /// seed-<c>true</c> CONSTANT for the AND-chain when OPEN has no initial cond
    /// (paired with per-loop-var <c>WHERE</c> + per-iter <c>AND</c> gating from a
    /// dynamic body break), per-iter <c>UNSQUEEZE</c> + final <c>CONCAT</c>
    /// for a non-empty scan output, and the scope-pair propagation arm that
    /// walks a nested <c>IF_CLOSE</c>'s paired <c>IF_OPEN</c> inputs back into
    /// the loop-dep set when the inner control flow's condition itself
    /// depends on the iteration index.
    /// </summary>
    [Fact]
    public void TestConstantLoopEdgeCasesCoverage()
    {
        Assert.True(AutoTest.AdvancedTestGraph<ZeroIterConstLoopLayer>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 42.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConstLoopWithScanOutput>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.0f)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConstLoopWithDynamicBreak>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.0f), TensorData(DType.Bool, [], true)]));
        Assert.True(AutoTest.AdvancedTestGraph<ConstLoopWithNestedIterDependentIf>(
            hyperparamInputs: [],
            runtimeInputs: [TensorData(DType.Float32, [], 0.0f)]));
    }

    /// <summary>
    /// Drives the <c>hasStructInput</c> pre-pass in <c>FastUnpackTensorStructs.Process</c>
    /// (~L1500-1584 of <c>FastProcessors.cs</c>): when the graph's input list contains a
    /// <c>MODEL_TENSORSTRUCT_INPUT</c> producer, that node is rewritten into one
    /// <c>MODEL_TENSOR_INPUT</c> per struct field and the graph's <c>Inputs</c> list is
    /// rebuilt to reference the per-field keys. <see cref="SimplePairSum"/> is the only
    /// existing [Module] whose signature takes a <c>GenericPairStruct</c> directly, so its
    /// <see cref="SimplePairSum.ComputationGraph"/> exercises this branch end-to-end. Runs
    /// the architecture pipeline only — execution requires a <c>TensorDataStruct</c>
    /// shape that <see cref="AutoTest.AdvancedTestGraph"/>'s flat <c>TensorData[]</c>
    /// API does not model.
    /// </summary>
    [Fact]
    public void TestTensorStructAsModuleInputCoverage()
    {
        var graph = SimplePairSum.ComputationGraph;
        Assert.Contains(graph.Nodes, n => n.OpCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT);

        var concreteArch = graph.ToConcreteArchitecture(new ModelParamList());

        Assert.DoesNotContain(concreteArch.Nodes, n => n.OpCode == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT);
        Assert.DoesNotContain(concreteArch.Nodes, n => n.OpCode == InternalOpCodes.TENSOR_STRUCT_CREATE);
        Assert.DoesNotContain(concreteArch.Nodes, n => n.OpCode == InternalOpCodes.TENSOR_STRUCT_GETFIELD);
    }
}
