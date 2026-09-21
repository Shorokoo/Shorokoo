using System.Collections.Immutable;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Tests;

/// <summary>
/// Autograd ENGINE path-checking semantics in <c>FastProcessAutoGrad</c>, the AD003
/// attribute-envelope guards on the gradient implementations, and the operator lowering the
/// engine runs over the graph first so an op with no <c>[AutoDiff]</c> rule reaches the reverse
/// walk as the primitives it decomposes into. Each module scenario drives a
/// module from <c>Modules/AutoGradEngineModules.cs</c> through
/// <see cref="AutoTest.AdvancedTestGraph{TModule}"/>; the AD003
/// <c>AutoDiffNotSupportedException</c> surfaces from the AUTO_GRAD lowering during
/// concretization, i.e. out of the <c>AdvancedTestGraph</c> call itself.
/// </summary>
[Trait("Domain", "AutoDiff")]
[Trait("Purpose", "Coverage")]
public class AutoGradEngineTests
{
    // An unsupported op on the loss→param path must throw AD003 at lowering — never silently cut
    // the chain and hand the parameter a zeros gradient.
    [Fact]
    public void TestAutoGradEngineThrowsAD003()
    {
        void AssertAD003<TModule>(long length, string messageFragment)
        {
            var ex = Assert.Throws<AutoDiffNotSupportedException>(() =>
                AutoTest.AdvancedTestGraph<TModule>(
                    hyperparamInputs: [],
                    runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [length])]));
            Assert.Equal(ErrorCodes.AD003, ex.ErrorCode);
            Assert.Contains(messageFragment, ex.Message);
        }

        AssertAD003<AutoGradEngineLoopOnParamPathCheck>(4L, "dynamic loops");
        AssertAD003<AutoGradEnginePadReflectThrowCheck>(4L, "constant");
        AssertAD003<AutoGradEngineScatterMulThrowCheck>(4L, "reduction");
    }

    // An unregistered op with NO parameter behind it is a legitimate gradient leaf (chain cut
    // there); Slice with steps != 1 scatters onto the exact flat offsets the forward selected; an
    // op lowered into its decomposition inside an IfElse arm is differentiated in that arm.
    [Fact]
    public void TestAutoGradEngineDifferentiates()
    {
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineRandomLeafCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [4L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineSliceStepsCheck>(
            hyperparamInputs: [], runtimeInputs: [TensorDataWithSmallVals(DType.Float32, [6L])]));
        Assert.True(AutoTest.AdvancedTestGraph<AutoGradEngineSoftsignInIfArmCheck>(
            hyperparamInputs: [], runtimeInputs: [
                TensorData(DType.Float32, [5L], 0f, 1f, -1f, 3f, -7f),
                TensorData(DType.Float32, [], 1f),
                TensorData(DType.Float32, [5L], 1f, 0.25f, 0.25f, 0.0625f, 0.015625f)]));
    }

    // An op with no [AutoDiff] rule but a registered lowering is rewritten into that
    // decomposition before the reverse walk, so every node the walk meets has a rule — the
    // primitives Softsign decomposes into all do, and the Constant is a leaf the walk cuts. A
    // graph with no AUTO_GRAD node never reaches the pass, so its Softsign survives to export.
    [Fact]
    public void TestAutoGradLowersAnOperatorThatHasNoGradientRuleOfItsOwn()
    {
        var rules = AutoDiffs.GetGradientOps();
        string[] primitives = [ABS, ADD, DIV, CAST_LIKE];
        Assert.False(rules.ContainsKey(SOFTSIGN));
        Assert.Contains(SOFTSIGN, FastProcessAutoGradProcessor.LoweredOpCodes);
        Assert.All(primitives, op => Assert.True(rules.ContainsKey(op)));

        var x = TensorData(DType.Float32, [5L], 0f, 1f, -1f, 3f, -7f);
        var expected = TensorData(DType.Float32, [5L], 1f, 0.25f, 0.25f, 0.0625f, 0.015625f);
        var training = AutoGradSoftsignLoweredGradientCheck.ComputationGraph.ToInternal();
        var trained = training.ToConcreteArchitecture(
            training.FromOrderedInputs([x, expected])).ToConcreteModel();
        Assert.DoesNotContain(trained.Nodes, n => n.OpCode == SOFTSIGN);
        Assert.DoesNotContain(trained.Nodes, n => n.OpCode == InternalOpCodes.AUTO_GRAD);

        var inference = QeeSoftsignLowered.ComputationGraph.ToInternal();
        var exported = inference.ToConcreteArchitecture(inference.FromOrderedInputs([x])).ToConcreteModel();
        FastProcessAutoGradProcessor.Process(exported);
        Assert.Equal(1, exported.Nodes.Count(n => n.OpCode == SOFTSIGN));
    }

    // Only what the domain's list names is lowered. A lowering is the fallback for an operator
    // the pass cannot differentiate, so an operator kept off the list — one whose hand-written
    // rule states a form the decomposition cannot — reaches the reverse walk as itself.
    [Fact]
    public void TestOnlyAnOperatorTheAutodiffListNamesIsLowered()
    {
        var x = TensorData(DType.Float32, [5L], 0f, 1f, -1f, 3f, -7f);
        var g = QeeSoftsignLowered.ComputationGraph.ToInternal();
        var concrete = g.ToConcreteArchitecture(g.FromOrderedInputs([x])).ToConcreteModel();

        var offTheList = concrete.Clone();
        FastLowerRegisteredOps.Process(offTheList, ImmutableHashSet<string>.Empty);
        var onIt = concrete.Clone();
        FastLowerRegisteredOps.Process(onIt, FastProcessAutoGradProcessor.LoweredOpCodes);

        Assert.Equal(1, offTheList.Nodes.Count(n => n.OpCode == SOFTSIGN));
        Assert.DoesNotContain(onIt.Nodes, n => n.OpCode == SOFTSIGN);
        Assert.Equal(1, onIt.Nodes.Count(n => n.OpCode == DIV));
    }
}
