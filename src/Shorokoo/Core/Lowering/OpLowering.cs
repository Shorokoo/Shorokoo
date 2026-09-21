using System.Reflection;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// Marks a static method in <see cref="OpLowerings"/> as the lowering of <see cref="OpName"/>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
internal sealed class OpLoweringAttribute : Attribute
{
    public string OpName { get; }

    public OpLoweringAttribute(string opName) => this.OpName = opName;
}

/// <summary>
/// How one operator is computed out of simpler ones — written once in <see cref="OpLowerings"/>,
/// run by every engine that lacks a direct implementation of it.
///
/// <para><b>Why.</b> An operator that is merely a composition of simpler operators used to be
/// written up to three times over: as the authoring-layer decomposition that keeps the exported
/// ONNX at the opset the framework emits, as a QuickExecutionEngine kernel, and as a gradient
/// rule. Nothing was shared, so the three could drift. A registered lowering is that
/// decomposition stated once, in plain Shorokoo code.</para>
///
/// <para><b>Operator lowering is not graph lowering.</b> "Lowering" elsewhere in this codebase
/// means the graph concretization pipeline — <c>ToConcreteArchitecture</c> and the
/// <c>FastLower*</c> passes, which rewrite the graph that is then exported (see
/// <c>Documentation/debugging.md</c>). Operator lowering is carried out by one of those passes,
/// <c>FastLowerRegisteredOps</c>, but what it costs differs by who asks for it. The
/// QuickExecutionEngine runs it over a clone it owns and throws away, so the graph handed to it
/// keeps its <c>Softsign</c> node. The autodiff pass runs it over the training graph it is
/// expanding, since a decomposition it cannot see is one it cannot differentiate — and a graph
/// with no <c>AUTO_GRAD</c> node never reaches that pass, so an inference model still exports a
/// <c>Softsign</c> as a <c>Softsign</c>.</para>
/// </summary>
/// <param name="OpCode">The op code this lowering expresses (e.g. "Softsign").</param>
/// <param name="Method">The <see cref="OpLoweringAttribute"/>-marked method that builds it.</param>
internal sealed record OpLowering(string OpCode, MethodInfo Method)
{
    /// <summary>
    /// Builds the decomposition over <paramref name="inputs"/> (a null entry is an omitted
    /// optional input) and <paramref name="attributes"/>, the bag of the operator BEING lowered,
    /// and returns one <see cref="Variable"/> per declared output.
    ///
    /// <para>Building is inert: <see cref="NodeBuilder"/> constructs nodes and hands back their
    /// outputs without registering anything anywhere, so the decomposition exists only as the
    /// values returned here and whatever is reachable from them — which is what the caller reads
    /// the nodes back off.</para>
    /// </summary>
    public Variable?[] Build(Variable?[] inputs, OnnxCSharpAttributes attributes)
        => AutoDiffs.CallRuleWithoutOutputGrads(this.Method, inputs, attributes);
}
