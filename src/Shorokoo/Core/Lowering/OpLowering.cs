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
/// <c>Documentation/debugging.md</c>). Operator lowering rewrites nothing: it is an
/// engine-internal fallback consulted while an engine is already running, and it leaves the
/// graph exactly as it found it. A <c>Softsign</c> node still exports as a <c>Softsign</c>
/// node.</para>
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
    /// values returned here and whatever is reachable from them — which is what
    /// <see cref="Trace"/> reads back.</para>
    /// </summary>
    public Variable?[] Build(Variable?[] inputs, OnnxCSharpAttributes attributes)
        => AutoDiffs.CallRuleWithoutOutputGrads(this.Method, inputs, attributes);

    /// <summary>
    /// The nodes <paramref name="outputs"/> were built from, in topological order — every node
    /// after the ones producing its inputs, so an engine can evaluate the list front to back.
    /// Each node appears exactly once however many places use its outputs. A lowering says
    /// nothing about what it is doing, so this is how both engines learn what it built.
    ///
    /// <para>The walk stops at <paramref name="inputs"/>, the values the lowering was handed, so
    /// nothing belonging to the caller's own graph is reported — including where a lowering hands
    /// one of them straight back, which builds nothing and traces to nothing. A value built from
    /// nothing, such as the literal a constant comes from, has no inputs and ends the walk by
    /// itself. Identity is by reference throughout: neither <see cref="Variable"/> nor
    /// <see cref="Node"/> overrides equality.</para>
    ///
    /// <para><b>Recursion.</b> A lowering that builds its own op code is refused here, which is
    /// what keeps a lowering from being defined in terms of itself. Building it could not have
    /// looped — a node is built, not run — so the refusal is in reading the result back, and it
    /// covers both engines because both reach their nodes through this one walk.</para>
    /// </summary>
    public List<Node> Trace(Variable?[] outputs, Variable?[] inputs)
    {
        var boundary = new HashSet<Variable>();
        foreach (var input in inputs)
            if (input is not null) boundary.Add(input);

        var ordered = new List<Node>();
        var visited = new HashSet<Node>();
        var pending = new Stack<(Node Node, bool Expanded)>();
        foreach (var output in outputs)
            if (output is not null && !boundary.Contains(output)) pending.Push((output.OwningNode, false));

        while (pending.Count > 0)
        {
            var (node, expanded) = pending.Pop();
            if (expanded) { ordered.Add(node); continue; }
            if (!visited.Add(node)) continue;

            if (string.Equals(node.OpCode, this.OpCode, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Operator lowering for '{this.OpCode}' built '{this.OpCode}', "
                    + "which it cannot be built from.");

            pending.Push((node, true));
            foreach (var input in node.Inputs)
                if (input is not null && !boundary.Contains(input)) pending.Push((input.OwningNode, false));
        }

        return ordered;
    }
}
