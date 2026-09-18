using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The gradient of an operator that has no <c>[AutoDiff]</c> rule of its own, read off its
/// <see cref="OpLowering"/>: build the decomposition as real nodes,
/// <see cref="OpLowering.Trace"/> the nodes it built, then run one reverse pass over that trace,
/// differentiating each primitive with the rule the framework already has for it.
///
/// <para>The result is a vector-Jacobian product and nothing more — a cotangent per forward
/// input, from a cotangent per forward output. No scalar loss is involved and no nested autograd
/// runs: the caller's incoming output gradients seed the walk, so what comes back is
/// indistinguishable from what a hand-written rule would have returned.</para>
///
/// <para><b>Recursion.</b> The reverse walk differentiates a traced primitive through that
/// primitive's own <c>[AutoDiff]</c> rule and never through a second lowering, so no cycle
/// between lowerings can form. The one shape that could still spin — a lowering building its own
/// op code — is refused by the trace.</para>
///
/// <para>The caller passes one distinct value per input slot, as
/// <c>FastProcessAutoGradProcessor</c> does with its fresh stand-ins; a value shared between two
/// slots would have the sum of both slots' cotangents reported in each.</para>
/// </summary>
internal static class LoweredGradient
{
    private static readonly HashSet<string> gradientOpsUsingOutputs = AutoDiffs.GetGradientOpsUsingOutputs();

    /// <summary>
    /// Cotangents of <paramref name="inputs"/> — one per slot, null where no gradient reaches
    /// that slot — given <paramref name="outputGrads"/>, the cotangents of the lowered
    /// operator's outputs. <paramref name="attributes"/> is the bag of the operator being
    /// lowered, and <paramref name="gradientOps"/> the caller's gradient-rule table, which the
    /// reverse walk looks the traced primitives up in.
    /// </summary>
    public static Variable?[] Compute(
        OpLowering lowering,
        Variable?[] inputs,
        Variable?[] outputGrads,
        OnnxCSharpAttributes attributes,
        IReadOnlyDictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>> gradientOps)
    {
        var loweredOutputs = lowering.Build(inputs, attributes);

        // One cotangent seed per output. A decomposition that produced a different number of them
        // than the operator declares would have the surplus seeds dropped and the gradient come
        // out short, which is the one thing this engine never does quietly.
        if (loweredOutputs.Length != outputGrads.Length)
            throw new InvalidOperationException(
                $"Operator lowering for '{lowering.OpCode}' produced {loweredOutputs.Length} output(s) "
                + $"for {outputGrads.Length} output gradient(s).");

        var tape = lowering.Trace(loweredOutputs, inputs);

        // Which built values a forward input reaches. A step none of them reaches computes
        // nothing the caller asked for — the literal a lowering builds its own constants from is
        // the standing example — so the walk leaves it alone rather than emitting dead gradient
        // nodes for it.
        var live = new HashSet<Variable>();
        foreach (var input in inputs)
            if (input is not null) live.Add(input);
        foreach (var step in tape)
            if (IsLive(step.Inputs, live))
                foreach (var output in step.Outputs)
                    if (output is not null) live.Add(output);

        var cotangents = new Dictionary<Variable, Variable>();
        for (int i = 0; i < loweredOutputs.Length; i++)
            if (loweredOutputs[i] is { } output && outputGrads[i] is { } seed)
                Accumulate(cotangents, output, seed);

        for (int s = tape.Count - 1; s >= 0; s--)
        {
            var step = tape[s];
            var stepInputs = step.Inputs;
            if (!IsLive(stepInputs, live)) continue;

            var stepOutputs = step.Outputs;
            var stepOutputGrads = new Variable?[stepOutputs.Length];
            bool anyOutputGrad = false;
            for (int i = 0; i < stepOutputs.Length; i++)
            {
                if (stepOutputs[i] is not { } output) continue;
                if (!cotangents.TryGetValue(output, out var g)) continue;
                stepOutputGrads[i] = g;
                anyOutputGrad = true;
            }
            if (!anyOutputGrad) continue;

            if (!gradientOps.TryGetValue(step.OpCode, out var gradientOp))
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, lowering.OpCode,
                    $"the op has no registered gradient, and the '{step.OpCode}' its decomposition "
                    + "is built from has none either. Register an [AutoDiff] gradient for "
                    + $"'{step.OpCode}' or for the op itself.");

            // A rule flagged UsesOutputs reads the forward outputs after the inputs, the same
            // extension the engine makes when it calls such a rule on a graph node.
            Variable?[] ruleInputs = gradientOpsUsingOutputs.Contains(step.OpCode)
                ? [.. stepInputs, .. stepOutputs]
                : [.. stepInputs];

            var stepInputGrads = gradientOp(ruleInputs, stepOutputGrads, step.Attributes);
            for (int i = 0; i < stepInputs.Length && i < stepInputGrads.Length; i++)
            {
                if (stepInputs[i] is not { } slot || !live.Contains(slot)) continue;
                if (stepInputGrads[i] is { } g) Accumulate(cotangents, slot, g);
            }
        }

        var result = new Variable?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is { } input && cotangents.TryGetValue(input, out var g)) result[i] = g;
        return result;
    }

    private static bool IsLive(IReadOnlyList<Variable?> values, HashSet<Variable> live)
    {
        foreach (var value in values)
            if (value is not null && live.Contains(value)) return true;
        return false;
    }

    private static void Accumulate(Dictionary<Variable, Variable> cotangents, Variable value, Variable grad)
        => cotangents[value] = cotangents.TryGetValue(value, out var existing)
            ? AutoDiffEngine.AccumulateGradients(existing, grad)
            : grad;
}
