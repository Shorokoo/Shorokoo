using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The gradient of an operator that has no <c>[AutoDiff]</c> rule of its own, read off its
/// <see cref="OpLowering"/>: build the decomposition as real nodes through
/// <see cref="VariableEmitter"/> while recording each primitive it emits, then run one reverse
/// pass over that recording, differentiating each primitive with the rule the framework already
/// has for it.
///
/// <para>The result is a vector-Jacobian product and nothing more — a cotangent per forward
/// input, from a cotangent per forward output. No scalar loss is involved and no nested autograd
/// runs: the caller's incoming output gradients seed the walk, so what comes back is
/// indistinguishable from what a hand-written rule would have returned.</para>
///
/// <para><b>Recursion.</b> The reverse walk differentiates a recorded primitive through that
/// primitive's own <c>[AutoDiff]</c> rule and never through a second lowering, so no cycle
/// between lowerings can form. The one shape that could still spin — a lowering emitting its own
/// op code — is refused when it is emitted, before any node is built.</para>
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
    /// reverse walk looks the emitted primitives up in.
    /// </summary>
    public static Variable?[] Compute(
        OpLowering lowering,
        Variable?[] inputs,
        Variable?[] outputGrads,
        OnnxCSharpAttributes attributes,
        IReadOnlyDictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>> gradientOps)
    {
        var recorder = new RecordingEmitter(lowering.OpCode);
        var loweredOutputs = lowering.Lower(recorder, inputs, attributes);

        // One cotangent seed per output. A decomposition that produced a different number of them
        // than the operator declares would have the surplus seeds dropped and the gradient come
        // out short, which is the one thing this engine never does quietly.
        if (loweredOutputs.Length != outputGrads.Length)
            throw new InvalidOperationException(
                $"Operator lowering for '{lowering.OpCode}' produced {loweredOutputs.Length} output(s) "
                + $"for {outputGrads.Length} output gradient(s).");

        // Which emitted values a forward input reaches. A step none of them reaches computes
        // nothing the caller asked for — the Constant a lowering builds its own literals from is
        // the standing example — so the walk leaves it alone rather than emitting dead gradient
        // nodes for it.
        var live = new HashSet<Variable>();
        foreach (var input in inputs)
            if (input is not null) live.Add(input);
        foreach (var step in recorder.Tape)
            if (IsLive(step.Inputs, live))
                foreach (var output in step.Outputs) live.Add(output);

        var cotangents = new Dictionary<Variable, Variable>();
        for (int i = 0; i < loweredOutputs.Length; i++)
            if (outputGrads[i] is { } seed) Accumulate(cotangents, loweredOutputs[i], seed);

        for (int s = recorder.Tape.Count - 1; s >= 0; s--)
        {
            var step = recorder.Tape[s];
            if (!IsLive(step.Inputs, live)) continue;

            var stepOutputGrads = new Variable?[step.Outputs.Length];
            bool anyOutputGrad = false;
            for (int i = 0; i < step.Outputs.Length; i++)
            {
                if (!cotangents.TryGetValue(step.Outputs[i], out var g)) continue;
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
            var stepInputs = gradientOpsUsingOutputs.Contains(step.OpCode)
                ? [.. step.Inputs, .. step.Outputs]
                : step.Inputs;

            var stepInputGrads = gradientOp(stepInputs, stepOutputGrads, step.Attributes);
            for (int i = 0; i < step.Inputs.Length && i < stepInputGrads.Length; i++)
            {
                if (step.Inputs[i] is not { } slot || !live.Contains(slot)) continue;
                if (stepInputGrads[i] is { } g) Accumulate(cotangents, slot, g);
            }
        }

        var result = new Variable?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is { } input && cotangents.TryGetValue(input, out var g)) result[i] = g;
        return result;
    }

    private static bool IsLive(Variable?[] values, HashSet<Variable> live)
    {
        foreach (var value in values)
            if (value is not null && live.Contains(value)) return true;
        return false;
    }

    private static void Accumulate(Dictionary<Variable, Variable> cotangents, Variable value, Variable grad)
        => cotangents[value] = cotangents.TryGetValue(value, out var existing)
            ? AutoDiffEngine.AccumulateGradients(existing, grad)
            : grad;

    /// <summary>One primitive the lowering emitted, with the values it was given and produced.</summary>
    private readonly record struct Step(
        string OpCode, Variable?[] Inputs, Variable[] Outputs, OnnxCSharpAttributes Attributes);

    /// <summary>
    /// <see cref="VariableEmitter"/> with a tape: each primitive becomes a real node exactly as it
    /// otherwise would, and is written down in emission order so the reverse walk can read it back.
    /// The attributes come off the built node, so they are the ones
    /// <see cref="NodeBuilder"/> resolved against that operator's own definition — defaults filled
    /// in — rather than the raw pairs the lowering wrote.
    /// </summary>
    private sealed class RecordingEmitter : IOpEmitter<Variable>
    {
        private readonly VariableEmitter inner = new();
        private readonly string loweredOpCode;

        public RecordingEmitter(string loweredOpCode) => this.loweredOpCode = loweredOpCode;

        public List<Step> Tape { get; } = [];

        public Variable[] Emit(
            string opCode, Variable?[] inputs, (string Name, object? Value)[] attrs, int outputCount)
        {
            if (string.Equals(opCode, this.loweredOpCode, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Operator lowering for '{opCode}' emitted '{opCode}', which it cannot be built from.");

            var outputs = this.inner.Emit(opCode, inputs, attrs, outputCount);
            if (outputs.Length > 0)
                this.Tape.Add(new Step(opCode, inputs, outputs, outputs[0].OwningNode.Attributes));
            return outputs;
        }
    }
}
