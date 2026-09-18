using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The value of an operator that has no <see cref="QuickOp"/> of its own, computed from its
/// <see cref="OpLowering"/>: build the decomposition over stand-ins carrying the real inputs'
/// dtypes and ranks, <see cref="OpLowering.Trace"/> the nodes it built, then run that trace front
/// to back, one <see cref="QuickOp"/> per node.
///
/// <para>A lowering's intermediates are its own locals and end with it — nothing built here
/// reaches the engine's tensor store, which keys tensors by the graph's own
/// <c>FastTensorKey</c>s, has none for an intermediate, and never evicts.</para>
///
/// <para>A built op code with no <see cref="QuickOp"/> is a dead end and throws: computing a node
/// this way is a single step down and never consults <see cref="OpLoweringRegistry"/> again, so
/// the only operators available are the ones the engine implements directly.</para>
/// </summary>
internal static class LoweredValue
{
    /// <summary>
    /// The lowered operator's outputs, one per value its decomposition returned, given
    /// <paramref name="inputs"/> (a null entry is an omitted optional input) and
    /// <paramref name="attributes"/>, the bag of the operator being lowered. Throws when any step
    /// of the decomposition cannot be carried out.
    /// </summary>
    public static IRuntimeTensor[] Compute(
        OpLowering lowering,
        IRuntimeTensor?[] inputs,
        OnnxCSharpAttributes attributes,
        int maxDataElements)
    {
        var standIns = new Variable?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is { } input)
                standIns[i] = InternalOp.RuntimeInput(
                    input.DType, (input as RuntimeTensor)?.Shape?.Dims.Length);

        var loweredOutputs = lowering.Build(standIns, attributes);

        // Keyed by reference identity, which is all a Variable has: it overrides neither Equals
        // nor GetHashCode.
        var values = new Dictionary<Variable, IRuntimeTensor>();
        for (int i = 0; i < standIns.Length; i++)
            if (standIns[i] is { } standIn) values[standIn] = inputs[i]!;

        foreach (var step in lowering.Trace(loweredOutputs, standIns))
        {
            var op = OpRegistry.Get(step.OpCode)
                ?? throw new InvalidOperationException(
                    $"Operator lowering for '{lowering.OpCode}' built '{step.OpCode}', which the "
                    + "QuickExecutionEngine has no operator for.");

            var stepInputs = new IRuntimeTensor?[step.Inputs.Length];
            for (int i = 0; i < stepInputs.Length; i++)
                if (step.Inputs[i] is { } slot) values.TryGetValue(slot, out stepInputs[i]);

            var stepOutputs = step.Outputs;
            var results = op.Invoke(stepInputs, step.Attributes, maxDataElements);
            if (results.Length != stepOutputs.Length)
                throw new InvalidOperationException(
                    $"Operator lowering for '{lowering.OpCode}' built '{step.OpCode}' for "
                    + $"{stepOutputs.Length} output(s) but got {results.Length}.");

            for (int i = 0; i < stepOutputs.Length; i++)
                if (stepOutputs[i] is { } output) values[output] = results[i];
        }

        var outputs = new IRuntimeTensor[loweredOutputs.Length];
        for (int i = 0; i < loweredOutputs.Length; i++)
            outputs[i] = loweredOutputs[i] is { } value && values.TryGetValue(value, out var result)
                ? result
                : throw new InvalidOperationException(
                    $"Operator lowering for '{lowering.OpCode}' left output {i} unfilled.");
        return outputs;
    }
}
