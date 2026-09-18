using System.Collections.Concurrent;
using System.Text;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes;
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
///
/// <para><b>The plan is cached, the values are not.</b> Building and tracing costs milliseconds —
/// <see cref="NodeBuilder"/> captures a file-and-line stack trace per node, and the decomposition
/// is reached by reflection — while running the trace costs microseconds, so an operator appearing
/// hundreds of times in a model would otherwise pay seconds of pure overhead. What is cached is
/// the inert part: the stand-ins, the output values and the node list, none of which a run
/// mutates. Each call still maps the stand-ins onto its own runtime tensors, in a table it owns,
/// and looks each node's <see cref="QuickOp"/> up afresh so a thread-scoped
/// <c>OpRegistry.Override</c> still takes effect.</para>
/// </summary>
internal static class LoweredValue
{
    // Field and group separators: control characters, so no op code, attribute name, dtype name or
    // attribute value can be mistaken for one.
    private const char Sep = '\u0001';
    private const char Group = '\u0002';

    private static readonly ConcurrentDictionary<string, LoweredPlan> Plans = new(StringComparer.Ordinal);

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
        var plan = GetPlan(lowering, inputs, attributes);

        // Keyed by reference identity, which is all a Variable has: it overrides neither Equals
        // nor GetHashCode.
        var values = new Dictionary<Variable, IRuntimeTensor>();
        for (int i = 0; i < plan.StandIns.Length; i++)
            if (plan.StandIns[i] is { } standIn) values[standIn] = inputs[i]!;

        foreach (var step in plan.Trace)
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

        var outputs = new IRuntimeTensor[plan.Outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
            outputs[i] = plan.Outputs[i] is { } value && values.TryGetValue(value, out var result)
                ? result
                : throw new InvalidOperationException(
                    $"Operator lowering for '{lowering.OpCode}' left output {i} unfilled.");
        return outputs;
    }

    /// <summary>
    /// The plan for this shape of call, built on first sight and kept. Only a plan that built and
    /// traced cleanly is kept: a decomposition refused as it is read back — one that builds its
    /// own op code, say — must be refused on every call and not just the first, so nothing is
    /// stored for a build that threw. A shape whose attributes cannot be keyed is built fresh each
    /// time rather than shared with a call that differs in one of them.
    /// </summary>
    private static LoweredPlan GetPlan(
        OpLowering lowering, IRuntimeTensor?[] inputs, OnnxCSharpAttributes attributes)
    {
        var key = TryBuildKey(lowering, inputs, attributes);
        if (key is not null && Plans.TryGetValue(key, out var cached)) return cached;

        var standIns = new Variable?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is { } input)
                standIns[i] = InternalOp.RuntimeInput(
                    input.DType, (input as RuntimeTensor)?.Shape?.Dims.Length);

        var loweredOutputs = lowering.Build(standIns, attributes);
        var plan = new LoweredPlan(standIns, loweredOutputs, [.. lowering.Trace(loweredOutputs, standIns)]);

        // Two threads racing on one key both build and one wins; a plan is published whole, so
        // neither can observe a torn one and either serves equally.
        if (key is not null) Plans[key] = plan;
        return plan;
    }

    /// <summary>
    /// What a plan may be reused for: the lowering itself, the dtype and rank of each input (and
    /// which inputs are absent), and the attribute VALUES. The attributes belong in the key
    /// because a lowering is ordinary C# and may branch on them, so the same operator with
    /// different attributes can legitimately decompose into different nodes. Null when some
    /// attribute has no stable rendering — a tensor or a subgraph — since a key that ignored it
    /// would serve one call's plan to another that differs only there.
    /// </summary>
    private static string? TryBuildKey(
        OpLowering lowering, IRuntimeTensor?[] inputs, OnnxCSharpAttributes attributes)
    {
        var key = new StringBuilder(lowering.OpCode)
            .Append(Sep).Append(lowering.Method.MethodHandle.Value);

        foreach (var input in inputs)
        {
            key.Append(Group);
            if (input is null) { key.Append('~'); continue; }
            key.Append(input.DType).Append(Sep).Append((input as RuntimeTensor)?.Shape?.Dims.Length ?? -1);
        }

        foreach (var (name, value) in attributes.GetAttributeVals().OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            key.Append(Group).Append(name).Append(Sep);
            if (!TryAppendValue(key, value)) return null;
        }
        return key.ToString();
    }

    private static bool TryAppendValue(StringBuilder key, object? value)
    {
        switch (value)
        {
            case null: key.Append('~'); return true;
            case bool b: key.Append(b ? 'T' : 'F'); return true;
            case long l: key.Append(l); return true;
            case int i: key.Append(i); return true;
            // By bits, so the rendering is exact and carries no culture or rounding of its own.
            case float f: key.Append(BitConverter.SingleToInt32Bits(f)); return true;
            case double d: key.Append(BitConverter.DoubleToInt64Bits(d)); return true;
            // Length-prefixed: a string is the one value that could otherwise hold a separator.
            case string s: key.Append(s.Length).Append('"').Append(s); return true;
            case DType t: key.Append(t); return true;
            case Enum e: key.Append(e.GetType().FullName).Append('.').Append(e); return true;
            case Array a:
                key.Append('[');
                foreach (var item in a)
                {
                    if (!TryAppendValue(key, item)) return false;
                    key.Append(Sep);
                }
                key.Append(']');
                return true;
            default: return false;
        }
    }

    /// <summary>
    /// Everything about one lowered call that does not depend on the values flowing through it:
    /// the stand-in per input slot, the decomposition's outputs, and the nodes to run in order.
    /// All three are inert once built — a run reads them and writes only its own value table — so
    /// one plan serves every thread.
    /// </summary>
    private sealed record LoweredPlan(Variable?[] StandIns, Variable?[] Outputs, Node[] Trace);
}
