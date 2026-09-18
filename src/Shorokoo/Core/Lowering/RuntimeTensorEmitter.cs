using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The QuickExecutionEngine's reading of <see cref="IOpEmitter{T}"/>: emitting an operator runs
/// it, there and then, through the <see cref="QuickOp"/> registered for that op code, and the
/// value it hands back is the computed <see cref="IRuntimeTensor"/>.
///
/// <para>Nothing emitted here reaches the engine's tensor store. A lowering's intermediates are
/// its own locals and end with it — the store keys tensors by the graph's own
/// <c>FastTensorKey</c>s, which an intermediate has none of, and the engine never evicts, so
/// anything written there would surface in every caller's returned dictionary.</para>
///
/// <para>An emitted op code with no <see cref="QuickOp"/> is a dead end and throws: the engine
/// computing a node this way has only operators it implements directly to work with, so
/// emitting is a single step and never itself consults <see cref="OpLoweringRegistry"/>. That
/// also settles recursion — a lowering that emitted its own op code would find no operator for
/// it and fail rather than loop.</para>
/// </summary>
internal sealed class RuntimeTensorEmitter : IOpEmitter<IRuntimeTensor>
{
    private readonly int maxDataElements;

    public RuntimeTensorEmitter(int maxDataElements) => this.maxDataElements = maxDataElements;

    public IRuntimeTensor[] Emit(
        string opCode, IRuntimeTensor?[] inputs, (string Name, object? Value)[] attrs, int outputCount)
    {
        var op = OpRegistry.Get(opCode)
            ?? throw new InvalidOperationException(
                $"Operator lowering emitted '{opCode}', which the QuickExecutionEngine has no operator for.");

        if (!Definitions.NodeDefinitions.TryGetValue(opCode, out var definition))
            throw new InvalidOperationException(
                $"Operator lowering emitted '{opCode}', which has no node definition.");

        var vals = new Dictionary<string, object?>(attrs.Length, StringComparer.Ordinal);
        foreach (var (name, value) in attrs) vals[name] = value;

        var results = op.Invoke(
            inputs, OnnxCSharpAttributes.FromCSharpVals(vals, definition.AttributeDefs), maxDataElements);

        return results.Length == outputCount
            ? results
            : throw new InvalidOperationException(
                $"Operator lowering emitted '{opCode}' for {outputCount} output(s) but got {results.Length}.");
    }
}
