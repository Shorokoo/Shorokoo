using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The graph side's reading of <see cref="IOpEmitter{T}"/>: emitting an operator builds a real
/// node for it and hands back its output <see cref="Variable"/>s. Nothing is computed — the
/// lowering's shape is recorded in the graph, and whichever engine runs that graph computes it.
///
/// <para>This is what lets a gradient rule be written as an <see cref="OpLowering"/> rather than
/// by hand: the same decomposition that the QuickExecutionEngine evaluates
/// (<see cref="RuntimeTensorEmitter"/>) becomes, here, the nodes a differentiated graph is built
/// from.</para>
///
/// <para>Each emitted operator's attributes are resolved against ITS OWN node definition, by
/// <see cref="NodeBuilder"/>; the bag belonging to the operator being lowered would not fit,
/// since a node definition accepts only the attributes it declares.</para>
/// </summary>
internal sealed class VariableEmitter : IOpEmitter<Variable>
{
    public Variable[] Emit(
        string opCode, Variable?[] inputs, (string Name, object? Value)[] attrs, int outputCount)
    {
        var outputs = NodeBuilder.BuildNodeMultiOut(opCode, inputs, attrs, outputNames: new string?[outputCount]);

        return outputs.Length == outputCount
            ? outputs
            : throw new InvalidOperationException(
                $"Operator lowering emitted '{opCode}' for {outputCount} output(s) but got {outputs.Length}.");
    }
}
