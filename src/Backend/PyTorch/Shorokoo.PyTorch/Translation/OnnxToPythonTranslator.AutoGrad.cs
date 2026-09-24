using Shorokoo.Core.Factory.IR;
using Shorokoo.PyTorch.Translation.Operators;

namespace Shorokoo.PyTorch.Translation;

/// <summary>
/// The one <c>ai.shorokoo.training::AutoGrad</c> node of a training step handed over in
/// <see cref="TrainingFormats.OnnxAutoGrad"/>, found and checked before anything is written:
/// inputs <c>[loss, wrt…]</c>, outputs one gradient per <c>wrt</c>, at the top level of the main
/// graph.
/// </summary>
internal sealed record AutoGradStep(int Index, NodeProto Node)
{
    private static readonly HashSet<int> FloatingPoint = [1, 10, 11, 16];

    public string Loss => Node.Inputs[0];

    public IEnumerable<string> Wrt => Node.Inputs.Skip(1);

    public static bool IsAutoGrad(NodeProto node)
        => node.Domain == TrainingFormats.AutoGradDomain && node.OpType == TrainingFormats.AutoGradOpType;

    /// <summary>The model's AutoGrad node, or null for a model with none.</summary>
    /// <exception cref="TorchUnsupportedModelException">The node is somewhere, or is something, the
    /// PyTorch backend does not differentiate: inside a function or a branch or loop body, one of
    /// several, of another version, with a non-floating-point tensor to differentiate with respect
    /// to, or with an operator the backend does not differentiate through between such a tensor and
    /// the loss.</exception>
    public static AutoGradStep? Find(
        ModelProto model, GraphProto graph, IReadOnlyDictionary<string, long> opsets, IEnumerable<FunctionProto> functions)
    {
        var functionList = functions.ToList();
        foreach (var function in functionList)
            if (function.Nodes.Any(ContainsAutoGrad))
                throw Refusal($"the function {function.Name} holds one, and it is taken only at the top level of the model's graph");
        foreach (var node in graph.Nodes)
            if (node.Attributes.Any(a => a.G is { } g && g.Nodes.Any(ContainsAutoGrad)))
                throw Refusal($"the {node.OpType} node '{node.Name}' holds one in its body, and it is taken only at the top level of the model's graph");

        var found = graph.Nodes.Select((node, index) => (node, index)).Where(n => IsAutoGrad(n.node)).ToList();
        if (found.Count == 0) return null;
        if (found.Count > 1)
            throw Refusal($"the graph holds {found.Count}, and a step takes one gradient");
        if (!opsets.TryGetValue(TrainingFormats.AutoGradDomain, out var version) || version != TrainingFormats.AutoGradDomainVersion)
            throw Refusal($"the model imports its domain at version {(opsets.ContainsKey(TrainingFormats.AutoGradDomain) ? version : 0)}, "
                + $"and the backend runs version {TrainingFormats.AutoGradDomainVersion}");

        var step = new AutoGradStep(found[0].index, found[0].node);
        var autoGrad = step.Node;
        if (autoGrad.Attributes.Count > 0)
            throw Refusal($"it carries the attribute '{autoGrad.Attributes[0].Name}', and it takes none");
        if (autoGrad.Inputs.Count == 0 || autoGrad.Inputs.Any(i => i.Length == 0))
            throw Refusal("it must read a loss and every tensor it differentiates with respect to");
        if (autoGrad.Outputs.Count != autoGrad.Inputs.Count - 1)
            throw Refusal($"it differentiates with respect to {autoGrad.Inputs.Count - 1} tensors and returns {autoGrad.Outputs.Count} gradients");
        foreach (var wrt in step.Wrt)
            if (ElementType(graph, wrt) is > 0 and var type && !FloatingPoint.Contains(type))
                throw Refusal($"it differentiates with respect to '{wrt}', a tensor of ONNX element type {type}, "
                    + "and a gradient is taken with respect to floating-point tensors only");

        var byKey = functionList.ToDictionary(f => FunctionKey(f.Domain, f.Name), StringComparer.Ordinal);
        CheckPath([.. graph.Nodes.Take(step.Index)], [.. step.Wrt], [step.Loss], byKey);
        return step;
    }

    private static bool ContainsAutoGrad(NodeProto node)
        => IsAutoGrad(node) || node.Attributes.Any(a => a.G is { } g && g.Nodes.Any(ContainsAutoGrad));

    private static int ElementType(GraphProto graph, string name)
    {
        var info = graph.Inputs.Concat(graph.ValueInfoes).FirstOrDefault(v => v.Name == name);
        if (info?.Type?.TensorType is { } tensor) return tensor.ElemType;
        return graph.Initializers.FirstOrDefault(i => i.Name == name)?.data_type ?? 0;
    }

    private static string FunctionKey(string domain, string name) => domain + "\u0001" + name;

    /// <summary>
    /// Refuses an operator the backend does not differentiate through (<see cref="TorchGradient.Refused"/>)
    /// that lies on a path from a value in <paramref name="dependent"/> to one of
    /// <paramref name="targets"/> among <paramref name="nodes"/> — inside a branch, a loop body or a
    /// called function too. A path ends at an operator whose outputs carry no gradient.
    /// </summary>
    private static void CheckPath(
        IReadOnlyList<NodeProto> nodes, HashSet<string> dependent, IEnumerable<string> targets,
        IReadOnlyDictionary<string, FunctionProto> functions)
    {
        var reached = new bool[nodes.Count];
        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];
            if (!Reads(node).Any(dependent.Contains) || !CarriesGradient(node, functions)) continue;
            reached[i] = true;
            dependent.UnionWith(node.Outputs.Where(o => o.Length > 0));
        }

        var needed = new HashSet<string>(targets, StringComparer.Ordinal);
        for (int i = nodes.Count - 1; i >= 0; i--)
        {
            var node = nodes[i];
            if (!node.Outputs.Any(needed.Contains)) continue;
            needed.UnionWith(Reads(node));
            if (reached[i]) CheckNode(node, dependent, functions);
        }
    }

    private static void CheckNode(
        NodeProto node, HashSet<string> dependent, IReadOnlyDictionary<string, FunctionProto> functions)
    {
        if (functions.TryGetValue(FunctionKey(node.Domain, node.OpType), out var function))
        {
            var seeds = function.Inputs
                .Where((_, i) => i < node.Inputs.Count && dependent.Contains(node.Inputs[i]))
                .ToHashSet(StringComparer.Ordinal);
            CheckPath(function.Nodes, seeds, function.Outputs, functions);
            return;
        }
        if (GradientOf(node) == TorchGradient.Refused)
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedUsage, node.Domain, node.OpType,
                $"The PyTorch backend does not differentiate through the {node.OpType} operator, and the {node.OpType} "
                + $"node '{node.Name}' lies between the loss of the training step's AutoGrad node and a tensor it "
                + "differentiates with respect to.");
        foreach (var body in node.Attributes.Select(a => a.G).OfType<GraphProto>())
        {
            var inner = new HashSet<string>(dependent, StringComparer.Ordinal);
            inner.UnionWith(body.Inputs.Select(i => i.Name));
            CheckPath(body.Nodes, inner, body.Outputs.Select(o => o.Name), functions);
        }
    }

    private static bool CarriesGradient(NodeProto node, IReadOnlyDictionary<string, FunctionProto> functions)
        => functions.ContainsKey(FunctionKey(node.Domain, node.OpType)) || GradientOf(node) != TorchGradient.NotDifferentiable;

    private static TorchGradient? GradientOf(NodeProto node)
        => node.Domain is "" or "ai.onnx" ? OperatorTable.GradientOf(node.OpType) : null;

    /// <summary>What a node reads: its inputs, and the values of enclosing scopes its bodies read.</summary>
    private static IEnumerable<string> Reads(NodeProto node)
        => node.Inputs.Where(i => i.Length > 0)
            .Concat(node.Attributes.Select(a => a.G).OfType<GraphProto>().SelectMany(Captured));

    private static IEnumerable<string> Captured(GraphProto graph)
    {
        var defined = graph.Inputs.Select(i => i.Name)
            .Concat(graph.Initializers.Select(i => i.Name))
            .Concat(graph.Nodes.SelectMany(n => n.Outputs))
            .ToHashSet(StringComparer.Ordinal);
        return graph.Nodes.SelectMany(Reads).Where(name => !defined.Contains(name));
    }

    private static TorchUnsupportedModelException Refusal(string what)
        => new(TorchUnsupportedReason.UnsupportedUsage, TrainingFormats.AutoGradDomain, TrainingFormats.AutoGradOpType,
            $"The PyTorch backend cannot run the training step's AutoGrad node: {what}.");
}

internal sealed partial class OnnxToPythonTranslator
{
    /// <summary>
    /// Writes the body of a training step's main function. Each tensor the AutoGrad node
    /// differentiates with respect to becomes a leaf where it enters — at the top for a graph input,
    /// right after the node that makes it otherwise; everything up to and including the AutoGrad node
    /// runs under <c>torch.enable_grad()</c>; after it each such tensor is detached, so the optimizer
    /// update runs under the run's <c>torch.no_grad()</c> with nothing recording; and every output is
    /// returned detached.
    /// </summary>
    private void EmitTrainingStep(GraphProto graph, Scope scope, AutoGradStep step)
    {
        var wrt = step.Wrt.Distinct(StringComparer.Ordinal).ToList();
        var wrtSet = wrt.ToHashSet(StringComparer.Ordinal);
        foreach (var name in wrt)
            if (scope.IsBoundHere(name)) Rebind(scope, name, "training.leaf");

        Line("with torch.enable_grad():");
        _indent++;
        for (int i = 0; i < step.Index; i++)
        {
            var node = graph.Nodes[i];
            EmitNode(node, scope);
            foreach (var output in node.Outputs)
                if (wrtSet.Contains(output)) Rebind(scope, output, "training.leaf");
        }
        var loss = scope.Lookup(step.Loss, step.Node);
        var arguments = step.Wrt.Select(w => scope.Lookup(w, step.Node)).ToList();
        var call = $"training.autograd({loss}, ({string.Concat(arguments.Select(a => a + ", "))}))";
        var gradients = step.Node.Outputs.Select(o => o.Length == 0 ? "_" : Define(scope, o)).ToList();
        Line(gradients.Count == 0 ? call : $"{string.Join(", ", gradients)}, = {call}");
        _indent--;

        foreach (var name in wrt) Rebind(scope, name, "training.detach");
        for (int i = step.Index + 1; i < graph.Nodes.Count; i++) EmitNode(graph.Nodes[i], scope);
        Return(graph.Outputs.Select(o => o.Name), scope, "training.detach");
    }

    private void Rebind(Scope scope, string onnxName, string function)
    {
        var current = scope.Lookup(onnxName, null);
        Line($"{Define(scope, onnxName)} = {function}({current})");
    }
}
