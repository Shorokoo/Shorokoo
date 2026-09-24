using System.Text;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PyTorch.Translation.Operators;

namespace Shorokoo.PyTorch.Translation;

/// <summary>A model translated to Python: the module's source, the constants it reads as
/// <c>_C[i]</c>, and its interface.</summary>
internal sealed record TranslatedModel(
    string Source,
    IReadOnlyList<TorchConstant> Constants,
    string[] InputNames,
    string[] OutputNames,
    ShorokooTensorElementType[] OutputSequenceElementTypes)
{
    /// <summary>The outputs a run may write into consumed inputs' memory (see
    /// <see cref="OnnxToPythonTranslator.Translate(ModelProto, IReadOnlyList{OutputAlias})"/>).</summary>
    public IReadOnlyList<AliasSlot> Aliases { get; init; } = [];
}

/// <summary>
/// The names a graph's values go by in the Python being written, scope by scope: a subgraph sees
/// its enclosing graph's values, as ONNX says it does, and Python's closures give the nested
/// function exactly that. Every value gets an identifier of its own, so no scope ever shadows
/// another.
/// </summary>
internal sealed class Scope(Scope? parent)
{
    private readonly Dictionary<string, string> _names = new(StringComparer.Ordinal);

    public void Bind(string onnxName, string expression) => _names[onnxName] = expression;

    public bool IsBoundHere(string onnxName) => _names.ContainsKey(onnxName);

    public string Lookup(string onnxName, NodeProto? user)
    {
        for (var scope = this; scope is not null; scope = scope.Parent)
            if (scope._names.TryGetValue(onnxName, out var expression)) return expression;
        throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedModel, user?.Domain, user?.OpType,
            $"The value '{onnxName}'{(user is null ? "" : $", read by the {user.OpType} node '{user.Name}',")} "
            + "is not produced before it is read: the graph is not in topological order, or reads a "
            + "value nothing makes.");
    }

    public Scope? Parent { get; } = parent;
}

/// <summary>
/// Writes a Python module that computes an ONNX model with the support package's helpers.
///
/// <para>The module defines <c>main</c>, taking the graph's inputs in order and returning a tuple of
/// its outputs; one function per <c>FunctionProto</c> the model carries (Shorokoo emits its
/// components as functions of the <c>Functions</c> domain); and, nested where they are used, one
/// function per <c>If</c> branch and <c>Loop</c> body. Each node becomes one statement: its
/// operator's emitter (<see cref="OperatorTable"/>) writes the expression, and the translator
/// assigns the outputs. Every node is read under the opset its graph imports, a function's own
/// imports for a function body.</para>
///
/// <para>A training step whose gradient is left to the backend carries one
/// <c>ai.shorokoo.training::AutoGrad</c> node, which becomes <c>torch.autograd.grad</c> over the
/// forward pass before it (see <see cref="AutoGradStep"/> and <c>shorokoo_torch/training.py</c>).</para>
///
/// <para>Everything the backend cannot translate is found here, while the session is being
/// created — an operator it has no entry for, an attribute an entry does not handle — and
/// refused with a <see cref="TorchUnsupportedModelException"/> naming it.</para>
/// </summary>
internal sealed partial class OnnxToPythonTranslator
{
    private const string FunctionsDomain = "Functions";

    private readonly StringBuilder _source = new();
    private readonly List<TorchConstant> _constants = [];
    private readonly Dictionary<string, (string Name, FunctionProto Proto)> _functions = new(StringComparer.Ordinal);
    private int _indent;
    private int _nextValue;
    private int _nextGraph;
    private IReadOnlyDictionary<string, long> _opsets = new Dictionary<string, long>();

    private OnnxToPythonTranslator() { }

    /// <summary>Translates <paramref name="model"/>.</summary>
    /// <exception cref="TorchUnsupportedModelException">The model uses something the backend
    /// cannot run.</exception>
    public static TranslatedModel Translate(ModelProto model)
    {
        ArgumentNullException.ThrowIfNull(model);
        var graph = model.Graph ?? throw new TorchUnsupportedModelException(
            TorchUnsupportedReason.UnsupportedModel, null, null, "The model has no graph.");
        return new OnnxToPythonTranslator().Run(model, graph);
    }

    private TranslatedModel Run(ModelProto model, GraphProto graph)
    {
        var modelOpsets = Opsets(model.OpsetImports, null);
        for (int i = 0; i < model.Functions.Count; i++)
        {
            var function = model.Functions[i];
            _functions[FunctionKey(function.Domain, function.Name)] = ($"f{i}", function);
        }
        var training = AutoGradStep.Find(model, graph, modelOpsets, _functions.Values.Select(f => f.Proto));

        Line("# Translated from ONNX by Shorokoo's PyTorch backend.");
        Line("import torch");
        Line($"from shorokoo_torch import {string.Join(", ", OperatorTable.Modules)}");
        if (training is not null) Line("from shorokoo_torch import training");
        foreach (var (name, function) in _functions.Values)
        {
            _opsets = Opsets(function.OpsetImports, modelOpsets);
            Line();
            EmitFunction(name, function);
        }

        _opsets = modelOpsets;
        var initializers = graph.Initializers.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        var inputs = graph.Inputs.Select(i => i.Name).Where(n => !initializers.Contains(n)).ToArray();
        Line();
        EmitGraph("main", graph, inputs, parent: null, training);

        return new TranslatedModel(
            _source.ToString(),
            _constants,
            inputs,
            [.. graph.Outputs.Select(o => o.Name)],
            [.. graph.Outputs.Select(o => (ShorokooTensorElementType)(o.Type?.SequenceType?.ElemType?.TensorType?.ElemType ?? 0))]);
    }

    private static Dictionary<string, long> Opsets(
        IEnumerable<OperatorSetIdProto> imports, IReadOnlyDictionary<string, long>? fallback)
    {
        var opsets = fallback is null
            ? new Dictionary<string, long>(StringComparer.Ordinal)
            : new Dictionary<string, long>(fallback, StringComparer.Ordinal);
        foreach (var import in imports)
            opsets[import.Domain == "ai.onnx" ? "" : import.Domain] = import.Version;
        return opsets;
    }

    private static string FunctionKey(string domain, string name) => domain + "\u0001" + name;

    private void EmitFunction(string name, FunctionProto function)
    {
        if (function.Attributes.Count > 0 || function.AttributeProtoes.Count > 0)
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedModel, function.Domain, function.Name,
                $"The function {function.Name} takes attributes, which the PyTorch backend does not bind.");
        var scope = new Scope(null);
        var parameters = function.Inputs.Select(input => Define(scope, input)).ToList();
        Line($"def {name}({string.Join(", ", parameters)}):");
        _indent++;
        foreach (var node in function.Nodes) EmitNode(node, scope);
        Return(function.Outputs, scope);
        _indent--;
    }

    /// <summary>Writes a graph as a function <paramref name="name"/> taking
    /// <paramref name="inputs"/>, reading values of <paramref name="parent"/> by closure.</summary>
    private void EmitGraph(
        string name, GraphProto graph, IReadOnlyList<string> inputs, Scope? parent, AutoGradStep? training = null)
    {
        if (graph.SparseInitializers.Count > 0)
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedModel, null, null,
                $"The graph '{graph.Name}' has sparse initializers, which the PyTorch backend does not read.");
        var scope = new Scope(parent);
        var parameters = inputs.Select(input => Define(scope, input)).ToList();
        Line($"def {name}({string.Join(", ", parameters)}):");
        _indent++;
        foreach (var initializer in graph.Initializers)
            if (!scope.IsBoundHere(initializer.Name))
                scope.Bind(initializer.Name, AddConstant(TorchConstant.FromTensor(initializer, null)));
        if (training is null)
        {
            foreach (var node in graph.Nodes) EmitNode(node, scope);
            Return(graph.Outputs.Select(o => o.Name), scope);
        }
        else
        {
            EmitTrainingStep(graph, scope, training);
        }
        _indent--;
    }

    /// <summary>Writes a subgraph as a nested function where the current statement is about to be
    /// written, and returns its name.</summary>
    internal string EmitSubgraph(GraphProto graph, Scope parent)
    {
        var name = $"g{_nextGraph++}";
        EmitGraph(name, graph, [.. graph.Inputs.Select(i => i.Name)], parent);
        return name;
    }

    internal string AddConstant(TorchConstant constant)
    {
        _constants.Add(constant);
        return $"_C[{_constants.Count - 1}]";
    }

    private void Return(IEnumerable<string> outputs, Scope scope, string wrap = "")
    {
        var values = outputs.Select(o => wrap.Length == 0 ? scope.Lookup(o, null) : $"{wrap}({scope.Lookup(o, null)})").ToList();
        Line(values.Count == 0 ? "return ()" : $"return ({string.Join(", ", values)},)");
    }

    private string Define(Scope scope, string onnxName)
    {
        var identifier = $"v{_nextValue++}";
        if (onnxName.Length > 0) scope.Bind(onnxName, identifier);
        return identifier;
    }

    private void EmitNode(NodeProto node, Scope scope)
    {
        BeforeNode(node);
        string expression;
        bool returnsTuple;
        var outputs = node.Outputs.ToList();
        if (_functions.TryGetValue(FunctionKey(node.Domain, node.OpType), out var function))
        {
            // Shorokoo annotates the calls it writes with shrk_* attributes (the structure and
            // types of the values passed) that the function body never reads; any other attribute
            // would be an argument, which is not bound.
            if (node.Attributes.FirstOrDefault(a => !a.Name.StartsWith("shrk_", StringComparison.Ordinal)) is { } argument)
                throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedUsage, node.Domain, node.OpType,
                    $"The call of function {node.OpType} passes the attribute '{argument.Name}', which the PyTorch backend does not bind.");
            var arguments = node.Inputs.Select(i => i.Length == 0 ? "None" : scope.Lookup(i, node));
            expression = $"{function.Name}({string.Join(", ", arguments)})";
            if (outputs.Count < function.Proto.Outputs.Count) expression += $"[:{outputs.Count}]";
            returnsTuple = true;
        }
        else if (node.Domain is "" or "ai.onnx" && OperatorTable.TryGet(node.OpType, out var entry))
        {
            var context = new NodeContext(this, node, scope, _opsets.GetValueOrDefault("", 21));
            expression = entry.Emit(context);
            returnsTuple = entry.ReturnsTuple;
        }
        else
        {
            var where = node.Domain is "" or "ai.onnx" ? "" : $" of domain '{node.Domain}'";
            throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnknownOperator, node.Domain, node.OpType,
                node.Domain == FunctionsDomain
                    ? $"The model calls the function {node.OpType}, which it does not define."
                    : $"The PyTorch backend has no translation for the operator {node.OpType}{where} "
                      + $"(node '{node.Name}').");
        }

        if (!returnsTuple)
        {
            while (outputs.Count > 1 && outputs[^1].Length == 0) outputs.RemoveAt(outputs.Count - 1);
            if (outputs.Count > 1)
                throw new TorchUnsupportedModelException(TorchUnsupportedReason.UnsupportedUsage, node.Domain, node.OpType,
                    $"The PyTorch backend cannot run the {node.OpType} node '{node.Name}': its optional output "
                    + $"'{outputs[1]}' is not one the translation produces.");
        }

        var targets = outputs.Select(o => o.Length == 0 ? "_" : Define(scope, o)).ToList();
        if (targets.Count == 0)
            Line(expression);
        else if (returnsTuple)
            Line($"{string.Join(", ", targets)}, = {expression}");
        else if (AliasedWrite(node, scope, targets))
            Line($"if {targets[0]} is None: {targets[0]} = {expression}");
        else
            Line($"{targets[0]} = {expression}");
    }

    private void Line(string text = "")
    {
        if (text.Length > 0) _source.Append(' ', 4 * _indent).Append(text);
        _source.Append('\n');
    }
}
