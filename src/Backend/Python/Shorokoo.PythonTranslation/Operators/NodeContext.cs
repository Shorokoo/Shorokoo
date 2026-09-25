using Shorokoo.Core.Factory.IR;

namespace Shorokoo.PythonTranslation.Operators;

/// <summary>
/// One node, as an operator's emitter sees it: its inputs as Python expressions, its attributes,
/// its opset version, and the translator's services for constants and subgraphs.
/// </summary>
internal sealed class NodeContext
{
    private readonly OnnxToPythonTranslator _translator;
    private readonly Scope _scope;

    internal NodeContext(OnnxToPythonTranslator translator, NodeProto node, Scope scope, long opset)
    {
        _translator = translator;
        Node = node;
        _scope = scope;
        Opset = opset;
        var count = node.Inputs.Count;
        while (count > 0 && node.Inputs[count - 1].Length == 0) count--;
        InputCount = count;
    }

    /// <summary>The node.</summary>
    public NodeProto Node { get; }

    /// <summary>The version of the standard domain the node is read under.</summary>
    public long Opset { get; }

    /// <summary>The node's inputs, not counting omitted ones at the end.</summary>
    public int InputCount { get; }

    /// <summary>The node's declared outputs, omitted ones included.</summary>
    public int OutputCount => Node.Outputs.Count;

    /// <summary>Input <paramref name="index"/> as a Python expression; an omitted one is <c>None</c>.</summary>
    public string Input(int index)
        => index < Node.Inputs.Count && Node.Inputs[index].Length > 0
            ? _scope.Lookup(Node.Inputs[index], Node)
            : "None";

    /// <summary>The attribute named <paramref name="name"/>, or null.</summary>
    public AttributeProto? Attribute(string name) => Node.Attributes.FirstOrDefault(a => a.Name == name);

    /// <summary>The graph attribute named <paramref name="name"/>.</summary>
    public GraphProto Graph(string name)
        => Attribute(name)?.G ?? throw Unsupported($"it has no graph attribute '{name}'");

    /// <summary>A constant the model holds, as the expression that reads it.</summary>
    public string Constant(PythonConstant constant) => _translator.AddConstant(constant);

    /// <summary>Writes the graph attribute <paramref name="name"/> as a nested function taking the
    /// graph's inputs, and returns its name.</summary>
    public string Subgraph(string name) => _translator.EmitSubgraph(Graph(name), _scope);

    /// <summary>A refusal of this node, naming the operator and <paramref name="what"/> about it.</summary>
    public NotSupportedException Unsupported(string what)
        => Dialect.Unsupported(UnsupportedReason.UnsupportedUsage, Node.Domain, Node.OpType,
            $"The {Dialect.BackendName} backend cannot run the {Node.OpType} node '{Node.Name}': {what}.");

    /// <summary>The backend the translation is written for.</summary>
    public PythonDialect Dialect => _translator.Dialect;

    /// <summary>
    /// The call <see cref="OperatorTable.Registry.Map"/> emits: <paramref name="function"/> with
    /// the node's inputs, its attributes named in <paramref name="accepted"/> as keywords, and the
    /// opset and output count where asked for.
    /// </summary>
    /// <exception cref="NotSupportedException">The node carries an attribute not in
    /// <paramref name="accepted"/>.</exception>
    public string Call(string function, IReadOnlyCollection<string> accepted, bool passOpset, bool passOutputs)
    {
        var arguments = new List<string>();
        for (int i = 0; i < InputCount; i++) arguments.Add(Input(i));
        foreach (var attribute in Node.Attributes)
        {
            if (!accepted.Contains(attribute.Name))
                throw Unsupported($"its attribute '{attribute.Name}' is not one the translation handles");
            arguments.Add($"{attribute.Name}={AttributeLiteral(attribute)}");
        }
        if (passOpset) arguments.Add($"_opset={PyLiteral.Int(Opset)}");
        if (passOutputs) arguments.Add($"_outputs={PyLiteral.Int(OutputCount)}");
        return $"{function}({string.Join(", ", arguments)})";
    }

    /// <summary>The Python literal for an attribute's value; a tensor becomes a constant.</summary>
    public string AttributeLiteral(AttributeProto attribute)
    {
        if (attribute.RefAttrName.Length > 0)
            throw Unsupported($"its attribute '{attribute.Name}' refers to a function's attribute '{attribute.RefAttrName}'");
        return KindOf(attribute) switch
        {
            AttributeProto.AttributeType.Int => PyLiteral.Int(attribute.I),
            AttributeProto.AttributeType.Float => PyLiteral.Float(attribute.F),
            AttributeProto.AttributeType.String => PyLiteral.Str(PythonConstant.Text(Dialect, attribute.S ?? [], AttributeDescription(attribute), Node.OpType)),
            AttributeProto.AttributeType.Ints => PyLiteral.List((attribute.Ints ?? []).Select(PyLiteral.Int)),
            AttributeProto.AttributeType.Floats => PyLiteral.List((attribute.Floats ?? []).Select(f => PyLiteral.Float(f))),
            AttributeProto.AttributeType.Strings => PyLiteral.List(attribute.Strings.Select(s => PyLiteral.Str(PythonConstant.Text(Dialect, s, AttributeDescription(attribute), Node.OpType)))),
            AttributeProto.AttributeType.Tensor => Constant(PythonConstant.FromTensor(Dialect, attribute.T, Node.OpType)),
            var kind => throw Unsupported($"its attribute '{attribute.Name}' is a {kind}, which has no literal"),
        };
    }

    /// <summary>How a refusal names <paramref name="attribute"/> of this node.</summary>
    public string AttributeDescription(AttributeProto attribute)
        => $"The attribute '{attribute.Name}' of the {Node.OpType} node '{Node.Name}'";

    /// <summary>An attribute's kind, read off the field it fills where a writer left the type
    /// unset; refused where it fills none, which leaves nothing to tell an empty list by.</summary>
    private AttributeProto.AttributeType KindOf(AttributeProto attribute)
    {
        if (attribute.Type != AttributeProto.AttributeType.Undefined) return attribute.Type;
        if (attribute.T is not null) return AttributeProto.AttributeType.Tensor;
        if (attribute.G is not null) return AttributeProto.AttributeType.Graph;
        if (attribute.S is not null) return AttributeProto.AttributeType.String;
        if (attribute.Ints is { Length: > 0 }) return AttributeProto.AttributeType.Ints;
        if (attribute.Floats is { Length: > 0 }) return AttributeProto.AttributeType.Floats;
        if (attribute.Strings.Count > 0) return AttributeProto.AttributeType.Strings;
        if (attribute.ShouldSerializeF()) return AttributeProto.AttributeType.Float;
        if (attribute.ShouldSerializeI()) return AttributeProto.AttributeType.Int;
        throw Unsupported($"its attribute '{attribute.Name}' has no type and no value to read one off");
    }
}
