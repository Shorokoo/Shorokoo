using System.Text;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonTranslation.Operators;

namespace Shorokoo.PythonTranslation;

/// <summary>A model translated to Python: the module's source, the constants it reads as
/// <c>_C[i]</c>, and its interface.</summary>
internal sealed record TranslatedModel(
    string Source,
    IReadOnlyList<PythonConstant> Constants,
    string[] InputNames,
    string[] OutputNames,
    ShorokooTensorElementType[] OutputSequenceElementTypes)
{
    /// <summary>The outputs a run may write into consumed inputs' memory (see
    /// <see cref="OnnxToPythonTranslator.Translate(ModelProto, IReadOnlyList{OutputAlias}, PythonDialect)"/>).</summary>
    public IReadOnlyList<AliasSlot> Aliases { get; init; } = [];
}

/// <summary>
/// The names a graph's values go by in the Python being written, scope by scope: a subgraph sees
/// its enclosing graph's values, as ONNX says it does, and Python's closures give the nested
/// function exactly that. Every value gets an identifier of its own, so no scope ever shadows
/// another.
///
/// <para>A scope is one Python function, written statement by statement, and it records the last
/// statement that reads each identifier it assigns, so that the function can let go of the value
/// there (see <see cref="OnnxToPythonTranslator"/>). A nested function reading an enclosing
/// scope's value reads it within the enclosing statement that holds and calls it.</para>
/// </summary>
internal sealed class Scope(Scope? parent, PythonDialect dialect)
{
    private readonly Dictionary<string, string> _names = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _lastRead = new(StringComparer.Ordinal);

    public void Bind(string onnxName, string expression) => _names[onnxName] = expression;

    /// <summary>The statement being written, counted from the function's header, statement 0.</summary>
    public int Statement { get; private set; }

    /// <summary>Where each statement written so far ends in the source, and at what indentation.</summary>
    public List<(int Position, int Indent)> Ends { get; } = [];

    /// <summary>Records <paramref name="identifier"/> as a local this function assigns, in the
    /// statement being written.</summary>
    public void Assign(string identifier) => _lastRead[identifier] = Statement;

    /// <summary>Records <paramref name="identifier"/>, a local this function assigns, as read by the
    /// statement being written, where the statement reads it other than through <see cref="Lookup"/>.</summary>
    public void Read(string identifier) => _lastRead[identifier] = Statement;

    public void EndStatement(int position, int indent)
    {
        Ends.Add((position, indent));
        Statement++;
    }

    /// <summary>The locals this function assigns, grouped by the statement that reads each last.</summary>
    public IEnumerable<IGrouping<int, string>> LastReads()
        => _lastRead.GroupBy(pair => pair.Value, pair => pair.Key);

    public bool IsBoundHere(string onnxName) => _names.ContainsKey(onnxName);

    public string Lookup(string onnxName, NodeProto? user)
    {
        for (var scope = this; scope is not null; scope = scope.Parent)
        {
            if (!scope._names.TryGetValue(onnxName, out var expression)) continue;
            if (scope._lastRead.ContainsKey(expression)) scope._lastRead[expression] = scope.Statement;
            return expression;
        }
        throw dialect.Unsupported(UnsupportedReason.UnsupportedModel, user?.Domain, user?.OpType,
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
/// components as functions of the <c>Functions</c> domain) — or, for one that takes attributes, one
/// per set of attribute values it is called with, its body's references to them resolved; and,
/// nested where they are used, one function per <c>If</c> branch, <c>Loop</c> body and
/// <c>SequenceMap</c> body. Each node becomes one statement: its
/// operator's emitter (<see cref="OperatorTable"/>) writes the expression, and the translator
/// assigns the outputs. Every node is read under the opset its graph imports, a function's own
/// imports for a function body.</para>
///
/// <para>Every function lets go of each value it assigns — a parameter or a node's output — right
/// after the statement that reads it last, with a <c>del</c>, so that a run holds only the values
/// something still reads, as ONNX Runtime frees a buffer after its last use, and not every value
/// the graph ever made. A value a nested function reads is read by the statement that calls it; a
/// value the function returns is never let go of.</para>
///
/// <para>A training step whose gradient is left to the backend carries one
/// <c>ai.shorokoo.training::AutoGrad</c> node, which becomes the framework's gradient of the forward
/// pass before it — <c>torch.autograd.grad</c> over a recorded tape, or <c>jax.value_and_grad</c> of
/// the forward pass written as a function (see <see cref="AutoGradStep"/>, <see cref="GradientStyle"/>
/// and each support package's <c>training.py</c>).</para>
///
/// <para>What the module calls is the same for every backend — one helper per operator, under the
/// same name in each support package — and what differs is the backend's <see cref="PythonDialect"/>.
/// Everything the backend cannot translate is found here, while the session is being created — an
/// operator it has no entry for or refuses, an attribute an entry does not handle, an element type
/// it cannot hold — and refused with the dialect's exception, naming it.</para>
/// </summary>
internal sealed partial class OnnxToPythonTranslator
{
    private const string FunctionsDomain = "Functions";

    private StringBuilder _source = new();
    private List<(int Position, string Text)> _releases = [];
    private readonly StringBuilder _specializations = new();
    private readonly Dictionary<string, string> _specialized = new(StringComparer.Ordinal);
    private readonly List<PythonConstant> _constants = [];
    private readonly Dictionary<string, (string Name, FunctionProto Proto)> _functions = new(StringComparer.Ordinal);
    private int _indent;
    private int _nextValue;
    private int _nextGraph;
    private IReadOnlyDictionary<string, long> _opsets = new Dictionary<string, long>();
    private IReadOnlyDictionary<string, long> _modelOpsets = new Dictionary<string, long>();

    private OnnxToPythonTranslator(PythonDialect dialect) => Dialect = dialect;

    /// <summary>The backend the translation is written for.</summary>
    public PythonDialect Dialect { get; }

    /// <summary>Translates <paramref name="model"/> for the backend <paramref name="dialect"/> names.</summary>
    /// <exception cref="NotSupportedException">The model uses something the backend cannot run: the
    /// dialect's own exception.</exception>
    public static TranslatedModel Translate(ModelProto model, PythonDialect dialect)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(dialect);
        var graph = model.Graph ?? throw dialect.Unsupported(
            UnsupportedReason.UnsupportedModel, null, null, "The model has no graph.");
        return new OnnxToPythonTranslator(dialect).Run(model, graph);
    }

    private TranslatedModel Run(ModelProto model, GraphProto graph)
    {
        var modelOpsets = _modelOpsets = Opsets(model.OpsetImports, null);
        for (int i = 0; i < model.Functions.Count; i++)
        {
            var function = model.Functions[i];
            if (!_functions.TryAdd(FunctionKey(function.Domain, function.Name, function.Overload), ($"f{i}", function)))
                throw Dialect.Unsupported(UnsupportedReason.UnsupportedModel, function.Domain, function.Name,
                    $"The model defines the function {function.Name} of domain '{function.Domain}'"
                    + (function.Overload.Length > 0 ? $", overload '{function.Overload}'," : "") + " more than once.");
        }
        var training = AutoGradStep.Find(Dialect, model, graph, modelOpsets, _functions.Values.Select(f => f.Proto));
        CheckInterface(graph);

        Line($"# Translated from ONNX by Shorokoo's {Dialect.BackendName} backend.");
        foreach (var import in Dialect.Imports) Line(import);
        Line($"from {Dialect.Package} import {string.Join(", ", OperatorTable.Modules)}");
        if (training is not null) Line($"from {Dialect.Package} import training");
        foreach (var (name, function) in _functions.Values)
        {
            if (TakesAttributes(function)) continue;
            _opsets = Opsets(function.OpsetImports, modelOpsets);
            Line();
            EmitFunction(name, function, null);
        }

        _opsets = modelOpsets;
        var initializers = graph.Initializers.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        var inputs = graph.Inputs.Select(i => i.Name).Where(n => !initializers.Contains(n)).ToArray();
        Line();
        EmitGraph("main", graph, inputs, parent: null, training);
        _source = WithReleases(_source, _releases);
        _source.Append(_specializations);

        return new TranslatedModel(
            _source.ToString(),
            _constants,
            inputs,
            [.. graph.Outputs.Select(o => o.Name)],
            [.. graph.Outputs.Select(o => (ShorokooTensorElementType)(o.Type?.SequenceType?.ElemType?.TensorType?.ElemType ?? 0))]);
    }

    /// <summary>Refuses a graph input or output of an element type the backend cannot hold.</summary>
    private void CheckInterface(GraphProto graph)
    {
        foreach (var value in graph.Inputs.Concat(graph.Outputs))
        {
            var type = value.Type?.TensorType?.ElemType ?? value.Type?.SequenceType?.ElemType?.TensorType?.ElemType ?? 0;
            if (type > 0 && Dialect.Refusal((ShorokooTensorElementType)type) is { } why)
                throw Dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, null,
                    $"The {Dialect.BackendName} backend cannot run a graph whose input or output '{value.Name}' is of "
                    + $"element type {(ShorokooTensorElementType)type}: {why}.");
        }
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

    /// <summary>What names a model-local function, and a call of it: its domain, name and overload.</summary>
    internal static string FunctionKey(string domain, string name, string overload) => domain + "\u0001" + name + "\u0001" + overload;

    private static bool TakesAttributes(FunctionProto function)
        => function.Attributes.Count > 0 || function.AttributeProtoes.Count > 0;

    /// <summary>Writes <paramref name="function"/> as a Python function, its body's references to the
    /// function's attributes resolved from <paramref name="attributes"/> where it takes any.</summary>
    private void EmitFunction(string name, FunctionProto function, IReadOnlyDictionary<string, AttributeProto>? attributes)
    {
        var scope = new Scope(null, Dialect);
        var parameters = function.Inputs.Select(input => Define(scope, input)).ToList();
        Line($"def {name}({string.Join(", ", parameters)}):");
        _indent++;
        EndStatement(scope);
        foreach (var node in function.Nodes)
            EmitNode(attributes is null ? node : FunctionAttributes.Resolve(node, attributes), scope);
        Return(function.Outputs, scope);
        Release(scope);
        _indent--;
    }

    /// <summary>
    /// The name of the Python function computing <paramref name="function"/> as <paramref name="call"/>
    /// calls it: one written for each distinct set of attribute values the model calls the function
    /// with — the call's, and the function's defaults for those it leaves out — after the rest of
    /// the module, since Python binds a module's functions by name when they run.
    /// </summary>
    private string Specialize(string name, FunctionProto function, NodeProto call)
    {
        var declared = function.Attributes.Concat(function.AttributeProtoes.Select(a => a.Name)).ToHashSet(StringComparer.Ordinal);
        var values = new SortedDictionary<string, AttributeProto>(StringComparer.Ordinal);
        foreach (var fallback in function.AttributeProtoes) values[fallback.Name] = fallback;
        foreach (var attribute in call.Attributes.Where(a => declared.Contains(a.Name)))
            values[attribute.Name] = attribute;

        var key = FunctionKey(function.Domain, function.Name, function.Overload) + "\u0001" + FunctionAttributes.Key(values);
        if (_specialized.TryGetValue(key, out var specialization)) return specialization;
        specialization = $"{name}_{_specialized.Count}";
        _specialized[key] = specialization;

        var (source, releases, indent, opsets) = (_source, _releases, _indent, _opsets);
        (_source, _releases, _indent, _opsets) = (new StringBuilder(), [], 0, Opsets(function.OpsetImports, _modelOpsets));
        Line();
        EmitFunction(specialization, function, values);
        _specializations.Append(WithReleases(_source, _releases));
        (_source, _releases, _indent, _opsets) = (source, releases, indent, opsets);
        return specialization;
    }

    /// <summary>Writes a graph as a function <paramref name="name"/> taking
    /// <paramref name="inputs"/>, reading values of <paramref name="parent"/> by closure.</summary>
    private void EmitGraph(
        string name, GraphProto graph, IReadOnlyList<string> inputs, Scope? parent, AutoGradStep? training = null)
    {
        if (graph.SparseInitializers.Count > 0)
            throw Dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, null,
                $"The graph '{graph.Name}' has sparse initializers, which the {Dialect.BackendName} backend does not read.");
        var scope = new Scope(parent, Dialect);
        var parameters = inputs.Select(input => Define(scope, input)).ToList();
        Line($"def {name}({string.Join(", ", parameters)}):");
        _indent++;
        foreach (var initializer in graph.Initializers)
            if (!scope.IsBoundHere(initializer.Name))
                scope.Bind(initializer.Name, AddConstant(PythonConstant.FromTensor(Dialect, initializer, null)));
        EndStatement(scope);
        if (training is null)
        {
            foreach (var node in graph.Nodes) EmitNode(node, scope);
            Return(graph.Outputs.Select(o => o.Name), scope);
        }
        else if (Dialect.Gradients == GradientStyle.Tape)
        {
            EmitTrainingStep(graph, scope, training);
        }
        else
        {
            EmitTransformedTrainingStep(graph, scope, training);
        }
        Release(scope);
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

    internal string AddConstant(PythonConstant constant)
    {
        if (Dialect.Refusal(constant.ElementType) is { } why)
            throw Dialect.Unsupported(UnsupportedReason.UnsupportedModel, null, null,
                $"The {Dialect.BackendName} backend cannot hold the model's constant of element type {constant.ElementType}: {why}.");
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
        scope.Assign(identifier);
        return identifier;
    }

    /// <summary>Ends the statement of <paramref name="scope"/>'s function just written: where it
    /// ends is where the values it reads last are let go of.</summary>
    private void EndStatement(Scope scope) => scope.EndStatement(_source.Length, _indent);

    /// <summary>
    /// Lets go of every value <paramref name="scope"/>'s function assigns right after the statement
    /// that reads it last — right after the header for a parameter nothing reads, right after its own
    /// statement for an output nothing reads — except those its return statement reads, which is the
    /// last one and ends no statement. The <c>del</c>s are inserted where the statements end once the
    /// whole source is written (<see cref="WithReleases"/>), so that nothing written so far moves.
    /// </summary>
    private void Release(Scope scope)
    {
        foreach (var statement in scope.LastReads().Where(g => g.Key < scope.Ends.Count))
        {
            var (position, indent) = scope.Ends[statement.Key];
            _releases.Add((position, $"{new string(' ', 4 * indent)}del {string.Join(", ", statement.Order(StringComparer.Ordinal))}\n"));
        }
    }

    /// <summary><paramref name="source"/> with each of <paramref name="releases"/> inserted at its
    /// position.</summary>
    private static StringBuilder WithReleases(StringBuilder source, List<(int Position, string Text)> releases)
    {
        var text = source.ToString();
        var result = new StringBuilder(text.Length + releases.Sum(r => r.Text.Length));
        var at = 0;
        foreach (var (position, release) in releases.OrderBy(r => r.Position))
        {
            result.Append(text, at, position - at).Append(release);
            at = position;
        }
        return result.Append(text, at, text.Length - at);
    }

    private void EmitNode(NodeProto node, Scope scope)
    {
        BeforeNode(node);
        string expression;
        bool returnsTuple;
        var outputs = node.Outputs.ToList();
        if (_functions.TryGetValue(FunctionKey(node.Domain, node.OpType, node.Overload), out var function))
        {
            // A call's attributes bind the function's declared ones. Shorokoo also annotates the
            // calls it writes with shrk_* attributes (the structure and types of the values
            // passed), which the body never reads; any other attribute would be an argument the
            // function has no parameter for.
            var declared = function.Proto.Attributes.Concat(function.Proto.AttributeProtoes.Select(a => a.Name)).ToHashSet(StringComparer.Ordinal);
            if (node.Attributes.FirstOrDefault(a => !declared.Contains(a.Name) && !a.Name.StartsWith("shrk_", StringComparison.Ordinal)) is { } argument)
                throw Dialect.Unsupported(UnsupportedReason.UnsupportedUsage, node.Domain, node.OpType,
                    $"The call of function {node.OpType} passes the attribute '{argument.Name}', which the function does not declare.");
            // A call may leave the function's trailing inputs and outputs off, an input left off
            // being an omitted optional one; it has no more of either than the function declares.
            while (outputs.Count > 0 && outputs[^1].Length == 0) outputs.RemoveAt(outputs.Count - 1);
            if (node.Inputs.Count > function.Proto.Inputs.Count || outputs.Count > function.Proto.Outputs.Count)
                throw Dialect.Unsupported(UnsupportedReason.UnsupportedModel, node.Domain, node.OpType,
                    $"The call '{node.Name}' of function {node.OpType} has {node.Inputs.Count} inputs and {outputs.Count} "
                    + $"outputs, and the function declares {function.Proto.Inputs.Count} and {function.Proto.Outputs.Count}.");
            var callee = TakesAttributes(function.Proto) ? Specialize(function.Name, function.Proto, node) : function.Name;
            var arguments = node.Inputs.Select(i => i.Length == 0 ? "None" : scope.Lookup(i, node))
                .Concat(Enumerable.Repeat("None", function.Proto.Inputs.Count - node.Inputs.Count));
            expression = $"{callee}({string.Join(", ", arguments)})";
            if (outputs.Count < function.Proto.Outputs.Count) expression += $"[:{outputs.Count}]";
            returnsTuple = true;
        }
        else if (node.Domain is "" or "ai.onnx" && OperatorTable.TryGet(node.OpType, out var entry)
                 && Dialect.Refusal(node) is null)
        {
            var context = new NodeContext(this, node, scope, _opsets.GetValueOrDefault("", 21));
            expression = entry.Emit(context);
            returnsTuple = entry.ReturnsTuple;
            if (returnsTuple) expression += $"[:{outputs.Count}]";
        }
        else
        {
            var where = node.Domain is "" or "ai.onnx" ? "" : $" of domain '{node.Domain}'";
            throw Dialect.Unsupported(UnsupportedReason.UnknownOperator, node.Domain, node.OpType,
                node.Domain == FunctionsDomain
                    ? $"The model calls the function {node.OpType}, which it does not define."
                    : node.Domain is "" or "ai.onnx" && OperatorTable.TryGet(node.OpType, out _) && Dialect.Refusal(node) is { } why
                        ? $"The {Dialect.BackendName} backend does not run the operator {node.OpType} (node '{node.Name}'): {why}."
                        : $"The {Dialect.BackendName} backend has no translation for the operator {node.OpType}{where} "
                          + $"(node '{node.Name}').");
        }

        if (!returnsTuple)
        {
            while (outputs.Count > 1 && outputs[^1].Length == 0) outputs.RemoveAt(outputs.Count - 1);
            if (outputs.Count > 1)
                throw Dialect.Unsupported(UnsupportedReason.UnsupportedUsage, node.Domain, node.OpType,
                    $"The {Dialect.BackendName} backend cannot run the {node.OpType} node '{node.Name}': its optional output "
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
        EndStatement(scope);
    }

    private void Line(string text = "")
    {
        if (text.Length > 0) _source.Append(' ', 4 * _indent).Append(text);
        _source.Append('\n');
    }
}
