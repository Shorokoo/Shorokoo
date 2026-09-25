namespace Shorokoo.PythonTranslation.Operators;

/// <summary>How a backend that computes a training step's gradient itself treats an operator it
/// differentiates through.</summary>
internal enum GradientRule
{
    /// <summary>The framework differentiates the translation as written.</summary>
    Differentiable,

    /// <summary>The operator's outputs carry no gradient — comparisons, shapes, indices — so
    /// nothing flows back through it.</summary>
    NotDifferentiable,

    /// <summary>The training backend refuses a step that would differentiate through it.</summary>
    Refused,
}

/// <summary>Writes the Python expression a node evaluates to.</summary>
internal delegate string OperatorEmitter(NodeContext node);

/// <summary>
/// One operator's translation: the expression it emits and whether that expression is a tuple of
/// the node's outputs (in which case the node's outputs are unpacked from its leading elements, so
/// a node may leave off optional outputs at the end) or the node's single output.
/// </summary>
internal sealed record OperatorEntry(string OpType, OperatorEmitter Emit, bool ReturnsTuple, GradientRule Gradient);

/// <summary>
/// The ONNX operators the Python-based backends translate, keyed by operator type (standard domain).
///
/// <para><b>One file per operator family.</b> Each family registers its operators in its own
/// file — <c>OperatorTable.Elementwise.cs</c>, <c>OperatorTable.Reduction.cs</c>, ... — by
/// implementing the partial method this file declares for it, and the ONNX semantics live in the
/// matching Python module of each backend's support package (<c>ops_elementwise.py</c>, ...).
/// Adding an operator therefore touches exactly those two files. An operator absent from the table
/// fails the session that uses it when it is created, naming it.</para>
///
/// <para>Most operators are one line: <see cref="Registry.Map"/> sends the node's inputs to a
/// Python function positionally (an omitted optional input as <c>None</c>) and its attributes as
/// keywords named as ONNX names them — only the attributes listed, so a node carrying any other is
/// refused rather than run with it ignored. Operators with graph attributes or other structure
/// register an <see cref="OperatorEmitter"/> of their own with <see cref="Registry.Custom"/>.</para>
/// </summary>
internal static partial class OperatorTable
{
    /// <summary>The Python modules a translated model imports, one per family.</summary>
    public static IReadOnlyList<string> Modules { get; } =
    [
        "ops_elementwise", "ops_logic", "ops_reduction", "ops_shape", "ops_indexing", "ops_conv_pool",
        "ops_norm", "ops_linalg", "ops_control", "ops_sequence", "ops_string", "ops_random",
        "ops_signal", "ops_image", "ops_quant", "ops_rnn",
    ];

    private static readonly Dictionary<string, OperatorEntry> Entries = Build();

    /// <summary>The translation of a standard-domain operator, if there is one.</summary>
    public static bool TryGet(string opType, out OperatorEntry entry) => Entries.TryGetValue(opType, out entry!);

    /// <summary>Every operator the table translates.</summary>
    public static IReadOnlyCollection<string> Supported => Entries.Keys;

    [ThreadStatic] private static Dictionary<string, GradientRule>? _gradientOverrides;

    /// <summary>How the training backend treats a standard-domain operator, or null for one the
    /// table does not translate.</summary>
    public static GradientRule? GradientOf(string opType)
        => _gradientOverrides?.TryGetValue(opType, out var overridden) == true ? overridden
            : Entries.TryGetValue(opType, out var entry) ? entry.Gradient : null;

    /// <summary>Treats <paramref name="opType"/> as <paramref name="gradient"/> on this thread until
    /// the result is disposed — for tests of what the training backend does with a classification
    /// no operator has yet.</summary>
    internal static IDisposable OverrideGradient(string opType, GradientRule gradient)
    {
        var overrides = _gradientOverrides ??= new Dictionary<string, GradientRule>(StringComparer.Ordinal);
        overrides[opType] = gradient;
        return new GradientOverride(opType);
    }

    private sealed class GradientOverride(string opType) : IDisposable
    {
        public void Dispose() => _gradientOverrides?.Remove(opType);
    }

    private static Dictionary<string, OperatorEntry> Build()
    {
        var registry = new Registry();
        RegisterElementwise(registry);
        RegisterLogic(registry);
        RegisterReduction(registry);
        RegisterShape(registry);
        RegisterIndexing(registry);
        RegisterConvPool(registry);
        RegisterNorm(registry);
        RegisterLinalg(registry);
        RegisterControl(registry);
        RegisterSequence(registry);
        RegisterString(registry);
        RegisterRandom(registry);
        RegisterSignal(registry);
        RegisterImage(registry);
        RegisterQuant(registry);
        RegisterRnn(registry);
        return registry.Entries;
    }

    static partial void RegisterElementwise(Registry table);
    static partial void RegisterLogic(Registry table);
    static partial void RegisterReduction(Registry table);
    static partial void RegisterShape(Registry table);
    static partial void RegisterIndexing(Registry table);
    static partial void RegisterConvPool(Registry table);
    static partial void RegisterNorm(Registry table);
    static partial void RegisterLinalg(Registry table);
    static partial void RegisterControl(Registry table);
    static partial void RegisterSequence(Registry table);
    static partial void RegisterString(Registry table);
    static partial void RegisterRandom(Registry table);
    static partial void RegisterSignal(Registry table);
    static partial void RegisterImage(Registry table);
    static partial void RegisterQuant(Registry table);
    static partial void RegisterRnn(Registry table);

    /// <summary>What a family file registers its operators through.</summary>
    internal sealed class Registry
    {
        public Dictionary<string, OperatorEntry> Entries { get; } = new(StringComparer.Ordinal);

        /// <summary>
        /// <paramref name="opType"/> as a call of <paramref name="function"/> (<c>module.name</c>):
        /// the node's inputs positionally, then the <paramref name="attributes"/> it carries as
        /// keywords. <paramref name="opset"/> also passes <c>_opset</c>, the node's opset version, for
        /// a function whose semantics changed between versions; <paramref name="outputs"/> passes
        /// <c>_outputs</c>, the node's output count, for a function that returns a tuple of that many;
        /// <paramref name="tuple"/> marks a function that returns a tuple of every output the
        /// operator has, of which the node takes as many as it declares.
        /// </summary>
        public void Map(
            string opType, string function, string[]? attributes = null, bool opset = false,
            bool outputs = false, bool tuple = false, GradientRule gradient = GradientRule.Differentiable)
        {
            var accepted = attributes ?? [];
            Custom(opType, node => node.Call(function, accepted, opset, outputs), outputs || tuple, gradient);
        }

        /// <summary><paramref name="opType"/> translated by an emitter of its own.</summary>
        public void Custom(
            string opType, OperatorEmitter emit, bool returnsTuple = false,
            GradientRule gradient = GradientRule.Differentiable)
        {
            if (!Entries.TryAdd(opType, new OperatorEntry(opType, emit, returnsTuple, gradient)))
                throw new InvalidOperationException($"The operator {opType} is registered twice.");
        }
    }
}
