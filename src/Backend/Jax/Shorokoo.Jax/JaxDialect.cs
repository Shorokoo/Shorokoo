using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PythonTranslation;

namespace Shorokoo.Jax;

/// <summary>
/// The JAX backend's translation: calls into <c>shorokoo_jax</c>; no stop points, since a run is
/// one compiled XLA program; a training step's gradient taken by <c>jax.value_and_grad</c> of its
/// forward pass; and refusals of what a program compiled once per input shape cannot hold —
/// strings, sequences and optionals, and the operators whose output shape their input's values
/// decide.
/// </summary>
internal sealed class JaxDialect : PythonDialect
{
    public static JaxDialect Instance { get; } = new();

    private JaxDialect() { }

    private const string NoStrings = "JAX has no string tensors";
    private const string NoSequences = "JAX has no sequences or optionals, and the backend holds a value as a JAX array";
    private const string DataDependent = "its output's shape is decided by its input's values, and the backend compiles a model once per input shape";

    private static readonly Dictionary<string, string> Refused = new(StringComparer.Ordinal)
    {
        ["StringConcat"] = NoStrings,
        ["StringSplit"] = NoStrings,
        ["StringNormalizer"] = NoStrings,
        ["RegexFullMatch"] = NoStrings,
        ["TfIdfVectorizer"] = NoStrings,
        ["ImageDecoder"] = DataDependent,
        ["SequenceConstruct"] = NoSequences,
        ["SequenceEmpty"] = NoSequences,
        ["SequenceInsert"] = NoSequences,
        ["SequenceErase"] = NoSequences,
        ["SequenceAt"] = NoSequences,
        ["SequenceLength"] = NoSequences,
        ["SplitToSequence"] = NoSequences,
        ["ConcatFromSequence"] = NoSequences,
        ["SequenceMap"] = NoSequences,
        ["Optional"] = NoSequences,
        ["OptionalHasElement"] = NoSequences,
        ["OptionalGetElement"] = NoSequences,
        ["NonZero"] = DataDependent,
        ["Unique"] = DataDependent,
        ["Compress"] = DataDependent,
        ["NonMaxSuppression"] = DataDependent,
    };

    /// <summary>The operators the table translates that the JAX backend refuses, and why.</summary>
    public static IReadOnlyDictionary<string, string> RefusedOperators => Refused;

    public override string BackendName => "JAX";

    public override string Package => "shorokoo_jax";

    public override IReadOnlyList<string> Imports { get; } = ["import jax", "import jax.numpy as jnp"];

    public override GradientStyle Gradients => GradientStyle.Transform;

    public override bool StopPoints => false;

    public override string? Refusal(NodeProto node) => Refused.GetValueOrDefault(node.OpType);

    public override string? Refusal(ShorokooTensorElementType elementType)
        => elementType == ShorokooTensorElementType.String ? NoStrings : null;

    public override NotSupportedException Unsupported(
        UnsupportedReason reason, string? domain, string? operatorType, string message)
        => new JaxUnsupportedModelException((JaxUnsupportedReason)reason, domain, operatorType, message);
}
