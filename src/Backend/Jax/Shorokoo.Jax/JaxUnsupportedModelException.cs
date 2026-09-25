namespace Shorokoo.Jax;

/// <summary>What about a model the JAX backend cannot run.</summary>
public enum JaxUnsupportedReason
{
    /// <summary>An operator the backend has no translation for at all, or one it does not run:
    /// strings, sequences and optionals, and the operators whose output shape their input's values
    /// decide.</summary>
    UnknownOperator,

    /// <summary>An operator the backend translates, used with an attribute, an input or an output
    /// its translation does not handle — or with a shape, a count or an axis the graph computes
    /// from an input's values, which a program compiled once per input shape cannot have.</summary>
    UnsupportedUsage,

    /// <summary>Something about the model itself rather than one operator: external tensor data,
    /// a sparse initializer, a string input.</summary>
    UnsupportedModel,
}

/// <summary>
/// The JAX backend cannot run a model. It is found while the model is translated, when its session
/// is created — and where the model's inputs have fixed shapes, compiling it then finds the rest;
/// where they do not, the run that first compiles it for a shape can find it too.
/// <see cref="Operator"/> names the operator at fault, where one is.
/// </summary>
public sealed class JaxUnsupportedModelException : NotSupportedException
{
    /// <summary>Creates the exception.</summary>
    public JaxUnsupportedModelException(
        JaxUnsupportedReason reason, string? domain, string? operatorType, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        Reason = reason;
        Domain = domain;
        Operator = operatorType;
    }

    /// <summary>What kind of thing the backend cannot run.</summary>
    public JaxUnsupportedReason Reason { get; }

    /// <summary>The operator's domain, empty for the standard one, or null when no operator is at
    /// fault.</summary>
    public string? Domain { get; }

    /// <summary>The operator at fault, or null.</summary>
    public string? Operator { get; }
}
