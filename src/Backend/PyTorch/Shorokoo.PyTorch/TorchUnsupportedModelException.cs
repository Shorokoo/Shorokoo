namespace Shorokoo.PyTorch;

/// <summary>What about a model the PyTorch backend cannot run.</summary>
public enum TorchUnsupportedReason
{
    /// <summary>An operator the backend has no translation for at all.</summary>
    UnknownOperator,

    /// <summary>An operator the backend translates, used with an attribute, an input or an output
    /// its translation does not handle.</summary>
    UnsupportedUsage,

    /// <summary>Something about the model itself rather than one operator: external tensor data,
    /// a sparse initializer, a value used before it is made.</summary>
    UnsupportedModel,
}

/// <summary>
/// The PyTorch backend cannot run a model, found while its session is being created — never on the
/// first run. <see cref="Operator"/> names the operator at fault, where one is.
/// </summary>
public sealed class TorchUnsupportedModelException : NotSupportedException
{
    /// <summary>Creates the exception.</summary>
    public TorchUnsupportedModelException(
        TorchUnsupportedReason reason, string? domain, string? operatorType, string message)
        : base(message)
    {
        Reason = reason;
        Domain = domain;
        Operator = operatorType;
    }

    /// <summary>What kind of thing the backend cannot run.</summary>
    public TorchUnsupportedReason Reason { get; }

    /// <summary>The operator's domain, empty for the standard one, or null when no operator is at
    /// fault.</summary>
    public string? Domain { get; }

    /// <summary>The operator at fault, or null.</summary>
    public string? Operator { get; }
}
