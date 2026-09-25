using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.PythonTranslation;

/// <summary>What about a model a Python-based backend cannot run; each backend's public reason
/// enum has these members, in this order.</summary>
internal enum UnsupportedReason
{
    /// <summary>An operator the backend has no translation for at all.</summary>
    UnknownOperator,

    /// <summary>An operator the backend translates, used with an attribute, an input or an output
    /// its translation does not handle.</summary>
    UnsupportedUsage,

    /// <summary>Something about the model itself rather than one operator.</summary>
    UnsupportedModel,
}

/// <summary>How a translated training step computes the gradient of its
/// <c>ai.shorokoo.training::AutoGrad</c> node.</summary>
internal enum GradientStyle
{
    /// <summary>The forward pass records a tape as it runs, under <c>torch.enable_grad()</c>, and
    /// <c>torch.autograd.grad</c> walks it back.</summary>
    Tape,

    /// <summary>The forward pass is written as a function of the tensors it is differentiated with
    /// respect to, and <c>jax.value_and_grad</c> transforms it.</summary>
    Transform,
}

/// <summary>
/// What a Python-based backend's translation differs in: the support package the translated
/// module calls, what it imports, which operators and element types the backend refuses on top of
/// those the operator table has no entry for, how it computes a training step's gradient, and the
/// exception it refuses a model with. The operator table and the calls it writes are shared: a
/// backend's support package implements every family module under the same function names.
/// </summary>
internal abstract class PythonDialect
{
    /// <summary>The backend's name as refusals say it: "The {BackendName} backend …".</summary>
    public abstract string BackendName { get; }

    /// <summary>The support package the translated module imports its operator families from.</summary>
    public abstract string Package { get; }

    /// <summary>The import lines the translated module starts with, before the support package's.</summary>
    public abstract IReadOnlyList<string> Imports { get; }

    /// <summary>How the backend computes a training step's gradient.</summary>
    public abstract GradientStyle Gradients { get; }

    /// <summary>Whether the translation calls <c>_stop()</c> before every node, so a run can be
    /// stopped between two.</summary>
    public abstract bool StopPoints { get; }

    /// <summary>Why the backend refuses <paramref name="node"/>, an operator the table translates,
    /// or null where it runs it.</summary>
    public virtual string? Refusal(NodeProto node) => null;

    /// <summary>Whether the backend holds sequences and optionals; one that does not refuses a graph
    /// that takes or returns one.</summary>
    public virtual bool HoldsSequences => true;

    /// <summary>Why the backend cannot hold a tensor of <paramref name="elementType"/>, or null
    /// where it can.</summary>
    public virtual string? Refusal(ShorokooTensorElementType elementType) => null;

    /// <summary>The exception the backend refuses a model with.</summary>
    public abstract NotSupportedException Unsupported(
        UnsupportedReason reason, string? domain, string? operatorType, string message);
}
