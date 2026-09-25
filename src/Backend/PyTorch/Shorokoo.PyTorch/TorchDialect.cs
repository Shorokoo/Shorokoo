using Shorokoo.PythonTranslation;

namespace Shorokoo.PyTorch;

/// <summary>
/// The PyTorch backend's translation: calls into <c>shorokoo_torch</c>, a stop point before every
/// node, the gradient of a training step taken by torch autograd over the tape its forward pass
/// records, and every operator the table translates run.
/// </summary>
internal sealed class TorchDialect : PythonDialect
{
    public static TorchDialect Instance { get; } = new();

    private TorchDialect() { }

    public override string BackendName => "PyTorch";

    public override string Package => "shorokoo_torch";

    public override IReadOnlyList<string> Imports { get; } = ["import torch"];

    public override GradientStyle Gradients => GradientStyle.Tape;

    public override bool StopPoints => true;

    public override NotSupportedException Unsupported(
        UnsupportedReason reason, string? domain, string? operatorType, string message)
        => new TorchUnsupportedModelException((TorchUnsupportedReason)reason, domain, operatorType, message);
}
