namespace Shorokoo.Core.Backends;

/// <summary>
/// One message a backend's runtime emitted while it built a session or ran one, as
/// <see cref="LogSettings.Sink"/> receives it: plain text, with no terminal colour codes.
/// </summary>
/// <param name="Severity">How severe the runtime judged it.</param>
/// <param name="Source">The runtime that emitted it: <c>ONNX Runtime</c>, <c>PyTorch</c> or
/// <c>JAX</c> for the backends Shorokoo ships.</param>
/// <param name="Category">The runtime's own name for the kind of message: ONNX Runtime's logging
/// category (<c>onnxruntime</c> for its own messages), or the class of a Python warning
/// (<c>UserWarning</c>, <c>DeprecationWarning</c>, ...).</param>
/// <param name="Location">Where in the runtime it was emitted, as the runtime names it (a source
/// file and line, and for ONNX Runtime the function), or empty where it names none.</param>
/// <param name="Text">The message itself.</param>
public sealed record RuntimeLogMessage(
    ShorokooLogSeverity Severity, string Source, string Category, string Location, string Text)
{
    /// <summary>One line: the runtime and the severity, where it was emitted, and the message —
    /// <c>[ONNX Runtime Warning] graph.cc:5607 CleanUnusedInitializersAndNodeArgs: Removing initializer ...</c>.</summary>
    public override string ToString()
        => Location.Length == 0 ? $"[{Source} {Severity}] {Text}" : $"[{Source} {Severity}] {Location}: {Text}";
}
