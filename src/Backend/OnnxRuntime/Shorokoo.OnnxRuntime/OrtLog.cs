using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Backends;

namespace Shorokoo.OnnxRuntime;

/// <summary>One message ONNX Runtime logged: its text, without colour codes, and where it came from.</summary>
/// <param name="Severity">How severe ONNX Runtime judged it.</param>
/// <param name="Category">ONNX Runtime's category for it, <c>onnxruntime</c> for its own messages.</param>
/// <param name="LogId">The logger it went through: the environment's, or a session's.</param>
/// <param name="CodeLocation">The file, line and function in ONNX Runtime that logged it.</param>
/// <param name="Message">The message itself.</param>
public sealed record OrtLogMessage(
    ShorokooLogSeverity Severity, string Category, string LogId, string CodeLocation, string Message)
{
    /// <summary>One line: the severity, where it came from, and the message.</summary>
    public override string ToString() => $"[ONNX Runtime {Severity}] {CodeLocation}: {Message}";
}

/// <summary>
/// Where ONNX Runtime's log messages go, and from which severity on, for every session of the four
/// ONNX Runtime backends (<c>Shorokoo.LinuxCPU</c>, <c>LinuxGPU</c>, <c>WinCPU</c>, <c>WinGPU</c>).
///
/// <para>Shorokoo makes ONNX Runtime's environment when the first of those backends is built, and
/// makes it to log through <see cref="Sink"/> rather than through ONNX Runtime's own sink, which
/// writes to the standard error stream wrapped in terminal colour codes. Messages the environment
/// logs (a kernel warning about a graph it is given, for instance) and messages a session logs both
/// arrive here. Both settings can be changed at any time and apply to the next message.</para>
///
/// <para>Where the program made ONNX Runtime's environment itself before the first backend was
/// built (<c>OrtEnv.CreateInstanceWithOptions</c>), that environment logs as it was made to and
/// these settings do not reach it. A backend loaded with <see cref="IsolatedBackend.Load"/> runs
/// its own copy of ONNX Runtime with its own copy of these settings, at their defaults.</para>
/// </summary>
public static class OrtLog
{
    private static volatile Action<OrtLogMessage>? _sink = WriteToStandardError;
    private static volatile int _severity = (int)ShorokooLogSeverity.Warning;

    [ThreadStatic] private static Capture? _captured;

    /// <summary>
    /// The least severe message passed on to <see cref="Sink"/>; <see cref="ShorokooLogSeverity.Warning"/>
    /// unless set otherwise. ONNX Runtime's environment is set to the same level, so it does not
    /// format what would be dropped. A session built at a severity of its own
    /// (<see cref="IShorokooBackend.CreateSession(ReadOnlyMemory{byte}, ShorokooGraphOptimization, ShorokooLogSeverity, DeviceMemorySettings)"/>)
    /// passes on only what is at least as severe as both; Shorokoo builds its own sessions at
    /// <see cref="ShorokooLogSeverity.Fatal"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not one of <see cref="ShorokooLogSeverity"/>'s values.</exception>
    public static ShorokooLogSeverity Severity
    {
        get => (ShorokooLogSeverity)_severity;
        set
        {
            if (!Enum.IsDefined(value)) throw new ArgumentOutOfRangeException(nameof(value), value, null);
            _severity = (int)value;
            OrtEnvironment.ApplyLogSeverity();
        }
    }

    /// <summary>
    /// Receives each message at least as severe as <see cref="Severity"/>; <c>null</c> drops them
    /// all. <see cref="WriteToStandardError"/> unless set otherwise. It is called on whichever
    /// thread ONNX Runtime logs from, possibly several at once, and an exception it throws is
    /// dropped: none can be raised through ONNX Runtime.
    /// </summary>
    public static Action<OrtLogMessage>? Sink
    {
        get => _sink;
        set => _sink = value;
    }

    /// <summary>The default <see cref="Sink"/>: writes <paramref name="message"/> as one line, with
    /// no colour codes, to the standard error stream.</summary>
    public static void WriteToStandardError(OrtLogMessage message) => Console.Error.WriteLine(message.ToString());

    /// <summary>Whether a message at <paramref name="severity"/> reaches a sink.</summary>
    internal static bool Passes(ShorokooLogSeverity severity)
        => severity >= (_captured?.Severity ?? Severity) && (_captured is not null || _sink is not null);

    /// <summary>The level ONNX Runtime's environment is to log from.</summary>
    internal static OrtLoggingLevel EnvironmentLevel => (OrtLoggingLevel)_severity;

    /// <summary>The logging function ONNX Runtime's environment is made with.</summary>
    internal static void Forward(IntPtr param, OrtLoggingLevel severity, string category, string logId, string codeLocation, string message)
    {
        try
        {
            var level = (ShorokooLogSeverity)severity;
            if (!Passes(level)) return;
            var sink = _captured is { } captured ? captured.Sink : _sink;
            sink?.Invoke(new OrtLogMessage(level, category, logId, codeLocation, message));
        }
        catch (Exception) { }
    }

    /// <summary>Sends the messages ONNX Runtime logs on this thread to <paramref name="sink"/> from
    /// <paramref name="severity"/> on, in place of <see cref="Sink"/> and <see cref="Severity"/>,
    /// until disposed.</summary>
    internal static IDisposable CaptureOnThisThread(Action<OrtLogMessage> sink, ShorokooLogSeverity severity)
        => _captured = new Capture(sink, severity, _captured);

    private sealed class Capture(Action<OrtLogMessage> sink, ShorokooLogSeverity severity, Capture? outer) : IDisposable
    {
        public Action<OrtLogMessage> Sink { get; } = sink;
        public ShorokooLogSeverity Severity { get; } = severity;
        public void Dispose() => _captured = outer;
    }
}
