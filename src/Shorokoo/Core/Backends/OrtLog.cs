namespace Shorokoo.Core.Backends;

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
/// ONNX Runtime backends (<c>Shorokoo.LinuxCPU</c>, <c>LinuxGPU</c>, <c>WinCPU</c>, <c>WinGPU</c>),
/// those <see cref="IsolatedBackend.Load"/> loads included.
///
/// <para>Shorokoo makes ONNX Runtime's environment when the first of those backends over a native
/// runtime is built, and makes it to log through <see cref="Sink"/> rather than through ONNX
/// Runtime's own sink, which writes to the standard error stream wrapped in terminal colour codes.
/// Messages the environment logs (a kernel warning about a graph it is given, for instance) and
/// messages a session logs both arrive here. Both settings can be changed at any time and apply to
/// the next message.</para>
///
/// <para>Where the program made ONNX Runtime's environment itself before the first backend was
/// built (<c>OrtEnv.CreateInstanceWithOptions</c>), that environment logs as it was made to and
/// these settings do not reach it.</para>
/// </summary>
public static class OrtLog
{
    private static volatile Action<OrtLogMessage>? _sink = WriteToStandardError;
    private static volatile int _severity = (int)ShorokooLogSeverity.Warning;
    private static readonly object _gate = new();
    private static Action[] _environments = [];

    [ThreadStatic] private static Capture? _captured;

    /// <summary>
    /// The least severe message passed on to <see cref="Sink"/>; <see cref="ShorokooLogSeverity.Warning"/>
    /// unless set otherwise. Every ONNX Runtime environment Shorokoo made is set to the same level, so
    /// it does not format what would be dropped. A session built at a severity of its own
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
            lock (_gate)
            {
                _severity = (int)value;
                foreach (var apply in _environments) apply();
            }
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

    /// <summary>Takes note of an ONNX Runtime environment made to log here: <paramref name="apply"/>
    /// sets its level to <see cref="Severity"/>, and is called now and on every change.</summary>
    internal static void Register(Action apply)
    {
        lock (_gate)
        {
            _environments = [.. _environments, apply];
            apply();
        }
    }

    /// <summary>Passes a message ONNX Runtime logged on to the sink, if it is severe enough. Never
    /// throws: it is called from ONNX Runtime.</summary>
    internal static void Deliver(ShorokooLogSeverity severity, string category, string logId, string codeLocation, string message)
    {
        try
        {
            var captured = _captured;
            if (severity < (captured?.Severity ?? Severity)) return;
            (captured is not null ? captured.Sink : _sink)?.Invoke(new OrtLogMessage(severity, category, logId, codeLocation, message));
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
