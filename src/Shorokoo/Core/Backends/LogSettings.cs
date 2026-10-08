namespace Shorokoo.Core.Backends;

/// <summary>
/// Where the messages a backend's runtime emits go, and from which severity on: what
/// <see cref="RunSettings.Log"/> carries, for one run or as a context's default.
///
/// <para>A <see cref="Shorokoo.Runtime.ComputeContext"/>'s own <see cref="RunSettings"/> carries
/// the settings its sessions are built under — what a runtime says while it builds a session goes
/// there — and those its runs use when a call names none; a
/// <see cref="Shorokoo.Runtime.CompiledGraph"/> run given a <see cref="RunSettings"/> of its own
/// sends what that run emits to that run's settings alone. Like the other settings it is a record,
/// so a variation is a <c>with</c> expression rather than a mutation something else can see:</para>
/// <code>
/// using Shorokoo.Core.Backends;
/// using Shorokoo.Runtime;
///
/// var ctx = new ComputeContext(backend)
/// {
///     RunSettings = new RunSettings { Log = new LogSettings { MinimumSeverity = ShorokooLogSeverity.Error } },
/// };
/// var compiled = ctx.Compile(graph);
/// compiled.Execute([x], ctx.RunSettings with
/// {
///     Log = new LogSettings { MinimumSeverity = ShorokooLogSeverity.Verbose, Sink = m => logger.Log(m.ToString()) },
/// });
/// </code>
/// </summary>
public sealed record LogSettings
{
    /// <summary>What a run gets when nothing names other settings: messages of
    /// <see cref="ShorokooLogSeverity.Warning"/> and above, written as plain lines to the standard
    /// error stream.</summary>
    public static LogSettings Default { get; } = new();

    /// <summary>Settings that drop every message: no <see cref="Sink"/>.</summary>
    public static LogSettings None { get; } = new() { Sink = null };

    private readonly ShorokooLogSeverity _minimumSeverity = ShorokooLogSeverity.Warning;

    /// <summary>The least severe message passed on to <see cref="Sink"/>;
    /// <see cref="ShorokooLogSeverity.Warning"/> unless set otherwise. A backend asks its runtime
    /// for nothing less severe, so what would be dropped is not formatted either.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not one of <see cref="ShorokooLogSeverity"/>'s values.</exception>
    public ShorokooLogSeverity MinimumSeverity
    {
        get => _minimumSeverity;
        init
        {
            if (!Enum.IsDefined(value)) throw new ArgumentOutOfRangeException(nameof(value), value, null);
            _minimumSeverity = value;
        }
    }

    /// <summary>
    /// Receives each message at least as severe as <see cref="MinimumSeverity"/>;
    /// <c>null</c> drops them all. <see cref="WriteToStandardError"/> unless set otherwise.
    ///
    /// <para>It is called on whichever thread the runtime emits from — the one that builds or runs
    /// the session, or one of the runtime's own workers — possibly on several at once, so it must
    /// be thread-safe. An exception it throws is dropped: none can be raised through a runtime's
    /// logging.</para>
    /// </summary>
    public Action<RuntimeLogMessage>? Sink { get; init; } = WriteToStandardError;

    /// <summary>The default <see cref="Sink"/>: writes <paramref name="message"/> as one line, with
    /// no colour codes, to the standard error stream.</summary>
    public static void WriteToStandardError(RuntimeLogMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        Console.Error.WriteLine(message.ToString());
    }

    /// <summary>The least severe message these settings pass on: <see cref="MinimumSeverity"/>, or
    /// <see cref="ShorokooLogSeverity.Fatal"/> where there is no sink — the least a runtime can be
    /// asked for.</summary>
    internal ShorokooLogSeverity EffectiveSeverity => Sink is null ? ShorokooLogSeverity.Fatal : MinimumSeverity;

    /// <summary>Passes <paramref name="message"/> on to the sink if it is severe enough. Never
    /// throws: it is called from inside a runtime.</summary>
    internal void Deliver(RuntimeLogMessage message)
    {
        try
        {
            if (Sink is { } sink && message.Severity >= MinimumSeverity) sink(message);
        }
        catch (Exception) { }
    }
}
