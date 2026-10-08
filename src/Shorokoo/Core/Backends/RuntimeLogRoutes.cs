using System.Collections.Concurrent;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The <see cref="LogSettings"/> a runtime's messages are delivered to, for a runtime whose logging
/// reaches the program through one callback for the whole process, as ONNX Runtime's does: each
/// message carries the id of the logger it went through, and a backend gives each session it builds
/// and each run it makes a logger of its own, named by a <see cref="Route"/> opened here.
///
/// <para>It lives in the core so that every copy of a backend — those
/// <see cref="IsolatedBackend.Load"/> loads included, each with a runtime of its own — opens its
/// routes in the one table, and each message is delivered to the settings of the session or run
/// that emitted it whichever copy that was. Nothing here is configuration: the table holds only
/// the settings of sessions and runs that are alive.</para>
///
/// <para>A runtime logs much of what happens during a run through the session's logger rather than
/// the run's, so a message whose logger is a session's goes to the settings of that session's run
/// in progress on the thread that logged it, where there is one, and to the session's own
/// otherwise. A message whose logger names no live route — one the runtime logs through its
/// process-wide logger — goes to the settings of the session build or run in progress on the thread
/// that logged it (<see cref="Route.Enter()"/>), and where there is none, as on one of the runtime's
/// own worker threads, to <see cref="LogSettings.Default"/>.</para>
/// </summary>
internal static class RuntimeLogRoutes
{
    private static readonly ConcurrentDictionary<string, LogSettings> _routes = new(StringComparer.Ordinal);
    private static long _lastId;

    private static readonly object _gate = new();
    // The live routes that pass on Verbose and Info messages, by severity: what the runtime's
    // process-wide logger must still log for (Floor).
    private static readonly int[] _belowWarning = new int[(int)ShorokooLogSeverity.Warning];
    private static Action<ShorokooLogSeverity>[] _floorWatchers = [];

    // The route whose session is building or running on this thread, and the settings of that
    // build or run.
    [ThreadStatic] private static string? _inProgressRoute;
    [ThreadStatic] private static LogSettings? _inProgress;

    /// <summary>The least severe message a live route, or <see cref="LogSettings.Default"/>, passes
    /// on: what a runtime's process-wide logger is set to log from.</summary>
    internal static ShorokooLogSeverity Floor
    {
        get { lock (_gate) return FloorNow(); }
    }

    private static ShorokooLogSeverity FloorNow()
    {
        for (int severity = 0; severity < _belowWarning.Length; severity++)
            if (_belowWarning[severity] > 0) return (ShorokooLogSeverity)severity;
        return LogSettings.Default.MinimumSeverity;
    }

    /// <summary>Takes note of a runtime's process-wide logger: <paramref name="apply"/> sets its
    /// level, and is called now with <see cref="Floor"/> and again whenever it changes.</summary>
    internal static void WatchFloor(Action<ShorokooLogSeverity> apply)
    {
        lock (_gate)
        {
            _floorWatchers = [.. _floorWatchers, apply];
            apply(FloorNow());
        }
    }

    /// <summary>Opens a route to <paramref name="log"/>, under an id no other live route has, for a
    /// session or a run to name its logger by. The caller disposes it when the session or run is
    /// over; a route never disposed is closed when it is collected.</summary>
    internal static Route Open(LogSettings log)
    {
        ArgumentNullException.ThrowIfNull(log);
        var route = new Route("shorokoo-" + Interlocked.Increment(ref _lastId), log);
        _routes[route.Id] = log;
        Count(log, +1);
        return route;
    }

    private static void Close(Route route)
    {
        if (_routes.TryRemove(route.Id, out _)) Count(route.Log, -1);
    }

    private static void Count(LogSettings log, int delta)
    {
        var severity = log.EffectiveSeverity;
        if (severity >= ShorokooLogSeverity.Warning) return;
        lock (_gate)
        {
            var before = FloorNow();
            _belowWarning[(int)severity] += delta;
            var after = FloorNow();
            if (after != before)
                foreach (var apply in _floorWatchers)
                    apply(after);
        }
    }

    /// <summary>Whether a route of id <paramref name="id"/> is live.</summary>
    internal static bool IsOpen(string id) => _routes.ContainsKey(id);

    /// <summary>The settings a message logged through the logger <paramref name="logId"/> goes to:
    /// those of a live route the id names — a run's logger is named by its session's id and its own,
    /// joined by colons, and the run's comes first — unless this thread is building or running under
    /// that route, when the settings of that build or run; or else the settings in progress on this
    /// thread; or else <see cref="LogSettings.Default"/>.</summary>
    internal static LogSettings Resolve(string? logId)
    {
        if (!string.IsNullOrEmpty(logId))
        {
            if (Named(logId) is { } exact) return exact;
            var colon = logId.LastIndexOf(':');
            while (colon >= 0)
            {
                if (Named(logId[(colon + 1)..]) is { } named) return named;
                logId = logId[..colon];
                if (Named(logId) is { } outer) return outer;
                colon = logId.LastIndexOf(':');
            }
        }
        return _inProgress ?? LogSettings.Default;
    }

    private static LogSettings? Named(string id)
        => !_routes.TryGetValue(id, out var log) ? null
            : _inProgress is { } inProgress && _inProgressRoute == id ? inProgress
            : log;

    /// <summary>Delivers a message a runtime logged through the logger <paramref name="logId"/> to
    /// the settings it belongs to (<see cref="Resolve"/>). Never throws: it is called from inside
    /// the runtime.</summary>
    internal static void Deliver(
        string? logId, ShorokooLogSeverity severity, string source, string category, string location, string text)
    {
        try
        {
            var log = Resolve(logId);
            if (log.Sink is null || severity < log.MinimumSeverity) return;
            log.Deliver(new RuntimeLogMessage(severity, source, category ?? "", location ?? "", text ?? ""));
        }
        catch (Exception) { }
    }

    /// <summary>
    /// A live entry of the table: the id a session's or a run's logger is named by, and the settings
    /// its messages go to. Closed when disposed, or when collected undisposed — a compiled graph its
    /// program dropped is collected without its session ever being disposed — so the table holds
    /// nothing of a session or run that is over.
    /// </summary>
    internal sealed class Route : IDisposable
    {
        private int _closed;

        internal Route(string id, LogSettings log)
        {
            Id = id;
            Log = log;
        }

        /// <summary>The id the runtime's logger is named by.</summary>
        internal string Id { get; }

        /// <summary>Where its messages go.</summary>
        internal LogSettings Log { get; }

        /// <summary>Makes <see cref="Log"/> the settings in progress on this thread until the
        /// returned scope is disposed, for the session it names to build or run under: where a
        /// message the runtime logs on this thread through its process-wide logger goes.</summary>
        internal InProgress Enter() => Enter(Log);

        /// <summary>Makes <paramref name="run"/> the settings in progress on this thread until the
        /// returned scope is disposed, for a run of the session this route names: where a message
        /// the runtime logs on this thread through the session's logger or its process-wide one
        /// goes.</summary>
        internal InProgress Enter(LogSettings run)
        {
            var outer = new InProgress(_inProgressRoute, _inProgress);
            _inProgressRoute = Id;
            _inProgress = run;
            return outer;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _closed, 1) == 0) Close(this);
            GC.SuppressFinalize(this);
        }

        ~Route()
        {
            if (Interlocked.Exchange(ref _closed, 1) == 0) Close(this);
        }
    }

    /// <summary>The scope <see cref="Route.Enter(LogSettings)"/> opens: restores what was in
    /// progress on the thread before it.</summary>
    internal readonly struct InProgress(string? outerRoute, LogSettings? outer) : IDisposable
    {
        public void Dispose()
        {
            _inProgressRoute = outerRoute;
            _inProgress = outer;
        }
    }
}
