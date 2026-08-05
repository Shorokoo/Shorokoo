using System.Reflection;

namespace Shorokoo.Core.Inference;

/// <summary>
/// Central registry mapping an ONNX op code to its <see cref="QuickOp"/> implementation.
/// Concrete op implementations under <c>Ops/</c> are auto-discovered via reflection on first
/// access.
/// </summary>
internal static class OpRegistry
{
    private static readonly Dictionary<string, QuickOp> _ops = new(StringComparer.Ordinal);
    private static volatile bool _initialized;
    private static readonly object _lock = new();

    /// <summary>
    /// Thread-scoped handler overrides installed by <see cref="Override"/>. Consulted ahead of
    /// the process-wide table so a caller can swap an op's implementation — for fault injection —
    /// without other threads observing the swap.
    /// </summary>
    [ThreadStatic]
    private static Dictionary<string, QuickOp>? _overrides;

    public static QuickOp? Get(string opCode)
    {
        if (_overrides is { } o && o.TryGetValue(opCode, out var overridden))
            return overridden;
        EnsureInitialized();
        return _ops.TryGetValue(opCode, out var op) ? op : null;
    }

    /// <summary>
    /// Replaces the handlers for <paramref name="ops"/> on the calling thread only, until the
    /// returned scope is disposed. Overrides are invisible to other threads, so concurrent
    /// callers keep seeing the real implementations — and, unlike mutating the shared table,
    /// this cannot race with the unsynchronized reads in <see cref="Get"/>.
    /// </summary>
    public static IDisposable Override(params QuickOp[] ops) => new OverrideScope(ops);

    private sealed class OverrideScope : IDisposable
    {
        private readonly Dictionary<string, QuickOp>? _previous;

        internal OverrideScope(QuickOp[] ops)
        {
            _previous = _overrides;
            var next = _previous is null
                ? new Dictionary<string, QuickOp>(StringComparer.Ordinal)
                : new Dictionary<string, QuickOp>(_previous, StringComparer.Ordinal);
            foreach (var op in ops)
                next[op.OpCode] = op;
            _overrides = next;
        }

        public void Dispose() => _overrides = _previous;
    }

    private static void EnsureInitialized()
    {
        if (_initialized) return;
        lock (_lock)
        {
            if (_initialized) return;
            DiscoverOps();
            _initialized = true;
        }
    }

    private static void DiscoverOps()
    {
        var asm = typeof(OpRegistry).Assembly;
        Type[] types;
        try { types = asm.GetTypes(); }
        catch (System.Reflection.ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t is not null).ToArray()!; }

        foreach (var type in types)
        {
            if (type is null || type.IsAbstract || !typeof(QuickOp).IsAssignableFrom(type)) continue;
            if (type.GetConstructor(Type.EmptyTypes) is null) continue;
            try
            {
                var op = (QuickOp)Activator.CreateInstance(type)!;
                _ops[op.OpCode] = op;
            }
            catch { /* skip ops that fail to construct */ }
        }
    }
}
