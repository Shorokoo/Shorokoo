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
    private static bool _initialized;
    private static readonly object _lock = new();

    public static QuickOp? Get(string opCode)
    {
        EnsureInitialized();
        return _ops.TryGetValue(opCode, out var op) ? op : null;
    }

    public static bool Contains(string opCode)
    {
        EnsureInitialized();
        return _ops.ContainsKey(opCode);
    }

    public static IReadOnlyCollection<string> RegisteredOpCodes
    {
        get { EnsureInitialized(); return _ops.Keys; }
    }

    public static void Register(QuickOp op)
    {
        lock (_lock)
        {
            _ops[op.OpCode] = op;
        }
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
