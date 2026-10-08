using Python.Runtime;
using Shorokoo.Core.Backends;

namespace Shorokoo.PythonHost;

/// <summary>
/// The Python warnings a call into a backend's runtime raised, as that runtime's warning hook
/// collected them: a list the .NET side hands the call, into which the hook appends one
/// <c>(category, message, location)</c> per warning shown instead of writing it to the standard
/// error stream. The interpreter's warning machinery is the whole process's, so the list travels
/// with the call — in a context variable on the Python side — and each run's warnings reach that
/// run's <see cref="LogSettings"/> alone.
/// </summary>
internal static class PythonWarnings
{
    /// <summary>What <paramref name="collected"/> holds, as messages from <paramref name="source"/>,
    /// read under the interpreter lock the caller holds.</summary>
    public static RuntimeLogMessage[] Read(PyList collected, string source)
    {
        var count = collected.Length();
        if (count == 0) return [];
        var messages = new RuntimeLogMessage[count];
        for (int i = 0; i < count; i++)
        {
            using var entry = collected[i];
            using var category = entry[0];
            using var message = entry[1];
            using var location = entry[2];
            messages[i] = new RuntimeLogMessage(
                ShorokooLogSeverity.Warning, source, category.As<string>(), location.As<string>(), message.As<string>());
        }
        return messages;
    }

    /// <summary>Delivers <paramref name="messages"/> to <paramref name="log"/>, outside the
    /// interpreter lock, so that a sink that calls into Python itself waits for nothing.</summary>
    public static void Deliver(LogSettings log, RuntimeLogMessage[] messages)
    {
        foreach (var message in messages) log.Deliver(message);
    }
}
