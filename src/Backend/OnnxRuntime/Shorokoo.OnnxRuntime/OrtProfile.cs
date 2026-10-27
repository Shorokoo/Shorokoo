using System.Globalization;
using System.Text.Json;
using Shorokoo.Core.Backends;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// Reads which execution provider ran each node out of an ONNX Runtime profile file.
///
/// <para>ORT writes one <c>cat: "Node"</c> event per node per run, and its <c>args</c> carry the
/// provider that ran it — literally <c>CPUExecutionProvider</c> or <c>CUDAExecutionProvider</c> —
/// beside the node's operator, its index in the execution order, and the bytes it read and
/// produced. Grouping those by provider is the list of nodes a device session left to the host,
/// with what they cost.</para>
/// </summary>
internal static class OrtProfile
{
    private const string NodeCategory = "Node";
    private const string KernelSuffix = "_kernel_time";

    /// <summary>How long a profile another handle holds open is waited for.</summary>
    internal static readonly TimeSpan HeldForAtMost = TimeSpan.FromSeconds(1);

    /// <summary>
    /// The profile at <paramref name="path"/>, opened for reading. A file ORT has just written is
    /// often opened by another process a moment later — a virus scanner or an indexer reading it —
    /// which holds it against a reader for as long as that takes, so an attempt that finds the file
    /// held is made again until <see cref="HeldForAtMost"/> has passed. A file that is not there is
    /// not waited for.
    /// </summary>
    internal static FileStream OpenOnceLetGo(string path)
    {
        var deadline = Environment.TickCount64 + (long)HeldForAtMost.TotalMilliseconds;
        while (true)
        {
            try
            {
                return File.OpenRead(path);
            }
            catch (IOException) when (File.Exists(path) && Environment.TickCount64 < deadline)
            {
                Thread.Sleep(20);
            }
        }
    }

    /// <summary>
    /// The node placement recorded in the profile at <paramref name="path"/>. A profile with no node
    /// events — a session that was never run — gives a placement with no nodes, which is the honest
    /// answer to "what ran": nothing did.
    /// </summary>
    /// <exception cref="IOException">The file cannot be read, or another handle still holds it
    /// <see cref="HeldForAtMost"/> after the first attempt.</exception>
    /// <exception cref="UnauthorizedAccessException">The file cannot be opened.</exception>
    /// <exception cref="JsonException">The file is not JSON.</exception>
    /// <exception cref="InvalidDataException">The file is JSON but not a profile: its root is not an
    /// array of events.</exception>
    internal static NodePlacement Read(string path)
    {
        using var file = OpenOnceLetGo(path);
        using var profile = JsonDocument.Parse(file);
        if (profile.RootElement.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException(
                $"'{path}' is not an ONNX Runtime profile: its root is a JSON {profile.RootElement.ValueKind}, not an array of events.");

        // One node runs once per run, so the file holds as many events per node as there were
        // runs. Keyed by name, first event wins: what is wanted is where each node ran, and
        // that does not change between runs of one session.
        var nodes = new Dictionary<string, NodeExecution>(StringComparer.Ordinal);
        foreach (var entry in profile.RootElement.EnumerateArray())
        {
            if (entry.ValueKind != JsonValueKind.Object) continue;
            if (Text(entry, "cat") != NodeCategory) continue;
            if (Text(entry, "name") is not { } name) continue;
            if (!entry.TryGetProperty("args", out var args) || args.ValueKind != JsonValueKind.Object) continue;
            // Only the kernel-time event carries the placement; the fence events around it
            // name the same node and carry nothing.
            if (Text(args, "provider") is not { } provider) continue;
            if (name.EndsWith(KernelSuffix, StringComparison.Ordinal))
                name = name[..^KernelSuffix.Length];
            if (nodes.ContainsKey(name)) continue;

            nodes[name] = new NodeExecution(
                name,
                Text(args, "op_name") ?? "",
                provider,
                Number(args, "node_index"),
                Number(args, "activation_size"),
                Number(args, "parameter_size"),
                Number(args, "output_size"));
        }

        return new NodePlacement([.. nodes.Values]);
    }

    private static string? Text(JsonElement element, string property)
        => element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    /// <summary>ORT writes the byte counts and the node index as decimal <i>strings</i>, so this
    /// takes either spelling and answers zero for anything else.</summary>
    private static long Number(JsonElement element, string property)
    {
        if (!element.TryGetProperty(property, out var value)) return 0;
        return value.ValueKind switch
        {
            JsonValueKind.Number => value.TryGetInt64(out var number) ? number : 0,
            JsonValueKind.String => long.TryParse(
                value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var text) ? text : 0,
            _ => 0,
        };
    }
}
