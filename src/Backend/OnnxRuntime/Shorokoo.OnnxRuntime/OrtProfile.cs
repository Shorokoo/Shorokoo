using System.Globalization;
using System.Text.Json;
using Shorokoo.Core.Inference.Abstractions;

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

    /// <summary>
    /// The node placement recorded in the profile at <paramref name="path"/>, or <c>null</c> when
    /// the file cannot be read or is not a profile. A profile with no node events — a session that
    /// was never run — gives a placement with no nodes rather than <c>null</c>, which is the
    /// honest answer to "what ran": nothing did.
    /// </summary>
    internal static NodePlacement? Read(string path)
    {
        try
        {
            using var file = File.OpenRead(path);
            using var profile = JsonDocument.Parse(file);
            if (profile.RootElement.ValueKind != JsonValueKind.Array) return null;

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
        // As broadly as the rest of this backend's probes: a trace that cannot be read costs the
        // diagnostic and nothing else, and the runs it was recorded from have already happened.
        catch (Exception) { return null; }
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
