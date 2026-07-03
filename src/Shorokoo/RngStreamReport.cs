using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Shorokoo;

/// <summary>The kind of consumer behind an RNG stream.</summary>
public enum RngStreamKind
{
    /// <summary>A trainable/state parameter's initialization noise (drawn once at materialization).</summary>
    ParamInit,
    /// <summary>A runtime uniform feed (drawn every execution).</summary>
    UniformFeed,
    /// <summary>A runtime normal feed (drawn every execution).</summary>
    NormalFeed,
}

/// <summary>One RNG stream of a concrete architecture: who draws, from where in the ModelId
/// tree, and (when a config is supplied) with which resolved key.</summary>
public sealed class RngStreamInfo
{
    /// <summary>Which seed collection the stream folds from (init vs runtime sub-master).</summary>
    public required RngCollection Collection { get; init; }

    /// <summary>The consumer's absolute ModelId path; <c>-1</c> marks a loop-iteration slot
    /// (one stream per iteration, split at runtime).</summary>
    public required IReadOnlyList<int> ModelIdPath { get; init; }

    /// <summary>What kind of consumer draws from the stream.</summary>
    public required RngStreamKind Kind { get; init; }

    /// <summary>The parameter's identifier (e.g. <c>Linear#0.weight</c>) when the consumer is
    /// a parameter; feeds have no source-level name — that is exactly the gap
    /// <c>Rng.Pin</c> fills (see the pin-skeleton emission).</summary>
    public string? Name { get; init; }

    /// <summary>The parameter's shape when known (feeds draw at runtime-computed shapes).</summary>
    public IReadOnlyList<long>? Shape { get; init; }

    /// <summary>The stream key ([k0, k1] 32-bit words) resolved under the supplied config;
    /// <c>null</c> when no config was supplied. For a path with <c>-1</c> slots this is the
    /// PREFIX key before the first iteration slot (per-iteration keys only exist at runtime).</summary>
    public IReadOnlyList<long>? KeyWords { get; init; }

    /// <summary>Whether <see cref="KeyWords"/> is a prefix key (path contains loop-iteration slots).</summary>
    public bool KeyIsPrefix => KeyWords is not null && ModelIdPath.Contains(-1);

    /// <summary>One human-readable line: collection, path, kind, name/shape, key.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append(Collection == RngCollection.Params ? "params " : "runtime");
        sb.Append("  [").Append(string.Join(", ", ModelIdPath)).Append(']');
        sb.Append("  ").Append(Kind switch
        {
            RngStreamKind.ParamInit => "init",
            RngStreamKind.UniformFeed => "uniform feed",
            _ => "normal feed",
        });
        if (Name is not null) sb.Append("  ").Append(Name);
        if (Shape is not null) sb.Append("  [").Append(string.Join(", ", Shape)).Append(']');
        if (KeyWords is not null)
        {
            sb.Append("  key=0x").Append(((uint)KeyWords[1]).ToString("x8"))
              .Append(((uint)KeyWords[0]).ToString("x8"));
            if (KeyIsPrefix) sb.Append(" (prefix)");
        }
        return sb.ToString();
    }
}

/// <summary>
/// The bind-time inventory of every RNG stream in a concrete architecture — the tool-side
/// half of the pinning workflow: it can describe every slot (ModelId path, consumer kind,
/// parameter name, shape, resolved key), and it can emit a sparse <c>Rng.Pin</c> skeleton
/// with the one thing no tool can know — the source-level variable name — left as a
/// placeholder for the author (see Documentation/rng-pinning.md).
/// </summary>
public sealed class RngStreamReport
{
    /// <summary>All streams, ordered by collection then ModelId path.</summary>
    public IReadOnlyList<RngStreamInfo> Streams { get; }

    internal RngStreamReport(IEnumerable<RngStreamInfo> streams)
    {
        Streams = streams
            .OrderBy(s => s.Collection)
            .ThenBy(s => s.ModelIdPath, ModelIdPathComparer.Instance)
            .ToArray();
    }

    /// <summary>
    /// Emits the sparse-form <c>Rng.Pin</c> skeleton for this architecture's streams: one
    /// entry per stream at its current id path (loop-iteration <c>-1</c> slots elided, per
    /// the sparse-form contract), with a descriptive comment and a <c>?</c> placeholder
    /// where the author must supply the captured variable. Deliberately non-compiling until
    /// every <c>?</c> is filled — a guessed pin is worse than none.
    /// </summary>
    public string EmitPinSkeleton()
    {
        var sb = new StringBuilder("Rng.Pin(");
        bool first = true;
        foreach (var s in Streams)
        {
            sb.Append(first ? "\n" : ",\n");
            first = false;

            var path = s.ModelIdPath.Where(v => v != -1);
            var desc = s.Kind switch
            {
                RngStreamKind.ParamInit => s.Name ?? "param",
                RngStreamKind.UniformFeed => "uniform feed",
                _ => "normal feed",
            };
            if (s.Kind == RngStreamKind.ParamInit && s.Shape is not null)
                desc += $"  [{string.Join(", ", s.Shape)}]";
            if (s.ModelIdPath.Contains(-1))
                desc += " (loop body)";

            sb.Append("    ([").Append(string.Join(", ", path)).Append("], /* ")
              .Append(desc).Append(" */ ?)");
        }
        sb.Append(");");
        return sb.ToString();
    }

    /// <summary>The report as one line per stream.</summary>
    public override string ToString()
        => string.Join(Environment.NewLine, Streams);

    private sealed class ModelIdPathComparer : IComparer<IReadOnlyList<int>>
    {
        public static readonly ModelIdPathComparer Instance = new();
        public int Compare(IReadOnlyList<int>? a, IReadOnlyList<int>? b)
        {
            if (a is null || b is null) return (a is null ? 0 : 1) - (b is null ? 0 : 1);
            for (int i = 0; i < Math.Min(a.Count, b.Count); i++)
                if (a[i] != b[i]) return a[i].CompareTo(b[i]);
            return a.Count.CompareTo(b.Count);
        }
    }
}
