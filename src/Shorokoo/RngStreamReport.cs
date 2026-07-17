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
    /// <c>null</c> when no config was supplied — or when <see cref="ModelIdPath"/> is an
    /// in-loop feed SITE (a <c>-1</c> iteration slot present): its per-iteration keys derive
    /// at runtime from the iteration index (iteration <c>i</c>'s key is the runtime master
    /// folded along the path with <c>i</c> in the slot), so no single key describes the row.</summary>
    public IReadOnlyList<long>? KeyWords { get; init; }

    /// <summary>
    /// The stream's SITE id (the ModelId with <c>-1</c> iteration placeholders) when
    /// <see cref="ModelIdPath"/> is a realized per-iteration stream of an in-loop consumer —
    /// an in-loop parameter's iteration copy; null when the path is its own site (parameters
    /// outside loops, and every feed row: a loop feed is reported once, as its site, since
    /// its per-iteration streams are runtime-derived rather than enumerated). Pinning
    /// addresses sites, so the pin skeleton groups by this.
    /// </summary>
    public IReadOnlyList<int>? SitePath { get; init; }

    /// <summary>
    /// True for a framework-injected consumer (the <c>RngExecutionCounter</c> drawBase state):
    /// listed for inventory completeness, but excluded from the pin skeleton — no source-level
    /// variable exists to pin it with, and the framework appends it after id assignment, so it
    /// never needs freezing.
    /// </summary>
    public bool FrameworkOwned { get; init; }

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
    /// Emits the sparse-form <c>Rng.Pin</c> skeleton, grouped by pinning <b>scope</b> — one
    /// <c>Rng.Pin(...)</c> per scope (the module body, and each loop body), matching how pins
    /// are scoped: a module-level pin at the end of <c>Inline</c>, and a loop-body pin written
    /// inside that loop. Each entry names a consumer's LOCAL slot in its scope with a
    /// descriptive comment and a <c>?</c> placeholder for the captured variable the author must
    /// supply. Deliberately non-compiling until every <c>?</c> is filled — a guessed pin is
    /// worse than none.
    /// </summary>
    public string EmitPinSkeleton()
    {
        // Group streams by scope (the leading ModelId elements through the last loop-iteration
        // -1 slot). Within a scope the pinnable unit is a consumer's LOCAL slot: the element
        // after that last -1, or the top-level slot when the stream is not in a loop. Multiple
        // params under one consumer collapse to that consumer's single slot.
        var byScope = new List<(IReadOnlyList<int> scope, SortedDictionary<int, RngStreamInfo> bySlot)>();
        foreach (var s in Streams)
        {
            if (s.FrameworkOwned) continue;   // not an author-nameable consumer — see FrameworkOwned
            var path = s.SitePath ?? s.ModelIdPath;
            if (path.Count == 0) continue;   // defensive: a consumer always carries ≥1 id element
            int lastNeg = -1;
            for (int i = 0; i < path.Count; i++) if (path[i] == -1) lastNeg = i;
            int localSlot = lastNeg + 1 < path.Count ? path[lastNeg + 1] : path[0];
            var scope = lastNeg >= 0 ? path.Take(lastNeg + 1).ToArray() : System.Array.Empty<int>();

            var entry = byScope.FirstOrDefault(e => e.scope.SequenceEqual(scope));
            if (entry.bySlot is null) { entry = (scope, new SortedDictionary<int, RngStreamInfo>()); byScope.Add(entry); }
            if (!entry.bySlot.ContainsKey(localSlot)) entry.bySlot[localSlot] = s;
        }

        var sb = new StringBuilder();
        bool firstScope = true;
        foreach (var (scope, bySlot) in byScope.OrderBy(e => e.scope.Count).ThenBy(e => string.Join(",", e.scope)))
        {
            if (!firstScope) sb.Append("\n\n");
            firstScope = false;
            sb.Append(scope.Count == 0
                ? "// at the end of Inline:\n"
                : $"// inside the loop body at ModelId path [{string.Join(", ", scope)}]:\n");
            sb.Append("Rng.Pin(");
            bool firstItem = true;
            foreach (var (slot, s) in bySlot)
            {
                sb.Append(firstItem ? "\n" : ",\n");
                firstItem = false;
                sb.Append("    ([").Append(slot).Append("], /* ").Append(DescribeStream(s)).Append(" */ ?)");
            }
            sb.Append(");");
        }
        return sb.ToString();
    }

    private static string DescribeStream(RngStreamInfo s)
    {
        var desc = s.Kind switch
        {
            RngStreamKind.ParamInit => s.Name ?? "param",
            RngStreamKind.UniformFeed => "uniform feed",
            _ => "normal feed",
        };
        if (s.Kind == RngStreamKind.ParamInit && s.Shape is not null)
            desc += $"  [{string.Join(", ", s.Shape)}]";
        return desc;
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
