using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Shorokoo;

/// <summary>One per-stream runtime override a model carries: the overridden stream's ModelId
/// path and the key that replaces its derived one.</summary>
public sealed class RngStreamOverride
{
    internal RngStreamOverride(IEnumerable<int> modelIdPath, ulong key)
    {
        ModelIdPath = [.. modelIdPath];
        Key = key;
    }

    /// <summary>The overridden stream's absolute ModelId path, as
    /// <see cref="RngConfig.Override"/> addressed it.</summary>
    public IReadOnlyList<int> ModelIdPath { get; }

    /// <summary>The key the override installed — it replaces the fully folded key, so it is
    /// the seed passed to <see cref="RngConfig.Override"/> verbatim.</summary>
    public ulong Key { get; }

    /// <summary>One human-readable line: path and key.</summary>
    public override string ToString()
        => $"[{string.Join(", ", ModelIdPath)}]  key=0x{Key:x16}";
}

/// <summary>
/// A model's bound <b>runtime</b> RNG identity, read back from the <c>RngSeed</c> parameter
/// that <see cref="Graph.ComputationGraph.WithRngConfig"/> wrote (see
/// <see cref="Graph.ComputationGraph.TryGetRngIdentity"/>). It says which bit generator the
/// model draws with, which runtime master key its feeds fold from, and which streams carry an
/// explicit override.
///
/// <para><b>An identity is not a config.</b> Two things a model genuinely does not record, and
/// that this type therefore does not offer:</para>
///
/// <list type="bullet">
/// <item><description><see cref="RngConfig.MasterSeed"/> is not recoverable. The identity stores
/// the <em>derived</em> runtime master key, not the seed it was folded from — so a model can tell
/// you which keys its streams use, never the seed a caller typed.</description></item>
/// <item><description>The <see cref="RngCollection.Params"/> tier is not recorded at all.
/// Initialization randomness is drawn once and baked into the weights, so nothing in a saved model
/// consumes it; re-running initialization under a chosen seed takes an explicit
/// <see cref="RngConfig"/>.</description></item>
/// </list>
///
/// <para>Both are why this is an identity rather than a round-trippable config. What it does
/// support exactly is re-keying a single runtime stream —
/// <see cref="Graph.ComputationGraph.WithRngOverride"/> — without reconstructing the config that
/// produced the model.</para>
/// </summary>
public sealed class RngIdentity
{
    internal RngIdentity(RngAlgorithm algorithm, ulong runMasterKey, IEnumerable<RngStreamOverride> overrides)
    {
        Algorithm = algorithm;
        RunMasterKey = runMasterKey;
        Overrides = [.. overrides];
    }

    /// <summary>The bit generator the model's draws are bound to.</summary>
    public RngAlgorithm Algorithm { get; }

    /// <summary>The runtime-collection master key every non-overridden feed folds from. This is
    /// the derived key, not <see cref="RngConfig.MasterSeed"/>; supply it as
    /// <see cref="RngConfig.RunMasterSeed"/> to reproduce this model's runtime streams under a
    /// fresh config.</summary>
    public ulong RunMasterKey { get; }

    /// <summary>Every runtime stream the model overrides, in the canonical (path-sorted) order the
    /// identity records them; empty when it overrides none.</summary>
    public IReadOnlyList<RngStreamOverride> Overrides { get; }

    /// <summary>The recorded key for an overridden stream, or <c>null</c> when that stream derives
    /// its key from <see cref="RunMasterKey"/> like any other.</summary>
    public ulong? TryGetOverride(IReadOnlyList<int> modelIdPath)
    {
        System.ArgumentNullException.ThrowIfNull(modelIdPath);
        foreach (var o in Overrides)
            if (o.ModelIdPath.SequenceEqual(modelIdPath)) return o.Key;
        return null;
    }

    /// <summary>A config that reproduces this identity's runtime tier exactly — same algorithm,
    /// same runtime master key, same overrides — and leaves the (unrecorded) init tier at its
    /// defaults. The basis for <see cref="Graph.ComputationGraph.WithRngOverride"/>.</summary>
    internal RngConfig ToRuntimeConfig()
    {
        var config = new RngConfig { Algorithm = Algorithm, RunMasterSeed = RunMasterKey };
        foreach (var o in Overrides)
            config = config.Override(RngCollection.Runtime, [.. o.ModelIdPath], o.Key);
        return config;
    }

    /// <summary>The identity as one line for the master key plus one per override.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append(Algorithm).Append("  runtime master key=0x").Append(RunMasterKey.ToString("x16"));
        foreach (var o in Overrides) sb.Append(System.Environment.NewLine).Append("  override  ").Append(o);
        return sb.ToString();
    }
}
