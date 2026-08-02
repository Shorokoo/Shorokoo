using System.Collections.Generic;
using Shorokoo.Core.Rng;

namespace Shorokoo.Tests;

/// <summary>
/// Host-side RNG key oracle — <b>test-only</b>.
///
/// <para>Production code performs no host-side Threefry (#136): the key tree is computed
/// exclusively in-graph by the algorithm's <c>SHRK_RNG_SPLIT</c> chain, and any host consumer
/// that needs a concrete key resolves it by <em>executing</em> that derivation
/// (<c>RngKeyResolver</c>). These helpers reimplement the fold independently, on the host, so
/// tests can assert the in-graph derivation against an oracle that does not share its
/// implementation — which is exactly what makes the assertions meaningful.</para>
///
/// <para>They deliberately mirror the old <c>RngConfig.FoldInitKey</c>/<c>FoldRunKey</c>
/// signatures so existing assertions read unchanged; the config now supplies only a derivation
/// <em>spec</em> (root key + path still to fold), and the oracle folds it here.</para>
/// </summary>
internal static class RngTestOracle
{
    /// <summary>One Threefry key-tree fold step: child = Bijection(counter: (index, 0), key).
    /// Bit-identical to the in-graph <c>SHRK_RNG_SPLIT</c> (whose counter word is the index's
    /// low 32 bits — the <c>uint</c> cast here matches its <c>Mask32</c>).</summary>
    public static (uint k0, uint k1) FoldKey((uint k0, uint k1) key, long index)
        => Threefry2x32.Bijection(unchecked((uint)index), 0u, key.k0, key.k1);

    private static (uint k0, uint k1) Fold(
        ((uint k0, uint k1) root, IReadOnlyList<int> foldPath) spec)
    {
        var key = spec.root;
        foreach (var v in spec.foldPath) key = FoldKey(key, v);
        return key;
    }

    /// <summary>A trainable parameter's init stream key (oracle for the in-graph derivation).</summary>
    public static (uint k0, uint k1) FoldInitKey(this RngConfig config, IReadOnlyList<int> modelIdVals)
        => Fold(config.InitKeySpec(modelIdVals));

    /// <summary>A runtime feed's stream key (oracle for the in-graph derivation).</summary>
    public static (uint k0, uint k1) FoldRunKey(this RngConfig config, IReadOnlyList<int> modelIdVals)
        => Fold(config.RunKeySpec(modelIdVals));

    /// <summary>A runtime feed's stream key under an encoded identity (oracle).</summary>
    public static (uint k0, uint k1) FoldRunKey(this RngRuntimeIdentity identity, IReadOnlyList<int> path)
        => Fold(identity.RunKeySpec(path));
}
