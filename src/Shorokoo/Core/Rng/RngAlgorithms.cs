using System;
using System.Collections.Generic;
using Shorokoo;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Rng;

/// <summary>
/// The registry of named RNG algorithms. An algorithm is a named, versioned <b>set of
/// functions</b> — <c>split</c> (index-based key split), <c>uniform</c> and <c>normal</c>
/// (keyed draws) — because both the bit generator <em>and</em> the distribution transforms
/// determine the produced values (two frameworks produce different normals from identical
/// bit streams when their transforms differ). The version is part of the name: improving a
/// transform later is a <em>new algorithm name</em>, never a silent change.
///
/// <para>Each function is an ordinary Shorokoo <see cref="Function"/> built from in-graph
/// integer/float math (see <see cref="RuntimeRng"/>), tagged with
/// <see cref="Function.RngAlgorithm"/> / <see cref="Function.RngFunctionKind"/>. Tagged
/// functions are <b>never inlined</b> by the function inliner and export as ONNX local
/// FunctionProtos carrying the tags in their metadata — so an exported model's randomness
/// is self-contained, deterministic on any runtime, and identifiable.</para>
/// </summary>
internal static class RngAlgorithms
{
    /// <summary>Threefry-2x32 (Random123, 20 rounds) + torch-convention 24-bit uniform + Box–Muller normal.</summary>
    public const string Threefry2x32BoxMullerV1 = "Threefry2x32-BoxMuller.v1";

    /// <summary>Threefry-2x32 with the reduced 13-round bit generator (Random123 <c>threefry2x32x13</c>,
    /// still BigCrush-resistant, ~35% faster) + the same 24-bit uniform + Box–Muller normal. Only the
    /// draw's round count differs from <see cref="Threefry2x32BoxMullerV1"/>; the key tree is shared.</summary>
    public const string Threefry2x32x13BoxMullerV1 = "Threefry2x32-13-BoxMuller.v1";

    /// <summary>The default algorithm for keyed draws.</summary>
    public const string Default = Threefry2x32BoxMullerV1;

    public const string KindSplit = "split";
    public const string KindUniform = "uniform";
    public const string KindNormal = "normal";

    /// <summary>The registry name of a configured <see cref="RngAlgorithm"/>.</summary>
    public static string NameOf(RngAlgorithm algorithm) => algorithm switch
    {
        RngAlgorithm.Threefry2x32 => Threefry2x32BoxMullerV1,
        RngAlgorithm.Threefry2x32Rounds13 => Threefry2x32x13BoxMullerV1,
        _ => throw new NotSupportedException($"Unknown RNG algorithm '{algorithm}'."),
    };

    /// <summary>The configured <see cref="RngAlgorithm"/> for a registry name, or null when the
    /// name is unknown (e.g. a carrier written by a newer version).</summary>
    public static RngAlgorithm? TryFromName(string algorithm) => algorithm switch
    {
        Threefry2x32BoxMullerV1 => RngAlgorithm.Threefry2x32,
        Threefry2x32x13BoxMullerV1 => RngAlgorithm.Threefry2x32Rounds13,
        _ => null,
    };

    // The draw (uniform/normal) bit-generator round count per algorithm. The key tree (split)
    // is deliberately NOT algorithm-dependent — see DrawRounds usage below.
    private static int DrawRounds(string algorithm) => algorithm switch
    {
        Threefry2x32BoxMullerV1 => Threefry2x32.Rounds,          // 20
        Threefry2x32x13BoxMullerV1 => Threefry2x32.Rounds13,     // 13
        _ => throw new NotSupportedException($"Unknown RNG algorithm '{algorithm}'."),
    };

    /// <summary>The draw bit-generator round count of a configured algorithm — used host-side for
    /// parameter-init noise so init draws match the same algorithm the runtime feeds use.</summary>
    public static int DrawRoundsOf(RngAlgorithm algorithm) => DrawRounds(NameOf(algorithm));

    private static readonly object Gate = new();
    private static readonly Dictionary<(string algorithm, string kind), Function> Cache = new();

    /// <summary>The named algorithm's function of the given kind (cached; built on first use).</summary>
    public static Function GetFunction(string algorithm, string kind)
    {
        // The key tree is algorithm-independent: switching the draw algorithm must not re-key
        // any stream, so split (the in-graph key fold) is always the default 20-round Threefry
        // regardless of the configured draw algorithm. Only uniform/normal vary by algorithm.
        if (kind == KindSplit) algorithm = Default;

        lock (Gate)
        {
            if (Cache.TryGetValue((algorithm, kind), out var fn)) return fn;

            int rounds = DrawRounds(algorithm);   // validates the algorithm name
            Delegate body = kind switch
            {
                KindSplit => (Func<Vector<int64>, Scalar<int64>, Vector<int64>>)SplitImpl,
                KindUniform => rounds == Threefry2x32.Rounds13
                    ? (Func<Vector<int64>, Scalar<int64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)Uniform13Impl
                    : (Func<Vector<int64>, Scalar<int64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)UniformImpl,
                KindNormal => rounds == Threefry2x32.Rounds13
                    ? (Func<Vector<int64>, Scalar<int64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)Normal13Impl
                    : (Func<Vector<int64>, Scalar<int64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)NormalImpl,
                _ => throw new NotSupportedException($"Unknown RNG function kind '{kind}'."),
            };

            // Sanitized, stable ONNX-safe name; the pretty algorithm name rides the metadata.
            var tag = algorithm == Threefry2x32x13BoxMullerV1 ? "Threefry2x32_13_BoxMuller_v1" : "Threefry2x32_BoxMuller_v1";
            var name = "ShrkRng_" + tag + "_" + kind;
            var graph = GraphBuilder.BuildFastComputationGraphFromDelegate(body);
            fn = new Function(graph, FunctionType.Function, name, name)
            {
                RngAlgorithm = algorithm,
                RngFunctionKind = kind,
            };
            Cache[(algorithm, kind)] = fn;
            return fn;
        }
    }

    private static Vector<int64> SplitImpl(Vector<int64> key, Scalar<int64> index)
    {
        var (k0, k1) = RuntimeRng.SplitKey(key[0], key[1], index);
        return [k0, k1];
    }

    private static Tensor<float32> UniformImpl(
        Vector<int64> key, Scalar<int64> drawBase, Vector<int64> shape,
        Scalar<float32> low, Scalar<float32> high)
        => RuntimeRng.Uniform(shape, key[0], key[1], drawBase, low, high, Threefry2x32.Rounds);

    private static Tensor<float32> NormalImpl(
        Vector<int64> key, Scalar<int64> drawBase, Vector<int64> shape,
        Scalar<float32> mean, Scalar<float32> scale)
        => RuntimeRng.Normal(shape, key[0], key[1], drawBase, mean, scale, Threefry2x32.Rounds);

    private static Tensor<float32> Uniform13Impl(
        Vector<int64> key, Scalar<int64> drawBase, Vector<int64> shape,
        Scalar<float32> low, Scalar<float32> high)
        => RuntimeRng.Uniform(shape, key[0], key[1], drawBase, low, high, Threefry2x32.Rounds13);

    private static Tensor<float32> Normal13Impl(
        Vector<int64> key, Scalar<int64> drawBase, Vector<int64> shape,
        Scalar<float32> mean, Scalar<float32> scale)
        => RuntimeRng.Normal(shape, key[0], key[1], drawBase, mean, scale, Threefry2x32.Rounds13);
}
