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
    /// <summary>The batched key-tree fold: one pass folds a whole level (M parent keys x M
    /// counters). Same math as <see cref="KindSplit"/> — see <c>RuntimeRng.BatchSplitKeys</c> —
    /// and, like it, algorithm-independent.</summary>
    public const string KindSplitBatch = "splitBatch";
    public const string KindUniform = "uniform";
    public const string KindNormal = "normal";
    public const string KindBits = "bits";

    /// <summary>The uint output dtypes a raw-bits draw supports.</summary>
    public static bool IsSupportedBitsDtype(DType dtype) =>
        dtype == DType.UInt8 || dtype == DType.UInt16 || dtype == DType.UInt32 || dtype == DType.UInt64;

    /// <summary>The registry name of a configured <see cref="RngAlgorithm"/>.</summary>
    public static string NameOf(RngAlgorithm algorithm) => algorithm switch
    {
        RngAlgorithm.Threefry2x32 => Threefry2x32BoxMullerV1,
        RngAlgorithm.Threefry2x32Rounds13 => Threefry2x32x13BoxMullerV1,
        _ => throw new NotSupportedException($"Unknown RNG algorithm '{algorithm}'."),
    };

    // The draw (uniform/normal) bit-generator round count per algorithm. The key tree (split)
    // is deliberately NOT algorithm-dependent — see DrawRounds usage below.
    private static int DrawRounds(string algorithm) => algorithm switch
    {
        Threefry2x32BoxMullerV1 => Threefry2x32.Rounds,          // 20
        Threefry2x32x13BoxMullerV1 => Threefry2x32.Rounds13,     // 13
        _ => throw new NotSupportedException($"Unknown RNG algorithm '{algorithm}'."),
    };

    private static readonly object Gate = new();
    private static readonly Dictionary<(string algorithm, string kind, DType? bitsDtype), Function> Cache = new();

    /// <summary>
    /// The named algorithm's function of the given kind (cached; built on first use). For
    /// <see cref="KindBits"/> the output uint width is part of the function's identity, so
    /// <paramref name="bitsDtype"/> is required (and must be a supported uint width); it must be
    /// null for every other kind.
    /// </summary>
    public static Function GetFunction(string algorithm, string kind, DType? bitsDtype = null)
    {
        // Validate the name before any remap: an unknown algorithm must fail loudly for every
        // kind — the split remap below must never launder an unrecognized name into Default.
        int rounds = DrawRounds(algorithm);

        if (kind == KindBits)
        {
            if (bitsDtype is null || !IsSupportedBitsDtype(bitsDtype))
                throw new NotSupportedException(
                    $"The '{KindBits}' RNG function requires a uint output dtype (U8/U16/U32/U64); " +
                    $"got '{bitsDtype?.ToString() ?? "none"}'.");
        }
        else if (bitsDtype is not null)
            throw new ArgumentException($"bitsDtype applies only to the '{KindBits}' kind, not '{kind}'.", nameof(bitsDtype));

        // The key tree is algorithm-independent: switching the draw algorithm must not re-key
        // any stream, so split (the in-graph key fold) is always the default 20-round Threefry
        // regardless of the configured draw algorithm. Only the draws vary by algorithm.
        if (kind == KindSplit || kind == KindSplitBatch) algorithm = Default;

        lock (Gate)
        {
            if (Cache.TryGetValue((algorithm, kind, bitsDtype), out var fn)) return fn;

            bool r13 = rounds == Threefry2x32.Rounds13;
            Delegate body = kind switch
            {
                KindSplit => (Func<Scalar<uint64>, Scalar<uint64>, Scalar<uint64>>)SplitImpl,
                KindSplitBatch => (Func<Vector<uint64>, Vector<uint64>, Vector<uint64>>)BatchSplitImpl,
                KindUniform => r13
                    ? (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)Uniform13Impl
                    : (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)UniformImpl,
                KindNormal => r13
                    ? (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)Normal13Impl
                    : (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)NormalImpl,
                KindBits => BitsBody(r13, bitsDtype!),
                _ => throw new NotSupportedException($"Unknown RNG function kind '{kind}'."),
            };

            // Sanitized, stable ONNX-safe name; the pretty algorithm name rides the metadata.
            var tag = algorithm == Threefry2x32x13BoxMullerV1 ? "Threefry2x32_13_BoxMuller_v1" : "Threefry2x32_BoxMuller_v1";
            var suffix = kind == KindBits ? "_" + bitsDtype!.ToString() : "";
            var name = "ShrkRng_" + tag + "_" + kind + suffix;
            var graph = GraphBuilder.BuildInternalComputationGraphFromDelegate(body);
            fn = new Function(graph, FunctionType.Function, name, name)
            {
                RngAlgorithm = algorithm,
                RngFunctionKind = kind,
            };
            Cache[(algorithm, kind, bitsDtype)] = fn;
            return fn;
        }
    }

    /// <summary>The bits draw delegate for a width and round count (a non-capturing static method,
    /// as <see cref="GraphBuilder.BuildInternalComputationGraphFromDelegate"/> requires).</summary>
    private static Delegate BitsBody(bool r13, DType dtype)
    {
        if (dtype == DType.UInt8)  return r13 ? (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Tensor<uint8>>)BitsU8Impl13   : BitsU8Impl;
        if (dtype == DType.UInt16) return r13 ? (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Tensor<uint16>>)BitsU16Impl13 : BitsU16Impl;
        if (dtype == DType.UInt32) return r13 ? (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Tensor<uint32>>)BitsU32Impl13 : BitsU32Impl;
        return r13 ? (Func<Scalar<uint64>, Scalar<uint64>, Vector<int64>, Tensor<uint64>>)BitsU64Impl13 : BitsU64Impl;
    }

    /// <summary>One pass folds a whole key-tree level; see <see cref="KindSplitBatch"/>.</summary>
    private static Vector<uint64> BatchSplitImpl(Vector<uint64> keys, Vector<uint64> indices)
        => RuntimeRng.BatchSplitKeys(keys, indices).Vec();

    private static Scalar<uint64> SplitImpl(Scalar<uint64> key, Scalar<uint64> index)
        => RuntimeRng.SplitKey(key, index);

    private static Tensor<float32> UniformImpl(
        Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape,
        Scalar<float32> low, Scalar<float32> high)
        => RuntimeRng.Uniform(shape, key, substreamIndex, low, high, Threefry2x32.Rounds);

    private static Tensor<float32> NormalImpl(
        Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape,
        Scalar<float32> mean, Scalar<float32> scale)
        => RuntimeRng.Normal(shape, key, substreamIndex, mean, scale, Threefry2x32.Rounds);

    private static Tensor<float32> Uniform13Impl(
        Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape,
        Scalar<float32> low, Scalar<float32> high)
        => RuntimeRng.Uniform(shape, key, substreamIndex, low, high, Threefry2x32.Rounds13);

    private static Tensor<float32> Normal13Impl(
        Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape,
        Scalar<float32> mean, Scalar<float32> scale)
        => RuntimeRng.Normal(shape, key, substreamIndex, mean, scale, Threefry2x32.Rounds13);

    private static Tensor<uint8>  BitsU8Impl  (Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU8(shape, key, substreamIndex, Threefry2x32.Rounds);
    private static Tensor<uint16> BitsU16Impl (Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU16(shape, key, substreamIndex, Threefry2x32.Rounds);
    private static Tensor<uint32> BitsU32Impl (Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU32(shape, key, substreamIndex, Threefry2x32.Rounds);
    private static Tensor<uint64> BitsU64Impl (Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU64(shape, key, substreamIndex, Threefry2x32.Rounds);
    private static Tensor<uint8>  BitsU8Impl13 (Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU8(shape, key, substreamIndex, Threefry2x32.Rounds13);
    private static Tensor<uint16> BitsU16Impl13(Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU16(shape, key, substreamIndex, Threefry2x32.Rounds13);
    private static Tensor<uint32> BitsU32Impl13(Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU32(shape, key, substreamIndex, Threefry2x32.Rounds13);
    private static Tensor<uint64> BitsU64Impl13(Scalar<uint64> key, Scalar<uint64> substreamIndex, Vector<int64> shape) => RuntimeRng.BitsU64(shape, key, substreamIndex, Threefry2x32.Rounds13);
}
