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

    /// <summary>The default algorithm for keyed draws.</summary>
    public const string Default = Threefry2x32BoxMullerV1;

    public const string KindSplit = "split";
    public const string KindUniform = "uniform";
    public const string KindNormal = "normal";

    private static readonly object Gate = new();
    private static readonly Dictionary<(string algorithm, string kind), Function> Cache = new();

    /// <summary>The named algorithm's function of the given kind (cached; built on first use).</summary>
    public static Function GetFunction(string algorithm, string kind)
    {
        lock (Gate)
        {
            if (Cache.TryGetValue((algorithm, kind), out var fn)) return fn;
            if (algorithm != Threefry2x32BoxMullerV1)
                throw new NotSupportedException($"Unknown RNG algorithm '{algorithm}'.");

            Delegate body = kind switch
            {
                KindSplit => (Func<Vector<int64>, Scalar<int64>, Vector<int64>>)SplitImpl,
                KindUniform => (Func<Vector<int64>, Scalar<int64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)UniformImpl,
                KindNormal => (Func<Vector<int64>, Scalar<int64>, Vector<int64>, Scalar<float32>, Scalar<float32>, Tensor<float32>>)NormalImpl,
                _ => throw new NotSupportedException($"Unknown RNG function kind '{kind}'."),
            };

            // Sanitized, stable ONNX-safe name; the pretty algorithm name rides the metadata.
            var name = "ShrkRng_Threefry2x32_BoxMuller_v1_" + kind;
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
        => RuntimeRng.Uniform(shape, key[0], key[1], drawBase, low, high);

    private static Tensor<float32> NormalImpl(
        Vector<int64> key, Scalar<int64> drawBase, Vector<int64> shape,
        Scalar<float32> mean, Scalar<float32> scale)
        => RuntimeRng.Normal(shape, key[0], key[1], drawBase, mean, scale);
}
