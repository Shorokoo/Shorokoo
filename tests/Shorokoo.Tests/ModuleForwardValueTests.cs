using System;
using System.Linq;
using Shorokoo.Graph;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;
using Shorokoo.Tests.Utils;
using static Shorokoo.Globals;

namespace Shorokoo.Tests;

// ---------------------------------------------------------------------------
// Forward-value coverage for the Shorokoo.Modules NN library.
//
// These replace the old "closed-form" self-checking modules (e.g.
// NNLinearMatchesManualMatMul / NNConv2dMatchesStaticConv), which validated a
// layer by RE-RUNNING the same math by hand and asserting the two agree — a
// tautology that still passes if the layer's math is wrong, as long as the hand
// recomputation is wrong the same way.
//
// Here the layer is materialized from a fixed seed, run forward, and its output
// compared *in-graph* to a frozen reference baked as a constant tensor — so the
// module returns Scalar<bit> and runs through AutoTest.AdvancedTestGraph, which
// exercises it across ONNX save/load, C# codegen, and multiple engines (the
// coverage the hand-recompute modules had). Each reference carries a provenance
// comment:
//   // REFERENCE: PyTorch — the equivalent PyTorch op on the same weights+input
//                           (tests/pytorch-reference/); an external source of truth.
//   // REFERENCE: golden  — self-generated: Shorokoo's own forward output, frozen.
//                           Catches regressions; upgraded to PyTorch over time.
//
// Outputs of <=30 values are compared elementwise. Larger outputs are collapsed
// in-graph to 19 numbers by a single MatMul against a baked [19, n] projection
// matrix W, where W[c, i] = w(i) when i mod 19 == c else 0 — i.e.
//     collapsed[c] = Σ_i W[c, i]·y[i]
// a fixed pseudo-random *linear* sketch: order- and value-sensitive (a wrong
// value or a swap moves a bucket) yet fp-stable (a rounding error ε in y[i] moves
// the result by only ε·w(i), |w| < 0.5). The weight w(i) is an integer hash of the
// POSITION i (not the value), so it is bit-identical across machines. The C# Collapse
// below is the same formula, used only to emit the frozen constants.
// ---------------------------------------------------------------------------

// Forward passes, shared between the raw fixtures (used to generate references) and
// the self-checking fixtures (used by the tests) so the two never drift.
file static class Forwards
{
    public static Tensor<float32> Linear(Tensor<float32> x)
        => Shorokoo.Modules.Layers.Linear.Call(Scalar(4L), Scalar(true), x);

    public static Tensor<float32> Conv2d(Tensor<float32> x)
        => Shorokoo.Modules.Layers.Conv2d
            .Model(Scalar(3L), Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true))
            .Call(x);
}

/// <summary>Raw forward output of Linear (for reference generation only).</summary>
[Module]
public partial class ParityLinearRaw
{
    public static Tensor<float32> Inline(Tensor<float32> x) => Forwards.Linear(x);
}

/// <summary>Raw forward output of Conv2d (for reference generation only).</summary>
[Module]
public partial class ParityConv2dRaw
{
    public static Tensor<float32> Inline(Tensor<float32> x) => Forwards.Conv2d(x);
}

/// <summary>Linear(out=4, bias=true): output [3,4]=12 compared elementwise to the reference.</summary>
[Module]
public partial class ParityLinear
{
    public static Scalar<bit> Inline(Tensor<float32> x)
        => ModuleForwardValueTests.MatchesReference(
            Forwards.Linear(x), 12, ModuleForwardValueTests.LinearReference, 1e-3f);
}

/// <summary>Conv2d(out=3,k=3,s=2,p=1): output [2,3,5,5]=150 collapsed to 19 and compared.</summary>
[Module]
public partial class ParityConv2d
{
    public static Scalar<bit> Inline(Tensor<float32> x)
        => ModuleForwardValueTests.MatchesReference(
            Forwards.Conv2d(x), 150, ModuleForwardValueTests.Conv2dReference, 1e-3f);
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModuleForwardValueTests
{
    internal const int P = 19;   // collapse width (prime)
    internal static readonly RngConfig ParitySeed = new() { MasterSeed = 12345 };

    // ----- Linear -----------------------------------------------------------
    // REFERENCE: PyTorch — F.linear(input, W, b) on the [4,5] KaimingUniform weight
    // Shorokoo materializes from MasterSeed=12345. Regenerate: tests/pytorch-reference/linear.py.
    internal static readonly float[] LinearReference =
    [
        -0.22925185f, -0.48421156f,  0.48000988f,  0.28337005f,  1.09443188f,
         0.80388707f, -0.47127473f, -0.57127398f, -1.82052422f, -1.02139914f,
         0.40264696f,  0.78657663f,
    ];

    [Fact]
    public void LinearForwardMatchesReference()
        => Assert.True(AutoTest.AdvancedTestGraph<ParityLinear>(
            hyperparamInputs: [], runtimeInputs: [SinInput([3L, 5L])], rngConfig: ParitySeed));

    // ----- Conv2d -----------------------------------------------------------
    // REFERENCE: golden — self-generated (Shorokoo forward, collapsed to 19). Regenerate
    // with the _GoldenReferenceGen harness. Output [2,3,5,5]=150 collapsed via the [19,150]
    // projection matrix.
    internal static readonly float[] Conv2dReference =
    [
        -0.21963434f,  0.95934266f, -0.44873038f,  0.10481805f, -0.28914124f,
        -0.26660570f, -0.01198603f, -0.89579510f,  0.42162973f, -0.58016930f,
        -0.46328200f, -0.50000070f, -0.21899788f, -0.34139730f,  0.02679433f,
        -0.00914705f,  0.23482558f, -0.09963442f, -0.69120497f,
    ];

    [Fact]
    public void Conv2dForwardMatchesReference()
        => Assert.True(AutoTest.AdvancedTestGraph<ParityConv2d>(
            hyperparamInputs: [], runtimeInputs: [SinInput([2L, 2L, 9L, 9L])], rngConfig: ParitySeed));

    // ----- in-graph reference check -----------------------------------------

    /// <summary>In-graph verdict: flattens <paramref name="y"/> (known length <paramref name="n"/>),
    /// collapses it to 19 numbers when large, and returns whether it matches
    /// <paramref name="reference"/> (baked as a constant) within <paramref name="tol"/> (max-abs).</summary>
    internal static Scalar<bit> MatchesReference(Tensor<float32> y, int n, float[] reference, float tol)
    {
        var flat = y.Reshape([Scalar((long)n)]);                 // [n]
        var collapsed = n <= 30 ? flat : Project(flat, n);       // [n] or [19]
        var r = Vector(reference);                               // [M] constant
        var diff = (collapsed - r).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar();
        return diff < Scalar(tol);
    }

    /// <summary>In-graph collapse: <c>collapsed = W · flat</c> with W the baked [19, n] projection.</summary>
    private static Tensor<float32> Project(Tensor<float32> flat, int n)
    {
        var w = Vector(Projection(n)).Reshape([Scalar((long)P), Scalar((long)n)]);   // [19, n]
        var col = flat.Reshape([Scalar((long)n), Scalar(1L)]);                        // [n, 1]
        return w.MatMul(col).Reshape([Scalar((long)P)]);                             // [19]
    }

    /// <summary>Row-major [19, n] projection matrix: entry (i mod 19, i) = w(i), else 0.</summary>
    private static float[] Projection(int n)
    {
        var m = new float[P * n];
        for (int i = 0; i < n; i++)
            m[(i % P) * n + i] = Weight(i);
        return m;
    }

    /// <summary>Fixed position-hash weight in [-0.5, 0.5) — a function of the index only, so it is
    /// bit-identical across machines and never touches the float data.</summary>
    internal static float Weight(int i)
    {
        uint h = unchecked((uint)(i + 1) * 2654435761u);
        return ((h >> 8) & 0xFFFF) / 65535.0f - 0.5f;
    }

    // ----- reference-generation helpers (used by _GoldenReferenceGen) --------

    /// <summary>Deterministic input of shape <paramref name="shape"/>: sin(0.7·i) row-major.</summary>
    internal static TensorData SinInput(long[] shape)
    {
        long n = shape.Aggregate(1L, (a, b) => a * b);
        return TensorData(shape, Enumerable.Range(0, (int)n).Select(i => (float)Math.Sin(i * 0.7)).ToArray());
    }

    /// <summary>Host-side twin of <see cref="Project"/> (same formula, same order-of-magnitude), used
    /// only to emit frozen golden constants. <=30 values pass through unchanged.</summary>
    internal static float[] Collapse(float[] y)
    {
        if (y.Length <= 30) return y;
        var acc = new float[P];
        for (int i = 0; i < y.Length; i++)
            acc[i % P] += y[i] * Weight(i);
        return acc;
    }

    /// <summary>Materializes <typeparamref name="TModule"/> at the parity seed and runs a forward
    /// pass on <paramref name="input"/>, returning the flattened float32 output.</summary>
    internal static float[] RunForward<TModule>(TensorData input)
    {
        var prop = typeof(TModule).GetProperty("ComputationGraph",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var g = (FastComputationGraph)prop.GetValue(null)!;
        var arch = g.ToConcreteArchitecture(g.FromOrderedInputs([input]));
        var model = arch.ToConcreteModel(ParitySeed);
        var outs = new ComputeContext().Execute(model, input);
        return outs[0].ToTensorData<float32>().AccessMemory().ToArray();
    }
}
