using System;
using System.Linq;
using Shorokoo.Graph;
using Shorokoo.Modules.Layers;
using Shorokoo.Runtime;
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
// compared to a frozen reference. Each reference carries a provenance comment:
//
//   // REFERENCE: PyTorch    — produced by the equivalent PyTorch op on the same
//                             weights+input (tests/pytorch-reference/); an external
//                             source of truth.
//   // REFERENCE: golden     — self-generated: Shorokoo's own forward output, frozen.
//                             Catches regressions but not a currently-wrong-but-stable
//                             implementation. Upgraded to a PyTorch reference over time.
//
// Outputs of <=30 values are compared elementwise; larger outputs are collapsed
// (Collapse) to 19 order- and value-sensitive, fp-stable numbers first.
// ---------------------------------------------------------------------------

/// <summary>Linear(out=4, bias=true) fixture: single tensor in/out for a forward-value check.</summary>
[Module]
public partial class ParityLinear
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Linear.Call(Scalar(4L), Scalar(true), x);
}

/// <summary>Conv2d(out=3, k=3, stride=2, pad=1, dilation=1, group=1, bias=true) fixture.</summary>
[Module]
public partial class ParityConv2d
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => Conv2d.Model(Scalar(3L), Scalar(3L), Scalar(2L), Scalar(1L), Scalar(1L), Scalar(1L), Scalar(true)).Call(x);
}

[Trait("Domain", "Modules")]
[Trait("Purpose", "Coverage")]
public class ModuleForwardValueTests
{
    internal static readonly RngConfig ParitySeed = new() { MasterSeed = 12345 };

    /// <summary>Deterministic input of shape <paramref name="shape"/>: sin(0.7·i) row-major.</summary>
    internal static TensorData SinInput(long[] shape)
    {
        long n = shape.Aggregate(1L, (a, b) => a * b);
        return TensorData(shape, Enumerable.Range(0, (int)n).Select(i => (float)Math.Sin(i * 0.7)).ToArray());
    }

    // ----- Linear -----------------------------------------------------------
    // REFERENCE: PyTorch — F.linear(input, W, b) on the [4,5] KaimingUniform weight
    // Shorokoo materializes from MasterSeed=12345. Regenerate: tests/pytorch-reference/linear.py.
    private static readonly float[] LinearReference =
    [
        -0.22925185f, -0.48421156f,  0.48000988f,  0.28337005f,  1.09443188f,
         0.80388707f, -0.47127473f, -0.57127398f, -1.82052422f, -1.02139914f,
         0.40264696f,  0.78657663f,
    ];

    [Fact]
    public void LinearForwardMatchesReference()
    {
        var y = RunForward<ParityLinear>(SinInput([3L, 5L]));
        AssertMatches(LinearReference, y);
    }

    // ----- Conv2d -----------------------------------------------------------
    // REFERENCE: golden — self-generated (Shorokoo forward, collapsed). Regenerate with
    // the _GoldenGen harness. Output [2,3,5,5]=150 values, collapsed to 19.
    private static readonly float[] Conv2dReference =
    [
        -0.21963434f,  0.95934266f, -0.44873038f,  0.10481805f, -0.28914124f,
        -0.26660570f, -0.01198603f, -0.89579510f,  0.42162973f, -0.58016930f,
        -0.46328200f, -0.50000070f, -0.21899788f, -0.34139730f,  0.02679433f,
        -0.00914705f,  0.23482558f, -0.09963442f, -0.69120497f,
    ];

    [Fact]
    public void Conv2dForwardMatchesReference()
    {
        var y = Collapse(RunForward<ParityConv2d>(SinInput([2L, 2L, 9L, 9L])));
        AssertMatches(Conv2dReference, y);
    }

    // ----- helpers ----------------------------------------------------------

    private static void AssertMatches(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 1e-3f);   // headroom for cross-machine kernel variation
    }

    /// <summary>Collapses a large flat output to 19 fp-stable numbers that still move when any
    /// value changes or two values are swapped: each of 19 prime-strided buckets accumulates a
    /// position-weighted sum (integer-hash weight, no transcendental → portable). Outputs of
    /// <=30 values are returned unchanged for elementwise comparison.</summary>
    internal static float[] Collapse(float[] y)
    {
        if (y.Length <= 30) return y;
        const int P = 19;
        var acc = new float[P];
        for (int i = 0; i < y.Length; i++)
        {
            uint h = unchecked((uint)(i + 1) * 2654435761u);
            float w = ((h >> 8) & 0xFFFF) / 65535.0f - 0.5f;   // deterministic bounded weight [-0.5,0.5)
            acc[i % P] += y[i] * w;
        }
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
