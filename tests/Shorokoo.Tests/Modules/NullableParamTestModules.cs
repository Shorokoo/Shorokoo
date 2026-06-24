namespace Shorokoo.Tests.Modules
{
    // Modules exercising the source-generated nullable surface. An OptionalTensor<T> parameter is
    // exposed to callers as Tensor<T>? (omit / pass null = absent), and a [Hyper(default)] scalar
    // is exposed as a nullable, omittable parameter that falls back to its attribute default. The
    // body itself uses ordinary, non-nullable parameters and the OptionalTensor API for branching.

    /// <summary>Optional bias input; defaults to zeros when absent.</summary>
    [Module]
    public partial class NullableBiasLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, OptionalTensor<float32> bias)
        {
            var b = bias.HasValue().IfElse(() => bias.TensorValue(), () => TensorFill(x.ShapeTensor(), 0f));
            return x + b;
        }
    }

    /// <summary>Two optional inputs, each defaulting independently (zeros / ones).</summary>
    [Module]
    public partial class TwoNullableLayer
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            OptionalTensor<float32> bias,
            OptionalTensor<float32> scale)
        {
            var b = bias.HasValue().IfElse(() => bias.TensorValue(), () => TensorFill(x.ShapeTensor(), 0f));
            var s = scale.HasValue().IfElse(() => scale.TensorValue(), () => TensorFill(x.ShapeTensor(), 1f));
            return x * s + b;
        }
    }

    /// <summary>Optional bias combined with a trainable weight (ones-initialized).</summary>
    [Module]
    public partial class NullableTrainableBiasLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, OptionalTensor<float32> bias)
        {
            var w = InitSimple.Init(x.ShapeTensor());
            var b = bias.HasValue().IfElse(() => bias.TensorValue(), () => TensorFill(x.ShapeTensor(), 0f));
            return x * w + b;
        }
    }

    /// <summary>A single [Hyper(default)] scalar the caller may omit (falls back to 3.0).</summary>
    [Module]
    public partial class DefaultedHyperLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper(3f)] Scalar<float32> factor)
            => x * factor;
    }

    /// <summary>Two [Hyper(default)] scalars — scale (2.0) and bias (1.0) — each omittable
    /// independently, so a caller can supply some and omit others.</summary>
    [Module]
    public partial class TwoDefaultedHyperLayer
    {
        public static Tensor<float32> Inline(
            Tensor<float32> x,
            [Hyper(2f)] Scalar<float32> scale,
            [Hyper(1f)] Scalar<float32> bias)
            => x * scale + bias;
    }

    // ── self-checking caller modules ──
    // Each calls a defaulted-hyper module — omitting some or all of its defaults — and returns a
    // Scalar<bit> that is FALSE when the wrong value comes out (a default was not applied as
    // expected). AutoTest.AdvancedTestGraph treats a Scalar<bit> output as a self-check and fails
    // the test on a false bit, so the validation lives in the module and the xUnit test is a one-liner.

    /// <summary>Omits the lone default → DefaultedHyperLayer.Model().Call(x) must equal x * 3.</summary>
    [Module]
    public partial class DefaultedHyperOmitCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var actual = DefaultedHyperLayer.Model().Call(x);              // factor omitted → x * 3
            var expected = x * Scalar(3f);
            return (actual - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

    /// <summary>Supplies the lone default → DefaultedHyperLayer.Model(Scalar(5f)).Call(x) must equal x * 5.</summary>
    [Module]
    public partial class DefaultedHyperSupplyCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var actual = DefaultedHyperLayer.Model(Scalar(5f)).Call(x);    // factor = 5 → x * 5
            var expected = x * Scalar(5f);
            return (actual - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

    /// <summary>Omits BOTH defaults → TwoDefaultedHyperLayer.Model().Call(x) must equal x*2 + 1.</summary>
    [Module]
    public partial class TwoDefaultedHyperOmitAllCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var actual = TwoDefaultedHyperLayer.Model().Call(x);           // scale & bias omitted → x*2 + 1
            var expected = x * Scalar(2f) + Scalar(1f);
            return (actual - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

    /// <summary>Omits SOME: supplies scale = 5 (positional), omits bias →
    /// TwoDefaultedHyperLayer.Model(Scalar(5f)).Call(x) must equal x*5 + 1 (bias defaulted to 1).</summary>
    [Module]
    public partial class TwoDefaultedHyperOmitBiasCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var actual = TwoDefaultedHyperLayer.Model(Scalar(5f)).Call(x); // scale = 5, bias omitted → x*5 + 1
            var expected = x * Scalar(5f) + Scalar(1f);
            return (actual - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

    /// <summary>Omits SOME the other way: omits scale (defaulted to 2), supplies bias = 7 via a named
    /// argument → TwoDefaultedHyperLayer.Model(bias: Scalar(7f)).Call(x) must equal x*2 + 7.</summary>
    [Module]
    public partial class TwoDefaultedHyperOmitScaleCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var actual = TwoDefaultedHyperLayer.Model(bias: Scalar(7f)).Call(x); // scale omitted → 2, bias = 7 → x*2 + 7
            var expected = x * Scalar(2f) + Scalar(7f);
            return (actual - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

    /// <summary>Passes a present bias through the Tensor?-accepting generated Call → must equal x + bias.</summary>
    [Module]
    public partial class NullableBiasPresentCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x, Tensor<float32> bias)
        {
            var actual = NullableBiasLayer.Call(x, bias);                  // present bias → x + bias
            var expected = x + bias;
            return (actual - expected).Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-6f);
        }
    }

}
