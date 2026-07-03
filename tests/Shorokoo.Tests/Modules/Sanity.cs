using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

/// <summary>
/// Lightweight "reasonable output" self-check for coverage <c>[Module]</c>s that only need to
/// exercise a layer end-to-end rather than re-derive its math. The former closed-form modules
/// re-implemented the layer's own computation and asserted equality — circular, and (under
/// per-parameter init RNG) reliant on a re-run initializer coinciding with the module's realized
/// weight. Instead these modules now just CALL the layer and assert the output is sane: finite
/// and within a generous hardcoded magnitude bound, and not identically zero. Deterministic under
/// the default master-seed RNG. The value-exact behaviour is covered elsewhere (the GetTrainableParam
/// closed-form checks that survived, plus the training-rig smoke tests).
/// </summary>
internal static class Sanity
{
    /// <summary>
    /// Bounded-output bit: <paramref name="y"/> is within ±<paramref name="bound"/> (NaN/Inf
    /// fail the comparison). No non-degeneracy requirement — use for outputs that can
    /// legitimately be all-zero, e.g. a relu over a tiny hidden state whose pre-activations
    /// all landed negative under some weight draw.
    /// </summary>
    public static Scalar<bit> Bounded(Tensor<float32> y, float bound = 1000f)
    {
        var maxv = y.Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var minv = y.Reduce(ReduceKind.Min, keepDims: false).Scalar();
        return (maxv < Scalar(bound)) & (minv > Scalar(-bound));
    }

    /// <summary>
    /// Reasonable-output bit: <paramref name="y"/> is bounded by ±<paramref name="bound"/> and
    /// carries non-zero mass. NaN/Inf fail (a NaN max compares false against the bound; a NaN
    /// abs-sum compares false against zero), so this also serves as a finiteness guard.
    /// </summary>
    public static Scalar<bit> Reasonable(Tensor<float32> y, float bound = 1000f)
    {
        var maxv = y.Reduce(ReduceKind.Max, keepDims: false).Scalar();
        var minv = y.Reduce(ReduceKind.Min, keepDims: false).Scalar();
        var absSum = y.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        var bounded = (maxv < Scalar(bound)) & (minv > Scalar(-bound));
        var nonDegenerate = absSum > Scalar(0f);
        return bounded & nonDegenerate;
    }
}
