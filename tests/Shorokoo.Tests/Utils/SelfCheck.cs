namespace Shorokoo.Tests.Utils
{
    /// <summary>
    /// Helpers for self-checking coverage modules. A coverage <c>[Module]</c> must make its
    /// returned <see cref="Scalar{T}"/> verdict depend on every operation it claims to cover —
    /// otherwise the exercised nodes are orphans, get pruned during concretization, never run
    /// shape inference, and the test validates nothing (see issue #4). These helpers fold an
    /// exercised result into a non-negative error contribution so the producing op stays a
    /// reachable graph output: a clean result contributes ~0, a wrong/NaN result contributes
    /// more, and a broken op (e.g. a shape fault) throws when the now-reachable node is
    /// concretized/executed.
    /// </summary>
    internal static class SelfCheck
    {
        /// <summary>L1 distance between two float tensors, reduced to a scalar.</summary>
        public static Scalar<float32> L1(Tensor<float32> a, Tensor<float32> b)
            => (a - b).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        /// <summary>
        /// Number of NaNs in a float tensor — 0 for any finite result. Reads the tensor so its
        /// producing op stays reachable (not pruned) while contributing 0 to a clean verdict;
        /// use for ops whose exact output value is impractical to reference (transcendentals,
        /// resize/rescale, random sampling), where reachability + finiteness is the bar.
        /// </summary>
        public static Scalar<float32> Nan(Tensor<float32> a)
            => a.IsNaN().Cast<float32>().Reduce(ReduceKind.Sum, keepDims: false).Scalar();

        /// <summary>
        /// Finiteness fold for a tensor of any element type: casts to float32 first, so a
        /// constructed value of any DType (or generic placeholder) is reachable and finiteness-
        /// checked. Use for dispatch-catalog coverage where the goal is "the construction executes
        /// and isn't pruned" rather than an exact per-DType value.
        /// </summary>
        public static Scalar<float32> NanAny<T>(Tensor<T> a) where T : IVarType
            => Nan(a.Cast<float32>());

        // Vector{T}/Scalar{T} are structs (no longer Tensor{T} subclasses), so generic
        // inference can't reach Tensor{T} through their implicit conversions — provide a
        // per-kind overload so T binds directly from the constructed value.
        public static Scalar<float32> NanAny<T>(Vector<T> a) where T : IVarType
            => Nan(a.Cast<float32>());

        public static Scalar<float32> NanAny<T>(Scalar<T> a) where T : IVarType
            => Nan(a.Cast<float32>());
    }
}
