using System.Collections.Immutable;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

namespace Shorokoo.Core.Lowering.KernelWorkarounds;

/// <summary>
/// The registered <see cref="KernelWorkaroundSet"/>s, by the name a backend gives
/// (<see cref="IShorokooBackend.KernelWorkaroundSet"/>). Built once and never changed, so every
/// thread reads the same sets.
/// </summary>
internal static class KernelWorkaroundRegistry
{
    /// <summary>
    /// The rewrites around ONNX Runtime's kernels, in the order they apply: the cross-entropy ONNX
    /// Runtime has no kernel for, as the log-softmax and negative log-likelihood it is made of,
    /// ahead of all of them, so the rewrites after it see those calls; then the reductions' axis
    /// handling, so the later reduction workarounds see normalised calls, with the negative
    /// axes made non-negative before a reduction of no axis is rewritten, since the axis that
    /// rewrite reduces is never negative; the crop-and-resize roi before the cubic resize layout,
    /// and both before the resize axes, since the calls they build keep the call's <c>axes</c>;
    /// the <c>MatMul</c> after the reductions, since the reductions it builds take no axes and
    /// need none of their rewrites. The int64 <c>Range</c> count comes after every other rewrite
    /// that builds a <c>Range</c>, so such a call is looked at like one written in the model, and
    /// the float16 and bfloat16 reducing scatters after every rewrite that builds a scatter.
    /// <c>Where</c> last, since earlier rewrites, the <c>Range</c> count among them, build
    /// <c>Where</c>s of their own.
    /// </summary>
    public static KernelWorkaroundSet OnnxRuntime { get; } = new(KernelWorkaroundSets.OnnxRuntime, OnnxRuntimeWorkarounds);

    /// <summary>
    /// The rewrites around ONNX Runtime's kernels on its CUDA execution provider: every one of
    /// <see cref="OnnxRuntime"/>, in its order, but with the reductions of no axis widened to every
    /// input for the four the provider passes through, in their place; with the crop-and-resize roi
    /// choosing its sequence by a <c>Where</c> rather than an <c>If</c>, in its place; with the cubic
    /// crop-and-resize layout widened to
    /// every linear and cubic resize the provider computes wrongly on its channels-last route, in
    /// its place: so a call is regrouped once, whichever of the two would take it, and the call a
    /// regrouping under a <c>not_larger</c> or <c>not_smaller</c> policy builds, which keeps its
    /// <c>axes</c>, comes before the resize axes as that layout's does; and with the float16 and
    /// bfloat16 reducing scatters narrowed to the bfloat16 <c>ScatterND</c> the provider has no
    /// kernel for, in their place, so the float32 call it builds runs on the card. Then the calls
    /// the provider runs on the CPU over values from outside their body, which covers such calls
    /// the earlier rewrites build too; then a tensor read straight back out of its <c>Optional</c>, which the provider cannot run; then
    /// the summed or averaged losses written out as the reductions they are; then the reductions
    /// with <c>keepdims</c> 0 over an input that may be empty, after every rewrite that builds a
    /// reduction; then the calls the provider takes and has no kernel for, kept on
    /// the host, which covers such calls every rewrite before builds; then a crop-and-resize's
    /// constant <c>roi</c> read as a value the card computes, after every rewrite that builds a
    /// resize; last, the linear uint8 resizes the provider computes wrongly, kept on the host.
    /// </summary>
    public static KernelWorkaroundSet OnnxRuntimeCuda { get; } = new(KernelWorkaroundSets.OnnxRuntimeCuda,
        [.. OnnxRuntimeWorkarounds.Select(w => w switch
            {
                ReduceNoopEmptyAxesWorkaround => new CudaReduceNoopAxesWorkaround(),
                CropAndResizeRoiWorkaround => new CudaCropAndResizeRoiWorkaround(),
                CubicResizeMiddleAxesWorkaround => new CudaResizeMiddleAxesWorkaround(),
                HalfScatterReductionWorkaround => new CudaBFloat16ScatterReductionWorkaround(),
                _ => w,
            }), .. CudaWorkarounds]);

    /// <summary>The rewrites <see cref="OnnxRuntimeCuda"/> adds after <see cref="OnnxRuntime"/>'s,
    /// in their order.</summary>
    internal static ImmutableArray<KernelWorkaround> CudaWorkarounds =>
    [
        new CudaHostFallbackBodyInputWorkaround(),
        new CudaOptionalRoundTripWorkaround(),
        new CudaLossReductionWorkaround(),
        new CudaReduceEmptyAxisWorkaround(),
        new CudaNoKernelFormOnHostWorkaround(),
        new CudaCropAndResizeConstantRoiWorkaround(),
        new CudaUInt8LinearResizeOnHostWorkaround(),
    ];

    private static ImmutableArray<KernelWorkaround> OnnxRuntimeWorkarounds =>
    [
        new CrossEntropyAsLogSoftmaxWorkaround(),
        new ReduceNegativeAxesWorkaround(),
        new ReduceNoopEmptyAxesWorkaround(),
        new BoolEmptyReduceExtremeWorkaround(),
        new MaxPoolLowestValueIndexWorkaround(),
        new PoolPaddingWorkaround(),
        new ConvTransposeSamePaddingWorkaround(),
        new Col2ImOneAxisWorkaround(),
        new CropAndResizeRoiWorkaround(),
        new CubicResizeMiddleAxesWorkaround(),
        new ResizeAxesSubsetWorkaround(),
        new IntegerRangeCountWorkaround(),
        new MatMulEmptyOperandWorkaround(),
        new HalfScatterReductionWorkaround(),
        new WhereTypesWorkaround(),
    ];

    /// <summary>The set named <paramref name="name"/>; the empty set for null or a name no set
    /// is registered under.</summary>
    public static KernelWorkaroundSet For(string? name) => name switch
    {
        KernelWorkaroundSets.OnnxRuntime => OnnxRuntime,
        KernelWorkaroundSets.OnnxRuntimeCuda => OnnxRuntimeCuda,
        _ => KernelWorkaroundSet.Empty,
    };
}
