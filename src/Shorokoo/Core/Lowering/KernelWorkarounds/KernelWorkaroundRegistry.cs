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
    /// The rewrites around ONNX Runtime's kernels, in the order they apply: the reductions' axis
    /// handling first, so the later reduction workarounds see normalised calls, with the negative
    /// axes made non-negative before a reduction of no axis is rewritten, since the axis that
    /// rewrite reduces is never negative; the crop-and-resize roi before the cubic resize layout,
    /// and both before the resize axes, since the calls they build keep the call's <c>axes</c>;
    /// the <c>MatMul</c> after the reductions, since the reductions it builds take no axes and
    /// need none of their rewrites. The integer <c>Range</c> count and the arithmetic with an
    /// empty constant come after every other rewrite that builds a <c>Range</c> or an <c>Add</c>,
    /// <c>Sub</c>, <c>Mul</c> or <c>Div</c>, so such a call is looked at like one written in the
    /// model; neither builds a call the other rewrites, so their order between them changes
    /// nothing. <c>Where</c> last, since earlier rewrites, the integer <c>Range</c> count among
    /// them, build <c>Where</c>s of their own.
    /// </summary>
    public static KernelWorkaroundSet OnnxRuntime { get; } = new(KernelWorkaroundSets.OnnxRuntime, OnnxRuntimeWorkarounds);

    /// <summary>
    /// The rewrites around ONNX Runtime's kernels on its CUDA execution provider: every one of
    /// <see cref="OnnxRuntime"/>, in its order, then the calls the provider runs on the CPU over
    /// values from outside their body, which covers such calls the earlier rewrites build too.
    /// </summary>
    public static KernelWorkaroundSet OnnxRuntimeCuda { get; } = new(KernelWorkaroundSets.OnnxRuntimeCuda,
        [.. OnnxRuntimeWorkarounds, new CudaHostFallbackBodyInputWorkaround()]);

    private static ImmutableArray<KernelWorkaround> OnnxRuntimeWorkarounds =>
    [
        new ReduceNegativeAxesWorkaround(),
        new ReduceNoopEmptyAxesWorkaround(),
        new Float16EmptyReduceWorkaround(),
        new IntegerEmptyReduceExtremeWorkaround(),
        new MaxPoolLowestValueIndexWorkaround(),
        new PoolPaddingWorkaround(),
        new ConvTransposeSamePaddingWorkaround(),
        new Col2ImOneAxisWorkaround(),
        new CropAndResizeRoiWorkaround(),
        new CubicResizeMiddleAxesWorkaround(),
        new ResizeAxesSubsetWorkaround(),
        new IntegerRangeCountWorkaround(),
        new MatMulEmptyOperandWorkaround(),
        new ArithmeticEmptyConstantWorkaround(),
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
