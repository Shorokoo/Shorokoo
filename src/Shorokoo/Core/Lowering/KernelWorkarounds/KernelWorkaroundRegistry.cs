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
    /// The rewrites around ONNX Runtime's kernels, in the order they apply: the reductions'
    /// axis handling first, so the later reduction workarounds see normalised calls, with the
    /// negative axes made non-negative before a reduction of no axis is rewritten, since the axis
    /// that rewrite reduces is never negative; the
    /// crop-and-resize roi before the cubic resize layout, and both before the resize axes, since
    /// the calls they build keep the call's <c>axes</c>; <c>Where</c> last, since earlier
    /// rewrites, the int64 <c>Range</c> count among them, build <c>Where</c>s of their own.
    /// </summary>
    public static KernelWorkaroundSet OnnxRuntime { get; } = new(KernelWorkaroundSets.OnnxRuntime,
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
        new Int64RangeCountWorkaround(),
        new WhereTypesWorkaround(),
    ]);

    /// <summary>The set named <paramref name="name"/>; the empty set for null or a name no set
    /// is registered under.</summary>
    public static KernelWorkaroundSet For(string? name) => name switch
    {
        KernelWorkaroundSets.OnnxRuntime => OnnxRuntime,
        _ => KernelWorkaroundSet.Empty,
    };
}
