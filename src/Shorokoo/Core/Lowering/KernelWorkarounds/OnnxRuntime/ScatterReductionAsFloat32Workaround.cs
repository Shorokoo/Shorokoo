using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOp;
using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A reducing <c>ScatterND</c> over float16 or bfloat16 data that an ONNX Runtime provider has
/// no kernel for, computed as float32 and cast back:
/// <c>Cast(ScatterND(Cast(data, float32), indices, Cast(updates, float32), reduction), dtype)</c>.
///
/// <para>Float32 holds every float16 and bfloat16 value, so the scatter reads its operands exactly
/// and a <c>max</c> or <c>min</c> comes out as the spec gives it. An <c>add</c> or <c>mul</c>
/// accumulates in float32 and is rounded to the data's type once, at the end: where several
/// updates reach one element, the result is the float32 sum or product rounded, which can differ
/// in the last place from one rounded after every update, and a product whose partial values pass
/// the type's range comes out finite where the type's own accumulation would not.</para>
/// </summary>
internal abstract class ScatterReductionAsFloat32Workaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } = new HashSet<string>([SCATTER_ND], StringComparer.Ordinal);

    public override bool Applies(WorkaroundSite site)
        => ReductionOf(site) is { } reduction && reduction != ScatterNDReduction.None && Takes(site.DTypeOf(0));

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
    {
        var dtype = site.DTypeOf(0);
        var scattered = NodeBuilder.BuildNodeSingleOut(site.OpCode,
            [Cast(inputs[0]!, null, DType.Float32), inputs[1]!, Cast(inputs[2]!, null, DType.Float32)],
            [(AttrReduction, site.Attributes.GetAttributeObj(AttrReduction))]);
        return [Cast(scattered, null, dtype)];
    }

    /// <summary>Whether a reducing call over <paramref name="dtype"/> data is rewritten.</summary>
    protected abstract bool Takes(DType dtype);

    /// <summary>The site's <c>reduction</c>: <see cref="ScatterNDReduction.None"/> where it is
    /// unset, null where it names none of the spec's.</summary>
    internal static ScatterNDReduction? ReductionOf(WorkaroundSite site)
        => !site.Attributes.IsAttributeDefined(AttrReduction)
            ? ScatterNDReduction.None
            : site.Attributes.GetAttributeObj(AttrReduction) switch
            {
                null => ScatterNDReduction.None,
                ScatterNDReduction reduction => reduction,
                string name => Enum.TryParse<ScatterNDReduction>(name, ignoreCase: true, out var parsed) ? parsed : null,
                _ => null,
            };

    protected static bool IsHalf(DType dtype)
        => dtype.IsSameElementTypeAs(DType.Float16) || dtype.IsSameElementTypeAs(DType.BFloat16);
}

/// <summary>
/// A reducing <c>ScatterND</c> over float16 or bfloat16 data, which ONNX Runtime's CPU provider
/// has no kernel for, computed as float32 (<see cref="ScatterReductionAsFloat32Workaround"/>).
/// The CUDA set takes <see cref="CudaBFloat16ScatterReductionWorkaround"/> in its place.
/// </summary>
internal sealed class HalfScatterReductionWorkaround : ScatterReductionAsFloat32Workaround
{
    protected override bool Takes(DType dtype) => IsHalf(dtype);
}

/// <summary>
/// A reducing <c>ScatterND</c> over bfloat16 data, which ONNX Runtime's CUDA execution provider
/// has no kernel for, computed as float32 (<see cref="ScatterReductionAsFloat32Workaround"/>).
/// Part of the CUDA set only
/// (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>), in the place of
/// <see cref="HalfScatterReductionWorkaround"/>: the provider's float16 kernel computes the call
/// as the spec does, accumulating in float16, so a float16 call is left to it.
/// </summary>
internal sealed class CudaBFloat16ScatterReductionWorkaround : ScatterReductionAsFloat32Workaround
{
    protected override bool Takes(DType dtype) => dtype.IsSameElementTypeAs(DType.BFloat16);
}
