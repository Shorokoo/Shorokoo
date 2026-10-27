using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering.KernelWorkarounds.OnnxRuntime;

using static OnnxOpAttributeNames;
using static OpCodes;

/// <summary>
/// A call ONNX Runtime's CUDA execution provider takes when the session is built and has no kernel
/// for when it runs, kept on the host (Shorokoo/Shorokoo#541):
/// <list type="bullet">
/// <item>a <c>ScatterND</c> with a reduction over data of any type but float32 and float16 —
/// the provider's reducing kernel computes those two only, and a float64 table's gradient is such a
/// call;</item>
/// <item>a blocked <c>QuantizeLinear</c> or <c>DequantizeLinear</c> (<c>block_size</c> set) of any
/// quantized type but int4 and uint4, the only ones the provider's blocked kernels compute; a
/// <c>QuantizeLinear</c> whose output type the graph does not tell counts as such.</item>
/// </list>
///
/// <para>The provider registers each operator for every type, so the session places these calls
/// on the card, and a run fails at the first one. Kept on the host, the call runs on the CPU
/// provider, which computes every form, at the cost of copying its operands to the host and its
/// result back. Every other call stays where the provider places it.</para>
///
/// <para>Part of the CUDA set only (<see cref="Shorokoo.Core.Backends.KernelWorkaroundSets.OnnxRuntimeCuda"/>),
/// last, so the calls earlier rewrites build are kept on the host too.</para>
/// </summary>
internal sealed class CudaNoKernelFormOnHostWorkaround : KernelWorkaround
{
    public override IReadOnlySet<string> OpCodes { get; } =
        new HashSet<string>([SCATTER_ND, QUANTIZE_LINEAR, DEQUANTIZE_LINEAR], StringComparer.Ordinal);

    public override bool KeepsOnHost => true;

    public override bool Applies(WorkaroundSite site) => site.OpCode switch
    {
        SCATTER_ND => Reduces(site) && !IsFloat32Or16(site.DTypeOf(0)),
        QUANTIZE_LINEAR => site.Attributes.GetLongVal(AttrBlockSize) is > 0 && !IsInt4(site.OutputDTypeOf(0)),
        _ => site.Attributes.GetLongVal(AttrBlockSize) is > 0 && !IsInt4(site.DTypeOf(0)),
    };

    public override Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs)
        => throw new InvalidOperationException("A call kept on the host is not rewritten.");

    private static bool Reduces(WorkaroundSite site)
        => site.Attributes.IsAttributeDefined(AttrReduction)
           && site.Attributes.GetAttributeObj(AttrReduction) switch
           {
               null => false,
               ScatterNDReduction reduction => reduction != ScatterNDReduction.None,
               string name => !name.Equals("none", StringComparison.OrdinalIgnoreCase),
               _ => true,
           };

    private static bool IsFloat32Or16(DType dtype)
        => dtype.IsSameElementTypeAs(DType.Float32) || dtype.IsSameElementTypeAs(DType.Float16);

    private static bool IsInt4(DType dtype)
        => dtype.IsSameElementTypeAs(DType.Int4) || dtype.IsSameElementTypeAs(DType.UInt4);
}
