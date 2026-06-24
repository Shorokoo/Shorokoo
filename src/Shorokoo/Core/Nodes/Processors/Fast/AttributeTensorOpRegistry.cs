using System.Collections.Generic;
using System.Collections.Immutable;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// One tensor-input → static-attribute mapping for an attribute-tensorized operator variant.
    /// <see cref="InputName"/> is the variant op's input parameter name (as declared in its
    /// NodeDefinition); <see cref="AttributeName"/> is the target attribute on the standard op;
    /// <see cref="IsScalar"/> selects a single <c>long</c> (vs a <c>long[]</c>) attribute value.
    /// </summary>
    internal sealed record AttributeTensorMapping(string InputName, string AttributeName, bool IsScalar);

    /// <summary>
    /// Describes how a Shorokoo-specific operator variant lowers back to a standard ONNX operator:
    /// the <see cref="StandardOpCode"/> to swap to, and the set of tensor inputs that must be
    /// resolved to constant values and written as static attributes. Any variant attribute the
    /// standard op also declares (and that is not produced by a mapping, e.g. Conv's
    /// <c>auto_pad</c>) is carried over verbatim; any variant input not covered by a mapping
    /// (e.g. Conv's <c>X</c>/<c>W</c>/<c>B</c>) passes through to the standard op unchanged.
    /// </summary>
    internal sealed record AttributeTensorSpec(string StandardOpCode, ImmutableArray<AttributeTensorMapping> TensorAttributes);

    /// <summary>
    /// Registry of attribute-tensorized operator variants, keyed by the variant op code. The
    /// generic <see cref="FastLowerAttributeTensorOps"/> pass is driven entirely by this table —
    /// adding a new variant is one definition + one entry here, with no new pass code.
    /// </summary>
    internal static class AttributeTensorOpRegistry
    {
        public static readonly ImmutableDictionary<string, AttributeTensorSpec> Specs =
            new Dictionary<string, AttributeTensorSpec>
            {
                [SHRK_CONV] = new AttributeTensorSpec(CONV, ImmutableArray.Create(
                    new AttributeTensorMapping("pads", AttrPads, IsScalar: false),
                    new AttributeTensorMapping("strides", AttrStrides, IsScalar: false),
                    new AttributeTensorMapping("dilations", AttrDilations, IsScalar: false),
                    new AttributeTensorMapping("kernel_shape", AttrKernelShape, IsScalar: false),
                    new AttributeTensorMapping("group", AttrGroup, IsScalar: true))),
            }.ToImmutableDictionary();
    }
}
