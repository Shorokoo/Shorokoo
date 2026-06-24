using Shorokoo.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// A lightweight, mutable representation of a <see cref="Node"/>. FastNodes reference
    /// inputs and outputs by <see cref="FastTensorKey"/> (i.e. GUID-based identifiers) instead of
    /// by tensor object. This makes FastNodes cheap to construct, to clone and to edit in
    /// place, which is useful for offline graph analyses and transformations that do not need
    /// the full rich object graph of a <c>ComputationGraph</c>.
    ///
    /// The node owns its op code, attributes and tensor-key references. Per-tensor metadata
    /// (dtype, structure, rank, user-facing name, etc.) is built on demand via
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastTensorInfoProcessor.BuildTensorInfoLookup"/>.
    ///
    /// Unlike <see cref="Node"/>, FastNode is fully mutable - every property has a public
    /// setter and the input/output dictionaries can be freely edited.
    /// </summary>
    public class FastNode
    {
        /// <summary>
        /// Globally unique identifier for this node. Preserves the identity of the source
        /// <see cref="Node"/> across round-trip conversions.
        /// </summary>
        public FastNodeKey Key { get; set; }

        /// <summary>
        /// The op code of this node (e.g. "Add", "Conv", or a Shorokoo internal op code).
        /// </summary>
        public string OpCode { get; set; } = string.Empty;

        /// <summary>
        /// The node's attributes. <see cref="OnnxCSharpAttributes"/> is itself immutable, but
        /// the reference is swappable via the setter - use <c>Attributes.SetAttributes(...)</c>
        /// to produce a new attribute bag and assign it back.
        /// </summary>
        public OnnxCSharpAttributes Attributes { get; set; } = null!;

        /// <summary>
        /// Inputs to this node, grouped by graph-attribute name. Input tensors are referenced
        /// by <see cref="FastTensorKey"/>; a null entry represents an explicit missing/optional
        /// input slot. For the typical case (no graph attributes) all inputs live under the
        /// empty-string key.
        /// </summary>
        public Dictionary<string, List<FastTensorKey?>> FullInputs { get; set; } = new();

        /// <summary>
        /// Outputs of this node, grouped the same way as <see cref="FullInputs"/>. A null
        /// entry represents a missing/optional output slot.
        /// </summary>
        public Dictionary<string, List<FastTensorKey?>> FullOutputs { get; set; } = new();

        /// <summary>
        /// Human-readable name for the node, if any.
        /// </summary>
        public string? FriendlyName { get; set; }

        /// <summary>
        /// Original stack trace captured when the node was first built, if any.
        /// </summary>
        public string? StackTrace { get; set; }

        /// <summary>
        /// For close nodes (e.g. IF_CLOSE, LOOP_CLOSE), the key of the matching open node.
        /// </summary>
        public FastNodeKey? GraphOpenNodeKey { get; set; }

        /// <summary>
        /// Serialized form of the node's <see cref="ModelParamIdentifierTemplate"/>, if any.
        /// </summary>
        public string? IdentifierTemplate { get; set; }

        /// <summary>
        /// Optional reference to the function this node targets. Functions themselves are not
        /// flattened by this representation - they are preserved as-is so that the round-trip
        /// converter can rebuild a faithful <c>ComputationGraph</c>.
        /// </summary>
        public Function? TargetFunction { get; set; }

        /// <summary>
        /// Flattened view of all input tensor keys, in the same deterministic order that
        /// <see cref="Node.Inputs"/> uses (graph-attribute groups sorted by ordinal key).
        /// </summary>
        public List<FastTensorKey?> Inputs =>
            FullInputs.OrderBy(x => x.Key, System.StringComparer.Ordinal)
                      .SelectMany(x => x.Value)
                      .ToList();

        /// <summary>
        /// Flattened view of all output tensor keys, in the same deterministic order that
        /// <see cref="Node.Outputs"/> uses.
        /// </summary>
        public List<FastTensorKey?> Outputs =>
            FullOutputs.OrderBy(x => x.Key, System.StringComparer.Ordinal)
                       .SelectMany(x => x.Value)
                       .ToList();

        public override string ToString() => $"FastNode({OpCode})";

        /// <summary>
        /// Returns the <see cref="TensorData"/> stored on this node's
        /// <see cref="Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkAttrTensorData"/>
        /// attribute when the node is a MODEL_PARAM_DATA node, or null otherwise.
        /// Mirrors <see cref="Node.GetTensorData"/>.
        /// </summary>
        public TensorData? GetTensorData() => this.OpCode != Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes.MODEL_PARAM_DATA
            ? null
            : this.Attributes.GetTensorVal(Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames.ShrkAttrTensorData);
    }
}
