using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Collections.Immutable;
using System.Linq;
using System.Threading;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The one place that reads and writes the <see cref="OnnxOpAttributeNames.ShrkAttrCheckpoint"/>
/// stamp on inlined nodes. A <c>[Module(Checkpoint = true)]</c> invoke becomes, after
/// inlining, a <b>segment</b>: the spliced-in nodes carry the segment's id, positive on
/// interior nodes and negative on the nodes that produce the invoke's outputs. The memory-aware
/// pass (<see cref="Rematerializer.ApplyCheckpointSegments"/>) recomputes every interior
/// tensor the backward pass reads from the segment's boundary, once per segment.
///
/// <para>The stamp is an ordinary attribute so it rides every lowering pass that keeps a node's
/// attribute bag; a pass that rebuilds a node from its op definition drops it, which merely
/// shrinks the segment (a node without the stamp is outside it), never breaks anything. It is
/// removed at ONNX emission (<see cref="Strip"/>) because no ORT kernel schema declares it.</para>
/// </summary>
internal static class CheckpointSegment
{
    private static long _nextId;

    private static readonly NodeDefAttributeDef Def = new()
    {
        AttributeName = OnnxOpAttributeNames.ShrkAttrCheckpoint,
        Type = AttributeType.Long,
        DefaultValue = null,
    };

    /// <summary>A segment id no other inlining anywhere in the process has used.</summary>
    public static long NewId() => Interlocked.Increment(ref _nextId);

    /// <summary>Whether <paramref name="invoke"/> (a MODEL_INVOKE) asks for checkpointing.</summary>
    public static bool IsRequested(FastNode invoke)
        => invoke.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrCheckpoint)
        && invoke.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrCheckpoint) is > 0;

    /// <summary>
    /// Whether a node can carry the stamp: the executable ops (those with an ONNX schema, plus
    /// the Conv variant that lowers to one). Module-stage machinery — parameter references,
    /// hyperparameter nodes, struct fields, RNG feeds — is rebuilt by later lowerings against
    /// its own op definition, which has no slot for the stamp, and none of it is ever
    /// recomputed anyway.
    /// </summary>
    public static bool IsStampable(FastNode node)
        => Definitions.VanillaOpNames.Contains(node.OpCode) || node.OpCode == InternalOpCodes.SHRK_CONV;

    /// <summary>Stamps <paramref name="node"/> as a member of segment <paramref name="id"/>.</summary>
    public static void Stamp(FastNode node, long id, bool producesSegmentOutput)
    {
        if (!IsStampable(node)) return;
        var value = producesSegmentOutput ? -id : id;
        var attrs = node.Attributes;
        if (attrs.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrCheckpoint))
        {
            node.Attributes = attrs.SetAttributes((OnnxOpAttributeNames.ShrkAttrCheckpoint, (object?)value));
            return;
        }
        var vals = attrs.GetAttributeVals().ToDictionary(kv => kv.Key, kv => kv.Value);
        vals[OnnxOpAttributeNames.ShrkAttrCheckpoint] = value;
        node.Attributes = OnnxCSharpAttributes.FromCSharpVals(vals, attrs.AttributeDefs.Add(Def));
    }

    /// <summary>The segment <paramref name="node"/> belongs to, or null when it is in none.</summary>
    public static long? IdOf(FastNode node)
    {
        if (!node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrCheckpoint)) return null;
        var value = node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrCheckpoint);
        return value is long v && v != 0 ? System.Math.Abs(v) : null;
    }

    /// <summary>True when <paramref name="node"/> produces one of its segment's outputs.</summary>
    public static bool ProducesSegmentOutput(FastNode node)
        => node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrCheckpoint)
        && node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrCheckpoint) is < 0;

    /// <summary>
    /// <paramref name="attrs"/> without the stamp — its value and its definition — so the bag
    /// is exactly what the op's own schema declares. Returns the same instance when there is
    /// nothing to strip.
    /// </summary>
    public static OnnxCSharpAttributes Strip(OnnxCSharpAttributes attrs)
    {
        if (!attrs.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrCheckpoint)) return attrs;
        var vals = attrs.GetAttributeVals()
            .Where(kv => kv.Key != OnnxOpAttributeNames.ShrkAttrCheckpoint)
            .ToDictionary(kv => kv.Key, kv => kv.Value);
        var defs = attrs.AttributeDefs.RemoveAll(d => d.AttributeName == OnnxOpAttributeNames.ShrkAttrCheckpoint);
        return OnnxCSharpAttributes.FromCSharpVals(vals, defs);
    }

    /// <summary>
    /// Carries the stamp of <paramref name="from"/> onto <paramref name="to"/> — for a pass
    /// that rebuilds a node's attribute bag from a different op definition.
    /// </summary>
    public static OnnxCSharpAttributes Carry(OnnxCSharpAttributes from, OnnxCSharpAttributes to)
    {
        if (!from.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrCheckpoint)) return to;
        var value = from.GetLongVal(OnnxOpAttributeNames.ShrkAttrCheckpoint);
        if (value is null) return to;
        var vals = to.GetAttributeVals().ToDictionary(kv => kv.Key, kv => kv.Value);
        vals[OnnxOpAttributeNames.ShrkAttrCheckpoint] = value;
        var defs = to.AttributeDefs.Any(d => d.AttributeName == OnnxOpAttributeNames.ShrkAttrCheckpoint)
            ? to.AttributeDefs : to.AttributeDefs.Add(Def);
        return OnnxCSharpAttributes.FromCSharpVals(vals, defs);
    }
}
