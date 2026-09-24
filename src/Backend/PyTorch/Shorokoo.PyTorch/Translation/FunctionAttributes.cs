using Shorokoo.Core.Factory.IR;

namespace Shorokoo.PyTorch.Translation;

/// <summary>
/// The attributes a call passes to a <c>FunctionProto</c>, bound into its body: a node attribute
/// that refers to one of the function's (<c>ref_attr_name</c>) takes the value the call — or the
/// function's default — gives it, under the node attribute's own name, and is left off the node
/// where neither gives one, as ONNX specifies. References are resolved in the body's subgraphs
/// too, which read the function's attributes as its nodes do.
/// </summary>
internal static class FunctionAttributes
{
    /// <summary><paramref name="node"/> with its references to the function's attributes resolved
    /// from <paramref name="values"/>; the node itself where it has none.</summary>
    public static NodeProto Resolve(NodeProto node, IReadOnlyDictionary<string, AttributeProto> values)
    {
        if (!Refers(node)) return node;
        var copy = ProtoBuf.Serializer.DeepClone(node);
        ResolveInPlace(copy, values);
        return copy;
    }

    /// <summary>A string that is the same for two sets of attribute values exactly when they are
    /// the same values.</summary>
    public static string Key(IReadOnlyDictionary<string, AttributeProto> values)
    {
        using var stream = new MemoryStream();
        foreach (var name in values.Keys.Order(StringComparer.Ordinal))
            ProtoBuf.Serializer.SerializeWithLengthPrefix(stream, values[name], ProtoBuf.PrefixStyle.Base128);
        return Convert.ToBase64String(stream.ToArray());
    }

    private static bool Refers(NodeProto node)
        => node.Attributes.Any(a => a.RefAttrName.Length > 0
            || (a.G is { } g && g.Nodes.Any(Refers))
            || a.Graphs.Any(g => g.Nodes.Any(Refers)));

    private static void ResolveInPlace(NodeProto node, IReadOnlyDictionary<string, AttributeProto> values)
    {
        for (int i = node.Attributes.Count - 1; i >= 0; i--)
        {
            var attribute = node.Attributes[i];
            if (attribute.RefAttrName.Length > 0)
            {
                if (values.TryGetValue(attribute.RefAttrName, out var value))
                {
                    var bound = ProtoBuf.Serializer.DeepClone(value);
                    bound.Name = attribute.Name;
                    node.Attributes[i] = bound;
                }
                else
                {
                    node.Attributes.RemoveAt(i);
                }
                continue;
            }
            if (attribute.G is { } graph)
                foreach (var inner in graph.Nodes) ResolveInPlace(inner, values);
            foreach (var each in attribute.Graphs)
                foreach (var inner in each.Nodes) ResolveInPlace(inner, values);
        }
    }
}
