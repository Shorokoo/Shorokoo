using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// Rewrites, in a freshly parsed model, the standard-domain nodes written at an opset
    /// before one of their operands moved from an attribute to an input, into the form the
    /// importer's operator definitions (the current opset) expect: the attribute becomes a
    /// <c>Constant</c> feeding the new input. Without it such a node keeps its old arity and
    /// the model fails to load in the runtime ("input size 1 not in range [min=2, max=2]").
    /// </summary>
    /// <remarks>
    /// Covered: <c>axes</c> of Squeeze and Unsqueeze and ReduceSum (inputs since opset 13) and of
    /// the other reductions (since 18); <c>split</c> of Split (since 13), and a Split with no
    /// sizes at all given <c>num_outputs</c> (required since 18). Other operands that moved the
    /// same way in older opsets (Slice before 10, Pad and Clip before 11, …) are not rewritten
    /// yet; see <c>Documentation/limitations.md</c>.
    /// </remarks>
    internal static class OnnxLegacyOperands
    {
        // The opset from which each operand is an input rather than an attribute.
        private static readonly Dictionary<string, (string Attribute, long InputSince)> Moved = new()
        {
            ["Squeeze"] = ("axes", 13),
            ["Unsqueeze"] = ("axes", 13),
            ["ReduceSum"] = ("axes", 13),
            ["ReduceMean"] = ("axes", 18),
            ["ReduceMax"] = ("axes", 18),
            ["ReduceMin"] = ("axes", 18),
            ["ReduceProd"] = ("axes", 18),
            ["ReduceL1"] = ("axes", 18),
            ["ReduceL2"] = ("axes", 18),
            ["ReduceLogSum"] = ("axes", 18),
            ["ReduceLogSumExp"] = ("axes", 18),
            ["ReduceSumSquare"] = ("axes", 18),
            ["Split"] = ("split", 13),
        };

        private const long SplitNumOutputsSince = 18;

        /// <summary>Rewrites <paramref name="model"/> in place; a model at a current opset is left as it is.</summary>
        public static void Upgrade(ModelProto model)
        {
            long? modelOpset = StandardOpset(model.OpsetImports);
            var names = new HashSet<string>(StringComparer.Ordinal);
            if (model.Graph is { } g)
                CollectNames(g, names);
            foreach (var function in model.Functions)
            {
                names.UnionWith(function.Inputs);
                names.UnionWith(function.ValueInfoes.Select(v => v.Name));
                CollectNodeNames(function.Nodes, names);
            }
            if (model.Graph is { } graph && modelOpset is { } opset)
                UpgradeNodes(graph.Nodes, opset, names);
            foreach (var function in model.Functions)
                if ((StandardOpset(function.OpsetImports) ?? modelOpset) is { } functionOpset)
                    UpgradeNodes(function.Nodes, functionOpset, names);
        }

        private static long? StandardOpset(IEnumerable<OperatorSetIdProto> imports) =>
            imports.FirstOrDefault(i => string.IsNullOrEmpty(i.Domain) || i.Domain == "ai.onnx")?.Version;

        // Every tensor name the model uses anywhere (a subgraph may read its parent's), so a new
        // constant never takes one.
        private static void CollectNames(GraphProto graph, HashSet<string> names)
        {
            foreach (var input in graph.Inputs)
                names.Add(input.Name);
            foreach (var initializer in graph.Initializers)
                names.Add(initializer.Name);
            foreach (var info in graph.ValueInfoes)
                names.Add(info.Name);
            CollectNodeNames(graph.Nodes, names);
        }

        private static void CollectNodeNames(List<NodeProto> nodes, HashSet<string> names)
        {
            foreach (var node in nodes)
            {
                names.UnionWith(node.Outputs);
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } subgraph)
                        CollectNames(subgraph, names);
                    foreach (var g in attribute.Graphs)
                        CollectNames(g, names);
                }
            }
        }

        // Named after the node's first output, with a counter where that name is taken, so an
        // import stays deterministic.
        private static string FreshName(string stem, HashSet<string> names)
        {
            string name = stem;
            for (int n = 2; !names.Add(name); n++)
                name = $"{stem}_{n}";
            return name;
        }

        private static void UpgradeNodes(List<NodeProto> nodes, long opset, HashSet<string> names)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                foreach (var attribute in node.Attributes)
                {
                    if (attribute.G is { } subgraph)
                        UpgradeNodes(subgraph.Nodes, opset, names);
                    foreach (var g in attribute.Graphs)
                        UpgradeNodes(g.Nodes, opset, names);
                }
                if (!string.IsNullOrEmpty(node.Domain) && node.Domain != "ai.onnx" || !Moved.TryGetValue(node.OpType, out var moved))
                    continue;

                if (opset < moved.InputSince && node.Attributes.FirstOrDefault(a => a.Name == moved.Attribute) is { } operand)
                {
                    string name = FreshName($"{node.Outputs[0]}__{moved.Attribute}", names);
                    // The operand is the node's second input; the first is always present.
                    while (node.Inputs.Count < 2)
                        node.Inputs.Add("");
                    node.Inputs[1] = name;
                    node.Attributes.Remove(operand);
                    // A single INT is how some exporters wrote a one-axis list; read it as one.
                    long[] values = operand.Type == AttributeProto.AttributeType.Int ? [operand.I] : operand.Ints ?? [];
                    nodes.Insert(i, Constant(name, values));
                    i++;
                }
                else if (node.OpType == "Split" && opset < SplitNumOutputsSince
                         && (node.Inputs.Count < 2 || node.Inputs[1] == "")
                         && !node.Attributes.Any(a => a.Name == "num_outputs"))
                {
                    // Equal parts, as many as the node has outputs: said by num_outputs since 18.
                    node.Attributes.Add(new AttributeProto
                    {
                        Name = "num_outputs",
                        Type = AttributeProto.AttributeType.Int,
                        I = node.Outputs.Count,
                    });
                }
            }
        }

        private static NodeProto Constant(string output, long[] values)
        {
            var node = new NodeProto { OpType = "Constant", Name = output };
            node.Outputs.Add(output);
            node.Attributes.Add(new AttributeProto
            {
                Name = "value",
                Type = AttributeProto.AttributeType.Tensor,
                T = new TensorProto
                {
                    Name = output,
                    data_type = (int)TensorProto.DataType.Int64,
                    Dims = [values.Length],
                    RawData = [.. values.SelectMany(BitConverter.GetBytes)],
                },
            });
            return node;
        }
    }
}
