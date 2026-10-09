using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Bakes into a session's model the integer values its graph computes from shapes alone, where
    /// the model is built for one set of input dimensions (<see cref="ConcreteValues"/>): each such
    /// value a node reads becomes a constant, and the arithmetic that computed it drops out of the
    /// model. The runtime would fold that arithmetic itself wherever its own shape inference follows
    /// every dimension, but it loses one at the first reshape to a computed shape, and on a GPU the
    /// arithmetic it keeps is a host computation the device's queue drains for, then copies back.
    ///
    /// <para>Two readings of a shape-decided value then do nothing at all, and are dropped: a
    /// <c>ReduceSum</c> over no axes under <c>noop_with_empty_axes</c>, or over axes that are each
    /// one long while it keeps them, and a <c>Reshape</c> to the shape its input already has. A
    /// broadcasting op's gradient makes both for every operand the op did not broadcast: it sums
    /// over the axes the operand was broadcast along, which it works out from the two shapes, then
    /// reshapes the sum to the operand's shape.</para>
    ///
    /// <para>A <c>Pad</c> whose pads and axes are both decided this way is given constant pads over
    /// every axis of its input and no axes (<see cref="PadsOverEveryAxis"/>).</para>
    ///
    /// <para>Only an <c>int64</c> value is baked, and only one small enough for the interpreter to
    /// hold (<see cref="AutoDiffCheckpointing.ShapeInferenceInterpreter.MaxSmallTensorElements"/>):
    /// shape arithmetic is integer arithmetic, and the interpreter computes it exactly. Only a node
    /// at the top level of the graph reads a baked value, and none that opens or closes a scope;
    /// a dropped node's result is passed on to the body nodes that read it, and one an output reads
    /// keeps the node, so an output never becomes an alias of another value.</para>
    /// </summary>
    internal static class FastBakeShapeArithmetic
    {
        public static void Process(InternalComputationGraph graph, ConcreteValues values) => Bake(graph, values, null);

        /// <summary>
        /// Bakes <paramref name="graph"/> as <see cref="Process"/> does, in a form <see cref="Reversal.Restore"/>
        /// turns back into a graph for every set of dimensions: the graph a session built for these
        /// dimensions runs, for a pass to rewrite before the rewrite is carried back to the graph it
        /// was given. Each baked value is a constant of its own, which stands for the value it
        /// replaces wherever a reader reads it, a copy of the reader included; a <c>ReduceSum</c>
        /// that does nothing is a <c>Reshape</c> to its input's shape, which a session's model drops
        /// as it drops the <c>ReduceSum</c>, rather than gone. Null where nothing is baked.
        /// </summary>
        public static Reversal? Reversibly(InternalComputationGraph graph, ConcreteValues values)
        {
            var reversal = new Reversal(graph.Clone());
            Bake(graph, values, reversal);
            return reversal.Constants.Count == 0 ? null : reversal;
        }

        private static void Bake(InternalComputationGraph graph, ConcreteValues values, Reversal? reversal)
        {
            var nodeByKey = graph.Nodes.ToDictionary(n => n.Key);
            var replacement = new Dictionary<FastTensorKey, FastTensorKey>();
            var baked = new Dictionary<FastTensorKey, FastTensorKey>();
            var constants = new List<FastNode>();
            int depth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.IsCloseNode()) depth--;
                bool topLevelBody = depth == 0 && !node.IsOpenNode() && !node.IsCloseNode()
                    && !InternalOpCodes.IsGraphOutputOp(node.OpCode) && node.OpCode != OpCodes.CONSTANT;
                if (node.IsOpenNode()) depth++;
                if (!topLevelBody) continue;

                if (Passes(node, values) is FastTensorKey passed && node.Outputs is [FastTensorKey result])
                {
                    if (reversal is null)
                    {
                        replacement[result] = passed;
                        continue;
                    }
                    if (node.OpCode == OpCodes.REDUCE_SUM && values.ShapeOf(passed) is { } passedShape)
                    {
                        var shapeNode = FastInternalOp.Constant(Int64Attribute(new Shape(passedShape.Dims.Length), [.. passedShape.Dims]));
                        constants.Add(shapeNode);
                        reversal.NoOps[shapeNode.Outputs[0]!.Value] = Copy(node);
                        node.OpCode = OpCodes.RESHAPE;
                        node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                            new Dictionary<string, object?>(), Definitions.NodeDefinitions[OpCodes.RESHAPE].AttributeDefs);
                        node.FullInputs = new Dictionary<string, List<FastTensorKey?>> { [""] = [passed, shapeNode.Outputs[0]] };
                        continue;
                    }
                }
                if (node.Outputs.All(o => o is not FastTensorKey k || IntegerValue(values, k) is not null))
                    continue;
                if (PadsOverEveryAxis(node, nodeByKey, values) is { } pads)
                {
                    var padsNode = FastInternalOp.Constant(Int64Attribute(new Shape(pads.Length), pads));
                    constants.Add(padsNode);
                    var slots = node.FullInputs[""];
                    reversal?.Pads.Add(padsNode.Outputs[0]!.Value, (slots[1], slots[3]));
                    slots[1] = padsNode.Outputs[0];
                    slots[3] = null;
                    continue;
                }
                foreach (var slots in node.FullInputs.Values)
                    for (int i = 0; i < slots.Count; i++)
                    {
                        if (slots[i] is not FastTensorKey key || IsConstant(nodeByKey, key)
                            || IntegerValue(values, key) is not { } value)
                            continue;
                        if (!baked.TryGetValue(key, out var constant))
                        {
                            var constantNode = FastInternalOp.Constant(Int64Attribute(value.Shape!, [.. value.IntData!.Value]));
                            constants.Add(constantNode);
                            baked[key] = constant = constantNode.Outputs[0]!.Value;
                            reversal?.Values.Add(constant, key);
                        }
                        slots[i] = constant;
                    }
            }
            if (replacement.Count == 0 && constants.Count == 0) return;
            reversal?.Constants.UnionWith(constants.Select(c => c.Key));

            var outputKeys = graph.OutputNodes.Select(n => n.Key).ToHashSet();
            foreach (var node in graph.Nodes)
            {
                if (outputKeys.Contains(node.Key)) continue;
                foreach (var slots in node.FullInputs.Values)
                    for (int i = 0; i < slots.Count; i++)
                        if (slots[i] is FastTensorKey k && Resolve(replacement, k) is var x && x != k)
                            slots[i] = x;
            }
            graph.InsertAtBodyStart(constants);
            FastProcessorHelper.RemoveUnreachableNodes(graph);
        }

        /// <summary>The value <paramref name="node"/> passes on unchanged, where its shape-decided
        /// inputs make it a no-op; null otherwise.</summary>
        private static FastTensorKey? Passes(FastNode node, ConcreteValues values)
        {
            if (node.OpCode == OpCodes.REDUCE_SUM && node.Inputs is [FastTensorKey data, FastTensorKey axesKey]
                && IntegerValue(values, axesKey) is { } axes && values.ShapeOf(data) is { } dataShape)
            {
                var attrs = node.Attributes;
                if (axes.Shape!.Count == 0)
                    return AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrNoopWithEmptyAxes, false) ? data : null;
                int rank = dataShape.Dims.Length;
                return AttrAccess.GetBool(attrs, OnnxOpAttributeNames.AttrKeepdims, true)
                    && axes.IntData!.Value.All(a => a >= -rank && a < rank && dataShape.Dims[a < 0 ? a + rank : a] == 1)
                    ? data : null;
            }
            if (node.OpCode == OpCodes.RESHAPE && node.Inputs is [FastTensorKey input, FastTensorKey shapeKey]
                && IntegerValue(values, shapeKey) is not null
                && node.Outputs is [FastTensorKey output]
                && values.ShapeOf(input) is { } from && values.ShapeOf(output) is { } to
                && from.Dims.SequenceEqual(to.Dims))
                return input;
            return null;
        }

        /// <summary>
        /// The pads of a <c>Pad</c> whose pads and axes its inputs' shapes decide, at least one of
        /// them computed, stated over every axis of its input: two zeros for each axis it does not
        /// name, so the <c>Pad</c> reads no axes. Constant pads beside constant axes make a
        /// <c>Pad</c> ONNX Runtime's CUDA kernel fails on at run time, counting the pads against
        /// every axis of the input. Null for any other node.
        /// </summary>
        private static long[]? PadsOverEveryAxis(FastNode node, Dictionary<FastNodeKey, FastNode> nodeByKey, ConcreteValues values)
        {
            if (node.OpCode != OpCodes.PAD || !node.FullInputs.TryGetValue("", out var slots) || slots.Count != 4
                || slots[0] is not FastTensorKey data || slots[1] is not FastTensorKey padsKey || slots[3] is not FastTensorKey axesKey
                || IsConstant(nodeByKey, padsKey) && IsConstant(nodeByKey, axesKey)
                || values.ShapeOf(data) is not { } shape
                || IntegerValue(values, padsKey) is not { IntData: { } pads }
                || IntegerValue(values, axesKey) is not { IntData: { } axes }
                || pads.Length != 2 * axes.Length)
                return null;
            int rank = shape.Dims.Length;
            var full = new long[2 * rank];
            for (int i = 0; i < axes.Length; i++)
            {
                if (axes[i] < -rank || axes[i] >= rank) return null;
                int axis = (int)(axes[i] < 0 ? axes[i] + rank : axes[i]);
                full[axis] = pads[i];
                full[rank + axis] = pads[axes.Length + i];
            }
            return full;
        }

        private static FastNode Copy(FastNode node)
        {
            var copy = new FastNode
            {
                Key = node.Key,
                OpCode = node.OpCode,
                Attributes = node.Attributes,
                FriendlyName = node.FriendlyName,
                CallStack = node.CallStack,
                GraphOpenNodeKey = node.GraphOpenNodeKey,
                IdentifierTemplate = node.IdentifierTemplate,
                TargetFunction = node.TargetFunction,
            };
            foreach (var (name, slot) in node.FullInputs) copy.FullInputs[name] = [.. slot];
            foreach (var (name, slot) in node.FullOutputs) copy.FullOutputs[name] = [.. slot];
            return copy;
        }

        /// <summary>
        /// What <see cref="Reversibly"/> baked, and the graph it baked, so that a rewrite of the baked
        /// graph — its nodes reordered, some recomputed by copies that read what they read — can be
        /// carried back to a graph for every set of dimensions (<see cref="Restore"/>).
        /// </summary>
        internal sealed class Reversal
        {
            internal Reversal(InternalComputationGraph original) => Original = original;

            /// <summary>The graph as it was before the bake.</summary>
            internal InternalComputationGraph Original { get; }

            /// <summary>The nodes the bake added: every constant below.</summary>
            internal HashSet<FastNodeKey> Constants { get; } = [];

            /// <summary>Each baked constant, with the value it stands for. A rewrite never copies one:
            /// a constant is resident for the whole step, so nothing recomputes it.</summary>
            internal Dictionary<FastTensorKey, FastTensorKey> Values { get; } = [];

            /// <summary>Each constant a <c>Pad</c> was given as pads over every axis, with the pads and
            /// axes it read.</summary>
            internal Dictionary<FastTensorKey, (FastTensorKey? Pads, FastTensorKey? Axes)> Pads { get; } = [];

            /// <summary>Each shape a <c>ReduceSum</c> that does nothing was made a <c>Reshape</c> to, with
            /// that <c>ReduceSum</c>.</summary>
            internal Dictionary<FastTensorKey, FastNode> NoOps { get; } = [];

            /// <summary>
            /// <paramref name="baked"/>, a rewrite of the baked graph, for every set of dimensions:
            /// each node reading a baked value, a copy of one included, reads the value again; each
            /// <c>Reshape</c> standing for a <c>ReduceSum</c> is that <c>ReduceSum</c> again, over its
            /// own input; and each node computing a value read again, which the bake let go, is put
            /// back just before its first reader, in the order the original graph had them. Every
            /// other node keeps its place, so a session built for the baked dimensions bakes it back
            /// into the graph it was rewritten as. A value put back that reads only the shape of one
            /// computed after it reads a copy of that value instead, computed just before it. Null
            /// where a value put back is computed in a scope or cannot otherwise be put before its
            /// first reader.
            /// </summary>
            internal InternalComputationGraph? Restore(InternalComputationGraph baked)
            {
                var restored = baked.Clone();
                var nodes = new List<FastNode>(restored.Nodes.Count);
                foreach (var node in restored.Nodes)
                {
                    if (Constants.Contains(node.Key)) continue;
                    if (node.FullInputs.TryGetValue("", out var slots))
                    {
                        if (node.OpCode == OpCodes.RESHAPE && slots is [FastTensorKey data, FastTensorKey shape] && NoOps.TryGetValue(shape, out var noOp))
                        {
                            node.OpCode = noOp.OpCode;
                            node.Attributes = noOp.Attributes;
                            node.FullInputs = noOp.FullInputs.ToDictionary(e => e.Key, e => new List<FastTensorKey?>(e.Value));
                            node.FullInputs[""][0] = data;
                        }
                        else if (node.OpCode == OpCodes.PAD && slots.Count == 4 && slots[1] is FastTensorKey pads && Pads.TryGetValue(pads, out var read))
                            (slots[1], slots[3]) = read;
                    }
                    foreach (var slot in node.FullInputs.Values)
                        for (int i = 0; i < slot.Count; i++)
                            if (slot[i] is FastTensorKey key && Values.TryGetValue(key, out var value))
                                slot[i] = value;
                    nodes.Add(node);
                }

                var produced = nodes.SelectMany(n => n.Outputs).OfType<FastTensorKey>().ToHashSet();
                var producerOf = new Dictionary<FastTensorKey, (FastNode Node, int Position)>();
                for (int i = 0; i < Original.Nodes.Count; i++)
                    foreach (var output in Original.Nodes[i].Outputs)
                        if (output is FastTensorKey key) producerOf[key] = (Original.Nodes[i], i);
                var putBack = new List<(FastNode Node, int Position)>();
                var pending = new Stack<FastTensorKey>(nodes.SelectMany(n => n.Inputs).OfType<FastTensorKey>());
                while (pending.Count > 0)
                {
                    var key = pending.Pop();
                    if (produced.Contains(key)) continue;
                    if (!producerOf.TryGetValue(key, out var producer) || producer.Node.GraphOpenNodeKey is not null
                        || producer.Node.IsOpenNode() || producer.Node.IsCloseNode() || InternalOpCodes.IsModelInputOp(producer.Node.OpCode))
                        return null;
                    putBack.Add((Copy(producer.Node), producer.Position));
                    produced.UnionWith(producer.Node.Outputs.OfType<FastTensorKey>());
                    foreach (var input in producer.Node.Inputs)
                        if (input is FastTensorKey read) pending.Push(read);
                }
                foreach (var (node, _) in putBack.OrderByDescending(p => p.Position))
                {
                    var outputs = node.Outputs.OfType<FastTensorKey>().ToHashSet();
                    var at = nodes.FindIndex(n => n.Inputs.Any(i => i is FastTensorKey k && outputs.Contains(k)));
                    if (at < 0) at = nodes.Count;
                    var copies = new List<FastNode>();
                    if (node.OpCode is OpCodes.SHAPE or OpCodes.SIZE && node.FullInputs.TryGetValue("", out var read) && read is [FastTensorKey of, ..])
                    {
                        var position = new Dictionary<FastTensorKey, int>();
                        for (int i = 0; i < nodes.Count; i++)
                            foreach (var output in nodes[i].Outputs)
                                if (output is FastTensorKey key) position[key] = i;
                        if (CopiedBefore(of, at, nodes, position, copies, []) is not { } copy) return null;
                        read[0] = copy;
                    }
                    nodes.InsertRange(at, [.. copies, node]);
                }

                restored.Nodes = nodes;
                restored.SetInputs(baked.Inputs);
                restored.MoveOutputsToEnd();
                return restored.IsLinearOrderValid() ? restored : null;
            }
        }

        /// <summary>
        /// <paramref name="key"/> where it is computed before <paramref name="at"/> in
        /// <paramref name="nodes"/>, and otherwise a copy of it, computed by copies of the nodes that
        /// compute it after <paramref name="at"/>, which are added to <paramref name="copies"/> in
        /// order: for a reader of its shape alone, which a draw's copy gives as well as the draw.
        /// Null where one of those nodes is a scope's.
        /// </summary>
        private static FastTensorKey? CopiedBefore(FastTensorKey key, int at, List<FastNode> nodes, Dictionary<FastTensorKey, int> position,
            List<FastNode> copies, Dictionary<FastTensorKey, FastTensorKey> copied)
        {
            if (!position.TryGetValue(key, out var index) || index < at) return key;
            if (copied.TryGetValue(key, out var done)) return done;
            var producer = nodes[index];
            if (producer.GraphOpenNodeKey is not null || producer.IsOpenNode() || producer.IsCloseNode()) return null;
            var copy = Copy(producer);
            copy.Key = FastNodeKey.New();
            copy.Attributes = AutoDiffCheckpointing.CheckpointSegment.Strip(copy.Attributes);
            foreach (var slot in copy.FullInputs.Values)
                for (int i = 0; i < slot.Count; i++)
                    if (slot[i] is FastTensorKey input)
                    {
                        if (CopiedBefore(input, at, nodes, position, copies, copied) is not { } before) return null;
                        slot[i] = before;
                    }
            foreach (var slot in copy.FullOutputs.Values)
                for (int i = 0; i < slot.Count; i++)
                    if (slot[i] is FastTensorKey { IsEmpty: false } output)
                        slot[i] = copied[output] = new FastTensorKey(copy.Key, output.OutputIndex);
            copies.Add(copy);
            return copied[key];
        }

        private static TensorAttribute Int64Attribute(Shape shape, long[] elements)
            => TensorAttribute.Create(shape, DType.Int64, elements.SelectMany(System.BitConverter.GetBytes).ToArray());

        private static FastTensorKey Resolve(Dictionary<FastTensorKey, FastTensorKey> replacement, FastTensorKey key)
        {
            while (replacement.TryGetValue(key, out var next)) key = next;
            return key;
        }

        private static bool IsConstant(Dictionary<FastNodeKey, FastNode> nodeByKey, FastTensorKey key)
            => nodeByKey.TryGetValue(key.FastNodeKey, out var producer) && producer.OpCode == OpCodes.CONSTANT;

        /// <summary>The <c>int64</c> value <paramref name="key"/> holds at the stated dimensions,
        /// where the interpreter worked it out; null otherwise.</summary>
        private static RuntimeTensor? IntegerValue(ConcreteValues values, FastTensorKey key)
            => values.ValueOf(key) is { DType: var dtype, Shape: { } shape, IntData: { } data } value
                && dtype == DType.Int64 && shape.Dims.All(d => d >= 0) && data.Length == shape.Count
                    ? value : null;
    }
}
