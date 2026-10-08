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
        public static void Process(InternalComputationGraph graph, ConcreteValues values)
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
                    replacement[result] = passed;
                    continue;
                }
                if (node.Outputs.All(o => o is not FastTensorKey k || IntegerValue(values, k) is not null))
                    continue;
                if (PadsOverEveryAxis(node, nodeByKey, values) is { } pads)
                {
                    var padsNode = FastInternalOp.Constant(Int64Attribute(new Shape(pads.Length), pads));
                    constants.Add(padsNode);
                    var slots = node.FullInputs[""];
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
                        }
                        slots[i] = constant;
                    }
            }
            if (replacement.Count == 0 && constants.Count == 0) return;

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
