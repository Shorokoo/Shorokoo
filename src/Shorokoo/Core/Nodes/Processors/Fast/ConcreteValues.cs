using System;
using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// What the values of a graph are when it runs on inputs of stated dimensions, as far as those
    /// dimensions alone decide them: Shorokoo's interpreter run over the graph with every input
    /// described by its dimensions and dtype and no data. A value's shape is known where the
    /// dimensions decide it, and its elements where it is small and computed from shapes and
    /// constants alone; nothing computed from an input's data is.
    ///
    /// <para>Every value computed from a scope's result the interpreter estimated rather than
    /// worked out is left out: an <c>If</c> it could not tell the branch of, and a <c>Loop</c> it
    /// could not tell the end of. The interpreter gives each a shape all the same — the one
    /// branch's it knows, the shape a loop carries after the few iterations it walks — which a run
    /// can contradict.</para>
    ///
    /// <para>Read off the graph before the pre-passes rewrite it: they keep each value they do not
    /// replace under its key, and a value they add is not known here.</para>
    /// </summary>
    internal sealed class ConcreteValues
    {
        private readonly Dictionary<FastTensorKey, IRuntimeTensor> _values;
        private readonly HashSet<FastTensorKey> _estimated;

        private ConcreteValues(Dictionary<FastTensorKey, IRuntimeTensor> values, HashSet<FastTensorKey> estimated)
        {
            _values = values;
            _estimated = estimated;
        }

        /// <summary>
        /// The values of <paramref name="graph"/> at <paramref name="inputDims"/>, the dimensions
        /// of its inputs, positionally. Null where they do not state one for every input.
        /// </summary>
        public static ConcreteValues? At(InternalComputationGraph graph, IReadOnlyList<long[]?>? inputDims)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            var inputNodes = graph.InputNodes;
            if (inputDims is null || inputDims.Any(d => d is null) || inputNodes.Count != inputDims.Count) return null;
            var initial = new Dictionary<FastTensorKey, IRuntimeTensor>();
            for (int i = 0; i < inputNodes.Count; i++)
                if (inputNodes[i].OpCode == InternalOpCodes.MODEL_TENSOR_INPUT
                    && inputNodes[i].Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) is { IsGenericType: false } dtype)
                    initial[InternalComputationGraph.InputKeyOf(inputNodes[i])] = RuntimeTensorFactory.Create(dtype, new Shape(inputDims[i]!));

            var values = new QuickExecutionEngine().Run(graph, initial);
            return new ConcreteValues(values, Estimated(graph, values));
        }

        /// <summary>The value <paramref name="key"/> holds, unless it was estimated.</summary>
        public RuntimeTensor? ValueOf(FastTensorKey key)
            => !_estimated.Contains(key) && _values.TryGetValue(key, out var value) ? value as RuntimeTensor : null;

        /// <summary>The shape of <paramref name="key"/>, where every dimension of it is known.</summary>
        public Shape? ShapeOf(FastTensorKey key)
            => ValueOf(key) is { Shape: { } shape } && shape.Dims.All(d => d >= 0) ? shape : null;

        /// <summary>Every value's shape that is known in full, by value.</summary>
        public Dictionary<FastTensorKey, Shape> Shapes()
        {
            var shapes = new Dictionary<FastTensorKey, Shape>(_values.Count);
            foreach (var key in _values.Keys)
                if (ShapeOf(key) is { } shape)
                    shapes[key] = shape;
            return shapes;
        }

        /// <summary>
        /// The values of <paramref name="graph"/> whose shape in <paramref name="values"/> the
        /// interpreter estimated: each output of a scope it did not work out, and every value
        /// computed from one. An <c>If</c> is worked out where its condition's value is known
        /// and the branch that holds gives the output; a <c>Loop</c> where its trip count is
        /// known and each condition it has is too, so its walk ran to the loop's end. Any other
        /// scope is taken as estimated.
        /// </summary>
        private static HashSet<FastTensorKey> Estimated(
            InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> values)
        {
            var nodeByKey = new Dictionary<FastNodeKey, FastNode>(graph.Nodes.Count);
            foreach (var node in graph.Nodes) nodeByKey[node.Key] = node;
            var estimated = new HashSet<FastTensorKey>();

            bool Known(FastTensorKey? key, Func<RuntimeTensor, bool> hasValue)
                => key is { IsEmpty: false } k && values.TryGetValue(k, out var v) && v is RuntimeTensor t && hasValue(t);
            static bool IsBool(RuntimeTensor t) => t.BoolData is { Length: > 0 };
            static bool IsInt(RuntimeTensor t) => t.IntData is { Length: > 0 };
            bool AbsentOrBool(IReadOnlyList<FastTensorKey?> keys, int slot)
                => slot >= keys.Count || keys[slot] is not { IsEmpty: false } || Known(keys[slot], IsBool);

            foreach (var node in graph.Nodes)
            {
                var outputs = node.Outputs;
                if (node.Inputs.Any(k => k is { } key && estimated.Contains(key)))
                {
                    foreach (var key in outputs) if (key is { } k) estimated.Add(k);
                    continue;
                }
                if (!FastOpsetResolver.IsCloseOpCode(node.OpCode)) continue;

                var open = node.GraphOpenNodeKey is { } openKey && nodeByKey.TryGetValue(openKey, out var o) ? o.Inputs : null;
                if (node.OpCode == OpCodes.IF_CLOSE && open is [{ } condition, ..]
                    && values.TryGetValue(condition, out var c) && c is RuntimeTensor { BoolData: { Length: > 0 } taken })
                {
                    var branch = node.FullInputs.TryGetValue(taken[0] ? OnnxOpAttributeNames.AttrThenBranch : OnnxOpAttributeNames.AttrElseBranch, out var b) ? b : [];
                    for (int i = 0; i < outputs.Count; i++)
                        if (outputs[i] is { } key && (i >= branch.Count || branch[i] is not { } from
                            || estimated.Contains(from) || !values.ContainsKey(from)))
                            estimated.Add(key);
                    continue;
                }
                if (node.OpCode == OpCodes.LOOP_CLOSE && open is { Count: >= 2 }
                    && Known(open[0], IsInt) && AbsentOrBool(open, 1) && AbsentOrBool(node.Inputs, 0))
                    continue;
                foreach (var key in outputs) if (key is { } k) estimated.Add(k);
            }
            return estimated;
        }
    }
}
