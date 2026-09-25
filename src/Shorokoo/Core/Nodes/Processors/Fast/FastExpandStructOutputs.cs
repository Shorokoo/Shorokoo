using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Expands every <see cref="InternalComputationGraph.Outputs"/> entry whose producing node
    /// emits a TensorStruct value (i.e. its <c>AttrDtype</c> has a non-null
    /// <see cref="DType.TensorStructDef"/>) into one TENSOR_STRUCT_GETFIELD output per
    /// field of the struct, in field-order.
    ///
    /// <para>
    /// Mirrors the CG-side <c>ExpandStructOutputs</c> helpers used by training-graph lowering.
    /// Mutates <c>graph</c> in place: appends new GETFIELD nodes to
    /// the end of the body and replaces the struct's output node with one output node per field.
    /// </para>
    /// </summary>
    internal static class FastExpandStructOutputs
    {
        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Build a producer-node lookup so we can read each output's dtype.
            var producerByOutput = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in graph.Nodes)
            {
                foreach (var (_, outs) in node.FullOutputs)
                {
                    foreach (var ok in outs)
                    {
                        if (ok is FastTensorKey k && !k.IsEmpty)
                            producerByOutput[k] = node;
                    }
                }
            }

            var newOutputNodes = new List<FastNode>(graph.OutputCount);
            var newNodes = new List<FastNode>();
            bool changed = false;

            foreach (var outputNode in graph.OutputNodes)
            {
                var outKey = InternalComputationGraph.OutputKeyOf(outputNode);
                if (!producerByOutput.TryGetValue(outKey, out var producer))
                {
                    // No matching producer (unlikely in a well-formed graph); leave as is.
                    newOutputNodes.Add(outputNode);
                    continue;
                }

                var structDef = ReadStructDefIfAny(producer);
                if (structDef is null)
                {
                    newOutputNodes.Add(outputNode);
                    continue;
                }

                // The struct output gives way to one output per field.
                changed = true;
                foreach (var fieldDef in structDef.Fields)
                {
                    var getField = FastInternalOp.TensorStructGetField(
                        outKey, fieldDef.Name, fieldDef.ElementType, fieldDef.Rank, fieldDef.Structure);
                    newNodes.Add(getField);
                    newOutputNodes.Add(InternalComputationGraph.NewOutputNode(new FastTensorKey(getField.Key, 0)));
                }
            }

            if (!changed) return;

            graph.InsertAtBodyEnd(newNodes);
            graph.Nodes.RemoveRange(graph.BodyEnd, graph.OutputCount);
            graph.Nodes.AddRange(newOutputNodes);
        }

        /// <summary>
        /// Returns the producer's TensorStructDef if it produces a struct value; null otherwise.
        /// Both TENSOR_STRUCT_CREATE and MODEL_TENSORSTRUCT_INPUT carry their struct dtype on
        /// <see cref="OnnxOpAttributeNames.AttrDtype"/>; TENSOR_STRUCT_GETFIELD with a struct-typed
        /// field carries it on <see cref="OnnxOpAttributeNames.ShrkAttrDtype"/>.
        ///
        /// <para>
        /// Producer ops that don't declare the relevant attribute in their schema (e.g. the
        /// loss output's ReduceSum) are not struct producers, so we treat that as "no struct"
        /// rather than letting <see cref="OnnxCSharpAttributes.GetDTypeVal"/> throw.
        /// </para>
        /// </summary>
        private static TensorStructDef? ReadStructDefIfAny(FastNode producer)
        {
            string attrName = producer.OpCode == InternalOpCodes.TENSOR_STRUCT_GETFIELD
                ? OnnxOpAttributeNames.ShrkAttrDtype
                : OnnxOpAttributeNames.AttrDtype;

            if (!producer.Attributes.IsAttributeDefined(attrName))
                return null;

            return producer.Attributes.GetDTypeVal(attrName)?.TensorStructDef;
        }
    }
}
