using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Fast version of the random-op lowering pass. Replaces every
    /// <c>SHRK_RANDOM_UNIFORM(shape)</c> with
    /// <c>ConstantOfShape(shape, 0f) → RandomUniformLike(placeholder)</c> and every
    /// <c>SHRK_RANDOM_NORMAL(shape)</c> with
    /// <c>ConstantOfShape(shape, 0f) → RandomNormalLike(placeholder)</c>. Random ops can
    /// appear both in the main graph and inside function bodies (e.g. TrainableParam
    /// initializers that call <c>Globals.RandomNormal</c>); both are lowered.
    /// </summary>
    internal static class FastLowerRandomOps
    {
        private static readonly TensorData ZeroScalar = new OnnxTensorData<float32>(
            new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { 0f }));

        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var functionRemap = new Dictionary<Function, Function>();

            ProcessGraph(graph, functionRemap);
        }

        private static void ProcessGraph(FastComputationGraph graph, Dictionary<Function, Function> functionRemap)
        {
            // Lower every Function reachable from this graph (post-order: leaves first).
            // Memoization on Function instance ensures shared Functions are lowered once.
            foreach (var node in graph.Nodes)
            {
                if (node.TargetFunction is { } fn)
                    LowerFunctionRecursive(fn, functionRemap);
            }

            // Lower main graph nodes by mutating SHRK_RANDOM_* in place and inserting a
            // new ConstantOfShape predecessor.
            var newNodes = new List<FastNode>(graph.Nodes.Count);
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM)
                {
                    LowerToRandomUniformLike(node, newNodes);
                }
                else if (node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL)
                {
                    LowerToRandomNormalLike(node, newNodes);
                }
                newNodes.Add(node);
            }
            graph.Nodes = newNodes;

            // Remap TargetFunction references on all nodes.
            if (functionRemap.Count > 0)
            {
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
            }
        }

        /// <summary>
        /// Lowers a Function's body in post-order. If the body has no random ops (directly or
        /// transitively via nested Functions), the original Function is reused.
        /// </summary>
        private static void LowerFunctionRecursive(Function fn, Dictionary<Function, Function> functionRemap)
        {
            if (functionRemap.ContainsKey(fn)) return;

            var bodyHasRandomOps = HasRandomOps(fn.OriginalFastGraph) ||
                                   fn.ReferencedFunctions.Any(x => HasRandomOps(x.OriginalFastGraph));
            if (!bodyHasRandomOps)
            {
                // Mark as visited (but not remapped) so we don't redo the check.
                functionRemap[fn] = fn;
                return;
            }

            // Clone the function's primary Fast body and lower in place,
            // recursively lowering nested Functions.
            var bodyFast = fn.OriginalFastGraph.Clone();
            ProcessGraph(bodyFast, functionRemap);

            var newFn = new Function(bodyFast, fn.FunctionType,
                defaultName: fn.DefaultName, friendlyName: fn.FriendlyName);
            functionRemap[fn] = newFn;
        }

        private static bool HasRandomOps(FastComputationGraph graph) =>
            graph.Nodes.Any(node =>
                node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL);

        private static void LowerToRandomUniformLike(FastNode node, List<FastNode> newNodes)
        {
            // SHRK_RANDOM_UNIFORM(shape) → ConstantOfShape(shape, 0f) → RandomUniformLike(placeholder)
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_UNIFORM has null shape input.");

            var placeholderKey = AppendConstantOfShape(shapeInput, newNodes);

            var dctAttrs = new Dictionary<string, object?>
            {
                [AttrHigh] = node.Attributes.GetFloatVal(AttrHigh),
                [AttrLow] = node.Attributes.GetFloatVal(AttrLow),
                [AttrSeed] = node.Attributes.GetFloatVal(AttrSeed),
            };
            var attrDefs = Definitions.NodeDefinitions[OpCodes.RANDOM_UNIFORM_LIKE].AttributeDefs;

            node.OpCode = OpCodes.RANDOM_UNIFORM_LIKE;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(dctAttrs, attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { placeholderKey }
            };
        }

        private static void LowerToRandomNormalLike(FastNode node, List<FastNode> newNodes)
        {
            // SHRK_RANDOM_NORMAL(shape) → ConstantOfShape(shape, 0f) → RandomNormalLike(placeholder)
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_NORMAL has null shape input.");

            var placeholderKey = AppendConstantOfShape(shapeInput, newNodes);

            var dctAttrs = new Dictionary<string, object?>
            {
                [AttrMean] = node.Attributes.GetFloatVal(AttrMean),
                [AttrScale] = node.Attributes.GetFloatVal(AttrScale),
                [AttrSeed] = node.Attributes.GetFloatVal(AttrSeed),
            };
            var attrDefs = Definitions.NodeDefinitions[OpCodes.RANDOM_NORMAL_LIKE].AttributeDefs;

            node.OpCode = OpCodes.RANDOM_NORMAL_LIKE;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(dctAttrs, attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { placeholderKey }
            };
        }

        private static FastTensorKey AppendConstantOfShape(FastTensorKey shapeInput, List<FastNode> newNodes)
        {
            var nodeKey = FastNodeKey.New();
            var outputKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT_OF_SHAPE].AttributeDefs;
            var attrs = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [AttrValue] = ZeroScalar },
                attrDefs);

            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.CONSTANT_OF_SHAPE,
                Attributes = attrs,
                FullInputs = { [""] = new List<FastTensorKey?> { shapeInput } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outputKey } },
            });

            return outputKey;
        }
    }
}
