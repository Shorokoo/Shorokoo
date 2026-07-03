using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Binds an <see cref="RngConfig"/> to a concrete model's runtime random feeds: every
    /// top-level <c>SHRK_RANDOM_UNIFORM/NORMAL</c> site (id-bearing since creation, its
    /// ModelId made absolute by module inlining) is rewritten to the keyed
    /// <c>SHRK_RNG_UNIFORM/NORMAL</c> draw whose stream key is the runtime master folded
    /// host-side along the feed's ModelId path — the RNG key tree IS the ModelId tree, so
    /// distinct feed sites decorrelate and a feed's key is reconstructible offline from its
    /// ModelId alone. drawBase is a constant 0 (per-step channel not yet wired).
    ///
    /// <para>Feeds inside loops are left untouched (they fall through to the ONNX
    /// random-op lowering): a loop body's node executes once per iteration with the same
    /// key and drawBase, which under the keyed path would repeat the same values every
    /// iteration; the ONNX fallback preserves fresh-per-iteration draws until per-iteration
    /// key splitting is plumbed.</para>
    /// </summary>
    internal static class FastApplyRngKeys
    {
        public static void Process(FastComputationGraph graph, RngConfig rngConfig)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            var newNodes = new List<FastNode>(graph.Nodes.Count);
            int loopDepth = 0;

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN) loopDepth++;
                else if (node.OpCode == OpCodes.LOOP_CLOSE) loopDepth--;

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if ((!isUniform && !isNormal) || loopDepth > 0)
                {
                    newNodes.Add(node);
                    continue;
                }

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                if (idVals is null || idVals.Length == 0)
                {
                    // Not id-bearing (e.g. built by a path that bypasses id assignment):
                    // leave for the ONNX fallback lowering.
                    newNodes.Add(node);
                    continue;
                }

                RewriteToKeyedDraw(node, isUniform, idVals, rngConfig, newNodes);
                newNodes.Add(node);
            }

            graph.Nodes = newNodes;
        }

        private static void RewriteToKeyedDraw(
            FastNode node, bool isUniform, int[] idVals, RngConfig rngConfig, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            var (k0, k1) = rngConfig.FoldRunKey(idVals);

            float a = isUniform
                ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
            float b = isUniform
                ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;

            var keyKey = AppendKeyConstant(k0, k1, newNodes);
            var drawBaseKey = AppendScalarInt64(0L, newNodes);
            var aKey = AppendScalarFloat32(a, newNodes);
            var bKey = AppendScalarFloat32(b, newNodes);

            var newOp = isUniform ? InternalOpCodes.SHRK_RNG_UNIFORM : InternalOpCodes.SHRK_RNG_NORMAL;
            var attrDefs = Definitions.NodeDefinitions[newOp].AttributeDefs;
            node.OpCode = newOp;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = RngAlgorithms.Default },
                attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { keyKey, drawBaseKey, shapeInput, aKey, bKey }
            };
        }

        private static FastTensorKey AppendKeyConstant(uint k0, uint k1, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(2),
                OnnxUtils.CreateTensorValue(new Shape(2), new long[] { k0, k1 }));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendScalarInt64(long value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), new[] { value }));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendScalarFloat32(float value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<float32>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), new[] { value }));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendConstant(TensorData data, List<FastNode> newNodes)
        {
            var constAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var key = FastNodeKey.New();
            var outKey = new FastTensorKey(key, 0);
            newNodes.Add(new FastNode
            {
                Key = key,
                OpCode = OpCodes.CONSTANT,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrValue] = data }, constAttrDefs),
                FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }
    }
}
