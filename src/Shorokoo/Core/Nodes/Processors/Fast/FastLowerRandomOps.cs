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
    /// Random-op lowering pass. Rewrites every runtime <c>SHRK_RANDOM_UNIFORM(shape)</c> /
    /// <c>SHRK_RANDOM_NORMAL(shape)</c> into an in-graph counter-based draw
    /// (<see cref="RuntimeRng"/>): the body of the captured <see cref="RuntimeRngFunctions"/>
    /// function is spliced directly at the site, fed the draw shape plus constant Threefry key
    /// words, the per-execution counter high word (<c>drawBase</c>), and the distribution
    /// parameters (low/high or mean/scale). The result is ordinary integer/float math —
    /// deterministic and identical on every execution provider and in the Quick Execution Engine,
    /// and self-contained on ONNX export (unlike ONNX's <c>RandomUniformLike</c>).
    ///
    /// <para>Each site's key is derived from <see cref="RngConfig"/> and a per-site ordinal so
    /// distinct draw sites decorrelate (fixing the shared-seed correlation of the old lowering).
    /// The <c>drawBase</c> is currently a fixed 0; a per-execution counter is wired separately.</para>
    /// </summary>
    internal static class FastLowerRandomOps
    {
        private sealed class Ordinal { public int Value; }

        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var ordinal = new Ordinal();
            var functionRemap = new Dictionary<Function, Function>();
            ProcessGraph(graph, functionRemap, ordinal);
        }

        private static void ProcessGraph(
            FastComputationGraph graph, Dictionary<Function, Function> functionRemap, Ordinal ordinal)
        {
            // Lower every Function reachable from this graph first (memoized per Function instance).
            foreach (var node in graph.Nodes)
                if (node.TargetFunction is { } fn)
                    LowerFunctionRecursive(fn, functionRemap, ordinal);

            var newNodes = new List<FastNode>(graph.Nodes.Count);
            var outputRemap = new Dictionary<FastTensorKey, FastTensorKey>();

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL)
                {
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal)
                {
                    newNodes.Add(node);
                    continue;
                }
                SpliceRuntimeDraw(node, isUniform, newNodes, outputRemap, ordinal);
            }

            graph.Nodes = newNodes;

            ApplyOutputRemap(graph, outputRemap);

            if (functionRemap.Count > 0)
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
        }

        private static void LowerFunctionRecursive(
            Function fn, Dictionary<Function, Function> functionRemap, Ordinal ordinal)
        {
            if (functionRemap.ContainsKey(fn)) return;

            var bodyHasRandomOps = HasRandomOps(fn.OriginalFastGraph) ||
                                   fn.ReferencedFunctions.Any(x => HasRandomOps(x.OriginalFastGraph));
            if (!bodyHasRandomOps)
            {
                functionRemap[fn] = fn;   // visited, unchanged
                return;
            }

            var bodyFast = fn.OriginalFastGraph.Clone();
            ProcessGraph(bodyFast, functionRemap, ordinal);

            functionRemap[fn] = new Function(bodyFast, fn.FunctionType,
                defaultName: fn.DefaultName, friendlyName: fn.FriendlyName);
        }

        private static bool HasRandomOps(FastComputationGraph graph) =>
            graph.Nodes.Any(node =>
                node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL);

        /// <summary>
        /// Rewrites a keyed SHRK_RNG_* node in place to a FUNCTION_INVOKE of the named
        /// algorithm's function of the matching kind. The node inputs already match the
        /// function's input order 1:1; RNG algorithm functions are never inlined and export
        /// as ONNX local FunctionProtos, so the call survives to the ONNX model as a
        /// Functions-domain call node (see FastOpsetResolver).
        /// </summary>
        private static void LowerKeyedRngToFunctionCall(FastNode node)
        {
            var algorithm = node.Attributes.GetStringVal(ShrkAttrRngAlgorithm)
                ?? RngAlgorithms.Default;
            var (kind, dtype, rank) = node.OpCode switch
            {
                InternalOpCodes.SHRK_RNG_SPLIT => (RngAlgorithms.KindSplit, DType.Int64, 1L),
                InternalOpCodes.SHRK_RNG_UNIFORM => (RngAlgorithms.KindUniform, DType.Float32, -1L),
                _ => (RngAlgorithms.KindNormal, DType.Float32, -1L),
            };
            var fn = RngAlgorithms.GetFunction(algorithm, kind);

            var invokeAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.FUNCTION_INVOKE].AttributeDefs;
            node.OpCode = InternalOpCodes.FUNCTION_INVOKE;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrStructure] = new[] { DataStructure.Tensor },
                    [ShrkAttrDtype] = new[] { dtype },
                    [ShrkAttrRank] = new[] { rank },
                    [ShrkAttrGenericTypeArgs] = null,
                },
                invokeAttrDefs);
            node.IdentifierTemplate = null;
            node.TargetFunction = fn;
        }

        /// <summary>
        /// Splices the <see cref="RuntimeRng"/> function body at a single <c>SHRK_RANDOM_*</c> site:
        /// appends the constant key/drawBase/param inputs, clones + rekeys the function body, remaps
        /// its six inputs to the site's inputs, and records the site output → body output remap so
        /// downstream consumers pick up the drawn tensor.
        /// </summary>
        private static void SpliceRuntimeDraw(
            FastNode node, bool isUniform, List<FastNode> newNodes,
            Dictionary<FastTensorKey, FastTensorKey> outputRemap, Ordinal ordinal)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            // Per-site key: distinct sites decorrelate. drawBase fixed 0 for now.
            int site = ordinal.Value++;
            var (k0, k1) = RngConfig.Default.ResolveKey(RngCollection.Runtime, "site" + site);

            float a = isUniform
                ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
            float b = isUniform
                ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;

            FastTensorKey?[] callerInputs =
            [
                shapeInput,
                AppendScalarInt64((long)k0, newNodes),
                AppendScalarInt64((long)k1, newNodes),
                AppendScalarInt64(0L, newNodes),          // drawBase
                AppendScalarFloat32(a, newNodes),
                AppendScalarFloat32(b, newNodes),
            ];

            var fn = isUniform ? RuntimeRngFunctions.Uniform : RuntimeRngFunctions.Normal;
            var sub = fn.GetFastFlattenedGraph().Clone();
            FastProcessorHelper.RekeySubgraph(sub);

            var inputRemap = new Dictionary<FastTensorKey, FastTensorKey>();
            for (int i = 0; i < sub.Inputs.Count && i < callerInputs.Length; i++)
                if (callerInputs[i] is FastTensorKey caller)
                    inputRemap[sub.Inputs[i]] = caller;

            foreach (var subNode in sub.Nodes)
            {
                foreach (var kvp in subNode.FullInputs)
                {
                    var list = kvp.Value;
                    for (int j = 0; j < list.Count; j++)
                        if (list[j] is FastTensorKey key && inputRemap.TryGetValue(key, out var repl))
                            list[j] = repl;
                }
                newNodes.Add(subNode);
            }

            // The SHRK node's output feeds downstream consumers; redirect it to the body's output.
            var siteOutput = node.FullOutputs[""][0]!.Value;
            outputRemap[siteOutput] = sub.Outputs[0];
            // The SHRK node itself is dropped (not added to newNodes).
        }

        private static void ApplyOutputRemap(
            FastComputationGraph graph, Dictionary<FastTensorKey, FastTensorKey> outputRemap)
        {
            if (outputRemap.Count == 0) return;

            // Resolve transitive chains (a→b→c ⇒ a→c).
            foreach (var key in outputRemap.Keys.ToList())
            {
                var target = outputRemap[key];
                while (outputRemap.TryGetValue(target, out var next)) target = next;
                outputRemap[key] = target;
            }

            foreach (var node in graph.Nodes)
                foreach (var kvp in node.FullInputs)
                {
                    var list = kvp.Value;
                    for (int j = 0; j < list.Count; j++)
                        if (list[j] is FastTensorKey key && outputRemap.TryGetValue(key, out var repl))
                            list[j] = repl;
                }

            for (int i = 0; i < graph.Outputs.Count; i++)
                if (outputRemap.TryGetValue(graph.Outputs[i], out var repl))
                    graph.Outputs[i] = repl;
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
