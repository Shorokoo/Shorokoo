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
    /// Random-op lowering pass (runs in the ONNX-export pre-passes).
    ///
    /// <para><b>Keyed feeds.</b> When the graph carries a <c>SHRK_RNG_KEY_VECTOR</c> (written
    /// by <see cref="FastBindRngConfig"/> when an <see cref="RngConfig"/> is bound), it is the
    /// single source of truth: this pass decodes it ONCE (<see cref="RngConfig.FromKeyVector"/>)
    /// and derives every id-bearing <c>SHRK_RANDOM_*</c> feed's keys HERE — and only here —
    /// from the carrier plus the feed's structural attributes (ModelId + realized stream ids).
    /// A feed outside any loop draws from its key materialized as an int64[2] constant; a
    /// loop-body feed draws from a dense per-stream key table ([N, 2] constant over the feed's
    /// enumerated iteration space, one row per grid cell) selected by the runtime iteration
    /// index — realized streams are static and individually addressable per the concreteness
    /// contract, so there is nothing to derive in-graph. Deferring key derivation to ONNX prep
    /// is what makes config binding cheap and re-bindable: the pre-export graph keeps its
    /// structure and the derivation happens on the export clone.</para>
    ///
    /// <para><b>Keyed draws</b> (<c>SHRK_RNG_SPLIT/UNIFORM/NORMAL</c>) lower to
    /// FUNCTION_INVOKEs of the named algorithm's non-inlined functions — the exported model
    /// calls tagged local FunctionProtos, so its randomness is deterministic, portable, and
    /// identifiable.</para>
    ///
    /// <para><b>Unkeyed draws</b> (no carrier — no config bound — or feeds without stream
    /// identity) take the ONNX fallback: <c>ConstantOfShape +
    /// RandomUniformLike/RandomNormalLike</c> — the pre-keyed-RNG behavior, non-reproducible
    /// by nature, with any user seed copied through and none synthesized.</para>
    /// </summary>
    internal static class FastLowerRandomOps
    {
        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var functionRemap = new Dictionary<Function, Function>();
            ProcessGraph(graph, functionRemap);
        }

        private static void ProcessGraph(
            FastComputationGraph graph, Dictionary<Function, Function> functionRemap)
        {
            // Lower every Function reachable from this graph first (memoized per Function instance).
            foreach (var node in graph.Nodes)
                if (node.TargetFunction is { } fn)
                    LowerFunctionRecursive(fn, functionRemap);

            // The graph's RNG identity, when bound: decode the carrier once into the config
            // view all key derivation below reads. Function bodies never carry a carrier, so
            // their feeds (e.g. inside un-run initializer functions) take the ONNX fallback.
            var carrier = graph.Nodes.FirstOrDefault(
                n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
            RngConfig? keyConfig = null;
            string algorithm = RngAlgorithms.Default;
            if (carrier?.Attributes.GetTensorVal(AttrValue) is { } carrierData)
            {
                keyConfig = RngConfig.FromKeyVector(
                    carrierData.As<int64>().AccessMemory().ToArray());
                algorithm = carrier.Attributes.GetStringVal(ShrkAttrRngAlgorithm)
                    ?? RngAlgorithms.Default;
            }

            var newNodes = new List<FastNode>(graph.Nodes.Count);

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

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                if (keyConfig is not null && idVals is { Length: > 0 })
                {
                    RewriteFeedToKeyedDraw(node, isUniform, idVals, keyConfig, algorithm, newNodes);
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                // Unkeyed random op (no RngConfig bound, or a feed without stream identity):
                // the ONNX fallback — ConstantOfShape + RandomUniformLike/NormalLike.
                // Non-portable/non-reproducible by nature; documented as the no-config behavior.
                LowerToOnnxRandomLike(node, isUniform, newNodes);
                newNodes.Add(node);
            }

            graph.Nodes = newNodes;

            if (functionRemap.Count > 0)
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
        }

        private static void LowerFunctionRecursive(
            Function fn, Dictionary<Function, Function> functionRemap)
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
            ProcessGraph(bodyFast, functionRemap);

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
                    [ShrkAttrStructure] = (DataStructure[])[DataStructure.Tensor],
                    [ShrkAttrDtype] = (DType[])[dtype],
                    [ShrkAttrRank] = (long[])[rank],
                    [ShrkAttrGenericTypeArgs] = null,
                },
                invokeAttrDefs);
            node.IdentifierTemplate = null;
            node.TargetFunction = fn;
        }

        /// <summary>
        /// Rewrites an id-bearing SHRK_RANDOM_* feed in place to the SHRK_RNG_UNIFORM/NORMAL
        /// form (inputs <c>[key, drawBase, shape, a, b]</c>), deriving the key from the
        /// decoded carrier config: a feed outside any loop gets its stream key as an int64[2]
        /// constant; a loop-body feed gets its dense per-stream key table with the row
        /// selected by the runtime iteration index. drawBase is the site's own counter input
        /// when wired (e.g. Dropout's per-execution state counter) else a constant 0, and the
        /// distribution bounds come off the node's attributes as f32 scalar constants.
        /// </summary>
        private static void RewriteFeedToKeyedDraw(
            FastNode node, bool isUniform, int[] idVals, RngConfig keyConfig, string algorithm,
            List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            float a = isUniform
                ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
            float b = isUniform
                ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;

            int depth = idVals.Count(v => v == -1);
            FastTensorKey keyKey = depth == 0
                ? AppendKeyConstant(keyConfig.FoldRunKey(idVals), newNodes)
                : AppendKeyTableSelect(node, idVals, depth, keyConfig, newNodes);

            var drawBaseKey = node.Inputs.Count > 1 && node.Inputs[1] is { } db
                ? db
                : AppendScalarInt64(0L, newNodes);
            var aKey = AppendScalarFloat32(a, newNodes);
            var bKey = AppendScalarFloat32(b, newNodes);

            var newOp = isUniform ? InternalOpCodes.SHRK_RNG_UNIFORM : InternalOpCodes.SHRK_RNG_NORMAL;
            var attrDefs = Definitions.NodeDefinitions[newOp].AttributeDefs;
            node.OpCode = newOp;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = algorithm },
                attrDefs);
            node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
            {
                [""] = new List<FastTensorKey?> { keyKey, drawBaseKey, shapeInput, aKey, bKey }
            };
        }

        /// <summary>
        /// Emits a loop-body feed's dense per-stream key table and the Gather that selects
        /// this iteration's [k0, k1] row. The table is a [Π counts, 2] int64 constant with one
        /// row per grid cell of the feed's enumerated iteration space (counts recorded at
        /// concretization), each row the fold of the cell's fully realized path — so every row
        /// sits at exactly the flat index Σ iterationIndex[j] · stride[j] the runtime
        /// computes, jagged observed sets included (cells no valid input ever reaches carry
        /// well-defined derived keys and are simply never gathered).
        /// </summary>
        private static FastTensorKey AppendKeyTableSelect(
            FastNode node, int[] idVals, int depth, RngConfig keyConfig, List<FastNode> newNodes)
        {
            var counts = node.Attributes.GetLongsVal(ShrkAttrRngIterCounts);
            if (counts is null || counts.Length != depth)
                throw new InvalidOperationException(
                    $"{node.OpCode}: ModelId [{string.Join(", ", idVals)}] has {depth} iteration " +
                    $"slot(s) but carries {(counts is null ? "no" : $"{counts.Length}")} iteration " +
                    "count(s) — the feed was not realized at concretization.");
            var iterationIndicesInput = node.Inputs.Count > 2 ? node.Inputs[2] : null;
            if (iterationIndicesInput is null)
                throw new InvalidOperationException(
                    $"{node.OpCode}: ModelId [{string.Join(", ", idVals)}] has an iteration slot " +
                    "but the node carries no iteration-indices input to select its stream.");

            long[] strides = new long[depth];
            long total = 1;
            for (int j = depth - 1; j >= 0; j--) { strides[j] = total; total *= counts[j]; }

            var table = new long[2 * total];
            var path = new int[idVals.Length];
            for (long flat = 0; flat < total; flat++)
            {
                int slot = 0;
                for (int i = 0; i < idVals.Length; i++)
                {
                    if (idVals[i] == -1)
                    {
                        path[i] = checked((int)(flat / strides[slot] % counts[slot]));
                        slot++;
                    }
                    else path[i] = idVals[i];
                }
                var (k0, k1) = keyConfig.FoldRunKey(path);
                table[2 * flat] = k0;
                table[2 * flat + 1] = k1;
            }

            var tableData = new OnnxTensorData<int64>(
                new Shape(total, 2),
                OnnxUtils.CreateTensorValue(new Shape(total, 2), table));
            var tableKey = AppendConstant(tableData, newNodes);

            FastTensorKey indexKey = default;
            bool first = true;
            for (int j = 0; j < depth; j++)
            {
                var term = AppendBinaryScalarInt64(OpCodes.MUL,
                    AppendGatherScalar(iterationIndicesInput.Value, j, newNodes),
                    AppendScalarInt64(strides[j], newNodes), newNodes);
                indexKey = first ? term : AppendBinaryScalarInt64(OpCodes.ADD, indexKey, term, newNodes);
                first = false;
            }

            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.GATHER].AttributeDefs;
            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.GATHER,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrAxis] = 0L }, attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { tableKey, indexKey } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        /// <summary>Gathers element <paramref name="index"/> of a rank-1 int64 vector as a rank-0 scalar.</summary>
        private static FastTensorKey AppendGatherScalar(
            FastTensorKey vector, int index, List<FastNode> newNodes)
        {
            var indexConst = AppendScalarInt64(index, newNodes);
            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[OpCodes.GATHER].AttributeDefs;
            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = OpCodes.GATHER,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrAxis] = 0L }, attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { vector, indexConst } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        private static FastTensorKey AppendKeyConstant((uint k0, uint k1) key, List<FastNode> newNodes)
        {
            long[] keyWords = [key.k0, key.k1];
            var data = new OnnxTensorData<int64>(
                new Shape(2),
                OnnxUtils.CreateTensorValue(new Shape(2), keyWords));
            return AppendConstant(data, newNodes);
        }

        /// <summary>Element-wise int64 scalar binary op node (MUL/ADD) for index arithmetic.</summary>
        private static FastTensorKey AppendBinaryScalarInt64(
            string opCode, FastTensorKey a, FastTensorKey b, List<FastNode> newNodes)
        {
            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[opCode].AttributeDefs;
            newNodes.Add(new FastNode
            {
                Key = nodeKey,
                OpCode = opCode,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>(), attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { a, b } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }

        private static FastTensorKey AppendScalarInt64(long value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (long[])[value]));
            return AppendConstant(data, newNodes);
        }

        private static FastTensorKey AppendScalarFloat32(float value, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<float32>(
                new Shape(Array.Empty<long>()),
                OnnxUtils.CreateTensorValue(new Shape(Array.Empty<long>()), (float[])[value]));
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

        private static readonly TensorData ZeroScalar = new OnnxTensorData<float32>(
            new Shape(1), OnnxUtils.CreateTensorValue((long[])[1], (float[])[0f]));

        /// <summary>
        /// Lowers an unkeyed SHRK_RANDOM_* node to <c>ConstantOfShape(shape, 0f)</c> +
        /// <c>RandomUniformLike/RandomNormalLike(placeholder)</c>, copying the distribution
        /// attrs and any user seed through (never synthesizing one).
        /// </summary>
        private static void LowerToOnnxRandomLike(FastNode node, bool isUniform, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            var placeholderKey = AppendConstantOfShape(shapeInput, newNodes);

            var dctAttrs = isUniform
                ? new Dictionary<string, object?>
                {
                    [AttrHigh] = node.Attributes.GetFloatVal(AttrHigh),
                    [AttrLow] = node.Attributes.GetFloatVal(AttrLow),
                    [AttrSeed] = node.Attributes.GetFloatVal(AttrSeed),
                }
                : new Dictionary<string, object?>
                {
                    [AttrMean] = node.Attributes.GetFloatVal(AttrMean),
                    [AttrScale] = node.Attributes.GetFloatVal(AttrScale),
                    [AttrSeed] = node.Attributes.GetFloatVal(AttrSeed),
                };
            var opCode = isUniform ? OpCodes.RANDOM_UNIFORM_LIKE : OpCodes.RANDOM_NORMAL_LIKE;
            var attrDefs = Definitions.NodeDefinitions[opCode].AttributeDefs;

            node.OpCode = opCode;
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
