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
    /// <para><b>Keyed draws</b> (<c>SHRK_RNG_SPLIT/UNIFORM/NORMAL</c>, produced by
    /// <see cref="FastApplyRngKeys"/> when an <see cref="RngConfig"/> is bound) lower to
    /// FUNCTION_INVOKEs of the named algorithm's non-inlined functions — the exported model
    /// calls tagged local FunctionProtos, so its randomness is deterministic, portable, and
    /// identifiable.</para>
    ///
    /// <para><b>Stamped feeds</b> (<c>SHRK_RANDOM_*</c> carrying the
    /// <c>shrk_rng_explicit_key</c> attribute <see cref="FastApplyRngKeys"/> stamps when an
    /// <see cref="RngConfig"/> is bound) are rewritten HERE — and only here — into the keyed
    /// draw: key/drawBase/bounds materialize as constants feeding the algorithm function
    /// call. Deferring the rewrite to ONNX prep is what makes config binding cheap and
    /// re-bindable: the pre-export graph keeps its structure, a re-stamp overrides a prior
    /// stamp, and constant folding of the key chain happens on the export clone.</para>
    ///
    /// <para><b>Unstamped draws</b> (no config bound, or in-loop feeds the stamping pass
    /// skips) take the ONNX fallback: <c>ConstantOfShape +
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

            var newNodes = new List<FastNode>(graph.Nodes.Count);

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_UNIFORM ||
                    node.OpCode == InternalOpCodes.SHRK_RNG_NORMAL)
                {
                    // A split carrying an explicit-key stamp draws from the stamped key
                    // instead of its parent_rng_key input — the split-side override hook
                    // (the config's own stamping targets the consumer draw ops).
                    if (node.OpCode == InternalOpCodes.SHRK_RNG_SPLIT &&
                        node.Attributes.GetLongsVal(ShrkAttrRngExplicitKey) is { Length: 2 } splitKey)
                    {
                        node.FullInputs[""][0] = AppendKeyConstant(splitKey, newNodes);
                    }
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

                if (node.Attributes.GetLongsVal(ShrkAttrRngExplicitKey) is { Length: 2 } stampedKey)
                {
                    // Config-stamped feed: materialize the stamped key (plus drawBase and
                    // distribution bounds) as constants and call the algorithm function.
                    // Ids with -1 iteration slots first realize the stamped PREFIX key into
                    // the site key via in-graph splits on the runtime iteration indices.
                    RewriteStampedToKeyedDraw(node, isUniform, stampedKey, newNodes);
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                // Unstamped random op (no RngConfig bound, or an in-loop feed FastApplyRngKeys
                // skipped): the ONNX fallback — ConstantOfShape + RandomUniformLike/NormalLike.
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
        /// Rewrites a stamped SHRK_RANDOM_* node in place to the SHRK_RNG_UNIFORM/NORMAL
        /// form (inputs <c>[key, drawBase, shape, a, b]</c>): the stamped key becomes an
        /// int64[2] constant, drawBase is the site's own counter input when wired (e.g.
        /// Dropout's per-execution state counter) else a constant 0, and the distribution
        /// bounds come off the node's attributes as f32 scalar constants.
        /// </summary>
        private static void RewriteStampedToKeyedDraw(
            FastNode node, bool isUniform, long[] stampedKey, List<FastNode> newNodes)
        {
            var shapeInput = node.Inputs[0]
                ?? throw new InvalidOperationException("SHRK_RANDOM_* has null shape input.");

            float a = isUniform
                ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
            float b = isUniform
                ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;
            var algorithm = node.Attributes.GetStringVal(ShrkAttrRngAlgorithm)
                ?? RngAlgorithms.Default;

            FastTensorKey keyKey;
            var keyTable = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngKeyTable);
            if (keyTable is { Length: > 2 })
            {
                // Per-stream key table (stamped when a realized stream carries an override):
                // select this iteration's [k0, k1] row by the flattened iteration index.
                keyKey = AppendKeyTableSelect(node, keyTable, newNodes);
            }
            else
            {
                keyKey = AppendKeyConstant(stampedKey, newNodes);
                keyKey = AppendIterationSplits(node, keyKey, algorithm, newNodes);
            }
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
        /// Realizes a stamped PREFIX key into the feed's site key when the feed's ModelId
        /// carries <c>-1</c> iteration slots: starting after the stamped prefix (everything
        /// before the first <c>-1</c>), each remaining path slot becomes one
        /// <c>SHRK_RNG_SPLIT</c> fold — a <c>-1</c> consumes the next scalar of the feed's
        /// iteration-indices input (Gather on element j; works identically whether the loop
        /// survives to runtime or was unrolled into constants), a concrete slot folds a
        /// constant. Ids without <c>-1</c> return the stamped key unchanged.
        /// </summary>
        private static FastTensorKey AppendIterationSplits(
            FastNode node, FastTensorKey keyKey, string algorithm, List<FastNode> newNodes)
        {
            var idVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId);
            if (idVals is null) return keyKey;
            int firstIterationSlot = Array.IndexOf(idVals, -1);
            if (firstIterationSlot < 0) return keyKey;

            var iterationIndicesInput = node.Inputs.Count > 2 ? node.Inputs[2] : null;
            if (iterationIndicesInput is null)
                throw new InvalidOperationException(
                    $"{node.OpCode}: ModelId [{string.Join(", ", idVals)}] has an iteration slot " +
                    "but the node carries no iteration-indices input to realize it.");

            int iterationScalarsUsed = 0;
            for (int j = firstIterationSlot; j < idVals.Length; j++)
            {
                FastTensorKey indexKey = idVals[j] == -1
                    ? AppendGatherScalar(iterationIndicesInput.Value, iterationScalarsUsed++, newNodes)
                    : AppendScalarInt64(idVals[j], newNodes);
                keyKey = AppendSplit(keyKey, indexKey, algorithm, newNodes);
            }
            return keyKey;
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

        /// <summary>
        /// One key fold: <c>child = split(key, index)</c> under the named algorithm, emitted
        /// directly as the lowered FUNCTION_INVOKE form (this pass will not revisit it).
        /// </summary>
        private static FastTensorKey AppendSplit(
            FastTensorKey keyKey, FastTensorKey indexKey, string algorithm, List<FastNode> newNodes)
        {
            var nodeKey = FastNodeKey.New();
            var outKey = new FastTensorKey(nodeKey, 0);
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.SHRK_RNG_SPLIT].AttributeDefs;
            var splitNode = new FastNode
            {
                Key = nodeKey,
                OpCode = InternalOpCodes.SHRK_RNG_SPLIT,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = algorithm }, attrDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { keyKey, indexKey } },
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            };
            LowerKeyedRngToFunctionCall(splitNode);
            newNodes.Add(splitNode);
            return outKey;
        }

        private static FastTensorKey AppendKeyConstant(long[] keyWords, List<FastNode> newNodes)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(2),
                OnnxUtils.CreateTensorValue(new Shape(2), keyWords));
            return AppendConstant(data, newNodes);
        }

        /// <summary>
        /// Selects this iteration's key row from a stamped per-stream key table: the table is a
        /// [N, 2] int64 constant (rows in lexicographic iteration order), the row index is
        /// Σ iterationIndices[j] · stride[j] over the stamped per-level strides, and a Gather on
        /// axis 0 yields the [2] key. Used instead of split folds when a realized stream of the
        /// feed carries a per-stream override (a single overridden iteration is inexpressible as
        /// a derivation chain).
        /// </summary>
        private static FastTensorKey AppendKeyTableSelect(
            FastNode node, long[] keyTable, List<FastNode> newNodes)
        {
            int n = keyTable.Length / 2;
            var tableData = new OnnxTensorData<int64>(
                new Shape(n, 2),
                OnnxUtils.CreateTensorValue(new Shape(n, 2), keyTable));
            var tableKey = AppendConstant(tableData, newNodes);

            var strides = node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRngIterStrides) ?? [];
            FastTensorKey indexKey;
            if (strides.Length == 0)
            {
                indexKey = AppendScalarInt64(0L, newNodes);
            }
            else
            {
                var iterationIndicesInput = node.Inputs.Count > 2 ? node.Inputs[2] : null;
                if (iterationIndicesInput is null)
                    throw new InvalidOperationException(
                        $"{node.OpCode}: a key table with iteration strides needs the feed's " +
                        "iteration-indices input to select a row.");
                indexKey = default;
                bool first = true;
                for (int j = 0; j < strides.Length; j++)
                {
                    var term = AppendBinaryScalarInt64(OpCodes.MUL,
                        AppendGatherScalar(iterationIndicesInput.Value, j, newNodes),
                        AppendScalarInt64(strides[j], newNodes), newNodes);
                    indexKey = first ? term : AppendBinaryScalarInt64(OpCodes.ADD, indexKey, term, newNodes);
                    first = false;
                }
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

        private static readonly TensorData ZeroScalar = new OnnxTensorData<float32>(
            new Shape(1), OnnxUtils.CreateTensorValue(new long[] { 1 }, new[] { 0f }));

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
