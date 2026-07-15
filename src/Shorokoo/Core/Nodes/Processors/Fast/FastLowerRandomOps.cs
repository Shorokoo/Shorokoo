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
    /// Random-op lowering pass (runs in the ONNX-export pre-passes). Pure wiring: key
    /// DERIVATION does not happen here — a feed site's keys are the VALUE of its
    /// <c>SHRK_RNG_KEY</c> entity, materialized from the bound <see cref="RngConfig"/> when
    /// the graph became a concrete model (see <see cref="FastMaterializeRngKeys"/>), exactly
    /// like trainable-parameter values.
    ///
    /// <para><b>Key entities</b> lower to plain int64 CONSTANTs (their materialized [N, 2]
    /// key table). An entity that was never materialized — e.g. a concrete architecture
    /// exported without a config — materializes here under the graph's carrier identity, or
    /// <see cref="RngConfig.Default"/> when none is bound: a concrete artifact is never
    /// unkeyed, and "no config" simply means the default deterministic identity.</para>
    ///
    /// <para><b>Keyed feeds</b> (id-bearing, wired to their key entity at concretization)
    /// rewrite to the keyed deterministic draw: Gather their iteration's [k0, k1] row from
    /// the key table by the runtime flat index Σ iterationIndex[j] · stride[j] (row 0 for a
    /// feed outside any loop), then call the named algorithm's non-inlined function — the
    /// exported model calls tagged local FunctionProtos, so its randomness is deterministic,
    /// portable, and identifiable.</para>
    ///
    /// <para><b>Feeds without stream identity</b> (no ModelId or no key entity — e.g. draws
    /// inside un-run initializer function bodies, or graphs that never went through
    /// concretization) take the ONNX fallback: <c>ConstantOfShape +
    /// RandomUniformLike/RandomNormalLike</c>, with any user seed copied through and none
    /// synthesized.</para>
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

            // The graph's recorded identity: the algorithm the keyed draws call, and the
            // fallback identity for any key entity not yet materialized. No carrier means
            // the default deterministic identity (never the ONNX random fallback).
            var carrier = graph.Nodes.FirstOrDefault(
                n => n.OpCode == InternalOpCodes.SHRK_RNG_KEY_VECTOR);
            RngConfig? carrierConfig = null;
            string algorithm = RngAlgorithms.Default;
            if (carrier?.Attributes.GetTensorVal(AttrValue) is { } carrierData)
            {
                carrierConfig = RngConfig.FromKeyVector(
                    carrierData.As<int64>().AccessMemory().ToArray());
                algorithm = carrier.Attributes.GetStringVal(ShrkAttrRngAlgorithm)
                    ?? RngAlgorithms.Default;
            }

            // Pre-scan the key entities: materialize any missing value, and keep each site's
            // iteration counts (needed for the feeds' index arithmetic) before the entity is
            // rewritten to a plain CONSTANT below.
            Dictionary<FastTensorKey, long[]>? countsByKeySource = null;
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.SHRK_RNG_KEY) continue;
                if (node.Attributes.GetTensorVal(AttrValue) is null)
                    FastMaterializeRngKeys.Materialize(node, carrierConfig ?? RngConfig.Default);
                (countsByKeySource ??= new Dictionary<FastTensorKey, long[]>())
                    [node.Outputs[0]!.Value] = node.Attributes.GetLongsVal(ShrkAttrRngIterCounts) ?? [];
            }

            var newNodes = new List<FastNode>(graph.Nodes.Count);

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == InternalOpCodes.SHRK_RNG_KEY)
                {
                    LowerKeyEntityToConstant(node);
                    newNodes.Add(node);
                    continue;
                }

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
                var keySource = node.Inputs.Count > 3 ? node.Inputs[3] : null;
                if (idVals is { Length: > 0 } && keySource is { } ks &&
                    countsByKeySource is not null && countsByKeySource.TryGetValue(ks, out var counts))
                {
                    RewriteFeedToKeyedDraw(node, isUniform, idVals, ks, counts, algorithm, newNodes);
                    LowerKeyedRngToFunctionCall(node);
                    newNodes.Add(node);
                    continue;
                }

                // A feed without stream identity (no ModelId, or never realized — e.g. inside
                // an initializer function body): the ONNX fallback — ConstantOfShape +
                // RandomUniformLike/NormalLike. Every legitimate fallback case carries NO key
                // input; a feed with a WIRED key input whose SHRK_RNG_KEY entity is missing
                // from the graph was concretized and then corrupted (entity pruned, sliced
                // graph), and silently lowering it here would turn a keyed site into real
                // backend randomness.
                System.Diagnostics.Debug.Assert(keySource is null,
                    $"FastLowerRandomOps: the feed at ModelId [{string.Join(", ", idVals ?? [])}] " +
                    "has a wired key input but its SHRK_RNG_KEY entity is not in the graph — " +
                    "the graph was modified since concretization; lowering it to the ONNX " +
                    "random fallback would silently make a keyed site non-deterministic.");
                LowerToOnnxRandomLike(node, isUniform, newNodes);
                newNodes.Add(node);
            }

            graph.Nodes = newNodes;

            if (functionRemap.Count > 0)
                foreach (var node in graph.Nodes)
                    if (node.TargetFunction is { } fn && functionRemap.TryGetValue(fn, out var newFn))
                        node.TargetFunction = newFn;
        }

        /// <summary>
        /// Rewrites a materialized SHRK_RNG_KEY entity in place to a plain CONSTANT of its
        /// key-table value, so every backend treats the keys as ordinary tensor data.
        /// </summary>
        private static void LowerKeyEntityToConstant(FastNode node)
        {
            var data = node.Attributes.GetTensorVal(AttrValue);
            var constDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            node.OpCode = OpCodes.CONSTANT;
            node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?> { [AttrValue] = data }, constDefs);
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
        /// form (inputs <c>[key, drawBase, shape, a, b]</c>). The key is this iteration's
        /// [k0, k1] row of the site's key entity (the feed's key input, already materialized),
        /// selected by Gather at the runtime flat index Σ iterationIndex[j] · stride[j] over
        /// the site's iteration counts — row 0 for a feed outside any loop. drawBase is the
        /// site's own counter input when wired (e.g. the injected per-execution state counter)
        /// else a constant 0, and the distribution bounds come off the node's attributes as
        /// f32 scalar constants.
        /// </summary>
        private static void RewriteFeedToKeyedDraw(
            FastNode node, bool isUniform, int[] idVals, FastTensorKey keySource, long[] counts,
            string algorithm, List<FastNode> newNodes)
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
            if (counts.Length != depth)
                throw new InvalidOperationException(
                    $"{node.OpCode}: ModelId [{string.Join(", ", idVals)}] has {depth} iteration " +
                    $"slot(s) but its key entity carries {counts.Length} iteration count(s).");

            FastTensorKey indexKey;
            if (depth == 0)
            {
                indexKey = AppendScalarInt64(0L, newNodes);
            }
            else
            {
                var iterationIndicesInput = node.Inputs.Count > 2 ? node.Inputs[2] : null;
                if (iterationIndicesInput is null)
                    throw new InvalidOperationException(
                        $"{node.OpCode}: ModelId [{string.Join(", ", idVals)}] has an iteration slot " +
                        "but the node carries no iteration-indices input to select its stream.");

                long[] strides = new long[depth];
                long total = 1;
                for (int j = depth - 1; j >= 0; j--) { strides[j] = total; total *= counts[j]; }

                indexKey = default;
                bool first = true;
                for (int j = 0; j < depth; j++)
                {
                    var term = AppendBinaryScalarInt64(OpCodes.MUL,
                        AppendGatherScalar(iterationIndicesInput.Value, j, newNodes),
                        AppendScalarInt64(strides[j], newNodes), newNodes);
                    indexKey = first ? term : AppendBinaryScalarInt64(OpCodes.ADD, indexKey, term, newNodes);
                    first = false;
                }
            }

            var gatherKey = FastNodeKey.New();
            var keyKey = new FastTensorKey(gatherKey, 0);
            var gatherDefs = Definitions.NodeDefinitions[OpCodes.GATHER].AttributeDefs;
            newNodes.Add(new FastNode
            {
                Key = gatherKey,
                OpCode = OpCodes.GATHER,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrAxis] = 0L }, gatherDefs),
                FullInputs = { [""] = new List<FastTensorKey?> { keySource, indexKey } },
                FullOutputs = { [""] = new List<FastTensorKey?> { keyKey } },
            });

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
