using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// Builds an ONNX <see cref="ModelProto"/> directly from a
    /// <see cref="FastComputationGraph"/>. Runs the standard pre-passes, then
    /// walks the graph emitting protos. Every step works against
    /// <see cref="FastNode"/>/<see cref="FastNodeKey"/>/<see cref="FastTensorKey"/>
    /// — never an <see cref="Variable"/>.
    ///
    /// <para>
    /// All five pre-passes have Fast-side implementations, so the main pipeline
    /// no longer round-trips through CG: the input graph is cloned, mutated by
    /// the Fast pre-passes (<see cref="FastLowerRandomOps"/>,
    /// <see cref="FastAddIdentityForOuterScopeValues"/>,
    /// <see cref="FastPrepForOnnx"/>, <see cref="FastStripCallStacks"/>,
    /// <see cref="FastUseUniqueNames"/>), then walked. Each function's body is
    /// taken from <see cref="Function.OriginalFastGraph"/> (cloned so the
    /// per-function pre-passes don't mutate the cached canonical form).
    /// </para>
    ///
    /// <para>
    /// IF/LOOP subgraphs are extracted from the implicit positional band between
    /// each open node and its matching close node via
    /// <see cref="FastSubgraphExtractor"/>. Because graph nodes are visited in
    /// topological order and a close node always sits after its body nodes, the
    /// inner subgraph's NodeProtos are already built by the time we hit the
    /// close, so they can be embedded as a graph attribute on the close node's
    /// emitted NodeProto.
    /// </para>
    /// </summary>
    public static class FastOnnxModelBuilder
    {
        /// <summary>
        /// Build an ONNX <see cref="ModelProto"/> from a
        /// <see cref="FastComputationGraph"/>. The input graph is not mutated —
        /// all pre-passes run on a clone.
        /// </summary>
        public static ModelProto BuildOnnxModel(
            FastComputationGraph fastGraph,
            OpSetVersion opset = OpSetVersion.OPS_21,
            IR_VERSION irVersion = IR_VERSION.IR_10,
            bool prepForOnnx = false)
        {
            if (fastGraph is null) throw new ArgumentNullException(nameof(fastGraph));

            // ----- 1. Clone so we never mutate the caller's graph.
            var prepFast = fastGraph.Clone();

            // ----- 2. Run the Fast pre-passes in place. Capture the rename map
            // so we can also remap the tensor-info lookup we'll build below.
            var tensorInfoLookup = RunPrePassesAndBuildLookup(prepFast, prepForOnnx);

            // Reorder so each IF body has then-block nodes positionally first
            // and else-block nodes positionally second. The Fast back-walk used
            // during subgraph extraction relies on this invariant.
            prepFast.ConfigureScopes(ScopeSize.Maximal, ScopeSize.Maximal, ScopePriority.Loop);

            // Raise the opset stamp just enough to cover post-opset-21 ops anywhere in
            // the model (main graph or function bodies); see FastOpsetResolver.RaiseToRequired.
            opset = FastOpsetResolver.RaiseToRequired(
                prepFast.Nodes
                    .Concat(CollectFunctionsPostOrder(prepFast)
                        .SelectMany(fn => fn.OriginalFastGraph.Nodes)),
                opset);

            // ----- 3. Build the main GraphProto by walking the Fast graph.
            var graphProto = BuildGraphProto(
                graphName: "",
                fastGraph: prepFast,
                opset: opset,
                isFunction: false,
                tensorInfoLookup: tensorInfoLookup);

            // ----- 4. Discover all reachable Functions in post order and emit
            // a FunctionProto for each.
            var functions = CollectFunctionsPostOrder(prepFast);
            var functionProtos = functions
                .Select(fn => BuildFunctionProto(fn, opset, prepForOnnx))
                .ToArray();

            var model = (ModelProto)OnnxIRFactory.CreateModel(graphProto, functionProtos, opset);

            // ----- 5. Lower deprecated Upsample nodes to Resize nodes so that
            // ONNX Runtime (opset 21) can execute them.
            LowerUpsampleToResize(model.Graph);

            // ----- 5b. Lower training-mode BatchNormalization nodes into primitive
            // ops. ONNX Runtime's CPU BatchNormalization kernel with training_mode=1
            // aliases its input_mean/input_var INPUT buffers as the running-stat
            // outputs and mutates them in place, persistently across session.Run
            // calls — any other consumer of the same tensors (e.g. an inference-mode
            // BN sharing the constants) then reads corrupted values. Lowering at
            // export time means the BN training kernel is never emitted. Like the
            // Upsample lowering above this always runs (not just for prepForOnnx)
            // so user-exported .onnx files are safe too.
            LowerTrainingBatchNormalization(model.Graph, BuildTensorMetaByName(tensorInfoLookup));

            // ----- 6. Attach TensorStructDef metadata for any struct-typed
            // inputs/outputs so the loader can reconstruct DType identity.
            AddTensorStructMetadata(model, prepFast, tensorInfoLookup);

            return model;
        }

        /// <summary>
        /// For every TensorStruct DType referenced by a graph input or output,
        /// attaches its <see cref="TensorStructDef"/> as a model-level metadata
        /// prop keyed by <c>shrk_tensorstruct_{ProtoTypeNum}</c>. The loader uses
        /// these props to recover the exact DType identity (and field schema)
        /// for struct-typed I/O on a save→load round-trip.
        /// </summary>
        private static void AddTensorStructMetadata(
            ModelProto model,
            FastComputationGraph fast,
            Dictionary<FastTensorKey, FastTensorInfo> tensorInfoLookup)
        {
            var seen = new HashSet<DType>();
            CollectStructDTypesFromGraph(fast, tensorInfoLookup, seen);

            // Also walk every Function reachable via TargetFunction. A sub-function
            // can have a TensorStruct input/output whose DType is never referenced
            // from the top-level graph's I/O (e.g. a TensorStruct is constructed
            // inside the outer module and only passed to the inner function as a
            // parameter). Without these the loader's ReconstructTensorStructDType
            // fails the metadata lookup when it walks the FunctionProto's inputs.
            var seenFunctions = new HashSet<Function>(ReferenceEqualityComparer.Instance);
            void VisitFunction(Function fn)
            {
                if (!seenFunctions.Add(fn)) return;
                var fnGraph = fn.OriginalFastGraph;
                var fnInfo = FastTensorInfoProcessor.BuildTensorInfoLookup(fnGraph);
                CollectStructDTypesFromGraph(fnGraph, fnInfo, seen);
                foreach (var node in fnGraph.Nodes)
                    if (node.TargetFunction is not null)
                        VisitFunction(node.TargetFunction);
            }
            foreach (var node in fast.Nodes)
                if (node.TargetFunction is not null)
                    VisitFunction(node.TargetFunction);

            foreach (var dtype in seen)
            {
                var def = dtype.TensorStructDef;
                if (def is null) continue;
                model.MetadataProps.Add(new StringStringEntryProto
                {
                    Key = $"{OnnxOpAttributeNames.ShrkMetaTensorStructDefPrefix}{dtype.ProtoTypeNum}",
                    Value = def.ToJson(),
                });
            }
        }

        private static void CollectStructDTypesFromGraph(
            FastComputationGraph graph,
            Dictionary<FastTensorKey, FastTensorInfo> infoLookup,
            HashSet<DType> seen)
        {
            // Walk every tensor in the graph (inputs, outputs, and all node-output keys)
            // — a TensorStruct DType can show up purely as the output of an internal
            // TENSOR_STRUCT_CREATE without ever appearing on a graph input/output.
            void AddIfStruct(FastTensorKey key)
            {
                if (!infoLookup.TryGetValue(key, out var info)) return;
                if (info.DType is null || !info.DType.IsTensorStructType) return;
                seen.Add(info.DType);
            }
            foreach (var key in graph.Inputs.Concat(graph.Outputs))
                AddIfStruct(key);
            foreach (var node in graph.Nodes)
                foreach (var slot in node.FullOutputs.Values)
                    foreach (var k in slot)
                        if (k is not null && !k.Value.IsEmpty)
                            AddIfStruct(k.Value);
        }

        /// <summary>
        /// Replaces deprecated Upsample nodes in the graph with equivalent Resize nodes.
        /// Upsample was deprecated in ONNX opset 9 and is no longer accepted by ONNX Runtime
        /// at opset 21. The mapping is:
        ///   Upsample(X, scales, mode=M) -> Resize(X, roi="", scales, sizes="",
        ///                                         mode=M, coordinate_transformation_mode="asymmetric")
        /// </summary>
        private static void LowerUpsampleToResize(GraphProto graph)
        {
            if (graph is null) return;
            foreach (var node in graph.Nodes)
            {
                foreach (var attr in node.Attributes)
                    if (attr.G is not null)
                        LowerUpsampleToResize(attr.G);

                if (node.OpType != OpCodes.UPSAMPLE)
                    continue;

                var x      = node.Inputs.Count > 0 ? node.Inputs[0] : "";
                var scales = node.Inputs.Count > 1 ? node.Inputs[1] : "";

                node.Inputs.Clear();
                node.Inputs.Add(x);
                node.Inputs.Add("");
                node.Inputs.Add(scales);
                node.Inputs.Add("");

                node.OpType = OpCodes.RESIZE;

                if (!node.Attributes.Any(a => a.Name == "coordinate_transformation_mode"))
                {
                    node.Attributes.Add(new AttributeProto
                    {
                        Name = "coordinate_transformation_mode",
                        Type = AttributeProto.AttributeType.String,
                        S    = System.Text.Encoding.UTF8.GetBytes("asymmetric")
                    });
                }
            }
        }

        /// <summary>
        /// Per-tensor metadata (rank + ONNX element type, either may be unknown)
        /// keyed by the tensor's emitted ONNX name, used by
        /// <see cref="LowerTrainingBatchNormalization"/>.
        /// </summary>
        private sealed record TensorMeta(int? Rank, int? ProtoElemType);

        /// <summary>
        /// Re-keys the <see cref="FastTensorInfo"/> lookup by the emitted ONNX
        /// tensor names (<c>"N{k}_T{s}"</c> via <see cref="FastTensorKey.ToString"/>),
        /// keeping entries where at least the rank or the dtype is known.
        /// </summary>
        private static Dictionary<string, TensorMeta> BuildTensorMetaByName(
            Dictionary<FastTensorKey, FastTensorInfo> tensorInfoLookup)
        {
            var byName = new Dictionary<string, TensorMeta>(tensorInfoLookup.Count);
            foreach (var (key, info) in tensorInfoLookup)
            {
                int? elemType = info.DType is { } dt && !dt.Equals(DType.Invalid)
                    ? dt.ProtoTypeNum
                    : null;
                if (info.Rank is null && elemType is null) continue;
                byName[key.ToString()] = new TensorMeta(info.Rank, elemType);
            }
            return byName;
        }

        /// <summary>
        /// Resolves rank/element-type metadata for the tensor named
        /// <paramref name="name"/>: first from the Fast tensor-info lookup, then —
        /// for whatever is still unknown (e.g. the rank of a rank-agnostic
        /// <c>Tensor&lt;T&gt;</c> graph input) — from the graph's declared
        /// ValueInfos, whose shapes carry the rank recorded at the model boundary.
        /// Returns null when neither source knows the tensor.
        /// </summary>
        private static TensorMeta? ResolveTensorMeta(
            GraphProto graph, Dictionary<string, TensorMeta> tensorMetaByName, string name)
        {
            tensorMetaByName.TryGetValue(name, out var meta);
            int? rank = meta?.Rank;
            int? elemType = meta?.ProtoElemType;
            if (rank is null || elemType is null)
            {
                var valueInfo = graph.Inputs.Concat(graph.ValueInfoes).Concat(graph.Outputs)
                    .FirstOrDefault(v => v.Name == name);
                var tensorType = valueInfo?.Type?.TensorType;
                if (tensorType is not null)
                {
                    // An absent/empty shape is ambiguous (unknown rank vs rank-0
                    // scalar) — treat it as unknown.
                    if (rank is null && tensorType.Shape is { Dims.Count: > 0 } shape)
                        rank = shape.Dims.Count;
                    if (elemType is null && tensorType.ElemType != 0)
                        elemType = tensorType.ElemType;
                }
            }
            return rank is null && elemType is null ? null : new TensorMeta(rank, elemType);
        }

        /// <summary>
        /// Replaces every training-mode (<c>training_mode=1</c>) BatchNormalization
        /// node in <paramref name="graph"/> with an equivalent chain of primitive
        /// ops (ReduceMean / Sub / Mul / Div / Sqrt / Add / Reshape / Constant — all
        /// opset-21 compatible), so ONNX Runtime's BN training kernel — which mutates
        /// its input_mean/input_var INPUT buffers in place when aliasing them as the
        /// running-stat outputs — is never executed. For node
        /// BN(x, scale, b, mean, var) with epsilon e and momentum m:
        /// <code>
        ///   axes     = Constant int64 [0, 2, 3, ..., R-1]      (R = rank of x)
        ///   bmKeep   = ReduceMean(x, axes, keepdims=1)          // [1,C,1,...]
        ///   centered = Sub(x, bmKeep)
        ///   bvKeep   = ReduceMean(centered*centered, axes, 1)   // biased variance per spec
        ///   y        = centered / Sqrt(bvKeep + e) * Reshape(scale, [1,-1,1,...])
        ///              + Reshape(b, [1,-1,1,...])
        ///   runMean  = mean * m + Reshape(bmKeep, [-1]) * (1-m) // if output declared
        ///   runVar   = var  * m + Reshape(bvKeep, [-1]) * (1-m) // if output declared
        /// </code>
        /// When the rank of x is statically unavailable (e.g. x is a rank-agnostic
        /// <c>Tensor&lt;T&gt;</c> graph input, whose ValueInfo carries no shape), the
        /// rank-dependent constants are instead computed at runtime —
        /// <c>axes = Concat([0], Range(2, Reshape(Shape(Shape(x)), []), 1))</c> and
        /// the scale/bias reshape target is <c>Shape(bmKeep)</c> — so the lowering
        /// still applies to every training-mode BN in the graph.
        ///
        /// The original output names are wired onto the final producing nodes and the
        /// replacement chain is inserted at the BN node's position, so topological
        /// order is preserved. Idempotent: a second pass finds no training-mode BN.
        ///
        /// Recurses into graph-attribute subgraphs (Loop/If bodies) — their tensor
        /// names come from the same Fast graph, so the same metadata lookup applies.
        /// FunctionProto bodies are NOT lowered, mirroring
        /// <see cref="LowerUpsampleToResize"/> (which has the same limitation): a
        /// BN training node inside a sub-module Function would still reach the ORT
        /// kernel, but function bodies operate on caller-scoped formal parameters,
        /// not on shared graph constants/initializers, so the in-place corruption
        /// cannot cross into other consumers the way the main-graph case does.
        /// </summary>
        private static void LowerTrainingBatchNormalization(
            GraphProto graph, Dictionary<string, TensorMeta> tensorMetaByName)
        {
            if (graph is null) return;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];

                foreach (var attr in node.Attributes)
                    if (attr.G is not null)
                        LowerTrainingBatchNormalization(attr.G, tensorMetaByName);

                if (node.OpType != OpCodes.BATCH_NORMALIZATION)
                    continue;
                var trainingAttr = node.Attributes.FirstOrDefault(
                    a => a.Name == OnnxOpAttributeNames.AttrTrainingMode);
                if (trainingAttr is null || trainingAttr.I == 0)
                    continue; // inference mode — the ORT kernel is safe there.
                if (node.Inputs.Count < 5 || node.Outputs.Count < 1 || node.Outputs[0].Length == 0)
                    continue; // malformed BN node — leave as-is.

                string x = node.Inputs[0], scale = node.Inputs[1], b = node.Inputs[2];
                string mean = node.Inputs[3], variance = node.Inputs[4];

                // Rank of x comes from the Fast tensor-info lookup or the graph's
                // declared ValueInfos (BN requires rank >= 2: (N, C, D1...Dn)).
                // When it is statically unavailable (e.g. x is a rank-agnostic
                // Tensor<T> graph input) we fall back to building the reduction
                // axes dynamically from Shape(x) — see below — so the poisonous
                // BN training kernel is never emitted either way.
                var xMeta = ResolveTensorMeta(graph, tensorMetaByName, x);
                int? rank = xMeta?.Rank;
                if (rank is < 2)
                    continue; // statically known invalid/degenerate BN input — leave as-is.
                int xElemType = xMeta?.ProtoElemType ?? (int)TensorProto.DataType.Float;

                float epsilon = node.Attributes.FirstOrDefault(
                    a => a.Name == OnnxOpAttributeNames.AttrEpsilon)?.F ?? 1e-5f;
                float momentum = node.Attributes.FirstOrDefault(
                    a => a.Name == OnnxOpAttributeNames.AttrMomentum)?.F ?? 0.9f;

                string yName = node.Outputs[0];
                string runMeanName = node.Outputs.Count > 1 ? node.Outputs[1] : "";
                string runVarName = node.Outputs.Count > 2 ? node.Outputs[2] : "";

                // Stat tensors (mean/var) may use a different float type (T2) than x
                // (T) per the BN signature; in practice they match. Momentum constants
                // get the stats' element type so the running-stat arithmetic typechecks.
                int statElemType = ResolveTensorMeta(graph, tensorMetaByName, mean)?.ProtoElemType
                    ?? xElemType;

                // All original tensor names are "N{k}"/"N{k}_T{s}", so suffixing the
                // (unique) node name cannot collide with existing names.
                string prefix = (node.Name.Length > 0 ? node.Name : yName) + "_bnlow";

                var lowered = new List<NodeProto>(26);

                string axesName = $"{prefix}_axes";
                if (rank is int r)
                {
                    // Static rank: axes = all dims except the channel dim (1).
                    var axes = new long[r - 1];
                    axes[0] = 0;
                    for (int d = 2; d < r; d++) axes[d - 1] = d;
                    lowered.Add(MakeInt64ConstNode($"{prefix}_axes_const", axesName, axes));
                }
                else
                {
                    // Unknown rank: build [0, 2, ..., R-1] at runtime as
                    // Concat([0], Range(2, R, 1)) with R = Reshape(Shape(Shape(x)), []).
                    string shapeX = $"{prefix}_shape_x";
                    lowered.Add(MakeNode($"{prefix}_shape_x_of", OpCodes.SHAPE, [x], [shapeX]));
                    string rank1d = $"{prefix}_rank_1d";
                    lowered.Add(MakeNode($"{prefix}_rank_of", OpCodes.SHAPE, [shapeX], [rank1d]));
                    string emptyShape = $"{prefix}_empty_shape";
                    lowered.Add(MakeInt64ConstNode($"{prefix}_empty_shape_const", emptyShape, []));
                    string rankScalar = $"{prefix}_rank_scalar";
                    lowered.Add(MakeNode($"{prefix}_rank_to_scalar", OpCodes.RESHAPE, [rank1d, emptyShape], [rankScalar]));
                    string twoScalar = $"{prefix}_two";
                    lowered.Add(MakeInt64ScalarConstNode($"{prefix}_two_const", twoScalar, 2L));
                    string oneScalar = $"{prefix}_one";
                    lowered.Add(MakeInt64ScalarConstNode($"{prefix}_one_const", oneScalar, 1L));
                    string zeroVec = $"{prefix}_zero_axis";
                    lowered.Add(MakeInt64ConstNode($"{prefix}_zero_axis_const", zeroVec, [0L]));
                    string tailAxes = $"{prefix}_tail_axes";
                    lowered.Add(MakeNode($"{prefix}_tail_axes_range", OpCodes.RANGE,
                        [twoScalar, rankScalar, oneScalar], [tailAxes]));
                    lowered.Add(MakeNode($"{prefix}_axes_concat", OpCodes.CONCAT,
                        [zeroVec, tailAxes], [axesName],
                        MakeIntAttr(OnnxOpAttributeNames.AttrAxis, 0)));
                }

                string bmKeep = $"{prefix}_batch_mean";
                lowered.Add(MakeNode($"{prefix}_reduce_mean", OpCodes.REDUCE_MEAN,
                    [x, axesName], [bmKeep],
                    MakeIntAttr(OnnxOpAttributeNames.AttrKeepdims, 1)));
                string centered = $"{prefix}_centered";
                lowered.Add(MakeNode($"{prefix}_center", OpCodes.SUB, [x, bmKeep], [centered]));
                string squared = $"{prefix}_squared";
                lowered.Add(MakeNode($"{prefix}_square", OpCodes.MUL, [centered, centered], [squared]));
                string bvKeep = $"{prefix}_batch_var";
                lowered.Add(MakeNode($"{prefix}_reduce_var", OpCodes.REDUCE_MEAN,
                    [squared, axesName], [bvKeep],
                    MakeIntAttr(OnnxOpAttributeNames.AttrKeepdims, 1)));
                string epsName = $"{prefix}_eps";
                lowered.Add(MakeFloatScalarConstNode($"{prefix}_eps_const", epsName, epsilon, xElemType));
                string varEps = $"{prefix}_var_eps";
                lowered.Add(MakeNode($"{prefix}_add_eps", OpCodes.ADD, [bvKeep, epsName], [varEps]));
                string std = $"{prefix}_std";
                lowered.Add(MakeNode($"{prefix}_sqrt", OpCodes.SQRT, [varEps], [std]));
                string normalized = $"{prefix}_normalized";
                lowered.Add(MakeNode($"{prefix}_div", OpCodes.DIV, [centered, std], [normalized]));
                // Reshape target lifting the [C] scale/bias vectors to the channel
                // dim: a constant [1, -1, 1, ..., 1] when the rank is known, else
                // Shape(bmKeep) (= [1, C, 1, ...]) computed at runtime.
                string chanShapeName = $"{prefix}_chan_shape";
                if (rank is int r2)
                {
                    var chanShape = new long[r2];
                    Array.Fill(chanShape, 1L);
                    chanShape[1] = -1;
                    lowered.Add(MakeInt64ConstNode($"{prefix}_chan_shape_const", chanShapeName, chanShape));
                }
                else
                {
                    lowered.Add(MakeNode($"{prefix}_chan_shape_of", OpCodes.SHAPE, [bmKeep], [chanShapeName]));
                }
                string scaleR = $"{prefix}_scale_r";
                lowered.Add(MakeNode($"{prefix}_scale_reshape", OpCodes.RESHAPE, [scale, chanShapeName], [scaleR]));
                string biasR = $"{prefix}_bias_r";
                lowered.Add(MakeNode($"{prefix}_bias_reshape", OpCodes.RESHAPE, [b, chanShapeName], [biasR]));
                string scaled = $"{prefix}_scaled";
                lowered.Add(MakeNode($"{prefix}_apply_scale", OpCodes.MUL, [normalized, scaleR], [scaled]));
                lowered.Add(MakeNode($"{prefix}_apply_bias", OpCodes.ADD, [scaled, biasR], [yName]));

                if (runMeanName.Length > 0 || runVarName.Length > 0)
                {
                    string flatName = $"{prefix}_flat_shape";
                    lowered.Add(MakeInt64ConstNode($"{prefix}_flat_shape_const", flatName, [-1L]));
                    string momName = $"{prefix}_momentum";
                    lowered.Add(MakeFloatScalarConstNode($"{prefix}_momentum_const", momName, momentum, statElemType));
                    string invMomName = $"{prefix}_one_minus_momentum";
                    lowered.Add(MakeFloatScalarConstNode($"{prefix}_one_minus_momentum_const", invMomName, 1f - momentum, statElemType));

                    if (runMeanName.Length > 0)
                    {
                        string bmFlat = $"{prefix}_batch_mean_flat";
                        lowered.Add(MakeNode($"{prefix}_bm_flatten", OpCodes.RESHAPE, [bmKeep, flatName], [bmFlat]));
                        string rmOld = $"{prefix}_rm_old";
                        lowered.Add(MakeNode($"{prefix}_rm_old_mul", OpCodes.MUL, [mean, momName], [rmOld]));
                        string rmNew = $"{prefix}_rm_new";
                        lowered.Add(MakeNode($"{prefix}_rm_new_mul", OpCodes.MUL, [bmFlat, invMomName], [rmNew]));
                        lowered.Add(MakeNode($"{prefix}_rm_add", OpCodes.ADD, [rmOld, rmNew], [runMeanName]));
                    }
                    if (runVarName.Length > 0)
                    {
                        string bvFlat = $"{prefix}_batch_var_flat";
                        lowered.Add(MakeNode($"{prefix}_bv_flatten", OpCodes.RESHAPE, [bvKeep, flatName], [bvFlat]));
                        string rvOld = $"{prefix}_rv_old";
                        lowered.Add(MakeNode($"{prefix}_rv_old_mul", OpCodes.MUL, [variance, momName], [rvOld]));
                        string rvNew = $"{prefix}_rv_new";
                        lowered.Add(MakeNode($"{prefix}_rv_new_mul", OpCodes.MUL, [bvFlat, invMomName], [rvNew]));
                        lowered.Add(MakeNode($"{prefix}_rv_add", OpCodes.ADD, [rvOld, rvNew], [runVarName]));
                    }
                }

                graph.Nodes.RemoveAt(i);
                graph.Nodes.InsertRange(i, lowered);
                i += lowered.Count - 1;
            }
        }

        private static AttributeProto MakeIntAttr(string name, long value)
            => new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

        private static NodeProto MakeNode(
            string name, string opType, string[] inputs, string[] outputs,
            params AttributeProto[] attributes)
        {
            var node = new NodeProto { Name = name, OpType = opType };
            node.Inputs.AddRange(inputs);
            node.Outputs.AddRange(outputs);
            node.Attributes.AddRange(attributes);
            return node;
        }

        private static NodeProto MakeInt64ConstNode(string nodeName, string outputName, long[] values)
        {
            var bytes = new byte[values.Length * sizeof(long)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            var tensor = new TensorProto
            {
                Dims = [values.Length],
                data_type = (int)TensorProto.DataType.Int64,
                RawData = bytes,
            };
            return MakeNode(nodeName, OpCodes.CONSTANT, [], [outputName],
                new AttributeProto
                {
                    Name = OnnxOpAttributeNames.AttrValue,
                    Type = AttributeProto.AttributeType.Tensor,
                    T = tensor,
                });
        }

        private static NodeProto MakeInt64ScalarConstNode(string nodeName, string outputName, long value)
        {
            var tensor = new TensorProto
            {
                Dims = [],
                data_type = (int)TensorProto.DataType.Int64,
                RawData = BitConverter.GetBytes(value),
            };
            return MakeNode(nodeName, OpCodes.CONSTANT, [], [outputName],
                new AttributeProto
                {
                    Name = OnnxOpAttributeNames.AttrValue,
                    Type = AttributeProto.AttributeType.Tensor,
                    T = tensor,
                });
        }

        /// <summary>
        /// Builds a rank-0 Constant node carrying <paramref name="value"/> in the
        /// requested float element type. Unrecognised element types fall back to
        /// float32 (the only FloatLike types Shorokoo emits are handled explicitly).
        /// </summary>
        private static NodeProto MakeFloatScalarConstNode(
            string nodeName, string outputName, float value, int protoElemType)
        {
            byte[] bytes;
            int dataType = protoElemType;
            switch (protoElemType)
            {
                case (int)TensorProto.DataType.Double:
                    bytes = BitConverter.GetBytes((double)value);
                    break;
                case (int)TensorProto.DataType.Float16:
                    bytes = BitConverter.GetBytes(BitConverter.HalfToInt16Bits((Half)value));
                    break;
                case (int)TensorProto.DataType.Bfloat16:
                    bytes = BitConverter.GetBytes((short)(BitConverter.SingleToInt32Bits(value) >> 16));
                    break;
                case (int)TensorProto.DataType.Float:
                    bytes = BitConverter.GetBytes(value);
                    break;
                default:
                    bytes = BitConverter.GetBytes(value);
                    dataType = (int)TensorProto.DataType.Float;
                    break;
            }
            var tensor = new TensorProto
            {
                Dims = [],
                data_type = dataType,
                RawData = bytes,
            };
            return MakeNode(nodeName, OpCodes.CONSTANT, [], [outputName],
                new AttributeProto
                {
                    Name = OnnxOpAttributeNames.AttrValue,
                    Type = AttributeProto.AttributeType.Tensor,
                    T = tensor,
                });
        }

        // ----------- pre-passes -----------

        /// <summary>
        /// Runs every Fast pre-pass on <paramref name="graph"/> in the canonical
        /// pre-pass order. Mutates the graph in place.
        /// </summary>
        private static void RunPrePasses(FastComputationGraph graph, bool prepForOnnx)
        {
            FastLowerAttributeTensorOps.Process(graph);
            FastLowerRandomOps.Process(graph);
            FastAddIdentityForOuterScopeValues.Process(graph);
            if (prepForOnnx) FastPrepForOnnx.Process(graph);
            FastStripCallStacks.Process(graph);
            Debug.Assert(graph.IsLinearOrderValid(),
                "FastOnnxModelBuilder.RunPrePasses: scope nesting must be valid by this point — " +
                "every Fast pass that mutates node order preserves the linear-order invariant.");
            FastUseUniqueNames.Process(graph);
        }

        /// <summary>
        /// Runs the Fast pre-passes, then builds a tensor-info lookup keyed by
        /// the renamed (post-FastUseUniqueNames) <see cref="FastTensorKey"/>s.
        /// The lookup is built before <see cref="FastUseUniqueNames"/> runs (so
        /// the round-trip-via-CG conversion sees real producer chains) and
        /// then remapped through the rename map.
        /// </summary>
        private static Dictionary<FastTensorKey, FastTensorInfo> RunPrePassesAndBuildLookup(
            FastComputationGraph graph, bool prepForOnnx)
        {
            FastLowerAttributeTensorOps.Process(graph);
            FastLowerRandomOps.Process(graph);
            FastAddIdentityForOuterScopeValues.Process(graph);
            if (prepForOnnx) FastPrepForOnnx.Process(graph);
            FastStripCallStacks.Process(graph);
            Debug.Assert(graph.IsLinearOrderValid(),
                "FastOnnxModelBuilder.RunPrePassesAndBuildLookup: scope nesting must be valid by this point — " +
                "every Fast pass that mutates node order preserves the linear-order invariant.");

            // Build the lookup before renaming so keys still match the
            // post-converter producer wiring.
            var preRenameLookup = FastTensorInfoProcessor.BuildTensorInfoLookup(graph);

            // Rename, capturing the map so we can rewrite the lookup.
            var oldToNew = FastUseUniqueNames.ProcessAndReturnMap(graph);

            var renamed = new Dictionary<FastTensorKey, FastTensorInfo>(preRenameLookup.Count);
            foreach (var (oldKey, info) in preRenameLookup)
            {
                if (oldKey.IsEmpty) continue;
                var newNodeKey = oldToNew.TryGetValue(oldKey.FastNodeKey, out var nk) ? nk : oldKey.FastNodeKey;
                var newKey = new FastTensorKey(newNodeKey, oldKey.OutputIndex);
                renamed[newKey] = info;
            }
            return renamed;
        }

        // ----------- function discovery -----------

        /// <summary>
        /// Collects every <see cref="Function"/> reachable from
        /// <paramref name="graph"/> via <see cref="FastNode.TargetFunction"/>
        /// references, in post order so callee functions land in the result
        /// before the functions that call them.
        /// </summary>
        private static List<Function> CollectFunctionsPostOrder(FastComputationGraph graph)
        {
            var seen = new HashSet<Function>(ReferenceEqualityComparer.Instance);
            var result = new List<Function>();

            void Visit(FastComputationGraph g)
            {
                foreach (var node in g.Nodes)
                {
                    var fn = node.TargetFunction;
                    if (fn is null) continue;
                    if (!seen.Add(fn)) continue;
                    Visit(fn.OriginalFastGraph);
                    result.Add(fn);
                }
            }

            Visit(graph);
            return result;
        }

        // ----------- function emission -----------

        private static FunctionProto BuildFunctionProto(Function function, OpSetVersion opset, bool prepForOnnx)
        {
            // Clone the function's primary Fast body and run the same pre-passes
            // on the copy. The function's body has its own ONNX-name namespace,
            // so the per-graph counter inside FastUseUniqueNames restarts at 1
            // for each function — matches how ONNX FunctionProtos are scoped.
            var fnFast = function.OriginalFastGraph.Clone();
            RunPrePasses(fnFast, prepForOnnx);
            fnFast.ConfigureScopes(ScopeSize.Maximal, ScopeSize.Maximal, ScopePriority.Loop);

            var fnGraphProto = BuildGraphProto(
                graphName: function.DefaultName,
                fastGraph: fnFast,
                opset: opset,
                isFunction: true);

            var fnProto = new FunctionProto();
            // Encode the name to dodge built-in ONNX op-name collisions (see OnnxFunctionName);
            // must match the encoded op_type emitted for call sites in FastOpsetResolver.
            fnProto.Name = OnnxFunctionName.Encode(function.DefaultName);
            fnProto.Domain = "Functions";

            var defaultOpset = new OperatorSetIdProto { Domain = "", Version = (int)opset };
            fnProto.OpsetImports.Add(defaultOpset);
            var fnOpset = new OperatorSetIdProto { Domain = "Functions", Version = 1 };
            fnProto.OpsetImports.Add(fnOpset);

            fnProto.Inputs.AddAll(fnGraphProto.Inputs.Select(x => x.Name));
            fnProto.Outputs.AddAll(fnGraphProto.Outputs.Select(x => x.Name));
            fnProto.ValueInfoes.AddAll(fnGraphProto.Inputs);
            fnProto.ValueInfoes.AddAll(fnGraphProto.Outputs);
            fnProto.Nodes.AddAll(fnGraphProto.Nodes);

            if (function.FunctionType != FunctionType.Function)
            {
                var typeMeta = new StringStringEntryProto
                {
                    Key = Function.IRFunctionTypeParamName,
                    Value = Function.ToComponentTypeName(function.FunctionType),
                };
                fnProto.MetadataProps.Add(typeMeta);
            }

            var nameMeta = new StringStringEntryProto
            {
                Key = Function.IRFunctionFriendlyName,
                Value = function.FriendlyName,
            };
            fnProto.MetadataProps.Add(nameMeta);

            return fnProto;
        }

        // ----------- graph emission -----------

        /// <summary>
        /// Walks <paramref name="fastGraph"/> in topological order, emitting each
        /// non-boundary node into the appropriate subgraph bucket. The top-level
        /// bucket becomes the resulting <see cref="GraphProto.Nodes"/>; nested
        /// buckets become graph-attribute subgraphs on their close-node's
        /// NodeProto.
        /// </summary>
        private static GraphProto BuildGraphProto(
            string graphName,
            FastComputationGraph fastGraph,
            OpSetVersion opset,
            bool isFunction,
            Dictionary<FastTensorKey, FastTensorInfo>? tensorInfoLookup = null)
        {
            var scopeIndex = FastSubgraphExtractor.BuildScopeIndex(fastGraph);

            // Build a key→node lookup for resolving GraphOpenNodeKey on the fly.
            var nodeByKey = new Dictionary<FastNodeKey, FastNode>();
            for (int i = 0; i < fastGraph.Nodes.Count; i++)
                nodeByKey[fastGraph.Nodes[i].Key] = fastGraph.Nodes[i];

            // Per-node-index → built NodeProto so close nodes can pull their body
            // contents out by index.
            var protoByIndex = new Dictionary<int, NodeProto>();

            for (int i = 0; i < fastGraph.Nodes.Count; i++)
            {
                var node = fastGraph.Nodes[i];

                // Skip boundary nodes: model inputs and parameter data are emitted
                // as ValueInfoProto/TensorProto, not as NodeProto. Open nodes are
                // never emitted as NodeProto either.
                if (FastOpsetResolver.IsBoundaryOrOpen(node))
                    continue;

                FastNode? graphOpenNode = null;
                if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                    nodeByKey.TryGetValue(openKey, out graphOpenNode);

                Dictionary<string, GraphProto>? graphAttrs = null;
                if (node.OpCode == OpCodes.LOOP_CLOSE)
                {
                    int loopOpenIdx = scopeIndex.CloseIdxToOpenIdx[i];
                    var loopOpen = fastGraph.Nodes[loopOpenIdx];
                    var bodyIdxs = FastSubgraphExtractor.BodyBand(scopeIndex, loopOpenIdx, i);
                    var bodyOutputs = FastSubgraphExtractor.LoopBodyOutputs(node);
                    // Loop body subgraph inputs: the matching LOOP_OPEN exposes
                    // them via FullOutputs[AttrBody]. ONNX requires the body
                    // subgraph to declare iter_num + cond + loop-carried as
                    // inputs by name.
                    var bodyInputs = SubgraphInputsFromOpen(loopOpen, OnnxOpAttributeNames.AttrBody);
                    var bodyProto = AssembleSubgraph(
                        graphName: $"subgraph_{node.Key}_{OnnxOpAttributeNames.AttrBody}",
                        bodyIdxs: bodyIdxs,
                        inputKeys: bodyInputs,
                        outputKeys: bodyOutputs,
                        protoByIndex: protoByIndex,
                        fastGraph: fastGraph,
                        tensorInfoLookup: tensorInfoLookup);
                    graphAttrs = new Dictionary<string, GraphProto> { [OnnxOpAttributeNames.AttrBody] = bodyProto };
                }
                else if (node.OpCode == OpCodes.IF_CLOSE)
                {
                    int ifOpenIdx = scopeIndex.CloseIdxToOpenIdx[i];
                    var ifOpen = fastGraph.Nodes[ifOpenIdx];
                    var (thenIdxs, elseIdxs) = FastSubgraphExtractor.BifurcateIfBody(fastGraph, scopeIndex, ifOpenIdx, i);
                    var thenOutputs = FastSubgraphExtractor.BranchOutputs(node, OnnxOpAttributeNames.AttrThenBranch);
                    var elseOutputs = FastSubgraphExtractor.BranchOutputs(node, OnnxOpAttributeNames.AttrElseBranch);
                    var thenInputs = SubgraphInputsFromOpen(ifOpen, OnnxOpAttributeNames.AttrThenBranch);
                    var elseInputs = SubgraphInputsFromOpen(ifOpen, OnnxOpAttributeNames.AttrElseBranch);
                    var thenProto = AssembleSubgraph(
                        graphName: $"subgraph_{node.Key}_{OnnxOpAttributeNames.AttrThenBranch}",
                        bodyIdxs: thenIdxs,
                        inputKeys: thenInputs,
                        outputKeys: thenOutputs,
                        protoByIndex: protoByIndex,
                        fastGraph: fastGraph,
                        tensorInfoLookup: tensorInfoLookup);
                    var elseProto = AssembleSubgraph(
                        graphName: $"subgraph_{node.Key}_{OnnxOpAttributeNames.AttrElseBranch}",
                        bodyIdxs: elseIdxs,
                        inputKeys: elseInputs,
                        outputKeys: elseOutputs,
                        protoByIndex: protoByIndex,
                        fastGraph: fastGraph,
                        tensorInfoLookup: tensorInfoLookup);
                    graphAttrs = new Dictionary<string, GraphProto>
                    {
                        [OnnxOpAttributeNames.AttrThenBranch] = thenProto,
                        [OnnxOpAttributeNames.AttrElseBranch] = elseProto,
                    };
                }

                var info = FastOpsetResolver.Resolve(node, graphOpenNode, opset);
                if (info is null) continue; // open node — already handled by IsBoundaryOrOpen, but defensive
                var nodeProto = FastOnnxProtoFactory.CreateNodeProto(node, info.Value, graphAttrs);
                protoByIndex[i] = nodeProto;
            }

            // A node is "top-level" if its index is outside every (open, close)
            // span — i.e. it's not strictly between any open/close pair.
            var swallowed = new HashSet<int>();
            foreach (var (openIdx, closeIdx) in scopeIndex.OpenIdxToCloseIdx)
                for (int j = openIdx + 1; j < closeIdx; j++)
                    swallowed.Add(j);

            var topLevelNodes = new List<NodeProto>();
            foreach (var (idx, proto) in protoByIndex.OrderBy(kv => kv.Key))
            {
                if (swallowed.Contains(idx)) continue;
                topLevelNodes.Add(proto);
            }

            var initializers = isFunction
                ? Array.Empty<TensorProto>()
                : CreateInitializerTensors(fastGraph);
            var inputInfos = CreateInputInfos(fastGraph);
            var outputInfos = CreateOutputInfos(fastGraph);

            return (GraphProto)OnnxIRFactory.CreateGraph(
                graphName,
                initializers,
                inputInfos,
                outputInfos,
                topLevelNodes.ToArray());
        }

        /// <summary>
        /// Wraps the NodeProtos at the given Fast indices into a
        /// <see cref="GraphProto"/> for use as a subgraph attribute. The
        /// <paramref name="inputKeys"/> are emitted as declared subgraph
        /// inputs (required for Loop body subgraphs; harmless and helpful
        /// for IF branches).
        /// </summary>
        private static GraphProto AssembleSubgraph(
            string graphName,
            List<int> bodyIdxs,
            IReadOnlyList<FastTensorKey?> inputKeys,
            IReadOnlyList<FastTensorKey?> outputKeys,
            Dictionary<int, NodeProto> protoByIndex,
            FastComputationGraph fastGraph,
            Dictionary<FastTensorKey, FastTensorInfo>? tensorInfoLookup)
        {
            var nodes = new List<NodeProto>(bodyIdxs.Count);
            foreach (var idx in bodyIdxs)
            {
                if (protoByIndex.TryGetValue(idx, out var p))
                    nodes.Add(p);
            }

            var inputInfos = inputKeys
                .Where(k => k is not null && !k.Value.IsEmpty)
                .Select(k => CreateSubgraphInputInfo(k!.Value, tensorInfoLookup))
                .ToArray();

            var outputInfos = outputKeys
                .Where(k => k is not null && !k.Value.IsEmpty)
                .Select(k => FastOnnxProtoFactory.CreateSubgraphOutputInfo(k!.Value))
                .ToArray();

            return (GraphProto)OnnxIRFactory.CreateGraph(
                graphName,
                Array.Empty<TensorProto>(),
                inputInfos,
                outputInfos,
                nodes.ToArray());
        }

        /// <summary>
        /// Builds a typed <see cref="ValueInfoProto"/> for a subgraph input,
        /// pulling dtype/structure/rank from <paramref name="tensorInfoLookup"/>
        /// when available. Falls back to a name-only ValueInfoProto when the
        /// lookup has no entry — ONNX type inference will then attempt to
        /// resolve the type from the surrounding context.
        /// </summary>
        private static ValueInfoProto CreateSubgraphInputInfo(
            FastTensorKey key,
            Dictionary<FastTensorKey, FastTensorInfo>? tensorInfoLookup)
        {
            if (tensorInfoLookup is not null && tensorInfoLookup.TryGetValue(key, out var info) && !info.DType.Equals(DType.Invalid))
            {
                var dims = info.Rank is int r
                    ? OnnxIRFactory.CreateDims(MakeUnnamedDims(r), key.ToString())
                    : null;
                return (ValueInfoProto)OnnxIRFactory.CreateTensorInfo(
                    dims: dims,
                    name: key.ToString(),
                    type: info.DType,
                    structure: info.Structure,
                    targetFunctionName: null,
                    inputTypeName: null);
            }
            return FastOnnxProtoFactory.CreateSubgraphOutputInfo(key);
        }

        private static TensorDim[] MakeUnnamedDims(int rank)
        {
            var dims = new TensorDim[rank];
            for (int i = 0; i < rank; i++) dims[i] = new TensorDim();
            return dims;
        }

        /// <summary>
        /// Returns the open node's <c>FullOutputs[group]</c> slot — the
        /// subgraph's interface inputs (e.g. iter_num + cond + loop-carried
        /// for a Loop body, or the branch's outer-scope refs for an IF).
        /// </summary>
        private static IReadOnlyList<FastTensorKey?> SubgraphInputsFromOpen(FastNode openNode, string group)
        {
            if (openNode.FullOutputs.TryGetValue(group, out var slot)) return slot;
            return Array.Empty<FastTensorKey?>();
        }

        // ----------- boundary-proto collection -----------

        private static TensorProto[] CreateInitializerTensors(FastComputationGraph fastGraph)
        {
            var list = new List<TensorProto>();
            foreach (var node in fastGraph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                var outputKey = node.Outputs.FirstOrDefault(k => k is not null && !k.Value.IsEmpty);
                if (outputKey is null) continue;
                list.Add(FastOnnxProtoFactory.CreateInitializer(node, outputKey.Value));
            }
            return list.ToArray();
        }

        private static ValueInfoProto[] CreateInputInfos(FastComputationGraph fastGraph)
        {
            // Map graph-input keys back to their producing node so we can read
            // dtype/rank/structure off the node's attributes.
            var producerByOutputKey = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in fastGraph.Nodes)
            {
                if (!FastOpsetResolver.IsModelInputOpCode(node.OpCode)
                 && node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                foreach (var slot in node.FullOutputs.Values)
                    foreach (var k in slot)
                        if (k is FastTensorKey tk && !tk.IsEmpty)
                            producerByOutputKey[tk] = node;
            }

            var infos = new List<ValueInfoProto>(fastGraph.Inputs.Count);
            foreach (var key in fastGraph.Inputs)
            {
                if (!producerByOutputKey.TryGetValue(key, out var producer))
                    throw new InvalidOperationException(
                        $"FastOnnxModelBuilder: graph input {key} has no producing node in the Fast graph.");
                infos.Add(FastOnnxProtoFactory.CreateGraphInputInfo(producer, key));
            }
            return infos.ToArray();
        }

        private static ValueInfoProto[] CreateOutputInfos(FastComputationGraph fastGraph)
        {
            var infos = new ValueInfoProto[fastGraph.Outputs.Count];
            for (int i = 0; i < fastGraph.Outputs.Count; i++)
                infos[i] = FastOnnxProtoFactory.CreateGraphOutputInfo(fastGraph.Outputs[i]);
            return infos;
        }
    }
}
