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
    /// <see cref="InternalComputationGraph"/>. Runs the standard pre-passes, then
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
    /// <see cref="FastUseUniqueNames"/>), then walked. Each function's body is a
    /// fresh copy of <see cref="Function.OriginalFastGraph"/>, or of its flattened
    /// form for the dialects that cannot carry module machinery inside a body, so
    /// the per-function pre-passes never touch the cached canonical form.
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
        /// Name given to every model's main <see cref="GraphProto"/>. The ONNX spec requires a
        /// non-empty graph name (the reference checker rejects an empty one), and a
        /// <see cref="Shorokoo.Graph.ComputationGraph"/> carries no name of its own, so a fixed
        /// one keeps repeated exports byte-identical. <c>main_graph</c> follows the PyTorch
        /// exporter's convention. Nothing reads it back on import.
        /// </summary>
        private const string MainGraphName = "main_graph";

        /// <summary>
        /// Build an externally loadable ("vanilla" dialect) ONNX <see cref="ModelProto"/>
        /// from a <see cref="Shorokoo.Graph.GraphKind.ConcreteModel"/>-kind
        /// <see cref="Shorokoo.Graph.ComputationGraph"/>. The input graph is not
        /// mutated — all pre-passes run on a clone.
        ///
        /// <para>
        /// This is the user-facing export API, and the vanilla dialect is a guarantee:
        /// every emitted node is a standard ONNX op or a call to an emitted
        /// <see cref="FunctionProto"/>, so the file loads in any stock ONNX runtime.
        /// A graph that cannot be expressed that way — a module-stage graph still
        /// carrying Shorokoo-internal orchestration ops (<c>ShrkCreateModule</c>, …) —
        /// fails here at export time with the offending ops named, instead of writing
        /// a file that only fails when a third-party runtime rejects the custom ops.
        /// Shorokoo's own persistence keeps the internal dialect through
        /// <see cref="BuildInternalOnnxModel"/>.
        /// </para>
        ///
        /// <para>
        /// Graph inputs and outputs are named from the graph's signature
        /// (<see cref="InternalComputationGraph.InputNames"/> /
        /// <see cref="InternalComputationGraph.OutputNames"/>), deduplicated
        /// deterministically; unnamed slots fall back to <c>input_{i}</c> /
        /// <c>output_{i}</c>. Dtypes are always stamped on the I/O ValueInfos, and
        /// dimensions are stamped wherever they are known (known rank → per-dim
        /// symbolic entries, unknown rank → fully dynamic); an output whose rank the ops do not
        /// tell takes the rank its output node declares.
        /// <see cref="Shorokoo.Onnx.OnnxModelImporter"/> round-trips the names.
        /// </para>
        /// </summary>
        public static ModelProto BuildOnnxModel(
            Shorokoo.Graph.ComputationGraph graph,
            OpSetVersion opset = OpSetVersion.OPS_21,
            bool prepForOnnx = false)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            // The reliable-kind form of the FW045 gate: only a concrete model can satisfy
            // the vanilla-dialect guarantee, so refuse everything else up front. The
            // op-scanning check in ThrowIfNotVanillaDialect stays as the emission-side
            // backstop (it also covers internal-graph callers with no stamped kind).
            if (graph.Kind != Shorokoo.Graph.GraphKind.ConcreteModel)
                throw new ModelException(ErrorCodes.FW045, "ONNX export",
                    $"only a '{Shorokoo.Core.Utils.SrkFileFormat.StageName(Shorokoo.Graph.GraphKind.ConcreteModel)}' " +
                    "graph exports to vanilla ONNX that any external runtime can load, but this graph is a " +
                    $"'{Shorokoo.Core.Utils.SrkFileFormat.StageName(graph.Kind)}'. Lower the graph first " +
                    "(ToConcreteArchitecture -> ToConcreteModel) and export that. Shorokoo's own .srk/.zsrk " +
                    "persistence (CompressedFormatUtils.SaveFastGraphToFile/SaveFastGraphToBinary) accepts every graph kind. " +
                    Shorokoo.Core.Utils.SrkFileFormat.WithKindRemedyHint);
            return BuildOnnxModelCore(graph.ToInternal(), opset, prepForOnnx, vanillaExport: true, stage: graph.Kind);
        }

        /// <summary>
        /// Internal-graph form of <see cref="BuildOnnxModel(Shorokoo.Graph.ComputationGraph, OpSetVersion, bool)"/> for callers below the
        /// readonly wrapper (no stamped kind — the vanilla-dialect op scan is the gate).
        /// </summary>
        internal static ModelProto BuildOnnxModel(
            InternalComputationGraph fastGraph,
            OpSetVersion opset = OpSetVersion.OPS_21,
            bool prepForOnnx = false)
            // The emitted model is always stamped IR_10 (OnnxIRFactory.CreateModel);
            // there is deliberately no irVersion parameter to suggest otherwise.
            => BuildOnnxModelCore(fastGraph, opset, prepForOnnx, vanillaExport: true);

        /// <summary>
        /// Internal-dialect ONNX serialization for Shorokoo's own persistence
        /// (<c>.srk</c>/<c>.zsrk</c> via <see cref="Shorokoo.Core.Utils.CompressedFormatUtils"/>)
        /// and the execution pipeline: tensors keep their internal <c>N{k}_T{s}</c>
        /// names and module-stage graphs serialize their Shorokoo-internal ops
        /// unchecked. Files produced this way are re-importable only by
        /// <see cref="Shorokoo.Onnx.OnnxModelImporter"/>; use
        /// <see cref="BuildOnnxModel(Shorokoo.Graph.ComputationGraph, OpSetVersion, bool)"/> for anything meant to leave Shorokoo.
        /// </summary>
        /// <param name="fastGraph">The graph to serialize.</param>
        /// <param name="opset">Default-domain opset stamp (raised as required).</param>
        /// <param name="prepForOnnx">Run the ONNX-executability prep pass.</param>
        /// <param name="stage">Graph kind to stamp into the model metadata, when known.</param>
        /// <param name="applyExecutionLowerings">Run the semantics-narrowing execution
        /// lowerings: <c>STATE_UPDATE_LINK</c> / <c>WITH_STATE_DEPS</c> → Identity (one-shot
        /// inference) and the <c>SHRK_RANDOM_*</c> / <c>SHRK_RNG_*</c> feed lowering to draw
        /// functions / ONNX random fallbacks. The execution-session and export paths need
        /// them; persistence (.srk) passes false — a saved graph must keep its state
        /// machinery and its runtime-feed ops verbatim, or a reloaded module/architecture
        /// would silently lose state semantics or keyed-RNG identity.</param>
        /// <param name="emitInputsAsNodes">Native <c>.srk</c> on-disk dialect only: emit every top-level
        /// model-input op as an ordinary <see cref="NodeProto"/> (carrying all its attributes, incl. the
        /// representative-input shape) in graph-input order, and emit no graph-input
        /// <see cref="ValueInfoProto"/>s; the loader rebuilds the input list from those nodes. The
        /// output nodes are emitted as nodes too, closing the node list, so an output's name and
        /// declared rank ride on them; the graph-output ValueInfos still name the output values. Off
        /// (default) for the execution/compile path, which must keep proper ONNX graph inputs for ONNX
        /// Runtime.</param>
        /// <param name="inputDims">Execution path only: the concrete dimensions to stamp on each top-level
        /// graph input, positionally over <see cref="InternalComputationGraph.Inputs"/> (a null entry, or a
        /// null list, keeps that input rank-only / symbolic). The pre-passes never add, drop or reorder
        /// top-level inputs, so the positions survive them. See
        /// <see cref="FastOnnxProtoFactory.CreateGraphInputInfo"/> for what it buys and when it is safe.</param>
        internal static ModelProto BuildInternalOnnxModel(
            InternalComputationGraph fastGraph,
            OpSetVersion opset = OpSetVersion.OPS_21,
            bool prepForOnnx = false,
            Shorokoo.Graph.GraphKind? stage = null,
            bool applyExecutionLowerings = true,
            bool emitInputsAsNodes = false,
            IReadOnlyList<long[]?>? inputDims = null)
            => BuildOnnxModelCore(fastGraph, opset, prepForOnnx, vanillaExport: false, stage: stage,
                applyExecutionLowerings: applyExecutionLowerings, emitInputsAsNodes: emitInputsAsNodes,
                inputDims: inputDims);

        /// <summary>
        /// Gives each top-level tensor or optional input of <paramref name="graph"/> that declares
        /// no rank the rank of its recorded representative shape (see
        /// <see cref="RepresentativeInputShapes"/>) — an optional's the rank of its element, where
        /// it was concretized present. Only the rank: the exported input keeps symbolic dims, so it
        /// still accepts any size.
        /// </summary>
        private static void DeclareRepresentativeRanks(InternalComputationGraph graph)
        {
            var producers = graph.BuildProducerByOutputMap();
            foreach (var key in graph.Inputs)
            {
                if (!producers.TryGetValue(key, out var node)
                    || !RepresentativeInputShapes.CarriesShape(node)
                    || node.Attributes.GetLongVal(OnnxOpAttributeNames.ShrkAttrRank) is not null
                    || RepresentativeInputShapes.Get(node) is not { } dims
                    || (node.OpCode == InternalOpCodes.MODEL_OPTIONAL_INPUT
                        && dims.AsSpan().SequenceEqual(RepresentativeInputShapes.AbsentOptionalShape)))
                    continue;
                node.Attributes = node.Attributes.SetAttributes(
                    (OnnxOpAttributeNames.ShrkAttrRank, (object?)(long?)dims.Length));
            }
        }

        /// <summary>
        /// Gives each tensor or optional output of <paramref name="prepFast"/> its rank: the one its
        /// output node declares, where it declares one — the rank its type fixes — and else the rank
        /// of the shape it recorded at the samples the graph was concretized at
        /// (<see cref="RecordedOutputShapes"/>), as an input takes the rank of its representative
        /// shape. An output whose rank varies with its inputs is so exported at its samples' rank.
        /// Only the rank: the exported output keeps symbolic dims. An absent optional recorded no
        /// element shape, and a sequence output is left as the lookup has it.
        /// </summary>
        private static void DeclareOutputRanks(
            InternalComputationGraph prepFast, Dictionary<FastTensorKey, FastTensorInfo> lookup)
        {
            var outputNodes = prepFast.OutputNodes;
            foreach (var node in outputNodes)
                if ((InternalComputationGraph.DeclaredRankOf(node) ?? RecordedOutputShapes.RankOf(node)) is int rank
                    && lookup.TryGetValue(InternalComputationGraph.OutputKeyOf(node), out var info)
                    && info.Structure is DataStructure.Tensor or DataStructure.Optional)
                    info.Rank = rank;
        }

        private static ModelProto BuildOnnxModelCore(
            InternalComputationGraph fastGraph,
            OpSetVersion opset,
            bool prepForOnnx,
            bool vanillaExport,
            Shorokoo.Graph.GraphKind? stage = null,
            bool applyExecutionLowerings = true,
            bool emitInputsAsNodes = false,
            IReadOnlyList<long[]?>? inputDims = null)
        {
            if (fastGraph is null) throw new ArgumentNullException(nameof(fastGraph));

            // ----- 1. Clone so we never mutate the caller's graph.
            var prepFast = fastGraph.Clone();

            // A model built for a session here gives every output memory of its own. ONNX Runtime
            // hands back an output that names a graph input, or repeats an earlier output, as the
            // very value it was fed or has already returned, so two tensors would name one buffer,
            // and a write into either -- by its owner, or by a later run writing an output over it --
            // would change the other: the copy of an input its later runs read, say. ONNX Runtime
            // keeps an Identity over such a value and gives its output memory of its own. Before the
            // pre-passes, so the Identity is renamed with the rest of the graph.
            if (prepForOnnx && !vanillaExport) FastIdentityWrapping.WrapAliasedOutputs(prepFast);

            // The ONNX checker requires every main-graph input and output to carry at least a rank
            // (Shorokoo/Shorokoo#387). An input declared without one takes the rank of the shape it
            // was concretized at, which every concrete graph records; its dims stay symbolic. Before
            // the pre-passes, so the tensor-info lookup they build carries the rank through to the
            // outputs derived from it.
            if (vanillaExport) DeclareRepresentativeRanks(prepFast);

            // ----- 2. Run the Fast pre-passes in place. Capture the rename map
            // so we can also remap the tensor-info lookup we'll build below.
            var tensorInfoLookup = RunPrePassesAndBuildLookup(prepFast, prepForOnnx, applyExecutionLowerings);

            // Each output likewise takes its declared rank, else the rank it recorded at the samples,
            // so an exported file gives every output a shape too (Shorokoo/Shorokoo#387).
            if (vanillaExport) DeclareOutputRanks(prepFast, tensorInfoLookup);

            // Reorder so each IF body has then-block nodes positionally first
            // and else-block nodes positionally second. The Fast back-walk used
            // during subgraph extraction relies on this invariant.
            prepFast.ConfigureScopes();

            // Raise the opset stamp just enough to cover post-opset-21 ops anywhere in
            // the model (main graph or function bodies); see FastOpsetResolver.RaiseToRequired.
            opset = FastOpsetResolver.RaiseToRequired(
                prepFast.Nodes
                    .Concat(CollectFunctionsPostOrder(prepFast)
                        .SelectMany(fn => fn.OriginalFastGraph.Nodes)),
                opset);

            // The activation-checkpoint stamp is Shorokoo-private: stripped from anything ORT will
            // run or a user will export, kept in the .srk dialect (no execution lowerings, not
            // vanilla) so a reloaded architecture still carries its [Module(Checkpoint = true)].
            bool stripCheckpointStamp = prepForOnnx || vanillaExport || applyExecutionLowerings;

            // Same split for function bodies: the dialects ORT will run or a user will export
            // write each body with the calls it makes inlined — an initializer calling another
            // initializer's Init, say — so those bodies go out flattened. The .srk dialect keeps
            // the body as authored, so a reloaded module still shows the calls it makes instead
            // of a copy of each callee inlined into every caller.
            bool flattenFunctionBodies = prepForOnnx || vanillaExport || applyExecutionLowerings;

            // The model a backend's session is built from, as against an exported file or the .srk
            // dialect: only it is rewritten around one runtime's optimizer.
            bool forSession = prepForOnnx && !vanillaExport;

            // ----- 3. Build the main GraphProto by walking the Fast graph.
            var graphProto = BuildGraphProto(
                graphName: MainGraphName,
                fastGraph: prepFast,
                opset: opset,
                isFunction: false,
                tensorInfoLookup: tensorInfoLookup,
                emitInputsAsNodes: emitInputsAsNodes,
                // A graph input has no attribute bag, so wherever the inputs are graph inputs each
                // one's representative shape rides in its ValueInfoProto metadata instead; that is what
                // lets the model import back as the concrete graph it was. (The .srk dialect emits
                // the input nodes themselves, attribute and all.)
                emitRepresentativeMetadata: !emitInputsAsNodes,
                // The internal dialects keep a graph input's raw tensor id as its ValueInfo name, so
                // its signature name rides in the ValueInfo's metadata; a vanilla file is named from
                // the signature outright (ApplySignatureIONames).
                emitInputNameMetadata: !vanillaExport && !emitInputsAsNodes,
                // Likewise an output's name and declared rank ride in its ValueInfo's metadata in the
                // internal dialects that keep no output nodes.
                emitOutputMetadata: !vanillaExport && !emitInputsAsNodes,
                inputDims: inputDims,
                stripCheckpointStamp: stripCheckpointStamp);

            // ----- 4. Discover all reachable Functions in post order and emit
            // a FunctionProto for each.
            var functions = CollectFunctionsPostOrder(prepFast);
            var functionProtos = functions
                .Select(fn => BuildFunctionProto(fn, opset, prepForOnnx, applyExecutionLowerings, stripCheckpointStamp, flattenFunctionBodies, forSession))
                .ToArray();

            var model = (ModelProto)OnnxIRFactory.CreateModel(graphProto, functionProtos, opset);

            // ----- 4b. Drop the FunctionProtos nothing in the emitted model reaches. The list
            // above comes from walking each function's un-flattened body, so a callee whose only
            // call flattening spliced away would ship with nothing left to call it
            // (Shorokoo/Shorokoo#288). Only the flattening dialects can orphan one — the .srk
            // body keeps its call — so only they prune, and the graph the loader reads back is
            // never trimmed on a guess.
            if (flattenFunctionBodies)
                RemoveUnreferencedFunctions(model);

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

            // ----- 5b, continued. Every dialect a runtime reads — a backend's session or an exported file,
            // not the .srk one — gets its recurrent nodes' activation_alpha/activation_beta
            // written in full, in the form ONNX Runtime reads as the spec does; see
            // RecurrentActivationArguments.
            if (flattenFunctionBodies)
                RecurrentActivationArguments.Normalize(model, forSession);

            // ----- 5b, continued. Execution dialect only: every DequantizeLinear ONNX Runtime's
            // optimizer would move past what reads it is written so that it keeps its input type;
            // see WriteDequantizeZeroPoints. A workaround for one runtime's optimizer, so an exported
            // file keeps its quantization as the user wrote it. Function bodies are written as they
            // are built (BuildFunctionProto).
            if (forSession)
                WriteDequantizeZeroPoints(model.Graph, BuildTensorMetaByName(tensorInfoLookup));

            // ----- 5c. Execution dialect only: the one AUTO_GRAD node a training step keeps when
            // its gradient is left to the execution backend goes out as that backend's operator.
            // Only such a step reaches here carrying one -- the compile gate refuses AUTO_GRAD in
            // every other format -- and the .srk dialect keeps the node as Shorokoo's own.
            if (forSession)
                EmitDeferredAutoGrad(model);

            // ----- 6. Attach TensorStructDef metadata for any struct-typed
            // inputs/outputs so the loader can reconstruct DType identity.
            AddTensorStructMetadata(model, prepFast, tensorInfoLookup);

            // ----- 6b. Stamp the graph kind into the model metadata when the caller
            // knows it, so a serialized graph reloads as the same kind instead of
            // being re-classified by op-scanning (which cannot see e.g. MODEL_PARAM
            // nodes — they serialize as encoded initializer-function calls).
            if (stage is { } s)
                model.MetadataProps.Add(new StringStringEntryProto
                {
                    Key = OnnxOpAttributeNames.ShrkMetaGraphKind,
                    Value = Shorokoo.Core.Utils.SrkFileFormat.StageName(s),
                });

            // (The model's RNG identity needs no side channel: it is the ordinary RngSeed
            // parameter at ModelId [0], serialized as a plain initializer like any other
            // MODEL_PARAM_DATA and reloaded the same way.)

            // ----- 7. User-facing export only: enforce the vanilla dialect, then
            // finish the model boundary — typed graph outputs and signature-derived
            // I/O names. The internal dialect (.srk persistence, execution pipeline)
            // keeps raw N{k}_T{s} names and skips the dialect check.
            if (vanillaExport)
            {
                ThrowIfNotVanillaDialect(model);
                StampGraphOutputTypes(model.Graph, prepFast, tensorInfoLookup);
                // Names come from prepFast, not the original fastGraph: the proto's
                // I/O slots are built positionally from prepFast's inputs and outputs, and
                // each name rides on its own input or output node — pairing against the
                // original graph would silently mislabel I/O the day a pre-pass adds or drops one.
                ApplySignatureIONames(model.Graph, prepFast.InputNames, prepFast.OutputNames);
            }

            return model;
        }

        /// <summary>
        /// Rewrites each top-level <c>AUTO_GRAD</c> NodeProto of <paramref name="model"/> into
        /// <see cref="TrainingFormats.AutoGradDomain"/>::<see cref="TrainingFormats.AutoGradOpType"/>,
        /// and — when there was one — imports that domain and records the step's format in the model
        /// metadata, which is how a backend recognises what it has been handed. Inputs and outputs
        /// are the node's own: <c>[loss, wrt…]</c> and <c>[grad…]</c>. A model without the node is
        /// left exactly as it was built.
        /// </summary>
        private static void EmitDeferredAutoGrad(ModelProto model)
        {
            var any = false;
            foreach (var node in model.Graph.Nodes)
            {
                if (node.OpType != InternalOpCodes.AUTO_GRAD || node.Domain != "") continue;
                node.OpType = TrainingFormats.AutoGradOpType;
                node.Domain = TrainingFormats.AutoGradDomain;
                any = true;
            }
            if (!any) return;
            model.OpsetImports.Add(new OperatorSetIdProto
            {
                Domain = TrainingFormats.AutoGradDomain,
                Version = TrainingFormats.AutoGradDomainVersion,
            });
            model.MetadataProps.Add(new StringStringEntryProto
            {
                Key = TrainingFormats.MetadataKey,
                Value = TrainingFormats.OnnxAutoGrad,
            });
        }

        // ----------- function pruning -----------

        /// <summary>
        /// Removes every <see cref="FunctionProto"/> the emitted <paramref name="model"/> cannot
        /// reach, transitively, from its main graph.
        ///
        /// <para>A proto is referenced in exactly three ways, and this walk honours all three —
        /// they are the three <see cref="Shorokoo.Onnx.OnnxModelImporter"/> resolves on the way
        /// back in, so anything it can bind, this keeps:</para>
        /// <list type="number">
        ///   <item>a node's <c>op_type</c>, in the <c>Functions</c> domain — a call site;</item>
        ///   <item>a node's <see cref="OnnxOpAttributeNames.ShrkAttrFunctionName"/> attribute — how
        ///     a node whose op type is its own (ShrkCreateModule, ShrkModelInvoke, the sequence ops)
        ///     names the function it carries;</item>
        ///   <item>a ValueInfo's <c>Signature</c> metadata prop — a module-typed value naming the
        ///     signature function it is typed by.</item>
        /// </list>
        ///
        /// <para>The walk descends into every graph-valued attribute, so a call inside an
        /// <c>If</c> or <c>Loop</c> body counts as a reference like any other.</para>
        /// </summary>
        private static void RemoveUnreferencedFunctions(ModelProto model)
        {
            if (model.Functions.Count == 0) return;

            var byName = new Dictionary<string, FunctionProto>(model.Functions.Count);
            foreach (var fn in model.Functions) byName[fn.Name] = fn;

            var reached = new HashSet<string>();
            var pending = new Queue<string>();

            void Reference(string? name)
            {
                if (string.IsNullOrEmpty(name)) return;
                if (!byName.ContainsKey(name!)) return;
                if (reached.Add(name!)) pending.Enqueue(name!);
            }

            void ScanNodes(IEnumerable<NodeProto> nodes)
            {
                foreach (var node in nodes)
                {
                    if (node.Domain == "Functions") Reference(node.OpType);
                    foreach (var attr in node.Attributes)
                    {
                        if (attr.Name == OnnxOpAttributeNames.ShrkAttrFunctionName && attr.S is { } bytes)
                            Reference(System.Text.Encoding.UTF8.GetString(bytes));
                        if (attr.G is { } g) ScanGraph(g);
                        foreach (var sub in attr.Graphs) ScanGraph(sub);
                    }
                }
            }

            void ScanValueInfos(IEnumerable<ValueInfoProto> infos)
            {
                foreach (var info in infos)
                    foreach (var prop in info.MetadataProps)
                        if (prop.Key == Function.IRFunctionSignatureParamName)
                            Reference(prop.Value);
            }

            void ScanGraph(GraphProto graph)
            {
                ScanNodes(graph.Nodes);
                ScanValueInfos(graph.Inputs);
                ScanValueInfos(graph.Outputs);
                ScanValueInfos(graph.ValueInfoes);
            }

            ScanGraph(model.Graph);
            while (pending.Count > 0)
            {
                var fn = byName[pending.Dequeue()];
                ScanNodes(fn.Nodes);
                ScanValueInfos(fn.ValueInfoes);
            }

            if (reached.Count == model.Functions.Count) return;
            var kept = model.Functions.Where(fn => reached.Contains(fn.Name)).ToArray();
            model.Functions.Clear();
            model.Functions.AddAll(kept);
        }

        // ----------- vanilla-dialect guarantee -----------

        /// <summary>
        /// True when <paramref name="opType"/> (a default-domain NodeProto op type) is a
        /// Shorokoo-internal op rather than a standard ONNX op. Registry-owned: an op is
        /// vanilla exactly when it is a registered non-internal definition
        /// (<see cref="Definitions.VanillaOpNames"/>); everything else — internal
        /// registrations and any op the registry does not know — fails closed as internal,
        /// so a future internal op with a plain, convention-free name cannot slip through
        /// the vanilla-dialect guarantee.
        /// </summary>
        private static bool IsInternalOpType(string opType)
            => !Definitions.VanillaOpNames.Contains(opType);

        /// <summary>
        /// Visits <paramref name="graph"/> and every subgraph nested under it via
        /// node graph-attributes (Loop/If bodies), in declaration order — the one
        /// shared traversal behind the vanilla-dialect check, the I/O rename and
        /// proto-lowering passes, and <c>ComputeContext</c>'s optional-op scan, so
        /// subgraph coverage is fixed in a single place. A visit may rewrite the
        /// graph's node list in place: nested subgraphs are enumerated only after
        /// their parent graph's visit returns.
        /// </summary>
        internal static void ForEachGraphRecursive(GraphProto graph, Action<GraphProto> visit)
        {
            visit(graph);
            foreach (var node in graph.Nodes)
                foreach (var attr in node.Attributes)
                {
                    if (attr.G is not null) ForEachGraphRecursive(attr.G, visit);
                    foreach (var g in attr.Graphs) ForEachGraphRecursive(g, visit);
                }
        }

        /// <summary>
        /// Walks every NodeProto in the model — main graph, nested subgraph
        /// attributes, and all FunctionProto bodies — and throws when any node is not
        /// expressible in the vanilla ONNX dialect: a Shorokoo-internal op in the
        /// default domain, or a node in a domain other than the standard one and the
        /// emitted <c>Functions</c> domain (whose calls resolve to emitted
        /// <see cref="FunctionProto"/>s). The error names the offending ops and the fix.
        /// </summary>
        private static void ThrowIfNotVanillaDialect(ModelProto model)
        {
            var offending = new SortedSet<string>(StringComparer.Ordinal);

            void CheckGraph(GraphProto graph)
            {
                foreach (var node in graph.Nodes)
                {
                    var domain = node.Domain;
                    if (domain.Length == 0)
                    {
                        if (IsInternalOpType(node.OpType))
                            offending.Add(node.OpType);
                    }
                    else if (domain != "Functions")
                    {
                        offending.Add($"{domain}:{node.OpType}");
                    }
                }
            }

            ForEachGraphRecursive(model.Graph, CheckGraph);
            foreach (var fn in model.Functions)
            {
                // A FunctionProto body is a bare node list; wrap it so the shared
                // walker also descends into subgraphs nested inside function bodies.
                var fnBody = new GraphProto();
                fnBody.Nodes.AddRange(fn.Nodes);
                ForEachGraphRecursive(fnBody, CheckGraph);
            }

            if (offending.Count == 0) return;
            throw new ModelException(ErrorCodes.FW045, "ONNX export",
                $"the graph contains Shorokoo-internal op(s) that no external ONNX runtime can load: " +
                $"{string.Join(", ", offending)}. Only a concrete model exports to vanilla ONNX — " +
                "lower the graph first (Specialize → ToConcreteArchitecture → ToConcreteModel) and export that. " +
                "Shorokoo's own .srk/.zsrk persistence (CompressedFormatUtils.SaveFastGraphToFile/SaveFastGraphToBinary) " +
                "still accepts module-stage graphs.");
        }

        // ----------- user-facing I/O finishing -----------

        /// <summary>
        /// Replaces the name-only graph-output ValueInfos with typed ones (dtype,
        /// structure, and — when the rank is known — symbolic per-dim entries) pulled
        /// from the Fast tensor-info lookup. Outputs the lookup doesn't know keep
        /// their name-only form for ONNX type inference to fill in at load time.
        /// </summary>
        private static void StampGraphOutputTypes(
            GraphProto graph,
            InternalComputationGraph prepFast,
            Dictionary<FastTensorKey, FastTensorInfo> tensorInfoLookup)
        {
            // CreateOutputInfos emits exactly one ValueInfoProto per prepFast output;
            // a mismatch is a builder bug, and silently truncating here would ship
            // untyped (or wrongly typed) outputs.
            var outputs = prepFast.Outputs;
            Debug.Assert(outputs.Count == graph.Outputs.Count,
                "StampGraphOutputTypes: graph.Outputs must mirror prepFast.Outputs 1:1.");
            for (int i = 0; i < outputs.Count; i++)
            {
                var typed = CreateTypedValueInfo(outputs[i], tensorInfoLookup);
                typed.MetadataProps.AddRange(graph.Outputs[i].MetadataProps);
                graph.Outputs[i] = typed;
            }
        }

        /// <summary>
        /// Signature names that are really serialized internal ids carry no meaning
        /// for an external consumer — treat them as absent so the deterministic
        /// positional fallback applies instead.
        /// </summary>
        private static string? UsableSignatureName(string? name)
        {
            if (string.IsNullOrWhiteSpace(name)) return null;
            // CG-side "guid:index" tensor keys and Fast-side "N{k}"/"N{k}_T{s}" ids.
            if (TensorKey.TryParse(name, out _)) return null;
            if (System.Text.RegularExpressions.Regex.IsMatch(name, "^N[0-9]+(_T[0-9]+)?$")) return null;
            return name;
        }

        /// <summary>
        /// Renames the graph's inputs and outputs from internal <c>N{k}_T{s}</c> ids
        /// to the model's signature names (<paramref name="inputNames"/> /
        /// <paramref name="outputNames"/>), rewriting every reference in the graph and
        /// its nested subgraphs. Deduplication is deterministic: the first claimant
        /// keeps the name, later ones get a <c>_{n}</c> suffix (n = 2, 3, …), and
        /// collisions with any remaining internal tensor name are suffixed the same
        /// way. An output slot whose tensor already carries a name (it is also a graph
        /// input, or a duplicate of an earlier output slot) is bridged with an
        /// <c>Identity</c> node so each declared output still gets its own name.
        /// </summary>
        private static void ApplySignatureIONames(
            GraphProto graph,
            IReadOnlyList<string?> inputNames,
            IReadOnlyList<string?> outputNames)
        {
            var used = new HashSet<string>(StringComparer.Ordinal);
            CollectTensorNames(graph, used);

            var rename = new Dictionary<string, string>(StringComparer.Ordinal);
            var claimed = new HashSet<string>(StringComparer.Ordinal);

            string Assign(string? preferred, string fallback)
            {
                var baseName = UsableSignatureName(preferred) ?? fallback;
                var candidate = baseName;
                for (int n = 2; claimed.Contains(candidate) || used.Contains(candidate); n++)
                    candidate = $"{baseName}_{n}";
                claimed.Add(candidate);
                return candidate;
            }

            for (int i = 0; i < graph.Inputs.Count; i++)
            {
                var oldName = graph.Inputs[i].Name;
                rename[oldName] = Assign(i < inputNames.Count ? inputNames[i] : null, $"input_{i}");
            }

            // An output slot whose tensor already got a name (it is also a graph
            // input, or a duplicate of an earlier output slot) is bridged: its
            // ValueInfo is renamed here directly — the assigned name can't be a
            // rename key (Assign refuses every name already in the graph), so the
            // blanket pass below leaves it alone — and an Identity node feeds it
            // from the tensor's one real name.
            var bridges = new List<(string SourceName, string OutputName)>();
            for (int i = 0; i < graph.Outputs.Count; i++)
            {
                var oldName = graph.Outputs[i].Name;
                var newName = Assign(i < outputNames.Count ? outputNames[i] : null, $"output_{i}");
                if (rename.TryGetValue(oldName, out var sourceName))
                {
                    bridges.Add((sourceName, newName));
                    RenameValueInfo(graph.Outputs[i], oldName, newName);
                }
                else
                    rename[oldName] = newName;
            }

            RenameTensorReferences(graph, rename);

            foreach (var (sourceName, outputName) in bridges)
                graph.Nodes.Add(MakeNode($"{outputName}_identity", OpCodes.IDENTITY, [sourceName], [outputName]));
        }

        /// <summary>Collects every tensor name referenced anywhere in
        /// <paramref name="graph"/> (I/O, value infos, initializers, node edges),
        /// including its nested subgraphs.</summary>
        private static void CollectTensorNames(GraphProto graph, HashSet<string> names)
            => ForEachGraphRecursive(graph, g =>
            {
                foreach (var info in g.Inputs.Concat(g.Outputs).Concat(g.ValueInfoes))
                    names.Add(info.Name);
                foreach (var init in g.Initializers)
                    names.Add(init.Name);
                foreach (var node in g.Nodes)
                {
                    foreach (var n in node.Inputs) names.Add(n);
                    foreach (var n in node.Outputs) names.Add(n);
                }
            });

        /// <summary>Applies the old-name → new-name map to every tensor reference in
        /// <paramref name="graph"/> and its nested subgraphs. Internal names are unique
        /// graph-wide (FastUseUniqueNames), so the blanket rewrite is unambiguous.</summary>
        private static void RenameTensorReferences(GraphProto graph, Dictionary<string, string> rename)
            => ForEachGraphRecursive(graph, g =>
            {
                foreach (var info in g.Inputs.Concat(g.Outputs).Concat(g.ValueInfoes))
                    if (rename.TryGetValue(info.Name, out var newName))
                        RenameValueInfo(info, info.Name, newName);
                foreach (var init in g.Initializers)
                    if (rename.TryGetValue(init.Name, out var newName))
                        init.Name = newName;
                foreach (var node in g.Nodes)
                {
                    for (int i = 0; i < node.Inputs.Count; i++)
                        if (rename.TryGetValue(node.Inputs[i], out var newName))
                            node.Inputs[i] = newName;
                    for (int i = 0; i < node.Outputs.Count; i++)
                        if (rename.TryGetValue(node.Outputs[i], out var newName))
                            node.Outputs[i] = newName;
                }
            });

        /// <summary>Renames a ValueInfoProto and keeps its symbolic dim params (the
        /// <c>{name}_dim{i}</c> placeholders stamped by <see cref="OnnxIRFactory.CreateDims"/>)
        /// in sync with the new name.</summary>
        private static void RenameValueInfo(ValueInfoProto info, string oldName, string newName)
        {
            info.Name = newName;
            var tensorType = info.Type?.TensorType
                ?? info.Type?.SequenceType?.ElemType?.TensorType
                ?? info.Type?.OptionalType?.ElemType?.TensorType;
            if (tensorType?.Shape is not { } shape) return;
            var oldPrefix = oldName + "_dim";
            foreach (var dim in shape.Dims)
                if (dim.ShouldSerializeDimParam() && dim.DimParam.StartsWith(oldPrefix, StringComparison.Ordinal))
                    dim.DimParam = newName + dim.DimParam.Substring(oldName.Length);
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
            InternalComputationGraph fast,
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
            var seenFunctions = new HashSet<Function>();
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
                    Key = OnnxOpAttributeNames.ShrkMetaTensorStructDefPrefix + dtype.ProtoTypeNum.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    Value = def.ToJson(),
                });
            }
        }

        private static void CollectStructDTypesFromGraph(
            InternalComputationGraph graph,
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
            ForEachGraphRecursive(graph, LowerUpsampleToResizeInGraph);
        }

        private static void LowerUpsampleToResizeInGraph(GraphProto graph)
        {
            foreach (var node in graph.Nodes)
            {
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
            ForEachGraphRecursive(graph, g => LowerTrainingBatchNormalizationInGraph(g, tensorMetaByName));
        }

        private static void LowerTrainingBatchNormalizationInGraph(
            GraphProto graph, Dictionary<string, TensorMeta> tensorMetaByName)
        {
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];

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

        /// <summary>
        /// Writes each <c>DequantizeLinear</c> that ONNX Runtime's optimizer would move past the
        /// operator reading it in a form it dequantizes as the spec does. For a model handed to a
        /// backend's session only: an exported file keeps its quantization as written.
        ///
        /// <para>ONNX Runtime moves a <c>DequantizeLinear</c> whose scale is a single value — of
        /// rank 0, or one element — forward past a <c>Reshape</c>, <c>Transpose</c>,
        /// <c>Squeeze</c>, <c>Unsqueeze</c>, <c>Slice</c> or <c>MaxPool</c> reading it (through
        /// any <c>Identity</c>, which it removes first), and one with a scale along an axis past a
        /// <c>Transpose</c>, by inserting after it a <c>QuantizeLinear</c>/<c>DequantizeLinear</c>
        /// pair built from the original's scale and zero point. With no zero point that
        /// <c>QuantizeLinear</c> quantizes to uint8, its own default, whatever the original's input
        /// type was — so an int8, int16, uint16 or int32 input read through such an operator comes
        /// back clamped to what uint8 can hold, or the session is refused over the pair's types.
        /// Every other <c>DequantizeLinear</c> — one read by a convolution or a matrix product, the
        /// patterns ONNX Runtime fuses — is left exactly as written.</para>
        ///
        /// <para>An int8, int16 or uint16 input is given the zero point the spec already implies:
        /// zeros of its own type, in the scale's shape (<c>ConstantOfShape(Shape(scale))</c>), so
        /// the inserted pair keeps the type. An int32 input cannot take that route — no
        /// <c>QuantizeLinear</c> produces int32, and one with an int32 zero point makes the graph
        /// invalid — and is written as the arithmetic it stands for, <c>Cast(x) * scale</c> (after
        /// <c>x - zero_point</c> in int32 where there is one), which is how ONNX Runtime's own
        /// kernel computes it: a one-element scale and zero point read as the single values they
        /// are, one along an axis reshaped along it to broadcast. That rewrite is made where it is
        /// exact: a float32 scale, no block size and no other output type.</para>
        ///
        /// <para>A <c>Transpose</c> that names no permutation and reads a <c>DequantizeLinear</c>
        /// along an axis is given its permutation — the reversal it defaults to: ONNX Runtime's
        /// transpose optimizer reads that permutation without checking there is one, and aborts the
        /// process. Where the rank that permutation needs is not known, the
        /// <c>DequantizeLinear</c> is written as its arithmetic instead, whatever its integer input
        /// type (a narrower one subtracting its zero point in float32, where no difference
        /// wraps), so there is none left for the optimizer to reach.</para>
        /// </summary>
        private static void WriteDequantizeZeroPoints(GraphProto graph, Dictionary<string, TensorMeta> tensorMetaByName)
        {
            if (graph is null) return;
            ForEachGraphRecursive(graph, g => WriteDequantizeZeroPointsInGraph(g, tensorMetaByName));
        }

        /// <summary>The operators ONNX Runtime moves a single-valued <c>DequantizeLinear</c> past.</summary>
        private static readonly HashSet<string> MovedPast =
            [OpCodes.RESHAPE, OpCodes.TRANSPOSE, OpCodes.SQUEEZE, OpCodes.UNSQUEEZE, OpCodes.SLICE, OpCodes.MAX_POOL];

        private static void WriteDequantizeZeroPointsInGraph(GraphProto graph, Dictionary<string, TensorMeta> tensorMetaByName)
        {
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                var node = graph.Nodes[i];
                if (node.OpType != OpCodes.DEQUANTIZE_LINEAR || node.Domain.Length != 0
                    || node.Inputs.Count < 2 || node.Outputs.Count != 1 || node.Outputs[0].Length == 0)
                    continue;
                string x = node.Inputs[0], scale = node.Inputs[1];
                string zeroPoint = node.Inputs.Count > 2 ? node.Inputs[2] : "";
                var scaleMeta = ResolveTensorMeta(graph, tensorMetaByName, scale);
                var scaleDims = ConstantDims(graph, scale);
                bool single = scaleMeta?.Rank == 0 || scaleDims is [] or [1];
                bool alongAxis = scaleDims is [> 1];
                var readers = Readers(graph, node.Outputs[0]);
                if (single ? !readers.Any(r => MovedPast.Contains(r.OpType))
                    : !alongAxis || !readers.Any(r => r.OpType == OpCodes.TRANSPOSE))
                    continue;

                var xMeta = ResolveTensorMeta(graph, tensorMetaByName, x);
                var xType = xMeta?.ProtoElemType;
                var unpermuted = alongAxis
                    ? readers.Where(r => r.OpType == OpCodes.TRANSPOSE && !r.Attributes.Any(a => a.Name == OnnxOpAttributeNames.AttrPerm)).ToList()
                    : [];
                // Where the permutation cannot be written for want of the rank, the node is written
                // as its arithmetic whatever its input type, so no DequantizeLinear is left for the
                // transpose optimizer to reach.
                bool asArithmetic = xType == (int)TensorProto.DataType.Int32;
                if (unpermuted.Count > 0)
                {
                    if ((xMeta?.Rank ?? KnownRank(graph, x)) is int rank)
                        foreach (var transpose in unpermuted)
                            transpose.Attributes.Add(new AttributeProto
                            {
                                Name = OnnxOpAttributeNames.AttrPerm,
                                Type = AttributeProto.AttributeType.Ints,
                                Ints = [.. Enumerable.Range(0, rank).Reverse().Select(d => (long)d)],
                            });
                    else
                        asArithmetic = xType is { } t && (t == (int)TensorProto.DataType.Uint8 || ZeroPointBytes(t) is not null);
                }

                string prefix = (node.Name.Length > 0 ? node.Name : node.Outputs[0]) + "_dqlow";
                if (asArithmetic)
                {
                    if (scaleMeta?.ProtoElemType != (int)TensorProto.DataType.Float
                        || node.Attributes.Any(a => a.Name == OnnxOpAttributeNames.AttrBlockSize && a.I != 0
                            || a.Name == OnnxOpAttributeNames.AttrOutputDtype && a.I != (int)TensorProto.DataType.Float))
                        continue;
                    var lowered = new List<NodeProto>(14);
                    if (scaleDims is [1])
                    {
                        // One element: the single value it holds, which broadcasts over x of any rank
                        // to x's own shape, as the spec's per-tensor reading does.
                        string scalarShape = $"{prefix}_scalar_shape";
                        lowered.Add(MakeInt64ConstNode($"{prefix}_scalar_shape_const", scalarShape, []));
                        lowered.Add(MakeNode($"{prefix}_scale_scalar", OpCodes.RESHAPE, [scale, scalarShape], [$"{prefix}_scale"]));
                        scale = $"{prefix}_scale";
                        if (zeroPoint.Length > 0)
                        {
                            lowered.Add(MakeNode($"{prefix}_zero_point_scalar", OpCodes.RESHAPE, [zeroPoint, scalarShape], [$"{prefix}_zero_point"]));
                            zeroPoint = $"{prefix}_zero_point";
                        }
                    }
                    else if (alongAxis)
                    {
                        // Per axis: the scale (and zero point) are laid along the axis for the
                        // elementwise arithmetic to broadcast, reshaped to
                        // [1] * leading ++ [-1] ++ [1] * trailing, where one of the two counts is
                        // fixed by the axis and the other is the rest of x's rank, read at run time.
                        long axis = node.Attributes.FirstOrDefault(a => a.Name == OnnxOpAttributeNames.AttrAxis)?.I ?? 1;
                        long fixedOnes = axis >= 0 ? axis : -axis - 1;
                        string alongName = $"{prefix}_along", rankName = $"{prefix}_rank";
                        string fixedName = $"{prefix}_fixed_ones", restName = $"{prefix}_rest_ones";
                        lowered.Add(MakeNode($"{prefix}_shape_x", OpCodes.SHAPE, [x], [$"{prefix}_shape_x_out"]));
                        lowered.Add(MakeNode($"{prefix}_rank_of", OpCodes.SHAPE, [$"{prefix}_shape_x_out"], [rankName]));
                        lowered.Add(MakeInt64ConstNode($"{prefix}_rest_less_const", $"{prefix}_rest_less", [fixedOnes + 1]));
                        lowered.Add(MakeNode($"{prefix}_rest_sub", OpCodes.SUB, [rankName, $"{prefix}_rest_less"], [$"{prefix}_rest_count"]));
                        lowered.Add(MakeNode($"{prefix}_rest", OpCodes.CONSTANT_OF_SHAPE, [$"{prefix}_rest_count"], [restName],
                            new AttributeProto
                            {
                                Name = OnnxOpAttributeNames.AttrValue,
                                Type = AttributeProto.AttributeType.Tensor,
                                T = new TensorProto { Dims = [1], data_type = (int)TensorProto.DataType.Int64, RawData = BitConverter.GetBytes(1L) },
                            }));
                        var ones = new long[fixedOnes];
                        Array.Fill(ones, 1L);
                        lowered.Add(MakeInt64ConstNode($"{prefix}_fixed_ones_const", fixedName, ones));
                        lowered.Add(MakeInt64ConstNode($"{prefix}_minus_one_const", $"{prefix}_minus_one", [-1L]));
                        string[] parts = axis >= 0
                            ? [fixedName, $"{prefix}_minus_one", restName]
                            : [restName, $"{prefix}_minus_one", fixedName];
                        lowered.Add(MakeNode($"{prefix}_along_concat", OpCodes.CONCAT, parts, [alongName],
                            MakeIntAttr(OnnxOpAttributeNames.AttrAxis, 0)));
                        lowered.Add(MakeNode($"{prefix}_scale_along", OpCodes.RESHAPE, [scale, alongName], [$"{prefix}_scale"]));
                        scale = $"{prefix}_scale";
                        if (zeroPoint.Length > 0)
                        {
                            lowered.Add(MakeNode($"{prefix}_zero_point_along", OpCodes.RESHAPE, [zeroPoint, alongName], [$"{prefix}_zero_point"]));
                            zeroPoint = $"{prefix}_zero_point";
                        }
                    }
                    // int32 subtracts its zero point in int32, as ONNX Runtime's kernel does; a
                    // narrower type in float32, which holds every difference of two of its values
                    // exactly and cannot wrap.
                    string floats = $"{prefix}_float";
                    if (zeroPoint.Length == 0)
                        lowered.Add(CastToFloat($"{prefix}_cast", x, floats));
                    else if (xType == (int)TensorProto.DataType.Int32)
                    {
                        lowered.Add(MakeNode($"{prefix}_sub", OpCodes.SUB, [x, zeroPoint], [$"{prefix}_centred"]));
                        lowered.Add(CastToFloat($"{prefix}_cast", $"{prefix}_centred", floats));
                    }
                    else
                    {
                        lowered.Add(CastToFloat($"{prefix}_cast", x, $"{prefix}_x_float"));
                        lowered.Add(CastToFloat($"{prefix}_zero_point_cast", zeroPoint, $"{prefix}_zero_point_float"));
                        lowered.Add(MakeNode($"{prefix}_sub", OpCodes.SUB, [$"{prefix}_x_float", $"{prefix}_zero_point_float"], [floats]));
                    }
                    lowered.Add(MakeNode($"{prefix}_mul", OpCodes.MUL, [floats, scale], [node.Outputs[0]]));
                    graph.Nodes.RemoveAt(i);
                    graph.Nodes.InsertRange(i, lowered);
                    i += lowered.Count - 1;
                }
                else if (zeroPoint.Length == 0 && xType is { } narrow && ZeroPointBytes(narrow) is int bytes)
                {
                    string shape = $"{prefix}_scale_shape", zeros = $"{prefix}_zero_point";
                    graph.Nodes.InsertRange(i,
                    [
                        MakeNode($"{prefix}_shape", OpCodes.SHAPE, [scale], [shape]),
                        MakeNode($"{prefix}_zeros", OpCodes.CONSTANT_OF_SHAPE, [shape], [zeros],
                            new AttributeProto
                            {
                                Name = OnnxOpAttributeNames.AttrValue,
                                Type = AttributeProto.AttributeType.Tensor,
                                T = new TensorProto { Dims = [1], data_type = narrow, RawData = new byte[bytes] },
                            }),
                    ]);
                    while (node.Inputs.Count < 3) node.Inputs.Add("");
                    node.Inputs[2] = zeros;
                    i += 2;
                }
            }
        }

        /// <summary>The nodes of <paramref name="graph"/> reading <paramref name="name"/>, looking
        /// through <c>Identity</c> nodes to what reads theirs.</summary>
        private static List<NodeProto> Readers(GraphProto graph, string name)
        {
            var readers = new List<NodeProto>();
            foreach (var reader in graph.Nodes.Where(n => n.Inputs.Contains(name)))
            {
                if (reader.OpType == OpCodes.IDENTITY && reader.Domain.Length == 0 && reader.Outputs.Count == 1)
                    readers.AddRange(Readers(graph, reader.Outputs[0]));
                else
                    readers.Add(reader);
            }
            return readers;
        }

        private static NodeProto CastToFloat(string name, string input, string output)
            => MakeNode(name, OpCodes.CAST, [input], [output],
                MakeIntAttr(OnnxOpAttributeNames.AttrTo, (int)TensorProto.DataType.Float));

        /// <summary>The rank of <paramref name="name"/> where <paramref name="graph"/> fixes it where
        /// it is written, as ONNX Runtime's shape inference would see it: a constant, or the output
        /// of a <c>Reshape</c> to a constant shape; else null.</summary>
        private static int? KnownRank(GraphProto graph, string name)
        {
            if (ConstantDims(graph, name) is { } dims) return dims.Length;
            var producer = graph.Nodes.FirstOrDefault(n => n.Outputs.Contains(name));
            return producer is { OpType: OpCodes.RESHAPE, Inputs.Count: >= 2 } && producer.Domain.Length == 0
                && ConstantDims(graph, producer.Inputs[1]) is [var length] ? (int)length : null;
        }

        /// <summary>The dimensions of <paramref name="name"/> where <paramref name="graph"/> holds it
        /// as a constant — a <c>Constant</c> node's tensor or an initializer — else null.</summary>
        private static long[]? ConstantDims(GraphProto graph, string name)
        {
            foreach (var node in graph.Nodes)
                if (node.OpType == OpCodes.CONSTANT && node.Domain.Length == 0 && node.Outputs.Contains(name))
                    return node.Attributes.FirstOrDefault(a => a.Name == OnnxOpAttributeNames.AttrValue)?.T?.Dims ?? null;
            return graph.Initializers.FirstOrDefault(t => t.Name == name)?.Dims;
        }
        /// <summary>The size of a zero point of <paramref name="protoElemType"/>, for the input
        /// types whose missing zero point <see cref="WriteDequantizeZeroPoints"/> writes out; null
        /// for every other (uint8 is already the default's type).</summary>
        private static int? ZeroPointBytes(int protoElemType) => protoElemType switch
        {
            (int)TensorProto.DataType.Int8 => 1,
            (int)TensorProto.DataType.Int16 or (int)TensorProto.DataType.Uint16 => 2,
            _ => null,
        };

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
        /// What this builder lowers on the way out: the operators it cannot emit — ones the rest
        /// of the framework builds, runs and differentiates as themselves, but that the single
        /// opset Shorokoo writes has no node for. That is a different question from the engines',
        /// which is what they cannot compute or differentiate, so this is a different list: an
        /// operator may be on one, the other, both or neither, and <c>Softsign</c> — emittable,
        /// so absent here — is why exporting an inference model still yields a <c>Softsign</c>.
        ///
        /// <para><c>TensorScatter</c> is the one entry: ONNX introduced it at opset 24 and
        /// opset 21 has no node for it, so a graph carrying one is written out as the concat and
        /// gather its lowering decomposes it into. The pass tests the list before it walks
        /// anything, so a graph with no such node pays nothing beyond one scan.</para>
        /// </summary>
        private static readonly ImmutableHashSet<string> DefaultExportLoweredOpCodes =
            ImmutableHashSet.Create(StringComparer.Ordinal, OpCodes.TENSOR_SCATTER);

        /// <summary>Thread-scoped substitute installed by <see cref="OverrideExportLoweredOpCodes"/>.</summary>
        [ThreadStatic]
        private static ImmutableHashSet<string>? exportLoweredOpCodesOverride;

        /// <summary>
        /// The op codes the export pre-pass decomposes on the calling thread — see
        /// <see cref="DefaultExportLoweredOpCodes"/>.
        /// </summary>
        internal static IReadOnlySet<string> ExportLoweredOpCodes
            => exportLoweredOpCodesOverride ?? DefaultExportLoweredOpCodes;

        /// <summary>
        /// Replaces the export list with <paramref name="opCodes"/> on the calling thread only,
        /// until the returned scope is disposed — so a caller exercising the export lowering over
        /// a decomposition of its own leaves concurrent builds reading the real list.
        /// </summary>
        internal static IDisposable OverrideExportLoweredOpCodes(params string[] opCodes)
            => new ExportLoweredOpCodesScope(opCodes);

        private sealed class ExportLoweredOpCodesScope : IDisposable
        {
            private readonly ImmutableHashSet<string>? previous;

            internal ExportLoweredOpCodesScope(string[] opCodes)
            {
                this.previous = exportLoweredOpCodesOverride;
                exportLoweredOpCodesOverride = ImmutableHashSet.Create(StringComparer.Ordinal, opCodes);
            }

            public void Dispose() => exportLoweredOpCodesOverride = this.previous;
        }

        /// <summary>
        /// Decomposes every operator on <see cref="ExportLoweredOpCodes"/>, and refuses the export
        /// when one survives the pass.
        ///
        /// <para>Unlike the engines, this caller has no graceful degradation to fall back on. An
        /// operator on this list is one the single opset Shorokoo writes has no node for, so a node
        /// left in place is emitted as itself and <see cref="FastOpsetResolver.RaiseToRequired"/> —
        /// which runs after the pre-passes — raises the file's stamp to that operator's own opset.
        /// What ships is then a model ONNX Runtime's CPU provider will not load, with nothing said
        /// about why. The pass reports a decomposition it merely cannot use by leaving the node
        /// alone and one that is wrong by throwing; here the two mean the same thing — there is
        /// nothing to emit — so both stop the export.</para>
        /// </summary>
        private static void LowerForExport(InternalComputationGraph graph)
        {
            FastLowerRegisteredOps.Process(graph, ExportLoweredOpCodes);

            foreach (var node in graph.Nodes)
                if (ExportLoweredOpCodes.Contains(node.OpCode))
                    throw new InvalidOperationException(
                        $"FastOnnxModelBuilder: '{node.OpCode}' has no node at the opset Shorokoo emits, "
                        + "and its registered lowering could not be built for this node, so there is "
                        + "nothing to export. Check the node's inputs and attributes against what the "
                        + "operator's documented domain covers.");
        }

        /// <summary>
        /// Runs every Fast pre-pass on <paramref name="graph"/> in the canonical
        /// pre-pass order. Mutates the graph in place.
        /// </summary>
        private static void RunPrePasses(InternalComputationGraph graph, bool prepForOnnx, bool applyExecutionLowerings)
        {
            FastLowerAttributeTensorOps.Process(graph);
            if (applyExecutionLowerings) FastLowerStateUpdateLinksForInference.Process(graph);
            if (applyExecutionLowerings) FastLowerRandomOps.Process(graph);
            // Decompose what this builder cannot emit, here and not elsewhere in the list:
            //   - after the lowerings above, so an operator one of THEM produces is still offered
            //     to this one, and so the attribute-tensor pass resolves its geometry against the
            //     graph as authored;
            //   - before FastPrepForOnnx, whose reshape composition (the ORT ReshapeFusion
            //     workaround) and close-input identity wrapping must see the decomposition's
            //     nodes, not just the operator they replaced;
            //   - before FastStripCallStacks, because a decomposition is built through
            //     NodeBuilder, which captures a stack trace per node — left after the strip, the
            //     export would embed a fresh one each time and stop being byte-for-byte
            //     reproducible;
            //   - before FastUseUniqueNames and the tensor-info lookup below, so the spliced
            //     nodes are named and typed with the rest of the graph.
            // Off for the persistence dialect — the one caller that turns the execution lowerings
            // off — because a saved architecture must load back as the operator it was authored
            // with rather than as its decomposition; the export that later runs over the reloaded
            // graph decomposes it then.
            if (applyExecutionLowerings) LowerForExport(graph);
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
            InternalComputationGraph graph, bool prepForOnnx, bool applyExecutionLowerings)
        {
            FastLowerAttributeTensorOps.Process(graph);
            if (applyExecutionLowerings) FastLowerStateUpdateLinksForInference.Process(graph);
            if (applyExecutionLowerings) FastLowerRandomOps.Process(graph);
            // Same position, and for the same reasons, as in RunPrePasses above.
            if (applyExecutionLowerings) LowerForExport(graph);
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
        private static List<Function> CollectFunctionsPostOrder(InternalComputationGraph graph)
        {
            var seen = new HashSet<Function>();
            var result = new List<Function>();

            void Visit(InternalComputationGraph g)
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

        private static FunctionProto BuildFunctionProto(
            Function function, OpSetVersion opset, bool prepForOnnx, bool applyExecutionLowerings,
            bool stripCheckpointStamp = true, bool flattenBody = true, bool forSession = false)
        {
            // Run the same pre-passes over the function's own body. That body has its own
            // ONNX-name namespace,
            // so the per-graph counter inside FastUseUniqueNames restarts at 1
            // for each function — matches how ONNX FunctionProtos are scoped.
            // Flattened for the dialects ORT runs or a user exports: every inlinable invoke in the
            // body — a nested initializer's Init is one — is spliced in, so the body the runtime
            // reads is the one the keyed-draw substitution keyed. An initializer body can hold no
            // module machinery for this to leave behind: building one that touches a model is
            // refused (FW055). Both forms hand back a fresh mutable copy, so there is nothing to
            // clone.
            var fnFast = flattenBody ? function.GetFastFlattenedGraph() : function.OriginalFastGraph;

            // Before the pre-passes, so the inserted Identity is renamed with the rest of the body.
            FastIdentityWrapping.WrapAliasedOutputs(fnFast);
            // A body carrying a Loop or an If needs a tensor-info lookup: it is what types that
            // subgraph's inputs, and without one they go out untyped and ORT refuses the model
            // ("does not have type information") — for every dialect, not just a flattened body.
            // Building it costs a round-trip conversion of the body, so a body with no subgraph in
            // it — most of them, and every body on the training hot path — skips it and runs the
            // pre-passes alone, as before. Asking before the pre-passes is safe: none of them
            // introduces control flow (FastIdentityWrapping only wraps the close nodes it finds).
            // A session's body that dequantizes needs one too, for WriteDequantizeZeroPoints to know
            // the input types.
            Dictionary<FastTensorKey, FastTensorInfo>? fnTensorInfoLookup = null;
            if (fnFast.Nodes.Any(n => n.OpCode == OpCodes.LOOP_CLOSE || n.OpCode == OpCodes.IF_CLOSE
                    || forSession && n.OpCode == OpCodes.DEQUANTIZE_LINEAR))
                fnTensorInfoLookup = RunPrePassesAndBuildLookup(fnFast, prepForOnnx, applyExecutionLowerings);
            else
                RunPrePasses(fnFast, prepForOnnx, applyExecutionLowerings);
            fnFast.ConfigureScopes();

            var fnGraphProto = BuildGraphProto(
                graphName: function.DefaultName,
                fastGraph: fnFast,
                opset: opset,
                isFunction: true,
                tensorInfoLookup: fnTensorInfoLookup,
                stripCheckpointStamp: stripCheckpointStamp,
                emitInputNameMetadata: true,
                emitOutputMetadata: true);
            if (forSession && fnTensorInfoLookup is not null)
                WriteDequantizeZeroPoints(fnGraphProto, BuildTensorMetaByName(fnTensorInfoLookup));

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

            if (function.StateOwnership is { } stateOwnership)
                fnProto.MetadataProps.Add(new StringStringEntryProto
                {
                    Key = Function.IRStateOwnershipParamName,
                    Value = stateOwnership.ToString(),
                });

            var nameMeta = new StringStringEntryProto
            {
                Key = Function.IRFunctionFriendlyName,
                Value = function.FriendlyName,
            };
            fnProto.MetadataProps.Add(nameMeta);

            if (function.RngAlgorithm is not null)
                fnProto.MetadataProps.Add(new StringStringEntryProto
                {
                    Key = Function.IRRngAlgorithmParamName,
                    Value = function.RngAlgorithm,
                });
            if (function.RngFunctionKind is not null)
                fnProto.MetadataProps.Add(new StringStringEntryProto
                {
                    Key = Function.IRRngFunctionKindParamName,
                    Value = function.RngFunctionKind,
                });

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
            InternalComputationGraph fastGraph,
            OpSetVersion opset,
            bool isFunction,
            Dictionary<FastTensorKey, FastTensorInfo>? tensorInfoLookup = null,
            bool emitInputsAsNodes = false,
            bool emitRepresentativeMetadata = false,
            bool emitInputNameMetadata = false,
            IReadOnlyList<long[]?>? inputDims = null,
            bool stripCheckpointStamp = true,
            bool emitOutputMetadata = false)
        {
            // .srk dialect (top-level graph only): every model-input op is emitted as an ordinary
            // NodeProto (carrying all its attributes — including the representative-input shape) instead
            // of a graph-input ValueInfoProto, so the saved graph is self-describing. The input NodeProtos
            // are built separately, in graph-input order, and prepended below; the reader rebuilds the
            // input list from them in that order. Function bodies keep formal-parameter inputs, never this.
            bool inputsAsNodes = emitInputsAsNodes && !isFunction;
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
                // never emitted as NodeProto either. (In the .srk dialect the input ops
                // are emitted separately as nodes and prepended after this walk.)
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

                var info = FastOpsetResolver.Resolve(node, graphOpenNode, opset, stripCheckpointStamp);
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
            // .srk dialect: prepend the input-op NodeProtos in graph-input order (they produce no inputs,
            // so they are valid at the front), and emit no graph-input ValueInfoProtos — the reader
            // reconstructs the input list from these nodes in this order.
            if (inputsAsNodes)
                topLevelNodes.AddRange(BuildBoundaryNodeProtos(fastGraph.InputNodes, opset, stripCheckpointStamp));
            foreach (var (idx, proto) in protoByIndex.OrderBy(kv => kv.Key))
            {
                if (swallowed.Contains(idx)) continue;
                topLevelNodes.Add(proto);
            }
            // .srk dialect: the output nodes close the node list the same way, in output order.
            if (inputsAsNodes)
                topLevelNodes.AddRange(BuildBoundaryNodeProtos(fastGraph.OutputNodes, opset, stripCheckpointStamp));

            var initializers = isFunction
                ? Array.Empty<TensorProto>()
                : CreateInitializerTensors(fastGraph);
            var inputInfos = inputsAsNodes
                ? Array.Empty<ValueInfoProto>()
                : CreateInputInfos(fastGraph, emitRepresentativeMetadata, emitInputNameMetadata, inputDims);
            var outputInfos = CreateOutputInfos(fastGraph, emitOutputMetadata, emitRepresentativeMetadata);

            return (GraphProto)OnnxIRFactory.CreateGraph(
                graphName,
                initializers,
                inputInfos,
                outputInfos,
                topLevelNodes.ToArray());
        }

        /// <summary>
        /// Builds the NodeProto for each of <paramref name="boundaryNodes"/> — the input nodes, in
        /// graph-input order, or the output nodes, in output order — for the native <c>.srk</c>
        /// dialect (which emits them as ordinary nodes rather than graph-input ValueInfoProtos). Each is
        /// resolved and emitted through the same path as any interior node, so it carries all of the op's
        /// attributes verbatim. The reader collects these nodes, in this order, as the graph's inputs
        /// and outputs.
        /// </summary>
        private static NodeProto[] BuildBoundaryNodeProtos(IReadOnlyList<FastNode> boundaryNodes, OpSetVersion opset, bool stripCheckpointStamp)
        {
            var protos = new List<NodeProto>();
            foreach (var node in boundaryNodes)
            {
                var info = FastOpsetResolver.Resolve(node, graphOpenNode: null, opset, stripCheckpointStamp)
                    ?? throw new InvalidOperationException(
                        $"FastOnnxModelBuilder: boundary op {node.OpCode} did not resolve to an emittable node.");
                protos.Add(FastOnnxProtoFactory.CreateNodeProto(node, info, graphAttributes: null));
            }
            return protos.ToArray();
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
            InternalComputationGraph fastGraph,
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
                .Select(k => CreateTypedValueInfo(k!.Value, tensorInfoLookup))
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
        /// Builds a typed <see cref="ValueInfoProto"/> for a tensor, pulling
        /// dtype/structure/rank from <paramref name="tensorInfoLookup"/> when
        /// available. Falls back to a name-only ValueInfoProto when the lookup has
        /// no entry — ONNX type inference will then attempt to resolve the type
        /// from the surrounding context. Two callers with different contracts
        /// share this: subgraph input declarations (<see cref="AssembleSubgraph"/>,
        /// where ONNX is permissive) and the user-facing model-boundary outputs
        /// (<see cref="StampGraphOutputTypes"/>, where the typed form is part of
        /// the documented export guarantee) — degrade it with both in mind.
        /// </summary>
        private static ValueInfoProto CreateTypedValueInfo(
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

        private static TensorProto[] CreateInitializerTensors(InternalComputationGraph fastGraph)
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

        private static ValueInfoProto[] CreateInputInfos(
            InternalComputationGraph fastGraph, bool emitRepresentativeMetadata, bool emitInputNameMetadata,
            IReadOnlyList<long[]?>? inputDims = null)
        {
            var inputNodes = fastGraph.InputNodes;
            if (inputDims is not null && inputDims.Count != inputNodes.Count)
                throw new InvalidOperationException(
                    $"FastOnnxModelBuilder: {inputDims.Count} concrete input shape(s) were supplied for a graph " +
                    $"with {inputNodes.Count} input(s); they must correspond one-to-one in input order.");
            // dtype/rank/structure are read off each input node's attributes.
            var infos = new ValueInfoProto[inputNodes.Count];
            for (int i = 0; i < inputNodes.Count; i++)
            {
                var node = inputNodes[i];
                infos[i] = FastOnnxProtoFactory.CreateGraphInputInfo(
                    node, InternalComputationGraph.InputKeyOf(node), emitRepresentativeMetadata, concreteDims: inputDims?[i]);
                // Written whether or not the input has a name — empty where it has none — so the
                // reader never takes the ValueInfo's raw tensor id for one.
                if (emitInputNameMetadata)
                    infos[i].MetadataProps.Add(new StringStringEntryProto
                    {
                        Key = OnnxOpAttributeNames.ShrkAttrInputName,
                        Value = InternalComputationGraph.InputNameOf(node) ?? "",
                    });
            }
            return infos;
        }

        /// <summary>
        /// One ValueInfo per output, named by the value it outputs. With
        /// <paramref name="emitOutputMetadata"/>, the output's name and declared rank ride in the
        /// ValueInfo's metadata, as an input's name does in its own: the dialects that keep no output
        /// nodes (a function body's formal outputs, the internal execution dialect) have nowhere else
        /// to carry them. With <paramref name="emitRecordedShapeMetadata"/> (every dialect whose
        /// main-graph outputs are graph outputs), its recorded shape (<see cref="RecordedOutputShapes"/>)
        /// rides there too, as an input's representative shape does, so the model imports back as the
        /// concrete graph it was.
        /// </summary>
        private static ValueInfoProto[] CreateOutputInfos(
            InternalComputationGraph fastGraph, bool emitOutputMetadata, bool emitRecordedShapeMetadata)
        {
            var outputNodes = fastGraph.OutputNodes;
            var infos = new ValueInfoProto[outputNodes.Count];
            for (int i = 0; i < outputNodes.Count; i++)
            {
                var node = outputNodes[i];
                infos[i] = FastOnnxProtoFactory.CreateGraphOutputInfo(InternalComputationGraph.OutputKeyOf(node));
                if (emitRecordedShapeMetadata && RecordedOutputShapes.Get(node) is { } recorded)
                    infos[i].MetadataProps.Add(new StringStringEntryProto
                    {
                        Key = OnnxOpAttributeNames.ShrkAttrRecordedOutputShape,
                        Value = RepresentativeInputMetadata.FormatDims(recorded),
                    });
                if (!emitOutputMetadata) continue;
                infos[i].MetadataProps.Add(new StringStringEntryProto
                {
                    Key = OnnxOpAttributeNames.ShrkAttrOutputName,
                    Value = InternalComputationGraph.OutputNameOf(node) ?? "",
                });
                if (InternalComputationGraph.DeclaredRankOf(node) is int rank)
                    infos[i].MetadataProps.Add(new StringStringEntryProto
                    {
                        Key = OnnxOpAttributeNames.ShrkAttrDeclaredRank,
                        Value = rank.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    });
            }
            return infos;
        }
    }
}
