using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Runtime;
using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Core.Nodes.Processors.AutoGrad;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Graph
{
    /// <summary>
    /// Extension surface that re-exposes the high-level operations the deleted
    /// <c>ComputationGraph</c> class used to provide (architecture lowering,
    /// trainable-parameter naming, state-update queries, C# emission). Each method
    /// drives the FastCG processor pipeline directly; the legacy CG wrapper is
    /// never materialized.
    /// </summary>
    public static class FastComputationGraphExtensions
    {
        /// <summary>
        /// Lowers a module graph to a <b>concrete architecture</b>: inlines every sub-module and
        /// function, resolves identifier templates, converts trainable-parameter references, unpacks
        /// model/tensor structs, lowers variant ops, and runs autodiff-ready simplification — so that
        /// every trainable parameter becomes visible at the top level. This is the prerequisite step
        /// for <see cref="ToConcreteModel(FastComputationGraph)"/>, <see cref="InitializeTrainableParams"/>,
        /// and <see cref="GetConcreteModelParamInfos"/>.
        /// </summary>
        /// <param name="graph">The module graph to lower (e.g. <c>MyModel.ComputationGraph</c>).</param>
        /// <param name="inputHints">Sample inputs (names + shapes/values) used as shape hints and as
        /// QEE/ORT resolution fallbacks during lowering; build one with <see cref="FromOrderedInputs"/>.</param>
        /// <param name="computeContext">Optional context used to resolve values while lowering.</param>
        /// <param name="debugRequests">Optional hook to dump the graph at each lowering stage.</param>
        /// <returns>A fully inlined, concrete architecture graph.</returns>
        public static FastComputationGraph ToConcreteArchitecture(
            this FastComputationGraph graph,
            ModelParamList inputHints,
            ComputeContext? computeContext = null,
            DebugRequests? debugRequests = null)
        {
            var fastGraph = graph.Clone();
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After Clone");

            FastApplyIdentifierTemplates.Process(fastGraph);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastApplyIdentifierTemplates");

            FastInlineModulesAndFunctions.Process(fastGraph);
            DebugPrintFast(fastGraph, debugRequests, GraphCreationPoint.AfterInlineAllModulesAndFunctions);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.MODEL_INVOKE, InternalOpCodes.FUNCTION_INVOKE },
                "After FastInlineModulesAndFunctions");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastInlineModulesAndFunctions");

            // Prune unreferenced nodes left by inlining (e.g. outer-module
            // MODULE_SET_HYPERPARAMS) so the next stages don't see ghost templates.
            FastProcessorHelper.RemoveUnreachableNodes(fastGraph);

            var identifierTemplatesInfo = FastExtractIdentifierTemplates.Process(fastGraph);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastExtractIdentifierTemplates");

            FastConvertToIdRefTrainableParams.Process(fastGraph);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.TRAINABLE_PARAM_REF, InternalOpCodes.TRAINABLE_PARAM_MODEL_REF },
                "After FastConvertToIdRefTrainableParams");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastConvertToIdRefTrainableParams");

            FastUnpackModelStruct.Process(fastGraph);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.MODULE_SET_HYPERPARAMS, InternalOpCodes.MODEL_HYPERPARAM, InternalOpCodes.GET_MODEL_ID },
                "After FastUnpackModelStruct");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastUnpackModelStruct");

            FastUnpackTensorStructs.Process(fastGraph);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.TENSOR_STRUCT_CREATE, InternalOpCodes.TENSOR_STRUCT_GETFIELD },
                "After FastUnpackTensorStructs");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastUnpackTensorStructs");

            FastConvertTrainableParamIdRefToTrainableParam.Process(
                fastGraph, identifierTemplatesInfo, inputHints, computeContext);
            DebugPrintFast(fastGraph, debugRequests, GraphCreationPoint.AfterProcessTrainableParameters);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.TRAINABLE_PARAM_ID_REF },
                "After FastConvertTrainableParamIdRefToTrainableParam");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastConvertTrainableParamIdRefToTrainableParam");

            FastSimplify.Process(fastGraph);
            DebugPrintFast(fastGraph, debugRequests, GraphCreationPoint.AfterFirstSimplify);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastSimplify #1");

            // Lower attribute-tensorized variant ops (e.g. SHRK_CONV) to their standard ONNX
            // counterparts before autograd: the variant ops have no gradient rule, and the first
            // FastSimplify above has already constant-folded and unrolled loops, so the geometry
            // inputs are resolvable here. inputHints supplies sample values for the QEE/ORT
            // resolution fallbacks; the following FastSimplify folds the lowered ops.
            FastLowerAttributeTensorOps.Process(fastGraph, inputHints, computeContext);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastLowerAttributeTensorOps");

            FastProcessAutoGradProcessor.Process(fastGraph);
            DebugPrintFast(fastGraph, debugRequests, GraphCreationPoint.AfterExpandAutoGrad);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastProcessAutoGradProcessor");

            FastSimplify.Process(fastGraph);
            DebugPrintFast(fastGraph, debugRequests, GraphCreationPoint.FinalGraph);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastSimplify #2");

            var allInternalOps = new[]
            {
                InternalOpCodes.MODULE_SET_HYPERPARAMS,
                InternalOpCodes.MODEL_INVOKE,
                InternalOpCodes.MODEL_HYPERPARAM,
                InternalOpCodes.GET_MODEL_ID,
                InternalOpCodes.TRAINABLE_PARAM_REF,
                InternalOpCodes.TRAINABLE_PARAM_MODEL_REF,
                InternalOpCodes.TRAINABLE_PARAM_ID_REF,
                InternalOpCodes.FUNCTION_INVOKE,
                InternalOpCodes.TENSOR_STRUCT_CREATE,
                InternalOpCodes.TENSOR_STRUCT_GETFIELD,
                InternalOpCodes.AUTO_GRAD,
            };
            FastProcessorHelper.RemoveUnreachableNodes(fastGraph);
            AssertFastGraphDoesNotContainOps(fastGraph, allInternalOps, "Fast final graph validation");

            return fastGraph;
        }

        /// <summary>
        /// Returns a copy of <paramref name="graph"/> with a partial set of inputs replaced by
        /// constants and the graph constant-folded to specialize those values — typically used to
        /// bake hyperparameters into the graph so they are no longer live inputs. Inputs not listed
        /// in <paramref name="inputValues"/> remain as live graph inputs. Does not modify the
        /// original graph; the returned copy is independent.
        /// </summary>
        /// <param name="graph">The graph to specialize (e.g. a concrete model or architecture).</param>
        /// <param name="inputValues">
        /// Values to bake in, matched by name against
        /// <see cref="FastComputationGraph.InputUniqueNames"/>. Build one with
        /// <see cref="FromOrderedInputs"/> when the names match positional order. Names that have
        /// no corresponding graph input, and non-tensor entries (sequences, structs), are silently
        /// ignored.
        /// </param>
        /// <returns>
        /// A new, constant-folded graph with the named inputs replaced by constants and removed
        /// from the graph's input list.
        /// </returns>
        public static FastComputationGraph Specialize(
            this FastComputationGraph graph,
            ModelParamList inputValues)
        {
            var fastGraph = graph.Clone();

            var valueByName = new Dictionary<string, TensorData>();
            foreach (var param in inputValues.ModelParams)
            {
                if (param is TensorDataModelParam tdParam)
                    valueByName[param.ParamName] = tdParam.ToTensorData();
            }

            if (valueByName.Count == 0)
                return fastGraph;

            var constantAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var specializedIndices = new List<int>();

            for (int i = 0; i < fastGraph.Inputs.Count; i++)
            {
                var name = i < fastGraph.InputUniqueNames.Count ? fastGraph.InputUniqueNames[i] : null;
                if (name is null || !valueByName.TryGetValue(name, out var td))
                    continue;

                var node = fastGraph.FindNode(fastGraph.Inputs[i].FastNodeKey);
                if (node is null)
                    continue;

                // Replace the input node in-place with a CONSTANT node, preserving its
                // output key so all downstream consumers remain valid without remapping.
                node.OpCode = OpCodes.CONSTANT;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [OnnxOpAttributeNames.AttrValue] = td },
                    constantAttrDefs);
                node.IdentifierTemplate = null;
                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>();
                node.TargetFunction = null;

                specializedIndices.Add(i);
            }

            if (specializedIndices.Count == 0)
                return fastGraph;

            // Remove baked-in entries from both lists in reverse order to keep indices stable.
            for (int j = specializedIndices.Count - 1; j >= 0; j--)
            {
                var idx = specializedIndices[j];
                fastGraph.Inputs.RemoveAt(idx);
                if (idx < fastGraph.InputUniqueNames.Count)
                    fastGraph.InputUniqueNames.RemoveAt(idx);
            }

            FastSimplify.Process(fastGraph);

            return fastGraph;
        }

        /// <summary>
        /// Binds the architecture's <b>default</b> trainable-parameter values (produced by each
        /// <c>[TrainableParamInitializer]</c>) into a runnable, weight-filled graph — equivalent to
        /// <c>graph.ToConcreteModel(graph.InitializeTrainableParams())</c>. Requires a concrete
        /// architecture from <see cref="ToConcreteArchitecture"/>.
        /// </summary>
        /// <returns>A concrete model graph with default weights, ready to execute.</returns>
        public static FastComputationGraph ToConcreteModel(this FastComputationGraph graph)
        {
            var defaultTrainableParams = graph.InitializeTrainableParams();
            return graph.ToConcreteModel(defaultTrainableParams);
        }

        /// <summary>
        /// Binds loaded trainable-parameter values into a concrete (weight-filled) graph, matching
        /// value names to graph parameters with the given framework's naming convention (default:
        /// Shorokoo's own scheme). For weights exported from another framework, use the overload that
        /// takes an explicit <see cref="ModuleParamSetNamingScheme"/>. Requires a concrete architecture
        /// from <see cref="ToConcreteArchitecture"/>; names that do not resolve are silently dropped.
        /// </summary>
        /// <param name="graph">The concrete architecture to bind weights into.</param>
        /// <param name="trainableParamValues">Values to bind (e.g. loaded from a SafeTensors file).</param>
        /// <param name="frameworkId">Naming convention of the value names; defaults to <c>FrameworkId.Shorokoo</c>.</param>
        /// <returns>A concrete model graph with the supplied weights bound by name.</returns>
        public static FastComputationGraph ToConcreteModel(
            this FastComputationGraph graph,
            ModelParamList trainableParamValues,
            FrameworkId? frameworkId = null)
        {
            frameworkId ??= FrameworkId.Shorokoo;

            var shorokooParamIds = graph.GetConcreteModelParamInfos();
            if (frameworkId == FrameworkId.Shorokoo)
            {
                var shorokooNamingScheme = ModuleParamSetNamingScheme.CreateShorokooNamingScheme(shorokooParamIds);
                return graph.ToConcreteModel(trainableParamValues, shorokooNamingScheme);
            }

            throw new System.ArgumentException(
                $"Framework '{frameworkId.Name}' requires an explicit naming scheme. Use the overload that accepts a ModuleParamSetNamingScheme parameter.",
                nameof(frameworkId));
        }

        /// <summary>
        /// Binds loaded trainable-parameter values into a concrete (weight-filled) graph using an
        /// explicit <paramref name="namingScheme"/> to remap value names onto graph parameter ids —
        /// the form to use for third-party (PyTorch/timm) checkpoints. Requires a concrete architecture
        /// from <see cref="ToConcreteArchitecture"/>; names that do not resolve are silently dropped.
        /// </summary>
        /// <param name="graph">The concrete architecture to bind weights into.</param>
        /// <param name="trainableParamValues">Values to bind.</param>
        /// <param name="namingScheme">Maps each value's name to a graph ModelId.</param>
        /// <returns>A concrete model graph with the supplied weights bound by name.</returns>
        public static FastComputationGraph ToConcreteModel(
            this FastComputationGraph graph,
            ModelParamList trainableParamValues,
            ModuleParamSetNamingScheme namingScheme)
        {
            var modelParams = graph.GetConcreteModelParamInfos();
            var modelIds = modelParams.ModelIds;

            var paramValuesAndModelId = trainableParamValues.ModelParams
                .Select(x => (ModelId: namingScheme.ToModelId(x.ParamName, modelIds), ParamValue: x.ToTensorData()))
                .Where(x => x.ModelId is not null)
                .ToList();

            // Deduplicate by ModelId. Same trainable parameter may appear multiple times
            // with the same id when the underlying model is invoked multiple times.
            var paramValuesById = paramValuesAndModelId
                .GroupBy(x => x.ModelId.AssertNotNull())
                .ToImmutableDictionary(g => g.Key, g => g.First().ParamValue);

            var fastGraph = graph.Clone();
            return FastApplyModelParamValues.Process(fastGraph, paramValuesById);
        }

        /// <summary>
        /// Returns metadata (ids and shapes) for every trainable parameter in a <b>concrete
        /// architecture</b> — the inventory used to build naming schemes and bind weights. Requires a
        /// concrete architecture from <see cref="ToConcreteArchitecture"/> (throws otherwise, because
        /// parameters nested inside un-inlined sub-functions would be missed).
        /// </summary>
        public static ConcreteModelParamInfos GetConcreteModelParamInfos(this FastComputationGraph graph)
        {
            AssertConcreteArchitecture(graph, nameof(GetConcreteModelParamInfos));
            var nodes = FastComputationGraphConverter.BuildNodes(graph).nodesInTopoOrder;
            var infos = nodes.Where(x => x.OpCode == InternalOpCodes.TRAINABLE_PARAM)
                .Select(x => new ConcreteModelParamInfo(x))
                .ToImmutableArray();
            return new ConcreteModelParamInfos(infos);
        }

        /// <summary>
        /// Runs each trainable parameter's initializer to produce its default values, named via the
        /// given scheme (default: Shorokoo's). This is what <see cref="ToConcreteModel(FastComputationGraph)"/>
        /// binds when called with no values. Requires a concrete architecture from
        /// <see cref="ToConcreteArchitecture"/>.
        /// </summary>
        /// <param name="graph">The concrete architecture whose initializers to run.</param>
        /// <param name="namingScheme">Optional scheme for the returned parameter names; defaults to Shorokoo's.</param>
        /// <param name="computeContext">Optional context used to evaluate the initializers.</param>
        /// <returns>The default trainable-parameter values, named.</returns>
        public static ModelParamList InitializeTrainableParams(
            this FastComputationGraph graph,
            ModuleParamSetNamingScheme? namingScheme = null,
            ComputeContext? computeContext = null)
        {
            AssertConcreteArchitecture(graph, nameof(InitializeTrainableParams));
            computeContext ??= ComputeContext.Default;

            var paramNamingInfo = graph.GetConcreteModelParamInfos();
            namingScheme ??= ModuleParamSetNamingScheme.CreateShorokooNamingScheme(paramNamingInfo);

            var initializedParams = FastInitializeModelParams.Process(graph, computeContext);
            var trainableParams = initializedParams
                .Select(x => new TensorDataModelParam(namingScheme.ToName(x.Key).AssertNotNull(), ModelParamType.TrainableParam, x.Value))
                .ToArray();

            return new ModelParamList(trainableParams);
        }

        /// <summary>
        /// Builds the <see cref="ModelIdNamingScheme"/> for this concrete architecture's parameters
        /// under Shorokoo's own naming convention — the basis for remapping third-party weight names
        /// onto the graph's parameter ids.
        /// </summary>
        public static ModelIdNamingScheme GetShorokooIdNamingScheme(this FastComputationGraph graph)
            => ModelIdNamingScheme.CreateShorokooNamingScheme(graph.GetConcreteModelParamInfos());

        /// <summary>
        /// Pairs the graph's input names (in declaration order) with the supplied values to produce a
        /// <see cref="ModelParamList"/> of named inputs — the <c>inputHints</c> argument for
        /// <see cref="ToConcreteArchitecture"/>, or the inputs for an <c>Execute</c> call.
        /// </summary>
        /// <param name="graph">The graph whose input names to pair with the values.</param>
        /// <param name="inputValues">Input values in the same order as the graph's declared inputs.</param>
        /// <returns>The inputs as a named <see cref="ModelParamList"/>.</returns>
        public static ModelParamList FromOrderedInputs(this FastComputationGraph graph, ImmutableArray<TensorData> inputValues)
        {
            return new ModelParamList(graph.InputUniqueNames.Zip(inputValues)
                .Select(x => new TensorDataModelParam(x.First.AssertNotNull(), ModelParamType.InputParam, x.Second)));
        }

        /// <summary>Returns every node of the graph in topological (dependency) order.</summary>
        public static ImmutableArray<Node> GetAllNodes(this FastComputationGraph graph)
            => FastComputationGraphConverter.BuildNodes(graph).nodesInTopoOrder;

        /// <summary>
        /// Returns the nodes that link a state variable to its per-step updated value — i.e. the
        /// <c>Globals.StateUpdate(state, newValue)</c> registrations in the graph.
        /// </summary>
        public static ImmutableArray<Node> GetStateUpdateLinkNodes(this FastComputationGraph graph)
            => graph.GetAllNodes().Where(x => x.IsStateUpdateLink).ToImmutableArray();

        /// <summary>Returns the nodes carrying state dependencies (those threaded through state updates).</summary>
        public static ImmutableArray<Node> GetWithStateDepsNodes(this FastComputationGraph graph)
            => graph.GetAllNodes().Where(x => x.IsWithStateDeps).ToImmutableArray();

        /// <summary>The number of state-update outputs in the graph (one per <c>StateUpdate</c> registration).</summary>
        public static int GetStateUpdateOutputCount(this FastComputationGraph graph)
            => graph.GetStateUpdateLinkNodes().Length;

        /// <summary>Returns each state update as an <c>(original, updated)</c> variable pair.</summary>
        public static ImmutableArray<(Variable original, Variable updated)> GetStateUpdatePairs(this FastComputationGraph graph)
            => graph.GetStateUpdateLinkNodes()
                .Select(node => (original: node.Inputs[0]!, updated: node.Inputs[1]!))
                .ToImmutableArray();

        /// <summary>Returns the non-trainable model-parameter data nodes (e.g. state buffers), excluding trainable parameters.</summary>
        public static ImmutableArray<Node> GetStateParamDataNodes(this FastComputationGraph graph)
            => graph.GetAllNodes()
                .Where(x => x.IsModelParamData && !x.GetIsTrainable())
                .ToImmutableArray();

        /// <summary>Returns the distinct model-parameter identifier templates referenced by the graph.</summary>
        public static ImmutableArray<ModelParamIdentifierTemplate> GetModelParamIdentifierTemplates(this FastComputationGraph graph)
            => graph.GetAllNodes()
                .Select(x => x.IdentifierTemplate)
                .NotNulls()
                .DistinctBy(x => x.SpecificModelId)
                .ToImmutableArray();

        /// <summary>
        /// Emits C# source that rebuilds this graph (via <c>CSharpModelBuilder</c>) and writes it to
        /// <paramref name="filename"/>, creating the target directory and overwriting any existing file.
        /// </summary>
        /// <param name="graph">The graph to emit as C#.</param>
        /// <param name="filename">Destination path for the generated C# file.</param>
        /// <returns>The path written (same as <paramref name="filename"/>).</returns>
        public static string SaveToCSharp(this FastComputationGraph graph, string filename)
        {
            if (File.Exists(filename))
                File.Delete(filename);

            string? directoryPath = Path.GetDirectoryName(filename);
            if (directoryPath is null)
                throw new InvalidTensorOperationException(ErrorCodes.VG022, "SaveToCSharp", filename,
                    "Directory path is null from filename - cannot create directory for file operations");

            Directory.CreateDirectory(directoryPath);

            var code = new CSharpModelBuilder().BuildFullGraph(graph, "CodeModel");
            using (var textWriter = File.CreateText(filename))
            {
                textWriter.WriteLine(code);
            }

            return filename;
        }

        private static void DebugPrintFast(FastComputationGraph fastGraph, DebugRequests? debugRequests, GraphCreationPoint point)
        {
            if (debugRequests is null) return;
            debugRequests.PrintDebug(fastGraph, point);
        }

        /// <summary>
        /// Trainable-parameter discovery only sees the graph's top-level nodes. On a graph that
        /// still contains <c>MODEL_INVOKE</c>/<c>FUNCTION_INVOKE</c> nodes the parameters are nested
        /// inside sub-functions and would be silently missed, so require a concrete architecture
        /// (produced by <see cref="ToConcreteArchitecture"/>) instead of returning an empty set.
        /// </summary>
        private static void AssertConcreteArchitecture(FastComputationGraph graph, string operation)
        {
            var invokeOps = new HashSet<string> { InternalOpCodes.MODEL_INVOKE, InternalOpCodes.FUNCTION_INVOKE };
            var invokeCount = graph.Nodes.Count(n => invokeOps.Contains(n.OpCode));
            if (invokeCount == 0) return;

            throw new System.InvalidOperationException(
                $"{operation} requires a concrete architecture graph, but this graph still contains "
                + $"{invokeCount} un-inlined module/function invocation(s) "
                + $"({InternalOpCodes.MODEL_INVOKE}/{InternalOpCodes.FUNCTION_INVOKE}). Its trainable parameters "
                + "are nested inside sub-functions and would be missed. Call ToConcreteArchitecture(inputHints, ...) "
                + "first, then run this on the returned graph.");
        }

        private static void AssertFastGraphDoesNotContainOps(FastComputationGraph fastGraph, string[] forbiddenOps, string stageName)
        {
            var forbiddenSet = new HashSet<string>(forbiddenOps);
            var found = fastGraph.Nodes.Where(n => forbiddenSet.Contains(n.OpCode)).ToList();
            if (found.Count == 0) return;

            var errorMsg = new System.Text.StringBuilder();
            errorMsg.AppendLine($"FastComputationGraph: {stageName} left forbidden ops in the graph:");
            foreach (var node in found.Take(10))
                errorMsg.AppendLine($"  {node.OpCode} (Key={node.Key})");
            if (found.Count > 10)
                errorMsg.AppendLine($"  ... and {found.Count - 10} more");

            System.Diagnostics.Debug.Fail(errorMsg.ToString());
            throw new System.InvalidOperationException(errorMsg.ToString());
        }
    }
}
