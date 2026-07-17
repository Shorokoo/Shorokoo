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
        ///
        /// <para><b>The concreteness contract: static ModelIds.</b> After this call, every
        /// ModelId-based component of the graph is statically known — trainable parameters, RNG
        /// streams (every runtime feed's per-iteration stream set is enumerated here, with loop
        /// slots filled from the iterations observed under <paramref name="inputHints"/>), and
        /// every other id-addressed consumer. That static id space <em>is</em> what makes the
        /// architecture concrete. Some of it derives from input hints that were not constant
        /// folded; executing the concrete artifact with inputs that would produce ModelIds that
        /// did not exist at concretization (e.g. driving a loop past the iteration space
        /// enumerated here) is <em>invalid use</em> — such ids are not re-derived at runtime,
        /// and anything enumerated per-id (parameter storage, RNG stream tables) has no entry
        /// for them. Enumeration failures are therefore hard build errors, never silent
        /// fallbacks (see <c>FastPipelineUnsupportedException</c>).</para>
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

            // Generator-managed drawBase: inject the model-global execution counter and wire
            // it into every runtime random feed, BEFORE template extraction so the counter's
            // state param rides the normal trainable/state param pipeline from here on.
            FastInjectRngDrawCounter.Process(fastGraph);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastInjectRngDrawCounter");

            var identifierTemplatesInfo = FastExtractIdentifierTemplates.Process(fastGraph);
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastExtractIdentifierTemplates");

            FastConvertToIdRefModelParams.Process(fastGraph);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.MODEL_PARAM_REF, InternalOpCodes.MODEL_PARAM_MODEL_REF },
                "After FastConvertToIdRefModelParams");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastConvertToIdRefModelParams");

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

            FastConvertModelParamIdRefToModelParam.Process(
                fastGraph, identifierTemplatesInfo, inputHints, computeContext);
            DebugPrintFast(fastGraph, debugRequests, GraphCreationPoint.AfterProcessTrainableParameters);
            AssertFastGraphDoesNotContainOps(fastGraph,
                new[] { InternalOpCodes.MODEL_PARAM_ID_REF },
                "After FastConvertModelParamIdRefToModelParam");
            FastGraphCycleDetector.AssertAcyclic(fastGraph, "After FastConvertModelParamIdRefToModelParam");

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
                InternalOpCodes.MODEL_PARAM_REF,
                InternalOpCodes.MODEL_PARAM_MODEL_REF,
                InternalOpCodes.MODEL_PARAM_ID_REF,
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
        /// Binds default weights initialized under the given <see cref="RngConfig"/> — equivalent to
        /// <c>graph.ToConcreteModel(graph.InitializeTrainableParams(rngConfig: rngConfig))</c>.
        /// Each random initializer draws keyed noise on its parameter's own stream, so
        /// same-shape parameters get distinct values and initialization is reproducible for
        /// a config. Requires a concrete architecture from <see cref="ToConcreteArchitecture"/>.
        /// </summary>
        public static FastComputationGraph ToConcreteModel(this FastComputationGraph graph, RngConfig rngConfig)
        {
            var defaultTrainableParams = graph.InitializeTrainableParams(rngConfig: rngConfig);
            var concrete = graph.ToConcreteModel(defaultTrainableParams);
            concrete.ApplyRngConfig(rngConfig);
            return concrete;
        }

        /// <summary>
        /// Binds <paramref name="rngConfig"/> to the graph: validates it against the graph's
        /// runtime random surface (fail-loud — see <see cref="RngConfig.Override"/>) and writes
        /// the config's runtime identity into the <c>RngSeed</c> parameter at reserved ModelId
        /// [0] — the binding <b>is</b> that parameter's initialization, exactly as
        /// <c>ToConcreteModel</c> fills weights. Every feed's key derives in-graph from that
        /// parameter (a split chain wired at concretization), so re-binding is a parameter
        /// write that re-keys every draw — on a freshly built and a loaded model alike — while
        /// parameter values stay untouched. Because the identity and the chains ride the graph
        /// through save/load, a loaded model's randomness is reproducible with no config
        /// object. Works on any graph concretized by <c>ToConcreteArchitecture</c>: a concrete
        /// architecture, a concrete model, or a training-rig step graph.
        /// </summary>
        public static void ApplyRngConfig(this FastComputationGraph graph, RngConfig rngConfig)
            => FastBindRngConfig.Process(graph, rngConfig);

        /// <summary>
        /// Reads the model's bound RNG identity — the <c>RngSeed</c> parameter's value (written
        /// by <see cref="ApplyRngConfig"/>; carried through save/load as an ordinary
        /// initializer). Null when the graph has no runtime random surface or no identity is
        /// bound yet. See <c>Core.Rng.RngRuntimeIdentity</c> for the encoding.
        /// </summary>
        public static long[]? TryGetRngSeed(this FastComputationGraph graph)
        {
            var node = FastWireRngKeyDerivation.FindRngSeedNode(graph);
            if (node is null || node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) return null;
            return node.Attributes.GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData)
                ?.As<int64>().AccessMemory().ToArray();
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
            var concrete = FastApplyModelParamValues.Process(fastGraph, paramValuesById);

            // A concrete model with RNG feed sites always carries an identity: when none was
            // bound, bind the default deterministic config now — the RngSeed parameter fills
            // with the default identity and the feeds draw keyed Threefry. "No config" means
            // the default identity, never the non-reproducible ONNX random fallback.
            if (concrete.TryGetRngSeed() is null &&
                FastWireRngKeyDerivation.FindRngSeedNode(concrete) is not null)
                concrete.ApplyRngConfig(RngConfig.Default);
            return concrete;
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
            var infos = nodes.Where(x => x.OpCode == InternalOpCodes.MODEL_PARAM)
                // The RngSeed parameter at reserved ModelId [0] is the model's RNG identity,
                // not a weight: it has no initializer to run, no name to bind values by, and
                // no init stream — ApplyRngConfig is its initialization.
                .Where(x => x.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId)
                    is not ([0]))
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
        /// <param name="rngConfig">
        /// Optional RNG configuration. Each random initializer draws in-graph keyed noise on
        /// its parameter's own stream (so same-shape parameters get distinct values,
        /// reproducibly for a config). When <c>null</c>,
        /// <see cref="RngConfig.Default"/> (master seed 0) is used — under the ALGORITHM the
        /// graph's bound RNG identity records, when one is present, so no-config init draws
        /// with the same bit generator the model's runtime feeds use. The init-collection
        /// identity itself is never persisted (initialization runs to concrete weight values),
        /// so re-running initialization under a specific seed takes an explicit config. An
        /// identity recording an unknown algorithm (e.g. a model written by a newer framework
        /// version) throws rather than initializing under a substitute; pass an explicit config
        /// to deliberately re-key.
        /// </param>
        /// <returns>The default trainable-parameter values, named.</returns>
        public static ModelParamList InitializeTrainableParams(
            this FastComputationGraph graph,
            ModuleParamSetNamingScheme? namingScheme = null,
            ComputeContext? computeContext = null,
            RngConfig? rngConfig = null)
        {
            AssertConcreteArchitecture(graph, nameof(InitializeTrainableParams));
            computeContext ??= ComputeContext.Default;
            if (rngConfig is null && graph.TryGetRngSeed() is { } identityVec)
            {
                var identity = Core.Rng.RngRuntimeIdentity.Decode(identityVec);
                rngConfig = new RngConfig
                {
                    Algorithm = identity.Algorithm
                        ?? throw new System.NotSupportedException(
                            "InitializeTrainableParams: the graph's RngSeed identity records the " +
                            $"unknown algorithm id {identity.AlgorithmId} (likely written by a newer " +
                            "framework version). Initializing under a substitute algorithm would " +
                            "silently diverge from the recorded identity; pass an explicit rngConfig " +
                            "to deliberately re-key the parameters."),
                };
            }
            rngConfig ??= RngConfig.Default;

            var paramNamingInfo = graph.GetConcreteModelParamInfos();
            namingScheme ??= ModuleParamSetNamingScheme.CreateShorokooNamingScheme(paramNamingInfo);

            var initializedParams = FastInitializeModelParams.Process(graph, computeContext, rngConfig, paramNamingInfo);
            var trainableParams = initializedParams
                .Select(x => new TensorDataModelParam(namingScheme.ToName(x.Key).AssertNotNull(), ModelParamType.TrainableParam, x.Value))
                .ToArray();

            return new ModelParamList(trainableParams);
        }

        /// <summary>
        /// Re-initializes a <b>concrete model's</b> trainable parameters in place under a new
        /// RNG identity: re-runs each trainable parameter's initializer with an in-graph draw keyed
        /// by <paramref name="rngConfig"/>'s init collection (validating
        /// <see cref="RngCollection.Params"/> overrides against the parameter inventory
        /// exactly as at first initialization) and overwrites the parameter values — bit-exact
        /// with a fresh build under the same config. An explicit opt-in: it overwrites trained
        /// weights. Model state (e.g. the RNG execution counter) and the model's runtime RNG
        /// identity are untouched — re-keying the runtime feeds is a separate, equally explicit
        /// <see cref="ApplyRngConfig"/> call.
        ///
        /// <para>Fails loudly when the parameter inventory needed for keyed initialization —
        /// the source architecture with its initializer functions — is unavailable: the
        /// inventory is in-memory only (initializers are not persisted), so a loaded model
        /// cannot be re-initialized in place; rebuild from its architecture instead.</para>
        /// </summary>
        public static void ReinitializeTrainableParams(
            this FastComputationGraph graph, RngConfig rngConfig)
        {
            if (rngConfig is null) throw new System.ArgumentNullException(nameof(rngConfig));
            var arch = graph.SourceArchitecture
                ?? throw new System.InvalidOperationException(
                    "ReinitializeTrainableParams: this graph carries no parameter inventory " +
                    "for keyed initialization (no source architecture with initializer " +
                    "functions). The inventory is in-memory only — a loaded model cannot be " +
                    "re-initialized in place. Rebuild the concrete model from its " +
                    "architecture (ToConcreteArchitecture + ToConcreteModel) under the new " +
                    "config instead.");

            // Freshly initialized values under the new identity — the exact first-init code
            // path (per-parameter keyed in-graph draw, Params-override validation included), so
            // the result is bit-exact with a fresh build under the same config.
            var paramInfos = arch.GetConcreteModelParamInfos();
            var initialized = FastInitializeModelParams.Process(
                arch, ComputeContext.Default, rngConfig, paramInfos);

            // Index this model's parameter-data nodes by their specific ModelId.
            var dataNodesById = new Dictionary<ModelId, FastNode>();
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                if (node.IdentifierTemplate is not { } template) continue;
                var parsed = new ModelParamIdentifierTemplate(template);
                dataNodesById[parsed.SpecificModelId] = node;
            }

            var trainableIds = paramInfos.ParamInfos
                .Where(p => dataNodesById.TryGetValue(p.ModelId, out var n)
                            && (n.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? false))
                .Select(p => p.ModelId)
                .ToList();
            if (trainableIds.Count == 0)
                throw new System.InvalidOperationException(
                    "ReinitializeTrainableParams: no trainable parameter of the source " +
                    "architecture resolves to a trainable MODEL_PARAM_DATA node of this graph " +
                    "— this graph is not a concrete model of that architecture.");

            var dataAttrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM_DATA].AttributeDefs;
            foreach (var id in trainableIds)
            {
                var node = dataNodesById[id];
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?>
                    {
                        [OnnxOpAttributeNames.ShrkAttrTensorData] = initialized[id],
                        [OnnxOpAttributeNames.ShrkAttrIsTrainable] = true,
                    }, dataAttrDefs);
            }
        }

        /// <summary>
        /// Builds the <see cref="ModelIdNamingScheme"/> for this concrete architecture's parameters
        /// under Shorokoo's own naming convention — the basis for remapping third-party weight names
        /// onto the graph's parameter ids.
        /// </summary>
        public static ModelIdNamingScheme GetShorokooIdNamingScheme(this FastComputationGraph graph)
            => ModelIdNamingScheme.CreateShorokooNamingScheme(graph.GetConcreteModelParamInfos());

        /// <summary>
        /// Builds the RNG stream inventory of a <b>concrete architecture</b>: one entry per
        /// stream — every parameter's init stream and every runtime random feed — with its
        /// ModelId path, consumer kind, parameter name/shape where known, and (when
        /// <paramref name="rngConfig"/> is supplied) the resolved stream key. The report also
        /// emits the sparse <c>Rng.Pin</c> skeleton (see <see cref="RngStreamReport.EmitPinSkeleton"/>).
        /// Requires a concrete architecture from <see cref="ToConcreteArchitecture"/>.
        /// </summary>
        public static RngStreamReport GetRngStreamReport(
            this FastComputationGraph graph, RngConfig? rngConfig = null)
        {
            var streams = new List<RngStreamInfo>();

            foreach (var info in graph.GetConcreteModelParamInfos().ParamInfos)
            {
                // A param's SITE is its identifier template's generic ModelId (loop-iteration
                // slots as -1) — the exact analogue of a runtime feed's site attribute, so
                // realized per-iteration params of one in-loop definition group by site in the
                // pin skeleton exactly like realized feed streams do.
                var siteVals = info.ParamIdentifier.ModelIdTemplate.Vals;
                var name = info.ToShorokooIdString();
                streams.Add(new RngStreamInfo
                {
                    Collection = RngCollection.Params,
                    ModelIdPath = info.ModelId.Vals,
                    SitePath = siteVals.SequenceEqual(info.ModelId.Vals) ? null : siteVals,
                    Kind = RngStreamKind.ParamInit,
                    Name = name,
                    Shape = info.Shape.Dims,
                    FrameworkOwned = name.Contains(FastInjectRngDrawCounter.CounterName),
                    KeyWords = rngConfig is null
                        ? null
                        : ToKeyWords(rngConfig.FoldInitKey(info.ModelId.Vals)),
                });
            }

            // Runtime feeds: one row per SITE, straight off the feed nodes — no stored
            // enumeration exists anymore. An in-loop site's row keeps the -1 iteration
            // placeholders (its per-iteration streams derive at runtime from the iteration
            // index, so the realized set is unbounded by construction); a static site's row
            // is the realized stream itself and carries its exact, override-aware key.
            // Unrolled clones of one in-loop site are distinct realized sites and each get
            // their own row. Chain wiring is enforced at concretization (the concreteness
            // contract), so an id-bearing feed without its chain means the graph is not a
            // concrete architecture.
            var seenFeedPaths = new HashSet<string>();
            foreach (var node in graph.Nodes)
            {
                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal) continue;
                var idVals = node.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId);
                if (idVals is null || idVals.Length == 0) continue;
                var kind = isUniform ? RngStreamKind.UniformFeed : RngStreamKind.NormalFeed;

                if (node.Inputs.Count < 4 || node.Inputs[3] is null)
                    throw new System.InvalidOperationException(
                        $"GetRngStreamReport: the runtime random feed at ModelId " +
                        $"[{string.Join(", ", idVals)}] carries no key derivation chain — " +
                        "the report requires a concrete architecture (ToConcreteArchitecture).");

                if (!seenFeedPaths.Add(string.Join(",", idVals))) continue;
                bool isRealized = System.Array.IndexOf(idVals, -1) < 0;
                streams.Add(new RngStreamInfo
                {
                    Collection = RngCollection.Runtime,
                    ModelIdPath = idVals,
                    SitePath = null,
                    Kind = kind,
                    KeyWords = rngConfig is null || !isRealized
                        ? null
                        : ToKeyWords(rngConfig.FoldRunKey(idVals)),
                });
            }

            return new RngStreamReport(streams);

            static long[] ToKeyWords((uint k0, uint k1) key) => [key.k0, key.k1];
        }

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

        /// <summary>Returns the non-trainable model-parameter data nodes (e.g. state buffers),
        /// excluding trainable parameters — and the RngSeed parameter (the model's RNG
        /// identity, at reserved ModelId [0]), which is non-trainable data but not state.</summary>
        public static ImmutableArray<Node> GetStateParamDataNodes(this FastComputationGraph graph)
            => graph.GetAllNodes()
                .Where(x => x.IsModelParamData && !x.GetIsTrainable()
                    && x.IdentifierTemplate?.ToString() != FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
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
