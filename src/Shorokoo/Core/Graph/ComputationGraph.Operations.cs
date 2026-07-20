using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Runtime;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Utils;
using Shorokoo.Modules;
using System.Collections.Immutable;

namespace Shorokoo.Graph
{
    /// <summary>
    /// The user-facing operations on a readonly <see cref="ComputationGraph"/> —
    /// architecture lowering, weight binding, RNG configuration, parameter queries,
    /// C# emission. Each operation first checks <see cref="Kind"/> and refuses
    /// inappropriate inputs with an error naming the actual and required kinds, then
    /// runs the corresponding pipeline on a mutable copy
    /// (<see cref="InternalComputationGraphExtensions"/>) and freezes the result with
    /// its new kind. The wrapped graph never escapes; a caller-held graph never
    /// observes mutation.
    /// </summary>
    public sealed partial class ComputationGraph
    {
        private const string LoweringOrderHint =
            "The lowering order is: module graph (e.g. MyModule.ComputationGraph) " +
            "-> ToConcreteArchitecture(inputHints, ...) -> ToConcreteModel(...).";

        /// <summary>
        /// Lowers a module graph to a <b>concrete architecture</b>: inlines every sub-module and
        /// function, resolves identifier templates, converts trainable-parameter references, unpacks
        /// model/tensor structs, lowers variant ops, and runs autodiff-ready simplification — so that
        /// every trainable parameter becomes visible at the top level. This is the prerequisite step
        /// for <see cref="ToConcreteModel()"/>, <see cref="InitializeTrainableParams"/>,
        /// and <see cref="GetConcreteModelParamInfos"/>. Requires a
        /// <see cref="GraphKind.Module"/> graph; the result is stamped
        /// <see cref="GraphKind.ConcreteArchitecture"/>.
        ///
        /// <para>See <see cref="InternalComputationGraphExtensions.ToConcreteArchitecture"/> for
        /// the concreteness contract (static ModelIds) the returned graph satisfies.</para>
        /// </summary>
        /// <param name="inputHints">Sample inputs (names + shapes/values) used as shape hints and as
        /// QEE/ORT resolution fallbacks during lowering; build one with <see cref="FromOrderedInputs"/>.</param>
        /// <param name="computeContext">Optional context used to resolve values while lowering.</param>
        /// <param name="debugRequests">Optional hook to dump the graph at each lowering stage.</param>
        /// <returns>A fully inlined, concrete architecture graph.</returns>
        public ComputationGraph ToConcreteArchitecture(
            ModelParamList inputHints,
            ComputeContext? computeContext = null,
            DebugRequests? debugRequests = null)
        {
            RequireKind(GraphKind.Module, nameof(ToConcreteArchitecture), LoweringOrderHint);
            return new ComputationGraph(
                _graph.ToConcreteArchitecture(inputHints, computeContext, debugRequests),
                GraphKind.ConcreteArchitecture);
        }

        /// <summary>
        /// Returns a copy of this graph with a partial set of inputs replaced by
        /// constants and the graph constant-folded to specialize those values — typically used to
        /// bake hyperparameters into the graph so they are no longer live inputs. Inputs not listed
        /// in <paramref name="inputValues"/> remain as live graph inputs. The graph's
        /// <see cref="Kind"/> is preserved.
        /// </summary>
        /// <param name="inputValues">
        /// Values to bake in, matched by name against the graph's input names. Build one with
        /// <see cref="FromOrderedInputs"/> when the names match positional order. Names that have
        /// no corresponding graph input, and non-tensor entries (sequences, structs), are silently
        /// ignored.
        /// </param>
        public ComputationGraph Specialize(ModelParamList inputValues)
            => new(_graph.Specialize(inputValues), Kind);

        /// <summary>
        /// Binds the architecture's <b>default</b> trainable-parameter values (produced by each
        /// <c>[TrainableParamInitializer]</c>) into a runnable, weight-filled graph. Requires a
        /// <see cref="GraphKind.ConcreteArchitecture"/> from <see cref="ToConcreteArchitecture"/>;
        /// the result is stamped <see cref="GraphKind.ConcreteModel"/>.
        /// </summary>
        /// <returns>A concrete model graph with default weights, ready to execute.</returns>
        public ComputationGraph ToConcreteModel()
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(ToConcreteModel), LoweringOrderHint);
            return new ComputationGraph(_graph.ToConcreteModel(), GraphKind.ConcreteModel);
        }

        /// <summary>
        /// Binds default weights initialized under the given <see cref="RngConfig"/>. Each random
        /// initializer draws keyed noise on its parameter's own stream, so same-shape parameters get
        /// distinct values and initialization is reproducible for a config. Requires a
        /// <see cref="GraphKind.ConcreteArchitecture"/>; the result is stamped
        /// <see cref="GraphKind.ConcreteModel"/> and carries the config's runtime RNG identity.
        /// </summary>
        public ComputationGraph ToConcreteModel(RngConfig rngConfig)
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(ToConcreteModel), LoweringOrderHint);
            return new ComputationGraph(_graph.ToConcreteModel(rngConfig), GraphKind.ConcreteModel);
        }

        /// <summary>
        /// Binds loaded trainable-parameter values into a concrete (weight-filled) graph, matching
        /// value names to graph parameters with the given framework's naming convention (default:
        /// Shorokoo's own scheme). For weights exported from another framework, use the overload that
        /// takes an explicit <see cref="ModuleParamSetNamingScheme"/>. Requires a
        /// <see cref="GraphKind.ConcreteArchitecture"/>; names that do not resolve are silently
        /// dropped. The result is stamped <see cref="GraphKind.ConcreteModel"/>.
        /// </summary>
        /// <param name="trainableParamValues">Values to bind (e.g. loaded from a SafeTensors file).</param>
        /// <param name="frameworkId">Naming convention of the value names; defaults to <c>FrameworkId.Shorokoo</c>.</param>
        public ComputationGraph ToConcreteModel(
            ModelParamList trainableParamValues,
            FrameworkId? frameworkId = null)
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(ToConcreteModel), LoweringOrderHint);
            return new ComputationGraph(
                _graph.ToConcreteModel(trainableParamValues, frameworkId), GraphKind.ConcreteModel);
        }

        /// <summary>
        /// Binds loaded trainable-parameter values into a concrete (weight-filled) graph using an
        /// explicit <paramref name="namingScheme"/> to remap value names onto graph parameter ids —
        /// the form to use for third-party (PyTorch/timm) checkpoints. Requires a
        /// <see cref="GraphKind.ConcreteArchitecture"/>; names that do not resolve are silently
        /// dropped. The result is stamped <see cref="GraphKind.ConcreteModel"/>.
        /// </summary>
        /// <param name="trainableParamValues">Values to bind.</param>
        /// <param name="namingScheme">Maps each value's name to a graph ModelId.</param>
        public ComputationGraph ToConcreteModel(
            ModelParamList trainableParamValues,
            ModuleParamSetNamingScheme namingScheme)
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(ToConcreteModel), LoweringOrderHint);
            return new ComputationGraph(
                _graph.ToConcreteModel(trainableParamValues, namingScheme), GraphKind.ConcreteModel);
        }

        /// <summary>
        /// Returns a copy of the graph with <paramref name="rngConfig"/> bound: validates it
        /// against the graph's runtime random surface (fail-loud — see
        /// <see cref="RngConfig.Override"/>) and writes the config's runtime identity into the
        /// <c>RngSeed</c> parameter at reserved ModelId [0] — the binding <b>is</b> that
        /// parameter's initialization, exactly as <c>ToConcreteModel</c> fills weights. Every
        /// feed's key derives in-graph from that parameter, so re-binding re-keys every draw — on
        /// a freshly built and a loaded model alike — while parameter values stay untouched.
        /// Because the identity rides the graph through save/load, a loaded model's randomness is
        /// reproducible with no config object. Works on any concretized graph (a concrete
        /// architecture, a concrete model, or a training-rig step graph); the kind is preserved.
        /// </summary>
        public ComputationGraph WithRngConfig(RngConfig rngConfig)
        {
            RequireConcretized(nameof(WithRngConfig),
                "The RNG key-derivation chains it binds to are wired at concretization; " +
                "lower the graph with ToConcreteArchitecture(inputHints, ...) first.");
            var copy = _graph.Clone();
            copy.ApplyRngConfig(rngConfig);
            return new ComputationGraph(copy, Kind);
        }

        /// <summary>
        /// Reads the model's bound RNG identity — the <c>RngSeed</c> parameter's value (written
        /// by <see cref="WithRngConfig"/>; carried through save/load as an ordinary initializer).
        /// Null when the graph has no runtime random surface or no identity is bound yet.
        /// </summary>
        public long[]? TryGetRngSeed() => _graph.TryGetRngSeed();

        /// <summary>
        /// Returns a copy of a <b>concrete model</b> with its trainable parameters re-initialized
        /// under a new RNG identity: re-runs each trainable parameter's initializer with an
        /// in-graph draw keyed by <paramref name="rngConfig"/>'s init collection and overwrites
        /// the parameter values — bit-exact with a fresh build under the same config. An explicit
        /// opt-in: the copy's trained weights are replaced. Model state and the model's runtime
        /// RNG identity are untouched — re-keying the runtime feeds is a separate, equally
        /// explicit <see cref="WithRngConfig"/> call.
        ///
        /// <para>Fails loudly when the parameter inventory needed for keyed initialization —
        /// the source architecture with its initializer functions — is unavailable: the
        /// inventory is in-memory only (initializers are not persisted), so a loaded model
        /// cannot be re-initialized; rebuild from its architecture instead.</para>
        /// </summary>
        public ComputationGraph WithReinitializedTrainableParams(RngConfig rngConfig)
        {
            RequireKind(GraphKind.ConcreteModel, nameof(WithReinitializedTrainableParams),
                "Only a concrete model carries trainable-parameter values to overwrite.");
            var copy = _graph.Clone();
            copy.ReinitializeTrainableParams(rngConfig);
            return new ComputationGraph(copy, GraphKind.ConcreteModel);
        }

        /// <summary>
        /// Returns metadata (ids and shapes) for every trainable parameter in a <b>concrete
        /// architecture</b> — the inventory used to build naming schemes and bind weights.
        /// Requires a <see cref="GraphKind.ConcreteArchitecture"/> from
        /// <see cref="ToConcreteArchitecture"/>.
        /// </summary>
        public ConcreteModelParamInfos GetConcreteModelParamInfos()
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(GetConcreteModelParamInfos), LoweringOrderHint);
            return _graph.GetConcreteModelParamInfos();
        }

        /// <summary>
        /// Runs each trainable parameter's initializer to produce its default values, named via the
        /// given scheme (default: Shorokoo's). This is what <see cref="ToConcreteModel()"/>
        /// binds when called with no values. Requires a <see cref="GraphKind.ConcreteArchitecture"/>.
        /// See <see cref="InternalComputationGraphExtensions.InitializeTrainableParams"/> for the
        /// RNG-config semantics.
        /// </summary>
        public ModelParamList InitializeTrainableParams(
            ModuleParamSetNamingScheme? namingScheme = null,
            ComputeContext? computeContext = null,
            RngConfig? rngConfig = null)
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(InitializeTrainableParams), LoweringOrderHint);
            return _graph.InitializeTrainableParams(namingScheme, computeContext, rngConfig);
        }

        /// <summary>
        /// Builds the <see cref="ModelIdNamingScheme"/> for this concrete architecture's parameters
        /// under Shorokoo's own naming convention — the basis for remapping third-party weight names
        /// onto the graph's parameter ids. Requires a <see cref="GraphKind.ConcreteArchitecture"/>.
        /// </summary>
        public ModelIdNamingScheme GetShorokooIdNamingScheme()
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(GetShorokooIdNamingScheme), LoweringOrderHint);
            return _graph.GetShorokooIdNamingScheme();
        }

        /// <summary>
        /// Builds the RNG stream inventory of a <b>concrete architecture</b>: one entry per
        /// stream — every parameter's init stream and every runtime random feed — with its
        /// ModelId path, consumer kind, parameter name/shape where known, and (when
        /// <paramref name="rngConfig"/> is supplied) the resolved stream key. Requires a
        /// <see cref="GraphKind.ConcreteArchitecture"/>.
        /// </summary>
        public RngStreamReport GetRngStreamReport(RngConfig? rngConfig = null)
        {
            RequireKind(GraphKind.ConcreteArchitecture, nameof(GetRngStreamReport), LoweringOrderHint);
            return _graph.GetRngStreamReport(rngConfig);
        }

        /// <summary>
        /// Pairs the graph's input names (in declaration order) with the supplied values to produce a
        /// <see cref="ModelParamList"/> of named inputs — the <c>inputHints</c> argument for
        /// <see cref="ToConcreteArchitecture"/>, or the inputs for an <c>Execute</c> call.
        /// </summary>
        /// <param name="inputValues">Input values in the same order as the graph's declared inputs.</param>
        public ModelParamList FromOrderedInputs(ImmutableArray<TensorData> inputValues)
            => _graph.FromOrderedInputs(inputValues);

        /// <summary>
        /// Emits C# source that rebuilds this graph (via <c>CSharpModelBuilder</c>) and writes it to
        /// <paramref name="filename"/>, creating the target directory and overwriting any existing file.
        /// </summary>
        public string SaveToCSharp(string filename) => _graph.SaveToCSharp(filename);
    }
}
