using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Immutable;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Gives every runtime random feed its generator-managed <c>substreamIndex</c>: one
    /// model-global execution counter — a framework-owned int64 state scalar
    /// (<c>RngExecutionCounter</c>, initialized 0, advanced +1 per execution via the
    /// ordinary StateUpdate machinery) — wired as the substreamIndex input of every
    /// <c>SHRK_RANDOM_*</c> feed that has none. Runs at concretization, right after module
    /// inlining, so the counter is a normal state parameter from then on: the training rig
    /// threads it through the checkpoint (masks vary per step, resumed runs draw exactly
    /// what the uninterrupted run would), while one-shot inference bakes it at 0
    /// (deterministic and effectively stateless, the STATE_UPDATE_LINK lowering to the
    /// original value at ONNX export).
    ///
    /// <para>The counter is the RNG system's responsibility, not the consumer's: modules
    /// just call <c>Globals.Random*</c> and per-step freshness comes from here. One counter
    /// serves all feeds — sites are already decorrelated by their stream KEYS, so sharing
    /// the substreamIndex channel loses nothing and costs the checkpoint a single scalar. The
    /// counter takes the next free top-level ModelId slot (its init is a draw-free zero
    /// fill, so it consumes no randomness and no config re-keys it).</para>
    /// </summary>
    internal static class FastInjectRngDrawCounter
    {
        public const string CounterName = "RngExecutionCounter";

        /// <summary>
        /// True when <paramref name="identifier"/> is the framework-injected execution counter
        /// (<see cref="CounterName"/>), matched <b>structurally</b> — the identifier's leaf
        /// parameter part is named <see cref="CounterName"/> under the <c>TrainableParam</c>
        /// category — rather than by a substring scan of the raw identifier string. The counter's
        /// ModelId slot is assigned dynamically (so, unlike the fixed-slot <c>RngSeed</c> identity
        /// that is matched by a constant template, the slot cannot anchor the match); the stable
        /// signal is the leaf parameter name, matched regardless of slot or module nesting. A
        /// substring scan, by contrast, false-positives whenever the counter name appears anywhere
        /// in the path — e.g. as a <em>module</em> segment or a mere name substring. (A user
        /// <c>TrainableParam</c> whose leaf name is exactly <see cref="CounterName"/> would still
        /// collide; removing even that needs a reserved slot/region and its user-slot renumbering,
        /// deliberately not done here.)
        /// </summary>
        public static bool IsExecutionCounter(ModelParamIdentifierTemplate? identifier)
            => identifier is not null
               && identifier.Category == ModelParamIdentifierTemplatePart.TrainableParamCategory.Name
               && identifier.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Param
               && identifier.Parts[^1].Name == CounterName;

        /// <summary>
        /// String overload of <see cref="IsExecutionCounter(ModelParamIdentifierTemplate)"/> for a
        /// node's raw <c>IdentifierTemplate</c>; a null, empty, or unparseable identifier is not
        /// the counter.
        /// </summary>
        public static bool IsExecutionCounter(string? identifierTemplate)
        {
            if (string.IsNullOrEmpty(identifierTemplate)) return false;
            try { return IsExecutionCounter(new ModelParamIdentifierTemplate(identifierTemplate)); }
            catch (ArgumentException) { return false; }
        }

        /// <summary>
        /// The execution counter's initial value — an <c>int64[1]</c> zero, mirroring
        /// <see cref="CounterInit"/>. Used as the materialization fallback when no value for the
        /// counter is supplied: the counter is framework bookkeeping (a draw counter), so a model
        /// built from a source that omits it — notably a <c>.safetensors</c> interchange file,
        /// which excludes it — starts it fresh at 0 rather than requiring it like a weight. A
        /// native <c>.skpt</c> still supplies its checkpointed value, which takes precedence.
        /// </summary>
        public static TensorData ExecutionCounterInitialValue()
            => TensorData.CreateFromRawBytes(new Shape([1L]), DType.Int64, BitConverter.GetBytes(0L));

        public static void Process(InternalComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var feeds = graph.Nodes.Where(n =>
                (n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                 n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                 n.OpCode == InternalOpCodes.SHRK_RANDOM_BITS) &&
                (n.Inputs.Count < 2 || n.Inputs[1] is null)).ToList();
            if (feeds.Count == 0) return;   // no feeds, or all already wired (idempotent)

            // The counter joins the graph's (fully assigned) id space at the next free
            // top-level slot — appended, so no existing stream re-keys.
            int counterSlot = 1;
            foreach (var n in graph.Nodes)
                if (n.Attributes.IsAttributeDefined(ShrkAttrLocalModelId) &&
                    n.Attributes.GetIntsVal(ShrkAttrLocalModelId) is { Length: > 0 } vals)
                    counterSlot = Math.Max(counterSlot, vals[0] + 1);

            // Trace the counter subgraph with the real machinery (state init + StateUpdate +
            // WithStateDeps output wrapping), then re-slot its param into the host id space.
            var counterGraph = GraphBuilder.BuildInternalComputationGraphFromDelegate(
                (Func<Scalar<int64>>)CounterBody);

            var refNode = counterGraph.Nodes.Single(
                n => n.OpCode == InternalOpCodes.MODEL_PARAM_REF);
            var attrs = refNode.Attributes.GetAttributeVals().ToDictionary();
            attrs[ShrkAttrLocalModelId] = (long[])[counterSlot];
            refNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(attrs, refNode.Attributes.AttributeDefs);
            refNode.IdentifierTemplate = ModelParamIdentifierTemplate.LocalTrainableParam(
                new ModelId(counterSlot), CounterName, 0, ImmutableArray<int>.Empty).ToString();

            // Prepend the counter nodes (top-level scope; they reference nothing in the host
            // graph) and wire its state-dependent int64 scalar output into every feed.
            var substreamIndexKey = counterGraph.Outputs[0];
            graph.Nodes.InsertRange(0, counterGraph.Nodes);
            foreach (var feed in feeds)
            {
                var inputs = feed.FullInputs[""];
                while (inputs.Count < 2) inputs.Add(null);
                inputs[1] = substreamIndexKey;
            }
        }

        private static Scalar<int64> CounterBody()
        {
            var counter = Globals.CallTrainableParamInitializer(
                (Func<Vector<int64>, Tensor<int64>>)CounterInit, CounterName,
                isTrainable: false, StateOwnership.ModuleOwned,
                Globals.Vector(1L)).ToValue<Tensor<int64>>();
            Globals.StateUpdate(counter, counter + Globals.Scalar(1L));
            return counter.Scalar();
        }

        // A [1] int64 buffer: +1 is exact at every step count (a float32 counter saturates
        // at 2^24, silently freezing per-step mask variation). This is the convention for
        // framework-injected counters: int64 state, end to end. The RNG interface takes a
        // uint64 draw position, so FastLowerRandomOps casts at that boundary rather than
        // letting an unsigned type leak into the framework's own state plumbing.
        private static Tensor<int64> CounterInit(Vector<int64> shape)
            => Globals.TensorFill(shape, 0L);
    }
}
