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
    /// Gives every runtime random feed its generator-managed <c>drawBase</c>: one
    /// model-global execution counter — a framework-owned state scalar
    /// (<c>RngExecutionCounter</c>, initialized 0, advanced +1 per execution via the
    /// ordinary StateUpdate machinery) — wired as the drawBase input of every
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
    /// the drawBase channel loses nothing and costs the checkpoint a single scalar. The
    /// counter takes the next free top-level ModelId slot (its init is a draw-free zero
    /// fill, so it consumes no randomness and no config re-keys it).</para>
    /// </summary>
    internal static class FastInjectRngDrawCounter
    {
        public const string CounterName = "RngExecutionCounter";

        public static void Process(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var feeds = graph.Nodes.Where(n =>
                (n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                 n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL) &&
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
            var counterGraph = GraphBuilder.BuildFastComputationGraphFromDelegate(
                (Func<Scalar<int64>>)CounterBody);

            var refNode = counterGraph.Nodes.Single(
                n => n.OpCode == InternalOpCodes.TRAINABLE_PARAM_REF);
            var attrs = refNode.Attributes.GetAttributeVals().ToDictionary();
            attrs[ShrkAttrLocalModelId] = (long[])[counterSlot];
            refNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(attrs, refNode.Attributes.AttributeDefs);
            refNode.IdentifierTemplate = ModelParamIdentifierTemplate.LocalTrainableParam(
                new ModelId(counterSlot), CounterName, 0, ImmutableArray<int>.Empty).ToString();

            // Prepend the counter nodes (top-level scope; they reference nothing in the host
            // graph) and wire its state-dependent int64 scalar output into every feed.
            var drawBaseKey = counterGraph.Outputs[0];
            graph.Nodes.InsertRange(0, counterGraph.Nodes);
            foreach (var feed in feeds)
            {
                var inputs = feed.FullInputs[""];
                while (inputs.Count < 2) inputs.Add(null);
                inputs[1] = drawBaseKey;
            }
        }

        private static Scalar<int64> CounterBody()
        {
            var counter = Globals.CallTrainableParamInitializer(
                (Func<Vector<int64>, Tensor<float32>>)CounterInit, CounterName,
                isTrainable: false, StateOwnership.ModuleOwned,
                Globals.Vector(1L)).ToValue<Tensor<float32>>();
            Globals.StateUpdate(counter, counter + Globals.Scalar(1.0f));
            return counter.Scalar().Cast<int64>();
        }

        // A [1] float32 buffer, like other module-owned state: float32 exactly represents
        // step counts well past any real run and casts to the int64 drawBase at the sites.
        private static Tensor<float32> CounterInit(Vector<int64> shape)
            => Globals.TensorFill(shape, 0.0f);
    }
}
