using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Rng;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Linq;
using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Binds an <see cref="RngConfig"/> to a graph: <c>ApplyRngConfig</c> <b>is</b> the
    /// <c>RngSeed</c> parameter's initialization — it validates the config against the graph's
    /// runtime random surface, then writes the config's encoded runtime identity (see
    /// <see cref="RngRuntimeIdentity"/>) into the <c>RngSeed</c> parameter at reserved ModelId
    /// [0] (<c>MODEL_PARAM</c> → <c>MODEL_PARAM_DATA</c>), exactly as <c>ToConcreteModel</c>
    /// fills weights. Every feed's key is a <c>SHRK_RNG_SPLIT</c> chain rooted at that
    /// parameter (wired at concretization — see <see cref="FastWireRngKeyDerivation"/>), so
    /// re-binding — on a freshly built and on a loaded model alike — is a parameter write that
    /// changes every draw by construction; weights are untouched. Changing the override
    /// <em>set</em> additionally re-runs the wiring pass (override routing is structural);
    /// changing override <em>values</em> or the master seed is a pure value write.
    ///
    /// <para>Validation is fail-loud, per the concreteness contract: every id-bearing feed
    /// must carry its derivation chain (wired at <c>ToConcreteArchitecture</c>), a loop-body
    /// feed without per-iteration identity is an error (a single key would repeat identical
    /// values every iteration), and a <see cref="RngCollection.Runtime"/> override that
    /// addresses no runtime stream of the graph throws instead of silently doing nothing.
    /// (<see cref="RngCollection.Params"/> overrides are validated where they are consumed:
    /// parameter initialization.) A legacy file — saved before the RngSeed representation,
    /// carrying baked key-table constants with nothing left to re-key — also fails loudly.</para>
    /// </summary>
    internal static class FastBindRngConfig
    {
        public static void Process(InternalComputationGraph graph, RngConfig rngConfig)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            var feeds = ValidateFeeds(graph);
            var seedNode = FastWireRngKeyDerivation.FindRngSeedNode(graph);
            var runtimeOverrides = rngConfig.RuntimeOverridesSorted();

            if (seedNode is null)
            {
                if (graph.Nodes.Any(n => n.FriendlyName == ShrkRngKeysTensorName))
                    throw new InvalidOperationException(
                        "ApplyRngConfig: this graph was loaded from a file saved before the " +
                        "RNG identity became the RngSeed parameter (it carries the legacy " +
                        $"'{ShrkRngKeysTensorName}' identity tensor and baked key-table " +
                        "constants). Its draws cannot be re-keyed — re-binding would update " +
                        "only the recorded identity while every draw kept the old seed. " +
                        "Load-and-run of the file is unaffected; to re-key, rebuild the " +
                        "concrete model from its architecture under the new config.");

                // No runtime random surface at all: binding is a no-op — but an override that
                // addresses a stream of THIS graph cannot exist, so any Runtime override is
                // the silent-no-op hazard and must fail loudly.
                if (runtimeOverrides.Count > 0)
                    throw new InvalidOperationException(
                        "RngConfig.Override(Runtime, ...) matches no runtime stream of this graph: " +
                        string.Join(", ", runtimeOverrides.Select(o => $"[{string.Join(",", o.path)}]")) +
                        ". This graph has no runtime random feeds.");
                return;
            }

            if (feeds.Count > 0)
            {
                // Fail-loud override validation: a Runtime override that addresses no feed
                // site of this graph would otherwise be a silent no-op — exactly the
                // re-keying hazard explicit seeding exists to prevent. (A site's iteration
                // slots realize at runtime, so an override addresses a site pattern: static
                // slots exactly, iteration slots by non-negative index.)
                var sites = feeds
                    .Select(f => f.Attributes.GetIntsVal(ShrkAttrLocalModelId)!)
                    .ToList();
                var unmatched = runtimeOverrides
                    .Where(o => !sites.Any(s => FastWireRngKeyDerivation.PathMatchesSite(o.path, s)))
                    .Select(o => $"[{string.Join(",", o.path)}]")
                    .ToArray();
                if (unmatched.Length > 0)
                    throw new InvalidOperationException(
                        "RngConfig.Override(Runtime, ...) matches no runtime stream of this graph: " +
                        string.Join(", ", unmatched) +
                        ". Available stream paths are listed by GetRngStreamReport(); overrides must " +
                        "use a reported path exactly.");

                // Structural override routing: if the override SET differs from the one the
                // chains were wired for, re-run the wiring pass (the orphaned chain nodes are
                // swept by ordinary reachability). Same set — the common case, including every
                // seed-only re-bind — leaves the graph untouched.
                var wiredPaths = CurrentlyWiredOverridePaths(seedNode);
                var newPaths = runtimeOverrides.Select(o => o.path).ToArray();
                if (!SamePathSet(wiredPaths, newPaths))
                {
                    var identityLayout = RngRuntimeIdentity.Decode(RngRuntimeIdentity.Build(rngConfig));
                    FastWireRngKeyDerivation.Process(graph,
                        identityLayout.Overrides.Select(o => (o.Path, o.KeyOffset)).ToArray());
                    FastProcessorHelper.RemoveUnreachableNodes(graph);
                }
            }
            else
            {
                // An already-lowered graph — e.g. a file saved before .srk persistence kept
                // the feed ops verbatim (current saves preserve them, so freshly written
                // files re-enter the feeds branch above on reload): the feeds are baked
                // draw-function calls, but the symbolic chains and the RngSeed parameter
                // persist — re-binding is a parameter write that re-keys every draw. What
                // CANNOT change on such a graph is anything structural: the override set
                // (its routing is wired) and the algorithm (its draw functions are baked).
                var current = DecodeCurrentIdentity(seedNode)
                    ?? throw new InvalidOperationException(
                        "ApplyRngConfig: the RngSeed parameter carries no identity value on a " +
                        "graph whose random feeds are already lowered — the graph was modified " +
                        "since concretization.");
                if (!current.HasSameOverridePaths(runtimeOverrides.Select(o => o.path).ToArray()))
                    throw new InvalidOperationException(
                        "ApplyRngConfig: this graph's random draws are already lowered (a loaded " +
                        "model), so its override routing is fixed. Re-binding may change seed " +
                        "values (master or per-stream) but not the override SET; rebuild from " +
                        "the architecture to change which streams are overridden.");
                if (current.AlgorithmId != RngRuntimeIdentity.AlgorithmIdOf(rngConfig.Algorithm))
                    throw new InvalidOperationException(
                        "ApplyRngConfig: this graph's random draws are already lowered (a loaded " +
                        "model), so its draw algorithm is baked. Re-binding cannot switch the " +
                        "algorithm; rebuild from the architecture under the new config.");
            }

            WriteIdentity(seedNode, RngRuntimeIdentity.Build(rngConfig));
        }

        /// <summary>
        /// Validates the graph's feeds and returns the id-bearing ones. A loop-body feed
        /// without an iteration slot, and an id-bearing feed without its derivation chain,
        /// are hard errors — never silent fallbacks.
        /// </summary>
        private static List<FastNode> ValidateFeeds(InternalComputationGraph graph)
        {
            var feeds = new List<FastNode>();
            int loopDepth = 0;
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode == OpCodes.LOOP_OPEN) loopDepth++;
                else if (node.OpCode == OpCodes.LOOP_CLOSE) loopDepth--;

                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                if (!isUniform && !isNormal)
                    continue;

                var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
                if (idVals is null || idVals.Length == 0)
                {
                    // Not id-bearing (e.g. built by a path that bypasses id assignment): it has
                    // no stream identity, so it stays on the ONNX fallback at lowering.
                    continue;
                }

                if (loopDepth > 0 && Array.IndexOf(idVals, -1) < 0)
                {
                    // A loop-body feed without an iteration slot has no per-iteration stream
                    // identity: a deterministic key would repeat the same values every
                    // iteration. Under a bound config that is an error, not a silent
                    // nondeterministic fallback.
                    throw new InvalidOperationException(
                        $"ApplyRngConfig: the runtime random feed at ModelId [{string.Join(", ", idVals)}] " +
                        "sits inside a loop but carries no iteration slot (no per-iteration " +
                        "stream identity), so it cannot draw deterministically per iteration.");
                }

                if (node.Inputs.Count < 4 || node.Inputs[3] is null)
                    throw new InvalidOperationException(
                        $"ApplyRngConfig: the runtime random feed at ModelId [{string.Join(", ", idVals)}] " +
                        "carries no key derivation chain (no realized stream ids). RNG chains " +
                        "are wired at concretization (ToConcreteArchitecture) — bind the " +
                        "config to a concrete architecture, concrete model, or training-rig " +
                        "step graph.");

                feeds.Add(node);
            }
            return feeds;
        }

        /// <summary>The override paths the existing chains were wired for: the records of the
        /// currently bound identity, or none when the parameter is still value-less (the
        /// architecture-time wiring carries no overrides).</summary>
        private static int[][] CurrentlyWiredOverridePaths(FastNode seedNode)
            => DecodeCurrentIdentity(seedNode)?.Overrides.Select(o => o.Path).ToArray()
               ?? Array.Empty<int[]>();

        private static RngRuntimeIdentity? DecodeCurrentIdentity(FastNode seedNode)
        {
            if (seedNode.OpCode != InternalOpCodes.MODEL_PARAM_DATA) return null;
            var data = seedNode.Attributes.GetTensorVal(ShrkAttrTensorData);
            if (data is null) return null;
            return RngRuntimeIdentity.Decode(data.As<int64>().AccessMemory().ToArray());
        }

        private static bool SamePathSet(int[][] a, int[][] b)
        {
            if (a.Length != b.Length) return false;
            var setA = a.Select(p => string.Join(",", p)).ToHashSet();
            return b.All(p => setA.Contains(string.Join(",", p)));
        }

        /// <summary>Writes the encoded identity into the RngSeed parameter — the parameter's
        /// initialization (MODEL_PARAM → MODEL_PARAM_DATA on first bind, a value replacement
        /// thereafter). Identifier template and output key are preserved, so downstream
        /// consumers (the chains) stay wired.</summary>
        private static void WriteIdentity(FastNode seedNode, long[] identity)
        {
            var data = new OnnxTensorData<int64>(
                new Shape(identity.Length),
                OnnxUtils.CreateTensorValue(new Shape(identity.Length), identity));
            var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_PARAM_DATA].AttributeDefs;
            seedNode.OpCode = InternalOpCodes.MODEL_PARAM_DATA;
            seedNode.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>
                {
                    [ShrkAttrTensorData] = data,
                    [ShrkAttrIsTrainable] = false,
                }, attrDefs);
            seedNode.FullInputs = new Dictionary<string, List<FastTensorKey?>>();
            seedNode.TargetFunction = null;
        }
    }
}
