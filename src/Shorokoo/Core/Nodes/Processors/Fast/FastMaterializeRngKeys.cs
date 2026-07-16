using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Runs the RNG key "initializers": materializes every <c>SHRK_RNG_KEY_PARAM</c> entity's value —
    /// the dense [N, 2] int64 key table over the site's enumerated iteration grid, each row the
    /// runtime master folded along the cell's fully realized ModelId path — from the given
    /// <see cref="RngConfig"/>, exactly as <see cref="FastInitializeModelParams"/> materializes
    /// trainable-parameter values by running their initializers. A key initializer's only inputs
    /// are the config identity and the site's ModelId structure — no model tensors — so
    /// re-running it under a new identity (re-bind) is always safe: <c>ApplyRngConfig</c>
    /// re-materializes keys and leaves parameter values untouched. (In-place re-initialization
    /// of trainable params under a new identity is the designed extension of the same
    /// operation, not yet wired to the re-apply path.)
    /// </summary>
    internal static class FastMaterializeRngKeys
    {
        public static void Process(FastComputationGraph graph, RngConfig rngConfig)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            if (rngConfig is null) throw new ArgumentNullException(nameof(rngConfig));

            foreach (var node in graph.Nodes)
                if (node.OpCode == InternalOpCodes.SHRK_RNG_KEY_PARAM)
                    Materialize(node, rngConfig);
        }

        /// <summary>
        /// One site's key initializer: fills the dense grid cell by cell — grid strides over the
        /// per-level iteration counts, so every row sits at exactly the flat index
        /// Σ iterationIndex[j] · stride[j] the feed computes at runtime, jagged observed sets
        /// included (cells no valid input ever reaches carry well-defined derived keys and are
        /// simply never selected). A site outside any loop is the 1-row grid of its own key.
        /// </summary>
        public static void Materialize(FastNode node, RngConfig rngConfig)
        {
            var idVals = node.Attributes.GetIntsVal(ShrkAttrLocalModelId);
            if (idVals is null || idVals.Length == 0)
                throw new InvalidOperationException(
                    "SHRK_RNG_KEY_PARAM: the key entity carries no site ModelId.");
            int depth = idVals.Count(v => v == -1);
            var counts = node.Attributes.GetLongsVal(ShrkAttrRngIterCounts) ?? [];
            if (counts.Length != depth)
                throw new InvalidOperationException(
                    $"SHRK_RNG_KEY_PARAM: site ModelId [{string.Join(", ", idVals)}] has {depth} " +
                    $"iteration slot(s) but carries {counts.Length} iteration count(s) — the " +
                    "site was not realized at concretization.");

            long[] strides = new long[depth];
            long total = 1;
            for (int j = depth - 1; j >= 0; j--) { strides[j] = total; total *= counts[j]; }

            var table = new long[2 * total];
            var path = new int[idVals.Length];
            for (long flat = 0; flat < total; flat++)
            {
                int slot = 0;
                for (int i = 0; i < idVals.Length; i++)
                {
                    if (idVals[i] == -1)
                    {
                        path[i] = checked((int)(flat / strides[slot] % counts[slot]));
                        slot++;
                    }
                    else path[i] = idVals[i];
                }
                var (k0, k1) = rngConfig.FoldRunKey(path);
                table[2 * flat] = k0;
                table[2 * flat + 1] = k1;
            }

            var data = new OnnxTensorData<int64>(
                new Shape(total, 2),
                OnnxUtils.CreateTensorValue(new Shape(total, 2), table));
            node.Attributes = node.Attributes.SetAttributes((AttrValue, (object?)data));
        }
    }
}
