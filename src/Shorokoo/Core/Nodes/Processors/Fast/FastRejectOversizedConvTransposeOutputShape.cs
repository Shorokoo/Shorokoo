using Shorokoo.Core.Graph;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Graph;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Refuses a <c>ConvTranspose</c> whose <c>output_shape</c> lies so far beyond the full extent
    /// <c>stride * (in - 1) + output_padding + (kernel - 1) * dilation + 1</c> that the pads ONNX
    /// auto-generates from it start with a negative begin pad.
    ///
    /// <para>ONNX derives the pads from <c>output_shape</c> (<c>total = full - output_shape</c>, split by
    /// <c>auto_pad</c>) and requires pads to be non-negative. Its own <c>output_shape</c> example goes one
    /// past the full extent, a negative <i>end</i> pad that reads as zero-extension at the end, and ONNX
    /// Runtime computes that as the specification's example does; that much is kept. A negative
    /// <i>begin</i> pad has no such reading, and ONNX Runtime returns scrambled values for it (NaN among
    /// them), so the graph is refused here instead of running.</para>
    ///
    /// <para>The check needs the input's spatial shape, which a graph does not carry, so it resolves the
    /// shapes the way <see cref="FastLowerAttributeTensorOps"/> resolves geometry: with the
    /// <see cref="QuickExecutionEngine"/>, on the sample inputs the model is concretized against. A node
    /// whose shapes cannot be resolved is left alone.</para>
    /// </summary>
    internal static class FastRejectOversizedConvTransposeOutputShape
    {
        public static void Process(InternalComputationGraph graph, ModelParamList? sampleInputs)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var nodes = graph.Nodes
                .Where(n => n.OpCode == OpCodes.CONV_TRANSPOSE
                            && n.Attributes.GetIntsVal(OnnxOpAttributeNames.AttrOutputShape) is { Length: > 0 }
                            && n.Inputs.Count >= 2 && n.Inputs[0] is not null && n.Inputs[1] is not null)
                .ToList();
            if (nodes.Count == 0) return;

            var store = ResolveShapes(graph, nodes, sampleInputs);
            if (store is null) return;

            foreach (var node in nodes)
                Check(node, Dims(store, node.Inputs[0]!.Value), Dims(store, node.Inputs[1]!.Value));
        }

        private static Dictionary<FastTensorKey, IRuntimeTensor>? ResolveShapes(
            InternalComputationGraph graph, List<FastNode> nodes, ModelParamList? sampleInputs)
        {
            var keys = nodes.SelectMany(n => (FastTensorKey[])[n.Inputs[0]!.Value, n.Inputs[1]!.Value]).Distinct().ToList();
            var resolver = graph.Clone();
            resolver.SetOutputs(keys);
            FastProcessorHelper.RemoveUnreachableNodes(resolver);

            IData[]? samples = null;
            if (resolver.Inputs.Count > 0)
            {
                samples = sampleInputs is null ? null : FastLowerAttributeTensorOps.BindSamplesToTheInputsItReads(resolver, sampleInputs);
                if (samples is null) return null;
            }

            try
            {
                return samples is null
                    ? new QuickExecutionEngine().Run(resolver)
                    : new QuickExecutionEngine().Run(resolver, samples);
            }
            catch
            {
                return null;
            }
        }

        private static long[]? Dims(Dictionary<FastTensorKey, IRuntimeTensor> store, FastTensorKey key)
            => store.TryGetValue(key, out var rt) && rt is RuntimeTensor { Shape: { } shape } ? shape.Dims : null;

        private static void Check(FastNode node, long[]? xDims, long[]? wDims)
        {
            if (xDims is null || xDims.Length < 3) return;
            var rank = xDims.Length - 2;
            var attrs = node.Attributes;
            var outputShape = attrs.GetIntsVal(OnnxOpAttributeNames.AttrOutputShape)!;
            if (outputShape.Length < rank) return;
            var kernelShape = attrs.GetIntsVal(OnnxOpAttributeNames.AttrKernelShape);
            if (kernelShape is not { Length: > 0 } && (wDims is null || wDims.Length != xDims.Length)) return;
            var strides = attrs.GetIntsVal(OnnxOpAttributeNames.AttrStrides);
            var dilations = attrs.GetIntsVal(OnnxOpAttributeNames.AttrDilations);
            var outputPadding = attrs.GetIntsVal(OnnxOpAttributeNames.AttrOutputPadding);
            var autoPad = attrs.GetAttributeObj(OnnxOpAttributeNames.AttrAutoPad) as AutoPad? ?? AutoPad.NotSet;

            var requested = new long[rank];
            var full = new long[rank];
            var begin = new long[rank];
            for (int d = 0; d < rank; d++)
            {
                long At(int[]? values, long fallback) => values is not null && d < values.Length ? values[d] : fallback;
                var kernel = kernelShape is { Length: > 0 } ? At(kernelShape, 1) : wDims![d + 2];
                full[d] = At(strides, 1) * (xDims[d + 2] - 1) + At(outputPadding, 0) + (kernel - 1) * At(dilations, 1) + 1;
                requested[d] = outputShape[outputShape.Length - rank + d];
                var total = full[d] - requested[d];
                var half = (long)Math.Floor(total / 2.0);
                begin[d] = autoPad == AutoPad.SameUpper ? half : total - half;
            }

            if (begin.All(b => b >= 0)) return;

            static string List(long[] v) => $"[{string.Join(", ", v)}]";
            throw new OnnxNodeException(ErrorCodes.FW054, "ConvTranspose",
                string.IsNullOrEmpty(node.FriendlyName) ? node.Key.ToString() : node.FriendlyName,
                $"output_shape {List(requested)} exceeds the full extent {List(full)} " +
                $"(stride * (input - 1) + output_padding + (kernel - 1) * dilation + 1, for input spatial shape " +
                $"{List(xDims[2..])}) so far that the pads ONNX derives from it begin at {List(begin)}. ONNX pads " +
                "must not be negative: output_shape may reach past the full extent only as far as the end padding " +
                "takes it, one element (none under auto_pad SAME_UPPER). Lower output_shape, " +
                "or raise output_padding to reach the size you want.");
        }
    }
}
