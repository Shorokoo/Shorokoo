using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;
using System;
using System.Collections.Generic;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Import-time lowering of the ONNX control-flow operators Shorokoo does not execute
    /// natively. <c>Scan</c> reaches this pass as the <c>SCAN_OPEN</c> … <c>SCAN_CLOSE</c>
    /// band the reader materializes for any subgraph-bearing node, and is rewritten in
    /// place into the equivalent <c>LOOP_OPEN</c> … <c>LOOP_CLOSE</c> band;
    /// <c>SequenceMap</c> is rejected with an actionable error.
    ///
    /// <para>
    /// <c>Scan</c> is a <c>Loop</c> with automatic per-iteration slicing of its scan inputs
    /// and stacking of its scan outputs:
    /// <list type="bullet">
    ///   <item>trip count = the first scan input's length along its scan axis
    ///         (<c>Shape</c> → <c>Gather</c>), emitted ahead of the open node;</item>
    ///   <item>each scan input is sliced inside the body by
    ///         <c>Gather(X, iter, axis=scan_input_axes[m])</c> — index <c>len-1-iter</c>
    ///         for a reverse direction — taking over the slot <c>SCAN_OPEN</c> used to
    ///         hand the body;</item>
    ///   <item>the body's scan-output slices become Loop scan variables, which
    ///         <c>LOOP_CLOSE</c> stacks along axis 0 in iteration order — exactly Scan's
    ///         semantics for the default <c>scan_output_axes = 0</c> /
    ///         <c>scan_output_directions = forward</c>.</item>
    /// </list>
    /// Supported envelope: any <c>scan_input_axes</c> and any
    /// <c>scan_input_directions</c>; <c>scan_output_axes</c> and
    /// <c>scan_output_directions</c> must be 0/forward (the overwhelmingly common case).
    /// Outside the envelope a <see cref="NotSupportedException"/> names the offending
    /// attribute.
    /// </para>
    ///
    /// <para>
    /// The loop's continue condition is a fresh <c>true</c> CONSTANT — the same shape the
    /// authoring path builds (<c>Loop.BuildLoopCloseNode</c> passes
    /// <c>continueWhileTensor ?? Scalar(true)</c>) and the same shape the exporter writes.
    /// It is deliberately NOT <c>LOOP_OPEN</c>'s condition passthrough: that output exists
    /// only to satisfy the ONNX body signature — the definition names it
    /// <c>VestigalTrue</c> and leaves it unnamed — and nothing may consume it. Feeding it
    /// back in as the close node's <c>break</c> input would let a value the engine
    /// fabricates decide when the loop stops.
    /// </para>
    /// </summary>
    internal static class FastLowerScanToLoop
    {
        public static void Process(InternalComputationGraph graph, long opset)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));
            ProcessGraph(graph, opset, new HashSet<Function>());
        }

        private static void ProcessGraph(
            InternalComputationGraph graph, long opset, HashSet<Function> seenFunctions)
        {
            foreach (var node in graph.Nodes)
                if (node.TargetFunction is { } fn && seenFunctions.Add(fn))
                    ProcessGraph(fn.OriginalFastGraph, opset, seenFunctions);

            RejectSequenceMap(graph);

            // Last-first: a nested Scan's open node sits after its enclosing one, so
            // taking the last remaining open each round lowers inner bands before the
            // bands that contain them, and the node inserts never land inside a band
            // still waiting to be rewritten.
            while (true)
            {
                int openIdx = graph.Nodes.FindLastIndex(n => n.OpCode == OpCodes.SCAN_OPEN);
                if (openIdx < 0) return;
                LowerScanBand(graph, openIdx, opset);
            }
        }

        private static void RejectSequenceMap(InternalComputationGraph graph)
        {
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != OpCodes.SEQUENCE_MAP_OPEN) continue;
                throw new NotSupportedException(
                    $"ONNX import: the 'SequenceMap' operator (node '{node.FriendlyName}') is not supported. " +
                    "Lowering SequenceMap to a Loop requires whole-graph type inference: its additional " +
                    "inputs are mapped per-element when sequence-typed but broadcast when tensor-typed " +
                    "(indistinguishable without inferring the element types), and the per-output " +
                    "accumulator sequences need a typed SequenceEmpty seed. The ONNX Runtime execution " +
                    "backend has no SequenceMap kernel either. Workaround: rewrite the model as an " +
                    "explicit Loop over SequenceLength using SequenceAt/SequenceInsert (in Shorokoo, " +
                    "build it with LoopAPI) — that form is fully supported. " +
                    "See Documentation/limitations.md.");
            }
        }

        private static void LowerScanBand(InternalComputationGraph graph, int openIdx, long opset)
        {
            var open = graph.Nodes[openIdx];
            string Where() => $"ONNX import: Scan node '{open.FriendlyName}'";

            if (opset < 9)
                throw new NotSupportedException(
                    $"{Where()}: Scan at opset {opset} (the opset-8 form with an implicit batch " +
                    "dimension and a sequence_lens input) is not supported. Re-export the model at " +
                    "opset 9 or later, where Scan has the modern batchless form.");

            int closeIdx = graph.Nodes.FindIndex(openIdx + 1, n =>
                n.OpCode == OpCodes.SCAN_CLOSE
                && n.GraphOpenNodeKey is FastNodeKey k && k.Equals(open.Key));
            if (closeIdx < 0)
                throw new InvalidOperationException($"{Where()}: no matching Scan close node.");
            var close = graph.Nodes[closeIdx];

            var scanNodeInputs = open.FullInputs[""];              // [states…, scanInputs…]
            var bodyEntries = open.FullOutputs[AttrBody];          // [bodyStates…, bodySlices…]
            var bodyExits = close.FullInputs[AttrBody];            // [stateOuts…, scanOutSlices…]

            int m = checked((int)(open.Attributes.GetLongVal(AttrNumScanInputs)
                ?? throw new InvalidOperationException(
                    $"{Where()}: required attribute 'num_scan_inputs' is missing.")));
            int n = scanNodeInputs.Count - m;
            if (m < 1 || n < 0)
                throw new InvalidOperationException(
                    $"{Where()}: num_scan_inputs={m} is inconsistent with {scanNodeInputs.Count} node input(s).");
            if (bodyEntries.Count != n + m)
                throw new InvalidOperationException(
                    $"{Where()}: body declares {bodyEntries.Count} input(s) but N+M = {n + m} " +
                    $"(N={n} state variables, M={m} scan inputs).");
            int k = bodyExits.Count - n;
            if (k < 0 || close.FullOutputs[""].Count != n + k)
                throw new InvalidOperationException(
                    $"{Where()}: body declares {bodyExits.Count} output(s) and the node " +
                    $"{close.FullOutputs[""].Count}; expected N+K body outputs and N+K node " +
                    $"outputs with N={n}.");

            var inputAxes = Axes(open, AttrScanInputAxes, m, Where);
            var inputDirections = Axes(open, AttrScanInputDirections, m, Where);
            var outputAxes = Axes(open, AttrScanOutputAxes, k, Where);
            var outputDirections = Axes(open, AttrScanOutputDirections, k, Where);

            // Loop stacks its scan outputs along axis 0 in iteration order; anything else
            // would need rank-aware Transpose / ReverseSequence rewrites.
            if (outputAxes.Any(a => a != 0))
                throw new NotSupportedException(
                    $"{Where()}: non-zero scan_output_axes [{string.Join(", ", outputAxes)}] are not " +
                    "supported by the Scan→Loop import lowering (Loop always stacks scan outputs along " +
                    "axis 0). Set scan_output_axes to 0 and transpose the scan output downstream instead.");
            if (outputDirections.Any(d => d != 0))
                throw new NotSupportedException(
                    $"{Where()}: reverse scan_output_directions [{string.Join(", ", outputDirections)}] are " +
                    "not supported by the Scan→Loop import lowering (Loop always stacks scan outputs in " +
                    "iteration order). Set scan_output_directions to 0 (forward) and reverse the scan " +
                    "output downstream instead.");

            var states = scanNodeInputs.Take(n).ToList();
            var scanInputs = scanNodeInputs.Skip(n).ToList();

            // ---- Prelude, ahead of the open node: trip count and the continue condition.
            var prelude = new List<FastNode>();
            var tripCount = EmitAxisLength(scanInputs[0], inputAxes[0], prelude);
            var condTrue = EmitConstant(TensorDataScalarBool(true), prelude);

            // ---- The open node becomes LOOP_OPEN, keeping its key so the body-state slots
            // it already owns stay put. Loop prepends two outputs the Scan band has no
            // counterpart for, so they take fresh slots after the ones already allocated.
            int nextSlot = bodyEntries.Count;
            var iterIndex = new FastTensorKey(open.Key, nextSlot++);
            var vestigalTrue = new FastTensorKey(open.Key, nextSlot++);

            var newBodyEntries = new List<FastTensorKey?> { iterIndex, vestigalTrue };
            newBodyEntries.AddRange(bodyEntries.Take(n));

            open.OpCode = OpCodes.LOOP_OPEN;
            open.Attributes = EmptyAttributes(OpCodes.LOOP_OPEN);
            // Loop takes no initial condition here: the trip count alone bounds a Scan,
            // which is what the authoring path emits too (LoopOpen(condition: null)).
            open.FullInputs[""] = [tripCount, null, .. states];
            open.FullOutputs[AttrBody] = newBodyEntries;

            // ---- In-body slicing takes over the slots SCAN_OPEN used to fill.
            var bodyPrelude = new List<FastNode>();
            var sliceRemap = new Dictionary<FastTensorKey, FastTensorKey>();
            for (int j = 0; j < m; j++)
            {
                var index = inputDirections[j] == 0
                    ? iterIndex
                    : EmitReverseIndex(scanInputs[j], inputAxes[j], iterIndex, bodyPrelude);
                var slice = EmitGather(scanInputs[j], index, inputAxes[j], bodyPrelude);
                if (bodyEntries[n + j] is FastTensorKey oldSlot) sliceRemap[oldSlot] = slice;
            }

            // ---- The close node becomes LOOP_CLOSE. Its outputs are untouched; the only
            // input change is the break condition prepended ahead of the body's exits.
            close.OpCode = OpCodes.LOOP_CLOSE;
            close.Attributes = EmptyAttributes(OpCodes.LOOP_CLOSE);
            close.FullInputs[AttrBody] = [condTrue, .. bodyExits];

            graph.Nodes.InsertRange(openIdx + 1, bodyPrelude);
            graph.Nodes.InsertRange(openIdx, prelude);

            RemapTensorKeys(graph, sliceRemap);
        }

        /// <summary>
        /// Repoints every consumer of a rewritten scan-input slot at the in-body Gather that
        /// now produces it. A tensor key names the node that owns it, so the slices cannot
        /// simply keep their old keys once the open node stops producing them.
        /// </summary>
        private static void RemapTensorKeys(
            InternalComputationGraph graph, Dictionary<FastTensorKey, FastTensorKey> remap)
        {
            if (remap.Count == 0) return;
            foreach (var node in graph.Nodes)
                foreach (var slots in node.FullInputs.Values)
                    for (int i = 0; i < slots.Count; i++)
                        if (slots[i] is FastTensorKey key && remap.TryGetValue(key, out var replacement))
                            slots[i] = replacement;
        }

        /// <summary>Length of <paramref name="tensor"/> along <paramref name="axis"/>, as a scalar int64.</summary>
        private static FastTensorKey EmitAxisLength(
            FastTensorKey? tensor, long axis, List<FastNode> into)
        {
            var shape = EmitNode(OpCodes.SHAPE, new Dictionary<string, object?>(), [tensor], into);
            var axisConst = EmitConstant(TensorDataScalarLong(axis), into);
            // Gather with a scalar (possibly negative) index along axis 0 of the 1-D shape
            // vector yields the scalar int64 trip count Loop expects.
            return EmitNode(OpCodes.GATHER,
                new Dictionary<string, object?> { [AttrAxis] = 0L }, [shape, axisConst], into);
        }

        /// <summary>The reverse-direction read position: <c>len - 1 - iter</c>.</summary>
        private static FastTensorKey EmitReverseIndex(
            FastTensorKey? scanInput, long axis, FastTensorKey iterIndex, List<FastNode> into)
        {
            var length = EmitAxisLength(scanInput, axis, into);
            var one = EmitConstant(TensorDataScalarLong(1L), into);
            var last = EmitNode(OpCodes.SUB, new Dictionary<string, object?>(), [length, one], into);
            return EmitNode(OpCodes.SUB, new Dictionary<string, object?>(), [last, iterIndex], into);
        }

        private static FastTensorKey EmitGather(
            FastTensorKey? data, FastTensorKey index, long axis, List<FastNode> into)
            => EmitNode(OpCodes.GATHER,
                new Dictionary<string, object?> { [AttrAxis] = axis }, [data, index], into);

        private static FastTensorKey EmitConstant(TensorData value, List<FastNode> into)
            => EmitNode(OpCodes.CONSTANT,
                new Dictionary<string, object?> { [AttrValue] = value }, [], into);

        private static FastTensorKey EmitNode(
            string opCode, Dictionary<string, object?> attrs, FastTensorKey?[] inputs, List<FastNode> into)
        {
            var nodeKey = FastNodeKey.New();
            into.Add(FastNodeCreationHelpers.CreateFastNode(nodeKey, opCode, attrs, inputs));
            return new FastTensorKey(nodeKey, 0);
        }

        private static OnnxCSharpAttributes EmptyAttributes(string opCode)
            => OnnxCSharpAttributes.FromCSharpVals(
                new Dictionary<string, object?>(), Definitions.NodeDefinitions[opCode].AttributeDefs);

        private static TensorData TensorDataScalarLong(long value)
        {
            long[] vals = [value];
            return Shorokoo.Globals.TensorData(Array.Empty<long>(), vals);
        }

        private static TensorData TensorDataScalarBool(bool value)
        {
            bool[] vals = [value];
            return Shorokoo.Globals.TensorData(Array.Empty<long>(), vals);
        }

        /// <summary>
        /// An omitted or empty axes/directions attribute means "all zero", per the Scan spec.
        /// </summary>
        private static long[] Axes(FastNode node, string name, int count, Func<string> where)
        {
            var vals = node.Attributes.GetLongsVal(name);
            if (vals is null || vals.Length == 0) return new long[count];
            if (vals.Length != count)
                throw new InvalidOperationException(
                    $"{where()}: attribute '{name}' has {vals.Length} entries but {count} were expected.");
            return vals;
        }
    }
}
