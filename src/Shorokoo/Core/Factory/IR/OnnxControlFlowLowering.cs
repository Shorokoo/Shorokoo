using System;
using System.Collections.Generic;
using System.Linq;

namespace Shorokoo.Core.Factory.IR
{
    /// <summary>
    /// Import-time lowering of ONNX control-flow operators that Shorokoo does not
    /// execute natively, run by <see cref="OnnxModelReader"/> on the raw
    /// <see cref="ModelProto"/> before any FastNode is materialized.
    ///
    /// <para>
    /// <c>Scan</c> (opset 9+) is semantically a <c>Loop</c> with automatic
    /// per-iteration slicing of its scan inputs and stacking of its scan outputs.
    /// <see cref="LowerScanToLoop"/> rewrites each incoming Scan node into the
    /// equivalent Loop:
    /// <list type="bullet">
    ///   <item>trip count = the first scan input's length along its scan axis
    ///         (<c>Shape</c> → <c>Gather</c>);</item>
    ///   <item>the body is wrapped so each scan input is sliced with
    ///         <c>Gather(X, iter, axis=scan_input_axes[m])</c> — for a reverse
    ///         direction the index is <c>len-1-iter</c>;</item>
    ///   <item>the body's scan-output slices become Loop scan outputs, which Loop
    ///         stacks along axis 0 in iteration order — exactly Scan's semantics
    ///         for the default <c>scan_output_axes = 0</c> /
    ///         <c>scan_output_directions = forward</c>.</item>
    /// </list>
    /// Supported envelope: any <c>scan_input_axes</c> and any
    /// <c>scan_input_directions</c>; <c>scan_output_axes</c> and
    /// <c>scan_output_directions</c> must be 0/forward (the overwhelmingly common
    /// case). Outside the envelope a <see cref="NotSupportedException"/> names the
    /// offending attribute.
    /// </para>
    ///
    /// <para>
    /// <c>SequenceMap</c> is rejected with an actionable error: lowering it would
    /// require whole-graph type inference (its variadic additional inputs are
    /// implicitly mapped element-wise when sequence-typed and broadcast when
    /// tensor-typed — indistinguishable at the proto level — and the per-output
    /// accumulator sequences need a typed <c>SequenceEmpty</c> seed), and the
    /// ONNX Runtime execution backend has no SequenceMap kernel either. See
    /// Documentation/limitations.md.
    /// </para>
    /// </summary>
    internal static class OnnxControlFlowLowering
    {
        private const int Int64ElemType = 7; // TensorProto.DataType.INT64
        private const int BoolElemType = 9;  // TensorProto.DataType.BOOL

        /// <summary>
        /// Rewrites the model in place: every Scan node (top-level graph, nested
        /// subgraphs, and function bodies) is replaced by its Loop lowering;
        /// every SequenceMap node raises a <see cref="NotSupportedException"/>.
        /// </summary>
        public static void Process(ModelProto model)
        {
            if (model is null) throw new ArgumentNullException(nameof(model));

            long opset = 0;
            if (model.OpsetImports is { Count: > 0 })
            {
                var defaultDomain = model.OpsetImports.FirstOrDefault(x => IsDefaultDomain(x.Domain));
                opset = (defaultDomain ?? model.OpsetImports[0]).Version;
            }

            int uid = 0;
            if (model.Graph is not null)
                LowerNodeList(model.Graph.Nodes, opset, ref uid);
            if (model.Functions is not null)
                foreach (var fn in model.Functions)
                    LowerNodeList(fn.Nodes, opset, ref uid);
        }

        private static bool IsDefaultDomain(string? domain)
            => string.IsNullOrEmpty(domain) || domain == "ai.onnx";

        // Deliberately NOT on FastOnnxModelBuilder.ForEachGraphRecursive (the shared
        // GraphProto walker): this pass needs child-first ordering (a nested Scan must be
        // lowered before its enclosing node is rewritten), operates on bare node lists so
        // it can also rewrite FunctionProto bodies, and threads a ref counter — none of
        // which the parent-first Action<GraphProto> walker can express. It already covers
        // both attr.G and attr.Graphs.
        private static void LowerNodeList(List<NodeProto> nodes, long opset, ref int uid)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];

                // Recurse into subgraph attributes first so nested Scan nodes
                // (including Scans inside another Scan's body) are lowered before
                // the enclosing node is rewritten.
                foreach (var attr in node.Attributes)
                {
                    if (attr.Type == AttributeProto.AttributeType.Graph && attr.G is not null)
                        LowerNodeList(attr.G.Nodes, opset, ref uid);
                    else if (attr.Type == AttributeProto.AttributeType.Graphs)
                        foreach (var g in attr.Graphs)
                            LowerNodeList(g.Nodes, opset, ref uid);
                }

                if (!IsDefaultDomain(node.Domain))
                    continue;

                if (node.OpType == "SequenceMap")
                {
                    throw new NotSupportedException(
                        $"ONNX import: the 'SequenceMap' operator (node '{node.Name}') is not supported. " +
                        "Lowering SequenceMap to a Loop requires whole-graph type inference: its additional " +
                        "inputs are mapped per-element when sequence-typed but broadcast when tensor-typed " +
                        "(indistinguishable at the proto level), and the per-output accumulator sequences " +
                        "need a typed SequenceEmpty seed. The ONNX Runtime execution backend has no " +
                        "SequenceMap kernel either. Workaround: rewrite the model as an explicit Loop over " +
                        "SequenceLength using SequenceAt/SequenceInsert (in Shorokoo, build it with LoopAPI) " +
                        "— that form is fully supported. See Documentation/limitations.md.");
                }

                if (node.OpType == "Scan")
                {
                    var replacement = LowerScanToLoop(node, opset, uid);
                    uid++;
                    nodes.RemoveAt(i);
                    nodes.InsertRange(i, replacement);
                    i += replacement.Count - 1;
                }
            }
        }

        /// <summary>
        /// Builds the Loop-based replacement node list for one Scan node:
        /// <c>[Shape, Constant(axis), Gather(trip), Constant(true), Loop]</c>.
        /// </summary>
        private static List<NodeProto> LowerScanToLoop(NodeProto scan, long opset, int uid)
        {
            string Where() => $"ONNX import: Scan node '{scan.Name}'";

            if (opset < 9)
                throw new NotSupportedException(
                    $"{Where()}: Scan at opset {opset} (the opset-8 form with an implicit batch " +
                    "dimension and a sequence_lens input) is not supported. Re-export the model at " +
                    "opset 9 or later, where Scan has the modern batchless form.");

            var bodyAttr = scan.Attributes.FirstOrDefault(
                x => x.Name == "body" && x.Type == AttributeProto.AttributeType.Graph);
            if (bodyAttr?.G is null)
                throw new InvalidOperationException($"{Where()}: required graph attribute 'body' is missing.");
            var body = bodyAttr.G;

            var numScanAttr = scan.Attributes.FirstOrDefault(
                x => x.Name == "num_scan_inputs" && x.Type == AttributeProto.AttributeType.Int);
            if (numScanAttr is null)
                throw new InvalidOperationException($"{Where()}: required attribute 'num_scan_inputs' is missing.");

            int m = checked((int)numScanAttr.I);
            int n = scan.Inputs.Count - m;
            if (m < 1 || n < 0)
                throw new InvalidOperationException(
                    $"{Where()}: num_scan_inputs={m} is inconsistent with {scan.Inputs.Count} node input(s).");
            if (body.Inputs.Count != n + m)
                throw new InvalidOperationException(
                    $"{Where()}: body declares {body.Inputs.Count} input(s) but N+M = {n + m} " +
                    $"(N={n} state variables, M={m} scan inputs).");
            int k = body.Outputs.Count - n;
            if (k < 0 || scan.Outputs.Count != n + k)
                throw new InvalidOperationException(
                    $"{Where()}: body declares {body.Outputs.Count} output(s) and the node {scan.Outputs.Count}; " +
                    $"expected N+K body outputs and N+K node outputs with N={n}.");

            var inputAxes = GetIntsAttr(scan, "scan_input_axes", m, Where);
            var inputDirections = GetIntsAttr(scan, "scan_input_directions", m, Where);
            var outputAxes = GetIntsAttr(scan, "scan_output_axes", k, Where);
            var outputDirections = GetIntsAttr(scan, "scan_output_directions", k, Where);

            // Loop stacks its scan outputs along axis 0 in iteration order; anything
            // else would need rank-aware Transpose / ReverseSequence rewrites that
            // are not expressible without shape inference at import time.
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

            // Unique-name prefix for everything the lowering synthesizes.
            var baseName = string.IsNullOrEmpty(scan.Name) ? "scan" : scan.Name;
            string P(string suffix) => $"{baseName}__shrk_scan2loop_{uid}__{suffix}";

            var result = new List<NodeProto>();

            // ---- Outer prelude: trip count = shape(scan_input_0)[scan_input_axes[0]].
            var firstScanInput = scan.Inputs[n];
            result.Add(MakeNode("Shape", P("shape0"),
                inputs: new[] { firstScanInput },
                outputs: new[] { P("shape0_out") }));
            result.Add(MakeConstant(P("axis0"), P("axis0_out"), Int64Scalar(inputAxes[0])));
            // Gather with a scalar (possibly negative) index along axis 0 of the
            // 1-D shape vector yields the scalar int64 trip count Loop expects.
            result.Add(MakeNode("Gather", P("trip"),
                inputs: new[] { P("shape0_out"), P("axis0_out") },
                outputs: new[] { P("trip_out") },
                IntAttr("axis", 0)));
            result.Add(MakeConstant(P("cond"), P("cond_out"), BoolScalar(true)));

            // ---- Loop body: (iter, cond, states…) → (cond, states…, scan slices…).
            var loopBody = new GraphProto { Name = P("body") };
            loopBody.Inputs.Add(ScalarInfo(P("iter"), Int64ElemType));
            loopBody.Inputs.Add(ScalarInfo(P("cond_in"), BoolElemType));
            for (int s = 0; s < n; s++)
                loopBody.Inputs.Add(body.Inputs[s]);

            // Loop bodies must emit a continue condition; pass the incoming one through.
            loopBody.Nodes.Add(MakeNode("Identity", P("cond_identity"),
                inputs: new[] { P("cond_in") },
                outputs: new[] { P("cond_body_out") }));

            // Per-iteration slices: Gather(X_m, idx, axis=scan_input_axes[m]) with a
            // scalar index removes the scan axis — exactly Scan's slice semantics.
            // Forward direction reads index = iter; reverse reads len-1-iter.
            for (int j = 0; j < m; j++)
            {
                var scanInput = scan.Inputs[n + j];        // outer-scope capture
                var sliceName = body.Inputs[n + j].Name;   // the name the body consumes
                string indexName;
                if (inputDirections[j] == 0)
                {
                    indexName = P("iter");
                }
                else
                {
                    loopBody.Nodes.Add(MakeNode("Shape", P($"shape_{j}"),
                        inputs: new[] { scanInput },
                        outputs: new[] { P($"shape_{j}_out") }));
                    loopBody.Nodes.Add(MakeConstant(P($"axis_{j}"), P($"axis_{j}_out"), Int64Scalar(inputAxes[j])));
                    loopBody.Nodes.Add(MakeNode("Gather", P($"len_{j}"),
                        inputs: new[] { P($"shape_{j}_out"), P($"axis_{j}_out") },
                        outputs: new[] { P($"len_{j}_out") },
                        IntAttr("axis", 0)));
                    loopBody.Nodes.Add(MakeConstant(P($"one_{j}"), P($"one_{j}_out"), Int64Scalar(1)));
                    loopBody.Nodes.Add(MakeNode("Sub", P($"last_{j}"),
                        inputs: new[] { P($"len_{j}_out"), P($"one_{j}_out") },
                        outputs: new[] { P($"last_{j}_out") }));
                    loopBody.Nodes.Add(MakeNode("Sub", P($"revidx_{j}"),
                        inputs: new[] { P($"last_{j}_out"), P("iter") },
                        outputs: new[] { P($"revidx_{j}_out") }));
                    indexName = P($"revidx_{j}_out");
                }
                loopBody.Nodes.Add(MakeNode("Gather", P($"slice_{j}"),
                    inputs: new[] { scanInput, indexName },
                    outputs: new[] { sliceName },
                    IntAttr("axis", inputAxes[j])));
            }

            // The original body nodes run on the states + freshly-gathered slices.
            loopBody.Nodes.AddRange(body.Nodes);
            loopBody.Initializers.AddRange(body.Initializers);
            loopBody.ValueInfoes.AddRange(body.ValueInfoes);

            loopBody.Outputs.Add(ScalarInfo(P("cond_body_out"), BoolElemType));
            foreach (var bodyOutput in body.Outputs)
                loopBody.Outputs.Add(bodyOutput);

            // ---- The Loop node itself: same outputs as the Scan node it replaces
            // (N final states followed by K stacked scan outputs).
            var loopNode = new NodeProto { OpType = "Loop", Name = P("loop") };
            loopNode.Inputs.Add(P("trip_out"));
            loopNode.Inputs.Add(P("cond_out"));
            for (int s = 0; s < n; s++)
                loopNode.Inputs.Add(scan.Inputs[s]);
            loopNode.Outputs.AddRange(scan.Outputs);
            loopNode.Attributes.Add(new AttributeProto
            {
                Name = "body",
                Type = AttributeProto.AttributeType.Graph,
                G = loopBody,
            });
            result.Add(loopNode);

            return result;
        }

        /// <summary>
        /// Reads an optional INTS attribute, defaulting to <paramref name="count"/>
        /// zeros (the spec default for all four Scan axes/directions attributes).
        /// </summary>
        private static long[] GetIntsAttr(NodeProto node, string name, int count, Func<string> where)
        {
            var attr = node.Attributes.FirstOrDefault(
                x => x.Name == name && x.Type == AttributeProto.AttributeType.Ints);
            if (attr?.Ints is null || attr.Ints.Length == 0)
                return new long[count];
            if (attr.Ints.Length != count)
                throw new InvalidOperationException(
                    $"{where()}: attribute '{name}' has {attr.Ints.Length} entries but {count} were expected.");
            return attr.Ints;
        }

        private static NodeProto MakeNode(
            string opType, string name, string[] inputs, string[] outputs, params AttributeProto[] attrs)
        {
            var node = new NodeProto { OpType = opType, Name = name };
            node.Inputs.AddRange(inputs);
            node.Outputs.AddRange(outputs);
            node.Attributes.AddRange(attrs);
            return node;
        }

        private static NodeProto MakeConstant(string name, string outputName, TensorProto value)
            => MakeNode("Constant", name,
                inputs: Array.Empty<string>(),
                outputs: new[] { outputName },
                new AttributeProto { Name = "value", Type = AttributeProto.AttributeType.Tensor, T = value });

        private static AttributeProto IntAttr(string name, long value)
            => new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

        private static TensorProto Int64Scalar(long value)
            => new TensorProto { data_type = Int64ElemType, RawData = BitConverter.GetBytes(value) };

        private static TensorProto BoolScalar(bool value)
            => new TensorProto { data_type = BoolElemType, RawData = new[] { value ? (byte)1 : (byte)0 } };

        private static ValueInfoProto ScalarInfo(string name, int elemType)
            => new ValueInfoProto
            {
                Name = name,
                Type = new TypeProto
                {
                    TensorType = new TypeProto.Tensor
                    {
                        ElemType = elemType,
                        Shape = new TensorShapeProto(),
                    },
                },
            };
    }
}
