using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Graph;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// The shape every output of a <see cref="GraphKind.ConcreteArchitecture"/> or
    /// <see cref="GraphKind.ConcreteModel"/> graph records — the outputs' counterpart of
    /// <see cref="RepresentativeInputShapes"/>: the dims the output has when the graph runs at the
    /// samples it was concretized at, recorded as
    /// <see cref="OnnxOpAttributeNames.ShrkAttrRecordedOutputShape"/> on its <c>GRAPH_OUTPUT</c>
    /// node. It is distinct from the output's declared rank
    /// (<see cref="OnnxOpAttributeNames.ShrkAttrDeclaredRank"/>), which its type fixes: this one a
    /// sample decides, so an output whose rank varies with its inputs records the rank it had at
    /// the samples. ONNX export reads an output's rank off it where the output declares none.
    ///
    /// <para>Established where a graph becomes concrete — <c>ToConcreteArchitecture</c> evaluates
    /// the outputs at the samples it was handed, a composed graph at the representative inputs it
    /// is built at, an ONNX import from the file — and verified wherever a concrete graph is frozen
    /// (<see cref="Verify"/>). The attribute rides on the node, so it survives every copy, the
    /// <c>.srk</c> round trip, <c>ToConcreteModel</c> and <c>Specialize</c>.</para>
    ///
    /// <para>Every output records one, whatever it holds: a tensor its shape; an optional its
    /// element's where present, and <see cref="RepresentativeInputShapes.AbsentOptionalShape"/>
    /// where absent; a sequence the shape its elements share, and
    /// <see cref="NoSharedElementShape"/> where they share none. So a missing attribute always
    /// means a graph that breaks the invariant.</para>
    /// </summary>
    internal static class RecordedOutputShapes
    {
        /// <summary>
        /// The single negative dim recorded for a sequence output whose elements share no shape (or
        /// that is empty). Distinct from <see cref="RepresentativeInputShapes.AbsentOptionalShape"/>,
        /// and, like it, never a concrete shape.
        /// </summary>
        internal static readonly long[] NoSharedElementShape = [-2L];

        /// <summary>Records <paramref name="dims"/> on the output node <paramref name="node"/>.</summary>
        internal static void Set(FastNode node, long[] dims)
            => node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrRecordedOutputShape, (object?)dims));

        /// <summary>The recorded shape of the output node <paramref name="node"/>, or <c>null</c>.</summary>
        internal static long[]? Get(FastNode node)
            => node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRecordedOutputShape)
                ? node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRecordedOutputShape)
                : null;

        /// <summary>
        /// The rank <paramref name="node"/>'s recorded shape gives the output, or <c>null</c> when it
        /// records none or records a marker (<see cref="RepresentativeInputShapes.AbsentOptionalShape"/>,
        /// <see cref="NoSharedElementShape"/>) rather than a shape.
        /// </summary>
        internal static int? RankOf(FastNode node)
            => Get(node) is { } dims && !IsMarker(dims) ? dims.Length : null;

        private static bool IsMarker(long[] dims) => dims is [< 0];

        /// <summary>
        /// Records the shape of each output of <paramref name="graph"/> — a graph just lowered by
        /// <c>ToConcreteArchitecture</c> — at <paramref name="samples"/>, the real sample values it
        /// was lowered at, bound to its inputs as the lowering binds them
        /// (<see cref="RepresentativeInputShapes.BindSamplesToLoweredInputs"/>).
        ///
        /// <para>Evaluated by the <see cref="QuickExecutionEngine"/>, shapes first: every value up to
        /// <see cref="ShapeInferenceInterpreter.MaxSmallTensorElements"/> elements is computed, which
        /// covers every small value a shape can hang on (a flag, an axes list, a target shape), and
        /// anything larger is carried as its shape and dtype alone. Only where that leaves an output
        /// unresolved — a shape hanging on the values of a large tensor — is the walk repeated with
        /// the samples' values in full. The parameters are not initialized at this stage, so each
        /// stands in by the shape and dtype its node declares.</para>
        /// </summary>
        internal static void RecordAtSamples(InternalComputationGraph graph, ModelParamList samples)
        {
            var unresolved = Evaluate(graph, samples, ShapeInferenceInterpreter.MaxSmallTensorElements);
            if (unresolved) Evaluate(graph, samples, int.MaxValue);
        }

        private static bool Evaluate(InternalComputationGraph graph, ModelParamList samples, int maxDataElements)
        {
            var engine = new QuickExecutionEngine { MaxDataElements = maxDataElements };
            var initial = FastProcessorHelper.SampleRuntimeInputs(graph, samples, engine) ?? [];
            AddParameterStandIns(graph, initial);
            return !RecordFrom(graph, engine.Run(graph, initial));
        }

        /// <summary>
        /// Stands each uninitialized parameter (<c>MODEL_PARAM</c>) of <paramref name="graph"/> in by
        /// the shape and dtype its node declares (<see cref="TrainingRig.RepresentativeRuntimeInputFor"/>),
        /// so the engine carries its shape rather than an unknown. One declaring no concrete shape is
        /// left unknown.
        /// </summary>
        private static void AddParameterStandIns(InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> initial)
        {
            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM
                    || node.Outputs.FirstOrDefault(k => k is not null) is not { } key
                    || node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrShape) is not { } dims
                    || dims.Any(d => d < 0)
                    || node.Attributes.GetDTypeVal(OnnxOpAttributeNames.ShrkAttrDtype) is not { } dtype)
                    continue;
                initial[key] = TrainingRig.RepresentativeRuntimeInputFor(new Shape(dims), dtype);
            }
        }

        /// <summary>
        /// Records the shape of each output of <paramref name="graph"/> at its own representative
        /// inputs (<see cref="TrainingRig.ReadRepresentativeInputs"/>): a graph composed from
        /// concrete graphs, which has no samples of its own but is built at the shapes its inputs
        /// record. Where the engine leaves an output unresolved it is left without a shape.
        /// </summary>
        internal static void RecordAtRepresentativeInputs(InternalComputationGraph graph)
        {
            var shapes = ShapesAtRepresentativeInputs(graph);
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count; i++)
                if (shapes[i] is { } dims) Set(outputNodes[i], dims);
        }

        /// <summary>Each output's shape at <paramref name="graph"/>'s representative inputs, in
        /// output order, <c>null</c> where the engine leaves it unknown.</summary>
        private static long[]?[] ShapesAtRepresentativeInputs(InternalComputationGraph graph)
        {
            var engine = new QuickExecutionEngine { MaxDataElements = ShapeInferenceInterpreter.MaxSmallTensorElements };
            var inputs = TrainingRig.ReadRepresentativeInputs(graph, representSequences: true);
            var initial = new Dictionary<FastTensorKey, IRuntimeTensor>();
            var keys = graph.Inputs;
            for (int i = 0; i < keys.Count; i++) initial[keys[i]] = inputs[i];
            AddParameterStandIns(graph, initial);
            return ShapesFrom(graph, engine.Run(graph, initial));
        }

        /// <summary>
        /// Records each output's shape from <paramref name="shapes"/> — a shape inference already
        /// run over <paramref name="graph"/> at the inputs it records — where it gives one.
        /// </summary>
        internal static void Record(InternalComputationGraph graph, ShapeInferenceResult shapes)
        {
            foreach (var node in graph.OutputNodes)
                if (shapes.GetTensorInfo(InternalComputationGraph.OutputKeyOf(node)) is { Shape: { } shape }
                    && shape.Dims.All(d => d >= 0))
                    Set(node, shape.Dims);
        }

        /// <summary>
        /// Records each output's shape from the engine's <paramref name="store"/>; whether every
        /// output got one.
        /// </summary>
        private static bool RecordFrom(InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> store)
        {
            var shapes = ShapesFrom(graph, store);
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count; i++)
                if (shapes[i] is { } dims) Set(outputNodes[i], dims);
            return shapes.All(dims => dims is not null);
        }

        private static long[]?[] ShapesFrom(InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> store)
            => [.. graph.Outputs.Select(key => store.TryGetValue(key, out var value) ? ShapeOf(value) : null)];

        /// <summary>The dims to record for one evaluated value, or <c>null</c> where the evaluation
        /// left its shape unknown.</summary>
        private static long[]? ShapeOf(IRuntimeTensor value) => value switch
        {
            RuntimeTensor tensor => Definite(tensor),
            RuntimeOptionalTensor { HasValue: false } => RepresentativeInputShapes.AbsentOptionalShape,
            RuntimeOptionalTensor { HasValue: true, ValueTensor: { } present } => Definite(present),
            RuntimeSequenceTensor sequence => SharedElementShapeOf(sequence),
            _ => null,
        };

        private static long[]? Definite(RuntimeTensor tensor)
            => tensor.DType != DType.Invalid && tensor.HasDefiniteShape ? tensor.Shape!.Dims : null;

        private static long[]? SharedElementShapeOf(RuntimeSequenceTensor sequence)
        {
            if (sequence.Tensors is { Length: > 0 } elements)
            {
                var first = Definite(elements[0]);
                return first is not null && elements.All(e => Definite(e) is { } dims && dims.AsSpan().SequenceEqual(first))
                    ? first
                    : NoSharedElementShape;
            }
            if (sequence.TemplateTensor is { } template && sequence.Count is > 0)
                return Definite(template) ?? NoSharedElementShape;
            return sequence.Count is 0 || sequence.Tensors is { Length: 0 } ? NoSharedElementShape : null;
        }

        /// <summary>
        /// Holds a concrete graph to the invariant: every output records a shape. Throws
        /// <see cref="ErrorCodes.FW057"/> naming the first output that does not; a
        /// <see cref="GraphKind.Module"/> graph is not checked.
        /// </summary>
        internal static void Verify(InternalComputationGraph graph, GraphKind kind)
        {
            if (kind is not (GraphKind.ConcreteArchitecture or GraphKind.ConcreteModel)) return;
            if (FirstOutputWithoutShape(graph) is not { } name) return;
            throw new ModelException(ErrorCodes.FW057, $"output '{name}'",
                $"this {Shorokoo.Core.Utils.SrkFileFormat.StageName(kind)} graph's output '{name}' records " +
                "no shape. Every output of a concrete graph records the shape it has at the samples the " +
                "graph was concretized at: lower the module again with ToConcreteArchitecture. A graph " +
                "saved without these shapes cannot be read; re-save it from its module.");
        }

        /// <summary>
        /// The name of the first output of <paramref name="graph"/> that records no shape, or
        /// <c>null</c> when every one does.
        /// </summary>
        internal static string? FirstOutputWithoutShape(InternalComputationGraph graph)
        {
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count; i++)
                if (Get(outputNodes[i]) is null)
                    return InternalComputationGraph.OutputNameOf(outputNodes[i]) ?? $"#{i}";
            return null;
        }

        /// <summary>
        /// Records a shape on each output of a graph read from the ONNX <paramref name="graphProto"/>
        /// that does not already carry the one a Shorokoo export wrote: the shape the file declares
        /// for it, taken as <see cref="RepresentativeInputShapes.RecordFromOnnx"/> takes an input's —
        /// each symbolic (<c>dim_param</c>), unset or negative dimension as <c>1</c> — an optional's
        /// element's, and a sequence's element's (<see cref="NoSharedElementShape"/> where it
        /// declares none). An output the file gives no rank is evaluated at the representative
        /// inputs the import recorded, where every input records one; one left without a shape even
        /// so is refused by the entry points that freeze the graph
        /// (<see cref="ThrowIfImportLeftAnOutputUnshaped"/>).
        /// </summary>
        internal static void RecordFromOnnx(InternalComputationGraph graph, Factory.IR.GraphProto graphProto)
        {
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count && i < graphProto.Outputs.Count; i++)
                if (Get(outputNodes[i]) is null && DeclaredShapeOf(graphProto.Outputs[i]) is { } dims)
                    Set(outputNodes[i], dims);

            if (FirstOutputWithoutShape(graph) is null || !EveryInputIsRepresented(graph)
                || graph.Nodes.Any(FastNodeClassification.IsModuleStageMachinery)) return;
            var evaluated = ShapesAtRepresentativeInputs(graph);
            for (int i = 0; i < outputNodes.Count; i++)
                if (Get(outputNodes[i]) is null && evaluated[i] is { } dims)
                    Set(outputNodes[i], dims);
        }

        /// <summary>Whether every input of <paramref name="graph"/> can be stood in for at its
        /// representative shape — a tensor or optional input recording one, or a sequence input.</summary>
        private static bool EveryInputIsRepresented(InternalComputationGraph graph)
            => graph.InputNodes.All(n => n.OpCode == InternalOpCodes.MODEL_SEQUENCE_INPUT
                || (RepresentativeInputShapes.CarriesShape(n) && RepresentativeInputShapes.Get(n) is not null));

        /// <summary>The shape an output's declared type gives, or <c>null</c> where it gives no rank.</summary>
        private static long[]? DeclaredShapeOf(Factory.IR.ValueInfoProto proto)
        {
            var type = proto.Type;
            if (type?.SequenceType?.ElemType is { } element)
                return element.TensorType?.Shape?.Dims is { } elementDims
                    ? [.. elementDims.Select(OneWhereOpen)]
                    : NoSharedElementShape;
            if (type?.OptionalType?.ElemType is { } optional) type = optional;
            return type?.TensorType?.Shape?.Dims is { } dims ? [.. dims.Select(OneWhereOpen)] : null;
        }

        private static long OneWhereOpen(Factory.IR.TensorShapeProto.Dimension dim)
            => dim.ShouldSerializeDimValue() && dim.DimValue >= 0 ? dim.DimValue : 1L;

        /// <summary>
        /// Refuses (<see cref="ErrorCodes.FW058"/>) an imported graph with an output left without a
        /// shape — one the file gives no rank and whose evaluation at the recorded input shapes
        /// gives none either — naming every such output.
        /// </summary>
        internal static void ThrowIfImportLeftAnOutputUnshaped(InternalComputationGraph graph, string origin)
        {
            var outputNodes = graph.OutputNodes;
            var unshaped = Enumerable.Range(0, outputNodes.Count)
                .Where(i => Get(outputNodes[i]) is null)
                .Select(i => InternalComputationGraph.OutputNameOf(outputNodes[i]) ?? $"#{i}")
                .ToList();
            if (unshaped.Count == 0) return;
            throw new ModelException(ErrorCodes.FW058, origin,
                $"output(s) {string.Join(", ", unshaped.Select(n => $"'{n}'"))} declare no shape, and " +
                "evaluating the model at its inputs' recorded shapes gives none either, so no shape can " +
                "be recorded for them — and every output of an imported model records one. Declare the " +
                "output's shape in the file, or give the inputs the shapes it is computed at with the " +
                "overload that takes input shapes.");
        }
    }
}
