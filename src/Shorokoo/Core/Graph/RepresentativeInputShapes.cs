using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// The representative shape every input of a <see cref="GraphKind.ConcreteArchitecture"/> or
    /// <see cref="GraphKind.ConcreteModel"/> graph carries: the dims of the sample the graph was
    /// concretized at, recorded as <see cref="OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape"/>
    /// on the input's <c>MODEL_TENSOR_INPUT</c> / <c>MODEL_OPTIONAL_INPUT</c> node. It makes a
    /// concrete graph self-describing: training shape inference rebuilds its sample inputs from
    /// it, and the exporter reads each input's rank off it where the input declares none.
    ///
    /// <para>Established where a graph becomes concrete — <c>ToConcreteArchitecture</c> records
    /// the sample it was handed for each input, an ONNX import derives one from the file — and
    /// verified wherever a concrete graph is frozen (<see cref="Verify"/>). The attribute rides on
    /// the node, so it survives every copy, the <c>.srk</c> round trip and <c>Specialize</c>,
    /// which only removes inputs.</para>
    /// </summary>
    internal static class RepresentativeInputShapes
    {
        /// <summary>
        /// The single negative dim recorded for an optional input supplied ABSENT. A concretized
        /// shape never holds one, so it cannot be confused with a real shape — and recording it
        /// rather than recording nothing keeps a MISSING attribute meaning what it means on a
        /// tensor input: a graph that breaks the invariant.
        /// </summary>
        internal static readonly long[] AbsentOptionalShape = [-1L];

        /// <summary>Whether <paramref name="node"/> is an input node that carries the shape.</summary>
        internal static bool CarriesShape(FastNode node)
            => node.OpCode is InternalOpCodes.MODEL_TENSOR_INPUT or InternalOpCodes.MODEL_OPTIONAL_INPUT;

        /// <summary>
        /// Refuses (<see cref="ErrorCodes.FW056"/>) a lowering of <paramref name="graph"/> whose
        /// <paramref name="samples"/> leave any data input without a sample, listing every such
        /// input, give more samples than it has data inputs, stating both counts, or name a sample
        /// for another input than the one at its position, listing every such sample. Samples bind
        /// to the data inputs by position, as the lowering binds them; a generic module's
        /// type-placeholder slots take none.
        /// </summary>
        internal static void RequireSampleForEveryInput(InternalComputationGraph graph, ModelParamList samples)
        {
            var dataInputs = DataInputIndices(graph);
            if (samples.ModelParams.Length > dataInputs.Count)
                throw new ModelException(ErrorCodes.FW056, "ToConcreteArchitecture",
                    $"the graph has {dataInputs.Count} input(s) " +
                    $"({string.Join(", ", dataInputs.Select(i => $"'{NameOf(graph, i) ?? $"#{i}"}'"))}) but " +
                    $"{samples.ModelParams.Length} sample(s) were given. Give exactly one sample per " +
                    "input, in declaration order; a generic module's type-placeholder slots take none.");
            if (samples.ModelParams.Length < dataInputs.Count)
            {
                var missing = dataInputs.Skip(samples.ModelParams.Length)
                    .Select(i => $"'{NameOf(graph, i) ?? $"#{i}"}'");
                throw new ModelException(ErrorCodes.FW056, "ToConcreteArchitecture",
                    $"no sample was given for input(s) {string.Join(", ", missing)}. Every input of the " +
                    "graph needs a sample, [Hyper] inputs included, one per input in declaration order: " +
                    "the lowering records each sample's shape on the concrete architecture, which is " +
                    "what training and ONNX export read the input's shape from. Build the list with " +
                    "graph.FromOrderedInputs([...]).");
            }

            // Samples bind by position; a name is not consulted to bind one, so a sample named for
            // another input is a sample in the wrong place, and binding it anyway would lower the
            // graph at the wrong shapes. An unnamed sample binds where it stands.
            var misnamed = dataInputs
                .Select((inputIndex, k) => (Position: k, Sample: samples.ModelParams[k].ParamName, Input: NameOf(graph, inputIndex)))
                .Where(x => !string.IsNullOrEmpty(x.Sample) && x.Input is not null && x.Sample != x.Input)
                .Select(x => $"sample #{x.Position} is named '{x.Sample}' but input #{x.Position} is '{x.Input}'")
                .ToList();
            if (misnamed.Count > 0)
                throw new ModelException(ErrorCodes.FW056, "ToConcreteArchitecture",
                    $"{string.Join("; ", misnamed)}. Samples bind to the graph's inputs by position, one per " +
                    "input in declaration order; name each after the input at its position, or build the " +
                    "list with graph.FromOrderedInputs([...]).");
        }

        /// <summary>
        /// Records each input's sample shape on the input it is bound to
        /// (<see cref="BindSamplesToLoweredInputs"/>). An input without a sample, or whose sample
        /// has no single shape, is left as it is (<see cref="Verify"/> names the former).
        /// </summary>
        internal static void Record(InternalComputationGraph graph, ModelParamList samples)
        {
            var bound = BindSamplesToLoweredInputs(graph, samples);
            var producers = graph.BuildProducerByOutputMap();
            for (int i = 0; i < graph.Inputs.Count; i++)
            {
                if (!producers.TryGetValue(graph.Inputs[i], out var node) || bound[i] is not { } value) continue;
                if (CarriesShape(node) && ShapeOf(value) is { } dims) Set(node, dims);
                else if (node.OpCode == InternalOpCodes.MODEL_SEQUENCE_INPUT && SharedElementShapeOf(value) is { } element)
                    Set(node, element);
            }
        }

        /// <summary>
        /// The shape every element of a sequence sample shares, or <c>null</c> for an empty
        /// sequence, one whose elements differ in shape, or a value that is no sequence. Recorded
        /// on a sequence input — as an aid, not as the invariant a tensor input's shape is — so
        /// shape inference over a concrete graph can stand the input in by elements of that shape.
        /// </summary>
        private static long[]? SharedElementShapeOf(IData value)
        {
            if (value is SharedInput shared) value = shared.Value;
            if (value is not TensorDataSequence { Count: > 0 } sequence) return null;
            var first = sequence[0].Shape.Dims;
            return sequence.All(e => e.Shape.Dims.AsSpan().SequenceEqual(first)) ? first : null;
        }

        /// <summary>
        /// The sample value bound to each input of <paramref name="graph"/> — a graph in lowering,
        /// whose struct inputs <c>FastUnpackTensorStructs</c> may already have expanded into one
        /// input per field — in input order, <c>null</c> where no sample reaches the input. The one
        /// binding every lowering stage uses, so the stages that evaluate the graph at the samples
        /// and the one that records their shapes cannot disagree about which sample is whose.
        ///
        /// <para>Samples bind by position, as <see cref="RequireSampleForEveryInput"/> checks them: a
        /// struct sample stands for its struct's fields, in declaration order, which is the order
        /// the unpacking gives them as inputs; a generic module's type-placeholder slots take none.
        /// Each value is the sample's own — a tensor, an optional, a sequence, or a field's value,
        /// whatever kind it is.</para>
        /// </summary>
        internal static IData?[] BindSamplesToLoweredInputs(InternalComputationGraph graph, ModelParamList? samples)
        {
            var bound = new IData?[graph.Inputs.Count];
            if (samples is null) return bound;
            var values = samples.ModelParams.SelectMany(ValuesOf).ToList();
            var producers = graph.BuildProducerByOutputMap();
            int next = 0;
            for (int i = 0; i < graph.Inputs.Count && next < values.Count; i++)
            {
                if (producers.TryGetValue(graph.Inputs[i], out var node) && node.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                    continue;
                bound[i] = values[next++];
            }
            return bound;
        }

        private static IEnumerable<IData?> ValuesOf(NamedModelParam sample) => sample switch
        {
            TensorStructModelParam structSample => structSample.Definition.Fields.Select(field =>
                structSample.StructData.Fields.TryGetValue(field.Name, out var value) ? value : null),
            OptionalTensorDataModelParam optional => [optional.ToOptionalTensorData()],
            TensorDataSequenceModelParam sequence => [sequence.ToTensorDataSequence()],
            { Structure: DataStructure.Tensor } => [sample.ToTensorData()],
            _ => [null],
        };

        /// <summary>
        /// The dims to record for one sample value, or <c>null</c> for a value with no single shape
        /// (a sequence, a struct). An absent optional records <see cref="AbsentOptionalShape"/>, not
        /// nothing: reading the shape off the value rather than converting it to a tensor is what
        /// lets an absent optional through at all, as it has no tensor value (Shorokoo/Shorokoo#314).
        /// </summary>
        private static long[]? ShapeOf(IData value) => value switch
        {
            SharedInput shared => ShapeOf(shared.Value),
            TensorData tensor => tensor.Shape.Dims,
            OptionalTensorData { HasValue: true, Value: { } present } => present.Shape.Dims,
            OptionalTensorData => AbsentOptionalShape,
            _ => null,
        };

        /// <summary>Indices of <paramref name="graph"/>'s inputs that take a value: all but a
        /// generic module's type placeholders.</summary>
        private static List<int> DataInputIndices(InternalComputationGraph graph)
        {
            var producers = graph.BuildProducerByOutputMap();
            return [.. Enumerable.Range(0, graph.Inputs.Count).Where(i =>
                !(producers.TryGetValue(graph.Inputs[i], out var node)
                  && node.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))];
        }

        /// <summary>
        /// Records the shape of each of <paramref name="exemplars"/> — one per graph input, in
        /// input order — on the input it stands for: a composed graph's inputs, whose exemplars its
        /// builder already holds for shape inference.
        /// </summary>
        internal static void Record(InternalComputationGraph graph, IReadOnlyList<IRuntimeTensor> exemplars)
        {
            var producers = graph.BuildProducerByOutputMap();
            for (int i = 0; i < graph.Inputs.Count && i < exemplars.Count; i++)
            {
                if (!producers.TryGetValue(graph.Inputs[i], out var node) || !CarriesShape(node)) continue;
                var dims = exemplars[i] switch
                {
                    RuntimeOptionalTensor { HasValue: false } => AbsentOptionalShape,
                    RuntimeOptionalTensor { ValueTensor.Shape: { } s } => s.Dims,
                    RuntimeTensor { Shape: { } s } => s.Dims,
                    _ => null,
                };
                if (dims is not null) Set(node, dims);
            }
        }

        /// <summary>Records <paramref name="dims"/> on the input node <paramref name="node"/>.</summary>
        internal static void Set(FastNode node, long[] dims)
            => node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape, (object?)dims));

        /// <summary>The recorded shape of <paramref name="node"/>, or <c>null</c> when it has none.</summary>
        internal static long[]? Get(FastNode node)
            => node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape)
                ? node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRepresentativeInputShape)
                : null;

        /// <summary>
        /// Holds a concrete graph to the invariant: every tensor or optional input carries a
        /// representative shape. Throws <see cref="ErrorCodes.FW057"/> naming the first input that
        /// does not; a <see cref="GraphKind.Module"/> graph is not checked.
        /// </summary>
        internal static void Verify(InternalComputationGraph graph, GraphKind kind)
        {
            if (kind is not (GraphKind.ConcreteArchitecture or GraphKind.ConcreteModel)) return;
            if (FirstInputWithoutShape(graph) is not { } name) return;
            throw new ModelException(ErrorCodes.FW057, $"input '{name}'",
                $"this {Shorokoo.Core.Utils.SrkFileFormat.StageName(kind)} graph's input '{name}' carries " +
                "no representative shape. Every input of a concrete graph records the shape of the " +
                "sample it was concretized at: lower the module again with ToConcreteArchitecture, " +
                "giving a sample for every input. A graph saved without these shapes cannot be read; " +
                "re-save it from its module.");
        }

        /// <summary>
        /// The name of the first tensor or optional input of <paramref name="graph"/> that carries
        /// no representative shape, or <c>null</c> when every one does.
        /// </summary>
        internal static string? FirstInputWithoutShape(InternalComputationGraph graph)
        {
            if (graph.Inputs.Count == 0) return null;
            var inputKeys = new HashSet<FastNodeKey>(graph.Inputs.Select(k => k.FastNodeKey));
            var nodes = new Dictionary<FastNodeKey, FastNode>();
            foreach (var node in graph.Nodes)
                if (inputKeys.Contains(node.Key)) nodes[node.Key] = node;
            for (int i = 0; i < graph.Inputs.Count; i++)
            {
                if (!nodes.TryGetValue(graph.Inputs[i].FastNodeKey, out var node) || !CarriesShape(node)) continue;
                if (Get(node) is null) return NameOf(graph, i) ?? node.FriendlyName ?? $"#{i}";
            }
            return null;
        }

        /// <summary>
        /// Records a representative shape on each tensor or optional input of a graph read from
        /// the ONNX <paramref name="graphProto"/>: the one in <paramref name="inputShapes"/> where
        /// it names the input, else the one a Shorokoo export carried (already on the node), else
        /// the shape the file declares, each symbolic (<c>dim_param</c>) or unset dimension taken
        /// as <c>1</c>, as is a negative <c>dim_value</c>. An input the file declares no shape for is left without one; the entry
        /// points that freeze the graph refuse it
        /// (<see cref="ThrowIfImportLeftAnInputUnshaped"/>). A name in
        /// <paramref name="inputShapes"/> that is no input, or a shape of another rank than the
        /// file declares or contradicting a dimension it fixes, is refused.
        /// </summary>
        internal static void RecordFromOnnx(
            InternalComputationGraph graph,
            Factory.IR.GraphProto graphProto,
            IReadOnlyDictionary<string, long[]>? inputShapes)
        {
            var protoByName = new Dictionary<string, Factory.IR.ValueInfoProto>(System.StringComparer.Ordinal);
            foreach (var info in graphProto.Inputs) protoByName.TryAdd(info.Name, info);

            var producers = graph.BuildProducerByOutputMap();
            var inputNodes = graph.Inputs
                .Select(k => producers.TryGetValue(k, out var n) ? n : null)
                .OfType<FastNode>()
                .Where(CarriesShape)
                .ToList();

            if (inputShapes is not null)
            {
                var names = inputNodes.Select(n => n.FriendlyName).ToHashSet(System.StringComparer.Ordinal);
                foreach (var (name, dims) in inputShapes)
                {
                    if (!names.Contains(name))
                        throw new System.ArgumentException(
                            $"inputShapes names '{name}', which is not an input of the model. Its inputs " +
                            $"are {string.Join(", ", names.Select(x => $"'{x}'"))}.", nameof(inputShapes));
                    if (dims is null || dims.Any(d => d < 0))
                        throw new System.ArgumentException(
                            $"the shape given for input '{name}' must be concrete dimensions, none negative.",
                            nameof(inputShapes));
                    if (protoByName.TryGetValue(name, out var proto) && DeclaredDimsOf(proto) is { } declared)
                    {
                        if (declared.Count != dims.Length)
                            throw new System.ArgumentException(
                                $"input '{name}' is declared with rank {declared.Count}, but the shape given " +
                                $"for it has rank {dims.Length}.", nameof(inputShapes));
                        var contradicted = Enumerable.Range(0, dims.Length)
                            .Where(d => FixedSizeOf(declared[d]) is { } size && size != dims[d])
                            .Select(d => $"dimension {d} is fixed at {declared[d].DimValue} but given as {dims[d]}")
                            .ToList();
                        if (contradicted.Count > 0)
                            throw new System.ArgumentException(
                                $"the shape given for input '{name}' contradicts the one the model declares: " +
                                $"{string.Join("; ", contradicted)}.", nameof(inputShapes));
                    }
                }
            }

            foreach (var node in inputNodes)
            {
                var name = node.FriendlyName ?? "";
                if (inputShapes is not null && inputShapes.TryGetValue(name, out var given))
                    Set(node, given);
                else if (Get(node) is null
                         && protoByName.TryGetValue(name, out var proto) && DeclaredDimsOf(proto) is { } declared)
                    Set(node, [.. declared.Select(d => FixedSizeOf(d) ?? 1L)]);
            }
        }

        /// <summary>
        /// The size a declared dimension fixes, or <c>null</c> for one it leaves open: a symbolic
        /// (<c>dim_param</c>) or unset one, and a negative <c>dim_value</c>, which some writers use
        /// for "unknown" and which no concrete shape can hold (and <c>-1</c> would read back as an
        /// absent optional's marker).
        /// </summary>
        private static long? FixedSizeOf(Factory.IR.TensorShapeProto.Dimension dim)
            => dim.ShouldSerializeDimValue() && dim.DimValue >= 0 ? dim.DimValue : null;

        /// <summary>
        /// Refuses (<see cref="ErrorCodes.FW058"/>) an imported graph with a tensor or optional
        /// input left without a representative shape — one whose declared type in the file has no
        /// shape, so no rank — naming every such input and the overload that supplies it.
        /// </summary>
        internal static void ThrowIfImportLeftAnInputUnshaped(InternalComputationGraph graph, string origin)
        {
            var producers = graph.BuildProducerByOutputMap();
            var unshaped = new List<string>();
            for (int i = 0; i < graph.Inputs.Count; i++)
                if (producers.TryGetValue(graph.Inputs[i], out var node) && CarriesShape(node) && Get(node) is null)
                    unshaped.Add(node.FriendlyName ?? NameOf(graph, i) ?? $"#{i}");
            if (unshaped.Count == 0) return;
            throw new ModelException(ErrorCodes.FW058, origin,
                $"input(s) {string.Join(", ", unshaped.Select(n => $"'{n}'"))} declare no shape, so their " +
                "rank is unknown and no representative shape can be derived for them — and every input " +
                "of an imported model records one. Give each its shape with the overload that takes " +
                "input shapes, e.g. Persistence.ImportOnnx(filePath, new Dictionary<string, long[]> " +
                $"{{ [\"{unshaped[0]}\"] = [1, 3, 224, 224] }}) or " +
                "OnnxModelImporter.FromOnnxModel(filePath, inputShapes).");
        }

        /// <summary>The dimensions a graph input's declared type gives, or <c>null</c> when it
        /// declares no shape (unknown rank). An optional input's are its element's.</summary>
        private static List<Factory.IR.TensorShapeProto.Dimension>? DeclaredDimsOf(Factory.IR.ValueInfoProto proto)
        {
            var type = proto.Type;
            if (type?.OptionalType?.ElemType is { } element) type = element;
            return type?.TensorType?.Shape?.Dims;
        }

        private static string? NameOf(InternalComputationGraph graph, int i)
            => i < graph.InputUniqueNames.Count ? graph.InputUniqueNames[i] : null;
    }
}
