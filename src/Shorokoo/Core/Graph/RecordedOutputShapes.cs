using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.Interpreter;
using Shorokoo.Core.Interpreter.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.Processors.Training;
using Shorokoo.Graph;
using Shorokoo.Runtime;

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
    /// (<see cref="Verify"/>). The attribute rides on the node, so it survives every copy and the
    /// <c>.srk</c> round trip; <c>ToConcreteModel</c> and <c>Specialize</c>, which change what an
    /// output's shape can hang on, record it again (<see cref="Rerecord"/>).</para>
    ///
    /// <para>Every output records one, whatever it holds: a tensor its shape; an optional its
    /// element's where present, and <see cref="RepresentativeInputShapes.AbsentOptionalShape"/>
    /// where absent; a sequence the shape its elements share, and
    /// <see cref="NoSharedElementShape"/> where they share none; and, on a concrete architecture
    /// only, <see cref="UnresolvedShape"/> for one whose shape its weights decide. So a missing
    /// attribute always means a graph that breaks the invariant.</para>
    /// </summary>
    internal static class RecordedOutputShapes
    {
        /// <summary>
        /// The single negative dim recorded for a sequence output whose elements share no shape (or
        /// that is empty). Distinct from <see cref="RepresentativeInputShapes.AbsentOptionalShape"/>,
        /// and, like it, never a concrete shape.
        /// </summary>
        internal static readonly long[] NoSharedElementShape = [-2L];

        /// <summary>
        /// The single negative dim recorded for an output whose shape could not be settled on a
        /// concrete architecture — one that hangs on the values of parameters not yet initialized,
        /// or on an op neither the engine nor a run could evaluate. Allowed there, and only there:
        /// <c>ToConcreteModel</c> records every such output again with the weights bound
        /// (<see cref="Rerecord"/>), and a concrete model carrying one is refused.
        /// </summary>
        internal static readonly long[] UnresolvedShape = [-3L];

        /// <summary>Whether <paramref name="dims"/> is <see cref="UnresolvedShape"/>.</summary>
        internal static bool IsUnresolved(long[]? dims) => dims is [-3L];

        /// <summary>Records <paramref name="dims"/> on the output node <paramref name="node"/>.</summary>
        internal static void Set(FastNode node, long[] dims)
            => node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrRecordedOutputShape, (object?)dims));

        private static void Clear(FastNode node)
            => node.Attributes = node.Attributes.SetAttributes(
                (OnnxOpAttributeNames.ShrkAttrRecordedOutputShape, (object?)null));

        /// <summary>The recorded shape of the output node <paramref name="node"/>, or <c>null</c>.</summary>
        internal static long[]? Get(FastNode node)
            => node.Attributes.IsAttributeDefined(OnnxOpAttributeNames.ShrkAttrRecordedOutputShape)
                ? node.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrRecordedOutputShape)
                : null;

        /// <summary>
        /// The rank <paramref name="node"/>'s recorded shape gives the output, or <c>null</c> when it
        /// records none or records a marker (<see cref="RepresentativeInputShapes.AbsentOptionalShape"/>,
        /// <see cref="NoSharedElementShape"/>, <see cref="UnresolvedShape"/>) rather than a shape.
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
        /// stands in by its declared shape and dtype alone, never by values: an output whose shape
        /// hangs on a parameter's values is not settled here but by <see cref="Rerecord"/>, once
        /// <c>ToConcreteModel</c> has bound the real ones.</para>
        ///
        /// <para>An output the engine cannot compute at all — an operator it has no kernel for, such
        /// as the string ops — is then taken from a real run of the graph at the samples on
        /// <paramref name="computeContext"/> (the default one when <c>null</c>), provided the graph
        /// has no weight to bind, as a run at stand-in values would record a shape the real ones
        /// need not give. That run costs a session, so it is made only when an output is left
        /// unresolved by the engine. Where no run is made, or the compute context cannot make it —
        /// it refuses the graph on this machine, say — an output whose rank the engine did settle
        /// records it, each dimension it left open taken as <c>1</c>, as an ONNX import takes a
        /// symbolic one (<see cref="RecordFromOnnx"/>). Any output left over records
        /// <see cref="UnresolvedShape"/>, which a concrete architecture may carry and a concrete
        /// model may not.</para>
        /// </summary>
        internal static void RecordAtSamples(
            InternalComputationGraph graph, ModelParamList samples, ComputeContext? computeContext)
        {
            foreach (var node in graph.OutputNodes) Clear(node);
            var store = EvaluateDefinite(graph,
                max => FastProcessorHelper.SampleRuntimeInputs(graph, samples, new QuickExecutionEngine { MaxDataElements = max }) ?? []);
            if (FirstOutputWithoutShape(graph) is null) return;
            if (!BindsAWeight(graph))
                RecordFromARun(graph, RepresentativeInputShapes.BindSamplesToLoweredInputs(graph, samples),
                    computeContext, out _);
            RecordFrom(graph, store, partial: true);
            foreach (var node in graph.OutputNodes)
                if (Get(node) is null) Set(node, UnresolvedShape);
        }

        /// <summary>
        /// Records each output's shape afresh on <paramref name="graph"/> — a concrete graph whose
        /// parameters were just bound (<c>ToConcreteModel</c>) or some of whose inputs were just
        /// baked in (<c>Specialize</c>) — so no output keeps a shape the change has made stale.
        ///
        /// <para>The engine evaluates it at its inputs' recorded shapes, their values unknown, with
        /// every bound parameter's values and every baked input's: a shape it settles there holds
        /// whatever values the inputs take, so it replaces the recorded one. An output it leaves
        /// open keeps the shape it recorded before — recorded at the samples' own values, which
        /// this walk does not have — unless that was <see cref="UnresolvedShape"/>; then it records
        /// the rank the engine settled, each open dimension as <c>1</c>. On a graph with no weight
        /// left to bind, an output still unresolved is taken from a run at zeros of its inputs'
        /// recorded shapes on <paramref name="computeContext"/>, and one that even this leaves
        /// unresolved is refused (<see cref="ErrorCodes.FW057"/>), naming it, with what the run
        /// threw as the inner exception. A graph with a weight still unbound records
        /// <see cref="UnresolvedShape"/> for it instead.</para>
        /// </summary>
        internal static void Rerecord(InternalComputationGraph graph, ComputeContext? computeContext = null)
        {
            var outputNodes = graph.OutputNodes;
            var previous = outputNodes.Select(Get).ToArray();
            foreach (var node in outputNodes) Clear(node);
            var store = EvaluateDefinite(graph, _ => ShapeOnlyInputs(graph));
            for (int i = 0; i < outputNodes.Count; i++)
                if (Get(outputNodes[i]) is null && previous[i] is { } kept && !IsUnresolved(kept))
                    Set(outputNodes[i], kept);
            RecordFrom(graph, store, partial: true);
            if (FirstOutputWithoutShape(graph) is null) return;

            Exception? failure = null;
            bool bound = !BindsAWeight(graph);
            if (bound) RecordFromARun(graph, ZeroInputsAtRecordedShapes(graph), computeContext, out failure);
            for (int i = 0; i < outputNodes.Count; i++)
            {
                if (Get(outputNodes[i]) is not null) continue;
                if (!bound)
                {
                    Set(outputNodes[i], UnresolvedShape);
                    continue;
                }
                var name = Describe(outputNodes[i], i);
                var reason = $"the shape of this concrete model's output {name} could not be settled: " +
                    "evaluating the model with its weights bound leaves it open, and so does a run at its " +
                    "inputs' recorded shapes" + (failure is null ? "." : $", which failed: {failure.Message}") +
                    " Every output of a concrete model records its shape.";
                throw failure is null
                    ? new ModelException(ErrorCodes.FW057, $"output {name}", reason)
                    : new ModelException(ErrorCodes.FW057, $"output {name}", reason, failure);
            }
        }

        /// <summary>
        /// Records each output still without a shape from a run of <paramref name="graph"/> — a
        /// graph with no weight to bind, whose RNG identity, where it has one unbound, is the
        /// default one — at <paramref name="inputs"/> on <paramref name="computeContext"/> (the
        /// default one when <c>null</c>). Left as it is where an input has no value to run at, and
        /// where the compute context cannot run the graph, <paramref name="failure"/> then holding
        /// why. A failure that is no run's — a fault in Shorokoo itself — is not caught.
        /// </summary>
        private static void RecordFromARun(
            InternalComputationGraph graph, IData?[] inputs, ComputeContext? computeContext, out Exception? failure)
        {
            failure = null;
            var shared = inputs.Select(value => value is null ? null : SharedOf(value)).ToArray();
            if (shared.Any(value => value is null)) return;

            var run = graph;
            if (FastWireRngKeyDerivation.FindRngSeedNode(graph) is { OpCode: InternalOpCodes.MODEL_PARAM })
            {
                run = FastApplyModelParamValues.Process(graph, new Dictionary<ModelId, TensorAttribute>());
                run.ApplyRngConfig(RngConfig.Default);
            }

            NamedModelParam[] outputs;
            try { outputs = (computeContext ?? ComputeContext.Default).Execute(run, shared!); }
            catch (Exception e) when (IsARunFailure(e))
            {
                failure = e;
                return;
            }
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count && i < outputs.Length; i++)
                if (Get(outputNodes[i]) is null && ShapeOf(outputs[i]) is { } dims)
                    Set(outputNodes[i], dims);
        }

        /// <summary>
        /// Whether <paramref name="e"/> is a compute context refusing or failing a run — a
        /// Shorokoo error, or one a backend raises of its own — rather than a fault in this code:
        /// a runtime error (<see cref="SystemException"/>) or a failed assertion, which propagate.
        /// </summary>
        private static bool IsARunFailure(Exception e)
            => e is ShorokooException
               || (e is not SystemException
                   && e.GetType().Assembly != typeof(RecordedOutputShapes).Assembly
                   && e.GetType().Assembly != typeof(object).Assembly);

        /// <summary>Whether <paramref name="graph"/> has a weight still to bind: a <c>MODEL_PARAM</c>
        /// other than the RNG identity at reserved ModelId <c>[0]</c>.</summary>
        private static bool BindsAWeight(InternalComputationGraph graph)
            => graph.Nodes.Any(n => n.OpCode == InternalOpCodes.MODEL_PARAM
                && n.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId) is not [0]);

        /// <summary>
        /// Each input of <paramref name="graph"/> as the engine takes it for <see cref="Rerecord"/>:
        /// the shape and dtype it records and no values. An input recording no shape is left unknown.
        /// </summary>
        private static Dictionary<FastTensorKey, IRuntimeTensor> ShapeOnlyInputs(InternalComputationGraph graph)
        {
            var inputs = new Dictionary<FastTensorKey, IRuntimeTensor>();
            foreach (var node in graph.InputNodes)
            {
                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype) ?? DType.Invalid;
                var dims = RepresentativeInputShapes.Get(node);
                IRuntimeTensor? value = node.OpCode switch
                {
                    InternalOpCodes.MODEL_SEQUENCE_INPUT => new RuntimeSequenceTensor
                    {
                        DType = dtype,
                        TemplateTensor = dims is null ? new RuntimeTensor { DType = dtype } : ShapeOnly(dims, dtype),
                    },
                    InternalOpCodes.MODEL_OPTIONAL_INPUT when dims is [-1L]
                        => new RuntimeOptionalTensor { DType = dtype, HasValue = false },
                    InternalOpCodes.MODEL_OPTIONAL_INPUT when dims is not null => new RuntimeOptionalTensor
                    {
                        DType = dtype,
                        HasValue = true,
                        ValueTensor = ShapeOnly(dims, dtype),
                    },
                    InternalOpCodes.MODEL_TENSOR_INPUT when dims is not null => ShapeOnly(dims, dtype),
                    _ => null,
                };
                if (value is not null) inputs[InternalComputationGraph.InputKeyOf(node)] = value;
            }
            return inputs;
        }

        private static RuntimeTensor ShapeOnly(long[] dims, DType dtype) => new() { DType = dtype, Shape = new Shape(dims) };

        /// <summary>
        /// Zeros — empty strings for a string — of each input's recorded shape, an optional
        /// recorded absent as absent: the one run <see cref="Rerecord"/> can make, having no
        /// samples. <c>null</c> for an input that records no single shape (a sequence) or whose
        /// dtype has no zero.
        /// </summary>
        private static IData?[] ZeroInputsAtRecordedShapes(InternalComputationGraph graph)
            => [.. graph.InputNodes.Select(node =>
            {
                var dtype = node.Attributes.GetDTypeVal(OnnxOpAttributeNames.AttrDtype);
                if (dtype is null || RepresentativeInputShapes.Get(node) is not { } dims) return null;
                bool optional = node.OpCode == InternalOpCodes.MODEL_OPTIONAL_INPUT;
                if (optional && dims is [-1L]) return OptionalTensorData.None(dtype);
                if (!RepresentativeInputShapes.CarriesShape(node) || ZerosOf(new Shape(dims), dtype) is not { } zeros)
                    return null;
                var value = zeros.CopyToTensorData();
                return optional ? OptionalTensorData.Some(value) : (IData?)value;
            })];

        /// <summary>An output as a message names it: by its name, or by its index where it has
        /// none of its own — only the key of the value it reads.</summary>
        internal static string Describe(FastNode outputNode, int index)
            => InternalComputationGraph.OutputNameOf(outputNode) is { } name && !TensorKey.TryParse(name, out _)
                ? $"'{name}'"
                : $"#{index}";

        /// <summary>Zeros — empty strings for a string — of <paramref name="shape"/>, or <c>null</c>
        /// for a dtype whose elements have no fixed whole-byte width.</summary>
        private static TensorAttribute? ZerosOf(Shape shape, DType dtype)
            => dtype == DType.Utf8
                ? TensorAttribute.OverStrings(shape, [.. Enumerable.Repeat("", checked((int)shape.Count))])
                : ByteWidthOf(dtype) is int width
                    ? TensorAttribute.OverBytes(shape, dtype, new byte[shape.Count * width])
                    : null;

        /// <summary>A sample as a run reads it without taking it: the caller keeps its samples.</summary>
        private static IData? SharedOf(IData value) => value switch
        {
            SharedInput shared => shared,
            TensorData tensor => tensor.Shared(),
            OptionalTensorData optional => optional.Shared(),
            TensorDataSequence sequence => sequence.Shared(),
            _ => null,
        };

        /// <summary>The dims to record for one output of a run.</summary>
        private static long[]? ShapeOf(NamedModelParam output) => output switch
        {
            OptionalTensorDataModelParam optional => optional.ToOptionalTensorData() is { HasValue: true, Value: { } present }
                ? present.Shape.Dims
                : RepresentativeInputShapes.AbsentOptionalShape,
            TensorDataSequenceModelParam sequence => sequence.ToTensorDataSequence() is { Count: > 0 } elements
                && elements.All(e => e.Shape.Dims.AsSpan().SequenceEqual(elements[0].Shape.Dims))
                    ? elements[0].Shape.Dims
                    : NoSharedElementShape,
            { Structure: DataStructure.Tensor } => output.ToTensorData().Shape.Dims,
            _ => null,
        };

        /// <summary>
        /// The bytes one element of <paramref name="dtype"/> takes, or <c>null</c> for a dtype
        /// with no fixed whole-byte width (a string, a packed 4-bit or a complex type), whose
        /// elements cannot be written as raw bytes.
        /// </summary>
        internal static int? ByteWidthOf(DType dtype)
            => dtype == DType.Utf8 || dtype == DType.Int4 || dtype == DType.UInt4
               || dtype == DType.Complex64 || dtype == DType.Complex128
               || dtype.IsGenericType || dtype == DType.Invalid
               || dtype == DType.Module || dtype == DType.Model || dtype.TensorStructDef is not null
                ? null
                : dtype.EncodingBitCount / 8;

        /// <summary>
        /// Records every output the engine settles exactly, evaluating <paramref name="graph"/> at
        /// <paramref name="inputsAt"/> — its inputs, read at a given small-tensor threshold — small
        /// values first and, where that leaves an output open, all of them. The last store, for a
        /// partial read.
        /// </summary>
        private static Dictionary<FastTensorKey, IRuntimeTensor> EvaluateDefinite(
            InternalComputationGraph graph, Func<int, Dictionary<FastTensorKey, IRuntimeTensor>> inputsAt)
        {
            var store = Evaluate(graph, inputsAt, ShapeInferenceInterpreter.MaxSmallTensorElements);
            if (RecordFrom(graph, store, partial: false)) return store;
            store = Evaluate(graph, inputsAt, int.MaxValue);
            RecordFrom(graph, store, partial: false);
            return store;
        }

        private static Dictionary<FastTensorKey, IRuntimeTensor> Evaluate(
            InternalComputationGraph graph, Func<int, Dictionary<FastTensorKey, IRuntimeTensor>> inputsAt, int maxDataElements)
        {
            var engine = new QuickExecutionEngine { MaxDataElements = maxDataElements };
            var initial = inputsAt(maxDataElements);
            AddParameterStandIns(graph, initial);
            return engine.Run(graph, initial);
        }

        /// <summary>
        /// Stands each uninitialized parameter (<c>MODEL_PARAM</c>) of <paramref name="graph"/> in by
        /// the shape and dtype its node declares and no values, so the engine carries its shape
        /// rather than an unknown and settles nothing on values the parameter has not been given.
        /// One declaring no concrete shape is left unknown.
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
                initial[key] = ShapeOnly(dims, dtype);
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
            => ShapesFrom(graph, StoreAtRepresentativeInputs(graph), partial: true);

        /// <summary>The engine's evaluation of <paramref name="graph"/> at its representative inputs.</summary>
        private static Dictionary<FastTensorKey, IRuntimeTensor> StoreAtRepresentativeInputs(InternalComputationGraph graph)
        {
            var engine = new QuickExecutionEngine { MaxDataElements = ShapeInferenceInterpreter.MaxSmallTensorElements };
            var inputs = TrainingRig.ReadRepresentativeInputs(graph, representSequences: true);
            var initial = new Dictionary<FastTensorKey, IRuntimeTensor>();
            var keys = graph.Inputs;
            for (int i = 0; i < keys.Count; i++) initial[keys[i]] = inputs[i];
            AddParameterStandIns(graph, initial);
            return engine.Run(graph, initial);
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
        /// Records the shape of each output still without one from the engine's
        /// <paramref name="store"/> — with <paramref name="partial"/>, one whose rank alone it
        /// settled too, each open dimension as <c>1</c>; whether every output then records one.
        /// </summary>
        private static bool RecordFrom(
            InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> store, bool partial)
        {
            var shapes = ShapesFrom(graph, store, partial);
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count; i++)
                if (Get(outputNodes[i]) is null && shapes[i] is { } dims) Set(outputNodes[i], dims);
            return FirstOutputWithoutShape(graph) is null;
        }

        private static long[]?[] ShapesFrom(
            InternalComputationGraph graph, Dictionary<FastTensorKey, IRuntimeTensor> store, bool partial)
            => [.. graph.Outputs.Select(key => store.TryGetValue(key, out var value) ? ShapeOf(value, partial) : null)];

        /// <summary>The dims to record for one evaluated value, or <c>null</c> where the evaluation
        /// left its shape — or, with <paramref name="partial"/>, its rank — unknown.</summary>
        private static long[]? ShapeOf(IRuntimeTensor value, bool partial) => value switch
        {
            RuntimeTensor tensor => Definite(tensor, partial),
            RuntimeOptionalTensor { HasValue: false } => RepresentativeInputShapes.AbsentOptionalShape,
            RuntimeOptionalTensor { HasValue: true, ValueTensor: { } present } => Definite(present, partial),
            RuntimeSequenceTensor sequence => SharedElementShapeOf(sequence) ?? (partial ? NoSharedElementShape : null),
            _ => null,
        };

        /// <summary>The dims of <paramref name="tensor"/>, or with <paramref name="partial"/>, where
        /// the engine settled only its rank — as a shape with open dims, or as a rank alone — that
        /// rank with each open dim as <c>1</c>.</summary>
        private static long[]? Definite(RuntimeTensor tensor, bool partial = false)
            => tensor.DType == DType.Invalid ? null
                : tensor.HasDefiniteShape ? tensor.Shape!.Dims
                : !partial ? null
                : tensor.Shape is { } shape ? [.. shape.Dims.Select(d => d < 0 ? 1L : d)]
                : tensor.Rank is int rank ? [.. Enumerable.Repeat(1L, rank)]
                : null;

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
        /// Holds a concrete graph to the invariant: every output records a shape, and every output
        /// of a concrete model a settled one, not <see cref="UnresolvedShape"/>. Throws
        /// <see cref="ErrorCodes.FW057"/> naming the first output that does not; a
        /// <see cref="GraphKind.Module"/> graph is not checked.
        /// </summary>
        internal static void Verify(InternalComputationGraph graph, GraphKind kind)
        {
            if (kind is not (GraphKind.ConcreteArchitecture or GraphKind.ConcreteModel)) return;
            if (kind == GraphKind.ConcreteModel && FirstUnresolvedOutput(graph) is { } unresolved)
                throw new ModelException(ErrorCodes.FW057, $"output {unresolved}",
                    $"this concrete-model graph's output {unresolved} records no settled shape, only the " +
                    "marker a concrete architecture carries for a shape its weights decide. Every output of " +
                    "a concrete model records its shape: make the model with ToConcreteModel, which " +
                    "settles each against the weights it binds.");
            if (FirstOutputWithoutShape(graph) is not { } name) return;
            throw new ModelException(ErrorCodes.FW057, $"output '{name}'",
                $"this {Shorokoo.Core.Utils.SrkFileFormat.StageName(kind)} graph's output '{name}' records " +
                "no shape. Every output of a concrete graph records the shape it has at the samples the " +
                "graph was concretized at: lower the module again with ToConcreteArchitecture.");
        }

        /// <summary>Whether every output of <paramref name="graph"/> records a shape, as every
        /// output of a concrete graph does.</summary>
        internal static bool RecordsEveryOutput(InternalComputationGraph graph) => FirstOutputWithoutShape(graph) is null;

        /// <summary>
        /// The first output of <paramref name="graph"/> that records <see cref="UnresolvedShape"/>,
        /// as a message names it (<see cref="Describe"/>), or <c>null</c> when none does.
        /// </summary>
        internal static string? FirstUnresolvedOutput(InternalComputationGraph graph)
        {
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count; i++)
                if (IsUnresolved(Get(outputNodes[i])))
                    return Describe(outputNodes[i], i);
            return null;
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
        /// Records a shape on each output of a graph read from a foreign ONNX
        /// <paramref name="graphProto"/> — one no Shorokoo builder wrote, so it carries no graph-kind
        /// tag, and whose ops make it concrete: the shape the file declares
        /// for it, taken as <see cref="RepresentativeInputShapes.RecordFromOnnx"/> takes an input's —
        /// each symbolic (<c>dim_param</c>), unset or negative dimension as <c>1</c> — an optional's
        /// element's, and a sequence's element's (<see cref="NoSharedElementShape"/> where it
        /// declares none). An output the file gives no rank is evaluated at the representative
        /// inputs the import recorded, where every input records one: by the engine, else — for an
        /// op the engine cannot compute — by a run at zeros of those shapes, else, where the engine
        /// settled its rank alone, as that rank with each dimension <c>1</c>. One left without a
        /// shape even so is refused by the entry points that freeze the graph
        /// (<see cref="ThrowIfImportLeftAnOutputUnshaped"/>).
        /// </summary>
        internal static void RecordFromOnnx(InternalComputationGraph graph, Factory.IR.GraphProto graphProto)
        {
            if (graph.Nodes.Any(FastNodeClassification.IsModuleStageMachinery)) return;
            var outputNodes = graph.OutputNodes;
            for (int i = 0; i < outputNodes.Count && i < graphProto.Outputs.Count; i++)
                if (Get(outputNodes[i]) is null && DeclaredShapeOf(graphProto.Outputs[i]) is { } dims)
                    Set(outputNodes[i], dims);

            if (FirstOutputWithoutShape(graph) is null || !EveryInputIsRepresented(graph)) return;
            var store = StoreAtRepresentativeInputs(graph);
            if (RecordFrom(graph, store, partial: false)) return;
            if (!BindsAWeight(graph))
                RecordFromARun(graph, ZeroInputsAtRecordedShapes(graph), computeContext: null, out _);
            RecordFrom(graph, store, partial: true);
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
            bool everyInputShaped = graph.InputNodes.All(n => RepresentativeInputShapes.Get(n) is not null);
            throw new ModelException(ErrorCodes.FW058, origin,
                $"output(s) {string.Join(", ", unshaped.Select(n => $"'{n}'"))} declare no shape, and " +
                "neither evaluating the model at its inputs' recorded shapes nor running it there gives " +
                "one, so no shape can be recorded for them — and every output of an imported model " +
                "records one. Declare the output's shape in the file" + (everyInputShaped ? "." :
                ", or give the inputs the shapes it is computed at with the overload that takes input shapes."));
        }
    }
}
