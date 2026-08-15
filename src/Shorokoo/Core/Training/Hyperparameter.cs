using System;
using System.Collections.Generic;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Training;
using Shorokoo.Graph;

namespace Shorokoo
{
    /// <summary>Which of the three sources drives a <see cref="Hyperparameter"/>'s value.</summary>
    public enum HyperparameterKind
    {
        /// <summary>A fixed constant compiled into the training-step graph. Changing it rebuilds the rig.</summary>
        Baked,

        /// <summary>Driven in-graph by a scheduler source — a built-in <see cref="Schedule"/> or a user module.</summary>
        Scheduled,

        /// <summary>Supplied by the host every step as a runtime input.</summary>
        Runtime,
    }

    /// <summary>
    /// The source bound to one optimizer hyperparameter, as supplied when building a
    /// <see cref="TrainingRig"/>. A hyperparameter is a declared signature (name/dtype/shape, from the
    /// optimizer's <c>[Hyper]</c> declarations) bound to <b>exactly one</b> of three sources — an
    /// explicit closed union with an exhaustive <see cref="Kind"/> and no representable invalid states:
    ///
    /// <list type="bullet">
    /// <item><description>
    /// <b><see cref="Baked(float)"/></b> (and its per-dtype siblings) — a fixed value compiled into the
    /// training-step graph as a <c>Constant</c>. Changing it requires rebuilding the rig. Use this for
    /// fixed hyperparameters (e.g. AdamW's betas). A bare <see cref="float"/>, <see cref="double"/>,
    /// <see cref="int"/>, <see cref="long"/> or <see cref="bool"/> converts implicitly to <c>Baked</c>.
    /// </description></item>
    /// <item><description>
    /// <b><see cref="Scheduled(Schedule)"/></b> / <b><see cref="Scheduled(ComputationGraph)"/></b> —
    /// driven in-graph by a <i>scheduler source</i>: a built-in <see cref="Schedule"/> (implicitly
    /// converted; lowered to graph math by <c>ScheduleLowering</c>) or a user scheduler <b>module</b>
    /// (a Shorokoo module taking the int64 counter input(s) and producing the scheduled value).
    /// Both are the single <c>Scheduled</c> kind — two <i>constructors</i> of the same thing — computed
    /// on the engine each step from the counter input(s), with no host evaluation.
    /// </description></item>
    /// <item><description>
    /// <b><see cref="Runtime()"/></b> — supplied by the host every step via
    /// <see cref="TrainingRig.MakeHyperparameters(float)"/> and the override <c>TrainStep</c> overload.
    /// Useful for manual control and tests. It carries no seed: the shape-inference placeholder is an
    /// internal concern.
    /// </description></item>
    /// </list>
    ///
    /// <para><b>Dtype and shape.</b> The optimizer's declared signature — <c>[Hyper(...)]</c> on a
    /// <c>Scalar&lt;T&gt;</c>, <c>Vector&lt;T&gt;</c> or <c>Tensor&lt;T&gt;</c> — is the single source of
    /// truth for a hyperparameter's dtype and rank; a hyperparameter can be any supported dtype, not just
    /// <c>float32</c>, and any shape, not just a scalar. The concrete <i>shape</i> comes from the binding:
    /// a baked constant's own shape, a scheduler module's output shape, or the shape declared by
    /// <see cref="Runtime(long[])"/>. A baked value carries the dtype it was built at
    /// (<see cref="BakedDType"/>) and is fitted to the declared dtype at rig build, which fails loud when
    /// the conversion would lose the value. Built-in <see cref="Schedule"/> math is continuous and scalar,
    /// so a scheduled built-in requires a <c>float32</c> scalar declaration; a scheduler <i>module</i> may
    /// produce any declared dtype at any shape. Only the <c>[Hyper(default)]</c> literal stays
    /// scalar-only — an attribute argument is a compile-time constant and cannot be a tensor.</para>
    ///
    /// Nothing on this type answers "what is the value" for a scheduled or runtime hyperparameter — those
    /// come only from evaluating the source's canonical graph representation (in-graph per step; QEE at
    /// build for optimizer state init). This mirrors how Keras (<c>Adam(learning_rate=schedule)</c>) and
    /// Optax (<c>adamw(learning_rate=schedule)</c>) let the value's type decide constant-vs-scheduled.
    /// There is deliberately no API that accepts an arbitrary host lambda: a compiled closure has no
    /// durable graph representation, so schedules come only from the built-in factories/combinators or a
    /// scheduler module.
    /// </summary>
    public readonly struct Hyperparameter
    {
        private readonly TensorData? _bakedValue;
        private readonly Schedule? _schedule;
        private readonly ComputationGraph? _schedulerModule;
        private readonly long[]? _runtimeShape;

        /// <summary>Which of the three sources drives this hyperparameter (exhaustive, no invalid states).</summary>
        public HyperparameterKind Kind { get; }

        private Hyperparameter(
            HyperparameterKind kind, TensorData? bakedValue, Schedule? schedule,
            ComputationGraph? schedulerModule, long[]? runtimeShape)
        {
            Kind = kind;
            _bakedValue = bakedValue;
            _schedule = schedule;
            _schedulerModule = schedulerModule;
            _runtimeShape = runtimeShape;
        }

        /// <summary>A fixed <c>float32</c> value baked into the graph as a constant.</summary>
        public static Hyperparameter Baked(float value) => Baked(HyperparameterValues.Of(value));

        /// <summary>A fixed <c>float64</c> value baked into the graph as a constant.</summary>
        public static Hyperparameter Baked(double value) => Baked(HyperparameterValues.Of(value));

        /// <summary>A fixed <c>int32</c> value baked into the graph as a constant.</summary>
        public static Hyperparameter Baked(int value) => Baked(HyperparameterValues.Of(value));

        /// <summary>A fixed <c>int64</c> value baked into the graph as a constant.</summary>
        public static Hyperparameter Baked(long value) => Baked(HyperparameterValues.Of(value));

        /// <summary>A fixed <c>bool</c> value baked into the graph as a constant.</summary>
        public static Hyperparameter Baked(bool value) => Baked(HyperparameterValues.Of(value));

        /// <summary>
        /// A fixed tensor of an explicitly chosen dtype and shape, baked into the graph as a constant —
        /// the general form behind the per-dtype overloads. Use it for a dtype with no natural C# literal
        /// (e.g. <c>float16</c> or <c>uint64</c>) and for any <b>non-scalar</b> hyperparameter, whose
        /// shape this value supplies.
        /// </summary>
        public static Hyperparameter Baked(TensorData value)
            => new(HyperparameterKind.Baked, value ?? throw new ArgumentNullException(nameof(value)),
                null, null, null);

        /// <summary>
        /// A built-in <see cref="Schedule"/>, lowered to graph math and evaluated in-graph from the
        /// counter input(s) each training step. Built-in schedule math is continuous, so the
        /// hyperparameter it drives must be declared <c>float32</c>.
        /// </summary>
        public static Hyperparameter Scheduled(Schedule schedule)
            => new(HyperparameterKind.Scheduled, null,
                schedule ?? throw new ArgumentNullException(nameof(schedule)), null, null);

        /// <summary>
        /// A user-supplied scheduler <b>module</b> — a Shorokoo module graph taking the int64 scalar
        /// counter input(s) (named from <c>step</c>, <c>epoch</c>, <c>batchIndex</c>) and producing the
        /// scheduled value at the hyperparameter's declared dtype. Inlined into the training-step graph
        /// as a constituent; its signature and purity are validated at rig build. Use this for schedules
        /// the built-in <see cref="Schedules"/> factories don't cover, or for a non-<c>float32</c>
        /// hyperparameter.
        /// </summary>
        public static Hyperparameter Scheduled(ComputationGraph schedulerModule)
            => new(HyperparameterKind.Scheduled, null, null,
                schedulerModule ?? throw new ArgumentNullException(nameof(schedulerModule)), null);

        /// <summary>
        /// A dynamic <b>scalar</b> with no schedule: the rig routes it as a runtime input and you must
        /// supply it explicitly each step (see <see cref="TrainingRig.MakeHyperparameters(float)"/>).
        /// Seedless: the shape-inference placeholder is internal, never user-visible, and the field's
        /// dtype comes from the optimizer's declaration.
        /// </summary>
        public static Hyperparameter Runtime() => Runtime([]);

        /// <summary>
        /// A dynamic value of the given <paramref name="shape"/> with no schedule. A non-scalar runtime
        /// hyperparameter must state its shape here: the rig compiles the training step once, so the
        /// shape has to be known at build even though the values are not (unlike a baked or scheduled
        /// hyperparameter, which carries its shape in its constant or its scheduler graph). Every
        /// per-step value must then match it exactly. Still seedless — this is a shape, not a value.
        /// </summary>
        public static Hyperparameter Runtime(params long[] shape)
        {
            if (shape is null) throw new ArgumentNullException(nameof(shape));
            foreach (var d in shape)
                if (d < 0)
                    throw new ArgumentException(
                        $"A runtime hyperparameter's shape must have non-negative dims; got [{string.Join(", ", shape)}].",
                        nameof(shape));
            return new(HyperparameterKind.Runtime, null, null, null, (long[])shape.Clone());
        }

        /// <summary>
        /// The baked constant value, as the <see cref="TensorData"/> it was built from — which also
        /// carries its shape. Defined only for a <see cref="HyperparameterKind.Baked"/> hyperparameter;
        /// throws otherwise (the value of a scheduled/runtime hyperparameter comes from evaluating its
        /// graph, never from this type).
        /// </summary>
        public TensorData BakedValue => Kind == HyperparameterKind.Baked
            ? _bakedValue ?? throw new InvalidOperationException(
                "This Baked hyperparameter carries no value; build it with Hyperparameter.Baked(...).")
            : throw new InvalidOperationException(
                $"BakedValue is defined only for a Baked hyperparameter, not {Kind}.");

        /// <summary>
        /// The dtype the baked value was built at — converted to the optimizer's declared dtype at rig
        /// build. Defined only for a <see cref="HyperparameterKind.Baked"/> hyperparameter.
        /// </summary>
        public DType BakedDType => BakedValue.DType;

        /// <summary>
        /// The shape a <see cref="HyperparameterKind.Runtime"/> hyperparameter's per-step values must
        /// have (empty for a scalar); throws for any other kind, whose shape comes from its constant or
        /// its scheduler graph instead.
        /// </summary>
        public IReadOnlyList<long> RuntimeShape => Kind == HyperparameterKind.Runtime
            ? _runtimeShape ?? Array.Empty<long>()
            : throw new InvalidOperationException(
                $"RuntimeShape is defined only for a Runtime hyperparameter, not {Kind}.");

        /// <summary>
        /// The built-in schedule driving this value, or <c>null</c> when it is a scheduler module or not
        /// a <see cref="HyperparameterKind.Scheduled"/> hyperparameter.
        /// </summary>
        public Schedule? AsSchedule => _schedule;

        /// <summary>
        /// The user scheduler module driving this value, or <c>null</c> when it is a built-in schedule or
        /// not a <see cref="HyperparameterKind.Scheduled"/> hyperparameter.
        /// </summary>
        public ComputationGraph? AsSchedulerModule => _schedulerModule;

        /// <summary>A plain float is a baked-in <see cref="Baked(float)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(float value) => Baked(value);
        /// <summary>A plain double is a baked-in <see cref="Baked(double)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(double value) => Baked(value);
        /// <summary>A plain int is a baked-in <see cref="Baked(int)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(int value) => Baked(value);
        /// <summary>A plain long is a baked-in <see cref="Baked(long)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(long value) => Baked(value);
        /// <summary>A plain bool is a baked-in <see cref="Baked(bool)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(bool value) => Baked(value);
        /// <summary>A <see cref="Schedule"/> becomes a <see cref="Scheduled(Schedule)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(Schedule schedule) => Scheduled(schedule);
    }

    /// <summary>
    /// Host values for hyperparameters: building a rank-0 <see cref="TensorData"/> from a boxed CLR
    /// primitive, reading a scalar back, and fitting a value to an optimizer's declared dtype (and, when
    /// the rig knows it, shape). The declaration is the source of truth, so every host-supplied value (a
    /// baked constant, a per-step <c>MakeHyperparameters</c> value) passes through <see cref="ConvertTo"/>,
    /// which fails loud rather than silently truncating or reshaping.
    /// </summary>
    internal static class HyperparameterValues
    {
        /// <summary>Builds a rank-0 <see cref="TensorData"/> at the boxed value's natural dtype.</summary>
        internal static TensorData Of(object value) => value switch
        {
            float v => Globals.TensorData([], v),
            double v => Globals.TensorData([], v),
            int v => Globals.TensorData([], v),
            long v => Globals.TensorData([], v),
            bool v => Globals.TensorData([], v),
            sbyte v => Globals.TensorData([], v),
            short v => Globals.TensorData([], v),
            byte v => Globals.TensorData([], v),
            ushort v => Globals.TensorData([], v),
            uint v => Globals.TensorData([], v),
            ulong v => Globals.TensorData([], v),
            Float16 v => Globals.TensorData([], v),
            BFloat16 v => Globals.TensorData([], v),
            TensorData v => v,
            _ => throw new ArgumentException(
                $"'{value?.GetType().Name ?? "null"}' is not a supported hyperparameter value type. Use a " +
                "numeric or bool value, or a TensorData of the declared dtype and shape."),
        };

        /// <summary>A <paramref name="shape"/>-shaped tensor of <paramref name="dtype"/>'s zero/false value.</summary>
        internal static TensorData Zero(DType dtype, IReadOnlyList<long> shape)
            => Globals.TensorDataWithDefaultVals(dtype, [.. shape]);

        /// <summary>The scalar's single element, boxed at its CLR storage type.</summary>
        internal static object Read(TensorData value)
        {
            var dtype = value.DType;
            if (dtype == DType.Float32) return value.As<float32>().AccessMemory<float>()[0];
            if (dtype == DType.Float64) return value.As<float64>().AccessMemory<double>()[0];
            if (dtype == DType.Float16) return value.As<float16>().AccessMemory<Float16>()[0];
            if (dtype == DType.BFloat16) return value.As<bfloat16>().AccessMemory<BFloat16>()[0];
            if (dtype == DType.Int8) return value.As<int8>().AccessMemory<sbyte>()[0];
            if (dtype == DType.Int16) return value.As<int16>().AccessMemory<short>()[0];
            if (dtype == DType.Int32) return value.As<int32>().AccessMemory<int>()[0];
            if (dtype == DType.Int64) return value.As<int64>().AccessMemory<long>()[0];
            if (dtype == DType.UInt8) return value.As<uint8>().AccessMemory<byte>()[0];
            if (dtype == DType.UInt16) return value.As<uint16>().AccessMemory<ushort>()[0];
            if (dtype == DType.UInt32) return value.As<uint32>().AccessMemory<uint>()[0];
            if (dtype == DType.UInt64) return value.As<uint64>().AccessMemory<ulong>()[0];
            if (dtype == DType.Bool) return value.As<bit>().AccessMemory<bool>()[0];
            throw new ArgumentException($"'{dtype}' is not a supported hyperparameter dtype.", nameof(value));
        }

        /// <summary>
        /// Fails loud when <paramref name="dtype"/> cannot carry a hyperparameter (a struct/string/complex
        /// or otherwise non-scalar-numeric type).
        /// </summary>
        internal static void AssertSupported(DType dtype, string name)
        {
            if (dtype == DType.Bool
                || dtype == DType.Float16 || dtype == DType.BFloat16
                || dtype == DType.Float32 || dtype == DType.Float64
                || dtype == DType.Int8 || dtype == DType.Int16 || dtype == DType.Int32 || dtype == DType.Int64
                || dtype == DType.UInt8 || dtype == DType.UInt16 || dtype == DType.UInt32 || dtype == DType.UInt64)
                return;
            throw new ArgumentException(
                $"Hyperparameter '{name}' is declared '{dtype}', which is not a supported hyperparameter " +
                "dtype. Declare it as a numeric or bool Scalar<T> / Vector<T> / Tensor<T>.", nameof(dtype));
        }

        /// <summary>
        /// Fits a host-supplied value to <paramref name="declared"/>, the optimizer's declared dtype. An
        /// exact-dtype value passes through unchanged, whatever its shape. A <b>scalar</b> at another
        /// dtype is converted, but only when the conversion round-trips exactly, so a truncating or
        /// out-of-range value (0.5 into an int32, 300 into an int8, a bool into a float) fails loud
        /// instead of silently changing the value. A <b>non-scalar</b> at another dtype is rejected
        /// rather than converted element-wise: it was built explicitly, so it can be built at the
        /// declared dtype.
        /// </summary>
        internal static TensorData ConvertTo(TensorData value, DType declared, string name)
        {
            AssertSupported(declared, name);
            if (value.DType == declared) return value;

            if (value.Shape.Dims.Length != 0)
                throw new ArgumentException(
                    $"Hyperparameter '{name}' is declared '{declared}' but was given a '{value.DType}' " +
                    "tensor. Only a scalar is converted between dtypes; build a non-scalar value at the " +
                    "declared dtype (Globals.TensorData(dtype, shape, …)).", nameof(value));

            if (value.DType == DType.Bool || declared == DType.Bool)
                throw new ArgumentException(
                    $"Hyperparameter '{name}' is declared '{declared}' but was given a '{value.DType}' " +
                    "value; a bool hyperparameter takes only bool values, and vice versa.", nameof(value));

            var host = Read(value);
            var converted = ConvertNumeric(host, declared, name, value.DType);
            if (!ConvertNumeric(converted, value.DType, name, declared).Equals(host))
                throw new ArgumentException(
                    $"Hyperparameter '{name}' is declared '{declared}', and the supplied '{value.DType}' " +
                    $"value {host} does not survive the conversion. Supply it at the declared dtype " +
                    "(e.g. Hyperparameter.Baked(Globals.TensorData([], …))).", nameof(value));
            return Of(converted);
        }

        /// <summary>
        /// Fails loud when <paramref name="value"/>'s shape is not <paramref name="expected"/> — the shape
        /// the rig compiled this hyperparameter at. The training step is built once, so a hyperparameter's
        /// shape is fixed from build; only its values vary per step.
        /// </summary>
        internal static void AssertShape(TensorData value, Shape expected, string name)
        {
            if (value.Shape.Dims.AsSpan().SequenceEqual(expected.Dims)) return;
            throw new ArgumentException(
                $"Hyperparameter '{name}' was built at shape [{string.Join(", ", expected.Dims)}], but the " +
                $"supplied value has shape [{string.Join(", ", value.Shape.Dims)}]. A hyperparameter's " +
                "shape is fixed when the rig is built; only its values vary per step.", nameof(value));
        }

        /// <summary>Converts a boxed numeric to <paramref name="target"/>'s CLR storage type.</summary>
        private static object ConvertNumeric(object host, DType target, string name, DType from)
        {
            try
            {
                if (target == DType.Float16) return (Float16)Convert.ToSingle(host, System.Globalization.CultureInfo.InvariantCulture);
                if (target == DType.BFloat16) return (BFloat16)Convert.ToSingle(host, System.Globalization.CultureInfo.InvariantCulture);
                if (host is Float16 h16) host = (float)h16;
                if (host is BFloat16 hb16) host = (float)hb16;
                return Convert.ChangeType(host, target.ToPrimitiveType(), System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception e) when (e is OverflowException or InvalidCastException or FormatException)
            {
                throw new ArgumentException(
                    $"Hyperparameter '{name}' is declared '{target}', and the supplied '{from}' value " +
                    $"{host} cannot be represented in it.", nameof(host), e);
            }
        }
    }

    /// <summary>
    /// A named, ordered set of optimizer hyperparameters. The source generator emits a strongly
    /// typed implementation (e.g. <c>AdamWOptimizerHyperparameters</c>) for every optimizer module
    /// whose hyperparameters are all scalars — of any supported dtype — giving named, defaulted,
    /// init-only properties of type <see cref="Hyperparameter"/>. Pass an instance to
    /// <see cref="TrainingRig.FromScratch(Shorokoo.Graph.ComputationGraph, Shorokoo.Graph.ComputationGraph, Shorokoo.Graph.ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, Shorokoo.RngConfig?, Shorokoo.Runtime.ComputeContext?, Shorokoo.Runtime.ComputeContext?)"/>.
    /// </summary>
    public interface IOptimizerHyperparameters
    {
        /// <summary>The hyperparameter sources in the optimizer's declared (<c>[Hyper]</c>) order.</summary>
        Hyperparameter[] InOptimizerOrder();

        /// <summary>The hyperparameter names in the same order, used for legible graph fields and named overrides.</summary>
        System.Collections.Generic.IReadOnlyList<string> HyperparameterNames { get; }
    }
}
