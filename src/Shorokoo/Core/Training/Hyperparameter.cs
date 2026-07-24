using System;
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
    /// <b><see cref="Baked(float)"/></b> — a fixed value compiled into the training-step graph as a
    /// <c>Constant</c>. Changing it requires rebuilding the rig. Use this for fixed hyperparameters
    /// (e.g. AdamW's betas). A bare <see cref="float"/> converts implicitly to <c>Baked</c>.
    /// </description></item>
    /// <item><description>
    /// <b><see cref="Scheduled(Schedule)"/></b> / <b><see cref="Scheduled(ComputationGraph)"/></b> —
    /// driven in-graph by a <i>scheduler source</i>: a built-in <see cref="Schedule"/> (implicitly
    /// converted; lowered to graph math by <c>ScheduleLowering</c>) or a user scheduler <b>module</b>
    /// (a Shorokoo module taking the int64 counter input(s) and producing the scheduled float32 scalar).
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
    /// Nothing on this type answers "what is the value" — values come only from evaluating the source's
    /// canonical graph representation (in-graph per step; QEE at build for optimizer state init). This
    /// mirrors how Keras (<c>Adam(learning_rate=schedule)</c>) and Optax
    /// (<c>adamw(learning_rate=schedule)</c>) let the value's type decide constant-vs-scheduled. There is
    /// deliberately no API that accepts an arbitrary host lambda: a compiled closure has no durable graph
    /// representation, so schedules come only from the built-in factories/combinators or a scheduler module.
    /// </summary>
    public readonly struct Hyperparameter
    {
        private readonly float _bakedValue;
        private readonly Schedule? _schedule;
        private readonly ComputationGraph? _schedulerModule;

        /// <summary>Which of the three sources drives this hyperparameter (exhaustive, no invalid states).</summary>
        public HyperparameterKind Kind { get; }

        private Hyperparameter(HyperparameterKind kind, float bakedValue, Schedule? schedule, ComputationGraph? schedulerModule)
        {
            Kind = kind;
            _bakedValue = bakedValue;
            _schedule = schedule;
            _schedulerModule = schedulerModule;
        }

        /// <summary>A fixed value baked into the graph as a constant.</summary>
        public static Hyperparameter Baked(float value) => new(HyperparameterKind.Baked, value, null, null);

        /// <summary>
        /// A built-in <see cref="Schedule"/>, lowered to graph math and evaluated in-graph from the
        /// counter input(s) each training step.
        /// </summary>
        public static Hyperparameter Scheduled(Schedule schedule)
            => new(HyperparameterKind.Scheduled, 0f,
                schedule ?? throw new ArgumentNullException(nameof(schedule)), null);

        /// <summary>
        /// A user-supplied scheduler <b>module</b> — a Shorokoo module graph taking the int64 scalar
        /// counter input(s) (named from <c>step</c>, <c>epoch</c>, <c>batchIndex</c>) and producing the
        /// scheduled float32 scalar value. Inlined into the training-step graph as a constituent; its
        /// signature and purity are validated at rig build. Use this for schedules the built-in
        /// <see cref="Schedules"/> factories don't cover.
        /// </summary>
        public static Hyperparameter Scheduled(ComputationGraph schedulerModule)
            => new(HyperparameterKind.Scheduled, 0f, null,
                schedulerModule ?? throw new ArgumentNullException(nameof(schedulerModule)));

        /// <summary>
        /// A dynamic value with no schedule: the rig routes it as a runtime input and you must supply
        /// it explicitly each step (see <see cref="TrainingRig.MakeHyperparameters(float)"/>). Seedless:
        /// the shape-inference placeholder is internal, never user-visible.
        /// </summary>
        public static Hyperparameter Runtime() => new(HyperparameterKind.Runtime, 0f, null, null);

        /// <summary>
        /// The baked constant value. Defined only for a <see cref="HyperparameterKind.Baked"/>
        /// hyperparameter; throws otherwise (the value of a scheduled/runtime hyperparameter comes from
        /// evaluating its graph, never from this type).
        /// </summary>
        public float BakedValue => Kind == HyperparameterKind.Baked
            ? _bakedValue
            : throw new InvalidOperationException(
                $"BakedValue is defined only for a Baked hyperparameter, not {Kind}.");

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
        /// <summary>A <see cref="Schedule"/> becomes a <see cref="Scheduled(Schedule)"/> hyperparameter.</summary>
        public static implicit operator Hyperparameter(Schedule schedule) => Scheduled(schedule);
    }

    /// <summary>
    /// A named, ordered set of optimizer hyperparameters. The source generator emits a strongly
    /// typed implementation (e.g. <c>AdamWOptimizerHyperparameters</c>) for every optimizer module
    /// whose hyperparameters are all scalar <c>float32</c>, giving named, defaulted, init-only
    /// properties of type <see cref="Hyperparameter"/>. Pass an instance to
    /// <see cref="TrainingRig.FromScratch(Shorokoo.Graph.ComputationGraph, Shorokoo.Graph.ComputationGraph, Shorokoo.Graph.ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, Shorokoo.RngConfig?)"/>.
    /// </summary>
    public interface IOptimizerHyperparameters
    {
        /// <summary>The hyperparameter sources in the optimizer's declared (<c>[Hyper]</c>) order.</summary>
        Hyperparameter[] InOptimizerOrder();

        /// <summary>The hyperparameter names in the same order, used for legible graph fields and named overrides.</summary>
        System.Collections.Generic.IReadOnlyList<string> HyperparameterNames { get; }
    }
}
