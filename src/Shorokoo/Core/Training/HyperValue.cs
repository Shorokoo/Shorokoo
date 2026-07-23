using System;
using Shorokoo.Core.Training;
using Shorokoo.Graph;

namespace Shorokoo
{
    /// <summary>
    /// The value of one optimizer hyperparameter, as supplied when building a
    /// <see cref="TrainingRig"/>. Its <i>kind</i> — not a separate boolean flag — decides how the
    /// value is wired into the compiled training-step graph:
    ///
    /// <list type="bullet">
    /// <item><description>
    /// A bare <see cref="float"/> (implicitly converted) is <b>baked</b> as a graph
    /// <c>Constant</c>. Changing it requires rebuilding the rig. Use this for fixed
    /// hyperparameters (e.g. AdamW's betas).
    /// </description></item>
    /// <item><description>
    /// A built-in <see cref="Schedule"/> (implicitly converted) is <b>scheduled in-graph</b>: the
    /// rig lowers it to graph math (<c>ScheduleLowering</c>) and inlines it into the training-step
    /// graph as a pure function of the int64 step counter, so the value is recomputed on the engine
    /// each step with no host evaluation. The graph compiles once; the schedule is a durable graph
    /// constituent that resumes from a checkpoint's step alone.
    /// </description></item>
    /// <item><description>
    /// A <b>scheduler module</b> (via <see cref="Scheduled(ComputationGraph)"/>) covers anything the
    /// built-ins don't: a Shorokoo module graph taking the int64 scalar step counter as input and
    /// producing the scheduled float32 scalar value, inlined into the training-step graph just like
    /// a built-in schedule. Its signature is validated at rig build.
    /// </description></item>
    /// <item><description>
    /// <see cref="Runtime"/> marks the hyperparameter dynamic but <i>schedule-less</i>: the rig
    /// has no schedule for it, so its value must be supplied explicitly each step via
    /// <see cref="TrainingRig.MakeHyperparams(float)"/> and the override
    /// <c>TrainStep</c> overload. Useful for manual control and tests.
    /// </description></item>
    /// </list>
    ///
    /// This mirrors how Keras (<c>Adam(learning_rate=schedule)</c>) and Optax
    /// (<c>adamw(learning_rate=schedule)</c>) let the value's type decide constant-vs-scheduled.
    /// There is deliberately no API that accepts an arbitrary host lambda: a compiled closure has
    /// no durable graph representation, so schedules come only from the built-in factories/combinators
    /// or a scheduler module.
    /// </summary>
    public readonly struct HyperValue
    {
        private readonly float _value;
        private readonly Schedule? _schedule;
        private readonly ComputationGraph? _schedulerModule;
        private readonly bool _isDynamic;

        private HyperValue(float value, Schedule? schedule, ComputationGraph? schedulerModule, bool isDynamic)
        {
            _value = value;
            _schedule = schedule;
            _schedulerModule = schedulerModule;
            _isDynamic = isDynamic;
        }

        /// <summary>A fixed value baked into the graph as a constant.</summary>
        public static HyperValue Constant(float value) => new(value, null, null, isDynamic: false);

        /// <summary>
        /// A built-in <see cref="Schedule"/>, lowered to graph math and evaluated in-graph from the
        /// step counter each training step.
        /// </summary>
        public static HyperValue Scheduled(Schedule schedule)
        {
            if (schedule is null) throw new ArgumentNullException(nameof(schedule));
            return new HyperValue(schedule.At(0), schedule, null, isDynamic: true);
        }

        /// <summary>
        /// A user-supplied scheduler <b>module</b> — a Shorokoo module graph taking the int64 scalar
        /// step counter as its single input and producing the scheduled float32 scalar value. Inlined
        /// into the training-step graph as a constituent; its signature is validated at rig build.
        /// Use this for schedules the built-in <see cref="Schedules"/> factories don't cover.
        /// </summary>
        public static HyperValue Scheduled(ComputationGraph schedulerModule)
        {
            if (schedulerModule is null) throw new ArgumentNullException(nameof(schedulerModule));
            return new HyperValue(0f, null, schedulerModule, isDynamic: true);
        }

        /// <summary>
        /// A dynamic value with no schedule: the rig routes it as a runtime input but you must
        /// supply it explicitly each step (see <see cref="TrainingRig.MakeHyperparams(float)"/>).
        /// <paramref name="seed"/> is used only to seed shape inference / graph optimization.
        /// </summary>
        public static HyperValue Runtime(float seed = 0f) => new(seed, null, null, isDynamic: true);

        /// <summary>Whether this hyperparameter flows as a runtime/in-graph value rather than a baked constant.</summary>
        public bool IsDynamic => _isDynamic;

        /// <summary>The built-in schedule driving this value, or <c>null</c> when baked, a scheduler module, or schedule-less runtime.</summary>
        public Schedule? AsSchedule => _schedule;

        /// <summary>The user scheduler module driving this value, or <c>null</c> when it is not a scheduler-module hyperparameter.</summary>
        public ComputationGraph? AsSchedulerModule => _schedulerModule;

        /// <summary>
        /// The value used to seed shape inference / graph optimization (and the baked constant
        /// when not dynamic): the built-in schedule's step-0 value when scheduled, else the stored
        /// value. For a scheduler module the step-0 value is only known by executing the module, so
        /// this returns the stored seed (0); the rig feeds the real step counter for the in-graph value.
        /// </summary>
        public float InitialValue => _schedule is not null ? _schedule.At(0) : _value;

        /// <summary>A plain float is a baked-in <see cref="Constant"/> hyperparameter.</summary>
        public static implicit operator HyperValue(float value) => Constant(value);
        /// <summary>A <see cref="Schedule"/> becomes a <see cref="Scheduled(Schedule)"/> hyperparameter.</summary>
        public static implicit operator HyperValue(Schedule schedule) => Scheduled(schedule);
    }

    /// <summary>
    /// A named, ordered set of optimizer hyperparameters. The source generator emits a strongly
    /// typed implementation (e.g. <c>AdamWOptimizerHyperparameters</c>) for every optimizer module
    /// whose hyperparameters are all scalar <c>float32</c>, giving named, defaulted, init-only
    /// properties of type <see cref="HyperValue"/>. Pass an instance to
    /// <see cref="TrainingRig.FromScratch(Shorokoo.Graph.ComputationGraph, Shorokoo.Graph.ComputationGraph, Shorokoo.Graph.ComputationGraph, NamedModelParam[], IOptimizerHyperparameters, Shorokoo.RngConfig?)"/>.
    /// </summary>
    public interface IOptimizerHyperparameters
    {
        /// <summary>The hyperparameter values in the optimizer's declared (<c>[Hyper]</c>) order.</summary>
        HyperValue[] InOptimizerOrder();

        /// <summary>The hyperparameter names in the same order, used for legible graph fields and named overrides.</summary>
        System.Collections.Generic.IReadOnlyList<string> HyperparameterNames { get; }
    }
}
