using System;
using Shorokoo.Core.Training;

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
    /// A <see cref="Schedule"/> (implicitly converted) is <b>scheduled</b>: it flows as a runtime
    /// input and the rig evaluates it at the checkpoint's step on every
    /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct, Shorokoo.Runtime.CompiledGraph)"/>.
    /// The graph compiles once; the value can change every step.
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
    /// </summary>
    public readonly struct HyperValue
    {
        private readonly float _value;
        private readonly Schedule? _schedule;
        private readonly bool _isDynamic;

        private HyperValue(float value, Schedule? schedule, bool isDynamic)
        {
            _value = value;
            _schedule = schedule;
            _isDynamic = isDynamic;
        }

        /// <summary>A fixed value baked into the graph as a constant.</summary>
        public static HyperValue Constant(float value) => new(value, null, isDynamic: false);

        /// <summary>A scheduled value evaluated at the current step on every training step.</summary>
        public static HyperValue Scheduled(Schedule schedule)
        {
            if (schedule is null) throw new ArgumentNullException(nameof(schedule));
            return new HyperValue(schedule.At(0), schedule, isDynamic: true);
        }

        /// <summary>
        /// A dynamic value with no schedule: the rig routes it as a runtime input but you must
        /// supply it explicitly each step (see <see cref="TrainingRig.MakeHyperparams(float)"/>).
        /// <paramref name="seed"/> is used only to seed shape inference / graph optimization.
        /// </summary>
        public static HyperValue Runtime(float seed = 0f) => new(seed, null, isDynamic: true);

        /// <summary>Whether this hyperparameter flows as a runtime input (scheduled or manual) rather than a baked constant.</summary>
        public bool IsDynamic => _isDynamic;

        /// <summary>The schedule driving this value, or <c>null</c> when baked or schedule-less runtime.</summary>
        public Schedule? AsSchedule => _schedule;

        /// <summary>
        /// The value used to seed shape inference / graph optimization (and the baked constant
        /// when not dynamic): the schedule's step-0 value when scheduled, else the stored value.
        /// </summary>
        public float InitialValue => _schedule is not null ? _schedule.At(0) : _value;

        /// <summary>A plain float is a baked-in <see cref="Constant"/> hyperparameter.</summary>
        public static implicit operator HyperValue(float value) => Constant(value);
        /// <summary>A <see cref="Schedule"/> becomes a <see cref="Scheduled"/> hyperparameter.</summary>
        public static implicit operator HyperValue(Schedule schedule) => Scheduled(schedule);
        /// <summary>A step → value function becomes a <see cref="Scheduled"/> hyperparameter.</summary>
        public static implicit operator HyperValue(Func<int, float> schedule) => Scheduled(new Schedule(schedule));
    }

    /// <summary>
    /// A named, ordered set of optimizer hyperparameters. The source generator emits a strongly
    /// typed implementation (e.g. <c>AdamWOptimizerHyperparameters</c>) for every optimizer module
    /// whose hyperparameters are all scalar <c>float32</c>, giving named, defaulted, init-only
    /// properties of type <see cref="HyperValue"/>. Pass an instance to
    /// <see cref="TrainingRig.FromScratch(Shorokoo.Graph.InternalComputationGraph, Shorokoo.Graph.InternalComputationGraph, Shorokoo.Graph.InternalComputationGraph, NamedModelParam[], IOptimizerHyperparameters, Shorokoo.RngConfig?)"/>.
    /// </summary>
    public interface IOptimizerHyperparameters
    {
        /// <summary>The hyperparameter values in the optimizer's declared (<c>[Hyper]</c>) order.</summary>
        HyperValue[] InOptimizerOrder();

        /// <summary>The hyperparameter names in the same order, used for legible graph fields and named overrides.</summary>
        System.Collections.Generic.IReadOnlyList<string> HyperparameterNames { get; }
    }
}
