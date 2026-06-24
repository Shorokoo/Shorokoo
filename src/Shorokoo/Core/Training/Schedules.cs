using System;

namespace Shorokoo.Core.Training
{
    /// <summary>
    /// A <c>step → value</c> schedule for a scalar hyperparameter (typically the learning rate),
    /// with fluent combinators. A <see cref="Schedule"/> implicitly converts to/from
    /// <c>Func&lt;int, float&gt;</c> and to <see cref="HyperValue"/>, so it can be handed straight to a
    /// strongly-typed optimizer hyperparameter set:
    ///
    /// <code>
    /// var rig = TrainingRig.FromScratch(model, loss, AdamWOptimizer.ComputationGraph, sample,
    ///     new AdamWOptimizerHyperparameters
    ///     {
    ///         LearningRate = Schedules.Cosine(3e-4f, totalSteps).WithWarmup(warmupSteps),
    ///         WeightDecay  = 1e-4f, // a bare float is baked
    ///     });
    /// var compiled = ctx.Compile(rig.TrainingStepPureGraph); // compile once
    /// var ckpt = rig.CreateDefaultCheckpoint();
    /// for (int step = 0; step &lt; totalSteps; step++)
    ///     ckpt = rig.TrainStep(ckpt, inS, outS, compiled).Checkpoint; // schedule applied automatically
    /// </code>
    /// </summary>
    public sealed class Schedule
    {
        private readonly Func<int, float> _fn;

        /// <summary>Wraps a step → value function as a schedule.</summary>
        public Schedule(Func<int, float> fn) => _fn = fn ?? throw new ArgumentNullException(nameof(fn));

        /// <summary>The scheduled value at <paramref name="step"/> (0-based global training step).</summary>
        public float At(int step) => _fn(step);

        /// <summary>Wraps a step → value function as a schedule.</summary>
        public static implicit operator Schedule(Func<int, float> fn) => new(fn);
        /// <summary>Unwraps the underlying step → value function.</summary>
        public static implicit operator Func<int, float>(Schedule schedule) => schedule._fn;

        // --- Combinators ---------------------------------------------------------------

        /// <summary>Multiplies every value by <paramref name="factor"/>.</summary>
        public Schedule Scale(float factor) => new(s => _fn(s) * factor);

        /// <summary>Clamps every value to <c>[<paramref name="min"/>, <paramref name="max"/>]</c>.</summary>
        public Schedule Clamp(float min, float max) => new(s => Math.Clamp(_fn(s), min, max));

        /// <summary>Shifts the schedule earlier by <paramref name="steps"/> (value at step <c>s</c> becomes value at <c>s + steps</c>).</summary>
        public Schedule Shift(int steps) => new(s => _fn(s + steps));

        /// <summary>
        /// Reinterprets this schedule as epoch-based: the value is held constant within each epoch
        /// of <paramref name="stepsPerEpoch"/> steps (step <c>s</c> reads the schedule at epoch
        /// <c>s / stepsPerEpoch</c>).
        /// </summary>
        public Schedule PerEpoch(int stepsPerEpoch)
        {
            if (stepsPerEpoch < 1) throw new ArgumentOutOfRangeException(nameof(stepsPerEpoch));
            return new Schedule(s => _fn(s / stepsPerEpoch));
        }

        /// <summary>
        /// Prepends a linear warmup of <paramref name="warmupSteps"/> steps that ramps from
        /// <paramref name="startFactor"/>·peak up to this schedule's step-0 value (the peak), then
        /// continues with this schedule (re-based so it starts after the warmup).
        /// </summary>
        public Schedule WithWarmup(int warmupSteps, float startFactor = 0f)
        {
            if (warmupSteps < 0) throw new ArgumentOutOfRangeException(nameof(warmupSteps));
            if (warmupSteps == 0) return this;
            float peak = _fn(0);
            var inner = _fn;
            return new Schedule(s =>
            {
                if (s < warmupSteps)
                {
                    float t = (s + 1) / (float)warmupSteps;
                    return peak * (startFactor + (1f - startFactor) * t);
                }
                return inner(s - warmupSteps);
            });
        }

        /// <summary>
        /// Switches to <paramref name="next"/> at <paramref name="atStep"/> (re-based to start there),
        /// the analogue of Optax's <c>join_schedules</c> with a single boundary.
        /// </summary>
        public Schedule Then(int atStep, Schedule next)
        {
            if (next is null) throw new ArgumentNullException(nameof(next));
            var inner = _fn;
            return new Schedule(s => s < atStep ? inner(s) : next._fn(s - atStep));
        }
    }

    /// <summary>
    /// Factory for common scalar-hyperparameter <see cref="Schedule"/>s (the discoverable entry
    /// point; combinators live on <see cref="Schedule"/> itself). Each factory returns a
    /// <c>step → value</c> schedule that can be assigned directly to a <see cref="HyperValue"/>
    /// optimizer hyperparameter.
    /// </summary>
    public static class Schedules
    {
        /// <summary>Constant value (the trivial schedule); dynamic but unchanging.</summary>
        public static Schedule Constant(float value) => new(_ => value);

        /// <summary>Linear interpolation from <paramref name="baseValue"/> to <paramref name="finalValue"/> over <paramref name="totalSteps"/> steps (then held).</summary>
        public static Schedule Linear(float baseValue, float finalValue, int totalSteps)
        {
            if (totalSteps < 1) throw new ArgumentOutOfRangeException(nameof(totalSteps));
            return new Schedule(step =>
            {
                float prog = Math.Clamp((float)step / totalSteps, 0f, 1f);
                return baseValue + (finalValue - baseValue) * prog;
            });
        }

        /// <summary>Cosine decay from <paramref name="baseValue"/> to ~0 over <paramref name="totalSteps"/> steps.</summary>
        public static Schedule Cosine(float baseValue, int totalSteps)
        {
            if (totalSteps < 1) throw new ArgumentOutOfRangeException(nameof(totalSteps));
            return new Schedule(step =>
            {
                float prog = Math.Clamp((float)step / totalSteps, 0f, 1f);
                return 0.5f * baseValue * (1f + MathF.Cos(MathF.PI * prog));
            });
        }

        /// <summary>
        /// Linear warmup from 0 to <paramref name="baseValue"/> over <paramref name="warmupSteps"/> steps,
        /// then a cosine decay to ~0 over the remaining steps. Mirrors the cosine schedule used by the
        /// PyTorch ViT reference. Equivalent to <c>Cosine(baseValue, totalSteps - warmupSteps).WithWarmup(warmupSteps)</c>.
        /// </summary>
        public static Schedule CosineWithWarmup(float baseValue, int warmupSteps, int totalSteps)
        {
            if (totalSteps < 1) throw new ArgumentOutOfRangeException(nameof(totalSteps));
            int warm = Math.Max(0, warmupSteps);
            return Cosine(baseValue, Math.Max(1, totalSteps - warm)).WithWarmup(warm);
        }

        /// <summary>Step decay: multiply <paramref name="baseValue"/> by <paramref name="gamma"/> every <paramref name="stepSize"/> steps.</summary>
        public static Schedule StepDecay(float baseValue, int stepSize, float gamma)
        {
            if (stepSize < 1) throw new ArgumentOutOfRangeException(nameof(stepSize));
            return new Schedule(step => baseValue * MathF.Pow(gamma, step / stepSize));
        }

        /// <summary>Exponential decay: <c><paramref name="baseValue"/> · <paramref name="gamma"/>^step</c>.</summary>
        public static Schedule Exponential(float baseValue, float gamma)
            => new(step => baseValue * MathF.Pow(gamma, step));

        /// <summary>
        /// The 1cycle policy (Smith): cosine-anneal up from <c>maxValue / divFactor</c> to
        /// <paramref name="maxValue"/> over the first <paramref name="pctStart"/> of training, then
        /// cosine-anneal down to <c>(maxValue / divFactor) / finalDivFactor</c>. Mirrors PyTorch's
        /// <c>OneCycleLR</c> with <c>anneal_strategy='cos'</c>.
        /// </summary>
        public static Schedule OneCycle(
            float maxValue, int totalSteps,
            float pctStart = 0.3f, float divFactor = 25f, float finalDivFactor = 1e4f)
        {
            if (totalSteps < 1) throw new ArgumentOutOfRangeException(nameof(totalSteps));
            float initial = maxValue / divFactor;
            float final = initial / finalDivFactor;
            int up = Math.Max(1, (int)MathF.Round(totalSteps * Math.Clamp(pctStart, 0f, 1f)));
            int down = Math.Max(1, totalSteps - up);
            return new Schedule(step =>
            {
                if (step < up)
                {
                    float t = (float)step / up;
                    return initial + (maxValue - initial) * 0.5f * (1f - MathF.Cos(MathF.PI * t));
                }
                float td = Math.Clamp((float)(step - up) / down, 0f, 1f);
                return final + (maxValue - final) * 0.5f * (1f + MathF.Cos(MathF.PI * td));
            });
        }
    }
}
