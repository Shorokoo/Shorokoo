using System;

namespace Shorokoo.Core.Training
{
    /// <summary>
    /// A <c>step → value</c> schedule for a scalar hyperparameter (typically the learning rate),
    /// with fluent combinators. Build one only through the <see cref="Schedules"/> factories and the
    /// combinators on this type — there is no public constructor from an arbitrary host lambda,
    /// because every schedule the rig accepts must lower to a durable graph (see
    /// <see cref="ScheduleLowering"/>). A <see cref="Schedule"/> implicitly converts to
    /// <see cref="Hyperparameter"/>, so it can be handed straight to a strongly-typed optimizer
    /// hyperparameter set:
    ///
    /// <code>
    /// var rig = TrainingRig.FromScratch(model, loss, AdamWOptimizer.ComputationGraph, sample,
    ///     new AdamWOptimizerHyperparameters
    ///     {
    ///         LearningRate = Schedules.Cosine(3e-4f, totalSteps).WithWarmup(warmupSteps),
    ///         WeightDecay  = 1e-4f, // a bare float is baked
    ///     });
    /// var ckpt = rig.CreateInitialCheckpoint();
    /// for (int step = 0; step &lt; totalSteps; step++)
    ///     ckpt = rig.TrainStep(ckpt, inS, outS); // compiled once internally; schedule applied automatically
    /// </code>
    /// </summary>
    public sealed class Schedule
    {
        /// <summary>
        /// The single structural truth of this schedule: the <see cref="ScheduleExpr"/> tree the
        /// factories and combinators build, from which both host evaluation
        /// (<see cref="ScheduleInterpreter"/>, via <see cref="At(long)"/>) and graph lowering
        /// (<see cref="ScheduleLowering"/>) derive — one description, two consumers. It is
        /// <c>null</c> only for an opaque, non-lowerable schedule, which cannot be built through the
        /// public API and is used solely defensively (e.g. in lowering tests).
        /// </summary>
        internal ScheduleExpr? Expr { get; }

        /// <summary>
        /// Builds a schedule from its structural <paramref name="expr"/> (the single graph-lowerable
        /// and host-interpretable description). Internal because every public schedule must carry a
        /// lowerable <paramref name="expr"/>; the factories and combinators are the only callers. A
        /// <c>null</c> <paramref name="expr"/> marks an opaque, non-lowerable schedule — used only
        /// defensively (e.g. in lowering tests), where <see cref="At(long)"/> throws.
        /// </summary>
        internal Schedule(ScheduleExpr? expr)
        {
            Expr = expr;
        }

        /// <summary>
        /// The scheduled value at <paramref name="step"/> (0-based global training step), evaluated by
        /// the single host interpreter (<see cref="ScheduleInterpreter"/>) over this schedule's
        /// <see cref="Expr"/> — the pinned mirror of the graph lowering. Throws for an opaque schedule
        /// with no structural description.
        /// </summary>
        public float At(long step)
            => ScheduleInterpreter.Evaluate(
                Expr ?? throw new InvalidOperationException(
                    "Schedule has no structural description and cannot be evaluated; only schedules " +
                    "built from the Schedules factories and Schedule combinators are evaluable."),
                step);

        // --- Combinators ---------------------------------------------------------------

        /// <summary>
        /// The derived schedule's structural record: <paramref name="make"/> applied to this
        /// schedule's <see cref="Expr"/>, or <c>null</c> when this schedule is opaque.
        /// Opaqueness always propagates through combinators, so the guard lives here rather
        /// than being repeated at each combinator.
        /// </summary>
        private ScheduleExpr? Derive(Func<ScheduleExpr, ScheduleExpr> make)
            => Expr is null ? null : make(Expr);

        /// <summary>Multiplies every value by <paramref name="factor"/>.</summary>
        public Schedule Scale(float factor) => new(Derive(e => new ScheduleExpr.Scale(e, factor)));

        /// <summary>
        /// Clamps every value to <c>[<paramref name="min"/>, <paramref name="max"/>]</c>;
        /// <paramref name="min"/> must not exceed <paramref name="max"/>.
        /// </summary>
        public Schedule Clamp(float min, float max)
        {
            // Rejected eagerly: deferred to Math.Clamp it would throw on every At call, while
            // the lowered graph's Clip (numpy-style, max wins) would silently produce values
            // the host contract rejects.
            if (min > max)
                throw new ArgumentException($"min ({min}) must not be greater than max ({max}).", nameof(min));
            return new Schedule(Derive(e => new ScheduleExpr.Clamp(e, min, max)));
        }

        /// <summary>Shifts the schedule earlier by <paramref name="steps"/> (value at step <c>s</c> becomes value at <c>s + steps</c>).</summary>
        public Schedule Shift(int steps) => new(Derive(e => new ScheduleExpr.Shift(e, steps)));

        /// <summary>
        /// Reinterprets this schedule as epoch-based: the value is held constant within each epoch
        /// of <paramref name="stepsPerEpoch"/> steps (step <c>s</c> reads the schedule at epoch
        /// <c>s / stepsPerEpoch</c>).
        /// </summary>
        public Schedule PerEpoch(int stepsPerEpoch)
        {
            if (stepsPerEpoch < 1) throw new ArgumentOutOfRangeException(nameof(stepsPerEpoch));
            return new Schedule(Derive(e => new ScheduleExpr.PerEpoch(e, stepsPerEpoch)));
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
            // The peak is the inner schedule's step-0 value, captured now (as a folded constant) via
            // the same interpreter At uses; opaque schedules propagate opaqueness without evaluating.
            if (Expr is null) return new Schedule((ScheduleExpr?)null);
            float peak = At(0);
            return new Schedule(new ScheduleExpr.Warmup(Expr, warmupSteps, startFactor, peak));
        }

        /// <summary>
        /// Switches to <paramref name="next"/> at <paramref name="atStep"/> (re-based to start there),
        /// the analogue of Optax's <c>join_schedules</c> with a single boundary.
        /// </summary>
        public Schedule Then(int atStep, Schedule next)
        {
            if (next is null) throw new ArgumentNullException(nameof(next));
            return new Schedule(
                next.Expr is { } nextExpr ? Derive(e => new ScheduleExpr.Then(e, atStep, nextExpr)) : null);
        }
    }

    /// <summary>
    /// Factory for common scalar-hyperparameter <see cref="Schedule"/>s (the discoverable entry
    /// point; combinators live on <see cref="Schedule"/> itself). Each factory returns a
    /// <c>step → value</c> schedule that can be assigned directly to a <see cref="Hyperparameter"/>
    /// optimizer hyperparameter.
    /// </summary>
    public static class Schedules
    {
        /// <summary>Constant value (the trivial schedule); dynamic but unchanging.</summary>
        public static Schedule Constant(float value) => new(new ScheduleExpr.Constant(value));

        /// <summary>Linear interpolation from <paramref name="baseValue"/> to <paramref name="finalValue"/> over <paramref name="totalSteps"/> steps (then held).</summary>
        public static Schedule Linear(float baseValue, float finalValue, int totalSteps)
        {
            if (totalSteps < 1) throw new ArgumentOutOfRangeException(nameof(totalSteps));
            return new Schedule(new ScheduleExpr.Linear(baseValue, finalValue, totalSteps));
        }

        /// <summary>Cosine decay from <paramref name="baseValue"/> to ~0 over <paramref name="totalSteps"/> steps.</summary>
        public static Schedule Cosine(float baseValue, int totalSteps)
        {
            if (totalSteps < 1) throw new ArgumentOutOfRangeException(nameof(totalSteps));
            return new Schedule(new ScheduleExpr.Cosine(baseValue, totalSteps));
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
            return new Schedule(new ScheduleExpr.StepDecay(baseValue, stepSize, gamma));
        }

        /// <summary>Exponential decay: <c><paramref name="baseValue"/> · <paramref name="gamma"/>^step</c>.</summary>
        public static Schedule Exponential(float baseValue, float gamma)
            => new(new ScheduleExpr.Exponential(baseValue, gamma));

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
            return new Schedule(new ScheduleExpr.OneCycle(initial, maxValue, final, up, down));
        }
    }
}
