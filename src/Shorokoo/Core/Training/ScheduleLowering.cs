using System;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Training
{
    /// <summary>
    /// Structural description of a <see cref="Schedule"/> — one record per <see cref="Schedules"/>
    /// factory shape and per <see cref="Schedule"/> combinator. Factories and combinators attach
    /// the matching node to every schedule they build, so any schedule constructed through the
    /// public API carries its own definition; only a schedule wrapping an opaque host
    /// <c>Func&lt;int, float&gt;</c> has none. <see cref="ScheduleLowering"/> walks this tree to
    /// emit the equivalent graph math.
    ///
    /// Numeric parameters are stored exactly as the host evaluator uses them (already-folded
    /// float32 constants, e.g. <see cref="Warmup.Peak"/> or <see cref="OneCycle.Up"/>), so the
    /// lowered graph mirrors the host's float32 arithmetic op for op.
    /// </summary>
    internal abstract record ScheduleExpr
    {
        /// <summary><see cref="Schedules.Constant"/>.</summary>
        internal sealed record Constant(float Value) : ScheduleExpr;

        /// <summary><see cref="Schedules.Linear"/>.</summary>
        internal sealed record Linear(float BaseValue, float FinalValue, int TotalSteps) : ScheduleExpr;

        /// <summary><see cref="Schedules.Cosine"/>.</summary>
        internal sealed record Cosine(float BaseValue, int TotalSteps) : ScheduleExpr;

        /// <summary><see cref="Schedules.StepDecay"/>.</summary>
        internal sealed record StepDecay(float BaseValue, int StepSize, float Gamma) : ScheduleExpr;

        /// <summary><see cref="Schedules.Exponential"/>.</summary>
        internal sealed record Exponential(float BaseValue, float Gamma) : ScheduleExpr;

        /// <summary>
        /// <see cref="Schedules.OneCycle"/>, with the factory's derived constants
        /// (<c>initial</c>, <c>final</c>, <c>up</c>, <c>down</c>) pre-folded exactly as the host
        /// closure captures them.
        /// </summary>
        internal sealed record OneCycle(float Initial, float MaxValue, float Final, int Up, int Down) : ScheduleExpr;

        /// <summary><see cref="Schedule.Scale"/>.</summary>
        internal sealed record Scale(ScheduleExpr Inner, float Factor) : ScheduleExpr;

        /// <summary><see cref="Schedule.Clamp"/>.</summary>
        internal sealed record Clamp(ScheduleExpr Inner, float Min, float Max) : ScheduleExpr;

        /// <summary><see cref="Schedule.Shift"/>.</summary>
        internal sealed record Shift(ScheduleExpr Inner, int Steps) : ScheduleExpr;

        /// <summary><see cref="Schedule.PerEpoch"/>.</summary>
        internal sealed record PerEpoch(ScheduleExpr Inner, int StepsPerEpoch) : ScheduleExpr;

        /// <summary>
        /// <see cref="Schedule.WithWarmup"/>. <paramref name="Peak"/> is the inner schedule's
        /// step-0 value, captured at construction time exactly as the host closure does.
        /// </summary>
        internal sealed record Warmup(ScheduleExpr Inner, int WarmupSteps, float StartFactor, float Peak) : ScheduleExpr;

        /// <summary><see cref="Schedule.Then"/>.</summary>
        internal sealed record Then(ScheduleExpr First, int AtStep, ScheduleExpr Next) : ScheduleExpr;
    }

    /// <summary>
    /// Lowers a <see cref="Schedule"/> to Shorokoo graph math: a pure function from the int64
    /// step counter to the float32 hyperparameter value, built from elementary ops (arithmetic,
    /// <c>Cos</c>, <c>Pow</c>, <c>Clip</c>, comparison + <c>Where</c> for piecewise boundaries).
    ///
    /// The lowering mirrors the host evaluator's float32 arithmetic operation for operation —
    /// including the host's int → float32 counter conversion — so on an execution engine whose
    /// elementary float32 ops match .NET (<c>MathF</c>) semantics the graph value is bit-identical
    /// to <see cref="Schedule.At"/> at every step. Engines with different transcendental
    /// implementations (e.g. ONNX Runtime's <c>Cos</c>/<c>Pow</c>) may differ by a few ulps; see
    /// the schedule-lowering parity tests for the measured tolerance contract.
    ///
    /// Every schedule constructible from the <see cref="Schedules"/> factories and
    /// <see cref="Schedule"/> combinators can be lowered. The one exception is a schedule built
    /// directly from an opaque host function (<see cref="Schedule(Func{int, float})"/> or the
    /// implicit conversion), whose definition exists only as compiled code —
    /// <see cref="CanLower"/> reports <c>false</c> and <see cref="LowerToGraph(Schedule, Scalar{int64})"/>
    /// throws. <see cref="Schedule.PerEpoch"/> needs no separate epoch input: the epoch index is
    /// derived in-graph from the step counter (<c>step / stepsPerEpoch</c>, the combinator's
    /// defining contract), which keeps arbitrary compositions (e.g. a <c>PerEpoch</c> inside a
    /// re-based <see cref="Schedule.Then"/> branch) exact.
    /// </summary>
    public static class ScheduleLowering
    {
        /// <summary>
        /// Whether <paramref name="schedule"/> can be lowered to graph math: true for every
        /// schedule built from the <see cref="Schedules"/> factories and <see cref="Schedule"/>
        /// combinators, false when it (or any composed part) wraps an opaque host function.
        /// </summary>
        public static bool CanLower(this Schedule schedule)
            => (schedule ?? throw new ArgumentNullException(nameof(schedule))).Expr is not null;

        /// <summary>
        /// Emits graph math evaluating <paramref name="schedule"/> at the rank-0 int64
        /// <paramref name="step"/> counter, returning the float32 scheduled value.
        /// </summary>
        public static Scalar<float32> LowerToGraph(this Schedule schedule, Scalar<int64> step)
            => Emit(RequireExpr(schedule), step).Scalar();

        /// <summary>
        /// Batch variant of <see cref="LowerToGraph(Schedule, Scalar{int64})"/>: evaluates the
        /// schedule elementwise over a rank-1 tensor of step counters. The emitted math is the
        /// same elementwise graph; this overload exists so parity harnesses and inspection tools
        /// can evaluate many steps in one execution.
        /// </summary>
        public static Vector<float32> LowerToGraph(this Schedule schedule, Vector<int64> steps)
        {
            var value = Emit(RequireExpr(schedule), steps);
            // A step-independent schedule (e.g. Schedules.Constant) emits a rank-0 value;
            // broadcast it up to the probe shape. x + 0f is value-preserving in float32.
            if (value.Rank != 1)
                value = value + steps.Cast<float32>() * Scalar(0f);
            return value.Vec();
        }

        private static ScheduleExpr RequireExpr(Schedule schedule)
        {
            if (schedule is null) throw new ArgumentNullException(nameof(schedule));
            return schedule.Expr ?? throw new InvalidOperationException(
                "Schedule wraps an opaque host function and cannot be lowered to graph math; " +
                "only schedules built from the Schedules factories and Schedule combinators are " +
                "lowerable (check CanLower first).");
        }

        /// <summary>
        /// Emits the graph math for <paramref name="expr"/> at the (possibly re-based) step
        /// counter <paramref name="step"/>. All ops are elementwise, so the same emission serves
        /// the rank-0 counter contract and the rank-1 batch overload. Piecewise shapes evaluate
        /// both branches and select with <c>Where</c>; the unselected branch sees an
        /// out-of-domain counter (e.g. a negative re-based step), which every shape tolerates.
        /// </summary>
        private static Tensor<float32> Emit(ScheduleExpr expr, Tensor<int64> step)
        {
            switch (expr)
            {
                case ScheduleExpr.Constant c:
                    return Scalar(c.Value);

                // Host: baseValue + (finalValue - baseValue) * clamp((float)step / totalSteps, 0, 1)
                case ScheduleExpr.Linear l:
                    return Scalar(l.BaseValue) + Scalar(l.FinalValue - l.BaseValue) * Progress(step, l.TotalSteps);

                // Host: (0.5f * baseValue) * (1 + cos(pi * clamp((float)step / totalSteps, 0, 1)))
                case ScheduleExpr.Cosine c:
                    return Scalar(0.5f * c.BaseValue)
                        * (Scalar(1f) + (Scalar(MathF.PI) * Progress(step, c.TotalSteps)).Cos());

                // Host: baseValue * pow(gamma, step / stepSize) with integer division
                case ScheduleExpr.StepDecay d:
                {
                    Tensor<float32> gamma = Scalar(d.Gamma);
                    return Scalar(d.BaseValue) * gamma.Pow((step / Scalar((long)d.StepSize)).Cast<float32>());
                }

                // Host: baseValue * pow(gamma, (float)step)
                case ScheduleExpr.Exponential e:
                {
                    Tensor<float32> gamma = Scalar(e.Gamma);
                    return Scalar(e.BaseValue) * gamma.Pow(step.Cast<float32>());
                }

                // Host: step < up ? initial + (max-initial)*0.5f*(1 - cos(pi * (float)step/up))
                //                 : final + (max-final)*0.5f*(1 + cos(pi * clamp((float)(step-up)/down, 0, 1)))
                case ScheduleExpr.OneCycle o:
                {
                    var t = step.Cast<float32>() / Scalar((float)o.Up);
                    var rise = Scalar(o.Initial)
                        + Scalar((o.MaxValue - o.Initial) * 0.5f) * (Scalar(1f) - (Scalar(MathF.PI) * t).Cos());
                    var td = ((step - Scalar((long)o.Up)).Cast<float32>() / Scalar((float)o.Down))
                        .Clip(Scalar(0f), Scalar(1f));
                    var fall = Scalar(o.Final)
                        + Scalar((o.MaxValue - o.Final) * 0.5f) * (Scalar(1f) + (Scalar(MathF.PI) * td).Cos());
                    return (step < Scalar((long)o.Up)).Where(rise, fall);
                }

                case ScheduleExpr.Scale s:
                    return Emit(s.Inner, step) * Scalar(s.Factor);

                case ScheduleExpr.Clamp cl:
                    return Emit(cl.Inner, step).Clip(Scalar(cl.Min), Scalar(cl.Max));

                case ScheduleExpr.Shift sh:
                    return Emit(sh.Inner, step + Scalar((long)sh.Steps));

                case ScheduleExpr.PerEpoch p:
                    return Emit(p.Inner, step / Scalar((long)p.StepsPerEpoch));

                // Host: step < warmupSteps ? peak * (startFactor + (1-startFactor) * (step+1)/(float)warmupSteps)
                //                          : inner(step - warmupSteps)
                case ScheduleExpr.Warmup w:
                {
                    var t = (step + Scalar(1L)).Cast<float32>() / Scalar((float)w.WarmupSteps);
                    var ramp = Scalar(w.Peak) * (Scalar(w.StartFactor) + Scalar(1f - w.StartFactor) * t);
                    return (step < Scalar((long)w.WarmupSteps))
                        .Where(ramp, Emit(w.Inner, step - Scalar((long)w.WarmupSteps)));
                }

                case ScheduleExpr.Then t:
                    return (step < Scalar((long)t.AtStep))
                        .Where(Emit(t.First, step), Emit(t.Next, step - Scalar((long)t.AtStep)));

                default:
                    throw new NotSupportedException($"Unhandled schedule expression '{expr.GetType().Name}'.");
            }
        }

        /// <summary>The clamped progress ratio <c>clamp((float)step / totalSteps, 0, 1)</c>.</summary>
        private static Tensor<float32> Progress(Tensor<int64> step, int totalSteps)
            => (step.Cast<float32>() / Scalar((float)totalSteps)).Clip(Scalar(0f), Scalar(1f));
    }
}
