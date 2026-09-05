using System;
using System.Diagnostics;
using Shorokoo.Runtime;

namespace Shorokoo.Graph
{
    /// <summary>
    /// The top-level phase a build report belongs to. Which phases a build reports follows from what
    /// it does: a <c>ToConcreteArchitecture</c> call reports <see cref="Concretize"/> only; a
    /// <c>TrainingRig.FromScratch</c> runs all three in order; a rig derivation (<c>With…</c>) or a
    /// <c>TrainingRig.Load</c> reuses its concrete architecture and so opens at
    /// <see cref="TrainingStep"/>. Each build's last report is <c>Done</c>, in whichever phase it ends.
    /// </summary>
    public enum BuildPhase
    {
        /// <summary>Lowering the module graph to a concrete architecture (<c>ToConcreteArchitecture</c>).</summary>
        Concretize,

        /// <summary>Composing the concrete architecture with the loss, autograd and the optimizer into
        /// the training-step graph, then lowering it to an executable graph.</summary>
        TrainingStep,

        /// <summary>Running the parameter / optimizer-state initializers, then shape inference and the
        /// memory-aware graph optimization of the training-step graph.</summary>
        Initialize,
    }

    /// <summary>
    /// One progress report from a build, raised as the build <b>enters</b> the named stage — so a
    /// build that has been quiet for minutes is stuck in the stage its last report named. The one
    /// exception is the terminal <c>Done</c>, the last report of a <em>completed</em> build, which
    /// names no stage: a stream sitting on <c>Done</c> is finished, not stuck. Attach a sink with
    /// <see cref="ComputeContext.Progress"/>:
    ///
    /// <code>
    /// var buildContext = new ComputeContext { Progress = new SynchronousBuildProgress(Console.WriteLine) };
    /// var rig = TrainingRig.FromScratch(model, loss, optimizer, sampleInputs, hyperparameters,
    ///                                   mergeContext: buildContext);
    /// </code>
    /// </summary>
    /// <param name="Phase">The build phase the stage belongs to.</param>
    /// <param name="Stage">The lowering/build stage being entered, named after the pass that runs it.</param>
    /// <param name="Elapsed">Time since the start of the build this report belongs to.</param>
    public readonly record struct BuildProgress(BuildPhase Phase, string Stage, TimeSpan Elapsed)
    {
        /// <summary>A one-line rendering — <c>[  12.3s] Concretize: InlineModulesAndFunctions</c>.</summary>
        public override string ToString() => $"[{Elapsed.TotalSeconds,6:F1}s] {Phase}: {Stage}";
    }

    /// <summary>
    /// An <see cref="IProgress{T}"/> that invokes its handler <b>synchronously</b>, on the thread
    /// doing the build. Prefer it over <see cref="Progress{T}"/> for console logging:
    /// <see cref="Progress{T}"/> posts to the captured synchronization context, so its reports can
    /// arrive out of order — or after the build returns — which is exactly what a liveness signal
    /// must not do. The handler runs inline, so keep it short; a slow handler slows the build.
    ///
    /// <para>Inline also means an exception from the handler is <b>not</b> contained: it propagates
    /// out of the build that raised it, discarding that build's work (nothing outside it is left
    /// inconsistent — the build owns everything it has touched). Reporting is never retried or
    /// suppressed on your behalf, so a handler that can fail — a writer over a filling disk, a
    /// pipe whose reader has exited — must catch its own faults if a lost log line should not cost
    /// a build.</para>
    /// </summary>
    public sealed class SynchronousBuildProgress : IProgress<BuildProgress>
    {
        private readonly Action<BuildProgress> _handler;

        /// <summary>Creates a sink that calls <paramref name="handler"/> on each report.</summary>
        public SynchronousBuildProgress(Action<BuildProgress> handler)
            => _handler = handler ?? throw new ArgumentNullException(nameof(handler));

        /// <inheritdoc/>
        public void Report(BuildProgress value) => _handler(value);
    }

    /// <summary>
    /// The build-internal half of the progress facility: holds the sink and the clock whose elapsed
    /// time every report of one build carries. Created once per build entry point and threaded down,
    /// so the whole of a <c>FromScratch</c> — concretization, training-step composition,
    /// initialization — reports against a single clock. <see cref="For"/> returns <c>null</c> when no
    /// sink is attached, making every reporting call site a null check.
    /// </summary>
    internal sealed class BuildProgressReporter
    {
        private readonly IProgress<BuildProgress> _sink;
        private readonly Stopwatch _clock = Stopwatch.StartNew();

        private BuildProgressReporter(IProgress<BuildProgress> sink) => _sink = sink;

        internal static BuildProgressReporter? For(ComputeContext? context)
            => context?.Progress is { } sink ? new BuildProgressReporter(sink) : null;

        internal void Report(BuildPhase phase, string stage)
            => _sink.Report(new BuildProgress(phase, stage, _clock.Elapsed));
    }
}
