using System;
using System.Diagnostics;
using Shorokoo.Runtime;

namespace Shorokoo.Graph
{
    /// <summary>
    /// The top-level phase a build report belongs to. A <c>ToConcreteArchitecture</c> call reports
    /// <see cref="Concretize"/> only; a <c>TrainingRig.FromScratch</c> build runs all three in order.
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
    /// build that has been quiet for minutes is stuck in the stage its last report named. Attach a
    /// sink with <see cref="ComputeContext.Progress"/>:
    ///
    /// <code>
    /// var buildContext = new ComputeContext { Progress = new BuildProgressHandler(Console.WriteLine) };
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
    /// </summary>
    public sealed class BuildProgressHandler : IProgress<BuildProgress>
    {
        private readonly Action<BuildProgress> _handler;

        /// <summary>Creates a sink that calls <paramref name="handler"/> on each report.</summary>
        public BuildProgressHandler(Action<BuildProgress> handler)
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
