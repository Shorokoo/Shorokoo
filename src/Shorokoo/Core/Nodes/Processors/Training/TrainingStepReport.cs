using System;

namespace Shorokoo
{
    /// <summary>Why a <see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?, Action{TrainingStepReport}?, System.Threading.CancellationToken)"/>
    /// or <see cref="TrainingRig.Train"/> run ended.</summary>
    public enum TrainingStopReason
    {
        /// <summary>Every epoch asked for was trained.</summary>
        Completed,

        /// <summary>The run's cancellation token was cancelled, and the run stopped between two
        /// steps.</summary>
        Cancelled,

        /// <summary>The step callback called <see cref="TrainingStepReport.RequestStop"/> before the
        /// last step, and the run stopped after that step.</summary>
        StopRequested,
    }

    /// <summary>
    /// What a <c>Fit</c> or <c>Train</c> run throws when its step callback throws: the callback's
    /// exception is the <see cref="Exception.InnerException"/>, and <see cref="Checkpoint"/> is the
    /// run's state after the last step it took — the step the callback was reporting — so the
    /// training done is not lost with the run. Pass it back as the initial checkpoint to resume
    /// exactly where the run stopped, as from a run stopped with
    /// <see cref="TrainingStepReport.RequestStop"/>.
    /// </summary>
    public sealed class TrainingCallbackException : Exception
    {
        internal TrainingCallbackException(TrainingCheckpoint checkpoint, Exception thrown)
            : base($"The training step callback threw after step {checkpoint.Step - 1}; the run stopped "
                   + "there. The state after that step is this exception's Checkpoint: resume from it.", thrown)
            => Checkpoint = checkpoint;

        /// <summary>The run's state after the step the callback was reporting, whole and the
        /// caller's, as <see cref="TrainingStepReport.TakeCheckpoint"/> hands it over.</summary>
        public TrainingCheckpoint Checkpoint { get; }
    }

    /// <summary>
    /// What one step of a <c>Fit</c> or <c>Train</c> run did, handed to the run's step callback
    /// right after the step — in order, one per step, on the thread running the loop, before the
    /// next step starts.
    ///
    /// <para>Everything here is already on the host: the step's loss and applied hyperparameters
    /// are read back with every step whether anyone watches or not, so reporting them costs no
    /// copy, and a run given no callback builds no report at all.</para>
    ///
    /// <para>The callback may end the run after this step (<see cref="RequestStop"/>) — for early
    /// stopping on a validation metric, say — and may take the run's state as a checkpoint
    /// (<see cref="TakeCheckpoint"/>), to save it or evaluate it. A report is valid only during the
    /// callback it was handed to.</para>
    /// </summary>
    public sealed class TrainingStepReport
    {
        private readonly ResidentTrainingRun _run;
        private bool _live = true;

        internal TrainingStepReport(ResidentTrainingRun run, TrainingHistoryEntry entry, TimeSpan elapsed)
        {
            _run = run;
            Entry = entry;
            Elapsed = elapsed;
        }

        /// <summary>The step: the counters it ran at, its loss and the value every optimizer
        /// hyperparameter had in it — the entry it appended to the run's
        /// <see cref="TrainingCheckpoint.History"/>.</summary>
        public TrainingHistoryEntry Entry { get; }

        /// <summary>The global step counter the step ran at.</summary>
        public long Step => Entry.Step;

        /// <summary>The epoch of the batch the step trained on (see
        /// <see cref="TrainingCheckpoint.Epoch"/>).</summary>
        public long? Epoch => Entry.Epoch;

        /// <summary>The index of the batch the step trained on within its epoch: for the array forms
        /// of <c>Fit</c> / <c>Train</c>, its index in the arrays.</summary>
        public long? BatchIndex => Entry.BatchIndex;

        /// <summary>The step's loss.</summary>
        public float Loss => Entry.Loss;

        /// <summary>The time since the run began, this step included.</summary>
        public TimeSpan Elapsed { get; }

        /// <summary>Whether <see cref="RequestStop"/> was called.</summary>
        public bool StopRequested { get; private set; }

        /// <summary>
        /// Ends the run after this step: no further step runs, and the run returns the state this
        /// step produced, with <see cref="TrainingStopReason.StopRequested"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">The callback this report was handed to has
        /// returned.</exception>
        public void RequestStop()
        {
            ThrowIfStale();
            StopRequested = true;
        }

        /// <summary>
        /// The run's state after this step, as a checkpoint — without copying anything, so on a GPU
        /// its tensors are in the training device's memory; see
        /// <see cref="ResidentTrainingRun.TakeCheckpoint"/>, which this is. The checkpoint is the
        /// caller's and stays whole after the run goes on and after it ends; while the caller holds
        /// it, later steps write their state beside it rather than over it.
        /// </summary>
        /// <exception cref="InvalidOperationException">The callback this report was handed to has
        /// returned.</exception>
        public TrainingCheckpoint TakeCheckpoint()
        {
            ThrowIfStale();
            return _run.TakeCheckpoint();
        }

        /// <summary>Ends the report's validity, once the callback it was handed to returns.</summary>
        internal void Expire() => _live = false;

        private void ThrowIfStale()
        {
            if (!_live)
                throw new InvalidOperationException(
                    "This training step report is no longer valid: a report can be acted on only "
                    + "during the step callback it was handed to, before the run's next step.");
        }
    }
}
