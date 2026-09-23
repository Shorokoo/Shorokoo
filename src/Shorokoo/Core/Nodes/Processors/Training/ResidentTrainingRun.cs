using Shorokoo.Core.Training;
using System;
using System.Collections.Generic;

namespace Shorokoo
{
    /// <summary>
    /// A training run that keeps its state — trainable parameters, model state and optimizer state —
    /// where the execution provider produced it, instead of moving it through host memory between
    /// steps.
    ///
    /// <para><b>Why it exists.</b> The checkpoint-in / checkpoint-out
    /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, IData, IData)"/>
    /// hands the host a fresh copy of every parameter and both optimizer moments after every step,
    /// and feeds them all back in on the next one. On a GPU that is the whole training state crossing
    /// the bus twice per step, so throughput tracks <i>parameter count</i> rather than arithmetic: a
    /// model with 3.3× the parameters and slightly fewer FLOPs trained 2.3× slower
    /// (Shorokoo/Shorokoo#325). A resident run moves the state once in, once out, and
    /// <see cref="Step(IData, IData)"/> in between costs the arithmetic only.</para>
    ///
    /// <para><b>Reading the state costs a download, so you ask for it.</b>
    /// <see cref="Step(IData, IData)"/> returns the step's loss — a scalar, always host-readable —
    /// and nothing else; <see cref="StepToCheckpoint(IData, IData)"/> runs the same step and brings
    /// the state back to the host as an ordinary <see cref="TrainingCheckpoint"/> you can read, save
    /// and resume from. So a run pays for exactly the checkpoints it takes: call
    /// <c>StepToCheckpoint</c> on the steps you want to save at (including the last one you care
    /// about), and <c>Step</c> on every other. State a run still holds is released by
    /// <see cref="Dispose"/>, so a run that never takes a checkpoint trains and discards.</para>
    ///
    /// <para><b>What each step consumes.</b> A step feeds its inputs the way
    /// <c>TrainStep</c> does: a batch passed as it is is consumed by the step, and one passed
    /// <c>.Shared()</c> is read. The state is the run's business: state a step of this run produced
    /// is consumed by the next step, which is what releases it as it is superseded; a checkpoint the
    /// run has handed out (<c>StepToCheckpoint</c>) is the caller's too, so the next step only reads
    /// it; and the checkpoint the run began from is fed to the first step as it was passed to
    /// <see cref="TrainingRig.BeginResidentRun"/> — consumed as it is, read if passed
    /// <c>.Shared()</c>.</para>
    ///
    /// <para><b>A failed step.</b> A step takes what it consumes when it starts, and a step that
    /// then fails cannot give it back. Where that was the run's own state there is nothing left to
    /// train from, and every later step says so; begin a new run from the last checkpoint you took.
    /// A checkpoint handed out is only read, so a step that fails after one leaves it — and the
    /// run — whole.</para>
    ///
    /// <para><b>On a CPU backend</b> there is no second memory to be resident in, so a resident run
    /// is an ordinary step loop that releases each step's state as the next supersedes it — same
    /// numbers, same checkpoints, no growth.</para>
    ///
    /// <para>Create one with <see cref="TrainingRig.BeginResidentRun(TrainingCheckpoint?)"/>. It is
    /// not thread-safe: a run is a position in a training loop, and one loop drives it.
    /// <see cref="TrainingRig.Train"/> and every <c>Fit</c> overload drive one internally, so they
    /// already train this way and return the host checkpoint their last step produced.</para>
    /// </summary>
    public sealed class ResidentTrainingRun : IDisposable
    {
        private readonly TrainingRig _rig;

        /// <summary>
        /// The state the next step trains from, fed as its <see cref="TrainingCheckpoint.FeedMode"/>
        /// says. Its tensors are device-resident whenever the last step retained them, in which case
        /// nothing outside this run may read them.
        /// </summary>
        private TrainingCheckpoint _current;

        /// <summary>
        /// Whether <see cref="_current"/>'s tensors are this run's alone — true only of state a step
        /// of this run produced and handed to nobody, which the next step consumes and
        /// <see cref="Dispose"/> releases. Never the checkpoint the run began from, which was the
        /// caller's, nor state the run has published in a checkpoint the caller now holds.
        /// </summary>
        private bool _ownsCurrent;

        /// <summary>Set when a step failed after it had consumed the run's state, so there is
        /// nothing left to train from.</summary>
        private bool _lost;

        private bool _disposed;

        internal ResidentTrainingRun(TrainingRig rig, TrainingCheckpoint initialCheckpoint)
        {
            _rig = rig;
            _current = initialCheckpoint;
            _ownsCurrent = false;
        }

        /// <summary>
        /// The global step the run sits at — the <see cref="TrainingCheckpoint.Step"/> the run's
        /// last step produced, and the value its scheduled hyperparameters see. The next step
        /// produces this plus one.
        /// </summary>
        public long CurrentStep => Current.Step;

        /// <summary>Trains on one batch and returns its loss, leaving the updated state resident.</summary>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>,
        /// consumed by the step, or one passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingTarget">Training target data, in the same forms.</param>
        public float Step(IData trainingInput, IData trainingTarget)
            => Advance(Stepped(c => _rig.ResidentStep(c, null, trainingInput, trainingTarget, retain: true))).Loss!.Value;

        /// <summary>
        /// Trains on one batch of a rig whose loss reads no target
        /// (<see cref="TrainingRig.HasTargets"/> is <c>false</c>), and returns its loss
        /// (Shorokoo/Shorokoo#331). Throws when the rig's loss does read a target.
        /// </summary>
        public float Step(IData trainingInput)
        {
            _rig.RequireTargetless(nameof(Step));
            return Step(trainingInput, _rig.TargetDef.FromOrderedData());
        }

        /// <summary>
        /// Trains on one batch with explicit values for the rig's schedule-less runtime
        /// hyperparameters (build them with <see cref="TrainingRig.MakeHyperparameters(float)"/>) and
        /// returns its loss, leaving the updated state resident. The hyperparameters are fed like
        /// the batch: consumed as they are, read when passed <c>.Shared()</c>.
        /// </summary>
        public float Step(IData hyperparameters, IData trainingInput, IData trainingTarget)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return Advance(Stepped(c => _rig.ResidentStep(c, hyperparameters, trainingInput, trainingTarget, retain: true)))
                .Loss!.Value;
        }

        /// <summary>
        /// Trains on the next batch drawn from <paramref name="loader"/> — taking the run's epoch and
        /// batch counters from it, as
        /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, IDataLoader)"/> does — and returns its
        /// loss, leaving the updated state resident.
        /// </summary>
        public float Step(IDataLoader loader)
        {
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            return Step(loader.Next());
        }

        /// <summary>
        /// Trains on an already-drawn batch, taking the run's epoch and batch counters from its
        /// <see cref="DataBatch.Position"/>, and returns its loss — for a loop that draws from its
        /// loader itself.
        /// </summary>
        public float Step(DataBatch batch)
            => Advance(Stepped(c => _rig.ResidentBatchStep(c, batch, retain: true))).Loss!.Value;

        /// <summary>
        /// Trains on one batch and brings the updated state back to the host as an ordinary
        /// checkpoint — the step to use where you want to save, resume or read the state. The
        /// returned checkpoint owns its tensors: the run goes on training from them but only ever
        /// reads them, so holding it is safe.
        /// </summary>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>,
        /// consumed by the step, or one passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingTarget">Training target data, in the same forms.</param>
        public TrainingCheckpoint StepToCheckpoint(IData trainingInput, IData trainingTarget)
            => Publish(Stepped(c => _rig.ResidentStep(c, null, trainingInput, trainingTarget, retain: false)));

        /// <summary>
        /// <see cref="StepToCheckpoint(IData, IData)"/> for a rig whose loss reads no target
        /// (Shorokoo/Shorokoo#331). Throws when the rig's loss does read one.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(IData trainingInput)
        {
            _rig.RequireTargetless(nameof(StepToCheckpoint));
            return StepToCheckpoint(trainingInput, _rig.TargetDef.FromOrderedData());
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(IData, IData)"/> with explicit values for the rig's
        /// schedule-less runtime hyperparameters.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(
            IData hyperparameters, IData trainingInput, IData trainingTarget)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return Publish(Stepped(c => _rig.ResidentStep(c, hyperparameters, trainingInput, trainingTarget, retain: false)));
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(IData, IData)"/> on the next batch drawn from
        /// <paramref name="loader"/>, so the checkpoint records the batch that was used and a later
        /// <see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?)"/> resumes after it.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(IDataLoader loader)
        {
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            return StepToCheckpoint(loader.Next());
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(IData, IData)"/> on an already-drawn batch, recording its
        /// <see cref="DataBatch.Position"/> as the batch that was used.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(DataBatch batch)
            => Publish(Stepped(c => _rig.ResidentBatchStep(c, batch, retain: false)));

        /// <summary>The state to train the next step from, with the run still usable.</summary>
        private TrainingCheckpoint Current
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(ResidentTrainingRun),
                        "This resident training run has been disposed and its state released. Take the " +
                        "checkpoint you need with StepToCheckpoint(...) before disposing the run.");
                if (_lost)
                    throw new InvalidOperationException(
                        "A step of this resident training run failed after it had consumed the run's "
                        + "state: a step takes the state it trains from when it starts, and one that "
                        + "fails cannot give it back, so there is nothing left to train from. Begin a new "
                        + "run from the last checkpoint you took with StepToCheckpoint(...); the run only "
                        + "ever reads a checkpoint it has handed out, so a failure leaves that one whole.");
                return _current;
            }
        }

        /// <summary>
        /// Runs one step from the current state, noting when a failed step took that state with it:
        /// a step consumes the state it is fed as it is before it computes, so a failure part-way
        /// leaves it dead.
        /// </summary>
        private TrainingCheckpoint Stepped(Func<TrainingCheckpoint, TrainingCheckpoint> step)
        {
            var current = Current;
            try
            {
                return step(current);
            }
            catch
            {
                if (IsSpent(current)) _lost = true;
                throw;
            }
        }

        /// <summary>Whether any tensor of <paramref name="checkpoint"/>'s state is dead.</summary>
        private static bool IsSpent(TrainingCheckpoint checkpoint)
        {
            foreach (var state in (TensorDataStruct[])[checkpoint.TrainableParams, checkpoint.ModelState, checkpoint.OptimizerState])
                foreach (var field in state.Fields.Values)
                    if (field is TensorData { IsDisposed: true }) return true;
            return false;
        }

        /// <summary>
        /// Takes over a step's result. The state it superseded was the step's to deal with: this
        /// run's own was consumed by it, and anything else it only read.
        /// </summary>
        private TrainingCheckpoint Advance(TrainingCheckpoint next)
        {
            _current = next;
            _ownsCurrent = true;
            return next;
        }

        /// <summary>
        /// Takes over a step's result and hands it to the caller: the run keeps training from it but
        /// only ever reads it from now on, since the caller holds it too.
        /// </summary>
        private TrainingCheckpoint Publish(TrainingCheckpoint next)
        {
            _current = next.Shared();
            _ownsCurrent = false;
            return next;
        }

        /// <summary>
        /// Releases the state the run still owns. State already published in a checkpoint is left
        /// alone — the caller holds it — so a checkpoint taken from this run stays readable after the
        /// run is gone.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            if (_ownsCurrent && !_lost) TrainingRig.ReleaseCheckpointState(_current);
            _ownsCurrent = false;
            // Drop the released checkpoint rather than pinning its whole object graph for the
            // lifetime of a run that is finished with it.
            _current = null!;
            _disposed = true;
        }
    }
}
