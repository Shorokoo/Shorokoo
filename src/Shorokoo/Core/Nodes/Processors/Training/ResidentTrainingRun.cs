using Shorokoo.Core.Training;
using System;
using System.Collections.Generic;

namespace Shorokoo
{
    /// <summary>
    /// A training run that owns its state — trainable parameters, model state and optimizer state —
    /// between steps, where the execution provider produced it.
    ///
    /// <para><b>What it is for.</b> A step's state comes back where the step ran — on a GPU, the
    /// card's memory — whether it is taken by
    /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, IData, IData)"/> or by a run, so neither
    /// moves the state through host memory between steps (Shorokoo/Shorokoo#325). A run also owns
    /// that state: each step consumes the state the step before it produced, releasing it as it is
    /// superseded rather than when a collection gets to it, and <see cref="Step(IData, IData)"/>
    /// hands back only the loss, so a loop that does not need every step's checkpoint never holds
    /// one.</para>
    ///
    /// <para><b>A checkpoint stays where the state is.</b> <see cref="Step(IData, IData)"/> returns
    /// the step's loss — a scalar, always host-readable — and nothing else;
    /// <see cref="StepToCheckpoint(IData, IData)"/> runs the same step and hands you the state as an
    /// ordinary <see cref="TrainingCheckpoint"/>, and <see cref="TakeCheckpoint"/> does so between
    /// steps without running one. Neither copies anything: on a GPU the checkpoint's tensors are in
    /// the training device's memory, exactly where the run left them. You can save it —
    /// <see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/> and the <c>.skpt</c> saves
    /// write it straight out of device memory through one bounded host buffer — resume from it, or
    /// feed it to another step, or read its tensors' values, which copies them to the host and
    /// leaves them where they are; <see cref="TrainingCheckpoint.ToHost"/> makes a copy of the whole
    /// state in host memory. State a run still holds is released by
    /// <see cref="Dispose"/>, so a run that never takes a checkpoint trains and discards.</para>
    ///
    /// <para><b>What each step consumes.</b> A step feeds its inputs the way
    /// <c>TrainStep</c> does: a batch passed as it is is consumed by the step, and one passed
    /// <c>.Shared()</c> is read. The state is the run's business: state a step of this run produced
    /// is consumed by the next step, which is what releases it as it is superseded; a checkpoint the
    /// run has handed out (<c>StepToCheckpoint</c>, <c>TakeCheckpoint</c>) is the caller's too, so
    /// the next step only reads it; and the checkpoint the run began from is fed to the first step as it was passed to
    /// <see cref="TrainingRig.BeginResidentRun"/> — consumed as it is, read if passed
    /// <c>.Shared()</c>.</para>
    ///
    /// <para><b>A failed step.</b> A step takes what it consumes when it starts, and a step that
    /// then fails cannot give it back. Where that was the run's own state there is nothing left to
    /// train from, and every later step says so: begin a new run from the last checkpoint you took,
    /// or, before the run has handed any out, from a checkpoint you still hold. A checkpoint handed
    /// out is only read, so a step that fails after one leaves it — and the run — whole. State
    /// something else took is not the run's loss: a checkpoint handed out shares its tensors with
    /// the state the run goes on training from, so fed as it is to another step it takes that state
    /// with it, and every later step of the run is refused over it, naming what took it.</para>
    ///
    /// <para><b>On a CPU backend</b> there is no second memory to be resident in, so a resident run
    /// is an ordinary step loop that releases each step's state as the next supersedes it — same
    /// numbers, same checkpoints, no growth.</para>
    ///
    /// <para>Create one with <see cref="TrainingRig.BeginResidentRun(TrainingCheckpoint?)"/>. It is
    /// not thread-safe: a run is a position in a training loop, and one loop drives it.
    /// <see cref="TrainingRig.Train"/> and every <c>Fit</c> overload drive one internally, so they
    /// already train this way and return the checkpoint their last step produced, its state where
    /// that step left it.</para>
    /// </summary>
    public sealed class ResidentTrainingRun : IDisposable
    {
        private readonly TrainingRig _rig;

        /// <summary>
        /// The state the next step trains from, fed as its <see cref="TrainingCheckpoint.FeedMode"/>
        /// says. Its tensors are where the last step left them — on a GPU, the card's memory.
        /// </summary>
        private TrainingCheckpoint _current;

        /// <summary>
        /// Whether <see cref="_current"/>'s tensors are this run's alone — true only of state a step
        /// of this run produced and handed to nobody, which the next step consumes and
        /// <see cref="Dispose"/> releases. Never the checkpoint the run began from, which was the
        /// caller's, nor state the run has published in a checkpoint the caller now holds.
        /// </summary>
        private bool _ownsCurrent;

        /// <summary>Set when a step of this run failed after it had consumed the run's state, so
        /// there is nothing left to train from: what every later step throws, saying what is left
        /// to begin again from.</summary>
        private string? _lost;

        /// <summary>Whether the run has handed out a checkpoint (<see cref="StepToCheckpoint(IData, IData)"/>),
        /// which it only ever reads and so cannot lose.</summary>
        private bool _handedOut;

        /// <summary>The checkpoint the caller holds for <see cref="_current"/>, which the run reads
        /// through a <c>.Shared()</c> view of it: handed back as it was handed out, so taking it again
        /// gives the caller the same checkpoint and not the run's view. <c>null</c> once a step has
        /// moved on from it.</summary>
        private TrainingCheckpoint? _published;

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

        /// <summary>
        /// The value every optimizer hyperparameter had in the run's last step, keyed by the names of
        /// the rig that ran it (<see cref="TrainingRig.HyperparameterNames"/>); that step ran at
        /// counter <see cref="CurrentStep"/> <c>- 1</c>. Before the run's first step, the values of
        /// the checkpoint it began from: <c>null</c> where no step produced that one.
        ///
        /// <para>Host values each step records as it runs — a scheduled value read back with the
        /// step's loss, a runtime one copied from what the step was fed, which for a value resident
        /// on a device is a download the step makes — so asking costs nothing further; see
        /// <see cref="TrainingCheckpoint.AppliedHyperparameters"/>.</para>
        /// </summary>
        public IReadOnlyDictionary<string, AppliedHyperparameter>? AppliedHyperparameters
            => Current.AppliedHyperparameters;

        /// <summary>
        /// The steps that led to where the run sits, oldest first: the history of the checkpoint the
        /// run began from, with one entry appended per step of the run — whether taken with
        /// <c>Step</c> or <c>StepToCheckpoint</c> — as trimmed by <see cref="ReplaceHistory"/> and
        /// <see cref="ClearHistory"/>. Host values, so asking costs no download; see
        /// <see cref="TrainingCheckpoint.History"/>.
        /// </summary>
        public TrainingHistory History => Current.History;

        /// <summary>
        /// Replaces the run's <see cref="History"/>, from which later steps go on appending —
        /// typically by a slice of it: <c>run.ReplaceHistory(run.History.TakeLast(1000))</c> keeps a
        /// long run's history, and the memory it takes, bounded. The state the run trains from is
        /// untouched, and so is every checkpoint the run has handed out, which keeps the history it
        /// was handed out with.
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="history"/> is <c>null</c>.</exception>
        /// <exception cref="ObjectDisposedException">The run has been disposed.</exception>
        /// <exception cref="InvalidOperationException">A step of the run failed after it had consumed
        /// the run's state, so there is no state left for the history to go with.</exception>
        public void ReplaceHistory(TrainingHistory history)
        {
            ArgumentNullException.ThrowIfNull(history);
            // A checkpoint over the very same tensors, fed the same way: which of them the run owns,
            // and what a failed step would lose, is unchanged.
            _current = Current.WithHistory(history);
            _published = _published?.WithHistory(history);
        }

        /// <summary>
        /// Empties the run's <see cref="History"/>; the next step starts it again. Otherwise as
        /// <see cref="ReplaceHistory"/>.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The run has been disposed.</exception>
        /// <exception cref="InvalidOperationException">A step of the run failed after it had consumed
        /// the run's state, so there is no state left for the history to go with.</exception>
        public void ClearHistory() => ReplaceHistory(TrainingHistory.Empty);

        /// <summary>Trains on one batch and returns its loss, leaving the updated state resident.</summary>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>,
        /// consumed by the step, or one passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingTarget">Training target data, in the same forms.</param>
        public float Step(IData trainingInput, IData trainingTarget)
            => Advance(Stepped(c => _rig.ResidentStep(c, null, trainingInput, trainingTarget, reclaimSuperseded: !_ownsCurrent))).Loss!.Value;

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
            return Advance(Stepped(c => _rig.ResidentStep(c, hyperparameters, trainingInput, trainingTarget, reclaimSuperseded: !_ownsCurrent)))
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
            => Advance(Stepped(c => _rig.ResidentBatchStep(c, batch, reclaimSuperseded: !_ownsCurrent))).Loss!.Value;

        /// <summary>
        /// Trains on one batch and hands you the updated state as an ordinary checkpoint — the step
        /// to use where you want to save, resume or read the state. Its tensors stay where the step
        /// left them, which on a GPU is the training device's memory (see
        /// <see cref="TakeCheckpoint"/>). The returned checkpoint owns its tensors: the run goes on
        /// training from them but only ever reads them, so holding it is safe.
        /// </summary>
        /// <param name="trainingInput">Training input data: a <see cref="TensorDataStruct"/>,
        /// consumed by the step, or one passed through <c>.Shared()</c> or <c>.TryConsume()</c>.</param>
        /// <param name="trainingTarget">Training target data, in the same forms.</param>
        public TrainingCheckpoint StepToCheckpoint(IData trainingInput, IData trainingTarget)
            => Publish(Stepped(c => _rig.ResidentStep(c, null, trainingInput, trainingTarget, reclaimSuperseded: !_ownsCurrent)));

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
            return Publish(Stepped(c => _rig.ResidentStep(c, hyperparameters, trainingInput, trainingTarget, reclaimSuperseded: !_ownsCurrent)));
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(IData, IData)"/> on the next batch drawn from
        /// <paramref name="loader"/>, so the checkpoint records the batch that was used and a later
        /// <see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?, Action{TrainingStepReport}?, System.Threading.CancellationToken)"/> resumes after it.
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
            => Publish(Stepped(c => _rig.ResidentBatchStep(c, batch, reclaimSuperseded: !_ownsCurrent)));

        /// <summary>
        /// The run's state as it stands, as a checkpoint — between steps, without running one and
        /// without copying anything: its tensors are the very ones the next step trains from, so on
        /// a GPU they are in the training device's memory. Before the run's first step it is the
        /// checkpoint the run began from.
        ///
        /// <para>This is how a loop that decides to stop — cancelled, or satisfied — keeps what it
        /// trained: the state after its last step is always whole between steps, and this hands it
        /// over. Save it (<see cref="TrainingCheckpoint.Save(string, CheckpointComponents?)"/> and the
        /// <c>.skpt</c> saves write it straight out of device memory), resume from it, or bring it
        /// into host memory with <see cref="TrainingCheckpoint.ToHost"/>.</para>
        ///
        /// <para>The checkpoint is the caller's from then on, as one from
        /// <see cref="StepToCheckpoint(IData, IData)"/> is: the run goes on training from it but only
        /// ever reads it, and disposing the run leaves it alone. A later step therefore writes its
        /// new state beside it rather than over it; on a card that is a second copy of the state for
        /// as long as the caller holds the checkpoint.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">The run has been disposed.</exception>
        /// <exception cref="InvalidOperationException">A step of the run failed after it had consumed
        /// the run's state, so there is no state left to hand over.</exception>
        public TrainingCheckpoint TakeCheckpoint()
        {
            var current = Current;
            // A checkpoint handed out before goes back as it was handed out, not as the run's
            // .Shared() view of it.
            if (_published is { } published) return published;
            // The run's own state is handed over as Publish hands a step's; a checkpoint the caller
            // passed in as it is would otherwise still be consumed by the next step, although the
            // caller now holds it again. One the caller passed .Shared() is handed back as it
            // stands.
            if (_ownsCurrent || current.FeedMode != SharedInputMode.Shared)
            {
                _current = current.Shared();
                _ownsCurrent = false;
                _handedOut = true;
                _published = current;
            }
            return current;
        }

        /// <summary>The state to train the next step from, with the run still usable.</summary>
        private TrainingCheckpoint Current
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(ResidentTrainingRun),
                        "This resident training run has been disposed and its state released. Take the " +
                        "checkpoint you need with TakeCheckpoint() or StepToCheckpoint(...) before disposing the run.");
                if (_lost is { } lost) throw new InvalidOperationException(lost);
                return _current;
            }
        }

        /// <summary>
        /// Runs one step from the current state, noting when a failed step took that state with it:
        /// a step consumes the state it is fed as it is before it computes, so a failure part-way
        /// leaves it dead.
        ///
        /// <para>Only a step that found the state whole and left it dead took it. State that was
        /// dead already was taken by something else — a checkpoint this run handed out, fed as it
        /// is to another step, consumes the state the run trains from with it — and is not this
        /// run's loss: the step is refused over it before it takes anything, with that state's own
        /// refusal, and so is every step after it.</para>
        ///
        /// <para>Each step asks the rig to count the state it supersedes towards a collection only
        /// where that state is not the run's own (<c>!_ownsCurrent</c>, read before the step): a
        /// checkpoint the run handed out, or one it began from, which is garbage once the caller
        /// drops it and whose memory — on a card, device memory — only a finalizer frees. State the
        /// run owns the step consumes, and there is nothing to count.</para>
        /// </summary>
        private TrainingCheckpoint Stepped(Func<TrainingCheckpoint, TrainingCheckpoint> step)
        {
            var current = Current;
            var whole = !IsSpent(current);
            try
            {
                return step(current);
            }
            catch
            {
                if (whole && IsSpent(current)) _lost = Lost(beganFrom: !_ownsCurrent);
                throw;
            }
            finally
            {
                // State the run does not own -- the checkpoint it began from, or one it handed out --
                // is read by this step alone, which moves the run on to state of its own. The copies
                // made to read it would otherwise stay with the caller's checkpoint for as long as
                // that lives: on a card, a second copy of the whole state.
                if (!_ownsCurrent) TrainingRig.ReleaseStateReadCopies(current);
            }
        }

        /// <summary>
        /// What every step after a lost one throws: that the run's state went with the step that
        /// failed, and what is left to begin again from — the last checkpoint the run handed out,
        /// where it has handed one out, and otherwise only what the caller still holds.
        /// </summary>
        /// <param name="beganFrom">Whether the state lost was the checkpoint the run began from,
        /// taken by the run's first step.</param>
        private string Lost(bool beganFrom)
            => "A step of this resident training run failed after it had consumed the run's state: a "
               + "step takes the state it trains from when it starts, and one that fails cannot give it "
               + "back, so there is nothing left to train from. "
               + (_handedOut
                   ? "Begin a new run from the last checkpoint you took with TakeCheckpoint() or StepToCheckpoint(...); the "
                     + "run only ever reads a checkpoint it has handed out, so a failure leaves that one "
                     + "whole."
                   : beganFrom
                       ? "That state was the checkpoint the run began from, which its first step consumed, "
                         + "and the run has handed out no checkpoint since. Begin a new run from a "
                         + "checkpoint you still hold; one passed to BeginResidentRun as .Shared() is only "
                         + "read, so a failure leaves it whole."
                       : "The run has handed out no checkpoint to begin again from -- StepToCheckpoint(...) "
                         + "takes one, and so does TakeCheckpoint() -- so begin a new run from a checkpoint you still hold. The one this "
                         + "run began from is whole only if it was passed .Shared(); passed as it is, the "
                         + "run's first step consumed it.");

        /// <summary>Whether any tensor of <paramref name="checkpoint"/>'s state is dead.</summary>
        private static bool IsSpent(TrainingCheckpoint checkpoint)
        {
            foreach (var state in (TensorDataStruct[])[checkpoint.TrainableParams, checkpoint.ModelState, checkpoint.OptimizerState])
                foreach (var field in state.Fields.Values)
                    if (field is TensorData { IsDisposed: true }) return true;
            return false;
        }

        /// <summary>
        /// Takes over a step's result. The state it superseded was the step's to deal with: it
        /// consumed this run's own, and a checkpoint the run began from passed as it is, and only
        /// read anything else, letting go of the copies it read it through.
        /// </summary>
        private TrainingCheckpoint Advance(TrainingCheckpoint next)
        {
            _current = next;
            _ownsCurrent = true;
            _published = null;
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
            _handedOut = true;
            _published = next;
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
            if (_ownsCurrent && _lost is null) TrainingRig.ReleaseCheckpointState(_current);
            _ownsCurrent = false;
            // Drop the released checkpoint rather than pinning its whole object graph for the
            // lifetime of a run that is finished with it.
            _current = null!;
            _published = null;
            _disposed = true;
        }
    }
}
