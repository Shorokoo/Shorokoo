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
    /// <see cref="TrainingRig.TrainStep(TrainingCheckpoint, TensorDataStruct, TensorDataStruct)"/>
    /// hands the host a fresh copy of every parameter and both optimizer moments after every step,
    /// and feeds them all back in on the next one. On a GPU that is the whole training state crossing
    /// the bus twice per step, so throughput tracks <i>parameter count</i> rather than arithmetic: a
    /// model with 3.3× the parameters and slightly fewer FLOPs trained 2.3× slower
    /// (Shorokoo/Shorokoo#325). A resident run moves the state once in, once out, and
    /// <see cref="Step(TensorDataStruct, TensorDataStruct)"/> in between costs the arithmetic only.</para>
    ///
    /// <para><b>Reading the state costs a download, so you ask for it.</b>
    /// <see cref="Step(TensorDataStruct, TensorDataStruct)"/> returns the step's loss — a scalar,
    /// always host-readable — and nothing else;
    /// <see cref="StepToCheckpoint(TensorDataStruct, TensorDataStruct)"/> runs the same step and
    /// brings the state back to the host as an ordinary <see cref="TrainingCheckpoint"/> you can read,
    /// save and resume from. So a run pays for exactly the checkpoints it takes: call
    /// <c>StepToCheckpoint</c> on the steps you want to save at (including the last one you care
    /// about), and <c>Step</c> on every other. State a run still holds is released by
    /// <see cref="Dispose"/>, so a run that never takes a checkpoint trains and discards.</para>
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
        /// The state the next step trains from. Its tensors are device-resident whenever the last
        /// step retained them, in which case nothing outside this run may read them.
        /// </summary>
        private TrainingCheckpoint _current;

        /// <summary>
        /// Whether <see cref="_current"/>'s tensors are this run's to free — true only of state a
        /// step of this run produced and a later step has superseded. The initial checkpoint is
        /// never freed however it was obtained: the caller may still be reading it, and a
        /// rig-created one shares the rig's own initial parameter tensors, which every checkpoint
        /// that rig creates is built over. Nor is state the run has published in a checkpoint the
        /// caller now holds.
        /// </summary>
        private bool _ownsCurrent;

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
        /// <param name="trainingInput">Training input data as a <see cref="TensorDataStruct"/>.</param>
        /// <param name="trainingTarget">Training target data as a <see cref="TensorDataStruct"/>.</param>
        public float Step(TensorDataStruct trainingInput, TensorDataStruct trainingTarget)
            => Advance(_rig.ResidentStep(Current, null, trainingInput, trainingTarget, retain: true)).Loss!.Value;

        /// <summary>
        /// Trains on one batch of a rig whose loss reads no target
        /// (<see cref="TrainingRig.HasTargets"/> is <c>false</c>), and returns its loss
        /// (Shorokoo/Shorokoo#331). Throws when the rig's loss does read a target.
        /// </summary>
        public float Step(TensorDataStruct trainingInput)
        {
            _rig.RequireTargetless(nameof(Step));
            return Step(trainingInput, _rig.TargetDef.FromOrderedData());
        }

        /// <summary>
        /// Trains on one batch with explicit values for the rig's schedule-less runtime
        /// hyperparameters (build them with <see cref="TrainingRig.MakeHyperparameters(float)"/>) and
        /// returns its loss, leaving the updated state resident.
        /// </summary>
        public float Step(
            TensorDataStruct hyperparameters, TensorDataStruct trainingInput, TensorDataStruct trainingTarget)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return Advance(_rig.ResidentStep(Current, hyperparameters, trainingInput, trainingTarget, retain: true))
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
            => Advance(_rig.ResidentBatchStep(Current, batch, retain: true)).Loss!.Value;

        /// <summary>
        /// Trains on one batch and brings the updated state back to the host as an ordinary
        /// checkpoint — the step to use where you want to save, resume or read the state. The
        /// returned checkpoint owns its tensors: the run goes on training from them but never frees
        /// them, so holding it is safe.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(TensorDataStruct trainingInput, TensorDataStruct trainingTarget)
            => Publish(_rig.ResidentStep(Current, null, trainingInput, trainingTarget, retain: false));

        /// <summary>
        /// <see cref="StepToCheckpoint(TensorDataStruct, TensorDataStruct)"/> for a rig whose loss
        /// reads no target (Shorokoo/Shorokoo#331). Throws when the rig's loss does read one.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(TensorDataStruct trainingInput)
        {
            _rig.RequireTargetless(nameof(StepToCheckpoint));
            return StepToCheckpoint(trainingInput, _rig.TargetDef.FromOrderedData());
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(TensorDataStruct, TensorDataStruct)"/> with explicit values for
        /// the rig's schedule-less runtime hyperparameters.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(
            TensorDataStruct hyperparameters, TensorDataStruct trainingInput, TensorDataStruct trainingTarget)
        {
            if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));
            return Publish(_rig.ResidentStep(Current, hyperparameters, trainingInput, trainingTarget, retain: false));
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(TensorDataStruct, TensorDataStruct)"/> on the next batch drawn
        /// from <paramref name="loader"/>, so the checkpoint records the batch that was used and a
        /// later <see cref="TrainingRig.Fit(IDataLoader, int, TrainingCheckpoint?)"/> resumes after it.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(IDataLoader loader)
        {
            if (loader is null) throw new ArgumentNullException(nameof(loader));
            return StepToCheckpoint(loader.Next());
        }

        /// <summary>
        /// <see cref="StepToCheckpoint(TensorDataStruct, TensorDataStruct)"/> on an already-drawn
        /// batch, recording its <see cref="DataBatch.Position"/> as the batch that was used.
        /// </summary>
        public TrainingCheckpoint StepToCheckpoint(DataBatch batch)
            => Publish(_rig.ResidentBatchStep(Current, batch, retain: false));

        /// <summary>The state to train the next step from, with the run still usable.</summary>
        private TrainingCheckpoint Current => _disposed
            ? throw new ObjectDisposedException(nameof(ResidentTrainingRun),
                "This resident training run has been disposed and its state released. Take the " +
                "checkpoint you need with StepToCheckpoint(...) before disposing the run.")
            : _current;

        /// <summary>Takes over a step's result, releasing the state it superseded.</summary>
        private TrainingCheckpoint Advance(TrainingCheckpoint next)
        {
            ReleaseCurrent();
            _current = next;
            _ownsCurrent = true;
            return next;
        }

        /// <summary>
        /// Takes over a step's result and hands it to the caller: the run keeps training from it but
        /// gives up the right to free it, since the caller now holds it too.
        /// </summary>
        private TrainingCheckpoint Publish(TrainingCheckpoint next)
        {
            ReleaseCurrent();
            _current = next;
            _ownsCurrent = false;
            return next;
        }

        private void ReleaseCurrent()
        {
            if (!_ownsCurrent) return;
            _ownsCurrent = false;
            TrainingRig.ReleaseCheckpointState(_current);
        }

        /// <summary>
        /// Releases the state the run still owns. State already published in a checkpoint is left
        /// alone — the caller holds it — so a checkpoint taken from this run stays readable after the
        /// run is gone.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            ReleaseCurrent();
            // Drop the released checkpoint rather than pinning its whole object graph for the
            // lifetime of a run that is finished with it.
            _current = null!;
            _disposed = true;
        }
    }
}
