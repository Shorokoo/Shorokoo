using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Shorokoo.Core.Backends;

namespace Shorokoo
{
    /// <summary>
    /// A tensor's life: whether it is alive, why it died, and the reader locks the runs reading it
    /// hold.
    ///
    /// <para>This is the tensor's own state and nobody else's. A tensor is its allocation — there is
    /// no second object naming the same memory and no count of names — so the one question the
    /// state answers is whether this tensor may still be read, and the one decision it makes is
    /// who ends that. <see cref="TryTake"/> is the single atomic way to end it deliberately; every
    /// deliberate death — deletion, consumption by a run, a move into an attribute — is a take plus
    /// what the taker does with the memory.</para>
    /// </summary>
    public abstract partial class TensorData
    {
        // Per tensor, and taken by nothing else. Deliberately not a process-wide gate: the lock and
        // unlock path runs on every feed of every run, and a delete has to be able to wait for
        // readers while holding nothing at all.
        private readonly object _gate = new();

        // Null while the tensor is alive; set once, under the gate, to the reason it died.
        private TensorDeath? _death;

        // Whether a take handed the memory to a caller who is now responsible for releasing it --
        // a run that consumed the tensor, or the move into an attribute. Distinguishes a tensor
        // whose memory is still its own to release (deleted while a run was reading it) from one
        // whose memory has gone elsewhere.
        private bool _taken;

        // Whether the memory has been released, which happens exactly once.
        private bool _released;

        // Reader locks: how many runs are reading this tensor right now.
        private int _locks;

        // Created on the first lock and never disposed: it is the signal a deliberate delete
        // raises, and a run registers on it for as long as it holds its lock. A tensor that is
        // never fed to a run never has one.
        private CancellationTokenSource? _eviction;

        // Completed once the memory has been released, which is what DeleteAsync waits on. Created
        // only when a delete finds the memory still out.
        private TaskCompletionSource? _reclaimed;

        // Frees that arrived while a lock was held: a write through AccessModifiable... retires the
        // runtime values built from these contents, and freeing them under a running read is the
        // same use-after-free as freeing the contents themselves. Run when the last lock drops.
        private List<Action>? _deferred;

        /// <summary>
        /// True once this tensor is dead — deleted, consumed by a run, or moved into an attribute.
        /// Its shape, dtype and <see cref="ToString"/> stay readable as metadata; every other access
        /// throws, saying which of the three it was.
        /// </summary>
        public bool IsDisposed => Volatile.Read(ref _death) is not null;

        /// <summary>Why this tensor died, or null while it is alive. What every refused access
        /// names.</summary>
        internal TensorDeath? Death => Volatile.Read(ref _death);

        /// <summary>Whether a run is reading this tensor right now — what <see cref="Delete"/>
        /// refuses on and <see cref="TryDelete"/> declines on.</summary>
        internal bool IsLocked
        {
            get { lock (_gate) return _locks > 0; }
        }

        /// <summary>
        /// Releases this tensor's memory — through <see cref="AllocatingBackend"/>, which made it.
        /// Called exactly once, by whoever the memory belongs to when the tensor dies.
        /// </summary>
        private protected abstract void ReleaseMemory();

        /// <summary>
        /// This tensor's contents as host bytes of the caller's own, without the liveness check:
        /// the tensor's own array copied, a host-readable value's buffer copied, or a device value
        /// read back through its allocating backend.
        /// </summary>
        private protected abstract byte[] CopyContentBytes();

        /// <summary>A string tensor's elements, without the liveness check.</summary>
        private protected abstract IReadOnlyList<string> CopyContentStrings();

        /// <summary>
        /// Guards every path to the tensor's elements. Call it before touching storage.
        ///
        /// <para>Visible to the framework as well as to subclasses, so that a caller holding the
        /// tensor refuses in these words — naming the tensor and why it died — rather than
        /// failing further down on memory that is not there.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">The tensor is dead.</exception>
        protected internal void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _death) is { } death) throw death.Refusal(this);
        }

        /// <summary>
        /// Deletes this tensor: marks it dead and releases its memory through the backend that made
        /// it. The same operation as <see cref="Delete"/>.
        ///
        /// <para>It throws while a run is reading the tensor, because releasing the memory then would
        /// free a buffer the run is in the middle of reading (Shorokoo/Shorokoo#366).
        /// <see cref="TryDelete"/> is the form that declines instead, and <see cref="DeleteAsync"/>
        /// the one that asks the readers to stop.</para>
        ///
        /// <para>A tensor that is already dead — deleted, consumed or moved — is left as it is, so
        /// disposing twice, or at the end of a <c>using</c> over a tensor a run consumed, is
        /// harmless.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">A run is reading this tensor.</exception>
        public void Dispose() => Delete();

        /// <summary>
        /// Deletes this tensor: marks it dead and releases its memory through the backend that made
        /// it. Every access afterwards throws, saying it was deleted. <see cref="Dispose"/> is the
        /// same operation.
        /// </summary>
        /// <exception cref="InvalidOperationException">A run is reading this tensor; wait for it
        /// to return, or use <see cref="DeleteAsync"/> to ask it to stop.</exception>
        public void Delete()
        {
            switch (TryTake(TensorDeath.Deleted))
            {
                case TakeOutcome.Taken:
                    ReleaseTaken();
                    return;
                case TakeOutcome.Dead:
                    return;
                default:
                    throw ReadByARun(nameof(Delete));
            }
        }

        /// <summary>
        /// Deletes this tensor if nothing is reading it: marks it dead, releases its memory and
        /// returns true. If a run holds a lock on it, nothing at all changes — no run is disturbed
        /// and the tensor stays readable — and it returns false.
        ///
        /// <para>Opportunistic, and that is the whole of it. It is not
        /// <c>DeleteAsync(TimeSpan.Zero)</c>: it signals no eviction and aborts nobody, so a
        /// caller can ask whether these bytes can go without committing to their going.</para>
        /// </summary>
        /// <returns>True when the tensor is dead — including when it already was.</returns>
        public bool TryDelete()
        {
            switch (TryTake(TensorDeath.Deleted))
            {
                case TakeOutcome.Taken:
                    ReleaseTaken();
                    return true;
                case TakeOutcome.Dead:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Deletes this tensor and waits up to <paramref name="timeout"/> for its memory to be
        /// released.
        /// </summary>
        /// <remarks>
        /// <para><b>The tensor is deleted either way.</b> Deletion is immediate and unconditional:
        /// from the moment this is called every access to the tensor throws, whatever this returns
        /// and however long the wait takes. It never ignores a reader, though: the memory goes back
        /// to the backend only once the last run reading it has stood down, and each such run is
        /// asked to stop.</para>
        /// <para><b>A false is a diagnostic, not a failure.</b> It says the memory has not come
        /// back yet because a reader has not stood down inside the budget. Retrying is waiting for
        /// something that has already happened; the memory comes back when that run ends, with no
        /// second call. A timeout never rolls the deletion back — eviction has been signalled and
        /// the compliant readers have already thrown their work away, and un-signalling cannot
        /// un-abort them.</para>
        /// <para><b>It is never prompt.</b> A backend is asked to stop, and what it can stop at is
        /// its own business: the ONNX Runtime backend stops between kernels, so the wait is bounded
        /// below by the longest single operator in flight — and by a whole run where the graph is
        /// one operator, which has no boundary to stop at. A backend that ignores the request at
        /// all makes this slow and never unsafe: the wait then ends when the run finishes
        /// naturally.</para>
        /// <para>A tensor that has already died another way is not deleted again: the wait is for
        /// whoever holds its memory now to release it.</para>
        /// </remarks>
        /// <param name="timeout">How long to wait for the memory. <see cref="TimeSpan.Zero"/>
        /// signals eviction and does not wait; <see cref="Timeout.InfiniteTimeSpan"/> waits
        /// forever. There is deliberately no parameterless overload — an unbounded wait on a
        /// reader that may never stand down is a hang, and a caller who wants one should be seen
        /// to have asked for it.</param>
        /// <param name="cancellationToken">Cancels <i>the wait</i>, never the deletion.</param>
        /// <returns>True when the memory was released inside the budget; false when the wait ran
        /// out with the tensor deleted all the same.</returns>
        /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was
        /// cancelled while waiting. The tensor stays deleted.</exception>
        public async Task<bool> DeleteAsync(TimeSpan timeout, CancellationToken cancellationToken = default)
        {
            bool releaseNow = false;
            CancellationTokenSource? eviction = null;
            Task? reclaimed = null;
            lock (_gate)
            {
                if (_death is null)
                {
                    _death = TensorDeath.Deleted;
                    // Unread, the memory is this call's to release, exactly as a take hands it
                    // over; read, it is released by the last reader to stand down.
                    if (_locks == 0)
                    {
                        _taken = true;
                        releaseNow = true;
                    }
                }
                if (!releaseNow && !_released)
                {
                    if (_locks > 0) eviction = _eviction;
                    reclaimed = (_reclaimed ??= new TaskCompletionSource(
                        TaskCreationOptions.RunContinuationsAsynchronously)).Task;
                }
            }

            if (releaseNow)
            {
                ReleaseTaken();
                return true;
            }

            // Outside the gate: cancelling runs every reader's callback on this thread, and a
            // reader standing down takes the gate to drop its lock.
            eviction?.Cancel();

            if (reclaimed is null || reclaimed.IsCompleted) return true;
            if (timeout == TimeSpan.Zero) return false;

            using var budget = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var ranOut = Task.Delay(timeout, budget.Token);
            var first = await Task.WhenAny(reclaimed, ranOut).ConfigureAwait(false);
            if (ReferenceEquals(first, reclaimed))
            {
                budget.Cancel();
                return true;
            }
            cancellationToken.ThrowIfCancellationRequested();
            return false;
        }

        /// <summary>
        /// Ends this tensor's life deliberately, if no run is reading it: marks it dead with
        /// <paramref name="death"/> and hands its memory to the caller, who must release it with
        /// <see cref="ReleaseTaken"/> — or feed it to a run first and release it after. If a run holds
        /// a lock nothing at all changes. The one atomic primitive every deliberate death is built
        /// on.
        /// </summary>
        internal TakeOutcome TryTake(TensorDeath death)
        {
            ArgumentNullException.ThrowIfNull(death);
            lock (_gate)
            {
                if (_death is not null) return TakeOutcome.Dead;
                if (_locks > 0) return TakeOutcome.Locked;
                _death = death;
                _taken = true;
                return TakeOutcome.Taken;
            }
        }

        /// <summary>
        /// Releases the memory a successful <see cref="TryTake"/> handed over, through the backend
        /// that made it. Once: a second call releases nothing.
        /// </summary>
        internal void ReleaseTaken()
        {
            TaskCompletionSource? reclaimed;
            lock (_gate)
            {
                if (!_taken || _released) return;
                _released = true;
                reclaimed = _reclaimed;
            }
            ReleaseMemory();
            // After the release, so a waiter that sees this has seen the memory go.
            reclaimed?.TrySetResult();
        }

        /// <summary>
        /// Takes this tensor for a run that consumes it: <see cref="TryTake"/>, refusing a tensor
        /// that is already dead or that another run is reading. The run then feeds
        /// <see cref="ValueForTakenRun"/> and releases the memory with <see cref="ReleaseTaken"/>
        /// once it returns, however it returns.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The tensor is dead.</exception>
        /// <exception cref="InvalidOperationException">Another run is reading it.</exception>
        internal void TakeForRun(TensorDeath consumption)
        {
            switch (TryTake(consumption))
            {
                case TakeOutcome.Taken:
                    return;
                case TakeOutcome.Dead:
                    ThrowIfDisposed();
                    return;
                default:
                    throw new InvalidOperationException(
                        $"Tensor {this} was donated to a run while another run is reading it, and a "
                        + "tensor that is being read cannot be consumed: the run it was donated to "
                        + "would take memory the other one is still reading. Feed the tensor itself "
                        + "rather than its donation, or wait for the other run to return.");
            }
        }

        /// <summary>The value a run that has taken this tensor feeds. See
        /// <see cref="ValueFor"/>.</summary>
        internal IShorokooTensorValue ValueForTakenRun(IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            return ValueFor(backend);
        }

        /// <summary>
        /// Takes a reader lock for the length of a run, and hands back the signal a deliberate
        /// delete raises. Refuses a tensor that is already dead: there is nothing left to read.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The tensor is dead.</exception>
        internal CancellationToken AcquireReadLock()
        {
            lock (_gate)
            {
                if (_death is { } death) throw death.Refusal(this);
                _locks++;
                return (_eviction ??= new CancellationTokenSource()).Token;
            }
        }

        /// <summary>
        /// Drops a reader lock. The memory goes now if this was the last lock on a tensor deleted
        /// while it was held — which is what a deleted-but-not-yet-released tensor is waiting for —
        /// and so do the frees that were deferred behind the readers.
        /// </summary>
        internal void ReleaseReadLock()
        {
            bool release = false;
            List<Action>? deferred = null;
            TaskCompletionSource? reclaimed = null;
            lock (_gate)
            {
                _locks--;
                if (_locks == 0)
                {
                    (deferred, _deferred) = (_deferred, null);
                    if (_death is not null && !_taken && !_released)
                    {
                        _released = true;
                        release = true;
                        reclaimed = _reclaimed;
                    }
                }
            }

            if (release) ReleaseMemory();
            if (deferred is not null) foreach (var pending in deferred) pending();
            // After the release, so a waiter that sees this has seen the memory go.
            reclaimed?.TrySetResult();
        }

        /// <summary>
        /// Runs <paramref name="free"/> now, or once the last reader of this tensor stands down. For
        /// a release that is not the tensor's own memory — the runtime values built from these
        /// contents, which a write retires — where freeing under a running read is the same fault
        /// as freeing the contents.
        /// </summary>
        internal void FreeWhenUnlocked(Action free)
        {
            ArgumentNullException.ThrowIfNull(free);
            lock (_gate)
            {
                if (_locks > 0)
                {
                    (_deferred ??= []).Add(free);
                    return;
                }
            }
            free();
        }

        /// <summary>What <see cref="Delete"/> throws for a tensor a run is reading.</summary>
        private InvalidOperationException ReadByARun(string operation) => new(
            $"Tensor {this} is being read by a run, so {operation} cannot release its memory: the "
            + "run is reading it. Wait for the run to return, use TryDelete() to delete it only if "
            + "nothing is reading it, or DeleteAsync(timeout) to ask the run to stop.");
    }

    /// <summary>What <see cref="TensorData.TryTake"/> did.</summary>
    internal enum TakeOutcome
    {
        /// <summary>The tensor is now dead, and its memory is the caller's to release.</summary>
        Taken,
        /// <summary>A run is reading the tensor; nothing changed.</summary>
        Locked,
        /// <summary>The tensor was already dead; nothing changed.</summary>
        Dead,
    }

    /// <summary>
    /// Why a tensor died, and the words every refused access says it in.
    ///
    /// <para>The message matters more than it looks. A tensor that died of something the caller
    /// did not see — consumed by a run it was donated to, moved into an attribute by a call that
    /// took it — throws far from the cause, and the exception is the only thing that can say what
    /// the cause was and what to do instead.</para>
    /// </summary>
    internal sealed class TensorDeath
    {
        private readonly Func<TensorData, string> _explain;

        private TensorDeath(string cause, Func<TensorData, string> explain)
        {
            Cause = cause;
            _explain = explain;
        }

        /// <summary>A short name for the cause, for a diagnostic that wants one.</summary>
        internal string Cause { get; }

        /// <summary>Deleted: <c>Delete</c>, <c>Dispose</c>, <c>TryDelete</c> or
        /// <c>DeleteAsync</c>.</summary>
        internal static TensorDeath Deleted { get; } = new("deleted", t =>
            $"Tensor {t} has been deleted -- by Delete(), Dispose(), TryDelete() or DeleteAsync() "
            + "-- so its memory is gone and nothing may read it.");

        /// <summary>Moved into a <see cref="TensorAttribute"/> by
        /// <see cref="TensorData.MoveToAttribute"/>.</summary>
        internal static TensorDeath MovedToAttribute { get; } = new("moved into an attribute", t =>
            $"Tensor {t} was moved into a TensorAttribute by MoveToAttribute(), which took its "
            + "contents, so nothing may read it any more. Read the attribute instead, or take a "
            + "CopyTo(...) before the move where both are needed.");

        /// <summary>
        /// Consumed by a run of <paramref name="graph"/> on the context running on
        /// <paramref name="backend"/>, which it was donated to.
        /// </summary>
        internal static TensorDeath ConsumedBy(string graph, BackendDescription backend)
            => new("consumed by a run", t =>
                $"Tensor {t} was consumed by a run of {graph} on the compute context over "
                + $"{backend}: it was donated to that run with Donate(), which gives the run its "
                + "memory, so nothing may read it any more. Feed the tensor itself rather than its "
                + "donation at that call if you need it afterwards.");

        /// <summary>The exception an access to <paramref name="tensor"/> throws.</summary>
        internal ObjectDisposedException Refusal(TensorData tensor)
            => new(tensor.GetType().Name, _explain(tensor));

        /// <inheritdoc/>
        public override string ToString() => Cause;
    }
}
