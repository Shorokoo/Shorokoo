using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Shorokoo.Core.Backends;

namespace Shorokoo
{
    /// <summary>
    /// A tensor's life: whether it is alive, why it died, the reader locks the runs reading it
    /// hold, and the copies of it that runs made where they could not read it as it stands.
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

        // Whether a take handed the memory to a caller who is now responsible for it -- a run that
        // consumed the tensor, or the move into an attribute. Distinguishes a tensor whose memory is
        // still its own to release (deleted while a run was reading it) from one whose memory has
        // gone elsewhere.
        private bool _taken;

        // Whether the memory has been dealt with -- released, or handed to a backend that releases
        // it itself -- which happens exactly once.
        private bool _released;

        // Reader locks: how many runs are reading this tensor right now.
        private int _locks;

        // Who holds those locks, where the holder said: what a run refused this tensor names as the
        // read it would have taken the memory from under. Holders that did not say are counted in
        // _locks and not listed.
        private List<object>? _readers;

        // Created on the first lock and never disposed: it is the signal a deliberate delete
        // raises, and a run registers on it for as long as it holds its lock. A tensor that is
        // never fed to a run never has one.
        private CancellationTokenSource? _eviction;

        // Completed once the memory has been released, which is what DeleteAsync waits on. Created
        // only when a delete finds the memory still out.
        private TaskCompletionSource? _reclaimed;

        // The copies runs made of this tensor in memory they could read, one per place they are in.
        // Created on the first copy: most tensors are read where they are, or never read at all.
        private RunCopies? _copies;

        // The sequence this tensor is an element of, where it is one of a list sequence's own: the
        // sequence values runs built from that sequence were copied from this tensor too, so a
        // write here retires them as well as this tensor's own copies.
        private TensorDataSequence? _sequence;

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

        /// <summary>How a refusal names this tensor: "Tensor [4]:float32".</summary>
        internal string Describe() => $"Tensor {this}";

        /// <summary>
        /// Releases this tensor's own memory — through <see cref="AllocatingBackend"/>, which made it.
        /// Called at most once, by whoever the memory belongs to when the tensor dies; never for
        /// memory a run consumed, which its backend releases.
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
            if (Volatile.Read(ref _death) is { } death) throw death.Refusal(this, Describe());
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
        /// whoever holds its memory now to release it — for a tensor a run consumed, that run,
        /// whose backend releases it before the run returns.</para>
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
        /// <paramref name="death"/> and hands its memory to the caller, who must either release it
        /// with <see cref="ReleaseTaken"/> or hand it to a backend and say so with
        /// <see cref="HandedToBackend"/>. If a run holds a lock nothing at all changes. The one
        /// atomic primitive every deliberate death is built on.
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
        /// that made it, and the copies runs made of it. Once: a second call releases nothing, and
        /// neither does a call after <see cref="HandedToBackend"/>.
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
            try
            {
                ReleaseMemory();
            }
            finally
            {
                RetireCopies();
                // After the release, so a waiter that sees this has seen the memory go.
                reclaimed?.TrySetResult();
            }
        }

        /// <summary>
        /// Records that the memory a successful <see cref="TryTake"/> handed over went to a backend,
        /// which releases it itself — a run that consumed this tensor, after its call into the
        /// backend returned or threw. Nothing here touches that memory; the copies runs made of this
        /// tensor are this tensor's own business still, and go now.
        /// </summary>
        internal void HandedToBackend()
        {
            TaskCompletionSource? reclaimed;
            lock (_gate)
            {
                if (!_taken || _released) return;
                _released = true;
                reclaimed = _reclaimed;
            }
            RetireCopies();
            // The backend released the memory before its run returned or threw, which is when this
            // is called, so a waiter that sees this has seen the memory go.
            reclaimed?.TrySetResult();
        }

        /// <summary>
        /// Ends a copy that its source no longer wants — the source was written to, or died. It is
        /// marked dead with <paramref name="death"/> and its memory released now, or when the last
        /// run reading it returns; unlike a deletion it asks no run to stop, since a run reading the
        /// old contents is reading what it was fed. A tensor already dead is left as it is: whoever
        /// ended it deals with its memory.
        /// </summary>
        internal void Retire(TensorDeath death)
        {
            bool releaseNow = false;
            lock (_gate)
            {
                if (_death is not null) return;
                _death = death;
                if (_locks == 0)
                {
                    _taken = true;
                    releaseNow = true;
                }
            }
            if (releaseNow) ReleaseTaken();
        }

        /// <summary>
        /// Takes a reader lock for the length of a run, and hands back the signal a deliberate
        /// delete raises. Refuses a tensor that is already dead: there is nothing left to read.
        /// </summary>
        /// <param name="reader">Who is reading, for a refusal of this tensor to name while the lock
        /// is held; null for a holder with nothing to say.</param>
        /// <exception cref="ObjectDisposedException">The tensor is dead.</exception>
        internal CancellationToken AcquireReadLock(object? reader = null)
        {
            lock (_gate)
            {
                if (_death is { } death) throw death.Refusal(this, Describe());
                _locks++;
                if (reader is not null) (_readers ??= []).Add(reader);
                return (_eviction ??= new CancellationTokenSource()).Token;
            }
        }

        /// <summary>
        /// Drops a reader lock. The memory goes now if this was the last lock on a tensor that died
        /// while it was held — deleted, or retired as a copy — which is what such a tensor is
        /// waiting for.
        /// </summary>
        internal void ReleaseReadLock(object? reader = null)
        {
            bool release = false;
            TaskCompletionSource? reclaimed = null;
            lock (_gate)
            {
                _locks--;
                if (reader is not null) _readers?.Remove(reader);
                if (_locks == 0 && _death is not null && !_taken && !_released)
                {
                    _released = true;
                    release = true;
                    reclaimed = _reclaimed;
                }
            }

            if (!release) return;
            try
            {
                ReleaseMemory();
            }
            finally
            {
                RetireCopies();
                // After the release, so a waiter that sees this has seen the memory go.
                reclaimed?.TrySetResult();
            }
        }

        /// <summary>The read a refusal of this tensor names: the first holder of a lock that said
        /// who it was, or null when none did.</summary>
        internal string? DescribeReader()
        {
            lock (_gate) return _readers is { Count: > 0 } readers ? readers[0].ToString() : null;
        }

        /// <summary>What <see cref="Delete"/> throws for a tensor a run is reading.</summary>
        private InvalidOperationException ReadByARun(string operation) => new(
            $"Tensor {this} is being read by {DescribeReader() ?? "a run"}, so {operation} cannot "
            + "release its memory: the run is reading it. Wait for the run to return, use "
            + "TryDelete() to delete it only if nothing is reading it, or DeleteAsync(timeout) to ask "
            + "the run to stop.");

        // ---- Copies made for runs ----

        /// <summary>
        /// Whether no copy of this tensor made for a run is held — the seam a test needs to see that
        /// a release freed the copies rather than merely forgetting the tensor.
        /// </summary>
        internal bool CopiesAreEmpty => Volatile.Read(ref _copies) is not { IsEmpty: false };

        /// <summary>
        /// The copy of this tensor held for runs in memory at <paramref name="where"/>, made by
        /// <paramref name="build"/> the first time one is asked for there and kept for the next —
        /// which is what a shared read of memory a run cannot read as it stands reuses.
        ///
        /// <para>It is a tensor in its own right: one allocation, allocated by the backend that
        /// built it and released through it, with a life and a reader lock of its own. It lives as
        /// long as this tensor, and a write to this tensor retires it. A copy that has died some
        /// other way — deleted by someone who found it on a context's list — is replaced.</para>
        ///
        /// <para>The caller vouches for this tensor's memory: a run holding its reader lock, or a
        /// caller that has just checked it is alive. <paramref name="build"/> reads the contents
        /// without a liveness check, since a run may go on reading a tensor deleted under its lock.
        /// </para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor's memory was released while the
        /// copy was being made, by a caller that held no lock on it.</exception>
        internal TensorData CopyAt(MemoryLocation where, Func<TensorData> build)
        {
            ArgumentNullException.ThrowIfNull(build);
            var copies = RunCopiesOf();
            TensorData made;
            lock (copies)
            {
                if (copies.TryGet(where, out var existing) && !existing.IsDisposed) return existing;
                made = build();
                copies.Set(where, made);
            }
            // A release that ran while this was being built found nothing to retire, and would leave
            // the copy outliving the memory it was copied from. Only an unlocked caller racing a
            // delete can get here, and it is told the tensor is gone.
            bool released;
            lock (_gate) released = _released;
            if (released)
            {
                RetireCopies();
                ThrowIfDisposed();
            }
            return made;
        }

        /// <summary>
        /// The live copy held for runs at <paramref name="where"/>, or null when there is none —
        /// without making one. What a run planning its device memory asks, to know whether a read
        /// there will reuse memory already held or allocate more.
        /// </summary>
        internal TensorData? CopyHeldAt(MemoryLocation where)
        {
            var copies = Volatile.Read(ref _copies);
            if (copies is null) return null;
            lock (copies) return copies.TryGet(where, out var copy) && !copy.IsDisposed ? copy : null;
        }

        /// <summary>
        /// Takes the copy held at <paramref name="where"/> for a run that is consuming this tensor,
        /// marking it dead with <paramref name="death"/>, or null when there is none it can take.
        /// This tensor has already been taken by that run, so nothing else can be reading it
        /// through the copy.
        /// </summary>
        internal TensorData? TakeCopyAt(MemoryLocation where, TensorDeath death)
        {
            var copies = Volatile.Read(ref _copies);
            if (copies is null) return null;
            TensorData? copy;
            lock (copies)
            {
                if (!copies.TryGet(where, out copy)) return null;
                copies.Remove(where);
            }
            if (copy.TryTake(death) == TakeOutcome.Taken) return copy;
            // Still being read by a run that has let go of this tensor but not of the copy yet: it
            // goes when that run returns, and the consuming run makes a copy of its own.
            copy.Retire(TensorDeath.Retired);
            return null;
        }

        /// <summary>Records that <paramref name="sequence"/> holds this tensor as one of its own
        /// elements, so that a write to this tensor reaches the copies runs built of it.</summary>
        internal void BelongsTo(TensorDataSequence sequence) => Volatile.Write(ref _sequence, sequence);

        /// <summary>
        /// Called by every accessor that hands out a writable view of the contents, before it does:
        /// the copies runs made from these contents are stale from here — this tensor's own, and
        /// those of the sequence it is an element of — and are retired, so the next run reads what
        /// was written.
        /// </summary>
        private protected void Written()
        {
            RetireCopies();
            Volatile.Read(ref _sequence)?.ElementWritten();
        }

        /// <summary>
        /// Retires every copy runs made of this tensor, because the contents they were copied from
        /// are about to change or are gone. Each is dead from here — the next run makes a fresh one
        /// — and released once the last run reading it returns.
        /// </summary>
        private protected void RetireCopies()
        {
            var copies = Volatile.Read(ref _copies);
            if (copies is null) return;
            List<TensorData> retired;
            lock (copies) retired = copies.TakeAll();
            foreach (var copy in retired) copy.Retire(TensorDeath.Retired);
        }

        private RunCopies RunCopiesOf()
            => Volatile.Read(ref _copies)
               ?? Interlocked.CompareExchange(ref _copies, new RunCopies(), null)
               ?? _copies!;

        /// <summary>The copies of one tensor, by the memory each is in. Locked on itself, which also
        /// serializes building them: two runs asking for the same copy at once build it once.</summary>
        private sealed class RunCopies
        {
            private Dictionary<MemoryLocation, TensorData>? _byLocation;

            internal bool IsEmpty
            {
                get { lock (this) return _byLocation is not { Count: > 0 }; }
            }

            internal bool TryGet(MemoryLocation where, out TensorData copy)
            {
                copy = null!;
                return _byLocation is not null && _byLocation.TryGetValue(where, out copy!);
            }

            internal void Set(MemoryLocation where, TensorData copy)
            {
                _byLocation ??= [];
                if (_byLocation.TryGetValue(where, out var replaced) && !ReferenceEquals(replaced, copy))
                    replaced.Retire(TensorDeath.Retired);
                _byLocation[where] = copy;
            }

            internal void Remove(MemoryLocation where) => _byLocation?.Remove(where);

            internal List<TensorData> TakeAll()
            {
                if (_byLocation is null) return [];
                List<TensorData> all = [.. _byLocation.Values];
                _byLocation = null;
                return all;
            }
        }
    }

    /// <summary>What <see cref="TensorData.TryTake"/> did.</summary>
    internal enum TakeOutcome
    {
        /// <summary>The tensor is now dead, and its memory is the caller's to deal with.</summary>
        Taken,
        /// <summary>A run is reading the tensor; nothing changed.</summary>
        Locked,
        /// <summary>The tensor was already dead; nothing changed.</summary>
        Dead,
    }

    /// <summary>
    /// Why a tensor or a sequence died, and the words every refused access says it in.
    ///
    /// <para>The message matters more than it looks. A tensor that died of something the caller
    /// did not see — consumed by a run it was fed to as it is, moved into an attribute by a call that
    /// took it — throws far from the cause, and the exception is the only thing that can say what
    /// the cause was and what to do instead. For consumption above all: a tensor fed as it is is
    /// consumed by default, and this message is what tells a caller who meant to use it again how
    /// to say so.</para>
    /// </summary>
    internal sealed class TensorDeath
    {
        private readonly Func<string, string> _explain;

        private TensorDeath(string cause, Func<string, string> explain)
        {
            Cause = cause;
            _explain = explain;
        }

        /// <summary>A short name for the cause, for a diagnostic that wants one.</summary>
        internal string Cause { get; }

        /// <summary>Deleted: <c>Delete</c>, <c>Dispose</c>, <c>TryDelete</c> or
        /// <c>DeleteAsync</c>.</summary>
        internal static TensorDeath Deleted { get; } = new("deleted", what =>
            $"{what} has been deleted -- by Delete(), Dispose(), TryDelete() or DeleteAsync() -- so "
            + "its memory is gone and nothing may read it.");

        /// <summary>A sequence ended by its <c>Dispose</c>.</summary>
        internal static TensorDeath Disposed { get; } = new("disposed", what =>
            $"{what} has been disposed, so its storage is gone and nothing may read it.");

        /// <summary>Moved into a <see cref="TensorAttribute"/> by
        /// <see cref="TensorData.MoveToAttribute"/>.</summary>
        internal static TensorDeath MovedToAttribute { get; } = new("moved into an attribute", what =>
            $"{what} was moved into a TensorAttribute by MoveToAttribute(), which took its "
            + "contents, so nothing may read it any more. Read the attribute instead, or take a "
            + "CopyTo(...) before the move where both are needed.");

        /// <summary>A copy a run made of a tensor it could not read where it was, retired because
        /// that tensor was written to or ended.</summary>
        internal static TensorDeath Retired { get; } = new("retired", what =>
            $"{what} was a copy a run made of another tensor, in memory the run could read, and was "
            + "released because the tensor it was copied from was written to or ended. Nothing may "
            + "read it any more; read the tensor it was copied from instead.");

        /// <summary>
        /// Consumed by <paramref name="run"/>, which it fed as <paramref name="inputs"/> — fed as it
        /// is, or through <c>.TryConsume()</c> when <paramref name="tried"/>, with nothing else
        /// reading it.
        /// </summary>
        /// <param name="run">The run, as a caller can recognise it: which graph, on which context.
        /// Described when the message is, not before.</param>
        /// <param name="inputs">The input or inputs it fed, already quoted.</param>
        /// <param name="tried">Whether it was fed as <c>.TryConsume()</c> rather than as it is.</param>
        internal static TensorDeath ConsumedBy(object run, string inputs, bool tried)
            => new("consumed by a run", what => tried
                ? $"{what} was consumed by {run}, which it fed as {inputs}: it was passed as "
                  + ".TryConsume() and nothing else was reading it when that run started, so the run "
                  + "took its memory and nothing may read it any more. To use it after that call, "
                  + "pass it there as .Shared() instead, and the run will only read it."
                : $"{what} was consumed by {run}, which it fed as {inputs}: fed to a run as it is, "
                  + "it is given to that run, which takes its memory, so nothing may read it any "
                  + "more. To use it after that call, pass it there as .Shared() -- or pass the "
                  + "struct, sequence or checkpoint that held it that way -- and the run will only "
                  + "read it.");

        /// <summary>The exception an access to <paramref name="subject"/>, described as
        /// <paramref name="what"/>, throws.</summary>
        internal ObjectDisposedException Refusal(object subject, string what)
            => new(subject.GetType().Name, _explain(what));

        /// <inheritdoc/>
        public override string ToString() => Cause;
    }
}
