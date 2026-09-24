using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using Shorokoo.Core.Backends;

namespace Shorokoo
{
    /// <summary>
    /// Who holds a reader lock taken outside any run, as a refusal of the tensor or sequence names
    /// it: a copy being made out of its memory, or a write into it. Either way the memory may not
    /// end while it is held.
    /// </summary>
    internal sealed class OutsideARun
    {
        private readonly string _what;

        private OutsideARun(string what) => _what = what;

        internal static OutsideARun CopyingOut { get; } = new("a copy being made of its contents");

        internal static OutsideARun Writing { get; } = new("a write into its contents");

        /// <inheritdoc/>
        public override string ToString() => _what;
    }

    /// <summary>
    /// What a <see cref="Lifetime"/> belongs to: a tensor or a sequence, each one allocation. The
    /// lifetime decides when the allocation ends; the owner knows what ending it releases.
    /// </summary>
    internal interface ILifetimeOwner
    {
        /// <summary>The owner's life.</summary>
        Lifetime Life { get; }

        /// <summary>How a refusal names the owner: "Tensor (4,):Float32".</summary>
        string Describe();

        /// <summary>Releases the owner's own memory, through the backend that made it. Called at
        /// most once, by whoever the memory belongs to when the owner dies; never for memory a run
        /// consumed, which its backend releases.</summary>
        void ReleaseOwnMemory();

        /// <summary>Retires every copy runs made of the owner, which live no longer than it.</summary>
        void RetireRunCopies();
    }

    /// <summary>
    /// The life of one allocation — a tensor's, or a sequence's: whether it is alive, why it died,
    /// the reader locks the runs reading it hold, and who deals with its memory when it ends.
    ///
    /// <para>This is the allocation's own state and nobody else's. There is no second object naming
    /// the same memory and no count of names, so the one question it answers is whether the
    /// allocation may still be read, and the one decision it makes is who ends that.
    /// <see cref="TryTake"/> is the single atomic way to end it deliberately; deletion, consumption
    /// by a run and a move into an attribute are each a take plus what the taker does with the
    /// memory. <see cref="DeleteAsync"/> alone, which does not wait for the readers to finish, marks
    /// the allocation dead at once and leaves the memory to the last of them.</para>
    ///
    /// <para>Locked on itself, per allocation, and taken by nothing else. Deliberately not a
    /// process-wide gate: the lock and unlock path runs on every feed of every run, and a delete has
    /// to be able to wait for readers while holding nothing at all.</para>
    /// </summary>
    internal sealed class Lifetime
    {
        private readonly ILifetimeOwner _owner;

        // Null while alive; set once, under the lock, to the reason it died.
        private TensorDeath? _death;

        // Whether a take handed the memory to a caller who is now responsible for it -- a run that
        // consumed it, or the move into an attribute. Distinguishes an allocation whose memory is
        // still its own to release (deleted while a run was reading it) from one whose memory has
        // gone elsewhere.
        private bool _taken;

        // Whether the memory has been dealt with -- released, or handed to a backend that releases
        // it itself -- which happens exactly once.
        private bool _released;

        // Reader locks: how many readers hold one right now.
        private int _locks;

        // Who holds those locks, where the holder said: what a refusal names as the read it would
        // have taken the memory from under. Holders that did not say are counted and not listed.
        private List<object>? _readers;

        // Created on the first lock and never disposed: it is the signal a deliberate delete raises,
        // and a run registers on it for as long as it holds its lock. An allocation never read by a
        // run never has one.
        private CancellationTokenSource? _eviction;

        // Completed once the memory has been released, which is what DeleteAsync waits on. Created
        // only when a delete finds the memory still out.
        private TaskCompletionSource? _reclaimed;

        internal Lifetime(ILifetimeOwner owner) => _owner = owner;

        /// <summary>Why the allocation died, or null while it is alive.</summary>
        internal TensorDeath? Death => Volatile.Read(ref _death);

        /// <summary>Whether a reader holds a lock right now.</summary>
        internal bool IsLocked
        {
            get { lock (this) return _locks > 0; }
        }

        /// <summary>Whether the memory has been dealt with: released, or handed to a backend.</summary>
        internal bool IsReleased
        {
            get { lock (this) return _released; }
        }

        /// <summary>Throws the refusal of an access to a dead allocation, naming why it died.</summary>
        /// <exception cref="ObjectDisposedException">The allocation is dead.</exception>
        internal void ThrowIfDead()
        {
            if (Volatile.Read(ref _death) is { } death) throw death.Refusal(_owner, _owner.Describe());
        }

        /// <summary>
        /// Ends the allocation's life deliberately, if no reader holds a lock: marks it dead with
        /// <paramref name="death"/> and hands its memory to the caller, who must either release it
        /// with <see cref="ReleaseTaken"/> or hand it to a backend and say so with
        /// <see cref="HandedToBackend"/>. If a reader holds a lock nothing at all changes.
        /// </summary>
        internal TakeOutcome TryTake(TensorDeath death)
        {
            ArgumentNullException.ThrowIfNull(death);
            lock (this)
            {
                if (_death is not null) return TakeOutcome.Dead;
                if (_locks > 0) return TakeOutcome.Locked;
                _death = death;
                _taken = true;
                return TakeOutcome.Taken;
            }
        }

        /// <summary>
        /// Releases the memory a successful <see cref="TryTake"/> handed over, through the owner,
        /// and retires the copies runs made of it. Once: a second call releases nothing, and neither
        /// does a call after <see cref="HandedToBackend"/>.
        /// </summary>
        internal void ReleaseTaken()
        {
            TaskCompletionSource? reclaimed;
            lock (this)
            {
                if (!_taken || _released) return;
                _released = true;
                reclaimed = _reclaimed;
            }
            Release(reclaimed);
        }

        /// <summary>
        /// Records that the memory a successful <see cref="TryTake"/> handed over went to a backend,
        /// which releases it itself — a run that consumed it, after its call into the backend
        /// returned or threw. Nothing here touches that memory; the copies runs made of it are the
        /// owner's business still, and go now.
        /// </summary>
        internal void HandedToBackend()
        {
            TaskCompletionSource? reclaimed;
            lock (this)
            {
                if (!_taken || _released) return;
                _released = true;
                reclaimed = _reclaimed;
            }
            try
            {
                _owner.RetireRunCopies();
            }
            finally
            {
                // The backend released the memory before its run returned or threw, which is when
                // this is called, so a waiter that sees this has seen the memory go.
                reclaimed?.TrySetResult();
            }
        }

        /// <summary>
        /// Ends a copy its source no longer wants — the source was written to, or died. It is marked
        /// dead with <paramref name="death"/> and its memory released now, or when the last reader
        /// returns; unlike a deletion it asks no run to stop, since a run reading the old contents
        /// is reading what it was fed. An allocation already dead is left as it is: whoever ended it
        /// deals with its memory.
        /// </summary>
        internal void Retire(TensorDeath death)
        {
            bool releaseNow = false;
            lock (this)
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
        /// Deletes the allocation and waits up to <paramref name="timeout"/> for its memory to be
        /// released: marks it dead now, asks every run holding a lock to stop, and releases the
        /// memory when the last of them has. An allocation that has already died another way is
        /// not deleted again; the wait is for whoever holds its memory now to release it. See
        /// <c>TensorData.DeleteAsync</c>.
        /// </summary>
        internal async Task<bool> DeleteAsync(TimeSpan timeout, CancellationToken cancellationToken)
        {
            bool releaseNow = false;
            CancellationTokenSource? eviction = null;
            Task? reclaimed = null;
            lock (this)
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

            // Outside the lock: cancelling runs every reader's callback on this thread, and a reader
            // standing down takes the lock to drop its own.
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
        /// Takes a reader lock, and hands back the signal a deliberate delete raises. Refuses an
        /// allocation that is already dead: there is nothing left to read.
        /// </summary>
        /// <param name="reader">Who is reading, for a refusal to name while the lock is held; null
        /// for a holder with nothing to say.</param>
        /// <exception cref="ObjectDisposedException">The allocation is dead.</exception>
        internal CancellationToken AcquireReadLock(object? reader)
        {
            lock (this)
            {
                if (_death is { } death) throw death.Refusal(_owner, _owner.Describe());
                _locks++;
                if (reader is not null) (_readers ??= []).Add(reader);
                return (_eviction ??= new CancellationTokenSource()).Token;
            }
        }

        /// <summary>
        /// Drops a reader lock. The memory goes now if this was the last lock on an allocation that
        /// died while it was held — deleted, or retired as a copy — which is what it is waiting for.
        /// </summary>
        internal void ReleaseReadLock(object? reader)
        {
            TaskCompletionSource? reclaimed;
            lock (this)
            {
                _locks--;
                if (reader is not null) _readers?.Remove(reader);
                if (_locks != 0 || _death is null || _taken || _released) return;
                _released = true;
                reclaimed = _reclaimed;
            }
            Release(reclaimed);
        }

        /// <summary>The read a refusal names: the first holder of a lock that said who it was, or
        /// null when none did.</summary>
        internal string? DescribeReader()
        {
            lock (this) return _readers is { Count: > 0 } readers ? readers[0].ToString() : null;
        }

        /// <summary>Releases the owner's memory and retires its copies, then tells a waiting delete
        /// the memory is back — after the release, so a waiter that sees it has seen the memory
        /// go.</summary>
        private void Release(TaskCompletionSource? reclaimed)
        {
            try
            {
                _owner.ReleaseOwnMemory();
            }
            finally
            {
                try
                {
                    _owner.RetireRunCopies();
                }
                finally
                {
                    reclaimed?.TrySetResult();
                }
            }
        }
    }

    /// <summary>
    /// The copies runs made of one allocation in memory they could read where they could not read it
    /// as it stands, one per place they are in. Each copy is an allocation in its own right, with a
    /// life of its own, that lives no longer than its source. Locked on itself, which also
    /// serializes building them: two runs asking for the same copy at once build it once.
    /// </summary>
    internal sealed class RunCopies<TCopy> where TCopy : class, ILifetimeOwner
    {
        private Dictionary<MemoryLocation, TCopy>? _byLocation;

        /// <summary>Whether no copy is held.</summary>
        internal bool IsEmpty
        {
            get { lock (this) return _byLocation is not { Count: > 0 }; }
        }

        /// <summary>
        /// The copy held at <paramref name="where"/>, made by <paramref name="build"/> the first
        /// time one is asked for there and kept for the next. A copy that has died some other way —
        /// deleted by someone who found it on a context's list — is replaced.
        ///
        /// <para>The caller vouches for the source's memory: a reader holding its lock, or a caller
        /// that has just checked it is alive. A release of the source that comes while the copy is
        /// being built waits for it, on this object, and retires it. One that finished before the
        /// copy was begun found nothing to retire, and would leave a copy read from memory already
        /// released held with nothing to end it; only a caller holding no lock can get there, and
        /// the copy is retired and the caller told the source is gone.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">The source's memory was released while the copy
        /// was being made.</exception>
        internal TCopy CopyAt(MemoryLocation where, Func<TCopy> build, Lifetime source)
        {
            ArgumentNullException.ThrowIfNull(build);
            TCopy made;
            lock (this)
            {
                if (TryGetLive(where) is { } existing) return existing;
                made = build();
                Set(where, made);
            }
            if (source.IsReleased)
            {
                RetireAll();
                source.ThrowIfDead();
            }
            return made;
        }

        /// <summary>The live copy held at <paramref name="where"/>, or null when there is none —
        /// without making one.</summary>
        internal TCopy? HeldAt(MemoryLocation where)
        {
            lock (this) return TryGetLive(where);
        }

        /// <summary>
        /// Takes the copy held at <paramref name="where"/> for a run that has taken the source, marking
        /// it dead with <paramref name="death"/>, or null when there is none it can take. No run can
        /// be reading it — a run lets go of a copy before the source it read through it — but a
        /// caller outside one can, having found it on a context's list: a copy being read that way is
        /// retired instead, going when that caller is done, and the consuming run makes one of its
        /// own.
        /// </summary>
        internal TCopy? Take(MemoryLocation where, TensorDeath death)
        {
            TCopy? copy;
            lock (this)
            {
                if (_byLocation is null || !_byLocation.Remove(where, out copy)) return null;
            }
            if (copy.Life.TryTake(death) == TakeOutcome.Taken) return copy;
            copy.Life.Retire(TensorDeath.Retired);
            return null;
        }

        /// <summary>Retires every copy held, because the contents they were copied from are about
        /// to change or are gone. Each is dead from here and released once its last reader
        /// returns: every one of them, even where releasing one throws, which is thrown once all
        /// are retired.</summary>
        internal void RetireAll()
        {
            List<TCopy> retired;
            lock (this)
            {
                if (_byLocation is null) return;
                retired = [.. _byLocation.Values];
                _byLocation = null;
            }
            // Every one of them, whatever one release does: a copy left unretired would stay on its
            // context's books, and in its memory, with nothing left that knows it is there.
            Exception? failed = null;
            foreach (var copy in retired)
            {
                try
                {
                    copy.Life.Retire(TensorDeath.Retired);
                }
                catch (Exception e)
                {
                    failed ??= e;
                }
            }
            if (failed is not null) ExceptionDispatchInfo.Throw(failed);
        }

        private TCopy? TryGetLive(MemoryLocation where)
            => _byLocation is not null && _byLocation.TryGetValue(where, out var copy) && copy.Life.Death is null
                ? copy : null;

        private void Set(MemoryLocation where, TCopy copy)
        {
            _byLocation ??= [];
            if (_byLocation.TryGetValue(where, out var replaced) && !ReferenceEquals(replaced, copy))
                replaced.Life.Retire(TensorDeath.Retired);
            _byLocation[where] = copy;
        }
    }
}
