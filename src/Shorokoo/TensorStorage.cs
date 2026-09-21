using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo
{
    /// <summary>
    /// An allocation: the bytes behind one or more tensors, the memory device they are in, and a
    /// count of what still names them.
    ///
    /// <para>Separate from <see cref="TensorData"/> because more than one tensor may name the same
    /// bytes: a same-space <c>TransferTo</c> re-wraps rather than copying, and <c>GiveAccessTo</c>
    /// exists precisely to hand out a second name for memory another tensor already holds. Two
    /// kinds of thing hold a reference — every <see cref="TensorData"/> handle naming the
    /// allocation, and every lock a compute context holds on it while it reads it. At zero the
    /// memory is freed, immediately rather than at a collection.</para>
    ///
    /// <para>Which is why liveness lives here rather than on the tensor. A handle has no way to
    /// know that the allocation has been deleted, and a raw pointer does not stop working when its
    /// memory is freed — it starts reading something else. <see cref="IsLive"/> turns that into an
    /// exception. It costs one predictable branch per access, against a class of bug that cannot
    /// be found from the outside at all.</para>
    ///
    /// <para>Deletion (<see cref="TryDelete"/>, <see cref="DeleteAsync"/>) ignores the handle count
    /// — that is what makes it deletion rather than disposal — and never ignores the lock count:
    /// the bytes go back to the allocator only once the last reader has stood down. See
    /// <see cref="TensorData.DeleteAsync"/> for what a caller is promised.</para>
    /// </summary>
    internal sealed class TensorStorage
    {
        // Per allocation, and taken by nothing else. Deliberately not a process-wide gate: the
        // lock/unlock path runs on every feed of every run, and a delete has to be able to wait
        // for lockers while holding nothing at all.
        private readonly object _gate = new();

        // Null once the bytes have gone back to whatever allocator they came from, and null from
        // the start for a storage that holds nothing (see None).
        private Action? _release;
        private bool _freed;

        // Handles plus locks. The handle count alone cannot decide a free: a run holding a lock is
        // reading these bytes whether or not the caller still has a name for them, which is the
        // whole of Shorokoo/Shorokoo#366.
        private int _refs;
        private int _locks;

        private bool _deleted;

        // Created on the first lock and never disposed: it is the signal a deliberate delete
        // raises, and a locker registers on it for as long as it holds its lease. A storage that
        // is never fed to a run never has one.
        private CancellationTokenSource? _eviction;

        // Completed once a delete's bytes have actually been reclaimed, which is what DeleteAsync
        // waits on. Created only when a delete finds a lock held.
        private TaskCompletionSource? _reclaimed;

        // Frees that arrived while a lock was held: a write through AccessModifiable... retires
        // the runtime values built from these bytes, and freeing them under a running read is the
        // same use-after-free as freeing the bytes. Run when the last lock drops.
        private List<Action>? _deferred;

        internal TensorStorage(MemorySpace space, Action release)
            : this(space, release ?? throw new ArgumentNullException(nameof(release)), holdsNothing: false)
        {
        }

        private TensorStorage(MemorySpace space, Action? release, bool holdsNothing)
        {
            Device = MemoryDevice.For(space);
            _holdsNothing = holdsNothing;
            _release = holdsNothing ? null : release;
        }

        // Whether there are no bytes here at all. Such a storage is shared (see None), so it can
        // neither be freed nor deleted: one tensor deleting it would strike every other tensor
        // over it dead, process-wide.
        private readonly bool _holdsNothing;

        /// <summary>
        /// Storage that holds nothing and releases nothing — a metadata-only tensor's. Shared by
        /// every such tensor, so it is never freed and <see cref="IsLive"/> is always true: one
        /// tensor letting go of it must not make every other one unreadable.
        /// </summary>
        internal static TensorStorage None { get; } = new(MemorySpace.Host, null, holdsNothing: true);

        /// <summary>The memory these bytes are in, shared with every other allocation in the same
        /// <see cref="MemorySpace"/>.</summary>
        internal MemoryDevice Device { get; }

        /// <summary>Where these bytes are — <see cref="MemoryDevice.Space"/>, kept here for
        /// call-site brevity.</summary>
        internal MemorySpace Space => Device.Space;

        /// <summary>
        /// False once these bytes may no longer be read: freed, or deleted and merely waiting for
        /// the last reader to stand down. Reading them afterwards is a use-after-free, and every
        /// accessor checks this to make it an exception instead.
        ///
        /// <para>Deletion counts from the instant it is asked for rather than from the moment the
        /// bytes go, because those come apart: a delete is immediate and unconditional, and only
        /// the reclamation waits. A handle that went on reading in between would be reading data
        /// its owner has already been told is gone.</para>
        /// </summary>
        internal bool IsLive
        {
            get { lock (_gate) return !_freed && !_deleted; }
        }

        /// <summary>Whether this allocation was deleted, as opposed to having had its last handle
        /// let go of it. Read only to name the cause in the exception a dead read throws.</summary>
        internal bool IsDeleted
        {
            get { lock (_gate) return _deleted; }
        }

        /// <summary>Whether a compute context is reading these bytes right now — the test
        /// <see cref="TryDelete"/> refuses on.</summary>
        internal bool IsLocked
        {
            get { lock (_gate) return _locks > 0; }
        }

        /// <summary>
        /// Whether the asking handle is the only name for these bytes, so taking them away from
        /// the allocation takes them away from nobody else. What a move has to know: surrendering
        /// this handle's name says nothing about another's, and an attribute that took an array a
        /// second handle can still write would not be immutable.
        /// </summary>
        internal bool IsSoleHandle
        {
            get { lock (_gate) return _refs == 1 && _locks == 0; }
        }

        /// <summary>Records one more name for these bytes. Called by every handle as it is
        /// built.</summary>
        internal void AddHandleReference()
        {
            lock (_gate) _refs++;
        }

        /// <summary>
        /// Drops one handle's name for these bytes, freeing them if it was the last reference.
        /// Never affects another handle, and never frees anything a lock is holding.
        /// </summary>
        internal void DropHandleReference()
        {
            Action? free;
            lock (_gate)
            {
                _refs--;
                free = _refs <= 0 && _locks == 0 ? TakeRelease() : null;
            }
            free?.Invoke();
        }

        /// <summary>
        /// Takes a lock for the length of a read, and hands back the signal a deliberate delete
        /// raises. Refuses an allocation that is already dead: there is nothing left to read.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The allocation has been deleted or
        /// freed.</exception>
        internal CancellationToken Lock()
        {
            lock (_gate)
            {
                if (_deleted || _freed)
                    throw new ObjectDisposedException(nameof(TensorStorage),
                        _deleted
                            ? "This allocation has been deleted, so nothing may read it."
                            : "This allocation has been freed, so nothing may read it.");
                _refs++;
                _locks++;
                return (_eviction ??= new CancellationTokenSource()).Token;
            }
        }

        /// <summary>
        /// Drops a lock. The bytes go if this was the last thing holding them, and go regardless
        /// of the handle count if the allocation was deleted while this lock was held — which is
        /// what a deleted-but-not-yet-reclaimed allocation is waiting for.
        /// </summary>
        internal void Unlock()
        {
            Action? free;
            List<Action>? deferred = null;
            TaskCompletionSource? reclaimed = null;
            lock (_gate)
            {
                _refs--;
                _locks--;
                if (_locks == 0)
                {
                    (deferred, _deferred) = (_deferred, null);
                    // Deletion ignores the handle count; ordinary disposal does not.
                    free = _deleted || _refs <= 0 ? TakeRelease() : null;
                    if (_deleted) reclaimed = _reclaimed;
                }
                else free = null;
            }

            free?.Invoke();
            if (deferred is not null) foreach (var pending in deferred) pending();
            // After the free, so a waiter that sees this has seen the bytes go.
            reclaimed?.TrySetResult();
        }

        /// <summary>
        /// Runs <paramref name="free"/> now, or once the last lock on these bytes drops. For a
        /// release that is not the allocation's own — the runtime values built from these
        /// contents, which a write retires — where freeing under a running read is the same fault
        /// as freeing the bytes.
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

        /// <summary>
        /// Deletes these bytes if nothing is reading them: frees them now, marks the allocation
        /// dead and returns true. If any lock is held nothing at all changes — no eviction is
        /// signalled, no run is aborted, every handle stays readable — and it returns false.
        /// </summary>
        internal bool TryDelete()
        {
            if (_holdsNothing) return true;
            Action? free;
            lock (_gate)
            {
                if (_locks > 0) return false;
                _deleted = true;
                free = TakeRelease();
            }
            free?.Invoke();
            return true;
        }

        /// <summary>
        /// Marks the allocation dead at once, signals eviction to every locker, and waits up to
        /// <paramref name="timeout"/> for the bytes to come back. See
        /// <see cref="TensorData.DeleteAsync"/>, which is where the contract is written.
        /// </summary>
        internal async Task<bool> DeleteAsync(TimeSpan timeout, CancellationToken cancellationToken)
        {
            if (_holdsNothing) return true;
            Action? free;
            CancellationTokenSource? eviction;
            Task? reclaimed = null;
            lock (_gate)
            {
                _deleted = true;
                eviction = _eviction;
                if (_locks == 0) free = TakeRelease();
                else
                {
                    free = null;
                    reclaimed = (_reclaimed ??= new TaskCompletionSource(
                        TaskCreationOptions.RunContinuationsAsynchronously)).Task;
                }
            }

            // Outside the gate: cancelling runs every locker's callback on this thread, and a
            // locker standing down takes the gate to drop its lock.
            eviction?.Cancel();

            if (free is not null)
            {
                free();
                return true;
            }
            if (reclaimed is null) return true;
            if (reclaimed.IsCompleted) return true;
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

        /// <summary>The release action, once, marking the bytes gone. Called under
        /// <see cref="_gate"/>; the action itself runs outside it, because it is a native call and
        /// has no business running under a lock a run's feed path takes.</summary>
        private Action? TakeRelease()
        {
            var release = _release;
            _release = null;
            if (release is null) return null;
            _freed = true;
            return release;
        }
    }
}
