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
    /// <para>This is the tensor's own state and nobody else's, kept by its <see cref="Lifetime"/>
    /// — the same machinery a sequence keeps its own life with. A tensor is its allocation — there
    /// is no second object naming the same memory and no count of names — so the one question the
    /// state answers is whether this tensor may still be read, and the one decision it makes is
    /// who ends that. <see cref="TryTake"/> is the single atomic way to end it deliberately; every
    /// deliberate death — deletion, consumption by a run, a move into an attribute — is a take plus
    /// what the taker does with the memory.</para>
    /// </summary>
    public abstract partial class TensorData : ILifetimeOwner
    {
        private readonly Lifetime _life;

        // The copies runs made of this tensor in memory they could read, one per place they are in.
        // Created on the first copy: most tensors are read where they are, or never read at all.
        private RunCopies<TensorData>? _copies;

        // The sequence this tensor is an element of, where it is one of a list sequence's own: the
        // sequence values runs built from that sequence were copied from this tensor too, so a
        // write here retires them as well as this tensor's own copies.
        private TensorDataSequence? _sequence;

        /// <inheritdoc/>
        Lifetime ILifetimeOwner.Life => _life;

        /// <inheritdoc/>
        void ILifetimeOwner.ReleaseOwnMemory() => ReleaseMemory();

        /// <inheritdoc/>
        void ILifetimeOwner.RetireRunCopies() => RetireCopies();

        /// <inheritdoc/>
        string ILifetimeOwner.Describe() => Describe();

        /// <summary>
        /// True once this tensor is dead — deleted, consumed by a run, or moved into an attribute.
        /// Its shape, dtype, <see cref="ToString"/> and where its memory was stay readable as
        /// metadata; every other access throws, saying how it died.
        /// </summary>
        public bool IsDisposed => _life.Death is not null;

        /// <summary>Whether a run is reading this tensor right now — what <see cref="Delete"/>
        /// refuses on and <see cref="TryDelete"/> declines on.</summary>
        internal bool IsLocked => _life.IsLocked;

        /// <summary>How a refusal names this tensor: "Tensor (4,):Float32".</summary>
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
        protected internal void ThrowIfDisposed() => _life.ThrowIfDead();

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
        public Task<bool> DeleteAsync(TimeSpan timeout, CancellationToken cancellationToken = default)
            => _life.DeleteAsync(timeout, cancellationToken);

        /// <summary>
        /// Ends this tensor's life deliberately, if no run is reading it: marks it dead with
        /// <paramref name="death"/> and hands its memory to the caller, who must either release it
        /// with <see cref="ReleaseTaken"/> or hand it to a backend and say so with
        /// <see cref="Lifetime.HandedToBackend"/>. If a run holds a lock nothing at all changes. The
        /// atomic primitive deleting, consuming and moving a tensor into an attribute are built on;
        /// only <see cref="DeleteAsync"/>, which does not wait for the lock to be free, marks a tensor
        /// dead without it.
        /// </summary>
        internal TakeOutcome TryTake(TensorDeath death) => _life.TryTake(death);

        /// <summary>
        /// Releases the memory a successful <see cref="TryTake"/> handed over, through the backend
        /// that made it, and the copies runs made of it. Once: a second call releases nothing, and
        /// neither does a call after <see cref="Lifetime.HandedToBackend"/>.
        /// </summary>
        internal void ReleaseTaken() => _life.ReleaseTaken();

        /// <summary>The read a refusal of this tensor names: the first holder of a lock that said
        /// who it was, or null when none did.</summary>
        internal string? DescribeReader() => _life.DescribeReader();

        /// <summary>What <see cref="Delete"/> throws for a tensor a run is reading.</summary>
        private InvalidOperationException ReadByARun(string operation) => new(
            $"Tensor {this} is being read by {DescribeReader() ?? "a run"}, so {operation} cannot "
            + "release its memory under that read. Wait for the read to end, use TryDelete() to "
            + "delete it only if nothing is reading it, or DeleteAsync(timeout) to ask a run reading "
            + "it to stop.");

        /// <summary>
        /// Runs <paramref name="read"/> — a copy out of this tensor's memory made outside any run —
        /// under a reader lock, so nothing ends that memory while it is read: a run that would
        /// consume the tensor is refused, and a delete refused or held back, exactly as while a run
        /// reads it. Refuses a tensor that is already dead.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The tensor is dead.</exception>
        private protected T Reading<T>(Func<T> read)
        {
            _life.AcquireReadLock(CopyingOut.Reader);
            try
            {
                return read();
            }
            finally
            {
                _life.ReleaseReadLock(CopyingOut.Reader);
            }
        }

        /// <summary>Who holds the lock <see cref="Reading"/> takes, as a refusal names it.</summary>
        private sealed class CopyingOut
        {
            internal static CopyingOut Reader { get; } = new();

            public override string ToString() => "a copy being made of its contents";
        }

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
            => RunCopiesOf().CopyAt(where, build, _life);

        /// <summary>
        /// The live copy held for runs at <paramref name="where"/>, or null when there is none —
        /// without making one. What a run planning its device memory asks, to know whether a read
        /// there will reuse memory already held or allocate more.
        /// </summary>
        internal TensorData? CopyHeldAt(MemoryLocation where) => Volatile.Read(ref _copies)?.HeldAt(where);

        /// <summary>
        /// Takes the copy held at <paramref name="where"/> for a run that is consuming this tensor,
        /// marking it dead with <paramref name="death"/>, or null when there is none it can take.
        /// This tensor has already been taken by that run, so nothing else can be reading it
        /// through the copy.
        /// </summary>
        internal TensorData? TakeCopyAt(MemoryLocation where, TensorDeath death)
            => Volatile.Read(ref _copies)?.Take(where, death);

        /// <summary>
        /// Retires every copy runs made of this tensor, as a write would, for a caller that knows the
        /// copies will not be read again soon: a training step letting go of the copies of the batch
        /// it read, which would otherwise stay in the run's memory for as long as the batch lives.
        /// </summary>
        internal void ReleaseRunCopies() => RetireCopies();

        /// <summary>Records that <paramref name="sequence"/> holds this tensor as one of its own
        /// elements, so that a write to this tensor reaches the copies runs built of it.</summary>
        internal void BelongsTo(TensorDataSequence sequence) => Volatile.Write(ref _sequence, sequence);

        /// <summary>Records that <paramref name="sequence"/>, which died while a run was reading this
        /// tensor on its own account, holds it no longer: the tensor lives on by itself.</summary>
        internal void Leaves(TensorDataSequence sequence) => Interlocked.CompareExchange(ref _sequence, null, sequence);

        /// <summary>
        /// A new value of <paramref name="backend"/>'s runtime, in host memory, holding this tensor's
        /// contents as they stand — the caller's to own. What a sequence value is built from, element
        /// by element. Without the liveness check: the caller holds this tensor, by a reader lock or
        /// by having taken it.
        /// </summary>
        internal IShorokooTensorValue HostCopyOn(IShorokooBackend backend)
            => DType.IsSameElementTypeAs(DType.Utf8)
                ? backend.CreateStringTensor(CopyContentStrings(), (long[])Shape)
                : backend.CreateTensorFromRawBytes(
                    (ShorokooTensorElementType)(int)DType, ContentBytesForCopy(), (long[])Shape);

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
        private protected void RetireCopies() => Volatile.Read(ref _copies)?.RetireAll();

        private RunCopies<TensorData> RunCopiesOf()
            => Volatile.Read(ref _copies)
               ?? Interlocked.CompareExchange(ref _copies, new RunCopies<TensorData>(), null)
               ?? _copies!;
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
                  + "pass it there as .Shared() instead -- or pass the struct, sequence or checkpoint "
                  + "that held it that way -- and the run will only read it."
                : $"{what} was consumed by {run}, which it fed as {inputs}: fed to a run as it is, "
                  + "it is given to that run, which takes its memory, so nothing may read it any "
                  + "more. To use it after that call, pass it there as .Shared() -- or pass the "
                  + "struct, sequence or checkpoint that held it that way -- and the run will only "
                  + "read it.");

        /// <summary>The exception an access to <paramref name="subject"/>, described as
        /// <paramref name="what"/>, throws: named by the public type a caller holds, not by the
        /// class behind it.</summary>
        internal ObjectDisposedException Refusal(object subject, string what)
            => new(subject is TensorDataSequence ? nameof(TensorDataSequence) : nameof(TensorData), _explain(what));

        /// <inheritdoc/>
        public override string ToString() => Cause;
    }
}
