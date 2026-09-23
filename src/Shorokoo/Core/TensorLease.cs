using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Shorokoo.Core.Backends;

namespace Shorokoo.Runtime
{
    /// <summary>
    /// A run's reader lock on one tensor or sequence, taken by the compute context running it, from
    /// <c>ComputeContext.Lock</c>. While it is held the tensor cannot be deleted — a delete is
    /// refused, declined, or negotiated through <see cref="Eviction"/> — and the lease holds the
    /// tensor itself, strongly, so nothing a run is reading can be collected under it either.
    ///
    /// <para>The one obligation a holder has is to listen for that signal on everything it has
    /// locked and to stop as soon as any of them is raised. A run discharges it by handing the
    /// linked signal to the backend as <c>RunSettings.CancellationToken</c>; a holder that ignores
    /// it is not unsafe, only slow — the delete then waits for the read to end on its own.</para>
    ///
    /// <para>Disposing is dropping the lock, and it is a count rather than a flag: the same
    /// tensor locked twice is released twice.</para>
    /// </summary>
    internal sealed class TensorLease : IDisposable
    {
        private readonly ComputeContext _holder;

        // Held strongly for as long as the lease is: a run keeps what it reads alive for the whole
        // run, so a tensor with a lock on it is never garbage.
        private readonly object _target;
        private int _released;

        internal TensorLease(ComputeContext holder, TensorData tensor, CancellationToken eviction)
        {
            _holder = holder;
            _target = tensor;
            Eviction = eviction;
        }

        internal TensorLease(ComputeContext holder, TensorDataSequence sequence)
        {
            _holder = holder;
            _target = sequence;
        }

        /// <summary>Raised when something asks for the tensor to be deleted. Stop reading it and
        /// drop this lease: the deleter is waiting for exactly that. A sequence is never deleted
        /// that way, so its lease has no signal to raise.</summary>
        internal CancellationToken Eviction { get; }

        /// <summary>What this lease holds.</summary>
        internal object Target => _target;

        /// <summary>Drops the lock. The memory goes now if the delete it was waiting for has
        /// already been asked for. Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0) return;
            GC.SuppressFinalize(this);
            // The tensor's lock first, then the context's count, and neither under the other: the
            // two gates are never nested, so a delete waiting for this lease cannot end up behind
            // a gate the run is queued on.
            switch (_target)
            {
                case TensorData tensor: tensor.ReleaseReadLock(); break;
                case TensorDataSequence sequence: sequence.ReleaseReadLock(); break;
            }
            _holder.ReleaseLease(_target);
        }

#if DEBUG
        // Collecting a locked tensor is a bug: a run holds a strong reference to everything it
        // reads for its whole duration, through this lease, so the only way a locked tensor becomes
        // garbage is with its lease, undisposed. This is where that is observable. Debug builds only
        // -- the release build carries no finalizer on anything here for it.
        ~TensorLease()
        {
            if (Volatile.Read(ref _released) == 0)
                Debug.Fail(
                    $"A reader lock on {_target} was collected without being released: the run that "
                    + "took it never gave it back, and the tensor it locked became garbage while "
                    + "still locked.");
        }
#endif
    }

    /// <summary>
    /// What one run holds of its inputs: a reader lock on every tensor and sequence it reads, and
    /// the tensors it consumes — each held, strongly, until the run returns however it returns, and
    /// then given up: the locks dropped, the consumed memory released through the backend that
    /// made it.
    ///
    /// <para>Both run paths feed through one of these, so that what a feed means — read or
    /// consumed — is decided in one place.</para>
    /// </summary>
    internal sealed class RunFeeds : IDisposable
    {
        private readonly ComputeContext _context;
        private readonly Func<TensorDeath> _consumption;
        private readonly List<TensorLease> _leases;
        private List<TensorData>? _consumed;
        private int _held;

        /// <param name="context">The context running, which takes the locks.</param>
        /// <param name="expected">How many inputs the run is being fed.</param>
        /// <param name="consumption">Why a tensor this run consumes died, built only if one does:
        /// naming the graph costs a string, and most runs consume nothing.</param>
        internal RunFeeds(ComputeContext context, int expected, Func<TensorDeath> consumption)
        {
            _context = context;
            _consumption = consumption;
            _leases = new List<TensorLease>(expected);
        }

        /// <summary>How many inputs are held, by a lock or by consumption.</summary>
        internal int Held => _held;

        /// <summary>The locks held, whose eviction signals the run listens for.</summary>
        internal IReadOnlyList<TensorLease> Leases => _leases;

        /// <summary>
        /// Holds <paramref name="input"/> for the run — a reader lock, or, for a donation, the
        /// tensor itself taken — and then builds the value <paramref name="backend"/>'s session
        /// is fed. The hold first, then the value: the other order leaves a window in which a
        /// delete on another thread frees what was just built (Shorokoo/Shorokoo#366).
        /// </summary>
        /// <exception cref="InvalidOperationException">A kind of parameter that can be fed and
        /// cannot be held, which would be a feed with nothing holding it; or a donation of a tensor
        /// another run is reading.</exception>
        /// <exception cref="ObjectDisposedException">The input is dead.</exception>
        internal IShorokooTensorValue Feed(NamedModelParam input, IShorokooBackend backend)
        {
            switch (input)
            {
                // Before the plain tensor parameter it derives from: a donation is consumed, not
                // read.
                case DonatedTensorModelParam donated:
                {
                    var tensor = donated.ToTensorData();
                    tensor.TakeForRun(_consumption());
                    (_consumed ??= []).Add(tensor);
                    _held++;
                    return tensor.ValueForTakenRun(backend);
                }
                case TensorDataModelParam tensor:
                    Hold(_context.Lock(tensor.ToTensorData()));
                    return input.ToTensorValue(backend);
                case OptionalTensorDataModelParam { Data.Value: { } present }:
                    Hold(_context.Lock(present));
                    return input.ToTensorValue(backend);
                // An absent optional holds nothing and the feed itself refuses it, naming the
                // engine that does support one. It counts as held all the same, so that the count
                // still matches the feeds and the refusal is the feed's rather than this one's.
                case OptionalTensorDataModelParam:
                    _held++;
                    return input.ToTensorValue(backend);
                case TensorDataSequenceModelParam sequence:
                    Hold(_context.Lock(sequence.ToTensorDataSequence()));
                    return input.ToTensorValue(backend);
                default:
                    throw new InvalidOperationException(
                        $"Input '{input.ParamName}' was fed to a run as a {input.GetType().Name}, "
                        + "which nothing knows how to hold. Every fed input has to be held for the "
                        + "length of the run, or deleting it from another thread frees what the run "
                        + "is reading.");
            }
        }

        private void Hold(TensorLease lease)
        {
            _leases.Add(lease);
            _held++;
        }

        /// <summary>Gives up everything held: the locks dropped, the consumed memory released.
        /// However the run ended.</summary>
        public void Dispose()
        {
            foreach (var lease in _leases) lease.Dispose();
            if (_consumed is not null)
                foreach (var tensor in _consumed) tensor.ReleaseTaken();
        }
    }
}
