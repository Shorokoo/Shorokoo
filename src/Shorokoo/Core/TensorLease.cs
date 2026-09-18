using System;
using System.Threading;

namespace Shorokoo.Runtime
{
    /// <summary>
    /// A compute context's claim on one allocation for as long as it is reading it, from
    /// <c>ComputeContext.Lock</c>. While it is held the bytes cannot be freed by anything letting
    /// go of a handle on them, and a deliberate delete of them signals <see cref="Eviction"/> and
    /// waits rather than pulling them away.
    ///
    /// <para>The one obligation a holder has is to listen for that signal on everything it has
    /// locked and to stop as soon as any of them is raised. A run discharges it by handing the
    /// linked signal to the backend as <c>RunSettings.CancellationToken</c>; a holder that ignores
    /// it is not unsafe, only slow — the delete then waits for the read to end on its own.</para>
    ///
    /// <para>Disposing is dropping the claim, and it is a count rather than a flag: the same
    /// allocation locked twice is released twice.</para>
    /// </summary>
    internal sealed class TensorLease : IDisposable
    {
        private readonly ComputeContext _context;
        private readonly TensorStorage _storage;
        private int _released;

        internal TensorLease(ComputeContext context, TensorStorage storage, CancellationToken eviction)
        {
            _context = context;
            _storage = storage;
            Eviction = eviction;
        }

        /// <summary>Raised when something asks for this allocation to be deleted. Stop reading it
        /// and drop this lease: the deleter is waiting for exactly that.</summary>
        internal CancellationToken Eviction { get; }

        /// <summary>Drops this claim. The bytes go now if the delete they were waiting for has
        /// already been asked for, or if nothing else names them. Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0) return;
            // The context's count first, then the allocation's, and neither under the other: the
            // two gates are never nested, so a delete waiting for this lease cannot end up behind
            // a gate the run is queued on.
            _context.ReleaseLease();
            _storage.Unlock();
        }
    }
}
