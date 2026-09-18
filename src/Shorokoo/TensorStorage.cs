using System;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo
{
    /// <summary>
    /// The bytes behind a tensor, and whether they are still there.
    ///
    /// <para>Separate from <see cref="TensorData"/> because more than one tensor may name the same
    /// bytes: a same-space <c>TransferTo</c> re-wraps rather than copying, and <c>GiveAccessTo</c>
    /// exists precisely to hand out a second name for memory the caller does not own. Exactly one
    /// of those tensors owns the storage and releases it; the rest are readers.</para>
    ///
    /// <para>Which is why liveness lives here rather than on the tensor. A reader has no way to
    /// know that the owner has been disposed, and a raw pointer does not stop working when its
    /// memory is freed — it starts reading something else. <see cref="IsLive"/> turns that into an
    /// exception. It costs one predictable branch per access, against a class of bug that cannot
    /// be found from the outside at all.</para>
    /// </summary>
    internal sealed class TensorStorage
    {
        private Action? _release;

        internal TensorStorage(MemorySpace space, Action release)
        {
            Space = space;
            // The framework's own host memory until something takes it over. That is what a
            // storage nobody has handed to a context is, and the host context never releases, so
            // starting here is exactly the "no owner" this used to spell as null.
            Owner = Shorokoo.Runtime.ComputeContext.Host;
            _release = release ?? throw new ArgumentNullException(nameof(release));
        }

        /// <summary>Storage that holds nothing and releases nothing — a metadata-only tensor.</summary>
        internal static TensorStorage None { get; } = new(MemorySpace.Host, static () => { });

        /// <summary>Where these bytes are.</summary>
        internal MemorySpace Space { get; }

        /// <summary>
        /// The context whose disposal releases these bytes.
        /// <see cref="Shorokoo.Runtime.ComputeContext.Host"/> — the default — means they belong to
        /// the framework's own host memory and outlive every context, that one never being
        /// disposed.
        ///
        /// <para>It follows the ownership, not the allocation. A tensor transferred out of the
        /// context that allocated it takes the bytes with it, and disposing that first context
        /// must then leave them alone — they are the target's now.</para>
        /// </summary>
        internal Shorokoo.Runtime.ComputeContext Owner { get; private set; }

        /// <summary>
        /// The one lock every change of ownership is made under, and that a context's disposal
        /// takes around its release loop.
        ///
        /// <para>It has to be shared rather than per-context, because a hand-off touches two
        /// books and the danger is in the gap between them. Taken first and each context's own
        /// disposal gate second, always, so the two orders cannot invert.</para>
        ///
        /// <para>Without it the hand-off was two unsynchronised steps: a disposal of the old owner
        /// landing between them found the storage still on its books, released bytes the new owner
        /// had already accepted, and left a live context holding freed memory.</para>
        /// </summary>
        internal static object OwnershipGate { get; } = new();

        /// <summary>Moves responsibility for these bytes to <paramref name="context"/>, off
        /// whoever had it. Exactly one context is on the hook at a time.</summary>
        internal void TransferOwnershipTo(Shorokoo.Runtime.ComputeContext context)
        {
            // Outside the lock, because this is the overwhelmingly common case and the gate is
            // process-wide: every literal a graph build creates is a tensor on the host context,
            // which is where a storage starts, and taking a shared monitor to discover that
            // serialized model construction across the whole process -- on the very two-device
            // workload this design is for. Volatile-free is fine here: the only transition this
            // can miss is one that would have to be racing this storage's construction, and a
            // storage nobody else has a reference to yet has no other writer.
            if (ReferenceEquals(Owner, context)) return;

            lock (OwnershipGate)
            {
                if (ReferenceEquals(Owner, context)) return;
                // The new owner first: TakeOwnership refuses a disposed context, and a hand-off
                // that fails after the old owner has let go would leave these bytes on nobody's
                // books -- freed by neither context, which is the leak disposal exists to prevent.
                context.TakeOwnership(this);
                Owner.ReleaseOwnership(this);
                Owner = context;
            }
        }

        /// <summary>False once the owner has released these bytes. Reading them afterwards is a
        /// use-after-free, and every accessor checks this to make it an exception instead.</summary>
        internal bool IsLive => _release is not null;

        /// <summary>
        /// Releases the bytes. Called by the owning tensor and by nobody else — a reader disposing
        /// itself must leave the storage alone, which is what ownership means here. Idempotent.
        /// </summary>
        internal void Release()
        {
            var release = _release;
            if (release is null) return;
            _release = null;
            release();
        }
    }
}
