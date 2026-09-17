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
            _release = release ?? throw new ArgumentNullException(nameof(release));
        }

        /// <summary>Storage that holds nothing and releases nothing — a metadata-only tensor.</summary>
        internal static TensorStorage None { get; } = new(MemorySpace.Host, static () => { });

        /// <summary>Where these bytes are.</summary>
        internal MemorySpace Space { get; }

        /// <summary>
        /// The context whose disposal releases these bytes, or null when they belong to the
        /// framework's own host memory and outlive every context.
        ///
        /// <para>It follows the ownership, not the allocation. A tensor transferred out of the
        /// context that allocated it takes the bytes with it, and disposing that first context
        /// must then leave them alone — they are the target's now.</para>
        /// </summary>
        internal Shorokoo.Runtime.ComputeContext? Owner { get; private set; }

        /// <summary>Moves responsibility for these bytes to <paramref name="context"/>, off
        /// whoever had it. Exactly one context is on the hook at a time.</summary>
        internal void TransferOwnershipTo(Shorokoo.Runtime.ComputeContext? context)
        {
            if (ReferenceEquals(Owner, context)) return;
            // The new owner first: TakeOwnership refuses a disposed context, and a hand-off that
            // fails after the old owner has let go would leave these bytes on nobody's books --
            // freed by neither context, which is the leak disposal exists to prevent.
            context?.TakeOwnership(this);
            Owner?.ReleaseOwnership(this);
            Owner = context;
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
