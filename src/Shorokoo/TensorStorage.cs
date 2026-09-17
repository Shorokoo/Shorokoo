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
