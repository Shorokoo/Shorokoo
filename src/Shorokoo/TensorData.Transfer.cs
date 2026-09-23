using System;
using System.Collections.Generic;
using Shorokoo.Core.Backends;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// Putting a tensor where a compute context can use it, and taking it out of there.
    ///
    /// <para>A tensor never moves: its memory is where it was allocated, for as long as it lives.
    /// What these do is decide whether another tensor has to be made. <see cref="To"/> hands the
    /// tensor itself over wherever the target's backend can read it as it stands, and copies only
    /// where it cannot; <see cref="CopyTo"/> always copies; <see cref="ToHost"/> is <c>To</c> for
    /// host memory. None of them changes or ends the source.</para>
    ///
    /// <para>Whether the target can read the tensor as it stands is asked of the target's backend
    /// (<see cref="IShorokooBackend.CanAddress"/>), given where the tensor's memory is: the same
    /// device and the same runtime. Any host-memory backend reads the framework's own managed
    /// host memory. Two backends over one loaded ONNX Runtime share a card allocation; two
    /// isolated runtimes on one card do not, and copy through the host.</para>
    /// </summary>
    public abstract partial class TensorData
    {
        /// <summary>
        /// This tensor where <paramref name="target"/> can use it, attached to
        /// <paramref name="target"/>: the very same object when the target's backend can read its
        /// memory as it stands, and otherwise a new copy in the target's memory. This tensor is
        /// untouched either way.
        ///
        /// <para>Attaching is not ownership. A context keeps a weak list of the tensors attached to
        /// it, for its own accounting; the list never keeps a tensor alive and never ends one's life,
        /// and disposing the context leaves every tensor attached to it as it was.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This tensor is dead, or
        /// <paramref name="target"/> has been disposed.</exception>
        /// <remarks>A copy onto a card is allocated under the target context's
        /// <see cref="ComputeContext.DeviceMemory"/>, so one that does not fit fails with the backend
        /// runtime's own allocation error — the budget working rather than failing.</remarks>
        public TensorData To(ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            ThrowIfDisposed();
            RefuseDisposedTarget(target, nameof(To));
            if (!target.CanAddress(this)) return CopyInto(target);
            target.Attach(this);
            return this;
        }

        /// <summary>
        /// An independent copy of this tensor in <paramref name="target"/>'s memory, attached to
        /// <paramref name="target"/>. Copies whether or not the target could have read this tensor
        /// as it stands, and leaves this tensor exactly as it was.
        ///
        /// <para>This is the operation to reach for when two independent tensors are wanted: two
        /// runs that must not see each other's writes, or a value that has to outlive this tensor's
        /// deletion.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This tensor is dead, or
        /// <paramref name="target"/> has been disposed.</exception>
        /// <remarks>The copy is allocated under the target context's
        /// <see cref="ComputeContext.DeviceMemory"/>, so one that does not fit fails with the backend
        /// runtime's own allocation error.</remarks>
        public TensorData CopyTo(ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            ThrowIfDisposed();
            RefuseDisposedTarget(target, nameof(CopyTo));
            return CopyInto(target);
        }

        /// <summary>
        /// This tensor where the host can read it: the very same object when its memory is
        /// host-readable already, and otherwise a new copy in the framework's own host memory,
        /// attached to nothing. This tensor is untouched either way.
        ///
        /// <para>What a tensor a run left on a card, or one put there with <see cref="To"/>, needs
        /// before its elements can be read. The copy is read back through the backend that made
        /// the memory, which is the only thing that knows how to reach it.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor is dead.</exception>
        public TensorData ToHost()
        {
            ThrowIfDisposed();
            return IsHostResident ? this : CopyToManagedHost();
        }

        /// <summary>
        /// This tensor given to the next run that is fed the result, rather than lent to it. The run
        /// <b>consumes</b> the tensor when it starts: from that moment the tensor is dead — every
        /// access throws, saying which run took it — and its memory is released as soon as that run
        /// returns, however it returns, rather than whenever the caller lets go of it.
        ///
        /// <para>Nothing happens to the tensor until then: donating marks nothing, and a run refused
        /// before it starts — a cancelled one, say — takes nothing either. A run that is fed the
        /// donation while another run is reading the tensor is refused, since consuming memory
        /// another run is reading would take it from under that run.</para>
        ///
        /// <para>Nothing is copied and nothing moves. What it buys is the end of the wait — a feed a
        /// caller keeps is held until that caller lets go, which for a batch built per step is until
        /// the next collection, while a donated one is released with the step that read it
        /// (Shorokoo/Shorokoo#359).</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor is dead.</exception>
        public TensorDonation Donate()
        {
            ThrowIfDisposed();
            return new TensorDonation(this);
        }

        /// <summary>
        /// This tensor's elements as a <see cref="TensorAttribute"/> — a tensor in a graph's
        /// description rather than a runtime value. The tensor is <b>moved</b>: it dies — every
        /// access afterwards throws, saying it was moved — and its memory is released, so binding a
        /// 165 M-parameter checkpoint into a graph costs no second set of bytes.
        ///
        /// <para>Where the tensor holds its own array — one built from a C# array — the attribute
        /// takes that array and nothing is copied. Anything else is copied out into the attribute on
        /// the way: a runtime value's buffer, read back through the backend that made it where it is
        /// not host memory, or a string tensor's elements. The memory the copy came from is released
        /// either way.</para>
        ///
        /// <para>An attribute is immutable and shared by every graph that captured it, so there is
        /// no way back that does not copy: <see cref="TensorAttribute.CopyToTensorData()"/> is
        /// it. To keep the tensor as well, move a copy of it: <c>CopyTo(ComputeContext.Host)</c>.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor is dead.</exception>
        /// <exception cref="InvalidOperationException">A run is reading this tensor.</exception>
        public TensorAttribute MoveToAttribute()
        {
            ThrowIfDisposed();

            // What cannot be taken without a copy is copied first, while the tensor is still alive:
            // the copy is the step that can fail -- a device buffer read back without the runtime
            // that made it, say -- and a failure then leaves a tensor that is still whole rather than
            // one that died with its contents lost. The generic placeholder's storage dtype is read
            // now for the same reason: afterwards there is no value left to ask.
            var strings = DType.IsSameElementTypeAs(DType.Utf8) ? CopyContentStrings() : null;
            var own = strings is null ? OwnBytes : null;
            var copied = strings is null && own is null ? CopyContentBytes() : null;
            var storageDType = StorageDType();

            switch (TryTake(TensorDeath.MovedToAttribute))
            {
                case TakeOutcome.Taken:
                    break;
                case TakeOutcome.Dead:
                    ThrowIfDisposed();
                    break;
                default:
                    throw ReadByARun(nameof(MoveToAttribute));
            }

            try
            {
                return strings is not null
                    ? TensorAttribute.OverStrings(Shape, [.. strings])
                    // The tensor's own array where it has one: it is the only name for it, so the
                    // attribute taking it takes it from nobody, and the tensor is dead from here so
                    // nothing can write it through this tensor again.
                    : TensorAttribute.OverBytes(Shape, DType, own ?? copied!, storageDType);
            }
            finally
            {
                ReleaseTaken();
            }
        }

        /// <summary>
        /// The dtype these bytes are laid out at, which for a generic placeholder is not
        /// <see cref="DType"/>: that names the type parameter the literal stands for, and only the
        /// runtime value knows what was actually written. Null where there is nothing to add.
        /// </summary>
        private DType? StorageDType()
            => DType.IsGenericType && this is IOnnxData onnx ? (DType)(int)onnx.Value.ElementType : null;

        /// <summary>The elements of a string tensor, however this one holds them.</summary>
        private protected IEnumerable<string> StringElements()
        {
            ThrowIfDisposed();
            var strings = CopyContentStrings();
            GC.KeepAlive(this);
            return strings;
        }

        /// <summary>
        /// Refuses a target that has been disposed, naming the operation the caller actually made.
        /// Attaching refuses one too, but only after a copy onto it would already have been paid for.
        /// </summary>
        private static void RefuseDisposedTarget(ComputeContext target, string operation)
        {
            if (!target.IsDisposed) return;
            throw new ObjectDisposedException(
                nameof(ComputeContext),
                $"{operation} was given a compute context that has been disposed, so it has already "
                + "released everything and can place nothing in its memory.");
        }

        /// <summary>
        /// A fresh tensor holding this one's contents in <paramref name="target"/>'s memory, attached
        /// to it. Goes through host bytes, which is the only route a backend-independent copy has: a
        /// value belongs to the runtime that made it, so the target has to be handed contents rather
        /// than a pointer.
        ///
        /// <para>The target allocates them itself, through <c>CreateTensorInBackendMemory</c> rather
        /// than <c>CreateTensorFromRawBytes</c>, so the bytes land in the memory the target names
        /// instead of in host memory wearing its name. That is the difference between a tensor that
        /// is on the card and one an execution provider has to copy there on every run. A target
        /// whose memory is the host's gets the framework's own managed memory, which every host
        /// backend reads.</para>
        /// </summary>
        private TensorData CopyInto(ComputeContext target)
        {
            var copy = target.MemorySpace.IsHost ? CopyToManagedHost() : CopyIntoBackendMemory(target);
            try
            {
                target.Attach(copy);
            }
            catch
            {
                // Nothing else references it, and on a card it is an allocation that has just been
                // filled across the bus.
                copy.Delete();
                throw;
            }
            return copy;
        }

        /// <summary>A copy of this tensor in the framework's own host memory, attached to
        /// nothing.</summary>
        private TensorData CopyToManagedHost()
            // Strings have no flat buffer to copy, so they take the route their own literals take:
            // the elements themselves, rebuilt on the other side. ONNX Runtime allocates every string
            // tensor on the host whatever its provider, so host memory is where a string tensor goes
            // whichever context it is for.
            => DType.IsSameElementTypeAs(DType.Utf8)
                ? NewHostStringTensor(Shape, [.. StringElements()])
                : NewHostTensor(Shape, DType, HostBytes());

        private TensorData CopyIntoBackendMemory(ComputeContext target)
        {
            if (DType.IsSameElementTypeAs(DType.Utf8)) return CopyToManagedHost();

            var backend = target.ResolvedBackend;
            var value = backend.CreateTensorInBackendMemory(
                (ShorokooTensorElementType)(int)DType, HostBytes(), (long[])Shape, target.DeviceMemory);
            try
            {
                return Create(Shape, DType, value, backend);
            }
            catch
            {
                // Nothing else references it yet, and on a card it is an allocation that has just
                // been filled across the bus -- left to a finalizer it is a device leak for as long
                // as that takes.
                backend.Release(value);
                throw;
            }
        }

        /// <summary>
        /// This tensor's contents as host bytes, whatever memory it is in: its own, copied, where the
        /// host can read them, and otherwise a copy the backend that made the memory reads back —
        /// the only thing that knows how to reach it.
        /// </summary>
        private byte[] HostBytes()
        {
            ThrowIfDisposed();
            var bytes = CopyContentBytes();
            GC.KeepAlive(this);
            return bytes;
        }
    }
}
