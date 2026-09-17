using System;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo
{
    /// <summary>
    /// Moving a tensor between compute contexts.
    ///
    /// <para>Three operations, differing in what happens to the ownership of the bytes rather than
    /// in what happens to the bytes. <see cref="TransferTo"/> hands ownership over;
    /// <see cref="CopyTo"/> makes a second set of bytes with an owner of its own; and
    /// <see cref="GiveAccessTo"/> hands over a reader and no ownership at all.</para>
    ///
    /// <para>Whether the bytes move is a separate question, answered by
    /// <see cref="MemorySpace"/> and not by which context is which. Two contexts in one space —
    /// two CUDA contexts on one device, or any two host contexts — share the memory as it stands,
    /// so a transfer between them re-wraps and copies nothing. That is true even when the two are
    /// separate backends over separate native runtimes.</para>
    /// </summary>
    public abstract partial class TensorData
    {
        /// <summary>
        /// This tensor's bytes, as a tensor of <paramref name="target"/>, with ownership moved to
        /// the result. Null means the framework's own host memory.
        ///
        /// <para>Within one memory space nothing is copied: the result names the same bytes. If
        /// this tensor owned them the result owns them now and this one does not; if it did not
        /// own them, neither does the result. Either way this tensor stops being an owner, so
        /// disposing it afterwards releases nothing.</para>
        ///
        /// <para>Across memory spaces the bytes really move, which only an owner may do. This
        /// tensor is spent afterwards: its storage has been released and every read of it throws.
        /// A tensor that does not own its bytes is refused rather than silently copied — moving
        /// what you do not own is precisely what the non-owning case exists to forbid, and
        /// <see cref="CopyTo"/> is the operation that was meant.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor, or the memory behind it, is gone.</exception>
        /// <exception cref="InvalidOperationException">The spaces differ and this tensor does not
        /// own its bytes.</exception>
        public TensorData TransferTo(ComputeContext? target)
        {
            ThrowIfDisposed();
            RefuseUnknownSpace(nameof(TransferTo));
            var to = SpaceOf(target);

            if (to == Space && CanShareWith(target))
            {
                RefuseUnownedNullContext(target, wouldOwn: OwnsMemory, operation: nameof(TransferTo));
                RefuseDisposedTarget(target, nameof(TransferTo));
                // The bytes stay exactly where they are; only the names on them change.
                var moved = CloneSharing(target, OwnsMemory);
                SurrenderOwnership(target);
                return moved;
            }

            if (!OwnsMemory)
                throw new InvalidOperationException(
                    $"This tensor ({this}) is in {Space} and does not own its memory, so it cannot "
                    + $"be moved to {to}: the move would free bytes that belong to something else. "
                    + "Use CopyTo to take a copy of them there instead.");

            var relocated = CopyAcross(target, to);
            Dispose();
            return relocated;
        }

        /// <summary>
        /// An independent copy of this tensor's bytes in <paramref name="target"/>'s memory, owned
        /// by the result. Copies whether or not the spaces differ, and leaves this tensor exactly
        /// as it was — its bytes, its context and its ownership all untouched.
        ///
        /// <para>This is the operation that works from a tensor that owns nothing, and the one to
        /// reach for when the data has to outlive the context that produced it.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor, or the memory behind it, is gone.</exception>
        public TensorData CopyTo(ComputeContext? target)
        {
            ThrowIfDisposed();
            return CopyAcross(target, SpaceOf(target));
        }

        /// <summary>
        /// A reader for this tensor's bytes, as a tensor of <paramref name="target"/>. The result
        /// never owns the memory and this tensor keeps whatever ownership it had, so disposing the
        /// result frees nothing and disposing this one still frees everything.
        ///
        /// <para>Only within one memory space. Reaching another means allocating there, and
        /// allocating means owning, which is the one thing this operation promises not to do —
        /// <see cref="CopyTo"/> is that operation.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor, or the memory behind it, is gone.</exception>
        /// <exception cref="InvalidOperationException">The spaces differ.</exception>
        public TensorData GiveAccessTo(ComputeContext? target)
        {
            ThrowIfDisposed();
            RefuseUnknownSpace(nameof(GiveAccessTo));
            var to = SpaceOf(target);
            if (to != Space || !CanShareWith(target))
                throw new InvalidOperationException(
                    $"This tensor ({this}) is in {Space} and cannot be reached from {to} without "
                    + "allocating there, which would make the result an owner. GiveAccessTo never "
                    + "takes ownership; use CopyTo, which does.");

            RefuseUnownedNullContext(target, wouldOwn: false, operation: nameof(GiveAccessTo));
            RefuseDisposedTarget(target, nameof(GiveAccessTo));
            return CloneSharing(target, ownsMemory: false);
        }

        /// <summary>
        /// Refuses a target that has been disposed.
        ///
        /// <para>Taking ownership already refuses one — <c>ComputeContext.TakeOwnership</c> does
        /// it — but the two non-owning results reach no such path, so they were handed back bound
        /// to a context that had already released everything. <see cref="Context"/> is what every
        /// later operation routes through (<c>CanShareWith</c> reads the target's backend, and
        /// bringing a device tensor home asks that backend for the copy), so such a tensor is one
        /// the API says is usable and nothing will ever reject.</para>
        /// </summary>
        private void RefuseDisposedTarget(ComputeContext? target, string operation)
        {
            if (target is not { IsDisposed: true }) return;
            throw new ObjectDisposedException(
                nameof(ComputeContext),
                $"{operation} was given a compute context that has been disposed, so the tensor it "
                + "returned would name a context that has already released everything and can "
                + "answer nothing about its memory.");
        }

        /// <summary>
        /// Refuses the one combination the two invariants forbid between them: a tensor with no
        /// context that does not own its bytes.
        ///
        /// <para>The null context is not a context at all — it is the framework's own host memory,
        /// and a tensor there owns what it holds, because there is no context whose lifetime could
        /// own it instead. So memory can arrive there only by being owned. That rules out giving
        /// the null context access to someone else's bytes, and transferring to it from a tensor
        /// that has none to give; both mean <see cref="CopyTo"/>, which owns what it makes.</para>
        /// </summary>
        private void RefuseUnownedNullContext(ComputeContext? target, bool wouldOwn, string operation)
        {
            if (target is not null || wouldOwn) return;
            throw new InvalidOperationException(
                $"{operation}(null) would leave this tensor's bytes in the framework's own host "
                + "memory with nothing owning them. A tensor with no compute context owns what it "
                + "holds -- there is no context whose lifetime could own it instead -- so memory "
                + "reaches it only by being owned. Use CopyTo(null), which makes a copy it owns.");
        }

        /// <summary>
        /// Refuses a tensor whose memory cannot be named. Two such tensors compare equal as spaces
        /// without being in the same place, so treating them as transferable would be a guess
        /// dressed as a no-op.
        /// </summary>
        private void RefuseUnknownSpace(string operation)
        {
            if (Space.IsKnown) return;
            throw new InvalidOperationException(
                $"{operation} cannot place this tensor ({this}): it is in {Space}, so there is no "
                + "telling whether another context shares it. It came back from a session without "
                + "the context that produced it being recorded.");
        }

        /// <summary>
        /// Whether <paramref name="target"/> could actually use these bytes as they stand, rather
        /// than merely sharing a memory space with them.
        ///
        /// <para>The two are not the same for a value the execution provider kept. Sharing a space
        /// makes the <i>bytes</i> reachable from both, but a re-wrap hands the target this tensor's
        /// runtime value, and a session recognises its own values by type identity: two isolated
        /// backends on one device compare equal as a space while their values are types from
        /// different assemblies. So a re-wrap there would report success and hand back a tensor the
        /// context it now belongs to cannot feed. Host bytes have no such problem — anything can
        /// rebuild them — and the same backend trivially accepts its own.</para>
        /// </summary>
        private bool CanShareWith(ComputeContext? target)
            => Space.IsHost
               || ReferenceEquals(Context?.ResolvedBackend, target?.ResolvedBackend);

        /// <summary>Where a context's tensors live; null is the framework's own host memory.</summary>
        private static MemorySpace SpaceOf(ComputeContext? context)
            => context?.MemorySpace ?? MemorySpace.Host;

        /// <summary>
        /// A fresh, owned tensor holding this one's contents in <paramref name="to"/>. Goes through
        /// host bytes, which is the only route a backend-independent copy has: a value belongs to
        /// the runtime that made it, so the target has to be handed contents rather than a pointer.
        ///
        /// <para>The target allocates them itself, through
        /// <see cref="IShorokooInferenceBackend.CreateTensorInBackendMemory"/> rather than
        /// <c>CreateTensorFromRawBytes</c>, so the bytes land in the memory
        /// <paramref name="to"/> names instead of in host memory wearing its name. That is the
        /// difference between a tensor that is on the card and one an execution provider has to
        /// copy there on every run.</para>
        /// </summary>
        private TensorData CopyAcross(ComputeContext? target, MemorySpace to)
        {
            // Strings have no flat buffer to copy, so they take the route their own literals take:
            // the elements themselves, rebuilt on the other side. A backend holding them is asked
            // for them the same way, through the value it made.
            if (DType == DType.String) return CopyStringsAcross(target);

            var bytes = HostBytes();

            if (to.IsHost)
                return NewHostTensor(Shape, DType, bytes, target);

            var value = target!.ResolvedBackend.CreateTensorInBackendMemory(
                (ShorokooTensorElementType)(int)DType, bytes, (long[])Shape);
            try
            {
                return Create(Shape, DType, value, target);
            }
            catch
            {
                // Nothing else references it yet, and on a card it is an allocation that has just
                // been filled across the bus -- left to a finalizer it is a device leak for as long
                // as that takes.
                value.Dispose();
                throw;
            }
        }

        /// <summary>
        /// This tensor's contents as host bytes, whatever memory it is in.
        ///
        /// <para>A host tensor reads its own. One the execution provider kept cannot be read here
        /// at all — the accessors would hand out a device address and dereference it as a host one
        /// — so the copy is asked of the backend that owns the allocation, which is the only thing
        /// that knows how to reach it.</para>
        /// </summary>
        private TensorData CopyStringsAcross(ComputeContext? target)
        {
            var strings = this switch
            {
                HostStringTensorData host => host.Strings,
                IOnnxData onnx => onnx.Value.GetStringTensorData(),
                _ => throw new InvalidOperationException(
                    $"This tensor ({this}) holds strings but carries neither the elements "
                    + "themselves nor a runtime value to read them from."),
            };
            return NewHostStringTensor(Shape, [.. strings], target);
        }

        // Taken through CopyRawMemory rather than a bare span: the span is this tensor's last
        // read, so copying out of one by hand races the collection that frees what it points at
        // (Shorokoo/Shorokoo#178).
        private byte[] HostBytes()
        {
            if (Space.IsHost) return CopyRawMemory();

            if (!Space.IsKnown || Context is null)
                throw new InvalidOperationException(
                    $"This tensor ({this}) is in {Space}, and the context that produced it was not "
                    + "recorded, so there is no backend to ask for a copy of it.");

            if (this is not IOnnxData onnx)
                throw new InvalidOperationException(
                    $"This tensor ({this}) is in {Space} but carries no runtime value, so nothing "
                    + "can read it back.");

            return Context.ResolvedBackend.CopyTensorToHost(onnx.Value);
        }
    }
}
