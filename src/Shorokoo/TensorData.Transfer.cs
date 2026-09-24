using System;
using System.Collections.Generic;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Utils;
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
    ///
    /// <para>A target whose device memory is under a budget
    /// (<see cref="ComputeContext.DeviceMemory"/>'s <see cref="DeviceMemorySettings.LimitBytes"/>)
    /// counts what is attached to it there, and refuses what would take that past the limit: a
    /// copy before it is allocated, and a tensor handed over as it stands before it is attached.
    /// The refusal names the budget, what is attached and what was asked for.</para>
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
        /// <exception cref="InvalidOperationException"><paramref name="target"/>'s device-memory
        /// budget cannot take this tensor: what is attached to it in its memory, plus this tensor —
        /// the copy, or this very tensor where it is already there and not yet on the target's
        /// books — would pass its <see cref="DeviceMemorySettings.LimitBytes"/>. Nothing is copied
        /// or attached.</exception>
        /// <exception cref="InvalidOperationException">This tensor is in device memory whose producer was
        /// not recorded (<see cref="Create(Shape, DType, IShorokooTensorValue)"/>), so nothing can
        /// read it back to copy.</exception>
        public TensorData To(ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            ThrowIfDisposed();
            RefuseDisposedTarget(target, nameof(To));
            if (!target.CanAddress(this)) return CopyInto(target, nameof(To));
            target.AttachAllWithinBudget([this], nameof(To));
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
        /// <exception cref="InvalidOperationException"><paramref name="target"/>'s device-memory
        /// budget cannot take the copy alongside what is attached to it in its memory
        /// (<see cref="DeviceMemorySettings.LimitBytes"/>). Nothing is copied.</exception>
        /// <exception cref="InvalidOperationException">This tensor is in device memory whose producer was
        /// not recorded (<see cref="Create(Shape, DType, IShorokooTensorValue)"/>), so nothing can
        /// read it back.</exception>
        public TensorData CopyTo(ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            ThrowIfDisposed();
            RefuseDisposedTarget(target, nameof(CopyTo));
            return CopyInto(target, nameof(CopyTo));
        }

        /// <summary>
        /// The bytes putting this tensor on <paramref name="target"/> — handed over as it stands
        /// where <see cref="To"/> would, unless <paramref name="copying"/>, and copied otherwise —
        /// adds to what the target's device-memory budget counts: this tensor's own where it is in
        /// the target's memory and not yet on its books, and a copy's where the copy lands there. A
        /// copy the host keeps adds nothing: one of a dtype the target's runs read from host memory,
        /// a string among them. What a composite placement is refused on before any of it is placed;
        /// <paramref name="handedOver"/> is what the composite hands over as it stands so far, so a
        /// tensor it holds twice is counted once.
        /// </summary>
        internal long BytesPlacedOnto(ComputeContext target, bool copying, HashSet<TensorData>? handedOver = null)
        {
            var space = target.MemorySpace;
            if (space.IsHost) return 0;
            if (!copying && target.CanAddress(this))
                return Space == space && !target.Attaches(this) && (handedOver?.Add(this) ?? true) ? ByteCount : 0;
            return RunMemoryOf(target.ResolvedBackend, DType).Space.IsHost ? 0 : ByteCount;
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
        /// <exception cref="InvalidOperationException">This tensor is in device memory whose producer was
        /// not recorded (<see cref="Create(Shape, DType, IShorokooTensorValue)"/>), so nothing can
        /// read it back.</exception>
        public TensorData ToHost()
        {
            ThrowIfDisposed();
            return IsHostResident ? this : CopyToManagedHost();
        }

        /// <summary>
        /// This tensor to be <b>read</b> by the run it is fed to, rather than consumed by it. Fed as
        /// it is, a tensor is given to the run, which takes it when it starts; fed as this, it is lent
        /// instead — the run takes a reader lock on it for as long as it runs, the running context is
        /// attached to it, and it is alive and unchanged when the run returns.
        ///
        /// <para>Nothing happens to the tensor here: the lock is taken when the run starts. What the
        /// run reads is the tensor itself where the run's backend can address its memory, and
        /// otherwise a copy in memory the backend can read — made on the first such read, held by
        /// this tensor for the next ones, and retired by a write to it.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor is dead.</exception>
        public SharedInput Shared()
        {
            ThrowIfDisposed();
            return new SharedInput(this, SharedInputMode.Shared);
        }

        /// <summary>
        /// This tensor to be consumed by the run it is fed to if nothing else is reading it when that
        /// run starts, and read by it — exactly as <see cref="Shared"/> would be — if something is.
        /// The decision is made when the run starts, not here: whether another run holds a lock can
        /// change in between.
        ///
        /// <para>For a feed that is spent if it can be and kept if it must be: consumed, its memory
        /// goes to the run and is released as soon as the run no longer needs it; read, it stays
        /// the caller's.</para>
        /// </summary>
        /// <exception cref="ObjectDisposedException">This tensor is dead.</exception>
        public SharedInput TryConsume()
        {
            ThrowIfDisposed();
            return new SharedInput(this, SharedInputMode.TryConsume);
        }

        /// <summary>
        /// The memory a run on <paramref name="backend"/> reads a tensor of <paramref name="dtype"/>
        /// in, as the backend answers it (<see cref="IShorokooBackend.RunMemoryOf"/>): its own memory in
        /// its own runtime, and host memory for a string tensor. What a copy made for such a run is
        /// keyed by, and where one is put.
        /// </summary>
        internal static MemoryLocation RunMemoryOf(IShorokooBackend backend, DType dtype)
            => backend.RunMemoryOf(dtype.IsSameElementTypeAs(DType.Utf8)
                ? ShorokooTensorElementType.String : (ShorokooTensorElementType)(int)dtype);

        /// <summary>
        /// Whether this tensor is exactly where a run on <paramref name="backend"/> reads one of its
        /// dtype, as the backend answers it (<see cref="RunMemoryOf"/>): that backend's context has
        /// nothing to move even where the backend cannot address the memory — a string tensor in the
        /// host memory of a card's runtime, say, which is where that runtime reads every string.
        ///
        /// <para>In a memory space Shorokoo has no name for, the location names no device in
        /// particular, and two backends over one runtime on two such devices would match it alike: there
        /// it answers only for the backend that allocated this tensor.</para>
        /// </summary>
        internal bool IsWhereRunsRead(IShorokooBackend backend)
            => Location == RunMemoryOf(backend, DType)
               && (Location.Space.IsKnown || ReferenceEquals(AllocatingBackend, backend));

        /// <summary>Host memory of <paramref name="backend"/>'s runtime: where the values it builds
        /// from host bytes are, and where it leaves an output it does not keep in memory of its
        /// own.</summary>
        internal static MemoryLocation HostMemoryOf(IShorokooBackend backend)
            => new(MemorySpace.Host, backend.RuntimeIdentity);

        /// <summary>
        /// The copy of this tensor a run on <paramref name="backend"/> reads where it cannot be
        /// handed this tensor itself (<see cref="FeedsInPlace"/>): held by this tensor, keyed by the
        /// memory it is in, and reused by every later read that wants it there. The first time it
        /// is made, <paramref name="admit"/> is asked first — the reading run holding the copy to its
        /// context's device-memory budget — and may refuse it by throwing.
        ///
        /// <para>Every context whose runs read it has it attached, so it counts on each of their
        /// budgets; only the one that makes it is asked whether it fits.</para>
        /// </summary>
        internal TensorData SharedCopyFor(IShorokooBackend backend, Action<TensorData>? admit)
        {
            var where = RunMemoryOf(backend, DType);
            // The copy already held is what nearly every read finds, and finding it builds nothing.
            return CopyHeldAt(where) ?? CopyAt(where, () =>
            {
                admit?.Invoke(this);
                return BuildRunCopy(backend);
            });
        }

        /// <summary>
        /// The copy a run on <paramref name="backend"/> that has taken this tensor consumes in its
        /// place, itself taken with <paramref name="death"/>: the one this tensor already holds in
        /// the run's memory, or a fresh one — which <paramref name="admit"/> is asked about first,
        /// as <see cref="SharedCopyFor"/> asks. This tensor's own memory is the caller's to release.
        /// </summary>
        internal TensorData TakeRunCopy(
            IShorokooBackend backend, Action<TensorData>? admit, TensorDeath death)
        {
            if (TakeCopyAt(RunMemoryOf(backend, DType), death) is { } held) return held;
            admit?.Invoke(this);
            var fresh = BuildRunCopy(backend);
            // Made for this run alone, so nothing else can hold it: the take cannot fail.
            if (fresh.TryTake(death) != TakeOutcome.Taken)
                throw new InvalidOperationException("A copy made for one run was held by another.");
            return fresh;
        }

        /// <summary>
        /// The copy a run on <paramref name="backend"/> that has taken this tensor consumes in its
        /// place where the runtime is to copy it into its own arena: a value of that backend's runtime
        /// in host memory, holding this tensor's contents, taken with <paramref name="death"/>. Reads
        /// the contents without the liveness check: the run has taken this tensor.
        /// </summary>
        internal TensorData HostRunCopy(IShorokooBackend backend, TensorDeath death)
        {
            var copy = BuiltBy(backend, HostCopyOn(backend));
            // Made for this run alone, so nothing else can hold it: the take cannot fail.
            if (copy.TryTake(death) != TakeOutcome.Taken)
                throw new InvalidOperationException("A copy made for one run was held by another.");
            return copy;
        }

        /// <summary>
        /// A new tensor holding this one's contents in the memory a run on
        /// <paramref name="backend"/> reads, allocated by that backend. Reads the contents
        /// without the liveness check: the caller holds this tensor's lock, or has taken it.
        /// </summary>
        private TensorData BuildRunCopy(IShorokooBackend backend)
            => BuiltBy(backend, DType.IsSameElementTypeAs(DType.Utf8)
                ? backend.CreateStringTensor(CopyContentStrings(), (long[])Shape)
                : backend.CreateTensorInBackendMemory(
                    (ShorokooTensorElementType)(int)DType, ContentBytesForCopy(), (long[])Shape));

        /// <summary>
        /// A tensor over <paramref name="value"/>, which <paramref name="backend"/> has just built as
        /// a copy of this one: this tensor's shape and dtype, allocated by that backend and released
        /// through it — at once, if the wrapping fails, since nothing else names it yet.
        /// </summary>
        private protected TensorData BuiltBy(IShorokooBackend backend, IShorokooTensorValue value)
            => OnnxUtils.CreateTensorDataFromValue(Shape, DType, value, DType, backend);

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
        /// <exception cref="InvalidOperationException">A run is reading this tensor; or it is in
        /// device memory whose producer was not recorded
        /// (<see cref="Create(Shape, DType, IShorokooTensorValue)"/>), so nothing can read it
        /// back.</exception>
        public TensorAttribute MoveToAttribute()
        {
            ThrowIfDisposed();

            // What cannot be taken without a copy is copied first, while the tensor is still alive:
            // the copy is the step that can fail -- a device buffer read back without the runtime
            // that made it, say -- and a failure then leaves a tensor that is still whole rather than
            // one that died with its contents lost. The generic placeholder's storage dtype is read
            // now for the same reason: afterwards there is no value left to ask.
            var strings = DType.IsSameElementTypeAs(DType.Utf8) ? Reading(CopyContentStrings) : null;
            var own = strings is null ? OwnBytes : null;
            var copied = strings is null && own is null ? Reading(CopyContentBytes) : null;
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
        private protected IEnumerable<string> StringElements() => Reading(CopyContentStrings);

        /// <summary>
        /// Refuses a target that has been disposed, naming the operation the caller actually made.
        /// Attaching refuses one too, but only after a copy onto it would already have been paid for,
        /// and a composite with nothing of its own to attach would not be refused at all.
        /// </summary>
        internal static void RefuseDisposedTarget(ComputeContext target, string operation)
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
        /// <para>The target allocates them itself, through
        /// <see cref="IShorokooBackend.CreateTensorInBackendMemory"/> rather than
        /// <c>CreateTensorFromRawBytes</c>, so the bytes land in the memory the target names
        /// instead of in host memory wearing its name. That is the difference between a tensor that
        /// is on the card and one an execution provider has to copy there on every run. A target
        /// whose memory is the host's gets the framework's own managed memory, which every host
        /// backend reads.</para>
        ///
        /// <para>Where the target's memory is under its device-memory budget, the copy is refused
        /// before anything is allocated when the budget cannot take it alongside what is attached to
        /// the target — checked and attached one at a time with everything else that spends that
        /// budget, so two copies cannot both be admitted into the same room.</para>
        /// </summary>
        private TensorData CopyInto(ComputeContext target, string operation)
            => target.MemorySpace.IsHost
                ? target.Attached(CopyToManagedHost())
                : target.Placed(BytesPlacedOnto(target, copying: true),
                    () => $"{operation}(context) of {Describe()}", () => CopyIntoBackendMemory(target));

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
            var backend = target.ResolvedBackend;
            // Where the target's runs read this dtype from the host -- a string, on any device --
            // the copy is the framework's own host memory, which they read as they read any.
            if (RunMemoryOf(backend, DType).Space.IsHost) return CopyToManagedHost();

            // Straight from the contents, under a reader lock for the length of the copy: a managed
            // tensor's own array, which the backend copies out of and keeps nothing of, rather than a
            // host copy of it first. The same builder a run's copy takes, so the two carry the same
            // dtype, and a value the wrapping fails on goes back to the backend.
            return Reading(() => BuiltBy(backend, backend.CreateTensorInBackendMemory(
                (ShorokooTensorElementType)(int)DType, ContentBytesForCopy(), (long[])Shape)));
        }

        /// <summary>
        /// This tensor's contents as host bytes, whatever memory it is in: its own, copied, where the
        /// host can read them, and otherwise a copy the backend that made the memory reads back —
        /// the only thing that knows how to reach it. Copied under a reader lock, so no run can
        /// consume the tensor, and no delete release it, from under the copy.
        /// </summary>
        private byte[] HostBytes() => Reading(CopyContentBytes);
    }
}
