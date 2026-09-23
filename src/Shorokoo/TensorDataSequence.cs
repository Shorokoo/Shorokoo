using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo;
using Shorokoo.Core.Backends;
using static Shorokoo.Globals;
using System.Collections;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Runtime;

namespace Shorokoo
{
    public abstract class TensorDataSequence<T> : TensorDataSequence, IReadOnlyList<TensorData<T>>
        where T : IVarType
    {
        internal TensorDataSequence() : base(OnnxUtils.GetDType<T>())
        {
        }

        public abstract new TensorData<T> this[int index] { get; }

        public abstract new IEnumerator<TensorData<T>> GetEnumerator();

        internal override IEnumerator<TensorData> InternalGetEnumerator()
        {
            // GetEnumerator() validates; calling it here rather than iterating `this` lazily is
            // what makes the non-generic path throw at the call too.
            var elements = GetEnumerator();
            return Widen(elements);

            static IEnumerator<TensorData> Widen(IEnumerator<TensorData<T>> inner)
            {
                // The `foreach` this replaced disposed the inner enumerator; a bare while loop
                // would not, on a full drain or an early one.
                try
                {
                    while (inner.MoveNext()) yield return inner.Current;
                }
                finally
                {
                    inner.Dispose();
                }
            }
        }

        internal override TensorData GetAt(int index) => this[index];

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public List<TensorData<T>> AsList => [.. this];
    }

    /// <summary>
    /// A sequence of tensors, as a run takes and gives one.
    ///
    /// <para>A sequence has a life of its own, like a tensor's: fed to a run as it is it is
    /// <b>consumed</b> by that run, as a tensor is — dead from then on, its memory given to the
    /// run's backend — and fed as <see cref="Shared"/> it is read, the run holding a reader lock on
    /// it for as long as it runs. <see cref="Dispose"/> ends it and throws while a run holds one,
    /// and a dead sequence says why on every path to its elements. It knows no compute context;
    /// <see cref="To"/>, <see cref="CopyTo"/> and <see cref="ToHost"/> put its elements where a
    /// context can use them, as the same operations on <see cref="TensorData"/> do.</para>
    /// </summary>
    public abstract class TensorDataSequence : IData, IDisposable, IReadOnlyList<TensorData>
    {
        public DType DType { get; private set; }

        public abstract int Count { get; }

        int IReadOnlyCollection<TensorData>.Count => this.Count;

        internal TensorDataSequence(DType dtype)
        {
            this.DType = dtype;
        }

        // The sequence's own life, kept the way a tensor keeps its own: per sequence, and taken by
        // nothing else. See TensorData.Lifetime for what each field means there; they mean the same
        // here.
        private readonly object _gate = new();
        private TensorDeath? _death;
        private bool _taken;
        private bool _released;
        private int _locks;
        private List<object>? _readers;

        // The sequence values runs built from this one where they could not be handed it as it is,
        // one per place they are in, each a sequence in its own right. Created on the first.
        private Dictionary<MemoryLocation, TensorDataSequence>? _copies;
        private readonly object _copyGate = new();

        /// <summary>
        /// True once this sequence is dead — disposed, or consumed by a run — and its storage gone.
        /// Its dtype and <see cref="ToString"/> stay readable as metadata; every path to the
        /// elements throws, saying which it was.
        /// </summary>
        public bool IsDisposed => Volatile.Read(ref _death) is not null;

        /// <summary>How this sequence died, or null while it lives.</summary>
        internal TensorDeath? Death => Volatile.Read(ref _death);

        /// <summary>How a refusal names this sequence.</summary>
        internal string Describe() => $"Sequence {this}";

        /// <summary>Guards every path to the sequence's elements.</summary>
        protected void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _death) is { } death) throw death.Refusal(this, Describe());
        }

        /// <summary>
        /// Ends this sequence and releases its storage through the backend that made it. A sequence
        /// already dead is left as it is.
        /// </summary>
        /// <exception cref="InvalidOperationException">A run is reading this sequence; releasing
        /// its storage now would free it under the run.</exception>
        public void Dispose()
        {
            lock (_gate)
            {
                if (_death is not null) return;
                if (_locks > 0)
                    throw new InvalidOperationException(
                        $"Sequence {this} is being read by {DescribeReaderUnlocked() ?? "a run"}, so it "
                        + "cannot be disposed: its storage would be freed under the run. Wait for the "
                        + "run to return.");
                _death = TensorDeath.Disposed;
                _taken = true;
            }
            ReleaseTaken();
        }

        /// <summary>
        /// This sequence to be <b>read</b> by the run it is fed to — every element of it — rather
        /// than consumed; see <see cref="TensorData.Shared"/>.
        /// </summary>
        /// <exception cref="ObjectDisposedException">This sequence is dead.</exception>
        public SharedInput Shared()
        {
            ThrowIfDisposed();
            return new SharedInput(this, SharedInputMode.Shared);
        }

        /// <summary>
        /// This sequence to be consumed by the run it is fed to if nothing else is reading it when
        /// that run starts, and read otherwise; see <see cref="TensorData.TryConsume"/>.
        /// </summary>
        /// <exception cref="ObjectDisposedException">This sequence is dead.</exception>
        public SharedInput TryConsume()
        {
            ThrowIfDisposed();
            return new SharedInput(this, SharedInputMode.TryConsume);
        }

        /// <summary>Releases this sequence's own storage. Called at most once, by whoever it
        /// belongs to when the sequence dies; never for storage a run consumed.</summary>
        private protected abstract void ReleaseMemory();

        /// <summary>Ends this sequence's life deliberately if no run is reading it; see
        /// <see cref="TensorData.TryTake"/>.</summary>
        internal TakeOutcome TryTake(TensorDeath death)
        {
            ArgumentNullException.ThrowIfNull(death);
            lock (_gate)
            {
                if (_death is not null) return TakeOutcome.Dead;
                if (_locks > 0) return TakeOutcome.Locked;
                _death = death;
                _taken = true;
                return TakeOutcome.Taken;
            }
        }

        /// <summary>Releases what a successful take handed over: this sequence's storage and the
        /// copies runs built of it. Once.</summary>
        internal void ReleaseTaken()
        {
            lock (_gate)
            {
                if (!_taken || _released) return;
                _released = true;
            }
            try
            {
                ReleaseMemory();
            }
            finally
            {
                RetireCopies();
            }
        }

        /// <summary>Records that what a take handed over went to a backend, which releases it
        /// itself; the copies runs built of this sequence go now.</summary>
        internal void HandedToBackend()
        {
            lock (_gate)
            {
                if (!_taken || _released) return;
                _released = true;
            }
            RetireCopies();
        }

        /// <summary>Ends a copy its source no longer wants: dead, and released now or when the last
        /// run reading it returns; see <see cref="TensorData.Retire"/>.</summary>
        internal void Retire(TensorDeath death)
        {
            bool releaseNow = false;
            lock (_gate)
            {
                if (_death is not null) return;
                _death = death;
                if (_locks == 0)
                {
                    _taken = true;
                    releaseNow = true;
                }
            }
            if (releaseNow) ReleaseTaken();
        }

        /// <summary>Takes a reader lock for the length of a run. Refuses a sequence that is dead:
        /// there is nothing left to read.</summary>
        /// <exception cref="ObjectDisposedException">The sequence is dead.</exception>
        internal void AcquireReadLock(object? reader = null)
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                _locks++;
                if (reader is not null) (_readers ??= []).Add(reader);
            }
        }

        /// <summary>Drops a reader lock. A copy retired while it was held is released with the
        /// last one.</summary>
        internal void ReleaseReadLock(object? reader = null)
        {
            bool release = false;
            lock (_gate)
            {
                _locks--;
                if (reader is not null) _readers?.Remove(reader);
                if (_locks == 0 && _death is not null && !_taken && !_released)
                {
                    _released = true;
                    release = true;
                }
            }
            if (!release) return;
            try
            {
                ReleaseMemory();
            }
            finally
            {
                RetireCopies();
            }
        }

        /// <summary>Whether a run is reading this sequence right now.</summary>
        internal bool IsLocked
        {
            get { lock (_gate) return _locks > 0; }
        }

        /// <summary>The read a refusal of this sequence names, or null when no holder said.</summary>
        internal string? DescribeReader()
        {
            lock (_gate) return DescribeReaderUnlocked();
        }

        private string? DescribeReaderUnlocked()
            => _readers is { Count: > 0 } readers ? readers[0].ToString() : null;

        public override string ToString()
        {
            return $"sequence:{this.DType.ToString()}";
        }

        /// <summary>
        /// This sequence as a value of the process-wide backend's runtime, for a caller with no
        /// context to name. <see cref="TensorData.ToTensorValue()"/>'s counterpart.
        /// </summary>
        internal IShorokooTensorValue ToTensorValue() => ToTensorValue(DefaultBackend.Instance);

        /// <summary>
        /// This sequence as a value of <paramref name="backend"/>'s runtime — the form the feed
        /// sites take, so a sequence is built by the backend whose session is about to read it.
        /// A sequence that already holds a runtime value hands it over and ignores the argument,
        /// since a value belongs to the runtime that made it; one that is only a list of tensors
        /// is built on that runtime here — a copy held by this sequence, which runs on that backend
        /// read too.
        ///
        /// <para>The value returned is the sequence's, or its copy's: read it, do not dispose
        /// it.</para>
        /// </summary>
        internal IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            ArgumentNullException.ThrowIfNull(backend);
            ThrowIfDisposed();
            return OwnValue ?? SharedCopyFor(backend).UncheckedValue;
        }

        /// <summary>The runtime value this sequence holds itself, without the liveness check, or
        /// null for one that holds none of its own.</summary>
        internal virtual IShorokooTensorValue? OwnValue => null;

        /// <summary>This sequence's own value, or its copy's, without the liveness check.</summary>
        internal IShorokooTensorValue UncheckedValue
            => OwnValue ?? throw new InvalidOperationException(
                $"Sequence {this} holds no runtime value of its own; a run reads it through a copy.");

        /// <summary>
        /// Whether a run on <paramref name="backend"/> can be handed this sequence's own value as it
        /// stands. False for a sequence that is only a list of tensors, which a run reads through a
        /// sequence value its backend builds.
        /// </summary>
        internal virtual bool FeedsInPlace(IShorokooBackend backend) => false;

        /// <summary>
        /// The memory a run on <paramref name="backend"/> reads a sequence in: host memory of its
        /// runtime, since ONNX Runtime reads a sequence's elements back with a host copy whatever its
        /// provider (see <c>IShorokooBackend.CreateSequence</c>). What a copy is keyed by.
        /// </summary>
        internal static MemoryLocation RunMemoryOf(IShorokooBackend backend)
            => new(MemorySpace.Host, backend.RuntimeIdentity);

        /// <summary>
        /// The copy of this sequence a run on <paramref name="backend"/> reads where it cannot be
        /// handed this one as it stands: built on that backend the first time, held by this
        /// sequence and reused by every later read. The caller holds this sequence's lock, or has
        /// checked it is alive.
        /// </summary>
        internal TensorDataSequence SharedCopyFor(IShorokooBackend backend)
        {
            var where = RunMemoryOf(backend);
            lock (_copyGate)
            {
                if (_copies is not null && _copies.TryGetValue(where, out var held) && !held.IsDisposed)
                    return held;
                var made = BuildCopy(backend);
                (_copies ??= [])[where] = made;
                return made;
            }
        }

        /// <summary>
        /// The copy a run on <paramref name="backend"/> that has taken this sequence consumes in
        /// its place, taken with <paramref name="death"/>: the one already held, or a fresh one.
        /// </summary>
        internal TensorDataSequence TakeRunCopy(IShorokooBackend backend, TensorDeath death)
        {
            var where = RunMemoryOf(backend);
            TensorDataSequence? held = null;
            lock (_copyGate)
                if (_copies is not null && _copies.Remove(where, out var found)) held = found;
            if (held is not null)
            {
                if (held.TryTake(death) == TakeOutcome.Taken) return held;
                held.Retire(TensorDeath.Retired);
            }
            var fresh = BuildCopy(backend);
            if (fresh.TryTake(death) != TakeOutcome.Taken)
                throw new InvalidOperationException("A copy made for one run was held by another.");
            return fresh;
        }

        /// <summary>A sequence over the value <see cref="BuildValueOn"/> makes on
        /// <paramref name="backend"/>, allocated by it and released through it.</summary>
        private TensorDataSequence BuildCopy(IShorokooBackend backend)
        {
            var value = BuildValueOn(backend);
            try
            {
                return OnnxUtils.CreateTensorDataSequenceFromValue(DType, value, backend);
            }
            catch
            {
                backend.Release(value);
                throw;
            }
        }

        /// <summary>
        /// This sequence's contents built as one sequence value of <paramref name="backend"/>'s
        /// runtime, which the caller owns. Without the liveness check.
        /// </summary>
        private protected abstract IShorokooTensorValue BuildValueOn(IShorokooBackend backend);

        /// <summary>
        /// Called when one of this sequence's own elements is written: every sequence value runs
        /// built from it was copied from the old contents, so each is retired and the next run builds
        /// a fresh one — the rule a tensor's own copies follow, applied to the copies of the sequence
        /// holding it.
        /// </summary>
        internal void ElementWritten() => RetireCopies();

        /// <summary>Retires every copy runs built of this sequence, for a caller that knows they will
        /// not be read again soon; see <see cref="TensorData.ReleaseRunCopies"/>.</summary>
        internal void ReleaseRunCopies() => RetireCopies();

        /// <summary>Retires every copy runs built of this sequence, which lives no longer than
        /// it.</summary>
        private void RetireCopies()
        {
            List<TensorDataSequence> retired;
            lock (_copyGate)
            {
                if (_copies is null) return;
                retired = [.. _copies.Values];
                _copies = null;
            }
            foreach (var copy in retired) copy.Retire(TensorDeath.Retired);
        }

        internal abstract TensorData GetAt(int index);

        public TensorData this[int index] => GetAt(index);

        internal abstract IEnumerator<TensorData> InternalGetEnumerator();

        public IEnumerator<TensorData> GetEnumerator() => InternalGetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();


        public TensorDataSequence<T> As<T>() where T : IVarType => (TensorDataSequence<T>)this;

        public static TensorDataSequence Empty(DType dtype)
        {
            return Create([], dtype);
        }

        /// <summary>
        /// Managed zero-element sequence. ONNX Runtime's C# binding cannot create a
        /// zero-element sequence value, so the empty case is represented purely on the
        /// managed side; it supports Count/DType/enumeration but cannot be fed to an
        /// ONNX Runtime session as an input (use the in-graph SequenceEmpty op there).
        /// </summary>
        private sealed class EmptyTensorDataSequence<T> : TensorDataSequence<T>
            where T : IVarType
        {
            public override int Count
            {
                get
                {
                    ThrowIfDisposed();
                    return 0;
                }
            }

            public override TensorData<T> this[int index]
            {
                get
                {
                    ThrowIfDisposed();
                    throw new ArgumentOutOfRangeException(nameof(index), "The sequence is empty.");
                }
            }

            // The validation cannot live in the iterator: an iterator method's body does not run
            // until the first MoveNext, so a disposed sequence would hand back an enumerator and
            // only throw once someone stepped it.
            public override IEnumerator<TensorData<T>> GetEnumerator()
            {
                ThrowIfDisposed();
                return Empty();

                static IEnumerator<TensorData<T>> Empty() { yield break; }
            }

            private protected override void ReleaseMemory() { }

            private protected override bool AddressableBy(ComputeContext target) => true;

            private protected override bool IsHostReadable => true;

            /// <summary>
            /// There is nothing to build: ONNX Runtime's binding cannot build a zero-element
            /// sequence value, which is why the empty case is represented on the managed side alone.
            /// </summary>
            private protected override IShorokooTensorValue BuildValueOn(IShorokooBackend backend)
                => throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorValue", ToString(),
                    "This sequence has no backend-runtime value to feed a session, and none can be "
                    + "built: ONNX Runtime cannot represent a zero-element sequence. Build an empty one "
                    + "inside the graph with the SequenceEmpty op instead of passing one in.");
        }

        /// <summary>
        /// A sequence that is just a list of tensors, holding the very tensors it was given — the
        /// shape a copy of a sequence takes, and the one sequence whose elements are tensors of its
        /// own rather than copies minted per read.
        ///
        /// <para>It is not <see cref="IOnnxData"/>, for the same reason
        /// <see cref="HostTensorData{T}"/> is not: there is no runtime value here until something
        /// asks for one. A run reads it through a sequence value its backend builds, held by this
        /// sequence for the next run — see <see cref="SharedCopyFor"/>.</para>
        /// </summary>
        private sealed class ListTensorDataSequence<T> : TensorDataSequence<T>
            where T : IVarType
        {
            private readonly List<TensorData<T>> _elements;

            internal ListTensorDataSequence(List<TensorData<T>> elements)
            {
                _elements = elements;
                // Each element is this sequence's own, and what a run builds from this sequence is
                // copied from them: a write to one has to retire that too.
                foreach (var element in elements) element.BelongsTo(this);
            }

            public override int Count
            {
                get { ThrowIfDisposed(); return _elements.Count; }
            }

            public override TensorData<T> this[int index]
            {
                get { ThrowIfDisposed(); return _elements[index]; }
            }

            public override IEnumerator<TensorData<T>> GetEnumerator()
            {
                ThrowIfDisposed();
                return _elements.GetEnumerator();
            }

            /// <summary>
            /// This sequence's elements as one sequence value of <paramref name="backend"/>'s
            /// runtime.
            ///
            /// <para>Each element is copied rather than handed over. <c>CreateSequence</c> takes
            /// the values it is given: the sequence owns them from then on and releases them with
            /// itself, which would free storage the elements still own and still read
            /// (Shorokoo/Shorokoo#180). The copy is the same one <c>TensorDataSequence.Create</c>
            /// makes for the same reason, taken on this backend rather than the process default.</para>
            /// </summary>
            private protected override IShorokooTensorValue BuildValueOn(IShorokooBackend backend)
            {
                ArgumentNullException.ThrowIfNull(backend);
                var inner = new List<IShorokooTensorValue>(_elements.Count);
                try
                {
                    // Each element on this backend first, so that a literal materializes here
                    // rather than somewhere else and is then dragged across; BackendTransfer then
                    // has nothing to move for an element already of this runtime.
                    foreach (var element in _elements)
                        inner.Add(BackendTransfer.CopyTo(backend, element.ToTensorValue(backend)));
                }
                catch
                {
                    // These copies belong to nobody yet; on failure nothing else will release them.
                    foreach (var copy in inner) backend.Release(copy);
                    throw;
                }

                // Outside the catch on purpose: CreateSequence takes the copies over, and releases
                // them itself if it cannot. Inside, a failure there would free each of them twice.
                return backend.CreateSequence(inner);
            }

            /// <summary>
            /// Ends the elements, which are this sequence's own — a copy of a sequence is made of
            /// copies — the way this sequence ended, so that an element read afterwards says which:
            /// consumed by a run, say, rather than merely gone. An element a run is reading on its
            /// own is retired instead — dead from here, and released when that run returns — since
            /// this sequence is gone either way.
            /// </summary>
            private protected override void ReleaseMemory()
            {
                var death = Death ?? TensorDeath.Deleted;
                foreach (var element in _elements)
                {
                    if (element.TryTake(death) == TakeOutcome.Taken) element.ReleaseTaken();
                    else element.Retire(death);
                }
            }

            private protected override bool AddressableBy(ComputeContext target)
                => _elements.TrueForAll(target.CanAddress);

            private protected override bool IsHostReadable
                => _elements.TrueForAll(static element => element.IsHostResident);

            // As a tensor's To attaches it: under the target's device-memory budget, all of them or
            // none, refused where the ones not yet on its books would take it past its limit.
            private protected override void AttachElementsTo(ComputeContext target)
                => target.AttachAllWithinBudget(_elements, nameof(To));
        }

        /// <summary>A sequence holding these tensors as they are, rather than rebuilding them
        /// through a runtime. The sequence owns them from then on.</summary>
        internal static TensorDataSequence OfElements(List<TensorData> data, DType dtype)
            => data.Count == 0
                ? CreateEmpty(dtype)
                : (TensorDataSequence)OnnxUtils.CallGeneric(
                    dtype.ToIVarType(), typeof(TensorDataSequence), nameof(internalOfElements), data);

        internal static TensorDataSequence internalOfElements<T>(List<TensorData> data) where T : IVarType
            => new ListTensorDataSequence<T>([.. data.Cast<TensorData<T>>()]);

        internal static TensorDataSequence CreateEmpty(DType dtype)
            => (TensorDataSequence)OnnxUtils.CallGeneric(dtype.ToIVarType(), typeof(TensorDataSequence), nameof(internalCreateEmpty));

        internal static TensorDataSequence internalCreateEmpty<T>() where T : IVarType
            => new EmptyTensorDataSequence<T>();

        public static TensorDataSequence Create(List<TensorData> data, DType? dtype)
        {
            if (dtype is null  && data.Count == 0)
                throw new InvalidTensorOperationException(ErrorCodes.CR002, "TensorDataSequence.Create", $"data count: {data.Count}, dtype: null",
                    "Data cannot be empty when dtype is null");

            dtype ??= data[0].DType;
            // ORT's C# binding cannot build a zero-element sequence value; represent
            // the empty case purely on the managed side instead.
            if (data.Count == 0)
                return CreateEmpty(dtype);
            return OnnxUtils.CreateTensorDataSequence(dtype, data);
        }

        /// <summary>
        /// This sequence where <paramref name="target"/> can use it: the very same object when the
        /// target's backend can read every element as it stands — its elements then attached to the
        /// target, as <see cref="TensorData.To"/> attaches a tensor — and otherwise a copy of the
        /// whole sequence in the target's memory. This sequence is untouched either way.
        ///
        /// <para>A copy of the whole sequence rather than of the elements that need it, because a
        /// sequence owns its elements: one whose elements were partly another sequence's would
        /// delete them from under it when disposed.</para>
        /// </summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This sequence has been disposed.</exception>
        public TensorDataSequence To(ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            ThrowIfDisposed();
            if (!AddressableBy(target)) return CopyTo(target);
            AttachElementsTo(target);
            return this;
        }

        /// <summary>An independent copy of this sequence, its elements copied into
        /// <paramref name="target"/>'s memory and attached to it. This sequence is untouched.</summary>
        /// <exception cref="ArgumentNullException"><paramref name="target"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">This sequence has been disposed.</exception>
        public TensorDataSequence CopyTo(ComputeContext target)
        {
            ArgumentNullException.ThrowIfNull(target);
            ThrowIfDisposed();
            return Rebuild(element => element.CopyTo(target));
        }

        /// <summary>
        /// This sequence where the host can read it: the very same object when every element
        /// already is host-readable, and otherwise a copy in the framework's own host memory.
        /// </summary>
        /// <exception cref="ObjectDisposedException">This sequence has been disposed.</exception>
        public TensorDataSequence ToHost()
        {
            ThrowIfDisposed();
            return IsHostReadable ? this : CopyTo(ComputeContext.Host);
        }

        /// <summary>A list sequence of what <paramref name="copyElement"/> makes of each element,
        /// releasing everything it made if one of them fails.</summary>
        private TensorDataSequence Rebuild(Func<TensorData, TensorData> copyElement)
        {
            List<TensorData> copies = new(Count);
            TensorData? minted = null;
            try
            {
                foreach (var element in this)
                {
                    // Held so it can be released: a sequence that mints its elements per read hands
                    // this loop a tensor nobody else will ever see, and letting go of it here is the
                    // only chance -- otherwise every copy of a session's sequence output leaves one
                    // runtime value per element to its finalizer.
                    minted = MintsElementsPerRead ? element : null;
                    var copy = copyElement(element);
                    copies.Add(copy);
                    if (minted is not null && !ReferenceEquals(copy, minted)) minted.Delete();
                    minted = null;
                }
            }
            catch
            {
                // What this loop made belongs to nobody: the sequence that would have owned it is
                // never constructed.
                minted?.TryDelete();
                foreach (var copy in copies) copy.TryDelete();
                throw;
            }
            return OfElements(copies, DType);
        }

        /// <summary>Whether <paramref name="target"/>'s backend can read every element as it
        /// stands.</summary>
        private protected abstract bool AddressableBy(ComputeContext target);

        /// <summary>Whether the host can read every element as it stands.</summary>
        private protected abstract bool IsHostReadable { get; }

        /// <summary>Attaches this sequence's own elements to <paramref name="target"/>, for a
        /// sequence whose elements are tensors of its own.</summary>
        private protected virtual void AttachElementsTo(ComputeContext target)
        {
        }

        /// <summary>
        /// Whether reading an element mints a tensor of its own rather than handing out one this
        /// sequence holds. True where a runtime copies the element out per read, which makes the
        /// reader responsible for releasing it.
        /// </summary>
        private protected virtual bool MintsElementsPerRead => false;
    }

    public sealed class OnnxTensorDataSequence<T> : TensorDataSequence<T>, IOnnxData, IDisposable
        where T : IVarType
    {
        private readonly IShorokooTensorValue backing;

        /// <summary>
        /// A sequence over <paramref name="value"/>, without saying which backend made it. The
        /// sequence takes the value over and releases it by disposing it; host memory is assumed,
        /// which is where a value with no producer to ask can be read.
        /// </summary>
        public OnnxTensorDataSequence(IShorokooTensorValue value)
            : this(value, UnrecordedBackend.Instance, MemorySpace.Host)
        {
        }

        /// <summary>
        /// A sequence over <paramref name="value"/>, which <paramref name="allocatingBackend"/>
        /// made and releases. It is in that backend's memory: a sequence value is not a tensor and
        /// cannot say where it is, so the backend is taken at its word, and a sequence a card's
        /// execution provider produced is not labelled host memory it may not be.
        /// </summary>
        internal OnnxTensorDataSequence(IShorokooTensorValue value, IShorokooBackend allocatingBackend)
            : this(value, allocatingBackend, allocatingBackend.MemorySpace)
        {
        }

        private OnnxTensorDataSequence(
            IShorokooTensorValue value, IShorokooBackend allocatingBackend, MemorySpace space)
        {
            this.backing = value ?? throw new ArgumentNullException(nameof(value));
            AllocatingBackend = allocatingBackend ?? throw new ArgumentNullException(nameof(allocatingBackend));
            Space = space;
        }

        /// <summary>The backend that made this sequence's value, and releases it.</summary>
        internal IShorokooBackend AllocatingBackend { get; }

        /// <summary>Where this sequence's value is.</summary>
        internal MemorySpace Space { get; }

        /// <summary>Where this sequence's value is, with the runtime that made it.</summary>
        private MemoryLocation Location => new(Space, AllocatingBackend.RuntimeIdentity);

        /// <summary>
        /// The backing backend-runtime sequence value, which this sequence owns: disposing the
        /// sequence releases it, and nothing else may hold or free it.
        /// </summary>
        public IShorokooTensorValue Value
        {
            get
            {
                ThrowIfDisposed();
                return backing;
            }
        }

        public override int Count
        {
            get { ThrowIfDisposed(); return backing.GetValueCount(); }
        }

        /// <summary>
        /// The element at <paramref name="index"/>, on storage of its own: the runtime copies the
        /// element out rather than aliasing the sequence, so the returned tensor owns what it
        /// hands back and deleting it leaves this sequence intact.
        ///
        /// <para>The copy is made by the runtime holding the sequence, so the element's allocating
        /// backend is this sequence's, and it is released through that backend.</para>
        /// </summary>
        public override TensorData<T> this[int index]
        {
            get
            {
                ThrowIfDisposed();
                var val = backing.GetValue(index);
                try
                {
                    return (TensorData<T>)OnnxUtils.CreateTensorDataFromValue(
                        new Shape(val.Shape), (DType)(int)val.ElementType, val, AllocatingBackend);
                }
                catch
                {
                    AllocatingBackend.Release(val);
                    throw;
                }
            }
        }

        private protected override bool MintsElementsPerRead => true;

        private protected override void ReleaseMemory() => AllocatingBackend.Release(backing);

        private protected override bool AddressableBy(ComputeContext target)
            => target.ResolvedBackend.CanAddress(Location);

        private protected override bool IsHostReadable => Space.IsHost;

        /// <summary>The value this sequence already holds, whatever backend is asked for.</summary>
        internal override IShorokooTensorValue? OwnValue => backing;

        /// <summary>
        /// Whether a run on <paramref name="backend"/> can be handed this value as it stands: the
        /// backend can address the memory it is in. A sequence of another runtime is copied into the
        /// running one — which reads the source, so only one in host memory can be.
        /// </summary>
        internal override bool FeedsInPlace(IShorokooBackend backend) => backend.CanAddress(Location);

        /// <summary>A copy of this sequence's value built on <paramref name="backend"/>, element by
        /// element through the host.</summary>
        private protected override IShorokooTensorValue BuildValueOn(IShorokooBackend backend)
            => BackendTransfer.CopyTo(backend, backing);

        public override IEnumerator<TensorData<T>> GetEnumerator()
        {
            ThrowIfDisposed();
            return Elements(this);

            static IEnumerator<TensorData<T>> Elements(OnnxTensorDataSequence<T> self)
            {
                for (int i = 0; i < self.Count; i++)
                    yield return self[i];
            }
        }

        // No finalizer, for the reason OnnxTensorData<T> has none: the backing value has its own.
    }
}
