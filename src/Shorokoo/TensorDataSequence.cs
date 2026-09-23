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
    /// <para>A sequence has a life of its own, like a tensor's: a run reading it holds a reader lock
    /// on it for as long as it runs, <see cref="Dispose"/> ends it and throws while a run holds one,
    /// and a disposed sequence says so on every path to its elements. It knows no compute context;
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
        // nothing else.
        private readonly object _gate = new();
        private bool _dead;
        private int _locks;

        /// <summary>
        /// True once this sequence has been disposed and its storage released. Its dtype and
        /// <see cref="ToString"/> stay readable as metadata; every path to the elements throws.
        /// </summary>
        public bool IsDisposed => Volatile.Read(ref _dead);

        /// <summary>Guards every path to the sequence's elements.</summary>
        protected void ThrowIfDisposed()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(GetType().Name,
                    $"Sequence {this} has been disposed; its element storage is gone and reading " +
                    "it would read freed memory.");
        }

        /// <summary>
        /// Ends this sequence and releases its storage through the backend that made it. A sequence
        /// already disposed is left as it is.
        /// </summary>
        /// <exception cref="InvalidOperationException">A run is reading this sequence; releasing
        /// its storage now would free it under the run.</exception>
        public void Dispose()
        {
            lock (_gate)
            {
                if (_dead) return;
                if (_locks > 0)
                    throw new InvalidOperationException(
                        $"Sequence {this} is being read by a run, so it cannot be disposed: its "
                        + "storage would be freed under the run. Wait for the run to return.");
                _dead = true;
            }
            ReleaseMemory();
        }

        /// <summary>Releases this sequence's storage. Called once, by <see cref="Dispose"/>.</summary>
        private protected abstract void ReleaseMemory();

        /// <summary>Takes a reader lock for the length of a run. Refuses a sequence that has been
        /// disposed: there is nothing left to read.</summary>
        /// <exception cref="ObjectDisposedException">The sequence has been disposed.</exception>
        internal void AcquireReadLock()
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                _locks++;
            }
        }

        /// <summary>Drops a reader lock.</summary>
        internal void ReleaseReadLock()
        {
            lock (_gate) _locks--;
        }

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
        /// builds it here.
        ///
        /// <para>The value returned is the sequence's own: read it, do not dispose it.</para>
        /// </summary>
        internal virtual IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
        {
            ThrowIfDisposed();
            // The empty sequence, and only it: ONNX Runtime's binding cannot build a zero-element
            // sequence value, which is why the empty case is represented on the managed side alone.
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorValue", ToString(),
                "This sequence has no backend-runtime value to feed a session, and none can be "
                + "built: ONNX Runtime cannot represent a zero-element sequence. Build an empty one "
                + "inside the graph with the SequenceEmpty op instead of passing one in.");
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
        }

        /// <summary>
        /// A sequence that is just a list of tensors, holding the very tensors it was given — the
        /// shape a copy of a sequence takes, and the one sequence whose elements are tensors of its
        /// own rather than copies minted per read.
        ///
        /// <para>It is not <see cref="IOnnxData"/>, for the same reason
        /// <see cref="HostTensorData{T}"/> is not: there is no runtime value here until something
        /// asks for one. Feeding such a sequence to a session builds it then, on that session's
        /// backend -- see <see cref="ToTensorValue(IShorokooBackend)"/>.</para>
        /// </summary>
        private sealed class ListTensorDataSequence<T> : TensorDataSequence<T>
            where T : IVarType
        {
            private readonly List<TensorData<T>> _elements;

            // What these elements have been built into, per backend.
            private readonly MaterializedValues _materialized = new();

            internal ListTensorDataSequence(List<TensorData<T>> elements)
            {
                _elements = elements;
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
            /// runtime, built the first time that backend asks and kept for the next time.
            ///
            /// <para>Each element is copied rather than handed over. <c>CreateSequence</c> takes
            /// the values it is given: the sequence owns them from then on and releases them with
            /// itself, which would free storage the elements still own and still read
            /// (Shorokoo/Shorokoo#180). The copy is the same one <c>TensorDataSequence.Create</c>
            /// makes for the same reason, taken on this backend rather than the process default.</para>
            /// </summary>
            internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
            {
                ArgumentNullException.ThrowIfNull(backend);
                ThrowIfDisposed();

                return _materialized.Get(backend, Build);
            }

            /// <summary>Builds this sequence's elements into one sequence value of
            /// <paramref name="backend"/>'s runtime.</summary>
            private IShorokooTensorValue Build(IShorokooBackend backend)
            {
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
            /// Releases the sequence values built from these elements, and deletes the elements —
            /// which are this sequence's own: a copy of a sequence is made of copies.
            ///
            /// <para>An element a run is reading on its own refuses its deletion; the others are
            /// deleted all the same, and the refusal is what this throws.</para>
            /// </summary>
            private protected override void ReleaseMemory()
            {
                _materialized.Invalidate();
                Exception? refused = null;
                foreach (var element in _elements)
                {
                    try { element.Dispose(); }
                    catch (InvalidOperationException ex) { refused ??= ex; }
                }
                if (refused is not null) throw refused;
            }

            private protected override bool AddressableBy(ComputeContext target)
                => _elements.TrueForAll(target.CanAddress);

            private protected override bool IsHostReadable
                => _elements.TrueForAll(static element => element.IsHostResident);

            private protected override void AttachElementsTo(ComputeContext target)
            {
                foreach (var element in _elements) target.Attach(element);
            }
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
            => target.ResolvedBackend.CanAddress(new MemoryLocation(Space, AllocatingBackend.RuntimeIdentity));

        private protected override bool IsHostReadable => Space.IsHost;

        /// <summary>
        /// The value this sequence already holds, whatever backend is asked for. It was made by
        /// one runtime and belongs to it; a session of another rebuilds it as it is fed, which is
        /// a thing only that session can do.
        /// </summary>
        internal override IShorokooTensorValue ToTensorValue(IShorokooBackend backend)
            => Value;

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
