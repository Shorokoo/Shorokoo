using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo;
using Shorokoo.Core.Inference.Abstractions;
using static Shorokoo.Globals;
using System.Collections;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

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

    public abstract class TensorDataSequence : IData, IDisposable, IReadOnlyList<TensorData>
    {
        public DType DType { get; private set; }

        public abstract int Count { get; }

        int IReadOnlyCollection<TensorData>.Count => this.Count;

        internal TensorDataSequence(DType dtype)
        {
            this.DType = dtype;
        }

        /// <summary>
        /// True once this sequence's storage has been released. Its dtype and
        /// <see cref="ToString"/> stay readable as metadata; every path to the elements throws.
        /// </summary>
        public bool IsDisposed { get; protected set; }

        /// <summary>Guards every path to the sequence's elements.</summary>
        protected void ThrowIfDisposed()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(GetType().Name,
                    $"Sequence {this} has been disposed; its element storage is gone and reading " +
                    "it would read freed memory.");
        }

        public override string ToString()
        {
            return $"sequence:{this.DType.ToString()}";
        }

        /// <summary>
        /// This sequence as a value of the process-wide backend's runtime, for a caller with no
        /// context to name. <see cref="TensorData.ToTensorValue()"/>'s counterpart.
        /// </summary>
        internal IShorokooTensorValue ToTensorValue() => ToTensorValue(InferenceBackend.Default);

        /// <summary>
        /// This sequence as a value of <paramref name="backend"/>'s runtime — the form the feed
        /// sites take, so a sequence is built by the backend whose session is about to read it.
        /// A sequence that already holds a runtime value hands it over and ignores the argument,
        /// since a value belongs to the runtime that made it; one that is only a list of tensors
        /// builds it here.
        ///
        /// <para>The value returned is the sequence's own: read it, do not dispose it.</para>
        /// </summary>
        internal virtual IShorokooTensorValue ToTensorValue(IShorokooInferenceBackend backend)
        {
            ThrowIfDisposed();
            // The empty sequence, and only it: ONNX Runtime's binding cannot build a zero-element
            // sequence value, which is why the empty case is represented on the managed side alone.
            throw new InvalidTensorOperationException(ErrorCodes.FW007, "ToTensorValue", ToString(),
                "This sequence has no inference-runtime value to feed a session, and none can be "
                + "built: ONNX Runtime cannot represent a zero-element sequence. Build an empty one "
                + "inside the graph with the SequenceEmpty op instead of passing one in.");
        }

        internal abstract TensorData GetAt(int index);

        /// <summary>
        /// The allocation behind this sequence's runtime value, which a run locks while it feeds
        /// it. <see cref="TensorStorage.None"/> where there is nothing a run could be reading --
        /// the empty sequence, which has no value to build at all.
        /// </summary>
        internal virtual TensorStorage Storage => TensorStorage.None;

        // Whether this sequence's reference on its allocation has already been dropped. A context's
        // disposal and an explicit Dispose can both drop it, and a reference dropped twice frees
        // bytes something else still names.
        private int _referenceDropped;

        /// <summary>Drops this sequence's reference on its allocation, freeing the runtime value
        /// behind it if nothing else names it and no run is reading it. Idempotent.</summary>
        internal void DropReference()
        {
            var storage = Storage;
            if (ReferenceEquals(storage, TensorStorage.None)) return;
            if (Interlocked.Exchange(ref _referenceDropped, 1) != 0) return;
            storage.DropHandleReference();
        }

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

            public override void Dispose() => IsDisposed = true;
        }

        /// <summary>
        /// A sequence that is just a list of tensors, holding the very tensors it was given.
        ///
        /// <para>Which is what the transfer operations need. Building one through the runtime --
        /// <see cref="Create"/>'s ordinary path -- makes a fresh sequence value and fresh elements
        /// owning it, so every element's context and ownership would be replaced by the act of
        /// rebuilding, and a GiveAccessTo would hand back owners. Holding the elements keeps what
        /// each of them decided.</para>
        ///
        /// <para>It is not <see cref="IOnnxData"/>, for the same reason
        /// <see cref="HostTensorData{T}"/> is not: there is no runtime value here until something
        /// asks for one. Feeding such a sequence to a session builds it then, on that session's
        /// backend -- see <see cref="ToTensorValue(IShorokooInferenceBackend)"/>.</para>
        /// </summary>
        private sealed class ListTensorDataSequence<T> : TensorDataSequence<T>
            where T : IVarType
        {
            private readonly List<TensorData<T>> _elements;

            // What these elements have been built into, per backend.
            private readonly MaterializedValues _materialized = new();

            // The allocation a run locks while it reads this sequence. Its bytes are the runtime's
            // copies of the elements, so freeing it is what Dispose means -- and a run holding a
            // lock on it defers that free until it returns.
            private readonly TensorStorage _storage;

            internal ListTensorDataSequence(List<TensorData<T>> elements)
            {
                _elements = elements;
                _storage = new TensorStorage(MemorySpace.Host, _materialized.Invalidate);
                _storage.AddHandleReference();
            }

            internal override TensorStorage Storage => _storage;

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
            internal override IShorokooTensorValue ToTensorValue(IShorokooInferenceBackend backend)
            {
                ArgumentNullException.ThrowIfNull(backend);
                ThrowIfDisposed();

                return _materialized.Get(backend, Build);
            }

            /// <summary>Builds this sequence's elements into one sequence value of
            /// <paramref name="backend"/>'s runtime.</summary>
            private IShorokooTensorValue Build(IShorokooInferenceBackend backend)
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
                    foreach (var copy in inner) copy.Dispose();
                    throw;
                }

                // Outside the catch on purpose: CreateSequence takes the copies over, and releases
                // them itself if it cannot. Inside, a failure there would free each of them twice.
                return backend.CreateSequence(inner);
            }

            /// <summary>
            /// Lets go of this sequence's element handles and of the sequence values it had built
            /// on backends.
            ///
            /// <para>Each element here is this sequence's own handle -- a rebuild gives every
            /// element one of its own, even where the bytes are shared -- so disposing them lets
            /// go of this sequence's names for those bytes and leaves any other handle on them
            /// reading.</para>
            /// </summary>
            public override void Dispose()
            {
                if (IsDisposed) return;
                IsDisposed = true;
                foreach (var element in _elements) element.Dispose();
                DropReference();
            }
        }

        /// <summary>A sequence holding these tensors as they are, rather than rebuilding them
        /// through a runtime.</summary>
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
        /// The compute context these elements belong to.
        /// <see cref="Shorokoo.Runtime.ComputeContext.Host"/> is the framework's own host memory.
        /// Set by the transfer operations; a sequence built any other way inherits nothing and
        /// reports the host context.
        /// </summary>
        public Shorokoo.Runtime.ComputeContext Context { get; internal set; }
            = Shorokoo.Runtime.ComputeContext.Host;

        /// <summary>Moves this sequence's elements to <paramref name="target"/>, element by
        /// element and under each element's own rules.</summary>
        public TensorDataSequence TransferTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target ?? Shorokoo.Runtime.ComputeContext.Host,
                static (t, c) => t.TransferTo(c));

        /// <summary>Copies this sequence's elements into <paramref name="target"/>'s memory,
        /// leaving this sequence untouched.</summary>
        public TensorDataSequence CopyTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target ?? Shorokoo.Runtime.ComputeContext.Host,
                static (t, c) => t.CopyTo(c));

        /// <summary>Hands <paramref name="target"/> a second handle on each of this sequence's
        /// elements, leaving this sequence's own handles alone.</summary>
        public TensorDataSequence GiveAccessTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target ?? Shorokoo.Runtime.ComputeContext.Host,
                static (t, c) => t.GiveAccessTo(c));

        /// <param name="target">The context the rebuilt sequence belongs to.</param>
        /// <param name="operation">The per-element operation to apply.</param>
        private TensorDataSequence Rebuild(
            Shorokoo.Runtime.ComputeContext target,
            Func<TensorData, Shorokoo.Runtime.ComputeContext, TensorData> operation)
        {
            ThrowIfDisposed();
            List<TensorData> moved = new(Count);
            // Only what this rebuild allocated may be released if it fails. A same-space move
            // hands the rebuilt element the source's own storage, so disposing it on the way out
            // would free bytes the source element still names -- turning a failed transfer into
            // destroyed data, which is worse than the straddling it was cleaning up after.
            List<TensorData> allocated = new(Count);
            // What each source element was before the move, so a failure can put it back. Without
            // it a failed transfer left the source elements handed over -- referenceless, naming
            // the target -- so the caller's sequence died with a context it was never given to.
            List<(TensorData Element, Shorokoo.Runtime.ComputeContext Context)> handedOver = [];
            TensorData? minted = null;
            try
            {
                foreach (var element in this)
                {
                    // Held so the cleanup below can release it: a sequence that mints its elements
                    // per read hands this loop a tensor nobody else will ever see, and an operation
                    // that throws on it would otherwise leave that one to a finalizer too.
                    minted = MintsElementsPerRead ? element : null;
                    var before = element.Context;
                    var rebuiltElement = operation(element, target);
                    if (!ReferenceEquals(element.Context, before) && !MintsElementsPerRead)
                        handedOver.Add((element, before));
                    minted = null;
                    moved.Add(rebuiltElement);
                    if (!ReferenceEquals(rebuiltElement, element)
                        && !ReferenceEquals(rebuiltElement.Storage, element.Storage))
                        allocated.Add(rebuiltElement);
                    // A sequence whose elements are copied out per read hands this loop a tensor
                    // nobody else will ever see again, so letting go of it here is the only chance
                    // -- otherwise every rebuild of a session's sequence output leaves one runtime
                    // value, a device allocation on a card, to its context's disposal, and the
                    // default context is never disposed. Safe where the rebuilt element shares
                    // these bytes: it holds a reference of its own, so this only drops a name.
                    if (MintsElementsPerRead && !ReferenceEquals(rebuiltElement, element))
                        element.Dispose();
                }
            }
            catch
            {
                // What this loop built belongs to nobody: the sequence that would have owned it is
                // never constructed, so without this each rebuilt element is a runtime value -- a
                // device allocation on a card -- left to its finalizer. The elements it did not
                // reach are untouched, and the ones it moved keep whatever the operation did to
                // them: a transfer that fails part-way leaves the source straddling two contexts,
                // which is a real loose end and not one a cleanup here can tie.
                minted?.Dispose();
                foreach (var element in allocated) element.Dispose();
                foreach (var (element, before) in handedOver) element.ReclaimReference(before);
                throw;
            }
            var rebuilt = OfElements(moved, DType);
            rebuilt.Context = target;
            return rebuilt;
        }

        /// <summary>
        /// Whether reading an element mints a tensor of its own rather than handing out one this
        /// sequence holds. True where a runtime copies the element out per read, which makes the
        /// reader responsible for releasing it.
        /// </summary>
        private protected virtual bool MintsElementsPerRead => false;

        /// <summary>Records which context this sequence belongs to. Overridden where there is a
        /// runtime value for that context to own.</summary>
        internal virtual void BindTo(Shorokoo.Runtime.ComputeContext context) => Context = context;

        public abstract void Dispose();
    }

    public sealed class OnnxTensorDataSequence<T> : TensorDataSequence<T>, IOnnxData, IDisposable
        where T : IVarType
    {
        private readonly IShorokooTensorValue backing;

        /// <summary>
        /// The backing inference-runtime sequence value, which this sequence owns: disposing the
        /// sequence releases it, and nothing else may hold or free it.
        /// </summary>
        public IShorokooTensorValue Value
        {
            get
            {
                ThrowIfGone();
                return backing;
            }
        }

        public override int Count
        {
            get { ThrowIfGone(); return backing.GetValueCount(); }
        }

        /// <summary>
        /// The element at <paramref name="index"/>, on storage of its own: the runtime copies the
        /// element out rather than aliasing the sequence, so the returned tensor owns what it
        /// hands back and disposing it leaves this sequence intact.
        ///
        /// <para>The copy is made by the runtime holding the sequence, in that runtime's memory, so
        /// the element belongs to this sequence's <see cref="TensorDataSequence.Context"/> — the
        /// context whose session produced the sequence. Wrapping it without one left an element the
        /// provider had kept in device memory unable to say where it was, and so unable to be moved
        /// anywhere at all.</para>
        /// </summary>
        public override TensorData<T> this[int index]
        {
            get
            {
                ThrowIfGone();
                var val = backing.GetValue(index);
                return (TensorData<T>)OnnxUtils.CreateTensorDataFromValue(
                    new Shape(val.Shape), (DType)(int)val.ElementType, val, Context);
            }
        }

        // The allocation behind the backing value: this sequence is its handle, and a run locks
        // it while it feeds it. Built here rather than at BindTo so there is always something to
        // lock, and always exactly one thing that frees the value.
        private readonly TensorStorage _storage;

        internal override TensorStorage Storage => _storage;

        private protected override bool MintsElementsPerRead => true;

        /// <summary>
        /// Puts this sequence on <paramref name="context"/>'s books, so that disposing the context
        /// drops this handle's reference on the backing value. Without it a sequence output named
        /// its context but nothing of it was ever released by that context, while its elements
        /// were -- so disposing the context took the elements and left the sequence, which then
        /// answered about data it could no longer reach.
        /// </summary>
        internal override void BindTo(Shorokoo.Runtime.ComputeContext context)
        {
            base.BindTo(context);
            context.AttachSequence(this);
        }

        /// <summary>Refuses a read once this sequence is disposed, or once the allocation behind
        /// its value has been freed.</summary>
        private void ThrowIfGone()
        {
            ThrowIfDisposed();
            if (!_storage.IsLive)
                throw new ObjectDisposedException(GetType().Name,
                    $"Sequence {this} was released with the compute context that produced it.");
        }

        public OnnxTensorDataSequence(IShorokooTensorValue value) : base()
        {
            this.backing = value;
            _storage = new TensorStorage(MemorySpace.Host, value.Dispose);
            _storage.AddHandleReference();
        }

        /// <summary>
        /// The value this sequence already holds, whatever backend is asked for. It was made by
        /// one runtime and belongs to it; a session of another rebuilds it as it is fed, which is
        /// a thing only that session can do.
        /// </summary>
        internal override IShorokooTensorValue ToTensorValue(IShorokooInferenceBackend backend)
            => Value;

        public override IEnumerator<TensorData<T>> GetEnumerator()
        {
            ThrowIfGone();
            return Elements(this);

            static IEnumerator<TensorData<T>> Elements(OnnxTensorDataSequence<T> self)
            {
                for (int i = 0; i < self.Count; i++)
                    yield return self[i];
            }
        }

        #region IDisposable

        /// <summary>
        /// Lets go of this sequence's handle on the backing value, which goes when nothing else
        /// names it and no run is reading it. Idempotent; every read afterwards throws
        /// <see cref="ObjectDisposedException"/>. No finalizer, for the reason
        /// <see cref="OnnxTensorData{T}"/> has none.
        /// </summary>
        public override void Dispose()
        {
            if (IsDisposed) return;
            IsDisposed = true;
            DropReference();
        }

        #endregion
    }
}
