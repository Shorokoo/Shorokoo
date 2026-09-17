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
        /// </summary>
        private sealed class ListTensorDataSequence<T> : TensorDataSequence<T>
            where T : IVarType
        {
            private readonly List<TensorData<T>> _elements;

            internal ListTensorDataSequence(List<TensorData<T>> elements) => _elements = elements;

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

            /// <summary>Disposes the elements, each of which then decides for itself whether that
            /// releases anything -- a reader among them releases nothing.</summary>
            public override void Dispose()
            {
                if (IsDisposed) return;
                IsDisposed = true;
                foreach (var element in _elements) element.Dispose();
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
        /// The compute context these elements belong to, or null for the framework's own host
        /// memory. Set by the transfer operations; a sequence built any other way inherits nothing
        /// and reports null.
        /// </summary>
        public Shorokoo.Runtime.ComputeContext? Context { get; internal set; }

        /// <summary>Moves this sequence's owned elements to <paramref name="target"/>, element by
        /// element and under each element's own rules. Elements it only has access to are left
        /// where they are: moving what you do not own is what the non-owning case forbids.</summary>
        public TensorDataSequence TransferTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target, static (t, c) => t.TransferTo(c));

        /// <summary>Copies this sequence's owned elements into <paramref name="target"/>'s memory,
        /// leaving this sequence untouched.</summary>
        public TensorDataSequence CopyTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target, static (t, c) => t.CopyTo(c));

        /// <summary>Hands <paramref name="target"/> a reader for this sequence's owned elements,
        /// taking no ownership of any of them.</summary>
        public TensorDataSequence GiveAccessTo(Shorokoo.Runtime.ComputeContext? target)
            => Rebuild(target, static (t, c) => t.GiveAccessTo(c));

        private TensorDataSequence Rebuild(
            Shorokoo.Runtime.ComputeContext? target,
            Func<TensorData, Shorokoo.Runtime.ComputeContext?, TensorData> operation)
        {
            ThrowIfDisposed();
            List<TensorData> moved = [.. this.Select(e => e.OwnsMemory ? operation(e, target) : e)];
            var rebuilt = OfElements(moved, DType);
            rebuilt.Context = target;
            return rebuilt;
        }

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
                ThrowIfDisposed();
                return backing;
            }
        }

        public override int Count => Value.GetValueCount();

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
                var val = Value.GetValue(index);
                return (TensorData<T>)OnnxUtils.CreateTensorDataFromValue(
                    new Shape(val.Shape), (DType)(int)val.ElementType, val, Context);
            }
        }

        public OnnxTensorDataSequence(IShorokooTensorValue value) : base()
        {
            this.backing = value;
        }

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

        #region IDisposable

        /// <summary>
        /// Releases the backing sequence value. Idempotent; every read afterwards throws
        /// <see cref="ObjectDisposedException"/>. No finalizer, for the reason
        /// <see cref="OnnxTensorData{T}.Dispose"/> gives.
        /// </summary>
        public override void Dispose()
        {
            if (IsDisposed) return;
            IsDisposed = true;
            backing.Dispose();
        }

        #endregion
    }
}
