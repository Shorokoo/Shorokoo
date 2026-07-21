using System;
using System.Buffers.Binary;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// Globally unique identifier for a node in a <see cref="InternalComputationGraph"/>.
    /// Wraps a <see cref="UInt128"/>, which is the same width (16 bytes) as a
    /// <see cref="Guid"/>, so a <see cref="FastNodeKey"/> can be losslessly converted
    /// to and from the CG-side <see cref="Shorokoo.Graph.NodeKey"/> by reinterpreting
    /// the 128 underlying bits.
    ///
    /// <para>
    /// The string form is <c>"N{Id}"</c> (decimal), matching the naming convention
    /// produced by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastUseUniqueNames"/>: when a Fast graph is built from a
    /// CG that has been through <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastUseUniqueNames"/> with the
    /// <see cref="InternalComputationGraphConverter.PopulateFromNodes"/>
    /// <c>useSequentialIds</c> flag, each node's <see cref="Id"/> is the small
    /// integer counter (1, 2, …) and <see cref="ToString"/> reproduces
    /// <c>"N1"</c>, <c>"N2"</c>, … directly.
    /// </para>
    /// </summary>
    public readonly struct FastNodeKey : IEquatable<FastNodeKey>, IComparable<FastNodeKey>
    {
        /// <summary>The unique identifier for this node.</summary>
        public readonly UInt128 Id;

        /// <summary>The empty/invalid <see cref="FastNodeKey"/>.</summary>
        public static readonly FastNodeKey Empty = new FastNodeKey(UInt128.Zero);

        public FastNodeKey(UInt128 id) { Id = id; }
        public FastNodeKey(ulong id) { Id = id; }

        /// <summary>
        /// Generates a new random <see cref="FastNodeKey"/>. Uses
        /// <see cref="Guid.NewGuid"/> as the entropy source so each call produces a
        /// fresh UUID-v4-quality 128-bit value.
        /// </summary>
        public static FastNodeKey New() => FromGuid(Guid.NewGuid());

        /// <summary>
        /// Reinterprets the 16 bytes of <paramref name="g"/> as a <see cref="UInt128"/>.
        /// </summary>
        public static FastNodeKey FromGuid(Guid g)
        {
            Span<byte> bytes = stackalloc byte[16];
            g.TryWriteBytes(bytes);
            ulong low = BinaryPrimitives.ReadUInt64LittleEndian(bytes.Slice(0, 8));
            ulong high = BinaryPrimitives.ReadUInt64LittleEndian(bytes.Slice(8, 8));
            return new FastNodeKey(new UInt128(high, low));
        }

        /// <summary>
        /// Reinterprets the 16 bytes of this key's <see cref="Id"/> as a <see cref="Guid"/>.
        /// </summary>
        public Guid ToGuid()
        {
            Span<byte> bytes = stackalloc byte[16];
            ulong low = (ulong)(Id & ulong.MaxValue);
            ulong high = (ulong)(Id >> 64);
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.Slice(0, 8), low);
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.Slice(8, 8), high);
            return new Guid(bytes);
        }

        public static FastNodeKey FromCgKey(NodeKey k) => FromGuid(k.Id);
        public NodeKey ToCgKey() => new NodeKey(ToGuid());

        public bool IsEmpty => Id == UInt128.Zero;

        public bool Equals(FastNodeKey other) => Id.Equals(other.Id);
        public override bool Equals(object? obj) => obj is FastNodeKey other && Equals(other);
        public override int GetHashCode() => Id.GetHashCode();
        public int CompareTo(FastNodeKey other) => Id.CompareTo(other.Id);

        /// <summary>
        /// String form is <c>"N{Id}"</c> (decimal). Mirrors <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastUseUniqueNames"/>
        /// so that small sequential ids map directly to ONNX-friendly names like
        /// <c>"N1"</c>, <c>"N2"</c>, ...
        /// </summary>
        public override string ToString() => $"N{Id}";

        public static bool operator ==(FastNodeKey left, FastNodeKey right) => left.Equals(right);
        public static bool operator !=(FastNodeKey left, FastNodeKey right) => !left.Equals(right);
    }

    /// <summary>
    /// Globally unique identifier for a tensor in a <see cref="InternalComputationGraph"/>.
    /// Composed of the producing node's <see cref="FastNodeKey"/> plus an output index.
    /// Losslessly convertible to/from <see cref="Shorokoo.Graph.TensorKey"/>.
    ///
    /// <para>
    /// String form is <c>"N{NodeKey.Id}_T{OutputIndex}"</c>, matching the naming
    /// convention produced by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastUseUniqueNames"/>.
    /// </para>
    /// </summary>
    public readonly struct FastTensorKey : IEquatable<FastTensorKey>, IComparable<FastTensorKey>
    {
        /// <summary>
        /// The producing node's key. Named <c>FastNodeKey</c> rather than
        /// <c>NodeKey</c> so callers can disambiguate from the CG-side
        /// <see cref="Shorokoo.Graph.TensorKey.NodeKey"/> field at a glance.
        /// </summary>
        public readonly FastNodeKey FastNodeKey { get; }

        /// <summary>
        /// The zero-based index of this tensor in the producing node's output array.
        /// A value of -1 indicates a connecting tensor (open/close pairing).
        /// </summary>
        public readonly int OutputIndex { get; }

        /// <summary>The empty/invalid <see cref="FastTensorKey"/>.</summary>
        public static readonly FastTensorKey Empty = new FastTensorKey(FastNodeKey.Empty, -1);

        public FastTensorKey(FastNodeKey nodeKey, int outputIndex)
        {
            FastNodeKey = nodeKey;
            OutputIndex = outputIndex;
        }

        /// <summary>Creates a connecting-tensor key (OutputIndex = -1).</summary>
        public static FastTensorKey ForConnectingTensor(FastNodeKey nodeKey)
            => new FastTensorKey(nodeKey, -1);

        public static FastTensorKey FromCgKey(TensorKey k)
            => new FastTensorKey(FastNodeKey.FromCgKey(k.NodeKey), k.OutputIndex);

        public TensorKey ToCgKey() => new TensorKey(FastNodeKey.ToCgKey(), OutputIndex);

        public bool IsEmpty => FastNodeKey.IsEmpty;
        public bool IsConnectingTensor => OutputIndex == -1;

        public bool Equals(FastTensorKey other)
            => FastNodeKey.Equals(other.FastNodeKey) && OutputIndex == other.OutputIndex;
        public override bool Equals(object? obj) => obj is FastTensorKey other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(FastNodeKey, OutputIndex);

        public int CompareTo(FastTensorKey other)
        {
            var c = FastNodeKey.CompareTo(other.FastNodeKey);
            return c != 0 ? c : OutputIndex.CompareTo(other.OutputIndex);
        }

        /// <summary>
        /// String form is <c>"N{FastNodeKey.Id}_T{OutputIndex}"</c>. For connecting
        /// tensors (OutputIndex == -1) the form is <c>"N{FastNodeKey.Id}_Tcx"</c>
        /// to keep it ONNX-name-safe (no negative numbers or colons).
        /// </summary>
        public override string ToString()
            => OutputIndex == -1
                ? $"N{FastNodeKey.Id}_Tcx"
                : $"N{FastNodeKey.Id}_T{OutputIndex}";

        public static bool operator ==(FastTensorKey left, FastTensorKey right) => left.Equals(right);
        public static bool operator !=(FastTensorKey left, FastTensorKey right) => !left.Equals(right);
    }
}
