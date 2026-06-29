using System;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Graph
{
    /// <summary>
    /// A globally unique identifier for a Node.
    /// NodeKeys remain stable across graph transformations that preserve node identity.
    /// </summary>
    public readonly struct NodeKey : IEquatable<NodeKey>, IComparable<NodeKey>
    {
        /// <summary>
        /// The unique identifier for this node.
        /// </summary>
        public readonly Guid Id;

        /// <summary>
        /// Represents an empty/invalid NodeKey.
        /// </summary>
        public static readonly NodeKey Empty = new NodeKey(Guid.Empty);

        /// <summary>
        /// Creates a new NodeKey with the specified Guid.
        /// </summary>
        public NodeKey(Guid id)
        {
            Id = id;
        }

        /// <summary>
        /// Creates a new NodeKey with a newly generated Guid.
        /// </summary>
        public static NodeKey New() => new NodeKey(Guid.NewGuid());

        /// <summary>
        /// Creates a NodeKey from a string representation.
        /// </summary>
        public static NodeKey Parse(string s) => new NodeKey(Guid.Parse(s));

        /// <summary>
        /// Attempts to parse a string into a NodeKey.
        /// </summary>
        public static bool TryParse(string? s, out NodeKey result)
        {
            if (Guid.TryParse(s, out var guid))
            {
                result = new NodeKey(guid);
                return true;
            }
            result = Empty;
            return false;
        }

        /// <summary>
        /// Returns true if this is an empty/invalid NodeKey.
        /// </summary>
        public bool IsEmpty => Id == Guid.Empty;

        public bool Equals(NodeKey other) => Id.Equals(other.Id);
        public override bool Equals(object? obj) => obj is NodeKey other && Equals(other);
        public override int GetHashCode() => Id.GetHashCode();
        public override string ToString() => Id.ToString("N"); // No hyphens for compactness

        public int CompareTo(NodeKey other) => Id.CompareTo(other.Id);

        public static bool operator ==(NodeKey left, NodeKey right) => left.Equals(right);
        public static bool operator !=(NodeKey left, NodeKey right) => !left.Equals(right);
    }

    /// <summary>
    /// A globally unique identifier for a Tensor (Variable).
    /// Composed of the parent node's NodeKey and the output index within that node.
    /// TensorKeys remain stable across graph transformations that preserve tensor identity.
    /// </summary>
    public readonly struct TensorKey : IEquatable<TensorKey>, IComparable<TensorKey>
    {
        /// <summary>
        /// The NodeKey of the parent node that produces this tensor.
        /// </summary>
        public readonly NodeKey NodeKey { get; }

        /// <summary>
        /// The zero-based index of this tensor in the parent node's output array.
        /// A value of -1 indicates this is a connecting tensor (for open/close nodes).
        /// </summary>
        public readonly int OutputIndex { get; }

        /// <summary>
        /// Represents an empty/invalid TensorKey.
        /// </summary>
        public static readonly TensorKey Empty = new TensorKey(NodeKey.Empty, -1);

        /// <summary>
        /// Creates a new TensorKey.
        /// </summary>
        public TensorKey(NodeKey nodeKey, int outputIndex)
        {
            NodeKey = nodeKey;
            OutputIndex = outputIndex;
        }

        /// <summary>
        /// Creates a TensorKey for a connecting tensor.
        /// Connecting tensors use OutputIndex = -1 to distinguish from regular outputs.
        /// </summary>
        public static TensorKey ForConnectingTensor(NodeKey nodeKey)
            => new TensorKey(nodeKey, -1);

        /// <summary>
        /// Creates a TensorKey from a string representation.
        /// Format: "{NodeKeyGuid}:{OutputIndex}"
        /// </summary>
        public static TensorKey Parse(string s)
        {
            var parts = s.Split(':');
            if (parts.Length != 2)
                throw new FormatException($"Invalid TensorKey format: {s}");

            return new TensorKey(NodeKey.Parse(parts[0]), int.Parse(parts[1]));
        }

        /// <summary>
        /// Attempts to parse a string into a TensorKey.
        /// </summary>
        public static bool TryParse(string? s, out TensorKey result)
        {
            if (string.IsNullOrEmpty(s))
            {
                result = Empty;
                return false;
            }

            var parts = s.Split(':');
            if (parts.Length == 2 &&
                NodeKey.TryParse(parts[0], out var nodeKey) &&
                int.TryParse(parts[1], out var outputIndex))
            {
                result = new TensorKey(nodeKey, outputIndex);
                return true;
            }

            result = Empty;
            return false;
        }

        /// <summary>
        /// Returns true if this is an empty/invalid TensorKey.
        /// </summary>
        public bool IsEmpty => NodeKey.IsEmpty;

        /// <summary>
        /// Returns true if this represents a connecting tensor (output index is -1).
        /// </summary>
        public bool IsConnectingTensor => OutputIndex == -1;

        public bool Equals(TensorKey other)
            => NodeKey.Equals(other.NodeKey) && OutputIndex == other.OutputIndex;

        public override bool Equals(object? obj)
            => obj is TensorKey other && Equals(other);

        public override int GetHashCode()
            => HashCode.Combine(NodeKey, OutputIndex);

        public override string ToString()
            => $"{NodeKey}:{OutputIndex}";

        public int CompareTo(TensorKey other)
        {
            var nodeComparison = NodeKey.CompareTo(other.NodeKey);
            return nodeComparison != 0 ? nodeComparison : OutputIndex.CompareTo(other.OutputIndex);
        }

        public static bool operator ==(TensorKey left, TensorKey right) => left.Equals(right);
        public static bool operator !=(TensorKey left, TensorKey right) => !left.Equals(right);
    }
}
