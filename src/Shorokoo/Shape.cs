using Shorokoo.Graph;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using static Shorokoo.Globals;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;

namespace Shorokoo
{
    /// <summary>
    /// A tensor shape: an ordered list of dimension sizes, with value equality.
    /// Rank 0 (no dims) is a scalar; -1 marks an unknown (dynamic) dimension.
    /// </summary>
    public class Shape : IEquatable<Shape>
    {
        /// <summary>The dimension sizes; -1 for an unknown (dynamic) dimension.</summary>
        public long[] Dims { get; private set; }

        // public long[] SignedDims => this.Dims.Convert<long>().ToArray();
        /// <summary>Total element count (product of <see cref="Dims"/>); 1 for a scalar shape, -1 if any dimension is unknown.</summary>
        public long Count { get; private set; }

        private static Shape? scalar;
        /// <summary>The shared rank-0 (scalar) shape.</summary>
        public static Shape Scalar
        {
            get
            {
                if (scalar is null)
                    scalar = new Shape();

                return scalar;
            }
        }

        /// <summary>Creates a rank-0 (scalar) shape.</summary>
        public Shape()
        {
            this.Dims = new long[0];
            this.Count = 1;
        }

        /// <summary>Creates a shape from TensorDims; dimensions without a known size become -1.</summary>
        public Shape(TensorDim[] dims) : this(dims.Select(x => x.Size ?? -1).ToArray())
        {
            if (this.Dims.Any(x => x == -1))
                this.Count = -1;
        }

        /// <summary>Creates a shape with the given dimension sizes.</summary>
        public Shape(params long[] dims)
        {
            this.Dims = dims;
            this.Count = 1;

            foreach (var dim in dims)
                this.Count *= dim;
        }

        /// <summary>Creates a shape with the given dimension sizes.</summary>
        public Shape(params ulong[] dims) : this(dims.Convert<long>().ToArray()) { }

        /// <summary>Returns the dimension sizes as an array.</summary>
        public static explicit operator ulong[](Shape shape) => shape.Dims.Select(x => (ulong)x).ToArray();
        /// <summary>Returns the dimension sizes as an array.</summary>
        public static explicit operator long[](Shape shape) => shape.Dims;

        /// <summary>Converts a dimension array to a Shape.</summary>
        public static implicit operator Shape(long [] dims) => new Shape(dims);
        /// <summary>Converts a dimension array to a Shape.</summary>
        public static implicit operator Shape(ulong[] dims) => new Shape(dims);

        /// <summary>Converts a single dimension size to a rank-1 Shape.</summary>
        public static implicit operator Shape(int dim) => new Shape(dim);
        /// <summary>Converts a single dimension size to a rank-1 Shape.</summary>
        public static implicit operator Shape(uint dim) => new Shape(dim);
        /// <summary>Converts a single dimension size to a rank-1 Shape.</summary>
        public static implicit operator Shape(long dim) => new Shape(dim);
        /// <summary>Converts a single dimension size to a rank-1 Shape.</summary>
        public static implicit operator Shape(ulong dim) => new Shape(dim);
        /// <summary>Converts a tuple of dimension sizes to a Shape.</summary>
        public static implicit operator Shape((ulong Item1, ulong Item2) dims) => new Shape(dims.Item1, dims.Item2);
        /// <summary>Converts a tuple of dimension sizes to a Shape.</summary>
        public static implicit operator Shape((ulong Item1, ulong Item2, ulong Item3) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3);
        /// <summary>Converts a tuple of dimension sizes to a Shape.</summary>
        public static implicit operator Shape((ulong Item1, ulong Item2, ulong Item3, ulong Item4) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);
        /// <summary>Converts a tuple of dimension sizes to a Shape.</summary>
        public static implicit operator Shape((ulong Item1, ulong Item2, ulong Item3, ulong Item4, ulong Item5) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);
        /// <summary>Converts a tuple of dimension sizes to a Shape.</summary>
        public static implicit operator Shape((ulong Item1, ulong Item2, ulong Item3, ulong Item4, ulong Item5, ulong Item6) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6);
        /// <summary>Converts a tuple of dimension sizes to a Shape.</summary>
        public static implicit operator Shape((ulong Item1, ulong Item2, ulong Item3, ulong Item4, ulong Item5, ulong Item6, ulong Item7) dims) => new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6, dims.Item7);


        /// <summary>Value equality over the dimension sizes.</summary>
        public static bool operator ==(Shape? left, Shape? right)
        {
            if (ReferenceEquals(left, right)) return true;
            if (ReferenceEquals(left, null)) return false;

            return left.Equals(right);
        }

        /// <summary>Value inequality over the dimension sizes.</summary>
        public static bool operator !=(Shape? left, Shape? right)
        {
            return !(left == right);
        }

        /// <summary>Hash code combining all dimensions, consistent with <see cref="Equals(object?)"/>.</summary>
        public override int GetHashCode()
        {
            unchecked {
                int hash = 17;
                foreach (var dim in this.Dims)
                    hash = hash * 31 + dim.GetHashCode();

                return hash;
            }
        }

        /// <summary>Structural equality: true when both shapes have the same dimensions.</summary>
        public override bool Equals(object? obj)
        {
            if (obj is Shape shape)
                return this.Equals(shape);

            return false;
        }

        /// <summary>True when both shapes have identical dimension sizes.</summary>
        public bool Equals(Shape? other)
        {
            if (other == null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            if (this.Dims.Length != other.Dims.Length)
                return false;

            return this.Dims.SequenceEqual(other.Dims);
        }

        /// <summary>Formats the shape as <c>(d1,d2,...)</c>; <c>()</c> for a scalar, <c>(d,)</c> for rank 1.</summary>
        public override string ToString()
        {
            if (this.Dims is null)
                return "()";

            if (this.Dims.Length == 0)
                return "()";

            if (this.Dims.Length == 1)
                return $"({this.Dims[0]},)";

            return $"({string.Join(",", this.Dims)})";
        }
    }
}
