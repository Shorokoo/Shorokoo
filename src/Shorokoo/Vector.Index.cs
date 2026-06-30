
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Net.Mail;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static Shorokoo.Globals;
using static Shorokoo.Core.InternalGlobals;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo
{
    /// <summary>
    /// Argument type for the <see cref="Vector{T}"/> slicing indexer. Implicitly built from a
    /// <see cref="Range"/>, a (range, step) or (from, to, step) tuple, an index vector, or a
    /// <c>long[]</c> of gather indices.
    /// </summary>
    public struct VectorIndexerParam
    {
        internal readonly Vector<int64>? Indices { get; init; }
        internal readonly bool IsFullRange { get; init; }
        internal readonly Scalar<int64>? ScalarStart { get; init; }
        internal readonly Scalar<int64>? ScalarEnd { get; init; }
        internal readonly Scalar<int64>? ScalarStep { get; init; }

        /// <summary>Wraps a <see cref="Range"/> as a slicing parameter (step 1).</summary>
        public static implicit operator VectorIndexerParam(Range range)
            => new VectorIndexerParam
            {
                ScalarStart = Scalar(OnnxUtils.FromIndex(range.Start)),
                ScalarEnd = Scalar(OnnxUtils.FromIndex(range.End)),
                ScalarStep = null, // Scalar(1L),
                IsFullRange = range.Equals(Range.All)
            };

        /// <summary>Wraps a <see cref="Range"/> plus an explicit step.</summary>
        public static implicit operator VectorIndexerParam((Range range, long step) tuple)
            => new VectorIndexerParam
            {
                ScalarStart = Scalar(OnnxUtils.FromIndex(tuple.range.Start)),
                ScalarEnd = Scalar(OnnxUtils.FromIndex(tuple.range.End)),
                ScalarStep = Scalar(tuple.step),
                IsFullRange = tuple.step == 1 && tuple.range.Equals(Range.All)
            };

        /// <summary>Wraps explicit from / to / step slice bounds (null <c>to</c> means to-the-end).</summary>
        public static implicit operator VectorIndexerParam((long from, long? to, long step) tuple)
            => new VectorIndexerParam
            {
                ScalarStart = Scalar(tuple.from),
                ScalarEnd = Scalar(tuple.to ?? long.MaxValue),
                ScalarStep = Scalar(tuple.step),
                IsFullRange = tuple.from == 0 && (tuple.to ?? long.MaxValue) == long.MaxValue
            };

        /// <summary>Wraps a runtime index vector — the indexer gathers those positions.</summary>
        public static implicit operator VectorIndexerParam(Vector<int64> indices)
            => new VectorIndexerParam { Indices = indices };

        /// <summary>Wraps constant gather indices.</summary>
        public static implicit operator VectorIndexerParam(long[] indices)
            => new VectorIndexerParam { Indices = Vector(indices) };
    }

    public partial struct Vector<T>
    {
        /// <summary>
        /// Slices or gathers the vector for reads (<c>Vector w = v[1..3];</c>), and replaces the
        /// indexed positions for writes (<c>v[1..3] = w;</c>). Because <see cref="Vector{T}"/> is a
        /// value type, an indexer-set rebinds this local handle and never mutates a caller's instance.
        /// </summary>
        public Vector<T> this[VectorIndexerParam index]
        {
            get
            {
                // Full range is the identity.
                if (index.IsFullRange)
                    return this;

                // Gather a list of positions (ONNX Gather along axis 0 keeps rank 1).
                if (index.Indices is not null)
                    return this.Gather(index.Indices.Value, 0).Vec();

                // Contiguous or strided slice (ORT clamps an open-ended bound; steps drive the stride).
                return this.Slice(index.ScalarStart!.Value, index.ScalarEnd!.Value, index.ScalarStep);
            }
            set
            {
                // Full range replaces the whole vector.
                if (index.IsFullRange)
                {
                    this = value;
                    return;
                }

                // Both the gather-by-index and the (possibly strided) slice cases reduce to
                // "scatter `value` at these row indices"; ScatterND wants the index list shaped [n, 1].
                Vector<int64> positions;
                if (index.Indices is not null)
                    positions = index.Indices.Value;
                else
                {
                    var end = index.ScalarEnd!.Value.Min(this.TShape[0]);   // clamp open-ended bound
                    positions = OnnxOp.Range(index.ScalarStart!.Value, end, index.ScalarStep ?? Globals.Scalar(1L));
                }
                this = this.ScatterND(((Tensor<int64>)positions).Unsqueeze(), value);
            }
        }

        /// <summary>Reads or writes a single element at a runtime index.</summary>
        public Scalar<T> this[Scalar<int64> index]
        {
            get => this.Gather(index.Unsqueeze(), 0).Squeeze().Scalar();
            set => this = this.ScatterND(index.Unsqueeze().Unsqueeze(), value.Unsqueeze());
        }

        /// <summary>Reads or writes a single element at a constant index.</summary>
        public Scalar<T> this[long index]
        {
            get => this[Globals.Scalar(index)];
            set => this[Globals.Scalar(index)] = value;
        }

        /// <summary>Reads or writes a single element at a constant index.</summary>
        public Scalar<T> this[int index]
        {
            get => this[(long)index];
            set => this[(long)index] = value;
        }

        /// <summary>Reads or writes a single element at an <see cref="Index"/> (supports from-end <c>^i</c>).</summary>
        public Scalar<T> this[Index index]
        {
            get => this[OnnxUtils.FromIndex(index)];
            set => this[OnnxUtils.FromIndex(index)] = value;
        }
    }
}
