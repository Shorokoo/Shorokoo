
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

    /// <summary>
    /// Deferred result of a <see cref="Vector{T}"/> slice indexer: implicitly converts to the
    /// sliced/gathered <see cref="Vector{T}"/> for reads, or writes via <see cref="Set"/>.
    /// </summary>
    public class VectorIndexerResult<TT> where TT : IVarType
    {
        private Vector<TT> gatherFrom;
        private VectorIndexerParam slice;

        /// <summary>The sliced/gathered vector (shorthand for the implicit conversion).</summary>
        public Vector<TT> T => (Vector<TT>)this;

        internal VectorIndexerResult(Vector<TT> gatherFrom, VectorIndexerParam slice)
        {
            this.gatherFrom = gatherFrom;
            this.slice = slice;
        }

        /// <summary>Materializes the slice/gather as a <see cref="Vector{T}"/>.</summary>
        public static implicit operator Vector<TT>(VectorIndexerResult<TT> result)
        {
            var tensor = result.gatherFrom;
            var slice = result.slice;

            // This is a no op.
            if (slice.IsFullRange)
                return result.gatherFrom;


            if (slice.Indices is not null)
                return tensor.GatherND(slice.Indices.Value, batchDims: 0);

            Debug.Assert(slice.ScalarStart is not null);
            Debug.Assert(slice.ScalarEnd is not null);
            if (slice.ScalarStep is not null)
                throw new InvalidTensorOperationException(ErrorCodes.CR006, "Vector Index", "step operation", 
                    "Step operations in vector indexing are not yet supported");

            return tensor.Slice(slice.ScalarStart!.Value, slice.ScalarEnd!.Value, slice.ScalarStep);
        }

        /// <summary>Returns a copy of the source vector with the indexed positions replaced by <paramref name="values"/>.</summary>
        public Vector<TT> Set(Vector<TT> values)
        {
            // This is a no op.
            if (slice.IsFullRange)
                return values;


            if (slice.Indices is not null)
                return gatherFrom.ScatterND(slice.Indices.Value, values);

            Debug.Assert(slice.ScalarStart is not null);
            Debug.Assert(slice.ScalarEnd is not null);

            if (slice.ScalarStep is null)
            {
                var toUpdateTensor = values.Pad(PadMode.Constant, slice.ScalarStart!.Value, slice.ScalarEnd!.Value, Scalar<TT>(values.Type.DefaultVal));
                var updateMask = VectorFill(values.TShape, true).Pad(PadMode.Constant, slice.ScalarStart!.Value.Unsqueeze(), slice.ScalarEnd!.Value.Unsqueeze());
                return updateMask.Where(toUpdateTensor, gatherFrom);
            }

            var targetShape = gatherFrom.TShape;

            var stepPattern = VectorFill(slice.ScalarStep!.Value, false);
            stepPattern = stepPattern[0].Set(true);

            var stepMask = stepPattern.Tile(values.TShape);
            stepMask = stepMask.Slice(0L, targetShape[0]);

            var blownValues = values.Resize(targetShape, transformMode: CoordinateTransformationMode.Asymmetric, mode: ResizeMode.Nearest);

            var result = stepMask.Where(blownValues, gatherFrom);
            return result;
        }
    }

    /// <summary>
    /// Deferred result of a single-element <see cref="Vector{T}"/> indexer: implicitly converts
    /// to the element <see cref="Scalar{T}"/> for reads, or writes via <see cref="Set"/>.
    /// </summary>
    public class ScalarIndexerResult<TT> where TT : IVarType
    {
        private Vector<TT> gatherFrom;
        private Scalar<int64> index;

        /// <summary>The indexed element (shorthand for the implicit conversion).</summary>
        public Scalar<TT> T => (Scalar<TT>)this;

        internal ScalarIndexerResult(Vector<TT> gatherFrom, long index)
        {
            this.gatherFrom = gatherFrom;
            this.index = Scalar(index);
        }

        internal ScalarIndexerResult(Vector<TT> gatherFrom, Scalar<int64> index)
        {
            this.gatherFrom = gatherFrom;
            this.index = index;
        }

        /// <summary>Materializes the indexed element as a <see cref="Scalar{T}"/>.</summary>
        public static implicit operator Scalar<TT>(ScalarIndexerResult<TT> result)
        {
            var tensor = result.gatherFrom;
            var index = result.index;
            return tensor.Gather(result.index.Unsqueeze(), 0).Squeeze().Scalar();
            // return tensor.Slice(index, index + 1).Squeeze();
        }

        /// <summary>Returns a copy of the source vector with the indexed element replaced by <paramref name="value"/>.</summary>
        public Vector<TT> Set(Scalar<TT> value)
            => this.gatherFrom.ScatterND(index.Unsqueeze().Unsqueeze(), value.Unsqueeze());
    }

    public partial struct Vector<T>
    {
        /// <summary>Slices or gathers the vector (range, stepped range, or index list).</summary>
        public VectorIndexerResult<T> this[VectorIndexerParam index]
            => new VectorIndexerResult<T>(this, index);

        /// <summary>Reads/writes a single element at a runtime index.</summary>
        public ScalarIndexerResult<T> this[Scalar<int64> index]
            => new ScalarIndexerResult<T>(this, index);

        /// <summary>Reads/writes a single element at a constant index.</summary>
        public ScalarIndexerResult<T> this[long index]
            => new ScalarIndexerResult<T>(this, index);

        /// <summary>Reads/writes a single element at a constant index.</summary>
        public ScalarIndexerResult<T> this[int index]
            => this[(long)index];

        /// <summary>Reads/writes a single element at a <see cref="Index"/> (supports from-end <c>^i</c>).</summary>
        public ScalarIndexerResult<T> this[Index index]
            => this[OnnxUtils.FromIndex(index)];
    }
}
