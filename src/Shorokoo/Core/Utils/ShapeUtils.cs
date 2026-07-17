using System;
using System.Collections.Generic;
using System.Text;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Utils
{
    internal static class ShapeUtils
    {
        /// <summary>
        /// Builds the ONNX Reshape shape input for the user-facing <c>Reshape</c> wrappers'
        /// <c>keepDims</c> form: inserts a constant <c>0</c> ("copy this dimension from the
        /// input", ONNX allowzero=0 semantics) at each output position listed in
        /// <paramref name="keepDims"/>, filling the remaining slots with the elements of
        /// <paramref name="newShape"/> in order. Works on dynamic-length shape vectors: the
        /// splice points are compile-time constants, so the result is a Slice/Concat chain
        /// that never needs the runtime length (ORT clamps the open-ended tail slice).
        /// </summary>
        internal static Vector<int64> InsertKeepDimZeros(Vector<int64> newShape, int[] keepDims)
        {
            if (keepDims.Length == 0)
                return newShape;

            int[] sorted = [.. keepDims];
            Array.Sort(sorted);
            if (sorted[0] < 0)
                throw new ArgumentOutOfRangeException(nameof(keepDims), sorted[0],
                    "keepDims positions must be non-negative.");
            for (int i = 1; i < sorted.Length; i++)
                if (sorted[i] == sorted[i - 1])
                    throw new ArgumentException(
                        $"keepDims lists output position {sorted[i]} more than once.", nameof(keepDims));

            var zero = Vector(0L);
            var pieces = new List<Vector<int64>>();
            long consumed = 0;
            for (int i = 0; i < sorted.Length; i++)
            {
                // Output position p with i zeros already inserted before it splits newShape at p - i.
                long split = sorted[i] - i;
                if (split > consumed)
                    pieces.Add(newShape.Slice(Scalar(consumed), Scalar(split)));
                pieces.Add(zero);
                consumed = split;
            }
            pieces.Add(newShape.Slice(Scalar(consumed), Scalar(long.MaxValue)));
            return [.. pieces];
        }
    }
}
