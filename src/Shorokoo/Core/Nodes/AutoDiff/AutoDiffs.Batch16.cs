using System.Diagnostics;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Col2Im =====
        //
        // Forward: output = Col2Im(input, image_shape, block_shape, dilations, pads, strides)
        //   input:  [N, C * ∏(block_shape), L]  — unfolded columns
        //   output: [N, C, *image_shape]          — folded image (overlapping blocks are summed)
        //
        // Gradient (Im2Col): extracts sliding-window patches from the output gradient.
        //   dInput[n, c·B + block_offset, window_pos] = padded_grad[n, c, spatial_source_pos]
        //   where B = ∏(block_shape), and source positions are determined by
        //   strides, dilations, and the block offset within the window.
        //
        //   dImageShape = null (int64, not differentiable)
        //   dBlockShape = null (int64, not differentiable)
        //
        // Implementation: Pad the gradient, compute flat gather indices for all
        // (block_offset, window_position) pairs, then GatherElements to extract.
        // The index tensor is built via Range + Reshape broadcasting for each
        // spatial dimension, then combined using the Horner scheme for flat indexing.

        [AutoDiff(COL2IM)]
        public static Variable?[] Col2Im<T1>(
            Tensor<T1> input, Tensor<int64> imageShape, Tensor<int64> blockShape,
            Tensor<T1> grad,
            long[]? dilations, long[]? pads, long[]? strides)
            where T1 : IVarType
        {
            // Determine number of spatial dimensions from attributes. OnnxOp.Col2Im's
            // C# wrapper requires non-null strides/dilations/pads, so the gradient never
            // sees a null strides attribute in practice — the dilations/pads/default
            // fallbacks are API-dead.
            Debug.Assert(strides is not null);
            int nDims = strides!.Length;

            var effectiveDilations = dilations ?? Enumerable.Repeat(1L, nDims).ToArray();
            var effectivePads = pads ?? new long[2 * nDims];
            var effectiveStrides = strides ?? Enumerable.Repeat(1L, nDims).ToArray();

            // Step 1: Pad the gradient image
            // grad shape: [N, C, d0, d1, ...]
            // ONNX Pad format: [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
            var totalRank = 2 + nDims;
            var padArray = new long[2 * totalRank];
            for (int d = 0; d < nDims; d++)
            {
                padArray[2 + d] = effectivePads[d];                  // begin pad for spatial dim d
                padArray[totalRank + 2 + d] = effectivePads[d + nDims]; // end pad for spatial dim d
            }
            var padTensor = Vector(padArray);
            Tensor<T1> paddedGrad = OnnxOp.Pad(grad, padTensor, null, axes: null, mode: null);

            // Step 2: Get N, C from grad shape and compute spatial stride factors
            var gradShape = OnnxOp.Shape(grad);
            Tensor<int64> N_s = OnnxOp.Gather(gradShape, Scalar(0L), axis: 0);
            Tensor<int64> C_s = OnnxOp.Gather(gradShape, Scalar(1L), axis: 0);

            // Compute padded spatial dimensions
            var paddedDims = new Variable[nDims];
            for (int d = 0; d < nDims; d++)
            {
                Tensor<int64> imageDimD = OnnxOp.Gather(imageShape, Scalar((long)d), axis: 0);
                // Store the Immutable* graph value (not the boxed struct handle) so later reads can
                // downcast it back to Variable.
                paddedDims[d] = (Variable)(imageDimD + Scalar(effectivePads[d] + effectivePads[d + nDims]));
            }

            // Compute spatial strides for flat indexing into padded gradient
            // spatialStrides[d] = ∏_{j > d} paddedDims[j]
            var spatialStrides = new Variable[nDims];
            spatialStrides[nDims - 1] = (Variable)Scalar(1L);
            for (int d = nDims - 2; d >= 0; d--)
                spatialStrides[d] = (Variable)(((Tensor<int64>)spatialStrides[d + 1]) * ((Tensor<int64>)paddedDims[d + 1]));

            // Step 3: Build flat index tensor
            // For each spatial dimension d, we have:
            //   - block offsets: kd_range = Range(0, block_shape[d]) * dilations[d]
            //   - window starts: od_range = Range(0, output_dim[d]) * strides[d]
            //   - source positions for dim d: kd_offset + od_start (broadcasted)
            //
            // The index tensor has shape [k0, o0, k1, o1, ..., k_{n-1}, o_{n-1}]
            // flat_idx = Σ_d source_d * spatialStrides[d]
            var totalIndexDims = 2 * nDims;
            Variable? flatIdx = null;

            for (int d = 0; d < nDims; d++)
            {
                Tensor<int64> kd = OnnxOp.Gather(blockShape, Scalar((long)d), axis: 0);

                // Compute output dimension for this spatial dim:
                // od = (paddedDims[d] - dilations[d] * (kd - 1) - 1) / strides[d] + 1
                var od = ((Variable)paddedDims[d] - Scalar(effectiveDilations[d]) * (kd - Scalar(1L)) - Scalar(1L)) / Scalar(effectiveStrides[d]) + Scalar(1L);

                // Create range tensors
                Tensor<int64> khRange = OnnxOp.Range(Scalar(0L), kd, Scalar(1L)); // [kd]
                Tensor<int64> ohRange = OnnxOp.Range(Scalar(0L), od, Scalar(1L)); // [od]

                // Scale by dilation/stride
                var khOffsets = khRange * Scalar(effectiveDilations[d]); // [kd]
                var ohStarts = ohRange * Scalar(effectiveStrides[d]);    // [od]

                // Reshape for broadcasting into the [k0, o0, k1, o1, ...] layout
                // khOffsets goes at position 2*d, ohStarts at position 2*d+1
                var khShape = new long[totalIndexDims];
                Array.Fill(khShape, 1L);
                khShape[2 * d] = -1;
                var ohShape = new long[totalIndexDims];
                Array.Fill(ohShape, 1L);
                ohShape[2 * d + 1] = -1;

                Tensor<int64> khReshaped = OnnxOp.Reshape(khOffsets, Vector(khShape), allowZero: false);
                Tensor<int64> ohReshaped = OnnxOp.Reshape(ohStarts, Vector(ohShape), allowZero: false);

                // source_d = kh_offset + oh_start (broadcast to [..., kd, od, ...])
                var sourceD = khReshaped + ohReshaped;

                // Multiply by spatial stride and accumulate
                var contribution = sourceD * (Variable)spatialStrides[d];
                flatIdx = flatIdx is null
                    ? (Variable)contribution
                    : (Variable)((Tensor<int64>)flatIdx + contribution);
            }

            // flatIdx shape: [k0, o0, k1, o1, ..., k_{n-1}, o_{n-1}]
            // We need to transpose to [k0, k1, ..., o0, o1, ...]
            // then reshape to [block_size, L]
            var perm = new long[totalIndexDims];
            for (int d = 0; d < nDims; d++)
            {
                perm[d] = 2 * d;           // k dims first
                perm[nDims + d] = 2 * d + 1; // o dims after
            }
            flatIdx = OnnxOp.Transpose(flatIdx!, perm);

            // Reshape to [block_size, L]
            Tensor<int64> blockSize = OnnxOp.ReduceProd(blockShape, keepdims: false);
            var inputShapeT = OnnxOp.Shape(input);
            Tensor<int64> L = OnnxOp.Gather(inputShapeT, Scalar(2L), axis: 0);

            Tensor<int64> indexShape2D = OnnxOp.Concat([
                OnnxOp.Reshape(blockSize, Vector(1L), allowZero: false),
                OnnxOp.Reshape(L, Vector(1L), allowZero: false)
            ], axis: 0);
            flatIdx = OnnxOp.Reshape(flatIdx, indexShape2D, allowZero: false);
            // flatIdx: [block_size, L]

            // Step 4: Gather from padded gradient
            // paddedGrad: [N, C, padded_d0, padded_d1, ...]
            // Reshape to [N*C, spatial_flat]
            var NC = N_s * C_s;
            var paddedGradShape = OnnxOp.Shape(paddedGrad);
            Tensor<int64> spatialShape = OnnxOp.Slice(paddedGradShape, Vector(2L), Vector((long)(2 + nDims)));
            Tensor<int64> spatialFlat = OnnxOp.ReduceProd(spatialShape, keepdims: false);

            Tensor<int64> reshapeTo2D = OnnxOp.Concat([
                OnnxOp.Reshape(NC, Vector(1L), allowZero: false),
                OnnxOp.Reshape(spatialFlat, Vector(1L), allowZero: false)
            ], axis: 0);
            Tensor<T1> paddedFlat = OnnxOp.Reshape(paddedGrad, reshapeTo2D, allowZero: false);
            // paddedFlat: [N*C, spatial_flat]

            // Flatten index tensor to 1D: [block_size * L]
            Tensor<int64> flatIdxFlat = OnnxOp.Reshape(flatIdx, Vector(-1L), allowZero: false);

            // Expand indices to [N*C, block_size * L] for GatherElements
            var blockTimesL = blockSize * L;
            Tensor<int64> expandShape = OnnxOp.Concat([
                OnnxOp.Reshape(NC, Vector(1L), allowZero: false),
                OnnxOp.Reshape(blockTimesL, Vector(1L), allowZero: false)
            ], axis: 0);
            Tensor<int64> flatIdxExpanded = OnnxOp.Expand(
                OnnxOp.Unsqueeze(flatIdxFlat, Vector(0L)),
                expandShape);
            // flatIdxExpanded: [N*C, block_size * L]

            // GatherElements axis=1: output[i,j] = paddedFlat[i, flatIdxExpanded[i,j]]
            Tensor<T1> gathered = OnnxOp.GatherElements(paddedFlat, flatIdxExpanded, axis: 1);
            // gathered: [N*C, block_size * L]

            // Step 5: Reshape to [N, C * block_size, L] to match input shape
            var CTimesB = C_s * blockSize;
            Tensor<int64> outputShape = OnnxOp.Concat([
                OnnxOp.Reshape(N_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(CTimesB, Vector(1L), allowZero: false),
                OnnxOp.Reshape(L, Vector(1L), allowZero: false)
            ], axis: 0);
            Tensor<T1> dInput = OnnxOp.Reshape(gathered, outputShape, allowZero: false);

            return [dInput, null, null];
        }
    }
}
