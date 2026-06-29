using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Det (matrix determinant) =====
        //
        // Forward: y = det(X), where X is [..., M, M] and y is [...]
        //
        // Gradient: dL/dX = dL/dY * cofactor_matrix(X)
        //   cofactor_matrix(X)_{ij} = (-1)^{i+j} * det(minor(X, i, j))
        //   where minor(X, i, j) is the (M-1)×(M-1) submatrix with row i and column j removed.
        //
        // Implementation:
        //   1. Flatten batch dims: reshape [..., M, M] → [B, M, M]
        //   2. Build "leave-one-out" (loo) indices [M, M-1] to select all-but-one rows/cols
        //   3. Use two Gather ops to build all M² minor submatrices as [B, M, M, M-1, M-1]
        //   4. Apply Det to all minors → [B, M, M]
        //   5. Apply checkerboard sign (-1)^(i+j) and multiply by upstream gradient
        //   6. Reshape back to original input shape

        internal static Variable?[] DetGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;           // [..., M, M]
            var grad = outputGrads[0]!;   // [...]

            var origShape = OnnxOp.Shape(x);  // [rank]

            // Extract M from last dimension
            var M_vec = OnnxOp.Shape(x, start: -1);           // [1] containing M
            var M = OnnxOp.Squeeze(M_vec, Vector(0L));         // scalar M
            var M_minus_1 = OnnxOp.Sub(M, Scalar(1L));         // scalar M-1

            // Flatten to 3D: [..., M, M] → [B, M, M] where B = product of batch dims
            var shape_3d = OnnxOp.Concat([Vector(-1L), M_vec, M_vec], axis: 0);
            var x_3d = OnnxOp.Reshape(x, shape_3d, allowZero: false);  // [B, M, M]

            // Build "leave-one-out" indices: shape [M, M-1]
            // loo[i] = [0, ..., i-1, i+1, ..., M-1] — all indices except i
            var base_range = OnnxOp.Range(Scalar(0L), M_minus_1, Scalar(1L));  // [M-1]
            var base_2d = OnnxOp.Unsqueeze(base_range, Vector(0L));            // [1, M-1]
            var i_range = OnnxOp.Range(Scalar(0L), M, Scalar(1L));             // [M]
            var i_col = OnnxOp.Unsqueeze(i_range, Vector(1L));                 // [M, 1]
            var mask = OnnxOp.GreaterOrEqual(base_2d, i_col);                  // [M, M-1] bool
            var skip = OnnxOp.Cast(mask, saturate: null, to: DType.Int64);     // [M, M-1]
            var loo = OnnxOp.Add(base_2d, skip);                               // [M, M-1]

            // Build all M² minor submatrices
            // Remove each row i: Gather on axis 1 → [B, M, M-1, M]
            var rows_removed = OnnxOp.Gather(x_3d, loo, axis: 1);

            // Remove each column j: Gather on axis 3 → [B, M, M-1, M, M-1]
            var minors_raw = OnnxOp.Gather(rows_removed, loo, axis: 3);

            // Transpose to [B, M, M, M-1, M-1] (swap axes 2↔3)
            var minors = OnnxOp.Transpose(minors_raw, perm: [0, 1, 3, 2, 4]);

            // Det of each minor submatrix → [B, M, M]
            var minor_dets = OnnxOp.Det(minors);

            // Checkerboard sign pattern: (-1)^(i+j) for [M, M]
            var i_f = OnnxOp.Cast(i_range, saturate: null, to: DType.Float32);   // [M]
            var i_col_f = OnnxOp.Unsqueeze(i_f, Vector(1L));                     // [M, 1]
            var j_row_f = OnnxOp.Unsqueeze(i_f, Vector(0L));                     // [1, M]
            var sum_ij = OnnxOp.Add(i_col_f, j_row_f);                           // [M, M]
            var signs_f = OnnxOp.Pow(Scalar(-1.0f), sum_ij);                     // [M, M] {1, -1}
            var signs = OnnxOp.Cast(signs_f, saturate: null, to: x.Type);

            // Cofactor matrix = signs * minor_dets: [B, M, M]
            // Cast minor_dets to ensure type match (Det preserves type, but be defensive)
            var minor_dets_typed = OnnxOp.Cast(minor_dets, saturate: null, to: x.Type);
            var cofactors = OnnxOp.Mul(signs, minor_dets_typed);

            // Multiply by upstream gradient: reshape grad [...] → [B, 1, 1] and broadcast
            var grad_flat = OnnxOp.Reshape(grad, Vector(-1L), allowZero: false);  // [B]
            var grad_bc = OnnxOp.Unsqueeze(
                OnnxOp.Unsqueeze(grad_flat, Vector(1L)),
                Vector(2L));  // [B, 1, 1]
            var grad_x_3d = OnnxOp.Mul(grad_bc, cofactors);  // [B, M, M]

            // Reshape back to original input shape
            var grad_x = OnnxOp.Reshape(grad_x_3d, origShape, allowZero: false);

            return [grad_x];
        }
    }
}
