using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== AFFINE_GRID =====
        //
        // Forward: grid = AffineGrid(theta [N, K, K+1], size [K+2], align_corners?)
        //   K = 2 (size = [N, C, H, W],   grid [N, H, W, 2]) or
        //   K = 3 (size = [N, C, D, H, W], grid [N, D, H, W, 3]).
        //   For every output position p the op computes
        //     grid[n, p, :] = theta[n] @ h[p]      with h[p] = [base coords of p; 1]
        //   where the base coordinates are the normalized positions an IDENTITY theta
        //   would produce (align_corners=true: 2·i/(dim−1) − 1; false: (2·i+1)/dim − 1).
        //
        // Gradient (both 2-D and 3-D):
        //   dtheta[n] = dY[n]^T @ H  →  [K, P] @ [P, K+1] with P = ∏ spatial dims.
        //
        //   Instead of rebuilding the base grid by hand (the previous implementation
        //   hardcoded the 2-D [N,C,H,W] layout), we recover it from the op itself:
        //   AffineGrid(identityTheta, size) returns exactly the base coordinates in the
        //   op's own ordering/convention, for any spatial rank and either align_corners
        //   value. Appending a ones channel completes the homogeneous vector h.
        //
        //   dL/d(size) = null (int64, not differentiable)

        [AutoDiff(AFFINE_GRID)]
        public static Variable?[] AffineGrid<T>(
            Tensor<T> theta, Tensor<int64> size, Tensor<T> grad,
            bool? align_corners)
            where T : IVarType
        {
            var floatType = grad.Type;
            var one = OnnxOp.Cast(Scalar(1.0f), saturate: null, to: floatType);

            // theta shape pieces: [N, K, K+1]
            var thetaShape = OnnxOp.Shape(theta);
            var nVec = OnnxOp.Slice(thetaShape, Vector(0L), Vector(1L));   // [N]
            var kVec = OnnxOp.Slice(thetaShape, Vector(1L), Vector(2L));   // [K]
            var k1Vec = OnnxOp.Add(kVec, Vector(1L));                      // [K+1]

            // identityTheta [N, K, K+1]: EyeLike on a [K, K+1] zeros block gives [I_K | 0],
            // broadcast over the batch.
            var zerosKxK1 = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: floatType),
                OnnxOp.Concat([kVec, k1Vec], axis: 0));
            var identity = OnnxOp.EyeLike(zerosKxK1);                      // [K, K+1]
            var identityTheta = OnnxOp.Expand(identity, thetaShape);       // [N, K, K+1]

            // Base grid in the op's own convention: [N, *spatial, K]
            var baseGrid = OnnxOp.AffineGrid(identityTheta, size, align_corners);

            // Flatten spatial dims: [N, P, K]
            var flatShape = OnnxOp.Concat([nVec, Vector(-1L), kVec], axis: 0);
            var baseFlat = OnnxOp.Reshape(baseGrid, flatShape, allowZero: false);

            // Homogeneous ones channel → [N, P, K+1]
            var baseFlatShape = OnnxOp.Shape(baseFlat);                    // [N, P, K]
            var npVec = OnnxOp.Slice(baseFlatShape, Vector(0L), Vector(2L));
            var onesShape = OnnxOp.Concat([npVec, Vector(1L)], axis: 0);   // [N, P, 1]
            var ones = OnnxOp.Expand(one, onesShape);
            var baseH = OnnxOp.Concat([baseFlat, ones], axis: 2);          // [N, P, K+1]

            // grad [N, *spatial, K] → [N, P, K] → [N, K, P]
            var gradFlat = OnnxOp.Reshape(grad, flatShape, allowZero: false);
            var gradT = OnnxOp.Transpose(gradFlat, [0L, 2L, 1L]);

            // dtheta [N, K, K+1] = gradT [N, K, P] @ baseH [N, P, K+1]
            var dtheta = OnnxOp.MatMul(gradT, baseH);

            return [dtheta, null]; // null for size (int64, not differentiable)
        }
    }
}
