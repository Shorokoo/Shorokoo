using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== GridSample (bilinear, zeros padding) =====
        //
        // Forward: output[n,c,h,w] = BilinearInterp(X[n,c,:,:], grid[n,h,w,:])
        //   grid[n,h,w] = (gx, gy) in [-1,1], denormalized to pixel coords (ix, iy):
        //     align_corners=true:  ix = (gx+1)/2*(W-1),  iy = (gy+1)/2*(H-1)
        //     align_corners=false: ix = ((gx+1)*W-1)/2,  iy = ((gy+1)*H-1)/2
        //   Then bilinear interpolation from 4 neighboring pixels:
        //     Y = (1-wy)(1-wx)*X[iy0,ix0] + (1-wy)*wx*X[iy0,ix1]
        //       + wy*(1-wx)*X[iy1,ix0] + wy*wx*X[iy1,ix1]
        //   where ix0=floor(ix), ix1=ix0+1, wx=ix-ix0, wy=iy-iy0
        //
        // Gradient w.r.t. X (dInput):
        //   For each corner, scatter dY weighted by bilinear weight and validity mask
        //   back to the input positions using ScatterElements with Add reduction.
        //   dX[n,c,iy,ix] += weight * mask * dY[n,c,h,w]  for each (h,w) that maps to (iy,ix)
        //
        // Gradient w.r.t. grid (dGrid):
        //   dGrid[n,h,w,0] = scale_x * ΣC dY[n,c,h,w] * d(interp)/d(gx)
        //   dGrid[n,h,w,1] = scale_y * ΣC dY[n,c,h,w] * d(interp)/d(gy)
        //   where d(interp)/d(ix) = (1-wy)*(X[iy0,ix1]-X[iy0,ix0]) + wy*(X[iy1,ix1]-X[iy1,ix0])
        //         d(interp)/d(iy) = (1-wx)*(X[iy1,ix0]-X[iy0,ix0]) + wx*(X[iy1,ix1]-X[iy0,ix1])
        //   and scale_x = d(ix)/d(gx), scale_y = d(iy)/d(gy).

        internal static Variable?[] GridSampleGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;          // [N, C, H_in, W_in]
            var grid = inputs[1]!;       // [N, H_out, W_out, 2]
            var dY = outputGrads[0];     // [N, C, H_out, W_out]

            if (dY is null) return [null, null];

            // Read attributes
            var modeRaw = attributes.GetAttributeObj(AttrMode);
            var mode = modeRaw as GridSampleMode? ?? GridSampleMode.Linear;
            var paddingModeRaw = attributes.GetAttributeObj(AttrPaddingMode);
            var paddingMode = paddingModeRaw as GridSamplePaddingMode? ?? GridSamplePaddingMode.Zeros;
            var alignCornersRaw = attributes.GetAttributeObj(AttrAlignCorners);
            var alignCorners = alignCornersRaw is true || (alignCornersRaw is long lv && lv != 0);

            if (mode != GridSampleMode.Linear)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, GRID_SAMPLE,
                    $"the gradient is only implemented for mode='linear' (bilinear; got "
                    + $"'{mode}'): nearest/bicubic adjoints are not implemented. This is an "
                    + "implementation limitation, not a mathematical one.");
            if (paddingMode != GridSamplePaddingMode.Zeros)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, GRID_SAMPLE,
                    $"the gradient is only implemented for padding_mode='zeros' (got "
                    + $"'{paddingMode}'): border/reflection padding adjoints are not "
                    + "implemented. This is an implementation limitation, not a mathematical "
                    + "one.");

            var floatType = x.Type;

            // --- Dimension extraction ---
            var xShape = OnnxOp.Shape(x);           // [4]: N, C, H_in, W_in
            var N_s  = OnnxOp.Gather(xShape, Scalar(2L), axis: 0);  // will reuse index 0 below
            var C_s  = OnnxOp.Gather(xShape, Scalar(1L), axis: 0);
            var H_s  = OnnxOp.Gather(xShape, Scalar(2L), axis: 0);  // H_in
            var W_s  = OnnxOp.Gather(xShape, Scalar(3L), axis: 0);  // W_in
            N_s = OnnxOp.Gather(xShape, Scalar(0L), axis: 0);       // fix: N

            var H_f  = OnnxOp.Cast(H_s, saturate: null, to: floatType);
            var W_f  = OnnxOp.Cast(W_s, saturate: null, to: floatType);

            var gradShape = OnnxOp.Shape(dY);        // [4]: N, C, H_out, W_out
            var Ho_s = OnnxOp.Gather(gradShape, Scalar(2L), axis: 0);
            var Wo_s = OnnxOp.Gather(gradShape, Scalar(3L), axis: 0);

            // --- Extract grid x/y components ---
            // grid [N, H_out, W_out, 2]  → gx, gy  [N, H_out, W_out]
            var gx = OnnxOp.Gather(grid, Scalar(0L), axis: 3);
            var gy = OnnxOp.Gather(grid, Scalar(1L), axis: 3);

            // Typed float constants
            var zero_f = OnnxOp.Cast(Scalar(0L), saturate: null, to: floatType);
            var one_f  = OnnxOp.Cast(Scalar(1L), saturate: null, to: floatType);
            var two_f  = OnnxOp.Cast(Scalar(2L), saturate: null, to: floatType);

            // --- Denormalize grid coordinates to pixel space ---
            Variable ix, iy, scale_x, scale_y;
            if (alignCorners)
            {
                var wm1 = OnnxOp.Sub(W_f, one_f);   // W_in - 1
                var hm1 = OnnxOp.Sub(H_f, one_f);   // H_in - 1
                ix = OnnxOp.Mul(OnnxOp.Div(OnnxOp.Add(gx, one_f), two_f), wm1);
                iy = OnnxOp.Mul(OnnxOp.Div(OnnxOp.Add(gy, one_f), two_f), hm1);
                scale_x = OnnxOp.Div(wm1, two_f);   // (W-1)/2
                scale_y = OnnxOp.Div(hm1, two_f);   // (H-1)/2
            }
            else
            {
                ix = OnnxOp.Div(OnnxOp.Sub(OnnxOp.Mul(OnnxOp.Add(gx, one_f), W_f), one_f), two_f);
                iy = OnnxOp.Div(OnnxOp.Sub(OnnxOp.Mul(OnnxOp.Add(gy, one_f), H_f), one_f), two_f);
                scale_x = OnnxOp.Div(W_f, two_f);   // W/2
                scale_y = OnnxOp.Div(H_f, two_f);   // H/2
            }

            // --- Floor indices and fractional parts ---
            var ix0_f = OnnxOp.Floor(ix);
            var iy0_f = OnnxOp.Floor(iy);
            var wx    = OnnxOp.Sub(ix, ix0_f);       // fractional x
            var wy    = OnnxOp.Sub(iy, iy0_f);       // fractional y
            var ix1_f = OnnxOp.Add(ix0_f, one_f);
            var iy1_f = OnnxOp.Add(iy0_f, one_f);

            // --- Validity masks (zeros padding) ---
            var w_max = OnnxOp.Sub(W_f, one_f);
            var h_max = OnnxOp.Sub(H_f, one_f);

            var v_ix0 = OnnxOp.And(OnnxOp.GreaterOrEqual(ix0_f, zero_f), OnnxOp.LessOrEqual(ix0_f, w_max));
            var v_ix1 = OnnxOp.And(OnnxOp.GreaterOrEqual(ix1_f, zero_f), OnnxOp.LessOrEqual(ix1_f, w_max));
            var v_iy0 = OnnxOp.And(OnnxOp.GreaterOrEqual(iy0_f, zero_f), OnnxOp.LessOrEqual(iy0_f, h_max));
            var v_iy1 = OnnxOp.And(OnnxOp.GreaterOrEqual(iy1_f, zero_f), OnnxOp.LessOrEqual(iy1_f, h_max));

            var mask_tl = OnnxOp.Cast(OnnxOp.And(v_iy0, v_ix0), saturate: null, to: floatType);
            var mask_tr = OnnxOp.Cast(OnnxOp.And(v_iy0, v_ix1), saturate: null, to: floatType);
            var mask_bl = OnnxOp.Cast(OnnxOp.And(v_iy1, v_ix0), saturate: null, to: floatType);
            var mask_br = OnnxOp.Cast(OnnxOp.And(v_iy1, v_ix1), saturate: null, to: floatType);

            // --- Clamp indices and cast to int64 ---
            var ix0_i = OnnxOp.Cast(OnnxOp.Clip(ix0_f, zero_f, w_max), saturate: null, to: DType.Int64);
            var ix1_i = OnnxOp.Cast(OnnxOp.Clip(ix1_f, zero_f, w_max), saturate: null, to: DType.Int64);
            var iy0_i = OnnxOp.Cast(OnnxOp.Clip(iy0_f, zero_f, h_max), saturate: null, to: DType.Int64);
            var iy1_i = OnnxOp.Cast(OnnxOp.Clip(iy1_f, zero_f, h_max), saturate: null, to: DType.Int64);

            // Flat spatial indices: flat = iy * W_in + ix   [N, H_out, W_out] int64
            var flat_tl = OnnxOp.Add(OnnxOp.Mul(iy0_i, W_s), ix0_i);
            var flat_tr = OnnxOp.Add(OnnxOp.Mul(iy0_i, W_s), ix1_i);
            var flat_bl = OnnxOp.Add(OnnxOp.Mul(iy1_i, W_s), ix0_i);
            var flat_br = OnnxOp.Add(OnnxOp.Mul(iy1_i, W_s), ix1_i);

            // --- Bilinear weights (masked) ---
            var omwx = OnnxOp.Sub(one_f, wx);        // 1 - wx
            var omwy = OnnxOp.Sub(one_f, wy);        // 1 - wy
            var mw_tl = OnnxOp.Mul(OnnxOp.Mul(omwy, omwx), mask_tl);
            var mw_tr = OnnxOp.Mul(OnnxOp.Mul(omwy, wx),   mask_tr);
            var mw_bl = OnnxOp.Mul(OnnxOp.Mul(wy,   omwx), mask_bl);
            var mw_br = OnnxOp.Mul(OnnxOp.Mul(wy,   wx),   mask_br);

            // ===== dInput: scatter weighted dY to input positions =====

            var HW_in  = OnnxOp.Mul(H_s, W_s);       // H_in * W_in  scalar
            var HW_out = OnnxOp.Mul(Ho_s, Wo_s);      // H_out * W_out scalar

            // Build 3D flat shapes [N, C, HW]
            var flatXShape = OnnxOp.Concat([
                OnnxOp.Reshape(N_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(HW_in, Vector(1L), allowZero: false)
            ], axis: 0);
            var flatOutShape = OnnxOp.Concat([
                OnnxOp.Reshape(N_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(HW_out, Vector(1L), allowZero: false)
            ], axis: 0);

            // zeros [N, C, H_in*W_in]
            var zeros_flat = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: floatType), flatXShape);

            // Helper: scatter one corner's contribution
            // weight [N,Ho,Wo] → [N,1,Ho,Wo] → broadcast with dY [N,C,Ho,Wo]
            // → reshape [N,C,HW_out], scatter into zeros_flat
            Variable ScatterCorner(Variable weight, Variable flatIdx)
            {
                var w4d = OnnxOp.Unsqueeze(weight, Vector(1L));          // [N,1,Ho,Wo]
                var wdY = OnnxOp.Mul(w4d, dY);                          // [N,C,Ho,Wo]
                var wdY_flat = OnnxOp.Reshape(wdY, flatOutShape, allowZero: false); // [N,C,HW_out]

                // flatIdx [N,Ho,Wo] → [N,1,HW_out] → expand [N,C,HW_out]
                var idx_flat = OnnxOp.Reshape(flatIdx,
                    OnnxOp.Concat([
                        OnnxOp.Reshape(N_s, Vector(1L), allowZero: false),
                        Vector(1L),
                        OnnxOp.Reshape(HW_out, Vector(1L), allowZero: false)
                    ], axis: 0), allowZero: false);
                var idx_exp = OnnxOp.Expand(idx_flat, flatOutShape);     // [N,C,HW_out]

                return OnnxOp.ScatterElements(zeros_flat, idx_exp, wdY_flat,
                    axis: 2, reduction: ScatterNDReduction.Add);
            }

            var dX_tl = ScatterCorner(mw_tl, flat_tl);
            var dX_tr = ScatterCorner(mw_tr, flat_tr);
            var dX_bl = ScatterCorner(mw_bl, flat_bl);
            var dX_br = ScatterCorner(mw_br, flat_br);

            // Sum contributions and reshape to [N, C, H_in, W_in]
            var dX_flat = OnnxOp.Add(OnnxOp.Add(dX_tl, dX_tr), OnnxOp.Add(dX_bl, dX_br));
            var dX = OnnxOp.Reshape(dX_flat, xShape, allowZero: false);

            // ===== dGrid: compute partial derivatives of interpolation =====

            // Look up input values at 4 corners: X_flat [N,C,HW_in], gather [N,C,HW_out]
            var X_flat = OnnxOp.Reshape(x, flatXShape, allowZero: false);

            Variable LookupCorner(Variable flatIdx, Variable mask)
            {
                var idx_flat = OnnxOp.Reshape(flatIdx,
                    OnnxOp.Concat([
                        OnnxOp.Reshape(N_s, Vector(1L), allowZero: false),
                        Vector(1L),
                        OnnxOp.Reshape(HW_out, Vector(1L), allowZero: false)
                    ], axis: 0), allowZero: false);
                var idx_exp = OnnxOp.Expand(idx_flat, flatOutShape);
                var vals = OnnxOp.GatherElements(X_flat, idx_exp, axis: 2); // [N,C,HW_out]
                var vals_4d = OnnxOp.Reshape(vals, gradShape, allowZero: false);  // [N,C,Ho,Wo]
                // Mask: multiply by mask [N,Ho,Wo] unsqueezed to [N,1,Ho,Wo]
                return OnnxOp.Mul(vals_4d, OnnxOp.Unsqueeze(mask, Vector(1L)));
            }

            var Xtl = LookupCorner(flat_tl, mask_tl);  // [N,C,Ho,Wo]
            var Xtr = LookupCorner(flat_tr, mask_tr);
            var Xbl = LookupCorner(flat_bl, mask_bl);
            var Xbr = LookupCorner(flat_br, mask_br);

            // d(interp)/d(ix) = (1-wy)*(Xtr - Xtl) + wy*(Xbr - Xbl)
            // Unsqueeze wy,wx to [N,1,Ho,Wo] for channel broadcast
            var wy4 = OnnxOp.Unsqueeze(wy, Vector(1L));
            var wx4 = OnnxOp.Unsqueeze(wx, Vector(1L));
            var omwy4 = OnnxOp.Unsqueeze(omwy, Vector(1L));
            var omwx4 = OnnxOp.Unsqueeze(omwx, Vector(1L));

            var dInterp_dx = OnnxOp.Add(
                OnnxOp.Mul(omwy4, OnnxOp.Sub(Xtr, Xtl)),
                OnnxOp.Mul(wy4, OnnxOp.Sub(Xbr, Xbl)));           // [N,C,Ho,Wo]

            // d(interp)/d(iy) = (1-wx)*(Xbl - Xtl) + wx*(Xbr - Xtr)
            var dInterp_dy = OnnxOp.Add(
                OnnxOp.Mul(omwx4, OnnxOp.Sub(Xbl, Xtl)),
                OnnxOp.Mul(wx4, OnnxOp.Sub(Xbr, Xtr)));           // [N,C,Ho,Wo]

            // dGrid_x = scale_x * ΣC(dY * dInterp_dx)   [N,Ho,Wo]
            // dGrid_y = scale_y * ΣC(dY * dInterp_dy)   [N,Ho,Wo]
            var dGx = OnnxOp.Mul(scale_x,
                OnnxOp.Squeeze(
                    OnnxOp.ReduceSum(OnnxOp.Mul(dY, dInterp_dx), Vector(1L), keepdims: true, noopWithEmptyAxes: false),
                    Vector(1L)));                                    // [N,Ho,Wo]
            var dGy = OnnxOp.Mul(scale_y,
                OnnxOp.Squeeze(
                    OnnxOp.ReduceSum(OnnxOp.Mul(dY, dInterp_dy), Vector(1L), keepdims: true, noopWithEmptyAxes: false),
                    Vector(1L)));                                    // [N,Ho,Wo]

            // Stack into dGrid [N, Ho, Wo, 2]
            var dGx_4 = OnnxOp.Unsqueeze(dGx, Vector(3L));          // [N,Ho,Wo,1]
            var dGy_4 = OnnxOp.Unsqueeze(dGy, Vector(3L));          // [N,Ho,Wo,1]
            var dGrid = OnnxOp.Concat([dGx_4, dGy_4], axis: 3);     // [N,Ho,Wo,2]

            return [dX, dGrid];
        }
    }
}
