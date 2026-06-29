using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== ROI_ALIGN (avg mode, bilinear interpolation) =====
        //
        // Forward: Y[r, c, ph, pw] = (1 / count) * Σ_{iy,ix} bilinear(X[batch_indices[r], c, :, :], y, x)
        //   For each ROI r with (x1, y1, x2, y2) scaled by spatial_scale:
        //     bin_size_h = (y2 - y1) / output_height
        //     bin_size_w = (x2 - x1) / output_width
        //     For each output bin (ph, pw), sampling_ratio^2 sample points are taken.
        //     Sample coordinate: y = y1 + ph * bin_h + (iy + 0.5) * bin_h / sampling_ratio
        //                        x = x1 + pw * bin_w + (ix + 0.5) * bin_w / sampling_ratio
        //     Bilinear interpolation from 4 neighbors (same as GridSample).
        //
        // Gradient w.r.t. X:
        //   dX = scatter back dY / count through bilinear weights to input positions,
        //   accumulated per batch element via batch_indices.
        //
        // Gradient w.r.t. rois: null (typically not differentiable in standard use).
        // Gradient w.r.t. batch_indices: null (int64).
        //
        // Supported: avg mode, sampling_ratio >= 1 (fixed). Default half_pixel coords.

        internal static Variable?[] RoiAlignGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;                  // [N, C, H, W]
            var rois = inputs[1]!;               // [num_rois, 4]  (x1, y1, x2, y2)
            var batchIndices = inputs[2]!;        // [num_rois] int64
            var dY = outputGrads[0];              // [num_rois, C, output_height, output_width]

            if (dY is null) return [null, null, null];

            // Read attributes
            var modeRaw = attributes.GetAttributeObj(AttrMode);
            var mode = modeRaw as RoiAlignMode? ?? RoiAlignMode.Avg;
            if (mode != RoiAlignMode.Avg)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, ROI_ALIGN,
                    $"the gradient is only implemented for mode='avg' (got '{mode}'): the max "
                    + "mode's argmax-routing adjoint is not implemented. This is an "
                    + "implementation limitation, not a mathematical one.");

            var coordModeRaw = attributes.GetAttributeObj(AttrCoordinateTransformationMode);
            var isHalfPixel = coordModeRaw is null
                || coordModeRaw is RoiAlignTransformationMode rtm && rtm == RoiAlignTransformationMode.Half_pixel
                // Graphs serialized before the dedicated RoiAlignTransformationMode enum may
                // still carry the Resize-wide enum here.
                || coordModeRaw is CoordinateTransformationMode ctm && ctm == CoordinateTransformationMode.Half_pixel
                || coordModeRaw is string s && s == "half_pixel";

            var outputHeight = attributes.GetAttributeObj(AttrOutputHeight) is long oh ? oh : 1L;
            var outputWidth  = attributes.GetAttributeObj(AttrOutputWidth)  is long ow ? ow : 1L;
            var samplingRatio = attributes.GetAttributeObj(AttrSamplingRatio) is long sr && sr > 0 ? sr : 2L;
            var spatialScaleVal = attributes.GetAttributeObj(AttrSpatialScale) is float ss ? ss : 1.0f;

            var floatType = x.Type;

            // --- Dimension extraction ---
            var xShape = OnnxOp.Shape(x);                                           // [4]
            var N_s  = OnnxOp.Gather(xShape, Scalar(0L), axis: 0);                 // N
            var C_s  = OnnxOp.Gather(xShape, Scalar(1L), axis: 0);                 // C
            var H_s  = OnnxOp.Gather(xShape, Scalar(2L), axis: 0);                 // H
            var W_s  = OnnxOp.Gather(xShape, Scalar(3L), axis: 0);                 // W

            var H_f  = OnnxOp.Cast(H_s, saturate: null, to: floatType);
            var W_f  = OnnxOp.Cast(W_s, saturate: null, to: floatType);

            var numRois_s = OnnxOp.Gather(OnnxOp.Shape(rois), Scalar(0L), axis: 0); // num_rois

            // Typed constants
            var zero_f = OnnxOp.Cast(Scalar(0L), saturate: null, to: floatType);
            var one_f  = OnnxOp.Cast(Scalar(1L), saturate: null, to: floatType);
            var half_f = OnnxOp.Cast(Scalar(0.5f), saturate: null, to: floatType);

            // --- ROI coordinates ---
            // rois [num_rois, 4]: (x1, y1, x2, y2) in image coordinates
            var spatialScale = OnnxOp.Cast(Scalar(spatialScaleVal), saturate: null, to: floatType);
            var rois_scaled = OnnxOp.Mul(rois, spatialScale);                       // [num_rois, 4]

            var x1 = OnnxOp.Gather(rois_scaled, Scalar(0L), axis: 1);              // [num_rois]
            var y1 = OnnxOp.Gather(rois_scaled, Scalar(1L), axis: 1);
            var x2 = OnnxOp.Gather(rois_scaled, Scalar(2L), axis: 1);
            var y2 = OnnxOp.Gather(rois_scaled, Scalar(3L), axis: 1);

            // Half-pixel offset
            if (isHalfPixel)
            {
                x1 = OnnxOp.Sub(x1, half_f);
                y1 = OnnxOp.Sub(y1, half_f);
                x2 = OnnxOp.Sub(x2, half_f);
                y2 = OnnxOp.Sub(y2, half_f);
            }

            // Bin sizes
            var oh_f = OnnxOp.Cast(Scalar(outputHeight), saturate: null, to: floatType);
            var ow_f = OnnxOp.Cast(Scalar(outputWidth), saturate: null, to: floatType);
            var bin_h = OnnxOp.Div(OnnxOp.Sub(y2, y1), oh_f);                      // [num_rois]
            var bin_w = OnnxOp.Div(OnnxOp.Sub(x2, x1), ow_f);

            // Sampling step within each bin
            var sr_f = OnnxOp.Cast(Scalar(samplingRatio), saturate: null, to: floatType);
            var step_h = OnnxOp.Div(bin_h, sr_f);                                  // [num_rois]
            var step_w = OnnxOp.Div(bin_w, sr_f);
            var count  = OnnxOp.Mul(sr_f, sr_f);                                   // total samples per bin

            // --- Build all sampling coordinates ---
            // For each output bin (ph, pw) and each sub-sample (iy, ix):
            //   sample_y = y1 + ph * bin_h + (iy + 0.5) * step_h
            //   sample_x = x1 + pw * bin_w + (ix + 0.5) * step_w
            //
            // Shape: [num_rois, output_height * sampling_ratio, output_width * sampling_ratio]

            // Sub-sample offsets: (0.5, 1.5, ...) * step / sr
            var rangeH = OnnxOp.Cast(
                OnnxOp.Range(Scalar(0L), Scalar(outputHeight * samplingRatio), Scalar(1L)),
                saturate: null, to: floatType);                                     // [OH*SR]
            var rangeW = OnnxOp.Cast(
                OnnxOp.Range(Scalar(0L), Scalar(outputWidth * samplingRatio), Scalar(1L)),
                saturate: null, to: floatType);                                     // [OW*SR]

            // sample_y offsets: for idx i in range(OH*SR), offset = (floor(i/SR)*SR + mod(i,SR) + 0.5) * step_h
            // Simplify: offset_h[i] = (i + 0.5) * step_h  where i goes 0..OH*SR-1
            // Then sample_y[r, i] = y1[r] + offset_h[i] * step_h...
            // Actually: sample_y[r, i] = y1[r] + (i + 0.5) * bin_h / SR
            //   = y1[r] + (i + 0.5) * step_h

            // Compute: offset_h = (rangeH + 0.5) → [OH*SR]
            var offset_h = OnnxOp.Add(rangeH, half_f);                              // [OH*SR]
            var offset_w = OnnxOp.Add(rangeW, half_f);                              // [OW*SR]

            // sample_y[r, h_idx] = y1[r] + offset_h[h_idx] * step_h[r]
            // y1: [num_rois] → [num_rois, 1]
            // step_h: [num_rois] → [num_rois, 1]
            // offset_h: [OH*SR] → [1, OH*SR]
            var y1_2d = OnnxOp.Unsqueeze(y1, Vector(1L));                           // [num_rois, 1]
            var x1_2d = OnnxOp.Unsqueeze(x1, Vector(1L));
            var step_h_2d = OnnxOp.Unsqueeze(step_h, Vector(1L));                   // [num_rois, 1]
            var step_w_2d = OnnxOp.Unsqueeze(step_w, Vector(1L));

            var offset_h_2d = OnnxOp.Unsqueeze(offset_h, Vector(0L));              // [1, OH*SR]
            var offset_w_2d = OnnxOp.Unsqueeze(offset_w, Vector(0L));              // [1, OW*SR]

            // sample_y: [num_rois, OH*SR]
            var sample_y = OnnxOp.Add(y1_2d, OnnxOp.Mul(offset_h_2d, step_h_2d));
            // sample_x: [num_rois, OW*SR]
            var sample_x = OnnxOp.Add(x1_2d, OnnxOp.Mul(offset_w_2d, step_w_2d));

            // Create meshgrid: [num_rois, OH*SR, OW*SR]
            var sample_y_3d = OnnxOp.Unsqueeze(sample_y, Vector(2L));              // [num_rois, OH*SR, 1]
            var sample_x_3d = OnnxOp.Unsqueeze(sample_x, Vector(1L));              // [num_rois, 1, OW*SR]

            var totalH = outputHeight * samplingRatio;
            var totalW = outputWidth  * samplingRatio;
            var meshShape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                Vector(totalH),
                Vector(totalW)
            ], axis: 0);

            var sy = OnnxOp.Expand(sample_y_3d, meshShape);                        // [num_rois, OH*SR, OW*SR]
            var sx = OnnxOp.Expand(sample_x_3d, meshShape);

            // --- Bilinear interpolation setup ---
            // ORT clamps sample coordinates to [0, H-1]×[0, W-1] before bilinear interpolation
            // (not zeroing out-of-bounds). Clamp before computing floor/frac to match.
            var h_max = OnnxOp.Sub(H_f, one_f);
            var w_max = OnnxOp.Sub(W_f, one_f);

            var sy_c = OnnxOp.Clip(sy, zero_f, h_max);                             // clamped y
            var sx_c = OnnxOp.Clip(sx, zero_f, w_max);                             // clamped x

            var sy0_f = OnnxOp.Floor(sy_c);
            var sx0_f = OnnxOp.Floor(sx_c);
            var wy = OnnxOp.Sub(sy_c, sy0_f);                                      // fractional y
            var wx = OnnxOp.Sub(sx_c, sx0_f);                                      // fractional x

            // Cast floor to int64 (always valid after clamping)
            var sy0_i = OnnxOp.Cast(sy0_f, saturate: null, to: DType.Int64);
            var sx0_i = OnnxOp.Cast(sx0_f, saturate: null, to: DType.Int64);
            // sy1/sx1 may equal H or W when sy_c == h_max; clamp to valid range
            // (weight wy/wx is 0 at that edge, so the clamped index value is unused)
            var sy1_i = OnnxOp.Cast(OnnxOp.Clip(OnnxOp.Add(sy0_f, one_f), zero_f, h_max),
                saturate: null, to: DType.Int64);
            var sx1_i = OnnxOp.Cast(OnnxOp.Clip(OnnxOp.Add(sx0_f, one_f), zero_f, w_max),
                saturate: null, to: DType.Int64);

            // Flat spatial indices: flat = y * W + x   [num_rois, OH*SR, OW*SR]
            var flat_tl = OnnxOp.Add(OnnxOp.Mul(sy0_i, W_s), sx0_i);
            var flat_tr = OnnxOp.Add(OnnxOp.Mul(sy0_i, W_s), sx1_i);
            var flat_bl = OnnxOp.Add(OnnxOp.Mul(sy1_i, W_s), sx0_i);
            var flat_br = OnnxOp.Add(OnnxOp.Mul(sy1_i, W_s), sx1_i);

            // Bilinear weights (no masks needed — coordinates are clamped)
            var omwx = OnnxOp.Sub(one_f, wx);
            var omwy = OnnxOp.Sub(one_f, wy);
            var mw_tl = OnnxOp.Mul(omwy, omwx);
            var mw_tr = OnnxOp.Mul(omwy, wx);
            var mw_bl = OnnxOp.Mul(wy,   omwx);
            var mw_br = OnnxOp.Mul(wy,   wx);

            // --- Scatter dY back to dX ---
            // dY: [num_rois, C, output_height, output_width]
            // We need to:
            // 1. Expand dY to match sampling points (average over SR^2 samples per bin)
            // 2. Scatter weighted dY to input spatial positions
            // 3. Accumulate across ROIs into the correct batch elements

            var HW_in = OnnxOp.Mul(H_s, W_s);                                     // H * W

            // Repeat dY for sub-sampling: [num_rois, C, OH, OW] → [num_rois, C, OH*SR, OW*SR]
            // Use Reshape + Expand to tile each bin over sampling_ratio in each spatial dim.
            // dY / count gives the per-sample gradient.
            var dY_avg = OnnxOp.Div(dY, count);                                    // [num_rois, C, OH, OW]

            // Reshape to [num_rois, C, OH, 1, OW, 1]
            var dY_6d_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                Vector(outputHeight, 1L, outputWidth, 1L)
            ], axis: 0);
            var dY_6d = OnnxOp.Reshape(dY_avg, dY_6d_shape, allowZero: false);

            // Expand to [num_rois, C, OH, SR, OW, SR]
            var dY_exp_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                Vector(outputHeight, samplingRatio, outputWidth, samplingRatio)
            ], axis: 0);
            var dY_expanded = OnnxOp.Expand(dY_6d, dY_exp_shape);

            // Reshape to [num_rois, C, OH*SR, OW*SR]
            var dY_sample_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                Vector(totalH, totalW)
            ], axis: 0);
            var dY_samples = OnnxOp.Reshape(dY_expanded, dY_sample_shape, allowZero: false);
            // dY_samples: [num_rois, C, OH*SR, OW*SR]

            // Flatten spatial dims: [num_rois, C, OH*SR*OW*SR]
            var totalSamples = totalH * totalW;
            var dY_flat_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                Vector(totalSamples)
            ], axis: 0);
            var dY_flat = OnnxOp.Reshape(dY_samples, dY_flat_shape, allowZero: false);

            // Flatten index and weight arrays: [num_rois, OH*SR*OW*SR]
            var flat_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                Vector(totalSamples)
            ], axis: 0);

            var flat_tl_2d = OnnxOp.Reshape(flat_tl, flat_shape, allowZero: false);
            var flat_tr_2d = OnnxOp.Reshape(flat_tr, flat_shape, allowZero: false);
            var flat_bl_2d = OnnxOp.Reshape(flat_bl, flat_shape, allowZero: false);
            var flat_br_2d = OnnxOp.Reshape(flat_br, flat_shape, allowZero: false);

            var mw_tl_2d = OnnxOp.Reshape(mw_tl, flat_shape, allowZero: false);
            var mw_tr_2d = OnnxOp.Reshape(mw_tr, flat_shape, allowZero: false);
            var mw_bl_2d = OnnxOp.Reshape(mw_bl, flat_shape, allowZero: false);
            var mw_br_2d = OnnxOp.Reshape(mw_br, flat_shape, allowZero: false);

            // For each ROI, scatter into [C, H*W]
            // Expand indices to [num_rois, C, totalSamples] by unsqueezing and expanding
            var idx_3d_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                Vector(1L),
                Vector(totalSamples)
            ], axis: 0);

            Variable ExpandIdx(Variable idx2d)
            {
                var idx3d = OnnxOp.Reshape(idx2d, idx_3d_shape, allowZero: false);
                return OnnxOp.Expand(idx3d, dY_flat_shape);                        // [num_rois, C, totalSamples]
            }

            Variable ExpandWeight(Variable w2d)
            {
                var w3d = OnnxOp.Reshape(w2d, idx_3d_shape, allowZero: false);
                return OnnxOp.Expand(w3d, dY_flat_shape);                          // [num_rois, C, totalSamples]
            }

            // zeros [num_rois, C, H*W]
            var zeros_shape = OnnxOp.Concat([
                OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(HW_in, Vector(1L), allowZero: false)
            ], axis: 0);
            var zeros_flat = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: floatType), zeros_shape);

            // Scatter each corner's contribution
            Variable ScatterCorner(Variable weight2d, Variable flatIdx2d)
            {
                var w_exp = ExpandWeight(weight2d);                                // [num_rois, C, totalSamples]
                var idx_exp = ExpandIdx(flatIdx2d);                                // [num_rois, C, totalSamples]
                var wdY = OnnxOp.Mul(w_exp, dY_flat);                              // weighted gradient
                return OnnxOp.ScatterElements(zeros_flat, idx_exp, wdY,
                    axis: 2, reduction: ScatterNDReduction.Add);
            }

            var dX_tl = ScatterCorner(mw_tl_2d, flat_tl_2d);
            var dX_tr = ScatterCorner(mw_tr_2d, flat_tr_2d);
            var dX_bl = ScatterCorner(mw_bl_2d, flat_bl_2d);
            var dX_br = ScatterCorner(mw_br_2d, flat_br_2d);

            // Sum all corners: [num_rois, C, H*W]
            var dX_roi = OnnxOp.Add(OnnxOp.Add(dX_tl, dX_tr), OnnxOp.Add(dX_bl, dX_br));

            // --- Accumulate per-ROI gradients into global dX using batch_indices ---
            // dX_roi: [num_rois, C, H*W]
            // batch_indices: [num_rois] int64
            // Need: dX[n, c, hw] = Σ_{r: batch_indices[r]==n} dX_roi[r, c, hw]
            //
            // Use ScatterND: indices [num_rois, 1], updates [num_rois, C, H*W]
            //   → scatter into [N, C, H*W] with Add reduction

            // Expand batch_indices to [num_rois, 1] for ScatterND
            var bi_2d = OnnxOp.Reshape(batchIndices,
                OnnxOp.Concat([
                    OnnxOp.Reshape(numRois_s, Vector(1L), allowZero: false),
                    Vector(1L)
                ], axis: 0), allowZero: false);                                    // [num_rois, 1]

            // Target zeros [N, C, H*W]
            var dX_global_shape = OnnxOp.Concat([
                OnnxOp.Reshape(N_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(C_s, Vector(1L), allowZero: false),
                OnnxOp.Reshape(HW_in, Vector(1L), allowZero: false)
            ], axis: 0);
            var dX_global_zeros = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: floatType), dX_global_shape);

            var dX_global = OnnxOp.ScatterND(dX_global_zeros, bi_2d, dX_roi,
                reduction: ScatterNDReduction.Add);

            // Reshape to [N, C, H, W]
            var dX = OnnxOp.Reshape(dX_global, xShape, allowZero: false);

            return [dX, null, null];  // null for rois and batch_indices
        }
    }
}
