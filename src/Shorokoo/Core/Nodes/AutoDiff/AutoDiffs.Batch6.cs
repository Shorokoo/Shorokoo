using System;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Resize (nearest mode, asymmetric coordinate transform) =====
        //
        // Forward: output[n,c,oh,ow] = input[n,c,ih,iw]
        //   where ih = floor(oh * H_in / H_out), iw = floor(ow * W_in / W_out)
        //   (asymmetric coordinate transform for nearest mode)
        //
        // Gradient: grad_x[n,c,ih,iw] += grad[n,c,oh,ow]
        //           for all (oh,ow) that map to (ih,iw)
        //
        // Implemented via two-pass ScatterElements with Add reduction:
        //   Pass 1: scatter grad along H axis (axis=2) → intermediate [N,C,H_in,W_out]
        //   Pass 2: scatter intermediate along W axis (axis=3) → grad_x [N,C,H_in,W_in]

        internal static Variable?[] ResizeGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var mode = attributes.GetAttributeObj("mode") as ResizeMode? ?? ResizeMode.Nearest;
            if (mode != ResizeMode.Nearest)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, RESIZE,
                    $"the gradient is only implemented for mode='nearest' (got '{mode}'): "
                    + "applying the nearest-style index scatter to linear/cubic would be "
                    + "silently wrong. This is an implementation limitation, not a "
                    + "mathematical one.");

            // The index math below assumes the 'asymmetric' coordinate transform; an
            // explicitly different transform is outside the implemented envelope.
            var coordMode = attributes.GetAttributeObj("coordinate_transformation_mode")
                as CoordinateTransformationMode?;
            if (coordMode is not null && coordMode != CoordinateTransformationMode.Asymmetric)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, RESIZE,
                    $"the gradient is only implemented for "
                    + $"coordinate_transformation_mode='asymmetric' (got '{coordMode}'). This is "
                    + "an implementation limitation, not a mathematical one.");

            var x = inputs[0]!;
            var grad = outputGrads[0]!;

            // Get shape vectors [N, C, H, W] as int64 tensors
            var xShape    = OnnxOp.Shape(x);    // [N, C, H_in, W_in]
            var gradShape = OnnxOp.Shape(grad); // [N, C, H_out, W_out]

            // Extract individual 1-element dimension vectors
            var N_vec     = OnnxOp.Slice(xShape,    Vector(0L), Vector(1L));
            var C_vec     = OnnxOp.Slice(xShape,    Vector(1L), Vector(2L));
            var H_in_vec  = OnnxOp.Slice(xShape,    Vector(2L), Vector(3L));
            var W_in_vec  = OnnxOp.Slice(xShape,    Vector(3L), Vector(4L));
            var H_out_vec = OnnxOp.Slice(gradShape, Vector(2L), Vector(3L));
            var W_out_vec = OnnxOp.Slice(gradShape, Vector(3L), Vector(4L));

            // Squeeze to scalars for Range / float arithmetic
            var H_in_s  = OnnxOp.Squeeze(H_in_vec,  Vector(0L));
            var W_in_s  = OnnxOp.Squeeze(W_in_vec,  Vector(0L));
            var H_out_s = OnnxOp.Squeeze(H_out_vec, Vector(0L));
            var W_out_s = OnnxOp.Squeeze(W_out_vec, Vector(0L));

            // Cast dim scalars to float32 for index arithmetic
            var H_in_f  = OnnxOp.Cast(H_in_s,  saturate: null, to: DType.Float32);
            var W_in_f  = OnnxOp.Cast(W_in_s,  saturate: null, to: DType.Float32);
            var H_out_f = OnnxOp.Cast(H_out_s, saturate: null, to: DType.Float32);
            var W_out_f = OnnxOp.Cast(W_out_s, saturate: null, to: DType.Float32);

            // H index mapping: ih = floor(oh * H_in / H_out)  [asymmetric coordinate mode]
            var h_range   = OnnxOp.Range(Scalar(0L), H_out_s, Scalar(1L));             // int64 [H_out]
            var h_range_f = OnnxOp.Cast(h_range, saturate: null, to: DType.Float32);   // float [H_out]
            var ih_f      = OnnxOp.Floor(OnnxOp.Div(OnnxOp.Mul(h_range_f, H_in_f), H_out_f)); // float [H_out]
            var ih        = OnnxOp.Cast(ih_f, saturate: null, to: DType.Int64);         // int64 [H_out]

            // W index mapping: iw = floor(ow * W_in / W_out)  [asymmetric coordinate mode]
            var w_range   = OnnxOp.Range(Scalar(0L), W_out_s, Scalar(1L));
            var w_range_f = OnnxOp.Cast(w_range, saturate: null, to: DType.Float32);
            var iw_f      = OnnxOp.Floor(OnnxOp.Div(OnnxOp.Mul(w_range_f, W_in_f), W_out_f));
            var iw        = OnnxOp.Cast(iw_f, saturate: null, to: DType.Int64);         // int64 [W_out]

            // --- Pass 1: Scatter grad along H axis (axis=2) ---
            // Reshape ih [H_out] → [1,1,H_out,1] and broadcast to [N,C,H_out,W_out]
            var ih_4d_shape = OnnxOp.Concat([Vector(1L), Vector(1L), H_out_vec, Vector(1L)], axis: 0);
            var ih_4d       = OnnxOp.Reshape(ih, ih_4d_shape, allowZero: false);  // [1, 1, H_out, 1]
            var h_indices   = OnnxOp.Expand(ih_4d, gradShape);                    // [N, C, H_out, W_out]

            // Intermediate tensor: [N, C, H_in, W_out]
            var intermediateShape  = OnnxOp.Concat([N_vec, C_vec, H_in_vec, W_out_vec], axis: 0);
            var zeros_intermediate = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type),
                intermediateShape);  // zeros [N, C, H_in, W_out]

            // ScatterElements axis=2 with Add: accumulate grad from H_out rows into H_in rows
            var intermediate = OnnxOp.ScatterElements(
                zeros_intermediate, h_indices, grad,
                axis: 2, reduction: ScatterNDReduction.Add);  // [N, C, H_in, W_out]

            // --- Pass 2: Scatter intermediate along W axis (axis=3) ---
            // Reshape iw [W_out] → [1,1,1,W_out] and broadcast to [N,C,H_in,W_out]
            var iw_4d_shape = OnnxOp.Concat([Vector(1L), Vector(1L), Vector(1L), W_out_vec], axis: 0);
            var iw_4d       = OnnxOp.Reshape(iw, iw_4d_shape, allowZero: false);  // [1, 1, 1, W_out]
            var w_indices   = OnnxOp.Expand(iw_4d, intermediateShape);            // [N, C, H_in, W_out]

            // Final output: [N, C, H_in, W_in]
            var zeros_x = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type),
                xShape);  // zeros [N, C, H_in, W_in]

            // ScatterElements axis=3 with Add: accumulate intermediate from W_out cols into W_in cols
            var grad_x = OnnxOp.ScatterElements(
                zeros_x, w_indices, intermediate,
                axis: 3, reduction: ScatterNDReduction.Add);  // [N, C, H_in, W_in]

            // Return: gradient for x, null for roi/scales/sizes
            return [grad_x, null, null, null];
        }


        // ===== Upsample (nearest mode — deprecated Resize op) =====
        //
        // Forward: output[n,c,oh,ow] = input[n,c,ih,iw]
        //   where ih = floor(oh / scale_h), iw = floor(ow / scale_w)
        //   (equivalent to Resize with asymmetric coordinate transform)
        //
        // Gradient: grad_x[n,c,ih,iw] += grad[n,c,oh,ow]
        //           for all (oh,ow) that map to (ih,iw)
        //
        // Implemented via the same two-pass ScatterElements approach as ResizeGradient.
        // Return: gradient for x only; scales has no gradient (non-differentiable index math).

        internal static Variable?[] UpsampleGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var mode = attributes.GetAttributeObj("mode") as ResizeMode? ?? ResizeMode.Nearest;
            if (mode != ResizeMode.Nearest)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, UPSAMPLE,
                    $"the gradient is only implemented for mode='nearest' (got '{mode}'): "
                    + "applying the nearest-style index scatter to linear would be silently "
                    + "wrong. This is an implementation limitation, not a mathematical one.");

            var x = inputs[0]!;
            var grad = outputGrads[0]!;

            // Get shape vectors [N, C, H, W] as int64 tensors
            var xShape    = OnnxOp.Shape(x);    // [N, C, H_in, W_in]
            var gradShape = OnnxOp.Shape(grad); // [N, C, H_out, W_out]

            // Extract individual 1-element dimension vectors
            var N_vec     = OnnxOp.Slice(xShape,    Vector(0L), Vector(1L));
            var C_vec     = OnnxOp.Slice(xShape,    Vector(1L), Vector(2L));
            var H_in_vec  = OnnxOp.Slice(xShape,    Vector(2L), Vector(3L));
            var W_in_vec  = OnnxOp.Slice(xShape,    Vector(3L), Vector(4L));
            var H_out_vec = OnnxOp.Slice(gradShape, Vector(2L), Vector(3L));
            var W_out_vec = OnnxOp.Slice(gradShape, Vector(3L), Vector(4L));

            // Squeeze to scalars for Range / float arithmetic
            var H_in_s  = OnnxOp.Squeeze(H_in_vec,  Vector(0L));
            var W_in_s  = OnnxOp.Squeeze(W_in_vec,  Vector(0L));
            var H_out_s = OnnxOp.Squeeze(H_out_vec, Vector(0L));
            var W_out_s = OnnxOp.Squeeze(W_out_vec, Vector(0L));

            // Cast dim scalars to float32 for index arithmetic
            var H_in_f  = OnnxOp.Cast(H_in_s,  saturate: null, to: DType.Float32);
            var W_in_f  = OnnxOp.Cast(W_in_s,  saturate: null, to: DType.Float32);
            var H_out_f = OnnxOp.Cast(H_out_s, saturate: null, to: DType.Float32);
            var W_out_f = OnnxOp.Cast(W_out_s, saturate: null, to: DType.Float32);

            // H index mapping: ih = floor(oh * H_in / H_out)  [equivalent to floor(oh / scale_h)]
            var h_range   = OnnxOp.Range(Scalar(0L), H_out_s, Scalar(1L));
            var h_range_f = OnnxOp.Cast(h_range, saturate: null, to: DType.Float32);
            var ih_f      = OnnxOp.Floor(OnnxOp.Div(OnnxOp.Mul(h_range_f, H_in_f), H_out_f));
            var ih        = OnnxOp.Cast(ih_f, saturate: null, to: DType.Int64);

            // W index mapping: iw = floor(ow * W_in / W_out)  [equivalent to floor(ow / scale_w)]
            var w_range   = OnnxOp.Range(Scalar(0L), W_out_s, Scalar(1L));
            var w_range_f = OnnxOp.Cast(w_range, saturate: null, to: DType.Float32);
            var iw_f      = OnnxOp.Floor(OnnxOp.Div(OnnxOp.Mul(w_range_f, W_in_f), W_out_f));
            var iw        = OnnxOp.Cast(iw_f, saturate: null, to: DType.Int64);

            // --- Pass 1: Scatter grad along H axis (axis=2) ---
            var ih_4d_shape = OnnxOp.Concat([Vector(1L), Vector(1L), H_out_vec, Vector(1L)], axis: 0);
            var ih_4d       = OnnxOp.Reshape(ih, ih_4d_shape, allowZero: false);
            var h_indices   = OnnxOp.Expand(ih_4d, gradShape);

            var intermediateShape  = OnnxOp.Concat([N_vec, C_vec, H_in_vec, W_out_vec], axis: 0);
            var zeros_intermediate = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type),
                intermediateShape);

            var intermediate = OnnxOp.ScatterElements(
                zeros_intermediate, h_indices, grad,
                axis: 2, reduction: ScatterNDReduction.Add);

            // --- Pass 2: Scatter intermediate along W axis (axis=3) ---
            var iw_4d_shape = OnnxOp.Concat([Vector(1L), Vector(1L), Vector(1L), W_out_vec], axis: 0);
            var iw_4d       = OnnxOp.Reshape(iw, iw_4d_shape, allowZero: false);
            var w_indices   = OnnxOp.Expand(iw_4d, intermediateShape);

            var zeros_x = OnnxOp.Expand(
                OnnxOp.Cast(Scalar(0.0f), saturate: null, to: x.Type),
                xShape);

            var grad_x = OnnxOp.ScatterElements(
                zeros_x, w_indices, intermediate,
                axis: 3, reduction: ScatterNDReduction.Add);

            // Return: gradient for x only; scales is not differentiable
            return [grad_x, null];
        }


        // ===== LRN (Local Response Normalization) =====
        //
        // Forward: y_c = x_c / (bias + alpha/size * sum_{j in window(c)} x_j^2)^beta
        // where window(c) = [max(0, c - floor((size-1)/2)), min(C-1, c + ceil((size-1)/2))]
        //
        // Gradient: dL/dx_k = dL/dy_k * pool_k^(-beta)
        //           - 2*alpha*beta/size * x_k * ChannelWindowSum(dL/dy * x * pool^(-(beta+1)))_k
        // where pool_k = bias + alpha/size * ChannelWindowSum(x^2)_k

        [AutoDiff(LRN)]
        public static Variable?[] Lrn<T>(
            Tensor<T> x, Tensor<T> grad, float? alpha, float? beta, float? bias, long? size)
            where T : IVarType
        {
            var effectiveAlpha = alpha ?? 0.0001f;
            var effectiveBeta = beta ?? 0.75f;
            var effectiveBias = bias ?? 1.0f;
            var effectiveSize = size ?? 5L;

            // Asymmetric split for even window sizes:
            // e.g., size=5: leftHalf=2, rightHalf=2 (symmetric)
            // e.g., size=4: leftHalf=1, rightHalf=2 (matching ONNX floor/ceil spec)
            var leftHalf = (effectiveSize - 1) / 2;
            var rightHalf = effectiveSize - 1 - leftHalf;

            // Get the number of channels dynamically
            Tensor<int64> cDim = OnnxOp.Slice(OnnxOp.Shape(x), Vector(1L), Vector(2L));

            // Helper: Channel window sum (sliding window sum along channel axis 1)
            // Uses Pad + unrolled Slice-and-Add since size is a static attribute
            Tensor<T> ChannelWindowSum(Tensor<T> tensor)
            {
                // Pad along channel dimension (axis 1) with zeros
                var pads = Vector(leftHalf, rightHalf);
                Tensor<T> padded = OnnxOp.Pad(tensor, pads, null, axes: Vector(1L), mode: PadMode.Constant);

                // Unrolled slice-and-add across the window
                // First slice establishes the accumulator
                Tensor<T> result = OnnxOp.Slice(padded, Vector(0L), cDim, Vector(1L));
                for (long i = 1; i < effectiveSize; i++)
                {
                    var start = Vector(i);
                    Tensor<int64> end = OnnxOp.Add(cDim, Vector(i));
                    Tensor<T> sliced = OnnxOp.Slice(padded, start, end, Vector(1L));
                    result = result + sliced;
                }

                return result;
            }

            // Step 1: Compute pool = bias + (alpha/size) * ChannelWindowSum(x^2)
            var xSquared = x * x;
            var windowSumXSq = ChannelWindowSum(xSquared);
            var alphaOverSize = TypedConst(effectiveAlpha / (float)effectiveSize, x);
            var biasConst = TypedConst(effectiveBias, x);
            var pool = biasConst + alphaOverSize * windowSumXSq;

            // Step 2: term1 = grad * pool^(-beta)
            var negBeta = TypedConst(-effectiveBeta, x);
            Tensor<T> poolNegBeta = OnnxOp.Pow(pool, negBeta);
            var term1 = grad * poolNegBeta;

            // Step 3: term2 = -2*alpha*beta/size * x * ChannelWindowSum(grad * x * pool^(-(beta+1)))
            var negBetaM1 = TypedConst(-(effectiveBeta + 1.0f), x);
            Tensor<T> poolNegBetaM1 = OnnxOp.Pow(pool, negBetaM1);
            var inner = grad * x * poolNegBetaM1;
            var windowSumInner = ChannelWindowSum(inner);
            var coeff = TypedConst(-2.0f * effectiveAlpha * effectiveBeta / (float)effectiveSize, x);
            var term2 = coeff * x * windowSumInner;

            // gradX = term1 + term2
            var gradX = term1 + term2;

            return [gradX];
        }
    }
}
