using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== DFT (variadic registration) =====
        //
        // Forward: DFT(X, dft_length?, axis?) → Y
        //   Computes the Discrete Fourier Transform of X along the specified axis.
        //   inverse=false (forward): Y[k] = Σ_n X[n] · exp(-2πi·n·k/N)
        //   inverse=true  (inverse): Y[n] = (1/N) · Σ_k X[k] · exp(2πi·n·k/N)
        //   Input/output last dimension is 1 (real) or 2 (complex: [real, imag]).
        //
        // Gradient (full, non-onesided case):
        //   The DFT matrix F has entries F[k,n] = exp(-2πi·n·k/N).
        //   The conjugate-transpose F^H = N · IDFT_matrix.
        //
        //   Forward DFT gradient: dL/dX = F^H · dL/dY = N · IDFT(dL/dY)
        //     → dX = N · DFT(dY, inverse=true)
        //
        //   Inverse DFT gradient: dL/dX = (1/N) · F · dL/dY
        //     → dX = (1/N) · DFT(dY, inverse=false)
        //
        //   If input was real (last dim=1), only the real part of the gradient
        //   is kept by slicing the last axis to match the input shape.
        //
        // onesided=1 (real-input RFFT, forward only — the spec forbids onesided with
        // inverse=1): the forward keeps only bins k = 0..K-1 with K = floor(N/2)+1.
        //   The op is then simply the full DFT followed by truncation, so its adjoint is
        //   ZERO-PADDING the upstream gradient back to N bins followed by the full-DFT
        //   adjoint above (NOT conjugate mirroring — the dropped bins are not outputs and
        //   receive no gradient; this matches PyTorch's rfft backward).
        //
        //   dL/d(dft_length) = null (int64, not differentiable)
        //   dL/d(axis) = null (int64, not differentiable)

        internal static Variable?[] DftGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var x = inputs[0]!;                                          // input tensor [..., N, 1 or 2]
            var dftLength = inputs.Length > 1 ? inputs[1] : null;        // optional dft_length (int64 scalar)
            var axisInput = inputs.Length > 2 ? inputs[2] : null;        // optional axis (int64 scalar)
            var grad = outputGrads[0]!;                                  // output gradient [..., N or K, 2]

            var inverseRaw = attributes.GetAttributeObj(AttrInverse);
            var inverse = inverseRaw is true || (inverseRaw is long lv && lv != 0);
            var onesidedRaw = attributes.GetAttributeObj(AttrOnesided);
            var onesided = onesidedRaw is true || (onesidedRaw is long lo && lo != 0);

            // OnnxOp.Dft substitutes Scalar(-2L) (the ONNX-spec default of rank-2) for a
            // null axis before the node is constructed, so by the time the gradient runs
            // axisInput is always non-null. The variadic gradient dispatcher synthesizes
            // float32 stand-ins for every input slot, so cast the int64 axis/dft_length
            // scalars back explicitly before they feed integer ops.
            System.Diagnostics.Debug.Assert(axisInput is not null,
                "OnnxOp.Dft fills in a default axis when null; axisInput should never be null here.");
            var effectiveAxis = OnnxOp.Cast(axisInput!, saturate: null, to: DType.Int64);

            // Determine N (the DFT length) for scaling and onesided zero-padding.
            // N is the size of the input along the transform axis.
            Variable nScalar;
            if (dftLength is not null)
            {
                // dft_length input is a scalar — use it directly
                nScalar = OnnxOp.Cast(dftLength, saturate: null, to: DType.Int64);
            }
            else
            {
                nScalar = OnnxOp.Gather(OnnxOp.Shape(x), effectiveAxis, axis: 0);
            }

            // onesided=1: zero-pad the K = floor(N/2)+1 gradient bins back to the full N
            // bins along the transform axis (the truncated bins are not outputs, so their
            // adjoint contribution is zero). Pad accepts runtime pads/axes inputs, which
            // keeps the (possibly negative) axis fully dynamic.
            var effectiveGrad = grad;
            if (onesided)
            {
                // The spec forbids inverse=1 with onesided=1, so this is the RFFT adjoint.
                System.Diagnostics.Debug.Assert(!inverse,
                    "ONNX DFT forbids onesided=1 with inverse=1.");
                var kScalar = OnnxOp.Gather(OnnxOp.Shape(grad), effectiveAxis, axis: 0);
                var padAmount = OnnxOp.Reshape(OnnxOp.Sub(nScalar, kScalar), Vector(1L), allowZero: false);
                var pads = OnnxOp.Concat([Vector(0L), padAmount], axis: 0);          // [begin, end]
                var padAxes = OnnxOp.Reshape(effectiveAxis, Vector(1L), allowZero: false);
                effectiveGrad = OnnxOp.Pad(grad, pads, constantValue: null, axes: padAxes);
            }

            // Apply the conjugate-transpose operation by swapping the inverse flag.
            // Do not pass dft_length to the backward DFT because the (padded) gradient
            // already has the correct length along the transform axis.
            var gradDft = OnnxOp.Dft(effectiveGrad, null, effectiveAxis, inverse: !inverse, onesided: false);

            // Cast N to the gradient's float type for multiplication
            var nFloat = OnnxOp.Cast(nScalar, saturate: null, to: grad.Type);

            Variable scaled;
            if (!inverse)
            {
                // Forward DFT: dX = N · IDFT(dY)
                scaled = OnnxOp.Mul(gradDft, nFloat);
            }
            else
            {
                // Inverse DFT: dX = (1/N) · DFT(dY)
                scaled = OnnxOp.Div(gradDft, nFloat);
            }

            // If the input was real (last dim = 1), the gradient from the inverse/forward
            // DFT will be complex (last dim = 2). Slice to keep only the real component.
            var xShape = OnnxOp.Shape(x);
            var xRank = OnnxOp.Squeeze(OnnxOp.Shape(xShape), Vector(0L));
            var lastDimIdx = OnnxOp.Sub(xRank, Scalar(1L));
            var xLastDim = OnnxOp.Gather(xShape, lastDimIdx, axis: 0);
            var sliceEnd = OnnxOp.Reshape(xLastDim, Vector(1L), allowZero: false);
            var gradX = OnnxOp.Slice(scaled, Vector(0L), sliceEnd, Vector(-1L), null);

            // Return: gradient for X, null for dft_length and axis (int64, not differentiable)
            return [gradX, null, null];
        }
    }
}
