
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo;
using Shorokoo.Core;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo
{
    /// <summary>
    /// Operators, or specif ways to use operators that are too nuanced to include in the main interface.
    /// </summary>
    public static class NN
    {
        /// <summary>Generates a 2D/3D sampling grid from batched affine matrices (ONNX AffineGrid).</summary>
        public static Tensor<T> AffineGrid<T>(Tensor<T> theta, Vector<int64> size, bool? alignCorners = false) where T : FloatLike
            => OnnxOp.AffineGrid(theta, size, alignCorners);

        /// <summary>Returns a Blackman window of the given size (ONNX BlackmanWindow).</summary>
        public static Vector<T> BlackmanWindow<T>(Scalar<int64> size, bool periodic = true) where T : NumLike
        {
            return ((Tensor<T>)OnnxOp.BlackmanWindow(size, OnnxUtils.GetDType<T>(), periodic)).Vec();
        }

        /// <summary>Returns a Blackman window of the given size (ONNX BlackmanWindow); int32-size overload.</summary>
        public static Vector<T> BlackmanWindow<T>(Scalar<int32> size, bool periodic = true) where T : NumLike
        {
            return ((Tensor<T>)OnnxOp.BlackmanWindow(size, OnnxUtils.GetDType<T>(), periodic)).Vec();
        }

        /// <summary>Rearranges sliding-block columns back into a batched image tensor (ONNX Col2Im).</summary>
        public static Tensor<T> Col2Im<T>(Tensor<T> input, Tensor<int64> imageShape, Tensor<int64> blockShape, long[] dilations, long[] pads, long[] strides) where T : NumLike
            => OnnxOp.Col2Im(input, imageShape, blockShape, dilations, pads, strides);

        /// <summary>Concatenates the tensors along the given axis (ONNX Concat).</summary>
        public static Tensor<T> Concat<T>(Tensor<T>[] inputs, long axis) where T : IVarType
            => OnnxOp.Concat([.. inputs.Select(x => (Variable)x)], axis);

        /// <summary>N-dimensional convolution (ONNX Conv) with geometry supplied as static attributes.</summary>
        public static Tensor<T> Conv<T>(Tensor<T> x, Tensor<T> w, Vector<T> b, AutoPad autoPad,
            long[] dilations, long group, long[] kernelShape, long[]? pads, long[] strides)
            where T : FloatLike
        {
            return OnnxOp.Conv(x, w, b, autoPad, dilations, group, kernelShape, pads, strides);
        }

        /// <summary>
        /// Conv overload whose geometry (dilations, group, kernel_shape, pads, strides) is supplied
        /// as int64 tensor inputs (computed in-graph) rather than static <c>long[]</c>/<c>long</c>
        /// attributes; the parameter order matches the static-attribute <c>Conv</c> overload above.
        /// Lowered to standard ONNX Conv before execution; the geometry tensors must be resolvable
        /// to constants at lowering time.
        /// </summary>
        public static Tensor<T> Conv<T>(Tensor<T> x, Tensor<T> w, Vector<T> b, AutoPad autoPad,
            Vector<int64> dilations, Scalar<int64> group, Vector<int64> kernelShape, Vector<int64> pads, Vector<int64> strides)
            where T : FloatLike
            => InternalOp.Conv(x, w, b, autoPad, pads, strides, dilations, kernelShape, group);

        /// <summary>Integer convolution with zero points, producing an int32 result (ONNX ConvInteger).</summary>
        public static Tensor<int32> ConvInteger<T1,T2>(Tensor<T1> x, Tensor<T2> w, Scalar<T1> xZeroPoint, Scalar<T2> wZeroPoint,
            AutoPad autoPad, long[] dilations, long group, long[] kernelShape, long[] pads, long[] strides)
            where T1 : Int8Like where T2 : Int8Like
            => OnnxOp.ConvInteger(x, w, xZeroPoint, wZeroPoint, autoPad, dilations, group, kernelShape, pads, strides);

        /// <summary>Transposed (fractionally strided) convolution (ONNX ConvTranspose).</summary>
        public static Tensor<T> ConvTranspose<T>(Tensor<T> x, Tensor<T> w, Vector<T> b, AutoPad autoPad, long[]? dilations, long group,
            long[]? kernelShape, long[]? outputPadding, long[]? outputShape, long[]? pads, long[]? strides)
            where T : FloatLike
            => OnnxOp.ConvTranspose(x, w, b, autoPad, dilations, group, kernelShape, outputPadding, outputShape, pads, strides);

        /// <summary>Discrete Fourier transform along one axis, optionally inverse and/or one-sided (ONNX DFT).</summary>
        public static Tensor<T1> Dft<T1, T2>(Tensor<T1> input, Scalar<T2> dft_length, Scalar<int64>? axis = null, bool inverse = false, bool onesided = false)
            where T1 : FloatLike
            where T2 : IndexLike
            => OnnxOp.Dft(input, dft_length, axis, inverse, onesided);

        /// <summary>Deformable convolution with learned sampling offsets and mask (ONNX DeformConv).</summary>
        public static Tensor<T> DeformConv<T>(Tensor<T> x, Tensor<T> w, Tensor<T> offset, Vector<T> b, Tensor<T> mask,
            long[] dilations, long group, long[] kernelShape, long offsetGroup, long[] pads, long[] strides)
            where T : FloatLike
            => OnnxOp.DeformConv(x, w, offset, b, mask, dilations, group, kernelShape, offsetGroup, pads, strides);
        
        /// <summary>Dequantizes integer data to float: <c>(x - xZeroPoint) * xScale</c> (ONNX DequantizeLinear).</summary>
        public static Tensor<TOut> DequantizeLinear<TIn, TOut>(Tensor<TIn> x, Tensor<TOut> xScale, Tensor<TIn>? xZeroPoint, long? axis = null, long? blockSize = null)
            where TIn : AnyIntLike
            where TOut : FloatLike
            => OnnxOp.DequantizeLinear(x, xScale, xZeroPoint, axis, blockSize);

        /// <summary>Determinant of the innermost 2-D square matrices (ONNX Det).</summary>
        public static Tensor<T> DeterminantMatrix<T>(Tensor<T> batchedMatrices)
            where T : FloatLike
            => OnnxOp.Det(batchedMatrices);

        /// <summary>Quantizes to uint8 with scale and zero point computed from the data (ONNX DynamicQuantizeLinear).</summary>
        public static (Scalar<uint8> y, Scalar<float32> y_scale, Scalar<uint8> y_zero_point) DynamicQuantizeLinear<T>(Scalar<T> x)
            where T : FloatLike
        {
            var (y, scale, zp) = OnnxOp.DynamicQuantizeLinear(x);
            return ((Variable)y, (Variable)scale, (Variable)zp);
        }

        /// <summary>Quantizes to uint8 with scale and zero point computed from the data (ONNX DynamicQuantizeLinear).</summary>
        public static (Vector<uint8> y, Scalar<float32> y_scale, Scalar<uint8> y_zero_point) DynamicQuantizeLinear<T>(Vector<T> x)
            where T : FloatLike
        {
            var (y, scale, zp) = OnnxOp.DynamicQuantizeLinear(x);
            return ((Variable)y, (Variable)scale, (Variable)zp);
        }

        /// <summary>Quantizes to uint8 with scale and zero point computed from the data (ONNX DynamicQuantizeLinear).</summary>
        public static (Tensor<uint8> y, Scalar<float32> y_scale, Scalar<uint8> y_zero_point) DynamicQuantizeLinear<T>(Tensor<T> x)
            where T : FloatLike
        {
            var (y, scale, zp) = OnnxOp.DynamicQuantizeLinear(x);
            return ((Variable)y, (Variable)scale, (Variable)zp);
        }

        /// <summary>Tensor of input's shape and dtype with ones on the k-th diagonal, zeros elsewhere (ONNX EyeLike).</summary>
        public static Tensor<T> EyeLike<T>(Tensor<T> input, long k = 0)
            where T : CommonLike
            => OnnxOp.EyeLike(input, null, k);

        /// <summary>EyeLike overload taking any tensor for the shape and producing element type <typeparamref name="T"/>.</summary>
        public static Tensor<T> EyeLike<T>(Variable input, long k = 0)
            where T : CommonLike
            => OnnxOp.EyeLike(input, OnnxUtils.GetDType<T>(), k);

        /// <summary>Averages each channel over all spatial dimensions (ONNX GlobalAveragePool).</summary>
        public static Scalar<T> GlobalAveragePool<T>(Scalar<T> input)
            where T : FloatLike
            => OnnxOp.GlobalAveragePool(input);

        /// <summary>Averages each channel over all spatial dimensions (ONNX GlobalAveragePool).</summary>
        public static Vector<T> GlobalAveragePool<T>(Vector<T> input)
            where T : FloatLike
            => OnnxOp.GlobalAveragePool(input);

        /// <summary>Averages each channel over all spatial dimensions (ONNX GlobalAveragePool).</summary>
        public static Tensor<T> GlobalAveragePool<T>(Tensor<T> input)
            where T : FloatLike
            => OnnxOp.GlobalAveragePool(input);

        /// <summary>Lp-norm of each channel over all spatial dimensions (ONNX GlobalLpPool).</summary>
        public static Scalar<T> GlobalLpPool<T>(Scalar<T> input, long p = 2)
            where T : FloatLike
            => OnnxOp.GlobalLpPool(input, p);

        /// <summary>Lp-norm of each channel over all spatial dimensions (ONNX GlobalLpPool).</summary>
        public static Vector<T> GlobalLpPool<T>(Vector<T> input, long p = 2)
            where T : FloatLike
            => OnnxOp.GlobalLpPool(input, p);

        /// <summary>Lp-norm of each channel over all spatial dimensions (ONNX GlobalLpPool).</summary>
        public static Tensor<T> GlobalLpPool<T>(Tensor<T> input, long p = 2)
            where T : FloatLike
            => OnnxOp.GlobalLpPool(input, p);

        /// <summary>Maximum of each channel over all spatial dimensions (ONNX GlobalMaxPool).</summary>
        public static Scalar<T> GlobalMaxPool<T>(Scalar<T> input)
            where T : FloatLike
            => OnnxOp.GlobalMaxPool(input);

        /// <summary>Maximum of each channel over all spatial dimensions (ONNX GlobalMaxPool).</summary>
        public static Vector<T> GlobalMaxPool<T>(Vector<T> input)
            where T : FloatLike
            => OnnxOp.GlobalMaxPool(input);

        /// <summary>Maximum of each channel over all spatial dimensions (ONNX GlobalMaxPool).</summary>
        public static Tensor<T> GlobalMaxPool<T>(Tensor<T> input)
            where T : FloatLike
            => OnnxOp.GlobalMaxPool(input);

        /// <summary>Samples the input at normalized grid coordinates with the given interpolation and padding (ONNX GridSample).</summary>
        public static Tensor<T1> GridSample<T1, T2>(Tensor<T1> input, Tensor<T2> grid, GridSampleMode mode = GridSampleMode.Linear, GridSamplePaddingMode paddingMode = GridSamplePaddingMode.Zeros, bool? alignCorners = false)
            where T1 : CommonLike
            where T2 : FloatLike
            => OnnxOp.GridSample(input, grid, alignCorners, mode, paddingMode);

        /// <summary>Normalizes over channel groups, then applies per-channel scale and bias (ONNX GroupNormalization).</summary>
        public static Tensor<T> GroupNormalization<T>(Tensor<T> x, Tensor<T> scale, Tensor<T> bias, long numGroups, long stashType = 1L, float epsilon = 1e-05f)
            where T : FloatLike
            => OnnxOp.GroupNormalization(x, scale, bias, epsilon, numGroups, stashType);

        /// <summary>Passes the variable through unchanged (ONNX Identity).</summary>
        public static T Identity<T>(T x) where T : IValue
        {
            var v = x.ToVariable();
            return OnnxOp.Identity(v, v.Rank).ToValue<T>();
        }

        /// <summary>Integer matrix product with zero points, producing an int32 result (ONNX MatMulInteger).</summary>
        public static Tensor<int32> MatMulInteger<T1, T2>(Tensor<T1> a, Tensor<T2> b, Tensor<T1> aZeroPoint, Tensor<T2> bZeroPoint)
            where T1 : Int8Like
            where T2 : Int8Like
            => OnnxOp.MatMulInteger(a, b, aZeroPoint, bZeroPoint);

        /// <summary>Element-wise maximum of the given tensors, with broadcasting (ONNX Max).</summary>
        public static Tensor<T> Max<T>(params Tensor<T>[] toMax)
            where T : NumLike
            => OnnxOp.Max([.. toMax.Select(x => (Variable)x)]);

        /// <summary>Max pooling over spatial windows (ONNX MaxPool).</summary>
        public static Tensor<T> MaxPool<T>(Tensor<T> x, bool ceilMode, long[]? dilations, long[]? kernelShape, long[]? pads, long storageOrder, long[]? strides, AutoPad autoPad = AutoPad.NotSet)
            where T : FloatLike
            => OnnxOp.MaxPool(x, autoPad, ceilMode, dilations, kernelShape, pads, storageOrder, strides);

        /// <summary>Max pooling that also returns the flattened indices of the selected elements (ONNX MaxPool, two outputs).</summary>
        public static (Tensor<T> result, Tensor<int64> indices) MaxPoolWithIndices<T>(Tensor<T> x, bool ceilMode, long[] kernelShape, long[] pads, long[] strides, AutoPad autoPad = AutoPad.NotSet)
            where T : FloatLike
        {
            (Variable result, Variable indices) = OnnxOp.MaxPoolWithIndices(x, autoPad, ceilMode,
                kernelShape: kernelShape, pads: pads, strides: strides);
            return ((Variable)result, (Variable)indices);
        }

        /// <summary>Element-wise minimum of the given tensors, with broadcasting (ONNX Min).</summary>
        public static Tensor<T> Min<T>(params Tensor<T>[] toMax)
            where T : NumLike
            => OnnxOp.Min([.. toMax.Select(x => (Variable)x)]);

        /// <summary>Element-wise integer remainder of a / b; fmod=true selects C-style fmod sign semantics (ONNX Mod).</summary>
        public static Tensor<T> Mod<T>(Tensor<T> a, Tensor<T> b, bool fmod = false)
            where T : IntLike
            => OnnxOp.Mod(a, b, fmod);

        /// <summary>Element-wise C-style fmod remainder of a / b (ONNX Mod with fmod=1), allowing float operands.</summary>
        public static Tensor<T> FMod<T>(Tensor<T> a, Tensor<T> b)
            where T : NumLike
            => OnnxOp.Mod(a, b, true);

        /// <summary>Greedily selects boxes by score, suppressing overlaps above the IoU threshold (ONNX NonMaxSuppression).</summary>
        public static Tensor<int64> NonMaxSuppression(Tensor<float32> boxes, Tensor<float32> scores, Tensor<int64>? maxBoxesPerClass, Tensor<float32>? iouThreashold = null, Tensor<float32>? scoresThreshold = null, bool? usesCenterBoxPoint = false)
            => OnnxOp.NonMaxSuppression(boxes, scores, maxBoxesPerClass, iouThreashold, scoresThreshold, usesCenterBoxPoint);

        /// <summary>Indices of the non-zero elements, one row per dimension (ONNX NonZero).</summary>
        public static Tensor<int64> NonZero<T>(Tensor<T> tensor)
            where T : SignedNumLike
            => OnnxOp.NonZero(tensor);

        /// <summary>Reduce overload taking the axes as a vector, without the noOp flag.</summary>
        public static Tensor<T> Reduce<T>(ReduceKind reduceKind, Tensor<T> tensor, Vector<int64>? axes, bool? keepDims)
            where T : IVarType
            => Reduce(reduceKind, tensor, (Tensor<int64>?)axes, keepDims, null);

        // Vector/Scalar input overloads: the value-struct handles no longer inherit from Tensor<T>,
        // so generic inference does not flow a Vector<T>/Scalar<T> argument into the Tensor<T>
        // parameter. These let emitted code (and callers) reduce a rank-1/rank-0 handle directly.
        /// <summary>Reduce a rank-1 <see cref="Vector{T}"/> (forwards to the tensor overload).</summary>
        public static Tensor<T> Reduce<T>(ReduceKind reduceKind, Vector<T> tensor, Tensor<int64>? axes, bool? keepDims, bool? noOp)
            where T : IVarType
            => Reduce(reduceKind, (Tensor<T>)tensor, axes, keepDims, noOp);

        /// <summary>Reduce a rank-0 <see cref="Scalar{T}"/> (forwards to the tensor overload).</summary>
        public static Tensor<T> Reduce<T>(ReduceKind reduceKind, Scalar<T> tensor, Tensor<int64>? axes, bool? keepDims, bool? noOp)
            where T : IVarType
            => Reduce(reduceKind, (Tensor<T>)tensor, axes, keepDims, noOp);

        /// <summary>
        /// Applies the reduction selected by <paramref name="reduceKind"/> along the given axes
        /// (dispatches to the corresponding ONNX Reduce* op); noOp makes empty axes a pass-through.
        /// </summary>
        public static Tensor<T> Reduce<T>(ReduceKind reduceKind, Tensor<T> tensor, Tensor<int64>? axes, bool? keepDims, bool? noOp)
            where T : IVarType
        {
            switch (reduceKind)
            {
                case ReduceKind.L1:
                    return OnnxOp.ReduceL1(tensor, axes, keepDims, noOp);
                case ReduceKind.L2:
                    return OnnxOp.ReduceL2(tensor, axes, keepDims, noOp);
                case ReduceKind.LogSum:
                    return OnnxOp.ReduceLogSum(tensor, axes, keepDims, noOp);
                case ReduceKind.LogSumExp:
                    return OnnxOp.ReduceLogSumExp(tensor, axes, keepDims, noOp);
                case ReduceKind.Max:
                    return OnnxOp.ReduceMax(tensor, axes, keepDims, noOp);
                case ReduceKind.Mean:
                    return OnnxOp.ReduceMean(tensor, axes, keepDims, noOp);
                case ReduceKind.Min:
                    return OnnxOp.ReduceMin(tensor, axes, keepDims, noOp);
                case ReduceKind.Prod:
                    return OnnxOp.ReduceProd(tensor, axes, keepDims, noOp);
                case ReduceKind.Sum:
                    return OnnxOp.ReduceSum(tensor, axes, keepDims, noOp);
                case ReduceKind.SumSquare:
                    return OnnxOp.ReduceSumSquare(tensor, axes, keepDims, noOp);
                default:
                    throw new InvalidTensorOperationException(ErrorCodes.CR004, "Reduce", reduceKind.ToString(), 
                        $"Reduction type '{reduceKind}' is not yet implemented");
            }
        }

        /// <summary>Resizes the tensor by per-axis scales or explicit output sizes (ONNX Resize).</summary>
        public static Tensor<T> Resize<T>(Tensor<T> x, Vector<float32>? scales,
        Vector<int64>? sizes, bool? antialias, long[]? axes,
        CoordinateTransformationMode? coordinateTransformationMode,
        float? cubicCoeffA, bool? excludeOutside,
        float? extrapolationValue, KeepAspectRatioPolicy? keepAspectRatioPolicy,
        ResizeMode? mode, NearestMode? nearestMode)
            where T : NumLike
            => OnnxOp.Resize(x, null, scales, sizes, antialias, axes,
                                coordinateTransformationMode, cubicCoeffA, excludeOutside,
                                extrapolationValue, keepAspectRatioPolicy, mode, nearestMode);

        /// <summary>Resize overload that also takes a region-of-interest input (ONNX Resize).</summary>
        public static Tensor<T1> Resize<T1, T2>(Tensor<T1> x, Vector<float32>? scales,
        Vector<int64>? sizes, bool? antialias, long[]? axes,
        CoordinateTransformationMode? coordinateTransformationMode,
        float? cubicCoeffA, bool? excludeOutside,
        float? extrapolationValue, KeepAspectRatioPolicy? keepAspectRatioPolicy,
        ResizeMode? mode, NearestMode? nearestMode, Vector<T2>? roi)
            where T1 : NumLike
            where T2 : FloatLike
            => OnnxOp.Resize(x, roi, scales, sizes, antialias, axes,
                                coordinateTransformationMode, cubicCoeffA, excludeOutside,
                                extrapolationValue, keepAspectRatioPolicy, mode, nearestMode);

        /// <summary>Top-k values and their indices along an axis (ONNX TopK); scalar-k overload.</summary>
        public static (Tensor<T> topK, Tensor<int64> indices) TopK<T>(Tensor<T> tensor, Scalar<int64> k, long? axis, bool? largest = null, bool? sorted = null)
            where T : IVarType
        {
            var (values, indices) = OnnxOp.TopK(tensor, k.Unsqueeze(), axis, largest, sorted);
            return ((Variable)values, (Variable)indices);
        }

        /// <summary>Top-k values and their indices along an axis (ONNX TopK); 1-element-vector-k overload.</summary>
        public static (Tensor<T> topK, Tensor<int64> indices) TopK<T>(Tensor<T> tensor, Vector<int64> k, long? axis, bool? largest = null, bool? sorted = null)
            where T : IVarType
        {
            var (values, indices) = OnnxOp.TopK(tensor, k, axis, largest, sorted);
            return ((Variable)values, (Variable)indices);
        }

        // -- New opset-21 operators ---------------------------------------------

        /// <summary>Returns a Hamming window of the given size (ONNX HammingWindow).</summary>
        public static Vector<T> HammingWindow<T>(Scalar<int64> size, bool periodic = true) where T : NumLike
            => ((Tensor<T>)OnnxOp.HammingWindow(size, OnnxUtils.GetDType<T>(), periodic)).Vec();

        /// <summary>Returns a Hamming window of the given size (ONNX HammingWindow); int32-size overload.</summary>
        public static Vector<T> HammingWindow<T>(Scalar<int32> size, bool periodic = true) where T : NumLike
            => ((Tensor<T>)OnnxOp.HammingWindow(size, OnnxUtils.GetDType<T>(), periodic)).Vec();

        /// <summary>Returns a Hann window of the given size (ONNX HannWindow).</summary>
        public static Vector<T> HannWindow<T>(Scalar<int64> size, bool periodic = true) where T : NumLike
            => ((Tensor<T>)OnnxOp.HannWindow(size, OnnxUtils.GetDType<T>(), periodic)).Vec();

        /// <summary>Returns a Hann window of the given size (ONNX HannWindow); int32-size overload.</summary>
        public static Vector<T> HannWindow<T>(Scalar<int32> size, bool periodic = true) where T : NumLike
            => ((Tensor<T>)OnnxOp.HannWindow(size, OnnxUtils.GetDType<T>(), periodic)).Vec();

        /// <summary>Decodes an encoded image byte stream (e.g. JPEG/PNG) into a uint8 pixel tensor (ONNX ImageDecoder).</summary>
        public static Tensor<uint8> ImageDecoder(Vector<uint8> encodedStream, string? pixelFormat = null)
            => OnnxOp.ImageDecoder(encodedStream, pixelFormat);

        /// <summary>Layer normalization, returning only the normalized output (ONNX LayerNormalization).</summary>
        public static Tensor<T> LayerNormalization<T>(Tensor<T> x, Tensor<T> scale, Tensor<T>? b = null,
            long? axis = null, float? epsilon = null, long? stashType = null)
            where T : FloatLike
            => OnnxOp.LayerNormalization(x, scale, b, axis, epsilon, stashType).y;

        /// <summary>Layer normalization that also returns the saved mean and inverse standard deviation.</summary>
        public static (Tensor<T> y, Tensor<T>? mean, Tensor<T>? invStdDev) LayerNormalizationFullOutputs<T>(
            Tensor<T> x, Tensor<T> scale, Tensor<T>? b = null,
            long? axis = null, float? epsilon = null, long? stashType = null)
            where T : FloatLike
        {
            var retval = OnnxOp.LayerNormalization(x, scale, b, axis, epsilon, stashType);
            return ((Variable)retval.y,
                    retval.mean is null ? null : (Tensor<T>?)retval.mean,
                    retval.invStdDev is null ? null : (Tensor<T>?)retval.invStdDev);
        }

        /// <summary>Builds the weight matrix mapping linear DFT bins to mel-frequency bins (ONNX MelWeightMatrix).</summary>
        public static Tensor<T> MelWeightMatrix<T, TInt, TFloat>(
            Scalar<TInt> numMelBins, Scalar<TInt> dftLength, Scalar<TInt> sampleRate,
            Scalar<TFloat> lowerEdgeHertz, Scalar<TFloat> upperEdgeHertz)
            where T : NumLike where TInt : IndexLike where TFloat : FloatLike
            => OnnxOp.MelWeightMatrix(
                numMelBins, dftLength, sampleRate, lowerEdgeHertz, upperEdgeHertz,
                OnnxUtils.GetDType<T>());

        /// <summary>Samples class indices from per-row unnormalized log-probabilities (ONNX Multinomial).</summary>
        public static Tensor<TOut> Multinomial<TIn, TOut>(Tensor<TIn> input, long? sampleSize = null, float? seed = null)
            where TIn : FloatLike where TOut : IndexLike
            => OnnxOp.Multinomial(input, OnnxUtils.GetDType<TOut>(), sampleSize, seed);

        /// <summary>Negative log-likelihood loss over class scores and target indices (ONNX NegativeLogLikelihoodLoss).</summary>
        public static Tensor<T> NegativeLogLikelihoodLoss<T, TInd>(
            Tensor<T> input, Tensor<TInd> target, Tensor<T>? weight = null,
            long? ignoreIndex = null, string? reduction = null)
            where T : FloatLike where TInd : IndexLike
            => OnnxOp.NegativeLogLikelihoodLoss(input, target, weight, ignoreIndex, reduction);

        /// <summary>One-hot encodes indices to the given depth using [off, on] values (ONNX OneHot).</summary>
        public static Tensor<T> OneHot<TInd, TDepth, T>(Tensor<TInd> indices, Scalar<TDepth> depth, Vector<T> values, long? axis = null)
            where TInd : NumLike where TDepth : NumLike where T : IVarType
            => OnnxOp.OneHot(indices, depth, values, axis);

        /// <summary>Parametric ReLU: x where positive, slope * x where negative (ONNX PRelu).</summary>
        public static Tensor<T> PRelu<T>(Tensor<T> x, Tensor<T> slope) where T : FloatLike
            => OnnxOp.PRelu(x, slope);

        /// <summary>Quantizes float data to integers using scale and optional zero point (ONNX QuantizeLinear).</summary>
        public static Tensor<TOut> QuantizeLinear<TIn, TOut>(
            Tensor<TIn> x, Tensor<TIn> yScale, Tensor<TOut>? yZeroPoint = null,
            long? axis = null, long? blockSize = null,
            bool? saturate = null, long? precision = null)
            where TIn : FloatLike where TOut : AnyIntLike
            => OnnxOp.QuantizeLinear(x, yScale, yZeroPoint, axis, blockSize,
                OnnxUtils.GetDType<TOut>(), saturate, precision);

        /// <summary>Quantized matrix product of int8-like operands with per-operand scale/zero point (ONNX QLinearMatMul).</summary>
        public static Tensor<TOut> QLinearMatMul<TA, TB, TOut, TScale>(
            Tensor<TA> a, Scalar<TScale> aScale, Scalar<TA> aZeroPoint,
            Tensor<TB> b, Scalar<TScale> bScale, Scalar<TB> bZeroPoint,
            Scalar<TScale> yScale, Scalar<TOut> yZeroPoint)
            where TA : Int8Like where TB : Int8Like where TOut : Int8Like where TScale : FloatLike
            => OnnxOp.QLinearMatMul(
                a, aScale, aZeroPoint, b, bScale, bZeroPoint, yScale, yZeroPoint);

        /// <summary>Quantized convolution of int8-like operands with per-operand scale/zero point (ONNX QLinearConv).</summary>
        public static Tensor<TOut> QLinearConv<TIn, TFilt, TOut>(
            Tensor<TIn> x, Scalar<float32> xScale, Scalar<TIn> xZeroPoint,
            Tensor<TFilt> w, Scalar<float32> wScale, Scalar<TFilt> wZeroPoint,
            Scalar<float32> yScale, Scalar<TOut> yZeroPoint, Vector<int32>? b = null,
            AutoPad? autoPad = null, long[]? dilations = null, long? group = null,
            long[]? kernelShape = null, long[]? pads = null, long[]? strides = null)
            where TIn : Int8Like where TFilt : Int8Like where TOut : Int8Like
            => OnnxOp.QLinearConv(
                x, xScale, xZeroPoint, w, wScale, wZeroPoint, yScale, yZeroPoint, b,
                autoPad, dilations, group, kernelShape, pads, strides);

        /// <summary>Whether each string element fully matches the regex pattern (ONNX RegexFullMatch).</summary>
        public static Tensor<bit> RegexFullMatch(Tensor<@string> x, string? pattern = null)
            => OnnxOp.RegexFullMatch(x, pattern);

        /// <summary>Softmax cross-entropy loss over scores and label indices, optionally returning the log-probabilities (ONNX SoftmaxCrossEntropyLoss).</summary>
        public static (Tensor<T> output, Tensor<T>? logProb) SoftmaxCrossEntropyLoss<T, TInd>(
            Tensor<T> scores, Tensor<TInd> labels, Tensor<T>? weights = null,
            long? ignoreIndex = null, string? reduction = null)
            where T : FloatLike where TInd : IndexLike
        {
            var retval = OnnxOp.SoftmaxCrossEntropyLoss(scores, labels, weights, ignoreIndex, reduction);
            return ((Variable)retval.output, retval.logProb is null ? null : (Tensor<T>?)retval.logProb);
        }

        /// <summary>Splits a tensor along an axis into a sequence of tensors (ONNX SplitToSequence).</summary>
        public static Variable SplitToSequence<T>(Tensor<T> input, Vector<int64>? split = null, long? axis = null, long? keepdims = null)
            where T : IVarType
            => OnnxOp.SplitToSequence(input, split, axis, keepdims);

        /// <summary>Short-time Fourier transform of the signal with the given frame step and optional window (ONNX STFT).</summary>
        public static Tensor<T> STFT<T>(Tensor<T> signal, Scalar<int64> frameStep, Vector<T>? window = null,
            Scalar<int64>? frameLength = null, bool? onesided = null)
            where T : FloatLike
            => OnnxOp.STFT(signal, frameStep, window, frameLength, onesided);

        /// <summary>Element-wise string concatenation (ONNX StringConcat).</summary>
        public static Tensor<@string> StringConcat(Tensor<@string> x, Tensor<@string> y)
            => OnnxOp.StringConcat(x, y);

        /// <summary>Case normalization and stopword removal on string elements (ONNX StringNormalizer).</summary>
        public static Tensor<@string> StringNormalizer(Tensor<@string> x,
            string? caseChangeAction = null, bool? isCaseSensitive = null,
            string? locale = null, string[]? stopwords = null)
            => OnnxOp.StringNormalizer(
                x, caseChangeAction,
                isCaseSensitive is null ? null : (isCaseSensitive.Value ? 1L : 0L),
                locale, stopwords);

        /// <summary>Splits each string by a delimiter, returning the parts and per-element split counts (ONNX StringSplit).</summary>
        public static (Tensor<@string> y, Tensor<int64> numSplits) StringSplit(Tensor<@string> x, string? delimiter = null, long? maxsplit = null)
        {
            var retval = OnnxOp.StringSplit(x, delimiter, maxsplit);
            return ((Variable)retval.y, (Variable)retval.numSplits);
        }

        /// <summary>Extracts n-gram TF/IDF/TFIDF features from the input sequence (ONNX TfIdfVectorizer).</summary>
        public static Tensor<float32> TfIdfVectorizer<T>(Tensor<T> x,
            long? maxGramLength = null, long? maxSkipCount = null, long? minGramLength = null,
            string? mode = null, long[]? ngramCounts = null, long[]? ngramIndexes = null,
            long[]? poolInt64s = null, string[]? poolStrings = null, float[]? weights = null)
            where T : NumLike
            => OnnxOp.TfIdfVectorizer(x,
                maxGramLength, maxSkipCount, minGramLength, mode,
                ngramCounts, ngramIndexes, poolInt64s, poolStrings, weights);

        // -- Post-opset-21 operators (the exporter raises the model opset stamp
        //    per-graph; see FastOpsetResolver.RaiseToRequired) -------------------

        /// <summary>Scaled dot-product attention returning Y only (ONNX Attention, opset 23+).</summary>
        public static Tensor<T> Attention<T>(Tensor<T> q, Tensor<T> k, Tensor<T> v,
            Variable? attnMask = null, bool? isCausal = null,
            long? kvNumHeads = null, long? qNumHeads = null, float? scale = null, float? softcap = null)
            where T : FloatLike
            => OnnxOp.Attention(q, k, v, attnMask, nonpadKvSeqlen: null,
                isCausal: isCausal, kvNumHeads: kvNumHeads, qNumHeads: qNumHeads,
                qkMatmulOutputMode: null, scale: scale, softcap: softcap, softmaxPrecision: null);

        /// <summary>Scaled dot-product attention with KV-cache update: (Y, present_key, present_value) (ONNX Attention, opset 23+).</summary>
        public static (Tensor<T> y, Tensor<T> presentKey, Tensor<T> presentValue) AttentionWithKVCache<T>(
            Tensor<T> q, Tensor<T> k, Tensor<T> v,
            Tensor<T> pastKey, Tensor<T> pastValue, Variable? attnMask = null,
            bool? isCausal = null, long? kvNumHeads = null, long? qNumHeads = null,
            float? scale = null, float? softcap = null)
            where T : FloatLike
        {
            var retval = OnnxOp.AttentionWithKVCache(q, k, v, attnMask, pastKey, pastValue,
                isCausal: isCausal, kvNumHeads: kvNumHeads, qNumHeads: qNumHeads,
                qkMatmulOutputMode: null, scale: scale, softcap: softcap, softmaxPrecision: null);
            return ((Variable)retval.y, (Variable)retval.presentKey, (Variable)retval.presentValue);
        }

        /// <summary>Root-mean-square layer normalization over the suffix axes from <paramref name="axis"/> (ONNX RMSNormalization, opset 23+).</summary>
        public static Tensor<T> RMSNormalization<T>(Tensor<T> x, Tensor<T> scale,
            long? axis = null, float? epsilon = null, long? stashType = null)
            where T : FloatLike
            => OnnxOp.RMSNormalization(x, scale, axis, epsilon, stashType);

        /// <summary>Rotary positional embedding (ONNX RotaryEmbedding, opset 23+); output has x's shape.</summary>
        public static Tensor<T> RotaryEmbedding<T>(Tensor<T> x, Tensor<T> cosCache, Tensor<T> sinCache,
            Tensor<int64>? positionIds = null, bool? interleaved = null, long? numHeads = null,
            long? rotaryEmbeddingDim = null)
            where T : FloatLike
            => OnnxOp.RotaryEmbedding(x, cosCache, sinCache, positionIds,
                interleaved, numHeads, rotaryEmbeddingDim);

        /// <summary>Swish activation y = x * sigmoid(alpha * x) (ONNX Swish, opset 24+; no ORT 1.26 kernel — QEE-only execution).</summary>
        public static Tensor<T> Swish<T>(Tensor<T> x, float? alpha = null)
            where T : FloatLike
            => OnnxOp.Swish(x, alpha);
    }
}
