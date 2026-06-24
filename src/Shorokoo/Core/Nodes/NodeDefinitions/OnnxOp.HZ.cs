using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using System.Collections.Immutable;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static partial class OnnxOp
{
    public static IVariable Identity(IVariable x, int? rank)
        => NodeBuilder.BuildNodeSingleOut(IDENTITY, [x], [(InternalAttrRank, (long?)rank)]);

    public static Node LoopOpen(IVariable? maxNumIterations, IVariable? condition, IVariable?[] loopVariableInitializers)
        => NodeBuilder.BuildNode(LOOP_OPEN, [maxNumIterations, condition, .. loopVariableInitializers], []);

    public static Node LoopClose(IVariable? continueWhile, IVariable[] loopVariableUpdaters, IVariable[] scanVariableInputs, Node openNode)
        => NodeBuilder.BuildNode(LOOP_CLOSE, [], [(AttrBody, (IVariable?[])[continueWhile, .. loopVariableUpdaters, .. scanVariableInputs])], openNode: openNode);

    public static Node IfOpen(IVariable condition)
        => NodeBuilder.BuildNode(IF_OPEN, [condition], []);

    public static IVariable[] IfClose(IVariable[] thenBranch, IVariable[] elseBranch, Node openNode)
     => NodeBuilder.BuildNodeMultiOut(IF_CLOSE, [], [(AttrThenBranch, thenBranch), (AttrElseBranch, elseBranch)], openNode: openNode);

    public static IVariable LeakyRelu(IVariable x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(LEAKY_RELU, [x], [(AttrAlpha, alpha)]);

    public static IVariable Less(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(LESS, [left, right], []);

    public static IVariable LessOrEqual(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(LESS_OR_EQUAL, [left, right], []);

    public static IVariable LoopIndexVariable()
        => NodeBuilder.BuildNodeSingleOut(LOOP_INDEX_VARIABLE, [], []);

    public static IVariable LoopFakeInput(DType type, int? rank, DataStructure structure)
        => NodeBuilder.BuildNodeSingleOut(LOOP_FAKE_INPUT, [], [(AttrDtype, type), (InternalAttrRank, (long?)rank), (InternalAttrStructure, structure)]);

    public static IVariable LoopScanZombie(IVariable scannee)
        => NodeBuilder.BuildNodeSingleOut(LOOP_SCAN_VARIABLE, [scannee], []);
    
    public static IVariable Log(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(LOG, [x], []);

    public static IVariable LpPool(IVariable x, AutoPad? autoPad, bool? ceilMode,
        long[]? dilations, long[] kernelShape, long? p, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(LP_POOL, [x], [
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrP, p),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IVariable MatMul(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(MATMUL, [left, right], []);

    public static IVariable MatMulInteger(IVariable a, IVariable b, IVariable? aZeroPoint = null, IVariable? bZeroPoint = null)
        => NodeBuilder.BuildNodeSingleOut(MATMUL_INTEGER, [a, b, aZeroPoint, bZeroPoint], []);

    public static IVariable Max(params IVariable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MAX, inputs, []);

    public static IVariable Mean(params IVariable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MEAN, inputs, []);

    public static IVariable MaxPool(IVariable x, AutoPad? autoPad = null, bool? ceilMode = null,
        long[]? dilations = null, long[]? kernelShape = null, long[]? pads = null,
        long? storageOrder = null, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(MAX_POOL, [x], [
            (InternalAttrHasOptionalOutputs, false),
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStorageOrder, storageOrder),
            (AttrStrides, strides)]);

    public static (IVariable y, IVariable indices) MaxPoolWithIndices(IVariable x, AutoPad? autoPad = null, bool? ceilMode = null,
        long[]? dilations = null, long[]? kernelShape = null, long[]? pads = null,
        long? storageOrder = null, long[]? strides = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(MAX_POOL, [x], [
            (InternalAttrHasOptionalOutputs, true),
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStorageOrder, storageOrder),
            (AttrStrides, strides)]);
        return (retval[0], retval[1]);
    }

    public static IVariable MaxUnpool(IVariable x, IVariable indices,
        long[] kernelShape, long[]? pads = null, long[]? strides = null, IVariable? outputShape = null)
        // The optional output_shape input is only emitted when present — ORT's MaxUnpool
        // kernel rejects a trailing empty-name optional input ("input count mismatch").
        => NodeBuilder.BuildNodeSingleOut(MAX_UNPOOL,
            outputShape is null ? [x, indices] : [x, indices, outputShape], [
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IVariable MaxRoiPool(IVariable x, IVariable rois,
        long[] pooledShape, float? spatialScale = null)
        => NodeBuilder.BuildNodeSingleOut(MAX_ROI_POOL, [x, rois], [
            (AttrPooledShape, pooledShape),
            (AttrSpatialScale, spatialScale)]);

    public static IVariable Min(params IVariable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MIN, inputs, []);

    public static IVariable Mod(IVariable a, IVariable b, bool? fmod = null)
        => NodeBuilder.BuildNodeSingleOut(MOD, [a, b], [(AttrFmod, fmod)]);

    public static IVariable Mul(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(MUL, [left, right], []);

    public static IVariable Neg(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(NEG, [x], []);

    public static IVariable NonMaxSuppression(IVariable boxes, IVariable scores,
        IVariable? maxOutputBoxesPerClass = null, IVariable? iouThreshold = null,
        IVariable? scoreThreshold = null, bool? centerPointBox = null)
        => NodeBuilder.BuildNodeSingleOut(NON_MAX_SUPPRESSION, 
            [boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold],
            [(AttrCenterPointBox, centerPointBox)]);

    public static IVariable NonZero(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(NON_ZERO, [x], []);

    public static IVariable Not(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(NOT, [x], []);

    public static IVariable Optional(IVariable? x, DataStructure structure, DType type)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL, x is null ? [] : [x], [(AttrType, (structure, type))]);

    public static IVariable OptionalGetElement(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL_GET_ELEMENT, [x], []);

    public static IVariable OptionalHasElement(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL_HAS_ELEMENT, [x], []);

    public static IVariable Or(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(OR, [left, right], []);

    public static IVariable Pad(IVariable data, IVariable pads, IVariable? constantValue = null,
        IVariable? axes = null, PadMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(PAD, [data, pads, constantValue, axes],
            [(AttrMode, mode)]);

    public static IVariable Pow(IVariable x, IVariable y)
        => NodeBuilder.BuildNodeSingleOut(POW, [x, y], []);

    public static IVariable Range(IVariable start, IVariable limit, IVariable delta)
        => NodeBuilder.BuildNodeSingleOut(RANGE, [start, limit, delta], []);

    public static IVariable RandomNormal(long[] shape, float? mean = null, float? scale = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_NORMAL, [], [
            (AttrShape, shape), (AttrDtype, dtype), (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    public static IVariable RandomNormalLike(IVariable input, float? mean = null, float? scale = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_NORMAL_LIKE, [input], [
            (AttrDtype, dtype), (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    public static IVariable RandomUniform(long[] shape, float? high = null, float? low = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_UNIFORM, [], [
            (AttrShape, shape), (AttrDtype, dtype), (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    public static IVariable RandomUniformLike(IVariable input, float? high = null, float? low = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_UNIFORM_LIKE, [input], [
            (AttrDtype, dtype), (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    public static IVariable Reciprocal(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(RECIPROCAL, [x], []);

    public static IVariable ReduceL1(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_L1, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceL2(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_L2, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceLogSum(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_LOG_SUM, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceLogSumExp(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_LOG_SUM_EXP, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceMax(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MAX, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceMean(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MEAN, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceMin(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MIN, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceProd(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_PROD, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceSum(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_SUM, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IVariable ReduceSumSquare(IVariable data, IVariable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_SUM_SQUARE, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static (IVariable y, IVariable yH) Rnn(IVariable x, IVariable w, IVariable r,
        IVariable? b, IVariable? sequenceLens, IVariable? initialH,
        float[]? activationAlpha, float[]? activationBeta, string[]? activations,
        float? clip, RNNDirection? direction, long? hiddenSize,
        bool? layout)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(RNN, [x, w, r, b, sequenceLens, initialH], [
            (AttrActivationAlpha, activationAlpha),
            (AttrActivationBeta, activationBeta),
            (AttrActivations, activations),
            (AttrClip, clip),
            (AttrDirection, direction),
            (AttrHiddenSize, hiddenSize),
            (AttrLayout, layout)]);
        return (retval[0], retval[1]);
    }

    public static IVariable Relu(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(RELU, [x], []);

    public static IVariable Reshape(IVariable data, IVariable shape, bool allowZero)
        => NodeBuilder.BuildNodeSingleOut(RESHAPE, [data, shape], [(AttrAllowzero, allowZero)]);

    public static IVariable ReverseSequence(IVariable input, IVariable sequenceLens, long? batchAxis = null, long? timeAxis = null)
        => NodeBuilder.BuildNodeSingleOut(REVERSE_SEQUENCE, [input, sequenceLens], [(AttrBatchAxis, batchAxis), (AttrTimeAxis, timeAxis)]);

    public static IVariable Resize(IVariable x, IVariable? roi, IVariable? scales,
        IVariable? sizes, bool? antialias, long[]? axes,
        CoordinateTransformationMode? coordinateTransformationMode,
        float? cubicCoeffA, bool? excludeOutside,
        float? extrapolationValue, KeepAspectRatioPolicy? keepAspectRatioPolicy,
        ResizeMode? mode, NearestMode? nearestMode)
        => NodeBuilder.BuildNodeSingleOut(RESIZE, [x, roi, scales, sizes], [
            (AttrAntialias, antialias),
            (AttrAxes, axes),
            (AttrCoordinateTransformationMode, coordinateTransformationMode),
            (AttrCubicCoeffA, cubicCoeffA),
            (AttrExcludeOutside, excludeOutside),
            (AttrExtrapolationValue, extrapolationValue),
            (AttrKeepAspectRatioPolicy, keepAspectRatioPolicy),
            (AttrMode, mode),
            (AttrNearestMode, nearestMode)]);

    public static IVariable RoiAlign(IVariable x, IVariable rois, IVariable batchIndices,
        RoiAlignTransformationMode? coordinateTransformationMode = null,
        RoiAlignMode? mode = null, long? outputHeight = null, long? outputWidth = null,
        long? samplingRatio = null, float? spatialScale = null)
        => NodeBuilder.BuildNodeSingleOut(ROI_ALIGN, [x, rois, batchIndices], [
            (AttrCoordinateTransformationMode, coordinateTransformationMode),
            (AttrMode, mode),
            (AttrOutputHeight, outputHeight),
            (AttrOutputWidth, outputWidth),
            (AttrSamplingRatio, samplingRatio),
            (AttrSpatialScale, spatialScale)]);

    public static IVariable ScatterElements(IVariable data, IVariable indices, IVariable updates,
        long? axis = null, ScatterNDReduction? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(SCATTER_ELEMENTS, [data, indices, updates],
            [(AttrAxis, axis), (AttrReduction, reduction)]);

    public static IVariable ScatterND(IVariable data, IVariable indices, IVariable updates,
        ScatterNDReduction? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(SCATTER_ND, [data, indices, updates],
            [(AttrReduction, reduction)]);

    public static IVariable Selu(IVariable x, float? alpha = null, float? gamma = null)
        => NodeBuilder.BuildNodeSingleOut(SELU, [x], [(AttrAlpha, alpha), (AttrGamma, gamma)]);

    public static IVariable SequenceAt(IVariable input, IVariable position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_AT, [input, position], []);

    public static IVariable SequenceConstruct(params IVariable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONSTRUCT, inputs, []);

    internal static IVariable SequenceConstruct(Function targetFunction, params IVariable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONSTRUCT, inputs, [], targetFunction: targetFunction);

    public static IVariable SequenceEmpty(DType type, Function? targetFunction = null)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_EMPTY, [], [(AttrDtype, type)], targetFunction: targetFunction);

    public static IVariable SequenceErase(IVariable input, IVariable? position = null)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_ERASE, [input, position], []);

    public static IVariable SequenceErase(Function targetFunction, IVariable input, IVariable? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_ERASE, [input, position], [], targetFunction: targetFunction);

    public static IVariable SequenceInsert(IVariable input, IVariable tensor, IVariable? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_INSERT, [input, tensor, position], []);

    public static IVariable SequenceInsert(Function targetFunction, IVariable input, IVariable tensor, IVariable? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_INSERT, [input, tensor, position], [], targetFunction: targetFunction);

    public static IVariable SequenceLength(IVariable input)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_LENGTH, [input], []);

    public static IVariable Shape(IVariable data, long? end = null, long? start = null)
        => NodeBuilder.BuildNodeSingleOut(SHAPE, [data], [(AttrEnd, end), (AttrStart, start)]);

    public static IVariable Sigmoid(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(SIGMOID, [x], []);

    public static IVariable Sign(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(SIGN, [x], []);

    public static IVariable Sin(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(SIN, [num], []);

    public static IVariable Sinh(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(SINH, [num], []);

    public static IVariable Slice(IVariable data, IVariable starts, IVariable ends,
        IVariable? axes = null, IVariable? steps = null)
        => NodeBuilder.BuildNodeSingleOut(SLICE, [data, starts, ends, axes, steps], []);

    public static IVariable Softmax(IVariable input, long? axis)
        => NodeBuilder.BuildNodeSingleOut(SOFTMAX, [input], [(AttrAxis, axis)]);

    public static IVariable[] Split(IVariable data, IVariable? splits, long? axis, long? numOutputs, long variadicOutputCount)
        => NodeBuilder.BuildNodeMultiOut(SPLIT, [data, splits], [(AttrAxis, axis), (AttrNumOutputs, numOutputs)],
            outputNames: Enumerable.Repeat((string?)null, (int)variadicOutputCount).ToArray());

    public static IVariable Sqrt(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(SQRT, [num], []);

    public static IVariable Squeeze(IVariable data, IVariable? axes)
        => NodeBuilder.BuildNodeSingleOut(SQUEEZE, [data, axes], []);

    public static IVariable Sub(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(SUB, [left, right], []);

    public static IVariable Sum(params IVariable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SUM, inputs, []);

    public static IVariable Tan(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(TAN, [x], []);

    public static IVariable Tanh(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(TANH, [x], []);

    public static IVariable Tile(IVariable input, IVariable repeats)
        => NodeBuilder.BuildNodeSingleOut(TILE, [input, repeats], []);

    public static (IVariable values, IVariable indices) TopK(IVariable x, IVariable k,
        long? axis = null, bool? largest = null, bool? sorted = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(TOPK, [x, k], [
            (AttrAxis, axis),
            (AttrLargest, largest),
            (AttrSorted, sorted)]);
        return (retval[0], retval[1]);
    }

    public static IVariable Transpose(IVariable data, long[]? perm = null)
        => NodeBuilder.BuildNodeSingleOut(TRANSPOSE, [data], [(AttrPerm, perm)]);

    public static (IVariable y, IVariable indices, IVariable inverseIndices, IVariable counts) Unique(
        IVariable x, long? axis = null, bool? sorted = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(UNIQUE, [x], [
            (AttrAxis, axis),
            (AttrSorted, sorted)]);
        return (retval[0], retval[1], retval[2], retval[3]);
    }

    public static IVariable Unsqueeze(IVariable data, IVariable axes)
        => NodeBuilder.BuildNodeSingleOut(UNSQUEEZE, [data, axes], []);

    public static IVariable Where(IVariable condition, IVariable x, IVariable y)
        => NodeBuilder.BuildNodeSingleOut(WHERE, [condition, x, y], []);

    public static IVariable Xor(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(XOR, [left, right], []);

    public static IVariable SpaceToDepth(IVariable input, long? blockSize)
        => NodeBuilder.BuildNodeSingleOut(SPACE_TO_DEPTH, [input], [(AttrBlocksize, blockSize)]);

    public static IVariable Trilu(IVariable input, IVariable? k = null, long? upper = null)
        => NodeBuilder.BuildNodeSingleOut(TRILU, [input, k], [(AttrUpper, upper)]);

    public static IVariable LpNormalization(IVariable input, long? axis = null, long? p = null)
        => NodeBuilder.BuildNodeSingleOut(LP_NORMALIZATION, [input], [(AttrAxis, axis), (AttrP, p)]);

    public static (IVariable y, IVariable yH, IVariable yC) Lstm(IVariable x, IVariable w, IVariable r,
        IVariable? b, IVariable? sequenceLens, IVariable? initialH, IVariable? initialC, IVariable? p,
        float[]? activationAlpha, float[]? activationBeta, string[]? activations,
        float? clip, LSTMDirection? direction, long? hiddenSize,
        bool? inputForget, bool? layout)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(LSTM, [x, w, r, b, sequenceLens, initialH, initialC, p], [
            (AttrActivationAlpha, activationAlpha),
            (AttrActivationBeta, activationBeta),
            (AttrActivations, activations),
            (AttrClip, clip),
            (AttrDirection, direction),
            (AttrHiddenSize, hiddenSize),
            (AttrInputForget, inputForget),
            (AttrLayout, layout)]);
        return (retval[0], retval[1], retval[2]);
    }

    public static IVariable Lrn(IVariable x, float? alpha = null, float? beta = null,
        float? bias = null, long? size = null)
        => NodeBuilder.BuildNodeSingleOut(LRN, [x], [
            (AttrAlpha, alpha),
            (AttrBeta, beta),
            (AttrBias, bias),
            (AttrSize, size)]);

    public static IVariable Upsample(IVariable x, IVariable scales, ResizeMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(UPSAMPLE, [x, scales], [(AttrMode, mode)]);

    // -- New opset-21 operators ---------------------------------------------

    public static IVariable Hardmax(IVariable input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(HARDMAX, [input], [(AttrAxis, axis)]);

    public static IVariable HardSigmoid(IVariable x, float? alpha = null, float? beta = null)
        => NodeBuilder.BuildNodeSingleOut(HARD_SIGMOID, [x], [(AttrAlpha, alpha), (AttrBeta, beta)]);

    public static IVariable HardSwish(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(HARD_SWISH, [x], []);

    public static IVariable HammingWindow(IVariable size, DType? outputDatatype = null, bool? periodic = null)
        => NodeBuilder.BuildNodeSingleOut(HAMMING_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static IVariable HannWindow(IVariable size, DType? outputDatatype = null, bool? periodic = null)
        => NodeBuilder.BuildNodeSingleOut(HANN_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static IVariable ImageDecoder(IVariable encodedStream, string? pixelFormat = null)
        => NodeBuilder.BuildNodeSingleOut(IMAGE_DECODER, [encodedStream], [(AttrPixelFormat, pixelFormat)]);

    public static IVariable IsInf(IVariable x, bool? detectNegative = null, bool? detectPositive = null)
        => NodeBuilder.BuildNodeSingleOut(IS_INF, [x], [(AttrDetectNegative, detectNegative), (AttrDetectPositive, detectPositive)]);

    public static IVariable IsNaN(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(IS_NAN, [x], []);

    public static (IVariable y, IVariable? mean, IVariable? invStdDev) LayerNormalization(
        IVariable x, IVariable scale, IVariable? b = null,
        long? axis = null, float? epsilon = null, long? stashType = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(LAYER_NORMALIZATION, [x, scale, b],
            [(AttrAxis, axis), (AttrEpsilon, epsilon), (AttrStashType, stashType)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null, retval.Length > 2 ? retval[2] : null);
    }

    public static IVariable LogSoftmax(IVariable input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(LOG_SOFTMAX, [input], [(AttrAxis, axis)]);

    public static IVariable MeanVarianceNormalization(IVariable x, long[]? axes = null)
        => NodeBuilder.BuildNodeSingleOut(MEAN_VARIANCE_NORMALIZATION, [x], [(AttrAxes, axes)]);

    public static IVariable MelWeightMatrix(
        IVariable numMelBins, IVariable dftLength, IVariable sampleRate,
        IVariable lowerEdgeHertz, IVariable upperEdgeHertz, DType? outputDatatype = null)
        => NodeBuilder.BuildNodeSingleOut(MEL_WEIGHT_MATRIX,
            [numMelBins, dftLength, sampleRate, lowerEdgeHertz, upperEdgeHertz],
            [(AttrOutputDatatype, outputDatatype)]);

    public static IVariable Mish(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(MISH, [x], []);

    public static IVariable Multinomial(IVariable input, DType? dtype = null, long? sampleSize = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(MULTINOMIAL, [input],
            [(AttrDtype, dtype), (AttrSampleSize, sampleSize), (AttrSeed, seed)]);

    public static IVariable NegativeLogLikelihoodLoss(
        IVariable input, IVariable target, IVariable? weight = null,
        long? ignoreIndex = null, string? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(NEGATIVE_LOG_LIKELIHOOD_LOSS, [input, target, weight],
            [(AttrIgnoreIndex, ignoreIndex), (AttrReduction, reduction)]);

    public static IVariable OneHot(IVariable indices, IVariable depth, IVariable values, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(ONE_HOT, [indices, depth, values], [(AttrAxis, axis)]);

    public static IVariable PRelu(IVariable x, IVariable slope)
        => NodeBuilder.BuildNodeSingleOut(P_RELU, [x, slope], []);

    public static IVariable QuantizeLinear(
        IVariable x, IVariable yScale, IVariable? yZeroPoint = null,
        long? axis = null, long? blockSize = null, DType? outputDatatype = null,
        bool? saturate = null, long? precision = null)
        => NodeBuilder.BuildNodeSingleOut(QUANTIZE_LINEAR, [x, yScale, yZeroPoint],
            [(AttrAxis, axis), (AttrBlockSize, blockSize), (AttrOutputDtype, outputDatatype),
             (AttrSaturate, saturate), (AttrPrecision, precision)]);

    public static IVariable QLinearMatMul(
        IVariable a, IVariable aScale, IVariable aZeroPoint,
        IVariable b, IVariable bScale, IVariable bZeroPoint,
        IVariable yScale, IVariable yZeroPoint)
        => NodeBuilder.BuildNodeSingleOut(QLINEAR_MATMUL,
            [a, aScale, aZeroPoint, b, bScale, bZeroPoint, yScale, yZeroPoint], []);

    public static IVariable QLinearConv(
        IVariable x, IVariable xScale, IVariable xZeroPoint,
        IVariable w, IVariable wScale, IVariable wZeroPoint,
        IVariable yScale, IVariable yZeroPoint, IVariable? b = null,
        AutoPad? autoPad = null, long[]? dilations = null, long? group = null,
        long[]? kernelShape = null, long[]? pads = null, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(QLINEAR_CONV,
            [x, xScale, xZeroPoint, w, wScale, wZeroPoint, yScale, yZeroPoint, b],
            [(AttrAutoPad, autoPad), (AttrDilations, dilations), (AttrGroup, group),
             (AttrKernelShape, kernelShape), (AttrPads, pads), (AttrStrides, strides)]);

    public static IVariable RegexFullMatch(IVariable x, string? pattern = null)
        => NodeBuilder.BuildNodeSingleOut(REGEX_FULL_MATCH, [x], [(AttrPattern, pattern)]);

    public static IVariable Round(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(ROUND, [x], []);

    public static IVariable Shrink(IVariable input, float? bias = null, float? lambd = null)
        => NodeBuilder.BuildNodeSingleOut(SHRINK, [input], [(AttrBias, bias), (AttrLambd, lambd)]);

    public static IVariable Size(IVariable data)
        => NodeBuilder.BuildNodeSingleOut(SIZE, [data], []);

    public static IVariable Softplus(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(SOFTPLUS, [x], []);

    public static IVariable Softsign(IVariable input)
        => NodeBuilder.BuildNodeSingleOut(SOFTSIGN, [input], []);

    public static (IVariable output, IVariable? logProb) SoftmaxCrossEntropyLoss(
        IVariable scores, IVariable labels, IVariable? weights = null,
        long? ignoreIndex = null, string? reduction = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(SOFTMAX_CROSS_ENTROPY_LOSS, [scores, labels, weights],
            [(AttrIgnoreIndex, ignoreIndex), (AttrReduction, reduction)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null);
    }

    public static IVariable SplitToSequence(IVariable input, IVariable? split = null, long? axis = null, long? keepdims = null)
        => NodeBuilder.BuildNodeSingleOut(SPLIT_TO_SEQUENCE, [input, split],
            [(AttrAxis, axis), (AttrKeepdims, keepdims)]);

    public static IVariable STFT(
        IVariable signal, IVariable frameStep, IVariable? window = null, IVariable? frameLength = null,
        bool? onesided = null)
        => NodeBuilder.BuildNodeSingleOut(OpCodes.STFT, [signal, frameStep, window, frameLength],
            [(AttrOnesided, onesided)]);

    public static IVariable StringConcat(IVariable x, IVariable y)
        => NodeBuilder.BuildNodeSingleOut(STRING_CONCAT, [x, y], []);

    public static IVariable StringNormalizer(IVariable x,
        string? caseChangeAction = null, long? isCaseSensitive = null,
        string? locale = null, string[]? stopwords = null)
        => NodeBuilder.BuildNodeSingleOut(STRING_NORMALIZER, [x],
            [(AttrCaseChangeAction, caseChangeAction), (AttrIsCaseSensitive, isCaseSensitive),
             (AttrLocale, locale), (AttrStopwords, stopwords)]);

    public static (IVariable y, IVariable numSplits) StringSplit(IVariable x, string? delimiter = null, long? maxsplit = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(STRING_SPLIT, [x],
            [(AttrDelimiter, delimiter), (AttrMaxsplit, maxsplit)]);
        return (retval[0], retval[1]);
    }

    public static IVariable TfIdfVectorizer(IVariable x,
        long? maxGramLength = null, long? maxSkipCount = null, long? minGramLength = null,
        string? mode = null, long[]? ngramCounts = null, long[]? ngramIndexes = null,
        long[]? poolInt64s = null, string[]? poolStrings = null, float[]? weights = null)
        => NodeBuilder.BuildNodeSingleOut(TFIDF_VECTORIZER, [x],
            [(AttrMaxGramLength, maxGramLength), (AttrMaxSkipCount, maxSkipCount),
             (AttrMinGramLength, minGramLength), (AttrMode, mode),
             (AttrNgramCounts, ngramCounts), (AttrNgramIndexes, ngramIndexes),
             (AttrPoolInt64s, poolInt64s), (AttrPoolStrings, poolStrings),
             (AttrWeights, weights)]);

    public static IVariable ThresholdedRelu(IVariable x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(THRESHOLDED_RELU, [x], [(AttrAlpha, alpha)]);

    /// <summary>Root-mean-square layer normalization over the suffix axes from <paramref name="axis"/> (ONNX RMSNormalization, opset 23+).
    /// Lowered inline to opset-21 primitives — <c>y = x / sqrt(mean(x², suffix axes) + epsilon) * scale</c>
    /// (ReduceMean/Sqrt/Div/Mul) — so the emitted ONNX stays at opset 21. The fused RMS_NORMALIZATION
    /// op definition and QEE kernel are retained; restore the fused emission here once a runtime
    /// registers it at a usable opset. (x and scale are assumed to share a dtype, the spec's common
    /// case.)</summary>
    public static IVariable RMSNormalization(IVariable x, IVariable scale,
        long? axis = null, float? epsilon = null, long? stashType = null)
    {
        var a = axis ?? -1;
        IVariable axesVar;
        if (a < 0)
        {
            // Negative axis: the suffix [axis, axis+1, ..., -1] is rank-independent.
            var count = (int)(-a);
            var axesArr = new long[count];
            for (var i = 0; i < count; i++) axesArr[i] = a + i;
            axesVar = Constant(axesArr);
        }
        else
        {
            // Non-negative axis: the suffix is [axis, ..., rank-1]. Compute it from the runtime
            // rank (Size of the shape vector) so it works on dynamic-rank inputs too.
            var rank = Size(Shape(x));
            axesVar = Range(Constant(a), rank, Constant(1L));
        }

        var meanSq = ReduceMean(Mul(x, x), axesVar, keepdims: true);
        var rms = Sqrt(Add(meanSq, CastLike(Constant(epsilon ?? 1e-5f), x, null)));
        return Mul(Div(x, rms), scale);
    }

    /// <summary>Rotary positional embedding (ONNX RotaryEmbedding, opset 23+); Y has X's shape.
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a faithful lowering of
    /// the fused op (position-id gather, interleaved vs half-split layouts, partial rotary dim) is
    /// intricate enough to belong in core (deferred core work) — so this
    /// throws rather than force a higher model opset. The ROTARY_EMBEDDING op definition and QEE kernel
    /// are retained; restore the fused emission here once a runtime supports it.</summary>
    public static IVariable RotaryEmbedding(IVariable x, IVariable cosCache, IVariable sinCache,
        IVariable? positionIds = null, bool? interleaved = null, long? numHeads = null,
        long? rotaryEmbeddingDim = null)
        => throw new System.NotImplementedException(
            "RotaryEmbedding (ONNX opset 23) has no opset-21 equivalent, and Shorokoo emits a single " +
            "opset-21 model. A faithful lowering (position-id gather, interleaved/half-split layouts, " +
            "partial rotary dim) is deferred to the core project. The op " +
            "definition is retained; re-enable the fused emission here when a runtime supports it.");

    /// <summary>Swish activation y = x * sigmoid(alpha * x) (ONNX Swish, opset 24+).
    /// Lowered inline to opset-21 primitives (Mul/Sigmoid) so the emitted ONNX stays at opset 21 —
    /// ONNX Runtime 1.26 registers no Swish kernel on any provider. The fused SWISH op definition
    /// is retained; restore the fused emission here once a runtime supports it.</summary>
    public static IVariable Swish(IVariable x, float? alpha = null)
    {
        var a = alpha ?? 1.0f;
        var scaled = a == 1.0f ? x : Mul(x, CastLike(Constant(a), x, null));
        return Mul(x, Sigmoid(scaled));
    }

    /// <summary>Writes <paramref name="update"/> into <paramref name="pastCache"/> along the sequence axis at the per-batch write indices (ONNX TensorScatter, opset 24+).
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a faithful lowering of
    /// the fused op (per-batch write indices, windowed/circular modes) is intricate enough to belong in
    /// core (deferred core work) — so this throws rather than force a
    /// higher model opset. The TENSOR_SCATTER op definition and (shape-only) QEE kernel are retained;
    /// restore the fused emission here once a runtime supports it.</summary>
    public static IVariable TensorScatter(IVariable pastCache, IVariable update,
        IVariable? writeIndices = null, long? axis = null, TensorScatterMode? mode = null)
        => throw new System.NotImplementedException(
            "TensorScatter (ONNX opset 24) has no opset-21 equivalent, and Shorokoo emits a single " +
            "opset-21 model. A faithful lowering (per-batch write indices, windowed/circular modes) is " +
            "deferred to the core project. The op definition is retained; " +
            "re-enable the fused emission here when a runtime supports it.");
}