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
    public static Variable Identity(Variable x, int? rank)
        => NodeBuilder.BuildNodeSingleOut(IDENTITY, [x], [(InternalAttrRank, (long?)rank)]);

    public static Node LoopOpen(Variable? maxNumIterations, Variable? condition, Variable?[] loopVariableInitializers)
        => NodeBuilder.BuildNode(LOOP_OPEN, [maxNumIterations, condition, .. loopVariableInitializers], []);

    public static Node LoopClose(Variable? continueWhile, Variable[] loopVariableUpdaters, Variable[] scanVariableInputs, Node openNode)
        => NodeBuilder.BuildNode(LOOP_CLOSE, [], [(AttrBody, (Variable?[])[continueWhile, .. loopVariableUpdaters, .. scanVariableInputs])], openNode: openNode);

    public static Node IfOpen(Variable condition)
        => NodeBuilder.BuildNode(IF_OPEN, [condition], []);

    public static Variable[] IfClose(Variable[] thenBranch, Variable[] elseBranch, Node openNode)
     => NodeBuilder.BuildNodeMultiOut(IF_CLOSE, [], [(AttrThenBranch, thenBranch), (AttrElseBranch, elseBranch)], openNode: openNode);

    public static Variable LeakyRelu(Variable x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(LEAKY_RELU, [x], [(AttrAlpha, alpha)]);

    public static Variable Less(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(LESS, [left, right], []);

    public static Variable LessOrEqual(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(LESS_OR_EQUAL, [left, right], []);

    public static Variable LoopIndexVariable()
        => NodeBuilder.BuildNodeSingleOut(LOOP_INDEX_VARIABLE, [], []);

    public static Variable LoopFakeInput(DType type, int? rank, DataStructure structure)
        => NodeBuilder.BuildNodeSingleOut(LOOP_FAKE_INPUT, [], [(AttrDtype, type), (InternalAttrRank, (long?)rank), (InternalAttrStructure, structure)]);

    public static Variable LoopScanZombie(Variable scannee)
        => NodeBuilder.BuildNodeSingleOut(LOOP_SCAN_VARIABLE, [scannee], []);
    
    public static Variable Log(Variable x)
        => NodeBuilder.BuildNodeSingleOut(LOG, [x], []);

    public static Variable LpPool(Variable x, AutoPad? autoPad, bool? ceilMode,
        long[]? dilations, long[] kernelShape, long? p, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(LP_POOL, [x], [
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrP, p),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable MatMul(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(MATMUL, [left, right], []);

    public static Variable MatMulInteger(Variable a, Variable b, Variable? aZeroPoint = null, Variable? bZeroPoint = null)
        => NodeBuilder.BuildNodeSingleOut(MATMUL_INTEGER, [a, b, aZeroPoint, bZeroPoint], []);

    public static Variable Max(params Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MAX, inputs, []);

    public static Variable Mean(params Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MEAN, inputs, []);

    public static Variable MaxPool(Variable x, AutoPad? autoPad = null, bool? ceilMode = null,
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

    public static (Variable y, Variable indices) MaxPoolWithIndices(Variable x, AutoPad? autoPad = null, bool? ceilMode = null,
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

    public static Variable MaxUnpool(Variable x, Variable indices,
        long[] kernelShape, long[]? pads = null, long[]? strides = null, Variable? outputShape = null)
        // The optional output_shape input is only emitted when present — ORT's MaxUnpool
        // kernel rejects a trailing empty-name optional input ("input count mismatch").
        => NodeBuilder.BuildNodeSingleOut(MAX_UNPOOL,
            outputShape is null ? [x, indices] : [x, indices, outputShape], [
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable MaxRoiPool(Variable x, Variable rois,
        long[] pooledShape, float? spatialScale = null)
        => NodeBuilder.BuildNodeSingleOut(MAX_ROI_POOL, [x, rois], [
            (AttrPooledShape, pooledShape),
            (AttrSpatialScale, spatialScale)]);

    public static Variable Min(params Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MIN, inputs, []);

    public static Variable Mod(Variable a, Variable b, bool? fmod = null)
        => NodeBuilder.BuildNodeSingleOut(MOD, [a, b], [(AttrFmod, fmod)]);

    public static Variable Mul(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(MUL, [left, right], []);

    public static Variable Neg(Variable x)
        => NodeBuilder.BuildNodeSingleOut(NEG, [x], []);

    public static Variable NonMaxSuppression(Variable boxes, Variable scores,
        Variable? maxOutputBoxesPerClass = null, Variable? iouThreshold = null,
        Variable? scoreThreshold = null, bool? centerPointBox = null)
        => NodeBuilder.BuildNodeSingleOut(NON_MAX_SUPPRESSION, 
            [boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold],
            [(AttrCenterPointBox, centerPointBox)]);

    public static Variable NonZero(Variable x)
        => NodeBuilder.BuildNodeSingleOut(NON_ZERO, [x], []);

    public static Variable Not(Variable x)
        => NodeBuilder.BuildNodeSingleOut(NOT, [x], []);

    public static Variable Optional(Variable? x, DataStructure structure, DType type)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL, x is null ? [] : [x], [(AttrType, (structure, type))]);

    public static Variable OptionalGetElement(Variable x)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL_GET_ELEMENT, [x], []);

    public static Variable OptionalHasElement(Variable x)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL_HAS_ELEMENT, [x], []);

    public static Variable Or(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(OR, [left, right], []);

    public static Variable Pad(Variable data, Variable pads, Variable? constantValue = null,
        Variable? axes = null, PadMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(PAD, [data, pads, constantValue, axes],
            [(AttrMode, mode)]);

    public static Variable Pow(Variable x, Variable y)
        => NodeBuilder.BuildNodeSingleOut(POW, [x, y], []);

    public static Variable Range(Variable start, Variable limit, Variable delta)
        => NodeBuilder.BuildNodeSingleOut(RANGE, [start, limit, delta], []);

    public static Variable RandomNormal(long[] shape, float? mean = null, float? scale = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_NORMAL, [], [
            (AttrShape, shape), (AttrDtype, dtype), (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    public static Variable RandomNormalLike(Variable input, float? mean = null, float? scale = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_NORMAL_LIKE, [input], [
            (AttrDtype, dtype), (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    public static Variable RandomUniform(long[] shape, float? high = null, float? low = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_UNIFORM, [], [
            (AttrShape, shape), (AttrDtype, dtype), (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    public static Variable RandomUniformLike(Variable input, float? high = null, float? low = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_UNIFORM_LIKE, [input], [
            (AttrDtype, dtype), (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    public static Variable Reciprocal(Variable x)
        => NodeBuilder.BuildNodeSingleOut(RECIPROCAL, [x], []);

    public static Variable ReduceL1(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_L1, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceL2(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_L2, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceLogSum(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_LOG_SUM, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceLogSumExp(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_LOG_SUM_EXP, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceMax(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MAX, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceMean(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MEAN, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceMin(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MIN, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceProd(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_PROD, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceSum(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_SUM, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static Variable ReduceSumSquare(Variable data, Variable? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_SUM_SQUARE, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static (Variable y, Variable yH) Rnn(Variable x, Variable w, Variable r,
        Variable? b, Variable? sequenceLens, Variable? initialH,
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

    public static Variable Relu(Variable x)
        => NodeBuilder.BuildNodeSingleOut(RELU, [x], []);

    public static Variable Reshape(Variable data, Variable shape, bool allowZero)
        => NodeBuilder.BuildNodeSingleOut(RESHAPE, [data, shape], [(AttrAllowzero, allowZero)]);

    public static Variable ReverseSequence(Variable input, Variable sequenceLens, long? batchAxis = null, long? timeAxis = null)
        => NodeBuilder.BuildNodeSingleOut(REVERSE_SEQUENCE, [input, sequenceLens], [(AttrBatchAxis, batchAxis), (AttrTimeAxis, timeAxis)]);

    public static Variable Resize(Variable x, Variable? roi, Variable? scales,
        Variable? sizes, bool? antialias, long[]? axes,
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

    public static Variable RoiAlign(Variable x, Variable rois, Variable batchIndices,
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

    public static Variable ScatterElements(Variable data, Variable indices, Variable updates,
        long? axis = null, ScatterNDReduction? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(SCATTER_ELEMENTS, [data, indices, updates],
            [(AttrAxis, axis), (AttrReduction, reduction)]);

    public static Variable ScatterND(Variable data, Variable indices, Variable updates,
        ScatterNDReduction? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(SCATTER_ND, [data, indices, updates],
            [(AttrReduction, reduction)]);

    public static Variable Selu(Variable x, float? alpha = null, float? gamma = null)
        => NodeBuilder.BuildNodeSingleOut(SELU, [x], [(AttrAlpha, alpha), (AttrGamma, gamma)]);

    public static Variable SequenceAt(Variable input, Variable position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_AT, [input, position], []);

    public static Variable SequenceConstruct(params Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONSTRUCT, inputs, []);

    internal static Variable SequenceConstruct(Function targetFunction, params Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONSTRUCT, inputs, [], targetFunction: targetFunction);

    public static Variable SequenceEmpty(DType type, Function? targetFunction = null)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_EMPTY, [], [(AttrDtype, type)], targetFunction: targetFunction);

    public static Variable SequenceErase(Variable input, Variable? position = null)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_ERASE, [input, position], []);

    public static Variable SequenceErase(Function targetFunction, Variable input, Variable? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_ERASE, [input, position], [], targetFunction: targetFunction);

    public static Variable SequenceInsert(Variable input, Variable tensor, Variable? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_INSERT, [input, tensor, position], []);

    public static Variable SequenceInsert(Function targetFunction, Variable input, Variable tensor, Variable? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_INSERT, [input, tensor, position], [], targetFunction: targetFunction);

    public static Variable SequenceLength(Variable input)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_LENGTH, [input], []);

    public static Variable Shape(Variable data, long? end = null, long? start = null)
        => NodeBuilder.BuildNodeSingleOut(SHAPE, [data], [(AttrEnd, end), (AttrStart, start)]);

    public static Variable Sigmoid(Variable x)
        => NodeBuilder.BuildNodeSingleOut(SIGMOID, [x], []);

    public static Variable Sign(Variable x)
        => NodeBuilder.BuildNodeSingleOut(SIGN, [x], []);

    public static Variable Sin(Variable num)
        => NodeBuilder.BuildNodeSingleOut(SIN, [num], []);

    public static Variable Sinh(Variable num)
        => NodeBuilder.BuildNodeSingleOut(SINH, [num], []);

    public static Variable Slice(Variable data, Variable starts, Variable ends,
        Variable? axes = null, Variable? steps = null)
        => NodeBuilder.BuildNodeSingleOut(SLICE, [data, starts, ends, axes, steps], []);

    public static Variable Softmax(Variable input, long? axis)
        => NodeBuilder.BuildNodeSingleOut(SOFTMAX, [input], [(AttrAxis, axis)]);

    public static Variable[] Split(Variable data, Variable? splits, long? axis, long? numOutputs, long variadicOutputCount)
        => NodeBuilder.BuildNodeMultiOut(SPLIT, [data, splits], [(AttrAxis, axis), (AttrNumOutputs, numOutputs)],
            outputNames: Enumerable.Repeat((string?)null, (int)variadicOutputCount).ToArray());

    public static Variable Sqrt(Variable num)
        => NodeBuilder.BuildNodeSingleOut(SQRT, [num], []);

    public static Variable Squeeze(Variable data, Variable? axes)
        => NodeBuilder.BuildNodeSingleOut(SQUEEZE, [data, axes], []);

    public static Variable Sub(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(SUB, [left, right], []);

    public static Variable Sum(params Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SUM, inputs, []);

    public static Variable Tan(Variable x)
        => NodeBuilder.BuildNodeSingleOut(TAN, [x], []);

    public static Variable Tanh(Variable x)
        => NodeBuilder.BuildNodeSingleOut(TANH, [x], []);

    public static Variable Tile(Variable input, Variable repeats)
        => NodeBuilder.BuildNodeSingleOut(TILE, [input, repeats], []);

    public static (Variable values, Variable indices) TopK(Variable x, Variable k,
        long? axis = null, bool? largest = null, bool? sorted = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(TOPK, [x, k], [
            (AttrAxis, axis),
            (AttrLargest, largest),
            (AttrSorted, sorted)]);
        return (retval[0], retval[1]);
    }

    public static Variable Transpose(Variable data, long[]? perm = null)
        => NodeBuilder.BuildNodeSingleOut(TRANSPOSE, [data], [(AttrPerm, perm)]);

    public static (Variable y, Variable indices, Variable inverseIndices, Variable counts) Unique(
        Variable x, long? axis = null, bool? sorted = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(UNIQUE, [x], [
            (AttrAxis, axis),
            (AttrSorted, sorted)]);
        return (retval[0], retval[1], retval[2], retval[3]);
    }

    public static Variable Unsqueeze(Variable data, Variable axes)
        => NodeBuilder.BuildNodeSingleOut(UNSQUEEZE, [data, axes], []);

    public static Variable Where(Variable condition, Variable x, Variable y)
        => NodeBuilder.BuildNodeSingleOut(WHERE, [condition, x, y], []);

    public static Variable Xor(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(XOR, [left, right], []);

    public static Variable SpaceToDepth(Variable input, long? blockSize)
        => NodeBuilder.BuildNodeSingleOut(SPACE_TO_DEPTH, [input], [(AttrBlocksize, blockSize)]);

    public static Variable Trilu(Variable input, Variable? k = null, long? upper = null)
        => NodeBuilder.BuildNodeSingleOut(TRILU, [input, k], [(AttrUpper, upper)]);

    public static Variable LpNormalization(Variable input, long? axis = null, long? p = null)
        => NodeBuilder.BuildNodeSingleOut(LP_NORMALIZATION, [input], [(AttrAxis, axis), (AttrP, p)]);

    public static (Variable y, Variable yH, Variable yC) Lstm(Variable x, Variable w, Variable r,
        Variable? b, Variable? sequenceLens, Variable? initialH, Variable? initialC, Variable? p,
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

    public static Variable Lrn(Variable x, float? alpha = null, float? beta = null,
        float? bias = null, long? size = null)
        => NodeBuilder.BuildNodeSingleOut(LRN, [x], [
            (AttrAlpha, alpha),
            (AttrBeta, beta),
            (AttrBias, bias),
            (AttrSize, size)]);

    public static Variable Upsample(Variable x, Variable scales, ResizeMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(UPSAMPLE, [x, scales], [(AttrMode, mode)]);

    // -- New opset-21 operators ---------------------------------------------

    public static Variable Hardmax(Variable input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(HARDMAX, [input], [(AttrAxis, axis)]);

    public static Variable HardSigmoid(Variable x, float? alpha = null, float? beta = null)
        => NodeBuilder.BuildNodeSingleOut(HARD_SIGMOID, [x], [(AttrAlpha, alpha), (AttrBeta, beta)]);

    public static Variable HardSwish(Variable x)
        => NodeBuilder.BuildNodeSingleOut(HARD_SWISH, [x], []);

    public static Variable HammingWindow(Variable size, DType? outputDatatype = null, bool? periodic = null)
        => NodeBuilder.BuildNodeSingleOut(HAMMING_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static Variable HannWindow(Variable size, DType? outputDatatype = null, bool? periodic = null)
        => NodeBuilder.BuildNodeSingleOut(HANN_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static Variable ImageDecoder(Variable encodedStream, string? pixelFormat = null)
        => NodeBuilder.BuildNodeSingleOut(IMAGE_DECODER, [encodedStream], [(AttrPixelFormat, pixelFormat)]);

    public static Variable IsInf(Variable x, bool? detectNegative = null, bool? detectPositive = null)
        => NodeBuilder.BuildNodeSingleOut(IS_INF, [x], [(AttrDetectNegative, detectNegative), (AttrDetectPositive, detectPositive)]);

    public static Variable IsNaN(Variable x)
        => NodeBuilder.BuildNodeSingleOut(IS_NAN, [x], []);

    public static (Variable y, Variable? mean, Variable? invStdDev) LayerNormalization(
        Variable x, Variable scale, Variable? b = null,
        long? axis = null, float? epsilon = null, long? stashType = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(LAYER_NORMALIZATION, [x, scale, b],
            [(AttrAxis, axis), (AttrEpsilon, epsilon), (AttrStashType, stashType)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null, retval.Length > 2 ? retval[2] : null);
    }

    public static Variable LogSoftmax(Variable input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(LOG_SOFTMAX, [input], [(AttrAxis, axis)]);

    public static Variable MeanVarianceNormalization(Variable x, long[]? axes = null)
        => NodeBuilder.BuildNodeSingleOut(MEAN_VARIANCE_NORMALIZATION, [x], [(AttrAxes, axes)]);

    public static Variable MelWeightMatrix(
        Variable numMelBins, Variable dftLength, Variable sampleRate,
        Variable lowerEdgeHertz, Variable upperEdgeHertz, DType? outputDatatype = null)
        => NodeBuilder.BuildNodeSingleOut(MEL_WEIGHT_MATRIX,
            [numMelBins, dftLength, sampleRate, lowerEdgeHertz, upperEdgeHertz],
            [(AttrOutputDatatype, outputDatatype)]);

    public static Variable Mish(Variable x)
        => NodeBuilder.BuildNodeSingleOut(MISH, [x], []);

    public static Variable Multinomial(Variable input, DType? dtype = null, long? sampleSize = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(MULTINOMIAL, [input],
            [(AttrDtype, dtype), (AttrSampleSize, sampleSize), (AttrSeed, seed)]);

    public static Variable NegativeLogLikelihoodLoss(
        Variable input, Variable target, Variable? weight = null,
        long? ignoreIndex = null, string? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(NEGATIVE_LOG_LIKELIHOOD_LOSS, [input, target, weight],
            [(AttrIgnoreIndex, ignoreIndex), (AttrReduction, reduction)]);

    public static Variable OneHot(Variable indices, Variable depth, Variable values, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(ONE_HOT, [indices, depth, values], [(AttrAxis, axis)]);

    public static Variable PRelu(Variable x, Variable slope)
        => NodeBuilder.BuildNodeSingleOut(P_RELU, [x, slope], []);

    public static Variable QuantizeLinear(
        Variable x, Variable yScale, Variable? yZeroPoint = null,
        long? axis = null, long? blockSize = null, DType? outputDatatype = null,
        bool? saturate = null, long? precision = null)
        => NodeBuilder.BuildNodeSingleOut(QUANTIZE_LINEAR, [x, yScale, yZeroPoint],
            [(AttrAxis, axis), (AttrBlockSize, blockSize), (AttrOutputDtype, outputDatatype),
             (AttrSaturate, saturate), (AttrPrecision, precision)]);

    public static Variable QLinearMatMul(
        Variable a, Variable aScale, Variable aZeroPoint,
        Variable b, Variable bScale, Variable bZeroPoint,
        Variable yScale, Variable yZeroPoint)
        => NodeBuilder.BuildNodeSingleOut(QLINEAR_MATMUL,
            [a, aScale, aZeroPoint, b, bScale, bZeroPoint, yScale, yZeroPoint], []);

    public static Variable QLinearConv(
        Variable x, Variable xScale, Variable xZeroPoint,
        Variable w, Variable wScale, Variable wZeroPoint,
        Variable yScale, Variable yZeroPoint, Variable? b = null,
        AutoPad? autoPad = null, long[]? dilations = null, long? group = null,
        long[]? kernelShape = null, long[]? pads = null, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(QLINEAR_CONV,
            [x, xScale, xZeroPoint, w, wScale, wZeroPoint, yScale, yZeroPoint, b],
            [(AttrAutoPad, autoPad), (AttrDilations, dilations), (AttrGroup, group),
             (AttrKernelShape, kernelShape), (AttrPads, pads), (AttrStrides, strides)]);

    public static Variable RegexFullMatch(Variable x, string? pattern = null)
        => NodeBuilder.BuildNodeSingleOut(REGEX_FULL_MATCH, [x], [(AttrPattern, pattern)]);

    public static Variable Round(Variable x)
        => NodeBuilder.BuildNodeSingleOut(ROUND, [x], []);

    public static Variable Shrink(Variable input, float? bias = null, float? lambd = null)
        => NodeBuilder.BuildNodeSingleOut(SHRINK, [input], [(AttrBias, bias), (AttrLambd, lambd)]);

    public static Variable Size(Variable data)
        => NodeBuilder.BuildNodeSingleOut(SIZE, [data], []);

    public static Variable Softplus(Variable x)
        => NodeBuilder.BuildNodeSingleOut(SOFTPLUS, [x], []);

    public static Variable Softsign(Variable input)
        => NodeBuilder.BuildNodeSingleOut(SOFTSIGN, [input], []);

    public static (Variable output, Variable? logProb) SoftmaxCrossEntropyLoss(
        Variable scores, Variable labels, Variable? weights = null,
        long? ignoreIndex = null, string? reduction = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(SOFTMAX_CROSS_ENTROPY_LOSS, [scores, labels, weights],
            [(AttrIgnoreIndex, ignoreIndex), (AttrReduction, reduction)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null);
    }

    public static Variable SplitToSequence(Variable input, Variable? split = null, long? axis = null, long? keepdims = null)
        => NodeBuilder.BuildNodeSingleOut(SPLIT_TO_SEQUENCE, [input, split],
            [(AttrAxis, axis), (AttrKeepdims, keepdims)]);

    public static Variable STFT(
        Variable signal, Variable frameStep, Variable? window = null, Variable? frameLength = null,
        bool? onesided = null)
        => NodeBuilder.BuildNodeSingleOut(OpCodes.STFT, [signal, frameStep, window, frameLength],
            [(AttrOnesided, onesided)]);

    public static Variable StringConcat(Variable x, Variable y)
        => NodeBuilder.BuildNodeSingleOut(STRING_CONCAT, [x, y], []);

    public static Variable StringNormalizer(Variable x,
        string? caseChangeAction = null, long? isCaseSensitive = null,
        string? locale = null, string[]? stopwords = null)
        => NodeBuilder.BuildNodeSingleOut(STRING_NORMALIZER, [x],
            [(AttrCaseChangeAction, caseChangeAction), (AttrIsCaseSensitive, isCaseSensitive),
             (AttrLocale, locale), (AttrStopwords, stopwords)]);

    public static (Variable y, Variable numSplits) StringSplit(Variable x, string? delimiter = null, long? maxsplit = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(STRING_SPLIT, [x],
            [(AttrDelimiter, delimiter), (AttrMaxsplit, maxsplit)]);
        return (retval[0], retval[1]);
    }

    public static Variable TfIdfVectorizer(Variable x,
        long? maxGramLength = null, long? maxSkipCount = null, long? minGramLength = null,
        string? mode = null, long[]? ngramCounts = null, long[]? ngramIndexes = null,
        long[]? poolInt64s = null, string[]? poolStrings = null, float[]? weights = null)
        => NodeBuilder.BuildNodeSingleOut(TFIDF_VECTORIZER, [x],
            [(AttrMaxGramLength, maxGramLength), (AttrMaxSkipCount, maxSkipCount),
             (AttrMinGramLength, minGramLength), (AttrMode, mode),
             (AttrNgramCounts, ngramCounts), (AttrNgramIndexes, ngramIndexes),
             (AttrPoolInt64s, poolInt64s), (AttrPoolStrings, poolStrings),
             (AttrWeights, weights)]);

    public static Variable ThresholdedRelu(Variable x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(THRESHOLDED_RELU, [x], [(AttrAlpha, alpha)]);

    /// <summary>Root-mean-square layer normalization over the suffix axes from <paramref name="axis"/> (ONNX RMSNormalization, opset 23+).
    /// Lowered inline to opset-21 primitives — <c>y = x / sqrt(mean(x², suffix axes) + epsilon) * scale</c>
    /// (ReduceMean/Sqrt/Div/Mul) — so the emitted ONNX stays at opset 21. The fused RMS_NORMALIZATION
    /// op definition and QEE kernel are retained; restore the fused emission here once a runtime
    /// registers it at a usable opset. (x and scale are assumed to share a dtype, the spec's common
    /// case.)</summary>
    public static Variable RMSNormalization(Variable x, Variable scale,
        long? axis = null, float? epsilon = null, long? stashType = null)
    {
        var a = axis ?? -1;
        Variable axesVar;
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
    public static Variable RotaryEmbedding(Variable x, Variable cosCache, Variable sinCache,
        Variable? positionIds = null, bool? interleaved = null, long? numHeads = null,
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
    public static Variable Swish(Variable x, float? alpha = null)
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
    public static Variable TensorScatter(Variable pastCache, Variable update,
        Variable? writeIndices = null, long? axis = null, TensorScatterMode? mode = null)
        => throw new System.NotImplementedException(
            "TensorScatter (ONNX opset 24) has no opset-21 equivalent, and Shorokoo emits a single " +
            "opset-21 model. A faithful lowering (per-batch write indices, windowed/circular modes) is " +
            "deferred to the core project. The op definition is retained; " +
            "re-enable the fused emission here when a runtime supports it.");
}