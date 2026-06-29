using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static partial class OnnxOp
{
    public static Variable Abs(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ABS, [num], []);

    public static Variable Acos(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ACOS, [num], []);

    public static Variable Acosh(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ACOSH, [num], []);

    public static Variable Add(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(ADD, [left, right], []);

    public static Variable AffineGrid(Variable theta, Variable size, bool? alignCorners)
        => NodeBuilder.BuildNodeSingleOut(AFFINE_GRID, [theta, size], [(AttrAlignCorners, alignCorners)]);

    public static Variable And(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(AND, [left, right], []);

    public static Variable ArgMax(Variable x, long? axis, bool? keepdims, bool? selectLastIndex)
        => NodeBuilder.BuildNodeSingleOut(ARG_MAX, [x], [(AttrAxis, axis), (AttrKeepdims, keepdims), (AttrSelectLastIndex, selectLastIndex)]);

    public static Variable ArgMin(Variable x, long? axis, bool? keepdims, bool? selectLastIndex)
        => NodeBuilder.BuildNodeSingleOut(ARG_MIN, [x], [(AttrAxis, axis), (AttrKeepdims, keepdims), (AttrSelectLastIndex, selectLastIndex)]);

    public static Variable Asin(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ASIN, [num], []);

    public static Variable Asinh(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ASINH, [num], []);

    public static Variable Atan(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ATAN, [num], []);

    public static Variable Atanh(Variable num)
        => NodeBuilder.BuildNodeSingleOut(ATANH, [num], []);

    public static Variable AveragePool(Variable x, AutoPad? autoPad, bool? ceilMode, bool? countIncludePad, 
        long[]? dilations, long[] kernelShape, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(AVERAGE_POOL, [x], [
            (AttrAutoPad, autoPad), 
            (AttrCeilMode, ceilMode), 
            (AttrCountIncludePad, countIncludePad),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable BatchNormalization(
        Variable x, Variable scale, Variable b, Variable inputMean, Variable inputVar,
        float? epsilon, float? momentum, bool? trainingMode)
    {
        var retval = NodeBuilder.BuildNodeSingleOut(BATCH_NORMALIZATION, [x, scale, b, inputMean, inputVar],
            [(AttrEpsilon, epsilon), (AttrMomentum, momentum), (AttrTrainingMode, trainingMode)]);

        return retval;
    }

    public static (Variable y, Variable runningMean, Variable runningVariance) BatchNormalizationFullOutputs(
        Variable x, Variable scale, Variable b, Variable inputMean, Variable inputVar,
        float? epsilon, float? momentum, bool? trainingMode)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(BATCH_NORMALIZATION, [x, scale, b, inputMean, inputVar],
            [(AttrEpsilon, epsilon), (AttrMomentum, momentum), (AttrTrainingMode, trainingMode)]);

        return (retval[0], retval[1], retval[2]);
    }

    public static Variable BlackmanWindow(Variable size, DType? outputDatatype, bool? periodic)
        => NodeBuilder.BuildNodeSingleOut(BLACKMAN_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static Variable Bernoulli(Variable x, DType? dtype, float? seed)
        => NodeBuilder.BuildNodeSingleOut(BERNOULLI, [x], [(AttrDtype, dtype), (AttrSeed, seed)]);

    public static Variable BitShift(Variable x, Variable y, BitShiftDirection? direction)
        => NodeBuilder.BuildNodeSingleOut(BIT_SHIFT, [x, y], [(AttrDirection, direction)]);

    public static Variable BitwiseAnd(Variable x, Variable y)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_AND, [x, y], []);

    public static Variable BitwiseNot(Variable x)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_NOT, [x], []);

    public static Variable BitwiseOr(Variable x, Variable y)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_OR, [x, y], []);

    public static Variable BitwiseXor(Variable x, Variable y)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_XOR, [x, y], []);

    public static Variable Cast(Variable input, bool? saturate, DType to)
        => NodeBuilder.BuildNodeSingleOut(CAST, [input], [/*(AttrSaturate, saturate), */(AttrTo, to)]);

    public static Variable CastLike(Variable input, Variable targetType, bool? saturate)
        => NodeBuilder.BuildNodeSingleOut(CAST_LIKE, [input, targetType], [(AttrSaturate, saturate)]);

    public static Variable Ceil(Variable x)
        => NodeBuilder.BuildNodeSingleOut(CEIL, [x], []);

    public static Variable Celu(Variable x, float? alpha)
        => NodeBuilder.BuildNodeSingleOut(CELU, [x], [(AttrAlpha, alpha)]);

    public static Variable CenterCropPad(Variable input, Variable shape, long[]? axes)
        => NodeBuilder.BuildNodeSingleOut(CENTER_CROP_PAD, [input, shape], [(AttrAxes, axes)]);

    public static Variable Clip(Variable input, Variable min, Variable max)
        => NodeBuilder.BuildNodeSingleOut(CLIP, [input, min, max], []);

    public static Variable Col2Im(Variable input, Variable imageShape, Variable blockShape,
        long[] dilations, long[] pads, long[] strides)
        => NodeBuilder.BuildNodeSingleOut(COL2IM, [input, imageShape, blockShape], 
            [(AttrDilations, dilations), (AttrPads, pads), (AttrStrides, strides)]);

    public static Variable Compress(Variable input, Variable condition, long? axis)
        => NodeBuilder.BuildNodeSingleOut(COMPRESS, [input, condition], [(AttrAxis, axis)]);

    public static Variable Concat(Variable[] inputs, long axis)
        => NodeBuilder.BuildNodeSingleOut(CONCAT, inputs, [(AttrAxis, axis)]);

    public static Variable ConcatFromSequence(Variable inputSequence, long axis, bool newAxis)
        => NodeBuilder.BuildNodeSingleOut(CONCAT_FROM_SEQUENCE, [inputSequence], [(AttrAxis, axis), (AttrNewAxis, newAxis)]);
    public static Variable Constant(TensorData value) => Constant(value, null, null, null, null, null, null);
    public static Variable Constant(float value) => Constant(null, value, null, null, null, null, null);
    public static Variable Constant(float[] value) => Constant(null, null, value, null, null, null, null);
    public static Variable Constant(long value) => Constant(null, null, null, value, null, null, null);
    public static Variable Constant(long[] value) => Constant(null, null, null, null, value, null, null);
    public static Variable Constant(string value) => Constant(null, null, null, null, null, value, null);
    public static Variable Constant(string[] value) => Constant(null, null, null, null, null, null, value);

    public static Variable Constant(TensorData? value, float? valueFloat, float[]? valueFloats,
        long? valueInt, long[]? valueInts, string? valueString, string[]? valueStrings)
        => NodeBuilder.BuildNodeSingleOut(CONSTANT, [], [
            (AttrValue, value),
            (AttrValueFloat, valueFloat),
            (AttrValueFloats, valueFloats),
            (AttrValueInt, valueInt),
            (AttrValueInts, valueInts),
            (AttrValueString, valueString),
            (AttrValueStrings, valueStrings)]);

    public static Variable ConstantOfShape(Variable shape, TensorData value, int? rank)
        => Identity(ConstantOfShape(shape, value), rank);

    public static Variable ConstantOfShape(Variable shape, TensorData value)
        => NodeBuilder.BuildNodeSingleOut(CONSTANT_OF_SHAPE, [shape], [(AttrValue, value)]);

    public static Variable Conv(Variable x, Variable w, Variable b, AutoPad autoPad,
        long[] dilations, long group, long[] kernelShape,
        long[]? pads, long[] strides)
        => NodeBuilder.BuildNodeSingleOut(CONV, [x, w, b], [
            (AttrAutoPad, autoPad),
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable ConvInteger(Variable x, Variable w, Variable xZeroPoint, Variable wZeroPoint,
        AutoPad autoPad, long[]? dilations, long group, long[]? kernelShape, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(CONV_INTEGER, [x, w, xZeroPoint, wZeroPoint], [
            (AttrAutoPad, autoPad),
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable ConvTranspose(Variable x, Variable w, Variable b, AutoPad autoPad, long[]? dilations, long group,
        long[]? kernelShape, long[]? outputPadding, long[]? outputShape, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(CONV_TRANSPOSE, [x, w, b], [
            (AttrAutoPad, autoPad),
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrOutputPadding, outputPadding),
            (AttrOutputShape, outputShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable Cos(Variable x)
        => NodeBuilder.BuildNodeSingleOut(COS, [x], []);

    public static Variable Cosh(Variable x)
        => NodeBuilder.BuildNodeSingleOut(COSH, [x], []);

    public static Variable CumSum(Variable x, Variable axis, bool exclusive, bool reverse)
        => NodeBuilder.BuildNodeSingleOut(CUM_SUM, [x, axis], [(AttrExclusive, exclusive), (AttrReverse, reverse)]);

    /// <summary>Scaled dot-product attention returning only Y (ONNX Attention, opset 23+); no KV cache.
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a faithful lowering of
    /// the fused op (causal/GQA/softcap/qk-output-mode variants, plus a proper non-fused autodiff rule)
    /// is intricate enough to belong in core (deferred core work) — so this
    /// throws rather than force a higher model opset. The ATTENTION op definition and QEE kernel are
    /// retained; restore the fused emission here once a runtime supports it at a usable opset.</summary>
    public static Variable Attention(Variable q, Variable k, Variable v,
        Variable? attnMask = null, Variable? nonpadKvSeqlen = null,
        bool? isCausal = null, long? kvNumHeads = null, long? qNumHeads = null,
        long? qkMatmulOutputMode = null, float? scale = null, float? softcap = null,
        long? softmaxPrecision = null)
        => throw new System.NotImplementedException(
            "Attention (ONNX opset 23) has no opset-21 equivalent, and Shorokoo emits a single " +
            "opset-21 model. A faithful lowering plus a proper non-fused autodiff rule are deferred " +
            "to the core project (build attention from primitives / a Transformer module). The op " +
            "definition is retained; re-enable the fused emission here " +
            "when a runtime supports it.");

    /// <summary>Scaled dot-product attention with KV-cache update: returns (Y, present_key, present_value) (ONNX Attention, opset 23+).
    /// Not emittable today for the same reason as <see cref="Attention"/> (single opset-21 export; the
    /// KV-cache update and multi-output lowering belong in core), so this throws. The ATTENTION op
    /// definition and QEE kernel are retained.</summary>
    public static (Variable y, Variable presentKey, Variable presentValue) AttentionWithKVCache(
        Variable q, Variable k, Variable v,
        Variable? attnMask = null, Variable? pastKey = null, Variable? pastValue = null,
        bool? isCausal = null, long? kvNumHeads = null, long? qNumHeads = null,
        long? qkMatmulOutputMode = null, float? scale = null, float? softcap = null,
        long? softmaxPrecision = null)
        => throw new System.NotImplementedException(
            "Attention with KV cache (ONNX opset 23) has no opset-21 equivalent, and Shorokoo emits a " +
            "single opset-21 model. The KV-cache update and a proper non-fused autodiff rule are " +
            "deferred to the core project. The op definition is retained; " +
            "re-enable the fused emission here when a runtime supports it.");

    /// <summary>Reinterprets the tensor's bits as the same-width dtype <paramref name="to"/> (ONNX BitCast, opset 26+).
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a bit-pattern
    /// reinterpretation has no opset-21 primitive equivalent, so this throws rather than force a
    /// higher model opset. The BIT_CAST op definition and QEE kernel are retained; restore the fused
    /// emission here once a runtime supports it at a usable opset.</summary>
    public static Variable BitCast(Variable input, DType to)
        => throw new System.NotImplementedException(
            "BitCast (ONNX opset 26) has no opset-21 primitive equivalent (no op reinterprets bit " +
            "patterns), and Shorokoo emits a single opset-21 model — it cannot be lowered. The op " +
            "definition is retained; re-enable the fused emission here when a runtime supports it.");

    /// <summary>Cumulative product along the axis given as a 0-D tensor input (ONNX CumProd, opset 26+).
    /// Not emittable today: there is no opset-21 CumProd, and a faithful general decomposition needs a
    /// Scan (multiply body) which is not yet implemented, so this throws rather than force a higher
    /// model opset. The CUM_PROD op definition and QEE kernel are retained.</summary>
    public static Variable CumProd(Variable x, Variable axis, bool exclusive = false, bool reverse = false)
        => throw new System.NotImplementedException(
            "CumProd (ONNX opset 26) has no opset-21 equivalent (a general decomposition needs a Scan " +
            "multiply body, not yet implemented), and Shorokoo emits a single opset-21 model. The op " +
            "definition is retained; re-enable the fused emission here when a runtime supports it.");
}