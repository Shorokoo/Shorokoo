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
    public static IVariable Abs(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ABS, [num], []);

    public static IVariable Acos(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ACOS, [num], []);

    public static IVariable Acosh(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ACOSH, [num], []);

    public static IVariable Add(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(ADD, [left, right], []);

    public static IVariable AffineGrid(IVariable theta, IVariable size, bool? alignCorners)
        => NodeBuilder.BuildNodeSingleOut(AFFINE_GRID, [theta, size], [(AttrAlignCorners, alignCorners)]);

    public static IVariable And(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(AND, [left, right], []);

    public static IVariable ArgMax(IVariable x, long? axis, bool? keepdims, bool? selectLastIndex)
        => NodeBuilder.BuildNodeSingleOut(ARG_MAX, [x], [(AttrAxis, axis), (AttrKeepdims, keepdims), (AttrSelectLastIndex, selectLastIndex)]);

    public static IVariable ArgMin(IVariable x, long? axis, bool? keepdims, bool? selectLastIndex)
        => NodeBuilder.BuildNodeSingleOut(ARG_MIN, [x], [(AttrAxis, axis), (AttrKeepdims, keepdims), (AttrSelectLastIndex, selectLastIndex)]);

    public static IVariable Asin(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ASIN, [num], []);

    public static IVariable Asinh(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ASINH, [num], []);

    public static IVariable Atan(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ATAN, [num], []);

    public static IVariable Atanh(IVariable num)
        => NodeBuilder.BuildNodeSingleOut(ATANH, [num], []);

    public static IVariable AveragePool(IVariable x, AutoPad? autoPad, bool? ceilMode, bool? countIncludePad, 
        long[]? dilations, long[] kernelShape, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(AVERAGE_POOL, [x], [
            (AttrAutoPad, autoPad), 
            (AttrCeilMode, ceilMode), 
            (AttrCountIncludePad, countIncludePad),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IVariable BatchNormalization(
        IVariable x, IVariable scale, IVariable b, IVariable inputMean, IVariable inputVar,
        float? epsilon, float? momentum, bool? trainingMode)
    {
        var retval = NodeBuilder.BuildNodeSingleOut(BATCH_NORMALIZATION, [x, scale, b, inputMean, inputVar],
            [(AttrEpsilon, epsilon), (AttrMomentum, momentum), (AttrTrainingMode, trainingMode)]);

        return retval;
    }

    public static (IVariable y, IVariable runningMean, IVariable runningVariance) BatchNormalizationFullOutputs(
        IVariable x, IVariable scale, IVariable b, IVariable inputMean, IVariable inputVar,
        float? epsilon, float? momentum, bool? trainingMode)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(BATCH_NORMALIZATION, [x, scale, b, inputMean, inputVar],
            [(AttrEpsilon, epsilon), (AttrMomentum, momentum), (AttrTrainingMode, trainingMode)]);

        return (retval[0], retval[1], retval[2]);
    }

    public static IVariable BlackmanWindow(IVariable size, DType? outputDatatype, bool? periodic)
        => NodeBuilder.BuildNodeSingleOut(BLACKMAN_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static IVariable Bernoulli(IVariable x, DType? dtype, float? seed)
        => NodeBuilder.BuildNodeSingleOut(BERNOULLI, [x], [(AttrDtype, dtype), (AttrSeed, seed)]);

    public static IVariable BitShift(IVariable x, IVariable y, BitShiftDirection? direction)
        => NodeBuilder.BuildNodeSingleOut(BIT_SHIFT, [x, y], [(AttrDirection, direction)]);

    public static IVariable BitwiseAnd(IVariable x, IVariable y)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_AND, [x, y], []);

    public static IVariable BitwiseNot(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_NOT, [x], []);

    public static IVariable BitwiseOr(IVariable x, IVariable y)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_OR, [x, y], []);

    public static IVariable BitwiseXor(IVariable x, IVariable y)
        => NodeBuilder.BuildNodeSingleOut(BITWISE_XOR, [x, y], []);

    public static IVariable Cast(IVariable input, bool? saturate, DType to)
        => NodeBuilder.BuildNodeSingleOut(CAST, [input], [/*(AttrSaturate, saturate), */(AttrTo, to)]);

    public static IVariable CastLike(IVariable input, IVariable targetType, bool? saturate)
        => NodeBuilder.BuildNodeSingleOut(CAST_LIKE, [input, targetType], [(AttrSaturate, saturate)]);

    public static IVariable Ceil(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(CEIL, [x], []);

    public static IVariable Celu(IVariable x, float? alpha)
        => NodeBuilder.BuildNodeSingleOut(CELU, [x], [(AttrAlpha, alpha)]);

    public static IVariable CenterCropPad(IVariable input, IVariable shape, long[]? axes)
        => NodeBuilder.BuildNodeSingleOut(CENTER_CROP_PAD, [input, shape], [(AttrAxes, axes)]);

    public static IVariable Clip(IVariable input, IVariable min, IVariable max)
        => NodeBuilder.BuildNodeSingleOut(CLIP, [input, min, max], []);

    public static IVariable Col2Im(IVariable input, IVariable imageShape, IVariable blockShape,
        long[] dilations, long[] pads, long[] strides)
        => NodeBuilder.BuildNodeSingleOut(COL2IM, [input, imageShape, blockShape], 
            [(AttrDilations, dilations), (AttrPads, pads), (AttrStrides, strides)]);

    public static IVariable Compress(IVariable input, IVariable condition, long? axis)
        => NodeBuilder.BuildNodeSingleOut(COMPRESS, [input, condition], [(AttrAxis, axis)]);

    public static IVariable Concat(IVariable[] inputs, long axis)
        => NodeBuilder.BuildNodeSingleOut(CONCAT, inputs, [(AttrAxis, axis)]);

    public static IVariable ConcatFromSequence(IVariable inputSequence, long axis, bool newAxis)
        => NodeBuilder.BuildNodeSingleOut(CONCAT_FROM_SEQUENCE, [inputSequence], [(AttrAxis, axis), (AttrNewAxis, newAxis)]);
    public static IVariable Constant(TensorData value) => Constant(value, null, null, null, null, null, null);
    public static IVariable Constant(float value) => Constant(null, value, null, null, null, null, null);
    public static IVariable Constant(float[] value) => Constant(null, null, value, null, null, null, null);
    public static IVariable Constant(long value) => Constant(null, null, null, value, null, null, null);
    public static IVariable Constant(long[] value) => Constant(null, null, null, null, value, null, null);
    public static IVariable Constant(string value) => Constant(null, null, null, null, null, value, null);
    public static IVariable Constant(string[] value) => Constant(null, null, null, null, null, null, value);

    public static IVariable Constant(TensorData? value, float? valueFloat, float[]? valueFloats,
        long? valueInt, long[]? valueInts, string? valueString, string[]? valueStrings)
        => NodeBuilder.BuildNodeSingleOut(CONSTANT, [], [
            (AttrValue, value),
            (AttrValueFloat, valueFloat),
            (AttrValueFloats, valueFloats),
            (AttrValueInt, valueInt),
            (AttrValueInts, valueInts),
            (AttrValueString, valueString),
            (AttrValueStrings, valueStrings)]);

    public static IVariable ConstantOfShape(IVariable shape, TensorData value, int? rank)
        => Identity(ConstantOfShape(shape, value), rank);

    public static IVariable ConstantOfShape(IVariable shape, TensorData value)
        => NodeBuilder.BuildNodeSingleOut(CONSTANT_OF_SHAPE, [shape], [(AttrValue, value)]);

    public static IVariable Conv(IVariable x, IVariable w, IVariable b, AutoPad autoPad,
        long[] dilations, long group, long[] kernelShape,
        long[]? pads, long[] strides)
        => NodeBuilder.BuildNodeSingleOut(CONV, [x, w, b], [
            (AttrAutoPad, autoPad),
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IVariable ConvInteger(IVariable x, IVariable w, IVariable xZeroPoint, IVariable wZeroPoint,
        AutoPad autoPad, long[]? dilations, long group, long[]? kernelShape, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(CONV_INTEGER, [x, w, xZeroPoint, wZeroPoint], [
            (AttrAutoPad, autoPad),
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IVariable ConvTranspose(IVariable x, IVariable w, IVariable b, AutoPad autoPad, long[]? dilations, long group,
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

    public static IVariable Cos(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(COS, [x], []);

    public static IVariable Cosh(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(COSH, [x], []);

    public static IVariable CumSum(IVariable x, IVariable axis, bool exclusive, bool reverse)
        => NodeBuilder.BuildNodeSingleOut(CUM_SUM, [x, axis], [(AttrExclusive, exclusive), (AttrReverse, reverse)]);

    /// <summary>Scaled dot-product attention returning only Y (ONNX Attention, opset 23+); no KV cache.
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a faithful lowering of
    /// the fused op (causal/GQA/softcap/qk-output-mode variants, plus a proper non-fused autodiff rule)
    /// is intricate enough to belong in core (deferred core work) — so this
    /// throws rather than force a higher model opset. The ATTENTION op definition and QEE kernel are
    /// retained; restore the fused emission here once a runtime supports it at a usable opset.</summary>
    public static IVariable Attention(IVariable q, IVariable k, IVariable v,
        IVariable? attnMask = null, IVariable? nonpadKvSeqlen = null,
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
    public static (IVariable y, IVariable presentKey, IVariable presentValue) AttentionWithKVCache(
        IVariable q, IVariable k, IVariable v,
        IVariable? attnMask = null, IVariable? pastKey = null, IVariable? pastValue = null,
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
    public static IVariable BitCast(IVariable input, DType to)
        => throw new System.NotImplementedException(
            "BitCast (ONNX opset 26) has no opset-21 primitive equivalent (no op reinterprets bit " +
            "patterns), and Shorokoo emits a single opset-21 model — it cannot be lowered. The op " +
            "definition is retained; re-enable the fused emission here when a runtime supports it.");

    /// <summary>Cumulative product along the axis given as a 0-D tensor input (ONNX CumProd, opset 26+).
    /// Not emittable today: there is no opset-21 CumProd, and a faithful general decomposition needs a
    /// Scan (multiply body) which is not yet implemented, so this throws rather than force a higher
    /// model opset. The CUM_PROD op definition and QEE kernel are retained.</summary>
    public static IVariable CumProd(IVariable x, IVariable axis, bool exclusive = false, bool reverse = false)
        => throw new System.NotImplementedException(
            "CumProd (ONNX opset 26) has no opset-21 equivalent (a general decomposition needs a Scan " +
            "multiply body, not yet implemented), and Shorokoo emits a single opset-21 model. The op " +
            "definition is retained; re-enable the fused emission here when a runtime supports it.");
}