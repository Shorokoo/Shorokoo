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
    public static Variable Dft(Variable x, Variable? length, Variable? axis, bool? inverse, bool? onesided = null)
    {
        // ORT 1.25/1.26 segfaults when the DFT node's axis input is omitted (the
        // empty-string placeholder in the protobuf), even though the ONNX spec
        // says the default is -2. Substitute the spec default explicitly so the
        // emitted graph never carries a null axis input.
        axis ??= Shorokoo.Globals.Scalar(-2L);
        return NodeBuilder.BuildNodeSingleOut(DFT, [x, length, axis], [(AttrInverse, inverse), (AttrOnesided, onesided)]);
    }

    public static Variable DeformConv(Variable x, Variable w, Variable offset, Variable? b, Variable? mask,
        long[]? dilations, long? group, long[]? kernelShape, long? offsetGroup,
        long[]? pads, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(DEFORM_CONV, [x, w, offset, b, mask], [
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrOffsetGroup, offsetGroup),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static Variable DepthToSpace(Variable input, long? blockSize, DepthColumnRowMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(DEPTH_TO_SPACE, [input], [(AttrBlocksize, blockSize), (AttrMode, mode)]);

    public static Variable DequantizeLinear(Variable x, Variable xScale, Variable? xZeroPoint,
        long? axis, long? blockSize = null)
        => NodeBuilder.BuildNodeSingleOut(DEQUANTIZE_LINEAR, [x, xScale, xZeroPoint], 
            [(AttrAxis, axis), (AttrBlockSize, blockSize)]);

    public static Variable Det(Variable x)
        => NodeBuilder.BuildNodeSingleOut(DET, [x], []);

    public static Variable Div(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(DIV, [left, right], []);

    public static (Variable output, Variable? mask) Dropout(Variable data, Variable? ratio, Variable? training_mode,
        long? seed = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(DROPOUT, [data, ratio, training_mode], [(AttrSeed, seed)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null);
    }

    // public static Variable Dropout(Variable data, Variable? ratio, Variable? training_mode,
    //     long? seed = null)
    // {
    //     var retval = NodeBuilder.BuildNodeMultiOut(DROPOUT, [data, ratio, training_mode], [(AttrSeed, seed)], outputNames: [null, ""]);
    //     return (retval[0], retval.Length > 1 ? retval[1] : null);
    // }

    public static (Variable y, Variable yScale, Variable yZeroPoint) DynamicQuantizeLinear(Variable x)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(DYNAMIC_QUANTIZE_LINEAR, [x], []);
        return (retval[0], retval[1], retval[2]);
    }

    public static Variable Einsum(Variable[] inputs, string? equation = null)
        => NodeBuilder.BuildNodeSingleOut(EINSUM, inputs, [(AttrEquation, equation)]);

    public static Variable Elu(Variable x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(ELU, [x], [(AttrAlpha, alpha)]);

    public static Variable Equal(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(EQUAL, [left, right], []);

    public static Variable Erf(Variable x)
        => NodeBuilder.BuildNodeSingleOut(ERF, [x], []);

    public static Variable Exp(Variable x)
        => NodeBuilder.BuildNodeSingleOut(EXP, [x], []);

    public static Variable Expand(Variable input, Variable shape)
        => NodeBuilder.BuildNodeSingleOut(EXPAND, [input, shape], []);

    public static Variable EyeLike(Variable input, DType? dtype = null, long? k = 0)
        => NodeBuilder.BuildNodeSingleOut(EYE_LIKE, [input], [(AttrDtype, dtype), (AttrK, k)]);

    public static Variable Flatten(Variable input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(FLATTEN, [input], [(AttrAxis, axis)]);

    public static Variable Floor(Variable x)
        => NodeBuilder.BuildNodeSingleOut(FLOOR, [x], []);

    public static (Variable y, Variable yH) Gru(Variable x, Variable w, Variable r, 
        Variable? b, Variable? sequenceLens, Variable? initialH,
        float[]? activationAlpha, float[]? activationBeta, string[]? activations,
        float? clip, GRUDirection? direction, long? hiddenSize,
        bool? layout, bool? linearBeforeReset = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(GRU, [x, w, r, b, sequenceLens, initialH], [
            (AttrActivationAlpha, activationAlpha),
            (AttrActivationBeta, activationBeta),
            (AttrActivations, activations),
            (AttrClip, clip),
            (AttrDirection, direction),
            (AttrHiddenSize, hiddenSize),
            (AttrLayout, layout),
            (AttrLinearBeforeReset, linearBeforeReset)]);
        return (retval[0], retval[1]);
    }

    public static Variable Gather(Variable data, Variable indices, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER, [data, indices], [(AttrAxis, axis)]);

    public static Variable GatherElements(Variable data, Variable indices, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER_ELEMENTS, [data, indices], [(AttrAxis, axis)]);

    public static Variable GatherND(Variable data, Variable indices, long? batchDims = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER_ND, [data, indices], [(AttrBatchDims, batchDims)]);

    public static Variable Gelu(Variable x, GeluApproximate? approximate = null)
        => NodeBuilder.BuildNodeSingleOut(GELU, [x], [(AttrApproximate, approximate)]);

    public static Variable GlobalAveragePool(Variable x)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_AVERAGE_POOL, [x], []);

    public static Variable GlobalLpPool(Variable x, long? p = null)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_LP_POOL, [x], [(AttrP, p)]);

    public static Variable GlobalMaxPool(Variable x)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_MAX_POOL, [x], []);

    public static Variable Greater(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(GREATER, [left, right], []);

    public static Variable GreaterOrEqual(Variable left, Variable right)
        => NodeBuilder.BuildNodeSingleOut(GREATER_OR_EQUAL, [left, right], []);

    public static Variable GridSample(Variable x, Variable grid, bool? alignCorners,
        GridSampleMode? mode, GridSamplePaddingMode? paddingMode = null)
        => NodeBuilder.BuildNodeSingleOut(GRID_SAMPLE, [x, grid], [
            (AttrAlignCorners, alignCorners),
            (AttrMode, mode),
            (AttrPaddingMode, paddingMode)]);

    public static Variable GroupNormalization(Variable x, Variable scale, Variable bias,
        float? epsilon, long? numGroups, long? stashType = null)
        => NodeBuilder.BuildNodeSingleOut(GROUP_NORMALIZATION, [x, scale, bias], [
            (AttrEpsilon, epsilon),
            (AttrNumGroups, numGroups),
            (AttrStashType, stashType)]);

    public static Variable Gemm(Variable a, Variable b, Variable? c = null,
        float? alpha = null, float? beta = null, long? transA = null, long? transB = null)
        => NodeBuilder.BuildNodeSingleOut(GEMM, [a, b, c], [
            (AttrAlpha, alpha),
            (AttrBeta, beta),
            (AttrTransA, transA),
            (AttrTransB, transB)]);

    public static Variable InstanceNormalization(Variable x, Variable scale, Variable bias,
        float? epsilon = null)
        => NodeBuilder.BuildNodeSingleOut(INSTANCE_NORMALIZATION, [x, scale, bias], [
            (AttrEpsilon, epsilon)]);
}