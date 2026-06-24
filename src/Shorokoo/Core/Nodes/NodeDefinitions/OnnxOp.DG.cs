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
    public static IVariable Dft(IVariable x, IVariable? length, IVariable? axis, bool? inverse, bool? onesided = null)
    {
        // ORT 1.25/1.26 segfaults when the DFT node's axis input is omitted (the
        // empty-string placeholder in the protobuf), even though the ONNX spec
        // says the default is -2. Substitute the spec default explicitly so the
        // emitted graph never carries a null axis input.
        axis ??= Shorokoo.Globals.Scalar(-2L);
        return NodeBuilder.BuildNodeSingleOut(DFT, [x, length, axis], [(AttrInverse, inverse), (AttrOnesided, onesided)]);
    }

    public static IVariable DeformConv(IVariable x, IVariable w, IVariable offset, IVariable? b, IVariable? mask,
        long[]? dilations, long? group, long[]? kernelShape, long? offsetGroup,
        long[]? pads, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(DEFORM_CONV, [x, w, offset, b, mask], [
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrOffsetGroup, offsetGroup),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IVariable DepthToSpace(IVariable input, long? blockSize, DepthColumnRowMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(DEPTH_TO_SPACE, [input], [(AttrBlocksize, blockSize), (AttrMode, mode)]);

    public static IVariable DequantizeLinear(IVariable x, IVariable xScale, IVariable? xZeroPoint,
        long? axis, long? blockSize = null)
        => NodeBuilder.BuildNodeSingleOut(DEQUANTIZE_LINEAR, [x, xScale, xZeroPoint], 
            [(AttrAxis, axis), (AttrBlockSize, blockSize)]);

    public static IVariable Det(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(DET, [x], []);

    public static IVariable Div(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(DIV, [left, right], []);

    public static (IVariable output, IVariable? mask) Dropout(IVariable data, IVariable? ratio, IVariable? training_mode,
        long? seed = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(DROPOUT, [data, ratio, training_mode], [(AttrSeed, seed)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null);
    }

    // public static IVariable Dropout(IVariable data, IVariable? ratio, IVariable? training_mode,
    //     long? seed = null)
    // {
    //     var retval = NodeBuilder.BuildNodeMultiOut(DROPOUT, [data, ratio, training_mode], [(AttrSeed, seed)], outputNames: [null, ""]);
    //     return (retval[0], retval.Length > 1 ? retval[1] : null);
    // }

    public static (IVariable y, IVariable yScale, IVariable yZeroPoint) DynamicQuantizeLinear(IVariable x)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(DYNAMIC_QUANTIZE_LINEAR, [x], []);
        return (retval[0], retval[1], retval[2]);
    }

    public static IVariable Einsum(IVariable[] inputs, string? equation = null)
        => NodeBuilder.BuildNodeSingleOut(EINSUM, inputs, [(AttrEquation, equation)]);

    public static IVariable Elu(IVariable x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(ELU, [x], [(AttrAlpha, alpha)]);

    public static IVariable Equal(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(EQUAL, [left, right], []);

    public static IVariable Erf(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(ERF, [x], []);

    public static IVariable Exp(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(EXP, [x], []);

    public static IVariable Expand(IVariable input, IVariable shape)
        => NodeBuilder.BuildNodeSingleOut(EXPAND, [input, shape], []);

    public static IVariable EyeLike(IVariable input, DType? dtype = null, long? k = 0)
        => NodeBuilder.BuildNodeSingleOut(EYE_LIKE, [input], [(AttrDtype, dtype), (AttrK, k)]);

    public static IVariable Flatten(IVariable input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(FLATTEN, [input], [(AttrAxis, axis)]);

    public static IVariable Floor(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(FLOOR, [x], []);

    public static (IVariable y, IVariable yH) Gru(IVariable x, IVariable w, IVariable r, 
        IVariable? b, IVariable? sequenceLens, IVariable? initialH,
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

    public static IVariable Gather(IVariable data, IVariable indices, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER, [data, indices], [(AttrAxis, axis)]);

    public static IVariable GatherElements(IVariable data, IVariable indices, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER_ELEMENTS, [data, indices], [(AttrAxis, axis)]);

    public static IVariable GatherND(IVariable data, IVariable indices, long? batchDims = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER_ND, [data, indices], [(AttrBatchDims, batchDims)]);

    public static IVariable Gelu(IVariable x, GeluApproximate? approximate = null)
        => NodeBuilder.BuildNodeSingleOut(GELU, [x], [(AttrApproximate, approximate)]);

    public static IVariable GlobalAveragePool(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_AVERAGE_POOL, [x], []);

    public static IVariable GlobalLpPool(IVariable x, long? p = null)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_LP_POOL, [x], [(AttrP, p)]);

    public static IVariable GlobalMaxPool(IVariable x)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_MAX_POOL, [x], []);

    public static IVariable Greater(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(GREATER, [left, right], []);

    public static IVariable GreaterOrEqual(IVariable left, IVariable right)
        => NodeBuilder.BuildNodeSingleOut(GREATER_OR_EQUAL, [left, right], []);

    public static IVariable GridSample(IVariable x, IVariable grid, bool? alignCorners,
        GridSampleMode? mode, GridSamplePaddingMode? paddingMode = null)
        => NodeBuilder.BuildNodeSingleOut(GRID_SAMPLE, [x, grid], [
            (AttrAlignCorners, alignCorners),
            (AttrMode, mode),
            (AttrPaddingMode, paddingMode)]);

    public static IVariable GroupNormalization(IVariable x, IVariable scale, IVariable bias,
        float? epsilon, long? numGroups, long? stashType = null)
        => NodeBuilder.BuildNodeSingleOut(GROUP_NORMALIZATION, [x, scale, bias], [
            (AttrEpsilon, epsilon),
            (AttrNumGroups, numGroups),
            (AttrStashType, stashType)]);

    public static IVariable Gemm(IVariable a, IVariable b, IVariable? c = null,
        float? alpha = null, float? beta = null, long? transA = null, long? transB = null)
        => NodeBuilder.BuildNodeSingleOut(GEMM, [a, b, c], [
            (AttrAlpha, alpha),
            (AttrBeta, beta),
            (AttrTransA, transA),
            (AttrTransB, transB)]);

    public static IVariable InstanceNormalization(IVariable x, IVariable scale, IVariable bias,
        float? epsilon = null)
        => NodeBuilder.BuildNodeSingleOut(INSTANCE_NORMALIZATION, [x, scale, bias], [
            (AttrEpsilon, epsilon)]);
}