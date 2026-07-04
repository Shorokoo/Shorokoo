using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    internal static partial class Definitions
    {
        private static List<NodeDefinitionMaker> GetDGMakers() => [
            Op(DFT)
                .Tensor<FloatLike>("T1")
                .Tensor<IndexLike>("T2")
                .Tensor<int64>("T3")
                .AttributeBool(AttrInverse)
                .AttributeBool(AttrOnesided)
                .Input("input", "T1", "R")
                .Input("dft_length", "T2?", 0)
                .Input("axis", "T3?", 0)
                // Output rank equals the input rank per spec (the axis dim is resized and the
                // trailing complex dim becomes 2, but no dims are added or removed).
                .Output("output", "T1", rank: "R")
                .Code("NN.Dft({1:param}{2:param}{3:param}{a:param}{b:param})"),

            Op(DEFORM_CONV)
                .Tensor<FloatLike>("T")
                .AttributeLongs(AttrDilations)
                .AttributeLong(AttrGroup)
                .AttributeLongs(AttrKernelShape)
                .AttributeLong(AttrOffsetGroup)
                .AttributeLongs(AttrPads)
                .AttributeLongs(AttrStrides)
                .Input("X", "T", "R")   // X
                .Input("W", "T", "R2")  // W
                .Input("offset", "T", "R3")  // offset
                .Input("B", "T?", 1)  // B (optional per spec)
                .Input("mask", "T?", "R4")  // mask (optional per spec)
                .Output("Y", "T", "R")
                .Code("NN.DeformConv({1:param}{2:param}{3:param}{4:param}{5:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param})"),

            Op(DEPTH_TO_SPACE)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrBlocksize)
                .AttributeEnum<DepthColumnRowMode>(AttrMode, ["DCR", "CRD"])
                .Input("input", "T", 4)
                .Output("output", "T", 4)
                .Code("{1:this}.DepthToSpace({a:param}{b:param})"),

            Op(DEQUANTIZE_LINEAR)
                .Tensor<AnyIntLike>("T1")
                .Tensor<FloatLike>("T2")
                .AttributeLong(AttrAxis)
                .AttributeLong(AttrBlockSize)
                // opset-23 output_dtype: declared as a plain int so imported models
                // setting it resolve cleanly; the QEE dtype rule honors it. The def-level
                // output group stays bound to x_scale (they only diverge for float8/4
                // dtypes, which Shorokoo does not support).
                .AttributeLong(AttrOutputDtype)

                .ConstraintIsSet(AttrBlockSize, true)
                .Input("x", "T1", "R") // x
                .Input("x_scale", "T2", "R3") // x_scale
                .Input("x_zero_point", "T1?", "R3") // x_zero_point
                .Output("output", "T2", "R")
                .Code("NN.DequantizeLinear({1:param}{2:param}{3:param}{a:param}{b:param})")

                .ConstraintIsSet(AttrAxis, true)
                .ConstraintIsSet(AttrBlockSize, false)
                .Input("x", "T1", "R") // x
                .Input("x_scale", "T2", 1) // x_scale
                .Input("x_zero_point", "T1?", 1) // x_zero_point
                .Output("output", "T2", "R")
                .Code("NN.DequantizeLinear({1:param}{2:param}{3:param}{a:param}{b:param})")

                .ConstraintIsSet(AttrAxis, false)
                .ConstraintIsSet(AttrBlockSize, false)
                .Input("x", "T1", "R") // x
                .Input("x_scale", "T2", "R2") // x_scale (rank can be 0 or 1)
                .Input("x_zero_point", "T1?", "R2") // x_zero_point (rank can be 0 or 1)
                .Output("output", "T2", "R")
                .Code("NN.DequantizeLinear({1:param}{2:param}{3:param}{a:param}{b:param})"),

            Op(DET)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", rankMinusTwo: "R")
                .Code("NN.DeterminantMatrix({1:param})"),

            Op(DIV)
                .Tensor<NumLike>("T")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T", rankBroadcast: "R")
                .Code("{1:low_op} / {2:low_op}"),

            Op(DROPOUT)
                .AttributeLong(AttrSeed)
                .Tensor<FloatLike>("T")
                .Tensor<FloatLike>("T1")
                .Tensor<bit>("T2")
                .Input("data", "T", "R")
                .Input("ratio", "T1?", 0)
                .Input("training_mode", "T2?", 0)
                .Output("output", "T", "R")
                .Output("mask", "T2?", "R")
                .Code("{1:this}.Dropout({2:param}{3:param})"),

            Op(DYNAMIC_QUANTIZE_LINEAR)
                .Tensor<float32>("T1")
                .Tensor<uint8>("T2")
                .Input("x", "T1", "R") // x
                .Output("y", "T2", "R") // y
                .Output("y_scale", "T1", 0)  // y_scale
                .Output("y_zero_point", "T2?", 0)
                .Code("NN.DynamicQuantizeLinear({1:param})"),

            Op(EINSUM)
                .AttributeString(AttrEquation)
                .Tensor<NumLike>("T")
                .Variadic("V", minCount: 1)
                .Input("", ["T", "V"]) // input
                .Output("", "T"), // output

            Op(ELU)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Elu({a:param})"),

            Op(EQUAL)
                .Tensor<AnyLike>("T")
                .Tensor<bit>("T2")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T2", rankBroadcast: "R")
                .Code("{1:low_op} == {2:low_op}"),

            Op(ERF)
                .Tensor<NumLike>("T")
                .Input("input", "T", "R")
                .Output("output", "T", "R")
                .Code("{1:this}.Erf()"),

            Op(EXP)
                .Tensor<FloatLike>("T")
                .Input("input", "T", "R")
                .Output("output", "T", "R")
                .Code("{1:this}.Exp()"),

            Op(EXPAND)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("T2")
                .Input("input", "T", "R")
                .Input("shape", "T2", 1)
                .Output("output", "T", "R3")
                .Code("{1:this}.Expand({2:param})"),

            Op(EYE_LIKE)
                .Tensor<AnyLike>("T1")
                .Tensor<AnyLike>("T2")
                .AttributeDType(AttrDtype, "T2")
                .AttributeLong(AttrK)

                .ConstraintIsSet(AttrDtype, true)
                .Input("input", "T1", 2)
                .Output("output", "T2", 2)
                .Code("NN.EyeLike<{T2:ivartype}>({1:param}{b:param})")

                .ConstraintIsSet(AttrDtype, false)
                .Input("input", "T1", 2)
                .Output("output", "T1", 2)
                .Code("NN.EyeLike({1:param}{b:param})"),

            Op(FLATTEN)
                .Tensor<AnyLike>("T")
                .AttributeLong(AttrAxis)
                .Input("input", "T", "R")
                .Output("output", "T", 2)
                .Code("{1:this}.Flatten({a:param})"),

            Op(FLOOR)
                .Tensor<FloatLike>("T")
                .Input("input", "T", "R")
                .Output("output", "T", "R")
                .Code("{1:this}.Floor()"),

            Op(GRU)
                .Tensor<FloatLike>("T")
                .Tensor<int32>("T1")
                .AttributeFloats(AttrActivationAlpha)
                .AttributeFloats(AttrActivationBeta)
                .AttributeStrings(AttrActivations)
                .AttributeFloat(AttrClip)
                .AttributeEnum<GRUDirection>(AttrDirection, ["forward", "reverse", "bidirectional"])
                .AttributeLong(AttrHiddenSize)
                .AttributeBool(AttrLayout)
                .AttributeBool(AttrLinearBeforeReset)
                .Input("X", "T", 3)           // X: [seq_length, batch_size, input_size]
                .Input("W", "T", 3)           // W: [num_directions, 3*hidden_size, input_size]
                .Input("R", "T", 3)           // R: [num_directions, 3*hidden_size, hidden_size]
                .Input("B", "T?", 2)          // B: [num_directions, 6*hidden_size]
                .Input("sequence_lens", "T1?", 1)          // sequence_lens: [batch_size]
                .Input("initial_h", "T?", 3)          // initial_h: [num_directions, batch_size, hidden_size]
                .Output("Y", "T?", 4)         // Y: [seq_length, num_directions, batch_size, hidden_size]
                .Output("Y_h", "T?", 3),        // Y_h: [num_directions, batch_size, hidden_size]
            
            Op(GATHER)
                .Tensor<AnyLike>("T")
                .Tensor<IndexLike>("Tind")
                .AttributeLong(AttrAxis)
                .Input("data", "T", "R1")  // Data
                .Input("indices", "Tind", "R2") // Indicess
                .Output("output", "T", "R3"),

            Op(GATHER_ELEMENTS)
                .Tensor<AnyLike>("T")
                .Tensor<IndexLike>("Tind")
                .AttributeLong(AttrAxis)
                .Input("data", "T", "R1") // Data
                .Input("indices", "Tind", "R2") // Indices (same rank as data per spec)
                .Output("output", "T", "R2"), // output: same shape (and rank) as indices

            Op(GATHER_ND)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("Tind")
                .AttributeLong(AttrBatchDims)
                .Input("data", "T", "R1") // Data
                .Input("indices", "Tind", "R2") // Indices
                .Output("output", "T", "R3"),

            Op(GELU)
                .Tensor<FloatLike>("T")
                .AttributeEnum<GeluApproximate>(AttrApproximate, ["none", "tanh"])
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("{1:this}.Gelu({a:param})"),

            Op(GLOBAL_AVERAGE_POOL)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("NN.GlobalAveragePool({1:param})"),

            Op(GLOBAL_LP_POOL)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrP)
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("NN.GlobalLpPool({1:param}{a:param})"),

            Op(GLOBAL_MAX_POOL)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("NN.GlobalMaxPool({1:param})"),

            Op(GREATER)
                .Tensor<NumLike>("T")
                .Tensor<bit>("T2")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T2", rankBroadcast: "R")
                .Code("{1:low_op} > {2:low_op}"),

            Op(GREATER_OR_EQUAL)
                .Tensor<NumLike>("T")
                .Tensor<bit>("T2")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T2", rankBroadcast: "R")
                .Code("{1:low_op} >= {2:low_op}"),

            Op(GRID_SAMPLE)
                .Tensor<CommonLike>("T1")          // Input X can be any tensor type
                .Tensor<FloatLike>("T2")        // Grid must be float type
                .AttributeBool(AttrAlignCorners)
                .AttributeEnum<GridSampleMode>(AttrMode, ["linear", "nearest", "cubic"])
                .AttributeEnum<GridSamplePaddingMode>(AttrPaddingMode, ["zeros", "border", "reflection"])
                .Input("X", "T1", "R")
                .Input("grid", "T2", "R2")
                // Y is [N, C, *grid spatial dims] — same rank as X (and as grid). Used to be
                // declared rankMinusOne of the grid's rank, i.e. one too low.
                .Output("Y", "T1", "R")
                .Code("NN.GridSample({1:param}{2:param}{b:param}{c:param}{a:param})"),

            Op(GROUP_NORMALIZATION)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrEpsilon)
                .AttributeLong(AttrNumGroups)
                .AttributeLong(AttrStashType)
                .Input("X", "T", "R")           // X: [N, C, D1, D2, ..., Dn]
                .Input("scale", "T", 1)            // scale: [C]
                .Input("bias", "T", 1)            // bias: [C]
                .Output("Y", "T", "R")        // Y: same shape as X
                .Code("NN.GroupNormalization({1:param}{2:param}{3:param}{c:param}{b:param}{a:param})"),

            Op(GEMM)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .AttributeFloat(AttrBeta)
                .AttributeLong(AttrTransA)
                .AttributeLong(AttrTransB)
                .Input("A", "T", "R")
                .Input("B", "T", "R2")
                .Input("C", "T?", "R3")
                .Output("Y", "T", "R4")
                .Code("NN.Gemm({1:param}{2:param}{3:param}{a:param}{b:param}{c:param}{d:param})"),
        ];
    }
}