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
        private static List<NodeDefinitionMaker> GetACMakers() => [
            Op(ABS)
                .Tensor<NumLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Abs()"),

            Op(ACOS)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Acos()"),

            Op(ACOSH)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Acosh()"),

            Op(ADD)
                .Tensor<NumLike>("T")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} + {2:low_op}"),

            Op(AFFINE_GRID)
                .AttributeBool(AttrAlignCorners)
                .Tensor<FloatLike>("T1")
                .Tensor<int64>("T2")
                .Input("theta", "T1", "R")
                .Input("size", "T2", 1)
                .Output("grid", "T1", "R2")
                .InputTestShapes("theta", [[2, 2, 3], [1, 3, 4]])
                .InputTestValues("size", [TensorData(4, 2L, 3L, 4L, 2L), TensorData(5, 1L, 2L, 3L, 4L, 3L)])
                .Code("NN.AffineGrid({1:param}{2:param}{a:param?})"),

            Op(AND)
                .Tensor<bit>("T")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T", rankBroadcast: "R")
                .Code("{1:low_op} & {2:low_op}"),

            Op(ARG_MAX)
                .AttributeLong(AttrAxis)
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrSelectLastIndex)
                .Tensor<NumLike>("Tin")
                .Tensor<int64>("Tout")

                // Default is 1
                .Constraint(AttrKeepdims, 1)
                .Input("data", "Tin", "R")
                .Output("reduced", "Tout", rank: "R")
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]])
                .Code("{1:this}.ArgMax({a:param?}{b:param?}{c:param?}){o1:torank}")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "Tin", "R")
                .Output("reduced", "Tout", rankMinusOne: "R")
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]])
                .Code("{1:this}.ArgMax({a:param?}{b:param?}{c:param?}){o1:torank}"),

            Op(ARG_MIN)
                .AttributeLong(AttrAxis)
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrSelectLastIndex)
                .Tensor<NumLike>("Tin")
                .Tensor<int64>("Tout")
                
                // Default is 1
                .Constraint(AttrKeepdims, 1)
                .Input("data", "Tin", "R")
                .Output("reduced", "Tout", rank: "R")
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]])
                .Code("{1:this}.ArgMin({a:param?}{b:param?}{c:param?}){o1:torank}")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "Tin", "R")
                .Output("reduced", "Tout", rankMinusOne: "R")
                .AttributeTestValues(AttrAxis, [0L, 1L, 2L])
                .InputTestShapes("data", [[5],[3,4],[2,3,4]])
                .Code("{1:this}.ArgMin({a:param?}{b:param?}{c:param?}){o1:torank}"),

            Op(ASIN)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Asin()"),

            Op(ASINH)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Asinh()"),

            Op(ATAN)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Atan()"),

            // Scaled dot-product attention (opset 23+). Two variants, selected by the
            // internal has-optional-outputs flag (same pattern as MAX_POOL):
            //   1 → KV-cache form: inputs Q, K, V, attn_mask?, past_key?, past_value?;
            //       outputs Y, present_key, present_value (past/present used together per spec).
            //   0 → basic form: output Y only. The spec's positional input list still
            //       reserves slots 4/5 for past_key/past_value (nonpad_kv_seqlen is input
            //       SIX), so both slots are declared here too — but never wired (per spec
            //       past/present must be used together, and nonpad_kv_seqlen must not be
            //       combined with past/present).
            // The optional qk_matmul_output (4th output) is not emitted by this framework;
            // qk_matmul_output_mode is still declared for completeness.
            Op(ATTENTION)
                .Tensor<FloatLike>("T1")        // Q, K, Y, present_key
                .Tensor<FloatLike>("T2")        // V, present_value
                .Tensor<CommonLike>("U")        // attn_mask: bool or numeric (added to scores)
                .Tensor<int64>("T3")            // nonpad_kv_seqlen
                .AttributeBool(AttrIsCausal)                // a
                .AttributeLong(AttrKvNumHeads)              // b (3-D inputs only)
                .AttributeLong(AttrQNumHeads)               // c (3-D inputs only)
                .AttributeLong(AttrQkMatmulOutputMode)      // d
                .AttributeFloat(AttrScale)                  // e (default 1/sqrt(head_size))
                .AttributeFloat(AttrSoftcap)                // f
                .AttributeLong(AttrSoftmaxPrecision)        // g
                .AttributeBool(InternalAttrHasOptionalOutputs)

                .Constraint(InternalAttrHasOptionalOutputs, 1)
                .AttributeTestValues(InternalAttrHasOptionalOutputs, [1L])
                .Input("Q", "T1", "R")
                .Input("K", "T1", "R2")
                .Input("V", "T2", "R3")
                .Input("attn_mask", "U?", "R4")
                .Input("past_key", "T1?", "R5")
                .Input("past_value", "T2?", "R6")
                .InputTestShapes("Q", [[1, 2, 3, 4]])
                .InputTestShapes("K", [[1, 2, 5, 4]])
                .InputTestShapes("V", [[1, 2, 5, 4]])
                .InputTestShapes("attn_mask", [[3, 7]])
                .InputTestShapes("past_key", [[1, 2, 2, 4]])
                .InputTestShapes("past_value", [[1, 2, 2, 4]])
                .Output("Y", "T1", rank: "R")
                .Output("present_key", "T1?", "R5")
                .Output("present_value", "T2?", "R6")

                .Constraint(InternalAttrHasOptionalOutputs, 0)
                .AttributeTestValues(InternalAttrHasOptionalOutputs, [0L])
                .Input("Q", "T1", "R")
                .Input("K", "T1", "R2")
                .Input("V", "T2", "R3")
                .Input("attn_mask", "U?", "R4")
                .Input("past_key", "T1?", "R5")      // positional placeholder — never wired in this variant
                .Input("past_value", "T2?", "R6")    // positional placeholder — never wired in this variant
                .Input("nonpad_kv_seqlen", "T3?", 1)
                .InputTestShapes("Q", [[1, 2, 3, 4]])
                .InputTestShapes("K", [[1, 2, 5, 4]])
                .InputTestShapes("V", [[1, 2, 5, 4]])
                .InputTestShapes("attn_mask", [[3, 5]])
                .InputTestValues("nonpad_kv_seqlen", [TensorData([1], 4L)])
                .Output("Y", "T1", rank: "R"),

            Op(ATANH)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Atanh()"),

            Op(AVERAGE_POOL)
                .Tensor<FloatLike>("T")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])  // a
                .AttributeBool(AttrCeilMode)        // b
                .AttributeBool(AttrCountIncludePad) // c
                .AttributeLongs(AttrDilations)      // d
                .AttributeLongs(AttrKernelShape)    // e
                .AttributeLongs(AttrPads)           // f
                .AttributeLongs(AttrStrides)        // g

                .Constraint(AttrAutoPad, "NOTSET")
                .Input("x", "T", "R")
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L],[2L,2L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L],[3L,3L]])
                .AttributeTestValues(AttrPads, (long[][])[[0L,0L,0L,0L],[1L,1L,1L,1L]])
                .AttributeTestValues(AttrStrides, (long[][])[[2L,2L],[2L,2L]])
                .InputTestShapes("x", [[1L,2L,5L,3L],[1L,3L,2L,5L]])
                .Output("y", "T", rank: "R")
                .Code("Shorokoo.Core.Nodes.NodeDefinitions.OnnxOp.AveragePool({1:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param}{g:param}){o1:fromvar}")

                .ConstraintIsSet(AttrAutoPad, true)
                .Input("x", "T", "R")
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L],[2L,2L]])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L],[3L,3L]])
                .AttributeTestValues(AttrPads, (long[]?[])[null, null])
                .AttributeTestValues(AttrStrides, (long[][])[[2L,2L],[2L,2L]])
                .InputTestShapes("x", [[1L,2L,5L,3L],[1L,3L,2L,5L]])
                .Output("y", "T", rank: "R")
                .Code("Shorokoo.Core.Nodes.NodeDefinitions.OnnxOp.AveragePool({1:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param}{g:param}){o1:fromvar}"),

            Op(BATCH_NORMALIZATION)
                .Tensor<FloatLike>("T")
                .Tensor<FloatLike>("T1")
                .Tensor<FloatLike>("T2")
                .AttributeFloat(AttrEpsilon)
                .AttributeFloat(AttrMomentum)
                .AttributeBool(AttrTrainingMode)

                .Constraint(AttrTrainingMode, 0)
                .Input("x", "T", "R") // x
                .Input("scale", "T1", 1)  // scale
                .Input("b", "T1", 1)  // B
                .Input("inputMean", "T2", 1)  // input mean
                .Input("inputVariance", "T2", 1)  // input var
                .Output("y", "T", rank: "R")  // y
                .InputTestShapes("x", [[3L,2L],[3L,2L,4L]])
                .InputTestShapes("scale", [[2L],[2L]])
                .InputTestShapes("b", [[2L],[2L]])
                .InputTestShapes("inputMean", [[2L],[2L]])
                .InputTestShapes("inputVariance", [[2L],[2L]])
                .Code("{1:this}.BatchNormalization({2:param}{3:param}{4:param}{5:param}{a:param}{b:param}{c:param})")

                .Constraint(AttrTrainingMode, 1)
                .Input("x", "T", "R") // x
                .Input("scale", "T1", 1)  // scale
                .Input("b", "T1", 1)  // B
                .Input("inputMean", "T2", 1)  // input mean
                .Input("inputVariance", "T2", 1)  // input var
                .Output("y", "T", rank: "R")  // y
                .Output("runningMean", "T2", 1)  // running mean
                .Output("runningVariance", "T2", 1) // running var
                .InputTestShapes("x", [[3L,2L],[3L,2L,4L]])
                .InputTestShapes("scale", [[2L],[2L]])
                .InputTestShapes("b", [[2L],[2L]])
                .InputTestShapes("inputMean", [[2L],[2L]])
                .InputTestShapes("inputVariance", [[2L],[2L]])
                .Code("{1:this}.BatchNormalizationFullOuputs({2:param}{3:param}{4:param}{5:param}{a:param}{b:param}{c:param})"),

            Op(BERNOULLI)
                .Tensor<FloatLike>("T1")
                .Tensor<CommonLike>("T2")
                .AttributeDType(AttrDtype, "T2")
                .AttributeFloat(AttrSeed)

                .ConstraintIsSet(AttrDtype, true)
                .Input("input", "T1", "R")
                .Output("output", "T2", "R")
                .Code("{1:this}.Bernoulli<{T2:ivartype}>({b:param})")

                .ConstraintIsSet(AttrDtype, false)
                .Input("input", "T1", "R")
                .Output("output", "T1", "R")
                .Code("{1:this}.Bernoulli({b:param})"),

            // Bitwise reinterpretation (opset 26+). The target dtype must have the same
            // bit-width as the source; output shape always equals the input shape (no
            // last-dim rescaling — the spec only allows same-width targets). String is
            // excluded by the spec; CommonLike covers the numeric + bool element types
            // representable in-framework.
            Op(BIT_CAST)
                .Tensor<CommonLike>("T1")
                .Tensor<CommonLike>("T2")
                .AttributeDType(AttrTo, "T2")
                .Input("input", "T1", "R")
                .Output("output", "T2", "R")
                .InputTestShapes("input", [[2, 3]]),

            Op(BIT_SHIFT)
                .Tensor<UnsignedIntLike>("T")
                .AttributeEnum<BitShiftDirection>(AttrDirection, ["LEFT", "RIGHT"])

                .Constraint(AttrDirection, "LEFT")
                .Input("X", "T", "R1") // X
                .Input("Y", "T", "R2") // Y
                .Output("Z", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} << {2:low_op}")

                .Constraint(AttrDirection, "RIGHT")
                .Input("X", "T", "R1") // X
                .Input("Y", "T", "R2") // Y
                .Output("Z", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} >> {2:low_op}"),

            Op(BITWISE_AND)
                .Tensor<UnsignedIntLike>("T")
                .Input("X", "T", "R1") // X
                .Input("Y", "T", "R2") // Y
                .Output("Z", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} & {2:low_op}"),

            Op(BITWISE_NOT)
                .Tensor<UnsignedIntLike>("T")
                .Input("X", "T", "R") // X
                .Output("Y", "T", "R") // same rank as X (used to be an unconstrained "R2")
                .Code("!{1:low_op}"),

            Op(BITWISE_OR)
                .Tensor<UnsignedIntLike>("T")
                .Input("X", "T", "R1") // X
                .Input("Y", "T", "R2") // Y
                .Output("Z", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} | {2:low_op}"),

            Op(BITWISE_XOR)
                .Tensor<UnsignedIntLike>("T")
                .Input("X", "T", "R1") // X
                .Input("Y", "T", "R2") // Y
                .Output("Z", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} ^ {2:low_op}"),

            Op(BLACKMAN_WINDOW)
                .Tensor<NumLike>("T1")
                .Tensor<IndexLike>("T2")
                .Tensor<float32>("T3")
                .AttributeDType(AttrOutputDatatype, "T1")
                .AttributeBool(AttrPeriodic)

                .ConstraintIsSet(AttrOutputDatatype, true)
                .Input("size", "T2", 0)
                .Output("output", "T1", 1)
                .InputTestValues("size", [TensorData([], 3L), TensorData([], 20L), TensorData([], 6L), TensorData([], 32L)])
                .Code("NN.BlackmanWindow<{T1:ivartype}>({1:param}{b:param})")

                .ConstraintIsSet(AttrOutputDatatype, false)
                .Input("size", "T2", 0)
                .Output("output", "T3", 1)
                .InputTestValues("size", [TensorData([], 3L), TensorData([], 20L), TensorData([], 6L), TensorData([], 32L)])
                .Code("NN.BlackmanWindow<float32>({1:param}{b:param})"),

            Op(CAST)
                .Tensor<AnyLike>("T1")
                .Tensor<AnyLike>("T2")
                // .AttributeBool(AttrSaturate)
                .AttributeDType(AttrTo, "T2")
                // opset-24 round_mode only affects float8e8m0 targets (unsupported
                // dtype); declared so imports tolerate and round-trip it.
                .AttributeString(AttrRoundMode)

                // .Constraint(AttrSaturate, 1)
                .Input("input", "T1", "R")
                .Output("output", "T2", "R")
                .Code("{1:this}.Cast<{T2:ivartype}>()"),

                // .Constraint(AttrSaturate, 0)
                // .Input("", "T1", "R")
                // .Output("", "T2", "R")
                // .Code("{1:this}.Cast<{T2:ivartype}>(false)"),

            Op(CAST_LIKE)
                .Tensor<AnyLike>("T1")
                .Tensor<AnyLike>("T2")
                .AttributeBool(AttrSaturate)
                // opset-24 round_mode: tolerated for import round-trips (float8e8m0 only).
                .AttributeString(AttrRoundMode)
                .Input("input", "T1", "R")
                // target_type only contributes its dtype; its shape/rank is unrelated to
                // the input's (the definition used to force both ranks to "R").
                .Input("target_type", "T2", "R2")
                .Output("output", "T2", "R"),

            Op(CEIL)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Ceiling()"),

            Op(CELU)
                .Tensor<float32>("T")
                .AttributeFloat(AttrAlpha)
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Celu({a:param?})"),

            Op(CENTER_CROP_PAD)
                .AttributeLongs(AttrAxes)
                .Tensor<NumLike>("T")
                .Tensor<int64>("Tind") // Pinned to int64; the ONNX spec also allows int32 indices (restrictive but safe).
                .Input("input_data", "T", "R") // input_data
                .Input("shape", "Tind", "R2") // shape
                .Output("output_data", "T", "R") // output_data
                .AttributeTestValues(AttrAxes, (long[]?[])[null, [-1L, 1L], [0L, 2L]])
                .InputTestValues("input_data", [TensorData([2,3], 1f, 2f, 3f, 4f, 5f, 6f), TensorData([1, 2, 3], 1L, 2L, 3L, 4L, 5L, 6L), TensorData([2,1,3], 1d, 2d, 3d, 4d, 5d, 6d)])
                .InputTestValues("shape", [TensorData([2], 1L, 2L), TensorData([2], 2L, 2L), TensorData([2], 1L, 2L)])
                .Code("{1:this}.CenterCropPad({2:param}{a:param}){o1:torank}"),

            Op(CLIP)
                .Tensor<NumLike>("T")
                .Input("input", "T", "R")
                .Input("min", "T?", 0)
                .Input("max", "T?", 0)
                .Output("output", "T", "R")
                .Code("{1:this}.Clip({2:param}{3:param})"),

            Op(COL2IM)
                .AttributeLongs(AttrDilations)
                .AttributeLongs(AttrPads)
                .AttributeLongs(AttrStrides)
                .Tensor<NumLike>("T")
                .Tensor<int64>("T2")
                .Input("input", "T", "R")
                .Input("image_shape", "T2", 1)
                .Input("block_shape", "T2", 1)
                // Output is [N, C, *image_shape] = rank 2 + len(image_shape), which differs
                // from the 3-D input's rank — it must NOT share the input's "R" token.
                .Output("output", "T", "R3")
                .InputTestShapes("input", [[1, 2, 15]])
                .InputTestValues("image_shape", [TensorData([2], 3L, 5L)])
                .InputTestValues("block_shape", [TensorData([2], 1L, 1L)])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L]])
                .AttributeTestValues(AttrPads, (long[][])[[0L, 0L, 0L, 0L]])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L]])
                .Code("NN.Col2Im({1:param}{2:param}{3:param}{a:param}{b:param}{c:param})"),

            Op(COMPRESS)
                .Tensor<AnyLike>("T")
                .Tensor<bit>("T1")
                .AttributeLong(AttrAxis)

                .ConstraintIsSet(AttrAxis, isSet: true)
                .Input("input", "T", "R")
                .Input("condition", "T1", 1)
                .Output("output", "T", "R")

                .ConstraintIsSet(AttrAxis, isSet: false)
                .Input("input", "T", "R")
                .Input("condition", "T1", 1)
                .Output("output", "T", 1),

            Op(CONCAT)
                .Tensor<AnyLike>("T")
                .Variadic("V", minCount: 1)
                .AttributeLong(AttrAxis)
                .Input("inputs", ["T", "V"], "R")
                .Output("concat_result", "T", rank: "R")
                .Code("{1:this}.Concat({a:param}{#:param}){o1:torank}", inline: true),

            Op(CONCAT_FROM_SEQUENCE)
                .VarType<AnyLike>("T")
                .Structure("S", DataStructure.Sequence)
                .Structure("Tensor", DataStructure.Tensor)
                .AttributeLong(AttrAxis)
                .AttributeBool(AttrNewAxis)
                .Input("inputs_sequence", ["T", "S"], "R")
                // Output rank is element rank (new_axis=0) or element rank + 1 (new_axis=1),
                // so it cannot be tied to the input's element rank var.
                .Output("concat_result", ["T", "Tensor"], rank: "R2")
                .Code("{1:this}.Concat({a:param}{b:param})"),

            Op(CONSTANT)
                .Tensor<float32>("T1")
                .Tensor<int64>("T2")
                .Tensor<@string>("T3")
                .Tensor<AnyLike>("T4")
                .AttributeTensor(AttrValue, "T4", "R")
                // .AttributeSparseTensor(AttrSparseValue)
                .AttributeFloat(AttrValueFloat)
                .AttributeFloats(AttrValueFloats)
                .AttributeLong(AttrValueInt)
                .AttributeLongs(AttrValueInts)
                .AttributeString(AttrValueString)
                .AttributeStrings(AttrValueStrings)

                .ConstraintIsSet(AttrValueFloat, true)
                .Code("Scalar({b:})")
                .Output("value", "T1", 0)

                .ConstraintIsSet(AttrValueFloats, true)
                .Code("Vector({c:params})")
                .Output("value", "T1", 1)

                .ConstraintIsSet(AttrValueInt, true)
                .Code("Scalar({d:})")
                .Output("value", "T2", 0)

                .ConstraintIsSet(AttrValueInts, true)
                .Code("Vector({e:params})")
                .Output("value", "T2", 1)

                .ConstraintIsSet(AttrValueString, true)
                .Code("Scalar({f:})")
                .Output("value", "T3", 0)

                .ConstraintIsSet(AttrValueStrings, true)
                .Code("Vector({g:})")
                .Output("value", "T3", 1)

                .ConstraintIsSet(AttrValue, true)
                .Code("MakeTensor<{T4:ivartype}>({a:dims}, \"{a:base64string}\"){o1:torank}")
                .AttributeTestValues(AttrValue, [TensorData([2,3], 1f, 2f, 3f, 4f, 5f, 6f)])
                .Output("value", "T4", rank: "R"),

            Op(CONSTANT_OF_SHAPE)
                .Tensor<int64>("T1")
                .Tensor<float32>("T2")
                .Tensor<AnyLike>("T3")
                .AttributeTensor(AttrValue, "T3", "R")

                .ConstraintIsSet(AttrValue, false)
                .Input("shape", "T1", 1)
                .Output("value", "T2", rank: "R2")
                .InputTestValues("shape", [TensorData([3], 1L, 2L, 3L), TensorData([2], 2L, 3L), TensorData([1], 3L), TensorData([1], 3L), TensorData([0], (long[])[])])
                .Code("TensorFill({1:param}0f){o1:torank}")

                .ConstraintIsSet(AttrValue, true)
                .Input("shape", "T1", 1)
                .Output("value", "T3", rank: "R2")
                .InputTestValues("shape", [TensorData([1], 1L, 2L, 3L), TensorData([1], 2L, 3L), TensorData([1], 3L), TensorData([1], 3L), TensorData([0L], (long[])[])])
                .AttributeTestValues(AttrValue, [TensorData([1], 1f), TensorData([1], 2L), TensorData([1], -1.01d), TensorData([1], (uint)2), TensorData([1], true)])
                .Code("Tensor<{T3:ivartype}>.Fill({1:param}{a:param})"),

            Op(CONV)
                .Tensor<FloatLike>("T")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .AttributeLongs(AttrDilations)
                .AttributeLong(AttrGroup)
                .AttributeLongs(AttrKernelShape)
                .AttributeLongs(AttrPads)
                .AttributeLongs(AttrStrides)
                .Input("X", "T", "R")
                .Input("W", "T", "R2")
                .Input("B", "T?", 1)
                .Output("Y", "T", "R")
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,5L,5L]])
                .InputTestShapes("W", [[1L,1L,2L,2L],[1L,1L,3L,3L]])
                .InputTestShapes("B", [[1L],[1L]])
                .AttributeTestValues(AttrAutoPad, ["NOTSET", "SAME_UPPER"])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L], [1L,1L]])
                .AttributeTestValues(AttrGroup, [1L, 1L])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L], [3L,3L]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0L,0L,0L,0L], null])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L], [1L,1L]])
                .Code("NN.Conv({1:param}{2:param}{3:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param})"),

            Op(CONV_INTEGER)
                .Tensor<Int8Like>("T1")
                .Tensor<Int8Like>("T2")
                .Tensor<int32>("T3")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .AttributeLongs(AttrDilations)
                .AttributeLong(AttrGroup)
                .AttributeLongs(AttrKernelShape)
                .AttributeLongs(AttrPads)
                .AttributeLongs(AttrStrides)
                .Input("X", "T1", "R")
                .Input("W", "T2", "R2")
                .Input("x_zero_point", "T1?", 0)
                .Input("w_zero_point", "T2?", 0)
                .Output("Y", "T3", "R")
                .InputTestShapes("X", [[1L,1L,5L,5L],[1L,1L,5L,5L]])
                .InputTestShapes("W", [[1L,1L,2L,2L],[1L,1L,3L,3L]])
                .AttributeTestValues(AttrAutoPad, ["NOTSET", "SAME_UPPER"])
                .AttributeTestValues(AttrDilations, (long[][])[[1L,1L], [1L,1L]])
                .AttributeTestValues(AttrGroup, [1L, 1L])
                .AttributeTestValues(AttrKernelShape, (long[][])[[2L,2L], [3L,3L]])
                .AttributeTestValues(AttrPads, (long[]?[])[[0L,0L,0L,0L], null])
                .AttributeTestValues(AttrStrides, (long[][])[[1L,1L], [1L,1L]])
                .Code("NN.ConvInteger({1:param}{2:param}{3:param}{4:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param})"),

            Op(CONV_TRANSPOSE)
                .Tensor<FloatLike>("T")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .AttributeLongs(AttrDilations)
                .AttributeLong(AttrGroup)
                .AttributeLongs(AttrKernelShape)
                .AttributeLongs(AttrOutputPadding)
                .AttributeLongs(AttrOutputShape)
                .AttributeLongs(AttrPads)
                .AttributeLongs(AttrStrides)
                .Input("X", "T", "R")
                .Input("W", "T", "R2")
                .Input("B", "T?", 1)
                .Output("Y", "T", "R")

                .InputTestShapes("X", [[1L, 1L, 5L, 5L], [1L, 1L, 5L, 5L]])  // Input Tensor (N, C, H, W)
                .InputTestShapes("W", [[1L, 1L, 2L, 2L], [1L, 1L, 3L, 3L]])  // Weight Tensor (C, M/group, kH, kW)
                .InputTestShapes("B", [[1L], [1L]])  // Bias Tensor (M)
                .AttributeTestValues(AttrAutoPad, ["NOTSET"]) // Padding mode
                .AttributeTestValues(AttrDilations, (long[][])[[1L, 1L]])   // No dilation
                .AttributeTestValues(AttrGroup, [1L])                   // Standard convolution (no grouping)
                .AttributeTestValues(AttrKernelShape, (long[][])[[3L, 3L]]) // Kernel sizes
                .AttributeTestValues(AttrOutputPadding, (long[][])[[0,0,0,0]])  // Controls extra padding in output
                .AttributeTestValues(AttrOutputShape, (long[][])[[11,11]])  // Explicit output shapes
                .AttributeTestValues(AttrPads, (long[]?[])[[0,0,0,0]])    // Explicit padding for "NOTSET"
                .AttributeTestValues(AttrStrides, (long[][])[[2L, 2L]])     // Added strides

                .Code("NN.ConvTranspose({1:param}{2:param}{3:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param}{g:param}{h:param})"),

            Op(COS)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Cos()", inline: true),

            Op(COSH)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Cosh()", inline: true),

            // Cumulative product along an axis (opset 26+). Mirrors CUM_SUM: axis is a
            // 0-D tensor input; exclusive/reverse attributes change the window.
            Op(CUM_PROD)
                .Tensor<NumLike>("T")
                .Tensor<IndexLike>("T2")
                .AttributeBool(AttrExclusive)
                .AttributeBool(AttrReverse)
                .Input("x", "T", "R")
                .Input("axis", "T2", 0)
                .InputTestShapes("x", [[5], [3, 4], [2, 3, 4]])
                .InputTestValues("axis", [TensorData([], 0L), TensorData([], 1L), TensorData([], 2L)])
                .Output("y", "T", rank: "R"),

            Op(CUM_SUM)
                .Tensor<NumLike>("T")
                .Tensor<IndexLike>("T2")
                .AttributeBool(AttrExclusive)
                .AttributeBool(AttrReverse)
                .Input("x", "T", "R")
                .Input("axis", "T2?", 0)
                .InputTestShapes("x", [[5],[3,4],[2,3,4]])
                .InputTestValues("axis", [TensorData([], 0L), TensorData([], 1L), TensorData([], 2L)])
                .Output("result", "T", rank: "R")
                .Code("{1:this}.CumSum({2:param}{a:param}{b:param}){o1:torank}"),
        ];
    }
} 