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
        private static List<NodeDefinitionMaker> GetMQMakers() => [
            Op(MATMUL)
                .Tensor<FloatLike>("T")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T", rankBroadcast: "R")
                .Code("{1:this}.MatMul({2:param})"),

            Op(MATMUL_INTEGER)
                .Tensor<Int8Like>("T1")
                .Tensor<Int8Like>("T2")
                .Tensor<int32>("T3")
                .Input("a", "T1", "R1") // A (rank independent of B's per matmul semantics)
                .Input("b", "T2", "R2") // B
                .Input("a_zero_point", "T1?", "R3") // a_zero_point (optional per spec)
                .Input("b_zero_point", "T2?", "R4") // b_zero_point (optional per spec)
                .Output("c", "T3", rankBroadcast: "R")
                .Code("NN.MatMulInteger({1:param}{2:param}{3:param}{4:param})"),

            Op(MAX)
                .Tensor<NumLike>("T")
                .Variadic("V", minCount: 1)
                .Input("x", ["T", "V"], "R")
                .Output("y", "T", rankBroadcast: "R")
                .Code("NN.Max({#:param})"),

            Op(MAX_POOL)
                .Tensor<FloatLike>("T")
                .Tensor<int64>("T2")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .AttributeBool(AttrCeilMode)
                .AttributeLongs(AttrDilations)
                .AttributeLongs(AttrKernelShape)
                .AttributeLongs(AttrPads)
                .AttributeLong(AttrStorageOrder)
                .AttributeLongs(AttrStrides)
                .AttributeBool(InternalAttrHasOptionalOutputs)

                .Constraint(InternalAttrHasOptionalOutputs, 1)
                .Input("X", "T", "R")  // X
                .Output("Y", "T", rank: "R") // Y
                .Output("Indices", "T2?", rank: "R2") // Indices (int64 per spec)
                .Code("NN.MaxPoolWithIndices({{1:param}{c:param}{d:param}{e:param}{f:param}{g:param}{h:param}{b:param})")

                .Constraint(InternalAttrHasOptionalOutputs, 0)
                .Input("X", "T", "R")  // X
                .Output("Y", "T", rank: "R") // Y
                .Output("Indices", "T2?", rank: "R2") // Indices (int64 per spec)
                .Code("NN.MaxPool({{1:param}{c:param}{d:param}{e:param}{f:param}{g:param}{h:param}{b:param})"),

            Op(MAX_UNPOOL)
                .Tensor<FloatLike>("T")
                .Tensor<int64>("T2")
                .AttributeLongs(AttrKernelShape)    // a
                .AttributeLongs(AttrPads)           // b
                .AttributeLongs(AttrStrides)        // c
                .Input("X", "T", "R")
                .Input("I", "T2", "R")  // indices (int64 per spec)
                .Input("output_shape", "T2?", 1)
                .Output("output", "T", "R2")
                .Code("NN.MaxUnpool({1:param}{2:param}{3:param}{a:param}{b:param}{c:param})"),

            Op(MAX_ROI_POOL)
                .Tensor<FloatLike>("T")
                .AttributeLongs(AttrPooledShape)                     // a: [pooled_h, pooled_w]
                .AttributeFloat(AttrSpatialScale)                    // b
                .Input("X", "T", 4)                                  // X: [N, C, H, W]
                .Input("rois", "T", 2)                               // rois: [num_rois, 5] — (batch_id, x1, y1, x2, y2)
                .Output("Y", "T", 4)                                 // Y: [num_rois, C, pooled_h, pooled_w]
                .Code("NN.MaxRoiPool({1:param}{2:param}{a:param}{b:param})"),

            Op(MIN)
                .Tensor<NumLike>("T")
                .Variadic("V", minCount: 1)
                .Input("x", ["T", "V"], "R")
                .Output("y", "T", rankBroadcast: "R")
                .Code("NN.Min({#:param})"),

            Op(MEAN)
                .Tensor<FloatLike>("T")
                .Variadic("V", minCount: 1)
                .Input("x", ["T", "V"], "R")
                .Output("y", "T", rankBroadcast: "R")
                .Code("NN.Mean({#:param})"),

            Op(MOD)
                .Tensor<NumLike>("T1")
                .Tensor<IntLike>("T2")
                .AttributeBool(AttrFmod)

                .Constraint(AttrFmod, 0)
                .Input("A", "T2", "R1")
                .Input("B", "T2", "R2")
                .Output("C", "T2", rankBroadcast: "R")
                .Code("{1:low_op} % {2:low_op}")
            
                .Constraint(AttrFmod, 1)
                .Input("A", "T1", "R1")
                .Input("B", "T1", "R2")
                .Output("C", "T1", rankBroadcast: "R")
                .Code("NN.FMod({1:param}{2:param}){o1:torank}"),

            Op(MUL)
                .Tensor<NumLike>("T")
                .Input("A", "T", "R1")
                .Input("B", "T", "R2")
                .Output("C", "T", rankBroadcast: "R")
                .Code("{1:low_op} * {2:low_op}"),

            Op(NEG)
                .Tensor<SignedNumLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("-{1:high_op}"),

            Op(NON_MAX_SUPPRESSION)
                .Tensor<float32>("T1")         // boxes
                .Tensor<int64>("T2")           // max_output_boxes_per_class
                .AttributeBool(AttrCenterPointBox)
                .Input("boxes", "T1", 3)              // boxes: [num_batches, spatial_dimension, 4]
                .Input("scores", "T1", 3)             // scores: [num_batches, num_classes, spatial_dimension]
                .Input("max_output_boxes_per_class", "T2?", 0)               // max_output_boxes_per_class: scalar
                .Input("iou_threshold", "T1?", 0)               // iou_threshold: scalar
                .Input("score_threshold", "T1?", 0)               // score_threshold: scalar
                .Output("selected_indices", "T2", 2)           // selected_indices: [num_selected_indices, 3] — rank 2 per spec (was wrongly 3)
                .Code("NN.NonMaxSuppression({1:param}{2:param}{3:param}{4:param}{5:param}{a:param})"),

            Op(NON_ZERO)
                .Tensor<SignedNumLike>("TIn")
                .Tensor<int64>("TOut")
                .Input("x", "TIn", "R")
                .Output("y", "TOut", 2)
                .Code("NN.NonZero({1:param})"),

            Op(NOT)
                .Tensor<bit>("T")
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("!{1:low_op}"),

            Op(OPTIONAL)
                .VarType<AnyLike>("T")
                .Structure("Tensor", DataStructure.Tensor)
                .Structure("Optional", DataStructure.Optional)
                .AttributeTypeProto(AttrType, "T")
                .Input("tensor", ["T", "Tensor"], "R")
                .Output("optional", ["T", "Optional"], "R")
                .Code("OptionalTensor<{T:ivartype}>({1:param?})"),


            Op(OPTIONAL_GET_ELEMENT)
                .VarType<AnyLike>("T")
                .Structure("AnyStruct", DataStructure.Optional)
                .Structure("Tensor", DataStructure.Tensor)
                .Input("input", ["T", "AnyStruct"], "R")
                .Output("output", ["T", "Tensor"], "R")
                .Code("{1:this}.TensorValue(){o1:torank}"),

            Op(OPTIONAL_HAS_ELEMENT)
                .VarType<AnyLike>("T")
                .Tensor<bit>("T2")
                .Structure("AnyStruct", DataStructure.Optional)
                .Input("input", ["T", "AnyStruct"], "R")
                .Output("output", "T2", 0)
                .Code("{1:this}.HasValue()"),

            Op(OR)
                .Tensor<bit>("T")
                .Input("A", "T", "R1")
                .Input("B", "T", "R2")
                .Output("C", "T", rankBroadcast: "R")
                .Code("{1:low_op} | {2:low_op}"),

            Op(PAD)
                .Tensor<AnyLike>("T")          // data
                .Tensor<int64>("T2")           // pads
                .Tensor<IndexLike>("T3")       // axes
                .AttributeEnum<PadMode>(AttrMode, ["constant", "reflect", "edge", "wrap"])
                .Input("data", "T", "R")               // data: input tensor
                .Input("pads", "T2", 1)                // pads: [2 * num_axes]
                .Input("constant_value", "T?", 0)                // constant_value (optional): scalar
                .Input("axes", "T3?", 1)               // axes (optional): [num_axes]
                .Output("output", "T", "R")// output: padded tensor
                .Code("{1:this}.Pad({a:param}{2:param}{3:param}{4:param})"),             
            
            Op(POW)
                .Tensor<FloatLike>("T1")
                .Tensor<NumLike>("T2")
                .Input("X", "T1", "R1")
                .Input("Y", "T2", "R2")
                .Output("Z", "T1", rankBroadcast: "R")
                .Code("{1:this}.Pow({2:param})"),

            Op(MISH)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(ONE_HOT)
                .Tensor<NumLike>("T1")
                .Tensor<NumLike>("T2")
                .Tensor<AnyLike>("T3")
                .AttributeLong(AttrAxis)
                .Input("indices", "T1", "R")
                .Input("depth", "T2", "R2")     // scalar per spec (kept loose: [1] is common in the wild)
                .Input("values", "T3", 1)       // [off_value, on_value]
                .Output("output", "T3", rankPlusOne: "R"), // rank(indices) + 1

            Op(P_RELU)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R1")
                .Input("slope", "T", "R2")
                .Output("Y", "T", rankBroadcast: "R"),

            Op(MEAN_VARIANCE_NORMALIZATION)
                .Tensor<FloatLike>("T")
                .AttributeLongs(AttrAxes)
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(MEL_WEIGHT_MATRIX)
                .Tensor<NumLike>("T1")
                .Tensor<IndexLike>("T2")
                .Tensor<FloatLike>("T3")
                .Tensor<float32>("T4")
                .AttributeDType(AttrOutputDatatype, "T1")

                .ConstraintIsSet(AttrOutputDatatype, true)
                .Input("num_mel_bins", "T2", 0)
                .Input("dft_length", "T2", 0)
                .Input("sample_rate", "T2", 0)
                .Input("lower_edge_hertz", "T3", 0)
                .Input("upper_edge_hertz", "T3", 0)
                .Output("output", "T1", 2)

                .ConstraintIsSet(AttrOutputDatatype, false)
                .Input("num_mel_bins", "T2", 0)
                .Input("dft_length", "T2", 0)
                .Input("sample_rate", "T2", 0)
                .Input("lower_edge_hertz", "T3", 0)
                .Input("upper_edge_hertz", "T3", 0)
                .Output("output", "T4", 2),

            Op(MULTINOMIAL)
                .Tensor<FloatLike>("T1")
                .Tensor<IndexLike>("T2")
                .Tensor<int32>("T3")
                .AttributeDType(AttrDtype, "T2")
                .AttributeLong(AttrSampleSize)
                .AttributeFloat(AttrSeed)

                .ConstraintIsSet(AttrDtype, true)
                .Input("input", "T1", 2)
                .Output("output", "T2", 2)

                // dtype unset: the spec default is int32 (the "T2" used here previously was
                // an unresolvable IndexLike token — nothing bound it without the attribute).
                .ConstraintIsSet(AttrDtype, false)
                .Input("input", "T1", 2)
                .Output("output", "T3", 2),

            Op(NEGATIVE_LOG_LIKELIHOOD_LOSS)
                .Tensor<FloatLike>("T")
                .Tensor<IndexLike>("Tind")
                .AttributeLong(AttrIgnoreIndex)
                .AttributeString(AttrReduction)

                .Constraint(AttrReduction, "none")
                .Input("input", "T", "R")
                .Input("target", "Tind", "R2")
                .Input("weight", "T?", 1)
                .Output("loss", "T", "R2")

                // reduction explicitly "mean"/"sum" must resolve to the scalar-loss
                // variant (it used to fall through to the "none" variant's R2 rank).
                .Constraint(AttrReduction, "mean")
                .Input("input", "T", "R")
                .Input("target", "Tind", "R2")
                .Input("weight", "T?", 1)
                .Output("loss", "T", 0)

                .Constraint(AttrReduction, "sum")
                .Input("input", "T", "R")
                .Input("target", "Tind", "R2")
                .Input("weight", "T?", 1)
                .Output("loss", "T", 0)

                .ConstraintIsSet(AttrReduction, false)
                .Input("input", "T", "R")
                .Input("target", "Tind", "R2")
                .Input("weight", "T?", 1)
                .Output("loss", "T", 0),

            Op(QUANTIZE_LINEAR)
                .Tensor<FloatLike>("T1")
                .Tensor<AnyIntLike>("T2")
                .AttributeLong(AttrAxis)
                .AttributeLong(AttrBlockSize)
                .AttributeDType(AttrOutputDtype, "T2") // spec wire name is "output_dtype"
                .AttributeBool(AttrSaturate)
                .AttributeLong(AttrPrecision)
                .Input("x", "T1", "R")
                .Input("y_scale", "T1", "R2")
                .Input("y_zero_point", "T2?", "R2")
                .Output("y", "T2", "R"),

            Op(QLINEAR_MATMUL)
                .Tensor<Int8Like>("T1")
                .Tensor<FloatLike>("TS")
                .Tensor<Int8Like>("T2")
                .Tensor<Int8Like>("T3")
                .Input("a", "T1", "R1")
                .Input("a_scale", "TS", "RA1")
                .Input("a_zero_point", "T1", "RA2")
                .Input("b", "T2", "R2")
                .Input("b_scale", "TS", "RB1")
                .Input("b_zero_point", "T2", "RB2")
                .Input("y_scale", "TS", "RY1")
                .Input("y_zero_point", "T3", "RY2")
                .Output("y", "T3", rankBroadcast: "R"),

            Op(QLINEAR_CONV)
                .Tensor<Int8Like>("T1")
                .Tensor<float32>("TS")
                .Tensor<Int8Like>("T2")
                .Tensor<Int8Like>("T3")
                .Tensor<int32>("T4")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .AttributeLongs(AttrDilations)
                .AttributeLong(AttrGroup)
                .AttributeLongs(AttrKernelShape)
                .AttributeLongs(AttrPads)
                .AttributeLongs(AttrStrides)
                .Input("x", "T1", "R")
                .Input("x_scale", "TS", 0)
                .Input("x_zero_point", "T1", 0)
                .Input("w", "T2", "R2")
                .Input("w_scale", "TS", "RWS")
                .Input("w_zero_point", "T2", "RWZ")
                .Input("y_scale", "TS", 0)
                .Input("y_zero_point", "T3", 0)
                .Input("B", "T4?", 1)
                .Output("y", "T3", "R"),

        ];
    }
}
