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
        private static List<NodeDefinitionMaker> GetRMakers() => [
            Op(RNN)
                .Tensor<FloatLike>("T")
                .Tensor<int32>("T1")
                .AttributeFloats(AttrActivationAlpha)
                .AttributeFloats(AttrActivationBeta)
                .AttributeStrings(AttrActivations)
                .AttributeFloat(AttrClip)
                .AttributeEnum<RNNDirection>(AttrDirection, ["forward", "reverse", "bidirectional"])
                .AttributeLong(AttrHiddenSize)
                .AttributeBool(AttrLayout)
                .Input("X", "T", 3)                    // X: [seq_length, batch_size, input_size]
                .Input("W", "T", 3)                    // W: [num_directions, hidden_size, input_size]
                .Input("R", "T", 3)                    // R: [num_directions, hidden_size, hidden_size]
                .Input("B", "T?", 2)                   // B: [num_directions, 2*hidden_size]
                .Input("sequence_lens", "T1?", 1)      // sequence_lens: [batch_size]
                .Input("initial_h", "T?", 3)           // initial_h: [num_directions, batch_size, hidden_size]
                .Output("Y", "T?", 4)                  // Y: [seq_length, num_directions, batch_size, hidden_size]
                .Output("Y_h", "T?", 3),               // Y_h: [num_directions, batch_size, hidden_size]

            Op(RANGE)
                .Tensor<SimpleNumLike>("T")
                .Input("start", "T", 0) // start
                .Input("limit", "T", 0) // limit
                .Input("delta", "T", 0) // delta
                .Output("output", "T", 1)
                .Code("VectorRange({1:param}{2:param}{3:param})"),

            // The dtype-bound token of the four Random* defs used to be a FIXED float32
            // type-def, so a non-float32 dtype attribute produced a float32-typed variable
            // while serializing dtype=<other> on the wire. The token is now FloatLike (the
            // spec range), with an explicit float32 default branch when dtype is absent.
            Op(RANDOM_NORMAL)
                .Tensor<FloatLike>("T1")
                .Tensor<float32>("T2")
                .AttributeLongs(AttrShape)
                .AttributeDType(AttrDtype, "T1")
                .AttributeFloat(AttrMean)
                .AttributeFloat(AttrScale)
                .AttributeFloat(AttrSeed)

                .ConstraintIsSet(AttrDtype, true)
                .Output("output", "T1", rank: "R")

                .ConstraintIsSet(AttrDtype, false)
                .Output("output", "T2", rank: "R"),

            Op(RANDOM_NORMAL_LIKE)
                .Tensor<AnyLike>("T1")
                .Tensor<FloatLike>("T2")
                .AttributeDType(AttrDtype, "T2")
                .AttributeFloat(AttrMean)
                .AttributeFloat(AttrScale)
                .AttributeFloat(AttrSeed)

                .ConstraintIsSet(AttrDtype, true)
                .Input("input", "T1", "R")
                .Output("output", "T2", "R")

                .ConstraintIsSet(AttrDtype, false)
                .Input("input", "T1", "R")
                .Output("output", "T1", "R"),

            Op(RANDOM_UNIFORM)
                .Tensor<FloatLike>("T1")
                .Tensor<float32>("T2")
                .AttributeLongs(AttrShape)
                .AttributeDType(AttrDtype, "T1")
                .AttributeFloat(AttrHigh)
                .AttributeFloat(AttrLow)
                .AttributeFloat(AttrSeed)

                .ConstraintIsSet(AttrDtype, true)
                .Output("output", "T1", rank: "R")

                .ConstraintIsSet(AttrDtype, false)
                .Output("output", "T2", rank: "R"),

            Op(RANDOM_UNIFORM_LIKE)
                .Tensor<AnyLike>("T1")
                .Tensor<FloatLike>("T2")
                .AttributeDType(AttrDtype, "T2")
                .AttributeFloat(AttrHigh)
                .AttributeFloat(AttrLow)
                .AttributeFloat(AttrSeed)

                .ConstraintIsSet(AttrDtype, true)
                .Input("input", "T1", "R")
                .Output("output", "T2", "R")

                .ConstraintIsSet(AttrDtype, false)
                .Input("input", "T1", "R")
                .Output("output", "T1", "R"),

            Op(RECIPROCAL)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("{1:this}.Reciprocal()"),

            Op(REDUCE_L1)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.L1, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.L1, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_L2)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.L2, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.L2, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_LOG_SUM)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.LogSum, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.LogSum, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_LOG_SUM_EXP)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.LogSumExp, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.LogSumExp, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_MAX)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.Max, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.Max, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_MEAN)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.Mean, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.Mean, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_MIN)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.Min, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.Min, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_PROD)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.Prod, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.Prod, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_SUM)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.Sum, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.Sum, {1:param}{2:param}{a:param}{b:param})"),

            Op(REDUCE_SUM_SQUARE)
                .Tensor<SimpleNumLike2>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrKeepdims)
                .AttributeBool(AttrNoopWithEmptyAxes)

                .Constraint(AttrKeepdims, 1)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R")
                .Code("NN.Reduce(ReduceKind.SumSquare, {1:param}{2:param}{a:param}{b:param})")

                .Constraint(AttrKeepdims, 0)
                .Input("data", "T1", "R")
                .Input("axes", "T2?", 1)
                .Output("reduced", "T1", "R2")
                .Code("NN.Reduce(ReduceKind.SumSquare, {1:param}{2:param}{a:param}{b:param})"),

            Op(RELU)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Relu()"),

            Op(RESHAPE)
                .AttributeBool(AttrAllowzero)
                .Tensor<AnyLike>("T1")
                .Tensor<int64>("T2")
                .Input("input", "T1", "R")
                .Input("shape", "T2", 1)
                .Output("output", "T1", "R2")
                .Code("{1:this}.Reshape({2:param})"),

            Op(REVERSE_SEQUENCE)
                .Tensor<NumLike>("T")
                .Tensor<int64>("T2")
                .AttributeLong(AttrBatchAxis)
                .AttributeLong(AttrTimeAxis)
                .Input("input", "T", "R")
                .Input("sequence_lens", "T2", 1)
                .Output("Y", "T", "R")
                .Code("NN.ReverseSequence({1:param}{2:param}{a:param}{b:param})"),

            Op(RESIZE)
                .Tensor<NumLike>("T1")         // X and Y
                .Tensor<FloatLike>("T2")       // roi
                .Tensor<float32>("T3")
                .Tensor<int64>("T4")
                .AttributeBool(AttrAntialias)
                .AttributeLongs(AttrAxes)
                .AttributeEnum<CoordinateTransformationMode>(AttrCoordinateTransformationMode,
                    ["half_pixel", "half_pixel_symmetric", "pytorch_half_pixel",
                    "align_corners", "asymmetric", "tf_crop_and_resize"])
                .AttributeFloat(AttrCubicCoeffA)
                .AttributeBool(AttrExcludeOutside)
                .AttributeFloat(AttrExtrapolationValue)
                .AttributeEnum<KeepAspectRatioPolicy>(AttrKeepAspectRatioPolicy,
                    ["stretch", "not_larger", "not_smaller"])
                .AttributeEnum<ResizeMode>(AttrMode, ["nearest", "linear", "cubic"])
                .AttributeEnum<NearestMode>(AttrNearestMode,
                    ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"])
                .Input("X", "T1", "R")              // X: N-D tensor
                .Input("roi", "T2?", 1)               // roi: [2 * rank(X)] or [2 * len(axes)]
                .Input("scales", "T3?", 1)               // scales: [rank(X)] or [len(axes)]
                .Input("sizes", "T4?", 1)               // sizes: [rank(X)] or [len(axes)]
                .Output("Y", "T1", "R")             // Y: same rank as X (used to be an unconstrained "R2")
                .Code("NN.Resize({1:param}{3:param}{4:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param}{g:param}{h:param}{i:param}{2:param?})"),

            Op(ROI_ALIGN)
                .Tensor<FloatLike>("T1")       // X and Y
                .Tensor<int64>("T2")           // batch_indices
                .AttributeEnum<RoiAlignTransformationMode>(AttrCoordinateTransformationMode,
                    ["half_pixel", "output_half_pixel"])
                .AttributeEnum<RoiAlignMode>(AttrMode, ["avg", "max"])
                .AttributeLong(AttrOutputHeight)
                .AttributeLong(AttrOutputWidth)
                .AttributeLong(AttrSamplingRatio)
                .AttributeFloat(AttrSpatialScale)
                .Input("X", "T1", 4)                         // X: [N, C, H, W]
                .Input("rois", "T1", 2)                      // rois: [num_rois, 4]
                .Input("batch_indices", "T2", 1)             // batch_indices: [num_rois]
                .Output("Y", "T1", 4)                        // Y: [num_rois, C, output_height, output_width]
                .Code("NN.RoiAlign({1:param}{2:param}{3:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param})"),

            Op(ROUND)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(REGEX_FULL_MATCH)
                .Tensor<@string>("T1")
                .Tensor<bit>("T2")
                .AttributeString(AttrPattern)
                .Input("X", "T1", "R")
                .Output("Y", "T2", "R"),

            // Root-mean-square layer normalization (opset 23+). Normalizes over the
            // suffix axes starting at `axis` (default -1); Y has X's shape and the
            // scale input's dtype (type group V per spec).
            Op(RMS_NORMALIZATION)
                .Tensor<FloatLike>("T")     // X
                .Tensor<FloatLike>("V")     // scale, Y
                .AttributeLong(AttrAxis)            // a (default -1)
                .AttributeFloat(AttrEpsilon)        // b (default 1e-5)
                .AttributeLong(AttrStashType)       // c (default 1)
                .Input("X", "T", "R")
                .Input("scale", "V", "R2")
                .Output("Y", "V", rank: "R")
                .Code("NN.RMSNormalization({1:param}{2:param}{a:param}{b:param}{c:param})"),

            // Rotary positional embedding (opset 23+). Y has X's shape and dtype.
            // X is 4-D (batch, num_heads, seq, head) or 3-D (batch, seq, hidden) —
            // the 3-D form requires the num_heads attribute. position_ids is optional;
            // when absent the caches are 3-D (batch, seq, rot/2).
            Op(ROTARY_EMBEDDING)
                .Tensor<FloatLike>("T")
                .Tensor<int64>("M")
                .AttributeBool(AttrInterleaved)             // a (default 0)
                .AttributeLong(AttrNumHeads)                // b (3-D X only)
                .AttributeLong(AttrRotaryEmbeddingDim)      // c (default 0 → full rotation)
                .Input("X", "T", "R")
                .Input("cos_cache", "T", "R2")
                .Input("sin_cache", "T", "R3")
                .Input("position_ids", "M?", 2)
                .Output("Y", "T", rank: "R")
                .Code("NN.RotaryEmbedding({1:param}{2:param}{3:param}{4:param}{a:param}{b:param}{c:param})"),

        ];
    }
}