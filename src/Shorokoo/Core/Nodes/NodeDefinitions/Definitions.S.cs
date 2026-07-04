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
        private static List<NodeDefinitionMaker> GetSMakers() => [

            Op(SCATTER_ELEMENTS)
                .Tensor<AnyLike>("T")          // data and updates can be any tensor type
                .Tensor<IndexLike>("Tind")     // indices must be index type
                .AttributeLong(AttrAxis)
                .AttributeEnum<ScatterNDReduction>(AttrReduction, ["none", "add", "mul", "max", "min"])
                .Input("data", "T", "R1")               // data: rank r >= 1
                .Input("indices", "Tind", "R2")          // indices: same rank as data
                .Input("updates", "T", "R3")             // updates: same shape as indices
                .Output("output", "T", "R1")             // output: same shape (and rank) as data
                .Code("NN.ScatterElements({1:param}{2:param}{3:param}{a:param}{b:param})"),

            Op(SCATTER_ND)
                .Tensor<AnyLike>("T")          // data and updates can be any tensor type
                .Tensor<int64>("Tind")         // indices must be int64
                .AttributeEnum<ScatterNDReduction>(AttrReduction, ["none", "add", "mul", "max", "min"])
                .Input("data", "T", "R1")               // data: rank r >= 1
                .Input("indices", "Tind", "R2")            // indices: rank q >= 1
                .Input("updates", "T", "R3")              // updates: rank q + r - indices_shape[-1] - 1
                .Output("output", "T", "R1")             // output: same shape (and rank) as data
                .Code("{1:this}.ScatterND({2:param}{3:param}{a:param})"),

            Op(SELU)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .AttributeFloat(AttrGamma)
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Selu({a:param}{b:param})"),

            Op(SEQUENCE_AT)
                .VarType<AnyLike>("T", tracksModuleFn : true)
                .Structure("S", DataStructure.Sequence)
                .Structure("Tensor", DataStructure.Tensor)
                .Tensor<IndexLike>("T2")
                .Input("input_tensors", ["T", "S"], "R1")
                .Input("position", "T2", 0)
                .Output("tensor", ["T", "Tensor"], "R2")
                .Code("{1:this}[{2:param}]{o1:torank}"),

            Op(SEQUENCE_CONSTRUCT)
                .VarType<AnyLike>("T")
                .Structure("S", DataStructure.Sequence)
                .Structure("Tensor", DataStructure.Tensor)
                .Variadic("V", minCount: 1)
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .Input("input_tensors", ["T", "Tensor", "V"], "R1")
                .Output("sequence", ["T", "S"], "R2")
                .Code("TensorSequence({#:param})"),

            Op(SEQUENCE_EMPTY)
                .Sequence<AnyLike>("T")
                .AttributeDType(AttrDtype, "T")
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .Output("sequence", "T", "R")
                .Code("TensorSequence<{T:ivartype}>()"),

            Op(SEQUENCE_ERASE)
                .Sequence<AnyLike>("T", tracksModuleFn: true)
                .Tensor<IndexLike>("T2")
                .Input("input_tensors", "T", "R1")
                // position is optional per spec: absent means "erase the last element".
                .Input("position", "T2?", 0)
                .Output("output_sequence", "T", "R2")
                .Code("{1:this}.RemoveAt({2:param})"),

            Op(SEQUENCE_INSERT)
                .VarType<AnyLike>("T", tracksModuleFn: true)
                .Structure("S", DataStructure.Sequence)
                .Structure("Tensor", DataStructure.Tensor)
                .Tensor<IndexLike>("T2")
                .Input("input_tensors", ["T", "S"], "R1")
                .Input("input", ["T", "Tensor"], "R2")
                .Input("position", "T2?", 0)
                .Output("output_sequence", ["T", "S"], "R3")
                .Code("{1:this}.InsertAt({2:param}{3:param})"),

            Op(SEQUENCE_LENGTH)
                .Sequence<AnyLike>("T1")
                .Tensor<int64>("T2")
                .Input("sequence", "T1")
                .Output("length", "T2", 0)
                .Code("{1:this}.Count"),

            Op(SHAPE)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("T1")
                .AttributeLong(AttrEnd)
                .AttributeLong(AttrStart)
                .Input("tensor", "T", "R")
                .Output("shape", "T1", 1)
                .Code("{1:this}.ShapeTensor({b:param}{a:param})"),

            Op(SIGMOID)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Sigmoid()"),

            Op(SIGN)
                .Tensor<NumLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Sign()"),

            Op(SIN)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Sin()", inline: true),

            Op(SINH)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", rank: "R")
                .Code("{1:this}.Sinh()", inline: true),

            Op(SLICE)
                .Tensor<AnyLike>("T")
                .Tensor<IndexLike>("Tind")
                .Input("x", "T", "R")   // x
                .Input("starts", "Tind", 1)  // starts
                .Input("ends", "Tind", 1)  // ends
                .Input("axes", "Tind?", 1) // axes
                .Input("steps", "Tind?", 1) // steps
                .Output("output", "T", rank: "R")
                .Code("{1:this}.Slice({2:param}{3:param}{4:param}{5:param})"),

            Op(SOFTMAX)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrAxis)
                .Input("input", "T", "R")
                .Output("output", "T", rank: "R")
                .Code("{1:this}.Softmax({a:param?})"),

            Op(SPLIT)
                .Tensor<AnyLike>("T")
                .Variadic("V", minCount: 1)
                .Tensor<int64>("T2")
                .AttributeLong(AttrAxis)
                .AttributeLong(AttrNumOutputs, variadicCount: "V")
                .Input("input", "T", "R1")
                .Input("split", "T2?", 1)
                .Output("outputs", ["T", "V"], rank: "R2")
                .Code("{1:this}.Split({2:param}{a:param}{numoutputs:param})"),

            Op(SQUEEZE)
                .Tensor<AnyLike>("T1")
                .Tensor<int64>("T2")
                .Input("data", "T1", "R") // data
                .Input("axes", "T2?", 1) // axes (optional per spec: absent → squeeze all size-1 dims)
                .Output("squeezed", "T1", "R2") // squeezed
                .Code("{1:this}.Squeeze({2:param})"),

            Op(SQRT)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.Sqrt()"),

            Op(SUB)
                .Tensor<NumLike>("T")
                .Input("A", "T", "R1")
                .Input("B", "T", "R2")
                .Output("C", "T", rankBroadcast: "R")
                .Code("{1:low_op} - {2:low_op}"),

            Op(SUM)
                .Tensor<FloatLike>("T")
                .Variadic("V", minCount: 1)
                .Input("x", ["T", "V"], "R")
                .Output("y", "T", rankBroadcast: "R")
                .Code("NN.Sum({#:param})"),

            // Swish activation (opset 24+): y = x * sigmoid(alpha * x), alpha default 1.
            // NOTE: ONNX Runtime 1.26 registers no Swish kernel on any execution
            // provider, so graphs containing Swish are QEE-executable only.
            Op(SWISH)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)      // a (default 1.0)
                .Input("X", "T", "R")
                .Output("Y", "T", rank: "R")
                .Code("NN.Swish({1:param}{a:param})"),

            Op(SPACE_TO_DEPTH)
                .Tensor<AnyLike>("T")
                .AttributeLong(AttrBlocksize)
                .Input("input", "T", 4)
                .Output("output", "T", 4)
                .Code("NN.SpaceToDepth({1:param}{a:param})"),

            Op(SHRINK)
                .Tensor<NumLike>("T")
                .AttributeFloat(AttrBias)
                .AttributeFloat(AttrLambd)
                .Input("input", "T", "R")
                .Output("output", "T", "R"),

            Op(SIZE)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("T1")
                .Input("data", "T", "R")
                .Output("size", "T1", 0),

            Op(SOFTPLUS)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(SOFTSIGN)
                .Tensor<FloatLike>("T")
                .Input("input", "T", "R")
                .Output("output", "T", "R"),

            Op(STFT)
                .Tensor<FloatLike>("T1")
                .Tensor<int64>("T2")
                .AttributeBool(AttrOnesided)
                .Input("signal", "T1", 3)
                .Input("frame_step", "T2", 0)
                .Input("window", "T1?", 1)
                .Input("frame_length", "T2?", 0)
                .Output("output", "T1", 4),

            Op(SOFTMAX_CROSS_ENTROPY_LOSS)
                .Tensor<FloatLike>("T")
                .Tensor<IndexLike>("Tind")
                .AttributeLong(AttrIgnoreIndex)
                .AttributeString(AttrReduction)

                .Constraint(AttrReduction, "none")
                .Input("scores", "T", "R")
                .Input("labels", "Tind", "R2")
                .Input("weights", "T?", 1)
                .Output("output", "T", "R2")
                .Output("log_prob", "T?", "R")

                // reduction explicitly "mean"/"sum" must resolve to the scalar-loss
                // variant (it used to fall through to the "none" variant's R2 rank).
                .Constraint(AttrReduction, "mean")
                .Input("scores", "T", "R")
                .Input("labels", "Tind", "R2")
                .Input("weights", "T?", 1)
                .Output("output", "T", 0)
                .Output("log_prob", "T?", "R")

                .Constraint(AttrReduction, "sum")
                .Input("scores", "T", "R")
                .Input("labels", "Tind", "R2")
                .Input("weights", "T?", 1)
                .Output("output", "T", 0)
                .Output("log_prob", "T?", "R")

                .ConstraintIsSet(AttrReduction, false)
                .Input("scores", "T", "R")
                .Input("labels", "Tind", "R2")
                .Input("weights", "T?", 1)
                .Output("output", "T", 0)
                .Output("log_prob", "T?", "R"),

            Op(SPLIT_TO_SEQUENCE)
                .VarType<AnyLike>("T")
                .Structure("Tensor", DataStructure.Tensor)
                .Structure("S", DataStructure.Sequence)
                .Tensor<IndexLike>("Tind")
                .AttributeLong(AttrAxis)
                .AttributeLong(AttrKeepdims)
                .Input("input", ["T", "Tensor"], "R")
                .Input("split", "Tind?", "R2")
                // The output is a SEQUENCE of tensors (was mistyped as a plain tensor).
                .Output("output_sequence", ["T", "S"], "R3"),

            Op(STRING_CONCAT)
                .Tensor<@string>("T")
                .Input("X", "T", "R1")
                .Input("Y", "T", "R2")
                .Output("Z", "T", rankBroadcast: "R"),

            Op(STRING_NORMALIZER)
                .Tensor<@string>("T")
                .AttributeString(AttrCaseChangeAction)
                .AttributeLong(AttrIsCaseSensitive)
                .AttributeString(AttrLocale)
                .AttributeStrings(AttrStopwords)
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(STRING_SPLIT)
                .Tensor<@string>("T1")
                .Tensor<int64>("T2")
                .AttributeString(AttrDelimiter)
                .AttributeLong(AttrMaxsplit)
                .Input("X", "T1", "R")
                // Y gains one trailing "tokens" dim over the input per spec.
                .Output("Y", "T1", rankPlusOne: "R")
                .Output("num_splits", "T2", "R"),

            Op(SCAN_OPEN)
                .Any<AnyLike>("StateAndScanInputs", tracksModuleFn: true, minVariadicCount: 0)
                .AttributeLong(AttrNumScanInputs)
                .AttributeLongs(AttrScanInputAxes)
                .AttributeLongs(AttrScanInputDirections)
                .AttributeLongs(AttrScanOutputAxes)
                .AttributeLongs(AttrScanOutputDirections)
                .SingleGraphOpen(AttrBody)
                .Input("initial_state_and_scan_inputs", "StateAndScanInputs?", "RIn")
                .Output("state_and_iter_slices", "StateAndScanInputs?", rank: "RIn"),

            Op(SCAN_CLOSE)
                .Any<AnyLike>("StateAndScanOutputs", tracksModuleFn: true, minVariadicCount: 0)
                .AttributeLong(AttrNumScanInputs)
                .AttributeLongs(AttrScanInputAxes)
                .AttributeLongs(AttrScanInputDirections)
                .AttributeLongs(AttrScanOutputAxes)
                .AttributeLongs(AttrScanOutputDirections)
                .SingleGraphClose(AttrBody)
                .Input("body_state_and_scan_output_slices", "StateAndScanOutputs?", "RBody")
                .Output("final_state_and_scan_outputs", "StateAndScanOutputs?", rankPlusOne: "RBody"),

            Op(SEQUENCE_MAP_OPEN)
                .Any<AnyLike>("InputSeqAndExtras", tracksModuleFn: true, minVariadicCount: 1)
                .SingleGraphOpen(AttrBody)
                .Input("input_seq_and_extras", "InputSeqAndExtras?", "RIn")
                .Output("body_iter_inputs", "InputSeqAndExtras?", rank: "RIn"),

            Op(SEQUENCE_MAP_CLOSE)
                .Any<AnyLike>("BodyOutputs", tracksModuleFn: true, minVariadicCount: 0)
                .SingleGraphClose(AttrBody)
                .Input("body_iter_outputs", "BodyOutputs?", "RBody")
                .Output("output_sequences", "BodyOutputs?", rank: "RBody"),

        ];
    }
}