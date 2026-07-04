using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    internal static partial class Definitions
    {
        private static List<NodeDefinitionMaker> GetHLMakers() => [
            Op(IDENTITY)
                .Any<AnyLike>("T", tracksModuleFn: true)
                .AttributeLong(InternalAttrRank, "Rout")

                .ConstraintIsSet(InternalAttrRank, false)
                .Input("input", "T", "Rin")
                .Output("output", "T", "Rin")
                .Code("{1:}")

                .ConstraintIsSet(InternalAttrRank, true)
                .Input("input", "T", "Rin")
                .Output("output", "T", "Rout")
                .Code("{1:}{o1:torank}"),

            Op(IF_OPEN)
                .Tensor<bit>("Cond")
                .PairGraphOpen(AttrElseBranch, AttrThenBranch)
                .Input("cond", "Cond", 0),

            Op(IF_CLOSE)
                .Any<AnyLike>("ternaryParams", minVariadicCount: 1)
                .PairGraphClose(AttrElseBranch, AttrThenBranch)
                .Input("else_tensors", "ternaryParams", "R")
                .Input("then_tensors", "ternaryParams", "R")
                .Output("outputs", "ternaryParams", "R"),

            Op(LEAKY_RELU)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("{1:this}.LeakyRelu({a:param})"),

            Op(LESS)
                .Tensor<NumLike>("T")
                .Tensor<bit>("T2")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T2", rankBroadcast: "R")
                .Code("{1:low_op} < {2:low_op}"),

            Op(LESS_OR_EQUAL)
                .Tensor<NumLike>("T")
                .Tensor<bit>("T2")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T2", rankBroadcast: "R")
                .Code("{1:low_op} <= {2:low_op}"),

            Op(LOOP_OPEN)
                .Any<AnyLike>("LoopVariables", tracksModuleFn: true, minVariadicCount: 0)
                .Tensor<AnyLike>("ScanVariables", minVariadicCount: 0)
                .Tensor<bit>("Cond")
                .Tensor<int64>("MaxIter")
                .Tensor<int64>("IterNum")
                .Tensor<bit>("VestigalTrue")
                .Tensor<bit>("Break")
                .SingleGraphOpen(AttrBody)
                .Input("maxIterations", "MaxIter", 0)
                .Input("cond", "Cond", 0)
                .Input("loopVariables", "LoopVariables?", "RLoops")
                .Output("iterationIndex", "IterNum", 0)
                .Output("", "VestigalTrue", 0)
                .Output("loopVariables", "LoopVariables?", rank: "RLoops"),

            Op(LOOP_CLOSE)
                .Any<AnyLike>("LoopVariables", tracksModuleFn: true, minVariadicCount: 0)
                .Tensor<AnyLike>("ScanVariables", minVariadicCount: 0)
                .Tensor<bit>("Cond")
                .Tensor<int64>("MaxIter")
                .Tensor<int64>("IterNum")
                .Tensor<bit>("VestigalTrue")
                .Tensor<bit>("Break")
                .SingleGraphClose(AttrBody)
                .Input("break", "Break")
                .Input("loopVariables", "LoopVariables?", "RLoops")
                .Input("scanVariables", "ScanVariables?", "RScans")
                .Output("loopedVariables", "LoopVariables?", rank: "RLoops")
                .Output("scannedVariables", "ScanVariables?", rankPlusOne: "RScans"),

            Op(LOOP_FAKE_INPUT)
                .Any<AnyLike>("T")
                .AttributeDType(AttrDtype, "T")
                .AttributeLong(InternalAttrRank, "R")
                .AttributeEnum<DataStructure>(InternalAttrStructure, ["Tensor", "Optional", "Sequence"], structureDefName: "T")
                .Output("output", "T", "R"),

            Op(LOOP_SCAN_VARIABLE)
                .Tensor<AnyLike>("scan")
                .Input("scannee", "scan", "R")
                .Output("scanned", "scan", rankPlusOne: "R"),

            Op(LOOP_INDEX_VARIABLE)
                .Tensor<int64>("IterNum")
                .Output("zombie", "IterNum", rank: 0),

            Op(LOG)
                .Tensor<FloatLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("{1:this}.Ln()"),

            Op(INSTANCE_NORMALIZATION)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrEpsilon)
                .Input("input", "T", "R")
                .Input("scale", "T", 1)
                .Input("B", "T", 1)
                .Output("output", "T", "R")
                .Code("NN.InstanceNormalization({1:param}{2:param}{3:param}{a:param})"),

            Op(LP_NORMALIZATION)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrAxis)
                .AttributeLong(AttrP)
                .Input("input", "T", "R")
                .Output("output", "T", "R")
                .Code("NN.LpNormalization({1:param}{a:param}{b:param})"),

            Op(LP_POOL)
                .Tensor<FloatLike>("T")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])  // a
                .AttributeBool(AttrCeilMode)        // b
                .AttributeLongs(AttrDilations)      // c
                .AttributeLongs(AttrKernelShape)    // d
                .AttributeLong(AttrP)               // e
                .AttributeLongs(AttrPads)           // f
                .AttributeLongs(AttrStrides)        // g

                .Constraint(AttrAutoPad, "NOTSET")
                .Input("X", "T", "R")
                .Output("Y", "T", rank: "R")
                .Code("Shorokoo.Core.Nodes.NodeDefinitions.OnnxOp.LpPool({1:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param}{g:param}){o1:fromvar}")

                .ConstraintIsSet(AttrAutoPad, true)
                .Input("X", "T", "R")
                .Output("Y", "T", rank: "R")
                .Code("Shorokoo.Core.Nodes.NodeDefinitions.OnnxOp.LpPool({1:param}{a:param}{b:param}{c:param}{d:param}{e:param}{f:param}{g:param}){o1:fromvar}"),

            Op(LSTM)
                .Tensor<FloatLike>("T")
                .Tensor<int32>("T1")
                .AttributeFloats(AttrActivationAlpha)
                .AttributeFloats(AttrActivationBeta)
                .AttributeStrings(AttrActivations)
                .AttributeFloat(AttrClip)
                .AttributeEnum<LSTMDirection>(AttrDirection, ["forward", "reverse", "bidirectional"])
                .AttributeLong(AttrHiddenSize)
                .AttributeBool(AttrInputForget)
                .AttributeBool(AttrLayout)
                .Input("X", "T", 3)                    // X: [seq_length, batch_size, input_size]
                .Input("W", "T", 3)                    // W: [num_directions, 4*hidden_size, input_size]
                .Input("R", "T", 3)                    // R: [num_directions, 4*hidden_size, hidden_size]
                .Input("B", "T?", 2)                   // B: [num_directions, 8*hidden_size]
                .Input("sequence_lens", "T1?", 1)      // sequence_lens: [batch_size]
                .Input("initial_h", "T?", 3)           // initial_h: [num_directions, batch_size, hidden_size]
                .Input("initial_c", "T?", 3)           // initial_c: [num_directions, batch_size, hidden_size]
                .Input("P", "T?", 2)                   // P: [num_directions, 3*hidden_size] (peephole)
                .Output("Y", "T?", 4)                  // Y: [seq_length, num_directions, batch_size, hidden_size]
                .Output("Y_h", "T?", 3)                // Y_h: [num_directions, batch_size, hidden_size]
                .Output("Y_c", "T?", 3),               // Y_c: [num_directions, batch_size, hidden_size]

            Op(LRN)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .AttributeFloat(AttrBeta)
                .AttributeFloat(AttrBias)
                .AttributeLong(AttrSize)
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("NN.Lrn({1:param}{a:param}{b:param}{c:param}{d:param})"),

            Op(HAMMING_WINDOW)
                .Tensor<NumLike>("T1")
                .Tensor<IndexLike>("T2")
                .Tensor<float32>("T3")
                .AttributeDType(AttrOutputDatatype, "T1")
                .AttributeBool(AttrPeriodic)

                .ConstraintIsSet(AttrOutputDatatype, true)
                .Input("size", "T2", 0)
                .Output("output", "T1", 1)

                .ConstraintIsSet(AttrOutputDatatype, false)
                .Input("size", "T2", 0)
                .Output("output", "T3", 1),

            Op(HANN_WINDOW)
                .Tensor<NumLike>("T1")
                .Tensor<IndexLike>("T2")
                .Tensor<float32>("T3")
                .AttributeDType(AttrOutputDatatype, "T1")
                .AttributeBool(AttrPeriodic)

                .ConstraintIsSet(AttrOutputDatatype, true)
                .Input("size", "T2", 0)
                .Output("output", "T1", 1)

                .ConstraintIsSet(AttrOutputDatatype, false)
                .Input("size", "T2", 0)
                .Output("output", "T3", 1),

            Op(HARD_SIGMOID)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .AttributeFloat(AttrBeta)
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(HARD_SWISH)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            Op(HARDMAX)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrAxis)
                .Input("input", "T", "R")
                .Output("output", "T", rank: "R"),

            Op(IS_INF)
                .Tensor<FloatLike>("T1")
                .Tensor<bit>("T2")
                .AttributeBool(AttrDetectNegative)
                .AttributeBool(AttrDetectPositive)
                .Input("X", "T1", "R")
                .Output("Y", "T2", "R"),

            Op(IS_NAN)
                .Tensor<FloatLike>("T1")
                .Tensor<bit>("T2")
                .Input("X", "T1", "R")
                .Output("Y", "T2", "R"),

            Op(LAYER_NORMALIZATION)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrAxis)
                .AttributeFloat(AttrEpsilon)
                .AttributeLong(AttrStashType)
                .Input("X", "T", "R")
                .Input("Scale", "T", "R2")
                .Input("B", "T?", "R3")
                .Output("Y", "T", rank: "R")
                .Output("Mean", "T?", "R4")
                .Output("InvStdDev", "T?", "R5"),

            Op(LOG_SOFTMAX)
                .Tensor<FloatLike>("T")
                .AttributeLong(AttrAxis)
                .Input("input", "T", "R")
                .Output("output", "T", rank: "R"),

            Op(IMAGE_DECODER)
                .Tensor<uint8>("T1")
                .Tensor<uint8>("T2")
                .AttributeString(AttrPixelFormat)
                .Input("encoded_stream", "T1", 1)
                .Output("image", "T2", 3),

        ];
    }
}