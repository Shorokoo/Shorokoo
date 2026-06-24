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
        private static List<NodeDefinitionMaker> GetTZMakers() => [
            Op(TAN)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", rank: "R")
                .Code("{1:this}.Tan()", inline: true),

            Op(TANH)
                .Tensor<FloatLike>("T")
                .Input("X", "T", "R")
                .Output("Y", "T", rank: "R")
                .Code("{1:this}.Tanh()", inline: true),

            Op(TILE)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("T1")
                .Input("input", "T", "R")
                .Input("repeats", "T1", 1)
                .InputTestShapes("input", [[2,3,4],[1,3,2,2]])
                .InputTestValues("repeats", [TensorData([3], 2L, 3L, 2L), TensorData([4], 3L, 1L, 1L, 2L)])
                .Output("ouput", "T", rank: "R") // output: same rank as input per spec
                .Code("{1:this}.Tile({2:param})"),

            Op(TOPK)
                .Tensor<NumLike>("T")          // X and Values
                .Tensor<int64>("T2")           // K
                .AttributeLong(AttrAxis)
                .AttributeBool(AttrLargest)
                .AttributeBool(AttrSorted)
                .Input("X", "T", "R")               // X: [a_0, a_1, ..., a_{n-1}]
                .Input("K", "T2", 1)                // K: [1]
                .InputTestShapes("X", [[2,3,4],[1,3,2,2,6]])
                .InputTestValues("K", [TensorData([1], 2L), TensorData([1], 3L)])
                .AttributeTestValues(AttrAxis, [2L, 4L])
                .Output("Values", "T", "R")             // Values: [a_0, ..., a_{axis-1}, k, a_{axis+1}, ..., a_{n-1}]
                .Output("Indices", "T2", "R")           // Indices: same rank as Values
                .Code("NN.TopK({1:param}{2:param}{a:param}{b:param}{c:param})"),
            
            Op(TRANSPOSE)
                .Tensor<AnyLike>("T")
                .AttributeLongs(AttrPerm)
                .Input("data", "T", "R")
                .Output("transposed", "T", rank: "R")
                .InputTestShapes("data", [[2,3,4],[1,2,2,2,2]])
                .AttributeTestValues(AttrPerm, (long[]?[])[[1,2,0], [3,4,2,1,0]])
                .Code("{1:this}.Transpose({a:param})"),

            Op(UNSQUEEZE)
                .Tensor<AnyLike>("T1")
                .Tensor<int64>("T2")
                .Input("data", "T1", "R") // data
                .Input("axes", "T2", 1) // axes
                .InputTestShapes("data", [[2,3,4],[1,2,2,2]])
                .InputTestValues("axes", [TensorData([2], 1L,3L), TensorData([3], 6L, 5L, 4L)])
                .Output("unsqueezed", "T1", "R2")
                .Code("{1:this}.Unsqueeze({2:param})"),

            Op(UPSAMPLE)
                .Tensor<NumLike>("T")
                .Tensor<float32>("T2")
                .AttributeEnum<ResizeMode>(AttrMode, ["nearest", "linear", "cubic"])
                .Input("X", "T", "R")              // X: N-D input tensor
                .Input("scales", "T2", 1)          // scales: [rank(X)] scale factors
                .Output("Y", "T", "R"),            // Y: same rank as X (used to be an unconstrained "R2")

            Op(WHERE)
                .Tensor<bit>("T1")
                .Tensor<AnyLike>("T2")
                .Input("condition", "T1", "R1") // Condition
                .Input("x_when_true", "T2", "R2") // X: when true
                .Input("y_when_false", "T2", "R3") // Y: when false
                .Output("output", ["T2", "R1", "R2", "R3"], rankBroadcast: "R"), // output

            Op(XOR)
                .Tensor<bit>("T")
                .Input("A", "T", "R1")
                .Input("B", "T", "R2")
                .Output("C", "T", rankBroadcast: "R")
                .WithBroadcastTestShapes()
                .Code("{1:low_op} ^ {2:low_op}"),

            Op(TRILU)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("T2")
                .AttributeLong(AttrUpper)
                .Input("input", "T", "R")
                .Input("k", "T2?", 0)
                .Output("output", "T", "R")
                .InputTestShapes("input", [[3, 3]])
                .Code("NN.Trilu({1:param}{2:param}{a:param})"),

            Op(UNIQUE)
                .Tensor<NumLike>("T")
                .Tensor<int64>("T2")
                .AttributeLong(AttrAxis)
                .AttributeBool(AttrSorted)
                .Input("X", "T", "R")
                .Output("Y", "T", "R2")
                .Output("indices", "T2", 1)
                .Output("inverse_indices", "T2", 1)
                .Output("counts", "T2", 1)
                .InputTestShapes("X", [[6], [3, 4]])
                .AttributeTestValues(AttrSorted, [true, true])
                .Code("NN.Unique({1:param}{a:param}{b:param})"),

            Op(THRESHOLDED_RELU)
                .Tensor<FloatLike>("T")
                .AttributeFloat(AttrAlpha)
                .Input("X", "T", "R")
                .Output("Y", "T", "R"),

            // KV-cache style in-place-update-as-a-function op (opset 24+).
            // present_cache has past_cache's shape and dtype; update writes a
            // sequence_length-long window along `axis` (default -2) at the per-batch
            // write_indices offsets (linear or circular).
            Op(TENSOR_SCATTER)
                .Tensor<AnyLike>("T")
                .Tensor<int64>("T2")
                .AttributeLong(AttrAxis)                                            // a (default -2)
                .AttributeEnum<TensorScatterMode>(AttrMode, ["linear", "circular"]) // b (default "linear")
                .Input("past_cache", "T", "R")
                .Input("update", "T", "R2")
                .Input("write_indices", "T2?", 1)
                .Output("present_cache", "T", rank: "R")
                .InputTestShapes("past_cache", [[2, 1, 4, 5]])
                .InputTestShapes("update", [[2, 1, 1, 5]])
                .InputTestValues("write_indices", [TensorData([2], 1L, 2L)]),

            Op(TFIDF_VECTORIZER)
                .Tensor<NumLike>("T")
                .Tensor<float32>("T1")
                .AttributeLong(AttrMaxGramLength)
                .AttributeLong(AttrMaxSkipCount)
                .AttributeLong(AttrMinGramLength)
                .AttributeString(AttrMode)
                .AttributeLongs(AttrNgramCounts)
                .AttributeLongs(AttrNgramIndexes)
                .AttributeLongs(AttrPoolInt64s)
                .AttributeStrings(AttrPoolStrings)
                .AttributeFloats(AttrWeights)
                .Input("X", "T", "R")
                .Output("Y", "T1", "R"),
        ];
    }
}
