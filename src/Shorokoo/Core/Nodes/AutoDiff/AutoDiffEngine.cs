using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using System.Diagnostics;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal class AutoDiffEngine
    {
        public static Variable AccumulateGradients(Variable a, Variable b)
        {
            if (a.Structure() == DataStructure.Tensor)
            {
                Debug.Assert(b.Structure() == DataStructure.Tensor);
                return OnnxOp.Add(a, b);
            }
            else if (a.Structure() == DataStructure.Sequence)
            {
                var sequenceResult = OnnxOp.SequenceEmpty(a.Type);
                foreach (var ctx in LoopAPI.Iterate(((Tensor<int64>)OnnxOp.SequenceLength(a)).Scalar()))
                {
                    var elementA = OnnxOp.SequenceAt(a, ctx.IterationIndex);
                    var elementB = OnnxOp.SequenceAt(b, ctx.IterationIndex);
                    var sum = OnnxOp.Add(elementA, elementB);
                    sequenceResult = OnnxOp.SequenceInsert(sequenceResult, sum, ctx.IterationIndex);
                }

                return sequenceResult;
            }
            else if (a.Structure() == DataStructure.Optional)
            {
                var isNotNull = ((Tensor<bit>)OnnxOp.OptionalHasElement(a)).Scalar();
                var sum = OnnxOp.Add(OnnxOp.OptionalGetElement(a), OnnxOp.OptionalGetElement(b));
                return Shorokoo.Core.Nodes.Ops.IfElse(isNotNull, sum, OnnxOp.Optional(null, DataStructure.Tensor, a.Type));
            }

            throw new AutoDiffNotSupportedException(ErrorCodes.AD002, "GradientAccumulation",
                $"gradients can be accumulated for tensors, tensor sequences, and optional tensors, " +
                $"but not for variables of type '{a.GetType().Name}'.");
        }
    }
}
