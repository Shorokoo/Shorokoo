using System.Diagnostics;
using Shorokoo.Core.Nodes.OnnxNodes;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== SequenceConstruct =====
        //
        // Forward: sequence = SequenceConstruct(tensor_0, tensor_1, ..., tensor_{N-1})
        //   Creates a sequence from N input tensors.
        //
        // Gradient: The output gradient is a sequence of gradients.
        //   For each input tensor i, its gradient is the i-th element
        //   of the output sequence gradient: dTensor_i = SequenceAt(dSeq, i).
        //
        //   dTensor_i = SequenceAt(dOutputSequence, i)  for i in [0, N)

        internal static Variable?[] SequenceConstructGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // dSeq is guaranteed non-null: FastProcessAutoGrad/AutoDiffEngine elide single-output
            // gradients whose only outputGrad is null. SequenceConstruct's variadic tensor inputs
            // are never null either — the variadic API rejects null positional entries.
            var dSeq = outputGrads[0]!;
            Debug.Assert(dSeq is not null);

            var result = new Variable?[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                Debug.Assert(inputs[i] is not null);
                result[i] = OnnxOp.SequenceAt(dSeq, Scalar((long)i));
            }
            return result;
        }

        // ===== SequenceAt =====
        //
        // Forward: tensor = SequenceAt(sequence, position)
        //   Extracts the tensor at the given position from the sequence.
        //
        // Gradient: The output gradient is a tensor dY.
        //   The input sequence gradient is a sequence of zeros with dY placed
        //   at the target position. This is achieved by building a zero-filled
        //   sequence from the input (preserving element shapes), then replacing
        //   the element at position with dY.
        //
        //   dSequence = zeros_like_sequence(input_sequence)
        //   dSequence[position] = dY
        //   dPosition = null (int64, not differentiable)

        internal static Variable?[] SequenceAtGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var inputSeq = inputs[0]!;
            var position = inputs[1]!;
            // dY is guaranteed non-null: dispatcher skips single-output nodes whose only
            // outputGrad is null before invoking the gradient method.
            var dY = outputGrads[0]!;
            Debug.Assert(dY is not null);

            // Build zero-filled sequence matching input element shapes
            var len = ((Tensor<int64>)OnnxOp.SequenceLength(inputSeq)).Scalar();
            var dSeq = OnnxOp.SequenceEmpty(inputSeq.Type);

            foreach (var ctx in LoopAPI.Iterate(len))
            {
                var element = OnnxOp.SequenceAt(inputSeq, ctx.IterationIndex);
                var zeros = OnnxOp.Sub(element, element);  // zeros_like
                dSeq = OnnxOp.SequenceInsert(dSeq, zeros, null);  // append
            }

            // Replace zeros at position with dY
            dSeq = OnnxOp.SequenceErase(dSeq, position);
            dSeq = OnnxOp.SequenceInsert(dSeq, dY, position);

            return [dSeq, null];
        }

        // ===== SequenceInsert =====
        //
        // Forward: output_sequence = SequenceInsert(input_sequence, tensor, position?)
        //   Inserts tensor at the given position (or appends if position is null).
        //   Output sequence has length = input_length + 1.
        //
        // Gradient: The output gradient is a sequence of length input_length + 1.
        //   dInputSequence = SequenceErase(dOutputSequence, position)
        //     — remove the gradient at the inserted position to restore original length
        //   dTensor = SequenceAt(dOutputSequence, position)
        //     — the gradient at the inserted position is the gradient for the tensor
        //   dPosition = null (int64, not differentiable)

        internal static Variable?[] SequenceInsertGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // inputs[0] = input_sequence, inputs[1] = tensor, inputs[2] = position (optional, may be null)
            // dOutputSeq is guaranteed non-null: dispatcher elides single-output gradients whose
            // only outputGrad is null.
            var dOutputSeq = outputGrads[0]!;
            Debug.Assert(dOutputSeq is not null);

            // Determine the effective position
            Variable effectivePosition;
            if (inputs.Length <= 2 || inputs[2] is null)
            {
                // Null position means append — inserted element is at the end
                var outLen = OnnxOp.SequenceLength(dOutputSeq);
                effectivePosition = OnnxOp.Sub(outLen, Scalar(1L));
            }
            else
            {
                effectivePosition = inputs[2]!;
            }

            var dTensor = OnnxOp.SequenceAt(dOutputSeq, effectivePosition);
            var dInputSeq = OnnxOp.SequenceErase(dOutputSeq, effectivePosition);

            return inputs.Length > 2 ? [dInputSeq, dTensor, null] : [dInputSeq, dTensor];
        }

        // ===== SequenceErase =====
        //
        // Forward: output_sequence = SequenceErase(input_sequence, position)
        //   Removes the tensor at the given position from the sequence.
        //   Output sequence has length = input_length - 1.
        //
        // Gradient: The output gradient is a sequence of length input_length - 1.
        //   To restore the input sequence gradient, insert a zero tensor at the
        //   erased position. The zero tensor has the same shape as the original
        //   erased element.
        //
        //   dInputSequence = SequenceInsert(dOutputSequence, zeros_like(erased_element), position)
        //   dPosition = null (int64, not differentiable)

        internal static Variable?[] SequenceEraseGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var inputSeq = inputs[0]!;
            var position = inputs[1]!;
            // dOutputSeq is guaranteed non-null: dispatcher elides single-output gradients whose
            // only outputGrad is null.
            var dOutputSeq = outputGrads[0]!;
            Debug.Assert(dOutputSeq is not null);

            // Get the shape of the erased element to create zeros of matching shape
            var erasedElement = OnnxOp.SequenceAt(inputSeq, position);
            var zeros = OnnxOp.Sub(erasedElement, erasedElement);  // zeros_like

            // Insert zeros back at the erased position
            var dInputSeq = OnnxOp.SequenceInsert(dOutputSeq, zeros, position);

            return [dInputSeq, null];
        }

        // ===== ConcatFromSequence =====
        //
        // Forward: tensor = ConcatFromSequence(sequence, axis, newAxis)
        //   Concatenates all tensors in the sequence along the given axis.
        //   If newAxis is true, a new axis is inserted before concatenation
        //   (like torch.stack).
        //
        // Gradient: The output gradient is a tensor dY.
        //   Split dY along the axis according to each input element's size,
        //   producing a sequence of gradient tensors.
        //
        //   For newAxis=false: slice dY along axis using cumulative sizes
        //   For newAxis=true: each element contributes size 1 along the new axis;
        //     slice and squeeze to remove the new axis dimension

        internal static Variable?[] ConcatFromSequenceGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            var inputSeq = inputs[0]!;
            // dY is guaranteed non-null: dispatcher elides single-output gradients whose only
            // outputGrad is null.
            var dY = outputGrads[0]!;
            Debug.Assert(dY is not null);

            var axis = (long)attributes.GetAttributeObj(AttrAxis)!;
            var newAxis = attributes.GetAttributeObj(AttrNewAxis) is bool na && na;

            var len = ((Tensor<int64>)OnnxOp.SequenceLength(inputSeq)).Scalar();
            var dSeq = OnnxOp.SequenceEmpty(inputSeq.Type);
            Variable offset = Scalar(0L);

            foreach (var ctx in LoopAPI.Iterate(len))
            {
                var element = OnnxOp.SequenceAt(inputSeq, ctx.IterationIndex);

                Variable elemSize;
                if (newAxis)
                {
                    // Each element contributes 1 along the new axis
                    elemSize = Scalar(1L);
                }
                else
                {
                    var elemShape = OnnxOp.Shape(element);
                    elemSize = OnnxOp.Gather(elemShape, Scalar(axis), axis: 0);
                }

                // Slice dY from offset to offset+elemSize along the concat axis
                var starts = OnnxOp.Reshape(offset, Vector(1L), allowZero: false);
                var end = OnnxOp.Add(offset, elemSize);
                var ends = OnnxOp.Reshape(end, Vector(1L), allowZero: false);
                var axes = Vector(axis);
                var sliced = OnnxOp.Slice(dY, starts, ends, axes);

                if (newAxis)
                {
                    // Remove the new axis dimension
                    sliced = OnnxOp.Squeeze(sliced, Vector(axis));
                }

                dSeq = OnnxOp.SequenceInsert(dSeq, sliced, null);  // append
                offset = end;
            }

            return [dSeq];
        }
    }
}
