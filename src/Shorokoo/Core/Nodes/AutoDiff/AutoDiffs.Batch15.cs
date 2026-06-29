using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== ReverseSequence =====
        //
        // Forward: output = ReverseSequence(input, sequence_lens, batch_axis, time_axis)
        //   Reverses the first sequence_lens[i] elements along time_axis for
        //   each batch element i (indexed by batch_axis). Elements beyond
        //   sequence_lens[i] are copied through unchanged.
        //
        // Gradient: ReverseSequence is its own inverse when applied with the
        //   same sequence_lens, batch_axis, and time_axis. Applying it to the
        //   output gradient reverses the gradient elements back to their
        //   original positions.
        //
        //   dInput = ReverseSequence(grad, sequence_lens, batch_axis, time_axis)
        //   dSequenceLens = null (int64, not differentiable)

        [AutoDiff(REVERSE_SEQUENCE)]
        public static Variable?[] ReverseSequence<T1, T2>(
            Tensor<T1> input, Tensor<T2> sequenceLens,
            Tensor<T1> grad,
            long? batch_axis, long? time_axis)
            where T1 : IVarType
            where T2 : IVarType
        {
            Tensor<T1> dInput = OnnxOp.ReverseSequence(grad, sequenceLens,
                batchAxis: batch_axis, timeAxis: time_axis);
            return [dInput, null];
        }
    }
}
