using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Open side of a <c>Loop</c> node pair. Layout of its inputs:
///   inputs[0]              — maxIterations (scalar int64, may be null)
///   inputs[1]              — initial continue condition (scalar bool, may be null)
///   inputs[2 .. 1+N_loop]  — initial loop variables (rank/shape preserved through iterations)
///
/// Outputs:
///   outputs[0]              — iteration index (scalar int64, value unknown at inference time)
///   outputs[1]              — vestigial "true" bool
///   outputs[2 .. 1+N_loop]  — loop variables as seen inside the body; on iteration 0 these
///                             equal the initializers.
/// </summary>
internal sealed class LoopOpenOp : QuickOp
{
    public override string OpCode => OpCodes.LOOP_OPEN;

    protected override IRuntimeTensor[] Compute(
        IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var results = new List<IRuntimeTensor>
        {
            // Iteration index: scalar int64. Always starts at 0.
            RuntimeTensorFactory.Create(DType.Int64, new Shape(Array.Empty<long>()))
                with { IntData = ImmutableArray.Create(0L) },
            // Vestigal "true" bool.
            new RuntimeTensor
            {
                DType = DType.Bool,
                Shape = new Shape(Array.Empty<long>()),
                MaxShape = new Shape(Array.Empty<long>()),
                Rank = 0,
                MaxRank = 0,
                BoolData = ImmutableArray.Create(true),
            },
        };

        // For each loop variable initializer (inputs starting at index 2), produce a tensor with
        // the same shape/dtype/data as the initializer. On iteration 0 the body reads these as
        // the starting values; subsequent iterations overwrite them via loop-back from CLOSE.
        //
        // Loop variables may be any IRuntimeTensor variant — plain tensors, sequences, or
        // optionals. Propagate sequences/optionals as-is so the body can observe their element
        // structure (Count, Tensors, HasValue …); otherwise the base class' default cast would
        // drop them to null and the body would see a DType.Invalid RuntimeTensor.
        for (int i = 2; i < inputs.Length; i++)
        {
            results.Add(MirrorInitializer(inputs[i]));
        }

        return results.ToArray();
    }

    protected override RuntimeTensor[] Compute(
        RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => throw new InvalidOperationException(
            $"{nameof(LoopOpenOp)} handles inputs via the IRuntimeTensor overload.");

    private static IRuntimeTensor MirrorInitializer(IRuntimeTensor? src) => src switch
    {
        null => RuntimeTensorFactory.Create(DType.Invalid, null),
        RuntimeSequenceTensor seq => seq,
        RuntimeOptionalTensor opt => opt,
        RuntimeTensor t => RuntimeTensorFactory.Create(t.DType, t.Shape) with
        {
            MaxShape = t.MaxShape ?? t.Shape,
            Rank = t.Rank,
            MaxRank = t.MaxRank,
            FloatData = t.FloatData,
            IntData = t.IntData,
            BoolData = t.BoolData,
            StringData = t.StringData,
        },
        _ => RuntimeTensorFactory.Create(DType.Invalid, null),
    };
}
