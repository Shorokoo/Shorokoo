using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Close side of an <c>If</c> node pair. <see cref="Execute"/> prepends the paired open node's
/// condition input so the pure <see cref="Compute(IRuntimeTensor?[], OnnxCSharpAttributes, int)"/>
/// sees a flat array:
///   inputs[0]           — condition (bool scalar)
///   inputs[1 .. N]      — <c>else_branch</c> outputs (N outputs)
///   inputs[N+1 .. 2N]   — <c>then_branch</c> outputs
/// where <c>N = (inputs.Length - 1) / 2</c>. Each output may be any <see cref="IRuntimeTensor"/>
/// variant — plain tensor, sequence, or optional. When the condition is statically known the
/// winning branch's value passes through (plain tensors get a fresh copy stripped of
/// iteration-only metadata; sequences/optionals propagate as-is). When the condition is
/// dynamic, plain-tensor branches shape-merge and sequence/optional branches pick one side
/// to forward (the Fast pipeline guarantees both branches have the same variant kind and
/// element dtype, so dynamic-cond sequence outputs only need element-type fidelity).
/// </summary>
internal sealed class IfCloseOp : QuickOp
{
    public override string OpCode => OpCodes.IF_CLOSE;

    public override (IRuntimeTensor[] results, bool loopBack) Execute(
        FastNode node, InternalComputationGraph graph, Dictionary<FastNodeKey, FastNode> nodeByKey,
        Dictionary<FastTensorKey, IRuntimeTensor> store, int maxDataElements)
    {
        var ownInputs = GatherInputs(node.Inputs, store);
        FastNode? openNode = null;
        if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
            nodeByKey.TryGetValue(openKey, out openNode);

        IRuntimeTensor? cond = null;
        if (openNode is not null)
        {
            var openNodeInputs = openNode.Inputs;
            if (openNodeInputs.Count > 0 && openNodeInputs[0] is FastTensorKey condKey)
                store.TryGetValue(condKey, out cond);
        }

        var merged = new IRuntimeTensor?[ownInputs.Length + 1];
        merged[0] = cond;
        Array.Copy(ownInputs, 0, merged, 1, ownInputs.Length);

        return RunCompute(merged, node, maxDataElements);
    }

    protected override IRuntimeTensor[] Compute(
        IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var cond = inputs.Length > 0 ? inputs[0] as RuntimeTensor : null;
        var n = Math.Max(0, (inputs.Length - 1) / 2);

        var results = new IRuntimeTensor[n];
        for (int i = 0; i < n; i++)
        {
            var elseIn = 1 + i < inputs.Length ? inputs[1 + i] : null;
            var thenIn = 1 + n + i < inputs.Length ? inputs[1 + n + i] : null;
            results[i] = MergeOne(thenIn, elseIn, cond);
        }
        return results;
    }

    protected override RuntimeTensor[] Compute(
        RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
        => throw new InvalidOperationException(
            $"{nameof(IfCloseOp)} handles inputs via the IRuntimeTensor overload.");

    private static IRuntimeTensor MergeOne(IRuntimeTensor? thenIn, IRuntimeTensor? elseIn, RuntimeTensor? cond)
    {
        // Cond is statically known: pass the winning branch through as-is.
        if (cond?.BoolData is { Length: > 0 } bd)
        {
            var picked = bd[0] ? thenIn : elseIn;
            if (picked is not null) return CopyOrPassThrough(picked);
        }

        // Cond is dynamic (or static-but-picked-branch-was-null): merge.
        return Merge(thenIn, elseIn);
    }

    /// <summary>
    /// Static-cond winning branch: rebuild plain tensors so callers don't observe
    /// iteration-only metadata that belongs to the branch's body context; sequences and
    /// optionals propagate by reference (immutable records, no need to clone).
    /// </summary>
    private static IRuntimeTensor CopyOrPassThrough(IRuntimeTensor src) => src switch
    {
        RuntimeSequenceTensor seq => seq,
        RuntimeOptionalTensor opt => opt,
        RuntimeTensor t => RuntimeTensorFactory.Create(t.DType, t.Shape) with
        {
            MaxShape = t.MaxShape,
            Rank = t.Rank,
            MaxRank = t.MaxRank,
            FloatData = t.FloatData,
            IntData = t.IntData,
            BoolData = t.BoolData,
            StringData = t.StringData,
        },
        _ => RuntimeTensorFactory.Create(DType.Invalid, null),
    };

    private static IRuntimeTensor Merge(IRuntimeTensor? thenIn, IRuntimeTensor? elseIn)
    {
        // Sequences and optionals propagate by picking the non-null branch — well-formed
        // Shorokoo graphs emit matching variants + matching element dtypes on both sides of
        // an IF_CLOSE branch slot, so a static analyzer at this point only needs to surface
        // the element-type information for downstream nodes.
        if (thenIn is RuntimeSequenceTensor || elseIn is RuntimeSequenceTensor)
            return (thenIn ?? elseIn)!;
        if (thenIn is RuntimeOptionalTensor || elseIn is RuntimeOptionalTensor)
            return (thenIn ?? elseIn)!;

        return MergeTensors(thenIn as RuntimeTensor, elseIn as RuntimeTensor);
    }

    private static RuntimeTensor MergeTensors(RuntimeTensor? thenRt, RuntimeTensor? elseRt)
    {
        var dtype = thenRt?.DType ?? elseRt?.DType ?? DType.Invalid;

        Shape? resultShape = null;
        Shape? maxShape = null;
        if (thenRt?.Shape is not null && elseRt?.Shape is not null
            && thenRt.Shape.Dims.Length == elseRt.Shape.Dims.Length)
        {
            var rank = thenRt.Shape.Dims.Length;
            var exact = new long[rank];
            var max = new long[rank];
            var allEqual = true;
            for (int d = 0; d < rank; d++)
            {
                exact[d] = thenRt.Shape.Dims[d] == elseRt.Shape.Dims[d] ? thenRt.Shape.Dims[d] : -1;
                max[d] = Math.Max(thenRt.Shape.Dims[d], elseRt.Shape.Dims[d]);
                if (exact[d] == -1) allEqual = false;
            }
            if (allEqual) resultShape = new Shape(exact);
            maxShape = new Shape(max);
        }
        else
        {
            resultShape = thenRt?.Shape ?? elseRt?.Shape;
            maxShape = resultShape;
        }

        var merged = RuntimeTensorFactory.Create(dtype, resultShape);
        var mergedMaxRank = Math.Max(thenRt?.MaxRank ?? 0, elseRt?.MaxRank ?? 0);
        int? maxRankOut = mergedMaxRank == 0 ? (thenRt?.Rank ?? elseRt?.Rank ?? merged.Rank) : mergedMaxRank;
        return merged with
        {
            MaxShape = maxShape ?? merged.MaxShape,
            Rank = thenRt?.Rank ?? elseRt?.Rank ?? merged.Rank,
            MaxRank = maxRankOut,
        };
    }
}
