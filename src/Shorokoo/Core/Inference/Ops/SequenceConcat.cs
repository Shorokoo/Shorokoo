using System.Collections.Immutable;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Shorokoo-internal <c>#SequenceConcat#</c>: concatenates multiple sequences into a single
/// sequence. Concrete mode is preserved when every input is concrete; otherwise falls back to
/// a merged template and summed count.
/// </summary>
internal sealed class SequenceConcatOp : QuickOp
{
    public override string OpCode => InternalOpCodes.SEQUENCE_CONCAT;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var seqs = inputs.OfType<RuntimeSequenceTensor>().ToList();
        if (seqs.Count == 0)
            return [new RuntimeSequenceTensor { DType = DType.Invalid }];

        var dtype = seqs.Select(s => s.DType).FirstOrDefault(d => d.IsValid) ?? DType.Invalid;
        bool allConcrete = seqs.All(s => s.Tensors is not null);

        if (allConcrete)
        {
            var merged = new List<RuntimeTensor>();
            foreach (var s in seqs) merged.AddRange(s.Tensors!.Value);
            return [new RuntimeSequenceTensor { DType = dtype, Count = merged.Count, Tensors = merged.ToImmutableArray() }];
        }

        long? totalCount = 0;
        foreach (var s in seqs)
        {
            if (s.Count is null) { totalCount = null; break; }
            totalCount += s.Count;
        }

        RuntimeTensor? template = null;
        foreach (var s in seqs) template = SequenceHelpers.MergeTemplates(template, SequenceHelpers.EffectiveTemplate(s));
        return [new RuntimeSequenceTensor { DType = dtype, Count = totalCount, TemplateTensor = template }];
    }
}
