using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Helpers for constructing and manipulating <see cref="RuntimeSequenceTensor"/> summaries.
/// </summary>
internal static class SequenceHelpers
{
    /// <summary>
    /// Computes a template tensor that summarises every tensor in <paramref name="tensors"/>:
    /// the strongest shape/rank information that is valid for every element.
    ///   - Exact Shape is kept only when every element agrees on every dimension.
    ///   - MaxShape is kept when every element has the same rank; each dim is the element-wise max.
    ///   - Rank is kept when every element agrees on rank; otherwise MaxRank is used.
    ///   - DType is the first non-invalid dtype seen (all tensors in a sequence share dtype by spec).
    /// </summary>
    public static RuntimeTensor BuildTemplate(DType dtypeHint, IEnumerable<RuntimeTensor?> tensors)
    {
        var list = tensors.Where(t => t is not null).Cast<RuntimeTensor>().ToList();
        var dtype = list.Select(t => t.DType).FirstOrDefault(d => d.IsValid) ?? DType.Invalid;
        if (!dtype.IsValid) dtype = dtypeHint;

        if (list.Count == 0)
            return RuntimeTensorFactory.Create(dtype, null);

        // Shape agreement: only keep an exact shape when every element has a known identical shape.
        var shapes = list.Select(t => t.Shape).ToList();
        Shape? sharedShape = null;
        if (shapes.All(s => s is not null))
        {
            var first = shapes[0]!;
            if (shapes.All(s => s!.Equals(first))) sharedShape = first;
        }

        // Rank agreement.
        var ranks = list.Select(t => t.Rank ?? t.Shape?.Dims.Length).ToList();
        int? sharedRank = null;
        if (ranks.All(r => r.HasValue))
        {
            var r0 = ranks[0]!.Value;
            if (ranks.All(r => r!.Value == r0)) sharedRank = r0;
        }

        // Max-shape when all ranks agree.
        Shape? maxShape = null;
        if (sharedRank is int rk)
        {
            var max = new long[rk];
            bool ok = true;
            for (int i = 0; i < list.Count; i++)
            {
                var s = list[i].Shape ?? list[i].MaxShape;
                if (s is null || s.Dims.Length != rk) { ok = false; break; }
                for (int d = 0; d < rk; d++) if (s.Dims[d] > max[d]) max[d] = s.Dims[d];
            }
            if (ok) maxShape = new Shape(max);
        }

        var maxRank = list.Max(t => t.MaxRank ?? t.Rank ?? t.Shape?.Dims.Length ?? 0);

        var template = RuntimeTensorFactory.Create(dtype, sharedShape);
        return template with
        {
            MaxShape = maxShape ?? sharedShape,
            Rank = sharedRank,
            MaxRank = sharedRank ?? (maxRank > 0 ? maxRank : null),
        };
    }

    /// <summary>
    /// Merges two templates (for example, when concatenating two sequences). The returned
    /// template describes the weakest-common-information tensor across both.
    /// </summary>
    public static RuntimeTensor MergeTemplates(RuntimeTensor? a, RuntimeTensor? b)
    {
        if (a is null && b is null) return RuntimeTensorFactory.Create(DType.Invalid, null);
        if (a is null) return CloneAsTemplate(b!);
        if (b is null) return CloneAsTemplate(a);

        var dtype = a.DType.IsValid ? a.DType : b.DType;

        var combined = new List<RuntimeTensor> { a, b };
        return BuildTemplate(dtype, combined);
    }

    private static RuntimeTensor CloneAsTemplate(RuntimeTensor src)
    {
        var rt = RuntimeTensorFactory.Create(src.DType, src.Shape);
        return rt with
        {
            MaxShape = src.MaxShape ?? src.Shape,
            Rank = src.Rank,
            MaxRank = src.MaxRank ?? src.Rank,
        };
    }

    /// <summary>
    /// Returns the template tensor for a sequence: its <see cref="RuntimeSequenceTensor.TemplateTensor"/>
    /// when in template mode, or a freshly computed template of its concrete elements.
    /// </summary>
    public static RuntimeTensor? EffectiveTemplate(RuntimeSequenceTensor seq)
    {
        if (seq.TemplateTensor is not null) return seq.TemplateTensor;
        if (seq.Tensors is { } tensors) return BuildTemplate(seq.DType, tensors);
        return null;
    }
}
