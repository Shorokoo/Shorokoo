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
/// ONNX <c>SequenceConstruct</c>: builds a sequence from its variadic tensor inputs. When
/// every input is a plain <see cref="RuntimeTensor"/> the resulting sequence is in concrete
/// mode. When some inputs are of unknown variant, the sequence falls back to template mode.
/// </summary>
internal sealed class SequenceConstructOp : QuickOp
{
    public override string OpCode => OpCodes.SEQUENCE_CONSTRUCT;

    protected override IRuntimeTensor[] Compute(IRuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var tensors = new List<RuntimeTensor>();
        bool allTensors = true;
        DType dtype = DType.Invalid;
        foreach (var input in inputs)
        {
            if (input is RuntimeTensor rt)
            {
                tensors.Add(rt);
                if (!dtype.IsValid) dtype = rt.DType;
            }
            else if (input is not null)
            {
                allTensors = false;
                if (!dtype.IsValid) dtype = input.DType;
            }
        }

        var seq = new RuntimeSequenceTensor
        {
            DType = dtype,
            Count = inputs.Length,
            Tensors = allTensors ? tensors.ToImmutableArray() : null,
            TemplateTensor = allTensors ? null : SequenceHelpers.BuildTemplate(dtype, tensors),
        };
        return [seq];
    }
}
