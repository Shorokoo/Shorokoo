using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.PyTorch;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// The PyTorch CPU backend's half of <see cref="QeeAudit"/>: an audit module that passed on ONNX
/// Runtime is run again on torch, and every value of its main graph must agree — bit for bit, or
/// within <see cref="AutoTest.Tolerance"/> for a floating-point one.
///
/// <para><b>Every value, not only the outputs.</b> An audit module checks itself, so its output is
/// usually one bit reduced from many comparisons, and a wrong operator can still leave that bit
/// true. So the ONNX model each backend is handed is rewritten to expose every node output of the
/// main graph as a graph output, and each is compared pairwise: same kind (tensor, sequence,
/// optional), dtype and shape, and the same elements. What happens inside an <c>If</c>/<c>Loop</c>
/// body or a function body is judged by what flows out of that node.</para>
///
/// <para><b>Blame.</b> A value that disagrees although every value its node reads agrees convicts
/// that node's operator; its consumers then disagree as a consequence and are not blamed. A node
/// with a subgraph reads every outer value the subgraph names as well.</para>
///
/// <para><b>Which modules run.</b> Every one whose operators the backend translates, decided by the
/// backend itself: session creation refuses a model using an operator absent from its operator
/// table (<see cref="TorchUnsupportedReason.UnknownOperator"/>), and that refusal alone skips the
/// module. Any other failure — an attribute the translation does not handle, a wrong value, an
/// exception in the run — fails the audit.</para>
///
/// <para><b>Known disagreements.</b> An operator torch computes differently from ONNX Runtime in a
/// module, for a reason recorded here, is listed in <see cref="KnownDisagreements"/> under the
/// module and the operator. The operators convicted in a module must be exactly those listed for
/// it: a new one fails the audit, and so does an entry that no longer disagrees.</para>
/// </summary>
internal static class QeeAuditOnTorch
{
    private static readonly Lazy<TorchCpuBackend> Backend = new(() => new TorchCpuBackend());

    private static readonly Dictionary<(Type Module, string Operator), string> KnownDisagreements = new()
    {
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMax")] = "ONNX: an empty reduction yields -inf or the type's minimum; torch does, ORT yields 0",
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMin")] = "ONNX: an empty reduction yields +inf or the type's maximum; torch does, ORT yields 0",
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMean")] = "ONNX: an empty ReduceMean is undefined; torch yields NaN, ORT 0",
    };

    public static bool Agrees<TModule>(InternalComputationGraph model, TensorData[] inputs)
    {
        IData[] feeds = [.. inputs.Select(static t => (IData)t.Shared())];
        List<NodeProto> nodes = [];
        var reference = Values(ComputeContext.Default.ExecuteRewritten(model, m => ExposeEveryValue(m, nodes), feeds));
        Dictionary<string, IData> onTorch;
        try
        {
            onTorch = Values(new ComputeContext(Backend.Value).ExecuteRewritten(model, m => ExposeEveryValue(m, []), feeds));
        }
        catch (Exception ex) when (IsUntranslatedOperator(ex))
        {
            return true;
        }
        catch (Exception)
        {
            return false;
        }
        var convicted = Convicted(nodes, reference, onTorch);
        return convicted.SetEquals(KnownDisagreements.Keys.Where(k => k.Module == typeof(TModule)).Select(k => k.Operator));
    }

    private static ModelProto ExposeEveryValue(ModelProto model, List<NodeProto> nodes)
    {
        var graph = model.Graph;
        nodes.AddRange(graph.Nodes);
        var exposed = graph.Outputs.Select(o => o.Name).Concat(graph.Inputs.Select(i => i.Name)).ToHashSet();
        foreach (var name in graph.Nodes.SelectMany(n => n.Outputs))
            if (name.Length > 0 && exposed.Add(name))
                graph.Outputs.Add(new ValueInfoProto { Name = name });
        return model;
    }

    private static Dictionary<string, IData> Values(NamedModelParam[] outputs)
        => outputs.ToDictionary(p => p.ParamName, p => p switch
        {
            TensorDataSequenceModelParam s => (IData)s.ToTensorDataSequence(),
            OptionalTensorDataModelParam o => o.Data,
            _ => p.ToTensorData(),
        });

    private static HashSet<string> Convicted(List<NodeProto> nodes, Dictionary<string, IData> reference, Dictionary<string, IData> onTorch)
    {
        var disagreeing = reference.Keys.Where(k => !onTorch.TryGetValue(k, out var got) || !Same(reference[k], got)).ToHashSet();
        return [.. nodes.Where(n => n.Outputs.Any(disagreeing.Contains) && !Reads(n).Any(disagreeing.Contains)).Select(n => n.OpType)];
    }

    private static IEnumerable<string> Reads(NodeProto node)
        => node.Inputs.Concat(node.Attributes.SelectMany(a => a.Graphs.Append(a.G)).OfType<GraphProto>().SelectMany(Names));

    private static IEnumerable<string> Names(GraphProto graph)
        => graph.Nodes.SelectMany(Reads);

    private static bool IsUntranslatedOperator(Exception? ex)
    {
        for (; ex is not null; ex = ex.InnerException)
            if (ex is TorchUnsupportedModelException { Reason: TorchUnsupportedReason.UnknownOperator })
                return true;
        return false;
    }

    private static bool Same(IData expected, IData actual) => (expected, actual) switch
    {
        (TensorData want, TensorData got) => Same(want, got),
        (TensorDataSequence want, TensorDataSequence got) => want.Count == got.Count && want.Zip(got).All(p => Same(p.First, p.Second)),
        (OptionalTensorData want, OptionalTensorData got) => want.HasValue == got.HasValue && (!want.HasValue || Same(want.Value!, got.Value!)),
        _ => false,
    };

    private static bool Same(TensorData expected, TensorData actual)
    {
        if (expected.DType != actual.DType || !expected.Shape.Equals(actual.Shape)) return false;
        if (expected.DType.IsSameElementTypeAs(DType.Utf8)) return expected.Data.SequenceEqual(actual.Data);
        var want = Widened(expected);
        var got = Widened(actual);
        if (want is null || got is null) return expected.CopyRawMemory().AsSpan().SequenceEqual(actual.CopyRawMemory());
        return want.Zip(got).All(p => p.First.Equals(p.Second)
            || Math.Abs(p.First - p.Second) <= AutoTest.Tolerance * Math.Max(1.0, Math.Abs(p.First)));
    }

    private static double[]? Widened(TensorData data)
    {
        var raw = data.CopyRawMemory();
        if (data.DType == DType.Float32) return [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw).ToArray().Select(v => (double)v)];
        if (data.DType == DType.Float64) return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(raw).ToArray();
        if (data.DType == DType.Float16) return [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Float16>(raw).ToArray().Select(v => (double)(float)v)];
        if (data.DType == DType.BFloat16) return [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, BFloat16>(raw).ToArray().Select(v => (double)(float)v)];
        return null;
    }
}
