using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Jax;
using Shorokoo.Jax.Cpu;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;
using Shorokoo.Tests.Modules;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// The Python-based backends' half of <see cref="QeeAudit"/>: an audit module that passed on ONNX
/// Runtime is run again on the PyTorch CPU backend (<see cref="Torch"/>) and the JAX CPU backend
/// (<see cref="Jax"/>), and every value of its main graph must agree — bit for bit, or within
/// <see cref="AutoTest.Tolerance"/> for a floating-point one.
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
/// <para><b>Random draws.</b> Neither torch's generator nor JAX's can reproduce ONNX Runtime's, so a
/// value that depends on a random draw — a random operator's output, and everything computed from
/// it, a node whose body or function draws included — is compared by kind, dtype and shape only. Shape, Size
/// and SequenceLength read nothing of their input but its shape, which is still compared, so they
/// end that dependency. Dropout draws only in training mode, so a Dropout of the main graph counts
/// as a draw only where the <c>training_mode</c> it was given — fed, an initializer, or computed —
/// is true; one inside a body counts unless it has none. Everything else, Shorokoo's own keyed
/// generator included (integer arithmetic in functions), is compared value for value.</para>
///
/// <para><b>Every module runs.</b> Every operator Shorokoo builds is one the backends translate, so
/// any failure — a model a backend refuses, an operator it has no translation for, an attribute the
/// translation does not handle, an exception in the run — fails the audit. The one exception is a
/// JAX refusal of what JAX cannot hold: an operator the JAX backend refuses outright
/// (<see cref="JaxDialect.RefusedOperators"/>, strings, sequences and data-dependent shapes), or one
/// listed in <see cref="JaxKnownRefusals"/> for a module whose graph computes a shape from an input's
/// values. A listed module that no longer refuses fails the audit as well.</para>
///
/// <para>The model is built once, the way a session receives it, and that one model is run on both
/// backends.</para>
///
/// <para><b>Known disagreements.</b> An operator a backend computes differently from ONNX Runtime in
/// a module, for a reason recorded here, is listed under the module and the operator. The operators
/// convicted in a module must be exactly those listed for it: a new one fails the audit, and so does
/// an entry that no longer disagrees.</para>
/// </summary>
internal sealed class QeeAuditOnBackend(
    Func<IShorokooBackend> create,
    Dictionary<(Type Module, string Operator), string> knownDisagreements,
    Dictionary<Type, string>? knownRefusals = null)
{
    private readonly Lazy<IShorokooBackend> _backend = new(create);

    private static readonly Dictionary<(Type Module, string Operator), string> JaxKnownDisagreements = new()
    {
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMax")] = "ONNX: an empty reduction yields -inf or the type's minimum; JAX does, ORT yields 0",
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMin")] = "ONNX: an empty reduction yields +inf or the type's maximum; JAX does, ORT yields 0",
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMean")] = "ONNX: an empty ReduceMean is undefined; JAX yields NaN, ORT 0",
    };

    /// <summary>The modules the JAX backend refuses because their graph computes, from an input's
    /// values, a number an operator needs when the model is compiled, and that operator.</summary>
    private static readonly Dictionary<Type, string> JaxKnownRefusals = new()
    {
    };

    private static readonly Dictionary<(Type Module, string Operator), string> TorchKnownDisagreements = new()
    {
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMax")] = "ONNX: an empty reduction yields -inf or the type's minimum; torch does, ORT yields 0",
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMin")] = "ONNX: an empty reduction yields +inf or the type's maximum; torch does, ORT yields 0",
        [(typeof(QeeEmptyReduceNoIdentityCheck), "ReduceMean")] = "ONNX: an empty ReduceMean is undefined; torch yields NaN, ORT 0",
    };

    // After the tables they read: static fields are initialized in the order they are written.

    /// <summary>The PyTorch CPU backend's audit.</summary>
    public static QeeAuditOnBackend Torch { get; } = new(() => new TorchCpuBackend(), TorchKnownDisagreements);

    /// <summary>The JAX CPU backend's audit.</summary>
    public static QeeAuditOnBackend Jax { get; } = new(() => new JaxCpuBackend(), JaxKnownDisagreements, JaxKnownRefusals);

    public bool Agrees<TModule>(InternalComputationGraph model, TensorData[] inputs, Func<ModelProto, ModelProto>? alterOnBackend = null)
    {
        IData[] feeds = [.. inputs.Select(static t => (IData)t.Shared())];
        var built = ExposeEveryValue(FastOnnxModelBuilder.BuildInternalOnnxModel(model, prepForOnnx: true));
        var reference = Values(ComputeContext.Default.ExecuteModel(model, built, feeds));
        var known = new Dictionary<string, IData>(reference);
        foreach (var (input, value) in built.Graph.Inputs.Zip(inputs)) known.TryAdd(input.Name, value);
        foreach (var initializer in built.Graph.Initializers) known.TryAdd(initializer.Name, Initializer(initializer));
        var onBackendModel = alterOnBackend is null ? built : alterOnBackend(ProtoBuf.Serializer.DeepClone(built));
        using var context = new ComputeContext(_backend.Value);
        Dictionary<string, IData> onBackend;
        try
        {
            onBackend = Values(context.ExecuteModel(model, onBackendModel, feeds));
        }
        catch (Exception ex)
        {
            return Refused(ex) is { } refused
                && (JaxDialect.RefusedOperators.ContainsKey(refused.Operator ?? "") && refused.Reason == JaxUnsupportedReason.UnknownOperator
                    || knownRefusals?.GetValueOrDefault(typeof(TModule)) == refused.Operator && refused.Reason == JaxUnsupportedReason.UnsupportedUsage);
        }
        if (knownRefusals?.ContainsKey(typeof(TModule)) == true) return false;
        var functions = built.Functions.ToDictionary(f => f.Domain + ":" + f.Name);
        var convicted = Convicted(built.Graph.Nodes, reference, onBackend, Drawn(built.Graph.Nodes, functions, known));
        return convicted.SetEquals(knownDisagreements.Keys.Where(k => k.Module == typeof(TModule)).Select(k => k.Operator));
    }

    private JaxUnsupportedModelException? Refused(Exception? ex)
    {
        if (knownRefusals is null) return null;
        for (; ex is not null; ex = ex.InnerException)
            if (ex is JaxUnsupportedModelException refused) return refused;
        return null;
    }

    private static ModelProto ExposeEveryValue(ModelProto model)
    {
        var graph = model.Graph;
        var exposed = graph.Outputs.Select(o => o.Name).Concat(graph.Inputs.Select(i => i.Name)).ToHashSet();
        foreach (var name in graph.Nodes.SelectMany(n => n.Outputs))
            if (name.Length > 0 && exposed.Add(name))
                graph.Outputs.Add(new ValueInfoProto { Name = name });
        return model;
    }

    /// <summary>An initializer's value, as far as <see cref="Draws"/> reads one: its bytes, the
    /// elements it holds in <c>int32_data</c> (where a bool is kept) standing in for raw ones.</summary>
    private static IData Initializer(TensorProto tensor)
        => Globals.TensorData(DType.UInt8, [tensor.RawData is { Length: > 0 } raw ? raw.Length : tensor.Int32Datas?.Length ?? 0],
            tensor.RawData is { Length: > 0 } bytes ? bytes : [.. (tensor.Int32Datas ?? []).Select(v => (byte)(v == 0 ? 0 : 1))]);

    private static Dictionary<string, IData> Values(NamedModelParam[] outputs)
        => outputs.ToDictionary(p => p.ParamName, p => p switch
        {
            TensorDataSequenceModelParam s => (IData)s.ToTensorDataSequence(),
            OptionalTensorDataModelParam o => o.Data,
            _ => p.ToTensorData(),
        });

    private static readonly HashSet<string> RandomOperators =
        ["RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike", "Bernoulli", "Multinomial"];

    private static readonly HashSet<string> ShapeReaders = ["Shape", "Size", "SequenceLength"];

    private static HashSet<string> Drawn(List<NodeProto> nodes, Dictionary<string, FunctionProto> functions, Dictionary<string, IData> known)
    {
        HashSet<string> drawn = [];
        foreach (var node in nodes)
            if (!ShapeReaders.Contains(node.OpType) && (Draws(node, functions, known) || Reads(node).Any(drawn.Contains)))
                drawn.UnionWith(node.Outputs);
        return drawn;
    }

    /// <summary>Whether <paramref name="node"/> draws. A Dropout of the main graph is judged by the
    /// training mode it was actually given — a graph input's fed value, an initializer's, or
    /// another node's output, all in <paramref name="known"/> — and one inside a body, whose value
    /// is not exposed, draws unless its mode is absent.</summary>
    private static bool Draws(NodeProto node, Dictionary<string, FunctionProto> functions, Dictionary<string, IData>? known)
        => RandomOperators.Contains(node.OpType)
            || node.OpType == "Dropout" && node.Inputs.Count > 2 && node.Inputs[2].Length > 0
                && (known is null
                    || (known.TryGetValue(node.Inputs[2], out var mode) && mode is TensorData flag
                        ? flag.CopyRawMemory().Any(b => b != 0)
                        : throw new InvalidOperationException($"The training mode '{node.Inputs[2]}' of Dropout '{node.Name}' has no value to judge it by.")))
            || node.Attributes.SelectMany(a => a.Graphs.Append(a.G)).OfType<GraphProto>()
                .SelectMany(g => g.Nodes).Any(n => Draws(n, functions, null))
            || functions.TryGetValue(node.Domain + ":" + node.OpType, out var function)
                && function.Nodes.Any(n => Draws(n, functions, null));

    private static HashSet<string> Convicted(
        List<NodeProto> nodes, Dictionary<string, IData> reference, Dictionary<string, IData> onBackend, HashSet<string> drawn)
    {
        var disagreeing = reference.Keys.Where(k => !onBackend.TryGetValue(k, out var got) || !Same(reference[k], got, !drawn.Contains(k))).ToHashSet();
        return [.. nodes.Where(n => n.Outputs.Any(disagreeing.Contains) && !Reads(n).Any(disagreeing.Contains)).Select(n => n.OpType)];
    }

    private static IEnumerable<string> Reads(NodeProto node)
        => node.Inputs.Concat(node.Attributes.SelectMany(a => a.Graphs.Append(a.G)).OfType<GraphProto>().SelectMany(Names));

    private static IEnumerable<string> Names(GraphProto graph)
        => graph.Nodes.SelectMany(Reads);

    private static bool Same(IData expected, IData actual, bool values) => (expected, actual) switch
    {
        (TensorData want, TensorData got) => Same(want, got, values),
        (TensorDataSequence want, TensorDataSequence got) => want.Count == got.Count && want.Zip(got).All(p => Same(p.First, p.Second, values)),
        (OptionalTensorData want, OptionalTensorData got) => want.HasValue == got.HasValue && (!want.HasValue || Same(want.Value!, got.Value!, values)),
        _ => false,
    };

    private static bool Same(TensorData expected, TensorData actual, bool values)
    {
        if (expected.DType != actual.DType || !expected.Shape.Equals(actual.Shape)) return false;
        if (!values) return true;
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

/// <summary>The PyTorch CPU backend's audit, as <see cref="QeeAuditOnBackend.Torch"/>.</summary>
internal static class QeeAuditOnTorch
{
    public static bool Agrees<TModule>(InternalComputationGraph model, TensorData[] inputs, Func<ModelProto, ModelProto>? alterOnBackend = null)
        => QeeAuditOnBackend.Torch.Agrees<TModule>(model, inputs, alterOnBackend);
}
