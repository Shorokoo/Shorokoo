using Shorokoo.Core.Backends;

namespace Shorokoo.OnnxRuntime;

/// <summary>
/// The severity a message ONNX Runtime logs is delivered at: its own, but for the messages listed
/// here, which report on something no user can act on and are delivered at
/// <see cref="ShorokooLogSeverity.Verbose"/> instead. They stay readable there, and at the default
/// severity every warning that reaches a sink is one worth reading.
///
/// <para>A message is known by where in ONNX Runtime it is logged — the source file and the
/// function, not the line, which moves from one release to the next — and by what it says. A
/// message from one of those places that says something else, and every message from anywhere
/// else, keeps the severity ONNX Runtime gave it.</para>
/// </summary>
internal static class OrtLogTriage
{
    private readonly record struct Known(string File, string Function, string Text);

    private static readonly Known[] NoUserCanActOn =
    [
        // Shorokoo has the runtime write the graph it optimized out, to read which values the
        // graph it runs still holds and which outputs it may write into its inputs. The file is
        // Shorokoo's own and is never handed to another machine.
        new("inference_session.cc", "onnxruntime::InferenceSession::Initialize",
            "Serializing optimized model with Graph Optimization level greater than ORT_ENABLE_EXTENDED"),

        // Which nodes a CUDA session leaves to the host, and the copies that adds: what
        // DiagnosticSettings.TraceNodePlacement and CompiledGraph.OutputPlacement report on
        // demand, node by node, rather than once per session built. A random draw is among them:
        // its bitwise operators have no CUDA kernel.
        new("session_state.cc", "onnxruntime::VerifyEachNodeIsAssignedToAnEp",
            "Some nodes were not assigned to the preferred execution providers"),
        new("session_state.cc", "onnxruntime::VerifyEachNodeIsAssignedToAnEp",
            "Rerunning with verbose output on a non-minimal build will show node assignments"),
        new("transformer_memcpy.cc", "onnxruntime::MemcpyTransformer::ApplyImpl",
            "Memcpy nodes are added to the graph"),

        // A node the runtime's optimizer could not fold into a constant, which the run then
        // computes: an operator of an IfElse branch the inputs never take, whose constants are out
        // of range in the branch alone, or a CastLike, which the runtime has no kernel to fold with.
        new("constant_folding.cc", "onnxruntime::ConstantFolding::ApplyImpl",
            "Failure during constant folding of "),
        new("constant_folding.cc", "onnxruntime::ConstantFolding::ApplyImpl",
            "Could not find a CPU kernel and hence can't constant fold "),

        // Two of the runtime's own shape inferences of one value disagreeing, which it settles
        // itself: about its inference over the graph Shorokoo lowered, which states no shape for
        // a value inside it.
        new("graph.cc", "onnxruntime::MergeShapeInfo", "Error merging shape info for output."),

        // A weight a session reads from memory it already holds (SuppliedInitializer), which the
        // model lists among its inputs too so that the runtime does not fold it into a constant.
        new("graph.cc", "onnxruntime::Graph::Graph", "appears in graph inputs and will not be treated as constant value/weight"),

        // A node failing as a run computes it, which fails the run: the exception it raises
        // carries the same message.
        new("sequential_executor.cc", "onnxruntime::ExecuteKernel", "Non-zero status code returned while running "),
    ];

    /// <summary>The severity the message <paramref name="text"/>, which ONNX Runtime logged at
    /// <paramref name="severity"/> from <paramref name="location"/> (<c>graph.cc:124
    /// onnxruntime::MergeShapeInfo</c>), is delivered at.</summary>
    internal static ShorokooLogSeverity Of(ShorokooLogSeverity severity, string? location, string? text)
    {
        if (severity == ShorokooLogSeverity.Verbose || string.IsNullOrEmpty(location) || string.IsNullOrEmpty(text))
            return severity;
        var space = location.IndexOf(' ');
        var file = location.AsSpan(0, space < 0 ? location.Length : space);
        if (file.IndexOf(':') is var colon and >= 0) file = file[..colon];
        var function = space < 0 ? ReadOnlySpan<char>.Empty : location.AsSpan(space + 1).Trim();
        foreach (var known in NoUserCanActOn)
            if (file.SequenceEqual(known.File) && function.SequenceEqual(known.Function)
                && text.Contains(known.Text, StringComparison.Ordinal))
                return ShorokooLogSeverity.Verbose;
        return severity;
    }
}
