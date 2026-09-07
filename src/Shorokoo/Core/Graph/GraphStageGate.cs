using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;

namespace Shorokoo.Graph
{
    /// <summary>
    /// The stage gate the execution paths share: a graph still carrying ops that lowering removes
    /// cannot run, and is refused here naming those ops and the lowering that removes them, rather
    /// than deep inside ONNX Runtime naming an op that appears nowhere in the public API
    /// (<c>No Op registered for ShrkCreateModule</c>).
    ///
    /// <para>Two callers, one message. <see cref="ComputationGraph.RequireConcretized"/> reads the
    /// stamp, which is the cheap and reliable answer where there is one; the op scan below serves
    /// the graphs built from output variables (the <c>Eval</c> family), which have no stamp, and
    /// backs up the ONNX Runtime session paths against a graph stamped past that check by
    /// <c>FromInternal</c> or <c>WithKind</c>. Only those session paths are gated:
    /// <c>QuickExecutionEngine</c> interprets a graph rather than building a session, so it never
    /// reaches the error this exists to pre-empt.</para>
    ///
    /// <para>The <c>Eval</c> family calls the scan again ahead of <c>ComputeContext</c>, which would
    /// catch the same graphs: the second call buys only the <c>operation</c> label, and
    /// the label is what makes the message open with the API the reader actually typed.</para>
    /// </summary>
    internal static class GraphStageGate
    {
        /// <summary>
        /// Refuses <paramref name="graph"/> when it carries ops the pipeline cannot lower away.
        /// <paramref name="operation"/> names the API the caller used, so the message opens with
        /// what the reader typed.
        /// </summary>
        internal static void RequireRunnableOps(this InternalComputationGraph graph, string operation)
        {
            foreach (var node in graph.Nodes)
                if (node.IsUnrunnableModuleOp())
                    throw Refusal(operation,
                        graph.Nodes.Select(n => (n.OpCode, n.TargetFunction, n.IsUnrunnableModuleOp())));
        }

        /// <summary>
        /// The refusal for a graph whose <paramref name="nodes"/> carry unrunnable machinery, as
        /// <c>(op code, target, is machinery)</c> triples — the shape both a
        /// <see cref="InternalComputationGraph"/> and a frozen <see cref="ComputationGraph"/> can
        /// offer without either knowing about the other's node type.
        /// </summary>
        internal static System.InvalidOperationException Refusal(
            string operation, IEnumerable<(string OpCode, Function? Target, bool IsMachinery)> nodes)
        {
            var machinery = nodes.Where(n => n.IsMachinery).ToList();

            // Ordinal order would lead with the '#Sigil#' names, which mean nothing outside the
            // codebase, and bury the Shrk* ops that actually say "module machinery".
            var ops = string.Join(", ", machinery
                .Select(n => n.OpCode)
                .Distinct()
                .OrderBy(op => op.StartsWith('#') ? 1 : 0)
                .ThenBy(op => op, System.StringComparer.Ordinal));

            // Named as a label, never spelled into code: the name a module carries is its
            // [Module] type's for a generated one, but only the declaring type's for a module built
            // from a delegate — which has no ComputationGraph of its own to lower.
            var modules = machinery
                .Where(n => n.Target is { FunctionType: FunctionType.Module })
                .Select(n => n.Target!.FriendlyName)
                .Distinct()
                .ToList();

            const string Lowering = "ToConcreteArchitecture(inputHints) then ToConcreteModel()";
            const string InOrder = "passing a value for each of its inputs in order "
                + "([Hyper] parameters come first)";

            var remedy = modules.Count switch
            {
                0 => $"Lower the graph the whole way — {Lowering} — and execute that, {InOrder}.",
                1 => $"It comes from module '{modules[0]}': lower that module's ComputationGraph the "
                     + $"whole way — {Lowering} — and execute that, {InOrder}.",
                _ => $"It comes from modules {string.Join(", ", modules.Select(m => $"'{m}'"))}: each "
                     + $"has to be lowered the whole way — {Lowering} — and run as its own model, "
                     + $"{InOrder}.",
            };

            return new System.InvalidOperationException(SrkFileFormat.MachineryMismatchMessage(
                operation, "a concretized graph (a 'concrete-architecture' or 'concrete-model')",
                GraphKind.Module,
                $"It still carries module machinery that lowering removes ({ops}). {remedy} "
                + "See Documentation/inference.md#running-a-module."));
        }
    }
}
