using Shorokoo.Core.Graph;

namespace Shorokoo.Core.Inference;

/// <summary>
/// The mutable bookkeeping one <see cref="QuickExecutionEngine"/> run accumulates, threaded
/// through <see cref="QuickOp.Execute"/>.
///
/// <see cref="OpRegistry"/> hands out one operator instance per op code for the whole process,
/// so every engine on every thread shares it: an op that has to remember something between
/// invocations must keep it here rather than in a field of its own. A fresh instance per run
/// also keeps one run's leftovers — a loop the engine gave up on part-way, say — out of the
/// next one.
/// </summary>
internal sealed class QuickRunState
{
    /// <summary>
    /// How many iterations each active loop has completed, keyed by its open node.
    /// <c>LOOP_CLOSE</c> reads it to decide whether to go round again, and drops the entry
    /// when the loop terminates.
    /// </summary>
    public Dictionary<FastNodeKey, int> LoopIterations { get; } = new();
}
