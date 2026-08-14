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

    /// <summary>
    /// One entry per loop the walk is currently inside, outermost first. Its depth is what
    /// tags newly-produced tensors as having come from inside a loop (see
    /// <see cref="IRuntimeTensor.IterationIndices"/>), and its frames let <c>LOOP_CLOSE</c>
    /// resolve the index of the node immediately after its paired <c>LOOP_OPEN</c>.
    ///
    /// A frame is only popped when its close node terminates, which a throwing or malformed
    /// close node never reaches — so this rides on the run rather than on the engine, and a
    /// run that gives up part-way cannot leave a later one believing it is inside a loop.
    /// </summary>
    public List<FastLoopFrame> LoopStack { get; } = new();
}

/// <summary>A <c>LOOP_OPEN</c> the walk is currently inside, and where it sits in the node
/// list — the loop-back jump target is the node after it.</summary>
internal readonly struct FastLoopFrame
{
    public readonly FastNode OpenNode;
    public readonly int OpenNodeIndex;

    public FastLoopFrame(FastNode openNode, int openNodeIndex)
    {
        OpenNode = openNode;
        OpenNodeIndex = openNodeIndex;
    }
}
