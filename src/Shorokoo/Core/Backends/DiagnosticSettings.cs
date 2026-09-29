namespace Shorokoo.Core.Backends;

/// <summary>
/// What a <see cref="Shorokoo.Runtime.ComputeContext"/> records about the sessions it compiles
/// and the runs they make. <b>Everything here is off by default</b>, and a context that leaves it
/// alone pays nothing: no figures are read, no session is built differently, and
/// <see cref="Shorokoo.Runtime.ComputeContext.RunStats"/> stays empty.
///
/// <para>Like <see cref="DeviceMemorySettings"/> and <see cref="RunSettings"/> it is a record, so
/// a variation is a <c>with</c> expression rather than a mutation something else can see:</para>
/// <code>
/// using Shorokoo.Core.Backends;
/// using Shorokoo.Runtime;
///
/// var ctx = new ComputeContext
/// {
///     Diagnostics = new DiagnosticSettings { CollectRunStatistics = true },
/// };
/// </code>
/// </summary>
public sealed record DiagnosticSettings
{
    /// <summary>What a context gets when it names nothing: every diagnostic off.</summary>
    public static DiagnosticSettings Default { get; } = new();

    /// <summary>
    /// Whether the context reads its sessions' arena figures either side of every run and folds
    /// them into <see cref="Shorokoo.Runtime.ComputeContext.RunStats"/>. Off by default, because
    /// on it costs two calls into the backend per run.
    ///
    /// <para>A backend that reports no arena figures records nothing, so this is free to set on a
    /// context that may run on one.</para>
    /// </summary>
    public bool CollectRunStatistics { get; init; }

    private readonly int _recentRunCapacity = 1000;

    /// <summary>
    /// How many per-run records <see cref="RunStatistics.RecentRuns"/> keeps, oldest dropped
    /// first. 1000 by default, and a ring buffer rather than a list because a training loop is
    /// six figures of runs and one record apiece would grow for the context's whole life.
    ///
    /// <para>It bounds the <i>detail</i> only. <see cref="RunStatistics"/>' aggregates —
    /// <see cref="RunStatistics.PeakBytes"/>, the counts, the totals — are folded in as each run
    /// finishes and stay exact over every run the context has ever made, not just the retained
    /// window. Zero keeps no per-run detail at all and leaves those aggregates untouched.</para>
    ///
    /// <para>The ring is allocated whole when collection starts rather than grown, so the capacity
    /// is paid for up front whether or not that many runs ever happen — about a hundred bytes a
    /// slot. A capacity in the millions is therefore a memory decision, not just a retention
    /// one.</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A negative capacity.</exception>
    public int RecentRunCapacity
    {
        get => _recentRunCapacity;
        init
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _recentRunCapacity = value;
        }
    }

    /// <summary>
    /// Whether the sessions this context compiles record which execution provider ran each node,
    /// readable afterwards through
    /// <see cref="Shorokoo.Runtime.CompiledGraph.ReadNodePlacement"/>. Off by default, and worth
    /// leaving off outside a diagnosis: it builds the session with the runtime's profiler on,
    /// which costs every run that session then makes.
    ///
    /// <para>The cheap question — did any of this graph run on the host — is answered without
    /// this by <see cref="Shorokoo.Runtime.CompiledGraph.OutputPlacement"/>.</para>
    /// </summary>
    public bool TraceNodePlacement { get; init; }

    /// <summary>
    /// Whether the sessions this context compiles compute bit for bit the same result every time
    /// they are run on the same inputs. Off by default, which on a CUDA card is not so: kernels there
    /// add up partial results in whatever order the card's threads finish, so two training runs from
    /// one seed and one data order agree only to about the last digit of the loss, from the first
    /// steps on. With this on, ONNX Runtime runs the deterministic kernel wherever an operator has
    /// both — at some cost in speed on the card — and two such runs, a run resumed from a
    /// checkpoint included, are bit-identical.
    ///
    /// <para>The ONNX Runtime CPU backends are bit-reproducible either way, so this changes nothing
    /// there. It is an ONNX Runtime session setting: the PyTorch and JAX backends build their
    /// sessions as they otherwise would. Like <see cref="TraceNodePlacement"/> it is read when a
    /// session is built, so it takes a context that carries it from the start — a training rig's
    /// <c>runtimeContext</c>, or the one a checkpoint is loaded onto.</para>
    /// </summary>
    public bool DeterministicCompute { get; init; }
}
