namespace Shorokoo.Core.Backends;

/// <summary>
/// What a single execution of a compiled session runs with, settled per call: one run can hand
/// back the memory it finished with while the next keeps it, on the same compiled graph and without
/// rebuilding anything.
///
/// <para>A <see cref="Shorokoo.Runtime.ComputeContext"/> holds the instance its runs use when a
/// call names none, and every <see cref="Shorokoo.Runtime.CompiledGraph"/> run entry point takes
/// one to override it for that call alone. A context's own one-shot entry points —
/// <c>Execute</c>, <c>Run</c>, <c>Eval</c>, <c>ExecuteWithState</c> — and a training rig's
/// <c>TrainStep</c> build or reuse a session per call and run on the context's instance, with no
/// per-call override; set it on the context they run on. Like
/// <see cref="DeviceMemorySettings"/> it is a record, so an override is a <c>with</c> expression
/// rather than a mutation something else can see:</para>
/// <code>
/// using Shorokoo.Core.Backends;
///
/// var compiled = ctx.Compile(graph);
/// compiled.Execute(x.Shared());                                           // the context's default
/// compiled.Execute([x], new RunSettings { ShrinkArenaAfterRun = true });   // this run only
/// </code>
/// </summary>
public sealed record RunSettings
{
    /// <summary>What a run gets when the call names nothing and its context holds no other
    /// instance: what the session keeps for its next runs is kept.</summary>
    public static RunSettings Default { get; } = new();

    /// <summary>
    /// Whether to hand back to the device, when this run finishes, the blocks the session keeps
    /// for its next runs. Off by default, since the blocks then have to be taken again on the next
    /// run, which on a card costs a synchronizing <c>cudaMalloc</c> per step. On, what the session
    /// keeps stops being a ratchet: what the run did not need stays available to the rest of the
    /// machine.
    ///
    /// <para>The memory is the one the run computes in: the card's on a CUDA backend, the host's
    /// on a CPU one. On ONNX Runtime it is what Shorokoo's allocator keeps cached for the session,
    /// and for the tensors placed on the device — which a run that hands its memory back hands back
    /// too — while the blocks still in use, the session's weights and the outputs a caller keeps,
    /// stay as they are.</para>
    /// </summary>
    public bool ShrinkArenaAfterRun { get; init; }

    /// <summary>
    /// Asks the run to stop early. A run whose token is already cancelled when it is handed over
    /// is refused before it costs anything; one cancelled while it is running is stopped at the
    /// next point the backend can stop at. A run stopped either way throws an
    /// <see cref="OperationCanceledException"/> carrying this token, so it is never mistaken for
    /// one whose outputs are real. <see cref="CancellationToken.None"/> by default, which is a run
    /// nothing can stop.
    ///
    /// <para>Stopping is an optimisation, not a guarantee, and every guarantee around a run is
    /// written so as not to need it: a backend that ignores this token runs to completion and
    /// returns its outputs, so whatever waits on the run waits longer and learns the same thing.
    /// A run that reaches its last kernel before the cancellation is seen therefore succeeds, and
    /// hands back the outputs it computed.</para>
    ///
    /// <para>What a backend can stop at is its own business. The ONNX Runtime backend stops
    /// between kernels — it sets ORT's <c>RunOptions.Terminate</c>, which the executor reads
    /// before each node — so the wait is whatever is left of the kernel that was running, and one
    /// kernel can be a whole matmul over a large batch. A graph that <i>is</i> one such kernel has
    /// no boundary to stop at and cannot be stopped at all.</para>
    /// </summary>
    public CancellationToken CancellationToken { get; init; }

    private readonly LogSettings _log = LogSettings.Default;

    /// <summary>
    /// Where the messages the backend's runtime emits during this run go, and from which severity
    /// on: <see cref="LogSettings.Default"/> — warnings and above, as plain lines on the standard
    /// error stream — unless set otherwise, and <see cref="LogSettings.None"/> to drop them all.
    ///
    /// <para>On a context it is also what the sessions it compiles are built under: a runtime's
    /// messages about a model it is handed — a kernel it warns about, an initializer nothing reads —
    /// go to the settings of the context that compiles it. A run of a compiled graph given settings
    /// of its own sends what the run emits there, and nothing of another run's, even of the same
    /// session at the same time. See <see cref="LogSettings"/>.</para>
    /// </summary>
    /// <exception cref="ArgumentNullException">A null settings object.</exception>
    public LogSettings Log
    {
        get => _log;
        init => _log = value ?? throw new ArgumentNullException(nameof(value));
    }
}
