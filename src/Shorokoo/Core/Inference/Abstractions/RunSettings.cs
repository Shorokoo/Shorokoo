namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// What a single execution of a compiled session runs with. ONNX Runtime reads these off the
/// run, not the session, so they are settled per call: one run can shrink the arena it
/// finished with while the next leaves it alone, on the same compiled graph and without
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
/// using Shorokoo.Core.Inference.Abstractions;
///
/// var compiled = ctx.Compile(graph);
/// compiled.Execute(inputs);                                                   // the context's default
/// compiled.Execute(inputs, new RunSettings { ShrinkArenaAfterRun = true });    // this run only
/// </code>
/// </summary>
public sealed record RunSettings
{
    /// <summary>What a run gets when the call names nothing and its context holds no other
    /// instance: the arena is left as the run found it.</summary>
    public static RunSettings Default { get; } = new();

    /// <summary>
    /// Whether to hand the arena's unused blocks back to the device when this run finishes —
    /// ORT's <c>memory.enable_memory_arena_shrinkage</c> run option. Off by default, since
    /// the blocks then have to be re-allocated on the next run, which costs a synchronizing
    /// <c>cudaMalloc</c> per step. On, the arena stops being a ratchet: what the run did not
    /// need stays available to the rest of the machine.
    ///
    /// <para>Only a session on a CUDA backend has such an arena; a CPU session ignores this.</para>
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
}
