namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// What a single execution of a compiled session runs with. ONNX Runtime reads these off the
/// run, not the session, so they are settled per call: one run can shrink the arena it
/// finished with while the next leaves it alone, on the same compiled graph and without
/// rebuilding anything.
///
/// <para>A <see cref="Shorokoo.Runtime.ComputeContext"/> holds the instance its runs use when a
/// call names none, and every run entry point takes one to override it for that call alone.
/// Like <see cref="DeviceMemorySettings"/> it is a record, so an override is a <c>with</c>
/// expression rather than a mutation something else can see:</para>
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
}
