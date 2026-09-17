namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// How ONNX Runtime's device arena grows when it needs a block it does not already hold.
/// </summary>
public enum ArenaExtendStrategy
{
    /// <summary>
    /// ORT's own default: each extension is at least as large as everything the arena already
    /// holds. Its regions are large, and get split and reused, which is what an unpredictable
    /// series of allocation sizes needs — but on a loop whose sizes have settled the doubling is
    /// pure overshoot, and that is what a long training run ends up holding.
    /// </summary>
    NextPowerOfTwo,

    /// <summary>
    /// Extend by exactly the requested size — Shorokoo's default. A loop whose allocation sizes
    /// have settled then tracks them instead of doubling past them. The cost is that an
    /// exactly-sized region cannot serve a later, larger request: a run whose input shapes keep
    /// growing strands each region it outgrows and can need <b>more</b> memory this way.
    /// </summary>
    SameAsRequested,
}

/// <summary>
/// The device-memory configuration a session is built with — ONNX Runtime's
/// <c>gpu_mem_limit</c> and <c>arena_extend_strategy</c> for the CUDA arena that session
/// allocates in. Both are ignored by the CPU backends, which have no device arena.
///
/// <para><b>This is session-scoped, and that is ORT's own shape, not a convention layered on
/// top.</b> ORT gives each session its own arena and reads both values once, while the session
/// is being created, after which the session keeps them for life. So an instance held by a
/// <see cref="Shorokoo.Runtime.ComputeContext"/> configures the sessions that context compiles
/// from then on, two contexts can differ, and a session already compiled is unaffected by any
/// later change. Nothing here is process-wide, and there is no way to reach a session that has
/// already been built: to run under a different budget, compile under a different one.</para>
///
/// <para>It is a record, so a variation is a <c>with</c> expression off an existing one rather
/// than a mutation of shared state:</para>
/// <code>
/// using Shorokoo.Core.Inference.Abstractions;
/// using Shorokoo.Runtime;
///
/// var ctx = new ComputeContext
/// {
///     DeviceMemory = new DeviceMemorySettings { LimitBytes = 16L * 1024 * 1024 * 1024 },
/// };
/// var tight = ctx.DeviceMemory with { ArenaExtend = ArenaExtendStrategy.NextPowerOfTwo };
/// </code>
/// </summary>
public sealed record DeviceMemorySettings
{
    /// <summary>
    /// What a session gets when nothing names otherwise: no budget, and exact-size arena
    /// extension. Immutable and shared — a record, so it cannot be altered in place by one
    /// caller on behalf of every other.
    /// </summary>
    public static DeviceMemorySettings Default { get; } = new();

    private readonly long? _limitBytes;

    /// <summary>
    /// The upper bound, in bytes, on what this session's CUDA arena may allocate — ORT's
    /// <c>gpu_mem_limit</c>. <c>null</c> (the default) leaves ORT free to take the whole
    /// card. A step that needs more than this fails with an ORT <c>BFCArena</c> allocation
    /// error rather than eating into what is left of the device, which is what makes a
    /// too-large configuration fail early and visibly instead of starving everything else
    /// on the machine.
    ///
    /// <para>It caps <b>one session's</b> arena. A process holding several live sessions — a
    /// compiled graph plus a rig, say — can hold this much more than once, so read it as the
    /// ceiling on any one of them and halve it accordingly when two must coexist.</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A limit of zero or less.</exception>
    public long? LimitBytes
    {
        get => _limitBytes;
        init
        {
            // ORT parses this into a size_t, where a negative reads back as SIZE_MAX -- an
            // uncapped arena from a caller who asked for the opposite. Refuse it at the
            // assignment, so no such value can reach a session.
            if (value is <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), value, "The device-memory limit must be positive.");
            _limitBytes = value;
        }
    }

    private readonly ArenaExtendStrategy _arenaExtend = ArenaExtendStrategy.SameAsRequested;

    /// <summary>
    /// How this session's arena extends itself — ORT's <c>arena_extend_strategy</c>. Defaults to
    /// <see cref="ArenaExtendStrategy.SameAsRequested"/>, <b>not</b> to ORT's own
    /// <see cref="ArenaExtendStrategy.NextPowerOfTwo"/>.
    ///
    /// <para>Neither strategy is better in general; the default is a bet on the workload
    /// Shorokoo exists for. A training run feeds one input shape to one compiled step for its
    /// whole length, and on that shape exact-size extension holds about 1.3-1.45x less than ORT's
    /// doubling. (A separate figure from the same report: under ORT's doubling the arena settled at
    /// roughly 1.8x what the run's own steps used — that is the waste being removed, not a ratio
    /// between the two strategies.) Where several
    /// allocation sizes are in play it is the doubling that holds less, but by 1.06-1.13x — an
    /// order of magnitude less at stake. The one case it loses badly is input shapes that keep
    /// growing without settling, where each outgrown region is stranded: set
    /// <see cref="ArenaExtendStrategy.NextPowerOfTwo"/> if that is your workload and device
    /// memory is tight.</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not one of the strategies.</exception>
    public ArenaExtendStrategy ArenaExtend
    {
        get => _arenaExtend;
        init
        {
            if (!Enum.IsDefined(value))
                throw new ArgumentOutOfRangeException(nameof(value), value, "Not an arena-extend strategy.");
            _arenaExtend = value;
        }
    }
}
