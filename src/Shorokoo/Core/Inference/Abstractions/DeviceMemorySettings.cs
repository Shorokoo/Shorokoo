namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// How ONNX Runtime's device arena grows when it needs a block it does not already hold.
/// </summary>
public enum ArenaExtendStrategy
{
    /// <summary>
    /// Let Shorokoo choose per session — the default. Which strategy wastes less depends on
    /// whether a session's allocation sizes settle: on a series that settles, exact-size
    /// extension holds about 1.3-1.45x less; on shapes that keep growing, ORT's doubling holds
    /// about 1.5x less and fits on a capped arena where exact-size extension does not. Between
    /// those lies a band where the two are within about 1.1x, or tie.
    ///
    /// <para>Whether a given session's sizes settle is a property of how its caller feeds it, so
    /// Shorokoo does not guess at it. <see cref="Auto"/> is <see cref="SameAsRequested"/> except
    /// where Shorokoo <i>knows</i> a session is reused across differing input shapes — today that
    /// is one case, a training rig's fallback step, which it reaches only after more distinct
    /// shapes have been fed than it keeps specialized steps for and which every later shape then
    /// shares. There the sizes demonstrably do not settle, and exact-size extension is the
    /// strategy that strands a region per outgrown input.</para>
    ///
    /// <para>So this departs from exact-size extension only on evidence, never on a guess. If
    /// your own session is fed shapes that keep growing, say <see cref="NextPowerOfTwo"/> —
    /// Shorokoo has no way to know that before the feeds arrive. What a session was built with
    /// is reported by <see cref="Shorokoo.Runtime.CompiledGraph.DeviceMemory"/>.</para>
    ///
    /// <para>It is a default, not a policy: naming either concrete strategy on a
    /// <see cref="DeviceMemorySettings"/> overrides it for the sessions built from that one, and
    /// no others. What a session actually got is reported by
    /// <see cref="Shorokoo.Runtime.CompiledGraph.DeviceMemory"/>, so the choice can be read back
    /// rather than inferred.</para>
    /// </summary>
    Auto,

    /// <summary>
    /// ORT's own default: each extension is at least as large as everything the arena already
    /// holds. Its regions are large, and get split and reused, which is what an unpredictable
    /// series of allocation sizes needs — but on a loop whose sizes have settled the doubling is
    /// pure overshoot, and that is what a long training run ends up holding.
    /// </summary>
    NextPowerOfTwo,

    /// <summary>
    /// Extend by exactly the requested size. A loop whose allocation sizes have settled then
    /// tracks them instead of doubling past them, and this is what <see cref="Auto"/> resolves to
    /// in every case but one. The cost is that an exactly-sized region cannot serve a later,
    /// larger request: a run whose input shapes keep growing strands each region it outgrows and
    /// can need <b>more</b> memory this way.
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
    /// What a session gets when nothing names otherwise: no budget, and
    /// <see cref="ArenaExtendStrategy.Auto"/>. Immutable and shared — a record, so it cannot be
    /// altered in place by one caller on behalf of every other.
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

    private readonly ArenaExtendStrategy _arenaExtend = ArenaExtendStrategy.Auto;

    /// <summary>
    /// How this session's arena extends itself — ORT's <c>arena_extend_strategy</c>. Defaults to
    /// <see cref="ArenaExtendStrategy.Auto"/>, which is not one of ORT's values but a choice
    /// between them made per session; <see cref="Resolve"/> is that choice.
    ///
    /// <para>Neither concrete strategy is better in general. A training run feeds one input shape
    /// to one compiled step for its whole length, and on that shape exact-size extension holds
    /// about 1.3-1.45x less than ORT's doubling. (A separate figure from the same report: under
    /// ORT's doubling the arena settled at roughly 1.8x what the run's own steps used — that is
    /// the waste being removed, not a ratio between the two strategies.) Where several allocation
    /// sizes are in play it is the doubling that holds less, but by 1.06-1.13x. The case it wins
    /// outright is input shapes that keep growing without settling, where each outgrown region is
    /// stranded: there the doubling holds about 1.5x less and fits on a capped arena where
    /// exact-size extension does not. Name a strategy here to decide it yourself.</para>
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

    /// <summary>
    /// These settings with <see cref="ArenaExtend"/> settled to one of the two strategies ORT
    /// accepts: unchanged when a concrete strategy was named, and otherwise
    /// <see cref="ArenaExtendStrategy.SameAsRequested"/> unless
    /// <paramref name="reusedAcrossShapes"/> says this session is known to be fed differing input
    /// shapes, in which case ORT's doubling.
    ///
    /// <para><paramref name="reusedAcrossShapes"/> is an assertion of fact, not a guess: pass it
    /// only where the differing shapes have already happened. A symbolic session is <b>not</b>
    /// enough on its own — a compiled graph fed one batch shape for its whole life is symbolic
    /// too, and it is the case exact-size extension wins by the widest measured margin.</para>
    ///
    /// <para>The resolution happens here rather than in the backend, so that the settings a
    /// session is built with are concrete by the time anything sees them: the backend never has
    /// to interpret <see cref="ArenaExtendStrategy.Auto"/>, and
    /// <see cref="Shorokoo.Runtime.CompiledGraph.DeviceMemory"/> reports what was chosen rather
    /// than what was asked for.</para>
    /// </summary>
    public DeviceMemorySettings Resolve(bool reusedAcrossShapes)
        => ArenaExtend is not ArenaExtendStrategy.Auto
            ? this
            : this with
            {
                ArenaExtend = reusedAcrossShapes
                    ? ArenaExtendStrategy.NextPowerOfTwo
                    : ArenaExtendStrategy.SameAsRequested,
            };
}
