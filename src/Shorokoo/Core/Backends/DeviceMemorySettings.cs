namespace Shorokoo.Core.Backends;

/// <summary>
/// How ONNX Runtime's device arena grows when it needs a block it does not already hold.
/// </summary>
public enum ArenaExtendStrategy
{
    /// <summary>
    /// Let Shorokoo choose per session — the default. Which strategy wastes less depends on
    /// whether a session's allocation sizes settle: on a series that settles, exact-size
    /// extension holds less — about 1.3-1.45x on a host arena, and 1.12x measured on a CUDA card
    /// over a real training step; on shapes that keep growing, ORT's doubling holds about 1.5x
    /// less and fits on a capped arena where exact-size extension does not. Between those lies a
    /// band where the two are within about 1.1x, or tie.
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
/// The device-memory configuration a CUDA arena is built with — ONNX Runtime's
/// <c>gpu_mem_limit</c> and <c>arena_extend_strategy</c>. It governs both arenas a compute
/// context has: the one each session it compiles allocates in, and the one the tensors it places
/// on its card come out of. Both are ignored by the CPU backends, which have no device arena.
///
/// <para><b>This is arena-scoped, and that is ORT's own shape, not a convention layered on
/// top.</b> ORT gives each session its own arena and reads both values once, while the session
/// is being created, after which the session keeps them for life. So an instance held by a
/// <see cref="Shorokoo.Runtime.ComputeContext"/> configures the sessions that context compiles
/// from then on, two contexts can differ, and a session already compiled is unaffected by any
/// later change: to run under a different budget, compile under a different one.</para>
///
/// <para>The same values settle the arena that context's <i>transfers</i> allocate from — the one
/// a tensor moved onto its card comes out of, which is not a session the program compiled and
/// which lives until the process ends. Assigning here reaches neither kind after the fact; both
/// read these values when they are built and never again.</para>
///
/// <para>It is a record, so a variation is a <c>with</c> expression off an existing one rather
/// than a mutation of shared state:</para>
/// <code>
/// using Shorokoo.Core.Backends;
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
    /// <para>It caps <b>one arena</b>, and a card carries one per live session compiled against
    /// it plus one per set of these settings a tensor has been placed on it under. So read it as
    /// the ceiling on any one of them and divide accordingly: a context that compiles a graph and
    /// a rig and also holds tensors moved onto its card can be holding this much three times over.
    /// Both kinds are countable from the program's own side — the sessions it compiled, and the
    /// configurations it placed tensors under — which is what makes the division possible; what
    /// each is actually holding is reported by
    /// <see cref="Shorokoo.Runtime.CompiledGraph.ReadArenaStatistics"/> and
    /// <see cref="Shorokoo.Runtime.ComputeContext.ReadTransferArenaStatistics"/>.</para>
    ///
    /// <para>They differ in how long they last. A session's arena goes when that session does; the
    /// arena a transfer allocates from is held until the process ends, because the tensors in it
    /// free themselves through it and can outlive every context. So a program that keeps building
    /// fresh settings objects for the same budget keeps opening arenas it can never close, which
    /// is why a card refuses to hold more than a handful of distinct configurations at once.</para>
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
    /// less: about 1.3-1.45x on a host arena over four chained matmuls, and 1.12x measured on a
    /// CUDA card over a 49 M-parameter transformer's training step. Where several allocation
    /// sizes are in play it is the doubling that holds less, but by 1.06-1.13x. The case it wins
    /// outright is input shapes that keep growing without settling, where each outgrown region is
    /// stranded: there the doubling holds about 1.5x less and fits on a capped arena where
    /// exact-size extension does not. Name a strategy here to decide it yourself.</para>
    ///
    /// <para><b>The strategy is the smaller half of what a training step's arena costs.</b> On the
    /// card, that step's arena roughly doubles between its first run and its second under
    /// <i>either</i> strategy — 1.75x and 1.89x — in one extension, and ends up holding about
    /// twice what its steps use (1.94x and 2.15x). The strategy trims that one block; only
    /// <see cref="LimitBytes"/> stops it being taken, and a budget near what the step uses keeps
    /// the run identical while giving back what the doubling would have held.</para>
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
