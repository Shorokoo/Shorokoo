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
/// The device-memory configuration of a compute context: a budget on what it holds in its device's
/// memory (<see cref="LimitBytes"/>), from which each of its sessions' ONNX Runtime
/// <c>gpu_mem_limit</c> is derived, and the <c>arena_extend_strategy</c> its sessions' CUDA arenas
/// are built with (<see cref="ArenaExtend"/>). Both are ignored where a context's memory is the
/// host's: the CPU backends have no device arena, and a device-memory budget does not govern host
/// memory.
///
/// <para><b>The budget is the context's, not one arena's.</b> It covers the tensors attached to
/// the context in its device's memory and, while one of its runs executes, the arena that run
/// computes in, whose limit is cut to what the attached tensors leave. What that means for a
/// transfer, a run and a session is on <see cref="LimitBytes"/>.</para>
///
/// <para><b>The arena settings are session-scoped, and that is ORT's own shape, not a convention
/// layered on top.</b> ORT gives each session its own arena and reads its limit and strategy once,
/// while the session is being created, after which the session keeps them for life. So an instance
/// held by a <see cref="Shorokoo.Runtime.ComputeContext"/> configures the sessions that context
/// compiles, two contexts can differ, and neither reaches the other's sessions: to run under a
/// different budget or strategy, compile on a context that carries it.</para>
///
/// <para>The tensors a context places on its card — <c>To</c>, <c>CopyTo</c>,
/// <c>AllocateUninitialized</c>, and the copies its runs make of memory they cannot read where it
/// is — come out of none of those arenas. They are allocated from one allocator per card, shared by
/// every context in the process whatever its settings and held for the life of the process; the
/// budget is kept by counting them against the context they are attached to, not by that
/// allocator.</para>
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
    /// What a context gets when nothing names otherwise: no budget, and
    /// <see cref="ArenaExtendStrategy.Auto"/>. Immutable and shared — a record, so it cannot be
    /// altered in place by one caller on behalf of every other.
    /// </summary>
    public static DeviceMemorySettings Default { get; } = new();

    private readonly long? _limitBytes;

    /// <summary>
    /// The compute context's budget, in bytes, on its device's memory: the most it may hold there at
    /// once, counting the tensors attached to it there and the arena of whichever of its runs is
    /// executing. <c>null</c> (the default) is no budget, and leaves ORT free to take the whole card.
    /// It is a budget, not a hint: what would pass it is refused, or fails, rather than eating into
    /// what is left of the device.
    ///
    /// <para><b>Transfers.</b> <c>To</c>, <c>CopyTo</c> and <c>AllocateUninitialized</c> onto the
    /// context, and the copies a run of it makes of memory it cannot read where it is, are refused
    /// with an <see cref="InvalidOperationException"/> naming the budget, what is attached and what
    /// was asked for, when what is attached plus what they would add passes the limit.
    /// <see cref="Shorokoo.Runtime.ComputeContext.ReadDeviceMemoryUse"/> reads what is attached
    /// against it.</para>
    ///
    /// <para><b>Runs.</b> A session's arena gets ORT's <c>gpu_mem_limit</c> of the budget less the
    /// <i>discount</i>: what the context holds in its memory outside that arena for the length of
    /// the run — the tensors attached to it there, and those the run reads there or copies there to
    /// read. A tensor already on the card is read where it is and never enters the arena, so it
    /// stays in the discount for the whole run; one the session's own earlier runs left in its arena
    /// is inside the limit already. What the run consumed is released as it returns, and drops out.
    /// A run that needs more arena than it was left fails with ORT's <c>BFCArena</c> error; one whose
    /// discount leaves no arena at all is refused before it takes anything it was fed.</para>
    ///
    /// <para><b>Sessions.</b> ORT fixes a session's <c>gpu_mem_limit</c> when the session is built,
    /// and building one costs about as much as the graph is large, so a session is built with the
    /// budget less the discount rounded up to the next sixty-fourth of the budget, and kept while
    /// the discount stays within what that left. A run that finds the discount grown past it builds
    /// the session again, with the lower limit; one that finds it fallen keeps the session and its
    /// lower limit. So the limit only ever comes down — at most sixty-four times over a compiled
    /// graph's life as the discount climbs through the budget, and once more each time what is left
    /// halves in its last sixty-fourth — and never in a loop whose discount holds steady.
    /// <see cref="Shorokoo.Runtime.CompiledGraph.DeviceMemory"/> reports the limit a session
    /// got.</para>
    ///
    /// <para><b>One at a time.</b> Under a budget the context's runs are serialized — a second
    /// waits for the first to return — and so are compiles on it and transfers onto it, and every
    /// run hands its arena's unused blocks back as it ends, whatever
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/> says.</para>
    ///
    /// <para><b>What it does not count.</b> The budget counts tensors, not arenas. The blocks an
    /// arena keeps spare, the one allocator per card that tensors are placed from — which holds the
    /// most it was ever asked for at once — and the weights a session keeps in its arena for as long
    /// as it lives are not in it: a session's weights count only against its own runs, so a context
    /// that has compiled several graphs with large weights is holding every one of them at once, and
    /// can be holding more than its budget between them.</para>
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
    /// twice what its steps use (1.94x and 2.15x). The strategy trims that one block; only an arena
    /// limit stops it being taken — the one <see cref="LimitBytes"/> leaves a session — and a limit
    /// near what the step uses keeps the run identical while giving back what the doubling would
    /// have held.</para>
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
