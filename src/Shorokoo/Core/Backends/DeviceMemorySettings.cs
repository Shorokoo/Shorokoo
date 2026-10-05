namespace Shorokoo.Core.Backends;

/// <summary>
/// The device-memory configuration of a compute context: a budget on what it holds in its device's
/// memory (<see cref="LimitBytes"/>), from which the limit on what each of its sessions' runs may
/// allocate there is derived. Ignored where a context's memory is the host's: a device-memory
/// budget does not govern host memory.
///
/// <para><b>The budget is the context's, not one session's.</b> It covers the tensors attached to
/// the context in its device's memory and, while one of its runs executes, what that run's session
/// allocates there, whose limit is cut to what the attached tensors leave. What that means for a
/// transfer, a run and a session is on <see cref="LimitBytes"/>.</para>
///
/// <para>The tensors a context places on its card — <c>To</c>, <c>CopyTo</c>, and the copies its
/// runs make of memory they cannot read where it is — come from the same allocator per card and
/// runtime as its sessions' allocations, shared by every context over that runtime — a backend
/// loaded in isolation has a runtime, and so an allocator, of its own — and held for the life of the
/// process; the budget is kept by counting them against the context they are attached to.</para>
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
/// var tighter = ctx.DeviceMemory with { LimitBytes = 8L * 1024 * 1024 * 1024 };
/// </code>
/// </summary>
public sealed record DeviceMemorySettings
{
    /// <summary>
    /// What a context gets when nothing names otherwise: no budget. Immutable and shared — a record,
    /// so it cannot be altered in place by one caller on behalf of every other.
    /// </summary>
    public static DeviceMemorySettings Default { get; } = new();

    private readonly long? _limitBytes;

    /// <summary>
    /// The compute context's budget, in bytes, on its device's memory: the most it may hold there at
    /// once, counting the tensors attached to it there and what the session of whichever of its runs
    /// is executing allocates there. <c>null</c> (the default) is no budget, and leaves its sessions
    /// free to take the whole card. It is a budget, not a hint: what would pass it is refused, or
    /// fails, rather than eating into what is left of the device.
    ///
    /// <para><b>Transfers.</b> <c>To</c> and <c>CopyTo</c> onto the context, and the copies a run
    /// of it makes of memory it cannot read where it is, are refused
    /// with an <see cref="InvalidOperationException"/> naming the budget, what is attached and what
    /// was asked for, when what is attached plus what they would add passes the limit.
    /// <see cref="Shorokoo.Runtime.ComputeContext.ReadDeviceMemoryUse"/> reads what is attached
    /// against it.</para>
    ///
    /// <para><b>Runs.</b> A run's session may allocate the budget less the <i>discount</i>: what the
    /// context holds in its memory apart from that session's allocations for the length of the run —
    /// the tensors attached to it there, and those the run reads there or copies there to read. A
    /// tensor already on the card is read where it is, so it stays in the discount for the whole
    /// run, and so does every earlier run's output, which the context holds from the moment the run
    /// that made it returns. A host tensor the run consumes is copied onto the card before the run,
    /// like one the run reads, and discounted — once, however many inputs it feeds. What the run
    /// consumed is released as it returns, and drops out. The session's weights, everything its
    /// runs compute and the outputs they make are what the session allocates. A run whose
    /// allocations would pass what it was left fails with an allocation error; one whose discount
    /// leaves nothing at all is refused before it takes anything it was fed.</para>
    ///
    /// <para><b>Sessions.</b> On ONNX Runtime every allocation a session makes on the card goes
    /// through Shorokoo's allocator, which holds the session to its limit as each block is asked for.
    /// So the limit is set before each run to exactly what the budget leaves it, with no need to
    /// build the session again. A backend that fixes a session's limit when the session is built —
    /// PyTorch's — gets the budget less the discount rounded up to the next sixty-fourth of the
    /// budget, keeps the session while the discount stays within what that left, and builds it
    /// again with a lower limit when the discount grows past it — never merely because it has
    /// fallen. <see cref="Shorokoo.Runtime.CompiledGraph.DeviceMemory"/> reports the limit a session
    /// runs under.</para>
    ///
    /// <para><b>One at a time.</b> Under a budget the context's runs are serialized — a second
    /// waits for the first to return — and so are compiles on it and transfers onto it, and every
    /// run hands back the memory its session keeps cached as it ends, whatever
    /// <see cref="RunSettings.ShrinkArenaAfterRun"/> says.</para>
    ///
    /// <para><b>What it does not count.</b> The budget counts tensors and what the running session
    /// allocates. A session's weights count only against its own runs, so a context that has
    /// compiled several graphs with large weights is holding every one of them at once, and can be
    /// holding more than its budget between them. Blocks an allocator keeps cached for other
    /// sessions, and the pinned host memory ONNX Runtime stages crossings through, are not in it
    /// either.</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A limit of zero or less.</exception>
    public long? LimitBytes
    {
        get => _limitBytes;
        init
        {
            if (value is <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), value, "The device-memory limit must be positive.");
            _limitBytes = value;
        }
    }
}
