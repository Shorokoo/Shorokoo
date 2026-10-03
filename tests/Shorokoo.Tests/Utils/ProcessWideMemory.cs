namespace Shorokoo.Tests;

/// <summary>
/// Collection for the tests that read memory as the system, the driver or a process-wide allocator
/// reports it rather than as a session's own account does: the process's private bytes, what the
/// process holds on the card, the card's use and <c>DeviceMemory</c>'s peak, and the figures of the
/// caching allocator PyTorch keeps for each card across the whole process. Anything else running in
/// the process moves every one of those figures, so this collection never runs alongside another.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessWideMemory
{
    public const string Name = "process-wide memory";
}
