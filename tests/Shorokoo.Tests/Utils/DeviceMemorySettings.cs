namespace Shorokoo.Tests;

/// <summary>
/// The xUnit collection for tests that read or write <c>DeviceMemory</c>'s process-wide settings.
/// They are one process's state, so a test asserting the shipped defaults and a test overriding
/// them cannot run at the same time — which they otherwise would, being in different classes and
/// so in different collections. Any new test touching those statics belongs here.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DeviceMemorySettings
{
    public const string Name = "device memory settings";
}
