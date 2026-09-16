namespace Shorokoo.Tests;

/// <summary>
/// The xUnit collection for classes that read or write <c>DeviceMemory</c>'s process-wide settings.
/// They are one process's state, so a class asserting the shipped defaults and one overriding them
/// cannot run at the same time — which they otherwise would, being in different classes and so in
/// different collections. Any new test touching those statics belongs in a class that joins this.
///
/// <para>Sharing the name is what serializes them against each other, and is all that is wanted:
/// <c>DisableParallelization</c> would additionally stop the collection running alongside every
/// other one, which costs the whole suite to solve a problem between two classes.</para>
/// </summary>
[CollectionDefinition(Name)]
public sealed class DeviceMemorySettings
{
    public const string Name = "device memory settings";
}
