namespace Shorokoo.Tests;

/// <summary>
/// The xUnit collection for classes that read or write <c>DeviceMemory</c>'s process-wide settings.
/// They are one process's state, so a class asserting the shipped defaults and one overriding them
/// cannot run at the same time — which they otherwise would, being in different classes and so in
/// different collections. Any new test touching those statics belongs in a class that joins this.
///
/// <para>It costs the coverage suite nothing today: the only other member is
/// <c>GpuExecutionTests</c>, which is <c>Purpose=Hardware</c> and never in a coverage run — so the
/// serialisation only binds on a machine running both purposes at once, which is exactly the case
/// it is for.</para>
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DeviceMemorySettings
{
    public const string Name = "device memory settings";
}
