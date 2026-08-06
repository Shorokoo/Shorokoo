namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// Collection shared by the Purpose=Benchmark classes. Both measure the machine — one wall
/// clock, one process-wide memory — so neither may run while the other does. A collection is
/// the only reliable lever: xunit never runs two classes in the same collection concurrently,
/// whatever the runner is configured to do. Passing xUnit.MaxParallelThreads=1 on the
/// dotnet test command line does NOT work; it is accepted and silently ignored.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class SerialMeasurement
{
    public const string Name = "serial measurement";
}
