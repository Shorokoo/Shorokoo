namespace Shorokoo.Tests;

/// <summary>
/// Collection for the tests that point the process's temp folder somewhere else. Every test that
/// makes a file there reads the same variable, so this collection never runs alongside another.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessWideTempFolder
{
    public const string Name = "process-wide temp folder";
}
