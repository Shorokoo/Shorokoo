namespace Shorokoo.Tests;

/// <summary>
/// Collection for the tests that write the process-wide backend state — which factory is live and
/// which one was remembered. Both are read by anything resolving
/// <see cref="Shorokoo.Runtime.ComputeContext.Default"/>, and the default context caches what it
/// resolves, so a test installing a stub while another runs can leave that stub as the default for
/// the rest of the assembly. Disabling parallelization is the only lever that stops it: xunit never
/// runs two classes of one collection at once, nor this collection alongside another.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessWideBackend
{
    public const string Name = "process-wide backend";
}
