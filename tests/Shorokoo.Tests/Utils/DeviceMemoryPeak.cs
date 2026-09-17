namespace Shorokoo.Tests;

/// <summary>
/// The xUnit collection for classes that read or write <c>DeviceMemory</c>'s peak. The peak is
/// one process's observation of its own run, so a class resetting it and one asserting what it
/// holds cannot run at the same time — which they otherwise would, being in different classes and
/// so in different collections. Any new test touching the peak belongs in a class that joins this.
///
/// <para>The device-memory <i>settings</i> needed no such arrangement once they stopped being
/// process-wide: a <c>DeviceMemorySettings</c> or <c>RunSettings</c> belongs to the session or the
/// run it was handed to, so two tests configuring them differently never meet.</para>
///
/// <para>Sharing the name is what serializes these against each other, and is all that is wanted:
/// <c>DisableParallelization</c> would additionally stop the collection running alongside every
/// other one, which costs the whole suite to solve a problem between two classes.</para>
/// </summary>
[CollectionDefinition(Name)]
public sealed class DeviceMemoryPeak
{
    public const string Name = "device memory peak";
}
