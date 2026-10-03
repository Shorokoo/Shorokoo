namespace Shorokoo.Tests;

/// <summary>
/// The xUnit collection for coverage classes that read or write <c>DeviceMemory</c>'s peak. The
/// peak is one process's observation of its own run, so a class resetting it and one asserting
/// what it holds cannot run at the same time — which they otherwise would, being in different
/// classes and so in different collections. Any new test touching the peak belongs in a class that
/// joins this, or in one of <see cref="ProcessWideMemory"/>, which runs alongside no other.
///
/// <para>The device-memory <i>settings</i> need no such arrangement: a <c>DeviceMemorySettings</c>
/// or <c>RunSettings</c> belongs to the session or the run it was handed to, so two tests
/// configuring them differently never meet.</para>
///
/// <para>Sharing the name is what serializes these against each other, and is all a coverage class
/// touching the peak needs: <c>DisableParallelization</c> would additionally stop the collection
/// running alongside every other one, which costs the whole suite to keep a few classes apart. The
/// hardware classes that read the card are in <see cref="ProcessWideMemory"/> instead, since what
/// they read moves with anything else the process runs.</para>
/// </summary>
[CollectionDefinition(Name)]
public sealed class DeviceMemoryPeak
{
    public const string Name = "device memory peak";
}
