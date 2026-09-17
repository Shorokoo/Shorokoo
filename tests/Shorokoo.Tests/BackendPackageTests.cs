using System.Runtime.InteropServices;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// A backend loads from a DLL at runtime, having been referenced by nothing at compile time, and a
/// backend that does not fit this machine says so as a value rather than throwing from inside the
/// loader.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class BackendPackageCoverageTests
{
    private static bool Windows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    private static string Beside(string assemblyName)
        => Path.Combine(AppContext.BaseDirectory, assemblyName + ".dll");

    /// <summary>The backend this deployment carries for the platform it is running on.</summary>
    private static string NativeBackend => Beside(Windows ? "Shorokoo.WinCPU" : "Shorokoo.LinuxCPU");

    /// <summary>The same backend built for the other platform. It is restored by the solution but
    /// not deployed, so the file is reached through the build output of its own project.</summary>
    private static string ForeignBackendName => Windows ? "Shorokoo.LinuxCPU" : "Shorokoo.WinCPU";

    [Fact]
    public void TestThisPlatformsBackendProbesAsSupported()
    {
        var probe = BackendPackage.Probe(NativeBackend);

        Assert.True(probe.Supported, probe.Detail);
        Assert.Equal(BackendRejection.None, probe.Reason);
        Assert.Equal(Windows ? "windows" : "linux", probe.Os);
        Assert.Equal("x64", probe.Architecture);
        Assert.Equal("cpu", probe.Device);
    }

    [Fact]
    public void TestEveryWayOfNotBeingALoadableBackendIsAnAnswerAndNotAThrow()
    {
        (string Path, BackendRejection Reason, string Contains)[] cases =
        [
            (Beside("no-such-backend"), BackendRejection.Unreadable, "no file"),
            // A real file, and not a managed assembly: the test project's own runner config.
            (Path.Combine(AppContext.BaseDirectory, "xunit.runner.json"),
                BackendRejection.Unreadable, "not a managed assembly"),
            // A managed assembly that is not a backend: the framework itself.
            (Beside("Shorokoo"), BackendRejection.NotABackend, "declares no [ShorokooBackend]"),
        ];

        foreach (var (path, reason, contains) in cases)
        {
            var probe = BackendPackage.Probe(path);
            Assert.False(probe.Supported);
            Assert.Equal(reason, probe.Reason);
            Assert.Contains(contains, probe.Detail, StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>
    /// The case the whole probe exists for: a backend built for the other operating system is
    /// turned away with a reason, rather than throwing a BadImageFormatException from the loader
    /// or a DllNotFoundException from the first P/Invoke.
    /// </summary>
    [Fact]
    public void TestABackendForTheOtherOperatingSystemIsRefusedWithAReason()
    {
        var foreign = Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "Backend", "OnnxRuntime", ForeignBackendName, "bin", "Release", "net10.0",
            ForeignBackendName + ".dll");
        foreign = Path.GetFullPath(foreign);
        Assert.True(File.Exists(foreign),
            $"The other platform's backend should be built beside this one, at '{foreign}'.");

        var probe = BackendPackage.Probe(foreign);

        Assert.False(probe.Supported);
        Assert.Equal(BackendRejection.WrongOperatingSystem, probe.Reason);
        Assert.Contains(Windows ? "linux backend" : "windows backend", probe.Detail);

        // And TryLoad says the same rather than attempting it.
        Assert.False(BackendPackage.TryLoad(foreign, out var factory, out var failure));
        Assert.Null(factory);
        Assert.Equal(BackendRejection.WrongOperatingSystem, failure.Reason);
    }

    [Fact]
    public void TestABackendLoadedFromAPathYieldsAContextThatRuns()
    {
        Assert.True(BackendPackage.TryLoad(NativeBackend, out var factory, out var failure),
            failure.Detail);
        Assert.NotNull(factory);
        Assert.Equal(ComputeDevice.Cpu, factory!.Description.Device);
        Assert.Equal(MemorySpace.Host, factory.MemorySpace);

        using var context = new ComputeContext(factory, detachesOutputs: true);
        var a = InputVector<float32>("a");
        var b = InputVector<float32>("b");
        var graph = new InternalComputationGraph([a, b], [a * b + a]);
        float[] av = [1f, 2f, 3f, 4f];
        float[] bv = [10f, 20f, 30f, 40f];

        var result = context.Execute(graph, TensorData([4L], av), TensorData([4L], bv))[0]
            .ToTensorData();

        Assert.Equal([.. av.Zip(bv, (x, y) => x * y + x)],
            result.As<float32>().AccessMemory<float>().ToArray());

        // Detached, so it is still readable once the context that made it is gone.
        context.Dispose();
        Assert.Null(result.Context);
        Assert.Equal(4, result.As<float32>().AccessMemory<float>().Length);
    }

    [Fact]
    public void TestProbingRefusesABlankPathRatherThanAnsweringAboutNothing()
        => Assert.Throws<ArgumentException>(() => BackendPackage.Probe("  "));
}
