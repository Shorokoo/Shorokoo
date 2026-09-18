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

        Assert.True(probe.Supported);
        Assert.Equal(BackendRejection.None, probe.Reason);
        Assert.Equal(Windows ? "windows" : "linux", probe.Os);
        // What the backend declares, which is what the projects build for -- not a claim about the
        // machine. Asserting the literal made an arm64 host fail here rather than at the one place
        // that would explain it.
        Assert.Equal(RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(), probe.Architecture);
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

    [Fact]
    public void TestAProbeAnswersForEveryFileItIsHandedAndAFailedLoadStrandsNoLoadContext()
    {
        var root = NewTempDirectory();
        try
        {
            // Walking a folder of candidates is the documented use, so no entry in one may throw.
            // A path Path.GetFullPath refuses is the case this can reproduce; the other is a file
            // the process may not open, which throws UnauthorizedAccessException -- not an
            // IOException, so it escaped the handler that named that one.
            (string Path, string Because)[] answered =
            [
                (Path.Combine(root, "na\0me.dll"), "a path no filesystem accepts"),
                (root, "a directory rather than a file"),
                (Path.Combine(root, "absent.dll"), "nothing there at all"),
            ];
            foreach (var (path, _) in answered)
            {
                var probe = BackendPackage.Probe(path);
                Assert.False(probe.Supported);
                Assert.Equal(BackendRejection.Unreadable, probe.Reason);
                Assert.False(BackendPackage.TryLoad(path, out _, out _));
            }

            var before = IsolatedBackend.LoadContexts;
            for (int i = 0; i < 3; i++)
                Assert.Throws<InvalidOperationException>(() => IsolatedBackend.Load(
                    new IsolatedBackendSpec
                    {
                        Name = $"not-a-backend-{i}",
                        BackendAssembly = "Shorokoo",
                        NativeRuntimePath = BackendPackage.ResolveNative(
                            AppContext.BaseDirectory, NativeFileName)!,
                        ProbeDirectory = AppContext.BaseDirectory,
                    }));
            // A load context can never be unloaded, so a failing load must not keep building them.
            Assert.True(IsolatedBackend.LoadContexts - before <= 1);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TestABackendForTheOtherOperatingSystemIsRefusedWithAReason()
    {
        var foreign = ForeignBackendAssembly();
        Assert.True(File.Exists(foreign));

        var probe = BackendPackage.Probe(foreign);

        Assert.False(probe.Supported);
        Assert.Equal(BackendRejection.WrongOperatingSystem, probe.Reason);
        Assert.Contains(Windows ? "linux backend" : "windows backend", probe.Detail);

        // And TryLoad says the same rather than attempting it.
        Assert.False(BackendPackage.TryLoad(foreign, out var backend, out var failure));
        Assert.Null(backend);
        Assert.Equal(BackendRejection.WrongOperatingSystem, failure.Reason);
    }

    [Fact]
    public void TestABackendLoadedFromAPathYieldsAContextThatRuns()
    {
        Assert.True(BackendPackage.TryLoad(NativeBackend, out var backend, out _));
        Assert.NotNull(backend);
        Assert.Equal(ComputeDevice.Cpu, backend!.Description.Device);
        Assert.Equal(MemorySpace.Host, backend.MemorySpace);

        using var context = new ComputeContext(backend, detachesOutputs: true);
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

    /// <summary>The native ONNX Runtime a backend binds, spelled for this platform.</summary>
    private static string NativeFileName => Windows ? "onnxruntime.dll" : "libonnxruntime.so";

    /// <summary>The portable runtime identifier a NuGet native package ships its folder under.
    /// Spelled out rather than taken from <c>RuntimeInformation</c>, so that the layout the test
    /// builds is the one a package really produces and not an echo of the code under test.</summary>
    private static string PortableRid
        => (Windows ? "win" : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "osx" : "linux")
            + "-" + RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();

    /// <summary>
    /// The native this deployment is running on, wherever the build put it. Found by looking in
    /// both layouts by hand: the fixture must not be located with the resolver it is testing.
    /// </summary>
    private static string DeployedNative
    {
        get
        {
            var flat = Path.Combine(AppContext.BaseDirectory, NativeFileName);
            return File.Exists(flat)
                ? flat
                : Path.Combine(AppContext.BaseDirectory, "runtimes", PortableRid, "native", NativeFileName);
        }
    }

    /// <summary>A throwaway folder to lay a deployment out in, deleted by the caller.</summary>
    private static string NewTempDirectory()
        => Directory.CreateDirectory(Path.Combine(
            Path.GetTempPath(), "shorokoo-probe-" + Guid.NewGuid().ToString("n"))).FullName;

    /// <summary>Creates an empty file at <paramref name="path"/>, folders and all. Enough for the
    /// resolver, which asks only whether a native is there.</summary>
    private static string Touch(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, []);
        return path;
    }

    [Fact]
    public void TestANativeUnderRuntimesRidNativeResolvesJustAsAFlatOneDoes()
    {
        Assert.True(File.Exists(DeployedNative));
        var root = NewTempDirectory();
        try
        {
            var backendName = Path.GetFileName(NativeBackend);
            var backend = Path.Combine(root, backendName);
            File.Copy(NativeBackend, backend);
            var native = Path.Combine(root, "runtimes", PortableRid, "native", NativeFileName);
            Directory.CreateDirectory(Path.GetDirectoryName(native)!);
            File.Copy(DeployedNative, native);
            // Nothing flat: the only copy of the native is the one under runtimes/.
            Assert.False(File.Exists(Path.Combine(root, NativeFileName)));

            var probe = BackendPackage.Probe(backend);

            Assert.True(probe.Supported);
            Assert.Equal(BackendRejection.None, probe.Reason);

            // And the path TryLoad hands to IsolatedBackend.Load, which opens it as a file, is
            // where the native really is rather than where a flat deployment would have put it.
            Assert.Equal(native, BackendPackage.ResolveNative(root, NativeFileName));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TestANativeThatIsInNeitherLayoutIsStillMissingNativeNamingTheFile()
    {
        var root = NewTempDirectory();
        try
        {
            var backend = Path.Combine(root, Path.GetFileName(NativeBackend));
            File.Copy(NativeBackend, backend);
            // A runtimes/ tree that carries everything except this platform's native.
            Touch(Path.Combine(root, "runtimes", PortableRid, "native", "some-other-library.bin"));

            var probe = BackendPackage.Probe(backend);

            Assert.False(probe.Supported);
            Assert.Equal(BackendRejection.MissingNative, probe.Reason);
            Assert.Contains(NativeFileName, probe.Detail);
            Assert.Null(BackendPackage.ResolveNative(root, NativeFileName));

            Assert.False(BackendPackage.TryLoad(backend, out var loaded, out var failure));
            Assert.Null(loaded);
            Assert.Equal(BackendRejection.MissingNative, failure.Reason);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TestTheWrongOperatingSystemIsAnsweredBeforeAnyNativeIsLookedFor()
    {
        var root = NewTempDirectory();
        try
        {
            var foreign = Path.Combine(root, ForeignBackendName + ".dll");
            File.Copy(ForeignBackendAssembly(), foreign);

            // A folder with no native of any kind, for either platform.
            var probe = BackendPackage.Probe(foreign);

            Assert.False(probe.Supported);
            Assert.Equal(BackendRejection.WrongOperatingSystem, probe.Reason);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TestNativeResolutionPrefersFlatAcceptsANarrowerRidAndCrossesNoPlatformBoundary()
    {
        var root = NewTempDirectory();
        try
        {
            var flat = Touch(Path.Combine(root, NativeFileName));
            var underRid = Touch(Path.Combine(root, "runtimes", PortableRid, "native", NativeFileName));
            Assert.Equal(flat, BackendPackage.ResolveNative(root, NativeFileName));

            File.Delete(flat);
            Assert.Equal(underRid, BackendPackage.ResolveNative(root, NativeFileName));

            // A RID this code cannot name in advance, but which is still this OS on this
            // architecture, is reached by looking at what the folder actually holds.
            File.Delete(underRid);
            var os = PortableRid[..PortableRid.LastIndexOf('-')];
            var architecture = PortableRid[(PortableRid.LastIndexOf('-') + 1)..];
            var narrower = Touch(Path.Combine(
                root, "runtimes", $"{os}22.04-{architecture}", "native", NativeFileName));
            Assert.Equal(narrower, BackendPackage.ResolveNative(root, NativeFileName));

            // Neither a foreign OS nor a foreign architecture is ever taken for this machine's.
            File.Delete(narrower);
            var foreignOs = Windows ? "linux" : "win";
            Touch(Path.Combine(root, "runtimes", $"{foreignOs}-{architecture}", "native", NativeFileName));
            Touch(Path.Combine(
                root, "runtimes", $"{os}-{(architecture == "x64" ? "arm64" : "x64")}", "native",
                NativeFileName));
            Assert.Null(BackendPackage.ResolveNative(root, NativeFileName));

            // And a folder with no runtimes/ tree at all, or no folder at all, is an answer
            // rather than a throw.
            Assert.Null(BackendPackage.ResolveNative(
                Path.Combine(root, "runtimes", PortableRid), NativeFileName));
            Assert.Null(BackendPackage.ResolveNative(
                Path.Combine(root, "no-such-deployment"), NativeFileName));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// The other operating system's backend assembly, wherever this build put it.
    ///
    /// <para>It is not this test project's output — the foreign backend is referenced only on its
    /// own platform — so it has to be found in the backend project's own bin. Searching both
    /// configurations rather than hard-coding Release matters: the repo sets no default, so
    /// `dotnet build` followed by the documented `dotnet test` is a Debug tree, and a hard-coded
    /// Release path turned that into a red suite that said nothing about the product.</para>
    /// </summary>
    private static string ForeignBackendAssembly()
    {
        var backendBin = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "Backend", "OnnxRuntime", ForeignBackendName, "bin"));
        var candidates = Directory.Exists(backendBin)
            ? Directory.GetFiles(backendBin, ForeignBackendName + ".dll", SearchOption.AllDirectories)
            : [];
        // This build's configuration first, so a stale tree from the other one is never preferred.
        var thisConfiguration = candidates.FirstOrDefault(
            p => p.Contains(Path.DirectorySeparatorChar + Configuration + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase));
        return thisConfiguration ?? candidates.FirstOrDefault() ?? Path.Combine(backendBin, "missing.dll");
    }

    /// <summary>The configuration this test assembly was built in, read off its own location.</summary>
    private static string Configuration =>
        AppContext.BaseDirectory.Contains(
            Path.DirectorySeparatorChar + "Debug" + Path.DirectorySeparatorChar,
            StringComparison.OrdinalIgnoreCase) ? "Debug" : "Release";

}
