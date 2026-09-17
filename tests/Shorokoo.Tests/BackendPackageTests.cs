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

    /// <summary>
    /// The layout the backends actually ship in on Linux and macOS, and the one every consumer of
    /// the NuGet package gets: the native under <c>runtimes/&lt;rid&gt;/native/</c> rather than
    /// flat beside the assembly.
    ///
    /// <para>The flat layout the probe used to insist on is an accident of ONNX Runtime's own
    /// build props, which copy the native to the output root on Windows only. Nothing in the
    /// backend asks for it and nothing on Linux produces it, so a probe that required it refused
    /// every Linux deployment there is — while the .NET host, resolving the P/Invoke through
    /// <c>deps.json</c>, would have loaded the very library the probe said was absent.</para>
    /// </summary>
    [Fact]
    public void TestANativeUnderRuntimesRidNativeResolvesJustAsAFlatOneDoes()
    {
        Assert.True(File.Exists(DeployedNative), $"no native to copy from '{DeployedNative}'");
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

            Assert.True(probe.Supported, probe.Detail);
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

    /// <summary>
    /// A native that is genuinely nowhere is still a <see cref="BackendRejection.MissingNative"/>
    /// naming the file. Widening the search must not turn a broken deployment into a silent pass,
    /// which is the way a fix like this goes wrong.
    /// </summary>
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

            Assert.False(BackendPackage.TryLoad(backend, out var factory, out var failure));
            Assert.Null(factory);
            Assert.Equal(BackendRejection.MissingNative, failure.Reason);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// Order matters: the operating system and the architecture are answered before the natives
    /// are looked for, so a backend built for another platform is refused for being that rather
    /// than for missing a library it was never going to be asked for. Without the ordering, the
    /// reason a caller prints is the wrong one and the remedy it suggests is useless.
    /// </summary>
    [Fact]
    public void TestTheWrongOperatingSystemIsAnsweredBeforeAnyNativeIsLookedFor()
    {
        var root = NewTempDirectory();
        try
        {
            var foreign = Path.Combine(root, ForeignBackendName + ".dll");
            File.Copy(
                Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                    "src", "Backend", "OnnxRuntime", ForeignBackendName, "bin", "Release", "net10.0",
                    ForeignBackendName + ".dll"),
                foreign);

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

    /// <summary>
    /// What the resolver will and will not accept. A flat native wins over a <c>runtimes/</c> one,
    /// because a build that flattened deliberately — the per-backend native folders the
    /// side-by-side tests deploy — means the file it put there. A RID folder narrower than the
    /// portable one still counts, since a package may ship <c>linux-musl-x64</c> or
    /// <c>win10-x64</c>. A folder for another operating system or another architecture never
    /// does: that is a library this process cannot load, and calling it found would turn a clear
    /// rejection into a crash inside the loader.
    /// </summary>
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
}
