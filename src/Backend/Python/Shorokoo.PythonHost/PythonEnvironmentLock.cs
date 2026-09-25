using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace Shorokoo.PythonHost;

/// <summary>
/// The exact set of packages a backend's Python environment holds: a <c>uv pip compile</c> lock,
/// the Python version it was compiled for, and the index arguments it has to be installed with.
///
/// <para>The index arguments are part of the lock, not an afterthought. A compiled lock records
/// versions such as <c>torch==2.14.0+cpu</c> but not where they came from, so installing it
/// without the index it was compiled against fails to find them. They are hashed together with
/// the requirements, so two locks that would install different things never share a cached
/// environment.</para>
/// </summary>
public sealed class PythonEnvironmentLock
{
    /// <summary>Creates a lock.</summary>
    /// <param name="name">A short name for the environment, used in its cache folder's name.</param>
    /// <param name="pythonVersion">The CPython version it was compiled for, as "major.minor".</param>
    /// <param name="requirements">The compiled requirements, as <c>uv pip compile</c> wrote them.</param>
    /// <param name="indexArguments">The arguments naming the indexes to install from, each a
    /// separate token (<c>--index-url</c>, then the URL).</param>
    public PythonEnvironmentLock(
        string name, string pythonVersion, string requirements, IReadOnlyList<string> indexArguments)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(pythonVersion);
        ArgumentNullException.ThrowIfNull(requirements);
        ArgumentNullException.ThrowIfNull(indexArguments);
        Name = name;
        PythonVersion = pythonVersion;
        Requirements = requirements;
        IndexArguments = [.. indexArguments];
        var digest = SHA256.HashData(Encoding.UTF8.GetBytes(
            string.Join('\n', [pythonVersion, requirements.ReplaceLineEndings("\n"), .. IndexArguments])));
        Hash = Convert.ToHexStringLower(digest)[..16];
    }

    /// <summary>The environment's short name, e.g. <c>cpu</c>.</summary>
    public string Name { get; }

    /// <summary>The CPython version the lock was compiled for, as "major.minor".</summary>
    public string PythonVersion { get; }

    /// <summary>The compiled requirements.</summary>
    public string Requirements { get; }

    /// <summary>The index arguments the requirements must be installed with.</summary>
    public IReadOnlyList<string> IndexArguments { get; }

    /// <summary>A digest of everything above, which names the cached environment built from it.</summary>
    public string Hash { get; }

    /// <summary>The cache folder's name: the lock's name and its hash.</summary>
    public string CacheKey => $"{Name}-{Hash}";

    /// <summary>The PyTorch CPU build for this machine's platform, from PyTorch's own <c>cpu</c>
    /// index.</summary>
    /// <exception cref="PlatformNotSupportedException">This is not Linux or Windows on x64, which
    /// are the platforms locks exist for.</exception>
    public static PythonEnvironmentLock Cpu => CpuLock.Value;

    /// <summary>The PyTorch build for CUDA 13 for this machine's platform: PyPI's default wheel on
    /// Linux, and on Windows — where PyPI's wheel is the CPU build — the <c>cu130</c> one from
    /// PyTorch's own index.</summary>
    /// <exception cref="PlatformNotSupportedException">This is not Linux or Windows on x64.</exception>
    public static PythonEnvironmentLock Cu13 => Cu13Lock.Value;

    // Lazy, so that a platform no lock exists for is refused where a lock is asked for rather than
    // in the type initializer, which would make every later use of this class a
    // TypeInitializationException that no longer says why.
    private static readonly Lazy<PythonEnvironmentLock> CpuLock = new(() => ForPlatform("cpu", CurrentPlatform()));
    private static readonly Lazy<PythonEnvironmentLock> Cu13Lock = new(() => ForPlatform("cu13", CurrentPlatform()));

    /// <summary>The platforms a lock exists for, by runtime identifier.</summary>
    public static IReadOnlyList<string> Platforms { get; } = ["linux-x64", "win-x64"];

    /// <summary>
    /// The runtime identifier of the platform this process runs on, as the lock folders name it
    /// (<c>linux-x64</c>, <c>win-x64</c>), or null when no lock exists for it.
    /// </summary>
    public static string? CurrentPlatform() => PlatformOf(
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? OSPlatform.Windows
        : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? OSPlatform.Linux
        : OSPlatform.OSX,
        RuntimeInformation.ProcessArchitecture);

    /// <summary>The lock folder's runtime identifier for <paramref name="os"/> on
    /// <paramref name="architecture"/>, or null when no lock exists for that platform.</summary>
    internal static string? PlatformOf(OSPlatform os, Architecture architecture)
        => architecture != Architecture.X64 ? null
            : os == OSPlatform.Windows ? "win-x64"
            : os == OSPlatform.Linux ? "linux-x64"
            : null;

    /// <summary>
    /// The lock <paramref name="name"/> (<c>cpu</c>, <c>cu13</c>) for <paramref name="platform"/>,
    /// one of <see cref="Platforms"/>: what <see cref="Cpu"/> and <see cref="Cu13"/> answer on that
    /// platform. Each platform's lock was compiled for it alone — wheels differ per platform, and so
    /// can the index — so a lock is never installed on a platform other than its own.
    /// </summary>
    /// <exception cref="PlatformNotSupportedException">No such lock exists for
    /// <paramref name="platform"/>, or <paramref name="platform"/> is null.</exception>
    public static PythonEnvironmentLock ForPlatform(string name, string? platform)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (platform is null || !Platforms.Contains(platform))
            throw new PlatformNotSupportedException(
                $"The Python-based backends have environment locks for {string.Join(" and ", Platforms)} only, "
                + $"and '{platform ?? RuntimeInformation.OSDescription + " on " + RuntimeInformation.ProcessArchitecture}' is not one of them.");
        var folder = $"Environments/{name}/{platform}";
        var requirements = ReadResource($"{folder}/requirements.txt");
        var index = ReadResource($"{folder}/uv-index.txt")
            .Split((char[])[' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        return new PythonEnvironmentLock(name, "3.12", requirements, index);
    }

    private static string ReadResource(string logicalName)
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(logicalName)
            ?? throw new PlatformNotSupportedException(
                $"The lock resource '{logicalName}' is missing from {typeof(PythonEnvironmentLock).Assembly.GetName().Name}.");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    }
}
