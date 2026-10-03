using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Shorokoo.Core.Backends;

/// <summary>One file of a pinned library: where its wheel installs it, and the SHA-256 (unpadded
/// base64url, as a wheel's <c>RECORD</c> writes it) and size it has.</summary>
internal sealed record CudaLibraryFile(string WheelPath, string Sha256, long Size)
{
    /// <summary>The file's name, which is the name a backend loads it by.</summary>
    public string FileName => WheelPath[(WheelPath.LastIndexOf('/') + 1)..];
}

/// <summary>
/// One NVIDIA library release every CUDA backend of a process shares: the PyPI wheel it comes from,
/// and the files of it a backend loads, in the order they are loaded.
/// </summary>
internal sealed record CudaLibraryPin(
    string Name, string Version, int CudaMajor, string Package, Uri Wheel, string WheelSha256, long WheelSize,
    IReadOnlyList<CudaLibraryFile> Files)
{
    /// <summary>The name of its folder in the shared cache: <c>cudnn-9.24.0.43-cu13</c>.</summary>
    public string CacheKey => $"{Name}-{Version}-cu{CudaMajor}";

    /// <summary>The library as a sentence names it: <c>cudnn 9.24.0.43 for CUDA 13</c>.</summary>
    public override string ToString() => $"{Name} {Version} for CUDA {CudaMajor}";
}

/// <summary>
/// The NVIDIA libraries pinned for one platform, read from the <c>cuda-libraries.txt</c> written
/// beside the lock of the CUDA Python environment, so the libraries every backend shares are the
/// releases that environment's PyTorch carries. On Windows the file also records the PyTorch wheel
/// the lock pins and its own copies of those files, which is how a test holds the two together
/// without the network.
/// </summary>
internal sealed record CudaLibraryPins(
    IReadOnlyList<CudaLibraryPin> Libraries, string? BundledBy, string? BundledWheelSha256,
    IReadOnlyList<CudaLibraryFile> Bundled)
{
    /// <summary>The platforms pins exist for, by runtime identifier.</summary>
    public static IReadOnlyList<string> Platforms { get; } = ["linux-x64", "win-x64"];

    /// <summary>The pins for this process's platform, or null where there are none.</summary>
    public static CudaLibraryPins? Current => CurrentPins.Value;

    private static readonly Lazy<CudaLibraryPins?> CurrentPins = new(() =>
        RuntimeInformation.ProcessArchitecture != Architecture.X64 ? null
        : OperatingSystem.IsWindows() ? ForPlatform("win-x64")
        : OperatingSystem.IsLinux() ? ForPlatform("linux-x64")
        : null);

    /// <summary>Each release's cache folder and wheel digest: text that differs between any two pin
    /// sets, which an environment's link marker records.</summary>
    public string Identity => string.Join('\n', Libraries.Select(l => $"{l.CacheKey} {l.WheelSha256}"));

    /// <summary>The pins for <paramref name="platform"/>, one of <see cref="Platforms"/>.</summary>
    public static CudaLibraryPins ForPlatform(string platform)
    {
        if (!Platforms.Contains(platform))
            throw new PlatformNotSupportedException($"No CUDA libraries are pinned for '{platform}'.");
        var name = $"CudaLibraries/cu13/{platform}/cuda-libraries.txt";
        using var stream = typeof(CudaLibraryPins).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"The resource '{name}' is missing from {typeof(CudaLibraryPins).Assembly.GetName().Name}.");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return Parse(reader.ReadToEnd());
    }

    /// <summary>Reads a <c>cuda-libraries.txt</c>.</summary>
    public static CudaLibraryPins Parse(string text)
    {
        var libraries = new List<(string[] Fields, List<CudaLibraryFile> Files)>();
        var bundled = new List<CudaLibraryFile>();
        string? bundledBy = null, bundledSha = null;
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var fields = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            switch (fields)
            {
                case ["library", _, _, _, _, _, _, _]:
                    libraries.Add((fields, []));
                    break;
                case ["file", var library, var path, var sha, var size]:
                    libraries.Single(l => l.Fields[1] == library).Files.Add(new(path, sha, long.Parse(size, CultureInfo.InvariantCulture)));
                    break;
                case ["bundled", var package, var sha]:
                    (bundledBy, bundledSha) = (package, sha);
                    break;
                case ["bundles", var path, var sha, var size]:
                    bundled.Add(new(path, sha, long.Parse(size, CultureInfo.InvariantCulture)));
                    break;
                default:
                    throw new FormatException($"'{line}' is not a line of a cuda-libraries.txt.");
            }
        }
        return new(
            [.. libraries.Select(l => new CudaLibraryPin(
                l.Fields[1], l.Fields[2], int.Parse(l.Fields[3], CultureInfo.InvariantCulture), l.Fields[4],
                new Uri(l.Fields[5]), l.Fields[6], long.Parse(l.Fields[7], CultureInfo.InvariantCulture), l.Files))],
            bundledBy, bundledSha, bundled);
    }
}
