using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The per-user cache the pinned NVIDIA libraries live in, one folder per release —
/// <c>%LOCALAPPDATA%\shorokoo\cuda\cudnn-9.24.0.43-cu13\</c> on Windows,
/// <c>$XDG_CACHE_HOME/shorokoo/cuda/…</c> (or <c>~/.cache/…</c>) elsewhere — beside the Python
/// environments, and filled the way they are: once, under a file lock beside the folder, which is
/// marked complete only after every file is in place and checked, so a folder an interrupted fill
/// left behind is filled again rather than used.
///
/// <para>A folder is filled from an installed copy of the release where one matches exactly — every
/// pinned file there, each with the SHA-256 its wheel's <c>RECORD</c> gives — hard-linked where the
/// volume allows and copied otherwise; and from the release's wheel on PyPI where none does. A copy
/// that differs in any file is never used, so no other release, and no build for another CUDA
/// major, is ever mixed in.</para>
/// </summary>
internal static partial class CudaLibraryCache
{
    /// <summary>The file a filled folder holds, naming the wheel it was filled from.</summary>
    internal const string CompleteMarker = ".shorokoo-provisioned";

    /// <summary>The file an environment whose copies of the pinned files are links into the cache
    /// holds, naming the pins it was linked against.</summary>
    internal const string LinkedMarker = ".shorokoo-cuda-libraries";

    /// <summary>How long a fill waits for another process's, and how long a download may take.</summary>
    internal static readonly TimeSpan DefaultTimeout = TimeSpan.FromHours(1);

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Gates = new(StringComparer.Ordinal);

    private static readonly Lazy<HttpClient> Http = new(() =>
    {
        var client = new HttpClient { Timeout = DefaultTimeout };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("Shorokoo");
        return client;
    });

    /// <summary>The cache folder of this user.</summary>
    internal static string DefaultRoot => Root(Environment.GetEnvironmentVariable, OperatingSystem.IsWindows());

    /// <summary>The cache folder, reading environment variables through <paramref name="variables"/>,
    /// on Windows or not as <paramref name="windows"/> says.</summary>
    internal static string Root(Func<string, string?> variables, bool windows) => Path.Combine(UserCache(variables, windows), "cuda");

    /// <summary>The user cache folder Shorokoo keeps its NVIDIA libraries and Python environments in.</summary>
    private static string UserCache(Func<string, string?> variables, bool windows)
    {
        var root = windows
            ? variables("LOCALAPPDATA") is { Length: > 0 } local ? local : Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
            : variables("XDG_CACHE_HOME") is { Length: > 0 } xdg ? xdg : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        return Path.Combine(root, "shorokoo");
    }

    /// <summary>Whether <paramref name="directory"/> is a filled folder of <paramref name="pin"/>.</summary>
    internal static bool IsComplete(string directory, CudaLibraryPin pin)
    {
        var marker = Path.Combine(directory, CompleteMarker);
        try
        {
            return File.Exists(marker) && File.ReadAllText(marker).Trim() == pin.WheelSha256;
        }
        catch (IOException)
        {
            return false;
        }
    }

    /// <summary>
    /// Where an installed copy of <paramref name="pin"/> is looked for, in order. First the provisioned
    /// Python environments of the same CUDA major beside the cache, whose PyTorch carries the pinned
    /// release — <c>torch\lib</c> on Windows, the NVIDIA wheels' own folders on Linux — and which, on
    /// the cache's volume, fill it by hard link; then the folders on <c>PATH</c>
    /// (<c>LD_LIBRARY_PATH</c> elsewhere); then where NVIDIA's installers put the library. For cuDNN that
    /// is <c>%CUDNN_PATH%</c> and every folder under <c>%ProgramFiles%\NVIDIA\CUDNN</c>; for cuBLAS,
    /// the CUDA toolkit's <c>bin</c>: <c>%CUDA_PATH%</c>'s and every CUDA 13 toolkit's under
    /// <c>%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA</c>. Folders that are not there are left
    /// in; they hold no match. Reading the environments' folders starts no Python and depends on
    /// nothing of the Python host: they are files, held to the pin like any other copy.
    /// </summary>
    internal static IReadOnlyList<string> InstalledCandidates(CudaLibraryPin pin, Func<string, string?> variables, bool windows)
    {
        IEnumerable<string> Split(string? list) => (list ?? "")
            .Split(windows ? ';' : ':', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        IEnumerable<string> Under(string? root, params string[] parts)
            => root is { Length: > 0 } ? [Path.Combine([root, .. parts])] : [];
        IEnumerable<string> Each(string? root, string pattern, Func<string, IEnumerable<string>> within)
        {
            if (root is not { Length: > 0 } || !Directory.Exists(root)) return [];
            try
            {
                return [.. Directory.EnumerateDirectories(root, pattern).Order(StringComparer.Ordinal).SelectMany(within)];
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return [];
            }
        }
        IEnumerable<string> AllWithin(string folder)
            => [folder, .. Directory.EnumerateDirectories(folder, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal)];

        IEnumerable<string> SitePackages(string environment) => windows
            ? [Path.Combine(environment, "Lib", "site-packages")]
            : Each(Path.Combine(environment, "lib"), "python3.*", lib => [Path.Combine(lib, "site-packages")]);
        var environments = Each(Path.Combine(UserCache(variables, windows), "python-envs"), $"cu{pin.CudaMajor}-*",
            environment => SitePackages(environment)
                .SelectMany(sitePackages => pin.Files.SelectMany(file => EnvironmentFiles(sitePackages, file)))
                .Select(path => Path.GetDirectoryName(path)!)
                .Distinct());

        var programFiles = variables("ProgramFiles");
        IEnumerable<string> installed = (windows, pin.Name) switch
        {
            (true, "cudnn") =>
            [
                .. Split(variables("PATH")), .. Under(variables("CUDNN_PATH")), .. Under(variables("CUDNN_PATH"), "bin"),
                .. Under(variables("CUDNN_PATH"), "bin", "x64"),
                .. Each(Under(programFiles, "NVIDIA", "CUDNN").SingleOrDefault(), "v*", AllWithin),
            ],
            (true, _) =>
            [
                .. Split(variables("PATH")), .. Under(variables("CUDA_PATH"), "bin"), .. Under(variables("CUDA_PATH"), "bin", "x64"),
                .. Each(Under(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA").SingleOrDefault(), $"v{pin.CudaMajor}.*",
                    toolkit => [Path.Combine(toolkit, "bin"), Path.Combine(toolkit, "bin", "x64")]),
            ],
            (false, "cudnn") =>
            [
                .. Split(variables("LD_LIBRARY_PATH")), .. Under(variables("CUDNN_PATH"), "lib"), .. Under(variables("CUDNN_PATH"), "lib64"),
                "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/cuda/lib64",
            ],
            _ =>
            [
                .. Split(variables("LD_LIBRARY_PATH")), .. Under(variables("CUDA_PATH"), "lib64"), .. Under(variables("CUDA_HOME"), "lib64"),
                .. Each("/usr/local", $"cuda-{pin.CudaMajor}*", toolkit => [Path.Combine(toolkit, "lib64")]),
                "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
            ],
        };
        return [.. environments.Concat(installed).Distinct(windows ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal)];
    }

    /// <summary>Whether <paramref name="directory"/> holds every file of <paramref name="pin"/>, each
    /// with the size and SHA-256 the pin gives it.</summary>
    internal static bool MatchesExactly(string directory, CudaLibraryPin pin)
    {
        try
        {
            // Every size before any hash: a folder of another release is turned away without
            // reading a gigabyte of it.
            return Directory.Exists(directory)
                && pin.Files.All(file => File.Exists(Path.Combine(directory, file.FileName))
                    && new FileInfo(Path.Combine(directory, file.FileName)).Length == file.Size)
                && pin.Files.All(file => Matches(Path.Combine(directory, file.FileName), file));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>Whether the file at <paramref name="path"/> is <paramref name="file"/>.</summary>
    internal static bool Matches(string path, CudaLibraryFile file)
    {
        using var stream = File.OpenRead(path);
        return stream.Length == file.Size && Sha256Of(stream) == file.Sha256;
    }

    /// <summary>A stream's SHA-256, written as a wheel's <c>RECORD</c> writes it.</summary>
    internal static string Sha256Of(Stream stream)
        => Convert.ToBase64String(SHA256.HashData(stream)).TrimEnd('=').Replace('+', '-').Replace('/', '_');

    /// <summary>
    /// The filled folder of <paramref name="pin"/> under <paramref name="root"/>, filling it first if
    /// it is not: from the first of <paramref name="localSources"/> that matches exactly, or else from
    /// the wheel. Serialised across threads and processes by a lock file beside the folder.
    /// </summary>
    /// <exception cref="InvalidOperationException">No source matches and the wheel could not be
    /// fetched; the message names the release, where it was looked for and the download's size.</exception>
    internal static string Provision(CudaLibraryPin pin, string root, IEnumerable<string> localSources, TimeSpan timeout)
    {
        var directory = Path.Combine(root, pin.CacheKey);
        if (IsComplete(directory, pin)) return directory;
        Directory.CreateDirectory(root);
        var gate = Gates.GetOrAdd(directory, static _ => new SemaphoreSlim(1, 1));
        var clock = Stopwatch.StartNew();
        if (!gate.Wait(timeout)) throw TimedOut(directory, timeout);
        try
        {
            using var fileLock = AcquireFileLock(directory + ".lock", timeout - clock.Elapsed, directory, timeout);
            if (!IsComplete(directory, pin)) Fill(pin, directory, localSources);
        }
        finally
        {
            gate.Release();
        }
        return directory;
    }

    private static void Fill(CudaLibraryPin pin, string directory, IEnumerable<string> localSources)
    {
        if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        Directory.CreateDirectory(directory);
        try
        {
            if (localSources.FirstOrDefault(source => MatchesExactly(source, pin)) is { } source)
                foreach (var file in pin.Files)
                    LinkOrCopy(Path.Combine(source, file.FileName), Path.Combine(directory, file.FileName));
            else
                Download(pin, directory);
            foreach (var file in pin.Files)
                if (!Matches(Path.Combine(directory, file.FileName), file))
                    throw new InvalidDataException(
                        $"{file.FileName} of {pin}, filled into '{directory}', is not the file its wheel records.");
            File.WriteAllText(Path.Combine(directory, CompleteMarker), pin.WheelSha256);
        }
        catch
        {
            try { Directory.Delete(directory, recursive: true); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            throw;
        }
    }

    private static void Download(CudaLibraryPin pin, string directory)
    {
        var wheel = directory + ".wheel";
        try
        {
            using (var target = new FileStream(wheel, FileMode.Create, FileAccess.ReadWrite, FileShare.None))
            {
                try
                {
                    if (pin.Wheel.IsFile)
                    {
                        using var source = File.OpenRead(pin.Wheel.LocalPath);
                        source.CopyTo(target);
                    }
                    else
                    {
                        // The synchronous send: this runs inside a backend's own synchronous call, and
                        // blocking on an asynchronous one there could wait on the caller's context.
                        using var request = new HttpRequestMessage(HttpMethod.Get, pin.Wheel);
                        using var response = Http.Value.Send(request, HttpCompletionOption.ResponseHeadersRead);
                        response.EnsureSuccessStatusCode();
                        using var source = response.Content.ReadAsStream();
                        source.CopyTo(target);
                    }
                }
                catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException
                                               or IOException or UnauthorizedAccessException)
                {
                    throw Unavailable(pin, directory, ex);
                }
                target.Position = 0;
                if (target.Length != pin.WheelSize || Convert.ToHexStringLower(SHA256.HashData(target)) != pin.WheelSha256)
                    throw new InvalidDataException($"The wheel fetched from {pin.Wheel} is not the one {pin} pins: its size or SHA-256 differs.");
            }
            using var archive = ZipFile.OpenRead(wheel);
            foreach (var file in pin.Files)
                (archive.GetEntry(file.WheelPath)
                    ?? throw new InvalidDataException($"The wheel {pin.Wheel} has no {file.WheelPath}."))
                    .ExtractToFile(Path.Combine(directory, file.FileName));
        }
        finally
        {
            File.Delete(wheel);
        }
    }

    private static InvalidOperationException Unavailable(CudaLibraryPin pin, string directory, Exception cause)
        => new($"The CUDA backends need {pin} ({string.Join(", ", pin.Files.Select(f => f.FileName))}), and this "
            + $"machine has no copy of it: the cache folder '{directory}' is not filled, and no installed copy "
            + $"matches it exactly (looked for {WhereLookedFor(pin, OperatingSystem.IsWindows())}). Downloading it, "
            + $"{pin.WheelSize / (1024 * 1024)} MiB from {pin.Wheel}, failed: {cause.Message} Connect to the network "
            + $"and run again, or install exactly that release ({pin.Package}=={pin.Version}).",
            cause);

    private static string WhereLookedFor(CudaLibraryPin pin, bool windows)
        => (windows, pin.Name) switch
        {
            (true, "cudnn") => @"on PATH, in %CUDNN_PATH% and under %ProgramFiles%\NVIDIA\CUDNN",
            (true, _) => $@"on PATH, in %CUDA_PATH% and in the CUDA {pin.CudaMajor} toolkits under %ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA",
            (false, "cudnn") => "on LD_LIBRARY_PATH, in $CUDNN_PATH and in the system's library folders",
            _ => $"on LD_LIBRARY_PATH, in $CUDA_PATH, in $CUDA_HOME, in the CUDA {pin.CudaMajor} toolkits under /usr/local and in the system's library folders",
        };

    private static FileStream AcquireFileLock(string path, TimeSpan timeout, string directory, TimeSpan total)
    {
        var clock = Stopwatch.StartNew();
        while (true)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException) when (clock.Elapsed < timeout)
            {
                Thread.Sleep(250);
            }
            catch (IOException ex)
            {
                throw TimedOut(directory, total, ex);
            }
        }
    }

    private static TimeoutException TimedOut(string directory, TimeSpan timeout, Exception? inner = null)
        => new($"Another process has been filling '{directory}' for longer than {timeout}. Its lock is released "
            + "when it finishes or ends, so wait for it and try again.", inner);

    /// <summary>Where an environment's copy of <paramref name="file"/> can be: where the library's own
    /// wheel installs it, as it does on Linux, and among the libraries PyTorch's wheel bundles in
    /// <c>torch/lib</c>, as it does on Windows.</summary>
    internal static IReadOnlyList<string> EnvironmentFiles(string sitePackages, CudaLibraryFile file)
        => [Path.Combine([sitePackages, .. file.WheelPath.Split('/')]), Path.Combine(sitePackages, "torch", "lib", file.FileName)];

    /// <summary>
    /// Makes the environment's copies of the pinned files hard links to the cache's, filling the cache
    /// from the environment's own copies where they match — no download — and from the
    /// <paramref name="installed"/> folders or the wheel otherwise. A copy that is another release is
    /// left as it is. Done once per pin set: the environment then holds <see cref="LinkedMarker"/>.
    /// A file another process has loaded cannot be replaced; it stays a copy until a later call, and
    /// is the very same release meanwhile. True when every copy is a link.
    /// </summary>
    internal static bool LinkEnvironment(
        string environment, string sitePackages, CudaLibraryPins pins, string root,
        Func<CudaLibraryPin, IEnumerable<string>> installed, TimeSpan timeout)
    {
        var marker = Path.Combine(environment, LinkedMarker);
        if (File.Exists(marker) && File.ReadAllText(marker) == pins.Identity) return true;
        var linked = true;
        foreach (var pin in pins.Libraries)
        {
            var copies = pin.Files.SelectMany(file => EnvironmentFiles(sitePackages, file)).Where(File.Exists).ToList();
            if (copies.Count == 0) continue;
            var cache = Path.Combine(root, pin.CacheKey);
            if (!IsComplete(cache, pin))
                cache = Provision(pin, root, [.. copies.Select(path => Path.GetDirectoryName(path)!).Distinct(), .. installed(pin)], timeout);
            foreach (var file in pin.Files)
                foreach (var copy in EnvironmentFiles(sitePackages, file).Where(File.Exists).Where(path => Matches(path, file)))
                    linked &= ReplaceWithLink(copy, Path.Combine(cache, file.FileName));
        }
        if (linked) File.WriteAllText(marker, pins.Identity);
        return linked;
    }

    /// <summary>Replaces <paramref name="path"/> with a hard link to <paramref name="target"/>. False
    /// where the file is in use by another process; where no link can be made at all — another
    /// volume — the copy is left, since it is the same release, and that is not a failure.</summary>
    private static bool ReplaceWithLink(string path, string target)
    {
        var link = path + ".shorokoo-link";
        File.Delete(link);
        if (!TryCreateHardLink(link, target)) return true;
        try
        {
            File.Move(link, path, overwrite: true);
            // POSIX rename does nothing when both names are already one file, as a copy the cache
            // was filled from is; the second name it leaves is removed here.
            File.Delete(link);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            File.Delete(link);
            return false;
        }
    }

    private static void LinkOrCopy(string source, string target)
    {
        var real = new FileInfo(source).ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? source;
        if (!TryCreateHardLink(target, real)) File.Copy(real, target);
    }

    /// <summary>Makes <paramref name="link"/> a second name of the file at <paramref name="existing"/>.</summary>
    internal static bool TryCreateHardLink(string link, string existing)
        => OperatingSystem.IsWindows()
            ? CreateHardLinkW(link, existing, IntPtr.Zero)
            : LinkUnix(existing, link) == 0;

    [LibraryImport("kernel32.dll", StringMarshalling = StringMarshalling.Utf16, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CreateHardLinkW(string newFileName, string existingFileName, IntPtr securityAttributes);

    [LibraryImport("libc", EntryPoint = "link", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int LinkUnix(string oldPath, string newPath);
}
