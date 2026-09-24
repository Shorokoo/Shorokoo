using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Shorokoo.PythonHost;

/// <summary>
/// Finds the Python environment a backend runs in, provisioning it when nothing names one.
///
/// <para>The order is fixed: an explicit <see cref="PythonEnvironmentOptions.EnvironmentPath"/>;
/// then the <c>SHOROKOO_PYTHON_ENV</c> environment variable; then a cached environment built from
/// the backend's <see cref="PythonEnvironmentLock"/> under
/// <c>&lt;cache&gt;/shorokoo/python-envs/&lt;lock-name&gt;-&lt;lock-hash&gt;/</c>, created with
/// <c>uv</c> the first time it is needed. A named environment is used as it is and never
/// modified.</para>
///
/// <para>Provisioning is serialised across threads and processes by a lock file beside the
/// environment, so two test hosts or two programs starting at once build it once. An environment
/// is marked complete only after every package is installed, so one a killed process left half
/// built is rebuilt rather than used.</para>
/// </summary>
public static class PythonEnvironmentResolver
{
    /// <summary>The environment variable naming an environment to use as it is.</summary>
    public const string EnvironmentVariable = "SHOROKOO_PYTHON_ENV";

    /// <summary>The environment variable naming the <c>uv</c> executable to provision with.</summary>
    public const string UvVariable = "SHOROKOO_UV";

    private const string CompleteMarker = ".shorokoo-provisioned";

    /// <summary>The environment for <paramref name="lockFile"/>, resolved in the order above.</summary>
    /// <exception cref="PythonEnvironmentException">No usable environment could be found or
    /// provisioned; the failure says why.</exception>
    public static PythonEnvironment Resolve(PythonEnvironmentLock lockFile, PythonEnvironmentOptions? options = null)
        => Resolve(lockFile, options ?? new PythonEnvironmentOptions(), Environment.GetEnvironmentVariable);

    /// <summary>The same, reading environment variables through <paramref name="variables"/>, so a
    /// test can stand for a process with or without them.</summary>
    internal static PythonEnvironment Resolve(
        PythonEnvironmentLock lockFile, PythonEnvironmentOptions options, Func<string, string?> variables)
    {
        ArgumentNullException.ThrowIfNull(lockFile);
        ArgumentNullException.ThrowIfNull(options);
        if (!string.IsNullOrWhiteSpace(options.EnvironmentPath))
            return PythonEnvironment.Open(options.EnvironmentPath, lockFile.PythonVersion, PythonEnvironmentSource.Explicit);
        if (variables(EnvironmentVariable) is { Length: > 0 } named)
            return PythonEnvironment.Open(named, lockFile.PythonVersion, PythonEnvironmentSource.EnvironmentVariable);
        return Provision(lockFile, options, variables);
    }

    /// <summary>Where <paramref name="lockFile"/>'s environment is cached, provisioned or not.</summary>
    public static string CachedEnvironmentPath(PythonEnvironmentLock lockFile, PythonEnvironmentOptions? options = null)
        => Path.Combine(CacheRoot(options ?? new PythonEnvironmentOptions(), Environment.GetEnvironmentVariable), lockFile.CacheKey);

    /// <summary>The folder cached environments live in.</summary>
    internal static string CacheRoot(PythonEnvironmentOptions options, Func<string, string?> variables)
    {
        if (!string.IsNullOrWhiteSpace(options.CacheDirectory)) return Path.GetFullPath(options.CacheDirectory);
        string root;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            root = variables("LOCALAPPDATA") is { Length: > 0 } local
                ? local
                : Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        else
            root = variables("XDG_CACHE_HOME") is { Length: > 0 } xdg
                ? xdg
                : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        return Path.Combine(root, "shorokoo", "python-envs");
    }

    // One gate per cache folder within this process, in front of the file lock: the file lock is
    // what serialises processes, and this keeps threads of one process from spinning on it.
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Gates = new(StringComparer.Ordinal);

    private static PythonEnvironment Provision(
        PythonEnvironmentLock lockFile, PythonEnvironmentOptions options, Func<string, string?> variables)
    {
        var root = CacheRoot(options, variables);
        var directory = Path.Combine(root, lockFile.CacheKey);
        if (IsComplete(directory, lockFile))
            return PythonEnvironment.Open(directory, lockFile.PythonVersion, PythonEnvironmentSource.Provisioned);

        var uv = FindUv(options, variables);
        try
        {
            System.IO.Directory.CreateDirectory(root);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new PythonEnvironmentException(PythonEnvironmentFailure.ProvisioningFailed,
                $"The environment cache '{root}' cannot be created: {ex.Message} Name another with "
                + $"PythonEnvironmentOptions.CacheDirectory, or name an environment with {EnvironmentVariable}.", ex);
        }

        var gate = Gates.GetOrAdd(directory, static _ => new SemaphoreSlim(1, 1));
        var deadline = Stopwatch.StartNew();
        if (!gate.Wait(options.ProvisioningTimeout))
            throw TimedOut(directory, options.ProvisioningTimeout);
        try
        {
            using var fileLock = AcquireFileLock(directory + ".lock", options.ProvisioningTimeout - deadline.Elapsed, directory);
            if (!IsComplete(directory, lockFile))
                Build(uv, lockFile, directory);
        }
        finally
        {
            gate.Release();
        }
        return PythonEnvironment.Open(directory, lockFile.PythonVersion, PythonEnvironmentSource.Provisioned);
    }

    private static bool IsComplete(string directory, PythonEnvironmentLock lockFile)
    {
        var marker = Path.Combine(directory, CompleteMarker);
        try
        {
            return File.Exists(marker) && File.ReadAllText(marker).Trim() == lockFile.Hash;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static FileStream AcquireFileLock(string path, TimeSpan timeout, string directory)
    {
        var clock = Stopwatch.StartNew();
        while (true)
        {
            try
            {
                // FileShare.None is an exclusive lock on every platform .NET runs on -- on Unix it is
                // taken with flock, which a second open in this process honours as well.
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException) when (clock.Elapsed < timeout)
            {
                Thread.Sleep(250);
            }
            catch (IOException ex)
            {
                throw TimedOut(directory, timeout, ex);
            }
        }
    }

    private static PythonEnvironmentException TimedOut(string directory, TimeSpan timeout, Exception? inner = null)
        => new(PythonEnvironmentFailure.ProvisioningTimedOut,
            $"Another process has been provisioning '{directory}' for longer than {timeout}. If none "
            + $"is, delete '{directory}.lock' and try again.", inner);

    private static void Build(string uv, PythonEnvironmentLock lockFile, string directory)
    {
        if (System.IO.Directory.Exists(directory))
            System.IO.Directory.Delete(directory, recursive: true);

        var requirements = directory + ".requirements.txt";
        File.WriteAllText(requirements, lockFile.Requirements);
        try
        {
            Run(uv, ["python", "install", lockFile.PythonVersion], null);
            Run(uv, ["venv", "--no-config", "--managed-python", "--python", lockFile.PythonVersion, directory], null);
            Run(uv, ["pip", "install", "--no-config", "-r", requirements, .. lockFile.IndexArguments], directory);
            File.WriteAllText(Path.Combine(directory, CompleteMarker), lockFile.Hash);
        }
        catch
        {
            try { System.IO.Directory.Delete(directory, recursive: true); }
            catch (Exception) { }
            throw;
        }
        finally
        {
            File.Delete(requirements);
        }
    }

    private static void Run(string uv, string[] arguments, string? virtualEnvironment)
    {
        var start = new ProcessStartInfo(uv)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        if (virtualEnvironment is not null) start.Environment["VIRTUAL_ENV"] = virtualEnvironment;
        start.Environment["UV_NO_PROGRESS"] = "1";

        var output = new StringBuilder();
        using var process = new Process { StartInfo = start };
        process.OutputDataReceived += (_, e) => { if (e.Data is not null) lock (output) output.AppendLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) lock (output) output.AppendLine(e.Data); };
        try
        {
            process.Start();
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException)
        {
            throw new PythonEnvironmentException(PythonEnvironmentFailure.UvNotFound,
                $"uv could not be started from '{uv}': {ex.Message}", ex);
        }
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        process.WaitForExit();

        if (process.ExitCode == 0) return;
        string text;
        lock (output) text = output.ToString().Trim();
        var command = $"uv {string.Join(' ', arguments)}";
        throw new PythonEnvironmentException(
            LooksLikeNetwork(text) ? PythonEnvironmentFailure.NetworkUnavailable : PythonEnvironmentFailure.ProvisioningFailed,
            LooksLikeNetwork(text)
                ? $"Provisioning the Python environment needs the network, and `{command}` could not reach "
                  + $"the package index. Connect, or provision an environment elsewhere and name it with "
                  + $"{EnvironmentVariable}. uv said:\n{text}"
                : $"`{command}` failed (exit code {process.ExitCode}). uv said:\n{text}");
    }

    private static readonly string[] NetworkSymptoms =
    [
        "error sending request", "dns error", "failed to lookup address", "connection refused",
        "network is unreachable", "could not connect", "failed to fetch", "operation timed out",
        "connection reset",
    ];

    internal static bool LooksLikeNetwork(string uvOutput)
        => NetworkSymptoms.Any(symptom => uvOutput.Contains(symptom, StringComparison.OrdinalIgnoreCase));

    /// <summary>The uv executable to provision with.</summary>
    internal static string FindUv(PythonEnvironmentOptions options, Func<string, string?> variables)
    {
        var named = !string.IsNullOrWhiteSpace(options.UvPath) ? options.UvPath : variables(UvVariable);
        if (!string.IsNullOrWhiteSpace(named))
            return File.Exists(named)
                ? named
                : throw new PythonEnvironmentException(PythonEnvironmentFailure.UvNotFound,
                    $"The uv named for provisioning, '{named}', does not exist.");

        var executable = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "uv.exe" : "uv";
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        IEnumerable<string> folders = (variables("PATH") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Concat([Path.Combine(home, ".local", "bin"), Path.Combine(home, ".cargo", "bin")]);
        return folders.Select(folder => Path.Combine(folder, executable)).FirstOrDefault(File.Exists)
            ?? throw new PythonEnvironmentException(PythonEnvironmentFailure.UvNotFound,
                "No Python environment is named and provisioning one needs uv, which is not on PATH. "
                + "Install it (https://docs.astral.sh/uv/getting-started/installation/), name it with "
                + $"{UvVariable}, or name an existing environment with {EnvironmentVariable}.");
    }
}
