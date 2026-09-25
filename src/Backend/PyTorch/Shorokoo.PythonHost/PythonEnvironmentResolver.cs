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
        => CacheRoot(options, variables, RuntimeInformation.IsOSPlatform(OSPlatform.Windows));

    /// <summary>The same, on Windows or not as <paramref name="windows"/> says, whatever this
    /// machine is.</summary>
    internal static string CacheRoot(PythonEnvironmentOptions options, Func<string, string?> variables, bool windows)
    {
        if (!string.IsNullOrWhiteSpace(options.CacheDirectory)) return Path.GetFullPath(options.CacheDirectory);
        string root;
        if (windows)
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
            // uv is looked for only once this process is the one to build: one that finds the
            // environment built by the process it waited for needs none.
            if (!IsComplete(directory, lockFile))
                Build(FindUv(options, variables), lockFile, directory, options.ProvisioningTimeout, deadline);
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
            $"Another process has been provisioning '{directory}' for longer than {timeout}. Its lock is "
            + "released when it finishes or ends, so wait for it and try again, or allow it longer with "
            + "PythonEnvironmentOptions.ProvisioningTimeout.", inner);

    private static void Build(
        string uv, PythonEnvironmentLock lockFile, string directory, TimeSpan timeout, Stopwatch deadline)
    {
        if (System.IO.Directory.Exists(directory))
            System.IO.Directory.Delete(directory, recursive: true);

        var requirements = directory + ".requirements.txt";
        File.WriteAllText(requirements, lockFile.Requirements);
        try
        {
            // Every step names what it acts on -- the Python version, the environment -- rather than
            // leaving uv to find it, and installs exactly the files the lock hashes.
            Run(uv, ["python", "install", "--no-config", lockFile.PythonVersion], timeout, deadline);
            Run(uv, ["venv", "--no-config", "--managed-python", "--python", lockFile.PythonVersion, directory], timeout, deadline);
            Run(uv, ["pip", "install", "--no-config", "--require-hashes", "--python", directory, "-r", requirements,
                .. lockFile.IndexArguments], timeout, deadline);
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

    /// <summary>
    /// The variables of the process a uv step runs in: this process's own, which carry the network's
    /// proxies and certificates, the user's home and the path, without those that would make uv
    /// install somewhere else or from somewhere else than the step says — every <c>UV_</c> variable
    /// but the few that only say where uv caches and how it reaches the network, and the variables
    /// that name another environment or Python — and with uv's progress output turned off.
    /// </summary>
    internal static Dictionary<string, string> UvEnvironment(IEnumerable<KeyValuePair<string, string>> inherited)
    {
        var environment = new Dictionary<string, string>(
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        foreach (var (name, value) in inherited)
        {
            var upper = name.ToUpperInvariant();
            if (upper.StartsWith("UV_", StringComparison.Ordinal) ? !KeptUvVariables.Contains(upper) : RedirectingVariables.Contains(upper))
                continue;
            environment[name] = value;
        }
        environment["UV_NO_PROGRESS"] = "1";
        return environment;
    }

    private static readonly HashSet<string> KeptUvVariables =
        new(StringComparer.Ordinal) { "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_NATIVE_TLS", "UV_HTTP_TIMEOUT" };

    private static readonly HashSet<string> RedirectingVariables =
        new(StringComparer.Ordinal) { "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH" };

    private static void Run(string uv, string[] arguments, TimeSpan timeout, Stopwatch deadline)
    {
        var start = new ProcessStartInfo(uv)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        var inherited = System.Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Select(entry => new KeyValuePair<string, string>((string)entry.Key, (string?)entry.Value ?? ""));
        start.Environment.Clear();
        foreach (var (name, value) in UvEnvironment(inherited)) start.Environment[name] = value;

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
        var command = $"uv {string.Join(' ', arguments)}";
        var left = timeout - deadline.Elapsed;
        if (left < TimeSpan.Zero || !process.WaitForExit(left))
        {
            try { process.Kill(entireProcessTree: true); }
            catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception) { }
            process.WaitForExit();
            throw new PythonEnvironmentException(PythonEnvironmentFailure.ProvisioningTimedOut,
                $"`{command}` had not finished when provisioning had taken {timeout}, so it was stopped. "
                + "Allow provisioning longer with PythonEnvironmentOptions.ProvisioningTimeout, or provision "
                + $"an environment elsewhere and name it with {EnvironmentVariable}.");
        }
        process.WaitForExit();

        if (process.ExitCode == 0) return;
        string text;
        lock (output) text = output.ToString().Trim();
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
