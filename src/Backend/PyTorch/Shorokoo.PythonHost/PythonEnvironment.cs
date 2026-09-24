using System.Globalization;
using System.Runtime.InteropServices;

namespace Shorokoo.PythonHost;

/// <summary>Where a resolved environment came from.</summary>
public enum PythonEnvironmentSource
{
    /// <summary>Named by <see cref="PythonEnvironmentOptions.EnvironmentPath"/>.</summary>
    Explicit,

    /// <summary>Named by the <c>SHOROKOO_PYTHON_ENV</c> environment variable.</summary>
    EnvironmentVariable,

    /// <summary>Provisioned from a lock file into the cache, now or by an earlier run.</summary>
    Provisioned,

    /// <summary>The environment this process's interpreter was already started over.</summary>
    Running,
}

/// <summary>
/// A virtual environment an interpreter can be embedded over: the environment itself, the base
/// CPython it was made from, and the shared library of that CPython.
///
/// <para>The base comes from the environment's <c>pyvenv.cfg</c>, whose <c>home</c> is the folder
/// holding the base interpreter. The library is found beside it — <c>lib/libpython3.12.so</c> on
/// Linux, <c>python312.dll</c> on Windows — because a virtual environment does not copy it.</para>
/// </summary>
public sealed class PythonEnvironment
{
    private PythonEnvironment(
        string directory, string pythonHome, string libPython, string sitePackages,
        Version version, PythonEnvironmentSource source)
    {
        Directory = directory;
        PythonHome = pythonHome;
        LibPython = libPython;
        SitePackages = sitePackages;
        PythonVersion = version;
        Source = source;
    }

    /// <summary>The virtual environment.</summary>
    public string Directory { get; }

    /// <summary>The base CPython installation the environment was made from.</summary>
    public string PythonHome { get; }

    /// <summary>The shared Python library the interpreter is loaded from.</summary>
    public string LibPython { get; }

    /// <summary>The environment's <c>site-packages</c>.</summary>
    public string SitePackages { get; }

    /// <summary>The environment's Python version.</summary>
    public Version PythonVersion { get; }

    /// <summary>Where this environment came from.</summary>
    public PythonEnvironmentSource Source { get; }

    /// <summary>The same environment, recorded as having come from <paramref name="source"/>.</summary>
    internal PythonEnvironment As(PythonEnvironmentSource source)
        => new(Directory, PythonHome, LibPython, SitePackages, PythonVersion, source);

    /// <inheritdoc/>
    public override string ToString() => $"{Directory} (CPython {PythonVersion}, {Source})";

    /// <summary>
    /// Reads the virtual environment at <paramref name="directory"/>, checking that it is one, that
    /// its Python is <paramref name="requiredVersion"/>, and that its base interpreter has a shared
    /// library to embed.
    /// </summary>
    /// <exception cref="PythonEnvironmentException">Any of those does not hold; the failure says
    /// which.</exception>
    public static PythonEnvironment Open(
        string directory, string requiredVersion, PythonEnvironmentSource source)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentException.ThrowIfNullOrWhiteSpace(requiredVersion);
        // Without a trailing separator, so that one folder named two ways is one environment.
        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory));
        if (!System.IO.Directory.Exists(full))
            throw new PythonEnvironmentException(PythonEnvironmentFailure.EnvironmentNotFound,
                $"There is no Python environment at '{full}' ({Describe(source)}). Create one with "
                + $"`uv venv -p {requiredVersion} {full}` and install the backend's packages into it, "
                + "or leave it unset to have one provisioned.");

        var config = Path.Combine(full, "pyvenv.cfg");
        if (!File.Exists(config))
            throw new PythonEnvironmentException(PythonEnvironmentFailure.NotAVirtualEnvironment,
                $"'{full}' ({Describe(source)}) is not a Python virtual environment: it has no "
                + "pyvenv.cfg. Name the environment's own folder, the one `uv venv` or "
                + "`python -m venv` created.");

        var settings = ReadConfig(config);
        var versionText = settings.GetValueOrDefault("version_info") ?? settings.GetValueOrDefault("version");
        if (versionText is null || !TryParseVersion(versionText, out var version))
            throw new PythonEnvironmentException(PythonEnvironmentFailure.WrongPythonVersion,
                $"'{config}' does not say which Python the environment runs, so it cannot be checked "
                + $"against the {requiredVersion} this backend needs.");
        var required = Version.Parse(requiredVersion);
        if (version.Major != required.Major || version.Minor != required.Minor)
            throw new PythonEnvironmentException(PythonEnvironmentFailure.WrongPythonVersion,
                $"The environment at '{full}' runs Python {version}, and this backend needs "
                + $"{required.Major}.{required.Minor}: its packages are built for that version. "
                + $"Recreate it with `uv venv -p {requiredVersion}`.");

        if (settings.GetValueOrDefault("home") is not { Length: > 0 } binHome)
            throw new PythonEnvironmentException(PythonEnvironmentFailure.LibPythonNotFound,
                $"'{config}' names no home for its base interpreter, so there is no Python library "
                + "to embed.");

        var windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        var pythonHome = PythonHomeOf(binHome, windows);
        var libPython = LibPythonCandidates(pythonHome, version, windows).FirstOrDefault(File.Exists)
            ?? throw new PythonEnvironmentException(PythonEnvironmentFailure.LibPythonNotFound,
                $"The base interpreter of '{full}', at '{pythonHome}', has no shared Python library "
                + $"({string.Join(", ", LibPythonCandidates(pythonHome, version, windows).Select(Path.GetFileName).Distinct())}) "
                + "to embed. A distribution's system Python often ships without one; a Python "
                + $"installed by `uv python install {requiredVersion}` has it, and `uv venv "
                + "--managed-python` makes the environment over that one.");

        return new PythonEnvironment(full, pythonHome, libPython, SitePackagesOf(full, version, windows), version, source);
    }

    private static string Describe(PythonEnvironmentSource source) => source switch
    {
        PythonEnvironmentSource.Explicit => "named by the backend's options",
        PythonEnvironmentSource.EnvironmentVariable => $"named by {PythonEnvironmentResolver.EnvironmentVariable}",
        PythonEnvironmentSource.Provisioned => "provisioned into the cache",
        _ => "the running environment",
    };

    /// <summary>
    /// The base installation a virtual environment's <c>pyvenv.cfg</c> <c>home</c> names: on Windows
    /// <c>home</c> is the installation's own folder, holding <c>python.exe</c> and the library; elsewhere
    /// it is the installation's <c>bin</c>, one level below it.
    /// </summary>
    internal static string PythonHomeOf(string binHome, bool windows)
        => windows
            ? Path.TrimEndingDirectorySeparator(binHome)
            : Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(binHome)) ?? binHome;

    /// <summary>Where the shared Python library of the installation at <paramref name="home"/> can
    /// be, in the order they are tried: <c>python312.dll</c> beside <c>python.exe</c> on Windows,
    /// <c>lib/libpython3.12.so</c> (or <c>.dylib</c>) elsewhere.</summary>
    internal static IEnumerable<string> LibPythonCandidates(string home, Version version, bool windows)
    {
        var mm = $"{version.Major}.{version.Minor}";
        if (windows)
        {
            yield return Path.Combine(home, $"python{version.Major}{version.Minor}.dll");
            yield break;
        }
        var extension = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "dylib" : "so";
        foreach (var lib in (string[])["lib", "lib64"])
        {
            yield return Path.Combine(home, lib, $"libpython{mm}.{extension}");
            yield return Path.Combine(home, lib, $"libpython{mm}.{extension}.1.0");
        }
    }

    /// <summary>A virtual environment's <c>site-packages</c>: <c>Lib\site-packages</c> on Windows,
    /// <c>lib/python3.12/site-packages</c> elsewhere.</summary>
    internal static string SitePackagesOf(string environment, Version version, bool windows)
        => windows
            ? Path.Combine(environment, "Lib", "site-packages")
            : Path.Combine(environment, "lib", $"python{version.Major}.{version.Minor}", "site-packages");

    private static Dictionary<string, string> ReadConfig(string path)
    {
        var settings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in File.ReadAllLines(path))
        {
            var equals = line.IndexOf('=');
            if (equals <= 0) continue;
            settings[line[..equals].Trim()] = line[(equals + 1)..].Trim();
        }
        return settings;
    }

    private static bool TryParseVersion(string text, out Version version)
    {
        // "3.12.11", or "3.12.0rc2" from a pre-release: only the numeric head matters here.
        var parts = text.Split('.');
        version = new Version(0, 0);
        if (parts.Length < 2) return false;
        if (!int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out var major)) return false;
        if (!int.TryParse(new string([.. parts[1].TakeWhile(char.IsAsciiDigit)]), NumberStyles.None,
                CultureInfo.InvariantCulture, out var minor)) return false;
        var patch = parts.Length > 2 && int.TryParse(new string([.. parts[2].TakeWhile(char.IsAsciiDigit)]),
            NumberStyles.None, CultureInfo.InvariantCulture, out var p) ? p : 0;
        version = new Version(major, minor, patch);
        return true;
    }
}
