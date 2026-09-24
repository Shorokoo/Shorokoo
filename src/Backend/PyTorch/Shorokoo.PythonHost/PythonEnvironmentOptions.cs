namespace Shorokoo.PythonHost;

/// <summary>
/// Where a Python-based backend gets its environment. Everything is optional: with nothing set the
/// environment is <c>SHOROKOO_PYTHON_ENV</c> when that is set, and otherwise one provisioned from the
/// backend's lock file into the user cache on first use.
/// </summary>
public sealed record PythonEnvironmentOptions
{
    /// <summary>A virtual environment to use as it is — a directory holding <c>pyvenv.cfg</c>. When
    /// set, nothing is provisioned and <c>SHOROKOO_PYTHON_ENV</c> is not consulted.</summary>
    public string? EnvironmentPath { get; init; }

    /// <summary>The <c>uv</c> executable to provision with. When null, <c>SHOROKOO_UV</c>, then
    /// <c>PATH</c>, then uv's own default install locations are searched.</summary>
    public string? UvPath { get; init; }

    /// <summary>The folder provisioned environments are cached under. When null, the user cache:
    /// <c>$XDG_CACHE_HOME</c> or <c>~/.cache</c>, or <c>%LOCALAPPDATA%</c> on Windows, then
    /// <c>shorokoo/python-envs</c>.</summary>
    public string? CacheDirectory { get; init; }

    /// <summary>How long to wait for another process that is provisioning the same environment.
    /// Provisioning downloads the framework, which is more than a gigabyte, so the default is
    /// generous.</summary>
    public TimeSpan ProvisioningTimeout { get; init; } = TimeSpan.FromMinutes(30);
}
