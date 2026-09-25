namespace Shorokoo.PythonHost;

/// <summary>Why a Python environment could not be found, provisioned or started.</summary>
public enum PythonEnvironmentFailure
{
    /// <summary>The environment named — by an option or by <c>SHOROKOO_PYTHON_ENV</c> — is not
    /// there.</summary>
    EnvironmentNotFound,

    /// <summary>The directory named is there but is not a virtual environment: it has no
    /// <c>pyvenv.cfg</c>.</summary>
    NotAVirtualEnvironment,

    /// <summary>The environment's Python is not the version the backend needs.</summary>
    WrongPythonVersion,

    /// <summary>The environment's base interpreter has no shared Python library to embed.</summary>
    LibPythonNotFound,

    /// <summary>No environment was named and provisioning one needs <c>uv</c>, which is not on
    /// <c>PATH</c> and not named by <c>SHOROKOO_UV</c> or an option.</summary>
    UvNotFound,

    /// <summary>Provisioning needed the network and could not reach the package index.</summary>
    NetworkUnavailable,

    /// <summary>Provisioning ran and failed for another reason, which the message quotes.</summary>
    ProvisioningFailed,

    /// <summary>Another process held the provisioning lock for longer than the timeout.</summary>
    ProvisioningTimedOut,

    /// <summary>The environment lacks a package the backend imports.</summary>
    MissingPackage,

    /// <summary>This process already runs an interpreter over a different environment, and a
    /// process has one.</summary>
    EnvironmentConflict,

    /// <summary>The interpreter itself failed to start.</summary>
    InterpreterFailed,

    /// <summary>The environment runs, but cannot serve the device asked for — a CUDA backend over
    /// a build of the framework without CUDA, say.</summary>
    DeviceUnavailable,

    /// <summary>No environment lock exists for this platform: the Python-based backends run on
    /// Linux and Windows on x64.</summary>
    UnsupportedPlatform,
}

/// <summary>
/// A Python environment could not be found, provisioned or started. <see cref="Failure"/> says
/// which, and the message names the thing that is missing and what to do about it.
/// </summary>
public sealed class PythonEnvironmentException : Exception
{
    /// <summary>Creates the exception.</summary>
    public PythonEnvironmentException(PythonEnvironmentFailure failure, string message, Exception? inner = null)
        : base(message, inner)
    {
        Failure = failure;
    }

    /// <summary>What went wrong.</summary>
    public PythonEnvironmentFailure Failure { get; }
}
