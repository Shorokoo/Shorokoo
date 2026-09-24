using Shorokoo.PythonHost;

namespace Shorokoo.PyTorch.Cpu;

/// <summary>
/// The Shorokoo backend that runs on the CPU through PyTorch. Each model is translated to Python
/// and run in the process's embedded CPython, in the PyTorch CPU build.
///
/// <para>Name it explicitly; it is never discovered:</para>
/// <code>
/// using var context = new ComputeContext(new TorchCpuBackend());
/// </code>
/// <para>With no options, the Python environment is the one <c>SHOROKOO_PYTHON_ENV</c> names, or
/// else one provisioned with <c>uv</c> from this package's lock file into the user cache on first
/// use (<see cref="PythonEnvironmentResolver"/>).</para>
/// </summary>
public sealed class TorchCpuBackend : TorchBackend
{
    /// <summary>Creates the backend, resolving its environment the default way.</summary>
    public TorchCpuBackend() : this(null) { }

    /// <summary>Creates the backend over the environment <paramref name="options"/> name.</summary>
    public TorchCpuBackend(PythonEnvironmentOptions? options)
        : base(PythonEnvironmentLock.Cpu, options, cudaDeviceId: null) { }
}
