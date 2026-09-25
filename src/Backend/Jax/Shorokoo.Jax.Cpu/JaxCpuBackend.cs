using Shorokoo.PythonHost;

namespace Shorokoo.Jax.Cpu;

/// <summary>
/// The Shorokoo backend that runs on the CPU through JAX. Each model is translated to Python calling
/// <c>jax.numpy</c>, and compiled by XLA for the CPU once per signature of input shapes.
///
/// <para>Name it explicitly; it is never discovered:</para>
/// <code>
/// using var context = new ComputeContext(new JaxCpuBackend());
/// </code>
/// <para>With no options, the Python environment is the one <c>SHOROKOO_PYTHON_ENV</c> names, or
/// else one provisioned with <c>uv</c> from this package's lock file into the user cache on first
/// use (<see cref="PythonEnvironmentResolver"/>) — the same environment the PyTorch CPU backend
/// uses, which holds both.</para>
/// </summary>
public sealed class JaxCpuBackend : JaxBackend
{
    /// <summary>Creates the backend, resolving its environment the default way.</summary>
    public JaxCpuBackend() : this(null) { }

    /// <summary>Creates the backend over the environment <paramref name="options"/> name.</summary>
    public JaxCpuBackend(PythonEnvironmentOptions? options)
        : base(() => PythonEnvironmentLock.Cpu, options, cudaDeviceId: null) { }
}
