using Shorokoo.PythonHost;

namespace Shorokoo.Jax.Cuda;

/// <summary>
/// The Shorokoo backend that runs on an NVIDIA GPU through JAX, built for CUDA 13, on Linux (JAX's
/// CUDA plugin is built for no other system). Each model is translated to Python calling
/// <c>jax.numpy</c> and compiled by XLA for the card once per signature of input shapes; its
/// tensors live on <c>cuda:N</c>, and a run hands outputs back in host memory unless it is asked to
/// keep them on the card.
///
/// <para>Name it explicitly; it is never discovered:</para>
/// <code>
/// using var context = new ComputeContext(new JaxCudaBackend());
/// </code>
/// <para>With no options, the Python environment is the one <c>SHOROKOO_PYTHON_ENV</c> names, or
/// else one provisioned with <c>uv</c> from this package's lock file into the user cache on first
/// use (<see cref="PythonEnvironmentResolver"/>) — the CUDA 13 environment the PyTorch CUDA backend
/// uses too. A process runs one Python environment, so where a CPU and a CUDA backend of either
/// family run together, the CUDA one should be started first (<see cref="JaxBackend.Start"/>).</para>
///
/// <para>JAX takes the card's memory as it needs it rather than most of the card up front
/// (<c>XLA_PYTHON_CLIENT_PREALLOCATE=false</c>, unless the program's environment sets it), since the
/// card is shared with the other backends in the process.</para>
/// </summary>
public sealed class JaxCudaBackend : JaxBackend
{
    /// <summary>Creates the backend on CUDA device 0.</summary>
    public JaxCudaBackend() : this(0) { }

    /// <summary>Creates the backend on CUDA device <paramref name="deviceId"/>, over the
    /// environment <paramref name="options"/> name.</summary>
    public JaxCudaBackend(int deviceId, PythonEnvironmentOptions? options = null)
        : base(() => PythonEnvironmentLock.Cu13, options, ValidDevice(deviceId)) { }

    private static int ValidDevice(int deviceId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        return deviceId;
    }
}
