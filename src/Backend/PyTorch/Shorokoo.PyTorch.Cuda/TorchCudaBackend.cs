using Shorokoo.PythonHost;

namespace Shorokoo.PyTorch.Cuda;

/// <summary>
/// The Shorokoo backend that runs on an NVIDIA GPU through PyTorch, built for CUDA 13. Each model
/// is translated to Python and run in the process's embedded CPython; its tensors live on
/// <c>cuda:N</c>, and a run hands outputs back in host memory unless it is asked to keep them on
/// the card.
///
/// <para>Name it explicitly; it is never discovered:</para>
/// <code>
/// using var context = new ComputeContext(new TorchCudaBackend());
/// </code>
/// <para>With no options, the Python environment is the one <c>SHOROKOO_PYTHON_ENV</c> names, or
/// else one provisioned with <c>uv</c> from this package's lock file into the user cache on first
/// use (<see cref="PythonEnvironmentResolver"/>). A process runs one Python environment, shared by
/// every torch backend in it, so where a CPU and a CUDA torch backend run together, the CUDA one
/// should be started first (<see cref="TorchBackend.Start"/>): the CUDA build serves the CPU as
/// well, and the CPU build cannot serve the card.</para>
/// </summary>
public sealed class TorchCudaBackend : TorchBackend
{
    /// <summary>Creates the backend on CUDA device 0.</summary>
    public TorchCudaBackend() : this(0) { }

    /// <summary>Creates the backend on CUDA device <paramref name="deviceId"/>, over the
    /// environment <paramref name="options"/> name.</summary>
    public TorchCudaBackend(int deviceId, PythonEnvironmentOptions? options = null)
        : base(() => PythonEnvironmentLock.Cu13, options, ValidDevice(deviceId)) { }

    private static int ValidDevice(int deviceId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        return deviceId;
    }
}
