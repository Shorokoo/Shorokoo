using Shorokoo.Core.Backends;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// Marks a test that requires a CUDA-capable Shorokoo backend and a card for it to run on.
/// Consults <see cref="DefaultBackend.Instance"/>: if the loaded
/// platform DLL's assembly name ends in "GPU" (e.g. Shorokoo.WinGPU) and a CUDA device answers,
/// the test runs; otherwise it is skipped with a clear message.
///
/// EP selection is now determined by which Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU}
/// project is referenced in the test deployment, not by a runtime preference.
/// To enable CUDA-gated tests on a dev box, add a project reference to a GPU
/// backend (Shorokoo.WinGPU on Windows, Shorokoo.LinuxGPU on Linux) and have
/// CUDA Toolkit 12.x + cuDNN 9.x installed.
///
/// <para>The backend alone is not enough: a GPU backend deploys and loads on a machine with no
/// card, and every test gated on it then failed inside ONNX Runtime's session creation instead of
/// skipping. The device is asked the way <see cref="SideBySideCudaFactAttribute"/> asks it.</para>
/// </summary>
public sealed class CudaFactAttribute : FactAttribute
{
    public CudaFactAttribute(string? extraNote = null)
    {
        string assemblyName;
        try
        {
            assemblyName = DefaultBackend.Instance.GetType().Assembly.GetName().Name ?? "?";
        }
        catch (Exception ex)
        {
            assemblyName = $"<resolver failed: {ex.GetType().Name}: {ex.Message}>";
        }

        string? skip = null;
        if (!assemblyName.EndsWith("GPU", StringComparison.OrdinalIgnoreCase))
            skip = $"CUDA EP not available -- loaded backend is '{assemblyName}'. " +
                   "Reference a GPU backend (Shorokoo.WinGPU on Windows, Shorokoo.LinuxGPU on " +
                   "Linux) with CUDA Toolkit 12.x + cuDNN 9.x to run.";
        else if (DeviceMemory.Read() is null)
            skip = $"No CUDA device answers on this machine -- the loaded backend is '{assemblyName}', " +
                   "but there is no card for it to run on. Run this on a CUDA machine.";

        if (skip is null) return;
        if (!string.IsNullOrEmpty(extraNote)) skip += " " + extraNote;
        Skip = skip;
    }
}
