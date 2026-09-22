using Shorokoo.Core.Backends;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// Marks a test that requires a CUDA-capable Shorokoo backend.
/// Consults <see cref="DefaultBackend.Instance"/>: if the loaded
/// platform DLL's assembly name ends in "GPU" (e.g. Shorokoo.WinGPU), the
/// test runs; otherwise it is skipped with a clear message.
///
/// EP selection is now determined by which Shorokoo.{WinCPU,WinGPU,LinuxCPU,LinuxGPU}
/// project is referenced in the test deployment, not by a runtime preference.
/// To enable CUDA-gated tests on a dev box, add a project reference to a GPU
/// backend (Shorokoo.WinGPU on Windows, Shorokoo.LinuxGPU on Linux) and have
/// CUDA Toolkit 12.x + cuDNN 9.x installed.
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

        if (!assemblyName.EndsWith("GPU", StringComparison.OrdinalIgnoreCase))
        {
            var msg = $"CUDA EP not available -- loaded backend is '{assemblyName}'. " +
                      "Reference a GPU backend (Shorokoo.WinGPU on Windows, Shorokoo.LinuxGPU on " +
                      "Linux) with CUDA Toolkit 12.x + cuDNN 9.x to run.";
            if (!string.IsNullOrEmpty(extraNote)) msg += " " + extraNote;
            Skip = msg;
        }
    }
}
