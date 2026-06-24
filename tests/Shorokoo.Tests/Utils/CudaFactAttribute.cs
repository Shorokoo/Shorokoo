using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// Marks a test that requires a CUDA-capable Shorokoo inference provider.
/// Consults <see cref="InferenceBackend.Factory"/>: if the loaded
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
            assemblyName = InferenceBackend.Factory.GetType().Assembly.GetName().Name ?? "?";
        }
        catch (Exception ex)
        {
            assemblyName = $"<resolver failed: {ex.GetType().Name}: {ex.Message}>";
        }

        if (!assemblyName.EndsWith("GPU", StringComparison.OrdinalIgnoreCase))
        {
            var msg = $"CUDA EP not available -- loaded inference provider is '{assemblyName}'. " +
                      "Reference a GPU backend (Shorokoo.WinGPU on Windows, Shorokoo.LinuxGPU on " +
                      "Linux) with CUDA Toolkit 12.x + cuDNN 9.x to run.";
            if (!string.IsNullOrEmpty(extraNote)) msg += " " + extraNote;
            Skip = msg;
        }
    }
}
