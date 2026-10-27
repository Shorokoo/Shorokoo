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
/// CUDA Toolkit 13.x + cuDNN 9.x installed.
///
/// <para>The backend alone is not enough: a GPU backend deploys and loads on a machine with no
/// card, and every test gated on it then failed inside ONNX Runtime's session creation instead of
/// skipping. The device is asked the way <see cref="SideBySideCudaFactAttribute"/> asks it.</para>
///
/// <para>A test that reads what this process holds on the card passes <c>needsProcessFigure</c>
/// and skips under WSL, where neither DXGI nor NVML gives a per-process figure
/// (<see href="https://github.com/Shorokoo/Shorokoo/issues/552">Shorokoo/Shorokoo#552</see>).
/// Everywhere else the figure exists, so such a test runs, and a figure that cannot be read
/// fails it.</para>
/// </summary>
public sealed class CudaFactAttribute : FactAttribute
{
    public CudaFactAttribute(string? extraNote = null, bool needsProcessFigure = false)
    {
        var skip = NoCard.Value;
        if (skip is null && needsProcessFigure && UnderWsl)
            skip = "Under WSL the driver attributes no device memory to a process, and this test reads " +
                   "that figure (Shorokoo/Shorokoo#552).";

        if (skip is null) return;
        if (!string.IsNullOrEmpty(extraNote)) skip += " " + extraNote;
        Skip = skip;
    }

    /// <summary>Whether the tests run on a card: a GPU backend is loaded and a CUDA device
    /// answers.</summary>
    public static bool RunsOnACard => NoCard.Value is null;

    // Why the tests do not run on a card, or null where they do.
    private static readonly Lazy<string?> NoCard = new(() =>
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
            return $"CUDA EP not available -- loaded backend is '{assemblyName}'. " +
                   "Reference a GPU backend (Shorokoo.WinGPU on Windows, Shorokoo.LinuxGPU on " +
                   "Linux) with CUDA Toolkit 13.x + cuDNN 9.x to run.";
        if (DeviceMemory.Read() is null)
            return $"No CUDA device answers on this machine -- the loaded backend is '{assemblyName}', " +
                   "but there is no card for it to run on. Run this on a CUDA machine.";
        return null;
    });

    /// <summary>Whether this process runs under WSL: Linux, with WSL's GPU device, its interop
    /// handler, or a kernel release WSL's kernel names itself in.</summary>
    public static bool UnderWsl { get; } = OperatingSystem.IsLinux()
        && (Path.Exists("/dev/dxg") || Path.Exists("/proc/sys/fs/binfmt_misc/WSLInterop") || KernelReleaseNamesMicrosoft());

    private static bool KernelReleaseNamesMicrosoft()
    {
        try
        {
            return File.ReadAllText("/proc/sys/kernel/osrelease").Contains("microsoft", StringComparison.OrdinalIgnoreCase);
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }
}
