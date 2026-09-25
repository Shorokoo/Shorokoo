using System.Globalization;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The NVIDIA driver, as far as a backend that brings its own CUDA libraries needs it: whether one
/// is installed, the newest CUDA it supports, and whether it sees a device. Read through the
/// driver API (<c>libcuda.so.1</c>, <c>nvcuda.dll</c>), which the driver itself installs — unlike
/// the CUDA runtime <see cref="CudaRuntime"/> binds, which comes with a toolkit — so it answers on
/// a machine that has a card and no toolkit, which is exactly the machine such a backend runs on.
///
/// <para>Bound lazily and best effort, as <see cref="CudaRuntime"/> is: every question answers
/// "no" rather than throwing on a machine with no driver at all.</para>
/// </summary>
internal static class CudaDriver
{
    private static string LibraryName => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? "nvcuda.dll"
        : "libcuda.so.1";

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int IntOut(out int value);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Init(uint flags);

    private static readonly object Gate = new();
    private static bool _read;
    private static CudaDriverReading _reading;

    /// <summary>What the driver says of itself, read once per process.</summary>
    internal static CudaDriverReading Read()
    {
        lock (Gate)
        {
            if (_read) return _reading;
            _reading = ReadNow();
            _read = true;
            return _reading;
        }
    }

    private static CudaDriverReading ReadNow()
    {
        try
        {
            if (!NativeLibrary.TryLoad(LibraryName, out var library)) return default;
            // The library stays loaded: a process that asked about the driver is about to use it.
            if (!NativeLibrary.TryGetExport(library, "cuDriverGetVersion", out var getVersion)
                || Marshal.GetDelegateForFunctionPointer<IntOut>(getVersion)(out var version) != 0)
                return new CudaDriverReading(true, 0, 0);
            if (!NativeLibrary.TryGetExport(library, "cuInit", out var init)
                || !NativeLibrary.TryGetExport(library, "cuDeviceGetCount", out var getCount)
                || Marshal.GetDelegateForFunctionPointer<Init>(init)(0) != 0
                || Marshal.GetDelegateForFunctionPointer<IntOut>(getCount)(out var count) != 0)
                return new CudaDriverReading(true, version, 0);
            return new CudaDriverReading(true, version, count);
        }
        // Best effort, as the class promises: a driver that cannot be asked is one that is not there.
        catch (Exception) { return default; }
    }

    /// <summary>
    /// The driver API's number for a CUDA version written "major.minor" — 13000 for "13.0", 12040
    /// for "12.4" — or null for anything else.
    /// </summary>
    internal static int? VersionNumber(string version)
    {
        var parts = version.Split('.');
        if (parts.Length is < 1 or > 2) return null;
        if (!int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out var major)) return null;
        var minor = 0;
        if (parts.Length == 2 && !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out minor))
            return null;
        return major * 1000 + minor * 10;
    }

    /// <summary>A driver API version number written the way CUDA spells versions: "13.0".</summary>
    internal static string VersionText(int number)
        => $"{number / 1000}.{number % 1000 / 10}";

    /// <summary>
    /// Why <paramref name="reading"/> cannot serve a backend needing CUDA
    /// <paramref name="required"/>, or null when it can.
    /// </summary>
    internal static string? Refusal(CudaDriverReading reading, string required)
    {
        // A requirement this cannot read is refused rather than skipped: skipping it would accept
        // the backend on any driver at all.
        if (VersionNumber(required) is not { } minimum)
            return $"'{required}' is not a CUDA version written major.minor, so no driver can be checked against it";
        if (!reading.Installed)
            return $"no NVIDIA driver is installed (there is no {LibraryName} to load), and the driver is "
                   + "the one part of CUDA it does not bring with it";
        if (reading.Version < minimum)
            return reading.Version == 0
                ? "the NVIDIA driver installed does not say which CUDA it supports"
                : $"the NVIDIA driver installed supports CUDA up to {VersionText(reading.Version)}, and it needs "
                  + $"{required} or later: update the driver";
        if (reading.DeviceCount == 0)
            return "the NVIDIA driver is installed but sees no CUDA device";
        return null;
    }
}

/// <summary>What the NVIDIA driver said of itself.</summary>
/// <param name="Installed">Whether the driver library loads at all.</param>
/// <param name="Version">The newest CUDA it supports, in the driver API's numbering (13000 for
/// 13.0), or 0 when it would not say.</param>
/// <param name="DeviceCount">The CUDA devices it sees.</param>
internal readonly record struct CudaDriverReading(bool Installed, int Version, int DeviceCount);
