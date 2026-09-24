using System;
using System.Runtime.CompilerServices;

namespace Shorokoo;

/// <summary>
/// Refuses to run on a big-endian machine. Tensor payloads, ONNX raw data and the SafeTensors
/// and checkpoint fields Shorokoo persists are all little-endian and are moved as raw native
/// memory, so on a big-endian host every one of them would be silently misread. Every platform
/// .NET officially supports is little-endian; this turns the one exception into a clear error.
/// </summary>
internal static class LittleEndianGuard
{
#pragma warning disable CA2255 // Deliberate: the check must run before any Shorokoo code touches tensor bytes.
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Check()
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException(
                "Shorokoo requires a little-endian platform: tensor data, ONNX raw data and the " +
                "SafeTensors and checkpoint formats it reads and writes are all little-endian.");
    }
}
