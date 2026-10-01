using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Backends;

/// <summary>
/// The cuDNN and cuBLAS every CUDA backend of a process runs on: one release of each per CUDA major,
/// pinned beside the lock of the CUDA Python environment — the releases its PyTorch carries — and
/// one copy of it, which every backend binds to.
///
/// <para>One release, because two do not mix. A library named without a path binds to the copy the
/// process already holds under that name, and cuDNN's sub-libraries import one another's internal
/// entry points by name, as cuBLAS imports cuBLASLt's, which another release does not all export.
/// One copy, because a process holding the same release twice holds every page of it twice.</para>
///
/// <para>The copy is the release's folder in a per-user cache, <c>%LOCALAPPDATA%\shorokoo\cuda\</c>
/// on Windows and <c>~/.cache/shorokoo/cuda/</c> elsewhere. <see cref="Prepare"/> fills it the first time it is needed — from an installed
/// copy that matches the pin exactly, or else from the release's wheel on PyPI — and loads it, by
/// full path, before a backend loads anything by name. The ONNX Runtime CUDA backend calls it before
/// its provider loads; a provisioned Python environment's own copies are hard links to the cache's
/// files, so PyTorch, loading them by path, loads the very same files.</para>
/// </summary>
public static class CudaLibraries
{
    private static readonly object Gate = new();
    private static bool _prepared;

    /// <summary>
    /// Fills the shared cache with the pinned cuDNN and cuBLAS where it does not hold them yet, and
    /// loads them into this process; returns at once when they are loaded already. The ONNX Runtime
    /// CUDA backend calls it before its execution provider loads. Call it yourself at startup to fetch
    /// them then rather than on the first CUDA run, or before a backend of your own loads them.
    /// </summary>
    /// <exception cref="InvalidOperationException">A pinned library is in neither the cache nor an
    /// exactly matching installed copy, and its wheel could not be fetched; the message names the
    /// library, where it was looked for and the size of the download.</exception>
    public static void Prepare()
    {
        if (CudaLibraryPins.Current is not { } pins) return;
        lock (Gate)
        {
            if (_prepared) return;
            var root = CudaLibraryCache.DefaultRoot;
            // cuBLAS first: cuDNN loads cuBLASLt by name, and has to find this one.
            foreach (var pin in pins.Libraries.OrderBy(pin => pin.Name == "cudnn"))
            {
                var directory = Path.Combine(root, pin.CacheKey);
                if (!CudaLibraryCache.IsComplete(directory, pin))
                    directory = CudaLibraryCache.Provision(pin, root,
                        CudaLibraryCache.InstalledCandidates(pin, Environment.GetEnvironmentVariable, OperatingSystem.IsWindows()),
                        CudaLibraryCache.DefaultTimeout);
                // In the pinned order, which puts what a file imports by name ahead of it. Loaded for
                // the life of the process, like every library a backend binds.
                foreach (var file in pin.Files)
                    NativeLibrary.Load(Path.Combine(directory, file.FileName));
            }
            _prepared = true;
        }
    }

    /// <summary>
    /// The pinned libraries <paramref name="directory"/> ships that this process already holds another
    /// release of, each named with that copy's folder and version, or null when it holds none: what
    /// makes a framework that loads that folder's libraries by path fail to load them.
    /// </summary>
    internal static string? Conflict(string directory)
    {
        if (!OperatingSystem.IsWindows() || CudaLibraryPins.Current is not { } pins) return null;
        using var process = Process.GetCurrentProcess();
        var loaded = new List<string>();
        foreach (ProcessModule module in process.Modules)
            using (module)
                loaded.Add(module.FileName);
        return Conflict(directory, pins.Libraries.SelectMany(pin => pin.Files).Select(file => file.FileName), loaded);
    }

    /// <summary>The same, for the files <paramref name="loaded"/> lists as the ones the process
    /// holds and the file names <paramref name="pinned"/> lists as the pinned ones.</summary>
    internal static string? Conflict(string directory, IEnumerable<string> pinned, IEnumerable<string> loaded)
    {
        if (!Directory.Exists(directory)) return null;
        var names = pinned.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var held = loaded
            .Where(path => names.Contains(Path.GetFileName(path)))
            .Where(path => File.Exists(Path.Combine(directory, Path.GetFileName(path))))
            // A copy of the very same file, under another path, binds as well as the folder's own.
            .Where(path => !SameContents(path, Path.Combine(directory, Path.GetFileName(path))))
            .Select(path => $"{Path.GetFileName(path)} {FileVersionInfo.GetVersionInfo(path).FileVersion} from '{Path.GetDirectoryName(path)}'")
            .ToList();
        return held.Count == 0 ? null : string.Join("; ", held);
    }

    private static bool SameContents(string a, string b)
    {
        using var first = File.OpenRead(a);
        using var second = File.OpenRead(b);
        return first.Length == second.Length && CudaLibraryCache.Sha256Of(first) == CudaLibraryCache.Sha256Of(second);
    }
}
