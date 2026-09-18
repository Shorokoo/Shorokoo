using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>Why a backend cannot be used here, or <see cref="None"/> when it can.</summary>
public enum BackendRejection
{
    /// <summary>It can be used.</summary>
    None = 0,

    /// <summary>The file is not there, or is not a managed assembly at all.</summary>
    Unreadable,

    /// <summary>A managed assembly, but not one declaring itself a Shorokoo backend.</summary>
    NotABackend,

    /// <summary>Built for a different operating system.</summary>
    WrongOperatingSystem,

    /// <summary>Built for a different processor architecture.</summary>
    WrongArchitecture,

    /// <summary>A native library it declares is not deployed anywhere the backend's own folder
    /// carries natives — neither flat beside it nor under <c>runtimes/&lt;rid&gt;/native/</c>.</summary>
    MissingNative,

    /// <summary>It needs a CUDA runtime this machine does not have.</summary>
    MissingCudaRuntime,

    /// <summary>It fits this machine, but loading it did not yield a usable backend.</summary>
    NotLoadable,
}

/// <summary>
/// What <see cref="BackendPackage.Probe"/> found: whether the backend can be used here, and if
/// not, which of the reasons it is and what to do about it.
/// </summary>
/// <param name="Supported">Whether this backend can be loaded on this machine.</param>
/// <param name="Reason">Why not, or <see cref="BackendRejection.None"/>.</param>
/// <param name="Detail">A sentence for a human, naming the thing that is wrong.</param>
/// <param name="Os">The operating system it declares, when it declared one.</param>
/// <param name="Architecture">The architecture it declares, when it declared one.</param>
/// <param name="Device">The device it declares, when it declared one.</param>
public readonly record struct BackendProbe(
    bool Supported, BackendRejection Reason, string Detail,
    string? Os = null, string? Architecture = null, string? Device = null)
{
    /// <inheritdoc/>
    public override string ToString() => Supported ? "supported" : $"{Reason}: {Detail}";
}

/// <summary>
/// Loading a Shorokoo backend from a DLL at runtime, without having referenced it at compile time
/// and without having to know in advance whether it will work here.
///
/// <para><see cref="Probe"/> answers that question by reading the file's metadata — it does not
/// load the assembly, resolve its references or run any of its code — so a backend for another
/// operating system is turned away as a value rather than as an exception thrown from somewhere
/// inside the loader. <see cref="TryLoad"/> does the same and then loads the ones that pass.</para>
///
/// <para>Several may be loaded at once. Each gets a load context of its own, so each binds its own
/// native ONNX Runtime — which is what lets one process drive a CPU backend and a CUDA one
/// together.</para>
/// </summary>
public static class BackendPackage
{
    private const string AttributeName = "ShorokooBackendAttribute";

    /// <summary>
    /// Whether the backend assembly at <paramref name="assemblyPath"/> can be used on this
    /// machine, and why not when it cannot.
    ///
    /// <para>Never throws for an answer of "no": a missing file, a file that is not an assembly,
    /// an assembly that is not a backend and a backend for the wrong platform are all results.</para>
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="assemblyPath"/> is blank.</exception>
    public static BackendProbe Probe(string assemblyPath)
    {
        if (string.IsNullOrWhiteSpace(assemblyPath))
            throw new ArgumentException("A backend's path is required.", nameof(assemblyPath));

        string full;
        try
        {
            full = Path.GetFullPath(assemblyPath);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return new(false, BackendRejection.Unreadable,
                $"'{assemblyPath}' is not a usable path: {ex.Message}");
        }

        if (!File.Exists(full))
            return new(false, BackendRejection.Unreadable, $"There is no file at '{full}'.");

        ImmutableDictionary<string, string> declared;
        try
        {
            declared = ReadManifest(full);
        }
        catch (BadImageFormatException)
        {
            return new(false, BackendRejection.Unreadable,
                $"'{full}' is not a managed assembly, so it cannot be a Shorokoo backend.");
        }
        // Every way a file can fail to be read as an assembly, not the three seen so far. This
        // method's whole promise is that walking a folder of candidates never throws, and the
        // caller who most needs that is the one whose folder holds something unexpected:
        // UnauthorizedAccessException does not derive from IOException, so a DLL this process may
        // not open came straight out of Probe -- and out of TryLoad, which calls it before its own
        // try. A module rather than an assembly does the same through InvalidOperationException.
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return new(false, BackendRejection.Unreadable, $"'{full}' could not be read: {ex.Message}");
        }

        if (declared.IsEmpty)
            return new(false, BackendRejection.NotABackend,
                $"'{Path.GetFileName(full)}' declares no [ShorokooBackend], so it is not a backend "
                + "-- or was built against a Shorokoo too old to declare one.");

        var os = declared.GetValueOrDefault("os", "");
        var arch = declared.GetValueOrDefault("architecture", "");
        var device = declared.GetValueOrDefault("device", "");
        BackendProbe No(BackendRejection reason, string detail)
            => new(false, reason, detail, os, arch, device);

        // Before OSPlatform.Create, which throws on an empty string: a manifest that names no
        // operating system is one this machine is not, and saying so is this method's whole job.
        // Nothing here may answer by throwing, least of all for a file a caller is merely probing.
        if (os.Length == 0)
            return No(BackendRejection.NotABackend,
                $"'{Path.GetFileName(full)}' carries a [ShorokooBackend] that names no operating "
                + "system, so there is no way to tell whether it fits this machine.");

        if (!OSPlatform.Create(os.ToUpperInvariant()).Equals(CurrentPlatform()))
            return No(BackendRejection.WrongOperatingSystem,
                $"'{Path.GetFileName(full)}' is a {os} backend and this is {CurrentOsName()}.");

        if (!arch.Equals(RuntimeInformation.ProcessArchitecture.ToString(), StringComparison.OrdinalIgnoreCase))
            return No(BackendRejection.WrongArchitecture,
                $"'{Path.GetFileName(full)}' is built for {arch} and this process is "
                + $"{RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant()}.");

        var directory = Path.GetDirectoryName(full)!;
        var missing = declared.GetValueOrDefault("natives", "")
            .Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(n => ResolveNative(directory, n) is null)
            .ToList();
        if (missing.Count > 0)
            return No(BackendRejection.MissingNative,
                $"'{Path.GetFileName(full)}' is missing {string.Join(", ", missing)}: not beside it "
                + $"in '{directory}', and not under '{RuntimesNativeFolder()}' there either.");

        if (declared.TryGetValue("requirescudaruntime", out var cuda)
            && !string.IsNullOrEmpty(cuda) && !CudaRuntime.TryMemGetInfo(out _, out _))
            return No(BackendRejection.MissingCudaRuntime,
                $"'{Path.GetFileName(full)}' needs a CUDA {cuda}.x runtime, which this machine "
                + "does not have -- no driver, no device, or the toolkit is not installed.");

        return new(true, BackendRejection.None, "supported", os, arch, device);
    }

    /// <summary>
    /// Loads the backend at <paramref name="assemblyPath"/> if it can be used here, and reports why
    /// not if it cannot. A backend that does not fit this machine is a <c>false</c>, not a throw —
    /// the caller decides whether that is a fallback, a warning or the end of the program.
    ///
    /// <para>The backend is loaded into isolation of its own, bound to the native ONNX Runtime
    /// found in its folder — flat beside it, or under <c>runtimes/&lt;rid&gt;/native/</c>, whichever
    /// the deployment used — so several loaded this way run side by side without sharing a
    /// runtime.</para>
    /// </summary>
    /// <param name="assemblyPath">The backend DLL.</param>
    /// <param name="backend">The loaded backend, or null.</param>
    /// <param name="failure">Why it was not loaded, meaningful only when this returns false.</param>
    public static bool TryLoad(
        string assemblyPath,
        out IShorokooInferenceBackend? backend,
        out BackendProbe failure)
    {
        backend = null;
        var probe = Probe(assemblyPath);
        if (!probe.Supported) { failure = probe; return false; }

        var full = Path.GetFullPath(assemblyPath);
        var directory = Path.GetDirectoryName(full)!;
        var name = Path.GetFileNameWithoutExtension(full);

        // Resolved rather than assumed: IsolatedBackend.Load binds this path as a file, so it has
        // to be where the native really is, not where a flat deployment would have put it.
        var native = ResolveNative(directory, NativeRuntimeFileName());

        if (native is null)
        {
            failure = probe with
            {
                Supported = false,
                Reason = BackendRejection.MissingNative,
                Detail = $"'{name}' passed its own checks but the native ONNX Runtime it binds "
                    + $"({NativeRuntimeFileName()}) is in neither '{directory}' nor "
                    + $"'{Path.Combine(directory, RuntimesNativeFolder())}'.",
            };
            return false;
        }

        try
        {
            backend = IsolatedBackend.Load(new IsolatedBackendSpec
            {
                Name = $"{name} ({probe.Device})",
                BackendAssembly = name,
                NativeRuntimePath = native,
                ProbeDirectory = directory,
            });
            failure = probe;
            return true;
        }
        // Every way loading can fail, not a list of the ones seen so far. A backend that fits the
        // machine on paper can still fail to load -- a native of the wrong bitness, a backend
        // constructor that reaches for a driver -- and a caller walking a folder of candidates has
        // to be able to skip that file rather than crash on it. That is what this method promises
        // by answering with a bool.
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            failure = probe with
            {
                Supported = false,
                Reason = BackendRejection.NotLoadable,
                Detail = $"'{name}' fits this machine but could not be loaded: "
                    + (ex is TargetInvocationException { InnerException: { } inner } ? inner : ex).Message,
            };
            return false;
        }
    }

    /// <summary>
    /// The backend's declaration, read straight out of the file's metadata tables. Nothing is
    /// loaded, resolved or executed — which is what makes a rejection cheap and total.
    /// </summary>
    private static ImmutableDictionary<string, string> ReadManifest(string path)
    {
        using var stream = File.OpenRead(path);
        using var pe = new PEReader(stream);
        if (!pe.HasMetadata) throw new BadImageFormatException("No CLI metadata.", path);

        var reader = pe.GetMetadataReader();
        foreach (var handle in reader.GetAssemblyDefinition().GetCustomAttributes())
        {
            var attribute = reader.GetCustomAttribute(handle);
            if (NameOf(reader, attribute) != AttributeName) continue;

            // Every value is a string, so the blob is walked rather than decoded through a type
            // provider: a provider would have to resolve types out of an assembly this is
            // deliberately not loading.
            return ReadAllStrings(reader.GetBlobReader(attribute.Value));
        }
        return ImmutableDictionary<string, string>.Empty;
    }

    private static string? NameOf(MetadataReader reader, CustomAttribute attribute)
        => attribute.Constructor.Kind switch
        {
            HandleKind.MemberReference => reader.GetString(
                reader.GetTypeReference((TypeReferenceHandle)reader
                    .GetMemberReference((MemberReferenceHandle)attribute.Constructor).Parent).Name),
            HandleKind.MethodDefinition => reader.GetString(
                reader.GetTypeDefinition(reader
                    .GetMethodDefinition((MethodDefinitionHandle)attribute.Constructor)
                    .GetDeclaringType()).Name),
            _ => null,
        };

    /// <summary>
    /// The attribute's three positional strings and then its named ones, keyed lower-case. The
    /// blob layout is fixed by ECMA-335: a 0x0001 prologue, the fixed arguments in declaration
    /// order, a count of named arguments, and then each named one as kind, type, name, value.
    /// </summary>
    private static ImmutableDictionary<string, string> ReadAllStrings(BlobReader blob)
    {
        var values = ImmutableDictionary.CreateBuilder<string, string>();
        if (blob.ReadUInt16() != 1) return values.ToImmutable();

        foreach (var name in (string[])["os", "architecture", "device"])
            values[name] = blob.ReadSerializedString() ?? "";

        var namedCount = blob.ReadUInt16();
        for (int i = 0; i < namedCount && blob.RemainingBytes > 0; i++)
        {
            blob.ReadByte();                       // field or property
            var elementType = blob.ReadByte();     // ELEMENT_TYPE_STRING for all of ours
            var name = blob.ReadSerializedString();
            if (elementType != 0x0E || name is null) break;   // not a string: stop rather than guess
            values[name.ToLowerInvariant()] = blob.ReadSerializedString() ?? "";
        }
        return values.ToImmutable();
    }

    /// <summary>
    /// Where <paramref name="nativeFileName"/> actually is for a backend deployed in
    /// <paramref name="directory"/>, or null when it is in neither place a .NET build puts one.
    ///
    /// <para>There are two layouts, and a backend has no say in which it gets. The standard one
    /// is <c>runtimes/&lt;rid&gt;/native/</c>, which is how a native NuGet package ships and how
    /// the host resolves a P/Invoke through <c>deps.json</c>. The flat one — the native sitting
    /// beside the managed assembly — is an artefact of a package's own build props copying it to
    /// the output root, and ONNX Runtime's props do that <i>on Windows only</i>. So these
    /// backends build flat on Windows and under <c>runtimes/</c> on Linux and macOS. A program
    /// that installs a backend's NuGet package gets <c>runtimes/</c> on every platform,
    /// Windows included, because those props sit in the package's <c>build/</c> folder rather
    /// than its <c>buildTransitive/</c> one and so never reach a consumer that did not reference
    /// ONNX Runtime itself. Assuming either layout makes the backend unloadable wherever the
    /// other one is what the deployment produced.</para>
    ///
    /// <para>Flat wins when both exist. A build that flattened deliberately — the
    /// <c>ShorokooBackendNatives</c> deployment that gives each isolated backend a native folder
    /// of its own, say — means the file it put there, not whatever a package happened to leave
    /// under <c>runtimes/</c> beside it.</para>
    /// </summary>
    internal static string? ResolveNative(string directory, string nativeFileName)
    {
        var flat = Path.Combine(directory, nativeFileName);
        if (File.Exists(flat)) return flat;

        var runtimes = Path.Combine(directory, "runtimes");
        foreach (var rid in CandidateRuntimeIdentifiers(runtimes))
        {
            var path = Path.Combine(runtimes, rid, "native", nativeFileName);
            if (File.Exists(path)) return path;
        }
        return null;
    }

    /// <summary>
    /// The runtime identifiers whose <c>native/</c> folder may hold this machine's copy of a
    /// native, most specific first.
    ///
    /// <para>The host's own identifier comes first, then the portable one it normally equals
    /// (<c>win-x64</c>, <c>linux-x64</c>, <c>osx-arm64</c>) for the case where the host reports
    /// something narrower than the folder a package shipped. Last, and only if neither was
    /// there, any folder actually present for this operating system and this architecture: that
    /// is what finds a package shipping under a RID this code cannot name in advance —
    /// <c>linux-musl-x64</c>, <c>win10-x64</c> — without ever crossing an OS or an architecture
    /// boundary, which is the part that would load a library this process cannot run.</para>
    /// </summary>
    private static IEnumerable<string> CandidateRuntimeIdentifiers(string runtimesDirectory)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var rid in (string[])[RuntimeInformation.RuntimeIdentifier, PortableRuntimeIdentifier()])
            if (!string.IsNullOrEmpty(rid) && seen.Add(rid))
                yield return rid;

        if (!Directory.Exists(runtimesDirectory)) yield break;

        var os = RuntimeIdentifierOs();
        var architecture = "-" + RuntimeIdentifierArchitecture();
        // Ordered, so that a folder set offering more than one match resolves to the same file on
        // every run rather than to whatever the file system happened to enumerate first.
        foreach (var candidate in Directory.EnumerateDirectories(runtimesDirectory)
                     .Select(Path.GetFileName)
                     .OfType<string>()
                     .Order(StringComparer.Ordinal))
            if (candidate.StartsWith(os, StringComparison.OrdinalIgnoreCase)
                && candidate.EndsWith(architecture, StringComparison.OrdinalIgnoreCase)
                && seen.Add(candidate))
                yield return candidate;
    }

    /// <summary>The path a <c>runtimes/</c>-layout native sits at, relative to the backend's
    /// folder. For naming the place in a rejection, not for probing — probing goes through
    /// <see cref="CandidateRuntimeIdentifiers"/>, which considers more than this one.</summary>
    private static string RuntimesNativeFolder()
        => Path.Combine("runtimes", PortableRuntimeIdentifier(), "native");

    private static string PortableRuntimeIdentifier()
        => $"{RuntimeIdentifierOs()}-{RuntimeIdentifierArchitecture()}";

    // "win", not "windows": a RID spells the operating system its own way, and this is the token
    // a runtimes/ folder is named with rather than the one the backend's manifest declares.
    private static string RuntimeIdentifierOs()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win"
            : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux"
            : "osx";

    private static string RuntimeIdentifierArchitecture()
        => RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();

    private static OSPlatform CurrentPlatform()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? OSPlatform.Windows
            : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? OSPlatform.Linux
            : OSPlatform.OSX;

    private static string CurrentOsName() => CurrentPlatform().ToString().ToLowerInvariant();

    private static string NativeRuntimeFileName()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "onnxruntime.dll" : "libonnxruntime.so";
}
