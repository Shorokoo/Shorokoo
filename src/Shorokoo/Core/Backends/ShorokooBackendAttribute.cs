using System;

namespace Shorokoo.Core.Backends;

/// <summary>
/// Declares an assembly to be a Shorokoo backend, and says what it needs to run.
///
/// <para>Carried in the assembly so it cannot drift from it, and written entirely in strings so
/// that <see cref="BackendPackage.Probe"/> can read it out of the file's metadata without loading
/// the assembly, resolving its references or running any of its code. That is the whole point: a
/// backend for the wrong operating system has to be turned away <i>before</i> anything native is
/// touched, because that is the last moment at which failing is cheap and says why.</para>
///
/// <code>
/// [assembly: ShorokooBackend("linux", "x64", "cuda",
///     Natives = "libonnxruntime.so;libonnxruntime_providers_cuda.so",
///     RequiresCudaRuntime = "12")]
/// </code>
/// </summary>
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = false)]
public sealed class ShorokooBackendAttribute : Attribute
{
    /// <summary>Declares this assembly a backend for <paramref name="os"/> on
    /// <paramref name="architecture"/>, running on <paramref name="device"/>.</summary>
    /// <param name="os">"windows", "linux" or "osx", as
    /// <see cref="System.Runtime.InteropServices.OSPlatform"/> spells them, lower-cased.</param>
    /// <param name="architecture">"x64", "arm64", as
    /// <see cref="System.Runtime.InteropServices.Architecture"/> spells them, lower-cased.</param>
    /// <param name="device">"cpu" or "cuda".</param>
    public ShorokooBackendAttribute(string os, string architecture, string device)
    {
        Os = os;
        Architecture = architecture;
        Device = device;
    }

    /// <summary>The operating system this backend's natives are built for.</summary>
    public string Os { get; }

    /// <summary>The processor architecture this backend's natives are built for.</summary>
    public string Architecture { get; }

    /// <summary>The device its sessions run on: "cpu" or "cuda".</summary>
    public string Device { get; }

    /// <summary>
    /// The native libraries this assembly's folder must carry for it to load, separated by
    /// semicolons. Checked by <see cref="BackendPackage.Probe"/>, so a deployment missing one is
    /// refused with the file named rather than failing at the first P/Invoke.
    ///
    /// <para>Names only, never paths: where the file lands is the deployment's business, not the
    /// backend's. A build that flattens its natives puts them beside this assembly, and one that
    /// does not leaves them under <c>runtimes/&lt;rid&gt;/native/</c> — the probe looks in both,
    /// because which one a backend gets depends on the platform it was built on rather than on
    /// anything declared here.</para>
    /// </summary>
    public string Natives { get; set; } = "";

    /// <summary>
    /// The major version of the CUDA runtime this backend needs, or null when it needs none.
    /// A probe on a machine without it answers no rather than throwing from a driver call.
    /// </summary>
    public string? RequiresCudaRuntime { get; set; }

    /// <summary>
    /// How a program comes to use this backend: null for a backend <see cref="DefaultBackend"/>
    /// may discover, or <see cref="ExplicitSelection"/> for one that is only ever used where a
    /// program names it — <c>new ComputeContext(new TorchCpuBackend())</c>.
    ///
    /// <para>An explicit backend can be deployed beside a discoverable one without making
    /// discovery ambiguous, because discovery never counts it, and
    /// <see cref="BackendPackage.TryLoad"/> does not load it: it binds no native ONNX Runtime for
    /// isolation to give it. <see cref="BackendPackage.Probe"/> reads it like any other.</para>
    /// </summary>
    public string? Selection { get; set; }

    /// <summary>The <see cref="Selection"/> of a backend a program must name to use.</summary>
    public const string ExplicitSelection = "explicit";
}
