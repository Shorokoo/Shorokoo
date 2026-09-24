using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Shorokoo.PythonHost;

/// <summary>
/// The exact set of packages a backend's Python environment holds: a <c>uv pip compile</c> lock,
/// the Python version it was compiled for, and the index arguments it has to be installed with.
///
/// <para>The index arguments are part of the lock, not an afterthought. A compiled lock records
/// versions such as <c>torch==2.14.0+cpu</c> but not where they came from, so installing it
/// without the index it was compiled against fails to find them. They are hashed together with
/// the requirements, so two locks that would install different things never share a cached
/// environment.</para>
/// </summary>
public sealed class PythonEnvironmentLock
{
    /// <summary>Creates a lock.</summary>
    /// <param name="name">A short name for the environment, used in its cache folder's name.</param>
    /// <param name="pythonVersion">The CPython version it was compiled for, as "major.minor".</param>
    /// <param name="requirements">The compiled requirements, as <c>uv pip compile</c> wrote them.</param>
    /// <param name="indexArguments">The arguments naming the indexes to install from, each a
    /// separate token (<c>--index-url</c>, then the URL).</param>
    public PythonEnvironmentLock(
        string name, string pythonVersion, string requirements, IReadOnlyList<string> indexArguments)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(pythonVersion);
        ArgumentNullException.ThrowIfNull(requirements);
        ArgumentNullException.ThrowIfNull(indexArguments);
        Name = name;
        PythonVersion = pythonVersion;
        Requirements = requirements;
        IndexArguments = [.. indexArguments];
        var digest = SHA256.HashData(Encoding.UTF8.GetBytes(
            string.Join('\n', [pythonVersion, requirements.ReplaceLineEndings("\n"), .. IndexArguments])));
        Hash = Convert.ToHexStringLower(digest)[..16];
    }

    /// <summary>The environment's short name, e.g. <c>cpu</c>.</summary>
    public string Name { get; }

    /// <summary>The CPython version the lock was compiled for, as "major.minor".</summary>
    public string PythonVersion { get; }

    /// <summary>The compiled requirements.</summary>
    public string Requirements { get; }

    /// <summary>The index arguments the requirements must be installed with.</summary>
    public IReadOnlyList<string> IndexArguments { get; }

    /// <summary>A digest of everything above, which names the cached environment built from it.</summary>
    public string Hash { get; }

    /// <summary>The cache folder's name: the lock's name and its hash.</summary>
    public string CacheKey => $"{Name}-{Hash}";

    /// <summary>The PyTorch CPU build, from PyTorch's own <c>cpu</c> index.</summary>
    public static PythonEnvironmentLock Cpu { get; } = Embedded("cpu");

    /// <summary>The PyTorch build for CUDA 13, from PyPI, whose default Linux wheel it is.</summary>
    public static PythonEnvironmentLock Cu13 { get; } = Embedded("cu13");

    /// <inheritdoc/>
    public override string ToString() => $"{CacheKey} (CPython {PythonVersion})";

    private static PythonEnvironmentLock Embedded(string name)
    {
        var requirements = ReadResource($"Environments/{name}/requirements.txt");
        var index = ReadResource($"Environments/{name}/uv-index.txt")
            .Split((char[])[' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        return new PythonEnvironmentLock(name, "3.12", requirements, index);
    }

    private static string ReadResource(string logicalName)
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(logicalName)
            ?? throw new InvalidOperationException(
                $"The lock resource '{logicalName}' is missing from {typeof(PythonEnvironmentLock).Assembly.GetName().Name}.");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    }
}
