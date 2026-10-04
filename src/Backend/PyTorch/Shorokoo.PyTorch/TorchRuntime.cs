using Python.Runtime;
using Shorokoo.Core.Backends;
using Shorokoo.PythonHost;

namespace Shorokoo.PyTorch;

/// <summary>
/// PyTorch, loaded once into this process's interpreter, and the support package's entry points.
///
/// <para>One per process, shared by every torch backend — CPU and CUDA alike — because a process
/// has one interpreter and so one torch. That is also why <see cref="Identity"/>, the runtime
/// identity of every torch backend, is a single object: a tensor either of them made is a torch
/// tensor the other's sessions can be handed as it stands.</para>
///
/// <para>The first backend to need torch decides the environment. A later backend that names none
/// of its own shares the one already running; one that names a different environment is refused,
/// since a process cannot run two.</para>
/// </summary>
internal sealed class TorchRuntime
{
    /// <summary>The runtime identity every torch backend in this process answers.</summary>
    public static object Identity { get; } = new();

    private static readonly object Gate = new();
    private static TorchRuntime? _instance;

    /// <summary>torch in this process, or null before any torch backend has started it.</summary>
    public static TorchRuntime? Current => Volatile.Read(ref _instance);

    private TorchRuntime(PythonEnvironment environment, PyObject runtime)
    {
        Environment = environment;
        FromHost = runtime.GetAttr("from_host");
        Empty = runtime.GetAttr("empty");
        Strings = runtime.GetAttr("strings");
        StringList = runtime.GetAttr("string_list");
        HostCopy = runtime.GetAttr("host_copy");
        SequenceElement = runtime.GetAttr("sequence_element");
        Describe = runtime.GetAttr("describe");
        Run = runtime.GetAttr("run");
        LoadModel = runtime.GetAttr("load_model");
        ConstantStorages = runtime.GetAttr("constant_storages");
        ConstantIds = runtime.GetAttr("constant_ids");
        ArenaStatistics = runtime.GetAttr("arena_statistics");
        TorchVersion = runtime.InvokeMethod("torch_version").As<string>();
        TorchCudaVersion = runtime.InvokeMethod("torch_cuda_version").As<string>();
        CudaDeviceCount = runtime.InvokeMethod("cuda_device_count").As<int>();
    }

    /// <summary>The environment torch was imported from.</summary>
    public PythonEnvironment Environment { get; }

    /// <summary>The torch version, e.g. <c>2.14.0+cpu</c>.</summary>
    public string TorchVersion { get; }

    /// <summary>The CUDA version torch was built for, e.g. <c>13.0</c>, or empty for a CPU build.</summary>
    public string TorchCudaVersion { get; }

    /// <summary>How many CUDA devices torch sees; zero for a CPU build or a machine with no card.</summary>
    public int CudaDeviceCount { get; }

    // The support package's runtime functions, looked up once. Held for the life of the process,
    // which is the interpreter's life too.
    public PyObject FromHost { get; }
    public PyObject Empty { get; }
    public PyObject Strings { get; }
    public PyObject StringList { get; }
    public PyObject HostCopy { get; }
    public PyObject SequenceElement { get; }
    public PyObject Describe { get; }
    public PyObject Run { get; }
    public PyObject LoadModel { get; }
    public PyObject ConstantStorages { get; }
    public PyObject ConstantIds { get; }
    public PyObject ArenaStatistics { get; }

    /// <summary>The name of the Python exception a run stopped between two nodes raises.</summary>
    public const string RunStopped = "RunStopped";

    /// <summary>
    /// torch in this process, started over the environment <paramref name="lockFile"/> and
    /// <paramref name="options"/> resolve to — or over the one already running, where they name
    /// none of their own.
    /// </summary>
    /// <exception cref="PythonEnvironmentException">No environment could be resolved or started,
    /// it lacks torch, or it is not the one already running.</exception>
    public static TorchRuntime Start(PythonEnvironmentLock lockFile, PythonEnvironmentOptions options)
    {
        lock (Gate)
        {
            var namesOne = !string.IsNullOrWhiteSpace(options.EnvironmentPath)
                || !string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable(PythonEnvironmentResolver.EnvironmentVariable));
            if (_instance is { } running && !namesOne) return running;

            var environment = PythonRuntime.Environment is { } started && !namesOne
                ? started.As(PythonEnvironmentSource.Running)
                : PythonEnvironmentResolver.Resolve(lockFile, options);
            PythonRuntime.Start(environment);
            return _instance ??= Import(environment);
        }
    }

    private static TorchRuntime Import(PythonEnvironment environment)
    {
        var cudaLibraries = environment.CudaLibraryDirectory;
        using (PythonRuntime.Gil())
        {
            try
            {
                PythonRuntime.InstallEmbeddedPackage(typeof(TorchRuntime).Assembly, "shorokoo_torch");
                using var runtime = Py.Import("shorokoo_torch.runtime");
                return new TorchRuntime(environment, runtime);
            }
            catch (PythonException ex) when (ImportFailure(
                ex.Type.Name, () => cudaLibraries is null ? null : CudaLibraries.Conflict(cudaLibraries), out var held) is { } failure)
            {
                if (failure == PythonEnvironmentFailure.MissingPackage)
                    throw new PythonEnvironmentException(PythonEnvironmentFailure.MissingPackage,
                        $"The Python environment at '{environment.Directory}' cannot import what the PyTorch "
                        + $"backend needs ({ex.Message}). Install torch and numpy into it, or leave "
                        + $"{PythonEnvironmentResolver.EnvironmentVariable} unset to have an environment provisioned.",
                        ex);
                throw new PythonEnvironmentException(PythonEnvironmentFailure.CudaLibraryConflict,
                    $"PyTorch cannot load the CUDA libraries in '{cudaLibraries}' ({ex.Message}). This process "
                    + $"already holds another release of some of them ({held}). Those libraries load one another "
                    + "by name, so PyTorch's copies bind to the ones held, and two releases do not mix. Shorokoo's "
                    + "CUDA backends load the pinned release, which a provisioned environment's PyTorch carries; "
                    + "so either something else in this process loaded the copies held, or this environment's "
                    + "PyTorch bundles another release than the pinned one.",
                    ex);
            }
        }
    }

    /// <summary>
    /// What an import of the support package that raised Python's <paramref name="exceptionType"/>
    /// failed of: the CUDA libraries, where this process holds another release of some of them
    /// (<paramref name="conflict"/> names those, as <paramref name="held"/>) — which fails the import
    /// as an <c>ImportError</c> as readily as an <c>OSError</c>; else a missing package, for a module
    /// that is not there or did not import; else nothing this can name. A module that is not there at
    /// all is a missing package whatever the process holds.
    /// </summary>
    internal static PythonEnvironmentFailure? ImportFailure(string exceptionType, Func<string?> conflict, out string? held)
    {
        held = exceptionType == "ModuleNotFoundError" ? null : conflict();
        if (held is not null) return PythonEnvironmentFailure.CudaLibraryConflict;
        return exceptionType is "ModuleNotFoundError" or "ImportError" ? PythonEnvironmentFailure.MissingPackage : null;
    }
}
