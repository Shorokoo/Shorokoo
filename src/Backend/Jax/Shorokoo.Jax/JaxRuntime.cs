using Python.Runtime;
using Shorokoo.PythonHost;

namespace Shorokoo.Jax;

/// <summary>
/// JAX, loaded once into this process's interpreter, and the support package's entry points.
///
/// <para>One per process, shared by every JAX backend — CPU and CUDA alike — because a process has
/// one interpreter and so one JAX. That is also why <see cref="Identity"/>, the runtime identity of
/// every JAX backend, is a single object: a value either of them made is one the other's sessions
/// can be handed as it stands, JAX moving it to their device.</para>
///
/// <para>JAX shares the interpreter, and so the environment, with every other Python-based backend
/// in the process: the first backend to need Python decides the environment, and the environments
/// Shorokoo provisions hold PyTorch and JAX together, built for one CUDA major.</para>
/// </summary>
internal sealed class JaxRuntime
{
    /// <summary>The runtime identity every JAX backend in this process answers.</summary>
    public static object Identity { get; } = new();

    private static readonly object Gate = new();
    private static JaxRuntime? _instance;

    /// <summary>JAX in this process, or null before any JAX backend has started it.</summary>
    public static JaxRuntime? Current => Volatile.Read(ref _instance);

    private JaxRuntime(PythonEnvironment environment, PyObject runtime)
    {
        Environment = environment;
        FromHost = runtime.GetAttr("from_host");
        Empty = runtime.GetAttr("empty");
        HostCopy = runtime.GetAttr("host_copy");
        Describe = runtime.GetAttr("describe");
        Run = runtime.GetAttr("run");
        LoadModel = runtime.GetAttr("load_model");
        Prepare = runtime.GetAttr("prepare");
        ArenaStatistics = runtime.GetAttr("arena_statistics");
        using var version = runtime.InvokeMethod("jax_version");
        JaxVersion = version.As<string>();
        using var cuda = new PyString("cuda");
        using var count = runtime.InvokeMethod("device_count", cuda);
        CudaDeviceCount = count.As<int>();
    }

    /// <summary>The environment JAX was imported from.</summary>
    public PythonEnvironment Environment { get; }

    /// <summary>The JAX version, e.g. <c>0.11.2</c>.</summary>
    public string JaxVersion { get; }

    /// <summary>How many CUDA devices JAX sees; zero where it has no CUDA plugin, or the machine no
    /// card.</summary>
    public int CudaDeviceCount { get; }

    // The support package's runtime functions, looked up once. Held for the life of the process,
    // which is the interpreter's life too.
    public PyObject FromHost { get; }
    public PyObject Empty { get; }
    public PyObject HostCopy { get; }
    public PyObject Describe { get; }
    public PyObject Run { get; }
    public PyObject LoadModel { get; }
    public PyObject Prepare { get; }
    public PyObject ArenaStatistics { get; }

    /// <summary>The name of the Python exception an operator raises where a model needs a number
    /// its inputs' values decide.</summary>
    public const string DataDependentShape = "DataDependentShape";

    /// <summary>
    /// JAX in this process, started over the environment <paramref name="lockFile"/> and
    /// <paramref name="options"/> resolve to — or over the one already running, where they name
    /// none of their own.
    /// </summary>
    /// <exception cref="PythonEnvironmentException">No environment could be resolved or started,
    /// it lacks JAX, or it is not the one already running.</exception>
    public static JaxRuntime Start(PythonEnvironmentLock lockFile, PythonEnvironmentOptions options)
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

    private static JaxRuntime Import(PythonEnvironment environment)
    {
        using (PythonRuntime.Gil())
        {
            try
            {
                PythonRuntime.InstallEmbeddedPackage(typeof(JaxRuntime).Assembly, "shorokoo_jax");
                using var runtime = Py.Import("shorokoo_jax.runtime");
                return new JaxRuntime(environment, runtime);
            }
            catch (PythonException ex) when (ex.Type.Name is "ModuleNotFoundError" or "ImportError")
            {
                throw new PythonEnvironmentException(PythonEnvironmentFailure.MissingPackage,
                    $"The Python environment at '{environment.Directory}' cannot import what the JAX "
                    + $"backend needs ({ex.Message}). Install jax and numpy into it, or leave "
                    + $"{PythonEnvironmentResolver.EnvironmentVariable} unset to have an environment provisioned.",
                    ex);
            }
        }
    }
}
