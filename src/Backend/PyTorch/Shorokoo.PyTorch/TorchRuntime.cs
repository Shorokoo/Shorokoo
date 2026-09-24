using System.Text;
using Python.Runtime;
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
        TorchVersion = runtime.InvokeMethod("torch_version").As<string>();
        CudaDeviceCount = runtime.InvokeMethod("cuda_device_count").As<int>();
    }

    /// <summary>The environment torch was imported from.</summary>
    public PythonEnvironment Environment { get; }

    /// <summary>The torch version, e.g. <c>2.14.0+cpu</c>.</summary>
    public string TorchVersion { get; }

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
        using (PythonRuntime.Gil())
        {
            try
            {
                using var sources = new PyDict();
                foreach (var (module, source) in SupportPackage())
                    sources[module] = new PyString(source);
                using var scope = Py.CreateScope();
                scope.Set("_sources", sources);
                scope.Exec(Bootstrap);
                using var runtime = Py.Import("shorokoo_torch.runtime");
                return new TorchRuntime(environment, runtime);
            }
            catch (PythonException ex) when (ex.Type.Name is "ModuleNotFoundError" or "ImportError")
            {
                throw new PythonEnvironmentException(PythonEnvironmentFailure.MissingPackage,
                    $"The Python environment at '{environment.Directory}' cannot import what the PyTorch "
                    + $"backend needs ({ex.Message}). Install torch and numpy into it, or leave "
                    + $"{PythonEnvironmentResolver.EnvironmentVariable} unset to have an environment provisioned.",
                    ex);
            }
        }
    }

    /// <summary>
    /// Makes the support package importable from the sources embedded in this assembly: a finder
    /// on <c>sys.meta_path</c> serves <c>shorokoo_torch</c> and its modules from memory, and gives
    /// tracebacks their source lines.
    /// </summary>
    private const string Bootstrap = """
        import importlib.abc, importlib.util, linecache, sys

        class _ShorokooTorchSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def __init__(self, sources):
                self.sources = dict(sources)

            def _filename(self, name):
                return "<embedded>/" + name.replace(".", "/") + (".py" if name != "shorokoo_torch" else "/__init__.py")

            def find_spec(self, name, path, target=None):
                if name not in self.sources:
                    return None
                return importlib.util.spec_from_loader(
                    name, self, origin=self._filename(name), is_package=(name == "shorokoo_torch"))

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                source = self.sources[module.__name__]
                filename = self._filename(module.__name__)
                linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
                module.__file__ = filename
                exec(compile(source, filename, "exec"), module.__dict__)

            def get_source(self, name):
                return self.sources[name]

        if not any(isinstance(f, _ShorokooTorchSources) for f in sys.meta_path):
            sys.meta_path.insert(0, _ShorokooTorchSources(_sources))
        """;

    /// <summary>The embedded support package, keyed by module name.</summary>
    private static IEnumerable<(string Module, string Source)> SupportPackage()
    {
        var assembly = typeof(TorchRuntime).Assembly;
        foreach (var name in assembly.GetManifestResourceNames())
        {
            if (!name.StartsWith("shorokoo_torch/", StringComparison.Ordinal) || !name.EndsWith(".py", StringComparison.Ordinal))
                continue;
            using var stream = assembly.GetManifestResourceStream(name)!;
            using var reader = new StreamReader(stream, Encoding.UTF8);
            var file = Path.GetFileNameWithoutExtension(name["shorokoo_torch/".Length..]);
            yield return (file == "__init__" ? "shorokoo_torch" : "shorokoo_torch." + file, reader.ReadToEnd());
        }
    }
}
