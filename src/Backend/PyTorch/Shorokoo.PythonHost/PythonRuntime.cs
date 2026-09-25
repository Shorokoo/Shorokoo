using Python.Runtime;

namespace Shorokoo.PythonHost;

/// <summary>
/// The one embedded CPython interpreter of this process.
///
/// <para>CPython cannot be started twice in a process, nor over two environments, so the first
/// backend that needs it starts it over the environment it resolved and every later one shares
/// it. A later request for a different environment is refused rather than quietly ignored: its
/// packages are not the ones that would run.</para>
///
/// <para><b>GIL discipline.</b> Once started, the interpreter's global lock is released
/// (<c>PythonEngine.BeginAllowThreads</c>), so any thread can take it. Every call into Python,
/// and every disposal of a <see cref="PyObject"/>, is made holding it — <see cref="Gil"/>. A
/// <see cref="PyObject"/> nobody disposed is not a leak and not a crash: pythonnet's finalizer
/// only queues its reference, which is dropped the next time some thread takes the lock.</para>
///
/// <para><b>Rooting.</b> pythonnet hands the C API a bare pointer out of a <see cref="PyObject"/>,
/// so the wrapper is only protected by being reachable. Where a caller reads a raw address out of
/// a Python object — a tensor's data pointer — it must keep the wrapper alive until it has
/// finished with the address, with <see cref="GC.KeepAlive"/> after the last use: a collection
/// before that queues the reference for release, and any thread taking the lock — including one
/// the framework's own kernels let in by releasing it — then frees the memory underneath.</para>
/// </summary>
public static class PythonRuntime
{
    private static readonly object Gate = new();
    private static PythonEnvironment? _environment;
    private static PythonEnvironmentException? _failure;

    /// <summary>The environment the interpreter runs over, or null before it is started.</summary>
    public static PythonEnvironment? Environment => _environment;

    /// <summary>
    /// Starts the interpreter over <paramref name="environment"/>, or confirms it already runs over
    /// it. Safe to call from any thread, any number of times.
    /// </summary>
    /// <exception cref="PythonEnvironmentException">The interpreter already runs over a different
    /// environment, or could not be started.</exception>
    public static PythonEnvironment Start(PythonEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);
        lock (Gate)
        {
            if (_environment is { } running)
            {
                if (!string.Equals(running.Directory, environment.Directory, StringComparison.Ordinal))
                    throw new PythonEnvironmentException(PythonEnvironmentFailure.EnvironmentConflict,
                        $"This process already runs Python over '{running.Directory}', and a process can "
                        + $"start only one interpreter, so '{environment.Directory}' cannot be used as "
                        + "well. Give every Python-based backend in the process the same environment.");
                return running;
            }

            // CPython is not started twice in a process, so one that failed to start stays failed.
            if (_failure is { } failed)
                throw new PythonEnvironmentException(PythonEnvironmentFailure.InterpreterFailed,
                    $"CPython failed to start earlier in this process, and a process cannot start it again: {failed.Message}",
                    failed);

            try
            {
                Runtime.PythonDLL = environment.LibPython;
                PythonEngine.PythonHome = environment.PythonHome;
                PythonEngine.Initialize();
                // Initializing leaves the lock held by this thread, and it is let go of however the
                // setup below ends: held, every other thread's Gil() would wait forever.
                try
                {
                    using (Py.GIL())
                    {
                        using var scope = Py.CreateScope();
                        scope.Set("_site_packages", environment.SitePackages);
                        scope.Set("_prefix", environment.Directory);
                        scope.Set("_executable", Executable(environment));
                        // The environment's packages first, ahead of anything the base installation or
                        // the user's own site-packages would offer, and its .pth files honoured; and
                        // sys.prefix made the environment's, as a venv's own interpreter would have it.
                        scope.Exec("""
                            import sys, site
                            sys.path.insert(0, _site_packages)
                            site.addsitedir(_site_packages)
                            sys.prefix = sys.exec_prefix = _prefix
                            sys.executable = _executable
                            """);
                    }
                }
                finally
                {
                    PythonEngine.BeginAllowThreads();
                }
            }
            catch (Exception ex) when (ex is not PythonEnvironmentException)
            {
                throw _failure = new PythonEnvironmentException(PythonEnvironmentFailure.InterpreterFailed,
                    $"CPython could not be started from '{environment.LibPython}' over "
                    + $"'{environment.Directory}': {ex.Message}", ex);
            }
            _environment = environment;
            return environment;
        }
    }

    /// <summary>
    /// Takes the global interpreter lock for the calling thread until the result is disposed:
    /// <c>using (PythonRuntime.Gil()) { ... }</c>. Reentrant on one thread.
    /// </summary>
    /// <exception cref="InvalidOperationException">The interpreter has not been started.</exception>
    public static IDisposable Gil()
    {
        if (_environment is null)
            throw new InvalidOperationException("The Python interpreter has not been started.");
        return Py.GIL();
    }

    /// <summary>Drops <paramref name="value"/>'s reference now, under the lock, rather than
    /// leaving it to the finalizer. Null is ignored.</summary>
    public static void Release(PyObject? value)
    {
        if (value is null) return;
        using (Gil()) value.Dispose();
    }

    private static string Executable(PythonEnvironment environment)
        => OperatingSystem.IsWindows()
            ? Path.Combine(environment.Directory, "Scripts", "python.exe")
            : Path.Combine(environment.Directory, "bin", "python");
}
