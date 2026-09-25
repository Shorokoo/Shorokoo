using Python.Runtime;

namespace Shorokoo.PyTorch;

/// <summary>
/// Calls into Python that release the objects they make for the call's arguments as the call
/// returns, under the interpreter lock the caller holds, rather than leaving them to the finalizer.
/// </summary>
internal static class PyCall
{
    /// <summary><paramref name="function"/> called with <paramref name="arguments"/>: each a
    /// <see cref="PyObject"/> passed as it is — the caller's to release — or an <see cref="int"/>,
    /// <see cref="long"/>, <see cref="bool"/> or <see cref="string"/> made a Python object for the
    /// call alone.</summary>
    public static PyObject Invoke(PyObject function, params object[] arguments)
    {
        var made = new List<PyObject>(arguments.Length);
        try
        {
            var values = new PyObject[arguments.Length];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = arguments[i] switch
                {
                    PyObject value => value,
                    var other => Made(made, Of(other)),
                };
            }
            return function.Invoke(values);
        }
        finally
        {
            foreach (var value in made) value.Dispose();
        }
    }

    /// <summary>Appends <paramref name="value"/> to <paramref name="list"/>, as a Python object made
    /// for the list alone.</summary>
    public static void Append(PyList list, object value)
    {
        using var item = Of(value);
        list.Append(item);
    }

    private static PyObject Made(List<PyObject> made, PyObject value)
    {
        made.Add(value);
        return value;
    }

    private static PyObject Of(object value) => value switch
    {
        int number => new PyInt(number),
        long number => new PyInt(number),
        bool flag => flag.ToPython(),
        string text => new PyString(text),
        _ => throw new ArgumentException($"A {value.GetType().Name} is not an argument a Python call is made with here.", nameof(value)),
    };
}
