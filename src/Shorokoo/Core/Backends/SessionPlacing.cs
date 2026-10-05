namespace Shorokoo.Core.Backends;

/// <summary>
/// Whether a session a backend builds on this thread now will place its runs' values in the memory
/// of the inputs they consume (<see cref="PlacementProof"/>): the framework says it will not, around
/// the build, where it would stop the session placing at once — a session built for one run, or
/// for a context that places nothing — so that a backend does none of the work placing asks of a
/// build: keeping the model to build from, writing out the graph it runs, copying its weights.
/// </summary>
internal static class SessionPlacing
{
    [ThreadStatic]
    private static bool t_suppressed;

    /// <summary>Whether the session being built on this thread will not place.</summary>
    internal static bool Suppressed => t_suppressed;

    /// <summary>Says, until disposed, that the session built on this thread will not place, where
    /// <paramref name="suppress"/>; says nothing where it is false.</summary>
    internal static Scope Suppress(bool suppress)
    {
        var outer = t_suppressed;
        t_suppressed = outer || suppress;
        return new Scope(outer);
    }

    /// <summary>What <see cref="Suppress"/> answers: restores what the thread said before.</summary>
    internal readonly struct Scope(bool outer) : IDisposable
    {
        public void Dispose() => t_suppressed = outer;
    }
}
