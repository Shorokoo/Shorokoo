using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo;

/// <summary>
/// The runtime values some managed contents have been built into, one per backend they were fed
/// to, held for the next feed and released together.
///
/// <para>It belongs to the <i>contents</i> rather than to the tensor naming them, which is what
/// makes it shareable: <see cref="TensorData.CloneSharing"/> hands the clone the same bytes, so
/// the clone must name the same materializations too. Given one cache each, a write through one
/// name would drop only that name's copies and the other would go on feeding the contents as they
/// stood before the write — a run reading data the tensor itself no longer reports.</para>
///
/// <para>Keyed by backend because contents may legitimately be fed to more than one — that is the
/// whole point of a program running two — and a value belongs to the runtime that made it.
/// Reference equality is the right comparison: two backends are the same backend exactly when
/// they are the same object.</para>
/// </summary>
internal sealed class MaterializedValues
{
    // Written from whichever thread is feeding a session, and the whole point of this is two
    // backends fed at once.
    private readonly object _gate = new();

    // Null until something is actually fed: most tensors never are -- a graph literal is read by
    // the builder and that is the end of it.
    private Dictionary<IShorokooInferenceBackend, IShorokooTensorValue>? _byBackend;

    /// <summary>Whether no runtime currently holds a copy of these contents. The test seam for
    /// the release paths: nothing else can observe that a value was freed rather than forgotten.
    /// </summary>
    internal bool IsEmpty
    {
        get { lock (_gate) return _byBackend is not { Count: > 0 }; }
    }

    /// <summary>
    /// The value <paramref name="backend"/>'s runtime holds for these contents, built by
    /// <paramref name="build"/> the first time that backend asks and kept for the next time.
    /// </summary>
    internal IShorokooTensorValue Get(
        IShorokooInferenceBackend backend,
        Func<IShorokooInferenceBackend, IShorokooTensorValue> build)
    {
        lock (_gate)
        {
            _byBackend ??= new Dictionary<IShorokooInferenceBackend, IShorokooTensorValue>(
                ReferenceEqualityComparer.Instance);

            if (_byBackend.TryGetValue(backend, out var existing)) return existing;

            var value = build(backend);
            _byBackend[backend] = value;
            return value;
        }
    }

    /// <summary>
    /// Drops every runtime's copy of these contents, freeing each one, because the contents are
    /// gone. Each copy was taken at the moment it was built, so contents mutated after being fed
    /// would otherwise keep feeding the old ones -- silently, since the tensor itself reads back
    /// the new ones.
    ///
    /// <para>This is the allocation's release action, so it runs when the last handle and the last
    /// lock on those contents have let go: a value handed to a session is a bare native pointer
    /// from that moment on, and freeing one while a run is reading it is a read of freed memory
    /// that nothing on this side could detect. Where the contents are merely being <i>written</i>
    /// rather than released, the copies come off the cache without being freed --
    /// <see cref="Retire"/> -- and the caller frees them once the readers are done.</para>
    /// </summary>
    internal void Invalidate()
    {
        Retire()?.Invoke();
    }

    /// <summary>
    /// Takes every runtime's copy off the cache without freeing it, and hands back the free as an
    /// action for the caller to run when it is safe to. Null when there was nothing cached.
    ///
    /// <para>The two halves come apart because they answer to different things. Retiring has to be
    /// immediate -- the next feed must rebuild from the contents as they now are -- while freeing
    /// has to wait for the last run still reading the old copies.</para>
    /// </summary>
    internal Action? Retire()
    {
        IShorokooTensorValue[] retired;
        lock (_gate)
        {
            if (_byBackend is null) return null;
            retired = [.. _byBackend.Values];
            _byBackend = null;
        }
        return () => { foreach (var value in retired) value.Dispose(); };
    }
}
