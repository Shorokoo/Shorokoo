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
/// Reference equality is the right comparison: two factories are the same backend exactly when
/// they are the same object.</para>
/// </summary>
internal sealed class MaterializedValues
{
    // Written from whichever thread is feeding a session, and the whole point of this is two
    // backends fed at once.
    private readonly object _gate = new();

    // Null until something is actually fed: most tensors never are -- a graph literal is read by
    // the builder and that is the end of it.
    private Dictionary<IShorokooInferenceSessionFactory, IShorokooTensorValue>? _byBackend;

    /// <summary>
    /// The value <paramref name="factory"/>'s runtime holds for these contents, built by
    /// <paramref name="build"/> the first time that backend asks and kept for the next time.
    /// </summary>
    internal IShorokooTensorValue Get(
        IShorokooInferenceSessionFactory factory,
        Func<IShorokooInferenceSessionFactory, IShorokooTensorValue> build)
    {
        lock (_gate)
        {
            _byBackend ??= new Dictionary<IShorokooInferenceSessionFactory, IShorokooTensorValue>(
                ReferenceEqualityComparer.Instance);

            if (_byBackend.TryGetValue(factory, out var existing)) return existing;

            var value = build(factory);
            _byBackend[factory] = value;
            return value;
        }
    }

    /// <summary>
    /// Drops every runtime's copy of these contents, because the contents are about to change or
    /// have just been handed out for writing. Each copy was taken at the moment it was built, so
    /// contents mutated after being fed would otherwise keep feeding the old ones -- silently,
    /// since the tensor itself reads back the new ones.
    /// </summary>
    internal void Invalidate()
    {
        lock (_gate)
        {
            if (_byBackend is null) return;
            foreach (var value in _byBackend.Values) value.Dispose();
            _byBackend = null;
        }
    }
}
