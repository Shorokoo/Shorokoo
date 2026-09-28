using Shorokoo.Core.Nodes;

namespace Shorokoo.Core.Lowering.KernelWorkarounds;

/// <summary>
/// One rewrite around a backend's kernel: a call of an operator the kernel computes otherwise than
/// the ONNX spec says, replaced by an equivalent call the backend computes as the spec does.
///
/// <para>A workaround is applied by <c>FastApplyKernelWorkarounds</c> to the model built for a
/// backend's session only, as part of the <see cref="KernelWorkaroundSet"/> the backend names
/// (<see cref="Shorokoo.Core.Backends.IShorokooBackend.KernelWorkaroundSet"/>). The graph, an
/// exported file, generated C# and <c>.srk</c> keep the operator as written.</para>
///
/// <para><b>Contract.</b> <see cref="Applies"/> decides from what the <see cref="WorkaroundSite"/>
/// tells about one call — its attributes, the dtype and rank of each input and output, which
/// outputs are read, the values of constant inputs. What only the run knows, an extent say, is
/// decided in the graph the rewrite builds, with an <c>If</c> or a branchless form.
/// <see cref="Rewrite"/> builds the replacement with the ordinary operator builders over
/// <c>inputs</c>, one stand-in per input slot (null for an absent optional input), and returns one
/// value per output slot of the call: null only for a slot the call leaves absent. A returned value
/// must equal the operator's output wherever the spec defines one. The rewrite may build the
/// operator it replaces; the pass never revisits what a workaround spliced in, so it is not
/// rewritten again by the same workaround, and later workarounds of the set do see it.</para>
///
/// <para>A workaround holds no state: one instance serves every graph on every thread.</para>
/// </summary>
internal abstract class KernelWorkaround
{
    /// <summary>The op codes whose calls this workaround looks at.</summary>
    public abstract IReadOnlySet<string> OpCodes { get; }

    /// <summary>Whether the call at <paramref name="site"/> is one this workaround rewrites.</summary>
    public abstract bool Applies(WorkaroundSite site);

    /// <summary>
    /// The replacement for the call at <paramref name="site"/>, built over
    /// <paramref name="inputs"/>: one value per output slot.
    /// </summary>
    public abstract Variable?[] Rewrite(WorkaroundSite site, Variable?[] inputs);

    /// <summary>The name this workaround is known by in a plan cache key and a message.</summary>
    public virtual string Name => GetType().FullName ?? GetType().Name;
}
