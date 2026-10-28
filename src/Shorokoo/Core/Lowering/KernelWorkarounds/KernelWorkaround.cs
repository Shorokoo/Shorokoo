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

    /// <summary>
    /// Whether <see cref="Applies"/> asks <see cref="WorkaroundSite.IsFromOutsideBody"/>. Working
    /// that out lays the graph's scopes out as the model is emitted, so a site of a workaround
    /// that does not ask answers false for every slot and the pass does no such work for it.
    /// </summary>
    public virtual bool ReadsBodies => false;

    /// <summary>
    /// Whether <see cref="Applies"/> asks <see cref="WorkaroundSite.ShapeOf"/>. Working that out
    /// runs the graph through Shorokoo's interpreter at the dimensions its inputs are stated with,
    /// so a site of a workaround that does not ask answers null for every slot and the build does
    /// no such work for it.
    /// </summary>
    public virtual bool ReadsShapes => false;

    /// <summary>
    /// Whether this workaround keeps each call it applies to on the host rather than rewriting it:
    /// the pass leaves the call as it stands, never asks <see cref="Rewrite"/>, and marks the call
    /// so the session runs it on the host (<see cref="Shorokoo.Core.Graph.FastNode.RunsOnHost"/>),
    /// where the runtime's host kernel computes it as the spec does. For a call a runtime places on
    /// a device whose kernel then fails it.
    /// </summary>
    public virtual bool KeepsOnHost => false;

    /// <summary>
    /// The op code of a producer this workaround looks through, or null. Where it is set, the pass
    /// offers the workaround only a call whose first input that producer gives, and offers it as a
    /// call over the producer's own first input: the site describes that value in slot 0, and the
    /// replacement reads it there. A producer nothing reads once the walk is done is dropped.
    /// </summary>
    public virtual string? LooksThrough => null;

    /// <summary>The name this workaround is known by in a plan cache key and a message.</summary>
    public virtual string Name => GetType().FullName ?? GetType().Name;
}
