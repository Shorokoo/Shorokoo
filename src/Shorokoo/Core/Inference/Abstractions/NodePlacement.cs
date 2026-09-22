namespace Shorokoo.Core.Inference.Abstractions;

/// <summary>
/// Where a session produces its outputs — the cheap half of the placement question, answered by
/// the session itself with no profiling and no extra run.
///
/// <para>On a GPU backend it is the one signal that costs nothing: a provider that cannot run
/// part of a graph leaves that part to the host, and an output the session reports in host memory
/// is the tail of a graph that ran there. <see cref="Mixed"/> says so outright.</para>
/// </summary>
public enum SessionOutputPlacement
{
    /// <summary>The backend does not report where its outputs are produced.</summary>
    Unknown,

    /// <summary>Every output is produced in host memory — a CPU session, or a device session
    /// whose whole graph fell back to the host.</summary>
    Host,

    /// <summary>Every output is produced in the execution provider's own memory.</summary>
    Device,

    /// <summary>Some outputs are produced on the device and some on the host, so part of this
    /// graph ran on the host and its results crossed the bus to get back.</summary>
    Mixed,
}

/// <summary>
/// One node of a session's graph, and the execution provider that ran it.
/// </summary>
/// <param name="Name">The node's name in the graph the session was built from, after the
/// runtime's own optimization passes have renamed and fused what they will.</param>
/// <param name="OpType">The operator the node runs.</param>
/// <param name="Provider">The execution provider that ran it, in the runtime's own spelling —
/// <c>CPUExecutionProvider</c>, <c>CUDAExecutionProvider</c>.</param>
/// <param name="NodeIndex">The node's index in the runtime's execution order.</param>
/// <param name="ActivationBytes">Bytes of activations the node read.</param>
/// <param name="ParameterBytes">Bytes of weights the node read.</param>
/// <param name="OutputBytes">Bytes the node produced.</param>
public readonly record struct NodeExecution(
    string Name,
    string OpType,
    string Provider,
    long NodeIndex,
    long ActivationBytes,
    long ParameterBytes,
    long OutputBytes);

/// <summary>What one execution provider ran, and what it moved.</summary>
/// <param name="Provider">The provider, in the runtime's own spelling.</param>
/// <param name="NodeCount">Nodes it ran.</param>
/// <param name="ActivationBytes">Their activations, summed.</param>
/// <param name="ParameterBytes">Their weights, summed.</param>
/// <param name="OutputBytes">Their outputs, summed.</param>
public readonly record struct ProviderShare(
    string Provider,
    int NodeCount,
    long ActivationBytes,
    long ParameterBytes,
    long OutputBytes);

/// <summary>
/// Which execution provider ran each node of a session's graph, with the bytes each one moved.
///
/// <para>This is the expensive half of the placement question and it is <b>off by default</b>:
/// it needs the session built with the runtime's profiler on, which costs every run it then
/// makes. Ask for it with
/// <see cref="Shorokoo.Runtime.ComputeContext.Diagnostics"/>'s
/// <see cref="DiagnosticSettings.TraceNodePlacement"/> and read it back from
/// <see cref="Shorokoo.Runtime.CompiledGraph.ReadNodePlacement"/>. For the free answer — did
/// <i>anything</i> run on the host — see <see cref="SessionOutputPlacement"/>, which needs
/// neither.</para>
///
/// <para><see cref="Providers"/> is the answer most callers want: more than one entry on a GPU
/// session is the list of nodes that fell back to the host, with the bytes they cost.</para>
/// </summary>
public sealed class NodePlacement
{
    /// <summary>Groups <paramref name="nodes"/> by the provider that ran each.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="nodes"/> is null.</exception>
    public NodePlacement(IReadOnlyList<NodeExecution> nodes)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        // Ordered here rather than left to whoever produced the list: a runtime writes its trace
        // as it runs and a graph run out of order is what this type exists to show, so the one
        // thing the caller cannot be asked to have done first is put it back in order. The sort is
        // stable, so nodes a runtime reports without an index keep the order it gave them.
        Nodes = [.. nodes.OrderBy(node => node.NodeIndex)];
        Providers =
        [
            .. Nodes.GroupBy(node => node.Provider, StringComparer.Ordinal)
                .Select(share => new ProviderShare(
                    share.Key,
                    share.Count(),
                    share.Sum(node => node.ActivationBytes),
                    share.Sum(node => node.ParameterBytes),
                    share.Sum(node => node.OutputBytes)))
                .OrderByDescending(share => share.NodeCount)
                .ThenBy(share => share.Provider, StringComparer.Ordinal),
        ];
    }

    /// <summary>Every node the session ran, in the runtime's execution order.</summary>
    public IReadOnlyList<NodeExecution> Nodes { get; }

    /// <summary>One entry per provider that ran anything, busiest first. A single entry is a
    /// graph that ran entirely on one provider.</summary>
    public IReadOnlyList<ProviderShare> Providers { get; }

    /// <summary>The nodes <paramref name="provider"/> ran, in execution order; empty when it ran
    /// none.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="provider"/> is null.</exception>
    public IReadOnlyList<NodeExecution> NodesOn(string provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        return [.. Nodes.Where(node => string.Equals(node.Provider, provider, StringComparison.Ordinal))];
    }
}
