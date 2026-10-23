using System.Runtime.CompilerServices;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;

namespace Shorokoo.Core.AutoDiffCheckpointing;

/// <summary>
/// The memory-aware pass's objective with the peak the backend a step runs on holds, as that
/// backend's own model of a run says (<see cref="Shorokoo.Core.Backends.IShorokooBackend"/>'s
/// modelled run peak), in place of the peak the pass's evaluator charges: the compute the
/// evaluator models, the memory the backend's model. It is normalized to the graph the pass was
/// handed — its modelled compute and the backend's peak of it — as the pass's own objective is to
/// the pass's figures, so the two weigh a trade alike. The backend is asked about each graph once,
/// however many of the pass's candidates are that graph (<see cref="Content"/>), and about the graph
/// the pass was handed before any other: where its model cannot tell that graph, no candidate can
/// be scored, so none is asked about. Where its model is quick to ask (<see cref="WeighsPlateaus"/>),
/// the rematerializer asks it about the trials a plateau of its own figures leaves; otherwise only
/// the steps each strategy takes and the strategies' ends are weighed by it.
/// </summary>
internal sealed class BackendJudge
{
    private readonly Func<InternalComputationGraph, long?> _peakOf;
    private readonly Dictionary<InternalComputationGraph, long?> _asked = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Content, long?> _peaks = [];
    private readonly double _computeWeight;
    private readonly double _memoryWeight;
    private readonly InternalComputationGraph _handed;
    private readonly double _handedComputeTime;
    private ComputeMemoryObjective? _objective;

    /// <summary>Whether the rematerializer weighs a plateau's trials by the backend's model.</summary>
    internal bool WeighsPlateaus { get; }

    /// <param name="peakOf">The backend's peak of a run of a graph, or null where its model cannot
    /// tell.</param>
    /// <param name="computeWeight">The objective's weight on compute.</param>
    /// <param name="memoryWeight">The objective's weight on memory.</param>
    /// <param name="handed">The graph the pass was handed, which the objective is normalized to.</param>
    /// <param name="handedEval">The evaluator's figures for <paramref name="handed"/>.</param>
    /// <param name="weighsPlateaus">Whether the backend's model is quick enough to weigh a plateau's
    /// trials by.</param>
    internal BackendJudge(Func<InternalComputationGraph, long?> peakOf, double computeWeight, double memoryWeight,
        InternalComputationGraph handed, GraphEvaluationResult handedEval, bool weighsPlateaus = true)
    {
        WeighsPlateaus = weighsPlateaus;
        _peakOf = peakOf;
        _computeWeight = computeWeight;
        _memoryWeight = memoryWeight;
        _handed = handed;
        _handedComputeTime = handedEval.TotalComputeTime;
    }

    /// <summary>The backend's peak of a run of <paramref name="graph"/>, or null where its model
    /// cannot tell.</summary>
    internal long? Peak(InternalComputationGraph graph)
    {
        if (_asked.TryGetValue(graph, out var peak)) return peak;
        var content = new Content(graph);
        if (!_peaks.TryGetValue(content, out peak))
            _peaks[content] = peak = _peakOf(graph);
        return _asked[graph] = peak;
    }

    /// <summary>The objective on <paramref name="graph"/>, evaluated as <paramref name="eval"/>,
    /// with the backend's peak of it, and that peak; null where the backend's model cannot tell the
    /// peak of the graph or of the graph the pass was handed.</summary>
    internal (double Score, long Peak)? Judge(InternalComputationGraph graph, GraphEvaluationResult eval)
    {
        if (_objective is null)
        {
            if (Peak(_handed) is not { } handedPeak) return null;
            _objective = new ComputeMemoryObjective(_computeWeight, _memoryWeight,
                new GraphEvaluationResult { TotalComputeTime = _handedComputeTime, PeakMemoryBytes = handedPeak, NodeDetails = [] });
        }
        if (Peak(graph) is not { } peak) return null;
        return (_objective.Value.Score(new GraphEvaluationResult { TotalComputeTime = eval.TotalComputeTime, PeakMemoryBytes = peak, NodeDetails = [] }), peak);
    }

    /// <summary>
    /// A graph the backend was asked about, as its nodes stand, every field of each node, with each
    /// node and tensor key standing for the order it first appears in: two are equal only where each
    /// node of one is the other's node in the same place, but for the keys it was minted, so the two
    /// lower to the same model, which names its values by their place. The rematerializer mints fresh
    /// keys for each copy it makes, so the same recomputation reached by two strategies is two graphs
    /// that differ in their keys alone. A field held by reference — the attributes, the call stack,
    /// the function a node targets — is compared by reference, which tells apart the odd pair of
    /// graphs whose nodes are equal but were built apart; the graphs the pass's strategies make of
    /// one another share them. The pass changes no graph it has made, so the graph is kept, not a
    /// copy of its fields.
    /// </summary>
    private sealed class Content : IEquatable<Content>
    {
        private readonly InternalComputationGraph _graph;
        private readonly int _hash;

        internal Content(InternalComputationGraph graph)
        {
            _graph = graph;
            var hash = new HashCode();
            foreach (var field in Fields(graph))
                hash.Add(field is null or string or ValueType ? field?.GetHashCode() ?? 0 : RuntimeHelpers.GetHashCode(field));
            _hash = hash.ToHashCode();
        }

        private static IEnumerable<object?> Fields(InternalComputationGraph graph)
        {
            var nodes = new Dictionary<FastNodeKey, int>();
            var tensors = new Dictionary<FastTensorKey, int>();
            static int Place<TKey>(Dictionary<TKey, int> places, TKey key) where TKey : notnull
                => places.TryGetValue(key, out var place) ? place : places[key] = places.Count;
            foreach (var node in graph.Nodes)
            {
                yield return Place(nodes, node.Key);
                yield return node.OpCode;
                yield return node.Attributes;
                yield return node.FriendlyName;
                yield return node.CallStack;
                yield return node.GraphOpenNodeKey is { } open ? Place(nodes, open) : null;
                yield return node.IdentifierTemplate;
                yield return node.TargetFunction;
                foreach (var ports in (Dictionary<string, List<FastTensorKey?>>[])[node.FullInputs, node.FullOutputs])
                {
                    yield return ports.Count;
                    foreach (var (name, keys) in ports)
                    {
                        yield return name;
                        yield return keys.Count;
                        foreach (var key in keys) yield return key is { } tensor ? Place(tensors, tensor) : null;
                    }
                }
            }
        }

        public bool Equals(Content? other)
        {
            if (other is null || other._hash != _hash || other._graph.Nodes.Count != _graph.Nodes.Count) return false;
            if (ReferenceEquals(other._graph, _graph)) return true;
            using var theirs = Fields(other._graph).GetEnumerator();
            foreach (var a in Fields(_graph))
            {
                if (!theirs.MoveNext()) return false;
                var b = theirs.Current;
                if (a is null or string or ValueType ? !Equals(a, b) : !ReferenceEquals(a, b)) return false;
            }
            return !theirs.MoveNext();
        }

        public override bool Equals(object? obj) => Equals(obj as Content);

        public override int GetHashCode() => _hash;
    }
}
