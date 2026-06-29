using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Graph
{
    /// <summary>
    /// Per-tensor metadata, keyed by <see cref="FastTensorKey"/>. Built on demand by
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastTensorInfoProcessor"/>.
    /// </summary>
    public class FastTensorInfo
    {
        public FastTensorKey Key { get; set; }
        public DType DType { get; set; } = DType.Invalid;
        public DataStructure Structure { get; set; } = DataStructure.Tensor;
        public int? Rank { get; set; }
        public string? UniqueName { get; set; }
        public Function? ModuleFn { get; set; }
    }

    /// <summary>
    /// A flat, mutable representation of a graph. Nodes reference tensors by
    /// <see cref="FastTensorKey"/> only. Per-tensor metadata is built on demand via
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastTensorInfoProcessor.BuildTensorInfoLookup"/>.
    ///
    /// Use <see cref="FastComputationGraphConverter"/> for higher-level helpers
    /// (BuildNodes, BuildTensorMapping, LocalFunctions, FunctionsPostOrder).
    /// </summary>
    public partial class FastComputationGraph
    {
        /// <summary>
        /// All nodes in the graph, in the same topological order as the source
        /// <c>ComputationGraph.TopologicalOrderNodes</c>.
        /// </summary>
        public List<FastNode> Nodes { get; set; } = new();

        /// <summary>
        /// Tensor keys corresponding to <c>ComputationGraph.Inputs</c>, in order.
        /// </summary>
        public List<FastTensorKey> Inputs { get; set; } = new();

        /// <summary>
        /// Tensor keys corresponding to <c>ComputationGraph.Outputs</c>, in order.
        /// </summary>
        public List<FastTensorKey> Outputs { get; set; } = new();

        /// <summary>
        /// Original <see cref="Variable.UniqueName"/> for each entry in <see cref="Inputs"/>,
        /// captured by <see cref="FastComputationGraphConverter.ToFastGraph"/> and re-applied
        /// by <see cref="FastComputationGraphConverter.ToComputationGraph"/>. Preserves
        /// human-readable input names across the round-trip.
        /// </summary>
        public List<string?> InputUniqueNames { get; set; } = new();

        /// <summary>
        /// Original <see cref="Variable.UniqueName"/> for each entry in <see cref="Outputs"/>.
        /// Same purpose as <see cref="InputUniqueNames"/>.
        /// </summary>
        public List<string?> OutputUniqueNames { get; set; } = new();

        /// <summary>
        /// Optional overrides for the rank of each output, mirroring
        /// <c>ComputationGraph.OutputRankOverrides</c>.
        /// </summary>
        public int?[]? OutputRankOverrides { get; set; }

        /// <summary>
        /// Empty-graph constructor. Callers populate <see cref="Nodes"/>, <see cref="Inputs"/>,
        /// <see cref="Outputs"/>, etc. directly (used by
        /// <see cref="FastComputationGraphConverter"/> and the Fast processors).
        /// </summary>
        public FastComputationGraph() { }

        /// <summary>
        /// Builds a <see cref="FastComputationGraph"/> directly from an
        /// <see cref="Variable"/>-shaped graph. Walks back from <paramref name="outputs"/>
        /// to collect every reachable <see cref="Node"/>, sorts by
        /// <see cref="Node.OrderingHintNumber"/> (each Node's monotonic creation counter,
        /// which by construction places every producer before its consumers and every
        /// LoopAPI body node between its scope's OPEN and CLOSE), and lowers the result
        /// to <see cref="FastNode"/>s. The post-build
        /// <c>Debug.Assert(IsLinearOrderValid())</c> catches any case where that
        /// assumption doesn't hold.
        ///
        /// <para>Pass <paramref name="externalInputKeys"/> to remap stand-in
        /// <c>MODEL_TENSOR_INPUT</c> leaves to host-graph <see cref="FastTensorKey"/>s
        /// instead of fresh ones — used by the AUTO_GRAD splice so that gradient body
        /// nodes reference the existing forward-graph tensors directly. Stand-ins
        /// listed in this map are dropped from <see cref="Nodes"/>; their host keys
        /// take the corresponding slots in <see cref="Inputs"/>.</para>
        /// </summary>
        public FastComputationGraph(
            ImmutableArray<Variable> inputs,
            ImmutableArray<Variable> outputs,
            ImmutableArray<int?>? outputRankOverrides = null,
            IReadOnlyDictionary<Variable, FastTensorKey>? externalInputKeys = null)
        {
            Debug.Assert(inputs.All(x => x.OwningNode.IsModelInput));

            var tensors = Visitors.ReversePreOrder(ImmutableArray<Variable>.Empty, outputs).ToHashSet();
            var orderedNodes = tensors.Select(x => x.OwningNode)
                                      .Concat(inputs.Select(x => x.OwningNode))
                                      .NotNulls()
                                      .Distinct()
                                      .OrderBy(n => n.OrderingHintNumber)
                                      .ToImmutableArray();

            var ranks = outputRankOverrides?.ToArray() ?? outputs.Select(x => x.Rank).ToArray();
            FastComputationGraphConverter.PopulateFromNodes(
                this, orderedNodes, inputs, outputs, ranks,
                useSequentialIds: false, externalInputKeys: externalInputKeys);
            Debug.Assert(IsLinearOrderValid(), "IsLinearOrderValid()");
        }

        /// <summary>
        /// Finds a node by its <see cref="FastNodeKey"/>. Returns null if not found.
        /// </summary>
        public FastNode? FindNode(FastNodeKey key)
        {
            for (int i = 0; i < Nodes.Count; i++)
                if (Nodes[i].Key == key)
                    return Nodes[i];
            return null;
        }

        /// <summary>
        /// Produces a deep copy of this <see cref="FastComputationGraph"/>. The returned graph
        /// shares no mutable state with the source - nodes and input/output lists are all
        /// duplicated. Immutable/value-typed values (<see cref="OnnxCSharpAttributes"/>,
        /// <see cref="FastTensorKey"/>, <see cref="FastNodeKey"/>, <see cref="Function"/>) are shared.
        /// </summary>
        public FastComputationGraph Clone()
        {
            var copy = new FastComputationGraph
            {
                Inputs = new List<FastTensorKey>(this.Inputs),
                Outputs = new List<FastTensorKey>(this.Outputs),
                InputUniqueNames = new List<string?>(this.InputUniqueNames),
                OutputUniqueNames = new List<string?>(this.OutputUniqueNames),
                OutputRankOverrides = this.OutputRankOverrides is null
                    ? null
                    : (int?[])this.OutputRankOverrides.Clone(),
            };

            foreach (var node in this.Nodes)
                copy.Nodes.Add(CloneNode(node));

            System.Diagnostics.Debug.Assert(copy.IsLinearOrderValid(), "copy.IsLinearOrderValid()");
            return copy;
        }

        private static FastNode CloneNode(FastNode node)
        {
            var copy = new FastNode
            {
                Key = node.Key,
                OpCode = node.OpCode,
                Attributes = node.Attributes,
                FriendlyName = node.FriendlyName,
                StackTrace = node.StackTrace,
                GraphOpenNodeKey = node.GraphOpenNodeKey,
                IdentifierTemplate = node.IdentifierTemplate,
                TargetFunction = node.TargetFunction,
            };

            foreach (var kvp in node.FullInputs)
                copy.FullInputs[kvp.Key] = new List<FastTensorKey?>(kvp.Value);
            foreach (var kvp in node.FullOutputs)
                copy.FullOutputs[kvp.Key] = new List<FastTensorKey?>(kvp.Value);

            return copy;
        }

        /// <summary>
        /// Replaces this graph's state with the contents of <paramref name="other"/>. Used by
        /// the Fast* processors to "write back" a new graph onto an existing FastComputationGraph
        /// reference so that callers holding the reference see the mutation.
        /// </summary>
        public void CopyFrom(FastComputationGraph other)
        {
            if (other is null) throw new System.ArgumentNullException(nameof(other));

            this.Nodes = other.Nodes;
            this.Inputs = other.Inputs;
            this.Outputs = other.Outputs;
            this.OutputRankOverrides = other.OutputRankOverrides;
        }

        /// <summary>
        /// Computes the module / model signature strings for this graph. Operates on Fast
        /// keys: the input/output IValues consumed by
        /// <see cref="ModuleHelper.CreateFunctionSignatureString(Variable[], Variable[], Variable[], int?[])"/>
        /// are pulled out of the converter's <see cref="FastTensorKey"/> →
        /// <see cref="Variable"/> mapping.
        /// </summary>
        internal (string moduleSignature, string modelSignature) GetSignatureStrings()
        {
            var tensorMapping = FastComputationGraphConverter.BuildTensorMapping(this);

            var actualInputs = this.Inputs
                .Where(k => this.FindNode(k.FastNodeKey) is { } node && node.OpCode != InternalOpCodes.GENERIC_TYPE_INPUT)
                .ToList();

            var hyperParamCount = actualInputs
                .TakeWhile(k => this.FindNode(k.FastNodeKey)!.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) == InputType.Hyperparam)
                .Count();

            var hyperParams = actualInputs.Take(hyperParamCount).Select(k => tensorMapping[k]).ToArray();
            var inputs = actualInputs.Skip(hyperParamCount).Select(k => tensorMapping[k]).ToArray();
            var outputs = this.Outputs.Select(k => tensorMapping[k]).ToArray();

            return ModuleHelper.CreateFunctionSignatureString(hyperParams, inputs, outputs, this.OutputRankOverrides);
        }

        /// <summary>
        /// Returns true iff <see cref="Nodes"/> is in a valid linear order: every node's
        /// data-input producers and (for close nodes) the matching open appear at strictly
        /// smaller indices, every LOOP/IF OPEN has a matching CLOSE referencing the same
        /// key with no two scopes overlapping, and every scope strictly containing a
        /// producer also contains the consumer (or is the consumer's own scope when the
        /// consumer is a close).
        ///
        /// <para>The Fast pipeline maintains this as a graph invariant — every constructor
        /// and every pass that mutates <see cref="Nodes"/> is expected to <c>Debug.Assert</c>
        /// on this method on exit. There is intentionally no throwing variant; failures
        /// are caught only in Debug builds via the assertion.</para>
        /// </summary>
        public bool IsLinearOrderValid() => TryValidateLinearOrder(out _);

        private bool TryValidateLinearOrder(out string? error)
        {
            var n = Nodes.Count;
            var outputToNode = new System.Collections.Generic.Dictionary<FastTensorKey, int>(n * 2);
            var nodeKeyToIndex = new System.Collections.Generic.Dictionary<FastNodeKey, int>(n);
            for (int i = 0; i < n; i++)
            {
                var node = Nodes[i];
                nodeKeyToIndex[node.Key] = i;
                foreach (var kvp in node.FullOutputs)
                    foreach (var ok in kvp.Value)
                        if (ok is not null && !ok.Value.IsEmpty)
                            outputToNode[ok.Value] = i;
            }

            // Pass 1: scope-nesting check, collecting (openIdx, closeIdx) pairs for
            // every successfully matched scope along the way.
            var scopes = new System.Collections.Generic.List<(int OpenIdx, int CloseIdx)>();
            var openStack = new System.Collections.Generic.Stack<int>();
            for (int i = 0; i < n; i++)
            {
                var nd = Nodes[i];
                if (Shorokoo.Core.Factory.FastOpsetResolver.IsOpenOpCode(nd.OpCode))
                {
                    openStack.Push(i);
                }
                else if (Shorokoo.Core.Factory.FastOpsetResolver.IsCloseOpCode(nd.OpCode))
                {
                    if (openStack.Count == 0)
                    {
                        error = $"FastComputationGraph: unmatched close node at index {i} ({nd.OpCode}).";
                        return false;
                    }
                    var topIdx = openStack.Pop();
                    if (Nodes[topIdx].Key != nd.GraphOpenNodeKey)
                    {
                        error = $"FastComputationGraph: non-nesting open/close at index {i} ({nd.OpCode}): scopes overlap.";
                        return false;
                    }
                    scopes.Add((topIdx, i));
                }
            }
            if (openStack.Count > 0)
            {
                error = "FastComputationGraph: unmatched open node — scope is missing its close.";
                return false;
            }

            // Pass 2: topological-order + scope-visibility check. For each node N at pos i:
            //   - every data-input producer index must be strictly less than i,
            //   - every close node's GraphOpenNodeKey must resolve to a strictly earlier index,
            //   - and for every scope S strictly containing the producer, S must also contain
            //     N (or S must be N's own scope if N is the close of S). Visibility says that
            //     a node can only consume tensors from its own scope or an enclosing scope —
            //     not from a sibling scope or a scope nested deeper than its own. (Fast CG
            //     does allow a close node to consume from an outer enclosing scope; that's
            //     the only place we differ from ONNX's strict subgraph-output rule, and the
            //     check naturally permits it because outer scopes also contain the close.)
            var graphInputs = new System.Collections.Generic.HashSet<FastTensorKey>(this.Inputs);

            for (int i = 0; i < n; i++)
            {
                var node = Nodes[i];

                foreach (var kvp in node.FullInputs)
                {
                    foreach (var inputKey in kvp.Value)
                    {
                        if (inputKey is null || inputKey.Value.IsEmpty) continue;
                        if (graphInputs.Contains(inputKey.Value)) continue;
                        if (!outputToNode.TryGetValue(inputKey.Value, out var srcIdx)) continue;

                        if (srcIdx >= i)
                        {
                            error = $"FastComputationGraph: node #{i} ({node.OpCode}) consumes output of node #{srcIdx} which appears later in graph.Nodes.";
                            return false;
                        }

                        foreach (var s in scopes)
                        {
                            // Producer is strictly inside scope s?
                            if (!(s.OpenIdx < srcIdx && srcIdx < s.CloseIdx)) continue;
                            // Consumer also strictly inside scope s? then visible.
                            if (s.OpenIdx < i && i < s.CloseIdx) continue;
                            // Or s is the consumer's own scope (consumer is the close of s)?
                            if (s.CloseIdx == i && node.GraphOpenNodeKey is FastNodeKey ck &&
                                Nodes[s.OpenIdx].Key == ck) continue;

                            error = $"FastComputationGraph: node #{i} ({node.OpCode}) consumes output of node #{srcIdx} which sits inside scope ({Nodes[s.OpenIdx].OpCode} #{s.OpenIdx} … #{s.CloseIdx}) that doesn't enclose the consumer — sibling/inner-scope reference.";
                            return false;
                        }
                    }
                }

                if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty &&
                    nodeKeyToIndex.TryGetValue(openKey, out var openIdx) && openIdx >= i)
                {
                    error = $"FastComputationGraph: close node #{i} ({node.OpCode}) precedes its open node #{openIdx}.";
                    return false;
                }
            }

            error = null;
            return true;
        }
    }
}
