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
    /// Use <see cref="InternalComputationGraphConverter"/> for higher-level helpers
    /// (BuildNodes, BuildTensorMapping, LocalFunctions, FunctionsPostOrder).
    /// </summary>
    public partial class InternalComputationGraph
    {
        /// <summary>
        /// All nodes in the graph, in the same topological order as the source
        /// <c>ComputationGraph.TopologicalOrderNodes</c>.
        /// </summary>
        public List<FastNode> Nodes { get; set; } = new();

        /// <summary>
        /// The graph's inputs, in order: the output key of each input node
        /// (<see cref="InternalOpCodes.IsModelInputOp"/>) in the prefix of <see cref="Nodes"/>
        /// those nodes form. The graph keeps no input list of its own — this is a read-only view,
        /// rebuilt on each access; add or remove an input by adding or removing its node
        /// (<see cref="InsertInput"/>). The prefix order is the input order: a generic
        /// <c>[Module]</c>'s type slots, then hyperparameters, then runtime inputs.
        /// </summary>
        public IReadOnlyList<FastTensorKey> Inputs
        {
            get
            {
                var count = InputCount;
                var keys = new FastTensorKey[count];
                for (int i = 0; i < count; i++)
                    keys[i] = InputKeyOf(Nodes[i]);
                return keys;
            }
        }

        /// <summary>The name of each input, in <see cref="Inputs"/> order, read off its node's
        /// <see cref="OnnxOpAttributeNames.ShrkAttrInputName"/> attribute (null where it has none).</summary>
        public IReadOnlyList<string?> InputNames
        {
            get
            {
                var count = InputCount;
                var names = new string?[count];
                for (int i = 0; i < count; i++)
                    names[i] = InputNameOf(Nodes[i]);
                return names;
            }
        }

        /// <summary>The input nodes: the prefix of <see cref="Nodes"/> they form.</summary>
        public IReadOnlyList<FastNode> InputNodes => Nodes.GetRange(0, InputCount);

        /// <summary>The number of inputs, which is also the index in <see cref="Nodes"/> at which the
        /// body — every node that is not an input, initializers included — starts.</summary>
        public int InputCount
        {
            get
            {
                int i = 0;
                while (i < Nodes.Count && InternalOpCodes.IsModelInputOp(Nodes[i].OpCode)) i++;
                return i;
            }
        }

        /// <summary>The tensor an input node produces: its one output.</summary>
        public static FastTensorKey InputKeyOf(FastNode inputNode)
        {
            Debug.Assert(InternalOpCodes.IsModelInputOp(inputNode.OpCode));
            foreach (var group in inputNode.FullOutputs.Values)
                foreach (var key in group)
                    if (key is { } k) return k;
            throw new System.InvalidOperationException($"Input node {inputNode.Key} ({inputNode.OpCode}) has no output.");
        }

        /// <summary>An input node's <see cref="OnnxOpAttributeNames.ShrkAttrInputName"/>, or null.</summary>
        public static string? InputNameOf(FastNode inputNode)
            => inputNode.Attributes.GetAttributeVals().TryGetValue(OnnxOpAttributeNames.ShrkAttrInputName, out var name)
                ? name as string
                : null;

        /// <summary>Sets an input node's <see cref="OnnxOpAttributeNames.ShrkAttrInputName"/>.</summary>
        public static void SetInputName(FastNode inputNode, string? name)
            => inputNode.Attributes = inputNode.Attributes.SetAttributes((OnnxOpAttributeNames.ShrkAttrInputName, name));

        /// <summary>Makes <paramref name="inputNode"/> the input at <paramref name="position"/>
        /// (0 ≤ position ≤ <see cref="InputCount"/>), shifting the later inputs along.</summary>
        public void InsertInput(int position, FastNode inputNode)
        {
            if (!InternalOpCodes.IsModelInputOp(inputNode.OpCode))
                throw new System.ArgumentException($"'{inputNode.OpCode}' is not an input op.", nameof(inputNode));
            if (position < 0 || position > InputCount)
                throw new System.ArgumentOutOfRangeException(nameof(position));
            Nodes.Insert(position, inputNode);
        }

        /// <summary>Makes <paramref name="inputNode"/> the last input.</summary>
        public void AddInput(FastNode inputNode) => InsertInput(InputCount, inputNode);

        /// <summary>Puts <paramref name="nodes"/> at the start of the body, right after the inputs.</summary>
        public void InsertAtBodyStart(IEnumerable<FastNode> nodes) => Nodes.InsertRange(InputCount, nodes);

        /// <summary>Puts <paramref name="node"/> at the start of the body, right after the inputs.</summary>
        public void InsertAtBodyStart(FastNode node) => Nodes.Insert(InputCount, node);

        /// <summary>
        /// Makes the graph's inputs exactly <paramref name="order"/>: the input nodes producing those
        /// keys, wherever they stand in <see cref="Nodes"/>, are moved to the front in that order.
        /// Any other input node is removed. Every other node keeps its relative order. For a pass
        /// that builds its new inputs piecemeal and settles their order at the end.
        /// </summary>
        public void SetInputs(IEnumerable<FastTensorKey> order)
        {
            var byKey = new Dictionary<FastTensorKey, FastNode>();
            foreach (var node in Nodes)
                if (InternalOpCodes.IsModelInputOp(node.OpCode))
                    byKey[InputKeyOf(node)] = node;
            var reordered = new List<FastNode>(Nodes.Count);
            foreach (var key in order)
            {
                if (!byKey.Remove(key, out var node))
                    throw new System.ArgumentException($"{key} is not produced by an input node of this graph, or is listed twice.", nameof(order));
                reordered.Add(node);
            }
            foreach (var node in Nodes)
                if (!InternalOpCodes.IsModelInputOp(node.OpCode))
                    reordered.Add(node);
            Nodes.Clear();
            Nodes.AddRange(reordered);
        }

        /// <summary>
        /// Tensor keys corresponding to <c>ComputationGraph.Outputs</c>, in order.
        /// </summary>
        public List<FastTensorKey> Outputs { get; set; } = new();

        /// <summary>
        /// Original <see cref="Variable.UniqueName"/> for each entry in <see cref="Outputs"/>,
        /// captured when the graph is built and re-applied when the Variable view is rebuilt.
        /// </summary>
        public List<string?> OutputUniqueNames { get; set; } = new();

        /// <summary>
        /// Optional overrides for the rank of each output, mirroring
        /// <c>ComputationGraph.OutputRankOverrides</c>.
        /// </summary>
        public int?[]? OutputRankOverrides { get; set; }

        /// <summary>
        /// Empty-graph constructor. Callers populate <see cref="Nodes"/> (input nodes first),
        /// <see cref="Outputs"/>, etc. directly (used by
        /// <see cref="InternalComputationGraphConverter"/> and the Fast processors).
        /// </summary>
        public InternalComputationGraph() { }

        /// <summary>
        /// Builds a <see cref="InternalComputationGraph"/> directly from an
        /// <see cref="Variable"/>-shaped graph. Walks back from <paramref name="outputs"/>
        /// to collect every reachable <see cref="Node"/>, puts the input nodes first in
        /// <paramref name="inputs"/> order, and sorts the rest by
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
        /// listed in this map are dropped from <see cref="Nodes"/>, so the result has no
        /// inputs of its own: it is a body fragment for the host to splice.</para>
        /// </summary>
        public InternalComputationGraph(
            ImmutableArray<Variable> inputs,
            ImmutableArray<Variable> outputs,
            ImmutableArray<int?>? outputRankOverrides = null,
            IReadOnlyDictionary<Variable, FastTensorKey>? externalInputKeys = null)
        {
            Debug.Assert(inputs.All(x => x.OwningNode.IsModelInput));

            // A loop marks every body value that cannot leave it invalid at termination, and
            // the node constructor refuses such a variable as an input. Returning one straight out
            // of the graph takes neither route, and what reaches the exporter is unusable — an
            // internal op no lowering can remove, or a node the loop's scope owns. Refuse the
            // shapes the loop names here, where what the user wrote is still nameable.
            foreach (var output in outputs)
                if (!output.IsValid)
                    throw new UnsupportedLoopVariableAssignmentException(
                        ErrorCodes.FW046, output.InvalidReason ?? Looper.BodyValueReadAfterLoopGuidance);

            var tensors = Visitors.ReversePreOrder(ImmutableArray<Variable>.Empty, outputs).ToHashSet();
            var inputNodes = inputs.Select(x => x.OwningNode).Distinct().ToList();
            var declared = inputNodes.ToHashSet();
            var bodyNodes = tensors.Select(x => x.OwningNode)
                                   .NotNulls()
                                   .Distinct()
                                   .Where(n => !declared.Contains(n))
                                   .OrderBy(n => n.OrderingHintNumber)
                                   .ToList();
            if (bodyNodes.FirstOrDefault(n => n.IsModelInput) is { } stray)
                throw new System.InvalidOperationException(
                    $"InternalComputationGraph: the graph reads input node '{stray.OpCode}' ({stray.Key}), " +
                    "which is not one of its declared inputs.");
            ImmutableArray<Node> orderedNodes = [.. inputNodes, .. bodyNodes];

            var ranks = outputRankOverrides?.ToArray() ?? outputs.Select(x => x.Rank).ToArray();
            InternalComputationGraphConverter.PopulateFromNodes(
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
        /// Produces a deep copy of this <see cref="InternalComputationGraph"/>. The returned graph
        /// shares no mutable state with the source - nodes and input/output lists are all
        /// duplicated. Immutable/value-typed values (<see cref="OnnxCSharpAttributes"/>,
        /// <see cref="FastTensorKey"/>, <see cref="FastNodeKey"/>, <see cref="Function"/>) are shared.
        /// </summary>
        public InternalComputationGraph Clone()
        {
            var copy = new InternalComputationGraph
            {
                Outputs = new List<FastTensorKey>(this.Outputs),
                OutputUniqueNames = new List<string?>(this.OutputUniqueNames),
                OutputRankOverrides = this.OutputRankOverrides is null
                    ? null
                    : (int?[])this.OutputRankOverrides.Clone(),
            };

            foreach (var node in this.Nodes)
                copy.Nodes.Add(CloneNode(node));

            System.Diagnostics.Debug.Assert(copy.TryValidateLinearOrder(out var cloneOrderError),
                "copy.IsLinearOrderValid(): " + cloneOrderError);
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
        /// Computes the module / model signature strings for this graph. Operates on Fast
        /// keys: the input/output IValues consumed by
        /// <see cref="ModuleHelper.CreateFunctionSignatureString(Variable[], Variable[], Variable[], int?[])"/>
        /// are pulled out of the converter's <see cref="FastTensorKey"/> →
        /// <see cref="Variable"/> mapping.
        /// </summary>
        internal (string moduleSignature, string modelSignature) GetSignatureStrings()
        {
            var tensorMapping = InternalComputationGraphConverter.BuildTensorMapping(this);

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
        /// Returns true iff <see cref="Nodes"/> is in a valid linear order: the input nodes form
        /// its prefix (no input node after the first body node), every node's
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

        internal bool TryValidateLinearOrder(out string? error)
        {
            var n = Nodes.Count;
            if (FindMisplacedInput() is int misplaced)
            {
                error = $"InternalComputationGraph: input node #{misplaced} ({Nodes[misplaced].OpCode}) follows body node #{misplaced - 1} ({Nodes[misplaced - 1].OpCode}) — input nodes must form a prefix of graph.Nodes.";
                return false;
            }
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
                        error = $"InternalComputationGraph: unmatched close node at index {i} ({nd.OpCode}).";
                        return false;
                    }
                    var topIdx = openStack.Pop();
                    if (Nodes[topIdx].Key != nd.GraphOpenNodeKey)
                    {
                        error = $"InternalComputationGraph: non-nesting open/close at index {i} ({nd.OpCode}): scopes overlap.";
                        return false;
                    }
                    scopes.Add((topIdx, i));
                }
            }
            if (openStack.Count > 0)
            {
                error = "InternalComputationGraph: unmatched open node — scope is missing its close.";
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
            for (int i = 0; i < n; i++)
            {
                var node = Nodes[i];

                foreach (var kvp in node.FullInputs)
                {
                    foreach (var inputKey in kvp.Value)
                    {
                        if (inputKey is null || inputKey.Value.IsEmpty) continue;
                        if (!outputToNode.TryGetValue(inputKey.Value, out var srcIdx)) continue;

                        if (srcIdx >= i)
                        {
                            error = $"InternalComputationGraph: node #{i} ({node.OpCode}) consumes output of node #{srcIdx} which appears later in graph.Nodes.";
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

                            error = $"InternalComputationGraph: node #{i} ({node.OpCode}) consumes output of node #{srcIdx} which sits inside scope ({Nodes[s.OpenIdx].OpCode} #{s.OpenIdx} … #{s.CloseIdx}) that doesn't enclose the consumer — sibling/inner-scope reference.";
                            return false;
                        }
                    }
                }

                if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty &&
                    nodeKeyToIndex.TryGetValue(openKey, out var openIdx) && openIdx >= i)
                {
                    error = $"InternalComputationGraph: close node #{i} ({node.OpCode}) precedes its open node #{openIdx}.";
                    return false;
                }
            }

            error = null;
            return true;
        }

        /// <summary>The index of the first input node that follows a body node, or null when the
        /// input nodes form a prefix of <see cref="Nodes"/>. One pass over the op codes.</summary>
        internal int? FindMisplacedInput()
        {
            int i = InputCount;
            for (; i < Nodes.Count; i++)
                if (InternalOpCodes.IsModelInputOp(Nodes[i].OpCode))
                    return i;
            return null;
        }
    }
}
