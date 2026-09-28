using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Splices a replacement built out of ordinary operator builders into a graph in place of one
    /// node: the machinery shared by the passes that rewrite one node at a time into several
    /// (<see cref="FastLowerRegisteredOps"/>, <see cref="FastApplyKernelWorkarounds"/>).
    ///
    /// <para><b>Building.</b> A replacement is built once per distinct shape of call, over
    /// stand-ins: one <see cref="InternalOp.RuntimeInput"/> per present input slot, at the dtype
    /// and rank the graph gives it. The built values become a small graph whose input prefix is the
    /// stand-ins and whose output suffix is the replacement's outputs; what lies between is the
    /// <see cref="Plan.Body"/>.</para>
    ///
    /// <para><b>Splicing.</b> Every spliced node is re-keyed, so the one plan can be spliced at
    /// every occurrence without two occurrences sharing a <see cref="FastNodeKey"/>; every
    /// reference to a body node — input and output tensor keys, connecting tensors, a close
    /// node's open-node key — is remapped to match, and every reference to a stand-in becomes the
    /// host node's own input. The body is inserted where the host stands, which keeps it inside
    /// the loop or branch scope the host sat in.</para>
    ///
    /// <para><b>In place.</b> Where the replacement's last node produces all its outputs, the host
    /// becomes that node: it keeps its <see cref="FastNode.Key"/>, its outputs, its friendly name and
    /// its stack trace, and takes everything else from the last node — operator, attributes, inputs,
    /// and the open node, function and identifier template that operator comes with. A replacement
    /// ending in an <c>If</c> so makes the host the <c>IF_CLOSE</c> paired with the <c>IF_OPEN</c>
    /// spliced before it, and one ending in a function call makes the host that call.</para>
    ///
    /// <para><b>Types.</b> A plan built with its tensor info carries the dtype, structure and rank of
    /// every value its body produces, and a splice given a lookup adds them, re-keyed, so a pass
    /// that splices keeps its lookup current without rebuilding it.</para>
    /// </summary>
    internal static class FastSplice
    {
        // Field and group separators for a plan cache key: control characters, so no op code,
        // attribute name, dtype name or attribute value can be mistaken for one.
        private const char Sep = '\u0001';
        private const char Group = '\u0002';

        /// <summary>
        /// A built replacement: its nodes in topological order, the tensor key standing in for
        /// each input slot of the host (null for an absent slot), and the key of each of the
        /// replacement's outputs; and, when it was built with them, the tensor info of every value
        /// the body produces.
        /// </summary>
        internal sealed record Plan(List<FastNode> Body, FastTensorKey?[] StandInKeyBySlot, FastTensorKey[] OutputKeys,
            IReadOnlyDictionary<FastTensorKey, FastTensorInfo>? TensorInfo = null)
        {
            /// <summary>The last node of the body.</summary>
            public FastNode Terminal => Body[^1];

            /// <summary>Whether <see cref="Terminal"/> produces every output of the replacement,
            /// in order and nothing else, so it can take the host node's place and keys.</summary>
            public bool TerminalProducesOutputs
            {
                get
                {
                    var outputs = Terminal.Outputs;
                    if (outputs.Count != OutputKeys.Length) return false;
                    for (int i = 0; i < outputs.Count; i++)
                        if (outputs[i] != OutputKeys[i]) return false;
                    return true;
                }
            }
        }

        /// <summary>One stand-in per described slot, null for an absent one.</summary>
        public static Variable?[] StandIns((DType DType, int? Rank)?[] inputs)
        {
            var standIns = new Variable?[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                if (inputs[i] is { } descriptor)
                    standIns[i] = InternalOp.RuntimeInput(descriptor.DType, descriptor.Rank);
            return standIns;
        }

        /// <summary>
        /// The plan of the replacement whose outputs are <paramref name="outputs"/>, built over
        /// <paramref name="standIns"/>, with the tensor info of its body when
        /// <paramref name="typed"/>; null when it holds no node at all.
        /// </summary>
        public static Plan? Build(Variable?[] standIns, Variable[] outputs, bool typed = false)
        {
            ImmutableArray<Variable> presentStandIns = [.. standIns.Where(x => x is not null).Select(x => x!)];
            var built = new InternalComputationGraph(presentStandIns, [.. outputs]);

            var standInKeyBySlot = new FastTensorKey?[standIns.Length];
            var builtInputs = built.Inputs;
            for (int i = 0, present = 0; i < standIns.Length; i++)
                if (standIns[i] is not null) standInKeyBySlot[i] = builtInputs[present++];

            List<FastNode> body = [.. built.Nodes.Take(built.BodyEnd).Skip(built.InputCount)];
            if (body.Count == 0) return null;
            if (!typed) return new Plan(body, standInKeyBySlot, [.. built.Outputs]);

            var bodyKeys = body.Select(n => n.Key).ToHashSet();
            var info = FastTensorInfoProcessor.BuildTensorInfoLookup(built)
                .Where(p => bodyKeys.Contains(p.Key.FastNodeKey))
                .ToDictionary(p => p.Key, p => p.Value);
            return new Plan(body, standInKeyBySlot, [.. built.Outputs], info);
        }

        /// <summary>
        /// Inserts <paramref name="plan"/>'s body for <paramref name="host"/> into
        /// <paramref name="newNodes"/> and mutates <paramref name="host"/> into the terminal node,
        /// keeping the host's key and output keys. Requires
        /// <see cref="Plan.TerminalProducesOutputs"/>. Each minted key is added to
        /// <paramref name="minted"/>, and the tensor info of each value the body produces to
        /// <paramref name="tensorInfo"/>, when it is given; the host's own outputs keep what
        /// <paramref name="tensorInfo"/> already says of them.
        /// </summary>
        public static void SpliceInPlace(FastNode host, Plan plan, List<FastNode> newNodes, ISet<FastNodeKey>? minted = null,
            IDictionary<FastTensorKey, FastTensorInfo>? tensorInfo = null)
        {
            var (tensorMap, nodeMap) = Maps(host, plan, host.Key, minted);
            AddTensorInfo(plan, tensorMap, nodeMap, tensorInfo);
            for (int i = 0; i < plan.Body.Count - 1; i++)
                newNodes.Add(Copy(plan.Body[i], tensorMap, nodeMap));

            var terminal = plan.Terminal;
            host.OpCode = terminal.OpCode;
            host.Attributes = terminal.Attributes;
            host.FullInputs = terminal.FullInputs.ToDictionary(g => g.Key, g => Remap(g.Value, tensorMap, nodeMap));
            host.GraphOpenNodeKey = RemapNode(terminal.GraphOpenNodeKey, nodeMap);
            host.TargetFunction = terminal.TargetFunction;
            host.IdentifierTemplate = terminal.IdentifierTemplate;
            newNodes.Add(host);
        }

        /// <summary>
        /// Inserts the whole of <paramref name="plan"/>'s body for <paramref name="host"/> into
        /// <paramref name="newNodes"/>, re-keyed, leaving <paramref name="host"/> out of it, and
        /// returns the key each of the replacement's outputs carries in the graph. Each minted key
        /// is added to <paramref name="minted"/>, and the tensor info of each value the body
        /// produces to <paramref name="tensorInfo"/>, when it is given.
        /// </summary>
        public static FastTensorKey[] SpliceBeside(FastNode host, Plan plan, List<FastNode> newNodes, ISet<FastNodeKey>? minted = null,
            IDictionary<FastTensorKey, FastTensorInfo>? tensorInfo = null)
        {
            var (tensorMap, nodeMap) = Maps(host, plan, null, minted);
            AddTensorInfo(plan, tensorMap, nodeMap, tensorInfo);
            foreach (var built in plan.Body)
                newNodes.Add(Copy(built, tensorMap, nodeMap));
            return [.. plan.OutputKeys.Select(k => RemapKey(k, tensorMap, nodeMap))];
        }

        private static (Dictionary<FastTensorKey, FastTensorKey> Tensors, Dictionary<FastNodeKey, FastNodeKey> Nodes) Maps(
            FastNode host, Plan plan, FastNodeKey? terminalKey, ISet<FastNodeKey>? minted)
        {
            var hostInputs = host.Inputs;
            var tensorMap = new Dictionary<FastTensorKey, FastTensorKey>();
            for (int i = 0; i < plan.StandInKeyBySlot.Length && i < hostInputs.Count; i++)
                if (plan.StandInKeyBySlot[i] is { } standIn && hostInputs[i] is { } input)
                    tensorMap[standIn] = input;

            var nodeMap = new Dictionary<FastNodeKey, FastNodeKey>(plan.Body.Count);
            for (int i = 0; i < plan.Body.Count; i++)
            {
                if (i == plan.Body.Count - 1 && terminalKey is { } kept)
                {
                    nodeMap[plan.Body[i].Key] = kept;
                    continue;
                }
                var fresh = FastNodeKey.New();
                nodeMap[plan.Body[i].Key] = fresh;
                minted?.Add(fresh);
            }
            return (tensorMap, nodeMap);
        }

        private static void AddTensorInfo(Plan plan, Dictionary<FastTensorKey, FastTensorKey> tensorMap,
            Dictionary<FastNodeKey, FastNodeKey> nodeMap, IDictionary<FastTensorKey, FastTensorInfo>? tensorInfo)
        {
            if (tensorInfo is null || plan.TensorInfo is null) return;
            foreach (var (key, info) in plan.TensorInfo)
            {
                var spliced = RemapKey(key, tensorMap, nodeMap);
                tensorInfo.TryAdd(spliced, new FastTensorInfo
                {
                    Key = spliced,
                    DType = info.DType,
                    Structure = info.Structure,
                    Rank = info.Rank,
                    UniqueName = info.UniqueName,
                    ModuleFn = info.ModuleFn,
                });
            }
        }

        private static FastNode Copy(
            FastNode built, Dictionary<FastTensorKey, FastTensorKey> tensorMap, Dictionary<FastNodeKey, FastNodeKey> nodeMap)
        {
            var copy = new FastNode
            {
                Key = nodeMap[built.Key],
                OpCode = built.OpCode,
                Attributes = built.Attributes,
                FriendlyName = built.FriendlyName,
                StackTrace = built.StackTrace,
                GraphOpenNodeKey = RemapNode(built.GraphOpenNodeKey, nodeMap),
                IdentifierTemplate = built.IdentifierTemplate,
                TargetFunction = built.TargetFunction,
            };
            foreach (var group in built.FullInputs) copy.FullInputs[group.Key] = Remap(group.Value, tensorMap, nodeMap);
            foreach (var group in built.FullOutputs) copy.FullOutputs[group.Key] = Remap(group.Value, tensorMap, nodeMap);
            return copy;
        }

        private static FastNodeKey? RemapNode(FastNodeKey? key, Dictionary<FastNodeKey, FastNodeKey> nodeMap)
            => key is { } k && nodeMap.TryGetValue(k, out var mapped) ? mapped : key;

        private static FastTensorKey RemapKey(
            FastTensorKey key, Dictionary<FastTensorKey, FastTensorKey> tensorMap, Dictionary<FastNodeKey, FastNodeKey> nodeMap)
        {
            if (tensorMap.TryGetValue(key, out var mapped)) return mapped;
            if (!key.IsEmpty && nodeMap.TryGetValue(key.FastNodeKey, out var node)) return new FastTensorKey(node, key.OutputIndex);
            return key;
        }

        private static List<FastTensorKey?> Remap(
            List<FastTensorKey?> keys, Dictionary<FastTensorKey, FastTensorKey> tensorMap, Dictionary<FastNodeKey, FastNodeKey> nodeMap)
        {
            var remapped = new List<FastTensorKey?>(keys.Count);
            foreach (var key in keys)
                remapped.Add(key is { } k ? RemapKey(k, tensorMap, nodeMap) : key);
            return remapped;
        }

        /// <summary>
        /// What a plan may be reused for: <paramref name="identity"/> (what builds it), the dtype
        /// and rank of each input (and which inputs are absent), how many outputs the operator
        /// declares, and the attribute VALUES — a builder is ordinary C# and may branch on them.
        /// Null when some attribute has no stable rendering — a tensor or a subgraph — since a key
        /// that ignored it would serve one node's plan to another that differs only there.
        /// </summary>
        public static StringBuilder? TryBuildKey(
            string identity,
            (DType DType, int? Rank)?[] inputs,
            OnnxCSharpAttributes attributes,
            int declaredOutputs)
        {
            // The slot and output counts are part of the key, so a cached plan's stand-in-per-slot
            // table always lines up with the node it is reused for.
            var key = new StringBuilder(identity)
                .Append(Sep).Append(inputs.Length)
                .Append(Sep).Append(declaredOutputs);

            foreach (var input in inputs)
            {
                key.Append(Group);
                if (input is not { } descriptor) { key.Append('~'); continue; }
                key.Append(descriptor.DType).Append(Sep).Append(descriptor.Rank ?? -1);
            }

            foreach (var (name, value) in attributes.GetAttributeVals().OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                key.Append(Group).Append(name).Append(Sep);
                if (!TryAppendValue(key, value)) return null;
            }
            return key;
        }

        /// <summary>Appends a stable rendering of <paramref name="value"/>; false when it has
        /// none.</summary>
        public static bool TryAppendValue(StringBuilder key, object? value)
        {
            switch (value)
            {
                case null: key.Append('~'); return true;
                case bool b: key.Append(b ? 'T' : 'F'); return true;
                case long l: key.Append(l); return true;
                case int i: key.Append(i); return true;
                // By bits, so the rendering is exact and carries no culture or rounding of its own.
                case float f: key.Append(BitConverter.SingleToInt32Bits(f)); return true;
                case double d: key.Append(BitConverter.DoubleToInt64Bits(d)); return true;
                // Length-prefixed: a string is the one value that could otherwise hold a separator.
                case string s: key.Append(s.Length).Append('"').Append(s); return true;
                case DType t: key.Append(t); return true;
                case Enum e: key.Append(e.GetType().FullName).Append('.').Append(e); return true;
                case Array a:
                    key.Append('[');
                    foreach (var item in a)
                    {
                        if (!TryAppendValue(key, item)) return false;
                        key.Append(Sep);
                    }
                    key.Append(']');
                    return true;
                default: return false;
            }
        }

        /// <summary>Appends <paramref name="value"/> to a key as a field of its own.</summary>
        public static StringBuilder AppendField(this StringBuilder key, string value)
            => key.Append(Group).Append(value.Length).Append('"').Append(value);
    }
}
