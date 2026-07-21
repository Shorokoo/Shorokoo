using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace Shorokoo.Core.Nodes.Processors.Training;

/// <summary>
/// Fast-native processor that replaces every trainable-parameter producer node
/// (<c>MODEL_PARAM</c>, <c>MODEL_PARAM_DATA</c>, <c>MODEL_PARAM_ID_REF</c> with
/// <c>shrk_is_trainable=true</c>) with a per-field <c>TENSOR_STRUCT_GETFIELD</c> consumer
/// of a single new <c>MODEL_TENSORSTRUCT_INPUT</c>, so the model's baked-in parameters
/// become an external TensorStruct input suitable for training.
///
/// <para>
/// Mutates <c>graph</c> in place: removes the original param-producer nodes,
/// inserts the new input + per-field GETFIELD nodes at the front of <see cref="InternalComputationGraph.Nodes"/>,
/// rewires every consumer (and graph output) that referenced an original param's
/// <see cref="FastTensorKey"/> to the matching GETFIELD output, and adds the struct input to
/// <see cref="InternalComputationGraph.Inputs"/>.
/// </para>
/// </summary>
internal static class FastReplaceTrainableParamsWithInputProcessor
{
    /// <summary>
    /// Result of <see cref="Process"/>: the discovered struct definition and the
    /// <see cref="FastTensorKey"/> of the new struct input plus its per-field GETFIELD outputs.
    /// </summary>
    public sealed class ProcessResult
    {
        public TensorStructDef TrainableParamStructDef { get; }
        public FastTensorKey TrainableParamStructInputKey { get; }
        public FastTensorKey[] ParamFieldKeys { get; }
        public ImmutableArray<FastDiscoveredParamInfo> ParamInfos { get; }

        internal ProcessResult(
            TensorStructDef def,
            FastTensorKey structInputKey,
            FastTensorKey[] paramFieldKeys,
            ImmutableArray<FastDiscoveredParamInfo> paramInfos)
        {
            TrainableParamStructDef = def;
            TrainableParamStructInputKey = structInputKey;
            ParamFieldKeys = paramFieldKeys;
            ParamInfos = paramInfos;
        }
    }

    public static ProcessResult Process(InternalComputationGraph graph)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));

        var paramInfos = FastDiscoverTrainableParamsProcessor.Process(graph);

        if (paramInfos.Length == 0)
            throw new InvalidOperationException(
                "No trainable parameters found in the computation graph. " +
                "Ensure the graph contains MODEL_PARAM or MODEL_PARAM_DATA nodes marked as trainable.");

        var structDef = FastBuildTrainableParamStructDefProcessor.Process(paramInfos, "TrainableParams");
        var structDType = DType.GetOrCreateForTensorStruct(structDef);

        var structInputNode = BuildTensorStructInputNode(structDType, "trainable_params");
        var structInputKey = new FastTensorKey(structInputNode.Key, 0);

        var fieldNodes = new List<FastNode>(paramInfos.Length);
        var fieldKeys = new FastTensorKey[paramInfos.Length];
        for (int i = 0; i < paramInfos.Length; i++)
        {
            var fieldDef = structDef.Fields[i];
            var node = BuildTensorStructGetFieldNode(structInputKey, fieldDef);
            fieldNodes.Add(node);
            fieldKeys[i] = new FastTensorKey(node.Key, 0);
        }

        var remap = new Dictionary<FastTensorKey, FastTensorKey>(paramInfos.Length);
        var paramNodeKeys = new HashSet<FastNodeKey>();
        for (int i = 0; i < paramInfos.Length; i++)
        {
            remap[paramInfos[i].OutputKey] = fieldKeys[i];
            paramNodeKeys.Add(paramInfos[i].Node.Key);
        }

        // Rewire every input slot of every remaining node, replacing keys that match
        // a former param output with the corresponding GETFIELD output.
        foreach (var node in graph.Nodes)
        {
            if (paramNodeKeys.Contains(node.Key)) continue;
            foreach (var (groupName, slots) in node.FullInputs)
            {
                for (int i = 0; i < slots.Count; i++)
                {
                    var k = slots[i];
                    if (k is FastTensorKey tk && remap.TryGetValue(tk, out var newKey))
                        slots[i] = newKey;
                }
            }
        }

        // Rewire graph outputs in case any output is itself a former param key.
        for (int i = 0; i < graph.Outputs.Count; i++)
            if (remap.TryGetValue(graph.Outputs[i], out var newKey))
                graph.Outputs[i] = newKey;

        graph.Nodes.RemoveAll(n => paramNodeKeys.Contains(n.Key));

        // Insert the new input and GETFIELD nodes at the front. The struct input
        // produces a graph input (no producer edges), and each GETFIELD only depends
        // on the struct input, so they're topologically valid at the head of Nodes.
        var prelude = new List<FastNode>(1 + fieldNodes.Count);
        prelude.Add(structInputNode);
        prelude.AddRange(fieldNodes);
        graph.Nodes.InsertRange(0, prelude);

        graph.Inputs.Add(structInputKey);
        graph.InputUniqueNames.Add("trainable_params");

        return new ProcessResult(structDef, structInputKey, fieldKeys, paramInfos);
    }

    private static FastNode BuildTensorStructInputNode(DType structDType, string defaultName)
    {
        var nodeKey = FastNodeKey.New();
        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.MODEL_TENSORSTRUCT_INPUT].AttributeDefs;
        var attrs = OnnxCSharpAttributes.FromCSharpVals(
            new Dictionary<string, object?>
            {
                [OnnxOpAttributeNames.AttrDtype] = structDType,
                [OnnxOpAttributeNames.ShrkAttrInputType] = (InputType?)null,
            },
            attrDefs);

        return new FastNode
        {
            Key = nodeKey,
            OpCode = InternalOpCodes.MODEL_TENSORSTRUCT_INPUT,
            Attributes = attrs,
            FriendlyName = defaultName,
            FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
        };
    }

    private static FastNode BuildTensorStructGetFieldNode(FastTensorKey structInputKey, TensorStructFieldDef field)
    {
        var nodeKey = FastNodeKey.New();
        var attrDefs = Definitions.NodeDefinitions[InternalOpCodes.TENSOR_STRUCT_GETFIELD].AttributeDefs;
        var attrs = OnnxCSharpAttributes.FromCSharpVals(
            new Dictionary<string, object?>
            {
                [OnnxOpAttributeNames.ShrkAttrFieldName] = field.Name,
                [OnnxOpAttributeNames.ShrkAttrDtype] = field.ElementType,
                [OnnxOpAttributeNames.ShrkAttrRank] = (long?)field.Rank,
                [OnnxOpAttributeNames.ShrkAttrStructure] = field.Structure,
            },
            attrDefs);

        return new FastNode
        {
            Key = nodeKey,
            OpCode = InternalOpCodes.TENSOR_STRUCT_GETFIELD,
            Attributes = attrs,
            FullInputs = { [""] = new List<FastTensorKey?> { structInputKey } },
            FullOutputs = { [""] = new List<FastTensorKey?> { new FastTensorKey(nodeKey, 0) } },
        };
    }
}
