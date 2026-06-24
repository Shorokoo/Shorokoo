using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.AutoGrad;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Per-op handlers for SEQUENCE_* nodes whose element / input dtype is
    /// <see cref="DType.Model"/>. Each handler translates a single op into parallel
    /// per-field SEQUENCE_* nodes on the FastComputationGraph, keeping the unpack
    /// state in <see cref="FastModelStructContext"/> in sync. Works for any hp field
    /// layout (Tensor / Optional / Sequence) — the per-field SEQUENCE_* we emit just
    /// carry whichever tensor keys the Model struct stores for that field.
    /// </summary>
    internal static class FastModelSequenceOpHandlers
    {
        /// <summary>
        /// SEQUENCE_EMPTY producing a Sequence&lt;Model&gt;. Emits one empty sequence per
        /// struct field (iterationIndices + hyperparams + modelId), registers the bundle in
        /// <see cref="FastModelStructContext.UnpackedStructSequences"/> and retires the
        /// original node.
        /// </summary>
        public static void HandleSequenceEmpty(FastNode fastNode, FastModelStructContext ctx)
        {
            var outputKey = fastNode.Outputs[0]!.Value;
            var targetFn = fastNode.TargetFunction
                ?? throw new System.InvalidOperationException(
                    "FastUnpackModelStruct: SEQUENCE_EMPTY with Model dtype is missing TargetFunction.");

            var fieldSeqKeys = new List<FastTensorKey>(targetFn.HyperparamInputs.Length + 2);

            // iterationIndices sequence at the front.
            fieldSeqKeys.Add(EmitSequenceEmpty(ctx, DType.Int64));

            foreach (var hp in targetFn.HyperparamInputs)
                fieldSeqKeys.Add(EmitSequenceEmpty(ctx, hp.DType));

            // modelId sequence at the back.
            fieldSeqKeys.Add(EmitSequenceEmpty(ctx, DType.Int64));

            ctx.UnpackedStructSequences[outputKey] = fieldSeqKeys;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        /// <summary>
        /// SEQUENCE_CONSTRUCT whose variadic inputs are unpacked Model structs. Transposes
        /// the per-element field lists and emits one SEQUENCE_CONSTRUCT per field.
        /// </summary>
        public static void HandleSequenceConstruct(FastNode fastNode, FastModelStructContext ctx)
        {
            var outputKey = fastNode.Outputs[0]!.Value;
            var inputs = fastNode.Inputs;

            var elementFields = new List<List<FastTensorKey?>>(inputs.Count);
            foreach (var inputKey in inputs)
            {
                Debug.Assert(inputKey is not null);
                var resolved = ctx.ResolveModelKey(inputKey!.Value);
                Debug.Assert(ctx.UnpackedStructs.ContainsKey(resolved),
                    "SEQUENCE_CONSTRUCT input is not a known unpacked Model struct.");
                elementFields.Add(ctx.UnpackedStructs[resolved]);
            }

            int numFields = elementFields[0].Count;
            Debug.Assert(elementFields.All(e => e.Count == numFields),
                "SEQUENCE_CONSTRUCT element field counts must match.");

            var fieldSeqKeys = new List<FastTensorKey>(numFields);
            for (int fieldIdx = 0; fieldIdx < numFields; fieldIdx++)
            {
                var elements = elementFields.Select(e => e[fieldIdx]).ToList();
                var seqKey = FastNodeKey.New();
                ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceConstruct(seqKey, elements));
                fieldSeqKeys.Add(new FastTensorKey(seqKey, 0));
            }

            ctx.UnpackedStructSequences[outputKey] = fieldSeqKeys;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        /// <summary>
        /// SEQUENCE_AT on a Sequence&lt;Model&gt;. Emits one SEQUENCE_AT per field sequence
        /// and records the resulting per-field tensors as the unpacked struct for the
        /// original output key.
        /// </summary>
        public static void HandleSequenceAt(FastNode fastNode, FastModelStructContext ctx)
        {
            var inputSeqKey = fastNode.Inputs[0]!.Value;
            var positionKey = fastNode.Inputs[1]!.Value;
            var outputKey = fastNode.Outputs[0]!.Value;

            var resolved = ctx.ResolveModelKey(inputSeqKey);
            Debug.Assert(ctx.UnpackedStructSequences.ContainsKey(resolved),
                "SEQUENCE_AT input is not a known unpacked Model sequence.");

            var fieldSeqKeys = ctx.UnpackedStructSequences[resolved];
            var fieldKeys = new List<FastTensorKey?>(fieldSeqKeys.Count);
            foreach (var fieldSeqKey in fieldSeqKeys)
            {
                var atKey = FastNodeKey.New();
                ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceAt(atKey, fieldSeqKey, positionKey));
                fieldKeys.Add(new FastTensorKey(atKey, 0));
            }

            ctx.UnpackedStructs[outputKey] = fieldKeys;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        /// <summary>
        /// SEQUENCE_ERASE on a Sequence&lt;Model&gt;. Emits one SEQUENCE_ERASE per field.
        /// </summary>
        public static void HandleSequenceErase(FastNode fastNode, FastModelStructContext ctx)
        {
            var inputSeqKey = fastNode.Inputs[0]!.Value;
            var positionKey = fastNode.Inputs[1]!.Value;
            var outputKey = fastNode.Outputs[0]!.Value;

            var resolved = ctx.ResolveModelKey(inputSeqKey);
            Debug.Assert(ctx.UnpackedStructSequences.ContainsKey(resolved),
                "SEQUENCE_ERASE input is not a known unpacked Model sequence.");

            var fieldSeqKeys = ctx.UnpackedStructSequences[resolved];
            var newFieldSeqKeys = new List<FastTensorKey>(fieldSeqKeys.Count);
            foreach (var fieldSeqKey in fieldSeqKeys)
            {
                var eraseKey = FastNodeKey.New();
                ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceErase(eraseKey, fieldSeqKey, positionKey));
                newFieldSeqKeys.Add(new FastTensorKey(eraseKey, 0));
            }

            ctx.UnpackedStructSequences[outputKey] = newFieldSeqKeys;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        /// <summary>
        /// SEQUENCE_INSERT on a Sequence&lt;Model&gt;. Requires the inserted element to be
        /// an unpacked Model struct; emits one SEQUENCE_INSERT per field. The position input
        /// is optional in ONNX semantics; a null slot means "append".
        /// </summary>
        public static void HandleSequenceInsert(FastNode fastNode, FastModelStructContext ctx)
        {
            var inputSeqKey = fastNode.Inputs[0]!.Value;
            var elementKey = fastNode.Inputs[1]!.Value;
            var positionKey = fastNode.Inputs.Count > 2 ? fastNode.Inputs[2] : null;
            var outputKey = fastNode.Outputs[0]!.Value;

            var resolvedSeq = ctx.ResolveModelKey(inputSeqKey);
            var resolvedElement = ctx.ResolveModelKey(elementKey);
            Debug.Assert(ctx.UnpackedStructSequences.ContainsKey(resolvedSeq),
                "SEQUENCE_INSERT input sequence is not a known unpacked Model sequence.");
            Debug.Assert(ctx.UnpackedStructs.ContainsKey(resolvedElement),
                "SEQUENCE_INSERT element is not a known unpacked Model struct.");

            var fieldSeqKeys = ctx.UnpackedStructSequences[resolvedSeq];
            var elementFields = ctx.UnpackedStructs[resolvedElement];
            Debug.Assert(fieldSeqKeys.Count == elementFields.Count,
                "SEQUENCE_INSERT sequence-field count must equal element-field count.");

            var newFieldSeqKeys = new List<FastTensorKey>(fieldSeqKeys.Count);
            for (int i = 0; i < fieldSeqKeys.Count; i++)
            {
                var fieldSeqKey = fieldSeqKeys[i];
                var elementField = elementFields[i]!.Value;
                var insertKey = FastNodeKey.New();
                ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceInsert(
                    insertKey, fieldSeqKey, elementField, positionKey));
                newFieldSeqKeys.Add(new FastTensorKey(insertKey, 0));
            }

            ctx.UnpackedStructSequences[outputKey] = newFieldSeqKeys;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        /// <summary>
        /// SEQUENCE_LENGTH on a Sequence&lt;Model&gt;. All parallel field sequences share the
        /// same length, so we read it off the first field and remap the original output.
        /// </summary>
        public static void HandleSequenceLength(FastNode fastNode, FastModelStructContext ctx)
        {
            var inputSeqKey = fastNode.Inputs[0]!.Value;
            var outputKey = fastNode.Outputs[0]!.Value;

            var resolved = ctx.ResolveModelKey(inputSeqKey);
            Debug.Assert(ctx.UnpackedStructSequences.ContainsKey(resolved),
                "SEQUENCE_LENGTH input is not a known unpacked Model sequence.");

            var firstFieldSeqKey = ctx.UnpackedStructSequences[resolved][0];
            var lenKey = FastNodeKey.New();
            ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceLength(lenKey, firstFieldSeqKey));

            ctx.Remap[outputKey] = new FastTensorKey(lenKey, 0);
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        private static FastTensorKey EmitSequenceEmpty(FastModelStructContext ctx, DType elementDType)
        {
            var key = FastNodeKey.New();
            ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceEmpty(key, elementDType));
            return new FastTensorKey(key, 0);
        }
    }
}
