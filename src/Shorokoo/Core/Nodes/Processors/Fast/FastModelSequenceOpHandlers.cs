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
    /// per-field SEQUENCE_* nodes on the InternalComputationGraph, keeping the unpack
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
                fieldSeqKeys.Add(BuildParallelSequence(
                    elementFields.Select(e => e[fieldIdx]).ToList(), ctx));

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
                fieldKeys.Add(TakeFieldAt(fieldSeqKey, positionKey, ctx));

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
                newFieldSeqKeys.Add(EraseFieldAt(fieldSeqKey, positionKey, ctx));

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
                newFieldSeqKeys.Add(InsertFieldInto(
                    fieldSeqKeys[i], elementFields[i]!.Value, positionKey, ctx));

            ctx.UnpackedStructSequences[outputKey] = newFieldSeqKeys;
            ctx.NodesToRemove.Add(fastNode.Key);
        }

        /// <summary>
        /// The parallel sequence for one struct field, over the elements' values for it. A field
        /// that is itself a Model — a <c>[Hyper] Model&lt;&gt;</c> — has no tensor to put in a
        /// sequence, so its own fields get the sequences instead, and the key returned names that
        /// bundle in <see cref="FastModelStructContext.UnpackedStructSequences"/>. That is what
        /// lets a sequence of such hosts be indexed by a value known only at run time: every
        /// tensor the elements own, however deep, ends up in a sequence the index reaches
        /// (Shorokoo/Shorokoo#300).
        /// </summary>
        private static FastTensorKey BuildParallelSequence(
            List<FastTensorKey?> elements, FastModelStructContext ctx)
        {
            var nested = elements
                .Select(e => e is null ? null : ctx.TryGetStruct(e.Value)).ToList();
            if (nested.Any(n => n is not null))
            {
                Debug.Assert(nested.All(n => n is not null),
                    "A struct field is a Model in some sequence elements and a tensor in others.");
                int subFieldCount = nested[0]!.Count;
                Debug.Assert(nested.All(n => n!.Count == subFieldCount),
                    "SEQUENCE_CONSTRUCT nested-element field counts must match.");

                var subSeqKeys = new List<FastTensorKey>(subFieldCount);
                for (int i = 0; i < subFieldCount; i++)
                    subSeqKeys.Add(BuildParallelSequence(
                        nested.Select(n => n![i]).ToList(), ctx));

                var bundleKey = ctx.NewNestedModelField();
                ctx.UnpackedStructSequences[bundleKey] = subSeqKeys;
                return bundleKey;
            }

            var seqKey = FastNodeKey.New();
            ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceConstruct(seqKey, elements));
            return new FastTensorKey(seqKey, 0);
        }

        /// <summary>Reads one field's value at <paramref name="positionKey"/>, recursing into a
        /// nested Model field's own parallel sequences.</summary>
        private static FastTensorKey TakeFieldAt(
            FastTensorKey fieldSeqKey, FastTensorKey positionKey, FastModelStructContext ctx)
        {
            if (ctx.NestedModelFields.Contains(fieldSeqKey))
            {
                var subKeys = ctx.UnpackedStructSequences[fieldSeqKey]
                    .Select(k => (FastTensorKey?)TakeFieldAt(k, positionKey, ctx)).ToList();
                var structKey = ctx.NewNestedModelField();
                ctx.UnpackedStructs[structKey] = subKeys;
                return structKey;
            }

            var atKey = FastNodeKey.New();
            ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceAt(atKey, fieldSeqKey, positionKey));
            return new FastTensorKey(atKey, 0);
        }

        /// <summary>Erases one field's element at <paramref name="positionKey"/>, recursing into a
        /// nested Model field's own parallel sequences.</summary>
        private static FastTensorKey EraseFieldAt(
            FastTensorKey fieldSeqKey, FastTensorKey positionKey, FastModelStructContext ctx)
        {
            if (ctx.NestedModelFields.Contains(fieldSeqKey))
            {
                var erased = ctx.UnpackedStructSequences[fieldSeqKey]
                    .Select(k => EraseFieldAt(k, positionKey, ctx)).ToList();
                var bundleKey = ctx.NewNestedModelField();
                ctx.UnpackedStructSequences[bundleKey] = erased;
                return bundleKey;
            }

            var eraseKey = FastNodeKey.New();
            ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceErase(eraseKey, fieldSeqKey, positionKey));
            return new FastTensorKey(eraseKey, 0);
        }

        /// <summary>Inserts one field's value at <paramref name="positionKey"/> (null appends),
        /// recursing into a nested Model field's own parallel sequences.</summary>
        private static FastTensorKey InsertFieldInto(
            FastTensorKey fieldSeqKey, FastTensorKey elementField, FastTensorKey? positionKey,
            FastModelStructContext ctx)
        {
            if (ctx.NestedModelFields.Contains(fieldSeqKey))
            {
                var subSeqKeys = ctx.UnpackedStructSequences[fieldSeqKey];
                var elementFields = ctx.TryGetStruct(elementField)
                    ?? throw new System.InvalidOperationException(
                        "SEQUENCE_INSERT: a Model-typed field's inserted element is not an unpacked Model struct.");
                Debug.Assert(subSeqKeys.Count == elementFields.Count,
                    "SEQUENCE_INSERT nested sequence-field count must equal element-field count.");

                var inserted = new List<FastTensorKey>(subSeqKeys.Count);
                for (int i = 0; i < subSeqKeys.Count; i++)
                    inserted.Add(InsertFieldInto(subSeqKeys[i], elementFields[i]!.Value, positionKey, ctx));

                var bundleKey = ctx.NewNestedModelField();
                ctx.UnpackedStructSequences[bundleKey] = inserted;
                return bundleKey;
            }

            var insertKey = FastNodeKey.New();
            ctx.RecordNewNode(FastNodeConstructionUtils.CreateSequenceInsert(
                insertKey, fieldSeqKey, elementField, positionKey));
            return new FastTensorKey(insertKey, 0);
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
