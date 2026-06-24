using Shorokoo.Graph;
using Shorokoo.Core.Graph;
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
    /// Per-op handlers for LOOP_OPEN / LOOP_CLOSE / IF_CLOSE nodes that carry Model-typed
    /// tensors through their variadic input / output slots. Each handler expands every
    /// Model / Sequence&lt;Model&gt; slot into its parallel per-field slots, rewires the
    /// node's FullInputs / FullOutputs in place (preserving the op's original key
    /// structure — LOOP_OPEN has three output groups, LOOP_CLOSE has two, IF_CLOSE has
    /// one), and records the old output keys in the context's unpack tables (or the
    /// simple remap, for non-Model slots whose positions shifted).
    ///
    /// Preserving the node's own <see cref="FastNodeKey"/> keeps the LOOP_CLOSE → LOOP_OPEN
    /// <see cref="FastNode.GraphOpenNodeKey"/> pointer valid without any fix-up.
    /// </summary>
    internal static class FastModelControlFlowOpHandlers
    {
        // LoopAPI.ProcessNode collapses LOOP_OPEN's three declared output groups
        // (iterationIndex / "" / loopVariables) into a single AttrBody group. LOOP_CLOSE
        // keeps its outputs under the default key (the loopedVariables + scannedVariables
        // declared groups are also collapsed by LoopAPI). IF_CLOSE's outputs likewise live
        // under the default key.
        private const string LoopOpenOutputGroup = OnnxOpAttributeNames.AttrBody;
        private const string LoopCloseOutputGroup = "";
        private const string IfCloseOutputGroup = "";

        /// <summary>
        /// LOOP_OPEN whose loop-variable inputs include at least one Model /
        /// Sequence&lt;Model&gt;. Inputs live under the default key as
        /// <c>[maxIterations, cond, ...loopVariables]</c>; outputs live under the
        /// <see cref="OnnxOpAttributeNames.AttrBody"/> key as
        /// <c>[iterationIndex, vestigialTrue, ...loopVariables]</c>. (The output grouping
        /// is flattened by <c>LoopAPI.ProcessNode</c> even though the Op definition
        /// declares three separate groups.) Only the trailing variadic slice needs
        /// expanding.
        /// </summary>
        public static void HandleLoopOpen(FastNode fastNode, FastModelStructContext ctx)
        {
            // FullInputs[""] = [maxIter, cond, ...loopVars]
            // FullOutputs["body"] = [iterationIndex, vestigialTrue, ...loopVars]
            var origDirectInputs = fastNode.FullInputs[""];
            Debug.Assert(origDirectInputs.Count >= 2);

            var origLoopVarInputs = new List<FastTensorKey?>(origDirectInputs.Count - 2);
            for (int i = 2; i < origDirectInputs.Count; i++)
                origLoopVarInputs.Add(origDirectInputs[i]);

            var origBodyOutputs = fastNode.FullOutputs[LoopOpenOutputGroup];
            Debug.Assert(origBodyOutputs.Count >= 2);

            var origLoopVarOutputs = new List<FastTensorKey?>(origBodyOutputs.Count - 2);
            for (int i = 2; i < origBodyOutputs.Count; i++)
                origLoopVarOutputs.Add(origBodyOutputs[i]);

            // Flat slot indices for the variadic group start at 2 (after iterationIndex
            // and vestigialTrue). We mirror them so remap lookups still match.
            int startOutputSlot = 2;

            var newLoopVarInputs = new List<FastTensorKey?>();
            var newLoopVarOutputs = new List<FastTensorKey?>();
            ExpandVariadicSlotsFlat(
                fastNode, ctx,
                origLoopVarInputs, origLoopVarOutputs,
                startOutputSlot,
                newInputs: newLoopVarInputs, newOutputs: newLoopVarOutputs);

            var newDirectInputs = new List<FastTensorKey?>(2 + newLoopVarInputs.Count)
            {
                origDirectInputs[0], origDirectInputs[1]
            };
            newDirectInputs.AddRange(newLoopVarInputs);

            var newBodyOutputs = new List<FastTensorKey?>(2 + newLoopVarOutputs.Count)
            {
                origBodyOutputs[0], origBodyOutputs[1]
            };
            newBodyOutputs.AddRange(newLoopVarOutputs);

            fastNode.FullInputs[""] = newDirectInputs;
            fastNode.FullOutputs[LoopOpenOutputGroup] = newBodyOutputs;

            // Remember the pre-expansion variadic count so the matching LOOP_CLOSE handler
            // can recover the loopVar / scanVar split of its own (still pre-expansion)
            // body inputs.
            ctx.SetLoopOpenOriginalVariadicCount(fastNode.Key, origLoopVarInputs.Count);
            ctx.NodeByKey[fastNode.Key] = fastNode;
        }

        /// <summary>
        /// LOOP_CLOSE whose loop-variable inputs include at least one Model /
        /// Sequence&lt;Model&gt;. Inputs live under the
        /// <see cref="OnnxOpAttributeNames.AttrBody"/> graph-attribute key as
        /// <c>[break, ...loopVars, ...scanVars]</c>; outputs live under the default key as
        /// <c>[...loopedVars, ...scannedVars]</c>. (LoopAPI.ProcessNode collapses
        /// the declared loopedVariables / scannedVariables output groups into a single
        /// key.) Only the loopVars slice can be Model-typed — scanVars are always plain
        /// tensors — so we expand just those.
        /// </summary>
        public static void HandleLoopClose(FastNode fastNode, FastModelStructContext ctx)
        {
            var openNodeKey = fastNode.GraphOpenNodeKey
                ?? throw new System.InvalidOperationException(
                    "FastUnpackModelStruct: LOOP_CLOSE with Model I/O is missing GraphOpenNodeKey.");

            // CLOSE's body inputs are still pre-expansion here. We've already mutated
            // OPEN's inputs, so its current variadic arity is the expanded count. Recover
            // the original variadic count via the stash in HandleLoopOpen.
            int originalLoopVarCount = ctx.GetLoopOpenOriginalVariadicCount(openNodeKey);

            // FullInputs["body"] = [break, ...loopVars, ...scanVars]
            // FullOutputs[""]   = [...loopedVars, ...scannedVars]
            var origBodyInputs = fastNode.FullInputs[OnnxOpAttributeNames.AttrBody];
            var origBreakInput = origBodyInputs[0];

            var origLoopVarInputs = new List<FastTensorKey?>(originalLoopVarCount);
            for (int i = 0; i < originalLoopVarCount; i++)
                origLoopVarInputs.Add(origBodyInputs[1 + i]);

            var origScanVarInputs = new List<FastTensorKey?>();
            for (int i = 1 + originalLoopVarCount; i < origBodyInputs.Count; i++)
                origScanVarInputs.Add(origBodyInputs[i]);

            var origFlatOutputs = fastNode.FullOutputs[LoopCloseOutputGroup];
            var origLoopVarOutputs = new List<FastTensorKey?>(originalLoopVarCount);
            for (int i = 0; i < originalLoopVarCount; i++)
                origLoopVarOutputs.Add(origFlatOutputs[i]);
            var origScanVarOutputs = new List<FastTensorKey?>();
            for (int i = originalLoopVarCount; i < origFlatOutputs.Count; i++)
                origScanVarOutputs.Add(origFlatOutputs[i]);

            // Flat output slot indices: loopedVariables start at 0.
            int startOutputSlot = 0;

            var newLoopVarInputs = new List<FastTensorKey?>();
            var newLoopVarOutputs = new List<FastTensorKey?>();
            ExpandVariadicSlotsFlat(
                fastNode, ctx,
                origLoopVarInputs, origLoopVarOutputs,
                startOutputSlot,
                newInputs: newLoopVarInputs, newOutputs: newLoopVarOutputs);

            // ScanVars pass through verbatim — but their output slot numbers shift by the
            // same delta as loopVars' expansion. Remap to keep downstream refs valid.
            int scanOutputBaseSlot = newLoopVarOutputs.Count;
            var newScanVarOutputs = new List<FastTensorKey?>(origScanVarOutputs.Count);
            for (int i = 0; i < origScanVarOutputs.Count; i++)
            {
                var origScanOut = origScanVarOutputs[i]!.Value;
                var newScanOut = new FastTensorKey(fastNode.Key, scanOutputBaseSlot + i);
                newScanVarOutputs.Add(newScanOut);
                if (!newScanOut.Equals(origScanOut))
                    ctx.Remap[origScanOut] = newScanOut;
            }

            var newBodyInputs = new List<FastTensorKey?>(1 + newLoopVarInputs.Count + origScanVarInputs.Count)
            {
                origBreakInput
            };
            newBodyInputs.AddRange(newLoopVarInputs);
            newBodyInputs.AddRange(origScanVarInputs);

            var newFlatOutputs = new List<FastTensorKey?>(newLoopVarOutputs.Count + newScanVarOutputs.Count);
            newFlatOutputs.AddRange(newLoopVarOutputs);
            newFlatOutputs.AddRange(newScanVarOutputs);

            fastNode.FullInputs[OnnxOpAttributeNames.AttrBody] = newBodyInputs;
            fastNode.FullOutputs[LoopCloseOutputGroup] = newFlatOutputs;
            ctx.NodeByKey[fastNode.Key] = fastNode;
        }

        /// <summary>
        /// IF_CLOSE whose branch inputs include at least one Model / Sequence&lt;Model&gt;.
        /// Inputs live under <see cref="OnnxOpAttributeNames.AttrElseBranch"/> /
        /// <see cref="OnnxOpAttributeNames.AttrThenBranch"/>; outputs live under the
        /// single <c>outputs</c> group. Both branches must expand to the same shape — we
        /// drive off the else branch's unpack state and assert parity.
        /// </summary>
        public static void HandleIfClose(FastNode fastNode, FastModelStructContext ctx)
        {
            var origElse = fastNode.FullInputs[OnnxOpAttributeNames.AttrElseBranch];
            var origThen = fastNode.FullInputs[OnnxOpAttributeNames.AttrThenBranch];
            Debug.Assert(origElse.Count == origThen.Count,
                "IF_CLOSE then/else branches must have matching arity.");

            var origOutputs = fastNode.FullOutputs[IfCloseOutputGroup];
            Debug.Assert(origOutputs.Count == origElse.Count,
                "IF_CLOSE output count must match branch arity.");

            var newElse = new List<FastTensorKey?>();
            var newThen = new List<FastTensorKey?>();
            var newOutputs = new List<FastTensorKey?>();

            int newOutputSlot = 0;
            for (int i = 0; i < origElse.Count; i++)
            {
                var elseKey = origElse[i]!.Value;
                var thenKey = origThen[i]!.Value;
                var resolvedElse = ctx.ResolveModelKey(elseKey);
                var resolvedThen = ctx.ResolveModelKey(thenKey);

                bool elseIsStruct = ctx.UnpackedStructs.ContainsKey(resolvedElse);
                bool thenIsStruct = ctx.UnpackedStructs.ContainsKey(resolvedThen);
                bool elseIsSeq = ctx.UnpackedStructSequences.ContainsKey(resolvedElse);
                bool thenIsSeq = ctx.UnpackedStructSequences.ContainsKey(resolvedThen);

                Debug.Assert(elseIsStruct == thenIsStruct && elseIsSeq == thenIsSeq,
                    "IF_CLOSE branch slot must be unpacked the same way on both sides.");

                if (elseIsStruct)
                {
                    var elseFields = ctx.UnpackedStructs[resolvedElse];
                    var thenFields = ctx.UnpackedStructs[resolvedThen];
                    Debug.Assert(elseFields.Count == thenFields.Count);

                    var origModelOutputKey = origOutputs[i]!.Value;
                    var newFieldOutputKeys = new List<FastTensorKey?>(elseFields.Count);
                    for (int f = 0; f < elseFields.Count; f++)
                    {
                        newElse.Add(elseFields[f]);
                        newThen.Add(thenFields[f]);
                        var newOutputKey = new FastTensorKey(fastNode.Key, newOutputSlot++);
                        newOutputs.Add(newOutputKey);
                        newFieldOutputKeys.Add(newOutputKey);
                    }
                    ctx.UnpackedStructs[origModelOutputKey] = newFieldOutputKeys;
                }
                else if (elseIsSeq)
                {
                    var elseSeqs = ctx.UnpackedStructSequences[resolvedElse];
                    var thenSeqs = ctx.UnpackedStructSequences[resolvedThen];
                    Debug.Assert(elseSeqs.Count == thenSeqs.Count);

                    var origModelOutputKey = origOutputs[i]!.Value;
                    var newFieldOutputKeys = new List<FastTensorKey>(elseSeqs.Count);
                    for (int f = 0; f < elseSeqs.Count; f++)
                    {
                        newElse.Add(elseSeqs[f]);
                        newThen.Add(thenSeqs[f]);
                        var newOutputKey = new FastTensorKey(fastNode.Key, newOutputSlot++);
                        newOutputs.Add(newOutputKey);
                        newFieldOutputKeys.Add(newOutputKey);
                    }
                    ctx.UnpackedStructSequences[origModelOutputKey] = newFieldOutputKeys;
                }
                else
                {
                    newElse.Add(elseKey);
                    newThen.Add(thenKey);
                    var origOutput = origOutputs[i]!.Value;
                    var newOutputKey = new FastTensorKey(fastNode.Key, newOutputSlot++);
                    newOutputs.Add(newOutputKey);
                    if (!newOutputKey.Equals(origOutput))
                        ctx.Remap[origOutput] = newOutputKey;
                }
            }

            fastNode.FullInputs[OnnxOpAttributeNames.AttrElseBranch] = newElse;
            fastNode.FullInputs[OnnxOpAttributeNames.AttrThenBranch] = newThen;
            fastNode.FullOutputs[IfCloseOutputGroup] = newOutputs;
            ctx.NodeByKey[fastNode.Key] = fastNode;
        }

        /// <summary>
        /// Flat variadic expansion: for each original slot in <paramref name="origInputs"/>
        /// / <paramref name="origOutputs"/>, if the input is an unpacked Model struct or
        /// Sequence&lt;Model&gt; it expands into its parallel field slots; otherwise it
        /// passes through with a freshly-allocated output FastTensorKey. Updates the context's
        /// unpack tables and remap accordingly.
        ///
        /// <paramref name="startOutputSlot"/> is the flat slot index assigned to the first
        /// emitted output in the node's overall <see cref="FastNode.Outputs"/> view (e.g.
        /// 2 for LOOP_OPEN's <c>loopVariables</c> group, 0 for LOOP_CLOSE's
        /// <c>loopedVariables</c> / IF_CLOSE's <c>outputs</c>). Keeping slot numbers
        /// consistent with the flat view means downstream remap lookups by
        /// <c>(nodeKey, flatSlot)</c> still match.
        /// </summary>
        private static void ExpandVariadicSlotsFlat(
            FastNode fastNode, FastModelStructContext ctx,
            List<FastTensorKey?> origInputs, List<FastTensorKey?> origOutputs,
            int startOutputSlot,
            List<FastTensorKey?> newInputs, List<FastTensorKey?> newOutputs)
        {
            Debug.Assert(origInputs.Count == origOutputs.Count);

            int slot = startOutputSlot;
            for (int i = 0; i < origInputs.Count; i++)
            {
                var origInput = origInputs[i]!.Value;
                var origOutput = origOutputs[i]!.Value;
                var resolved = ctx.ResolveModelKey(origInput);

                if (ctx.UnpackedStructs.TryGetValue(resolved, out var fields))
                {
                    var newFieldOutputKeys = new List<FastTensorKey?>(fields.Count);
                    foreach (var field in fields)
                    {
                        newInputs.Add(field);
                        var newOutputKey = new FastTensorKey(fastNode.Key, slot++);
                        newOutputs.Add(newOutputKey);
                        newFieldOutputKeys.Add(newOutputKey);
                    }
                    ctx.UnpackedStructs[origOutput] = newFieldOutputKeys;
                }
                else if (ctx.UnpackedStructSequences.TryGetValue(resolved, out var seqs))
                {
                    var newFieldOutputKeys = new List<FastTensorKey>(seqs.Count);
                    foreach (var seq in seqs)
                    {
                        newInputs.Add(seq);
                        var newOutputKey = new FastTensorKey(fastNode.Key, slot++);
                        newOutputs.Add(newOutputKey);
                        newFieldOutputKeys.Add(newOutputKey);
                    }
                    ctx.UnpackedStructSequences[origOutput] = newFieldOutputKeys;
                }
                else
                {
                    newInputs.Add(origInput);
                    var newOutputKey = new FastTensorKey(fastNode.Key, slot++);
                    newOutputs.Add(newOutputKey);
                    if (!newOutputKey.Equals(origOutput))
                        ctx.Remap[origOutput] = newOutputKey;
                }
            }
        }
    }
}
