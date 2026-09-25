using Shorokoo.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;

namespace Shorokoo.Core.Nodes.Processors.Fast
{
    /// <summary>
    /// Per-parameter initialization RNG. Rewrites the abstract <c>SHRK_RANDOM_*</c> draw
    /// inside a trainable-parameter's initializer to the keyed in-graph draw
    /// (<c>SHRK_RNG_UNIFORM/NORMAL</c>) on the parameter's own stream — the same
    /// counter-based Threefry lowering the runtime feeds use — so that same-shape
    /// parameters no longer receive identical values, initialization is reproducible for a
    /// config, and the draw executes on the compute backend (values are never staged as
    /// host-generated constants, so init is not bounded by the protobuf constant-size
    /// ceiling and parallelizes on GPU).
    ///
    /// <para>The draw node is rewritten in place: its key input is the parameter's folded
    /// init key as a uint64 scalar constant — split once per enclosing loop by that loop's
    /// runtime iteration index, so a draw inside a loop body is a fresh sample on every trip
    /// rather than one sample re-derived (Shorokoo/Shorokoo#343, the initialization-side twin of
    /// the runtime defect #289; the split is the same <c>SHRK_RNG_SPLIT</c> a runtime feed's
    /// chain folds with, on the same algorithm-independent default) — its substreamIndex is the
    /// draw's ordinal within the initializer (a distinct sub-stream per draw SITE; every shipping
    /// initializer has exactly one, so ordinal 0 in practice), and its shape input and declared
    /// distribution bounds carry over — the initializer's downstream scaling math is unchanged.
    /// <see cref="FastLowerRandomOps"/> later lowers the keyed node to a call of the named
    /// algorithm's exported function, exactly as for a runtime feed.</para>
    ///
    /// <para>The substitution runs on the initializer's <b>flattened</b> body
    /// (<see cref="Function.GetFastFlattenedGraph"/>), so a draw factored into a called
    /// initializer is inlined to the top level and keyed like an inline draw — each inlined
    /// call site becomes its own node and its own sub-stream ordinal. Inside an initializer
    /// body a nested <c>Init</c> call is emitted as an ordinary invoke of the called
    /// initializer's body rather than as a second parameter definition, so the shipped
    /// parameterized initializers are reachable from a custom one and draw on the parameter
    /// being created (Shorokoo/Shorokoo#323). Nothing else an initializer body may call
    /// survives flattening except the RNG algorithm functions themselves: a body that calls a
    /// module is refused when it is built (FW055). A draw inside any other call left standing
    /// is still refused loudly rather than drawn unkeyed.</para>
    /// </summary>
    internal static class FastInitKeyedDraws
    {
        /// <summary>Whether an initializer draws randomness (so it needs a stream key).
        /// Mirrors the flatten-then-scan the rewrite itself performs.</summary>
        public static bool DrawsRandomness(Function fn)
            => fn.GetFastFlattenedGraph().Nodes.Any(n =>
                   n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                   n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                   n.OpCode == InternalOpCodes.SHRK_RANDOM_BITS)
               || fn.ReferencedFunctions.Any(f => f.OriginalFastGraph.Nodes.Any(n =>
                   n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                   n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                   n.OpCode == InternalOpCodes.SHRK_RANDOM_BITS));

        /// <summary>
        /// Returns a new initializer <see cref="Function"/> whose random draws are rewritten
        /// to keyed in-graph draws on the parameter's own stream <paramref name="streamKey"/>
        /// (resolved by the caller by EXECUTING the derivation — see
        /// <c>FastInitializeModelParams.ResolveInitKeys</c>; the host folds nothing itself, #136)
        /// under the named <paramref name="algorithm"/>, or <c>null</c> if it contains no
        /// random ops (the caller then keeps the original). Draws nested in called
        /// initializers are reached by flattening the body first.
        /// </summary>
        public static Function? BuildKeyedDraws(
            Function fn, ulong streamKey, string streamName, string algorithm)
        {
            // Flatten so a draw factored into a called initializer becomes a top-level node the
            // substitution below can intercept. Shipping initializers contain no calls, so their
            // flattened body is node-identical to the original.
            var body = fn.GetFastFlattenedGraph().Clone();

            // Backstop: anything still invoked after flattening must not smuggle a draw past the
            // top-level scan. An initializer body can only call other initializers (inlined above)
            // and the RNG algorithm functions (never inlined, excluded here) — calling a module is
            // refused when the body is built or read back from a file (FW055) — so this is not
            // expected to fire; it stays
            // so that a call some future path leaves standing fails loudly instead of drawing
            // unkeyed, non-reproducible backend randomness.
            var nested = body.Nodes
                .Where(n => n.OpCode == InternalOpCodes.FUNCTION_INVOKE ||
                            n.OpCode == InternalOpCodes.MODEL_INVOKE ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM_REF ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM_ID_REF ||
                            n.OpCode == InternalOpCodes.MODEL_PARAM_MODEL_REF)
                .Select(n => n.TargetFunction)
                .Where(f => f is not null && f.RngFunctionKind is null)
                .SelectMany(f => f!.ReferencedFunctions.Concat([f]))
                .FirstOrDefault(f => f!.OriginalFastGraph.Nodes.Any(n =>
                    n.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM ||
                    n.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL ||
                    n.OpCode == InternalOpCodes.SHRK_RANDOM_BITS));
            if (nested is not null)
                throw new NotSupportedException(
                    $"Initializer '{fn.FriendlyName}' of parameter '{streamName}' draws randomness " +
                    $"inside the called function '{nested.FriendlyName}', which could not be inlined. " +
                    "The nested draw keeps no parameter key and would fall back to unkeyed, " +
                    "non-reproducible backend randomness. Make the draw the initializer's own: " +
                    "call another initializer's Init (whose body IS inlined and keyed on this " +
                    "parameter), or move the draw (RandomUniform/RandomNormal/RandomBits) directly " +
                    "into the initializer's body.");

            var newNodes = new List<FastNode>(body.Nodes.Count);
            int randomOrdinal = 0;

            // The loops enclosing the node being visited, outermost first. Scope membership in the
            // Fast pipeline is positional, so this is what tells a draw which iteration indices
            // its key has to fold in.
            var enclosingLoops = new List<FastNode>();

            foreach (var node in body.Nodes)
            {
                bool isUniform = node.OpCode == InternalOpCodes.SHRK_RANDOM_UNIFORM;
                bool isNormal = node.OpCode == InternalOpCodes.SHRK_RANDOM_NORMAL;
                bool isBits = node.OpCode == InternalOpCodes.SHRK_RANDOM_BITS;
                if (!isUniform && !isNormal && !isBits)
                {
                    if (node.OpCode == OpCodes.LOOP_OPEN)
                        enclosingLoops.Add(node);
                    else if (node.OpCode == OpCodes.LOOP_CLOSE)
                    {
                        // Swallowing an underflow would leave the stack shifted for every later
                        // draw and still end at zero, so it would report nothing.
                        Debug.Assert(enclosingLoops.Count > 0,
                            "FastInitKeyedDraws: LOOP_CLOSE before its LOOP_OPEN.");
                        if (enclosingLoops.Count > 0)
                            enclosingLoops.RemoveAt(enclosingLoops.Count - 1);
                    }
                    newNodes.Add(node);
                    continue;
                }

                var shapeInput = node.Inputs[0]
                    ?? throw new InvalidOperationException("Random init node has null shape input.");

                // The parameter's own stream key as a scalar constant, folded once per enclosing
                // loop by that loop's iteration index, and a distinct sub-stream
                // (substreamIndex = ordinal) per draw within one initializer. Every node is emitted
                // per draw so it always sits in the draw's own control-flow scope — the splits in
                // particular MUST stay inside the loop body, since they read its iteration index.
                var keyKey = AppendConstant(
                    Shorokoo.Globals.TensorData([], streamKey).MoveToAttribute(), newNodes);
                // The fold is FastWireRngKeyDerivation's own split, not a look-alike: a per-trip
                // init key is the same bijection of the key tree a runtime feed's chain applies.
                //
                // The one asymmetry with that chain is overrides. A feed's chain selects an
                // override at runtime, per iteration, because its iteration slots are ModelId path
                // elements an override can address. A parameter's override is applied when
                // streamKey is resolved on the host, so it is already baked into the constant
                // these splits fold — which re-seeds every trip together and leaves no way to
                // address one trip (see Documentation/rng-configuration.md).
                //
                // Splitting by the index is exactly the key a parameter at ModelId
                // `path ++ [index]` would get. No such parameter exists: parameter slots and
                // sub-module slots are numbered from one counter per parent, so an id that names a
                // parameter is a leaf and nothing extends it. That invariant is what keeps the two
                // key spaces disjoint — an id-allocation change that let one parameter's path be a
                // prefix of another's would alias their streams silently.
                foreach (var loopOpen in enclosingLoops)
                    keyKey = FastWireRngKeyDerivation.AppendSplit(
                        keyKey,
                        // LOOP_OPEN's output 0 is the iteration index.
                        FastWireRngKeyDerivation.AppendCastToUInt64(loopOpen.Outputs[0]!.Value, newNodes),
                        newNodes);

                var substreamIndexKey = AppendConstant(
                    Shorokoo.Globals.TensorData([], (ulong)randomOrdinal).MoveToAttribute(), newNodes);

                if (isBits)
                {
                    // Raw-bits init draw: keyed on the parameter's own stream, no distribution
                    // bounds; the uint output width rides the shrk_dtype attribute. Mirrors the
                    // runtime RewriteBitsFeedToKeyedDraw.
                    var bitsDtype = node.Attributes.GetDTypeVal(ShrkAttrDtype)
                        ?? throw new InvalidOperationException(
                            "FastInitKeyedDraws: a SHRK_RANDOM_BITS init draw is missing its shrk_dtype (output width) attribute.");
                    node.OpCode = InternalOpCodes.SHRK_RNG_BITS;
                    node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                        new Dictionary<string, object?>
                        {
                            [ShrkAttrRngAlgorithm] = algorithm,
                            [ShrkAttrDtype] = bitsDtype,
                        },
                        Definitions.NodeDefinitions[InternalOpCodes.SHRK_RNG_BITS].AttributeDefs);
                    node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                    {
                        [""] = new List<FastTensorKey?> { keyKey, substreamIndexKey, shapeInput }
                    };
                    newNodes.Add(node);
                    randomOrdinal++;
                    continue;
                }

                float a = isUniform
                    ? node.Attributes.GetFloatVal(AttrLow) ?? 0.0f
                    : node.Attributes.GetFloatVal(AttrMean) ?? 0.0f;
                float b = isUniform
                    ? node.Attributes.GetFloatVal(AttrHigh) ?? 1.0f
                    : node.Attributes.GetFloatVal(AttrScale) ?? 1.0f;

                // A draw whose distribution is computed in the initializer body (e.g. a
                // fan-in-derived bound or standard deviation, or UniformRange's own low/high
                // parameters) carries its parameters as inputs; they go straight to the keyed draw,
                // which takes them as inputs anyway. Only literals are materialized as constants here.
                var bounds = FastLowerRandomOps.TensorBounds(node);
                var aKey = bounds?.low ?? AppendConstant(
                    Shorokoo.Globals.TensorData([], a).MoveToAttribute(), newNodes);
                var bKey = bounds?.high ?? AppendConstant(
                    Shorokoo.Globals.TensorData([], b).MoveToAttribute(), newNodes);

                // Rewrite the random node in place to the keyed draw (inputs
                // [key, substreamIndex, shape, a, b]), preserving its output key so downstream
                // consumers stay valid. FastLowerRandomOps lowers it to the algorithm's
                // function call at ONNX prep.
                var newOp = isUniform ? InternalOpCodes.SHRK_RNG_UNIFORM : InternalOpCodes.SHRK_RNG_NORMAL;
                node.OpCode = newOp;
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [ShrkAttrRngAlgorithm] = algorithm },
                    Definitions.NodeDefinitions[newOp].AttributeDefs);
                node.FullInputs = new Dictionary<string, List<FastTensorKey?>>
                {
                    [""] = new List<FastTensorKey?> { keyKey, substreamIndexKey, shapeInput, aKey, bKey }
                };
                newNodes.Add(node);
                randomOrdinal++;
            }

            // The per-node check above catches a SHIFTED stack; this catches a phantom one — a
            // LOOP_OPEN the sweep never saw closed, which would have silently split every later
            // draw's key by an index that is not in scope.
            Debug.Assert(enclosingLoops.Count == 0,
                "FastInitKeyedDraws: a LOOP_OPEN in the initializer body was never closed.");

            if (randomOrdinal == 0)
                return null; // no random ops; caller keeps the shared original

            body.Nodes = newNodes;

            // Flattening leaves dead ShrkCreateModule / ShrkModuleSetHyperparams metadata
            // behind whose TargetFunction still names the (un-rewritten) inlined module.
            // Prune it: it is unreachable from the initializer's outputs, and leaving it in
            // would make FastLowerRandomOps recurse into the original module body and trip
            // on its now-orphaned id-bearing draw (which this pass already keyed at top level).
            FastProcessorHelper.RemoveUnreachableNodes(body);

            // Give the per-parameter initializer a unique name. The original name
            // ("KaimingUniform", ...) is shared across every parameter using that
            // initializer; leaving it unchanged makes ONNX function emission dedupe the
            // distinct per-parameter bodies (with their distinct keys) down to one, so
            // every same-initializer parameter would collapse to identical values.
            string suffix = new string(streamName.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray());
            return new Function(body, fn.FunctionType,
                defaultName: fn.DefaultName + "__rng__" + suffix,
                friendlyName: fn.FriendlyName + "__rng__" + suffix,
                fn.StateOwnership);
        }

        private static FastTensorKey AppendConstant(TensorAttribute data, List<FastNode> newNodes)
        {
            var constAttrDefs = Definitions.NodeDefinitions[OpCodes.CONSTANT].AttributeDefs;
            var key = FastNodeKey.New();
            var outKey = new FastTensorKey(key, 0);
            newNodes.Add(new FastNode
            {
                Key = key,
                OpCode = OpCodes.CONSTANT,
                Attributes = OnnxCSharpAttributes.FromCSharpVals(
                    new Dictionary<string, object?> { [AttrValue] = data }, constAttrDefs),
                FullInputs = new Dictionary<string, List<FastTensorKey?>>(),
                FullOutputs = { [""] = new List<FastTensorKey?> { outKey } },
            });
            return outKey;
        }
    }
}
