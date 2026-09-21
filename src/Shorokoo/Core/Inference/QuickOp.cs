using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference;

/// <summary>
/// Base class for every operator implementation in the QuickExecutionEngine. A single
/// instance is registered per op code and invoked for every node bearing that op code — by
/// every engine, on every thread — so an op holds no mutable state. What has to survive
/// between invocations lives in the per-run <see cref="QuickRunState"/> that
/// <see cref="Execute"/> receives; anything an <see cref="Execute"/> override needs to hand
/// its own helpers travels as an argument.
///
/// Three layers, from specific to general:
///   - <see cref="Compute(RuntimeTensor?[], OnnxCSharpAttributes, int)"/> (required): pure
///     shape/dtype inference from tensor inputs + attributes. Almost every op implements just
///     this.
///   - <see cref="Compute(IRuntimeTensor?[], OnnxCSharpAttributes, int)"/> (optional): same
///     shape but accepts the full runtime-tensor hierarchy (plain tensors, optionals,
///     sequences). Default casts each input to <see cref="RuntimeTensor"/> (nulling anything
///     that isn't a plain tensor) and delegates to the tensor-only overload. Ops that natively
///     work with optionals / sequences (SequenceConstruct, OptionalGetElement, etc.) override
///     this instead.
///   - <see cref="Execute"/> (optional): the orchestration layer. Has access to the graph node
///     and tensor store, so it can resolve inputs from places that aren't the node's own input
///     array — e.g. a close node reading its paired open node's inputs — and it alone can ask
///     the engine to rewind a loop body. Default gathers <c>node.Inputs</c>, calls
///     <see cref="RunCompute"/> and never loops back. Only control-flow close ops and
///     <c>SplitOp</c> override this.
/// </summary>
internal abstract class QuickOp
{
    /// <summary>The op code this operator handles (e.g., "Add", "Relu", "Loop#CLOSE").</summary>
    public abstract string OpCode { get; }

    /// <summary>
    /// Runs the op for the given node. Default gathers inputs by <see cref="FastTensorKey"/>
    /// and delegates to <see cref="RunCompute"/>. Control-flow close ops (LoopClose, IfClose)
    /// override this to pull in their paired open node's data; only LoopClose ever asks for a
    /// loop-back.
    /// </summary>
    public virtual (IRuntimeTensor[] results, bool loopBack) Execute(
        FastNode node,
        InternalComputationGraph graph,
        Dictionary<FastNodeKey, FastNode> nodeByKey,
        Dictionary<FastTensorKey, IRuntimeTensor> store,
        int maxDataElements,
        QuickRunState state)
    {
        var inputs = GatherInputs(node.Inputs, store);
        return (RunCompute(inputs, node, maxDataElements), false);
    }

    /// <summary>
    /// Shared tail for the default <see cref="Execute"/> and for overrides whose results come
    /// straight from <c>Compute</c> (<c>IfCloseOp</c>) — an override that builds its results some
    /// other way calls <see cref="FinalizeOutputs"/> itself, as <c>SplitOp</c> and
    /// <c>LoopCloseOp</c> do. Delegates to
    /// <see cref="Compute(IRuntimeTensor?[], OnnxCSharpAttributes, int)"/>, narrows each integer
    /// output to its declared width, and enforces the per-output data-size limit. Each op is
    /// expected to emit <see cref="IRuntimeTensor"/> results with their dtype already populated
    /// (no ReferenceTensor wiring — FastNode has no Variable objects), but not to remember that
    /// QEE's shared 64-bit integer buffer is wider than most of the dtypes it carries — see
    /// <see cref="RuntimeTensorFactory.NarrowToDeclaredWidth(IRuntimeTensor)"/>.
    /// </summary>
    protected IRuntimeTensor[] RunCompute(
        IRuntimeTensor?[] inputs,
        FastNode node,
        int maxDataElements)
    {
        var results = Compute(inputs, node.Attributes, maxDataElements);
        FinalizeOutputs(results, maxDataElements);
        return results;
    }

    /// <summary>
    /// The per-output tail every op's results must pass through, in place: enforce the data-size
    /// limit first (a discarded buffer needs no further work), then narrow each surviving integer
    /// buffer to its declared width — see <see cref="RuntimeTensorFactory.NarrowToDeclaredWidth(IRuntimeTensor)"/>.
    /// <see cref="RunCompute"/> applies it for the ordinary path; an <see cref="Execute"/> override
    /// that builds its results some other way must call this itself.
    /// </summary>
    protected static void FinalizeOutputs(IRuntimeTensor[] results, int maxDataElements)
    {
        for (int i = 0; i < results.Length; i++)
        {
            var rt = results[i];
            if (rt is null) continue;
            rt = RuntimeTensorFactory.EnforceDataSizeLimit(rt, maxDataElements);
            results[i] = RuntimeTensorFactory.NarrowToDeclaredWidth(rt);
        }
    }

    /// <summary>
    /// Resolves a list of tensor keys to runtime tensors. Null keys stay null.
    /// </summary>
    protected static IRuntimeTensor?[] GatherInputs(
        System.Collections.Generic.IReadOnlyList<FastTensorKey?> keys,
        Dictionary<FastTensorKey, IRuntimeTensor> store)
    {
        var rs = new IRuntimeTensor?[keys.Count];
        for (int i = 0; i < keys.Count; i++)
        {
            var k = keys[i];
            if (k is null) { rs[i] = null; continue; }
            store.TryGetValue(k.Value, out var rt);
            rs[i] = rt;
        }
        return rs;
    }

    /// <summary>
    /// IRuntimeTensor-layer compute. Default casts each input to <see cref="RuntimeTensor"/>
    /// (nulling anything that isn't a plain tensor — e.g. a sequence or optional) and forwards
    /// to the tensor-only <see cref="Compute(RuntimeTensor?[], OnnxCSharpAttributes, int)"/>.
    /// Ops that natively handle non-tensor structures override this.
    /// </summary>
    protected virtual IRuntimeTensor[] Compute(
        IRuntimeTensor?[] inputs,
        OnnxCSharpAttributes attrs,
        int maxDataElements)
    {
        var rtInputs = new RuntimeTensor?[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
            rtInputs[i] = inputs[i] as RuntimeTensor;
        var results = Compute(rtInputs, attrs, maxDataElements);
        var asInterface = new IRuntimeTensor[results.Length];
        for (int i = 0; i < results.Length; i++) asInterface[i] = results[i];
        return asInterface;
    }

    /// <summary>
    /// Implements the operator. Receives one plain runtime tensor per node input (null when
    /// the input variable itself was null in the graph, or when the stored runtime value is a
    /// sequence / optional rather than a plain tensor). Must produce one runtime tensor per
    /// node output, in declaration order. Every output's DType must be determinable from the
    /// inputs or attributes alone — the base class handles ReferenceTensor wiring.
    ///
    /// Default returns an empty array. Ops that handle plain tensors override this. Ops that
    /// natively handle sequences / optionals override the IRuntimeTensor overload above and
    /// leave the default in place — the engine never routes through this method for those.
    /// </summary>
    protected virtual RuntimeTensor[] Compute(
        RuntimeTensor?[] inputs,
        OnnxCSharpAttributes attributes,
        int maxDataElements) => Array.Empty<RuntimeTensor>();
}
