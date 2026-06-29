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
/// instance is registered per op code and invoked for every node bearing that op code.
///
/// Four layers, from specific to general:
///   - <see cref="Compute(RuntimeTensor?[], OnnxCSharpAttributes, int)"/> (required): pure
///     shape/dtype inference from tensor inputs + attributes. Almost every op implements just
///     this.
///   - <see cref="Compute(IRuntimeTensor?[], OnnxCSharpAttributes, int)"/> (optional): same
///     shape but accepts the full runtime-tensor hierarchy (plain tensors, optionals,
///     sequences). Default casts each input to <see cref="RuntimeTensor"/> (nulling anything
///     that isn't a plain tensor) and delegates to the tensor-only overload. Ops that natively
///     work with optionals / sequences (SequenceConstruct, OptionalGetElement, etc.) override
///     this instead.
///   - <see cref="ComputeWithLoopBack"/> (optional): adds a loop-back signal; default delegates
///     to the IRuntimeTensor overload of <see cref="Compute(IRuntimeTensor?[], OnnxCSharpAttributes, int)"/>.
///   - <see cref="Execute"/> (optional): the orchestration layer. Has access to the graph node
///     and tensor store, so it can resolve inputs from places that aren't the node's own input
///     array — e.g. a close node reading its paired open node's inputs. Default gathers
///     <c>node.Inputs</c> and calls <see cref="ComputeWithLoopBack"/>. Only control-flow close
///     ops need to override this.
/// </summary>
internal abstract class QuickOp
{
    /// <summary>The op code this operator handles (e.g., "Add", "Relu", "Loop#CLOSE").</summary>
    public abstract string OpCode { get; }

    /// <summary>
    /// Runs the op for the given node. Default gathers inputs by <see cref="FastTensorKey"/>
    /// and delegates to <see cref="ComputeWithLoopBack"/>. Control-flow close ops (LoopClose,
    /// IfClose) override this to pull in their paired open node's data.
    /// </summary>
    public virtual (IRuntimeTensor[] results, bool loopBack) Execute(
        FastNode node,
        FastComputationGraph graph,
        Dictionary<FastNodeKey, FastNode> nodeByKey,
        Dictionary<FastTensorKey, IRuntimeTensor> store,
        int maxDataElements)
    {
        var inputs = GatherInputs(node.Inputs, store);
        return RunCompute(inputs, node, maxDataElements);
    }

    /// <summary>
    /// Shared tail used by every <see cref="Execute"/> override: delegates to
    /// <see cref="ComputeWithLoopBack"/> and enforces the per-output data-size limit. Each op
    /// is expected to emit <see cref="IRuntimeTensor"/> results with their dtype already
    /// populated (no ReferenceTensor wiring — FastNode has no Variable objects).
    /// </summary>
    protected (IRuntimeTensor[] results, bool loopBack) RunCompute(
        IRuntimeTensor?[] inputs,
        FastNode node,
        int maxDataElements)
    {
        var (results, loopBack) = ComputeWithLoopBack(inputs, node.Attributes, maxDataElements);

        for (int i = 0; i < results.Length; i++)
        {
            var rt = results[i];
            if (rt is null) continue;
            results[i] = RuntimeTensorFactory.EnforceDataSizeLimit(rt, maxDataElements);
        }

        return (results, loopBack);
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
    /// Loop-aware computation entry. Default delegates to the <see cref="IRuntimeTensor"/>
    /// overload of <see cref="Compute(IRuntimeTensor?[], OnnxCSharpAttributes, int)"/> and
    /// never requests a loop-back. <c>LoopCloseOp</c> overrides this to signal when the engine
    /// should rewind the loop body.
    /// </summary>
    protected virtual (IRuntimeTensor[] results, bool loopBack) ComputeWithLoopBack(
        IRuntimeTensor?[] inputs,
        OnnxCSharpAttributes attrs,
        int maxDataElements)
    {
        return (Compute(inputs, attrs, maxDataElements), false);
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
