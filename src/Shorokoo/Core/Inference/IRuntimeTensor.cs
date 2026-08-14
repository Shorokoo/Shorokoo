using System.Collections.Immutable;
using Shorokoo.Core.Graph;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core.Inference;

/// <summary>
/// Common surface for every runtime tensor object tracked by the QuickExecutionEngine. This is
/// implemented by <see cref="RuntimeTensor"/> (plain tensor), <see cref="RuntimeOptionalTensor"/>
/// (optional wrapper), and <see cref="RuntimeSequenceTensor"/> (sequence of tensors).
///
/// Instances are immutable once constructed. All concrete types are <c>record class</c>es, so
/// callers produce modified copies via C#'s <c>with</c> expression. Array-valued payloads are
/// <see cref="ImmutableArray{T}"/> so no consumer can mutate them.
/// </summary>
public interface IRuntimeTensor
{
    /// <summary>
    /// The logical element data type. For a sequence/optional this is the element dtype.
    /// </summary>
    DType DType { get; init; }

    /// <summary>
    /// The tensor object from the computation graph that this runtime tensor corresponds to.
    /// </summary>
    Variable? ReferenceTensor { get; init; }

    /// <summary>
    /// One slot per loop enclosing this tensor when it was produced, outermost first. Null for
    /// tensors created outside every loop. Only the length is meaningful — the nesting depth;
    /// the engine fills the slots with zeros rather than the iteration each loop was on, and
    /// nothing reads them.
    /// </summary>
    ImmutableArray<long>? IterationIndices { get; init; }

    /// <summary>
    /// All runtime tensors previously produced for the same <see cref="Graph.FastTensorKey"/>,
    /// earliest-first. Only populated for tensors produced inside a loop body.
    /// </summary>
    ImmutableArray<IRuntimeTensor>? History { get; init; }
}
