using Shorokoo.Core.AutoDiffCheckpointing;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Input context provided to op performance estimators.
/// Contains shape information and liveness data needed for estimating
/// compute cost and determining in-place buffer reuse opportunities.
/// </summary>
internal class OpPerfInput
{
    /// <summary>
    /// Shape information for each input tensor. Null entries indicate optional inputs not present.
    /// </summary>
    public required TensorShapeInfo?[] InputShapes { get; init; }

    /// <summary>
    /// Shape information for each output tensor (from shape inference).
    /// Null entries indicate optional outputs not present.
    /// </summary>
    public required TensorShapeInfo?[] OutputShapes { get; init; }

    /// <summary>
    /// For each input tensor, whether it must remain intact after the operation.
    /// True means the tensor is still needed by subsequent operations and cannot be overwritten.
    /// False means this is the last use and the buffer can potentially be reused in-place.
    /// </summary>
    public required bool[] InputMustRemainIntact { get; init; }

    /// <summary>
    /// The operation code (e.g., "Add", "MatMul").
    /// </summary>
    public required string OpCode { get; init; }

    /// <summary>
    /// Operation attributes as key-value pairs, for ops that need attribute values
    /// to estimate performance (e.g., axis for reduction, perm for transpose).
    /// </summary>
    public required IReadOnlyDictionary<string, object?> Attributes { get; init; }
}
