namespace Shorokoo.Core.Inference.Abstractions;

public enum ShorokooGraphOptimization
{
    DisableAll = 0,
    EnableBasic = 1,
    EnableExtended = 2,
    EnableAll = 99,

    /// <summary>
    /// The profile for a lowered training step: <see cref="EnableAll"/>, except that the backend
    /// must run the graph's duplicated work as written rather than merging identical
    /// subexpressions — the memory-aware pass recomputes activations on purpose, a clone is the
    /// same op on the same inputs, and a common-subexpression pass would fold every clone back
    /// into the tensor it exists to free. Only the training rig's own step sessions use it; a
    /// user-compiled graph runs <see cref="EnableAll"/>.
    /// </summary>
    TrainingStep = 100,
}
