namespace Shorokoo.Core.Inference.Abstractions;

public enum ShorokooGraphOptimization
{
    DisableAll = 0,
    EnableBasic = 1,
    EnableExtended = 2,
    EnableAll = 99,

    /// <summary>
    /// The profile for a lowered training step: <see cref="EnableAll"/>, with two differences
    /// that only a training step wants. The backend must run the graph's duplicated work as
    /// written rather than merging identical subexpressions — the memory-aware pass recomputes
    /// activations on purpose, a clone is the same op on the same inputs, and a
    /// common-subexpression pass would fold every clone back into the tensor it exists to free.
    /// And denormal floats are flushed to zero: gradients are full of them, and a GEMM on
    /// denormal operands runs several times slower. Flushing is a visible change for a graph
    /// that draws or computes values inside the denormal range (a uniform draw over
    /// <c>[0, float.Epsilon)</c>, say), which is why it is not applied to every session.
    /// </summary>
    TrainingStep = 100,
}
