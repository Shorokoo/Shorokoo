using Shorokoo.Graph;

namespace Shorokoo;

/// <summary>
/// Static hub for built-in loss computation graphs.
/// Each property returns the same graph as <c>XxxLoss.ComputationGraph</c>,
/// but at the call site you write <c>Losses.L2Loss</c> rather than
/// <c>L2Loss.ComputationGraph</c>.
/// </summary>
/// <example>
/// <code>
/// var rig = TrainingRig.FromScratch(model, Losses.L2Loss, Optimizers.Adam, …);
/// </code>
/// </example>
public static class Losses
{

    /// <summary>Mean squared error — <c>mean((predictions - targets)²)</c>.</summary>
    public static ComputationGraph L2Loss          => global::Shorokoo.Modules.Losses.L2Loss.ComputationGraph;

    /// <summary>Mean absolute error — <c>mean(|predictions - targets|)</c>.</summary>
    public static ComputationGraph L1Loss          => global::Shorokoo.Modules.Losses.L1Loss.ComputationGraph;

    /// <summary>Softmax cross-entropy. Targets are class indices (int64).</summary>
    public static ComputationGraph CrossEntropy    => global::Shorokoo.Modules.Losses.CrossEntropyLoss.ComputationGraph;

    /// <summary>Binary cross-entropy over probabilities in [0, 1].</summary>
    public static ComputationGraph BCE             => global::Shorokoo.Modules.Losses.BCELoss.ComputationGraph;

    /// <summary>Binary cross-entropy applied to raw logits (numerically stable).</summary>
    public static ComputationGraph BCEWithLogits   => global::Shorokoo.Modules.Losses.BCEWithLogitsLoss.ComputationGraph;

    /// <summary>
    /// Smooth L1 (Huber with delta = 1). L2 for small errors, L1 for large ones.
    /// Rig-safe (no hyperparameter input).
    /// </summary>
    public static ComputationGraph SmoothL1        => global::Shorokoo.Modules.Losses.SmoothL1Loss.ComputationGraph;

    /// <summary>
    /// Huber loss with a runtime <c>delta</c> hyperparameter. Requires 3 inputs
    /// (predictions, targets, delta) — not directly rig-safe; use <see cref="SmoothL1"/>
    /// for delta = 1, or wrap in a 2-input module with a fixed delta.
    /// </summary>
    public static ComputationGraph Huber           => global::Shorokoo.Modules.Losses.HuberLoss.ComputationGraph;

    /// <summary>Hinge loss for binary classification.</summary>
    public static ComputationGraph Hinge           => global::Shorokoo.Modules.Losses.HingeLoss.ComputationGraph;

    /// <summary>Squared hinge loss.</summary>
    public static ComputationGraph SquaredHinge    => global::Shorokoo.Modules.Losses.SquaredHingeLoss.ComputationGraph;

    /// <summary>Kullback-Leibler divergence.</summary>
    public static ComputationGraph KLDiv           => global::Shorokoo.Modules.Losses.KLDivLoss.ComputationGraph;

    /// <summary>Negative log-likelihood loss.</summary>
    public static ComputationGraph NLL             => global::Shorokoo.Modules.Losses.NLLLoss.ComputationGraph;

    /// <summary>Poisson negative log-likelihood loss.</summary>
    public static ComputationGraph PoissonNLL      => global::Shorokoo.Modules.Losses.PoissonNLLLoss.ComputationGraph;

    /// <summary>Log-cosh loss — smooth approximation of MAE.</summary>
    public static ComputationGraph LogCosh         => global::Shorokoo.Modules.Losses.LogCoshLoss.ComputationGraph;

    /// <summary>Cosine embedding loss.</summary>
    public static ComputationGraph CosineEmbedding => global::Shorokoo.Modules.Losses.CosineEmbeddingLoss.ComputationGraph;

    /// <summary>Triplet margin loss.</summary>
    public static ComputationGraph TripletMargin   => global::Shorokoo.Modules.Losses.TripletMarginLoss.ComputationGraph;

    /// <summary>Binary focal loss for class-imbalanced problems.</summary>
    public static ComputationGraph BinaryFocal     => global::Shorokoo.Modules.Losses.BinaryFocalLoss.ComputationGraph;
}
