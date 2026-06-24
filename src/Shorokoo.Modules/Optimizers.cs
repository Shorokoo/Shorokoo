using Shorokoo.Graph;

namespace Shorokoo;

/// <summary>
/// Static hub for built-in optimizer computation graphs.
/// Each property returns the same graph as <c>XxxOptimizer.ComputationGraph</c>,
/// but at the call site you write <c>Optimizers.Adam</c> rather than
/// <c>AdamOptimizer.ComputationGraph</c>.
/// </summary>
/// <example>
/// <code>
/// var rig = TrainingRig.FromScratch(model, Losses.L2Loss, Optimizers.Adam, …);
/// </code>
/// </example>
public static class Optimizers
{

    /// <summary>Stochastic gradient descent without momentum.</summary>
    public static FastComputationGraph SGD         => global::Shorokoo.Modules.Optimizers.SGDOptimizer.ComputationGraph;

    /// <summary>SGD with momentum.</summary>
    public static FastComputationGraph SGDMomentum => global::Shorokoo.Modules.Optimizers.SGDMomentumOptimizer.ComputationGraph;

    /// <summary>Adam (adaptive moment estimation).</summary>
    public static FastComputationGraph Adam        => global::Shorokoo.Modules.Optimizers.AdamOptimizer.ComputationGraph;

    /// <summary>AdamW — Adam with decoupled weight decay.</summary>
    public static FastComputationGraph AdamW       => global::Shorokoo.Modules.Optimizers.AdamWOptimizer.ComputationGraph;

    /// <summary>Adamax — Adam variant using infinity norm.</summary>
    public static FastComputationGraph Adamax      => global::Shorokoo.Modules.Optimizers.AdamaxOptimizer.ComputationGraph;

    /// <summary>NAdam — Nesterov-accelerated Adam.</summary>
    public static FastComputationGraph NAdam       => global::Shorokoo.Modules.Optimizers.NAdamOptimizer.ComputationGraph;

    /// <summary>Adagrad — per-parameter adaptive learning rates using accumulated gradient squares.</summary>
    public static FastComputationGraph Adagrad     => global::Shorokoo.Modules.Optimizers.AdagradOptimizer.ComputationGraph;

    /// <summary>Adadelta — adaptive learning rate that doesn't require a global learning rate.</summary>
    public static FastComputationGraph Adadelta    => global::Shorokoo.Modules.Optimizers.AdadeltaOptimizer.ComputationGraph;

    /// <summary>RMSprop — divides learning rate by exponential moving average of squared gradients.</summary>
    public static FastComputationGraph RMSprop     => global::Shorokoo.Modules.Optimizers.RMSpropOptimizer.ComputationGraph;

    /// <summary>RAdam — Rectified Adam with variance correction in early training.</summary>
    public static FastComputationGraph RAdam       => global::Shorokoo.Modules.Optimizers.RAdamOptimizer.ComputationGraph;

    /// <summary>LAMB — Layer-wise Adaptive Moments for large-batch training.</summary>
    public static FastComputationGraph Lamb        => global::Shorokoo.Modules.Optimizers.LambOptimizer.ComputationGraph;

    /// <summary>Lion — EvoLved Sign Momentum optimizer.</summary>
    public static FastComputationGraph Lion        => global::Shorokoo.Modules.Optimizers.LionOptimizer.ComputationGraph;

    /// <summary>Adafactor — memory-efficient optimizer with factored second-moment estimates.</summary>
    public static FastComputationGraph Adafactor   => global::Shorokoo.Modules.Optimizers.AdafactorOptimizer.ComputationGraph;
}
