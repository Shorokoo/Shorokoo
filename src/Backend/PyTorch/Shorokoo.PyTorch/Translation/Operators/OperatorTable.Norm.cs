namespace Shorokoo.PyTorch.Translation.Operators;

// Normalization, dropout and losses; the semantics are in shorokoo_torch/ops_norm.py.
internal static partial class OperatorTable
{
    static partial void RegisterNorm(Registry table)
    {
        const string M = "ops_norm.";
        table.Map("BatchNormalization", M + "batch_normalization", ["epsilon", "momentum", "training_mode"], outputs: true);
        table.Map("InstanceNormalization", M + "instance_normalization", ["epsilon"]);
        table.Map("LayerNormalization", M + "layer_normalization", ["axis", "epsilon", "stash_type"], outputs: true);
        table.Map("GroupNormalization", M + "group_normalization", ["epsilon", "num_groups", "stash_type"], opset: true);
        table.Map("RMSNormalization", M + "rms_normalization", ["axis", "epsilon", "stash_type"]);
        table.Map("LRN", M + "lrn", ["alpha", "beta", "bias", "size"]);
        table.Map("LpNormalization", M + "lp_normalization", ["axis", "p"]);
        table.Map("MeanVarianceNormalization", M + "mean_variance_normalization", ["axes"]);
        table.Map("Dropout", M + "dropout", ["seed"], outputs: true);
        table.Map("NegativeLogLikelihoodLoss", M + "negative_log_likelihood_loss", ["ignore_index", "reduction"]);
        table.Map("SoftmaxCrossEntropyLoss", M + "softmax_cross_entropy_loss", ["ignore_index", "reduction"], outputs: true);
    }
}
