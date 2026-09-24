namespace Shorokoo.PyTorch.Translation.Operators;

// Random draws, and Dropout, which draws its mask; the semantics are in shorokoo_torch/ops_random.py.
// A draw carries no gradient; Dropout's output carries its input's through the mask.
internal static partial class OperatorTable
{
    static partial void RegisterRandom(Registry table)
    {
        const string M = "ops_random.";
        const TorchGradient none = TorchGradient.NotDifferentiable;
        table.Map("RandomNormal", M + "random_normal", ["shape", "dtype", "mean", "scale", "seed"], gradient: none);
        table.Map("RandomUniform", M + "random_uniform", ["shape", "dtype", "high", "low", "seed"], gradient: none);
        table.Map("RandomNormalLike", M + "random_normal_like", ["dtype", "mean", "scale", "seed"], gradient: none);
        table.Map("RandomUniformLike", M + "random_uniform_like", ["dtype", "high", "low", "seed"], gradient: none);
        table.Map("Bernoulli", M + "bernoulli", ["dtype", "seed"], gradient: none);
        table.Map("Multinomial", M + "multinomial", ["dtype", "sample_size", "seed"], gradient: none);
        table.Map("Dropout", M + "dropout", ["seed", "ratio"], outputs: true);
    }
}
