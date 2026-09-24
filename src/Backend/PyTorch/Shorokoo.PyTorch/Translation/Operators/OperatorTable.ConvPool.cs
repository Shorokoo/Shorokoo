namespace Shorokoo.PyTorch.Translation.Operators;

// Convolution and pooling; the semantics go in shorokoo_torch/ops_conv_pool.py. None is translated yet, so a model
// using one is refused when its session is created, naming the operator.
internal static partial class OperatorTable
{
    static partial void RegisterConvPool(Registry table)
    {
    }
}
