using Shorokoo.Core.Nodes.NodeDefinitions;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// Every operator the framework knows how to compute out of simpler ones, one
/// <see cref="OpLoweringAttribute"/>-marked method each.
///
/// <para>A lowering is ordinary Shorokoo code, written exactly as a <c>[Module]</c> function or an
/// <c>[AutoDiff]</c> gradient rule is: it takes the operator's inputs and attributes and combines
/// them with normal Shorokoo operations. It may therefore branch and loop over ranks or
/// attributes like any other C#. What it must not do is compute: the values it returns are graph
/// values, and which engine evaluates them — the QuickExecutionEngine on the spot, the autodiff
/// engine as nodes to differentiate — is not its concern.</para>
/// </summary>
internal static class OpLowerings
{
    /// <summary>
    /// <c>Softsign(x) = x / (1 + |x|)</c>.
    ///
    /// <para>The <c>1</c> is cast to x's type rather than built at x's type: casting is a runtime
    /// step, so it follows whatever dtype x actually turns out to have. Reading
    /// <c>x.Type</c> in C# instead would fix the constant at whatever the engine's stand-in for x
    /// happened to carry, which on the autodiff path is float32 for every input — a float64
    /// Softsign would then add a float32 one to a float64 magnitude.</para>
    /// </summary>
    [OpLowering(SOFTSIGN)]
    public static Variable?[] Softsign<T>(Tensor<T> x) where T : IVarType
        => [x / (OneLike(x) + x.Abs())];

    private static Tensor<T> OneLike<T>(Tensor<T> like) where T : IVarType
        => OnnxOp.CastLike(Scalar(1.0f), like, saturate: null);
}
