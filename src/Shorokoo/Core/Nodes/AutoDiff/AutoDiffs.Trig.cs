using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal static partial class AutoDiffs
    {
        // ===== Sin =====

        [AutoDiff(SIN)]
        public static IVariable?[] Sin<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(sin(x))/dx = cos(x)
            return [grad * x.Cos()];
        }

        // ===== Cos =====

        [AutoDiff(COS)]
        public static IVariable?[] Cos<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(cos(x))/dx = -sin(x)
            return [-(grad * x.Sin())];
        }

        // ===== Tan =====

        [AutoDiff(TAN)]
        public static IVariable?[] Tan<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(tan(x))/dx = 1/cos²(x) = sec²(x)
            var cosX = x.Cos();
            return [grad / (cosX * cosX)];
        }

        // ===== Asin =====

        [AutoDiff(ASIN)]
        public static IVariable?[] Asin<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(asin(x))/dx = 1/√(1-x²)
            var one = TypedConst(1.0f, x);
            return [grad / (one - x * x).Sqrt()];
        }

        // ===== Acos =====

        [AutoDiff(ACOS)]
        public static IVariable?[] Acos<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(acos(x))/dx = -1/√(1-x²)
            var one = TypedConst(1.0f, x);
            return [-(grad / (one - x * x).Sqrt())];
        }

        // ===== Atan =====

        [AutoDiff(ATAN)]
        public static IVariable?[] Atan<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(atan(x))/dx = 1/(1+x²)
            var one = TypedConst(1.0f, x);
            return [grad / (one + x * x)];
        }

        // ===== Sinh =====

        [AutoDiff(SINH)]
        public static IVariable?[] Sinh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(sinh(x))/dx = cosh(x)
            return [grad * x.Cosh()];
        }

        // ===== Cosh =====

        [AutoDiff(COSH)]
        public static IVariable?[] Cosh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(cosh(x))/dx = sinh(x)
            return [grad * x.Sinh()];
        }

        // ===== Asinh =====

        [AutoDiff(ASINH)]
        public static IVariable?[] Asinh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(asinh(x))/dx = 1/√(x²+1)
            var one = TypedConst(1.0f, x);
            return [grad / (x * x + one).Sqrt()];
        }

        // ===== Acosh =====

        [AutoDiff(ACOSH)]
        public static IVariable?[] Acosh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(acosh(x))/dx = 1/√(x²-1)
            var one = TypedConst(1.0f, x);
            return [grad / (x * x - one).Sqrt()];
        }

        // ===== Atanh =====

        [AutoDiff(ATANH)]
        public static IVariable?[] Atanh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(atanh(x))/dx = 1/(1-x²)
            var one = TypedConst(1.0f, x);
            return [grad / (one - x * x)];
        }
    }
}
