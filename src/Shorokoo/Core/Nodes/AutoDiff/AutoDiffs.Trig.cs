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
        public static Variable?[] Sin<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(sin(x))/dx = cos(x)
            return [grad * x.Cos()];
        }

        // ===== Cos =====

        [AutoDiff(COS)]
        public static Variable?[] Cos<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(cos(x))/dx = -sin(x)
            return [-(grad * x.Sin())];
        }

        // ===== Tan =====

        [AutoDiff(TAN)]
        public static Variable?[] Tan<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(tan(x))/dx = 1/cos²(x) = sec²(x)
            var cosX = x.Cos();
            return [grad / (cosX * cosX)];
        }

        // ===== Asin =====

        [AutoDiff(ASIN)]
        public static Variable?[] Asin<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(asin(x))/dx = 1/√(1-x²)
            var one = TypedConst(1.0f, x);
            return [grad / (one - x * x).Sqrt()];
        }

        // ===== Acos =====

        [AutoDiff(ACOS)]
        public static Variable?[] Acos<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(acos(x))/dx = -1/√(1-x²)
            var one = TypedConst(1.0f, x);
            return [-(grad / (one - x * x).Sqrt())];
        }

        // ===== Atan =====

        [AutoDiff(ATAN)]
        public static Variable?[] Atan<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(atan(x))/dx = 1/(1+x²)
            var one = TypedConst(1.0f, x);
            return [grad / (one + x * x)];
        }

        // ===== Sinh =====

        [AutoDiff(SINH)]
        public static Variable?[] Sinh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(sinh(x))/dx = cosh(x)
            return [grad * x.Cosh()];
        }

        // ===== Cosh =====

        [AutoDiff(COSH)]
        public static Variable?[] Cosh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(cosh(x))/dx = sinh(x)
            return [grad * x.Sinh()];
        }

        // ===== Asinh =====

        [AutoDiff(ASINH)]
        public static Variable?[] Asinh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(asinh(x))/dx = 1/√(x²+1)
            var one = TypedConst(1.0f, x);
            return [grad / (x * x + one).Sqrt()];
        }

        // ===== Acosh =====

        [AutoDiff(ACOSH)]
        public static Variable?[] Acosh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(acosh(x))/dx = 1/√(x²-1)
            var one = TypedConst(1.0f, x);
            return [grad / (x * x - one).Sqrt()];
        }

        // ===== Atanh =====

        [AutoDiff(ATANH)]
        public static Variable?[] Atanh<T>(Tensor<T> x, Tensor<T> grad) where T : IVarType
        {
            // d(atanh(x))/dx = 1/(1-x²)
            var one = TypedConst(1.0f, x);
            return [grad / (one - x * x)];
        }
    }
}
