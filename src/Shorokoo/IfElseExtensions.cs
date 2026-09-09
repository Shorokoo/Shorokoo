using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using System.Collections.Immutable;
using System.Linq;

namespace Shorokoo
{
    /// <summary>
    /// Extension-method form of <c>IfElse</c> on <see cref="Scalar{bit}"/>, enabling
    /// the natural call syntax <c>condition.IfElse(whenTrue, whenFalse)</c>. Each overload
    /// forwards to the corresponding non-extension overload on
    /// <c>Shorokoo.Core.Nodes.Ops</c>, which holds the implementation. Lives in
    /// <see cref="Shorokoo"/> so user-facing code only needs <c>using Shorokoo;</c>.
    ///
    /// <para>A branch expression is an ordinary argument, so it is traced before the
    /// <c>IfElse</c> that selects it. Lowering then moves everything that serves one branch
    /// and nothing else inside that branch, so it runs only when the condition picks it —
    /// see
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastIfBranchScoper"/>. Callers
    /// therefore write both branches as plain expressions, whatever they contain: a whole
    /// loop, a nested <c>IfElse</c>, or an operation only valid on its own path, such as
    /// unwrapping an <c>OptionalTensor</c> that is absent on the other.</para>
    /// </summary>
    public static class IfElseExtensions
    {
        public static A IfElse<A>(this Scalar<bit> condition, A aWhenTrue, A aWhenFalse) where A : IValue
            => Ops.IfElse(condition, aWhenTrue, aWhenFalse);

        public static (A, B) IfElse<A, B>(this Scalar<bit> condition, (A a, B b) whenTrue, (A a, B b) whenFalse)
            where A : IValue
            where B : IValue
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C) IfElse<A, B, C>(this Scalar<bit> condition, (A a, B b, C c) whenTrue, (A a, B b, C c) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C, D) IfElse<A, B, C, D>(this Scalar<bit> condition, (A a, B b, C c, D d) whenTrue, (A a, B b, C c, D d) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C, D, E) IfElse<A, B, C, D, E>(this Scalar<bit> condition, (A a, B b, C c, D d, E e) whenTrue, (A a, B b, C c, D d, E e) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            where E : IValue
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C, D, E, F) IfElse<A, B, C, D, E, F>(this Scalar<bit> condition, (A a, B b, C c, D d, E e, F f) whenTrue, (A a, B b, C c, D d, E e, F f) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            where E : IValue
            where F : IValue
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static ImmutableArray<Variable?> IfElse(this Scalar<bit> condition, ImmutableArray<Variable?> whenTrue, ImmutableArray<Variable?> whenFalse)
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            return OnnxOp.IfClose(whenTrue.AssertNotNulls().ToArray(), whenFalse.AssertNotNulls().ToArray(), ifOpen).Select(x => (Variable?)x).ToImmutableArray();
        }
    }
}
