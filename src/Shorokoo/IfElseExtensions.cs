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
    /// </summary>
    public static class IfElseExtensions
    {
        public static A IfElse<A>(this Scalar<bit> condition, A aWhenTrue, A aWhenFalse) where A : IVariable
            => Ops.IfElse(condition, aWhenTrue, aWhenFalse);

        /// <summary>
        /// Lazy <c>IfElse</c>: each branch is built inside the If scope (lowering to an ONNX <c>If</c>
        /// subgraph), so a branch may safely contain an op that is invalid off-branch — e.g.
        /// <c>optional.TensorValue()</c> when the optional may be absent.
        /// </summary>
        public static A IfElse<A>(this Scalar<bit> condition, System.Func<A> whenTrue, System.Func<A> whenFalse) where A : IVariable
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B) IfElse<A, B>(this Scalar<bit> condition, (A a, B b) whenTrue, (A a, B b) whenFalse)
            where A : IVariable
            where B : IVariable
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C) IfElse<A, B, C>(this Scalar<bit> condition, (A a, B b, C c) whenTrue, (A a, B b, C c) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C, D) IfElse<A, B, C, D>(this Scalar<bit> condition, (A a, B b, C c, D d) whenTrue, (A a, B b, C c, D d) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C, D, E) IfElse<A, B, C, D, E>(this Scalar<bit> condition, (A a, B b, C c, D d, E e) whenTrue, (A a, B b, C c, D d, E e) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            where E : IVariable
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static (A, B, C, D, E, F) IfElse<A, B, C, D, E, F>(this Scalar<bit> condition, (A a, B b, C c, D d, E e, F f) whenTrue, (A a, B b, C c, D d, E e, F f) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            where E : IVariable
            where F : IVariable
            => Ops.IfElse(condition, whenTrue, whenFalse);

        public static ImmutableArray<IVariable?> IfElse(this Scalar<bit> condition, ImmutableArray<IVariable?> whenTrue, ImmutableArray<IVariable?> whenFalse)
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            return OnnxOp.IfClose(whenTrue.AssertNotNulls().ToArray(), whenFalse.AssertNotNulls().ToArray(), ifOpen).Select(x => (IVariable?)x).ToImmutableArray();
        }
    }
}
