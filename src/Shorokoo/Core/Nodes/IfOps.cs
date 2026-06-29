using Shorokoo.Graph;
using System;
using System.Collections.Generic;
using System.Text;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System.Collections.Immutable;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Training;

namespace Shorokoo.Core.Nodes
{
    public static partial class Ops
    {
        public static A IfElse<A>(Scalar<bit> condition, A aWhenTrue, A aWhenFalse) where A : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            return OnnxOp.IfClose([aWhenTrue.ToVariable()], [aWhenFalse.ToVariable()], ifOpen)[0].ToValue<A>();
        }

        // Graph-side overload: internal callers already hold non-generic Variable nodes (not user
        // handles), so they bind here rather than the IValue-constrained generic above.
        public static Variable IfElse(Scalar<bit> condition, Variable aWhenTrue, Variable aWhenFalse)
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            return OnnxOp.IfClose([aWhenTrue], [aWhenFalse], ifOpen)[0];
        }

        /// <summary>
        /// Lazy <c>IfElse</c>: each branch expression is built <b>inside</b> the If scope (the
        /// delegate runs between the open and close), so it lowers to an ONNX <c>If</c> subgraph and
        /// is only evaluated when its branch is taken. Use this when a branch contains an operation
        /// that is invalid off-branch — most importantly <c>OptionalGetElement</c> on an optional
        /// that may be absent (eagerly unwrapping an absent optional is a runtime error).
        /// </summary>
        public static A IfElse<A>(Scalar<bit> condition, System.Func<A> whenTrue, System.Func<A> whenFalse) where A : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var t = whenTrue();
            var f = whenFalse();
            return OnnxOp.IfClose([t.ToVariable()], [f.ToVariable()], ifOpen)[0].ToValue<A>();
        }

        public static (A, B) IfElse<A, B>(Scalar<bit> condition, (A a, B b) whenTrue, (A a, B b) whenFalse)
            where A : IValue
            where B : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>());
        }

        public static (A, B, C) IfElse<A, B, C>(Scalar<bit> condition, (A a, B b, C c) whenTrue, (A a, B b, C c) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable(), whenTrue.c.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable(), whenFalse.c.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>(), retval[2].ToValue<C>());
        }

        public static (A, B, C, D) IfElse<A, B, C, D>(Scalar<bit> condition, (A a, B b, C c, D d) whenTrue, (A a, B b, C c, D d) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable(), whenTrue.c.ToVariable(), whenTrue.d.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable(), whenFalse.c.ToVariable(), whenFalse.d.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>(), retval[2].ToValue<C>(), retval[3].ToValue<D>());
        }

        public static (A, B, C, D, E) IfElse<A, B, C, D, E>(Scalar<bit> condition, (A a, B b, C c, D d, E e) whenTrue, (A a, B b, C c, D d, E e) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            where E : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable(), whenTrue.c.ToVariable(), whenTrue.d.ToVariable(), whenTrue.e.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable(), whenFalse.c.ToVariable(), whenFalse.d.ToVariable(), whenFalse.e.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>(), retval[2].ToValue<C>(), retval[3].ToValue<D>(), retval[4].ToValue<E>());
        }

        public static (A, B, C, D, E, F) IfElse<A, B, C, D, E, F>(Scalar<bit> condition, (A a, B b, C c, D d, E e, F f) whenTrue, (A a, B b, C c, D d, E e, F f) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            where E : IValue
            where F : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable(), whenTrue.c.ToVariable(), whenTrue.d.ToVariable(), whenTrue.e.ToVariable(), whenTrue.f.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable(), whenFalse.c.ToVariable(), whenFalse.d.ToVariable(), whenFalse.e.ToVariable(), whenFalse.f.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>(), retval[2].ToValue<C>(), retval[3].ToValue<D>(), retval[4].ToValue<E>(), retval[5].ToValue<F>());
        }

        public static (A, B, C, D, E, F, G) IfElse<A, B, C, D, E, F, G>(Scalar<bit> condition, (A a, B b, C c, D d, E e, F f, G g) whenTrue, (A a, B b, C c, D d, E e, F f, G g) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            where E : IValue
            where F : IValue
            where G : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable(), whenTrue.c.ToVariable(), whenTrue.d.ToVariable(), whenTrue.e.ToVariable(), whenTrue.f.ToVariable(), whenTrue.g.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable(), whenFalse.c.ToVariable(), whenFalse.d.ToVariable(), whenFalse.e.ToVariable(), whenFalse.f.ToVariable(), whenFalse.g.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>(), retval[2].ToValue<C>(), retval[3].ToValue<D>(), retval[4].ToValue<E>(), retval[5].ToValue<F>(), retval[6].ToValue<G>());
        }

        public static (A, B, C, D, E, F, G, H) IfElse<A, B, C, D, E, F, G, H>(Scalar<bit> condition, (A a, B b, C c, D d, E e, F f, G g, H h) whenTrue, (A a, B b, C c, D d, E e, F f, G g, H h) whenFalse)
            where A : IValue
            where B : IValue
            where C : IValue
            where D : IValue
            where E : IValue
            where F : IValue
            where G : IValue
            where H : IValue
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var retval = OnnxOp.IfClose([whenTrue.a.ToVariable(), whenTrue.b.ToVariable(), whenTrue.c.ToVariable(), whenTrue.d.ToVariable(), whenTrue.e.ToVariable(), whenTrue.f.ToVariable(), whenTrue.g.ToVariable(), whenTrue.h.ToVariable()], [whenFalse.a.ToVariable(), whenFalse.b.ToVariable(), whenFalse.c.ToVariable(), whenFalse.d.ToVariable(), whenFalse.e.ToVariable(), whenFalse.f.ToVariable(), whenFalse.g.ToVariable(), whenFalse.h.ToVariable()], ifOpen);

            return (retval[0].ToValue<A>(), retval[1].ToValue<B>(), retval[2].ToValue<C>(), retval[3].ToValue<D>(), retval[4].ToValue<E>(), retval[5].ToValue<F>(), retval[6].ToValue<G>(), retval[7].ToValue<H>());
        }

        public static Variable[] IfElse(Scalar<bit> condition, IValue[] whenTrue, IValue[] whenFalse)
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            return OnnxOp.IfClose([.. whenTrue.Select(v => v.ToVariable())], [.. whenFalse.Select(v => v.ToVariable())], ifOpen);
        }

        public static Tensor<T> IfElse<T>(Scalar<bit> condition, Tensor<T>[] whenTrue, Tensor<T>[] whenFalse)
            where T : IVarType
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var results = OnnxOp.IfClose([.. whenTrue.Select(t => t.ToVariable())], [.. whenFalse.Select(t => t.ToVariable())], ifOpen);

            return results[0];
        }

        public static IValue<T> IfElse<T>(Scalar<bit> condition, IValue<T>[] whenTrue, IValue<T>[] whenFalse)
            where T : IVarType
        {
            var ifOpen = OnnxOp.IfOpen(condition);
            var results = OnnxOp.IfClose([.. whenTrue.Select(t => t.ToVariable())], [.. whenFalse.Select(t => t.ToVariable())], ifOpen);

            return (IValue<T>)results[0].ToValue();
        }

    }
}
