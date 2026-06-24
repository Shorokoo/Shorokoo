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
        public static A IfElse<A>(Scalar<bit> condition, A aWhenTrue, A aWhenFalse) where A : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            return (A)OnnxOp.IfClose([aWhenTrue], [aWhenFalse], ifOpen)[0];
        }

        /// <summary>
        /// Lazy <c>IfElse</c>: each branch expression is built <b>inside</b> the If scope (the
        /// delegate runs between the open and close), so it lowers to an ONNX <c>If</c> subgraph and
        /// is only evaluated when its branch is taken. Use this when a branch contains an operation
        /// that is invalid off-branch — most importantly <c>OptionalGetElement</c> on an optional
        /// that may be absent (eagerly unwrapping an absent optional is a runtime error).
        /// </summary>
        public static A IfElse<A>(Scalar<bit> condition, System.Func<A> whenTrue, System.Func<A> whenFalse) where A : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var t = whenTrue();
            var f = whenFalse();
            return (A)OnnxOp.IfClose([t], [f], ifOpen)[0];
        }

        public static (A, B) IfElse<A, B>(Scalar<bit> condition, (A a, B b) whenTrue, (A a, B b) whenFalse)
            where A : IVariable
            where B : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b], [whenFalse.a, whenFalse.b], ifOpen);

            return ((A)retval[0], (B)retval[1]);
        }

        public static (A, B, C) IfElse<A, B, C>(Scalar<bit> condition, (A a, B b, C c) whenTrue, (A a, B b, C c) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b, whenTrue.c], [whenFalse.a, whenFalse.b, whenFalse.c], ifOpen);

            return ((A)retval[0], (B)retval[1], (C)retval[2]);
        }

        public static (A, B, C, D) IfElse<A, B, C, D>(Scalar<bit> condition, (A a, B b, C c, D d) whenTrue, (A a, B b, C c, D d) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b, whenTrue.c, whenTrue.d], [whenFalse.a, whenFalse.b, whenFalse.c, whenFalse.d], ifOpen);

            return ((A)retval[0], (B)retval[1], (C)retval[2], (D)retval[3]);
        }

        public static (A, B, C, D, E) IfElse<A, B, C, D, E>(Scalar<bit> condition, (A a, B b, C c, D d, E e) whenTrue, (A a, B b, C c, D d, E e) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            where E : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b, whenTrue.c, whenTrue.d, whenTrue.e], [whenFalse.a, whenFalse.b, whenFalse.c, whenFalse.d, whenFalse.e], ifOpen);

            return ((A)retval[0], (B)retval[1], (C)retval[2], (D)retval[3], (E)retval[4]);
        }

        public static (A, B, C, D, E, F) IfElse<A, B, C, D, E, F>(Scalar<bit> condition, (A a, B b, C c, D d, E e, F f) whenTrue, (A a, B b, C c, D d, E e, F f) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            where E : IVariable
            where F : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b, whenTrue.c, whenTrue.d, whenTrue.e, whenTrue.f], [whenFalse.a, whenFalse.b, whenFalse.c, whenFalse.d, whenFalse.e, whenFalse.f], ifOpen);

            return ((A)retval[0], (B)retval[1], (C)retval[2], (D)retval[3], (E)retval[4], (F)retval[5]);
        }

        public static (A, B, C, D, E, F, G) IfElse<A, B, C, D, E, F, G>(Scalar<bit> condition, (A a, B b, C c, D d, E e, F f, G g) whenTrue, (A a, B b, C c, D d, E e, F f, G g) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            where E : IVariable
            where F : IVariable
            where G : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b, whenTrue.c, whenTrue.d, whenTrue.e, whenTrue.f, whenTrue.g], [whenFalse.a, whenFalse.b, whenFalse.c, whenFalse.d, whenFalse.e, whenFalse.f, whenFalse.g], ifOpen);

            return ((A)retval[0], (B)retval[1], (C)retval[2], (D)retval[3], (E)retval[4], (F)retval[5], (G)retval[6]);
        }

        public static (A, B, C, D, E, F, G, H) IfElse<A, B, C, D, E, F, G, H>(Scalar<bit> condition, (A a, B b, C c, D d, E e, F f, G g, H h) whenTrue, (A a, B b, C c, D d, E e, F f, G g, H h) whenFalse)
            where A : IVariable
            where B : IVariable
            where C : IVariable
            where D : IVariable
            where E : IVariable
            where F : IVariable
            where G : IVariable
            where H : IVariable
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var retval = OnnxOp.IfClose([whenTrue.a, whenTrue.b, whenTrue.c, whenTrue.d, whenTrue.e, whenTrue.f, whenTrue.g, whenTrue.h], [whenFalse.a, whenFalse.b, whenFalse.c, whenFalse.d, whenFalse.e, whenFalse.f, whenFalse.g, whenFalse.h], ifOpen);

            return ((A)retval[0], (B)retval[1], (C)retval[2], (D)retval[3], (E)retval[4], (F)retval[5], (G)retval[6], (H)retval[7]);
        }

        public static ITensor[] IfElse(Scalar<bit> condition, ITensor[] whenTrue, ITensor[] whenFalse)
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            return (ITensor[])OnnxOp.IfClose(whenTrue, whenFalse, ifOpen);
        }

        public static IVariable[] IfElse(Scalar<bit> condition, IVariable[] whenTrue, IVariable[] whenFalse)
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            return (IVariable[])OnnxOp.IfClose(whenTrue, whenFalse, ifOpen);
        }

        public static Tensor<T> IfElse<T>(Scalar<bit> condition, Tensor<T>[] whenTrue, Tensor<T>[] whenFalse)
            where T : IVarType
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var results = OnnxOp.IfClose(whenTrue, whenFalse, ifOpen);

            return (Tensor<T>)results[0];
        }

        public static IVariable<T> IfElse<T>(Scalar<bit> condition, IVariable<T>[] whenTrue, IVariable<T>[] whenFalse)
            where T : IVarType
        {
            var ifOpen = OnnxOp.IfOpen((Scalar<bit>)condition);
            var results = OnnxOp.IfClose(whenTrue, whenFalse, ifOpen);

            return (IVariable<T>)results[0];
        }

    }
}
