using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo
{
    // Overloads that the value structs need because they share no base type:
    //   * Scalar, Vector and Tensor are separate structs, so generic inference cannot reach a
    //     Tensor<T1> parameter from a Scalar<T1> or Vector<T1> argument (e.g. Pow<T1>(Tensor<T1>)
    //     called with a Scalar<T1>); each needs an explicit overload taking the concrete struct.
    //   * `struct Scalar<T>` cannot declare a member named `Scalar` (CS0542), so the
    //     `.Scalar()` reinterpret is an extension.
    public partial struct Tensor<T> where T : IVarType
    {
        /// <summary>Element-wise power with a scalar exponent (overload so T1 infers from a <see cref="Scalar{T1}"/>).</summary>
        public Tensor<T> Pow<T1>(Scalar<T1> power) where T1 : IVarType => this.Pow<T1>((Tensor<T1>)power);
        /// <summary>Element-wise power with a vector exponent (overload so T1 infers from a <see cref="Vector{T1}"/>).</summary>
        public Tensor<T> Pow<T1>(Vector<T1> power) where T1 : IVarType => this.Pow<T1>((Tensor<T1>)power);
    }

    public partial struct Vector<T> where T : IVarType
    {
        // Vector declares its own Reduce(ReduceKind) -> Scalar, so the op-generator emits none of
        // Tensor's Reduce overloads for it; this is Tensor's axis/keepDims overload.
        // The no-arg call binds to Vector.Reduce(ReduceKind) (exact arity wins).
        public Tensor<T> Reduce(ReduceKind reduceKind, Vector<int64>? axes = null, bool keepDims = false)
            => ((Tensor<T>)this).Reduce(reduceKind, axes, keepDims);

        public Tensor<T> Pow<T1>(Scalar<T1> power) where T1 : IVarType => ((Tensor<T>)this).Pow<T1>((Tensor<T1>)power);
        public Tensor<T> Pow<T1>(Vector<T1> power) where T1 : IVarType => ((Tensor<T>)this).Pow<T1>((Tensor<T1>)power);

        // Vector declares its own Concat(params Vector[]) -> Vector, so the op-generator emits none of
        // Tensor's Concat overloads for it; this is Tensor's axis/params overload.
        public Tensor<T> Concat(long axis, params Tensor<T>[] others)
            => ((Tensor<T>)this).Concat(axis, others);
    }

    public partial struct Scalar<T> where T : IVarType
    {
        /// <summary>Power with a tensor exponent — the base broadcasts.</summary>
        public Tensor<T> Pow<T1>(Tensor<T1> power) where T1 : IVarType => ((Tensor<T>)this).Pow<T1>(power);
        /// <summary>Power with a vector exponent — the base broadcasts (overload so T1 infers from a <see cref="Vector{T1}"/>).</summary>
        public Tensor<T> Pow<T1>(Vector<T1> power) where T1 : IVarType => ((Tensor<T>)this).Pow<T1>((Tensor<T1>)power);
    }

    public static class ValueStructCompatExtensions
    {
        /// <summary>Identity reinterpret of an already-rank-0 scalar, matching <c>Scalar()</c> on the other
        /// value types (a struct named <c>Scalar</c> cannot declare a same-named member, so it is an extension).</summary>
        public static Scalar<T> Scalar<T>(this Scalar<T> self) where T : IVarType => self;

        /// <summary>Reinterpret a handle as a <c>Tensor</c>, matching <c>Variable.Tensor()</c>;
        /// a struct named <c>Tensor</c> cannot declare a same-named member, so it is an extension).</summary>
        public static Tensor<T> Tensor<T>(this Tensor<T> self) where T : IVarType => self;
        public static Tensor<T> Tensor<T>(this Vector<T> self) where T : IVarType => self;
        public static Tensor<T> Tensor<T>(this Scalar<T> self) where T : IVarType => self;

        /// <summary>Element-wise select driven by a scalar condition (the <c>Where</c> extension requires a
        /// <c>Tensor&lt;bit&gt;</c> receiver; a <see cref="Scalar{bit}"/> reaches it via this overload).</summary>
        public static Tensor<V> Where<V>(this Scalar<bit> cond, Tensor<V> whenTrue, Tensor<V> whenFalse)
            where V : IVarType
            => ((Tensor<bit>)cond).Where(whenTrue, whenFalse);
    }
}
