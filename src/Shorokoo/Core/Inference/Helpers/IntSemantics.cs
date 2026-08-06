namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Signedness-aware primitives for the QuickExecutionEngine's single <c>long</c> integer buffer.
///
/// <para>QEE holds every integer width in that one buffer, so a <c>UInt64</c> above
/// <c>long.MaxValue</c> is stored as a negative bit-pattern long. The representation round-trips
/// exactly, but any kernel whose result depends on the sign — divide, modulo, ordering, magnitude,
/// conversion to float — must reinterpret through <c>ulong</c> first. These helpers are that
/// reinterpretation; kernels take the <c>unsigned</c> flag from
/// <see cref="DTypeHelpers.IsUnsignedInt(DType)"/> on the operand dtype.</para>
///
/// <para>The sub-64-bit unsigned widths land in <c>[0, 2^w)</c> via
/// <see cref="RuntimeTensorFactory.NarrowToDeclaredWidth(RuntimeTensor)"/>, where signed and
/// unsigned operators already agree, so routing them here changes nothing.</para>
/// </summary>
internal static class IntSemantics
{
    /// <summary>Reinterprets a buffer lane as unsigned.</summary>
    public static ulong U(long v) => unchecked((ulong)v);

    /// <summary>Stores an unsigned value back into the buffer's signed lane.</summary>
    public static long S(ulong v) => unchecked((long)v);

    /// <summary>Converts a lane to its true numeric magnitude.</summary>
    public static double ToDouble(bool unsigned, long v) => unsigned ? U(v) : v;

    /// <summary>Orders two lanes under the dtype's own signedness.</summary>
    public static int Compare(bool unsigned, long a, long b)
        => unsigned ? U(a).CompareTo(U(b)) : a.CompareTo(b);

    public static long Max(bool unsigned, long a, long b) => Compare(unsigned, a, b) >= 0 ? a : b;

    public static long Min(bool unsigned, long a, long b) => Compare(unsigned, a, b) <= 0 ? a : b;

    /// <summary><see cref="Compare"/> as an <see cref="IComparer{T}"/>, for the sorting kernels.</summary>
    public static IComparer<long> Comparer(bool unsigned)
        => unsigned ? UnsignedComparer.Instance : System.Collections.Generic.Comparer<long>.Default;

    /// <summary>
    /// The truncation to <paramref name="dtype"/>'s declared width, or null for the widths that
    /// are the buffer's own (Int64/UInt64) and the non-integer dtypes. Single definition of
    /// "declared width", shared by the per-op narrowing tail
    /// (<see cref="RuntimeTensorFactory.NarrowToDeclaredWidth(RuntimeTensor)"/>) and by the
    /// kernels that must re-enter the width mid-computation.
    /// </summary>
    public static Func<long, long>? Narrower(DType dtype)
    {
        if (dtype == DType.Int32) return v => unchecked((int)v);
        if (dtype == DType.Int16) return v => unchecked((short)v);
        if (dtype == DType.Int8) return v => unchecked((sbyte)v);
        if (dtype == DType.UInt32) return v => unchecked((uint)v);
        if (dtype == DType.UInt16) return v => unchecked((ushort)v);
        if (dtype == DType.UInt8) return v => unchecked((byte)v);
        return null;
    }

    /// <summary>
    /// Truncates a single value to <paramref name="dtype"/>'s declared width. Needed wherever a
    /// kernel applies an operation that does <em>not</em> commute with that truncation — division
    /// and ordering — to an intermediate it computed itself in the 64-bit buffer. Addition,
    /// subtraction and multiplication do commute, so they can overflow freely and let the op's
    /// tail narrow the result.
    /// </summary>
    public static long NarrowToWidth(DType dtype, long value)
        => Narrower(dtype) is { } narrow ? narrow(value) : value;

    private sealed class UnsignedComparer : IComparer<long>
    {
        public static readonly UnsignedComparer Instance = new();
        public int Compare(long a, long b) => U(a).CompareTo(U(b));
    }
}
