using System.Collections.Immutable;
using System.Runtime.InteropServices;
using System.Linq;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Helpers for creating <see cref="RuntimeTensor"/> instances. Centralizes the &gt;256
/// elements rule: data is only retained when the element count is small.
/// </summary>
internal static class RuntimeTensorFactory
{
    /// <summary>
    /// Creates a runtime tensor with known shape and dtype, attaching the reference variable
    /// when provided. Rank and MaxRank are derived from the shape.
    /// </summary>
    public static RuntimeTensor Create(DType dtype, Shape? shape, Variable? reference = null)
    {
        return new RuntimeTensor
        {
            DType = dtype,
            Shape = shape,
            MaxShape = shape,
            Rank = shape?.Dims.Length,
            MaxRank = shape?.Dims.Length,
            ReferenceTensor = reference,
        };
    }

    /// <summary>
    /// Creates a runtime tensor whose dtype (and optionally exact rank) are known but whose
    /// shape is not. Used by ops whose output dims are data-dependent or whose inputs are not
    /// concrete enough at QEE time — per the audit contract the shape must degrade to unknown
    /// rather than carry guessed / negative placeholder dims.
    /// </summary>
    public static RuntimeTensor CreateRankOnly(DType dtype, int? rank)
    {
        return new RuntimeTensor
        {
            DType = dtype,
            Shape = null,
            MaxShape = null,
            Rank = rank,
            MaxRank = rank,
        };
    }

    /// <summary>
    /// Decides whether the given shape is small enough for data to be retained. Returns true if
    /// the total element count is &lt;= <paramref name="maxElements"/>.
    /// </summary>
    public static bool ShouldStoreData(Shape? shape, int maxElements)
    {
        if (shape is null) return false;
        var count = shape.Count;
        return count >= 0 && count <= maxElements;
    }

    /// <summary>
    /// Applies the "data only for small tensors" rule: returns a copy with all data fields
    /// nulled out if the shape is larger than <paramref name="maxElements"/>. Otherwise returns
    /// the input unchanged.
    /// </summary>
    public static RuntimeTensor EnforceDataSizeLimit(RuntimeTensor rt, int maxElements)
    {
        if (ShouldStoreData(rt.Shape, maxElements)) return rt;
        if (rt.FloatData is null && rt.IntData is null && rt.StringData is null && rt.BoolData is null)
            return rt;
        return rt with { FloatData = null, IntData = null, StringData = null, BoolData = null };
    }

    /// <summary>
    /// Narrows an integer tensor's data to its own declared width. QEE holds every integer
    /// width in one <c>long</c> buffer, so an op that computes at 64 bits — a left shift, a
    /// bitwise or/xor of already-widened operands — can leave bits above the declared width
    /// set. Those bits are invisible to a later add (exact mod 2^w either way) but not to a
    /// right shift, which pulls them straight down into the result. Applied to every op's
    /// output, so no op has to remember to narrow; it is idempotent for the ops that already do.
    ///
    /// <para>Unsigned widths land in <c>[0, 2^w)</c> and signed widths sign-extend, matching
    /// how <see cref="TensorDataConverter.ToRuntimeTensor"/> loads them. <c>Int64</c> and
    /// <c>UInt64</c> are the buffer's own width and pass through untouched — which also means a
    /// <c>UInt64</c> value above <c>long.MaxValue</c> stays a negative bit-pattern long, and the
    /// kernels that use signed C# operators on it (Div, Mod, Less/Greater, Sign, Abs) read it as
    /// negative. Narrowing gives the sub-64-bit unsigned widths correct signed-operator behaviour
    /// for free; UInt64 is the one gap, tracked separately.</para>
    /// </summary>
    public static RuntimeTensor NarrowToDeclaredWidth(RuntimeTensor rt)
    {
        if (rt.IntData is not { } d || d.Length == 0) return rt;

        System.Func<long, long>? narrow = null;
        var t = rt.DType;
        if (t == DType.Int32) narrow = v => unchecked((int)v);
        else if (t == DType.Int16) narrow = v => unchecked((short)v);
        else if (t == DType.Int8) narrow = v => unchecked((sbyte)v);
        else if (t == DType.UInt32) narrow = v => unchecked((uint)v);
        else if (t == DType.UInt16) narrow = v => unchecked((ushort)v);
        else if (t == DType.UInt8) narrow = v => unchecked((byte)v);
        if (narrow is null) return rt;

        long[]? buf = null;
        for (int i = 0; i < d.Length; i++)
        {
            var n = narrow(d[i]);
            // Once buf exists the fast path is disabled, so no later index can be skipped.
            if (n == d[i] && buf is null) continue;
            buf ??= d.ToArray();
            buf[i] = n;
        }
        // AsImmutableArray wraps buf in place; ImmutableArray.Create would copy it a second time.
        return buf is null ? rt : rt with { IntData = ImmutableCollectionsMarshal.AsImmutableArray(buf) };
    }

    /// <summary>
    /// <see cref="IRuntimeTensor"/>-aware dispatch that returns a copy of <paramref name="rt"/>
    /// with data-size limit enforced on the plain tensor, the optional's value tensor, or every
    /// element / template tensor of a sequence.
    /// </summary>
    public static IRuntimeTensor EnforceDataSizeLimit(IRuntimeTensor rt, int maxElements)
    {
        return rt switch
        {
            RuntimeTensor plain => EnforceDataSizeLimit(plain, maxElements),
            RuntimeOptionalTensor opt => opt.ValueTensor is null
                ? opt
                : opt with { ValueTensor = EnforceDataSizeLimit(opt.ValueTensor, maxElements) },
            RuntimeSequenceTensor seq => seq with
            {
                Tensors = seq.Tensors is { } ts
                    ? ts.Select(t => EnforceDataSizeLimit(t, maxElements)).ToImmutableArray()
                    : null,
                TemplateTensor = seq.TemplateTensor is null
                    ? null
                    : EnforceDataSizeLimit(seq.TemplateTensor, maxElements),
            },
            _ => rt,
        };
    }
}
