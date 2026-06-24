using System;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Inference.Abstractions;

// Google brain float16: the top 16 bits of an IEEE 754 binary32. Two-byte
// layout matches ORT's Microsoft.ML.OnnxRuntime.Tensors.BFloat16, so spans
// reinterpret-cast freely between the two.
[StructLayout(LayoutKind.Sequential)]
public readonly struct BFloat16 : IEquatable<BFloat16>
{
    private readonly ushort _bits;

    public BFloat16(ushort bits) { _bits = bits; }

    public ushort Bits => _bits;

    public static BFloat16 Zero => new(0x0000);
    public static BFloat16 One => new(0x3F80);
    public static BFloat16 NegativeOne => new(0xBF80);
    public static BFloat16 NaN => new(0x7FC0);
    public static BFloat16 MaxValue => new(0x7F7F);
    public static BFloat16 MinValue => new(0xFF7F);
    public static BFloat16 Epsilon => new(0x0080);
    public static BFloat16 PositiveInfinity => new(0x7F80);
    public static BFloat16 NegativeInfinity => new(0xFF80);

    public static explicit operator float(BFloat16 value)
    {
        uint upper = (uint)value._bits << 16;
        return BitConverter.UInt32BitsToSingle(upper);
    }

    public static explicit operator BFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        // NaN: preserve a NaN encoding without dropping the payload entirely.
        if (float.IsNaN(value)) return NaN;
        // Round-to-nearest-even: bias by (0x7FFF + lsb) before truncation.
        uint lsb = (bits >> 16) & 1u;
        uint rounded = bits + 0x7FFFu + lsb;
        return new BFloat16((ushort)(rounded >> 16));
    }

    public bool Equals(BFloat16 other) => _bits == other._bits;
    public override bool Equals(object? obj) => obj is BFloat16 b && Equals(b);
    public override int GetHashCode() => _bits.GetHashCode();
    public override string ToString() => ((float)this).ToString();

    public static bool operator ==(BFloat16 a, BFloat16 b) => a._bits == b._bits;
    public static bool operator !=(BFloat16 a, BFloat16 b) => a._bits != b._bits;
}
