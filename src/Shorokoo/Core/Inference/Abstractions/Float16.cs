using System;
using System.Runtime.InteropServices;

namespace Shorokoo.Core.Inference.Abstractions;

// IEEE 754 binary16. Two-byte layout matches both ORT's
// Microsoft.ML.OnnxRuntime.Tensors.Float16 and System.Half, so spans can be
// reinterpret-cast across all three.
[StructLayout(LayoutKind.Sequential)]
public readonly struct Float16 : IEquatable<Float16>
{
    private readonly ushort _bits;

    public Float16(ushort bits) { _bits = bits; }

    public ushort Bits => _bits;

    public static Float16 Zero => new(0x0000);
    public static Float16 One => new(0x3C00);
    public static Float16 NegativeOne => new(0xBC00);
    public static Float16 NaN => new(0x7E00);
    public static Float16 MaxValue => new(0x7BFF);
    public static Float16 MinValue => new(0xFBFF);
    public static Float16 Epsilon => new(0x0001);
    public static Float16 PositiveInfinity => new(0x7C00);
    public static Float16 NegativeInfinity => new(0xFC00);

    public static explicit operator float(Float16 value) =>
        (float)BitConverter.UInt16BitsToHalf(value._bits);

    public static explicit operator Float16(float value) =>
        new(BitConverter.HalfToUInt16Bits((Half)value));

    public bool Equals(Float16 other) => _bits == other._bits;
    public override bool Equals(object? obj) => obj is Float16 f && Equals(f);
    public override int GetHashCode() => _bits.GetHashCode();
    public override string ToString() => ((float)this).ToString();

    public static bool operator ==(Float16 a, Float16 b) => a._bits == b._bits;
    public static bool operator !=(Float16 a, Float16 b) => a._bits != b._bits;
}
