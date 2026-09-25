using System.Globalization;
using System.Text;

namespace Shorokoo.PyTorch.Translation;

/// <summary>Python source literals for the values an ONNX attribute can hold.</summary>
internal static class PyLiteral
{
    public static string Int(long value) => value.ToString(CultureInfo.InvariantCulture);

    public static string Float(double value)
    {
        if (double.IsNaN(value)) return "float('nan')";
        if (double.IsPositiveInfinity(value)) return "float('inf')";
        if (double.IsNegativeInfinity(value)) return "float('-inf')";
        var text = value.ToString("R", CultureInfo.InvariantCulture);
        return text.Contains('.') || text.Contains('E') ? text : text + ".0";
    }

    /// <summary>A Python string literal: single-quoted, with every character that could end it or
    /// break a line escaped, and anything outside printable ASCII as a \u or \U escape.</summary>
    public static string Str(string value)
    {
        var builder = new StringBuilder("'");
        foreach (var rune in value.EnumerateRunes())
        {
            switch (rune.Value)
            {
                case '\\': builder.Append(@"\\"); break;
                case '\'': builder.Append(@"\'"); break;
                case '\n': builder.Append(@"\n"); break;
                case '\r': builder.Append(@"\r"); break;
                case '\t': builder.Append(@"\t"); break;
                case >= 0x20 and < 0x7F: builder.Append((char)rune.Value); break;
                case <= 0xFFFF: builder.Append(CultureInfo.InvariantCulture, $"\\u{rune.Value:x4}"); break;
                default: builder.Append(CultureInfo.InvariantCulture, $"\\U{rune.Value:x8}"); break;
            }
        }
        return builder.Append('\'').ToString();
    }

    public static string List(IEnumerable<string> items) => "[" + string.Join(", ", items) + "]";
}
