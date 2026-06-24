using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Inference.Helpers;

/// <summary>
/// Tolerant attribute readers. ONNX attributes coming through Shorokoo's pipeline may be
/// represented as either the C#-side type (e.g., <see cref="bool"/>) or the proto-side type
/// (e.g., <see cref="long"/>). These helpers accept either and fall back to a default when the
/// attribute is missing or the value is null.
/// </summary>
internal static class AttrAccess
{
    public static bool GetBool(OnnxCSharpAttributes attrs, string name, bool defaultValue = false)
    {
        if (!attrs.IsAttributeDefined(name)) return defaultValue;
        var obj = attrs.GetAttributeObj(name);
        return obj switch
        {
            bool b => b,
            long l => l != 0,
            _ => defaultValue,
        };
    }

    public static long GetLong(OnnxCSharpAttributes attrs, string name, long defaultValue = 0)
    {
        if (!attrs.IsAttributeDefined(name)) return defaultValue;
        var obj = attrs.GetAttributeObj(name);
        return obj switch
        {
            long l => l,
            _ => defaultValue,
        };
    }

    public static long[]? GetLongs(OnnxCSharpAttributes attrs, string name)
    {
        if (!attrs.IsAttributeDefined(name)) return null;
        return attrs.GetLongsVal(name);
    }

    public static float GetFloat(OnnxCSharpAttributes attrs, string name, float defaultValue = 0f)
    {
        if (!attrs.IsAttributeDefined(name)) return defaultValue;
        var obj = attrs.GetAttributeObj(name);
        return obj switch
        {
            float f => f,
            double d => (float)d,
            long l => l,
            _ => defaultValue,
        };
    }
}
