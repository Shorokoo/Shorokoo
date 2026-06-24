using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// Encodes/decodes the ONNX <c>op_type</c> / <see cref="IR.FunctionProto.Name"/> used to
    /// emit a Shorokoo <see cref="Function"/> call so it never collides with a built-in
    /// ONNX operator name.
    ///
    /// <para>
    /// A Shorokoo function is emitted as a NodeProto whose <c>op_type</c> is the function's name
    /// in the <c>"Functions"</c> domain, paired with a same-named <see cref="IR.FunctionProto"/>.
    /// ONNX Runtime, however, resolves a node whose <c>op_type</c> matches a built-in operator to
    /// that built-in <em>regardless of the node's domain</em> (its constant-folding optimizer keys
    /// on the bare op_type). So a user function named after a standard op — e.g. the
    /// <c>Constant</c> initializer, which collides with the ONNX <c>Constant</c> operator — is
    /// dispatched to the built-in, which takes no inputs and requires a <c>value</c> attribute the
    /// call node lacks, yielding a corrupt (String, degenerate-shape) result instead of the
    /// function's real output.
    /// </para>
    ///
    /// <para>
    /// To avoid the collision we prefix colliding names with <see cref="Prefix"/> on emit. The
    /// prefix is itself not a built-in op name, so encoding is idempotent. <see cref="Decode"/>
    /// restores the original name on load, so a function's <see cref="Function.DefaultName"/>
    /// round-trips byte-for-byte. Names that don't collide are emitted unchanged, so the on-disk
    /// shape of every existing model (none of whose function names collide) is unaffected.
    /// </para>
    /// </summary>
    internal static class OnnxFunctionName
    {
        /// <summary>Marker prepended to a function name that collides with a built-in ONNX op.
        /// Chosen to be illegal as a built-in op name (contains a '.') so it can never itself
        /// collide and so <see cref="Encode"/> is idempotent.</summary>
        public const string Prefix = "shrk.fn.";

        /// <summary>True when <paramref name="name"/> matches a registered (built-in or internal)
        /// op code — i.e. emitting it verbatim as an ONNX function op_type would shadow that op.</summary>
        private static bool CollidesWithBuiltinOp(string name) =>
            Definitions.NodeDefinitions.ContainsKey(name);

        /// <summary>Maps a function's <c>DefaultName</c> to the <c>op_type</c> /
        /// <c>FunctionProto.Name</c> used in the emitted ONNX model: prefixed when it would
        /// collide with a built-in op, unchanged otherwise.</summary>
        public static string Encode(string functionName) =>
            CollidesWithBuiltinOp(functionName) ? Prefix + functionName : functionName;

        /// <summary>Inverse of <see cref="Encode"/>: strips the collision-avoidance prefix so the
        /// reconstructed function <c>DefaultName</c> equals the original name.</summary>
        public static string Decode(string emittedName) =>
            emittedName.StartsWith(Prefix, System.StringComparison.Ordinal)
                ? emittedName.Substring(Prefix.Length)
                : emittedName;
    }
}
