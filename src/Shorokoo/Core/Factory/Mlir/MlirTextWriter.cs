using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Factory.Mlir
{
    /// <summary>
    /// Prints a <see cref="FastComputationGraph"/> as flat, MLIR-flavored assembly text.
    ///
    /// <para>The text is a near-isomorph of <see cref="FastComputationGraph.Nodes"/>: one op
    /// per node, SSA names carrying <see cref="FastTensorKey"/> identity, and
    /// <c>*#OPEN</c>/<c>*#CLOSE</c> ops delimiting scopes purely by position (no nested
    /// regions). See <c>src/docs/design/mlir-assembly-parser.md</c>. The inverse is
    /// <see cref="MlirTextReader"/>; <c>Read(Write(g))</c> is expected to reproduce the same
    /// printed text.</para>
    ///
    /// <para>Phase 1 scope: structural core plus scalar attribute types (Long/Longs, Float/Floats,
    /// Bool/Bools, String/Strings, DType/DTypes, Enum/Enums). Nodes carrying
    /// <see cref="AttributeType.Tensor"/>, <see cref="AttributeType.Graph"/> or
    /// <see cref="AttributeType.TypeProto"/> attributes, a non-null
    /// <see cref="FastNode.TargetFunction"/>, or graph-attribute input/output groups throw
    /// <see cref="NotSupportedException"/> naming the offending opcode/attribute.</para>
    /// </summary>
    public static class MlirTextWriter
    {
        /// <summary>Prints <paramref name="graph"/> to MLIR-flavored assembly text.</summary>
        public static string Write(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            var sb = new StringBuilder();
            sb.Append("graph {\n");

            WriteDirective(sb, "inputs", graph.Inputs.Select(TensorRef));
            WriteDirective(sb, "outputs", graph.Outputs.Select(TensorRef));
            WriteDirective(sb, "input_names", graph.InputUniqueNames.Select(StringOrHole));
            WriteDirective(sb, "output_names", graph.OutputUniqueNames.Select(StringOrHole));
            if (graph.OutputRankOverrides is not null)
                WriteDirective(sb, "output_rank_overrides",
                    graph.OutputRankOverrides.Select(r => r is null ? "_" : r.Value.ToString(CultureInfo.InvariantCulture)));

            foreach (var node in graph.Nodes)
                WriteNode(sb, node);

            sb.Append("}\n");
            return sb.ToString();
        }

        private static void WriteDirective(StringBuilder sb, string name, IEnumerable<string> items)
            => sb.Append("  ").Append(name).Append(" = [").Append(string.Join(", ", items)).Append("]\n");

        private static void WriteNode(StringBuilder sb, FastNode node)
        {
            RequireSingleDefaultGroup(node.FullInputs, node, "inputs");
            RequireSingleDefaultGroup(node.FullOutputs, node, "outputs");
            if (node.TargetFunction is not null)
                throw new NotSupportedException(
                    $"MlirTextWriter: node '{node.OpCode}' ({node.Key}) has a TargetFunction, which Phase 1 does not serialize.");

            sb.Append("  ").Append(NodeRef(node.Key)).Append(" = ").Append('"').Append(node.OpCode).Append('"');

            sb.Append('(').Append(string.Join(", ", SlotRefs(node.FullInputs))).Append(')');
            sb.Append(" -> (").Append(string.Join(", ", SlotRefs(node.FullOutputs))).Append(')');

            var attrText = WriteAttributes(node);
            if (attrText.Length > 0)
                sb.Append(' ').Append(attrText);

            if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                sb.Append(" open ").Append(NodeRef(openKey));
            if (node.FriendlyName is not null)
                sb.Append(" name ").Append(Quote(node.FriendlyName));
            if (node.IdentifierTemplate is not null)
                sb.Append(" template ").Append(Quote(node.IdentifierTemplate));

            sb.Append('\n');
        }

        private static IEnumerable<string> SlotRefs(Dictionary<string, List<FastTensorKey?>> group)
        {
            // RequireSingleDefaultGroup has already guaranteed a single "" group (or none).
            if (group.Count == 0) return [];
            return group[string.Empty].Select(k => k is null || k.Value.IsEmpty ? "_" : TensorRef(k.Value));
        }

        private static string WriteAttributes(FastNode node)
        {
            // Iterate the stable AttributeDefs order (not the hash-ordered value dict) so output is
            // deterministic and Read(Write(g)) reprints identically.
            var vals = node.Attributes.GetAttributeVals();
            var parts = new List<string>();

            foreach (var def in node.Attributes.AttributeDefs)
            {
                if (!vals.TryGetValue(def.AttributeName, out var value) || value is null) continue; // absent/default
                parts.Add($"{Quote(def.AttributeName)} = {WriteAttributeValue(node.OpCode, def, value)}");
            }

            if (parts.Count == 0) return string.Empty;
            return "{" + string.Join(", ", parts) + "}";
        }

        private static string WriteAttributeValue(string opCode, NodeDefAttributeDef def, object value) => def.Type switch
        {
            AttributeType.Long => $"{Convert.ToInt64(value)} : i64",
            AttributeType.Longs => $"[{string.Join(", ", ((long[])value))}] : i64",
            AttributeType.Float => $"{Float((float)value)} : f32",
            AttributeType.Floats => $"[{string.Join(", ", ((float[])value).Select(Float))}] : f32",
            AttributeType.Bool => (bool)value ? "true" : "false",
            AttributeType.Bools => $"[{string.Join(", ", ((bool[])value).Select(b => b ? "true" : "false"))}]",
            AttributeType.String => Quote((string)value),
            AttributeType.Strings => $"[{string.Join(", ", ((string[])value).Select(Quote))}]",
            AttributeType.DType => DTypeRef((DType)value),
            AttributeType.DTypes => $"[{string.Join(", ", ((DType[])value).Select(DTypeRef))}]",
            AttributeType.Enum => EnumRef(def, value),
            AttributeType.Enums => $"[{string.Join(", ", ((System.Collections.IEnumerable)value).Cast<object>().Select(v => EnumRef(def, v)))}]",
            AttributeType.Tensor => DenseTensor((TensorData)value),
            _ => throw new NotSupportedException(
                $"MlirTextWriter: attribute '{def.AttributeName}' on node '{opCode}' has type {def.Type}, which Phase 1 does not serialize.")
        };

        private static string EnumRef(NodeDefAttributeDef def, object value)
        {
            if (def.EnumDef is null)
                throw new NotSupportedException($"MlirTextWriter: enum attribute '{def.AttributeName}' has no EnumDef.");
            return $"enum<{def.EnumDef.EnumType.Name}, {def.EnumDef.ToOnnxName(value)}>";
        }

        private static string DTypeRef(DType dtype) => $"dtype<{dtype.ProtoTypeNum.ToString(CultureInfo.InvariantCulture)}>";

        private static string DenseTensor(TensorData td)
        {
            var dims = (long[])td.Shape;
            var bytes = MlirTensorCodec.ToRawBytes(td);
            return $"dense<[{string.Join(", ", dims)}], {DTypeRef(td.DType)}, {Quote(Convert.ToBase64String(bytes))}>";
        }

        private static string NodeRef(FastNodeKey key) => "%N" + key.Id.ToString(CultureInfo.InvariantCulture);

        private static string TensorRef(FastTensorKey key)
            => "%N" + key.FastNodeKey.Id.ToString(CultureInfo.InvariantCulture)
               + "_T" + key.OutputIndex.ToString(CultureInfo.InvariantCulture);

        private static string StringOrHole(string? s) => s is null ? "_" : Quote(s);

        private static string Float(float f)
        {
            // Round-trippable ("R") form; guarantee a decimal point / exponent so the token
            // reads back as a float rather than an integer.
            var s = f.ToString("R", CultureInfo.InvariantCulture);
            if (s.IndexOf('.') < 0 && s.IndexOf('E') < 0 && s.IndexOf('e') < 0
                && s != "NaN" && s != "Infinity" && s != "-Infinity")
                s += ".0";
            return s;
        }

        private static string Quote(string s)
        {
            var sb = new StringBuilder(s.Length + 2);
            sb.Append('"');
            foreach (var c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default: sb.Append(c); break;
                }
            }
            sb.Append('"');
            return sb.ToString();
        }

        private static void RequireSingleDefaultGroup(Dictionary<string, List<FastTensorKey?>> groups, FastNode node, string which)
        {
            if (groups.Count == 0) return;
            if (groups.Count > 1 || !groups.ContainsKey(string.Empty))
                throw new NotSupportedException(
                    $"MlirTextWriter: node '{node.OpCode}' ({node.Key}) has graph-attribute {which} groups "
                    + $"[{string.Join(", ", groups.Keys.Select(k => k.Length == 0 ? "\"\"" : k))}], which Phase 1 does not serialize.");
        }
    }
}
