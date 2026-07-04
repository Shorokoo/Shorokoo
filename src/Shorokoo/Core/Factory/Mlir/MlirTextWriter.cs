using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;

namespace Shorokoo.Core.Factory.Mlir
{
    /// <summary>
    /// Prints a <see cref="FastComputationGraph"/> as flat, MLIR-flavored assembly text.
    ///
    /// <para>The text is a near-isomorph of <see cref="FastComputationGraph.Nodes"/>: one op
    /// per node, SSA names carrying <see cref="FastTensorKey"/> identity, and
    /// <c>*#OPEN</c>/<c>*#CLOSE</c> ops delimiting scopes purely by position (no nested
    /// regions). Functions reachable from the graph (<see cref="FastNode.TargetFunction"/>,
    /// transitively) are emitted first as a <c>func @fnN { … }</c> symbol table and referenced
    /// by nodes via a <c>tgtfn @fnN</c> suffix. See
    /// <c>src/docs/design/mlir-assembly-parser.md</c>. The inverse is
    /// <see cref="MlirTextReader"/>; <c>Read(Write(g))</c> is expected to reproduce the same text.</para>
    ///
    /// <para>Not yet serialized: <see cref="AttributeType.TypeProto"/> attributes, and
    /// float16/bfloat16/bool/string/complex tensor constants — these throw
    /// <see cref="NotSupportedException"/> naming the offending opcode/attribute.</para>
    /// </summary>
    public static class MlirTextWriter
    {
        /// <summary>Prints <paramref name="graph"/> to MLIR-flavored assembly text.</summary>
        public static string Write(FastComputationGraph graph)
        {
            if (graph is null) throw new ArgumentNullException(nameof(graph));

            // Assign a stable @fnN symbol to every function reachable from the graph, in
            // dependency (post) order so a function only ever references earlier ones.
            var fnIds = new Dictionary<Function, string>(ReferenceEqualityComparer.Instance);
            var fns = FastComputationGraphConverter.FunctionsPostOrder(graph);
            for (int i = 0; i < fns.Length; i++)
                fnIds[fns[i]] = "@fn" + i.ToString(CultureInfo.InvariantCulture);

            var sb = new StringBuilder();
            foreach (var fn in fns)
            {
                sb.Append("func ").Append(fnIds[fn]).Append(" type ").Append(fn.FunctionType);
                sb.Append(" default ").Append(Quote(fn.DefaultName));
                sb.Append(" friendly ").Append(Quote(fn.FriendlyName));
                if (fn.StateOwnership is StateOwnership so) sb.Append(" ownership ").Append(so);
                sb.Append(" {\n");
                WriteGraphBody(sb, fn.OriginalFastGraph, fnIds);
                sb.Append("}\n");
            }

            sb.Append("graph {\n");
            WriteGraphBody(sb, graph, fnIds);
            sb.Append("}\n");
            return sb.ToString();
        }

        private static void WriteGraphBody(StringBuilder sb, FastComputationGraph graph, Dictionary<Function, string> fnIds)
        {
            WriteDirective(sb, "inputs", graph.Inputs.Select(TensorRef));
            WriteDirective(sb, "outputs", graph.Outputs.Select(TensorRef));
            WriteDirective(sb, "input_names", graph.InputUniqueNames.Select(StringOrHole));
            WriteDirective(sb, "output_names", graph.OutputUniqueNames.Select(StringOrHole));
            if (graph.OutputRankOverrides is not null)
                WriteDirective(sb, "output_rank_overrides",
                    graph.OutputRankOverrides.Select(r => r is null ? "_" : r.Value.ToString(CultureInfo.InvariantCulture)));

            foreach (var node in graph.Nodes)
                WriteNode(sb, node, fnIds);
        }

        private static void WriteDirective(StringBuilder sb, string name, IEnumerable<string> items)
            => sb.Append("  ").Append(name).Append(" = [").Append(string.Join(", ", items)).Append("]\n");

        private static void WriteNode(StringBuilder sb, FastNode node, Dictionary<Function, string> fnIds)
        {
            sb.Append("  ").Append(NodeRef(node.Key)).Append(" = ").Append('"').Append(node.OpCode).Append('"');

            AppendGroups(sb, node.FullInputs, "in");
            sb.Append(" ->");
            AppendGroups(sb, node.FullOutputs, "out");

            var attrText = WriteAttributes(node);
            if (attrText.Length > 0)
                sb.Append(' ').Append(attrText);

            if (node.GraphOpenNodeKey is FastNodeKey openKey && !openKey.IsEmpty)
                sb.Append(" open ").Append(NodeRef(openKey));
            if (node.FriendlyName is not null)
                sb.Append(" name ").Append(Quote(node.FriendlyName));
            if (node.IdentifierTemplate is not null)
                sb.Append(" template ").Append(Quote(node.IdentifierTemplate));
            if (node.TargetFunction is Function tf)
            {
                if (!fnIds.TryGetValue(tf, out var id))
                    throw new NotSupportedException(
                        $"MlirTextWriter: node '{node.OpCode}' references a function not reachable via FunctionsPostOrder.");
                sb.Append(" tgtfn ").Append(id);
            }

            sb.Append('\n');
        }

        /// <summary>
        /// Appends a group section: the default (empty-name) group as a bare <c>(…)</c> when present,
        /// followed by each named group as <c>&lt;keyword&gt; "name"(…)</c>. An absent default group emits
        /// no bare parens, so the read side can distinguish "no default group" from "empty default group".
        /// </summary>
        private static void AppendGroups(StringBuilder sb, Dictionary<string, List<FastTensorKey?>> groups, string keyword)
        {
            if (groups.TryGetValue(string.Empty, out var defaultSlots))
                sb.Append(" (").Append(string.Join(", ", defaultSlots.Select(SlotRef))).Append(')');

            foreach (var key in groups.Keys.Where(k => k.Length != 0).OrderBy(k => k, StringComparer.Ordinal))
                sb.Append(' ').Append(keyword).Append(' ').Append(Quote(key))
                  .Append('(').Append(string.Join(", ", groups[key].Select(SlotRef))).Append(')');
        }

        private static string SlotRef(FastTensorKey? k) => k is null || k.Value.IsEmpty ? "_" : TensorRef(k.Value);

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
            AttributeType.Graph => GraphAttr((BestGraphAttribute)value),
            _ => throw new NotSupportedException(
                $"MlirTextWriter: attribute '{def.AttributeName}' on node '{opCode}' has type {def.Type}, which is not serialized yet.")
        };

        private static string GraphAttr(BestGraphAttribute g)
        {
            var sb = new StringBuilder("graphattr<");
            sb.Append(Quote(g.GraphAttributeName));
            if (g.DefaultGraphName is not null) sb.Append(", default ").Append(Quote(g.DefaultGraphName));
            if (g.DefautGraphInputNames is not null)
                sb.Append(", names [").Append(string.Join(", ", g.DefautGraphInputNames.Select(Quote))).Append(']');
            sb.Append('>');
            return sb.ToString();
        }

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
    }
}
