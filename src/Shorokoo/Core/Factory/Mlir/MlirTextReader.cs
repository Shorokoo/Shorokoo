using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Graph;

namespace Shorokoo.Core.Factory.Mlir
{
    /// <summary>
    /// Parses flat, MLIR-flavored assembly text (as produced by <see cref="MlirTextWriter"/>)
    /// back into a <see cref="FastComputationGraph"/>. Per-opcode attribute schemas are recovered
    /// from <see cref="Definitions.NodeDefinitions"/>, so the reconstructed
    /// <see cref="OnnxCSharpAttributes"/> match what the runtime builds directly.
    ///
    /// <para>Scoping is implicit: close nodes carry their matching open via the <c>open %N…</c>
    /// suffix, exactly as the printer emits it, and the reconstructed graph is checked against
    /// <see cref="FastComputationGraph.IsLinearOrderValid"/> in debug builds. See
    /// <c>src/docs/design/mlir-assembly-parser.md</c>.</para>
    /// </summary>
    public static class MlirTextReader
    {
        /// <summary>Parses <paramref name="text"/> into a <see cref="FastComputationGraph"/>.</summary>
        public static FastComputationGraph Parse(string text)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            var parser = new Parser(Tokenizer.Tokenize(text));
            return parser.ParseGraph();
        }

        // ─────────────────────────────── tokenizer ───────────────────────────────

        private enum TokKind { Word, Percent, Str, Number, Punct, Arrow, End }

        private readonly struct Tok
        {
            public readonly TokKind Kind;
            public readonly string Text;
            public Tok(TokKind kind, string text) { Kind = kind; Text = text; }
            public override string ToString() => $"{Kind}:{Text}";
        }

        private static class Tokenizer
        {
            public static List<Tok> Tokenize(string s)
            {
                var toks = new List<Tok>();
                int i = 0, n = s.Length;
                while (i < n)
                {
                    char c = s[i];
                    if (char.IsWhiteSpace(c)) { i++; continue; }

                    if (c == '-' && i + 1 < n && s[i + 1] == '>') { toks.Add(new Tok(TokKind.Arrow, "->")); i += 2; continue; }

                    if (c == '"')
                    {
                        var (str, next) = ReadString(s, i);
                        toks.Add(new Tok(TokKind.Str, str));
                        i = next;
                        continue;
                    }

                    if (c == '%')
                    {
                        int start = i++;
                        while (i < n && (char.IsLetterOrDigit(s[i]) || s[i] == '_' || s[i] == '-')) i++;
                        toks.Add(new Tok(TokKind.Percent, s.Substring(start + 1, i - start - 1)));
                        continue;
                    }

                    if (char.IsLetter(c) || c == '_')
                    {
                        int start = i++;
                        while (i < n && (char.IsLetterOrDigit(s[i]) || s[i] == '_')) i++;
                        toks.Add(new Tok(TokKind.Word, s.Substring(start, i - start)));
                        continue;
                    }

                    if (char.IsDigit(c) || c == '.')
                    {
                        int start = i++;
                        while (i < n && (char.IsDigit(s[i]) || s[i] == '.')) i++;
                        if (i < n && (s[i] == 'e' || s[i] == 'E'))
                        {
                            i++;
                            if (i < n && (s[i] == '+' || s[i] == '-')) i++;
                            while (i < n && char.IsDigit(s[i])) i++;
                        }
                        toks.Add(new Tok(TokKind.Number, s.Substring(start, i - start)));
                        continue;
                    }

                    if ("{}()[]=,<>:-".IndexOf(c) >= 0) { toks.Add(new Tok(TokKind.Punct, c.ToString())); i++; continue; }

                    throw new FormatException($"MlirTextReader: unexpected character '{c}' at offset {i}.");
                }
                toks.Add(new Tok(TokKind.End, string.Empty));
                return toks;
            }

            private static (string value, int next) ReadString(string s, int i)
            {
                var sb = new StringBuilder();
                i++; // opening quote
                while (i < s.Length)
                {
                    char c = s[i++];
                    if (c == '"') return (sb.ToString(), i);
                    if (c == '\\')
                    {
                        if (i >= s.Length) break;
                        char e = s[i++];
                        sb.Append(e switch { 'n' => '\n', 'r' => '\r', 't' => '\t', '"' => '"', '\\' => '\\', _ => e });
                    }
                    else sb.Append(c);
                }
                throw new FormatException("MlirTextReader: unterminated string literal.");
            }
        }

        // ─────────────────────────────── parser ───────────────────────────────

        private sealed class Parser
        {
            private readonly List<Tok> _toks;
            private int _pos;

            public Parser(List<Tok> toks) { _toks = toks; }

            private Tok Peek => _toks[_pos];
            private Tok Next() => _toks[_pos++];

            private Tok Expect(TokKind kind)
            {
                if (Peek.Kind != kind) throw Fail($"expected {kind}");
                return Next();
            }

            private void ExpectPunct(string p)
            {
                if (Peek.Kind != TokKind.Punct || Peek.Text != p) throw Fail($"expected '{p}'");
                Next();
            }

            private void ExpectWord(string w)
            {
                if (Peek.Kind != TokKind.Word || Peek.Text != w) throw Fail($"expected '{w}'");
                Next();
            }

            private bool TryPunct(string p)
            {
                if (Peek.Kind == TokKind.Punct && Peek.Text == p) { Next(); return true; }
                return false;
            }

            private FormatException Fail(string what) => new($"MlirTextReader: {what} but found {Peek}.");

            public FastComputationGraph ParseGraph()
            {
                ExpectWord("graph");
                ExpectPunct("{");

                var graph = new FastComputationGraph();
                while (Peek.Kind == TokKind.Word)
                {
                    switch (Peek.Text)
                    {
                        case "inputs": Next(); ExpectPunct("="); graph.Inputs = ParseList(ParseTensorKey); break;
                        case "outputs": Next(); ExpectPunct("="); graph.Outputs = ParseList(ParseTensorKey); break;
                        case "input_names": Next(); ExpectPunct("="); graph.InputUniqueNames = ParseList(ParseStringOrHole); break;
                        case "output_names": Next(); ExpectPunct("="); graph.OutputUniqueNames = ParseList(ParseStringOrHole); break;
                        case "output_rank_overrides": Next(); ExpectPunct("="); graph.OutputRankOverrides = ParseList(ParseIntOrHole).ToArray(); break;
                        default: throw Fail($"unexpected directive '{Peek.Text}'");
                    }
                }

                while (Peek.Kind == TokKind.Percent)
                    graph.Nodes.Add(ParseNode());

                ExpectPunct("}");
                Expect(TokKind.End);

                Debug.Assert(graph.IsLinearOrderValid(), "MlirTextReader: parsed graph is not in valid linear order.");
                return graph;
            }

            private List<T> ParseList<T>(Func<T> parseItem)
            {
                var items = new List<T>();
                ExpectPunct("[");
                if (!TryPunct("]"))
                {
                    do { items.Add(parseItem()); } while (TryPunct(","));
                    ExpectPunct("]");
                }
                return items;
            }

            private FastNode ParseNode()
            {
                var nodeKey = ParseNodeKey();
                ExpectPunct("=");
                var opCode = Expect(TokKind.Str).Text;

                var inputs = ParseSlotGroup();
                Expect(TokKind.Arrow);
                var outputs = ParseSlotGroup();

                var node = new FastNode { Key = nodeKey, OpCode = opCode };
                node.FullInputs[string.Empty] = inputs;
                node.FullOutputs[string.Empty] = outputs;

                var attrVals = (Peek.Kind == TokKind.Punct && Peek.Text == "{") ? ParseAttributes(opCode) : new();

                while (Peek.Kind == TokKind.Word)
                {
                    switch (Peek.Text)
                    {
                        case "open": Next(); node.GraphOpenNodeKey = ParseNodeKey(); break;
                        case "name": Next(); node.FriendlyName = Expect(TokKind.Str).Text; break;
                        case "template": Next(); node.IdentifierTemplate = Expect(TokKind.Str).Text; break;
                        default: goto done;
                    }
                }
            done:
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(attrVals, ResolveDefs(opCode));
                return node;
            }

            private List<FastTensorKey?> ParseSlotGroup()
            {
                var slots = new List<FastTensorKey?>();
                ExpectPunct("(");
                if (!TryPunct(")"))
                {
                    do
                    {
                        if (Peek.Kind == TokKind.Word && Peek.Text == "_") { Next(); slots.Add(null); }
                        else slots.Add(ParseTensorKey());
                    } while (TryPunct(","));
                    ExpectPunct(")");
                }
                return slots;
            }

            private Dictionary<string, object?> ParseAttributes(string opCode)
            {
                var defs = ResolveDefs(opCode).ToDictionary(d => d.AttributeName);
                var vals = new Dictionary<string, object?>();
                ExpectPunct("{");
                if (!TryPunct("}"))
                {
                    do
                    {
                        var name = Expect(TokKind.Str).Text;
                        ExpectPunct("=");
                        if (!defs.TryGetValue(name, out var def))
                            throw Fail($"attribute '{name}' is not defined on op '{opCode}'");
                        vals[name] = ParseAttributeValue(def);
                    } while (TryPunct(","));
                    ExpectPunct("}");
                }
                return vals;
            }

            private object? ParseAttributeValue(NodeDefAttributeDef def)
            {
                switch (def.Type)
                {
                    case AttributeType.Long: { var v = ParseLong(); ExpectPunct(":"); ExpectWord("i64"); return v; }
                    case AttributeType.Longs: { var v = ParseArray(ParseLong).ToArray(); ExpectPunct(":"); ExpectWord("i64"); return v; }
                    case AttributeType.Float: { var v = ParseFloat(); ExpectPunct(":"); ExpectWord("f32"); return v; }
                    case AttributeType.Floats: { var v = ParseArray(ParseFloat).ToArray(); ExpectPunct(":"); ExpectWord("f32"); return v; }
                    case AttributeType.Bool: return ParseBool();
                    case AttributeType.Bools: return ParseArray(ParseBool).ToArray();
                    case AttributeType.String: return Expect(TokKind.Str).Text;
                    case AttributeType.Strings: return ParseArray(() => Expect(TokKind.Str).Text).ToArray();
                    case AttributeType.DType: return ParseDType();
                    case AttributeType.DTypes: return ParseArray(ParseDType).ToArray();
                    case AttributeType.Enum: return ParseEnum(def);
                    case AttributeType.Enums: return ParseArray(() => ParseEnum(def)).ToArray();
                    default:
                        throw new NotSupportedException(
                            $"MlirTextReader: attribute '{def.AttributeName}' has type {def.Type}, which Phase 1 does not parse.");
                }
            }

            private List<T> ParseArray<T>(Func<T> parseItem)
            {
                var items = new List<T>();
                ExpectPunct("[");
                if (!TryPunct("]"))
                {
                    do { items.Add(parseItem()); } while (TryPunct(","));
                    ExpectPunct("]");
                }
                return items;
            }

            private long ParseLong() => long.Parse(SignedNumberText(), CultureInfo.InvariantCulture);

            private float ParseFloat() => float.Parse(SignedNumberText(), NumberStyles.Float, CultureInfo.InvariantCulture);

            private string SignedNumberText()
            {
                var sb = new StringBuilder();
                if (Peek.Kind == TokKind.Punct && Peek.Text == "-") { Next(); sb.Append('-'); }
                if (Peek.Kind == TokKind.Number) return sb.Append(Next().Text).ToString();
                if (Peek.Kind == TokKind.Word && (Peek.Text is "Infinity" or "NaN"))
                    return sb.Append(Next().Text).ToString();
                throw Fail("expected a number");
            }

            private bool ParseBool()
            {
                if (Peek.Kind != TokKind.Word || (Peek.Text != "true" && Peek.Text != "false")) throw Fail("expected 'true' or 'false'");
                return Next().Text == "true";
            }

            private DType ParseDType()
            {
                ExpectWord("dtype");
                ExpectPunct("<");
                int protoNum = (int)ParseLong();
                ExpectPunct(">");
                return DType.FromProtoTypeNum(protoNum);
            }

            private object ParseEnum(NodeDefAttributeDef def)
            {
                if (def.EnumDef is null) throw new NotSupportedException($"MlirTextReader: enum attribute '{def.AttributeName}' has no EnumDef.");
                ExpectWord("enum");
                ExpectPunct("<");
                Expect(TokKind.Word); // enum type name — informational, ignored on reconstruction
                ExpectPunct(",");
                var onnxName = Expect(TokKind.Word).Text;
                ExpectPunct(">");
                return def.EnumDef.ToCSharpVal(onnxName);
            }

            private System.Collections.Immutable.ImmutableList<NodeDefAttributeDef> ResolveDefs(string opCode)
            {
                if (!Definitions.NodeDefinitions.TryGetValue(opCode, out var resolver))
                    throw new NotSupportedException($"MlirTextReader: unknown op code '{opCode}' (no NodeDefinition registered).");
                return resolver.AttributeDefs;
            }

            private FastNodeKey ParseNodeKey()
            {
                var text = Expect(TokKind.Percent).Text;
                if (text.IndexOf("_T", StringComparison.Ordinal) >= 0)
                    throw Fail($"expected a node key (%N…) but found tensor key '%{text}'");
                return new FastNodeKey(ParseNodeId(text));
            }

            private FastTensorKey ParseTensorKey()
            {
                var text = Expect(TokKind.Percent).Text;
                int t = text.IndexOf("_T", StringComparison.Ordinal);
                if (t < 0) throw Fail($"expected a tensor key (%N…_T…) but found '%{text}'");
                var nodeId = ParseNodeId(text.Substring(0, t));
                int outIdx = int.Parse(text.Substring(t + 2), CultureInfo.InvariantCulture);
                return new FastTensorKey(new FastNodeKey(nodeId), outIdx);
            }

            private UInt128 ParseNodeId(string nText)
            {
                if (nText.Length < 2 || nText[0] != 'N') throw Fail($"malformed node id '%{nText}'");
                return UInt128.Parse(nText.AsSpan(1), CultureInfo.InvariantCulture);
            }

            private string? ParseStringOrHole()
            {
                if (Peek.Kind == TokKind.Word && Peek.Text == "_") { Next(); return null; }
                return Expect(TokKind.Str).Text;
            }

            private int? ParseIntOrHole()
            {
                if (Peek.Kind == TokKind.Word && Peek.Text == "_") { Next(); return null; }
                return (int)ParseLong();
            }
        }
    }
}
