using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    /// Parses flat, MLIR-flavored assembly text (as produced by <see cref="MlirTextWriter"/>)
    /// back into a <see cref="FastComputationGraph"/>. Per-opcode attribute schemas are recovered
    /// from <see cref="Definitions.NodeDefinitions"/>, so the reconstructed
    /// <see cref="OnnxCSharpAttributes"/> match what the runtime builds directly. Leading
    /// <c>func @fnN { … }</c> blocks are parsed first into a symbol table that <c>tgtfn @fnN</c>
    /// node suffixes resolve against. See <c>src/docs/design/mlir-assembly-parser.md</c>.
    /// </summary>
    public static class MlirTextReader
    {
        /// <summary>Parses <paramref name="text"/> into a <see cref="FastComputationGraph"/>.</summary>
        public static FastComputationGraph Parse(string text)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            var parser = new Parser(Tokenizer.Tokenize(text));
            return parser.ParseTop();
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

                    // '@' introduces a function symbol (@fn0); keep the sigil in the Word text.
                    if (c == '@')
                    {
                        int start = i++;
                        while (i < n && (char.IsLetterOrDigit(s[i]) || s[i] == '_')) i++;
                        toks.Add(new Tok(TokKind.Word, s.Substring(start, i - start)));
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
            private readonly Dictionary<string, Function> _functions = new();

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

            private bool PeekWord(string w) => Peek.Kind == TokKind.Word && Peek.Text == w;

            private FormatException Fail(string what) => new($"MlirTextReader: {what} but found {Peek}.");

            public FastComputationGraph ParseTop()
            {
                while (PeekWord("func"))
                    ParseFunc();

                ExpectWord("graph");
                ExpectPunct("{");
                var graph = ParseGraphBody();
                ExpectPunct("}");
                Expect(TokKind.End);
                return graph;
            }

            private void ParseFunc()
            {
                ExpectWord("func");
                var id = ParseFuncSymbol();
                ExpectWord("type");
                var type = Enum.Parse<FunctionType>(Expect(TokKind.Word).Text);
                ExpectWord("default");
                var defaultName = Expect(TokKind.Str).Text;
                ExpectWord("friendly");
                var friendlyName = Expect(TokKind.Str).Text;

                StateOwnership? ownership = null;
                if (PeekWord("ownership")) { Next(); ownership = Enum.Parse<StateOwnership>(Expect(TokKind.Word).Text); }

                ExpectPunct("{");
                var body = ParseGraphBody();
                ExpectPunct("}");

                _functions[id] = new Function(body, type, defaultName, friendlyName, ownership);
            }

            private FastComputationGraph ParseGraphBody()
            {
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

                var node = new FastNode { Key = nodeKey, OpCode = opCode };
                foreach (var kv in ParseGroups("in")) node.FullInputs[kv.Key] = kv.Value;
                Expect(TokKind.Arrow);
                foreach (var kv in ParseGroups("out")) node.FullOutputs[kv.Key] = kv.Value;

                var attrVals = (Peek.Kind == TokKind.Punct && Peek.Text == "{") ? ParseAttributes(opCode) : new();

                while (Peek.Kind == TokKind.Word)
                {
                    switch (Peek.Text)
                    {
                        case "open": Next(); node.GraphOpenNodeKey = ParseNodeKey(); break;
                        case "name": Next(); node.FriendlyName = Expect(TokKind.Str).Text; break;
                        case "template": Next(); node.IdentifierTemplate = Expect(TokKind.Str).Text; break;
                        case "tgtfn": Next(); node.TargetFunction = ResolveFunction(ParseFuncSymbol()); break;
                        default: goto done;
                    }
                }
            done:
                node.Attributes = OnnxCSharpAttributes.FromCSharpVals(attrVals, ResolveDefs(opCode));
                return node;
            }

            /// <summary>
            /// Parses a group section: an optional default group as a bare <c>(…)</c>, then zero or more
            /// named groups <c>&lt;keyword&gt; "name"(…)</c>. Absent default group ⇒ no "" key in the result.
            /// </summary>
            private Dictionary<string, List<FastTensorKey?>> ParseGroups(string keyword)
            {
                var groups = new Dictionary<string, List<FastTensorKey?>>();
                if (Peek.Kind == TokKind.Punct && Peek.Text == "(")
                    groups[string.Empty] = ParseSlotList();
                while (PeekWord(keyword))
                {
                    Next();
                    var name = Expect(TokKind.Str).Text;
                    groups[name] = ParseSlotList();
                }
                return groups;
            }

            private List<FastTensorKey?> ParseSlotList()
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
                    case AttributeType.Tensor: return ParseDenseTensor();
                    case AttributeType.Graph: return ParseGraphAttr();
                    default:
                        throw new NotSupportedException(
                            $"MlirTextReader: attribute '{def.AttributeName}' has type {def.Type}, which is not parsed yet.");
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

            private TensorData ParseDenseTensor()
            {
                ExpectWord("dense");
                ExpectPunct("<");
                var dims = ParseArray(ParseLong).ToArray();
                ExpectPunct(",");
                var dtype = ParseDType();
                ExpectPunct(",");
                var data = Convert.FromBase64String(Expect(TokKind.Str).Text);
                ExpectPunct(">");
                return MlirTensorCodec.FromRawBytes(dims, dtype, data);
            }

            private BestGraphAttribute ParseGraphAttr()
            {
                ExpectWord("graphattr");
                ExpectPunct("<");
                var name = Expect(TokKind.Str).Text;
                string? defaultGraphName = null;
                string[]? inputNames = null;
                while (TryPunct(","))
                {
                    if (PeekWord("default")) { Next(); defaultGraphName = Expect(TokKind.Str).Text; }
                    else if (PeekWord("names")) { Next(); inputNames = ParseArray(() => Expect(TokKind.Str).Text).ToArray(); }
                    else throw Fail("expected 'default' or 'names' in graphattr");
                }
                ExpectPunct(">");
                return new BestGraphAttribute { GraphAttributeName = name, DefaultGraphName = defaultGraphName, DefautGraphInputNames = inputNames };
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

            private string ParseFuncSymbol()
            {
                if (Peek.Kind != TokKind.Word || !Peek.Text.StartsWith("@", StringComparison.Ordinal))
                    throw Fail("expected a function symbol (@fnN)");
                return Next().Text;
            }

            private Function ResolveFunction(string symbol)
            {
                if (!_functions.TryGetValue(symbol, out var fn))
                    throw Fail($"reference to undefined function symbol '{symbol}'");
                return fn;
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
