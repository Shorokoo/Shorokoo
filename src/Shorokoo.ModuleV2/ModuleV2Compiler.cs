using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Shorokoo.ModuleV2
{
    /// <summary>
    /// Prototype <c>[ModuleV2]</c> frontend: statically lowers a restricted, straight-line C#
    /// method to flat MLIR-flavored assembly text (the form consumed by
    /// <c>Shorokoo.Core.Factory.Mlir.MlirTextReader</c>). It reads the method as syntax and never
    /// executes it, which is what lets native control flow lower to graph ops in later phases.
    ///
    /// <para><b>Phase 2 slice.</b> Supported today: a single static method whose parameters are
    /// <c>Tensor&lt;T&gt;</c>/<c>Vector&lt;T&gt;</c>/<c>Scalar&lt;T&gt;</c>, a straight-line body of
    /// <c>var</c> declarations and a final <c>return</c>, and expressions built from:
    /// the binary operators <c>+ - * /</c>; a small set of input-only method calls (<c>MatMul</c>,
    /// <c>Relu</c>, <c>Sqrt</c>, <c>Exp</c>, <c>Abs</c>); constant constructors
    /// (<c>Scalar(1.0f)</c>, <c>Scalar(1L)</c>, <c>Vector(2L, 3L)</c>) → <c>Constant</c> nodes; and
    /// external initializer calls (<c>InitSimple.Init(...)</c>) referenced by name and bound at
    /// link time, never inlined. Attribute arguments, module <c>.Call(...)</c> invocations, and
    /// control flow are not handled yet and raise <see cref="NotSupportedException"/>. See
    /// <c>src/docs/design/mlir-assembly-parser.md</c>.</para>
    /// </summary>
    public static class ModuleV2Compiler
    {
        private static readonly Dictionary<SyntaxKind, string> BinaryOps = new()
        {
            [SyntaxKind.AddExpression] = "Add",
            [SyntaxKind.SubtractExpression] = "Sub",
            [SyntaxKind.MultiplyExpression] = "Mul",
            [SyntaxKind.DivideExpression] = "Div",
        };

        // Method-call ops: name -> opcode. The receiver is the first operand, remaining arguments
        // follow in order. Restricted to input-only ops (no attribute arguments) for this slice.
        private static readonly Dictionary<string, string> MethodOps = new()
        {
            ["MatMul"] = "MatMul",
            ["Relu"] = "Relu",
            ["Sqrt"] = "Sqrt",
            ["Exp"] = "Exp",
            ["Abs"] = "Abs",
        };

        private static readonly Dictionary<string, int> DTypeProtoByName = new(StringComparer.Ordinal)
        {
            ["float32"] = 1, ["float64"] = 11, ["float16"] = 10, ["bfloat16"] = 16,
            ["int8"] = 3, ["uint8"] = 2, ["int16"] = 5, ["uint16"] = 4,
            ["int32"] = 6, ["int64"] = 7, ["uint32"] = 12, ["uint64"] = 13,
            ["bit"] = 9,
        };

        /// <summary>Lowers the first method declaration found in <paramref name="methodSource"/> to MLIR text.</summary>
        public static string CompileToMlir(string methodSource)
        {
            if (methodSource is null) throw new ArgumentNullException(nameof(methodSource));

            // Wrap in a synthetic class so a bare method parses as a MethodDeclaration rather than
            // a top-level local function.
            var root = CSharpSyntaxTree.ParseText("class __ModuleV2Wrapper__ {\n" + methodSource + "\n}").GetRoot();
            var method = root.DescendantNodes().OfType<MethodDeclarationSyntax>().FirstOrDefault()
                ?? throw new NotSupportedException("ModuleV2Compiler: no method declaration found.");

            return new Lowering().Compile(method);
        }

        private sealed class Lowering
        {
            private int _counter;
            private readonly List<string> _nodeLines = new();
            private readonly Dictionary<string, string> _env = new(StringComparer.Ordinal); // local/param name -> SSA tensor ref
            private readonly List<string> _inputRefs = new();
            private readonly List<string> _inputNames = new();

            public string Compile(MethodDeclarationSyntax method)
            {
                if (method.Body is null)
                    throw new NotSupportedException("ModuleV2Compiler: expression-bodied methods are not supported yet; use a block body.");

                foreach (var p in method.ParameterList.Parameters)
                    EmitInput(p);

                string? outputRef = null;
                foreach (var stmt in method.Body.Statements)
                {
                    switch (stmt)
                    {
                        case LocalDeclarationStatementSyntax decl:
                            foreach (var v in decl.Declaration.Variables)
                            {
                                if (v.Initializer is null)
                                    throw new NotSupportedException($"ModuleV2Compiler: local '{v.Identifier}' has no initializer.");
                                _env[v.Identifier.Text] = Lower(v.Initializer.Value);
                            }
                            break;

                        case ReturnStatementSyntax ret:
                            if (ret.Expression is null)
                                throw new NotSupportedException("ModuleV2Compiler: bare 'return;' is not supported.");
                            outputRef = Lower(ret.Expression);
                            break;

                        default:
                            throw new NotSupportedException(
                                $"ModuleV2Compiler: statement '{stmt.Kind()}' is not supported in this slice.");
                    }
                }

                if (outputRef is null)
                    throw new NotSupportedException("ModuleV2Compiler: method has no 'return' statement.");

                return Assemble(outputRef);
            }

            private void EmitInput(ParameterSyntax p)
            {
                var proto = DTypeProtoOf(p.Type ?? throw new NotSupportedException($"ModuleV2Compiler: parameter '{p.Identifier}' has no type."));
                int k = ++_counter;
                _nodeLines.Add($"  %N{k} = \"#ModelTensorInput#\"() -> (%N{k}_T0) {{\"dtype\" = dtype<{proto}>}}");
                var refTxt = $"%N{k}_T0";
                _inputRefs.Add(refTxt);
                _inputNames.Add(p.Identifier.Text);
                _env[p.Identifier.Text] = refTxt;
            }

            private string Lower(ExpressionSyntax expr)
            {
                switch (expr)
                {
                    case ParenthesizedExpressionSyntax paren:
                        return Lower(paren.Expression);

                    case IdentifierNameSyntax id:
                        if (_env.TryGetValue(id.Identifier.Text, out var r)) return r;
                        throw new NotSupportedException($"ModuleV2Compiler: unknown identifier '{id.Identifier.Text}'.");

                    case BinaryExpressionSyntax bin when BinaryOps.TryGetValue(bin.Kind(), out var op):
                        return EmitNode(op, [Lower(bin.Left), Lower(bin.Right)]);

                    // Constant constructors: Scalar(literal) / Vector(literal, ...).
                    case InvocationExpressionSyntax inv when inv.Expression is IdentifierNameSyntax callee && callee.Identifier.Text is "Scalar" or "Vector":
                        return LowerConstant(callee.Identifier.Text, inv.ArgumentList.Arguments);

                    case InvocationExpressionSyntax inv when inv.Expression is MemberAccessExpressionSyntax ma:
                        // A member call whose receiver is an identifier not bound as a value is a
                        // type-qualified submodule/initializer call (e.g. InitSimple.Init(...)); a
                        // receiver that is a bound value is an op call (e.g. x.MatMul(y)).
                        if (ma.Expression is IdentifierNameSyntax recvId && !_env.ContainsKey(recvId.Identifier.Text))
                            return LowerSubmoduleCall(recvId.Identifier.Text, ma.Name.Identifier.Text, inv.ArgumentList.Arguments);

                        var name = ma.Name.Identifier.Text;
                        if (!MethodOps.TryGetValue(name, out var opcode))
                            throw new NotSupportedException($"ModuleV2Compiler: method '{name}' is not in the supported op set.");
                        var operands = new List<string> { Lower(ma.Expression) };
                        operands.AddRange(inv.ArgumentList.Arguments.Select(a => Lower(a.Expression)));
                        return EmitNode(opcode, operands);

                    default:
                        throw new NotSupportedException(
                            $"ModuleV2Compiler: expression '{expr.Kind()}' is not supported in this slice.");
                }
            }

            private string EmitNode(string opcode, IReadOnlyList<string> operandRefs)
            {
                int k = ++_counter;
                _nodeLines.Add($"  %N{k} = \"{opcode}\"({string.Join(", ", operandRefs)}) -> (%N{k}_T0)");
                return $"%N{k}_T0";
            }

            // Scalar(literal) / Vector(literal, ...) → a Constant node carrying a dense value.
            private string LowerConstant(string ctor, SeparatedSyntaxList<ArgumentSyntax> args)
            {
                if (ctor == "Scalar")
                {
                    if (args.Count != 1) throw new NotSupportedException("ModuleV2Compiler: Scalar(...) takes exactly one literal.");
                    var (proto, bytes) = ScalarConst(args[0].Expression);
                    return EmitConstant([], proto, bytes);
                }

                // Vector(long, long, ...) → rank-1 int64 constant.
                var buf = new List<byte>();
                foreach (var a in args)
                    buf.AddRange(BitConverter.GetBytes(LongLiteral(a.Expression)));
                return EmitConstant([args.Count], 7, buf.ToArray());
            }

            private string EmitConstant(long[] dims, int proto, byte[] bytes)
            {
                int k = ++_counter;
                var dense = $"dense<[{string.Join(", ", dims)}], dtype<{proto}>, \"{Convert.ToBase64String(bytes)}\">";
                _nodeLines.Add($"  %N{k} = \"Constant\"() -> (%N{k}_T0) {{\"value\" = {dense}}}");
                return $"%N{k}_T0";
            }

            private static (int proto, byte[] bytes) ScalarConst(ExpressionSyntax e)
            {
                var (neg, lit) = Unminus(e);
                return lit.Token.Value switch
                {
                    float f => (1, BitConverter.GetBytes(neg ? -f : f)),
                    double d => (1, BitConverter.GetBytes((float)(neg ? -d : d))),
                    long l => (7, BitConverter.GetBytes(neg ? -l : l)),
                    int i => (7, BitConverter.GetBytes(neg ? -(long)i : i)),
                    _ => throw new NotSupportedException($"ModuleV2Compiler: unsupported Scalar literal '{lit.Token.Text}' (use e.g. 1.0f or 1L).")
                };
            }

            private static long LongLiteral(ExpressionSyntax e)
            {
                var (neg, lit) = Unminus(e);
                long v = lit.Token.Value switch
                {
                    long l => l,
                    int i => i,
                    _ => throw new NotSupportedException($"ModuleV2Compiler: expected an integer literal, got '{lit.Token.Text}'.")
                };
                return neg ? -v : v;
            }

            private static (bool neg, LiteralExpressionSyntax lit) Unminus(ExpressionSyntax e)
            {
                bool neg = false;
                if (e is PrefixUnaryExpressionSyntax u && u.IsKind(SyntaxKind.UnaryMinusExpression)) { neg = true; e = u.Operand; }
                if (e is not LiteralExpressionSyntax lit)
                    throw new NotSupportedException($"ModuleV2Compiler: expected a numeric literal, got '{e.Kind()}'.");
                return (neg, lit);
            }

            // A call to an *external* module/initializer (e.g. InitSimple.Init(...)). Per design, the
            // callee is referenced by name and bound at link time, not inlined at codegen — the node
            // carries the callee's name; no function body is emitted. (Full arg-shape lowering and
            // module .Call() invocations are future work.)
            private string LowerSubmoduleCall(string typeName, string methodName, SeparatedSyntaxList<ArgumentSyntax> args)
            {
                if (methodName != "Init")
                    throw new NotSupportedException($"ModuleV2Compiler: submodule call '{typeName}.{methodName}' is not supported yet (only initializer '.Init' calls).");

                var operands = args.Select(a => Lower(a.Expression)).ToList();
                int k = ++_counter;
                _nodeLines.Add(
                    $"  %N{k} = \"#TrainableParamRef#\"({string.Join(", ", operands)}) -> (%N{k}_T0) " +
                    $"{{\"shrk_function_name\" = {Quote(typeName)}, \"shrk_domain_name\" = \"Functions\", \"shrk_is_trainable\" = true}}");
                return $"%N{k}_T0";
            }

            private string Assemble(string outputRef)
            {
                var sb = new StringBuilder();
                sb.Append("graph {\n");
                sb.Append("  inputs = [").Append(string.Join(", ", _inputRefs)).Append("]\n");
                sb.Append("  outputs = [").Append(outputRef).Append("]\n");
                sb.Append("  input_names = [").Append(string.Join(", ", _inputNames.Select(Quote))).Append("]\n");
                sb.Append("  output_names = [_]\n");
                foreach (var line in _nodeLines) sb.Append(line).Append('\n');
                sb.Append("}\n");
                return sb.ToString();
            }

            private static int DTypeProtoOf(TypeSyntax type)
            {
                if (type is GenericNameSyntax g && g.TypeArgumentList.Arguments.Count == 1)
                {
                    var arg = g.TypeArgumentList.Arguments[0].ToString();
                    if (DTypeProtoByName.TryGetValue(arg, out var proto)) return proto;
                    throw new NotSupportedException($"ModuleV2Compiler: unsupported element type '{arg}'.");
                }
                throw new NotSupportedException($"ModuleV2Compiler: parameter type '{type}' must be a tensor type like Tensor<float32>.");
            }

            private static string Quote(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }
    }
}
