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
    /// <c>var</c> declarations and a final <c>return</c>, and expressions built from the binary
    /// operators <c>+ - * /</c> and a small set of unary/binary method calls (e.g. <c>MatMul</c>,
    /// <c>Relu</c>). Literals, submodule calls, and control flow are not handled yet and raise
    /// <see cref="NotSupportedException"/>. See <c>src/docs/design/mlir-assembly-parser.md</c>.</para>
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

                    case InvocationExpressionSyntax inv when inv.Expression is MemberAccessExpressionSyntax ma:
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
