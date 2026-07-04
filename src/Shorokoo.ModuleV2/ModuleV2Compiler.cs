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
    /// the binary operators <c>+ - * /</c>; method-call ops whose arguments are classified per an
    /// <c>OpSpec</c> as inputs or host-constant attributes (<c>MatMul</c>/<c>Relu</c>/<c>Sqrt</c>/
    /// <c>Exp</c>/<c>Abs</c> input-only; <c>Transpose([1L,0L])</c> → <c>perm</c>; <c>Softmax(axis)</c>
    /// → <c>axis</c>); constant constructors (<c>Scalar(1.0f)</c>, <c>Scalar(1L)</c>,
    /// <c>Vector(2L, 3L)</c>) → <c>Constant</c> nodes; <c>[Hyper]</c> parameters (typed and ordered
    /// hyperparameters-first); and external initializer calls (<c>InitSimple.Init(...)</c>)
    /// referenced by name and bound at link time, never inlined; and native <c>if</c>/<c>else</c>
    /// lowered to an <c>If#OPEN</c>/<c>If#CLOSE</c> scope via SSA merge; and native <c>while</c>
    /// lowered to a <c>Loop#OPEN</c>/<c>Loop#CLOSE</c> scope with loop-carried variables. Module
    /// <c>.Call(...)</c> invocations, dynamic shape-collection arguments, and <c>for</c> loops are
    /// not handled yet and raise <see cref="NotSupportedException"/>. See
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

        private enum AttrKind { Longs, Long, Float, Bool }

        /// <summary>A method argument is either a graph input (an IValue) or a host-constant attribute.</summary>
        private sealed record ArgRole(bool IsInput, string? AttrName = null, AttrKind Kind = default);

        /// <summary>Maps a method call to its opcode and the roles of its arguments after the receiver.</summary>
        private sealed record OpSpec(string Opcode, ArgRole[] Args);

        private static ArgRole Input { get; } = new(true);
        private static ArgRole Attr(string name, AttrKind kind) => new(false, name, kind);

        // Method-call ops: the receiver is the first operand; the remaining arguments are consumed
        // per the ArgRole list (input operand vs host-constant attribute). Trailing roles beyond the
        // supplied argument count are treated as omitted optionals.
        private static readonly Dictionary<string, OpSpec> MethodOps = new()
        {
            ["MatMul"] = new("MatMul", [Input]),
            ["Relu"] = new("Relu", []),
            ["Sqrt"] = new("Sqrt", []),
            ["Exp"] = new("Exp", []),
            ["Abs"] = new("Abs", []),
            ["Transpose"] = new("Transpose", [Attr("perm", AttrKind.Longs)]),
            ["Softmax"] = new("Softmax", [Attr("axis", AttrKind.Long)]),
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
            private Dictionary<string, string> _env = new(StringComparer.Ordinal); // local/param name -> SSA tensor ref (swapped per branch)
            private readonly List<string> _inputRefs = new();
            private readonly List<string> _inputNames = new();

            public string Compile(MethodDeclarationSyntax method)
            {
                if (method.Body is null)
                    throw new NotSupportedException("ModuleV2Compiler: expression-bodied methods are not supported yet; use a block body.");

                // Graph inputs are ordered hyperparameters-first, matching the tracer's convention
                // (even though the source signature lists tensor inputs first, hypers last).
                var parameters = method.ParameterList.Parameters;
                foreach (var p in parameters.Where(IsHyper)) EmitInput(p, isHyper: true);
                foreach (var p in parameters.Where(p => !IsHyper(p))) EmitInput(p, isHyper: false);

                string? outputRef = null;
                foreach (var stmt in method.Body.Statements)
                {
                    if (stmt is ReturnStatementSyntax ret)
                    {
                        if (ret.Expression is null)
                            throw new NotSupportedException("ModuleV2Compiler: bare 'return;' is not supported.");
                        outputRef = Lower(ret.Expression);
                        continue;
                    }
                    ProcessStatement(stmt);
                }

                if (outputRef is null)
                    throw new NotSupportedException("ModuleV2Compiler: method has no 'return' statement.");

                return Assemble(outputRef);
            }

            /// <summary>Processes a non-return statement, mutating the current environment (<see cref="_env"/>).</summary>
            private void ProcessStatement(StatementSyntax stmt)
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

                    case ExpressionStatementSyntax { Expression: AssignmentExpressionSyntax { Left: IdentifierNameSyntax lhs } asg }
                        when asg.IsKind(SyntaxKind.SimpleAssignmentExpression):
                        _env[lhs.Identifier.Text] = Lower(asg.Right);
                        break;

                    case IfStatementSyntax ifStmt:
                        LowerIf(ifStmt);
                        break;

                    case WhileStatementSyntax whileStmt:
                        LowerWhile(whileStmt);
                        break;

                    default:
                        throw new NotSupportedException(
                            $"ModuleV2Compiler: statement '{stmt.Kind()}' is not supported in this slice.");
                }
            }

            /// <summary>
            /// Lowers <c>if (cond) { … } else { … }</c> to an <c>If#OPEN</c>/<c>If#CLOSE</c> scope. Both
            /// branches are evaluated in the enclosing scope (the flat model the tracer produces for
            /// <c>.IfElse</c>); the SSA merge — every variable that ends up bound differently on the two
            /// branches — becomes the close node's outputs, wired through its <c>then_branch</c> /
            /// <c>else_branch</c> input groups. Variables assigned on only one path must have a prior
            /// value (so the other path can yield it); a branch-local temporary simply does not escape.
            /// </summary>
            private void LowerIf(IfStatementSyntax ifStmt)
            {
                var condRef = Lower(ifStmt.Condition);
                var pre = new Dictionary<string, string>(_env, StringComparer.Ordinal);

                var outer = _env;
                _env = new Dictionary<string, string>(pre, StringComparer.Ordinal);
                ProcessBranch(ifStmt.Statement);
                var thenEnv = _env;

                _env = new Dictionary<string, string>(pre, StringComparer.Ordinal);
                if (ifStmt.Else is not null) ProcessBranch(ifStmt.Else.Statement);
                var elseEnv = _env;

                _env = outer;

                // Merge = variables bound on both paths to different SSA values. (A var new to only one
                // path can't be selected, so it does not escape the if.)
                var mergeVars = thenEnv.Keys
                    .Where(k => elseEnv.ContainsKey(k) && thenEnv[k] != elseEnv[k])
                    .OrderBy(k => k, StringComparer.Ordinal)
                    .ToList();

                int open = ++_counter;
                _nodeLines.Add($"  %N{open} = \"If#OPEN\"({condRef}) -> out \"else_branch\"() out \"then_branch\"()");

                int close = ++_counter;
                var thenRefs = mergeVars.Select(v => thenEnv[v]);
                var elseRefs = mergeVars.Select(v => elseEnv[v]);
                var outs = mergeVars.Select((_, i) => $"%N{close}_T{i}");
                _nodeLines.Add(
                    $"  %N{close} = \"If#CLOSE\" in \"else_branch\"({string.Join(", ", elseRefs)}) " +
                    $"in \"then_branch\"({string.Join(", ", thenRefs)}) -> ({string.Join(", ", outs)}) " +
                    $"{{\"else_branch\" = graphattr<\"else_branch\">, \"then_branch\" = graphattr<\"then_branch\">}} open %N{open}");

                for (int i = 0; i < mergeVars.Count; i++)
                    _env[mergeVars[i]] = $"%N{close}_T{i}";
            }

            private void ProcessBranch(StatementSyntax branch)
            {
                if (branch is BlockSyntax block)
                {
                    foreach (var s in block.Statements) ProcessStatement(s);
                }
                else ProcessStatement(branch);
            }

            /// <summary>
            /// Lowers <c>while (cond) { … }</c> to a <c>Loop#OPEN</c>/<c>Loop#CLOSE</c> scope. The
            /// loop-carried variables are those bound before the loop and reassigned in the body;
            /// their initial values become the open node's initializers, their per-iteration values
            /// are the open's <c>body</c>-group outputs (index, condition, then one per carried var),
            /// and the body-computed updates plus the re-evaluated condition become the close node's
            /// <c>body</c>-group inputs, with the close's outputs the post-loop values.
            /// </summary>
            private void LowerWhile(WhileStatementSyntax w)
            {
                var pre = new Dictionary<string, string>(_env, StringComparer.Ordinal);
                var carried = CollectAssignedNames(w.Statement)
                    .Where(pre.ContainsKey)
                    .OrderBy(k => k, StringComparer.Ordinal)
                    .ToList();

                // Initial condition, evaluated in the enclosing scope over the pre-loop values.
                var condInit = Lower(w.Condition);

                int open = ++_counter;
                var openInputs = new List<string> { "_", condInit };            // [maxIterations(unset), condition, …inits]
                openInputs.AddRange(carried.Select(v => pre[v]));
                var bodyOuts = new List<string> { $"%N{open}_T0", $"%N{open}_T1" }; // [iterationIndex, conditionIn, …loopVars]
                for (int i = 0; i < carried.Count; i++) bodyOuts.Add($"%N{open}_T{2 + i}");
                _nodeLines.Add($"  %N{open} = \"Loop#OPEN\"({string.Join(", ", openInputs)}) -> out \"body\"({string.Join(", ", bodyOuts)})");

                // Body: carried variables resolve to their per-iteration open outputs.
                var outer = _env;
                _env = new Dictionary<string, string>(pre, StringComparer.Ordinal);
                for (int i = 0; i < carried.Count; i++) _env[carried[i]] = $"%N{open}_T{2 + i}";
                ProcessBranch(w.Statement);
                var continueWhile = Lower(w.Condition);                          // re-evaluated over the updated values
                var updaters = carried.Select(v => _env[v]).ToList();
                _env = outer;

                int close = ++_counter;
                var closeIns = new List<string> { continueWhile };
                closeIns.AddRange(updaters);
                var closeOuts = carried.Select((_, i) => $"%N{close}_T{i}");
                _nodeLines.Add(
                    $"  %N{close} = \"Loop#CLOSE\" in \"body\"({string.Join(", ", closeIns)}) -> ({string.Join(", ", closeOuts)}) " +
                    $"{{\"body\" = graphattr<\"body\">}} open %N{open}");

                for (int i = 0; i < carried.Count; i++) _env[carried[i]] = $"%N{close}_T{i}";
            }

            /// <summary>Names assigned anywhere in <paramref name="body"/> (via <c>x = …</c> or <c>var x = …</c>).</summary>
            private static HashSet<string> CollectAssignedNames(StatementSyntax body)
            {
                var names = new HashSet<string>(StringComparer.Ordinal);
                foreach (var n in body.DescendantNodes())
                {
                    if (n is AssignmentExpressionSyntax { Left: IdentifierNameSyntax id }) names.Add(id.Identifier.Text);
                    else if (n is VariableDeclaratorSyntax vd) names.Add(vd.Identifier.Text);
                }
                return names;
            }

            private void EmitInput(ParameterSyntax p, bool isHyper)
            {
                var type = p.Type ?? throw new NotSupportedException($"ModuleV2Compiler: parameter '{p.Identifier}' has no type.");
                var proto = DTypeProtoOf(type);
                var inputType = isHyper ? "Hyperparam" : "ReadyInput";

                var attrs = new List<string> { $"\"dtype\" = dtype<{proto}>" };
                if (RankOf(type) is int rank) attrs.Add($"\"shrk_rank\" = {rank} : i64");
                attrs.Add($"\"shrk_input_type\" = enum<InputType, {inputType}>");

                int k = ++_counter;
                _nodeLines.Add($"  %N{k} = \"#ModelTensorInput#\"() -> (%N{k}_T0) {{{string.Join(", ", attrs)}}}");
                var refTxt = $"%N{k}_T0";
                _inputRefs.Add(refTxt);
                _inputNames.Add(p.Identifier.Text);
                _env[p.Identifier.Text] = refTxt;
            }

            private static bool IsHyper(ParameterSyntax p)
                => p.AttributeLists.SelectMany(al => al.Attributes).Any(a => a.Name.ToString() is "Hyper" or "HyperAttribute");

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
                        if (!MethodOps.TryGetValue(name, out var spec))
                            throw new NotSupportedException($"ModuleV2Compiler: method '{name}' is not in the supported op set.");
                        return LowerOpCall(spec, ma.Expression, inv.ArgumentList.Arguments);

                    default:
                        throw new NotSupportedException(
                            $"ModuleV2Compiler: expression '{expr.Kind()}' is not supported in this slice.");
                }
            }

            private string LowerOpCall(OpSpec spec, ExpressionSyntax receiver, SeparatedSyntaxList<ArgumentSyntax> args)
            {
                if (args.Count > spec.Args.Length)
                    throw new NotSupportedException($"ModuleV2Compiler: op '{spec.Opcode}' got {args.Count} arguments but expects at most {spec.Args.Length}.");

                var operands = new List<string> { Lower(receiver) };
                var attrs = new List<string>();
                for (int i = 0; i < args.Count; i++)
                {
                    var role = spec.Args[i];
                    if (role.IsInput) operands.Add(Lower(args[i].Expression));
                    else attrs.Add($"{Quote(role.AttrName!)} = {AttrLiteral(role.Kind, args[i].Expression)}");
                }
                return EmitNode(spec.Opcode, operands, attrs);
            }

            private string EmitNode(string opcode, IReadOnlyList<string> operandRefs, IReadOnlyList<string>? attrs = null)
            {
                int k = ++_counter;
                var attrText = attrs is { Count: > 0 } ? $" {{{string.Join(", ", attrs)}}}" : "";
                _nodeLines.Add($"  %N{k} = \"{opcode}\"({string.Join(", ", operandRefs)}) -> (%N{k}_T0){attrText}");
                return $"%N{k}_T0";
            }

            private static string AttrLiteral(AttrKind kind, ExpressionSyntax e) => kind switch
            {
                AttrKind.Long => $"{LongLiteral(e)} : i64",
                AttrKind.Longs => $"[{string.Join(", ", LongCollection(e))}] : i64",
                AttrKind.Float => $"{FloatLiteral(e)} : f32",
                AttrKind.Bool => BoolLiteral(e) ? "true" : "false",
                _ => throw new NotSupportedException($"ModuleV2Compiler: attribute kind {kind} not supported.")
            };

            private static IEnumerable<long> LongCollection(ExpressionSyntax e) => e switch
            {
                CollectionExpressionSyntax coll => coll.Elements.Select(el =>
                    el is ExpressionElementSyntax ee ? LongLiteral(ee.Expression)
                    : throw new NotSupportedException("ModuleV2Compiler: only plain elements are supported in an attribute list.")),
                _ => throw new NotSupportedException($"ModuleV2Compiler: expected a collection literal like [1L, 0L], got '{e.Kind()}'.")
            };

            private static float FloatLiteral(ExpressionSyntax e)
            {
                var (neg, lit) = Unminus(e);
                float v = lit.Token.Value switch
                {
                    float f => f,
                    double d => (float)d,
                    int i => i,
                    long l => l,
                    _ => throw new NotSupportedException($"ModuleV2Compiler: expected a float literal, got '{lit.Token.Text}'.")
                };
                return neg ? -v : v;
            }

            private static bool BoolLiteral(ExpressionSyntax e) => e.Kind() switch
            {
                SyntaxKind.TrueLiteralExpression => true,
                SyntaxKind.FalseLiteralExpression => false,
                _ => throw new NotSupportedException($"ModuleV2Compiler: expected 'true' or 'false', got '{e.Kind()}'.")
            };

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

            /// <summary>Statically known rank of a value type: Scalar → 0, Vector → 1, Tensor → unknown (null).</summary>
            private static int? RankOf(TypeSyntax type) => (type as GenericNameSyntax)?.Identifier.Text switch
            {
                "Scalar" => 0,
                "Vector" => 1,
                _ => null,
            };

            private static string Quote(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }
    }
}
