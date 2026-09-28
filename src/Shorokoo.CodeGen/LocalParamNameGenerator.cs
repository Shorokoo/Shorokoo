using System.Collections.Immutable;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace System.Runtime.CompilerServices
{
    // netstandard2.0 lacks the marker type records need.
    internal static class IsExternalInit { }
}

namespace Shorokoo.CodeGen
{
    /// <summary>
    /// Names a parameter or sub-model after the local it is assigned to. For every
    /// <c>var proj = Linear.Model(...)</c> / <c>var w = Normal02.Init(...)</c> in a <c>[Module]</c>
    /// class, emits an interceptor that makes the same call and records the local's name with
    /// <c>ParamNames.NamedByLocal</c>, so the parameter is saved as <c>proj</c> or <c>w</c> rather
    /// than under its class name and a creation-order number.
    ///
    /// <para>Only a local whose whole initializer is the call is named: in
    /// <c>var x = Normal02.Init(...).Gather(t)</c>, <c>x</c> is not the parameter. Generic
    /// <c>Init</c>/<c>Model</c> methods, and those of generic types, keep their class names.</para>
    ///
    /// <para>A called <c>Init</c>/<c>Model</c> defined in this same project is written by
    /// <see cref="ModuleSourceGenerator"/>, whose output this generator cannot see: the call does
    /// not bind yet. Its signature is taken from that generator's own analysis of the class
    /// instead (<see cref="ModuleSourceGenerator.GeneratedSignature"/>).</para>
    /// </summary>
    [Generator]
    public sealed class LocalParamNameGenerator : IIncrementalGenerator
    {
        internal const string InterceptorNamespace = "Shorokoo.Generated.ParamNames";

        private static readonly DiagnosticDescriptor InterceptorsNotEnabled = new(
            id: "MSG006",
            title: "Parameters cannot be named after their locals",
            messageFormat: "This project does not enable interceptors in namespace '" + InterceptorNamespace + "', so parameters and sub-models assigned to locals keep their class names. Add <InterceptorsNamespaces>$(InterceptorsNamespaces);" + InterceptorNamespace + "</InterceptorsNamespaces> to the project, or name them with .Named(\"...\").",
            category: "SourceGeneration",
            DiagnosticSeverity.Warning,
            isEnabledByDefault: true);

        /// <summary>One call to name. <see cref="Target"/> is the method called, fully qualified;
        /// <see cref="Signature"/> is its return and parameter types when the call binds, and null
        /// when the method is generated in this project (<see cref="ReceiverType"/> then names the
        /// class to look it up on).</summary>
        private sealed record Site(string Attribute, string Target, string ReceiverType, string Method,
            string Name, Signature? Signature);

        private sealed record Signature(string ReturnType, EquatableArray Parameters);

        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var sites = context.SyntaxProvider
                .CreateSyntaxProvider(
                    predicate: static (node, _) => IsLocalAssignedCall(node),
                    transform: static (ctx, ct) => Analyze(ctx, ct))
                .Where(static s => s is not null)
                .Collect();

            var generated = context.SyntaxProvider
                .CreateSyntaxProvider(
                    predicate: static (node, _) => ModuleSourceGenerator.IsPotentialModuleClass(node),
                    transform: static (ctx, _) => GeneratedSignatures(ctx))
                .Where(static s => s is not null)
                .Collect();

            var enabled = context.ParseOptionsProvider.Select(static (options, _) => IsEnabled(options));

            context.RegisterSourceOutput(sites.Combine(generated).Combine(enabled), static (spc, input) =>
            {
                var ((found, classes), isEnabled) = input;
                if (found.IsEmpty) return;

                var bySource = new Dictionary<(string type, string method), Signature>();
                foreach (var entry in classes)
                    foreach (var (method, signature) in entry!.Value.entries)
                        bySource[(entry.Value.type, method)] = signature;

                var resolved = new List<(Site site, Signature signature)>();
                foreach (var site in found)
                {
                    var signature = site!.Signature
                        ?? (bySource.TryGetValue((site.ReceiverType, site.Method), out var s) ? s : null);
                    if (signature is not null) resolved.Add((site, signature));
                }
                if (resolved.Count == 0) return;

                if (!isEnabled)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(InterceptorsNotEnabled, Location.None));
                    return;
                }
                spc.AddSource("ShorokooLocalParamNames.g.cs", SourceText.From(Emit(resolved), Encoding.UTF8));
            });
        }

        private static bool IsLocalAssignedCall(SyntaxNode node)
            => node is InvocationExpressionSyntax
            {
                Expression: MemberAccessExpressionSyntax { Name: IdentifierNameSyntax { Identifier.Text: "Init" or "Model" } },
                Parent: EqualsValueClauseSyntax
                {
                    Parent: VariableDeclaratorSyntax { Parent: VariableDeclarationSyntax { Parent: LocalDeclarationStatementSyntax } }
                },
            };

        private static Site? Analyze(GeneratorSyntaxContext ctx, CancellationToken ct)
        {
            var invocation = (InvocationExpressionSyntax)ctx.Node;
            var access = (MemberAccessExpressionSyntax)invocation.Expression;
            var methodName = access.Name.Identifier.Text;
            var name = ((VariableDeclaratorSyntax)invocation.Parent!.Parent!).Identifier.Text;
            if (name == "_") return null;

            var model = ctx.SemanticModel;
            if (!InModuleClass(invocation, model, ct)) return null;
            if (model.GetSymbolInfo(access.Expression, ct).Symbol is not INamedTypeSymbol receiver) return null;
            for (var t = receiver; t is not null; t = t.ContainingType)
                if (t.IsGenericType) return null;
            if (!HasAttribute(receiver, methodName == "Model"
                    ? ["ModuleAttribute"]
                    : ["TrainableParamInitializerAttribute", "StateInitializerAttribute"]))
                return null;
            if (!model.Compilation.IsSymbolAccessibleWithin(receiver, model.Compilation.Assembly)) return null;

            var location = model.GetInterceptableLocation(invocation, ct);
            if (location is null) return null;

            var fq = SymbolDisplayFormat.FullyQualifiedFormat;
            Signature? signature = null;
            if (model.GetSymbolInfo(invocation, ct).Symbol is IMethodSymbol method)
            {
                if (!method.IsStatic || method.IsGenericMethod || method.IsExtensionMethod) return null;
                if (method.Parameters.Any(p => p.RefKind != RefKind.None)) return null;
                if (!method.ReturnType.AllInterfaces.Any(i => i.ToDisplayString() == "Shorokoo.IModuleParam")) return null;
                signature = new Signature(method.ReturnType.ToDisplayString(fq),
                    new EquatableArray(method.Parameters.Select(p => p.Type.ToDisplayString(fq)).ToImmutableArray()));
            }
            else if (!receiver.Locations.Any(l => l.IsInSource))
                return null;

            return new Site(location.GetInterceptsLocationAttributeSyntax(),
                receiver.ToDisplayString(fq) + "." + methodName,
                receiver.ToDisplayString(), methodName, name, signature);
        }

        /// <summary>The <c>Init</c>/<c>Model</c> signatures <see cref="ModuleSourceGenerator"/> writes
        /// for one class of this project, keyed by the class's display name.</summary>
        private static (string type, EquatableArray<(string method, Signature signature)> entries)? GeneratedSignatures(GeneratorSyntaxContext ctx)
        {
            if (ModuleSourceGenerator.GetModuleClassInfo(ctx) is not { } info) return null;
            if (ctx.SemanticModel.GetDeclaredSymbol((ClassDeclarationSyntax)ctx.Node) is not INamedTypeSymbol symbol) return null;
            var entries = new List<(string, Signature)>();
            foreach (var method in new[] { "Init", "Model" })
                if (ModuleSourceGenerator.GeneratedSignature(info, method) is { } s)
                    entries.Add((method, new Signature(s.returnType, new EquatableArray(s.parameterTypes.ToImmutableArray()))));
            return entries.Count == 0 ? null : (symbol.ToDisplayString(), new EquatableArray<(string, Signature)>(entries.ToImmutableArray()));
        }

        private static bool InModuleClass(SyntaxNode node, SemanticModel model, CancellationToken ct)
        {
            foreach (var cls in node.Ancestors().OfType<ClassDeclarationSyntax>())
                if (model.GetDeclaredSymbol(cls, ct) is { } symbol && HasAttribute(symbol, ["ModuleAttribute"]))
                    return true;
            return false;
        }

        private static bool HasAttribute(INamedTypeSymbol type, string[] names)
            => type.GetAttributes().Any(a => a.AttributeClass is { } c && names.Contains(c.Name));

        private static bool IsEnabled(ParseOptions options)
        {
            foreach (var key in new[] { "InterceptorsNamespaces", "InterceptorsPreviewNamespaces" })
                if (options.Features.TryGetValue(key, out var value) &&
                    value.Split(';').Any(ns => ns.Trim() == InterceptorNamespace))
                    return true;
            return false;
        }

        private static string Emit(List<(Site site, Signature signature)> sites)
        {
            var sb = new StringBuilder()
                .AppendLine("// <auto-generated/>")
                .AppendLine("#nullable enable")
                .AppendLine("namespace System.Runtime.CompilerServices")
                .AppendLine("{")
                .AppendLine("    [global::System.AttributeUsage(global::System.AttributeTargets.Method, AllowMultiple = true)]")
                .AppendLine("    file sealed class InterceptsLocationAttribute : global::System.Attribute")
                .AppendLine("    {")
                .AppendLine("        public InterceptsLocationAttribute(int version, string data) { }")
                .AppendLine("    }")
                .AppendLine("}")
                .AppendLine()
                .AppendLine($"namespace {InterceptorNamespace}")
                .AppendLine("{")
                .AppendLine("    file static class LocalNames")
                .AppendLine("    {");
            int i = 0;
            foreach (var (site, signature) in sites)
            {
                var parameters = string.Join(", ", signature.Parameters.Items.Select((t, k) => $"{t} a{k}"));
                var arguments = string.Join(", ", signature.Parameters.Items.Select((_, k) => $"a{k}"));
                sb.AppendLine($"        {site.Attribute}")
                  .AppendLine($"        public static {signature.ReturnType} N{i++}({parameters})")
                  .AppendLine($"            => global::Shorokoo.ParamNames.NamedByLocal({site.Target}({arguments}), {SymbolDisplay.FormatLiteral(site.Name, quote: true)});");
            }
            return sb.AppendLine("    }").AppendLine("}").ToString();
        }
    }

    /// <summary>An array compared by its elements, so the incremental pipeline can cache on it.</summary>
    internal sealed class EquatableArray<T>(ImmutableArray<T> items) : IEquatable<EquatableArray<T>>
    {
        public ImmutableArray<T> Items { get; } = items;
        public bool Equals(EquatableArray<T>? other) => other is not null && Items.SequenceEqual(other.Items);
        public override bool Equals(object? obj) => Equals(obj as EquatableArray<T>);
        public override int GetHashCode()
        {
            int h = 17;
            foreach (var item in Items) h = h * 31 + (item?.GetHashCode() ?? 0);
            return h;
        }
        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)Items).GetEnumerator();
    }

    /// <summary>An <see cref="EquatableArray{T}"/> of strings.</summary>
    internal sealed class EquatableArray(ImmutableArray<string> items) : IEquatable<EquatableArray>
    {
        public ImmutableArray<string> Items { get; } = items;
        public bool Equals(EquatableArray? other) => other is not null && Items.SequenceEqual(other.Items);
        public override bool Equals(object? obj) => Equals(obj as EquatableArray);
        public override int GetHashCode()
        {
            int h = 17;
            foreach (var item in Items) h = h * 31 + item.GetHashCode();
            return h;
        }
    }
}
