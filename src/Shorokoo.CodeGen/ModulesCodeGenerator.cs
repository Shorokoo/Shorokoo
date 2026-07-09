using System.Diagnostics;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

[Generator]
public class ModuleSourceGenerator : IIncrementalGenerator
{
    // Diagnostics
    private static readonly DiagnosticDescriptor GeneratorError = new(
        id: "MSG001",
        title: "Module Source Generator Error",
        messageFormat: "An exception occurred while generating code for class '{0}': {1}",
        category: "SourceGeneration",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor InvalidModuleMethod = new(
        id: "MSG002",
        title: "Invalid Module Method Format",
        messageFormat: "Method '{0}' in class '{1}' has an invalid module signature or naming",
        category: "SourceGeneration",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor InitializerClassNamedInit = new(
        id: "MSG003",
        title: "Initializer class must not be named 'Init'",
        messageFormat: "The [TrainableParamInitializer]/[StateInitializer] class '{0}' cannot be named 'Init': the generator adds an 'Init(...)' method to it, and C# forbids a member named like its enclosing type (CS0542). Rename the class (e.g. 'ConstInit').",
        category: "SourceGeneration",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor RngPinAvailable = new(
        id: "MSG004",
        title: "RNG streams of this module can be pinned",
        messageFormat: "Module '{0}' has RNG stream consumers capturable by Rng.Pin. To freeze their stream identities against refactoring, {1}.",
        category: "SourceGeneration",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Add a simple marker to verify the generator is running
        context.RegisterPostInitializationOutput(ctx => {
            ctx.AddSource(
                "ShorokooGeneratorMarker.g.cs", 
                SourceText.From("// This file indicates the Shorokoo.CodeGen generator ran successfully at " + System.DateTime.Now.ToString() + "\n", Encoding.UTF8));
        });

        // Find all partial classes with [Module] attribute (both static and non-static).
        var moduleClasses = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => IsPotentialModuleClass(node),
                transform: static (ctx, _) => GetModuleClassInfo(ctx))
            .Where(static info => info is not null)
            .Collect();

        context.RegisterSourceOutput(moduleClasses, static (spc, classes) =>
        {
            foreach (var classInfo in classes)
            {
                if (classInfo is null) continue;

                // The generated initializer member is named Init; a class also named Init
                // would make that member collide with its enclosing type (CS0542 deep in
                // generated code). Fail with a clear, actionable error instead.
                if (classInfo.IsNewStyleInitializer && classInfo.ClassName == "Init")
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InitializerClassNamedInit,
                        classInfo.Location ?? Location.None,
                        classInfo.ClassName));
                    continue;
                }

                try
                {
                    var generatedCode = GenerateCode(classInfo);
                    spc.AddSource($"{classInfo.ClassName}_V2Generated.cs", SourceText.From(generatedCode, Encoding.UTF8));

                    // Rng.Pin discoverability: for straight-line Inline bodies whose random
                    // consumers (X.Model(...) / X.Init(...) results) are all captured in
                    // locals, surface the ready-to-paste pin statement as an Info diagnostic.
                    if (!classInfo.IsNewStyleInitializer &&
                        classInfo.Location?.SourceTree?.GetRoot().FindNode(classInfo.Location.SourceSpan)
                            is ClassDeclarationSyntax classSyntax)
                    {
                        var pinSuggestion = TryBuildRngPinSuggestion(classSyntax);
                        if (pinSuggestion is not null)
                            spc.ReportDiagnostic(Diagnostic.Create(
                                RngPinAvailable, classInfo.Location, classInfo.ClassName, pinSuggestion));
                    }

                    
                    // Report warnings for ignored methods (bad format / explicitly ignored)
                    foreach (var fm in classInfo.FullModules)
                    {
                        if (fm.Kind == ModuleKind.Ignore && fm.Location is not null)
                        {
                            spc.ReportDiagnostic(Diagnostic.Create(
                                InvalidModuleMethod,
                                fm.Location,
                                fm.FnName,
                                classInfo.ClassName));
                        }
                    }
                }
                catch (Exception e)
                {
                    // Emit as Roslyn error diagnostic instead of writing a diagnostics .cs file
                    spc.ReportDiagnostic(Diagnostic.Create(
                        GeneratorError,
                        classInfo.Location ?? Location.None,
                        classInfo.ClassName,
                        e.Message));
                }
            }
        });
    }

    /// <summary>
    /// Builds the ready-to-paste <c>Rng.Pin(...)</c> guidance for a module whose RNG stream
    /// consumers are all provably nameable. Each scope — the <c>Inline</c> body and every
    /// <c>LoopAPI.Iterate(...)</c> loop body, at any nesting — is pinned independently: the
    /// nameable <c>X.Model(...)</c> / <c>X.Init(...)</c> / <c>RandomUniform(...)</c> /
    /// <c>RandomNormal(...)</c> captures directly in that scope take its local id slots in
    /// source order (a captured feed tensor is pinnable exactly like a captured Init result —
    /// feeds share the module's id address space), and a nested loop takes one local slot too.
    /// So the guidance is one compilable <c>Rng.Pin(...)</c> per scope, placed at the end of
    /// that scope (loop-body pins go inside the loop, where those variables are in scope).
    /// Within a scope, when a nested loop occupies a local slot the pin uses the <b>sparse</b>
    /// form so the named items keep their exact slots and the loop's slot is left to be filled
    /// positionally; otherwise the terse positional form is exact.
    ///
    /// <para>Returns null (no diagnostic) for anything not provably capturable in <em>any</em>
    /// scope: C# control flow (<c>if</c>/<c>for</c>/<c>while</c>/<c>switch</c>/a non-<c>Iterate</c>
    /// <c>foreach</c>), an existing <c>Rng.Pin</c>, an uncaptured Model/Init/feed, any
    /// <c>.Call</c> whose receiver is not a counted <c>Recv.Model(...)</c> capture (static
    /// <c>TypeName.Call</c>, chained <c>Model(...).Call(...)</c>, or an unknown receiver that
    /// may be a stream-creating helper), or an opaque static-helper call that may create
    /// streams. A wrong pin silently re-keys, so anything uncertain refuses. (Method-call
    /// "control flow" like <c>cond.IfElse(...)</c> is an ordinary expression and is fine.)</para>
    /// </summary>
    internal static string? TryBuildRngPinSuggestion(ClassDeclarationSyntax classDecl)
    {
        var inline = classDecl.Members.OfType<MethodDeclarationSyntax>()
            .FirstOrDefault(m => m.Identifier.Text == "Inline" && m.Body is not null);
        if (inline?.Body is null) return null;

        // Already pinned anywhere in the body: no suggestion needed.
        foreach (var inv in inline.Body.DescendantNodes().OfType<InvocationExpressionSyntax>())
            if (inv.Expression is MemberAccessExpressionSyntax pinMa &&
                pinMa.Name.Identifier.Text == "Pin" &&
                pinMa.Expression is IdentifierNameSyntax { Identifier.Text: "Rng" })
                return null;

        var scopes = new List<(string placement, string pin)>();
        var modelCaptures = new HashSet<string>(StringComparer.Ordinal);
        if (!TryAnalyzeScope(inline.Body.Statements, "at the end of Inline", scopes, modelCaptures, out _))
            return null;
        if (scopes.Count == 0) return null;

        if (scopes.Count == 1 && scopes[0].placement == "at the end of Inline")
            return "paste " + scopes[0].placement + ": " + scopes[0].pin;   // common single-scope case

        return "paste a pin at the end of each scope — " +
            string.Join(" ", scopes.Select(s => s.placement + ": " + s.pin));
    }

    /// <summary>
    /// Recursively analyzes one scope's statements (the module body or a loop body). On success
    /// appends this scope's compilable <c>Rng.Pin(...)</c> — module scope first, then nested
    /// scopes in source order — to <paramref name="scopes"/> and reports via
    /// <paramref name="occupies"/> whether the scope contains any id-bearing consumer (so its
    /// parent counts it as one local slot). <paramref name="modelCaptures"/> accumulates the
    /// names of locals captured via <c>Recv.Model(...)</c> — the only receivers a stream-free
    /// <c>.Call(...)</c> can be proven against (see <see cref="SubtreeHasUnnameableStream"/>);
    /// each loop body recurses on a copy so sibling scopes never see each other's captures.
    /// Returns false if anything in this scope (or a nested one) is not provably capturable,
    /// so the whole suggestion is withheld.
    /// </summary>
    private static bool TryAnalyzeScope(
        IReadOnlyList<StatementSyntax> statements,
        string placement,
        List<(string placement, string pin)> scopes,
        HashSet<string> modelCaptures,
        out bool occupies)
    {
        occupies = false;
        var pinned = new List<(int slot, string name)>();
        int slot = 0;
        bool anyLoopSlot = false;
        var childScopes = new List<(string placement, string pin)>();

        foreach (var stmt in statements)
        {
            switch (stmt)
            {
                case LocalDeclarationStatementSyntax decl:
                    foreach (var v in decl.Declaration.Variables)
                    {
                        var init = v.Initializer?.Value;
                        if (init is InvocationExpressionSyntax inv &&
                            (IsSimpleModelOrInit(inv) || IsSimpleFeed(inv)))
                        {
                            // The receiver is a plain name, but the ARGUMENTS could still create
                            // streams (a nested Model/Init/feed, static Call, opaque helper) — those
                            // would be uncounted siblings, so refuse rather than emit a mis-slotting pin.
                            if (inv.ArgumentList is { } args && SubtreeHasUnnameableStream(args, modelCaptures))
                                return false;
                            slot++;
                            pinned.Add((slot, v.Identifier.Text));
                            if (inv.Expression is MemberAccessExpressionSyntax { Name.Identifier.Text: "Model" })
                                modelCaptures.Add(v.Identifier.Text);
                        }
                        else if (init is not null && SubtreeHasUnnameableStream(init, modelCaptures))
                        {
                            return false;   // uncaptured Model/Init/feed, unproven .Call, opaque helper
                        }
                        // else: pure math (ShapeTensor, arithmetic, .Call on a counted model) — 0 slots
                    }
                    break;

                case ExpressionStatementSyntax es:
                    if (SubtreeHasUnnameableStream(es.Expression, modelCaptures)) return false;
                    break;

                case ReturnStatementSyntax rs:
                    if (rs.Expression is not null && SubtreeHasUnnameableStream(rs.Expression, modelCaptures)) return false;
                    break;

                case ForEachStatementSyntax fe when IsLoopApiIterate(fe):
                    // The Iterate(...) header's arguments could create streams too (a feed or
                    // Model/Init used to compute the trip count) — scan them, but not the
                    // Iterate call itself (its uppercase receiver would always trip the check).
                    if (fe.Expression is InvocationExpressionSyntax iterInv &&
                        iterInv.ArgumentList is { } iterArgs && SubtreeHasUnnameableStream(iterArgs, modelCaptures))
                        return false;
                    var body = fe.Statement is BlockSyntax b
                        ? (IReadOnlyList<StatementSyntax>)b.Statements
                        : (StatementSyntax[])[fe.Statement];
                    if (!TryAnalyzeScope(body, LoopPlacement(fe), childScopes,
                            new HashSet<string>(modelCaptures, StringComparer.Ordinal), out var childOccupies))
                        return false;
                    if (childOccupies) { slot++; anyLoopSlot = true; }
                    // A loop with no id-bearing content takes no local slot.
                    break;

                default:
                    return false;   // C# control flow / unsupported statement
            }
        }

        occupies = pinned.Count > 0 || anyLoopSlot;

        if (pinned.Count > 0)
        {
            // A nested loop occupying a local slot forces the sparse form so the named items
            // keep their exact slots; otherwise the terse positional form is exact.
            var pin = anyLoopSlot
                ? "Rng.Pin(" + string.Join(", ", pinned.Select(p => "([" + p.slot + "], " + p.name + ")")) + ");"
                : "Rng.Pin(" + string.Join(", ", pinned.Select(p => p.name)) + ");";
            scopes.Add((placement, pin));
        }
        scopes.AddRange(childScopes);
        return true;
    }

    /// <summary>Human-readable placement for a loop-body pin, echoing the loop header.</summary>
    private static string LoopPlacement(ForEachStatementSyntax fe)
        => $"inside `foreach (var {fe.Identifier.Text} in {fe.Expression})`";

    /// <summary>A direct top-level capture: <c>Recv.Model(...)</c> / <c>Recv.Init(...)</c> whose
    /// receiver is a plain dotted name (not a chained invocation).</summary>
    private static bool IsSimpleModelOrInit(InvocationExpressionSyntax inv)
    {
        if (inv.Expression is not MemberAccessExpressionSyntax ma) return false;
        var name = ma.Name.Identifier.Text;
        if (name != "Model" && name != "Init") return false;
        // Receiver must contain no invocation (excludes chained X.Model(...).Call(...) etc.).
        return !ma.Expression.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>().Any();
    }

    /// <summary>A direct top-level feed capture: bare <c>RandomUniform(...)</c> /
    /// <c>RandomNormal(...)</c> (via <c>using static Globals</c>) or the
    /// <c>Globals.</c>-qualified form. A captured feed tensor is pinnable exactly like a
    /// captured Init result — feeds share the module's id address space — so it takes a
    /// local slot instead of refusing the whole suggestion.</summary>
    private static bool IsSimpleFeed(InvocationExpressionSyntax inv)
    {
        if (inv.Expression is IdentifierNameSyntax bare)
            return bare.Identifier.Text is "RandomUniform" or "RandomNormal";
        return inv.Expression is MemberAccessExpressionSyntax ma
            && ma.Name.Identifier.Text is "RandomUniform" or "RandomNormal"
            && ma.Expression is IdentifierNameSyntax { Identifier.Text: "Globals" };
    }

    /// <summary>Whether a subtree touches an RNG stream that an end-of-body pin cannot name:
    /// an (uncaptured) Model/Init call, a runtime feed, any <c>.Call</c> whose receiver is not
    /// a COUNTED model capture (static <c>TypeName.Call</c> creates an anonymous sub-model;
    /// a chained or unknown receiver — even a lowercase one — may be a helper whose Call
    /// creates streams: a lowercase name alone proves nothing), or an opaque uppercase
    /// static-helper call that may create streams internally. Only <c>m.Call(...)</c> where
    /// <c>m</c> is in <paramref name="modelCaptures"/> (captured via <c>Recv.Model(...)</c> in
    /// this analysis) is provably a stream-free re-invocation. Boundary of the syntax-only
    /// analysis: instance methods with OTHER names on lowercase receivers (tensor math,
    /// <c>ctx.ContinueWhile</c>) and bare using-static calls other than the feeds are trusted —
    /// a user helper smuggling stream creation through those shapes is not detectable without
    /// semantic info.</summary>
    private static bool SubtreeHasUnnameableStream(SyntaxNode node, HashSet<string> modelCaptures)
    {
        foreach (var inv in node.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>())
        {
            // Bare feed via `using static Globals`: RandomUniform(...) / RandomNormal(...).
            if (inv.Expression is IdentifierNameSyntax bare &&
                (bare.Identifier.Text == "RandomUniform" || bare.Identifier.Text == "RandomNormal"))
                return true;

            if (inv.Expression is MemberAccessExpressionSyntax ma)
            {
                var name = ma.Name.Identifier.Text;
                if (name is "Model" or "Init" or "RandomUniform" or "RandomNormal") return true;
                if (name == "Call")
                {
                    // Stream-free only when provably re-invoking a model this analysis counted.
                    if (ma.Expression is IdentifierNameSyntax rcv &&
                        modelCaptures.Contains(rcv.Identifier.Text))
                        continue;
                    return true;
                }
                // Opaque uppercase static-helper call (DropoutMasking.Mask, Recurrent.RNN, ...):
                // may create streams internally — can't name them, so refuse.
                if (ma.Expression is IdentifierNameSyntax uc && char.IsUpper(uc.Identifier.Text[0]))
                    return true;
            }
        }
        return false;
    }

    private static bool IsIterateInvocation(InvocationExpressionSyntax inv)
        => inv.Expression is MemberAccessExpressionSyntax ma && ma.Name.Identifier.Text == "Iterate";

    private static bool IsLoopApiIterate(ForEachStatementSyntax fe)
        => fe.Expression is InvocationExpressionSyntax inv && IsIterateInvocation(inv);

    // V2: Partial class with [Module] attribute (both static and non-static)
    private static bool IsPotentialModuleClass(SyntaxNode node)
        => node is ClassDeclarationSyntax classDecl &&
           classDecl.Modifiers.Any(SyntaxKind.PartialKeyword);

    private static ModuleClassInfo? GetModuleClassInfo(GeneratorSyntaxContext context)
    {
        var classDeclaration = (ClassDeclarationSyntax)context.Node;
        var semanticModel = context.SemanticModel;
        var classSymbol = semanticModel.GetDeclaredSymbol(classDeclaration) as INamedTypeSymbol;
        if (classSymbol is null) return null;

        // Check for the [Module] attribute by simple name match.TrainableParamInitializer 
        var hasModuleAttribute = classSymbol.GetAttributes()
            .Any(attr => attr.AttributeClass?.Name == "ModuleAttribute");

        // Check for new-style class-level [TrainableParamInitializer] or [StateInitializer] attributes
        var hasClassTrainableParamInitializer = classSymbol.GetAttributes()
            .Any(attr => attr.AttributeClass?.Name == "TrainableParamInitializerAttribute");
        var stateInitializerAttr = classSymbol.GetAttributes()
            .FirstOrDefault(attr => attr.AttributeClass?.Name == "StateInitializerAttribute");
        var hasClassStateInitializer = stateInitializerAttr is not null;

        // Read the [StateInitializer(Ownership = ...)] named argument; the enum value's integer
        // is mapped back to the StateOwnership member name when the Init method is emitted.
        // Defaults to ModuleOwned, matching the attribute's own default.
        var stateOwnershipName = "ModuleOwned";
        if (stateInitializerAttr is not null)
        {
            foreach (var named in stateInitializerAttr.NamedArguments)
            {
                if (named.Key == "Ownership" && named.Value.Value is int ownershipValue)
                    stateOwnershipName = ownershipValue == 1 ? "OptimizerOwned" : "ModuleOwned";
            }
        }

        // If none of the recognized attributes, skip this class
        if (!hasModuleAttribute && !hasClassTrainableParamInitializer && !hasClassStateInitializer) return null;

        var isStatic = classDeclaration.Modifiers.Any(SyntaxKind.StaticKeyword);
        
        // Handle new-style class-level TrainableParamInitializer or StateInitializer
        if (hasClassTrainableParamInitializer || hasClassStateInitializer)
        {
            // Look for a static method named "Inline"
            var inlineMethod = classSymbol.GetMembers()
                .OfType<IMethodSymbol>()
                .FirstOrDefault(m => m.Name == "Inline" && m.IsStatic);

            if (inlineMethod is null) return null;

            var moduleInfo = ProcessMethod(inlineMethod, classDeclaration.SyntaxTree);
            if (moduleInfo is null) return null;

            // Extract generic type parameters from the Inline method
            var (typeParameterList, typeConstraintClauses) = ExtractGenericTypeInfoFromMethod(inlineMethod);

            // Use TrainableParamInitializer kind for both (StateInitializer treated same way for code gen)
            var fullModuleInfo = new FullModuleInfo(
                classSymbol.Name,
                "Inline",
                ModuleKind.TrainableParamInitializer,
                moduleInfo.ReturnParams,
                moduleInfo.Hyperparams,
                moduleInfo.InputParams,
                moduleInfo.Location);

            return new ModuleClassInfo(
                classSymbol.Name,
                classSymbol.ContainingNamespace.ToDisplayString(),
                new List<FullModuleInfo> { fullModuleInfo },
                classDeclaration.GetLocation(),
                isStaticClass: true,  // These are always static classes
                typeParameterList,
                typeConstraintClauses,
                inlineMethod,
                isNewStyleInitializer: true,  // Flag for new-style initializer
                isStateInitializer: hasClassStateInitializer,  // True if [StateInitializer]
                stateOwnershipName: stateOwnershipName);
        }
        
        if (isStatic)
        {
            // For static classes with [Module], process all methods (TrainableParamInitializers)
            // Extract generic type parameters from class (legacy behavior for static classes)
            var (typeParameterList, typeConstraintClauses) = ExtractGenericTypeInfo(classSymbol, classDeclaration);
            
            var moduleInfos = classSymbol.GetMembers()
                .OfType<IMethodSymbol>()
                .Select(m => ProcessMethod(m, classDeclaration.SyntaxTree))
                .Where(static x => x is not null)
                .Select(static x => x!)
                .ToList();

            return new ModuleClassInfo(
                classSymbol.Name,
                classSymbol.ContainingNamespace.ToDisplayString(),
                moduleInfos,
                classDeclaration.GetLocation(),
                isStaticClass: true,
                typeParameterList,
                typeConstraintClauses);
        }
        else
        {
            // For non-static classes with [Module], look for a static method named "Inline"
            var inlineMethod = classSymbol.GetMembers()
                .OfType<IMethodSymbol>()
                .FirstOrDefault(m => m.Name == "Inline" && m.IsStatic);

            if (inlineMethod is null) return null;

            var moduleInfo = ProcessMethod(inlineMethod, classDeclaration.SyntaxTree);
            if (moduleInfo is null) return null;

            // Extract generic type parameters from the Inline method, not the class
            var (typeParameterList, typeConstraintClauses) = ExtractGenericTypeInfoFromMethod(inlineMethod);

            // The module name is the class name itself
            var fullModuleInfo = new FullModuleInfo(
                classSymbol.Name,
                "Inline",
                ModuleKind.ModuleImplementation,
                moduleInfo.ReturnParams,
                moduleInfo.Hyperparams,
                moduleInfo.InputParams,
                moduleInfo.Location);

            return new ModuleClassInfo(
                classSymbol.Name,
                classSymbol.ContainingNamespace.ToDisplayString(),
                new List<FullModuleInfo> { fullModuleInfo },
                classDeclaration.GetLocation(),
                isStaticClass: false,
                typeParameterList,
                typeConstraintClauses,
                inlineMethod);  // Pass the inline method
        }
    }

    private static (string typeParameterList, string typeConstraintClauses) ExtractGenericTypeInfo(INamedTypeSymbol classSymbol, ClassDeclarationSyntax classDeclaration)
    {
        if (!classSymbol.IsGenericType || classSymbol.TypeParameters.Length == 0)
        {
            return (string.Empty, string.Empty);
        }

        // Build type parameter list: <T> or <T1, T2>
        var typeParams = string.Join(", ", classSymbol.TypeParameters.Select(tp => tp.Name));
        var typeParameterList = $"<{typeParams}>";

        // Build constraint clauses from syntax (more reliable than semantic model for this)
        var constraintClauses = new System.Collections.Generic.List<string>();
        if (classDeclaration.ConstraintClauses != null)
        {
            foreach (var clause in classDeclaration.ConstraintClauses)
            {
                // Get the full clause text as written in source
                constraintClauses.Add(clause.ToString());
            }
        }

        var typeConstraintClauses = constraintClauses.Count > 0 
            ? " " + string.Join(" ", constraintClauses)
            : string.Empty;

        return (typeParameterList, typeConstraintClauses);
    }

    private static (string typeParameterList, string typeConstraintClauses) ExtractGenericTypeInfoFromMethod(IMethodSymbol methodSymbol)
    {
        if (!methodSymbol.IsGenericMethod || methodSymbol.TypeParameters.Length == 0)
        {
            return (string.Empty, string.Empty);
        }

        // Build type parameter list: <T> or <T1, T2>
        var typeParams = string.Join(", ", methodSymbol.TypeParameters.Select(tp => tp.Name));
        var typeParameterList = $"<{typeParams}>";

        // Build constraint clauses from type parameters
        var constraintClauses = new System.Collections.Generic.List<string>();
        foreach (var typeParam in methodSymbol.TypeParameters)
        {
            var constraints = new System.Collections.Generic.List<string>();
            
            // Check for class/struct constraint
            if (typeParam.HasReferenceTypeConstraint)
                constraints.Add("class");
            if (typeParam.HasValueTypeConstraint)
                constraints.Add("struct");
            if (typeParam.HasUnmanagedTypeConstraint)
                constraints.Add("unmanaged");
            if (typeParam.HasNotNullConstraint)
                constraints.Add("notnull");
            
            // Add type constraints
            foreach (var constraint in typeParam.ConstraintTypes)
            {
                constraints.Add(constraint.ToDisplayString());
            }
            
            // Check for constructor constraint
            if (typeParam.HasConstructorConstraint)
                constraints.Add("new()");
            
            if (constraints.Count > 0)
            {
                constraintClauses.Add($"where {typeParam.Name} : {string.Join(", ", constraints)}");
            }
        }

        var typeConstraintClauses = constraintClauses.Count > 0 
            ? " " + string.Join(" ", constraintClauses)
            : string.Empty;

        return (typeParameterList, typeConstraintClauses);
    }

    // Helper to find which generic type parameters are used in a list of parameters
    private static HashSet<string> FindUsedTypeParameters(IEnumerable<ParamData> parameters, IMethodSymbol method)
    {
        var usedTypeParams = new HashSet<string>();
        if (!method.IsGenericMethod) return usedTypeParams;

        var typeParamNames = new HashSet<string>(method.TypeParameters.Select(tp => tp.Name));

        foreach (var param in parameters)
        {
            // Check if any type parameter name appears in the type declaration
            foreach (var typeParamName in typeParamNames)
            {
                if (param.TypeDeclaration.Contains($"<{typeParamName}>") || 
                    param.TypeDeclaration.Contains($"<{typeParamName},") ||
                    param.TypeDeclaration.Contains($", {typeParamName}>") ||
                    param.TypeDeclaration.Contains($", {typeParamName},"))
                {
                    usedTypeParams.Add(typeParamName);
                }
            }
        }

        return usedTypeParams;
    }

    // Build type parameter list and constraints for a subset of type parameters
    private static (string typeParameterList, string typeConstraintClauses) BuildTypeParamsSubset(
        IMethodSymbol method, 
        HashSet<string> typeParamNamesToInclude)
    {
        if (!method.IsGenericMethod || typeParamNamesToInclude.Count == 0)
        {
            return (string.Empty, string.Empty);
        }

        var includedTypeParams = method.TypeParameters
            .Where(tp => typeParamNamesToInclude.Contains(tp.Name))
            .ToList();

        if (includedTypeParams.Count == 0)
        {
            return (string.Empty, string.Empty);
        }

        // Build type parameter list
        var typeParams = string.Join(", ", includedTypeParams.Select(tp => tp.Name));
        var typeParameterList = $"<{typeParams}>";

        // Build constraint clauses
        var constraintClauses = new System.Collections.Generic.List<string>();
        foreach (var typeParam in includedTypeParams)
        {
            var constraints = new System.Collections.Generic.List<string>();
            
            if (typeParam.HasReferenceTypeConstraint)
                constraints.Add("class");
            if (typeParam.HasValueTypeConstraint)
                constraints.Add("struct");
            if (typeParam.HasUnmanagedTypeConstraint)
                constraints.Add("unmanaged");
            if (typeParam.HasNotNullConstraint)
                constraints.Add("notnull");
            
            foreach (var constraint in typeParam.ConstraintTypes)
            {
                constraints.Add(constraint.ToDisplayString());
            }
            
            if (typeParam.HasConstructorConstraint)
                constraints.Add("new()");
            
            if (constraints.Count > 0)
            {
                constraintClauses.Add($"where {typeParam.Name} : {string.Join(", ", constraints)}");
            }
        }

        var typeConstraintClauses = constraintClauses.Count > 0 
            ? " " + string.Join(" ", constraintClauses)
            : string.Empty;

        return (typeParameterList, typeConstraintClauses);
    }

    private static FullModuleInfo? ProcessMethod(IMethodSymbol m, SyntaxTree tree)
    {
        var isTrainableParam = m.GetAttributes().Any(attr => attr.AttributeClass?.Name == "TrainableParamInitializerAttribute");

        var paramsData = m.Parameters.Select(ProcessParameter).ToList();
        // Input (tensor) parameters come first, hyperparameters last. A [Hyper] parameter
        // followed by a (later) non-hyper input is a malformed signature.
        var hasBadHyperparams = paramsData
            .SkipWhile(p => p.Kind != ParamKind.Hyperparam)
            .Any(p => p.Kind != ParamKind.Hyperparam);

        var inputParams = paramsData.TakeWhile(p => p.Kind != ParamKind.Hyperparam).ToList();
        var hyperParams = paramsData.SkipWhile(p => p.Kind != ParamKind.Hyperparam).ToList();
        var returnParams = ProcessReturnType(m.ReturnType)
            .Where(static x => x is not null)
            .Select(static x => x!)
            .ToList();

        var methodName = m.Name;
        var isBadFormat = hasBadHyperparams;

        if (isTrainableParam)
        {
            isBadFormat = isBadFormat ||
                          hyperParams.Count > 0 ||
                          (!char.IsLower(m.Name, 0) && m.Name[0] != '_');

            if (char.IsLower(m.Name, 0))
                methodName = char.ToUpper(m.Name[0]) + m.Name.Substring(1);
            else if (m.Name.Length > 1 && m.Name[0] == '_')
                methodName = m.Name.Substring(1);
            else
                methodName = "Create" + m.Name;
        }

        var moduleKind = isBadFormat
            ? ModuleKind.Ignore
            : isTrainableParam ? ModuleKind.TrainableParamInitializer : ModuleKind.ModuleImplementation;

        // Best-effort: use the declaring syntax reference for precise location
        var location = m.DeclaringSyntaxReferences.FirstOrDefault()?.GetSyntax() is { } syntax
            ? syntax.GetLocation()
            : Location.Create(tree, default);

        return new FullModuleInfo(methodName, m.Name, moduleKind, returnParams, hyperParams, inputParams, location);
    }

    private static ParamData ProcessParameter(IParameterSymbol p)
    {
        var attributes = p.GetAttributes();
        var hyperAttr = attributes.FirstOrDefault(x => x.AttributeClass?.Name == "HyperAttribute");
        var attributeKind = hyperAttr is not null ? ParamKind.Hyperparam : ParamKind.InputParam;

        string? defaultLiteral = null;
        if (hyperAttr is not null && hyperAttr.ConstructorArguments.Length > 0)
            defaultLiteral = FormatFloatLiteral(hyperAttr.ConstructorArguments[0].Value);

        return new ParamData(p.Name, p.Type.ToDisplayString(), attributeKind, defaultLiteral);
    }

    /// <summary>
    /// Formats a boxed numeric constant (from a <c>[Hyper(default)]</c> argument) as a C# float
    /// literal, e.g. <c>0.9f</c> or <c>1E-08f</c>. Returns <c>null</c> if the value isn't numeric.
    /// </summary>
    private static string? FormatFloatLiteral(object? value)
    {
        if (value is null) return null;
        try
        {
            float f = Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture);
            return f.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "f";
        }
        catch
        {
            return null;
        }
    }

    /// <summary>PascalCase a camelCase parameter name for use as a generated property name.</summary>
    private static string Pascalize(string name)
        => string.IsNullOrEmpty(name) ? name : char.ToUpperInvariant(name[0]) + name.Substring(1);

    // ─────────────────── nullable / omittable caller-facing parameters ───────────────────
    // The Inline method declares non-nullable parameters; the generated Model/Call surface
    // exposes "omittable" ones as nullable with a `= null` default so callers can leave them out:
    //   * a [Hyper(default)] Scalar<float32> becomes `Scalar<float32>? = null`; when null the
    //     attribute's default constant is substituted.
    //   * an OptionalTensor<T> becomes `Tensor<T>? = null`; null is wrapped as an absent optional,
    //     a tensor as a present one.

    /// <summary>True for an <c>OptionalTensor&lt;T&gt;</c> parameter.</summary>
    private static bool ParamIsOptionalTensor(ParamData p) => p.TypeDeclaration.Contains("OptionalTensor<");

    /// <summary>True for a <c>[Hyper(default)] Scalar&lt;float32&gt;</c> parameter.</summary>
    private static bool ParamIsDefaultedFloatHyper(ParamData p)
        => p.Kind == ParamKind.Hyperparam && p.DefaultLiteral is not null
           && p.TypeDeclaration.Replace(" ", "").Contains("Scalar<") && p.TypeDeclaration.Contains("float32");

    /// <summary>True when the parameter is exposed as a nullable, omittable caller-facing parameter.</summary>
    private static bool ParamIsOmittable(ParamData p) => ParamIsOptionalTensor(p) || ParamIsDefaultedFloatHyper(p);

    /// <summary>The caller-facing parameter type (nullable for omittable parameters).</summary>
    private static string CallerFacingType(ParamData p)
    {
        if (ParamIsOptionalTensor(p))
        {
            var idx = p.TypeDeclaration.IndexOf("OptionalTensor<", System.StringComparison.Ordinal);
            var inner = p.TypeDeclaration.Substring(idx + "OptionalTensor<".Length);
            inner = inner.Substring(0, inner.LastIndexOf('>'));
            return $"global::Shorokoo.Tensor<{inner}>?";
        }
        if (ParamIsDefaultedFloatHyper(p))
            return $"{p.TypeDeclaration}?";
        return p.TypeDeclaration;
    }

    /// <summary>The expression converting a caller-facing argument to the module-facing value.</summary>
    private static string CallerToModuleArg(ParamData p)
    {
        if (ParamIsOptionalTensor(p))
            return $"global::Shorokoo.Globals.OptionalTensor({p.Name})";
        if (ParamIsDefaultedFloatHyper(p))
            return $"({p.Name} ?? global::Shorokoo.Globals.Scalar({p.DefaultLiteral}))";
        return p.Name;
    }

    /// <summary>
    /// Renders a caller-facing parameter declaration list, giving the trailing contiguous run of
    /// omittable parameters a <c>= null</c> default (a non-trailing omittable parameter stays
    /// nullable but required, so C#'s optional-parameters-last rule is respected).
    /// </summary>
    private static string CallerSignatureList(System.Collections.Generic.IReadOnlyList<ParamData> ps)
    {
        int n = ps.Count;
        var omittableTail = new bool[n];
        var tail = true;
        for (int i = n - 1; i >= 0; i--)
        {
            tail = tail && ParamIsOmittable(ps[i]);
            omittableTail[i] = tail;
        }
        var parts = new System.Collections.Generic.List<string>(n);
        for (int i = 0; i < n; i++)
            parts.Add($"{CallerFacingType(ps[i])} {ps[i].Name}" + (omittableTail[i] ? " = null" : ""));
        return string.Join(", ", parts);
    }

    /// <summary>
    /// True when every hyperparameter is a scalar <c>float32</c> — the schedulable optimizer shape
    /// for which we emit a strongly-typed <see cref="ParamData"/>-named hyperparameter set.
    /// </summary>
    private static bool AllHyperparamsAreFloatScalars(FullModuleInfo module)
        => module.Hyperparams.Count > 0
           && module.Hyperparams.All(h =>
               h.TypeDeclaration.Replace(" ", "").Contains("Scalar<")
               && h.TypeDeclaration.Contains("float32"));

    private static List<ParamData?> ProcessReturnType(ITypeSymbol returnType)
    {
        // Handle tuple return types
        if (returnType is INamedTypeSymbol namedType && namedType.IsTupleType)
            return [.. namedType.TupleElements.Select(p => new ParamData(p.Name, p.Type.ToDisplayString(), ParamKind.OutputParam))];

        return [new ParamData("", returnType.ToDisplayString(), ParamKind.OutputParam)];
    }

    private static string GenerateCode(ModuleClassInfo classInfo)
    {
        var sb = new StringBuilder();
        
        // For static classes with [Module], generate TrainableParamInitializer code
        if (classInfo.IsStaticClass)
        {
            return GenerateStaticClassCode(classInfo);
        }
        
        // For non-static classes, there should be exactly one module (the Inline method)
        var fullModule = classInfo.FullModules.FirstOrDefault();
        if (fullModule is null) return string.Empty;

        var hasHyperparams = fullModule.Hyperparams.Count > 0;
        var hasInputs = fullModule.InputParams.Count > 0;

        string HyperparamTypeString() => fullModule.Hyperparams.Count switch
        {
            > 1 => $"({string.Join(", ", fullModule.Hyperparams.Select(p => $"{p.TypeDeclaration} {p.Name}"))})",
            1 => fullModule.Hyperparams[0].TypeDeclaration,
            _ => string.Empty
        };

        string InputTypeList() => fullModule.InputParams.Count switch
        {
            > 1 => string.Join(", ", fullModule.InputParams.Select(p => p.TypeDeclaration)),
            1 => fullModule.InputParams[0].TypeDeclaration,
            _ => string.Empty
        };

        string InputTypeString() => fullModule.InputParams.Count switch
        {
            > 1 => $"({string.Join(", ", fullModule.InputParams.Select(p => $"{p.TypeDeclaration} {p.Name}"))})",
            1 => fullModule.InputParams[0].TypeDeclaration,
            _ => string.Empty
        };

        string OutputTypeString() => fullModule.ReturnParams.Count switch
        {
            > 1 => $"({string.Join(", ", fullModule.ReturnParams.Select(p => $"{p.TypeDeclaration} {p.Name}"))})",
            1 => fullModule.ReturnParams[0].TypeDeclaration,
            _ => "Tensor<int64>"
        };

        string HyperparamReference() => fullModule.Hyperparams.Count switch
        {
            > 1 => string.Join(", ", fullModule.Hyperparams.Select(p => $"hyperparams.{p.Name}")),
            1 => "hyperparams",
            _ => string.Empty
        };

        string HyperparamNamesCommaSeparated() => fullModule.Hyperparams.Count switch
        {
            > 1 => string.Join(", ", fullModule.Hyperparams.Select(p => p.Name)),
            1 => fullModule.Hyperparams[0].Name,
            _ => string.Empty
        };

        string InputReference() => fullModule.InputParams.Count switch
        {
            > 1 => string.Join(", ", fullModule.InputParams.Select(p => $"inputs.{p.Name}")),
            1 => "inputs",
            _ => string.Empty
        };

        var architectureType = hasInputs ? "Module" : "CallbackModule";
        var hyperComma = hasHyperparams ? ", " : string.Empty;
        var inputComma = hasInputs ? ", " : string.Empty;
        var hyperInputComma = hasHyperparams && hasInputs ? ", " : string.Empty;
        var hyperparamsVarName = hasHyperparams ? "hyperparams" : string.Empty;
        var inputsVarName = hasInputs ? "inputs" : string.Empty;

        var className = classInfo.ClassName;
        var moduleClassName = $"{className}Module";
        var modelClassName = $"{className}Model";
        
        // Generic type parameters from Inline method
        var typeParams = classInfo.TypeParameterList;
        var constraints = classInfo.TypeConstraintClauses;
        
        // Determine which type parameters are used in hyperparams vs inputs vs outputs
        string modelTypeParams = typeParams;
        string modelConstraints = constraints;
        string callTypeParams = typeParams;
        string callConstraints = constraints;
        
        if (classInfo.InlineMethod != null && classInfo.InlineMethod.IsGenericMethod)
        {
            // Find type parameters used in hyperparameters
            var hyperTypeParams = FindUsedTypeParameters(fullModule.Hyperparams, classInfo.InlineMethod);
            
            // Find type parameters used in inputs
            var inputTypeParams = FindUsedTypeParameters(fullModule.InputParams, classInfo.InlineMethod);
            
            // Find type parameters used in outputs
            var outputTypeParams = FindUsedTypeParameters(fullModule.ReturnParams, classInfo.InlineMethod);
            
            // Model class needs type parameters from inputs AND outputs (not hyperparams)
            // This is because Model<TInput, TOutput> is the base class
            var modelTypeParamsSet = new HashSet<string>(inputTypeParams);
            modelTypeParamsSet.UnionWith(outputTypeParams);
            
            if (modelTypeParamsSet.Count > 0)
            {
                var (modelTp, modelConstr) = BuildTypeParamsSubset(classInfo.InlineMethod, modelTypeParamsSet);
                modelTypeParams = modelTp;
                modelConstraints = modelConstr;
            }
            else
            {
                // Edge case: no type params in inputs or outputs
                modelTypeParams = string.Empty;
                modelConstraints = string.Empty;
            }
            
            // Model() static method and Call() static method need ALL type parameters
            // from the Inline method, not just the ones used in the signature.
            // This is because:
            // 1. They need to instantiate the Module class which has all type parameters
            // 2. They need to call the Inline method which has all type parameters
            // 3. Type parameters might be used internally even if not in the signature
            // So callTypeParams should always equal typeParams
            callTypeParams = typeParams;
            callConstraints = constraints;
        }
        
        // Module and Model classes remain generic
        var moduleClassNameWithGenerics = moduleClassName + typeParams;
        var modelClassNameWithGenerics = modelClassName + modelTypeParams;
        
        // For calling Inline method, we need to pass type parameters
        var classNameWithGenerics = className + typeParams;

        if (classInfo.Namespace is not null)
        {
            sb.AppendLine($"namespace {classInfo.Namespace}")
              .AppendLine("{");
        }

        sb.AppendLine("    using Shorokoo;")
          .AppendLine("    using Shorokoo.Core;\n    using Shorokoo.Graph;\n    using Shorokoo.Core.Nodes;\n    using Shorokoo.Core.Nodes.NodeDefinitions;\n    using Shorokoo.Core.Nodes.OnnxNodes;\n    using Shorokoo.Core.Utils;\n    using Shorokoo.Modules;\n    using Shorokoo.Onnx;")
          .AppendLine();

        // Generate the Module class (still generic)
        var lambdaParams = $"{hyperparamsVarName}{hyperInputComma}{inputsVarName}";
        var inlineMethodRef = $"{className}.Inline{typeParams}";
        // Inline's parameters are inputs-first, hyperparameters-last, so the call into it
        // passes the input references before the hyperparameter references.
        var inlineCall = $"{inlineMethodRef}({InputReference()}{hyperInputComma}{HyperparamReference()})";

        // Caller-facing argument conversions: hyperparameters bound via SetHyperparams and inputs
        // passed to the model's Call, each wrapping/defaulting omittable parameters.
        string HyperSetArg() => fullModule.Hyperparams.Count switch
        {
            > 1 => "(" + string.Join(", ", fullModule.Hyperparams.Select(CallerToModuleArg)) + ")",
            1 => CallerToModuleArg(fullModule.Hyperparams[0]),
            _ => string.Empty
        };
        string InputCallArg() => fullModule.InputParams.Count switch
        {
            > 1 => string.Join(", ", fullModule.InputParams.Select(CallerToModuleArg)),
            1 => CallerToModuleArg(fullModule.InputParams[0]),
            _ => string.Empty
        };
        var hasOptionalTensorInput = fullModule.InputParams.Any(ParamIsOptionalTensor);
        var combinedCallerSig = CallerSignatureList([.. fullModule.Hyperparams, .. fullModule.InputParams]);

        sb.AppendLine($"    public class {moduleClassNameWithGenerics} : {architectureType}<{HyperparamTypeString()}{hyperComma}{InputTypeString()}{inputComma}{OutputTypeString()}>{constraints}")
          .AppendLine("    {")
          .AppendLine($"        public {moduleClassName}()")
          .AppendLine($"            : base(({lambdaParams}) => {inlineCall}, {inlineMethodRef})")
          .AppendLine("        {")
          .AppendLine("        }")
          .AppendLine("    }")
          .AppendLine();

        // Generate the Model class (still generic, but only with hyperparameter type params)
        sb.AppendLine($"    public class {modelClassNameWithGenerics} : Model<{InputTypeList()}{inputComma}{OutputTypeString()}>{modelConstraints}")
          .AppendLine("    {")
          .AppendLine($"        public {modelClassName}(Scalar<IModelVarType> modelVariable) : base(modelVariable)")
          .AppendLine("        {")
          .AppendLine("        }");
        // When the model has OptionalTensor inputs, expose a Tensor?-accepting Call overload so a
        // held model is called the same way as the static Foo.Call shortcut (omit / pass null for
        // the absent case). The base Model<...>.Call(OptionalTensor) remains available.
        if (hasInputs && hasOptionalTensorInput)
        {
            sb.AppendLine($"        public {OutputTypeString()} Call({CallerSignatureList(fullModule.InputParams)})")
              .AppendLine($"            => base.Call({InputCallArg()});");
        }
        sb.AppendLine("    }")
          .AppendLine();

        // Generate the partial class extension (non-generic class with generic methods)
        sb.AppendLine($"    [global::System.Runtime.CompilerServices.CompilerGenerated]")
          .AppendLine($"    public partial class {className}")
          .AppendLine("    {");
        
        // Model method. [Hyper(default)] parameters are exposed as nullable+omittable and an
        // OptionalTensor hyperparameter as Tensor?; SetHyperparams receives the substituted/wrapped
        // values (see CallerToModuleArg).
        if (hasHyperparams)
        {
            sb.AppendLine($"        public static {modelClassNameWithGenerics} Model{callTypeParams}({CallerSignatureList(fullModule.Hyperparams)}){callConstraints}")
              .AppendLine($"        {{")
              .AppendLine($"            var module = new {moduleClassName}{typeParams}();")
              .AppendLine($"            return module.SetHyperparams<{modelClassNameWithGenerics}>({HyperSetArg()});")
              .AppendLine($"        }}")
              .AppendLine();
        }
        else
        {
            sb.AppendLine($"        public static {modelClassNameWithGenerics} Model{callTypeParams}(){callConstraints}")
              .AppendLine($"        {{")
              .AppendLine($"            var module = new {moduleClassName}{typeParams}();")
              .AppendLine($"            return module.SetHyperparams<{modelClassNameWithGenerics}>();")
              .AppendLine($"        }}")
              .AppendLine();
        }

        // Combined static Call shortcut. Argument order is unchanged (hyperparameters first, then
        // inputs); omittable parameters in the trailing run get `= null` defaults. Hyperparameter
        // names pass straight through to Model (which substitutes defaults); OptionalTensor inputs
        // are wrapped before the model's Call.
        var callArgs = hasHyperparams
            ? HyperparamNamesCommaSeparated()
            : string.Empty;

        if (hasInputs)
        {
            sb.AppendLine($"        public static {OutputTypeString()} Call{callTypeParams}({combinedCallerSig}){callConstraints}")
              .AppendLine($"            => Model{callTypeParams}({callArgs}).Call({InputCallArg()});");
        }
        else
        {
            sb.AppendLine($"        public static {OutputTypeString()} Call{callTypeParams}({combinedCallerSig}){callConstraints}")
              .AppendLine($"            => Model{callTypeParams}({callArgs}).Call();");
        }

        sb.AppendLine();
        
        // ComputationGraph property - constructs FastComputationGraph by passing method info
        // This is non-generic and constructs the graph directly without using Module.
        // The property is still named ComputationGraph for backwards source-compat, but its
        // type is FastComputationGraph so consumers can use the Fast processor pipeline
        // directly. Consumers that still want the legacy ComputationGraph form should call
        // FastComputationGraphConverter.ToComputationGraph(...) on the result.
        //
        // Returns a deep clone of a cached template each call. The Fast processors mutate
        // graphs in place (FastChangeGenericTypeSpecialization.Process rewrites node.Attributes,
        // etc.), and node identity is preserved through ToComputationGraph/ToFastGraph round-trips,
        // so handing out the cached instance would let one caller's mutations leak into the next.
        sb.AppendLine($"        private static Shorokoo.Graph.FastComputationGraph? _computationGraphTemplate;")
          .AppendLine($"        public static Shorokoo.Graph.FastComputationGraph ComputationGraph")
          .AppendLine("        {")
          .AppendLine("            get")
          .AppendLine("            {")
          .AppendLine("                if (_computationGraphTemplate == null)")
          .AppendLine("                {")
          .AppendLine($"                    var methodInfo = typeof({className}).GetMethod(nameof(Inline));")
          .AppendLine("                    if (methodInfo == null)")
          .AppendLine($"                        throw new System.InvalidOperationException(\"Could not find Inline method on {className}\");")
          .AppendLine("                    _computationGraphTemplate = Shorokoo.Core.GraphBuilder.BuildFastComputationGraphFromMethodInfo(methodInfo);")
          .AppendLine("                }")
          .AppendLine("                return _computationGraphTemplate.Clone();")
          .AppendLine("            }")
          .AppendLine("        }");

        sb.AppendLine("    }");

        // Strongly-typed, named hyperparameter set for optimizer-shaped modules (all hyperparameters
        // are scalar float32). Gives named, defaulted, init-only HyperValue properties so a rig can be
        // built as `new XxxHyperparameters { LearningRate = Schedules.Cosine(...), WeightDecay = 1e-4f }`.
        if (AllHyperparamsAreFloatScalars(fullModule))
        {
            EmitHyperparameterSet(sb, className, fullModule);
        }

        if (classInfo.Namespace is not null)
        {
            sb.AppendLine("}");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Emits <c>{className}Hyperparameters</c>: a sealed <see cref="global::Shorokoo.IOptimizerHyperparameters"/>
    /// with one named, init-only <c>HyperValue</c> property per hyperparameter (defaulted from
    /// <c>[Hyper(default)]</c>, else <c>required</c>), plus the optimizer-order accessor and names.
    /// </summary>
    private static void EmitHyperparameterSet(StringBuilder sb, string className, FullModuleInfo fullModule)
    {
        var recordName = $"{className}Hyperparameters";

        sb.AppendLine()
          .AppendLine($"    /// <summary>Strongly-typed, named hyperparameters for <see cref=\"{className}\"/>.")
          .AppendLine("    /// A bare <c>float</c> is baked as a constant; a <c>Schedule</c> is applied per step.</summary>")
          .AppendLine($"    public sealed class {recordName} : global::Shorokoo.IOptimizerHyperparameters")
          .AppendLine("    {");

        foreach (var h in fullModule.Hyperparams)
        {
            var prop = Pascalize(h.Name);
            if (h.DefaultLiteral is not null)
                sb.AppendLine($"        public global::Shorokoo.HyperValue {prop} {{ get; init; }} = {h.DefaultLiteral};");
            else
                sb.AppendLine($"        public required global::Shorokoo.HyperValue {prop} {{ get; init; }}");
        }

        var orderRefs = string.Join(", ", fullModule.Hyperparams.Select(h => Pascalize(h.Name)));
        var nameLits = string.Join(", ", fullModule.Hyperparams.Select(h => $"\"{h.Name}\""));

        sb.AppendLine()
          .AppendLine($"        public global::Shorokoo.HyperValue[] InOptimizerOrder() => new global::Shorokoo.HyperValue[] {{ {orderRefs} }};")
          .AppendLine($"        private static readonly string[] _names = new string[] {{ {nameLits} }};")
          .AppendLine("        public System.Collections.Generic.IReadOnlyList<string> HyperparameterNames => _names;")
          .AppendLine("    }");
    }

    private static string GenerateStaticClassCode(ModuleClassInfo classInfo)
    {
        var sb = new StringBuilder();
        
        if (classInfo.Namespace is not null)
        {
            sb.AppendLine($"namespace {classInfo.Namespace}")
              .AppendLine("{");
        }

        sb.AppendLine("    using Shorokoo;")
          .AppendLine("    using Shorokoo.Core;\n    using Shorokoo.Graph;\n    using Shorokoo.Core.Nodes;\n    using Shorokoo.Core.Nodes.NodeDefinitions;\n    using Shorokoo.Core.Nodes.OnnxNodes;\n    using Shorokoo.Core.Utils;\n    using Shorokoo.Modules;\n    using Shorokoo.Onnx;")
          .AppendLine($"    [global::System.Runtime.CompilerServices.CompilerGenerated]")
          .AppendLine($"    public static partial class {classInfo.ClassName}")
          .AppendLine("    {");

        var isFirst = true;
        foreach (var fullModule in classInfo.FullModules)
        {
            if (fullModule.Kind == ModuleKind.Ignore) continue;

            if (!isFirst) sb.AppendLine();
            isFirst = false;

            if (fullModule.Kind == ModuleKind.TrainableParamInitializer)
            {
                var inputNamedTypeList = string.Join(", ", fullModule.InputParams.Select(p => $"{p.TypeDeclaration} {p.Name}"));
                var inputsReferenceString = string.Join(", ", fullModule.InputParams.Select(p => p.Name));
                var outputTypeName = fullModule.ReturnParams[0].TypeDeclaration;

                // For new-style class-level initializers, generate "Init" method
                // For old-style method-level initializers, generate PascalCase method name
                var methodName = classInfo.IsNewStyleInitializer ? "Init" : fullModule.ModuleName;

                // Handle generic type parameters
                var typeParamList = classInfo.TypeParameterList;
                var typeConstraints = classInfo.TypeConstraintClauses;

                // Check if this is a generic initializer
                var hasGenericParams = !string.IsNullOrEmpty(typeParamList);
                
                // isTrainable is false for [StateInitializer], true for [TrainableParamInitializer] and legacy initializers.
                // State initializers also pass their declared ownership (from
                // [StateInitializer(Ownership = ...)]) so the TrainingRig can tell module-owned
                // state (running stats) apart from optimizer-owned state (per-param moments).
                var isTrainableValue = classInfo.IsStateInitializer ? "false" : "true";

                // Build the CallTrainableParamInitializer argument list incrementally so a
                // no-parameter initializer (e.g. a scalar Inline() => Scalar(0f)) emits no
                // trailing comma — the variadic `inputs` slot just gets nothing.
                var callArgs = new List<string>
                {
                    $"{fullModule.FnName}{typeParamList}",
                    $"defaultName: \"{classInfo.ClassName}\"",
                    $"isTrainable: {isTrainableValue}",
                };
                if (classInfo.IsStateInitializer)
                    callArgs.Add($"stateOwnership: global::Shorokoo.Modules.StateOwnership.{classInfo.StateOwnershipName}");
                if (!string.IsNullOrEmpty(inputsReferenceString))
                    callArgs.Add(inputsReferenceString);
                var callArgList = string.Join(", ", callArgs);

                if (hasGenericParams && classInfo.IsNewStyleInitializer)
                {
                    // For generic new-style initializers, extract the first type parameter to use with
                    // Globals.CallTrainableParamInitializer<T> which returns Tensor<T> properly.
                    // typeParamList is produced by ExtractGenericTypeInfoFromMethod and is always
                    // well-formed like "<T>" or "<T1, T2>", so this simple parsing is safe.
                    var firstTypeParam = typeParamList.TrimStart('<').TrimEnd('>').Split(',')[0].Trim();

                    sb.AppendLine($"        public static {outputTypeName} {methodName}{typeParamList}({inputNamedTypeList}){typeConstraints}")
                      .AppendLine($"            => Globals.CallTrainableParamInitializer<{firstTypeParam}>({callArgList});");
                }
                else
                {
                    // For non-generic initializers, wrap the Variable result into the value-struct
                    // return type. A plain cast would unbox the interface to a struct and throw;
                    // Variable.ToValue invokes the node→struct conversion instead.
                    sb.AppendLine($"        public static {outputTypeName} {methodName}{typeParamList}({inputNamedTypeList}){typeConstraints}")
                      .AppendLine($"            => Globals.CallTrainableParamInitializer({callArgList}).ToValue<{outputTypeName}>();");
                }
            }
        }

        sb.AppendLine("    }");
        if (classInfo.Namespace is not null)
        {
            sb.AppendLine("}");
        }

        return sb.ToString();
    }
}

public class ModuleClassInfo
{
    public string ClassName { get; }
    public string? Namespace { get; }
    public List<FullModuleInfo> FullModules { get; }
    public Location? Location { get; }
    public bool IsStaticClass { get; }
    public string TypeParameterList { get; }
    public string TypeConstraintClauses { get; }
    public IMethodSymbol? InlineMethod { get; }  // For non-static classes, store the Inline method
    public bool IsNewStyleInitializer { get; }  // True for class-level [TrainableParamInitializer] or [StateInitializer]
    public bool IsStateInitializer { get; }  // True for [StateInitializer], false for [TrainableParamInitializer]
    public string StateOwnershipName { get; }  // StateOwnership member name from [StateInitializer(Ownership = ...)]

    public ModuleClassInfo(string className, string fullyQualifiedNamespace, List<FullModuleInfo> fullModules, Location? location, bool isStaticClass = false, string typeParameterList = "", string typeConstraintClauses = "", IMethodSymbol? inlineMethod = null, bool isNewStyleInitializer = false, bool isStateInitializer = false, string stateOwnershipName = "ModuleOwned")
    {
        ClassName = className;
        Namespace = string.IsNullOrEmpty(fullyQualifiedNamespace) || fullyQualifiedNamespace == "<global namespace>"
            ? null
            : fullyQualifiedNamespace;
        FullModules = fullModules;
        Location = location;
        IsStaticClass = isStaticClass;
        TypeParameterList = typeParameterList;
        TypeConstraintClauses = typeConstraintClauses;
        InlineMethod = inlineMethod;
        IsNewStyleInitializer = isNewStyleInitializer;
        IsStateInitializer = isStateInitializer;
        StateOwnershipName = stateOwnershipName;
    }
}

public enum ParamKind
{
    Hyperparam,
    InputParam,
    OutputParam
}

public enum ModuleKind
{
    TrainableParamInitializer,
    CallbackModuleImplementation,
    ModuleImplementation,
    Ignore
}

public class ParamData
{
    public string Name { get; }
    public string TypeDeclaration { get; }
    public ParamKind Kind { get; }

    /// <summary>
    /// For a hyperparameter declared with <c>[Hyper(default)]</c>, the C# float literal of that
    /// default (e.g. <c>"0.9f"</c>); <c>null</c> when no default was supplied.
    /// </summary>
    public string? DefaultLiteral { get; }

    public ParamData(string name, string typeDeclaration, ParamKind kind, string? defaultLiteral = null)
    {
        Name = name;
        TypeDeclaration = typeDeclaration;
        Kind = kind;
        DefaultLiteral = defaultLiteral;
    }
}

public class FullModuleInfo
{
    public string ModuleName { get; }
    public string FnName { get; }
    public ModuleKind Kind { get; }
    public List<ParamData> ReturnParams { get; }
    public List<ParamData> Hyperparams { get; }
    public List<ParamData> InputParams { get; }
    public Location? Location { get; }

    public FullModuleInfo(string moduleName, string fnName, ModuleKind kind, List<ParamData> returnParams, List<ParamData> hyperparams, List<ParamData> inputParams, Location? location)
    {
        ModuleName = moduleName;
        FnName = fnName;
        Kind = kind;
        ReturnParams = returnParams;
        Hyperparams = hyperparams;
        InputParams = inputParams;
        Location = location;
    }
}