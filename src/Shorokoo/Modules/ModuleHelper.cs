using Shorokoo;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using static Shorokoo.Globals;
using E = System.Linq.Expressions.Expression;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core
{
    internal static class ModuleHelper
    {
        private static Dictionary<MethodInfo, Function> targetFunctionCache = new();
        private static Dictionary<string, Function> signatureFunctionCache = new();

        private static Function? LoadCachedFunction(MethodInfo method)
        {
            lock (targetFunctionCache)
            {
                if (targetFunctionCache.ContainsKey(method))
                {
                    return targetFunctionCache[method];
                }

                return null;
            }
        }

        private static Function StoreCachedFunction(MethodInfo method, Function fn)
        {
            lock (targetFunctionCache)
            {
                if (targetFunctionCache.ContainsKey(method))
                {
                    return targetFunctionCache[method];
                }

                targetFunctionCache[method] = fn;
                return fn;
            }
        }

        private static Function? LoadCachedSignature(string signature)
        {
            lock (targetFunctionCache)
            {
                if (signatureFunctionCache.ContainsKey(signature))
                {
                    return signatureFunctionCache[signature];
                }

                return null;
            }
        }

        private static Function StoreCachedSignature(string signature, FastComputationGraph graph)
        {
            lock (targetFunctionCache)
            {
                if (signatureFunctionCache.ContainsKey(signature))
                    return signatureFunctionCache[signature];

                var name = $"Signature{signatureFunctionCache.Count + 1}";

                var retval = new Function(graph, FunctionType.ModuleSignature, name, name);
                signatureFunctionCache[signature] = retval;

                return retval;
            }
        }

        private static IEnumerable<Type> FlattenTuples(this IEnumerable<Type> typesWithTuples)
        {
            foreach (var type in typesWithTuples)
            {
                if (IsValueTuple(type))
                {
                    foreach (var subType in FlattenTuples(type.GenericTypeArguments))
                        yield return subType;
                }
                else
                {
                    yield return type;
                }
            }
        }

        private static InputType GetInputType(this ParameterInfo paramInfo)
        {
            if (paramInfo.GetCustomAttribute<HyperAttribute>() is not null)
                return InputType.Hyperparam;

            return InputType.ReadyInput;
        }

        /// <summary>The <c>[Hyper(defaultValue)]</c> default declared on this parameter, or null when it declared none.</summary>
        private static float? GetHyperDefaultValue(this ParameterInfo paramInfo)
            => paramInfo.GetCustomAttribute<HyperAttribute>() is { HasDefault: true } hyper ? hyper.DefaultValue : null;

        // The signature string is computed from the internal graph node; callers cross the module
        // boundary explicitly with IModuleParam.ToVariable() before reaching here.
        internal static string ToSignatureString(Variable variable)
            => ToSignatureStringWithOverride(variable, variable.Rank);

        internal static string ToSignatureStringWithOverride(Variable variable, int? rank)
        {
            // Model/module params are scalar nodes distinguished by their runtime DType (formerly the
            // generic ImmutableScalar<IModelVarType> / ImmutableScalar<IModuleVarType>).
            if (variable.Type == DType.Model)
                return $"[{variable.ModuleFn!.ModelSignatureString}]";

            if (variable.Type == DType.Module)
                return $"[{variable.ModuleFn!.ModuleSignatureString}]";

            var rankString = rank is null || rank == -1 ? "" : $"#{rank}";

            // Dispatch on the structural kind (a single Variable node satisfies every marker
            // interface, so kind must come from Structure()/Kind rather than ordered `is` checks).
            switch (variable.Structure())
            {
                case DataStructure.TensorStruct:
                    // "struct:{typeName}" where typeName is the IStruct interface name.
                    return $"struct:{variable.Definition?.TypeName ?? "anonymous"}";
                case DataStructure.Tensor:
                    return $"{variable.Type.ToString().ToLower()}{rankString}";
                case DataStructure.Optional:
                    return $"{variable.Type.ToString().ToLower()}?{rankString}";
                case DataStructure.Sequence:
                    return $"{variable.Type.ToString().ToLower()}/seq{rankString}";
            }

            throw new InvalidTensorOperationException(ErrorCodes.FW002, "Module Parameter Type Processing", $"parameter type {variable.GetType().Name}", "Unsupported module parameter structure");
        }

        internal static (string moduleSignature, string modelSignature) CreateFunctionSignatureString(Type[] hyperparams, Type[] inputs, Type[] outputs)
        {
            var hyperparamInputs = FlattenTuples(hyperparams).Select((x, i) => ModuleParamInputBasedOn(x, InputType.Hyperparam, $"h{i}").ToVariable()).ToArray();
            var inputInputs = FlattenTuples(inputs).Select((x, i) => ModuleParamInputBasedOn(x, InputType.ReadyInput, $"h{i}").ToVariable()).ToArray();
            var outputInputs = FlattenTuples(outputs).Select((x, i) => ModuleParamInputBasedOn(x, InputType.ReadyInput, $"h{i}").ToVariable()).ToArray();

            return CreateFunctionSignatureString(hyperparamInputs, inputInputs, outputInputs, null);
        }

        internal static (string moduleSignature, string modelSignature) CreateFunctionSignatureString(Variable[] hyperparams, Variable[] inputs, Variable[] outputs, int?[]? outputOverrideRanks)
        {
            var signatureHyperparamPart = string.Join(", ", hyperparams.Select(ToSignatureString));
            var signatureInputPart = string.Join(", ", inputs.Select(ToSignatureString));
            var signatureOutputPart = outputOverrideRanks is null ?
                    string.Join(", ", outputs.Select(ToSignatureString)) :
                    string.Join(",", outputs.Zip(outputOverrideRanks).Select(x => ToSignatureStringWithOverride(x.First, x.Second)));

            return ($"{signatureHyperparamPart} | {signatureInputPart} > {signatureOutputPart}",
                $"{signatureInputPart} > {signatureOutputPart}");
        }

        /// <summary>
        /// Reject the internal graph node type <see cref="Variable"/> in a module signature: modules
        /// must declare their inputs and outputs with the user-facing value handles
        /// (<see cref="Tensor{T}"/>, <see cref="Vector{T}"/>, <see cref="Scalar{T}"/>,
        /// <see cref="OptionalTensor{T}"/>, <see cref="TensorSequence{T}"/>, <see cref="TensorStruct{T}"/>).
        /// <see cref="Variable"/> is the graph's internal representation only — the framework converts
        /// to and from it behind the module boundary, never across it.
        /// </summary>
        internal static void RejectVariableParam(Type type)
        {
            if (typeof(Variable).IsAssignableFrom(type))
                throw new InvalidTensorOperationException(ErrorCodes.FW002, "module signature", type.Name,
                    "the internal graph node type 'Variable' cannot be a module input or output; use a value handle " +
                    "(Tensor<T>, Vector<T>, Scalar<T>, OptionalTensor<T>, TensorSequence<T>, TensorStruct<T>) instead");

            // Walk into composite shapes so a Variable nested in a tuple ((Variable, Tensor<T>)) or an
            // array (Variable[]) is rejected too — modules speak in value handles all the way down.
            if (IsValueTuple(type))
                foreach (var arg in type.GenericTypeArguments) RejectVariableParam(arg);
            else if (type.IsArray)
                RejectVariableParam(type.GetElementType().AssertNotNull());
        }


        internal static Function CreateFunctionSignature(Type[] hyperparams, Type[] inputs, Type[] outputs)
        {
            var signature = CreateFunctionSignatureString(hyperparams, inputs, outputs).moduleSignature;
            var cachedFunction = LoadCachedSignature(signature);
            if (cachedFunction is not null)
                return cachedFunction;

            var hyperparamInputs = FlattenTuples(hyperparams).Select((x, i) => ModuleParamInputBasedOn(x, InputType.Hyperparam, $"h{i}").ToVariable()).ToArray();
            var inputInputs = FlattenTuples(inputs).Select((x, i) => ModuleParamInputBasedOn(x, InputType.ReadyInput, $"h{i}").ToVariable()).ToArray();

            var outputTypes = FlattenTuples(outputs).ToList();
            var outputVariables = outputTypes.Select((x) => InternalGlobals.DefaultVariable(x)).ToArray();
            var rankOverrides = outputTypes.Select((x) =>
                                x.IsAssignableTo(typeof(IVector)) ? 1 :
                                x.IsAssignableTo(typeof(IScalar)) ? 0 :
                                (int?)null).ToArray();


            var graph = new FastComputationGraph([.. hyperparamInputs, .. inputInputs], [..outputVariables], [..rankOverrides]);

            return StoreCachedSignature(signature, graph);
        }

        internal static Function CreateTargetFunction(Delegate? method, bool isTrainableParamInitializer = false, bool isStateParamInitializer = false, MethodInfo? referenceMethod = null, string? defaultName = null, object? invokeTarget = null, StateOwnership stateOwnership = StateOwnership.ModuleOwned)
        {
            // Validate mutual exclusivity
            if (isTrainableParamInitializer && isStateParamInitializer)
                throw new InvalidOperationException("Cannot specify both isTrainableParamInitializer and isStateParamInitializer as true. A function can only be one type.");

            // When no explicit reference method is supplied, the delegate's own method is both the
            // signature source and the body invoked at graph-build time. Compiler-generated lambda
            // methods are instance methods on a display-class singleton, so carry the delegate's
            // bound target for the reflection invoke (a static method group has a null Target).
            if (referenceMethod is null)
                invokeTarget ??= method?.Target;
            EnsureNonCapturingDelegate(invokeTarget, referenceMethod?.Name ?? method?.Method.Name ?? "<unknown>");

            referenceMethod ??= method?.Method;
            referenceMethod = referenceMethod.AssertNotNull();

            if (referenceMethod.DeclaringType is null)
                throw new ReflectionException(ErrorCodes.FW002, $"method {referenceMethod.Name}", "declaring type", "Reference method declaring type is null");

            if (referenceMethod.DeclaringType.AssertNotNull().AssemblyQualifiedName is null)
                throw new ReflectionException(ErrorCodes.FW002, $"method {referenceMethod.Name}", $"type {referenceMethod.DeclaringType.Name}", "Assembly qualified name is null for reference method declaring type");

            // For generic methods, use the generic method definition as the cache key
            // This ensures that GenericModule.Call<T1>() and GenericModule.Call<T2>() reference the same Function
            var cacheKey = referenceMethod.IsGenericMethod && !referenceMethod.IsGenericMethodDefinition
                ? referenceMethod.GetGenericMethodDefinition()
                : referenceMethod;
                
            var cachedFunction = LoadCachedFunction(cacheKey);
            if (cachedFunction is not null)
                return cachedFunction;

            if (method is not null)
            {
                var actualMethodInfo = method.Method;

                var actualInputs = actualMethodInfo.GetParameters().Select(x => x.ParameterType).FlattenTuples().ToArray();
                var referenceInputs = referenceMethod.GetParameters().Select(x => x.ParameterType).ToArray();
                // Compare as multisets, not sequences: the wrapper delegate groups its parameters
                // as (THypers, TInputs) while the reference Inline method lists them inputs-first /
                // hyperparameters-last, so the two agree on the set of flattened parameter types
                // but not their order.
                Debug.Assert(
                    actualInputs.OrderBy(t => t.AssemblyQualifiedName, StringComparer.Ordinal)
                        .SequenceEqual(referenceInputs.OrderBy(t => t.AssemblyQualifiedName, StringComparer.Ordinal)));
            }

            // For generic methods, build the graph from the generic method definition
            // This ensures GENERIC_TYPE_INPUT nodes are created
            var methodToBuild = referenceMethod.IsGenericMethod && !referenceMethod.IsGenericMethodDefinition
                ? referenceMethod.GetGenericMethodDefinition()
                : referenceMethod;

            // Use the factored GraphBuilder code to build the function body in its
            // primary FastCG form. The Function ctor stores it directly; the legacy
            // CG view is materialized lazily on demand.
            var fastGraph = GraphBuilder.BuildFastComputationGraphFromMethod(methodToBuild, invokeTarget);

            var fnType = FunctionType.Module;
            if (isStateParamInitializer)
            {
                // Debug.Assert(graph.VirtualGraph.IsWeightless);
                fnType = FunctionType.StateParamInitializer;
            }
            else if (isTrainableParamInitializer)
            {
                // Debug.Assert(graph.VirtualGraph.IsWeightless);
                fnType = FunctionType.TrainableParamInitializer;
            }

            var name = defaultName ?? FriendlyDeclaringTypeName(referenceMethod) ?? referenceMethod.Name;
            var fn = new Function(fastGraph, fnType, name, name,
                isStateParamInitializer ? stateOwnership : (StateOwnership?)null);

            return StoreCachedFunction(cacheKey, fn);
        }

        /// <summary>
        /// Returns the name of the nearest non-compiler-generated declaring type of
        /// <paramref name="method"/>. For ordinary static methods this is the declaring class
        /// name (the codegen convention for module names); for lambda methods it skips the
        /// compiler display classes (<c>&lt;&gt;c</c>, ...) so the function isn't named after them.
        /// </summary>
        private static string? FriendlyDeclaringTypeName(MethodInfo method)
        {
            var type = method.DeclaringType;
            while (type is not null && type.Name.StartsWith("<"))
                type = type.DeclaringType;
            return type?.Name;
        }

        /// <summary>
        /// Validates that a delegate used as a module/initializer body carries no captured state.
        /// Module bodies are invoked once by reflection at graph-build time and the resulting
        /// <see cref="Function"/> is cached by the body's <see cref="MethodInfo"/> — captured
        /// locals would be invisible to the graph and silently shared across closures that
        /// compile to the same method. A static method group has a null target; a non-capturing
        /// (<c>static</c>) lambda binds the compiler's field-less display-class singleton —
        /// both pass. A capturing lambda or an instance-bound delegate is rejected.
        /// </summary>
        internal static void EnsureNonCapturingDelegate(object? target, string methodName)
        {
            if (target is null)
                return;

            var hasInstanceState = target.GetType()
                .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                .Length > 0;

            if (hasInstanceState)
                throw new InvalidOperationException(
                    $"Module delegate '{methodName}' captures local state (or is bound to an object instance). " +
                    "Codegen-free module bodies must be a static method group or a non-capturing (static) lambda: " +
                    "the body is invoked once by reflection at graph-build time and cached by its MethodInfo, so " +
                    "captured locals would be baked invisibly into the graph and shared across closures. " +
                    "Pass varying values as [Hyper] parameters or runtime inputs instead.");
        }

        internal static Scalar<IModuleVarType> CreateModule(Function targetFunction, Type? moduleType = null)
        {
            DType[]? genericTypeArgs = null;
            if (moduleType != null)
            {
                genericTypeArgs = ExtractGenericTypeArgsFromType(moduleType);
            }
            return InternalOp.CreateModule(targetFunction, genericTypeArgs);
        }

        /// <summary>Creates the module input placeholder corresponding to a module method parameter type (model/module, tensor struct, sequence, optional, scalar, vector, or tensor).</summary>
        /// <param name="type">The module method parameter's declared type, which selects the kind of input placeholder created.</param>
        /// <param name="inputType">Whether the placeholder is a hyperparameter or a regular input.</param>
        /// <param name="paramName">The parameter's name, used as the input node's default name (may be <see langword="null"/>).</param>
        /// <param name="hyperDefaultValue">The <c>[Hyper(defaultValue)]</c> default, when the parameter declared one;
        /// recorded as declarative metadata on a scalar hyperparameter input so it survives serialization.</param>
        public static IModuleParam ModuleParamInputBasedOn(Type type, InputType inputType, string? paramName, float? hyperDefaultValue = null)
        {
            RejectVariableParam(type);

            if (type.IsAssignableTo(typeof(IModel)))
                return (IModel)type.GetConstructor([typeof(InputType)]).AssertNotNull().Invoke([inputType]);
            else if (type.IsAssignableTo(typeof(IModule)))
                return (IModule)type.GetConstructor([typeof(InputType)]).AssertNotNull().Invoke([inputType]);

            // Check for ITensorStruct BEFORE extracting DType (TensorStruct<T> has IStruct as type arg, not a numeric type)
            if (type.IsAssignableTo(typeof(ITensorStruct)))
            {
                var (_, structDType) = StructDefExtractor.ExtractFromTensorStructType(type, "module input creation");
                return InternalOp.TensorStructInput(structDType, inputType, targetFunction: null, defaultName: paramName).ToValue();
            }

            // Check for IStruct types (interfaces like RealGenericPairStruct<U, V> or records like GenericPairRecord<U, V>).
            // These define tensor struct field shapes — not TensorStruct<T> handles — and have no value handle
            // of their own. The created node rides inside a TensorStruct<T> carrier (handed across the module
            // boundary as an IModuleParam); InvokeAndFormat converts it back to the Variable and wraps that in a DispatchProxy
            // (for interfaces) or constructs via the record constructor before passing to method.Invoke.
            if (typeof(IStruct).IsAssignableFrom(type) && type != typeof(IStruct) && type != typeof(IVarType) && !typeof(IValue).IsAssignableFrom(type))
            {
                var structDef = StructDefExtractor.ExtractFromType(type);
                var structDType = DType.GetOrCreateForTensorStruct(structDef);
                return InternalOp.TensorStructInput(structDType, inputType, targetFunction: null, defaultName: paramName).ToValue();
            }

            var dtype = OnnxUtils.GetDType(type.GetGenericArguments()[0]);
            if (dtype is null)
                throw new InvalidTensorOperationException(ErrorCodes.GC007, "CreateComputeGraphInput", type.Name,
                    "Unable to determine DType from generic argument type");

            if (dtype == DType.Model || dtype == DType.Module)
                throw new InvalidTensorOperationException(ErrorCodes.GC007, "CreateComputeGraphInput", dtype.ToString(),
                    "Model and Module types are not supported as compute graph inputs");

            if (type.IsAssignableTo(typeof(ITensorSequence)))
                return InternalOp.ModuleSequenceInput(dtype, inputType, null, paramName).ToValue();
            else if (type.IsAssignableTo(typeof(IOptionalTensor)))
                return InternalOp.ModuleOptionalInput(dtype, inputType, null, paramName).ToValue();
            else if (type.IsAssignableTo(typeof(IScalar)))
                return InternalOp.ModuleTensorInput(dtype, rank: 0, inputType, null, paramName, hyperDefaultValue).ToValue();
            else if (type.IsAssignableTo(typeof(IVector)))
                return InternalOp.ModuleTensorInput(dtype, rank: 1, inputType, null, paramName).ToValue();
            else if (type.IsAssignableTo(typeof(ITensor)))
                return InternalOp.ModuleTensorInput(dtype, rank: null, inputType, null, paramName).ToValue();

            throw new UnsupportedDTypeException(ErrorCodes.GC006, type.Name, "CreateComputeGraphInput",
                $"Type '{type.FullName}' is not supported for compute graph input creation");
        }

        /// <summary>
        /// Creates input module parameters from method parameter information.
        /// Public helper for GraphBuilder to construct VirtualGraphs.
        /// </summary>
        public static IModuleParam[] CreateInputParams(ParameterInfo[] parameters)
            => parameters.Select(param => ModuleParamInputBasedOn(param.ParameterType, param.GetInputType(), param.Name, param.GetHyperDefaultValue())).ToArray();

        /// <summary>
        /// Invokes a method with the given inputs and formats the outputs.
        /// Public helper for GraphBuilder to construct VirtualGraphs.
        /// For parameters typed as IStruct interfaces (not <see cref="TensorStruct{T}"/>), wraps
        /// Variable inputs in DispatchProxy so property access creates graph operations.
        /// <paramref name="target"/> is the invocation receiver: null for static methods, or the
        /// delegate's bound target for compiler-generated (non-capturing) lambda methods.
        /// </summary>
        public static Variable[] InvokeAndFormat(MethodInfo method, IModuleParam[] inputs, object? target = null)
        {
            var parameters = method.GetParameters();
            var invokeArgs = new object?[inputs.Length];
            
            for (int i = 0; i < inputs.Length; i++)
            {
                var paramType = parameters[i].ParameterType;
                
                if (typeof(IStruct).IsAssignableFrom(paramType)
                    && paramType != typeof(IStruct)
                    && paramType != typeof(IVarType)
                    && !typeof(IValue).IsAssignableFrom(paramType)
                    && inputs[i] is IValue structCarrier && structCarrier.ToVariable() is Variable tensorStruct)
                {
                    if (paramType.IsInterface)
                    {
                        // IStruct interface — wrap in DispatchProxy so property access creates graph operations.
                        invokeArgs[i] = TensorStructProxyFactory.Create(paramType, tensorStruct, tensorStruct.Definition);
                    }
                    else
                    {
                        // IStruct record/class — construct via record constructor with TensorStructGetField values.
                        invokeArgs[i] = ConstructStructFromTensorStruct(paramType, tensorStruct);
                    }
                }
                else
                {
                    // inputs[i] already arrives as the value the body expects — an IValue handle, or an
                    // IModel/IModule — because ModuleParamInputBasedOn converts the graph node before this
                    // point. Nothing further to do; reflective Invoke boxes it.
                    invokeArgs[i] = inputs[i];
                }
            }

            // DoNotWrapExceptions: a module body that throws (e.g. StateUpdate rejecting a
            // non-state variable with instructions) must surface its own exception to the
            // author, not a TargetInvocationException wrapper.
            return Format(method.Invoke(
                obj: target,
                invokeAttr: BindingFlags.DoNotWrapExceptions,
                binder: null,
                parameters: invokeArgs,
                culture: null));
        }

        /// <summary>
        /// Constructs a record/class instance implementing IStruct from a Variable,
        /// using TensorStructGetField operations to extract field values for the constructor.
        /// </summary>
        private static object ConstructStructFromTensorStruct(Type structType, Variable tensorStruct)
        {
            var def = tensorStruct.Definition;
            var ctor = structType.GetConstructors()
                .OrderByDescending(c => c.GetParameters().Length)
                .FirstOrDefault()
                ?? throw new InvalidOperationException(
                    $"Cannot construct {structType.Name}: no public constructors found.");
            var ctorParams = ctor.GetParameters();
            var args = new object[ctorParams.Length];

            for (int i = 0; i < ctorParams.Length; i++)
            {
                var paramName = ctorParams[i].Name
                    ?? throw new InvalidOperationException(
                        $"Cannot construct {structType.Name}: constructor parameter at index {i} has no name.");
                // Positional record constructor params match property names.
                // Try exact match first, then PascalCase for camelCase params.
                var fieldDef = def.GetField(paramName)
                    ?? (paramName.Length > 0 ? def.GetField(char.ToUpper(paramName[0]) + paramName.Substring(1)) : null);

                if (fieldDef == null)
                    throw new InvalidOperationException(
                        $"Cannot construct {structType.Name}: constructor parameter '{paramName}' does not match any field in TensorStructDef. " +
                        $"Available fields: {string.Join(", ", def.Fields.Select(f => f.Name))}");

                // The record ctor parameters are value-struct IValues; convert the field's Variable to one
                // so reflective Invoke can bind it (it does not apply the implicit conversion). This must
                // be driven by the declared ctor-parameter type, not the value's natural handle: a field
                // declared Tensor<U> (generic standin) or a general Tensor<T> over a low-rank value would
                // not match the value's own rank-specific handle.
                args[i] = InternalOp.TensorStructGetField(
                        tensorStruct,
                        fieldDef.Name,
                        fieldDef.ElementType,
                        fieldDef.Rank,
                        fieldDef.Structure)
                    .ToValue(ctorParams[i].ParameterType);
            }

            return ctor.Invoke(args);
        }

        internal static bool IsValueTuple<T>() => IsValueTuple(typeof(T));

        internal static bool IsValueTuple(Type type)
        {
            if (!type.IsGenericType)
                return false;
            var genericDefinition = type.GetGenericTypeDefinition();
            return genericDefinition.Namespace == "System" && genericDefinition.Name.StartsWith("ValueTuple`");
        }

        internal static Variable[] Format(object? retval)
        {
            if (retval is null)
                throw new InvalidTensorOperationException(ErrorCodes.FW002, "Method Invocation Result", "method return value", "Method invocation returned null when non-null result expected");

            if (retval is Variable[] vars)
                return vars;

            if (retval is IModuleParam[] moduleParams)
                return [.. moduleParams.Select(x => x.ToVariable())];

            if (retval is IModuleParam moduleParam)
                return [moduleParam.ToVariable()];

            // Handle TensorStructProxy (DispatchProxy wrapping a Variable).
            // When a module passes an IStruct interface object (created by DispatchProxy) to
            // another module's Call method, extract the backing TensorStruct Variable.
            if (retval is ITensorStructProxy proxy)
                return [(Variable)proxy.BackingTensorStruct];

            // Handle IStruct records/classes — extract field values and create a TensorStruct graph node.
            // When a module passes a record implementing IStruct to another module's Call method,
            // we convert it to a TensorStruct by reading its property values (which are Variable graph nodes).
            if (retval is IStruct)
            {
                var structType = retval.GetType();
                var def = StructDefExtractor.ExtractFromType(structType);
                var dtype = DType.GetOrCreateForTensorStruct(def);

                var fieldValues = new Variable[def.Fields.Length];
                for (int i = 0; i < def.Fields.Length; i++)
                {
                    var prop = structType.GetProperty(def.Fields[i].Name);
                    if (prop == null)
                        throw new InvalidOperationException(
                            $"Property '{def.Fields[i].Name}' not found on type '{structType.Name}'");
                    // The IStruct field is an IValue; convert it to its Variable.
                    fieldValues[i] = ((IValue)prop.GetValue(retval)!).ToVariable();
                }

                return [InternalOp.TensorStructCreate(dtype, fieldValues)];
            }

            if (retval is System.Collections.IEnumerable enumerable)
                return enumerable.Cast<IModuleParam>().Select(x => x.ToVariable()).ToArray();

            if (retval is ITuple tuple)
                return tuple.Cast<IModuleParam>().Select(x => x.ToVariable()).ToArray();

            throw new InvalidTensorOperationException(ErrorCodes.FW002, "Return Value Processing", $"return type {retval.GetType().Name}", "Unsupported return value type - expected Variable[], IModuleParam[], IModuleParam, or ITuple");
        }

        internal static T Reformat<T>(Variable[] vars)
        {
            // ValueTuple outputs need exactly one graph output per slot — guard with a clear error rather
            // than letting the compiled converter throw IndexOutOfRange on a mismatch.
            if (ReformatImpl<T>.TupleArity is int n && vars.Length != n)
                throw new InvalidTensorOperationException(ErrorCodes.FW002, "Type Reformat",
                    $"variables count: {vars.Length}, type arguments: {n}", "Variable count must match generic type argument count");
            return ReformatImpl<T>.Run(vars);
        }

        // Variable → targetType via the handle's op_Implicit(Variable), lifting through Nullable&lt;&gt;
        // for a nullable handle slot (e.g. a Tensor&lt;float32&gt;? tuple element). Building block for the
        // compiled delegates in ReformatImpl&lt;T&gt;.
        private static E NodeToHandle(E node, Type targetType)
        {
            var underlying = Nullable.GetUnderlyingType(targetType);
            return underlying is null
                ? E.Convert(node, targetType)
                : E.Convert(E.Convert(node, underlying), targetType);
        }

        /// <summary>
        /// Per-<typeparamref name="T"/> compiled converter from a module's raw graph outputs
        /// (<see cref="Variable"/>[]) to its declared return type — a single value handle, a ValueTuple
        /// of handles, or an array of handles. The delegate is built once (the static initializer of the
        /// closed generic runs on first use) from an expression tree that emits the handles'
        /// <c>op_Implicit(Variable)</c> directly, so every later call is a plain delegate invocation with
        /// no per-call reflection.
        /// </summary>
        private static class ReformatImpl<T>
        {
            internal static readonly Func<Variable[], T> Run = Build();

            /// <summary>The required graph-output count for a ValueTuple return type (one per slot), or
            /// null when no fixed count applies (single handle / array).</summary>
            internal static readonly int? TupleArity =
                IsValueTuple<T>() ? typeof(T).GetConstructors().First().GetParameters().Length : null;

            private static Func<Variable[], T> Build()
            {
                var tType = typeof(T);
                var vars = E.Parameter(typeof(Variable[]), "vars");

                // Single value handle: (T)vars[0].
                if (tType.IsAssignableTo(typeof(IValue)))
                    return E.Lambda<Func<Variable[], T>>(
                        NodeToHandle(E.ArrayIndex(vars, E.Constant(0)), tType), vars).Compile();

                // ValueTuple of handles: new (T1,…,Tn)((T1)vars[0], …, (Tn)vars[n-1]).
                if (IsValueTuple<T>())
                {
                    var ctor = tType.GetConstructors().First();
                    var ctorParams = ctor.GetParameters();
                    // C# represents tuples with more than 7 elements as ValueTuple`8 whose 8th slot
                    // (TRest) is itself a ValueTuple; nested-tuple slots have no single graph output to
                    // map onto. Reject with a clear message (deferred, matching the unsupported-type arm)
                    // rather than emitting a broken NodeToHandle on the nested tuple.
                    if (ctorParams.Any(p => IsValueTuple(p.ParameterType)))
                        return _ => throw new UnsupportedDTypeException(ErrorCodes.FW002, tType.Name, "Reformat",
                            $"ValueTuple return types with a nested ValueTuple slot (e.g. more than 7 elements) are not supported. Received: {tType.Name}");
                    var args = ctorParams
                        .Select((p, i) => NodeToHandle(E.ArrayIndex(vars, E.Constant(i)), p.ParameterType));
                    return E.Lambda<Func<Variable[], T>>(E.New(ctor, args), vars).Compile();
                }

                // Array of handles: arr = new Elem[vars.Length]; for (i…) arr[i] = (Elem)vars[i].
                if (tType.IsArray)
                {
                    var elemType = tType.GetElementType().AssertNotNull();
                    var len = E.Variable(typeof(int), "len");
                    var arr = E.Variable(elemType.MakeArrayType(), "arr");
                    var i = E.Variable(typeof(int), "i");
                    var done = E.Label("done");
                    var body = E.Block(
                        [len, arr, i],
                        E.Assign(len, E.ArrayLength(vars)),
                        E.Assign(arr, E.NewArrayBounds(elemType, len)),
                        E.Assign(i, E.Constant(0)),
                        E.Loop(
                            E.IfThenElse(
                                E.LessThan(i, len),
                                E.Block(
                                    E.Assign(E.ArrayAccess(arr, i), NodeToHandle(E.ArrayIndex(vars, i), elemType)),
                                    E.PostIncrementAssign(i)),
                                E.Break(done)),
                            done),
                        arr);
                    return E.Lambda<Func<Variable[], T>>(body, vars).Compile();
                }

                return _ => throw new UnsupportedDTypeException(ErrorCodes.FW002, tType.Name, "Reformat",
                    $"Unsupported target type for reformatting. Expected array, ValueTuple, or assignable to IValue. Received: {tType.Name}");
            }
        }

        internal static (DataStructure[] structures, DType[] varTypes, int[] ranks) InfosFromTouts<Touts>()
        {
            var tType = typeof(Touts);
            Type[] types;
            if (ModuleHelper.IsValueTuple<Touts>())
            {
                if (tType.GenericTypeArguments.Any(x => ModuleHelper.IsValueTuple(x)))
                    throw new UnsupportedDTypeException(ErrorCodes.FW002, "nested ValueTuple", "InfosFromTouts", "Nested ValueTuple types are not supported in generic type arguments");

                types = tType.GenericTypeArguments;
            }
            else
            {
                if (!tType.IsAssignableTo(typeof(IValue)))
                    throw new UnsupportedDTypeException(ErrorCodes.FW002, tType.Name, "InfosFromTouts", $"Type must be assignable to IValue. Received: {tType.Name}");
                types = [tType];
            }

            List<DataStructure> structures = new List<DataStructure>();
            List<DType> varTypes = new List<DType>();
            List<int> ranks = new List<int>();

            foreach (var typeForEach in types)
            {
                var type = typeForEach;
                // A model-typed output (a model returning a model) isn't wired up end-to-end yet —
                // surface it as not-implemented rather than remapping to a node type that falls through
                // to the generic "unsupported type" error below.
                if (type.IsAssignableTo(typeof(IModel)))
                    throw new System.NotImplementedException("Model-typed outputs (a model returning a model) are not yet supported.");

                if (type.IsAssignableTo(typeof(ITensorStruct)))
                {
                    // TensorStruct<T> - extract the struct definition and create its DType
                    structures.Add(DataStructure.TensorStruct);
                    ranks.Add(-1);
                    
                    try
                    {
                        var (_, structDType) = StructDefExtractor.ExtractFromTensorStructType(type, "type information inference");
                        varTypes.Add(structDType);
                    }
                    catch (InvalidOperationException ex)
                    {
                        throw new UnsupportedDTypeException(ErrorCodes.FW002, "DTypeStruct", "InfosFromTouts", ex.Message);
                    }
                }
                else if (type.IsAssignableTo(typeof(ITensorSequence)))
                {
                    structures.Add(DataStructure.Sequence);
                    ranks.Add(-1);
                    
                    var dtype = OnnxUtils.GetDType(type.GenericTypeArguments[0]);
                    if (dtype is null)
                        throw new UnsupportedDTypeException(ErrorCodes.FW002, type.GenericTypeArguments[0].Name, "InfosFromTouts", $"Unable to determine DType for generic type argument: {type.GenericTypeArguments[0].Name}");
                    varTypes.Add(dtype);
                }
                else if (type.IsAssignableTo(typeof(IOptionalTensor)))
                {
                    structures.Add(DataStructure.Optional);
                    ranks.Add(-1);
                    
                    var dtype = OnnxUtils.GetDType(type.GenericTypeArguments[0]);
                    if (dtype is null)
                        throw new UnsupportedDTypeException(ErrorCodes.FW002, type.GenericTypeArguments[0].Name, "InfosFromTouts", $"Unable to determine DType for generic type argument: {type.GenericTypeArguments[0].Name}");
                    varTypes.Add(dtype);
                }
                else if (type.IsAssignableTo(typeof(ITensor)))
                {
                    structures.Add(DataStructure.Tensor);
                    if (type.IsAssignableTo(typeof(IVector)))
                        ranks.Add(1);
                    else if (type.IsAssignableTo(typeof(IScalar)))
                        ranks.Add(0);
                    else
                        ranks.Add(-1);
                    
                    var dtype = OnnxUtils.GetDType(type.GenericTypeArguments[0]);
                    if (dtype is null)
                        throw new UnsupportedDTypeException(ErrorCodes.FW002, type.GenericTypeArguments[0].Name, "InfosFromTouts", $"Unable to determine DType for generic type argument: {type.GenericTypeArguments[0].Name}");
                    varTypes.Add(dtype);
                }
                else
                {
                    throw new UnsupportedDTypeException(ErrorCodes.FW002, type.Name, "InfosFromTouts", 
                        $"Unsupported type for InfosFromTouts. Supported types: Tensor<T>, OptionalTensor<T>, TensorSequence<T>, TensorStruct<T>. Received: {type.Name}");
                }
            }

            return (structures.ToArray(), varTypes.ToArray(), ranks.ToArray());
        }

        /// <summary>
        /// Extracts generic type arguments from a Module or Model C# Type.
        /// For generated modules like GenericScaleLayerModule&lt;T&gt;, extract the T type parameters.
        /// </summary>
        internal static DType[]? ExtractGenericTypeArgsFromType(Type moduleOrModelType)
        {
            if (!moduleOrModelType.IsGenericType)
                return null;

            // Raw framework base types (Module<...> / CallbackModule<...> instances created
            // directly, e.g. via ModuleFactory) carry their hyperparam/input/output SIGNATURE
            // types as generic arguments — those are not generic-module specialization args.
            // Only user-defined generic subclasses (e.g. GenericScaleLayerModule<float32>)
            // carry genuine type arguments.
            var genericDefinition = moduleOrModelType.GetGenericTypeDefinition();
            if (genericDefinition == typeof(Module<,>) || genericDefinition == typeof(Module<,,>) ||
                genericDefinition == typeof(CallbackModule<>) || genericDefinition == typeof(CallbackModule<,>))
                return null;

            // Get the generic type arguments from the Module type
            // For GenericScaleLayerModule<float32>, this gives us [float32]
            var typeArgs = moduleOrModelType.GetGenericArguments();
            
            if (typeArgs.Length == 0)
                return null;

            var dtypes = new List<DType>();
            
            foreach (var typeArg in typeArgs)
            {
                // Extract DType from the type argument
                // For Tensor<float32>, we need to get float32
                // For Scalar<float32>, we need to get float32
                // For float32 directly, we need float32
                var dtype = TryExtractDTypeFromType(typeArg);
                if (dtype != null)
                {
                    dtypes.Add(dtype);
                }
            }

            return dtypes.Count > 0 ? dtypes.ToArray() : null;
        }

        /// <summary>
        /// Tries to extract a DType from a C# Type.
        /// Handles Tensor&lt;T&gt;, Scalar&lt;T&gt;, Vector&lt;T&gt;, and direct types like float32.
        /// For IGenericType placeholders, returns the corresponding DType.GenericTypeX constant.
        /// The stand-in processor will later add GenericTypeParamName based on the method's actual parameter names.
        /// </summary>
        private static DType? TryExtractDTypeFromType(Type type)
        {
            // If it's a generic type like Tensor<float32>, get the float32 part
            if (type.IsGenericType)
            {
                var genericArgs = type.GetGenericArguments();
                if (genericArgs.Length > 0)
                {
                    // Recursively extract from the first generic argument
                    return TryExtractDTypeFromType(genericArgs[0]);
                }
            }

            // Use OnnxUtils.GetDType which handles all types including IGenericType placeholders
            // GenericTypeParamName will be added later by the stand-in processor when it has
            // access to the method's actual parameter names
            var dtype = OnnxUtils.GetDType(type);
            return dtype;
        }

    }
}


