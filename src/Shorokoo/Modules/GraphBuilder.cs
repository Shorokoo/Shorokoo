using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Training;

namespace Shorokoo.Core
{
    /// <summary>
    /// Helper class for building computation graphs from module methods.
    /// Used by source-generated module code to create VirtualGraph instances.
    /// </summary>
    public static class GraphBuilder
    {
        /// <summary>
        /// Ordered list of Shorokoo datatype primitives, ordered by how common/typical they are in ML.
        /// This is used to select concrete types for generic type parameters.
        /// </summary>
        internal static readonly Type[] OrderedMLDataTypes = new[]
        {
            // Most common float types first
            typeof(float32),   // Most common in ML
            typeof(float64),   // Double precision
            typeof(float16),   // Half precision (common in modern ML)
            typeof(bfloat16),  // Brain float (increasingly common)
            
            // Common integer types
            typeof(int64),     // Most common integer type in ML frameworks
            typeof(int32),     // Standard integer
            typeof(int16),     // Short integer
            typeof(int8),      // Byte integer
            typeof(int4),      // 4-bit integer
            
            // Unsigned integers
            typeof(uint64),
            typeof(uint32),
            typeof(uint16),
            typeof(uint8),
            typeof(uint4),
            
            // Complex types (rare)
            typeof(complex64),
            typeof(complex128),
            
            // Other types
            typeof(bit),        // Boolean-like
            typeof(@string),   // String type
        };

        /// <summary>
        /// Builds a <see cref="FastComputationGraph"/> from a MethodInfo (typically the Inline method).
        /// For generic methods, this instantiates the method with concrete types chosen
        /// based on generic type parameter constraints.
        /// This is the main entry point used by generated code.
        /// </summary>
        public static FastComputationGraph BuildFastComputationGraphFromMethodInfo(MethodInfo methodInfo)
        {
            if (methodInfo == null)
                throw new ArgumentNullException(nameof(methodInfo));

            return BuildFastComputationGraphFromMethod(methodInfo);
        }

        /// <summary>
        /// Builds a <see cref="FastComputationGraph"/> from a delegate's underlying method —
        /// the codegen-free equivalent of the source generator's <c>ComputationGraph</c>
        /// property (which routes through <see cref="BuildFastComputationGraphFromMethodInfo"/>).
        ///
        /// <para>The delegate must be a static method group or a non-capturing
        /// (<c>static</c>) lambda: the body is invoked once by reflection to trace the graph,
        /// so captured locals would be baked invisibly into the result. Trainable-parameter
        /// initializers, <c>Globals.StateUpdate</c>, <c>LoopAPI.Iterate</c>, and sub-module
        /// calls all work inside the delegate body exactly as in a <c>[Module]</c> class's
        /// <c>Inline</c> method. See also <see cref="Shorokoo.Modules.ModuleFactory"/> for the
        /// cached, Module-object-producing entry points.</para>
        /// </summary>
        /// <param name="fn">The module body. Parameters must be flattened tensor parameters
        /// (no tuples); leading hyperparameters are marked with <c>[Hyper]</c>.</param>
        /// <returns>A freshly built graph (not cached — callers own the instance).</returns>
        public static FastComputationGraph BuildFastComputationGraphFromDelegate(Delegate fn)
        {
            if (fn == null)
                throw new ArgumentNullException(nameof(fn));

            ModuleHelper.EnsureNonCapturingDelegate(fn.Target, fn.Method.Name);
            return BuildFastComputationGraphFromMethod(fn.Method, fn.Target);
        }

        /// <summary>
        /// Core method to build a <see cref="FastComputationGraph"/> from a MethodInfo,
        /// handling generic type resolution. For generic methods, this uses IGenericType
        /// placeholders to preserve generic type information through the graph building process.
        /// <paramref name="invokeTarget"/> is the reflection-invoke receiver: null for static
        /// methods, or the delegate's bound target for compiler-generated lambda methods.
        /// </summary>
        internal static FastComputationGraph BuildFastComputationGraphFromMethod(MethodInfo methodInfo, object? invokeTarget = null)
        {
            if (methodInfo == null)
                throw new ArgumentNullException(nameof(methodInfo));

            // Extract generic parameter information before instantiation
            MethodInfo originalGenericMethod = methodInfo;
            Dictionary<Type, string>? genericTypeToParamName = null;
            Dictionary<int, string[]>? genericConstraints = null;

            if (methodInfo.IsGenericMethodDefinition)
            {
                originalGenericMethod = methodInfo;
                var genericArgs = methodInfo.GetGenericArguments();
                genericTypeToParamName = new Dictionary<Type, string>();

                // For each generic parameter, store its name
                // We'll map the Type object representing the generic parameter to its name
                for (int i = 0; i < genericArgs.Length; i++)
                {
                    genericTypeToParamName[genericArgs[i]] = genericArgs[i].Name;
                }

                // Resolve to IGenericType placeholders instead of concrete types
                // This allows constants created with Scalar<T> to preserve generic type information
                methodInfo = ResolveGenericMethodWithPlaceholders(methodInfo, out genericConstraints);
            }

            // Clear any leftover state updates from previous invocations to prevent state leakage
            // This is important because state updates registered outside of a module graph context
            // should not affect subsequent module/initializer graph builds
            Shorokoo.Core.InternalGlobals.GetAndClearStateUpdates();

            // Modules speak in value handles, not the internal graph node type. (Inputs are validated
            // per-parameter inside CreateInputParams.)
            ModuleHelper.RejectVariableParam(methodInfo.ReturnType);

            // Create input parameters based on the method signature
            var fnInputs = ModuleHelper.CreateInputParams(methodInfo.GetParameters());

            LoopAPI.PushLooperContext();
            try
            {
                Variable[] fnOutputs;
                Variable[] stateUpdates;
                try
                {
                    fnOutputs = ModuleHelper.InvokeAndFormat(methodInfo, fnInputs, invokeTarget);
                }
                finally
                {
                    // Always clear state updates, even if an exception occurred
                    // This prevents state leakage between module invocations
                    stateUpdates = Shorokoo.Core.InternalGlobals.GetAndClearStateUpdates();
                }

                // Check for registered state updates and wrap outputs with WithStateDeps if any exist
                // This ensures state update tensors are included in the graph when outputs are used
                if (stateUpdates.Length > 0)
                {
                    // Wrap each output with WithStateDeps to create dependencies on state update tensors
                    fnOutputs = WrapOutputsWithStateDeps(fnOutputs, stateUpdates);
                }

                // For generic methods, prepend GENERIC_TYPE_INPUT nodes for each generic type parameter
                var allInputs = new List<Variable>();

                if (originalGenericMethod != null)
                {
                    var genericArgs = originalGenericMethod.GetGenericArguments();
                    for (int i = 0; i < genericArgs.Length; i++)
                    {
                        var genericArg = genericArgs[i];
                        var paramName = genericArg.Name;
                        var genericIndex = i + 1;

                        // Get the IGenericTypeX placeholder that was used
                        var placeholderType = GetGenericTypePlaceholder(genericIndex);
                        var dtype = OnnxUtils.GetDType(placeholderType).AssertNotNull();

                        // Get constraints if any
                        string[]? constraints = null;
                        genericConstraints?.TryGetValue(genericIndex, out constraints);

                        // Create GENERIC_TYPE_INPUT node
                        var genericTypeInput = InternalOp.GenericTypeInput(
                            dtype,
                            rank: 0, // Generic type info is a scalar
                            constraints: constraints,
                            defaultName: $"GenericType_{paramName}");
                        allInputs.Add(genericTypeInput);
                    }
                }

                // Add input parameters after the generic type inputs. User Inline signatures
                // are written inputs-first / hyperparameters-last, but the framework keeps its
                // graph input list ordered hyperparameters-first (every downstream consumer —
                // module-call inlining, signatures, concretization — relies on that order). The
                // body was already invoked with fnInputs in declaration order above, so only the
                // graph's input *list* is reordered here; the produced graph is identical to the
                // legacy hyperparameters-first layout.
                var fnInputVars = fnInputs.Select(x => x.ToVariable()).ToList();
                allInputs.AddRange(fnInputVars.Where(IsHyperparamInput));
                allInputs.AddRange(fnInputVars.Where(v => !IsHyperparamInput(v)));

                var fastGraph = new FastComputationGraph(
                    [.. allInputs],
                    [.. fnOutputs]);

                if (originalGenericMethod?.IsGenericMethodDefinition == true)
                {
                    var genericArgs = originalGenericMethod.GetGenericArguments();
                    var genericIndexToParamName = new Dictionary<int, string>();

                    // Map generic type index (1-based) to parameter name
                    for (int i = 0; i < genericArgs.Length; i++)
                        genericIndexToParamName[i + 1] = genericArgs[i].Name;

                    Shorokoo.Core.Nodes.Processors.Fast.FastConvertPlaceholderGenericTypesToDefaultGenericTypes
                        .Process(fastGraph, genericIndexToParamName);
                }

                Shorokoo.Core.Nodes.Processors.Fast.FastApplyIdentifierTemplates.Process(fastGraph);
                return fastGraph;
            }
            finally
            {
                LoopAPI.PopLooperContext();
            }
        }

        /// <summary>
        /// Resolves a generic method definition by instantiating it with IGenericType placeholders
        /// instead of concrete types. This allows operators without input-based type inference
        /// (like Cast, SequenceEmpty) to properly participate in type specialization.
        /// </summary>
        /// <param name="methodInfo">The generic method definition</param>
        /// <param name="genericConstraints">Output dictionary mapping generic type index to constraint type names</param>
        /// <returns>A method instance with IGenericTypeX placeholders for type parameters</returns>
        internal static MethodInfo ResolveGenericMethodWithPlaceholders(
            MethodInfo methodInfo,
            out Dictionary<int, string[]> genericConstraints)
        {
            genericConstraints = new Dictionary<int, string[]>();
            
            if (!methodInfo.IsGenericMethodDefinition)
                return methodInfo;

            var genericArgs = methodInfo.GetGenericArguments();
            var placeholderTypes = new Type[genericArgs.Length];
            
            // Map each generic parameter to IGenericTypeN and extract constraints
            for (int i = 0; i < genericArgs.Length; i++)
            {
                placeholderTypes[i] = GetGenericTypePlaceholder(i + 1);
                
                // Extract constraint information
                var constraints = genericArgs[i].GetGenericParameterConstraints();
                if (constraints.Length > 0)
                {
                    genericConstraints[i + 1] = constraints.Select(c => c.Name).ToArray();
                }
            }
            
            return methodInfo.MakeGenericMethod(placeholderTypes);
        }

        /// <summary>
        /// Resolves a generic method definition by instantiating it with IGenericType placeholders
        /// (backward compatibility overload without constraint output).
        /// </summary>
        internal static MethodInfo ResolveGenericMethodWithPlaceholders(MethodInfo methodInfo)
        {
            return ResolveGenericMethodWithPlaceholders(methodInfo, out _);
        }

        /// <summary>
        /// Gets the IGenericTypeN interface for the given index (1-based).
        /// </summary>
        /// <param name="index">The generic type parameter index (1-8)</param>
        /// <returns>The IGenericTypeN interface type</returns>
        public static Type GetGenericTypePlaceholder(int index)
        {
            return index switch
            {
                1 => typeof(IGenericType1),
                2 => typeof(IGenericType2),
                3 => typeof(IGenericType3),
                4 => typeof(IGenericType4),
                5 => typeof(IGenericType5),
                6 => typeof(IGenericType6),
                7 => typeof(IGenericType7),
                8 => typeof(IGenericType8),
                _ => throw new ArgumentException($"Generic type index {index} exceeds maximum. Supported range is 1-8.")
            };
        }

        /// <summary>
        /// True when an input variable was created for a <c>[Hyper]</c> parameter (its owning
        /// model-input node carries <see cref="InputType.Hyperparam"/>). Used to keep the graph's
        /// input list ordered hyperparameters-first even though Inline signatures are inputs-first.
        /// </summary>
        private static bool IsHyperparamInput(Variable v)
            => v.OwningNode.Attributes.GetEnumVal<InputType>(OnnxOpAttributeNames.ShrkAttrInputType) == InputType.Hyperparam;

        /// <summary>
        /// Wraps output tensors with WithStateDeps to create graph dependencies on state update tensors.
        /// This ensures that if any output is part of the graph, all state updates are included.
        /// </summary>
        /// <param name="outputs">The original output tensors from the module's Inline method</param>
        /// <param name="stateUpdates">The registered state update tensors</param>
        /// <returns>The wrapped output tensors</returns>
        private static Variable[] WrapOutputsWithStateDeps(Variable[] outputs, Variable[] stateUpdates)
        {
            if (stateUpdates.Length == 0)
                return outputs;

            // Wrap all output tensors with WithStateDeps to create a dependency on all state updates
            // This ensures that when any output is used, the state updates are also included in the graph
            // We wrap all outputs (not just the first) to handle cases where only some outputs are used
            var wrappedOutputs = new Variable[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                wrappedOutputs[i] = InternalOp.WithStateDeps(outputs[i], stateUpdates);
            }

            return wrappedOutputs;
        }
    }
}
