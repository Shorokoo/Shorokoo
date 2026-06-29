using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo.Core
{
    /// <summary>
    /// DispatchProxy-based implementation that allows IStruct interface property access
    /// on TensorStruct variables. When module Inline methods declare parameters with
    /// IStruct interface types (e.g. RealGenericPairStruct&lt;U, V&gt; pair), this proxy
    /// intercepts property getters and translates them to TensorStructGetField graph operations.
    /// </summary>
    internal class TensorStructProxy : DispatchProxy, ITensorStructProxy
    {
        private Variable? _backingTensorStruct;
        private TensorStructDef? _definition;
        private Dictionary<string, Variable>? _fieldCache;

        /// <summary>
        /// Gets the backing TensorStruct Variable that this proxy wraps. Typically an
        /// <see cref="Variable"/> but may be a plain <see cref="Variable"/> when the
        /// proxy was built from the result of a struct-typed graph op (e.g. SequenceAt over
        /// a struct sequence) whose Variable instance carries struct-shaped data but is
        /// not declared as Variable in the type system.
        /// </summary>
        public Variable BackingTensorStruct => _backingTensorStruct
            ?? throw new InvalidOperationException("TensorStructProxy has not been initialized. Call Initialize() first.");

        /// <summary>
        /// Gets the TensorStructDef describing this struct's fields.
        /// </summary>
        public TensorStructDef Definition => _definition
            ?? throw new InvalidOperationException("TensorStructProxy has not been initialized. Call Initialize() first.");

        /// <summary>
        /// Initializes the proxy with the backing TensorStruct and definition.
        /// Must be called after DispatchProxy.Create() since the constructor is parameterless.
        /// </summary>
        internal void Initialize(Variable backingTensorStruct, TensorStructDef definition)
        {
            _backingTensorStruct = backingTensorStruct ?? throw new ArgumentNullException(nameof(backingTensorStruct));
            _definition = definition ?? throw new ArgumentNullException(nameof(definition));
            _fieldCache = new Dictionary<string, Variable>();
        }

        /// <summary>
        /// Intercepts method calls on the IStruct interface. Property getters (get_FieldName)
        /// are translated to TensorStructGetField operations on the backing TensorStruct.
        /// </summary>
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod == null)
                throw new ArgumentNullException(nameof(targetMethod));

            // Property getter: get_First → extract field "First"
            if (targetMethod.Name.StartsWith("get_") && (args == null || args.Length == 0))
            {
                var fieldName = targetMethod.Name.Substring(4);
                // The property return type is a value-struct IValue; convert the field's Variable to it
                // so the DispatchProxy can assign it. This must be driven by the declared return type, not
                // the value's natural handle (which may differ for a generic-standin or more general
                // declared type).
                return GetOrCreateFieldVariable(fieldName).ToValue(targetMethod.ReturnType);
            }

            // IValue.ToVariable support — convert to the backing Variable
            // This shouldn't normally be called on the proxy, but handle it for safety
            if (targetMethod.Name == "ToVariable" && targetMethod.DeclaringType == typeof(IValue))
            {
                return _backingTensorStruct;
            }

            throw new NotSupportedException(
                $"TensorStructProxy only supports property getters on IStruct interfaces. " +
                $"Unsupported method call: {targetMethod.DeclaringType?.Name}.{targetMethod.Name}");
        }

        /// <summary>
        /// Gets or creates the Variable for a field, using TensorStructGetField to create
        /// graph operations for field access.
        /// </summary>
        private Variable GetOrCreateFieldVariable(string fieldName)
        {
            if (_fieldCache!.TryGetValue(fieldName, out var cached))
                return cached;

            var fieldDef = _definition!.GetField(fieldName);
            if (fieldDef == null)
                throw new KeyNotFoundException(
                    $"Field '{fieldName}' not found in TensorStruct definition '{_definition.TypeName}'. " +
                    $"Available fields: {string.Join(", ", GetFieldNames())}");

            // Create the TENSOR_STRUCT_GETFIELD graph node
            var fieldVariable = InternalOp.TensorStructGetField(
                _backingTensorStruct!,
                fieldName,
                fieldDef.ElementType,
                fieldDef.Rank,
                fieldDef.Structure);

            _fieldCache[fieldName] = fieldVariable;
            return fieldVariable;
        }

        private IEnumerable<string> GetFieldNames()
        {
            if (_definition == null) yield break;
            foreach (var field in _definition.Fields)
                yield return field.Name;
        }
    }

    /// <summary>
    /// Interface to identify and access TensorStructProxy instances regardless of the
    /// IStruct interface type they implement.
    /// </summary>
    internal interface ITensorStructProxy
    {
        /// <summary>
        /// Gets the backing TensorStruct Variable that this proxy wraps.
        /// </summary>
        Variable BackingTensorStruct { get; }

        /// <summary>
        /// Gets the struct definition.
        /// </summary>
        TensorStructDef Definition { get; }
    }

    /// <summary>
    /// Factory for creating TensorStructProxy instances that implement IStruct interfaces.
    /// </summary>
    internal static class TensorStructProxyFactory
    {
        /// <summary>
        /// Creates a DispatchProxy instance that implements the given IStruct interface type,
        /// backed by the given TensorStruct variable. Property access on the returned object
        /// will be translated to TensorStructGetField graph operations.
        /// </summary>
        /// <param name="interfaceType">The IStruct interface type to implement (must be an interface)</param>
        /// <param name="backingTensorStruct">The TensorStruct variable to wrap</param>
        /// <param name="definition">The TensorStructDef describing the struct</param>
        /// <returns>An object implementing the interface with property access wired to graph operations</returns>
        public static object Create(Type interfaceType, Variable backingTensorStruct, TensorStructDef definition)
        {
            if (!interfaceType.IsInterface)
                throw new ArgumentException(
                    $"TensorStructProxy only supports interface types. '{interfaceType.Name}' is not an interface. " +
                    "For record types, use record constructor. For classes with settable properties, use property setters.",
                    nameof(interfaceType));

            // DispatchProxy.Create<TInterface, TProxy>() requires generic call.
            // We use reflection to call the generic method with the runtime interface type.
            // Use GetMethods to avoid AmbiguousMatchException since DispatchProxy may have multiple Create overloads.
            var createMethod = typeof(DispatchProxy)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == nameof(DispatchProxy.Create) && m.GetGenericArguments().Length == 2)
                .MakeGenericMethod(interfaceType, typeof(TensorStructProxy));

            var proxy = createMethod.Invoke(null, null)!;

            // Initialize the proxy with the backing TensorStruct
            var tensorStructProxy = (TensorStructProxy)proxy;
            tensorStructProxy.Initialize(backingTensorStruct, definition);

            return proxy;
        }

    }
}
