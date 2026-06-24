using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Shorokoo.Core
{
    internal static class InternalGlobals
    {
        /// <summary>
        /// Thread-local storage for state update pairs during module invocation.
        /// Each pair represents (original state, updated state) registered via StateUpdate.
        /// </summary>
        [ThreadStatic]
        private static List<(IVariable original, IVariable updated)>? _stateUpdatePairs;

        /// <summary>
        /// Gets the current state update pairs collection, creating it if necessary.
        /// </summary>
        private static List<(IVariable original, IVariable updated)> StateUpdatePairs
            => _stateUpdatePairs ??= new List<(IVariable original, IVariable updated)>();

        /// <summary>
        /// Registers a state update relationship between an original state tensor and its updated value.
        /// Called by Globals.StateUpdate to track state updates during module execution.
        /// </summary>
        /// <param name="original">The original state tensor from a state initializer</param>
        /// <param name="updated">The computed updated value for the state</param>
        /// <returns>The linked updated state tensor (output of STATE_UPDATE_LINK node)</returns>
        internal static IVariable RegisterStateUpdate(IVariable original, IVariable updated)
        {
            
            // Create the STATE_UPDATE_LINK node to track the relationship in the graph
            var linkedUpdated = InternalOp.StateUpdateLink(original, updated);
            
            // Store the pair for later retrieval when wrapping module outputs
            StateUpdatePairs.Add((original, linkedUpdated));
            
            return linkedUpdated;
        }

        /// <summary>
        /// Retrieves and clears all registered state update pairs for the current module invocation.
        /// Called after the module's Inline method returns to wrap outputs with WithStateDeps.
        /// </summary>
        /// <returns>Array of updated state tensors (the linked versions)</returns>
        internal static IVariable[] GetAndClearStateUpdates()
        {
            if (_stateUpdatePairs == null || _stateUpdatePairs.Count == 0)
                return Array.Empty<IVariable>();

            var updates = _stateUpdatePairs.Select(p => p.updated).ToArray();
            _stateUpdatePairs.Clear();
            
            return updates;
        }

        /// <summary>
        /// Checks if there are any pending state updates that need to be wrapped.
        /// </summary>
        internal static bool HasPendingStateUpdates
            => _stateUpdatePairs != null && _stateUpdatePairs.Count > 0;

        internal static ITensor Tensor(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name = null, int? rank = null)
            => (ITensor)OnnxUtils.CallGeneric(dtype.ToIVarType(), typeof(InternalGlobals), nameof(CreateTensorWithShapeFn), [shapeFn, dtype, owningNode, moduleFn, name, rank]);

        private static Tensor<T> CreateTensorWithShapeFn<T>(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name, int? rank) where T : IVarType
            => new Tensor<T>(shapeFn, dtype, owningNode, moduleFn, name, rank: rank);

        internal static ITensor Vector(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name = null)
            => (ITensor)OnnxUtils.CallGeneric(dtype.ToIVarType(), typeof(InternalGlobals), nameof(CreateVectorWithShapeFn), [shapeFn, dtype, owningNode, moduleFn, name]);

        private static Vector<T> CreateVectorWithShapeFn<T>(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name) where T : IVarType
            => new Vector<T>(shapeFn, dtype, owningNode, moduleFn, name);

        internal static IScalar Scalar(DType dtype, Node? owningNode, Function? moduleFn = null, string? name = null)
            => (IScalar)OnnxUtils.CallGeneric(dtype.ToIVarType(), typeof(InternalGlobals), nameof(CreateScalar), [owningNode, dtype, moduleFn, name]);

        private static Scalar<T> CreateScalar<T>(Node owningNode, DType dtype, Function? moduleFn, string? name = null) where T : IVarType
            => new Scalar<T>(owningNode, dtype, moduleFn, name);

        private static Scalar<T> CreateScalarValForObj<T>(object val) where T : IVarType
            => Shorokoo.Globals.Scalar<T>(val);

        internal static TensorSequence<T> TensorSequence<T>(Node owningNode, DType dtype, Function? moduleFn, string? name) where T : IVarType
            => new TensorSequence<T>(dtype, owningNode, moduleFn, name);

        internal static ITensorSequence TensorSequence(DType dtype, Node owningNode, Function? moduleFn, string? name = null)
            => (ITensorSequence)OnnxUtils.CallGeneric(dtype.ToIVarType(), typeof(InternalGlobals), nameof(CreateTensorSequence), [owningNode, dtype, moduleFn, name]);

        private static TensorSequence<T> CreateTensorSequence<T>(Node owningNode, DType dtype, Function? moduleFn, string? name) where T : IVarType
            => TensorSequence<T>(owningNode, dtype, moduleFn, name);

        internal static OptionalTensor<T> OptionalTensor<T>(Node owningNode, DType dtype, Function? moduleFn, string? name = null) where T : IVarType
            => new OptionalTensor<T>(dtype, owningNode, moduleFn, name);

        internal static IOptionalTensor OptionalTensor(DType dtype, Node owningNode, Function? moduleFn, string? name = null)
            => (IOptionalTensor)OnnxUtils.CallGeneric(dtype.ToIVarType(), typeof(InternalGlobals), nameof(CreateOptionalTensor), [owningNode, dtype, moduleFn, name]);

        internal static IVector EmptyVector(DType type) => (IVector)Shorokoo.Core.Nodes.NodeDefinitions.OnnxOp.Constant(Globals.TensorData(type));

        private static OptionalTensor<T> CreateOptionalTensor<T>(Node owningNode, DType dtype, Function? moduleFn, string? name = null) where T : IVarType
            => OptionalTensor<T>(owningNode, dtype, moduleFn, name);

        /// <summary>
        /// Creates a TensorStruct for a given DType (which must be a TensorStruct type).
        /// </summary>
        internal static ITensorStruct TensorStruct(DType type, Node owningNode, Function? moduleFn, string? name = null)
        {
            if (!type.IsTensorStructType)
                throw new InvalidOperationException($"DType {type} is not a TensorStruct type");
            
            var structDef = type.TensorStructDef;
            if (structDef == null)
                throw new InvalidOperationException($"TensorStruct DType {type} has no associated TensorStructDef");

            // Use DTypeStruct for dynamically typed TensorStruct
            return new TensorStruct<DTypeStruct>(type, owningNode, moduleFn, name, structDef);
        }

        /// <summary>
        /// Creates a TensorStruct with a specific IStruct type.
        /// </summary>
        internal static TensorStruct<T> TensorStruct<T>(DType type, Node owningNode, Function? moduleFn, string? name, TensorStructDef definition) where T : IStruct
            => new TensorStruct<T>(type, owningNode, moduleFn, name, definition);

        internal static OnnxTensorData<bit> OnnxTensorData(Shape shape, params bool[] data) => new OnnxTensorData<bit>(shape, OnnxUtils.CreateTensorValue<bool>(shape, data));
        internal static OnnxTensorData<int8> OnnxTensorData(Shape shape, params sbyte[] data) => new OnnxTensorData<int8>(shape, OnnxUtils.CreateTensorValue<sbyte>(shape, data));
        internal static OnnxTensorData<int16> OnnxTensorData(Shape shape, params short[] data) => new OnnxTensorData<int16>(shape, OnnxUtils.CreateTensorValue<short>(shape, data));
        internal static OnnxTensorData<int32> OnnxTensorData(Shape shape, params int[] data) => new OnnxTensorData<int32>(shape, OnnxUtils.CreateTensorValue<int>(shape, data));
        internal static OnnxTensorData<int64> OnnxTensorData(Shape shape, params long[] data) => new OnnxTensorData<int64>(shape, OnnxUtils.CreateTensorValue<long>(shape, data));
        internal static OnnxTensorData<uint8> OnnxTensorData(Shape shape, params byte[] data) => new OnnxTensorData<uint8>(shape, OnnxUtils.CreateTensorValue<byte>(shape, data));
        internal static OnnxTensorData<uint16> OnnxTensorData(Shape shape, params ushort[] data) => new OnnxTensorData<uint16>(shape, OnnxUtils.CreateTensorValue<ushort>(shape, data));
        internal static OnnxTensorData<uint32> OnnxTensorData(Shape shape, params uint[] data) => new OnnxTensorData<uint32>(shape, OnnxUtils.CreateTensorValue<uint>(shape, data));
        internal static OnnxTensorData<uint64> OnnxTensorData(Shape shape, params ulong[] data) => new OnnxTensorData<uint64>(shape, OnnxUtils.CreateTensorValue<ulong>(shape, data));
        internal static OnnxTensorData<bfloat16> OnnxTensorData(Shape shape, params BFloat16[] data) => new OnnxTensorData<bfloat16>(shape, OnnxUtils.CreateTensorValue<BFloat16>(shape, data));
        internal static OnnxTensorData<float16> OnnxTensorData(Shape shape, params Float16[] data) => new OnnxTensorData<float16>(shape, OnnxUtils.CreateTensorValue<Float16>(shape, data));
        internal static OnnxTensorData<float32> OnnxTensorData(Shape shape, params float[] data) => new OnnxTensorData<float32>(shape, OnnxUtils.CreateTensorValue<float>(shape, data));
        internal static OnnxTensorData<float64> OnnxTensorData(Shape shape, params double[] data) => new OnnxTensorData<float64>(shape, OnnxUtils.CreateTensorValue<double>(shape, data));

        internal static T IdentityOp<T>(T var) where T : IVariable
            => (T)OnnxOp.Identity(var, var.Rank());
    }
}
