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
    /// <summary>
    /// The state-update pairs registered during one module-body trace: a dead-simple
    /// accumulate-then-harvest list. What a state update means lives here, next to
    /// <see cref="InternalGlobals.RegisterStateUpdate"/>; the ambient trace merely carries
    /// an instance (see <see cref="GraphTrace"/>), and the graph builder harvests it with
    /// <see cref="Take"/> after the body returns to wrap the module outputs with
    /// WithStateDeps.
    /// </summary>
    internal sealed class StateUpdateRegistry
    {
        private List<(Variable original, Variable updated)>? _pairs;

        /// <summary>Records one pair; <paramref name="linkedUpdated"/> is the
        /// STATE_UPDATE_LINK output wrapping the user's updated value.</summary>
        internal void Add(Variable original, Variable linkedUpdated)
            => (_pairs ??= new List<(Variable, Variable)>()).Add((original, linkedUpdated));

        /// <summary>Harvests (and clears) the linked updated-state tensors, in registration order.</summary>
        internal Variable[] Take()
        {
            if (_pairs is null || _pairs.Count == 0)
                return Array.Empty<Variable>();
            var updates = _pairs.Select(p => p.updated).ToArray();
            _pairs = null;
            return updates;
        }
    }

    internal static class InternalGlobals
    {
        /// <summary>
        /// Registers a state update relationship between an original state tensor and its updated value.
        /// Called by Globals.StateUpdate to track state updates during module execution. The pair is
        /// recorded on the current module build (see <see cref="GraphTrace"/>), where the graph builder
        /// harvests it after the body returns to wrap the module outputs with WithStateDeps; with no
        /// module build in progress the registration could never be harvested, so this throws instead.
        /// </summary>
        /// <param name="original">The original state tensor from a state initializer</param>
        /// <param name="updated">The computed updated value for the state</param>
        /// <returns>The linked updated state tensor (output of STATE_UPDATE_LINK node)</returns>
        internal static Variable RegisterStateUpdate(Variable original, Variable updated)
        {
            GraphTrace.EnsureStateUpdateRecordable();

            // Create the STATE_UPDATE_LINK node to track the relationship in the graph
            var linkedUpdated = InternalOp.StateUpdateLink(original, updated);

            // Store the pair for later retrieval when wrapping module outputs
            GraphTrace.RecordStateUpdate(original, linkedUpdated);

            return linkedUpdated;
        }

        // Every kind is now the single non-generic Variable, built directly from the runtime DType +
        // kind + rank — no per-output CallGeneric/MakeGenericType reflection round-trip.
        internal static Variable Tensor(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name = null, int? rank = null)
            => new Variable(dtype, owningNode, moduleFn, name, DataStructure.Tensor, rank: rank, shapeFn: shapeFn);

        internal static Variable Vector(Func<Vector<int64>>? shapeFn, DType dtype, Node owningNode, Function? moduleFn, string? name = null)
            => new Variable(dtype, owningNode, moduleFn, name, DataStructure.Tensor, rank: 1, shapeFn: shapeFn);

        internal static Variable Scalar(DType dtype, Node? owningNode, Function? moduleFn = null, string? name = null)
            => new Variable(dtype, owningNode!, moduleFn, name, DataStructure.Tensor, rank: 0);

        private static Scalar<T> CreateScalarValForObj<T>(object val) where T : IVarType
            => Shorokoo.Globals.Scalar<T>(val);

        internal static Variable TensorSequence(DType dtype, Node owningNode, Function? moduleFn, string? name = null)
            => new Variable(dtype, owningNode, moduleFn, name, DataStructure.Sequence);

        internal static Variable OptionalTensor(DType dtype, Node owningNode, Function? moduleFn, string? name = null)
            => new Variable(dtype, owningNode, moduleFn, name, DataStructure.Optional);

        internal static Variable EmptyVector(DType type) => (Variable)Shorokoo.Core.Nodes.NodeDefinitions.OnnxOp.Constant(Globals.TensorData(type));

        /// <summary>
        /// Creates a TensorStruct for a given DType (which must be a TensorStruct type).
        /// </summary>
        internal static Variable TensorStruct(DType type, Node owningNode, Function? moduleFn, string? name = null)
        {
            if (!type.IsTensorStructType)
                throw new InvalidOperationException($"DType {type} is not a TensorStruct type");
            
            var structDef = type.TensorStructDef;
            if (structDef == null)
                throw new InvalidOperationException($"TensorStruct DType {type} has no associated TensorStructDef");

            // The struct node is the single non-generic Variable; the field layout lives in the runtime TensorStructDef.
            return new Variable(type, owningNode, moduleFn, name, DataStructure.TensorStruct, structDef: structDef);
        }

        /// <summary>
        /// Creates a TensorStruct with a specific IStruct type. The element type parameter is kept for
        /// caller convenience but the node itself is non-generic (its layout is the runtime definition).
        /// </summary>
        internal static Variable TensorStruct<T>(DType type, Node owningNode, Function? moduleFn, string? name, TensorStructDef definition) where T : IStruct
            => new Variable(type, owningNode, moduleFn, name, DataStructure.TensorStruct, structDef: definition);

        /// <summary>
        /// The established default <see cref="Variable"/> for a value-handle type: a zero scalar (rank 0),
        /// a zero-length vector (rank 1 / unknown), default-filled struct fields, or an absent optional /
        /// empty sequence. Used to materialise a defaulted/absent handle.
        /// </summary>
        internal static Variable DefaultVariable(Type type)
        {
            ModuleHelper.RejectVariableParam(type);
            if (type.IsAssignableTo(typeof(ITensorStruct)))
            {
                var (structDef, structDType) = StructDefExtractor.ExtractFromTensorStructType(type, "default value creation");

                // Create default tensor variables for each field (empty tensors).
                var fieldVars = structDef.Fields.Select(f =>
                {
                    long[] shape = f.Rank.HasValue && f.Rank.Value == 0 ? [] : [0];
                    return Globals.DefaultTensor(f.ElementType, shape);
                }).ToArray();

                return InternalOp.TensorStructCreate(structDType, fieldVars);
            }

            if (type.IsAssignableTo(typeof(ITensor)))
            {
                var dtype = OnnxUtils.GetDType(type.GenericTypeArguments[0]).AssertNotNull();
                // A scalar (rank 0) defaults to a zero scalar; a vector / general tensor (rank >= 1 or
                // unknown) to a zero-length vector.
                long[] shape = type.IsAssignableTo(typeof(IScalar)) ? [] : [0];
                return Globals.DefaultTensor(dtype, shape);
            }

            if (type.IsAssignableTo(typeof(IOptionalTensor)))
                return Globals.OptionalTensor(OnnxUtils.GetDType(type.GenericTypeArguments[0]).AssertNotNull()); // Empty optional tensor.

            if (type.IsAssignableTo(typeof(ITensorSequence)))
                return Globals.TensorSequence(OnnxUtils.GetDType(type.GenericTypeArguments[0]).AssertNotNull()); // Empty sequence.

            throw new UnsupportedDTypeException(ErrorCodes.FW002, type.Name, "DefaultVariable", $"Unsupported type for default value creation. Supported types: Tensor<T>, OptionalTensor<T>, TensorSequence<T>, TensorStruct<T>. Received: {type.Name}");
        }

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

        internal static T IdentityOp<T>(T var) where T : Variable
            => (T)OnnxOp.Identity(var, var.Rank);
    }
}
