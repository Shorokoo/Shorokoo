using Shorokoo.Core;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Text;

namespace Shorokoo
{
    /// <summary>
    /// Static entry points of the Shorokoo graph-building API: constant/input/trainable
    /// tensor constructors, TensorData factories, and module helpers. Typically imported
    /// via <c>using static Shorokoo.Globals;</c>.
    /// </summary>
    public static partial class Globals
    {
        /// <summary>
        /// Creates a trainable parameter node whose value is produced by the given
        /// [TrainableParamInitializer] delegate, invoked with the supplied inputs.
        /// </summary>
        public static Variable CallTrainableParamInitializer(Delegate trainableParamInitializerImplementation, params Variable[] inputs)
        {
            return (Variable)CallTrainableParamInitializer(trainableParamInitializerImplementation, defaultName: null, isTrainable: true, inputs);
        }

        /// <summary>
        /// Creates a parameter node produced by the given initializer delegate, with an explicit
        /// default name; <paramref name="isTrainable"/> selects a trainable parameter (true)
        /// or a state parameter (false). State parameters default to
        /// <see cref="StateOwnership.ModuleOwned"/>; use the overload taking a
        /// <see cref="StateOwnership"/> for optimizer-owned state.
        /// </summary>
        public static Variable CallTrainableParamInitializer(Delegate trainableParamInitializerImplementation, string? defaultName, bool isTrainable, params Variable[] inputs)
            => CallTrainableParamInitializer(trainableParamInitializerImplementation, defaultName, isTrainable, StateOwnership.ModuleOwned, inputs);

        /// <summary>
        /// Creates a parameter node produced by the given initializer delegate.
        /// <paramref name="isTrainable"/> selects a trainable parameter (true) or a state
        /// parameter (false); for state parameters, <paramref name="stateOwnership"/> records who
        /// updates the state — the module's own forward logic
        /// (<see cref="StateOwnership.ModuleOwned"/>) or an optimizer module
        /// (<see cref="StateOwnership.OptimizerOwned"/>).
        /// </summary>
        public static Variable CallTrainableParamInitializer(Delegate trainableParamInitializerImplementation, string? defaultName, bool isTrainable, StateOwnership stateOwnership, params Variable[] inputs)
        {
            Vector<int64> iterationIndices = [.. LoopAPI.IterationIndices];

            var targetFn = ModuleHelper.CreateTargetFunction(trainableParamInitializerImplementation,
                isTrainableParamInitializer: isTrainable,
                isStateParamInitializer: !isTrainable,
                defaultName: defaultName,
                stateOwnership: stateOwnership);

            return InternalOp.TrainableParamRef(inputs, iterationIndices, localModelId: null, targetFn.Outputs[0].Type, targetFn.Outputs[0].Rank, targetFn, isTrainable);
        }

        /// <summary>
        /// Calls a trainable parameter initializer and returns a result with the specified generic type.
        /// This version handles generic type parameters properly by returning Tensor&lt;T&gt;.
        /// State parameters default to <see cref="StateOwnership.ModuleOwned"/>; use the overload
        /// taking a <see cref="StateOwnership"/> for optimizer-owned state.
        /// </summary>
        /// <typeparam name="T">The element type of the returned tensor</typeparam>
        /// <param name="trainableParamInitializerImplementation">The delegate for the initializer</param>
        /// <param name="defaultName">The default name for the trainable parameter</param>
        /// <param name="isTrainable">True for trainable parameters, false for state parameters</param>
        /// <param name="inputs">The input variables to the initializer</param>
        /// <returns>A tensor with the correct generic type</returns>
        public static Tensor<T> CallTrainableParamInitializer<T>(Delegate trainableParamInitializerImplementation, string? defaultName, bool isTrainable, params Variable[] inputs) where T : IVarType
            => CallTrainableParamInitializer<T>(trainableParamInitializerImplementation, defaultName, isTrainable, StateOwnership.ModuleOwned, inputs);

        /// <summary>
        /// Calls a trainable or state parameter initializer and returns a result with the specified
        /// generic type; for state parameters (<paramref name="isTrainable"/> false),
        /// <paramref name="stateOwnership"/> records who updates the state.
        /// </summary>
        public static Tensor<T> CallTrainableParamInitializer<T>(Delegate trainableParamInitializerImplementation, string? defaultName, bool isTrainable, StateOwnership stateOwnership, params Variable[] inputs) where T : IVarType
        {
            Vector<int64> iterationIndices = [.. LoopAPI.IterationIndices];

            var targetFn = ModuleHelper.CreateTargetFunction(trainableParamInitializerImplementation,
                isTrainableParamInitializer: isTrainable,
                isStateParamInitializer: !isTrainable,
                defaultName: defaultName,
                stateOwnership: stateOwnership);

            // Check if T is a generic standin type (IGenericType1, IGenericType2, etc.)
            // When called from within a module with a generic type, T will be a standin.
            // When called directly with a concrete type, T will be concrete (e.g., float32).
            var typeOfT = typeof(T);
            var isGenericStandin = typeOfT.IsAssignableTo(typeof(IGenericType));

            // For generic standins: use the target function's dtype (which is also a standin with GenericTypeParamName set)
            // For concrete types: use OnnxUtils.GetDType<T>() to get the concrete DType.
            // GetDType<T>() returns null for unknown types, so we fallback to targetFn's dtype.
            var dtype = OnnxUtils.GetDType<T>();
            var rank = targetFn.Outputs[0].Rank;


            // Build generic type args array (mirrors MODEL_INVOKE behavior).
            // When T is a generic standin, pass the dtype (which has GenericTypeParamName set)
            // so ProcessGenericSpecialization can update it to concrete type during specialization.
            DType[]? genericTypeArgs = null;
            if (isGenericStandin)
            {
                genericTypeArgs = [dtype];
            }

            // Create the trainable param ref with the appropriate dtype and generic type args
            var result = InternalOp.TrainableParamRef(inputs, iterationIndices, localModelId: null, dtype, rank, targetFn, isTrainable, genericTypeArgs);

            // The result is Variable but we know it's a tensor with the specified dtype.
            // Cast through Variable first (the interface), then to the concrete Tensor<T>.
            // This cast succeeds because TrainableParamRef creates a Variable with the correct dtype.
            return (Variable)result;
        }

        /// <summary>
        /// Guidance appended to every <see cref="InvalidStateUpdateException"/>: how to declare a
        /// state variable correctly.
        /// </summary>
        private const string StateUpdateGuidance =
            "StateUpdate may only target a state variable created by a [StateInitializer] class's " +
            "Init method (state variables are created by state initializers the same way trainable " +
            "parameters are created by [TrainableParamInitializer]s). Declare one, e.g.:\n" +
            "    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]  // OptimizerOwned for optimizer state\n" +
            "    public static partial class MyStateInit\n" +
            "    {\n" +
            "        public static Tensor<float32> Inline(Vector<int64> shape) => Globals.TensorFill(shape, 0.0f);\n" +
            "    }\n" +
            "then create the state inside the module body and register its update:\n" +
            "    var state = MyStateInit.Init(shape);   // in an optimizer: MyStateInit.Init(currentParam.ShapeTensor())\n" +
            "    Globals.StateUpdate(state, newValue);\n" +
            "State must not be declared as an Inline method parameter — method parameters are " +
            "runtime inputs, not state.";

        /// <summary>
        /// Registers an update relationship between an original state tensor and its updated value.
        /// This allows the system to track state updates and ensure they are not pruned during graph optimization.
        ///
        /// When called, this method:
        /// 1. Validates that <paramref name="originalState"/> is a state variable, i.e. was created
        ///    by a [StateInitializer] class's Init method (throws
        ///    <see cref="InvalidStateUpdateException"/> otherwise)
        /// 2. Creates a STATE_UPDATE_LINK node connecting the original and updated state tensors
        /// 3. Registers the update pair for later use when wrapping module outputs with WithStateDeps
        /// 4. Ensures that if the module output is used, the state update will be included in the graph
        /// </summary>
        /// <typeparam name="T">The tensor type (must implement Variable)</typeparam>
        /// <param name="originalState">The original state tensor from a state initializer</param>
        /// <param name="updatedState">The computed updated value for the state</param>
        /// <exception cref="InvalidStateUpdateException">
        /// <paramref name="originalState"/> is not a state variable (e.g. it is a module input or a
        /// trainable parameter). The message explains how to declare a state variable via a
        /// [StateInitializer] class.
        /// </exception>
        public static void StateUpdate<T>(T originalState, T updatedState) where T : IValue
        {
            // Unwrap the user-facing handles to their backing graph-side Variable nodes. A defaulted
            // handle materialises a default node, which then fails the state-variable check below (SU001).
            var originalVar = originalState.ToVariable();
            var updatedVar = updatedState.ToVariable();

            // Resolve the node that actually produced the state, tracing through the Identity
            // nodes that rank casts like .Vec() / .Scalar() insert.
            var producer = originalVar.OwningNode;
            while (producer.OpCode == OpCodes.IDENTITY
                   && producer.Inputs.Length > 0
                   && producer.Inputs[0] is Variable identityInput)
            {
                producer = identityInput.OwningNode;
            }

            var isParamNode = producer.OpCode
                is InternalOpCodes.TRAINABLE_PARAM_REF
                or InternalOpCodes.TRAINABLE_PARAM_MODEL_REF
                or InternalOpCodes.TRAINABLE_PARAM_ID_REF
                or InternalOpCodes.TRAINABLE_PARAM
                or InternalOpCodes.MODEL_PARAM_DATA;

            if (!isParamNode)
            {
                throw new InvalidStateUpdateException(ErrorCodes.SU001,
                    $"produced by a '{producer.OpCode}' node",
                    "the tensor was not created by a state initializer. " + StateUpdateGuidance);
            }

            var isTrainable = producer.Attributes.GetBoolVal(OnnxOpAttributeNames.ShrkAttrIsTrainable) ?? true;
            if (isTrainable)
            {
                throw new InvalidStateUpdateException(ErrorCodes.SU002,
                    $"produced by a '{producer.OpCode}' node",
                    "the tensor is a trainable parameter. Trainable parameters are updated by the " +
                    "optimizer through gradients, never via StateUpdate. If this tensor really is " +
                    "state, declare its initializer with [StateInitializer] instead of " +
                    "[TrainableParamInitializer]. " + StateUpdateGuidance);
            }

            // Register the state update, which creates the STATE_UPDATE_LINK node
            InternalGlobals.RegisterStateUpdate(originalVar, updatedVar);
        }
    }
}
