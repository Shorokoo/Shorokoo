using System.Collections.Generic;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    internal static partial class Definitions
    {
        private static List<NodeDefinitionMaker> GetInternalMakers() => [
            Op(AUTO_GRAD)
                .Any<FloatLike>("TInputs", minVariadicCount: 1)
                .Tensor<FloatLike>("TLoss")
                .Input("loss", "TLoss")
                .Input("inputs", "TInputs", rank: "R")
                .Output("inputGrads", "TInputs", rank: "R"),

            Op(MODEL_OPTIONAL_INPUT)
                .Optional<AnyLike>("T")
                .AttributeDType(AttrDtype, "T")
                .AttributeEnum<InputType>(ShrkAttrInputType, ["Hyperparam", "ReadyInput", "ModelInput", "GenericType"], defaultValue: "ReadyInput")
                .Output("modelInput", "T", rank: "R"),

            Op(MODEL_TENSOR_INPUT)

            .Tensor<AnyLike>("T")
                .AttributeDType(AttrDtype, "T")
                .AttributeLong(ShrkAttrRank, "R")
                .AttributeEnum<InputType>(ShrkAttrInputType, ["Hyperparam", "ReadyInput", "ModelInput", "GenericType"], defaultValue: "ReadyInput")
                // Optional: the [Hyper(defaultValue)] default for a defaulted hyperparameter input.
                .AttributeFloat(ShrkAttrDefaultValue)
                .Output("modelInput", "T", rank: "R"),

            Op(MODEL_SEQUENCE_INPUT)
                .Sequence<AnyLike>("T")
                .AttributeDType(AttrDtype, "T")
                .AttributeEnum<InputType>(ShrkAttrInputType, ["Hyperparam", "ReadyInput", "ModelInput", "GenericType"], defaultValue: "ReadyInput")
                .Output("modelInput", "T"),

            Op(GENERIC_TYPE_INPUT)
                .Tensor<AnyLike>("T", tracksModuleFn: true)
                .AttributeDType(AttrDtype, "T")
                .AttributeEnum<InputType>(ShrkAttrInputType, ["Hyperparam", "ReadyInput", "ModelInput", "GenericType"], defaultValue: "GenericType")
                .AttributeLong(ShrkAttrRank, "R")
                .AttributeStrings(ShrkAttrGenericTypeConstraints)
                .Output("genericTypeInfo", "T", rank: "R"),

            Op(MODEL_PARAM_DATA)
                .Tensor<AnyLike>("T")
                .AttributeTensor(ShrkAttrTensorData, "T", "R")
                .AttributeBool(ShrkAttrIsTrainable)
                .Output("modelParam", "T", rank: "R"),

            Op(MODULE_SET_HYPERPARAMS)
                .Tensor<IModuleVarType>("T1", tracksModuleFn: true)
                .Any<AnyLike>("T2", minVariadicCount: 0)
                .Tensor<IModelVarType>("T3")
                .Tensor<int64>("T4")
                .AttributeLongs(ShrkAttrLocalModelId)
                .Input("inputModule", "T1", rank: 0)
                .Input("iterationIndices", "T4", rank: 1)
                .Input("hyperParams", "T2")
                .Output("outputModule", "T3", rank: 0)
                .Code("{1:this}.SetHyperparams({2:ignore}{#:param})"),

            Op(NEW_MODEL_LIKE)
                .Tensor<IModelVarType>("T")
                .Input("inputModule", "T", rank: 0)
                .Output("outputModel", "T", rank: 0),

            Op(GET_MODEL_ID)
                .Tensor<IModelVarType>("T1")
                .Tensor<int64>("T2")
                .Input("inputModule", "T1", rank: 0)
                .Output("modelId", "T2", rank: 1),

            Op(MODEL_INVOKE)
                .Tensor<IModelVarType>("T1")
                .Any<AnyLike>("T2", minVariadicCount: 0)
                .VarType<AnyLike>("T3")
                .Variadic("T3", minCount: 1)
                .AttributeEnums<DataStructure>(ShrkAttrStructure, ["Tensor", "Optional", "Sequence"], structureDefName: "T3")
                .AttributeDTypes(ShrkAttrDtype, "T3")
                .AttributeDTypes(ShrkAttrGenericTypeArgs)
                .AttributeLongs(ShrkAttrRank, "R")
                .Input("inputModel", "T1", rank: 0)
                .Input("inputs", "T2")
                .Output("outputs", "T3", rank: "R"),

            Op(FUNCTION_INVOKE)
                .Any<AnyLike>("T2", minVariadicCount: 0)
                .VarType<AnyLike>("T3")
                .Variadic("T3", minCount: 1)
                .AttributeEnums<DataStructure>(ShrkAttrStructure, ["Tensor", "Optional", "Sequence"], structureDefName: "T3")
                .AttributeDTypes(ShrkAttrDtype, "T3")
                .AttributeDTypes(ShrkAttrGenericTypeArgs)
                .AttributeLongs(ShrkAttrRank, "R")
                .Input("inputs", "T2")
                .Output("outputs", "T3", rank: "R"),

            Op(SEQUENCE_CONCAT)
                .Sequence<AnyLike>("T")
                .Variadic("V", minCount: 1)
                .Input("input_sequences", ["T", "V"], "R1")
                .Output("output_sequence", ["T"], "R2"),

            Op(SEQUENCE_SLICE)
                .Sequence<AnyLike>("T")
                .Tensor<int64>("T2")
                .Input("input", "T")
                .Input("start", "T2", rank: 0)
                .Input("end", "T2", rank: 0)
                .Output("output", "T"),

            Op(SUBMODEL)
                .Tensor<IModelVarType>("T1")
                .Tensor<IModelVarType>("T3") // They have a different ModuleFn
                .Any<AnyLike>("T2", minVariadicCount: 0)
                .AttributeLongs(ShrkAttrRelativeModelId)
                .Input("inputModel", "T1", rank: 0)
                .Input("hyperParams", "T2")
                .Output("outputModel", "T3", rank: 0),

            Op(MODEL_HYPERPARAM)
                .Tensor<IModelVarType>("T1")
                .Tensor<AnyLike>("T2")
                .AttributeLong(ShrkAttrHyperparamIndex)
                .AttributeDType(ShrkAttrDtype, "T2")
                .AttributeLong(ShrkAttrRank, rank: "R")
                .AttributeEnums<DataStructure>(ShrkAttrStructure, ["Tensor", "Optional", "Sequence"], structureDefName: "T2")
                .Input("inputModel", "T1", rank: 0)
                .Output("weightTensor", "T2", rank: "R"),

            Op(TRAINABLE_PARAM)
                .Tensor<int64>("T1")
                .Tensor<AnyLike>("T2", minVariadicCount: 0)
                .Tensor<AnyLike>("T3")
                .AttributeLongs(ShrkAttrLocalModelId)
                .AttributeDType(ShrkAttrDtype, "T3")
                .AttributeLong(ShrkAttrRank, rank: "R")
                .AttributeLongs(ShrkAttrShape)
                .AttributeBool(ShrkAttrIsTrainable)
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .Input("initializerParams", "T2")
                .Output("weightTensor", "T3", rank: "R"),

            Op(TRAINABLE_PARAM_REF)
                .Tensor<int64>("T1")
                .Tensor<AnyLike>("T2", minVariadicCount: 0)
                .Tensor<AnyLike>("T3")
                .AttributeLongs(ShrkAttrLocalModelId)
                .AttributeDType(ShrkAttrDtype, "T3")
                .AttributeLong(ShrkAttrRank, rank: "R")
                .AttributeDTypes(ShrkAttrGenericTypeArgs)
                .AttributeBool(ShrkAttrIsTrainable)
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .Input("iterationIndices", "T1", rank: 1)
                .Input("initializerParams", "T2")
                .Output("weightTensor", "T3", rank: "R"),

            Op(TRAINABLE_PARAM_ID_REF)
                .Tensor<int64>("T1")
                .Tensor<AnyLike>("T2", minVariadicCount: 0)
                .Tensor<AnyLike>("T3")
                .AttributeDType(ShrkAttrDtype, "T3")
                .AttributeLong(ShrkAttrRank, rank: "R")
                .AttributeDTypes(ShrkAttrGenericTypeArgs)
                .AttributeBool(ShrkAttrIsTrainable)
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .Input("modelId", "T1", rank: 1)
                .Input("initializerParams", "T2")
                .Output("weightTensor", "T3", rank: "R"),

            Op(TRAINABLE_PARAM_MODEL_REF)
                .Tensor<IModelVarType>("T1")
                .Tensor<int64>("T4")
                .Tensor<AnyLike>("T2", minVariadicCount: 0)
                .Tensor<AnyLike>("T3")
                .AttributeLongs(ShrkAttrRelativeModelId)
                .AttributeDType(ShrkAttrDtype, "T3")
                .AttributeLong(ShrkAttrRank, rank: "R")
                .AttributeDTypes(ShrkAttrGenericTypeArgs)
                .AttributeBool(ShrkAttrIsTrainable)
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .Input("model", "T1?", rank: 0)
                .Input("iterationIndices", "T4", rank: 1)
                .Input("initializerParams", "T2")
                .Output("weightTensor", "T3", rank: "R"),

            Op(CREATE_MODULE)
                .Tensor<IModuleVarType>("T")
                .AttributeString(ShrkAttrFunctionName)
                .AttributeString(ShrkAttrDomainName)
                .AttributeDTypes(ShrkAttrGenericTypeArgs)
                .Output("outputModule", "T", rank: 0),

            // State management operators

            // STATE_UPDATE_LINK: Creates a traceable link between original and updated state in the graph.
            // Inputs: original state tensor, updated state tensor (both must be same type)
            // Outputs: the updated state tensor (pass-through)
            Op(STATE_UPDATE_LINK)
                .Tensor<AnyLike>("T")
                .Input("originalState", "T", rank: "R")
                .Input("updatedState", "T", rank: "R")
                .Output("linkedUpdatedState", "T", rank: "R"),

            // WITH_STATE_DEPS: Creates graph dependencies from updated state tensors to module output.
            // Inputs: main output tensor (first), variadic updated state tensors (rest)
            // Outputs: main output tensor only (pass-through)
            // This ensures state tensors are included when output is used.
            Op(WITH_STATE_DEPS)
                .Tensor<AnyLike>("T1")
                .Tensor<AnyLike>("T2", minVariadicCount: 0)
                .Input("mainOutput", "T1", rank: "R1")
                .Input("stateDeps", "T2", rank: "R2")
                .Output("output", "T1", rank: "R1"),

            // TensorStruct operations

            // MODEL_TENSORSTRUCT_INPUT: Creates a TensorStruct model input.
            // This is the input node for TensorStruct parameters, parallel to MODEL_TENSOR_INPUT.
            Op(MODEL_TENSORSTRUCT_INPUT)
                .TensorStruct<IStruct>("T")
                .AttributeDType(AttrDtype, "T")
                .AttributeEnum<InputType>(ShrkAttrInputType, ["Hyperparam", "ReadyInput", "ModelInput", "GenericType"], defaultValue: "ReadyInput")
                .Output("modelInput", "T"),

            // TENSOR_STRUCT_GETFIELD: Extracts a single field from a TensorStruct.
            // Input: A TensorStruct
            // Output: The field value (can be Tensor, Sequence, Optional, or nested TensorStruct)
            Op(TENSOR_STRUCT_GETFIELD)
                .TensorStruct<IStruct>("TStruct")
                .Any<AnyLike>("TField")
                .AttributeString(ShrkAttrFieldName)
                .AttributeDType(ShrkAttrDtype, "TField")
                .AttributeLong(ShrkAttrRank, rank: "R")
                .AttributeEnum<DataStructure>(ShrkAttrStructure, ["Tensor", "Optional", "Sequence", "TensorStruct"], structureDefName: "TField")
                .Input("structInput", "TStruct")
                .Output("fieldOutput", "TField", rank: "R"),

            // TENSOR_STRUCT_CREATE: Creates a TensorStruct from multiple input IValues.
            // Inputs: Variadic list of field values (order matches TensorStructDef)
            // Output: The created TensorStruct
            Op(TENSOR_STRUCT_CREATE)
                .TensorStruct<IStruct>("TStruct")
                .Any<AnyLike>("TFields", minVariadicCount: 1)
                .AttributeDType(AttrDtype, "TStruct")
                .Input("fields", "TFields")
                .Output("structOutput", "TStruct"),

            // SHRK_RANDOM_UNIFORM: a uniform runtime feed with dynamic shape input. An
            // id-bearing feed is wired at concretization to its SHRK_RNG_KEY_PARAM entity ("key"
            // input) and lowers to a keyed deterministic draw selecting its iteration's key
            // row; a feed without stream identity (e.g. inside an initializer function body)
            // lowers to ConstantOfShape + RandomUniformLike.
            Op(SHRK_RANDOM_UNIFORM)
                .Tensor<int64>("T1")
                .Tensor<float32>("T2")
                .AttributeFloat(AttrHigh)
                .AttributeFloat(AttrLow)
                .AttributeFloat(AttrSeed)
                .AttributeLongs(ShrkAttrLocalModelId)
                .Input("shape", "T1", 1)
                .Input("drawBase", "T1", 0)
                .Input("iterationIndices", "T1", 1)
                .Input("key", "T1?", 2)
                .Output("output", "T2", rank: "R"),

            // SHRK_RANDOM_NORMAL: the normal-distribution runtime feed; see SHRK_RANDOM_UNIFORM.
            Op(SHRK_RANDOM_NORMAL)
                .Tensor<int64>("T1")
                .Tensor<float32>("T2")
                .AttributeFloat(AttrMean)
                .AttributeFloat(AttrScale)
                .AttributeFloat(AttrSeed)
                .AttributeLongs(ShrkAttrLocalModelId)
                .Input("shape", "T1", 1)
                .Input("drawBase", "T1", 0)
                .Input("iterationIndices", "T1", 1)
                .Input("key", "T1?", 2)
                .Output("output", "T2", rank: "R"),

            // SHRK_RNG_KEY_VECTOR: the model's compact RNG key vector (int64, tiered — see
            // RngConfig.BuildKeyVector) plus the algorithm name: the recorded RNG identity.
            // Key entities (SHRK_RNG_KEY_PARAM) materialize their values from it at bind — or at
            // lowering, for a graph never bound (no config = the default identity). No inputs;
            // lowered to a plain CONSTANT at ONNX prep so every backend treats it as ordinary
            // (unused) data.
            Op(SHRK_RNG_KEY_VECTOR)
                .Tensor<int64>("T1")
                .AttributeTensor(AttrValue, "T1", "R")
                .AttributeString(ShrkAttrRngAlgorithm)
                .Output("keys", "T1", 1),

            // SHRK_RNG_KEY_PARAM: a feed site's key entity — the param-like carrier of the site's
            // realized stream set (site ModelId + realized stream ids + iteration counts),
            // whose value ([N, 2] int64 key table over the site's dense iteration grid) is
            // materialized from the bound RngConfig at concrete-model time and re-materialized
            // on re-bind — exactly as a trainable parameter's value comes from running its
            // initializer. Value absent until materialization. Feeds select their iteration's
            // row by runtime index; lowered to a plain CONSTANT at ONNX prep.
            Op(SHRK_RNG_KEY_PARAM)
                .Tensor<int64>("T1")
                .AttributeTensor(AttrValue, "T1", "R")
                .AttributeLongs(ShrkAttrLocalModelId)
                .AttributeLongs(ShrkAttrRngRealizedIds)
                .AttributeLongs(ShrkAttrRngIterCounts)
                .Output("keys", "T1", 2),

            // SHRK_RNG_SPLIT: index-based RNG key split, child = Bijection(key, counter: index)
            // under the named algorithm. Key = int64[2] (32-bit words). Lowered at ONNX export
            // to a call of the algorithm's non-inlined "split" function; QEE computes it host-side.
            Op(SHRK_RNG_SPLIT)
                .Tensor<int64>("T1")
                .AttributeString(ShrkAttrRngAlgorithm)
                .Input("key", "T1", 1)
                .Input("index", "T1", 0)
                .Output("childKey", "T1", 1),

            // SHRK_RNG_UNIFORM: keyed deterministic U(low, high) draw of dynamic shape under
            // the named algorithm. Counter = (flat element index, drawBase). Lowered at ONNX
            // export to a call of the algorithm's non-inlined "uniform" function.
            Op(SHRK_RNG_UNIFORM)
                .Tensor<int64>("T1")
                .Tensor<float32>("T2")
                .AttributeString(ShrkAttrRngAlgorithm)
                .Input("key", "T1", 1)
                .Input("drawBase", "T1", 0)
                .Input("shape", "T1", 1)
                .Input("low", "T2", 0)
                .Input("high", "T2", 0)
                .Output("output", "T2", rank: "R"),

            // SHRK_RNG_NORMAL: keyed deterministic N(mean, scale) draw of dynamic shape under
            // the named algorithm (per-element-pair Box-Muller). See SHRK_RNG_UNIFORM.
            Op(SHRK_RNG_NORMAL)
                .Tensor<int64>("T1")
                .Tensor<float32>("T2")
                .AttributeString(ShrkAttrRngAlgorithm)
                .Input("key", "T1", 1)
                .Input("drawBase", "T1", 0)
                .Input("shape", "T1", 1)
                .Input("mean", "T2", 0)
                .Input("scale", "T2", 0)
                .Output("output", "T2", rank: "R"),

            // SHRK_CONV: Conv variant taking geometry (pads/strides/dilations/kernel_shape/group)
            // as int64 tensor inputs instead of static attributes. Lowered to standard ONNX Conv
            // (with those resolved to static attributes) by FastLowerAttributeTensorOps.
            Op(SHRK_CONV)
                .Tensor<FloatLike>("T")
                .Tensor<int64>("TI")
                .AttributeEnum<AutoPad>(AttrAutoPad, ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .Input("X", "T", "R")
                .Input("W", "T", "R2")
                .Input("B", "T?", 1)
                .Input("pads", "TI", 1)
                .Input("strides", "TI", 1)
                .Input("dilations", "TI", 1)
                .Input("kernel_shape", "TI", 1)
                .Input("group", "TI", 0)
                .Output("Y", "T", "R"),
        ];
    }
}