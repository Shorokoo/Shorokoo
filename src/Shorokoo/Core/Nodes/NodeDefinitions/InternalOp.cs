using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using System.Collections.Immutable;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.InternalOpCodes;
using System.Diagnostics;
using System;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

internal static partial class InternalOp
{
    public static Variable?[] AutoGrad(Variable?[] inputs, Variable loss)
        => NodeBuilder.BuildNodeMultiOut(AUTO_GRAD, [loss, ..inputs], []);

    public static Variable RuntimeInput(DType dtype, int? rank, string? defaultName = null, Function? moduleFn = null)
        => NodeBuilder.BuildNodeSingleOut(MODEL_TENSOR_INPUT, [], [(AttrDtype, dtype), (ShrkAttrRank, (long?)rank)], outputNames: defaultName is null ? null : [defaultName], targetFunction: moduleFn);

    public static Variable ModuleTensorInput(DType dtype, int? rank, InputType? inputType, Function? targetFunction, string? defaultName = null, float? defaultValue = null)
    {
        Debug.Assert(targetFunction is null || dtype == DType.Model || dtype == DType.Module || dtype == DType.Int64);

        var attributes = new List<(string, object?)>
        {
            (AttrDtype, dtype),
            (ShrkAttrInputType, inputType),
            (ShrkAttrRank, (long?)rank)
        };

        // A [Hyper(defaultValue)] default is recorded as declarative metadata so it survives
        // serialization; only present when the parameter actually declared a default.
        if (defaultValue is float dv)
            attributes.Add((ShrkAttrDefaultValue, dv));

        return NodeBuilder.BuildNodeSingleOut(MODEL_TENSOR_INPUT, [], [.. attributes], outputNames: defaultName is null ? null : [defaultName], targetFunction: targetFunction);
    }

    public static Variable ModuleOptionalInput(DType dtype, InputType? inputType, Function? targetFunction, string? defaultName = null)
    {
        Debug.Assert(targetFunction is not null || (dtype != DType.Model && dtype != DType.Module));
        
        var attributes = new List<(string, object?)>
        {
            (AttrDtype, dtype),
            (ShrkAttrInputType, inputType)
        };
        
        return NodeBuilder.BuildNodeSingleOut(MODEL_OPTIONAL_INPUT, [], [.. attributes], outputNames: defaultName is null ? null : [defaultName], targetFunction: targetFunction);
    }

    public static Variable ModuleSequenceInput(DType dtype, InputType? inputType, Function? targetFunction, string? defaultName = null)
    {
        Debug.Assert(targetFunction is not null || (dtype != DType.Model && dtype != DType.Module));
        
        var attributes = new List<(string, object?)>
        {
            (AttrDtype, dtype),
            (ShrkAttrInputType, inputType)
        };
        
        return NodeBuilder.BuildNodeSingleOut(MODEL_SEQUENCE_INPUT, [], [.. attributes], outputNames: defaultName is null ? null : [defaultName], targetFunction: targetFunction);
    }

    public static Variable GenericTypeInput(DType dtype, int? rank, string[]? constraints = null, string? defaultName = null)
    {
        var attributes = new List<(string, object?)>
        {
            (AttrDtype, dtype),
            (ShrkAttrInputType, InputType.GenericType),
            (ShrkAttrRank, (long?)rank)
        };
        
        // Add constraints if provided
        if (constraints != null)
            attributes.Add((ShrkAttrGenericTypeConstraints, constraints));
        
        return NodeBuilder.BuildNodeSingleOut(GENERIC_TYPE_INPUT, [], [.. attributes], outputNames: defaultName is null ? null : [defaultName]);
    }

    public static Variable ModelParamData(TensorData data, bool isTrainable, string? identifierTemplateString, string? defaultName)
        => NodeBuilder.BuildNodeSingleOut(MODEL_PARAM_DATA, [], [(ShrkAttrTensorData, data), (ShrkAttrIsTrainable, isTrainable)], identifierTemplateString: identifierTemplateString, outputNames: defaultName is null ? null : [defaultName]);
    public static Variable ModuleSetHyperparams(Variable inputModule, Variable?[] moduleParams, Variable? iterationIndices, int[]? localModelId, string? identifierTemplateString)
        => NodeBuilder.BuildNodeSingleOut(MODULE_SET_HYPERPARAMS, [inputModule, iterationIndices, .. moduleParams], [(ShrkAttrLocalModelId, localModelId)], identifierTemplateString: identifierTemplateString);

    public static Variable GetModelId(Variable inputModule)
        => NodeBuilder.BuildNodeSingleOut(GET_MODEL_ID, [inputModule], []);

    public static Variable CreateModule(Function targetFunction, DType[]? genericTypeArgs = null)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrFunctionName, targetFunction.DefaultName),
            (ShrkAttrDomainName, "Functions")
        };

        // Add generic type arguments if provided
        if (genericTypeArgs != null && genericTypeArgs.Length > 0)
        {
            attributes.Add((ShrkAttrGenericTypeArgs, genericTypeArgs));
        }

        return NodeBuilder.BuildNodeSingleOut(InternalOpCodes.CREATE_MODULE, [], [.. attributes], targetFunction: targetFunction);
    }

    public static Variable NodeModelLike(Variable inputModule)
        => NodeBuilder.BuildNodeSingleOut(NEW_MODEL_LIKE, [inputModule], []);

    public static Variable[] ModelInvoke(Variable inputModule, Variable?[] inputs, DataStructure[] dataStructures, DType[] dtypes, int[] ranks, DType[]? genericTypeArgs = null)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrStructure, dataStructures),
            (ShrkAttrDtype, dtypes),
            (ShrkAttrRank, ranks)
        };
        
        // Add generic type arguments if provided
        if (genericTypeArgs != null && genericTypeArgs.Length > 0)
        {
            attributes.Add((ShrkAttrGenericTypeArgs, genericTypeArgs));
        }
        
        return NodeBuilder.BuildNodeMultiOut(MODEL_INVOKE, [inputModule, .. inputs], [.. attributes]);
    }

    public static Variable[] FunctionInvoke(Variable?[] inputs, DataStructure[] dataStructures, DType[] dtypes, int[] ranks, Function targetFn, DType[]? genericTypeArgs = null)
     => NodeBuilder.BuildNodeMultiOut(
                    FUNCTION_INVOKE, 
                    [ .. inputs], 
                        [(ShrkAttrStructure, dataStructures),
                         (ShrkAttrDtype, dtypes),
                         (ShrkAttrRank, ranks),
                         (ShrkAttrGenericTypeArgs, genericTypeArgs)],
                        targetFunction: targetFn);

    public static Variable SequenceConcat(Variable[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONCAT, inputs, []);

    public static Variable SequenceSlice(Variable input_sequence, Variable start, Variable end)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_SLICE, [input_sequence, start, end], []);

    public static Variable SubModel(Variable inputModule, Variable[] hyperparams, int[] relativeModelId, Function fnModule)
        => NodeBuilder.BuildNodeSingleOut(SUBMODEL, [inputModule], [(ShrkAttrRelativeModelId, relativeModelId)], targetFunction: fnModule);

    public static Variable ModelHyperparam(Variable inputModel, int hyperparamIndex, DType dtype, int? rank)
        => NodeBuilder.BuildNodeSingleOut(MODEL_HYPERPARAM, [inputModel], [(ShrkAttrHyperparamIndex, hyperparamIndex), (ShrkAttrDtype, dtype), (ShrkAttrRank, rank)]);

    public static Variable TrainableParamRef(Variable? model, Variable[] initializerParams, Variable? iterationIndices, int[] modelId, DType dtype, int? rank, Function initializerFn, bool isTrainable, DType[]? genericTypeArgs = null)
    {
        if (model is null)
            return TrainableParamRef(initializerParams, iterationIndices, modelId, dtype, rank, initializerFn, isTrainable, genericTypeArgs);

        if (model.Type == DType.Model)
            return TrainableParamModelRef(model, initializerParams, iterationIndices, modelId, dtype, rank, initializerFn, isTrainable, genericTypeArgs);

        Debug.Assert(model.Type == DType.Int64);
        return TrainableParamIdRef(model, initializerParams, iterationIndices, modelId, dtype, rank, initializerFn, isTrainable, genericTypeArgs);
    }

    public static Variable TrainableParam(Variable[] initializerParams, int[]? localModelId, DType dtype, int? rank, Shape shape, Function initializerFn, bool isTrainable, string? identifierTemplateString = null)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrLocalModelId, localModelId),
            (ShrkAttrDtype, dtype),
            (ShrkAttrRank, rank),
            (ShrkAttrShape, shape.Dims),
            (ShrkAttrIsTrainable, isTrainable),
            (ShrkAttrFunctionName, initializerFn.DefaultName),
            (ShrkAttrDomainName, "Functions")
        };

        return NodeBuilder.BuildNodeSingleOut(TRAINABLE_PARAM, [.. initializerParams], [.. attributes], targetFunction: initializerFn, identifierTemplateString: identifierTemplateString);
    }

    public static Variable TrainableParamRef(Variable[] initializerParams, Variable? iterationIndices, int[]? localModelId, DType dtype, int? rank, Function initializerFn, bool isTrainable, DType[]? genericTypeArgs = null, string? identifierTemplateString = null)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrLocalModelId, localModelId),
            (ShrkAttrDtype, dtype),
            (ShrkAttrRank, rank),
            (ShrkAttrIsTrainable, isTrainable),
            (ShrkAttrFunctionName, initializerFn.DefaultName),
            (ShrkAttrDomainName, "Functions")
        };

        // Add generic type arguments if provided (mirrors MODEL_INVOKE behavior)
        if (genericTypeArgs != null && genericTypeArgs.Length > 0)
        {
            attributes.Add((ShrkAttrGenericTypeArgs, genericTypeArgs));
        }

        return NodeBuilder.BuildNodeSingleOut(TRAINABLE_PARAM_REF, [iterationIndices, .. initializerParams], [.. attributes], targetFunction: initializerFn, identifierTemplateString: identifierTemplateString);
    }

    public static Variable TrainableParamModelRef(Variable model, Variable[] initializerParams, Variable? iterationIndices, int[] relativeModelId, DType dtype, int? rank, Function initializerFn, bool isTrainable, DType[]? genericTypeArgs = null, string? identifierTemplateString = null)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrRelativeModelId, relativeModelId),
            (ShrkAttrDtype, dtype),
            (ShrkAttrRank, rank),
            (ShrkAttrIsTrainable, isTrainable),
            (ShrkAttrFunctionName, initializerFn.DefaultName),
            (ShrkAttrDomainName, "Functions")
        };

        // Add generic type arguments if provided (mirrors MODEL_INVOKE behavior)
        if (genericTypeArgs != null && genericTypeArgs.Length > 0)
        {
            attributes.Add((ShrkAttrGenericTypeArgs, genericTypeArgs));
        }

        return NodeBuilder.BuildNodeSingleOut(TRAINABLE_PARAM_MODEL_REF, [model, iterationIndices, .. initializerParams], [.. attributes], targetFunction: initializerFn, identifierTemplateString: identifierTemplateString);
    }

    public static Variable TrainableParamIdRef(Variable modelIndexId, Variable[] initializerParams, Variable? iterationIndices, int[] relativeModelId, DType dtype, int? rank, Function initializerFn, bool isTrainable, DType[]? genericTypeArgs = null, string? identifierTemplateString = null)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrRelativeModelId, relativeModelId),
            (ShrkAttrDtype, dtype),
            (ShrkAttrRank, rank),
            (ShrkAttrIsTrainable, isTrainable),
            (ShrkAttrFunctionName, initializerFn.DefaultName),
            (ShrkAttrDomainName, "Functions")
        };

        // Add generic type arguments if provided (mirrors MODEL_INVOKE behavior)
        if (genericTypeArgs != null && genericTypeArgs.Length > 0)
        {
            attributes.Add((ShrkAttrGenericTypeArgs, genericTypeArgs));
        }

        return NodeBuilder.BuildNodeSingleOut(TRAINABLE_PARAM_ID_REF, [modelIndexId, iterationIndices, ..initializerParams], [.. attributes], targetFunction: initializerFn, identifierTemplateString: identifierTemplateString);
    }

    /// <summary>
    /// Creates a STATE_UPDATE_LINK node that marks the relationship between original and updated state tensors.
    /// This creates a traceable link in the graph that can be used during graph lowering.
    /// </summary>
    /// <param name="originalState">The original state tensor from a state initializer</param>
    /// <param name="updatedState">The computed updated value for the state</param>
    /// <returns>The updated state tensor (pass-through)</returns>
    public static Variable StateUpdateLink(Variable originalState, Variable updatedState)
        => NodeBuilder.BuildNodeSingleOut(STATE_UPDATE_LINK, [originalState, updatedState], []);

    /// <summary>
    /// Creates a WITH_STATE_DEPS node that creates explicit graph dependencies from updated state tensors to the output.
    /// This ensures that if the module output is part of the operation graph for any whole-graph output,
    /// then all updated state tensors are also included in the graph.
    /// </summary>
    /// <param name="mainOutput">The main output tensor (value to pass through)</param>
    /// <param name="stateDeps">Updated state tensors that should be linked to this output</param>
    /// <returns>The main output tensor (pass-through semantics)</returns>
    public static Variable WithStateDeps(Variable mainOutput, params Variable[] stateDeps)
        => NodeBuilder.BuildNodeSingleOut(WITH_STATE_DEPS, [mainOutput, ..stateDeps], []);

    // TensorStruct operations

    /// <summary>
    /// Creates a TensorStruct input node for module inputs.
    /// </summary>
    /// <param name="dtype">The DType of the TensorStruct (must be a TensorStruct type)</param>
    /// <param name="inputType">The type of input (Hyperparam, ReadyInput, ModelInput)</param>
    /// <param name="targetFunction">Optional target function for the input</param>
    /// <param name="defaultName">Optional default name for the input</param>
    /// <returns>The TensorStruct input variable</returns>
    public static Variable TensorStructInput(DType dtype, InputType? inputType = null, Function? targetFunction = null, string? defaultName = null)
    {
        var attributes = new List<(string, object?)>
        {
            (AttrDtype, dtype),
            (ShrkAttrInputType, inputType)
        };
        
        return NodeBuilder.BuildNodeSingleOut(MODEL_TENSORSTRUCT_INPUT, [], [.. attributes], 
            outputNames: defaultName is null ? null : [defaultName], targetFunction: targetFunction);
    }

    /// <summary>
    /// Extracts a field from a TensorStruct by field name.
    /// </summary>
    /// <param name="structInput">The TensorStruct to extract the field from</param>
    /// <param name="fieldName">The name of the field to extract</param>
    /// <param name="fieldDType">The DType of the field</param>
    /// <param name="fieldRank">The rank of the field (for tensor fields)</param>
    /// <param name="fieldStructure">The structure of the field (Tensor, Sequence, Optional, TensorStruct)</param>
    /// <returns>The extracted field variable</returns>
    public static Variable TensorStructGetField(Variable structInput, string fieldName, DType fieldDType, int? fieldRank, DataStructure fieldStructure)
    {
        var attributes = new List<(string, object?)>
        {
            (ShrkAttrFieldName, fieldName),
            (ShrkAttrDtype, fieldDType),
            (ShrkAttrRank, (long?)fieldRank),
            (ShrkAttrStructure, fieldStructure)
        };
        
        return NodeBuilder.BuildNodeSingleOut(TENSOR_STRUCT_GETFIELD, [structInput], [.. attributes]);
    }

    /// <summary>
    /// Creates a TensorStruct from multiple field values.
    /// Field values must be ordered to match the fields in the TensorStructDef of structDType.
    /// </summary>
    /// <param name="structDType">The DType of the TensorStruct to create (must contain TensorStructDef)</param>
    /// <param name="fieldValues">The values for each field, ordered to match TensorStructDef.Fields</param>
    /// <returns>The created TensorStruct variable</returns>
    public static Variable TensorStructCreate(DType structDType, Variable[] fieldValues)
    {
        var attributes = new List<(string, object?)>
        {
            (AttrDtype, structDType),
        };

        return NodeBuilder.BuildNodeSingleOut(TENSOR_STRUCT_CREATE, fieldValues, [.. attributes]);
    }

    /// <summary>
    /// Creates a tensor filled with random values from a uniform distribution in [low, high).
    /// Takes shape as a tensor input (dynamic shape support).
    /// Lowered to ONNX ConstantOfShape + RandomUniformLike before execution.
    /// </summary>
    public static Variable RandomUniform(Variable shape, float? high = null, float? low = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(SHRK_RANDOM_UNIFORM, [shape], [
            (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    /// <summary>
    /// Creates a tensor filled with random values from a normal distribution.
    /// Takes shape as a tensor input (dynamic shape support).
    /// Lowered to ONNX ConstantOfShape + RandomNormalLike before execution.
    /// </summary>
    public static Variable RandomNormal(Variable shape, float? mean = null, float? scale = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(SHRK_RANDOM_NORMAL, [shape], [
            (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    /// <summary>
    /// Conv variant whose geometry (pads, strides, dilations, kernel_shape, group) is supplied
    /// as int64 tensor inputs rather than static attributes, so it can be computed in-graph.
    /// Lowered to standard ONNX Conv (geometry resolved to static attributes) by
    /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastLowerAttributeTensorOps"/>. The geometry
    /// inputs must be resolvable to constants (directly, via constant folding, or from sample
    /// inputs) at lowering time. <paramref name="autoPad"/> stays a static attribute.
    /// </summary>
    public static Variable Conv(Variable x, Variable w, Variable b, AutoPad autoPad,
        Variable pads, Variable strides, Variable dilations, Variable kernelShape, Variable group)
        => NodeBuilder.BuildNodeSingleOut(SHRK_CONV, [x, w, b, pads, strides, dilations, kernelShape, group],
            [(AttrAutoPad, autoPad)]);
}