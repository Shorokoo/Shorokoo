using Shorokoo.Graph;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Diagnostics;
using static Shorokoo.Globals;
using Shorokoo;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Training;

namespace Shorokoo.Core
{
    /// <summary>
    /// Internal base types for the module system: the <c>IModel</c>/<c>IModule</c> interfaces,
    /// the <c>BaseModel&lt;&gt;</c> abstract bases that the user-facing <c>Model&lt;&gt;</c>
    /// wrappers in <c>Shorokoo.Modules</c> extend, and the <c>CallbackModule&lt;&gt;</c> /
    /// <c>Module&lt;&gt;</c> base classes consumed by source-generated module code. Lives in
    /// <c>Shorokoo.Core</c> because module authors never name these directly — they're framework
    /// plumbing.
    /// </summary>
    public interface IModel : IModuleParam
    {
        Scalar<IModelVarType> ModelVariable { get; }
    }

    /// <summary>Model-level helpers usable on any concrete <see cref="IModel"/>.</summary>
    public static class ModelExtensions
    {
        /// <summary>
        /// References an existing trainable parameter of <paramref name="model"/> by its relative
        /// <paramref name="relativeModelId"/> path, typed <typeparamref name="T"/> with the given
        /// <paramref name="rank"/>. The path is the chain of relative ModelIds from this model down
        /// to the parameter; parameters and sub-models are numbered from <c>1</c> in the order they
        /// are created within a model's body. So <c>[1]</c> is the model's own first parameter, and
        /// <c>[1, 1]</c> is the first parameter of the first sub-model it creates (had the model
        /// created a parameter of its own before that sub-model, the sub-model would be <c>[2]</c>
        /// and its first parameter <c>[2, 1]</c>). Unlike re-invoking an initializer, this points at
        /// the *same* parameter the model uses, so a reference computation built from it matches the
        /// model's own forward exactly — independent of how initialization is keyed. A low-level
        /// escape hatch (you count ids by hand), for tests and tooling that need realized weights.
        /// </summary>
        public static Tensor<T> GetTrainableParam<T>(this IModel model, int[] relativeModelId, int rank)
            where T : IVarType
        {
            Variable modelVar = model.ModelVariable;
            // The module function lives on the upstream CREATE_MODULE node; the model variable
            // itself is produced by MODULE_SET_HYPERPARAMS. Walk the producer chain to find it.
            var producer = modelVar.OwningNode;
            while (producer.TargetFunction is null && producer.Inputs.Length > 0
                   && producer.Inputs[0] is Variable inp)
                producer = inp.OwningNode;
            var moduleFn = producer.TargetFunction
                ?? throw new System.InvalidOperationException(
                    "Model variable has no target function; cannot reference its trainable parameters.");
            // The ref node must carry a trainable-param-initializer function (it is matched to the
            // model's actual parameter by ModelId during lowering; the initializer is metadata).
            var initializerFn = System.Linq.Enumerable.FirstOrDefault(
                    moduleFn.ReferencedFunctions,
                    f => f.FunctionType == Shorokoo.Core.Nodes.OnnxNodes.FunctionType.TrainableParamInitializer)
                ?? throw new System.InvalidOperationException(
                    "Model has no trainable-parameter initializers to reference.");
            // The reference's identifier name must be unique per parameter (distinct model ids
            // must not collapse to one canonical name), so encode the id path into the name.
            var template = ModelParamIdentifierTemplate
                .LocalTrainableParam(new ModelId(relativeModelId),
                    "ParamRef_" + string.Join("_", relativeModelId), 0,
                    System.Collections.Immutable.ImmutableArray<int>.Empty)
                .ToString();
            var dtype = OnnxUtils.GetDType(typeof(T))
                ?? throw new System.InvalidOperationException($"No DType for type {typeof(T).Name}.");
            var v = InternalOp.TrainableParamModelRef(
                modelVar, initializerParams: [], iterationIndices: null,
                relativeModelId: relativeModelId, dtype: dtype,
                rank: rank, initializerFn: initializerFn, isTrainable: true,
                genericTypeArgs: null, identifierTemplateString: template);
            return (Tensor<T>)v;
        }
    }

    internal interface IModule : IModuleParam
    {
        Scalar<IModuleVarType> ModuleVariable { get; }
    }

    public abstract class BaseModel<Tout> : IModel
    {
        Scalar<IModelVarType> IModel.ModelVariable => this.ModelVariable;

        private Scalar<IModelVarType> ModelVariable { get; set; }
        internal DType[]? GenericTypeArgs { get; set; }

        public BaseModel(InputType inputType)
        {
            var targetFunction = ModuleHelper.CreateFunctionSignature([], [], [typeof(Tout)]);
            this.ModelVariable = InternalOp.ModuleTensorInput(DType.Model, rank: 0, inputType, targetFunction, null);
        }

        public BaseModel(Scalar<IModelVarType> modelVariable, DType[]? genericTypeArgs = null)
        {
            this.ModelVariable = modelVariable;
            this.GenericTypeArgs = genericTypeArgs;
        }

        protected Tout internalCall()
        {
            (var structures, var dtypes, var ranks) = ModuleHelper.InfosFromTouts<Tout>();
            var retvals = InternalOp.ModelInvoke(this.ModelVariable, [], structures, dtypes, ranks, this.GenericTypeArgs);
            return ModuleHelper.Reformat<Tout>(retvals);
        }
    }

    public abstract class BaseModel<Tin, Tout> : IModel
    {
        Scalar<IModelVarType> IModel.ModelVariable => this.ModelVariable;

        private Scalar<IModelVarType> ModelVariable { get; set; } = default!;
        internal DType[]? GenericTypeArgs { get; set; }

        public BaseModel(InputType inputType)
        {
            var targetFunction = ModuleHelper.CreateFunctionSignature([], [typeof(Tin)], [typeof(Tout)]);
            this.ModelVariable = InternalOp.ModuleTensorInput(DType.Model, rank: 0, inputType, targetFunction, null);
        }

        public BaseModel(Scalar<IModelVarType> modelVariable, DType[]? genericTypeArgs = null)
        {
            this.ModelVariable = modelVariable;
            this.GenericTypeArgs = genericTypeArgs;
        }

        protected Tout internalCall(Tin inputs)
        {
            var inputVariables = ModuleHelper.Format(inputs);
            (var structures, var dtypes, var ranks) = ModuleHelper.InfosFromTouts<Tout>();
            var retvals = InternalOp.ModelInvoke(this.ModelVariable, inputVariables, structures, dtypes, ranks, this.GenericTypeArgs);
            return ModuleHelper.Reformat<Tout>(retvals);
        }
    }

    public class CallbackModule<TOutputs> : IModule
    {
        public Scalar<IModuleVarType> ModuleVariable { get; private set; }

        public CallbackModule(InputType inputType)
        {
            var targetFunction = ModuleHelper.CreateFunctionSignature([], [], [typeof(TOutputs)]);
            this.ModuleVariable = InternalOp.ModuleTensorInput(DType.Module, 0, inputType, targetFunction, null);
        }

        public CallbackModule(Func<TOutputs> fnModule, Delegate? referenceMethod = null, string? name = null)
        {
            var targetFunction = ModuleHelper.CreateTargetFunction(fnModule, referenceMethod: referenceMethod?.Method, defaultName: name, invokeTarget: referenceMethod?.Target);
            this.ModuleVariable = ModuleHelper.CreateModule(targetFunction, this.GetType());
        }

        public Model<TOutputs> SetHyperparams()
        {
            return this.SetHyperparams<Model<TOutputs>>();
        }

        public T SetHyperparams<T>() where T : BaseModel<TOutputs>
        {
            Vector<int64> iterationIndices = [.. LoopAPI.IterationIndices];
            Scalar<IModelVarType> modelVariable = InternalOp.ModuleSetHyperparams(this.ModuleVariable, [], iterationIndices, localModelId: null, identifierTemplateString: null);

            var genericTypeArgs = ModuleHelper.ExtractGenericTypeArgsFromType(this.GetType());

            var modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>), typeof(DType[]) });
            if (modelVariableConstructor == null)
            {
                modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>) });
                var model = (T)modelVariableConstructor.AssertNotNull().Invoke(new object[] { modelVariable });
                model.GenericTypeArgs = genericTypeArgs;
                return model;
            }
            return (T)modelVariableConstructor.Invoke(new object?[] { modelVariable, genericTypeArgs });
        }
    }

    public class CallbackModule<THyperparams, TOutputs> : IModule
    {
        public Scalar<IModuleVarType> ModuleVariable { get; private set; } = default!;

        public CallbackModule(InputType inputType)
        {
            var targetFunction = ModuleHelper.CreateFunctionSignature([typeof(THyperparams)], [], [typeof(TOutputs)]);
            this.ModuleVariable = InternalOp.ModuleTensorInput(DType.Module, 0, inputType, targetFunction, null);
        }

        public CallbackModule(Func<THyperparams, TOutputs>? fnModule, Delegate? referenceMethod = null, string? name = null)
        {
            var targetFunction = ModuleHelper.CreateTargetFunction(fnModule, referenceMethod: referenceMethod?.Method, defaultName: name, invokeTarget: referenceMethod?.Target);
            this.ModuleVariable = ModuleHelper.CreateModule(targetFunction, this.GetType());
        }

        public Model<TOutputs> SetHyperparams(THyperparams hyperparams)
        {
            return this.SetHyperparams<Model<TOutputs>>(hyperparams);
        }

        public T SetHyperparams<T>(THyperparams hyperparams) where T : BaseModel<TOutputs>
        {
            Vector<int64> iterationIndices = [.. LoopAPI.IterationIndices];
            var inputVariables = ModuleHelper.Format(hyperparams);
            Scalar<IModelVarType> modelVariable = InternalOp.ModuleSetHyperparams(this.ModuleVariable, inputVariables, iterationIndices, localModelId: null, identifierTemplateString: null);

            var genericTypeArgs = ModuleHelper.ExtractGenericTypeArgsFromType(this.GetType());

            var modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>), typeof(DType[]) });
            if (modelVariableConstructor == null)
            {
                modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>) });
                var model = (T)modelVariableConstructor.AssertNotNull().Invoke(new object[] { modelVariable });
                model.GenericTypeArgs = genericTypeArgs;
                return model;
            }
            return (T)modelVariableConstructor.Invoke(new object?[] { modelVariable, genericTypeArgs });
        }
    }

    public class Module<TInputs, TOutputs> : IModule
    {
        public Scalar<IModuleVarType> ModuleVariable { get; private set; } = default!;

        public Module(InputType inputType)
        {
            var targetFunction = ModuleHelper.CreateFunctionSignature([], [typeof(TInputs)], [typeof(TOutputs)]);
            this.ModuleVariable = InternalOp.ModuleTensorInput(DType.Module, 0, inputType, targetFunction, null);
        }

        public Module(Func<TInputs, TOutputs>? fnModule, Delegate? referenceMethod = null, string? name = null)
        {
            var targetFunction = ModuleHelper.CreateTargetFunction(fnModule, referenceMethod: referenceMethod?.Method, defaultName: name, invokeTarget: referenceMethod?.Target);
            this.ModuleVariable = ModuleHelper.CreateModule(targetFunction, this.GetType());
        }

        public Model<TInputs, TOutputs> SetHyperparams()
        {
            return this.SetHyperparams<Model<TInputs, TOutputs>>();
        }

        public T SetHyperparams<T>() where T : BaseModel<TInputs, TOutputs>
        {
            Vector<int64> iterationIndices = [.. LoopAPI.IterationIndices];
            Scalar<IModelVarType> modelVariable = InternalOp.ModuleSetHyperparams(this.ModuleVariable, [], iterationIndices, localModelId: null, identifierTemplateString: null);

            var genericTypeArgs = ModuleHelper.ExtractGenericTypeArgsFromType(this.GetType());

            var modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>), typeof(DType[]) });
            if (modelVariableConstructor == null)
            {
                modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>) });
                var model = (T)modelVariableConstructor.AssertNotNull().Invoke(new object[] { modelVariable });
                model.GenericTypeArgs = genericTypeArgs;
                return model;
            }
            return (T)modelVariableConstructor.Invoke(new object?[] { modelVariable, genericTypeArgs });
        }
    }

    public class Module<THyperparams, TInputs, TOutputs> : IModule
    {
        public Scalar<IModuleVarType> ModuleVariable { get; private set; } = default!;

        public Module(InputType inputType)
        {
            var targetFunction = ModuleHelper.CreateFunctionSignature([typeof(THyperparams)], [typeof(TInputs)], [typeof(TOutputs)]);
            this.ModuleVariable = InternalOp.ModuleTensorInput(DType.Module, 0, inputType, targetFunction, null);
        }

        public Module(Func<THyperparams, TInputs, TOutputs> fnModule, Delegate? referenceMethod = null, string? name = null)
        {
            var targetFunction = ModuleHelper.CreateTargetFunction(fnModule, referenceMethod: referenceMethod?.Method, defaultName: name, invokeTarget: referenceMethod?.Target);
            this.ModuleVariable = ModuleHelper.CreateModule(targetFunction, this.GetType());
        }

        public Model<TInputs, TOutputs> SetHyperparams(THyperparams hyperparams)
        {
            return this.SetHyperparams<Model<TInputs, TOutputs>>(hyperparams);
        }

        public T SetHyperparams<T>(THyperparams hyperparams) where T : BaseModel<TInputs, TOutputs>
        {
            Vector<int64> iterationIndices = [.. LoopAPI.IterationIndices];
            var inputVariables = ModuleHelper.Format(hyperparams);
            Scalar<IModelVarType> modelVariable = InternalOp.ModuleSetHyperparams(this.ModuleVariable, inputVariables, iterationIndices, localModelId: null, identifierTemplateString: null);

            var genericTypeArgs = ModuleHelper.ExtractGenericTypeArgsFromType(this.GetType());

            var modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>), typeof(DType[]) });
            if (modelVariableConstructor == null)
            {
                modelVariableConstructor = typeof(T).GetConstructor(new Type[] { typeof(Scalar<IModelVarType>) });
                var model = (T)modelVariableConstructor.AssertNotNull().Invoke(new object[] { modelVariable });
                model.GenericTypeArgs = genericTypeArgs;
                return model;
            }
            return (T)modelVariableConstructor.Invoke(new object?[] { modelVariable, genericTypeArgs });
        }
    }
}
