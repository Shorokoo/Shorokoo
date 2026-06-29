using Shorokoo.Graph;
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
