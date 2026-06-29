using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Text;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Training;

namespace Shorokoo.Modules
{
    public interface IModelSequence
    {

    }

    public interface IModelSequence<T> : IModelSequence where T : IModel
    {

    }

    public class ModelSequence : IModelSequence
    {
        internal TensorSequence<IModelVarType> modelSequenceVariable;
        internal Function targetFunction;

        internal ModelSequence(Function targetFunction)
        {
            this.modelSequenceVariable = TensorSequence<IModelVarType>.CreateEmpty(targetFunction);
            this.targetFunction = targetFunction;
        }

        internal ModelSequence(Scalar<IModelVarType>[] models, Function targetFunction)
        {
            this.modelSequenceVariable = TensorSequence<IModelVarType>.Create([.. models.Select(m => (Tensor<IModelVarType>)m)], targetFunction);
            this.targetFunction = targetFunction;
        }

        internal ModelSequence(TensorSequence<IModelVarType> models, Function targetFunction)
        {
            this.modelSequenceVariable = models;
            this.targetFunction = targetFunction;
        }

        public static ModelSequence<T> Empty<T>(T modelTemplate) where T : IModel
            => new ModelSequence<T>(modelTemplate, isModelTemplate: true);

        public static ModelSequence<T> Create<T>(params T[] inputModels) where T : IModel
            => new ModelSequence<T>(inputModels);
    }

    public class ModelSequence<T> : ModelSequence, IModelSequence<T> where T : IModel
    {

        public ModelSequence(T modelTemplate, bool isModelTemplate) : base(modelTemplate.ModelVariable.ModuleFn.AssertNotNull())
        {
        }

        public ModelSequence(params T[] inputModels)
            : base(inputModels.Select(x => x.ModelVariable).ToArray(), inputModels[0].ModelVariable.ModuleFn.AssertNotNull())
        {
        }

        internal ModelSequence(TensorSequence<IModelVarType> models, Function targetFunction) : base(models, targetFunction)
        {
        }

        public T this[Scalar<int64> index]
        {
            get
            {
                var modelVariable = base.modelSequenceVariable[index].Scalar();
                var type = typeof(T);
                // The generated model constructor takes Scalar<IModelVarType>. That type is statically
                // known here, so convert from Variable to IValue with the implicit operator directly;
                // reflective Invoke can't apply that conversion itself, it only receives the result.
                var ctor = type.GetConstructor([typeof(Scalar<IModelVarType>)]).AssertNotNull();
                return (T)ctor.Invoke([(Scalar<IModelVarType>)(Variable)modelVariable]);
            }
        }

        public ModelSequence<T> RemoveAt(Scalar<int64> index)
            => new ModelSequence<T>(base.modelSequenceVariable.RemoveAt(index), base.targetFunction);

        public ModelSequence<T> InsertAt(T model, Scalar<int64> index)
            => new ModelSequence<T>(base.modelSequenceVariable.InsertAt(model.ModelVariable, index), base.targetFunction);

        public ModelSequence<T> Append(T model)
            => new ModelSequence<T>(base.modelSequenceVariable.Append(model.ModelVariable), base.targetFunction);
    }
}