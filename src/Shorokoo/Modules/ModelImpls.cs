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
using System;
using System.Collections.Generic;
using System.Text;

namespace Shorokoo.Modules
{
    /// <summary>
    /// Concrete model taking no runtime inputs and producing <typeparamref name="TOutputs"/>.
    /// Obtained from a module via <c>SetHyperparams</c>, or declared as a model-typed graph input.
    /// </summary>
    public class Model<TOutputs> : BaseModel<TOutputs>
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model and returns its outputs.</summary>
        public TOutputs Call()
            => base.internalCall();
    }

    /// <summary>Concrete model taking one typed input and producing <typeparamref name="TOutputs"/>.</summary>
    public class Model<TInput1, TOutputs> : BaseModel<TInput1, TOutputs>
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs)
            => base.internalCall(inputs);
    }

    /// <summary>Concrete model taking two typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TOutputs> : BaseModel<(TInput1, TInput2), TOutputs> 
        where TInput1 : IModuleParam where TInput2 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2)
            => this.internalCall((inputs1, inputs2));
    }

    /// <summary>Concrete model taking three typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TInput3, TOutputs> : BaseModel<(TInput1, TInput2, TInput3), TOutputs>
        where TInput1 : IModuleParam where TInput2 : IModuleParam where TInput3 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2, TInput3 inputs3)
            => this.internalCall((inputs1, inputs2, inputs3));
    }

    /// <summary>Concrete model taking four typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TInput3, TInput4, TOutputs> : BaseModel<(TInput1, TInput2, TInput3, TInput4), TOutputs>
        where TInput1 : IModuleParam where TInput2 : IModuleParam where TInput3 : IModuleParam where TInput4 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2, TInput3 inputs3, TInput4 inputs4)
            => this.internalCall((inputs1, inputs2, inputs3, inputs4));
    }

    /// <summary>Concrete model taking five typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TInput3, TInput4, TInput5, TOutputs> : BaseModel<(TInput1, TInput2, TInput3, TInput4, TInput5), TOutputs>
        where TInput1 : IModuleParam where TInput2 : IModuleParam where TInput3 : IModuleParam where TInput4 : IModuleParam where TInput5 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2, TInput3 inputs3, TInput4 inputs4, TInput5 inputs5)
            => this.internalCall((inputs1, inputs2, inputs3, inputs4, inputs5));
    }

    /// <summary>Concrete model taking six typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TInput3, TInput4, TInput5, TInput6, TOutputs> : BaseModel<(TInput1, TInput2, TInput3, TInput4, TInput5, TInput6), TOutputs>
        where TInput1 : IModuleParam where TInput2 : IModuleParam where TInput3 : IModuleParam where TInput4 : IModuleParam where TInput5 : IModuleParam where TInput6 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2, TInput3 inputs3, TInput4 inputs4, TInput5 inputs5, TInput6 inputs6)
            => this.internalCall((inputs1, inputs2, inputs3, inputs4, inputs5, inputs6));
    }

    /// <summary>Concrete model taking seven typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TInput3, TInput4, TInput5, TInput6, TInput7, TOutputs> : BaseModel<(TInput1, TInput2, TInput3, TInput4, TInput5, TInput6, TInput7), TOutputs>
        where TInput1 : IModuleParam where TInput2 : IModuleParam where TInput3 : IModuleParam where TInput4 : IModuleParam where TInput5 : IModuleParam where TInput6 : IModuleParam where TInput7 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2, TInput3 inputs3, TInput4 inputs4, TInput5 inputs5, TInput6 inputs6, TInput7 inputs7)
            => this.internalCall((inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7));
    }

    /// <summary>Concrete model taking eight typed inputs (passed to the base as a tuple).</summary>
    public class Model<TInput1, TInput2, TInput3, TInput4, TInput5, TInput6, TInput7, TInput8, TOutputs> : BaseModel<(TInput1, TInput2, TInput3, TInput4, TInput5, TInput6, TInput7, TInput8), TOutputs>
        where TInput1 : IModuleParam where TInput2 : IModuleParam where TInput3 : IModuleParam where TInput4 : IModuleParam where TInput5 : IModuleParam where TInput6 : IModuleParam where TInput7 : IModuleParam where TInput8 : IModuleParam
    {
        /// <summary>Declares this model as a graph input of the given input type.</summary>
        public Model(InputType inputType) : base(inputType) { }
        /// <summary>Wraps an existing model-typed variable.</summary>
        public Model(Scalar<IModelVarType> modelVariable) : base(modelVariable) { }

        /// <summary>Invokes the model on the given inputs and returns its outputs.</summary>
        public TOutputs Call(TInput1 inputs1, TInput2 inputs2, TInput3 inputs3, TInput4 inputs4, TInput5 inputs5, TInput6 inputs6, TInput7 inputs7, TInput8 inputs8)
            => this.internalCall((inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8));
    }
}
