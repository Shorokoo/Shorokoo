
using Shorokoo.Core.Factory.IR;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;

namespace Shorokoo.Core
{
    internal abstract class ModelArchitectureVariant
    {
        // public abstract ConcreteModelBase ToConcreteModel(ModelParamSet paramSet);
        public abstract ModelParamSetSpecification GetModelParamSpecifications();
    }

    internal abstract class ModelArchitecture
    {
        public abstract ModelArchitectureVariant ToArchitectureVariant(ModelParamList variantHyperParameters);
        public abstract ModelParamSetSpecification GetHyperameterSpecifications();
        //public ConcreteModelBase ToConcreteModel(ModelParamSet hyperParameters)
        //    => this.ToArchitectureVariant(hyperParameters).ToConcreteModel(hyperParameters);
    }

    //public class ModuleArchitecture<T> : ModelArchitecture where T : ModuleBase
    //{
    //
    //}
    //
    //public class ModuleArchitectureVariant<T> : ModelArchitectureVariant where T : ModuleBase
    //{
    //
    //}
}
