using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Text;

namespace Shorokoo
{
    public interface IModelParamSetResolver
    {
        public abstract ModelParamNameMapping? ResolveNames(ModelParamSetSourceInfo sourceInfo, ModelParamSetSpecification? specification = null);
    }


    /// <summary>
    /// Maps model parameters from one naming convention to another naming convention.
    /// 
    /// Typically used when loading parameters created by one model implementation into
    /// a different model implementation (from same or different library/framework) with
    /// same or similar computation graphs, but differently named model parameters.
    /// 
    /// Applies to both trainable parameters and hyperparameters.
    /// </summary>
    public class ModelParamNameMapping
    { 
        // public BiDictionary<string, string> 
    }
}
