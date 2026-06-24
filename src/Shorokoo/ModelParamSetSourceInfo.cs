
using Shorokoo.Core.Factory.IR;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Runtime.InteropServices;
using System.Text;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;

namespace Shorokoo
{
    /// <summary>
    /// Metadata about a source for a set of model parameters.
    /// </summary>
    public class ModelParamSetSourceInfo
    {
        public ModelParamSetSourceInfo(string providerName, Type? providerType, ImmutableArray<ModelParamInfo> modelParams)
        {
            this.ProviderName = providerName;
            this.ProviderType = providerType;
            this.ModelParams = modelParams;
        }

        public string ProviderName { get; private set; }
        
        public Type? ProviderType { get; private set; } 

        public ImmutableArray<ModelParamInfo> ModelParams { get; private set; }
    }
}
