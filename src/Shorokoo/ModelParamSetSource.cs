using Shorokoo;
using Shorokoo.Modules;
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

namespace Shorokoo
{
    /// <summary>
    /// Metadata about a source for a set of model parameters.
    /// </summary>
    public interface IModelParamSetSource
    {
        public ModelParamSetSourceInfo GetInfo();
        public ModelParamList LoadParams(ModelParamNameMapping? nameMapping);
    }
}
