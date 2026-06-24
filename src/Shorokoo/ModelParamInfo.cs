
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
    public enum ModelParamType
    {
        Undefined,
        HyperParam,
        TrainableParam,
        InputParam,
        OutputParam
    }

    public class ModelParamInfo
    {
        public required Shape? Shape { get; init; }
        public required string Name { get; init; }
        public required ModelParamType ParamType { get; init; }
    }
}
