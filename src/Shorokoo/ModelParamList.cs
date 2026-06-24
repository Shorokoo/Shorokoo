
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
    public class ModelParamList
    {
        public ImmutableArray<NamedModelParam> ModelParams { get; private set; }

        public ModelParamList()
        {
            this.ModelParams = ImmutableArray<NamedModelParam>.Empty;
        }

        public ModelParamList(IEnumerable<NamedModelParam> modelParams)
        {
            this.ModelParams = modelParams.ToImmutableArray();
        }

        public ModelParamList(IEnumerable<Tuple<string, TensorData>> paramVals, ModelParamType paramType = ModelParamType.InputParam)
        {
            this.ModelParams = paramVals.Select(x => (NamedModelParam)new TensorDataModelParam(x.Item1, paramType, x.Item2)).ToImmutableArray();
        }

        public ModelParamList(IEnumerable<(string name, TensorData data)> paramVals, ModelParamType paramType = ModelParamType.InputParam)
        {
            this.ModelParams = paramVals.Select(x => (NamedModelParam)new TensorDataModelParam(x.name, paramType, x.data)).ToImmutableArray();
        }

        public ModelParamList(IEnumerable<KeyValuePair<string, TensorData>> paramVals, ModelParamType paramType = ModelParamType.InputParam)
        {
            this.ModelParams = paramVals.Select(x => (NamedModelParam)new TensorDataModelParam(x.Key, paramType, x.Value)).ToImmutableArray();
        }

        public NamedModelParam? Find(string paramName)
        {
            return this.ModelParams.FirstOrDefault(x => x.ParamName == paramName);
        }
    }
}
