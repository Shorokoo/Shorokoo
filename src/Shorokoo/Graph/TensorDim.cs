
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Graph
{
    public class TensorDim
    {
        public string? Symbol { get; init; }
        public long? Size { get; init; }

        public TensorDim()
        {
            Symbol = null;
            Size = null;
        }

        public TensorDim(string dimSymbol)
        {
            this.Symbol = dimSymbol;
        }

        public TensorDim(long dimVal)
        {
            this.Size = dimVal;
        }
    }
}
