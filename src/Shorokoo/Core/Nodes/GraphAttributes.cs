using Shorokoo.Graph;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Text;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Factory;

namespace Shorokoo.Core.Nodes
{

    internal class GraphInfo
    {
        public string AttributeName { get; private set; }
        public string? DefaultName { get; private set; }
        public Variable[] InputTensors { get; private set; }
        public Variable?[] OutputTensors { get; internal set; }

        public GraphInfo(string attributeName, Variable[] inputTensors, Variable?[] outputTensors)
        {
            this.AttributeName = attributeName;
            this.InputTensors = inputTensors;
            this.OutputTensors = outputTensors;
        }
    }
}
