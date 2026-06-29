using Shorokoo;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Shorokoo.Core.Nodes
{
    internal class NodeInfo
    {
        public string[] Inputs { get; private set; }

        public string[] Outputs { get; private set; }

        public OnnxProtoAttributes Attributes { get; private set; }

        public OpSetVersion Version { get; private set; }

        public string Domain { get; private set; }

        public string OpCode { get; private set; }

        public string? StackTrace { get; private set; }

        public string? IdentifierTemplateString { get; private set; }

        public NodeInfo(Variable?[] inputs, Variable?[] outputs, OnnxProtoAttributes attributes, string? identifierTemplateString, string? stackTrace, OpSetVersion version, string opCode, string domain = "")
        {
            this.Inputs = inputs.Select(x => x?.UniqueName ?? "").ToArray();
            this.Outputs = outputs.Select(x => x?.UniqueName ?? "").ToArray();
            this.Attributes = attributes;
            this.IdentifierTemplateString = identifierTemplateString;
            this.Version = version;
            this.OpCode = opCode;
            this.Domain = domain;
            this.StackTrace = stackTrace;
        }
    }
}
