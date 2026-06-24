using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using System.Diagnostics;
using System.Net.NetworkInformation;
using static Shorokoo.Core.InternalGlobals;
using static Shorokoo.Globals;
using System.Net;
using System.Collections.Immutable;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using Microsoft.VisualBasic;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    internal static partial class Definitions
    {
        private static NodeDefinitionMaker Op(string opName) => new NodeDefinitionMaker().Op(opName);

        private static ImmutableDictionary<string, NodeDefinitionResolver>? nodeDefinitions;

        public static ImmutableArray<string> ModuleOps = [InternalOpCodes.CREATE_MODULE, InternalOpCodes.MODULE_SET_HYPERPARAMS];

        public static ImmutableDictionary<string, NodeDefinitionResolver> NodeDefinitions
        {
            get
            {
                if (nodeDefinitions is null)
                {
                    var allMakers = GetACMakers()
                        .Concat(GetDGMakers())
                        .Concat(GetHLMakers())
                        .Concat(GetMQMakers())
                        .Concat(GetRMakers())
                        .Concat(GetSMakers())
                        .Concat(GetTZMakers())
                        .Concat(GetInternalMakers());

                    nodeDefinitions = allMakers
                        .Select(x => x.Finish())
                        .ToDictionary(x => x.OpName)
                        .ToImmutableDictionary();
                }

                return nodeDefinitions;
            }
        }
    }
}
