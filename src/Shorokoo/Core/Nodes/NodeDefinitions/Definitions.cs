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

        // Out-of-line test data. The structural op definitions carry no test data; each
        // Definitions.<Group>.Tests.cs (runtime-only) implements its RegisterXTestData partial to
        // register, keyed by op code, an action that applies that op's test data to its maker.
        // A group with no companion file leaves its partial unimplemented (a no-op), so migration
        // is safely incremental. See src/docs/design/mlir-assembly-parser.md.
        static partial void RegisterACTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterDGTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterHLTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterMQTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterRTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterSTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterTZTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);
        static partial void RegisterInternalTestData(Dictionary<string, System.Action<NodeDefinitionMaker>> r);

        private static Dictionary<string, System.Action<NodeDefinitionMaker>> BuildTestDataRegistry()
        {
            var r = new Dictionary<string, System.Action<NodeDefinitionMaker>>();
            RegisterACTestData(r);
            RegisterDGTestData(r);
            RegisterHLTestData(r);
            RegisterMQTestData(r);
            RegisterRTestData(r);
            RegisterSTestData(r);
            RegisterTZTestData(r);
            RegisterInternalTestData(r);
            return r;
        }

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
                        .Concat(GetInternalMakers())
                        .ToList();

                    var testData = BuildTestDataRegistry();
                    var appliedTestData = 0;
                    foreach (var maker in allMakers)
                        if (maker.OpName is string op && testData.TryGetValue(op, out var apply))
                        {
                            apply(maker);
                            appliedTestData++;
                        }
                    // Every out-of-line test-data entry must key to an op; a mismatch means a
                    // relocated entry silently lost its op (e.g. a wrong op-code constant).
                    Debug.Assert(appliedTestData == testData.Count,
                        $"NodeDefinitions: {testData.Count - appliedTestData} test-data key(s) matched no op code.");

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
