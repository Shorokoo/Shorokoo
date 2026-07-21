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
        private static ImmutableHashSet<string>? vanillaOpNames;

        public static ImmutableArray<string> ModuleOps = [InternalOpCodes.CREATE_MODULE, InternalOpCodes.MODULE_SET_HYPERPARAMS];

        public static ImmutableDictionary<string, NodeDefinitionResolver> NodeDefinitions
        {
            get
            {
                if (nodeDefinitions is null)
                {
                    var standardResolvers = GetACMakers()
                        .Concat(GetDGMakers())
                        .Concat(GetHLMakers())
                        .Concat(GetMQMakers())
                        .Concat(GetRMakers())
                        .Concat(GetSMakers())
                        .Concat(GetTZMakers())
                        .Select(x => x.Finish());

                    var internalResolvers = GetInternalMakers()
                        .Select(x => x.Finish(isInternal: true));

                    nodeDefinitions = standardResolvers
                        .Concat(internalResolvers)
                        .ToDictionary(x => x.OpName)
                        .ToImmutableDictionary();
                }

                return nodeDefinitions;
            }
        }

        /// <summary>
        /// The op names expressible in the vanilla ONNX dialect: the emitted op name of every
        /// registered non-<see cref="NodeDefinitionResolver.IsInternal"/> definition (an
        /// <c>#OPEN</c>/<c>#CLOSE</c> marker pair collapses to the standard ONNX op it emits as,
        /// e.g. <c>Loop</c>). The vanilla-export guarantee treats any default-domain op
        /// <b>not</b> in this set as Shorokoo-internal, so the classification is registry-owned
        /// and fails closed: a new internal op is non-vanilla by default even if someone gives
        /// it a plain, convention-free name.
        /// </summary>
        public static ImmutableHashSet<string> VanillaOpNames
        {
            get
            {
                vanillaOpNames ??= NodeDefinitions.Values
                    .Where(x => !x.IsInternal)
                    .Select(x => x.VariantDefinitions[0].FullNodeOpName)
                    .ToImmutableHashSet(StringComparer.Ordinal);
                return vanillaOpNames;
            }
        }
    }
}
