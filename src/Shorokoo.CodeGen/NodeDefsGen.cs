using System.Collections.Generic;

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Generator-side counterpart of the runtime's Definitions.Op helper, plus an accessor that
    // exposes the shared group makers (whose Get*Makers methods are private) to the op-table reader.
    // This file is compiled only into the generator; the runtime supplies its own Op helper.
    internal static partial class Definitions
    {
        private static NodeDefinitionMaker Op(string opName) => new NodeDefinitionMaker().Op(opName);

        /// <summary>Every op maker from the definition groups currently shared into the generator.</summary>
        internal static IEnumerable<NodeDefinitionMaker> AllSharedMakers()
        {
            foreach (var m in GetProofMakers()) yield return m;
            foreach (var m in GetTZMakers()) yield return m;
        }
    }
}
