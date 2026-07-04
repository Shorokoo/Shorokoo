using System.Collections.Generic;
using System.Linq;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.ModuleV2
{
    /// <summary>
    /// Reads the op-spelling table from the shared <c>NodeDefinitions</c> source at generator build
    /// time — the single source of truth that will eventually replace <see cref="ModuleV2Compiler"/>'s
    /// hand-maintained <c>MethodOps</c>. Proof of concept over <c>Definitions.GetProofMakers()</c>:
    /// the same fluent definition calls that build full node definitions in the runtime bind here to
    /// the dummy maker (<c>NodeDefsShim.cs</c>) and yield just op code + inputs + Code template.
    /// </summary>
    internal static class ModuleV2OpTable
    {
        internal sealed class OpSpelling
        {
            public string OpCode { get; set; } = "";
            public IReadOnlyList<string> Inputs { get; set; } = System.Array.Empty<string>();
            public string? CodeTemplate { get; set; }
        }

        /// <summary>Builds op code → spelling from the shared definition makers.</summary>
        public static Dictionary<string, OpSpelling> BuildFromSharedDefs()
        {
            var table = new Dictionary<string, OpSpelling>();
            foreach (var m in Definitions.AllSharedMakers())
            {
                if (m.OpName is not string op) continue;
                table[op] = new OpSpelling
                {
                    OpCode = op,
                    Inputs = m.Inputs.ToArray(),
                    CodeTemplate = m.CodeTemplate,
                };
            }
            return table;
        }
    }
}
