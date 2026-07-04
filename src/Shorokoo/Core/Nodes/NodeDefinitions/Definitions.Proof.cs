using System.Collections.Generic;
using Shorokoo;                          // IVarType markers (NumLike, …)
using Shorokoo.Core.Nodes.AutoDiff;      // unused here — exercises the empty-namespace shim in the generator

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    // Proof-of-concept that ONE definition source compiles into both Shorokoo.dll (real
    // NodeDefinitionMaker + types) and Shorokoo.CodeGen (the dummy stand-ins in NodeDefsShim.cs).
    // The generator reads the op-spelling from these makers; the runtime would build full node
    // definitions from the same calls. See src/docs/design/mlir-assembly-parser.md.
    internal static partial class Definitions
    {
        internal static List<NodeDefinitionMaker> GetProofMakers() =>
        [
            new NodeDefinitionMaker()
                .Op("Abs")
                .Tensor<NumLike>("T")
                .Input("x", "T", "R")
                .Output("y", "T", "R")
                .Code("{1:this}.Abs()"),

            new NodeDefinitionMaker()
                .Op("Add")
                .Tensor<NumLike>("T")
                .Input("a", "T", "R1")
                .Input("b", "T", "R2")
                .Output("c", "T", "R")
                .Code("{1:low_op} + {2:low_op}"),

            // Exercises the attribute surface (enum / longs / bool) and generic markers, so the
            // dummy maker is validated against real-shaped usage, not just input-only ops.
            new NodeDefinitionMaker()
                .Op("ProofPool")
                .Tensor<FloatLike>("T")
                .AttributeEnum<AutoPad>("auto_pad", ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"])
                .AttributeLongs("kernel_shape")
                .AttributeBool("ceil_mode")
                .Input("X", "T", "R")
                .Output("Y", "T", "R")
                .Code("{1:this}.ProofPool({a:param}{b:param}{c:param})"),
        ];
    }
}
