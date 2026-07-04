// Generator-side stand-ins for the runtime types that the shared NodeDefinitions source
// mentions. Because the generator (netstandard2.0) cannot reference Shorokoo.dll, it provides its
// own copies of these types in the SAME namespaces the runtime uses, so the one shared definition
// file compiles in both projects (binding to whichever copy its project supplies). The dummy
// NodeDefinitionMaker records only the op-spelling the frontend needs (op code, input/attribute
// names, the Code template); the runtime build uses the real, full maker instead.
//
// Namespaces a shared definition file `using`s but that the generator does not populate with real
// symbols are declared empty here, so the `using` resolves rather than erroring (the trick that lets
// one file be interpreted slightly differently per project).

namespace Shorokoo
{
    // Minimal IVarType marker hierarchy — only the markers the currently-shared definitions use.
    public interface IVarType { }
    public sealed class AnyLike : IVarType { }
    public sealed class NumLike : IVarType { }
    public sealed class FloatLike : IVarType { }
}

namespace Shorokoo.Core.Nodes.AutoDiff { }  // used by some definition files; no symbols needed here

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    public enum DataStructure { Tensor, Optional, Sequence, TensorStruct }

    /// <summary>
    /// Generator-side <c>NodeDefinitionMaker</c>: the fluent surface the shared definitions call,
    /// recording only what the op-spelling table needs. Every builder method returns <c>this</c> so
    /// the fluent chains type-check; most inputs are ignored.
    /// </summary>
    internal sealed class NodeDefinitionMaker
    {
        public string? OpName { get; private set; }
        public System.Collections.Generic.List<string> Inputs { get; } = new();
        public System.Collections.Generic.List<string> Attributes { get; } = new();
        public string? CodeTemplate { get; private set; }

        public NodeDefinitionMaker Op(string opName) { OpName = opName; return this; }

        public NodeDefinitionMaker Tensor<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType => this;

        public NodeDefinitionMaker Input(string paramName, string defsRef, string rank, DataStructure? structure = null, bool isOptional = false)
        { Inputs.Add(paramName); return this; }

        public NodeDefinitionMaker Output(string paramName, string defsRef, string rank,
            string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null,
            string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false) => this;

        public NodeDefinitionMaker Code(string codeTemplate, bool inline = false) { CodeTemplate = codeTemplate; return this; }
    }
}
