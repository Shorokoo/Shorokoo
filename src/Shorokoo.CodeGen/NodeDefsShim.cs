// Generator-side stand-ins for the runtime types the shared NodeDefinitions source mentions.
// The generator (netstandard2.0) cannot reference Shorokoo.dll, so it provides its own copies in the
// SAME namespaces the runtime uses; the one shared definition file then compiles in both projects,
// binding to whichever copy its project supplies. The dummy NodeDefinitionMaker records only the
// op-spelling the frontend needs (op code, input/attribute names + kinds, the Code template); the
// runtime build uses the real, full maker instead. Namespaces a shared file `using`s but that the
// generator does not populate are declared empty (below) so the `using` resolves.
using System.Collections.Generic;

namespace Shorokoo
{
    // IVarType marker hierarchy — the element-type markers the definitions use as `Tensor<X>` etc.
    public interface IVarType { }
    public interface IStruct { }
    public interface IModelVarType : IVarType { }
    public interface IModuleVarType : IVarType { }
    public sealed class AnyLike : IVarType { }
    public sealed class NumLike : IVarType { }
    public sealed class FloatLike : IVarType { }
    public sealed class IndexLike : IVarType { }
    public sealed class IntLike : IVarType { }
    public sealed class AnyIntLike : IVarType { }
    public sealed class SignedNumLike : IVarType { }
    public sealed class UnsignedIntLike : IVarType { }
    public sealed class Int8Like : IVarType { }
    public sealed class SimpleNumLike : IVarType { }
    public sealed class SimpleNumLike2 : IVarType { }
    public sealed class CommonLike : IVarType { }
    public sealed class int64 : IVarType { }
    public sealed class int32 : IVarType { }
    public sealed class int16 : IVarType { }
    public sealed class int8 : IVarType { }
    public sealed class uint8 : IVarType { }
    public sealed class float32 : IVarType { }
    public sealed class bit : IVarType { }

    // Referenced in AttributeDType/AttributeDTypes signatures only.
    public sealed class DType { }

    // A shared definition file may `using static Shorokoo.Globals;` (vestigial in the structural
    // defs); the generator only needs the type to exist for the directive to resolve.
    public static class Globals { }
}

// Empty shims for namespaces the shared definition / OpCodes / AttributeNames files `using` but that
// the generator does not otherwise populate, so those `using`s resolve.
namespace Shorokoo.Core.Nodes.AutoDiff { }
namespace Shorokoo.Modules { }
namespace Shorokoo.Core.Training { }
namespace Shorokoo.Core { public static class InternalGlobals { } }
namespace Shorokoo.Core.Nodes { }
namespace Shorokoo.Core.Factory { }
namespace Shorokoo.Core.Factory.IR { }
namespace Shorokoo.Onnx { }

namespace Shorokoo.Core.Nodes.NodeDefinitions
{
    public enum DataStructure { Tensor, Optional, Sequence, TensorStruct }
    public enum InputType { Hyperparam, ReadyInput, ModelInput, GenericType }

    // Attribute enums used as `AttributeEnum<X>` type args (and a few whose members appear as values).
    public enum AutoPad { NotSet, SameUpper, SameLower, Valid }
    public enum ScatterNDReduction { None }
    public enum ResizeMode { None }
    public enum TensorScatterMode { None }
    public enum RoiAlignTransformationMode { None }
    public enum RoiAlignMode { None }
    public enum RNNDirection { None }
    public enum PadMode { None }
    public enum NearestMode { None }
    public enum KeepAspectRatioPolicy { None }
    public enum GridSamplePaddingMode { None }
    public enum GridSampleMode { None }
    public enum GeluApproximate { None }
    public enum GRUDirection { None }
    public enum DepthColumnRowMode { None }
    public enum CoordinateTransformationMode { None }
    public enum BitShiftDirection { None }
    public enum LSTMDirection { None }

    /// <summary>
    /// Generator-side <c>NodeDefinitionMaker</c>: the full fluent surface the shared definitions call,
    /// recording only what the op-spelling table needs. Every builder returns <c>this</c> so the fluent
    /// chains type-check; type/rank/structure/constraint details are ignored.
    /// </summary>
    internal sealed class NodeDefinitionMaker
    {
        public string? OpName { get; private set; }
        public List<string> Inputs { get; } = new();
        public List<(string Name, string Kind)> Attributes { get; } = new();
        public string? CodeTemplate { get; private set; }

        public NodeDefinitionMaker Op(string opName) { OpName = opName; return this; }

        public NodeDefinitionMaker Tensor<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType => this;
        public NodeDefinitionMaker Optional<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType => this;
        public NodeDefinitionMaker Sequence<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType => this;
        public NodeDefinitionMaker Any<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IVarType => this;
        public NodeDefinitionMaker TensorStruct<T>(string typeName, int minVariadicCount = -1, bool tracksModuleFn = false) where T : IStruct => this;
        public NodeDefinitionMaker VarType<T>(string typeName, bool tracksModuleFn = false) => this;

        public NodeDefinitionMaker Structure(string typeName, DataStructure structure) => this;
        public NodeDefinitionMaker Structure(string typeName, DataStructure[] structures) => this;
        public NodeDefinitionMaker Variadic(string name, int minCount = 0, int maxCount = int.MaxValue) => this;
        public NodeDefinitionMaker Rank(string name, int minRank = 0, int maxRank = int.MaxValue) => this;

        public NodeDefinitionMaker SingleGraphOpen(string graphAttributeName) => AttributeGraph(graphAttributeName);
        public NodeDefinitionMaker SingleGraphClose(string graphAttributeName) => AttributeGraph(graphAttributeName);
        public NodeDefinitionMaker PairGraphOpen(string g1, string g2) { AttributeGraph(g1); return AttributeGraph(g2); }
        public NodeDefinitionMaker PairGraphClose(string g1, string g2) { AttributeGraph(g1); return AttributeGraph(g2); }

        public NodeDefinitionMaker Input(string paramName, string defsRef, int? rank = null, DataStructure? structure = null, bool isOptional = false) { Inputs.Add(paramName); return this; }
        public NodeDefinitionMaker Input(string paramName, string defsRef, string rank, DataStructure? structure = null, bool isOptional = false) { Inputs.Add(paramName); return this; }
        public NodeDefinitionMaker Input(string paramName, string[] defsRef, string rank, DataStructure? structure = null, bool isOptional = false) { Inputs.Add(paramName); return this; }
        public NodeDefinitionMaker Input(string paramName, string[] defsRefs, int? rank = null, DataStructure? structure = null, bool isOptional = false) { Inputs.Add(paramName); return this; }

        public NodeDefinitionMaker Output(string paramName, string defsRef, string rank, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false) => this;
        public NodeDefinitionMaker Output(string paramName, string[] defsRefs, string rank, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false) => this;
        public NodeDefinitionMaker Output(string paramName, string defsRef, int? rank = null, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false) => this;
        public NodeDefinitionMaker Output(string paramName, string[] defsRefs, int? rank = null, string? rankPlusOne = null, string? rankMinusOne = null, string? rankMinusTwo = null, string? rankBroadcast = null, DataStructure? structure = null, bool isOptional = false) => this;

        public NodeDefinitionMaker AttributeLong(string attributeName, string? rank = null, object? defaultValue = null, string? variadicCount = null) => Attr(attributeName, "Long");
        public NodeDefinitionMaker AttributeLongs(string attributeName, string? rank = null, object? defaultValue = null) => Attr(attributeName, "Longs");
        public NodeDefinitionMaker AttributeFloat(string attributeName, object? defaultValue = null, bool gtz = false, bool gtez = false) => Attr(attributeName, "Float");
        public NodeDefinitionMaker AttributeFloats(string attributeName, object? defaultValue = null) => Attr(attributeName, "Floats");
        public NodeDefinitionMaker AttributeString(string attributeName, object? defaultValue = null) => Attr(attributeName, "String");
        public NodeDefinitionMaker AttributeStrings(string attributeName, object? defaultValue = null) => Attr(attributeName, "Strings");
        public NodeDefinitionMaker AttributeGraph(string attributeName) => Attr(attributeName, "Graph");
        public NodeDefinitionMaker AttributeDType(string attributeName, string? tensorTypeName = null, DType? defaultValue = null) => Attr(attributeName, "DType");
        public NodeDefinitionMaker AttributeDTypes(string attributeName, string? tensorTypeName = null, DType[]? defaultValue = null) => Attr(attributeName, "DTypes");
        public NodeDefinitionMaker AttributeTypeProto(string attributeName, string? tensorTypeName = null, string? structureDefName = null) => Attr(attributeName, "TypeProto");
        public NodeDefinitionMaker AttributeBool(string attributeName, object? defaultValue = null) => Attr(attributeName, "Bool");
        public NodeDefinitionMaker AttributeBools(string attributeName) => Attr(attributeName, "Bools");
        public NodeDefinitionMaker AttributeEnum<T>(string attributeName, string[] enumValues, string? defaultValue = null, string? structureDefName = null) => Attr(attributeName, "Enum");
        public NodeDefinitionMaker AttributeEnums<T>(string attributeName, string[] enumValues, string[]? defaultValue = null, string? structureDefName = null) => Attr(attributeName, "Enums");
        public NodeDefinitionMaker AttributeTensor(string attributeName, string tensorTypeName, string rank) => Attr(attributeName, "Tensor");

        public NodeDefinitionMaker Constraint(string attributeName, string attributeValue) => this;
        public NodeDefinitionMaker Constraint(string attributeName, int? attributeValue) => this;
        public NodeDefinitionMaker ConstraintIsSet(string attributeName, bool isSet) => this;
        public NodeDefinitionMaker Code(string codeTemplate, bool inline = false) { CodeTemplate = codeTemplate; return this; }
        public NodeDefinitionMaker AlternateDefinitionName(string alternateName) => this;

        private NodeDefinitionMaker Attr(string name, string kind) { Attributes.Add((name, kind)); return this; }
    }
}
