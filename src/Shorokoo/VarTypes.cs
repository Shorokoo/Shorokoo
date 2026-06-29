
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using Shorokoo.Core.Nodes.NodeDefinitions;
using System.Collections.Immutable;
using System.Linq;

#pragma warning disable CS8981

namespace Shorokoo
{
    public interface IVarType;

    public interface SimpleFloatLike : FloatLike, SimpleNumLike, SimpleNumLike2;
    public interface SimpleNumLike : AnyNumLike;
    public interface SimpleNumLike2 : AnyNumLike;

    public interface bfloat16 : FloatLike;
    public interface float16 : FloatLike;
    public interface float32 : SimpleFloatLike;
    public interface float64 : SimpleFloatLike;
    public interface int4 : AnySignedIntLike;
    public interface int8 : SignedIntLike, Int8Like;
    public interface int16 : SignedIntLike, SimpleNumLike;
    public interface int32 : IndexLike, SimpleNumLike, SimpleNumLike2;
    public interface int64 : IndexLike, SimpleNumLike, SimpleNumLike2;
    public interface uint4 : AnyUnsignedIntLike;
    public interface uint8 : UnsignedIntLike, Int8Like;
    public interface uint16 : UnsignedIntLike;
    public interface uint32 : UnsignedIntLike, SimpleNumLike2;
    public interface uint64 : UnsignedIntLike, SimpleNumLike2;
    // Variable-length UTF-8 string tensor element. Maps to ONNX
    // TensorProto.DataType.STRING (8). Element-of, not array-of: a Tensor<@string>
    // is a tensor whose individual elements are .NET strings.
    public interface @string : IVarType;
    public interface bit : CommonLike;
    public interface complex64 : ComplexLike;
    public interface complex128 : ComplexLike;
    public interface invalid : IVarType;

    public interface Int8Like : IntLike;
    public interface IndexLike : SignedIntLike;
    public interface SignedIntLike : AnySignedIntLike, IntLike;
    public interface UnsignedIntLike : AnyUnsignedIntLike, IntLike;
    public interface IntLike : AnyIntLike, NumLike;
    public interface AnyUnsignedIntLike : AnyUnsignedNumLike, AnyIntLike;
    public interface AnySignedIntLike : AnySignedNumLike, AnyIntLike;
    public interface AnyIntLike : AnyNumLike;

    public interface ComplexLike : AnySignedNumLike;
    public interface FloatLike : SignedNumLike;

    public interface IModuleVarType : ParamLike;
    public interface IModelVarType : ParamLike;

    // Generic type placeholders - used during VirtualGraph construction for unresolved generic type parameters
    // These derive from all primitive type interfaces so operators will accept them as valid
    public interface IGenericType : 
        IVarType,
        FloatLike,
        SignedIntLike,
        UnsignedIntLike,
        IntLike,
        NumLike,
        AnyNumLike,
        AnyLike,
        CommonLike,
        ParamLike
    { }

    // Specific generic type markers (one per generic type parameter: T, Q, R, etc.)
    public interface IGenericType1 : IGenericType { }
    public interface IGenericType2 : IGenericType { }
    public interface IGenericType3 : IGenericType { }
    public interface IGenericType4 : IGenericType { }
    public interface IGenericType5 : IGenericType { }
    public interface IGenericType6 : IGenericType { }
    public interface IGenericType7 : IGenericType { }
    public interface IGenericType8 : IGenericType { }

    public interface UnsignedNumLike : NumLike;
    public interface SignedNumLike : AnySignedNumLike, NumLike;
    public interface AnySignedNumLike : AnyNumLike;
    public interface AnyUnsignedNumLike : AnyNumLike;
    public interface AnyNumLike : ParamLike;
    public interface NumLike : AnyNumLike, CommonLike;
    public interface CommonLike : AnyLike;
    public interface AnyLike : ParamLike;
    public interface ParamLike : IVarType;

    /// <summary>
    /// Marker interface for TensorStruct types. All user-defined struct interfaces must derive from IStruct.
    /// IStruct has no associated DType - it's an abstract category marker like FloatLike or NumLike.
    /// </summary>
    public interface IStruct : IVarType;

    /// <summary>
    /// Built-in marker interface for TensorStruct types where the struct definition is not known at C# compile time.
    /// When using TensorStruct&lt;DTypeStruct&gt;, the struct definition comes from the DType at runtime rather than from interface property declarations.
    /// </summary>
    public interface DTypeStruct : IStruct;

    public interface IValue<out T> : IValue where T : IVarType;
}

#pragma warning restore CS8981
