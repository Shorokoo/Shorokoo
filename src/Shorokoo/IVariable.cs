
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static RandN.Distributions.Uniform;
using System.Diagnostics.CodeAnalysis;

namespace Shorokoo
{
    public interface IModuleParam
    {
    }

    public interface IVariable : IModuleParam
    {
        public Node OwningNode { get; }

        public Node ParentNode => this.OwningNode;
        public DType Type { get; }
        public DType DType => this.Type;
        public Variable<V> As<V>() where V : IVarType;
        public Function? ModuleFn { get; }
        
        /// <summary>
        /// The unique name for this tensor. Defaults to Key.ToString() but can be set to human-readable
        /// names like "N1_T0" by processors during construction. Used for ONNX serialization.
        /// </summary>
        string UniqueName { get; }
        
        /// <summary>
        /// Obsolete: Use UniqueName instead. DefaultName now redirects to UniqueName for backwards compatibility.
        /// </summary>
        [Obsolete("Use UniqueName instead. DefaultName is deprecated and will be removed in a future version.")]
        string DefaultName => UniqueName;
        
        /// <summary>
        /// Deprecated: FriendlyName is no longer used. Use UniqueName for ONNX names or Key for stable identifiers.
        /// </summary>
        [Obsolete("FriendlyName is deprecated. Use UniqueName for ONNX names or Key.ToString() for stable identifiers.")]
        string? FriendlyName { get; }
        
        bool IsValid { get; set; }

        /// <summary>
        /// A globally unique identifier for this tensor, composed of the parent node's key and the output index.
        /// </summary>
        TensorKey Key { get; }

        bool IsConnectingTensor => OwningNode.IsOpenNode && OwningNode.ConnectingTensor == this;

        InputType? InputType
        {
            get
            {
                if (!this.OwningNode.IsModelInput)
                    return null;

                if (this.OwningNode.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                    return Shorokoo.Core.Nodes.NodeDefinitions.InputType.GenericType;

                var inputType = this.OwningNode.Attributes.GetEnumVal<Shorokoo.Core.Nodes.NodeDefinitions.InputType>(OnnxOpAttributeNames.ShrkAttrInputType);
                Debug.Assert(inputType != Shorokoo.Core.Nodes.NodeDefinitions.InputType.ModelInput); // Not really supported yet.

                return inputType ?? Shorokoo.Core.Nodes.NodeDefinitions.InputType.ReadyInput;
            }
        }

        /// <summary>
        /// The <c>[Hyper(defaultValue)]</c> default recorded on this model-input node, or null when
        /// the input is not a defaulted hyperparameter. Lets serializers re-emit the default (e.g. the
        /// C# emitter writes <c>[Hyper(defaultValue)]</c>). The attribute is optional, so a node that
        /// never carried it (any non-defaulted input) reads back as null rather than throwing.
        /// </summary>
        float? HyperDefaultValue
            => this.OwningNode.IsModelInput
                && this.OwningNode.Attributes.GetAttributeVals().TryGetValue(OnnxOpAttributeNames.ShrkAttrDefaultValue, out var dv)
                ? (float?)dv
                : null;

        TensorDim[]? TensorDims 
        {
            get
            {
                var tensorData = this.OwningNode.GetTensorData();
                if (tensorData is not null)
                    return tensorData.Shape.Dims.Select(x => new TensorDim(x)).ToArray();

                var rank = this.Rank();
                if (rank is null)
                    return null;

                return Enumerable.Range(1, rank.AssertNotNull()).Select(x => new TensorDim()).ToArray();
            }
        }
    }

    /// <summary>
    /// Interface for TensorStruct instances. TensorStruct is a mechanism for grouping multiple IVariables together into a single composite IVariable.
    /// Parallel to ITensor, ITensorSequence, and IOptionalTensor.
    /// </summary>
    public interface ITensorStruct : IVariable
    {
        /// <summary>
        /// Gets the definition describing the structure of this TensorStruct (field names, types, order).
        /// </summary>
        TensorStructDef Definition { get; }

        /// <summary>
        /// Gets a field from this TensorStruct by name.
        /// </summary>
        /// <param name="name">The name of the field to retrieve</param>
        /// <returns>The IVariable for the specified field</returns>
        IVariable GetField(string name);
    }

    public static class IVariableExtensions
    {
        public static Variable<T> As<T>(this IVariable var) where T : IVarType => ((Variable<T>)var);
        public static Variable<uint4> uint4(this IVariable var) => ((Variable<uint4>)var);
        public static Variable<uint8> uint8(this IVariable var) => ((Variable<uint8>)var);
        public static Variable<uint16> uint16(this IVariable var) => ((Variable<uint16>)var);
        public static Variable<uint32> uint32(this IVariable var) => ((Variable<uint32>)var);
        public static Variable<uint64> uint64(this IVariable var) => ((Variable<uint64>)var);
        public static Variable<int4> int4(this IVariable var) => ((Variable<int4>)var);
        public static Variable<int8> int8(this IVariable var) => ((Variable<int8>)var);
        public static Variable<int16> int16(this IVariable var) => ((Variable<int16>)var);
        public static Variable<int32> int32(this IVariable var) => ((Variable<int32>)var);
        public static Variable<int64> int64(this IVariable var) => ((Variable<int64>)var);
        public static Variable<float16> float16(this IVariable var) => ((Variable<float16>)var);
        public static Variable<bfloat16> bfloat16(this IVariable var) => ((Variable<bfloat16>)var);
        public static Variable<float32> float32(this IVariable var) => ((Variable<float32>)var);
        public static Variable<float64> float64(this IVariable var) => ((Variable<float64>)var);

        public static DataStructure Structure(this IVariable var)
            => var is ITensorStruct ? DataStructure.TensorStruct :
               var is ITensor ? DataStructure.Tensor :
               var is IOptionalTensor ? DataStructure.Optional :
               DataStructure.Sequence;

        public static int? Rank(this IVariable var)
            => var is ITensor tensor ? tensor.Rank : null;

        public static bool IsModelInput(this IVariable var) => var.OwningNode.IsModelInput;

        internal static IVariable ToVariable(this IModuleParam param) => 
                        param is IVariable var ? var :
                        param is IModel model ? model.ModelVariable :
                        param is IModule module ? module.ModuleVariable :
                        throw new InvalidTensorOperationException(ErrorCodes.CR001, "ToVariable", param?.GetType()?.Name ?? "null", 
                            "Invalid IModuleParam type for variable conversion");
    }
}
