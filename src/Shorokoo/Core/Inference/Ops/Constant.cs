using System.Collections.Immutable;
using System.Diagnostics;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Inference.Helpers;
using Shorokoo.Core.Nodes.AutoDiff;

namespace Shorokoo.Core.Inference.Ops;

/// <summary>
/// Implements the ONNX Constant op, which can draw its value from one of the supported
/// attributes (<c>value</c>, <c>value_float(s)</c>, <c>value_int(s)</c>,
/// <c>value_string(s)</c>). <c>sparse_value</c> is not representable in-framework (no sparse
/// tensor support); an unsupported attribute is asserted in Debug builds.
/// </summary>
internal sealed class ConstantOp : QuickOp
{
    public override string OpCode => OpCodes.CONSTANT;

    protected override RuntimeTensor[] Compute(RuntimeTensor?[] inputs, OnnxCSharpAttributes attrs, int maxDataElements)
    {
        var tensor = attrs.GetTensorVal(OnnxOpAttributeNames.AttrValue);
        if (tensor is not null)
            return [TensorDataConverter.ToRuntimeTensor(tensor, maxDataElements)];

        var vInt = attrs.GetLongVal(OnnxOpAttributeNames.AttrValueInt);
        if (vInt.HasValue)
        {
            return [RuntimeTensorFactory.Create(DType.Int64, new Shape(Array.Empty<long>()))
                with { IntData = ImmutableArray.Create(vInt.Value) }];
        }

        var vInts = attrs.GetLongsVal(OnnxOpAttributeNames.AttrValueInts);
        if (vInts is not null)
        {
            var shape = new Shape(new long[] { vInts.Length });
            var rt = RuntimeTensorFactory.Create(DType.Int64, shape);
            if (RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
                rt = rt with { IntData = ImmutableArray.Create(vInts) };
            return [rt];
        }

        var vFloat = attrs.GetFloatVal(OnnxOpAttributeNames.AttrValueFloat);
        if (vFloat.HasValue)
        {
            return [RuntimeTensorFactory.Create(DType.Float32, new Shape(Array.Empty<long>()))
                with { FloatData = ImmutableArray.Create(vFloat.Value) }];
        }

        var vFloats = attrs.GetFloatsVal(OnnxOpAttributeNames.AttrValueFloats);
        if (vFloats is not null)
        {
            var shape = new Shape(new long[] { vFloats.Length });
            var rt = RuntimeTensorFactory.Create(DType.Float32, shape);
            if (RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
                rt = rt with { FloatData = ImmutableArray.Create(vFloats) };
            return [rt];
        }

        var vString = attrs.GetStringVal(OnnxOpAttributeNames.AttrValueString);
        if (vString is not null)
        {
            return [RuntimeTensorFactory.Create(DType.String, new Shape(Array.Empty<long>()))
                with { StringData = ImmutableArray.Create(vString) }];
        }

        var vStrings = attrs.GetStringsVal(OnnxOpAttributeNames.AttrValueStrings);
        if (vStrings is not null)
        {
            var shape = new Shape(new long[] { vStrings.Length });
            var rt = RuntimeTensorFactory.Create(DType.String, shape);
            if (RuntimeTensorFactory.ShouldStoreData(shape, maxDataElements))
                rt = rt with { StringData = ImmutableArray.Create(vStrings) };
            return [rt];
        }

        Debug.Assert(false,
            "Constant op missing a supported value attribute. sparse_value is not representable "
            + "in-framework; a Constant must carry value, value_int(s), value_float(s), or value_string(s).");
        return [RuntimeTensorFactory.Create(DType.Invalid, null)];
    }
}

