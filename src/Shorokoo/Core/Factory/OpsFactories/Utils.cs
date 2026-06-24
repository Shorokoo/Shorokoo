using Shorokoo;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using IR = Shorokoo.Core.Factory.IR;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shorokoo.Core.Factory.OpsFactories
{
    public static class Helpers
    {
        public static DType[] Numeric14 = new DType[]
        {
            DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
            DType.Int8, DType.Int16, DType.Int32, DType.Int64,
            DType.Float16, DType.Float32, DType.Float64, DType.BFloat16
        };

        public static DType[] Numeric13 = new DType[]
        {
            DType.UInt32, DType.UInt64,
            DType.Int32, DType.Int64,
            DType.Float16, DType.Float32, DType.Float64, DType.BFloat16
        };

        public static DType[] Numeric6 = new DType[]
        {
            DType.UInt32, DType.UInt64,
            DType.Int32, DType.Int64,
            DType.Float16, DType.Float32, DType.Float64
        };

        public static DType[] Numeric1 = new DType[]
        {
            DType.Float16, DType.Float32, DType.Float64
        };

        public static DType[] All2 = new DType[]
        {
            DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
            DType.Int8, DType.Int16, DType.Int32, DType.Int64,
            DType.Float16, DType.Float32, DType.Float64,
            DType.Bool, DType.String, 
            DType.Complex64, DType.Complex128
        };

        public static DType[] All13 = new DType[]
        {
            DType.UInt8, DType.UInt16, DType.UInt32, DType.UInt64,
            DType.Int8, DType.Int16, DType.Int32, DType.Int64,
            DType.Float16, DType.Float32, DType.Float64, DType.BFloat16,
            DType.Bool, DType.String,
            DType.Complex64, DType.Complex128
        };

        public static Shorokoo.Core.Factory.IR.AttributeProto.AttributeType ToProto(this Shorokoo.Core.Nodes.NodeDefinitions.AttributeType type)
        {
            switch (type)
            {
                case AttributeType.Bool: return IR.AttributeProto.AttributeType.Int;
                case AttributeType.Bools: return IR.AttributeProto.AttributeType.Ints;
                case AttributeType.Long: return IR.AttributeProto.AttributeType.Int;
                case AttributeType.Longs: return IR.AttributeProto.AttributeType.Ints;
                case AttributeType.DType: return IR.AttributeProto.AttributeType.Int;
                case AttributeType.DTypes: return IR.AttributeProto.AttributeType.Ints;
                case AttributeType.Float: return IR.AttributeProto.AttributeType.Float;
                case AttributeType.Floats: return IR.AttributeProto.AttributeType.Floats;
                case AttributeType.Graph: return IR.AttributeProto.AttributeType.Graph;
                case AttributeType.String: return IR.AttributeProto.AttributeType.String;
                case AttributeType.Strings: return IR.AttributeProto.AttributeType.Strings;
                case AttributeType.Enum: return IR.AttributeProto.AttributeType.String;
                case AttributeType.Enums: return IR.AttributeProto.AttributeType.Strings;
                case AttributeType.Tensor: return IR.AttributeProto.AttributeType.Tensor;
                case AttributeType.TypeProto: return IR.AttributeProto.AttributeType.TypeProto;
                default: throw new UnsupportedDTypeException(ErrorCodes.FW044, type.ToString(), "utils operation", 
                    $"Utils operation not implemented for attribute type '{type}'");
            }
        }
    }
}
