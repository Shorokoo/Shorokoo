using Shorokoo.Core.Inference.Abstractions;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Text;
using static Shorokoo.Globals;

namespace Shorokoo.Core
{
    public class PrimitiveParam
    {
        private object boxed;

        private PrimitiveParam(object boxed)
        {
            this.boxed = boxed; 
        }

        public object ParamVal { get { return boxed; } } 

        public static implicit operator PrimitiveParam(bool param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(int param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(uint param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(long param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(ulong param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(short param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(ushort param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(sbyte param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(byte param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(float param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(double param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(BFloat16 param) => new PrimitiveParam(param);
        public static implicit operator PrimitiveParam(Float16 param) => new PrimitiveParam(param);

        public static implicit operator Scalar<bit>(PrimitiveParam param) => Scalar<bit>(param.ParamVal);
        public static implicit operator Scalar<int8>(PrimitiveParam param) => Scalar<int8>(param.ParamVal);
        public static implicit operator Scalar<int16>(PrimitiveParam param) => Scalar<int16>(param.ParamVal);
        public static implicit operator Scalar<int32>(PrimitiveParam param) => Scalar<int32>(param.ParamVal);
        public static implicit operator Scalar<int64>(PrimitiveParam param) => Scalar<int64>(param.ParamVal);
        public static implicit operator Scalar<uint8>(PrimitiveParam param) => Scalar<uint8>(param.ParamVal);
        public static implicit operator Scalar<uint16>(PrimitiveParam param) => Scalar<uint16>(param.ParamVal);
        public static implicit operator Scalar<uint32>(PrimitiveParam param) => Scalar<uint32>(param.ParamVal);
        public static implicit operator Scalar<uint64>(PrimitiveParam param) => Scalar<uint64>(param.ParamVal);
        public static implicit operator Scalar<bfloat16>(PrimitiveParam param) => Scalar<bfloat16>(param.ParamVal);
        public static implicit operator Scalar<float16>(PrimitiveParam param) => Scalar<float16>(param.ParamVal);
        public static implicit operator Scalar<float32>(PrimitiveParam param) => Scalar<float32>(param.ParamVal);
        public static implicit operator Scalar<float64>(PrimitiveParam param) => Scalar<float64>(param.ParamVal);

        public static implicit operator Tensor<bit>(PrimitiveParam param) => Scalar<bit>(param.ParamVal);
        public static implicit operator Tensor<int8>(PrimitiveParam param) => Scalar<int8>(param.ParamVal);
        public static implicit operator Tensor<int16>(PrimitiveParam param) => Scalar<int16>(param.ParamVal);
        public static implicit operator Tensor<int32>(PrimitiveParam param) => Scalar<int32>(param.ParamVal);
        public static implicit operator Tensor<int64>(PrimitiveParam param) => Scalar<int64>(param.ParamVal);
        public static implicit operator Tensor<uint8>(PrimitiveParam param) => Scalar<uint8>(param.ParamVal);
        public static implicit operator Tensor<uint16>(PrimitiveParam param) => Scalar<uint16>(param.ParamVal);
        public static implicit operator Tensor<uint32>(PrimitiveParam param) => Scalar<uint32>(param.ParamVal);
        public static implicit operator Tensor<uint64>(PrimitiveParam param) => Scalar<uint64>(param.ParamVal);
        public static implicit operator Tensor<bfloat16>(PrimitiveParam param) => Scalar<bfloat16>(param.ParamVal);
        public static implicit operator Tensor<float16>(PrimitiveParam param) => Scalar<float16>(param.ParamVal);
        public static implicit operator Tensor<float32>(PrimitiveParam param) => Scalar<float32>(param.ParamVal);
        public static implicit operator Tensor<float64>(PrimitiveParam param) => Scalar<float64>(param.ParamVal);
    }
}
