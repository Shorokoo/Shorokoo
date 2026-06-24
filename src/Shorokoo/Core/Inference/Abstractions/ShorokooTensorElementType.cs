namespace Shorokoo.Core.Inference.Abstractions;

// Integer values mirror ORT's Microsoft.ML.OnnxRuntime.Tensors.TensorElementType
// (and ONNX's onnx.TensorProto.DataType). DType internally casts (DType)(int)
// directly to these values, so the numeric encoding must stay aligned.
public enum ShorokooTensorElementType
{
    Float = 1,
    UInt8 = 2,
    Int8 = 3,
    UInt16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    UInt32 = 12,
    UInt64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16,
    Float8E4M3FN = 17,
    Float8E4M3FNUZ = 18,
    Float8E5M2 = 19,
    Float8E5M2FNUZ = 20,
    UInt4 = 21,
    Int4 = 22,
    DataTypeMax = 23,
}
