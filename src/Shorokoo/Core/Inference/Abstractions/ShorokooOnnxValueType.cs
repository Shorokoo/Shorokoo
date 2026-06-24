namespace Shorokoo.Core.Inference.Abstractions;

// Integer values mirror ORT's Microsoft.ML.OnnxRuntime.OnnxValueType so a
// platform implementation can cast between them without a translation table.
public enum ShorokooOnnxValueType
{
    Unknown = 0,
    Tensor = 1,
    Sequence = 2,
    Map = 3,
    Optional = 4,
}
