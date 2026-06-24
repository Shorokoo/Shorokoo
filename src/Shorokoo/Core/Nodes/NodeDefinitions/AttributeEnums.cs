using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using System.Diagnostics;
using System.Net.NetworkInformation;
using static Shorokoo.Core.InternalGlobals;
using static Shorokoo.Globals;
using System.Net;
using System.Collections.Immutable;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;

namespace Shorokoo;

// Member order must match the ONNX-name list in Definitions.DG.cs (["none", "tanh"]):
// NodeDefEnumDef maps positionally, and the spec default ("none") comes first by
// convention. (The order used to be { Tanh, None }, which serialized
// GeluApproximate.None as approximate="tanh" and vice versa.)
public enum GeluApproximate
{
    None,
    Tanh
}

public enum BitShiftDirection
{
    Left,
    Right
}

public enum AutoPad
{
    NotSet,
    SameUpper,
    SameLower,
    Valid
}

public enum RoundMode
{
    Floor,
    Ceiling
}

public enum DepthColumnRowMode
{
    DCR,
    CRD
}

public enum GRUDirection
{
    Forward,
    Reverse,
    Bidirectional
}

public enum RNNDirection
{
    Forward,
    Reverse,
    Bidirectional
}

public enum LSTMDirection
{
    Forward,
    Reverse,
    Bidirectional
}

public enum GridSampleMode
{
    Linear,
    Nearest,
    Cubic
}

public enum GridSamplePaddingMode
{
    Zeros,
    Border,
    Reflection
}

public enum PadMode
{
    Constant,
    Reflect,
    Edge,
    Wrap
}

public enum CoordinateTransformationMode
{
    Half_pixel,              // default
    Half_pixel_symmetric,
    Pytorch_half_pixel,
    Align_corners,
    Asymmetric,
    Tf_crop_and_resize
}

public enum KeepAspectRatioPolicy
{
    stretch,                 // default
    not_larger,
    not_smaller
}

public enum ResizeMode2
{
    Nearest,                 // default
    Linear,
    Cubic
}

public enum NearestMode
{
    Round_prefer_floor,      // default
    Round_prefer_ceil,
    Floor,
    Ceil
}

public enum ScatterNDReduction
{
    None,   // default
    Add,
    Mul,
    Max,
    Min
}

public enum RoiAlignMode
{
    Avg,
    Max
}

// RoiAlign's coordinate_transformation_mode only has two spec values
// ("half_pixel" / "output_half_pixel"); it previously reused the Resize
// CoordinateTransformationMode enum whose positional mapping turned
// Half_pixel_symmetric into "output_half_pixel" (and made members >= 2
// unserializable for RoiAlign).
public enum RoiAlignTransformationMode
{
    Half_pixel,              // default
    Output_half_pixel
}


public enum StorageOrder
{
    RowMajor,
    ColumnMajor
}

public enum Reduction
{
    None,
    Add,
    Mul,
    Max,
    Min
}

public enum ResizeTransformMode
{
    HalfPixel,
    HalfPixelAsymmetric,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TfCropAndResize
}

public enum ResizeAspectRatio
{
    Stretch,
    NotLarger,
    NotSmaller
}


public enum ResizeMode
{
    Nearest,
    Linear,
    Cubic
}

public enum ResizeNearestMode
{
    RoundHalfDown,
    RoundHalfUp,
    Floor,
    Ceiling
}

public enum ReduceKind
{
    Sum,
    L1,
    L2,
    LogSum,
    LogSumExp,
    Max,
    Mean,
    Min,
    Prod,
    SumSquare
}

// Member order must match the ONNX-name list in Definitions.TZ.cs (["linear", "circular"]).
public enum TensorScatterMode
{
    Linear,
    Circular
}