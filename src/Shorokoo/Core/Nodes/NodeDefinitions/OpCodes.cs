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
using System.Runtime.CompilerServices;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

internal static class InternalOpCodes
{
    public const string MODEL_OPTIONAL_INPUT = "#ModelOptionalInput#";
    public const string MODEL_TENSOR_INPUT = "#ModelTensorInput#";
    public const string MODEL_SEQUENCE_INPUT = "#ModelSequenceInput#";
    public const string GENERIC_TYPE_INPUT = "#GenericTypeInput#";
    public const string MODEL_PARAM_DATA = "#ModelParamData#";

    public const string MODULE_SET_HYPERPARAMS = "ShrkModuleSetHyperparams";
    // public const string READY_MODULE_TO_MODEL = "ShrkReadyModuleToModel";
    public const string MODEL_INVOKE = "ShrkModelInvoke";
    // public const string MODEL_AS_READY_MODULE = "ShrkModelAsReadyModule";
    public const string NEW_MODEL_LIKE = "ShrkNewModelLike";
    public const string GET_MODEL_ID = "ShrkGetModelId";

    public const string FUNCTION_INVOKE = "ShrkFunctionInvoke";

    public const string SUBMODEL = "ShrkSubModel#";
    public const string CREATE_MODULE = "ShrkCreateModule";

    public const string MODEL_HYPERPARAM = "ShrkModelHyperparam";

    public const string TRAINABLE_PARAM = "#TrainableParam#";
    public const string TRAINABLE_PARAM_REF = "#TrainableParamRef#";
    public const string TRAINABLE_PARAM_ID_REF = "#TrainableParamIdRef#";
    public const string TRAINABLE_PARAM_MODEL_REF = "#TrainableParamModelRef#";

    public const string SEQUENCE_CONCAT = "#SequenceConcat#";
    public const string SEQUENCE_SLICE = "#SequenceSlice#";

    public const string AUTO_GRAD = "#AutoGrad#";

    /// <summary>
    /// Marks the relationship between original and updated state tensors.
    /// Creates a traceable link between original and updated state in the graph.
    /// </summary>
    public const string STATE_UPDATE_LINK = "#StateUpdateLink#";

    /// <summary>
    /// Creates graph dependencies from updated state tensors to module output.
    /// Takes main output tensor (first input) + variadic updated state tensors.
    /// Returns only the first input (pass-through semantics).
    /// </summary>
    public const string WITH_STATE_DEPS = "#WithStateDeps#";

    // TensorStruct operations
    
    /// <summary>
    /// Input operation for TensorStruct - creates a model input that is a TensorStruct.
    /// </summary>
    public const string MODEL_TENSORSTRUCT_INPUT = "#ModelTensorStructInput#";

    /// <summary>
    /// Extracts a single field from a TensorStruct by field name.
    /// </summary>
    public const string TENSOR_STRUCT_GETFIELD = "shrk_TENSOR_STRUCT_GETFIELD";

    /// <summary>
    /// Generates a tensor filled with random values from a uniform distribution.
    /// Takes shape as a tensor input (unlike ONNX RandomUniform which takes shape as an attribute).
    /// Attributes: high (float), low (float), seed (float, optional).
    /// Lowered to ONNX ConstantOfShape + RandomUniformLike before execution.
    /// </summary>
    public const string SHRK_RANDOM_UNIFORM = "shrk_RandomUniform";

    /// <summary>
    /// Generates a tensor filled with random values from a normal distribution.
    /// Takes shape as a tensor input (unlike ONNX RandomNormal which takes shape as an attribute).
    /// Attributes: mean (float), scale (float), seed (float, optional).
    /// Lowered to ONNX ConstantOfShape + RandomNormalLike before execution.
    /// </summary>
    public const string SHRK_RANDOM_NORMAL = "shrk_RandomNormal";

    /// <summary>
    /// Creates a TensorStruct from multiple input IValues.
    /// </summary>
    public const string TENSOR_STRUCT_CREATE = "shrk_TENSOR_STRUCT_CREATE";

    /// <summary>
    /// Shorokoo-specific variant of ONNX <c>Conv</c> that takes its geometry
    /// (pads, strides, dilations, kernel_shape, group) as int64 tensor inputs
    /// instead of static attributes, so they can be computed in-graph. Lowered
    /// back to the standard ONNX <c>Conv</c> (with those values resolved to
    /// static attributes) by <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastLowerAttributeTensorOps"/>
    /// before autograd/execution/ONNX export. <c>auto_pad</c> stays a static attribute.
    /// </summary>
    public const string SHRK_CONV = "shrk_Conv";
}

internal static class OpCodes
{
    public const string ABS = "Abs";
    public const string ACOS = "Acos";
    public const string ACOSH = "Acosh";
    public const string ADD = "Add";
    public const string AFFINE_GRID = "AffineGrid";
    public const string AND = "And";
    public const string ARG_MAX = "ArgMax";
    public const string ARG_MIN = "ArgMin";
    public const string ASIN = "Asin";
    public const string ASINH = "Asinh";
    public const string ATAN = "Atan";
    public const string ATANH = "Atanh";
    public const string ATTENTION = "Attention";
    public const string AVERAGE_POOL = "AveragePool";
    public const string BATCH_NORMALIZATION = "BatchNormalization";
    public const string BERNOULLI = "Bernoulli";
    public const string BLACKMAN_WINDOW = "BlackmanWindow";
    public const string BIT_CAST = "BitCast";
    public const string BIT_SHIFT = "BitShift";
    public const string BITWISE_AND = "BitwiseAnd";
    public const string BITWISE_NOT = "BitwiseNot";
    public const string BITWISE_OR = "BitwiseOr";
    public const string BITWISE_XOR = "BitwiseXor";
    public const string CAST = "Cast";
    public const string CAST_LIKE = "CastLike";
    public const string CEIL = "Ceil";
    public const string CELU = "Celu";
    public const string CENTER_CROP_PAD = "CenterCropPad";
    public const string CLIP = "Clip";
    public const string COL2IM = "Col2Im";
    public const string COMPRESS = "Compress";
    public const string CONCAT = "Concat";
    public const string CONCAT_FROM_SEQUENCE = "ConcatFromSequence";
    public const string CONSTANT = "Constant";
    public const string CONSTANT_OF_SHAPE = "ConstantOfShape";
    public const string CONV = "Conv";
    public const string CONV_INTEGER = "ConvInteger";
    public const string CONV_TRANSPOSE = "ConvTranspose";
    public const string COS = "Cos";
    public const string COSH = "Cosh";
    public const string CUM_PROD = "CumProd";
    public const string CUM_SUM = "CumSum";
    public const string DFT = "DFT";
    public const string DEFORM_CONV = "DeformConv";
    public const string DEPTH_TO_SPACE = "DepthToSpace";
    public const string DEQUANTIZE_LINEAR = "DequantizeLinear";
    public const string DET = "Det";
    public const string DIV = "Div";
    public const string DROPOUT = "Dropout";
    public const string DYNAMIC_QUANTIZE_LINEAR = "DynamicQuantizeLinear";
    public const string EINSUM = "Einsum";
    public const string ELU = "Elu";
    public const string EQUAL = "Equal";
    public const string ERF = "Erf";
    public const string EXP = "Exp";
    public const string EXPAND = "Expand";
    public const string EYE_LIKE = "EyeLike";
    public const string FLATTEN = "Flatten";
    public const string FLOOR = "Floor";
    public const string GRU = "GRU";
    public const string GATHER = "Gather";
    public const string GATHER_ELEMENTS = "GatherElements";
    public const string GATHER_ND = "GatherND";
    public const string GEMM = "Gemm";
    public const string GELU = "Gelu";
    public const string GLOBAL_AVERAGE_POOL = "GlobalAveragePool";
    public const string GLOBAL_LP_POOL = "GlobalLpPool";
    public const string GLOBAL_MAX_POOL = "GlobalMaxPool";
    public const string GREATER = "Greater";
    public const string GREATER_OR_EQUAL = "GreaterOrEqual";
    public const string GRID_SAMPLE = "GridSample";
    public const string GROUP_NORMALIZATION = "GroupNormalization";
    public const string HAMMING_WINDOW = "HammingWindow";
    public const string HANN_WINDOW = "HannWindow";
    public const string HARD_SIGMOID = "HardSigmoid";
    public const string HARD_SWISH = "HardSwish";
    public const string HARDMAX = "Hardmax";
    public const string IDENTITY = "Identity";
    public const string IF = "If";
    public const string IF_OPEN = "If#OPEN";
    public const string IF_CLOSE = "If#CLOSE";
    public const string IMAGE_DECODER = "ImageDecoder";
    public const string INSTANCE_NORMALIZATION = "InstanceNormalization";
    public const string IS_INF = "IsInf";
    public const string IS_NAN = "IsNaN";
    public const string LRN = "LRN";
    public const string LSTM = "LSTM";
    public const string LAYER_NORMALIZATION = "LayerNormalization";
    public const string LEAKY_RELU = "LeakyRelu";
    public const string LESS = "Less";
    public const string LESS_OR_EQUAL = "LessOrEqual";
    public const string LOG = "Log";
    public const string LOG_SOFTMAX = "LogSoftmax";
    public const string LOOP = "Loop";
    public const string LOOP_OPEN = "Loop#OPEN";
    public const string LOOP_CLOSE = "Loop#CLOSE";
    public const string LOOP_FAKE_INPUT = "#LoopFakeInput#";
    public const string LOOP_SCAN_VARIABLE = "#LoopScanVariable#";
    public const string LOOP_INDEX_VARIABLE = "#LoopIndexVariable#";
    public const string LP_NORMALIZATION = "LpNormalization";
    public const string LP_POOL = "LpPool";
    public const string MATMUL = "MatMul";
    public const string MATMUL_INTEGER = "MatMulInteger";
    public const string MAX = "Max";
    public const string MAX_POOL = "MaxPool";
    public const string MAX_ROI_POOL = "MaxRoiPool";
    public const string MAX_UNPOOL = "MaxUnpool";
    public const string MEAN = "Mean";
    public const string MEAN_VARIANCE_NORMALIZATION = "MeanVarianceNormalization";
    public const string MEL_WEIGHT_MATRIX = "MelWeightMatrix";
    public const string MIN = "Min";
    public const string MISH = "Mish";
    public const string MOD = "Mod";
    public const string MUL = "Mul";
    public const string MULTINOMIAL = "Multinomial";
    public const string NEG = "Neg";
    public const string NEGATIVE_LOG_LIKELIHOOD_LOSS = "NegativeLogLikelihoodLoss";
    public const string NON_MAX_SUPPRESSION = "NonMaxSuppression";
    public const string NON_ZERO = "NonZero";
    public const string NOT = "Not";
    public const string ONE_HOT = "OneHot";
    public const string OPTIONAL = "Optional";
    public const string OPTIONAL_GET_ELEMENT = "OptionalGetElement";
    public const string OPTIONAL_HAS_ELEMENT = "OptionalHasElement";
    public const string OR = "Or";
    public const string P_RELU = "PRelu";
    public const string PAD = "Pad";
    public const string POW = "Pow";
    public const string QLINEAR_CONV = "QLinearConv";
    public const string QLINEAR_MATMUL = "QLinearMatMul";
    public const string QUANTIZE_LINEAR = "QuantizeLinear";
    public const string RMS_NORMALIZATION = "RMSNormalization";
    public const string RNN = "RNN";
    public const string RANDOM_NORMAL = "RandomNormal";
    public const string RANDOM_NORMAL_LIKE = "RandomNormalLike";
    public const string RANDOM_UNIFORM = "RandomUniform";
    public const string RANDOM_UNIFORM_LIKE = "RandomUniformLike";
    public const string RANGE = "Range";
    public const string RECIPROCAL = "Reciprocal";
    public const string REDUCE_L1 = "ReduceL1";
    public const string REDUCE_L2 = "ReduceL2";
    public const string REDUCE_LOG_SUM = "ReduceLogSum";
    public const string REDUCE_LOG_SUM_EXP = "ReduceLogSumExp";
    public const string REDUCE_MAX = "ReduceMax";
    public const string REDUCE_MEAN = "ReduceMean";
    public const string REDUCE_MIN = "ReduceMin";
    public const string REDUCE_PROD = "ReduceProd";
    public const string REDUCE_SUM = "ReduceSum";
    public const string REDUCE_SUM_SQUARE = "ReduceSumSquare";
    public const string REGEX_FULL_MATCH = "RegexFullMatch";
    public const string RELU = "Relu";
    public const string RESHAPE = "Reshape";
    public const string RESIZE = "Resize";
    public const string REVERSE_SEQUENCE = "ReverseSequence";
    public const string ROI_ALIGN = "RoiAlign";
    public const string ROTARY_EMBEDDING = "RotaryEmbedding";
    public const string ROUND = "Round";
    public const string STFT = "STFT";
    public const string SCAN = "Scan";
    public const string SCAN_OPEN = "Scan#OPEN";
    public const string SCAN_CLOSE = "Scan#CLOSE";
    public const string SCATTER_ELEMENTS = "ScatterElements";
    public const string SCATTER_ND = "ScatterND";
    public const string SELU = "Selu";
    public const string SEQUENCE_AT = "SequenceAt";
    public const string SEQUENCE_CONSTRUCT = "SequenceConstruct";
    public const string SEQUENCE_EMPTY = "SequenceEmpty";
    public const string SEQUENCE_ERASE = "SequenceErase";
    public const string SEQUENCE_INSERT = "SequenceInsert";
    public const string SEQUENCE_LENGTH = "SequenceLength";
    public const string SEQUENCE_MAP = "SequenceMap";
    public const string SEQUENCE_MAP_OPEN = "SequenceMap#OPEN";
    public const string SEQUENCE_MAP_CLOSE = "SequenceMap#CLOSE";
    public const string SHAPE = "Shape";
    public const string SHRINK = "Shrink";
    public const string SIGMOID = "Sigmoid";
    public const string SIGN = "Sign";
    public const string SIN = "Sin";
    public const string SINH = "Sinh";
    public const string SIZE = "Size";
    public const string SLICE = "Slice";
    public const string SOFTMAX = "Softmax";
    public const string SOFTMAX_CROSS_ENTROPY_LOSS = "SoftmaxCrossEntropyLoss";
    public const string SOFTPLUS = "Softplus";
    public const string SOFTSIGN = "Softsign";
    public const string SPACE_TO_DEPTH = "SpaceToDepth";
    public const string SPLIT = "Split";
    public const string SPLIT_TO_SEQUENCE = "SplitToSequence";
    public const string SQRT = "Sqrt";
    public const string SQUEEZE = "Squeeze";
    public const string STRING_CONCAT = "StringConcat";
    public const string STRING_NORMALIZER = "StringNormalizer";
    public const string STRING_SPLIT = "StringSplit";
    public const string SUB = "Sub";
    public const string SUM = "Sum";
    public const string SWISH = "Swish";
    public const string TAN = "Tan";
    public const string TANH = "Tanh";
    public const string TENSOR_SCATTER = "TensorScatter";
    public const string TFIDF_VECTORIZER = "TfIdfVectorizer";
    public const string THRESHOLDED_RELU = "ThresholdedRelu";
    public const string TILE = "Tile";
    public const string TOPK = "TopK";
    public const string TRANSPOSE = "Transpose";
    public const string TRILU = "Trilu";
    public const string UNIQUE = "Unique";
    public const string UNSQUEEZE = "Unsqueeze";
    public const string UPSAMPLE = "Upsample";
    public const string WHERE = "Where";
    public const string XOR = "Xor";
}