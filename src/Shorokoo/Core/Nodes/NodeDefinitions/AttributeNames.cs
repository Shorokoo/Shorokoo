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

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static class OnnxOpAttributeNames
{
    public const string AttrDirection = "direction";
    public const string AttrKeepdims = "keepdims";
    public const string AttrAxis = "axis";
    public const string AttrSelectLastIndex = "select_last_index";

    public const string AttrAutoPad = "auto_pad";
    public const string AttrCeilMode = "ceil_mode";
    public const string AttrCountIncludePad = "count_include_pad";
    public const string AttrDilations = "dilations";
    public const string AttrKernelShape = "kernel_shape";
    public const string AttrPads = "pads";
    public const string AttrStrides = "strides";

    public const string AttrEpsilon = "epsilon";
    public const string AttrMomentum = "momentum";
    public const string AttrTrainingMode = "training_mode";
    public const string AttrDtype = "dtype";
    public const string AttrSeed = "seed";
    public const string AttrHigh = "high";
    public const string AttrLow = "low";
    public const string AttrMean = "mean";
    public const string AttrScale = "scale";
    public const string AttrShape = "shape";
    public const string AttrOutputDatatype = "output_datatype";
    // QuantizeLinear's output-type attribute: the ONNX wire name is "output_dtype"
    // (unlike MelWeightMatrix / the window ops, whose spec name is "output_datatype").
    public const string AttrOutputDtype = "output_dtype";
    public const string AttrPeriodic = "periodic";

    public const string AttrK = "k";

    public const string AttrTo = "to";
    public const string AttrSaturate = "saturate";

    public const string AttrAlpha = "alpha";
    public const string AttrGamma = "gamma";
    public const string AttrApproximate = "approximate";

    public const string AttrP = "p";

    public const string AttrAlignCorners = "align_corners";
    public const string AttrAxes = "axes";
    public const string AttrExclusive = "exclusive";
    public const string AttrGroup = "group";
    public const string AttrOffsetGroup = "offset_group";
    public const string AttrNewAxis = "new_axis";
    public const string AttrOutputPadding = "output_padding";
    public const string AttrOutputShape = "output_shape";
    public const string AttrReverse = "reverse";
    public const string AttrSparseValue = "sparse_value";
    public const string AttrValue = "value";
    public const string AttrValueFloat = "value_float";
    public const string AttrValueFloats = "value_floats";
    public const string AttrValueInt = "value_int";
    public const string AttrValueInts = "value_ints";
    public const string AttrValueString = "value_string";
    public const string AttrValueStrings = "value_strings";

    public const string AttrInverse = "inverse";
    public const string AttrOnesided = "onesided";

    public const string AttrBlockSize = "block_size";
    public const string AttrBlocksize = "blocksize";

    public const string AttrEquation = "equation";

    public const string AttrActivationAlpha = "activation_alpha";
    public const string AttrActivationBeta = "activation_beta";
    public const string AttrActivations = "activations";
    public const string AttrClip = "clip";
    public const string AttrHiddenSize = "hidden_size";
    public const string AttrInputForget = "input_forget";
    public const string AttrLayout = "layout";
    public const string AttrLinearBeforeReset = "linear_before_reset";

    public const string AttrBatchDims = "batch_dims";

    public const string AttrPaddingMode = "padding_mode";

    public const string AttrNumGroups = "num_groups";
    public const string AttrStashType = "stash_type";

    public const string AttrMode = "mode";
    public const string AttrStorageOrder = "storage_order";
    public const string AttrCenterPointBox = "center_point_box";
    public const string AttrType = "type";
    public const string AttrNoopWithEmptyAxes = "noop_with_empty_axes";

    // From REVERSE_SEQUENCE
    public const string AttrBatchAxis = "batch_axis";
    public const string AttrTimeAxis = "time_axis";

    // From ROI_ALIGN
    public const string AttrOutputHeight = "output_height";
    public const string AttrOutputWidth = "output_width";
    public const string AttrSamplingRatio = "sampling_ratio";
    public const string AttrSpatialScale = "spatial_scale";

    // From MAX_ROI_POOL
    public const string AttrPooledShape = "pooled_shape";

    // From TOPK
    public const string AttrLargest = "largest";
    public const string AttrSorted = "sorted";

    // From GEMM
    public const string AttrBeta = "beta";
    public const string AttrTransA = "transA";
    public const string AttrTransB = "transB";

    // From TRILU
    public const string AttrUpper = "upper";

    // From LRN
    public const string AttrBias = "bias";
    public const string AttrSize = "size";

    // From SCATTER_ND
    public const string AttrReduction = "reduction";

    // From RESIZE
    public const string AttrAntialias = "antialias";
    public const string AttrCoordinateTransformationMode = "coordinate_transformation_mode";
    public const string AttrCubicCoeffA = "cubic_coeff_a";
    public const string AttrExcludeOutside = "exclude_outside";
    public const string AttrExtrapolationValue = "extrapolation_value";
    public const string AttrKeepAspectRatioPolicy = "keep_aspect_ratio_policy";
    public const string AttrNearestMode = "nearest_mode";

    public const string AttrAllowzero = "allowzero";
    public const string AttrStart = "start";
    public const string AttrEnd = "end";
    public const string AttrPerm = "perm";

    public const string AttrNumOutputs = "num_outputs";
    public const string AttrFmod = "fmod";

    public const string AttrMin = "min";

    // From IsInf
    public const string AttrDetectNegative = "detect_negative";
    public const string AttrDetectPositive = "detect_positive";

    // From Shrink
    public const string AttrLambd = "lambd";

    // From MelWeightMatrix
    public const string AttrNumMelBins = "num_mel_bins";
    public const string AttrDftLength = "dft_length";
    public const string AttrSampleRate = "sample_rate";
    public const string AttrLowerEdgeHertz = "lower_edge_hertz";
    public const string AttrUpperEdgeHertz = "upper_edge_hertz";

    // From Multinomial
    public const string AttrSampleSize = "sample_size";

    // From QuantizeLinear
    public const string AttrPrecision = "precision";
    public const string AttrRoundMode = "round_mode";

    // From RegexFullMatch
    public const string AttrPattern = "pattern";

    // From STFT — uses AttrOnesided which already exists

    // From SplitToSequence — uses AttrAxis, AttrKeepdims (both exist)

    // From StringNormalizer
    public const string AttrCaseChangeAction = "case_change_action";
    public const string AttrIsCaseSensitive = "is_case_sensitive";
    public const string AttrLocale = "locale";
    public const string AttrStopwords = "stopwords";

    // From StringSplit
    public const string AttrMaxsplit = "maxsplit";
    public const string AttrDelimiter = "delimiter";

    // From TfIdfVectorizer
    public const string AttrMaxGramLength = "max_gram_length";
    public const string AttrMaxSkipCount = "max_skip_count";
    public const string AttrMinGramLength = "min_gram_length";
    public const string AttrNgramCounts = "ngram_counts";
    public const string AttrNgramIndexes = "ngram_indexes";
    public const string AttrPoolInt64s = "pool_int64s";
    public const string AttrPoolStrings = "pool_strings";
    public const string AttrWeights = "weights";

    // From ImageDecoder
    public const string AttrPixelFormat = "pixel_format";

    // From NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss
    public const string AttrIgnoreIndex = "ignore_index";

    // From Scan
    public const string AttrNumScanInputs = "num_scan_inputs";
    public const string AttrScanInputAxes = "scan_input_axes";
    public const string AttrScanInputDirections = "scan_input_directions";
    public const string AttrScanOutputAxes = "scan_output_axes";
    public const string AttrScanOutputDirections = "scan_output_directions";

    // From Attention (opset 23/24)
    public const string AttrIsCausal = "is_causal";
    public const string AttrKvNumHeads = "kv_num_heads";
    public const string AttrQNumHeads = "q_num_heads";
    public const string AttrQkMatmulOutputMode = "qk_matmul_output_mode";
    public const string AttrSoftcap = "softcap";
    public const string AttrSoftmaxPrecision = "softmax_precision";

    // From RotaryEmbedding (opset 23)
    public const string AttrInterleaved = "interleaved";
    public const string AttrNumHeads = "num_heads";
    public const string AttrRotaryEmbeddingDim = "rotary_embedding_dim";

    public const string AttrElseBranch = "else_branch";
    public const string AttrThenBranch = "then_branch";
    public const string InternalAttrRank = "#rank#";
    public const string InternalAttrStructure = "#structure#";

    public const string InternalAttrHasOptionalOutputs = "#has_optional_outputs#";

    //public const string InternalAttrRank = "shrk_rank";
    //public const string InternalAttrTensorData = "shrk_tensor_data";
    //public const string InternalAttrStructure = "shrk_structure";
    //
    //public const string InternalAttrDtype = "shrk_dtype";
    //
    //public const string InternalAttrModuleDataIndex = "shrk_model_data_index";
    //
    //public const string InternalAttrModuleTypeName = "shrk_module_type_name";
    //
    //public const string InternalAttrFunctionName = "shrk_function_name";
    //public const string InternalAttrDomainName = "shrk_domain_name";
    //
    //public const string InternalAttrModelLocalId = "shrk_model_local_id";



    public const string ShrkAttrRank = "shrk_rank";
    public const string ShrkAttrShape = "shrk_shape";
    public const string ShrkAttrTensorData = "shrk_tensor_data";
    public const string ShrkAttrStructure = "shrk_structure";
    public const string ShrkAttrDtype = "shrk_dtype";
    public const string ShrkAttrFunctionName = "shrk_function_name";
    public const string ShrkAttrDomainName = "shrk_domain_name";
    public const string ShrkAttrLocalModelId = "shrk_local_model_id";

    /// <summary>The named RNG algorithm ("Threefry2x32-BoxMuller.v1") a SHRK_RNG_* op draws with.</summary>
    public const string ShrkAttrRngAlgorithm = "shrk_rng_algorithm";

    /// <summary>
    /// The resolved stream key ([k0, k1] 32-bit words as two longs) an RngConfig stamped on a
    /// SHRK_RANDOM_* feed. Pure metadata: stamping changes no graph structure, so a different
    /// config can re-stamp at any time; the ONNX-prep lowering reads it to emit the keyed
    /// deterministic draw (absent = ONNX random fallback).
    /// </summary>
    public const string ShrkAttrRngExplicitKey = "shrk_rng_explicit_key";
    public const string ShrkAttrRelativeModelId = "shrk_relative_model_id";
    public const string ShrkAttrInputType = "shrk_input_type";
    public const string ShrkAttrHyperparamIndex = "shrk_hyperparam_index";
    public const string ShrkAttrGenericTypeConstraints = "shrk_generic_type_constraints";
    public const string ShrkAttrGenericTypeArgs = "shrk_generic_type_args";
    public const string ShrkAttrIsTrainable = "shrk_is_trainable";
    public const string ShrkAttrFieldName = "shrk_field_name";

    /// <summary>
    /// Default value carried by a model-input node for a <c>[Hyper(defaultValue)]</c> parameter.
    /// Purely declarative metadata so the default survives serialization (ONNX / C#); the actual
    /// substitution-when-omitted is done by the source generator, which fills the omitted argument
    /// with <c>Scalar(defaultValue)</c>.
    /// </summary>
    public const string ShrkAttrDefaultValue = "shrk_default_value";



    public const string AttrBody = "body";

    // Metata Attribute Names
    public const string ShrkMetaNodeStackTrace = "StackTrace";
    public const string ShrkMetaIdentityNodeEthereal = "EtherealIdentity";
    public const string ShrkMetaNodeIdentifierTemplate = "IdentifierTemplate";
    public const string ShrkMetaIsTrainable = "IsTrainable";
    
    // NodeKey metadata - stores the GUID for stable node identification
    public const string ShrkMetaNodeKey = "NodeKey";
    
    // TensorStruct metadata prefix - used to store TensorStructDef in model metadata
    // Format: "shrk_tensorstruct_{ProtoTypeNum}" -> JSON representation of TensorStructDef
    public const string ShrkMetaTensorStructDefPrefix = "shrk_tensorstruct_";
}

