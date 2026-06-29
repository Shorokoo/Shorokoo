using System.Collections.Generic;
using System.Diagnostics;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;
using static Shorokoo.Globals;
using System.Reflection;
using System.Linq;
using System;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    internal class AutoDiffAttribute : Attribute
    {
        public string OpName { get; set; }
        public AutoDiffAttribute(string opName) { OpName = opName; }
    }

    internal static partial class AutoDiffs
    {
        public static Tensor<T> ReverseBroadcast<T>(Tensor<T> broadcastedInputGrads, Vector<int64> originalShape) where T : IVarType
        {
            var broadcastedShape = broadcastedInputGrads.DShape;
            var broadcastedRank = broadcastedShape.DShape.Squeeze();
            var originalRank = originalShape.DShape.Squeeze();

            var padding = broadcastedRank - originalRank;

            var paddedInputShape = originalShape.Pad(PadMode.Constant, pads: [padding, Scalar(0L)], val: Scalar(1L));
            var indices = VectorRange(Scalar(0L), broadcastedRank, Scalar(1L));
            var axesToReduce = ((Tensor<int64>)indices).Compress((paddedInputShape == 1) & (broadcastedShape != 1));

            // Use noopWithEmptyAxes=true so that when no axes need reducing (e.g., same-shape
            // operands), the tensor passes through unchanged instead of reducing all axes.
            var summedGrad = NN.Reduce(ReduceKind.Sum, broadcastedInputGrads, axes: axesToReduce, keepDims: true, noOp: true);
            var finalGrad = summedGrad.Reshape(originalShape, allowZero: true);

            return finalGrad;
        }

        [AutoDiff(ADD)]
        public static Variable?[] Add<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
        {
            var aGrad = ReverseBroadcast(grad, a.DShape);
            var bGrad = ReverseBroadcast(grad, b.DShape);

            return [aGrad, bGrad];
        }

        [AutoDiff(MUL)]
        public static Variable?[] Mul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> grad) where T : IVarType
        {
            var aGrad = ReverseBroadcast(grad * b, a.DShape);
            var bGrad = ReverseBroadcast(grad * a, b.DShape);

            return [aGrad, bGrad];
        }

        [AutoDiff(CAST)]
        public static Variable? Cast<TFrom, TTo>(Tensor<TFrom> input, Tensor<TTo> grad) 
            where TFrom : IVarType 
            where TTo : IVarType
        {
            // For Cast operation, the gradient flows back through a cast in the opposite direction
            var inputGrad = grad.Cast<TFrom>();
            
            return inputGrad;
        }

        [AutoDiff(CAST_LIKE)]
        public static Variable?[] CastLike<TFrom, TTo>(Tensor<TFrom> input, Tensor<TTo> targetType, Tensor<TTo> grad, bool? saturate)
            where TFrom : IVarType
            where TTo : IVarType
        {
            // CastLike casts input to the same type as targetType
            // Gradient flows back through a cast in the opposite direction
            var inputGrad = grad.Cast<TFrom>();
            return [inputGrad, null];
        }

        [AutoDiff(CONSTANT_OF_SHAPE)]
        public static Variable? ConstantOfShape<T>(Tensor<int64> shape, Tensor<T> grad) where T : IVarType
        {
            // The only input is `shape` (int64), which is non-differentiable.
            // The constant `value` is a fixed attribute, not a learnable input.
            // Returning null stops gradient propagation here.
            return null;
        }

        private static bool IsGenericTypeOrContainsGeneric(Type type, Type genericParam)
        {
            if (type == genericParam)
                return true;

            if (type.IsGenericType)
            {
                foreach (var arg in type.GetGenericArguments())
                {
                    if (IsGenericTypeOrContainsGeneric(arg, genericParam))
                        return true;
                }
            }

            return false;
        }


        private static Variable?[] CallGradientOps(MethodInfo gradientOp, Variable?[] inputs, Variable?[] outputs, OnnxCSharpAttributes attributes)
        {
            // MethodInfo input params has the following structure:
            // (input0, input1, ..., outputGrad0, outputGrad1, ..., attr0, attr1, ...)
            // The attributes are in alphabetical order of their names.
            // Inputs and outputs may use generic parameters as seen in the Add method above.
            // A single method may have multiple generic parameters.
            // We need to construct the generic method with the correct types before invoking it.

            // The leading graph-input parameters are everything before the output-grad params and the
            // trailing attribute (non-Variable) params. The forward op may have omitted trailing optional
            // inputs (e.g. Trilu's diagonal), so derive the expected input-parameter count from the method
            // signature; the provided inputs are padded with nulls below to keep the input/output-grad/
            // attribute slots aligned. Computed up front (off the generic definition's parameters —
            // IsAssignableFrom recognises an open-generic Tensor<T> as an IValue) so the type-inference
            // loop below classifies params against the same boundary as the value-filling loop.
            var sigParams = gradientOp.GetParameters();
            int attrParamCount = 0;
            for (int p = sigParams.Length - 1; p >= 0; p--)
            {
                var pt = Nullable.GetUnderlyingType(sigParams[p].ParameterType) ?? sigParams[p].ParameterType;
                if (typeof(Variable).IsAssignableFrom(pt) || typeof(IValue).IsAssignableFrom(pt))
                    break;
                attrParamCount++;
            }
            int expectedInputCount = Math.Max(inputs.Length, sigParams.Length - outputs.Length - attrParamCount);

            // Step 1: Determine the generic type arguments from the inputs or outputs
            Type[] genericTypeArgs = Array.Empty<Type>();
            if (gradientOp.IsGenericMethodDefinition)
            {
                var genericParams = gradientOp.GetGenericArguments();
                genericTypeArgs = new Type[genericParams.Length];

                // Try to infer each generic parameter type from method parameters
                for (int i = 0; i < genericParams.Length; i++)
                {
                    var genericParam = genericParams[i];
                    Type? inferredType = null;

                    // Look through method parameters to find one that uses this generic parameter
                    for (int paramIdx = 0; paramIdx < sigParams.Length; paramIdx++)
                    {
                        var paramType = sigParams[paramIdx].ParameterType;

                        // Check if this parameter type uses the current generic parameter
                        if (IsGenericTypeOrContainsGeneric(paramType, genericParam))
                        {
                            // Determine which input / output-grad slot this corresponds to. Use
                            // expectedInputCount (not inputs.Length) as the boundary so an omitted
                            // trailing optional input doesn't misclassify an output-grad parameter.
                            Variable? var = null;
                            if (paramIdx < expectedInputCount)
                                var = paramIdx < inputs.Length ? inputs[paramIdx] : null;
                            else if (paramIdx < expectedInputCount + outputs.Length)
                                var = outputs[paramIdx - expectedInputCount];

                            if (var != null)
                            {
                                inferredType = var.Type.ToIVarType();
                                break;
                            }
                        }
                    }

                    genericTypeArgs[i] = inferredType ?? typeof(int64); // Default to int64 for unresolvable types (e.g., null optional axes)
                }

                // Make the generic method concrete
                gradientOp = gradientOp.MakeGenericMethod(genericTypeArgs);
            }

            // Step 2: Build the parameter list
            var methodParams = gradientOp.GetParameters();
            var paramValues = new object?[methodParams.Length];

            // Validate parameter counts match expectations
            // Internal contract: every registered gradient method has a parameter count
            // between (inputs+outputs) and (inputs+outputs+attributes). Violations indicate
            // a misregistered AutoDiff entry, not a runtime/user error.
            int minExpectedParamCount = expectedInputCount + outputs.Length;
            int maxExpectedParamCount = minExpectedParamCount + attributes.AttributeDefs.Count;
            Debug.Assert(methodParams.Length >= minExpectedParamCount,
                $"Method {gradientOp.Name} expects at least {minExpectedParamCount} parameters " +
                $"(inputs: {expectedInputCount}, outputs: {outputs.Length}) but only has {methodParams.Length} parameters");
            Debug.Assert(methodParams.Length <= maxExpectedParamCount,
                $"Method {gradientOp.Name} expects {methodParams.Length} parameters but only {maxExpectedParamCount} can be provided " +
                $"(inputs: {expectedInputCount}, outputs: {outputs.Length}, attributes: {attributes.AttributeDefs.Count})");

            int paramIndex = 0;

            // Add inputs (converting each Variable to the value-struct IValue the parameter expects —
            // reflective Invoke does not apply the Variable→IValue implicit conversion), padding absent
            // trailing optional inputs with null.
            for (int j = 0; j < expectedInputCount; j++)
            {
                var inputVal = j < inputs.Length ? inputs[j] : null;
                paramValues[paramIndex] = inputVal?.ToValue(methodParams[paramIndex].ParameterType);
                paramIndex++;
            }

            // Add output gradients (same conversion).
            foreach (var output in outputs)
            {
                paramValues[paramIndex] = output?.ToValue(methodParams[paramIndex].ParameterType);
                paramIndex++;
            }
            
            // Add attributes in alphabetical order
            var attrDefs = attributes.AttributeDefs.OrderBy(x => x.AttributeName).ToArray();
            foreach (var attrDef in attrDefs)
            {
                if (paramIndex >= paramValues.Length)
                    break;
                    
                var attrValue = attributes.GetAttributeObj(attrDef.AttributeName);
                paramValues[paramIndex++] = attrValue;
            }
            
            // Step 3: Invoke the method. Unwrap reflection's TargetInvocationException so
            // deliberate gradient-side guards (e.g. AD003 AutoDiffNotSupportedException for
            // unsupported attribute combinations) surface as their original exception type.
            object? result;
            try
            {
                result = gradientOp.Invoke(null, paramValues);
            }
            catch (TargetInvocationException tie) when (tie.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(tie.InnerException).Throw();
                throw; // unreachable
            }

            // Step 4: Convert the result to Variable?[]. Gradient methods return either
            //   - Variable?[] (most cases) — pass straight through, including a non-null
            //     array containing nulls
            //   - Variable? — handle both a non-null Variable and the all-inputs-non-
            //     differentiable case (e.g. ConstantOfShape returns null).
            if (result is null)
                return [];
            if (result is Variable?[] array)
                return array;
            Debug.Assert(result is Variable,
                $"Unexpected gradient return type: {result.GetType()}");
            return [(Variable)result];
        }

        public static Dictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>> GetGradientOps()
        {
            var retval = new Dictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>>();

            // Use reflection to find all methods with AutoDiffAttribute
            var autoDiffMethods = typeof(AutoDiffs).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.GetCustomAttribute<AutoDiffAttribute>() != null)
                .ToArray();

            foreach (var method in autoDiffMethods)
            {
                var attribute = method.GetCustomAttribute<AutoDiffAttribute>().AssertNotNull();
                var opCode = attribute.OpName;
                var methodInfo = method;
                retval[opCode] = (inputs, outputs, attributes) => CallGradientOps(methodInfo, inputs, outputs, attributes);
            }

            // Register variadic gradient ops that cannot use the [AutoDiff] reflection pattern
            // due to variable input count (e.g., Concat accepts N tensors to concatenate)
            RegisterVariadicGradientOps(retval);

            return retval;
        }

        // ===== Concat (variadic) =====

        private static void RegisterVariadicGradientOps(
            Dictionary<string, Func<Variable?[], Variable?[], OnnxCSharpAttributes, Variable?[]>> retval)
        {
            retval[CONCAT] = ConcatGradient;
            retval[SPLIT] = SplitGradient;
            retval[BATCH_NORMALIZATION] = BatchNormalizationGradient;
            retval[MAX] = MaxGradient;
            retval[MIN] = MinGradient;
            retval[SUM] = SumGradient;
            retval[MEAN] = MeanGradient;
            retval[DROPOUT] = DropoutGradient;
            retval[CONV] = ConvGradient;
            retval[CONV_TRANSPOSE] = ConvTransposeGradient;
            retval[AVERAGE_POOL] = AveragePoolGradient;
            retval[MAX_POOL] = MaxPoolGradient;
            retval[RESIZE] = ResizeGradient;
            retval[UPSAMPLE] = UpsampleGradient;
            retval[DET] = DetGradient;
            retval[EINSUM] = EinsumGradient;
            retval[GRU] = GruGradient;
            retval[RNN] = RnnGradient;
            retval[LSTM] = LstmGradient;
            retval[TOPK] = TopKGradient;
            retval[UNIQUE] = UniqueGradient;
            retval[LP_POOL] = LpPoolGradient;
            retval[MAX_UNPOOL] = MaxUnpoolGradient;
            retval[DFT] = DftGradient;
            retval[GRID_SAMPLE] = GridSampleGradient;
            retval[ROI_ALIGN] = RoiAlignGradient;
            retval[MAX_ROI_POOL] = MaxRoiPoolGradient;
            retval[OPTIONAL] = OptionalGradient;
            retval[OPTIONAL_GET_ELEMENT] = OptionalGetElementGradient;

            // Control flow
            // IfCloseGradient now takes a typed Scalar<bit> cond as its first arg so the
            // FastProcessAutoGrad slot-dtype resolver can synthesize a Scalar<bit> stand-in
            // automatically. The flat (cond, else..., then...) layout from the autograd
            // engines is preserved by splitting inputs[0] from inputs[1..] here.
            retval[IF_CLOSE] = (inputs, outputGrads, attrs) =>
                IfCloseGradient((Variable)inputs[0]!, inputs.Skip(1).ToArray(), outputGrads, attrs);

            // Sequence operations
            retval[SEQUENCE_CONSTRUCT] = SequenceConstructGradient;
            retval[SEQUENCE_AT] = SequenceAtGradient;
            retval[SEQUENCE_INSERT] = SequenceInsertGradient;
            retval[SEQUENCE_ERASE] = SequenceEraseGradient;
            retval[CONCAT_FROM_SEQUENCE] = ConcatFromSequenceGradient;

            // Multi-output / multi-input differentiable ops that don't fit the [AutoDiff]
            // reflection shape (see AutoDiffs.Batch30.cs).
            retval[LAYER_NORMALIZATION] = LayerNormalizationGradient;
            retval[NEGATIVE_LOG_LIKELIHOOD_LOSS] = NegativeLogLikelihoodLossGradient;
            retval[SOFTMAX_CROSS_ENTROPY_LOSS] = SoftmaxCrossEntropyLossGradient;
            retval[STFT] = STFTGradient;
            retval[DEFORM_CONV] = DeformConvGradient;
            retval[SPLIT_TO_SEQUENCE] = SplitToSequenceGradient;

            // Opset 23/24 ops whose adjoints are not implemented: registered AD003
            // guards (DeformConv pattern, see AutoDiffs.Batch31.cs) so a loss→param
            // path through them fails with the op-specific message instead of the
            // engine's generic unregistered-op AD003.
            retval[ATTENTION] = AttentionGradient;
            retval[ROTARY_EMBEDDING] = RotaryEmbeddingGradient;
            retval[TENSOR_SCATTER] = TensorScatterGradient;

            // Non-differentiable / structural ops whose inputs receive no gradient.
            // Each just returns nulls of the right arity so the autograd dispatcher
            // doesn't throw NotImplementedException.
            retval[BIT_CAST] = NullInputGradient;  // bitwise reinterpretation — non-differentiable
            retval[CONV_INTEGER] = NullInputGradient;
            retval[DEQUANTIZE_LINEAR] = NullInputGradient;
            retval[DYNAMIC_QUANTIZE_LINEAR] = NullInputGradient;
            retval[HAMMING_WINDOW] = NullInputGradient;
            retval[HANN_WINDOW] = NullInputGradient;
            retval[IMAGE_DECODER] = NullInputGradient;
            retval[MATMUL_INTEGER] = NullInputGradient;
            retval[MEL_WEIGHT_MATRIX] = NullInputGradient;
            retval[NON_MAX_SUPPRESSION] = NullInputGradient;
            retval[OPTIONAL_HAS_ELEMENT] = NullInputGradient;
            retval[QLINEAR_CONV] = NullInputGradient;
            retval[QLINEAR_MATMUL] = NullInputGradient;
            retval[QUANTIZE_LINEAR] = NullInputGradient;
            retval[REGEX_FULL_MATCH] = NullInputGradient;
            retval[SEQUENCE_LENGTH] = NullInputGradient;
            retval[STRING_CONCAT] = NullInputGradient;
            retval[STRING_NORMALIZER] = NullInputGradient;
            retval[STRING_SPLIT] = NullInputGradient;
            retval[TFIDF_VECTORIZER] = NullInputGradient;

            // Shorokoo-internal sentinel ops that participate in graph wiring but carry no
            // float gradient.
            retval[InternalOpCodes.STATE_UPDATE_LINK] = NullInputGradient;
            retval[InternalOpCodes.WITH_STATE_DEPS] = NullInputGradient;
            retval[InternalOpCodes.TRAINABLE_PARAM_ID_REF] = NullInputGradient;
            retval[InternalOpCodes.SEQUENCE_CONCAT] = NullInputGradient;
            retval[InternalOpCodes.SEQUENCE_SLICE] = NullInputGradient;
        }

        private static Variable?[] ConcatGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Concat(inputs, axis) → output
            // Gradient: split the output gradient along axis according to each input's size
            var axis = (long)attributes.GetAttributeObj("axis")!;
            var grad = outputGrads[0]!;

            // Compute split sizes from each input's dimension along the concat axis
            var splitSizeList = new List<Variable>();
            foreach (var input in inputs)
            {
                if (input is null) continue;
                var shape = OnnxOp.Shape(input);
                var dimSize = OnnxOp.Gather(shape, Scalar(axis), axis: 0);
                splitSizeList.Add(OnnxOp.Unsqueeze(dimSize, Vector(0L)));
            }
            var splits = OnnxOp.Concat(splitSizeList.ToArray(), axis: 0);

            // Split gradient along the concat axis
            var nonNullCount = inputs.Count(i => i is not null);
            var gradSplits = OnnxOp.Split(grad, splits, axis: axis, numOutputs: null,
                variadicOutputCount: nonNullCount);

            // Return one gradient per input (null for null inputs)
            var result = new Variable?[inputs.Length];
            var splitIdx = 0;
            for (var i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is not null)
                    result[i] = gradSplits[splitIdx++];
            }

            return result;
        }

        private static Variable?[] SplitGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Split(input, split?, axis) → [output_0, output_1, ..., output_N]
            // Gradient: concatenate all output gradients along the split axis. Outputs that
            // never reached the loss have a null gradient — those slots must contribute a
            // zero block of the matching split size (dropping them would mis-place every
            // later split's gradient along the axis), so we re-split the forward input and
            // substitute Sub(piece, piece) zeros for each missing gradient.
            var axis = (long)attributes.GetAttributeObj("axis")!;

            // outputGrads has at least one non-null entry: FastProcessAutoGrad's all-null
            // skip elides the gradient method before this point.
            Debug.Assert(outputGrads.Any(g => g is not null));

            Variable gradInput;
            if (outputGrads.All(g => g is not null))
            {
                gradInput = OnnxOp.Concat(outputGrads!, axis: axis);
            }
            else
            {
                var split = inputs.Length > 1 ? inputs[1] : null;
                var numOutputs = attributes.GetAttributeObj("num_outputs") as long?;
                var pieces = OnnxOp.Split(inputs[0]!, split, axis: axis,
                    numOutputs: numOutputs, variadicOutputCount: outputGrads.Length);

                var parts = new Variable[outputGrads.Length];
                for (var i = 0; i < outputGrads.Length; i++)
                    parts[i] = outputGrads[i] ?? OnnxOp.Sub(pieces[i], pieces[i]);
                gradInput = OnnxOp.Concat(parts, axis: axis);
            }

            // Return gradient for input (first param), null for split sizes (second param)
            return [gradInput, null];
        }

        private static Variable?[] BatchNormalizationGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // BatchNormalization(x, scale, b, inputMean, inputVar) → y [, runningMean, runningVar]
            //
            // Inference mode (training_mode=0): y = scale * (x - input_mean) / sqrt(input_var + eps) + bias
            //   dx = grad * scale / sqrt(var + eps)
            //   dscale = sum(grad * (x - mean) / sqrt(var + eps)) over batch and spatial dims
            //   dbias = sum(grad) over batch and spatial dims
            //   dmean = null, dvar = null (not trainable inputs)
            //
            // Training mode (training_mode=1): y normalizes by the CURRENT batch statistics
            //   μ_B = mean(x), σ²_B = mean((x-μ_B)²) over batch+spatial per channel (the
            //   biased/population variance, per spec). Since ONNX ≥14 no longer exposes
            //   saved_mean/saved_inv_std as outputs, the gradient recomputes them from x.
            //   Standard training BN backward (with x̂ = (x-μ_B)·invStd, gy = grad·scale):
            //     dx = invStd · (gy − mean(gy) − x̂ · mean(gy·x̂))
            //   dscale/dbias use the batch-stat x̂ in the same reductions as inference mode.
            //   input_mean/input_var (the running stats) only feed the running-stat
            //   OUTPUTS in training mode, so their input gradients stay null.
            //
            // Gradients flowing back into the running-stat outputs (outputs 1/2) would
            // chain into x through μ_B/σ²_B with extra momentum-scaled terms that are NOT
            // modeled here — guarded below (consuming the running stats in a loss is
            // exotic; no silent wrongness).
            if (outputGrads.Skip(1).Any(g => g is not null))
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, BATCH_NORMALIZATION,
                    "gradients flowing into BatchNormalization's running-stat outputs "
                    + "(running_mean/running_var) are not implemented — only the Y output "
                    + "is differentiable. This is an implementation limitation, not a "
                    + "mathematical one — detach the running stats from the loss path.");

            var trainingModeAttr = attributes.GetAttributeObj("training_mode");
            var isTrainingMode = trainingModeAttr is true
                || (trainingModeAttr is long tmLong && tmLong != 0);

            var x = inputs[0]!;
            var scale = inputs[1]!;
            var inputMean = inputs[3]!;
            var inputVar = inputs[4]!;
            var grad = outputGrads[0]!;

            var epsilon = attributes.GetAttributeObj("epsilon") as float? ?? 1e-5f;
            var epsConst = OnnxOp.Cast(Scalar(epsilon), saturate: null, to: x.Type);

            // Build broadcast shape [1, C, 1, 1, ...] matching x's rank for channel tensors
            var xShape = OnnxOp.Shape(x);                                                    // [N, C, H, W, ...]
            var xRank = OnnxOp.Shape(xShape);                                                // [1] containing rank
            var onesShape = OnnxOp.Expand(Scalar(1L), xRank);                                // [1, 1, 1, ...] (rank elements)
            var cDim = OnnxOp.Slice(xShape, Vector(1L), Vector(2L));                          // [1] containing C
            var scatterIdx = OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);    // [[1]]
            var broadcastShape = OnnxOp.ScatterND(onesShape, scatterIdx, cDim);               // [1, C, 1, 1, ...]

            // Reshape 1-d channel tensors to broadcast shape
            var scaleBC = OnnxOp.Reshape(scale, broadcastShape, allowZero: false);

            // Build reduce axes: [0, 2, 3, ..., rank-1] (all except channel dim 1)
            var xRankScalar = OnnxOp.Squeeze(xRank, null);  // [1] → scalar: squeeze single-element vector
            var allAxes = OnnxOp.Range(Scalar(0L), xRankScalar, Scalar(1L));
            var axis0 = OnnxOp.Slice(allAxes, Vector(0L), Vector(1L));
            var axesSuffix = OnnxOp.Slice(allAxes, Vector(2L), xRank);  // xRank is [1] tensor with rank as value = end index
            var reduceAxes = OnnxOp.Concat([axis0, axesSuffix], axis: 0);

            // Normalization statistics: the running stats (inputs) in inference mode, the
            // recomputed CURRENT batch statistics in training mode (biased variance).
            Variable meanBC, varBC;
            if (isTrainingMode)
            {
                meanBC = OnnxOp.ReduceMean(x, reduceAxes, keepdims: true);             // [1,C,1,...]
                var xcStat = OnnxOp.Sub(x, meanBC);
                varBC = OnnxOp.ReduceMean(OnnxOp.Mul(xcStat, xcStat), reduceAxes, keepdims: true);
            }
            else
            {
                meanBC = OnnxOp.Reshape(inputMean, broadcastShape, allowZero: false);
                varBC = OnnxOp.Reshape(inputVar, broadcastShape, allowZero: false);
            }

            // inv_std = 1 / sqrt(var + eps)
            var invStd = OnnxOp.Reciprocal(OnnxOp.Sqrt(OnnxOp.Add(varBC, epsConst)));

            // normalized_x = (x - mean) * inv_std
            var normalizedX = OnnxOp.Mul(OnnxOp.Sub(x, meanBC), invStd);

            Variable gradX;
            if (isTrainingMode)
            {
                // Training mode: μ_B/σ²_B depend on x, so the backprop carries the
                // mean-subtraction and variance terms (standard BN training backward):
                //   dx = invStd · (gy − mean(gy) − x̂ · mean(gy·x̂)),  gy = grad·scale
                var gy = OnnxOp.Mul(grad, scaleBC);
                var meanGy = OnnxOp.ReduceMean(gy, reduceAxes, keepdims: true);
                var meanGyXhat = OnnxOp.ReduceMean(OnnxOp.Mul(gy, normalizedX), reduceAxes, keepdims: true);
                gradX = OnnxOp.Mul(invStd,
                    OnnxOp.Sub(
                        OnnxOp.Sub(gy, meanGy),
                        OnnxOp.Mul(normalizedX, meanGyXhat)));
            }
            else
            {
                // Inference mode: the stats are constants → dx = grad * scale_BC * inv_std
                gradX = OnnxOp.Mul(OnnxOp.Mul(grad, scaleBC), invStd);
            }

            // dscale = sum(grad * normalized_x) over batch + spatial dims → [C]
            var gradScale = OnnxOp.ReduceSum(OnnxOp.Mul(grad, normalizedX), reduceAxes, keepdims: false);

            // dbias = sum(grad) over batch + spatial dims → [C]
            var gradBias = OnnxOp.ReduceSum(grad, reduceAxes, keepdims: false);

            // Return: [dx, dscale, dbias, dmean=null, dvar=null]
            return [gradX, gradScale, gradBias, null, null];
        }

        // ===== Max (variadic) =====

        private static Variable?[] MaxGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Max(x0, x1, ..., xN) → element-wise maximum across inputs
            // Gradient flows to the input(s) that achieved the max value.
            // If there are ties, gradient is shared equally among all max elements.
            var grad = outputGrads[0]!;

            // Compute max output by calling Max on all non-null inputs. NodeBuilder rejects
            // variadic ops with zero non-null operands, so we always have at least one.
            var nonNullInputs = inputs.Where(i => i is not null).Select(i => i!).ToArray();
            Debug.Assert(nonNullInputs.Length > 0);

            var maxVal = OnnxOp.Max(nonNullInputs);

            // For each input, create a mask where input == max, then normalize by tie count
            var masks = new Variable?[inputs.Length];
            Variable? totalMask = null;
            for (var i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is null) continue;
                var mask = OnnxOp.Cast(OnnxOp.Equal(inputs[i]!, maxVal), saturate: null, to: inputs[i]!.Type);
                masks[i] = mask;
                totalMask = totalMask is null ? mask : OnnxOp.Add(totalMask, mask);
            }

            // Compute gradient for each input: mask / totalMask * grad
            var result = new Variable?[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                if (masks[i] is null) continue;
                var normalizedMask = OnnxOp.Div(masks[i]!, totalMask!);
                result[i] = ReverseBroadcast((Tensor<float32>)(OnnxOp.Mul(normalizedMask, grad)),
                    ((Variable)inputs[i]!).DShape);
            }

            return result;
        }

        // ===== Min (variadic) =====

        private static Variable?[] MinGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Min(x0, x1, ..., xN) → element-wise minimum across inputs
            // Gradient flows to the input(s) that achieved the min value.
            // If there are ties, gradient is shared equally among all min elements.
            var grad = outputGrads[0]!;

            // Compute min output by calling Min on all non-null inputs. NodeBuilder rejects
            // variadic ops with zero non-null operands, so we always have at least one.
            var nonNullInputs = inputs.Where(i => i is not null).Select(i => i!).ToArray();
            Debug.Assert(nonNullInputs.Length > 0);

            var minVal = OnnxOp.Min(nonNullInputs);

            // For each input, create a mask where input == min, then normalize by tie count
            var masks = new Variable?[inputs.Length];
            Variable? totalMask = null;
            for (var i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is null) continue;
                var mask = OnnxOp.Cast(OnnxOp.Equal(inputs[i]!, minVal), saturate: null, to: inputs[i]!.Type);
                masks[i] = mask;
                totalMask = totalMask is null ? mask : OnnxOp.Add(totalMask, mask);
            }

            // Compute gradient for each input: mask / totalMask * grad
            var result = new Variable?[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                if (masks[i] is null) continue;
                var normalizedMask = OnnxOp.Div(masks[i]!, totalMask!);
                result[i] = ReverseBroadcast((Tensor<float32>)(OnnxOp.Mul(normalizedMask, grad)),
                    ((Variable)inputs[i]!).DShape);
            }

            return result;
        }

        // ===== Sum (variadic) =====

        private static Variable?[] SumGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Sum(x0, x1, ..., xN) → element-wise sum across inputs (with broadcasting)
            // Gradient: each input receives grad, reduced to its original shape
            var grad = outputGrads[0]!;

            var result = new Variable?[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is null) continue;
                result[i] = ReverseBroadcast(
                    (Tensor<float32>)grad,
                    ((Variable)inputs[i]!).DShape);
            }

            return result;
        }

        // ===== Mean (variadic) =====

        private static Variable?[] MeanGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Mean(x0, x1, ..., xN) → element-wise mean across inputs (with broadcasting)
            // Gradient: each input receives grad / N, reduced to its original shape
            var grad = outputGrads[0]!;

            // NodeBuilder rejects variadic ops with zero non-null operands.
            var nonNullCount = inputs.Count(i => i is not null);
            Debug.Assert(nonNullCount > 0);

            var nConst = OnnxOp.Cast(Scalar((float)nonNullCount), saturate: null, to: inputs.First(i => i is not null)!.Type);
            var scaledGrad = OnnxOp.Div(grad, nConst);

            var result = new Variable?[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is null) continue;
                result[i] = ReverseBroadcast(
                    (Tensor<float32>)scaledGrad,
                    ((Variable)inputs[i]!).DShape);
            }

            return result;
        }

        // ===== Dropout (variadic registration for multi-output handling) =====

        /// <summary>
        /// Stand-in type pinning only — never dispatched (the dict registration of
        /// <see cref="DropoutGradient"/> overrides it, mirroring the IF_CLOSE pattern).
        /// FastProcessAutoGrad reads this signature's parameter types so the
        /// <c>training_mode</c> input gets a <c>Scalar&lt;bit&gt;</c> stand-in and the
        /// plumbed-through forward <c>mask</c> output (appended as trailing input slot 3)
        /// gets a <c>Tensor&lt;bit&gt;</c> stand-in.
        /// </summary>
        [AutoDiff(DROPOUT)]
        public static Variable?[] DropoutGradientStandInTypes(
            Tensor<float32> data, Tensor<float32>? ratio, Scalar<bit>? trainingMode, Tensor<bit>? mask,
            Tensor<float32> gradOutput, Tensor<bit>? gradMask, long? seed)
            => throw new InvalidOperationException(
                "DropoutGradientStandInTypes exists only for stand-in dtype resolution; "
                + "dispatch goes through the dict-registered DropoutGradient.");

        private static Variable?[] DropoutGradient(Variable?[] inputs, Variable?[] outputGrads, OnnxCSharpAttributes attributes)
        {
            // Dropout(data, ratio, training_mode) → (output, mask)
            // Inference mode (training_mode absent/false): output = data → identity gradient.
            // Training mode: output = data * mask / (1 - ratio), so
            //   grad_data = grad_output * mask / (1 - ratio).
            // training_mode is a runtime INPUT (its value is unknown at gradient-build time),
            // so whenever it is wired the gradient must be built to be correct for both
            // runtime values:
            //   grad_data = Where(training_mode, Where(mask, grad_output/(1-ratio), 0), grad_output)
            // The mask is a forward OUTPUT: FastProcessAutoGrad plumbs it through as a
            // trailing extra input slot (inputs[3]). When the mask isn't available, throw —
            // silently passing grad_output through would be wrong whenever training_mode
            // evaluates to true at runtime.
            var gradOutput = outputGrads[0]!;

            var trainingMode = inputs.Length > 2 ? inputs[2] : null;
            if (trainingMode is null)
                return [gradOutput, null, null];

            var mask = inputs.Length > 3 ? inputs[3] : null;
            if (mask is null)
                throw new AutoDiffNotSupportedException(ErrorCodes.AD003, DROPOUT,
                    "the gradient of Dropout with a wired training_mode input needs the forward "
                    + "mask output, which was not plumbed through on this path. This is an "
                    + "implementation limitation, not a mathematical one — use inference-mode "
                    + "Dropout (omit the training_mode input) when no mask is available.");

            var ratio = inputs.Length > 1 ? inputs[1] : null;
            var one = OnnxOp.Cast(Scalar(1.0f), saturate: null, to: gradOutput.Type);
            var keepProb = ratio is null
                ? OnnxOp.Cast(Scalar(0.5f), saturate: null, to: gradOutput.Type) // ONNX default ratio
                : OnnxOp.Sub(one, ratio);
            var scaledGrad = OnnxOp.Div(gradOutput, keepProb);
            var zeros = OnnxOp.Sub(gradOutput, gradOutput);
            var trainingGrad = OnnxOp.Where(mask, scaledGrad, zeros);
            var gradData = OnnxOp.Where(trainingMode, trainingGrad, gradOutput);

            // (data, ratio, training_mode, mask) — only data is differentiable.
            return [gradData, null, null, null];
        }

    }
} 