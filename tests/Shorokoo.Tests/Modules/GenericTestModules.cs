namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Generic module test definitions.
    /// These modules use generic type parameters to support different numeric types.
    /// </summary>

    /// <summary>
    /// Simplest possible generic module - just returns input as-is.
    /// This is the foundational test case for generic module support.
    /// </summary>
    [Module]
    public partial class SimpleGenericLayer
    {
        public static Tensor<T> Inline<T>(Tensor<T> input) where T : FloatLike
        {
            return input;
        }
    }

    /// <summary>
    /// Generic module with a simple mathematical operation.
    /// Tests that generic modules can perform operations on generic tensors.
    /// </summary>
    [Module]
    public partial class GenericScaleLayer
    {
        public static Tensor<T> Inline<T>(Tensor<T> input, [Hyper] Scalar<T> scale) where T : FloatLike
        {
            // Scale the input by the hyperparameter
            return input * scale;
        }
    }

    /// <summary>
    /// Generic module that adds two inputs.
    /// Tests that generic modules can handle multiple inputs of the same type.
    /// </summary>
    [Module]
    public partial class GenericAddLayer
    {
        public static Tensor<T> Inline<T>(Tensor<T> input1, Tensor<T> input2) where T : FloatLike
        {
            return input1 + input2;
        }
    }

    /// <summary>
    /// Generic module that composes other generic modules.
    /// Tests nested generic module composition.
    /// </summary>
    [Module]
    public partial class GenericComposedLayer
    {
        public static Tensor<T> Inline<T>(Tensor<T> input, [Hyper] Scalar<T> scale) where T : FloatLike
        {
            // First scale the input
            var scaled = GenericScaleLayer.Model<T>(scale).Call(input);
            
            // Then add it to itself
            var doubled = GenericAddLayer.Call<T>(scaled, scaled);
            
            return doubled;
        }
    }

    /// <summary>
    /// Generic module with three type parameters and mixed hyperparameters.
    /// Tests that the source generator can handle multiple generic type parameters with different constraints.
    /// </summary>
    [Module]
    public partial class GenericThreeTypeParamLayer
    {
        public static Scalar<T> Inline<T, Q, R>(
            Scalar<T> param4,
            Scalar<R> param5,
            Scalar<R> param6,
            [Hyper] Scalar<T> param1,
            [Hyper] Scalar<Q> param2,
            [Hyper] Scalar<Q> param3)
            where T : FloatLike
            where Q : IntLike 
            where R : AnyLike
        {
            // Just return param1 for simplicity
            // This tests that the generator correctly handles all three type parameters with different constraints
            return param1;
        }
    }

    /// <summary>
    /// Generic module that adds a constant to an input.
    /// Tests that generic types can be used with Scalar<T>(object) to create constants.
    /// </summary>
    [Module]
    public partial class AddThree
    {
        public static Tensor<T> Inline<T>(Tensor<T> input) where T : NumLike
        {
            // Create a scalar constant with value 3 (as ushort)
            // This will test that Scalar<T> can handle IGenericTypeX with actual data type uint16
            var three = Scalar<T>((ushort)3);
            return input + three;
        }
    }

    /// <summary>
    /// Generic module that adds a concrete float constant to a generic input.
    /// This should fail during standin processing because the concrete Float32 from the constant
    /// won't match the generic standin Float32 (with GenericTypeParamName) from the input.
    /// Tests that the type system correctly detects mismatches between generic and concrete types.
    /// </summary>
    [Module]
    public partial class AddFive
    {
        // Helper to create a concrete float32 scalar that bypasses generic type inference
        private static Scalar<T> CreateConcreteScalar<T>(float value) where T : IVarType
        {
            // Create a concrete Scalar<float32> first
            var concreteScalar = Scalar(value);
            // Use unsafe cast to convert to Scalar<T>
            // This will work at runtime but the constant node will have concrete Float32 DType
            return (Scalar<T>)(IScalar)concreteScalar;
        }

        public static Tensor<T> Inline<T>(Tensor<T> input) where T : NumLike
        {
            // Create a scalar constant with concrete float32 value
            // This bypasses Scalar<T> and creates a concrete DType constant
            var five = CreateConcreteScalar<T>(5.0f);
            return input + five;
        }
    }

    /// <summary>
    /// Generic module that casts a tensor from one type to another.
    /// Tests that the Cast operator with generic types is properly specialized.
    /// The Cast operator requires explicit type specification via the 'to' attribute,
    /// which must be specialized when using generic types.
    /// </summary>
    [Module]
    public partial class GenericCastLayer
    {
        public static Tensor<TOut> Inline<TIn, TOut>(Tensor<TIn> input) 
            where TIn : FloatLike 
            where TOut : FloatLike
        {
            // Cast the input to the output type
            // This will create a CAST node with the 'to' attribute set to TOut
            return input.Cast<TOut>();
        }
    }

    /// <summary>
    /// Generic module that creates an empty tensor sequence and appends an element.
    /// Tests that the SequenceEmpty operator with generic types is properly specialized.
    /// The SequenceEmpty operator has no inputs, so the element type must be specified
    /// via the 'dtype' attribute, making it critical for generic type handling.
    /// </summary>
    [Module]
    public partial class GenericSequenceLayer
    {
        public static TensorSequence<T> Inline<T>(Tensor<T> element) where T : FloatLike
        {
            // Create an empty sequence with generic type T
            // This will create a SEQUENCE_EMPTY node with 'dtype' attribute set to T
            var emptySeq = Shorokoo.TensorSequence<T>.CreateEmpty();
            
            // Append the element to the sequence
            var seq = emptySeq.Append(element);
            
            return seq;
        }
    }

    /// <summary>
    /// Generic module that creates an identity-like matrix with a different type.
    /// Tests that the EyeLike operator with generic types is properly specialized.
    /// The EyeLike operator can optionally specify output type via 'dtype' attribute,
    /// allowing conversion to a different type than the input.
    /// </summary>
    [Module]
    public partial class GenericEyeLikeLayer
    {
        public static Tensor<TOut> Inline<TIn, TOut>(Tensor<TIn> input) 
            where TIn : FloatLike 
            where TOut : FloatLike
        {
            // Create an identity-like matrix with type TOut
            // This will create an EYE_LIKE node with 'dtype' attribute set to TOut
            return NN.EyeLike<TOut>(input);
        }
    }

    /// <summary>
    /// Generic module that samples from Bernoulli distribution with a different output type.
    /// Tests that the Bernoulli operator with generic types is properly specialized.
    /// The Bernoulli operator can optionally specify output type via 'dtype' attribute,
    /// allowing conversion to a different type than the input probabilities.
    /// </summary>
    [Module]
    public partial class GenericBernoulliLayer
    {
        public static Tensor<TOut> Inline<TIn, TOut>(Tensor<TIn> probabilities) 
            where TIn : FloatLike 
            where TOut : CommonLike
        {
            // Sample from Bernoulli distribution with output type TOut
            // This will create a BERNOULLI node with 'dtype' attribute set to TOut
            return probabilities.Bernoulli<TOut>();
        }
    }

    /// <summary>
    /// Generic module that creates a Blackman window with a specific numeric type.
    /// Tests that the BlackmanWindow operator with generic types is properly specialized.
    /// The BlackmanWindow operator can optionally specify output type via 'output_datatype' attribute,
    /// allowing specification of the output precision.
    /// </summary>
    [Module]
    public partial class GenericBlackmanWindowLayer
    {
        public static Vector<T> Inline<T>(Scalar<int64> size) where T : NumLike
        {
            // Create a Blackman window with output type T
            // This will create a BLACKMAN_WINDOW node with 'output_datatype' attribute set to T
            return NN.BlackmanWindow<T>(size);
        }
    }

    /// <summary>
    /// Generic module that creates a tensor filled with a specific value.
    /// Tests that the ConstantOfShape operator with generic types is properly specialized.
    /// The ConstantOfShape operator has an optional 'value' attribute of type TensorData
    /// that specifies the fill value. When this TensorData has a generic type parameter,
    /// it must be specialized during generic type specialization.
    /// This mirrors the exact process as CONSTANT with Scalar<T>(object).
    /// </summary>
    [Module]
    public partial class GenericConstantOfShapeLayer
    {
        public static Tensor<T> Inline<T>(Vector<int64> shape) where T : NumLike
        {
            // Use TensorFill<T> with object value, mirroring Scalar<T>(object) pattern
            // This will create TensorData<T> with generic type parameter
            // The fill value will be converted to type T during ProcessGenericStandInTypeInference
            return TensorFill<T>(shape, (ushort)5);
        }
    }

    /// <summary>
    /// Original test module that uses unsafe cast from TensorData<float32> to TensorData<T>.
    /// This was the original implementation before TensorFill<T>(object) was added.
    /// This module demonstrates an issue: casting TensorData<float32> to TensorData<T> 
    /// (where T is IGenericType1) should fail at runtime, but needs investigation.
    /// </summary>
    [Module]
    public partial class GenericConstantOfShapeLayerOriginal
    {
        public static Tensor<T> Inline<T>(Vector<int64> shape) where T : NumLike
        {
            // Create a scalar fill value with generic type T
            // The Scalar<T> constructor creates TensorData<T> with the generic type
            var fillValue = Scalar<T>((ushort)5);
            
            // We need to extract the TensorData<T> from the Scalar<T>
            // However, Scalar<T> is a Variable, not TensorData<T>
            // The TensorData is stored as an attribute in the CONSTANT node backing the Scalar
            
            // Instead, let's create the TensorData<T> directly using TensorData with generic type
            // The challenge is that we can't directly instantiate TensorData<T> from the test
            // But we can use type inference from TensorFill<T> which expects TensorData<T>
            
            // For now, use a helper that will be processed by generic type inference
            // Create a constant TensorData - it will be converted to generic type during processing
            var fillValueData = TensorData(1, 5.0f);  // Creates TensorData<float32>
            
            // Use TensorFill<T> which expects TensorData<T>
            // This will force the type system to convert fillValueData to TensorData<T>
            // The cast is unsafe but needed to create the graph structure
            // The actual conversion will happen during ProcessGenericStandInTypeInference
            // NOTE: This cast should fail at runtime - investigating why it doesn't
            return TensorFill<T>(shape, (TensorData<T>)(object)fillValueData);
        }
    }

    /// <summary>
    /// Generic trainable-parameter initializer.
    /// Tests that trainable parameter initializers with generic types are properly specialized.
    /// </summary>
    [TrainableParamInitializer]
    public static partial class GenericTrainableParamInitializers
    {
        public static Tensor<T> Inline<T>(Vector<int64> shape) where T : FloatLike
        {
            // Use TensorFill<T> with object value - it handles generic types correctly
            return Globals.TensorFill<T>(shape, 0.0f);
        }
    }

    /// <summary>
    /// Generic module that uses trainable parameters with generic types.
    /// Tests that TRAINABLE_PARAM_REF operator with generic types is properly specialized.
    /// </summary>
    [Module]
    public partial class GenericLayerWithTrainableParams
    {
        public static Tensor<T> Inline<T>(Tensor<T> input, [Hyper] Vector<int64> shape) where T : FloatLike
        {
            // Create trainable parameters with generic type T
            var weights = GenericTrainableParamInitializers.Init<T>(shape).Vec();
            
            // Add weights to input (simple operation to test trainable param usage)
            return input + weights;
        }
    }

    /// <summary>
    /// Non-generic module that returns a single output.
    /// Used to test MODEL_INVOKE when called by a generic module.
    /// </summary>
    [Module]
    public partial class NonGenericDoubleLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            return input * Scalar(2.0f);
        }
    }

    /// <summary>
    /// Generic module that calls a non-generic module.
    /// Tests MODEL_INVOKE operator with generic types when:
    /// - Caller uses generic types (this module)
    /// - Callee does NOT use generic types (NonGenericDoubleLayer)
    /// This validates that the shrk_dtype array attribute on MODEL_INVOKE is properly handled
    /// during generic specialization for the caller's return type.
    /// </summary>
    [Module]
    public partial class GenericCallerOfNonGenericModule
    {
        public static Tensor<T> Inline<T>(Tensor<T> input) where T : FloatLike
        {
            // Cast the input to float32 since the callee only accepts float32
            var inputF32 = input.Cast<float32>();
            
            // Call the non-generic module
            var doubledF32 = NonGenericDoubleLayer.Call(inputF32);
            
            // Cast back to generic type T
            return doubledF32.Cast<T>();
        }
    }

    /// <summary>
    /// Generic module that can be called by another module.
    /// Used to test MODEL_INVOKE with generic type arguments.
    /// </summary>
    [Module]
    public partial class GenericTargetModule
    {
        public static Tensor<T> Inline<T>(Tensor<T> input) where T : FloatLike
        {
            // Simple operation - multiply by 2
            return input * Scalar<T>((ushort)2);
        }
    }

    /// <summary>
    /// Non-generic module that calls a generic module.
    /// Tests MODEL_INVOKE operator when:
    /// - Caller does NOT use generic types (this module)
    /// - Callee uses generic types (GenericTargetModule)
    /// This validates that the shrk_generic_type_args attribute on MODEL_INVOKE is properly handled
    /// and that GENERIC_TYPE_INPUT nodes are created for the callee's generic type parameters.
    /// </summary>
    [Module]
    public partial class NonGenericCallerOfGenericModule
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            // Call the generic module with float32 type argument
            var doubled = GenericTargetModule.Call<float32>(input);
            return doubled;
        }
    }

    /// <summary>
    /// Nested generic module that performs type casting and simple operations.
    /// Uses three generic type parameters: A (hyperparam), B (input), C (internal only).
    /// This module is called by NestedGenericCallerModule.
    /// </summary>
    [Module]
    public partial class NestedGenericCalleeModule
    {
        public static Tensor<A> Inline<A, B, C>(
            Tensor<B> input,
            [Hyper] Scalar<A> hyperParam)
            where A : FloatLike
            where B : FloatLike
            where C : FloatLike
        {
            // Cast input from type B to type A
            var castInput = input.Cast<A>();
            
            // Scale by hyperparam
            return castInput * hyperParam;
        }
    }

    /// <summary>
    /// Nested generic module that calls another generic module with type parameter mapping.
    /// Uses three generic type parameters: D (hyperparam), E (input), F (internal only).
    /// Calls NestedGenericCalleeModule mapping: D→A, E→B, F→C.
    /// This tests that generic type arguments flow correctly through nested calls.
    /// </summary>
    [Module]
    public partial class NestedGenericCallerModule
    {
        public static Tensor<D> Inline<D, E, F>(
            Tensor<E> input,
            [Hyper] Scalar<D> scale)
            where D : FloatLike
            where E : FloatLike
            where F : FloatLike
        {
            // Call the nested module, mapping type parameters:
            // D (our first type) -> A (callee's first type)
            // E (our second type) -> B (callee's second type)
            // F (our third type) -> C (callee's third type)
            var result = NestedGenericCalleeModule.Call<D, E, F>(scale, input);
            
            return result;
        }
    }

    #region TensorStruct Generic Modules

    /// <summary>
    /// Example user-defined struct interface with generic-typed fields for testing.
    /// Uses Scalar fields to keep tests simple.
    /// </summary>
    public interface GenericPairStruct : IStruct
    {
        Scalar<float32> First { get; }
        Scalar<float32> Second { get; }
    }

    public interface RealGenericPairStruct<T, V> : IStruct 
                where T : IVarType 
                where V : IVarType
    {
        Scalar<T> First { get; }
        Scalar<V> Second { get; }
    }

    /// <summary>
    /// Generic module that creates a TensorStruct internally and returns the sum of scaled fields.
    /// Tests that TensorStruct CREATE and GETFIELD operations work in generic module context.
    /// The module is generic, and the TensorStruct contains concrete Float32 fields.
    /// The generic parameter T is used in an operation on the extracted fields.
    /// </summary>
    [Module]
    public partial class GenericTensorStructProcessor
    {
        public static Scalar<T> Inline<T>(Scalar<float32> inputFirst, Scalar<float32> inputSecond, [Hyper] Scalar<T> scale) where T : FloatLike
        {
            // Globals.TensorStruct<T> builds a DispatchProxy implementing the IStruct
            // interface — property access translates to TENSOR_STRUCT_GETFIELD graph nodes.
            var pair = TensorStruct<GenericPairStruct>(inputFirst, inputSecond);

            // Sum the fields and scale by generic scale factor
            var sum = (pair.First + pair.Second).Cast<T>();
            return sum * scale;
        }
    }

    /// <summary>
    /// Generic module that demonstrates TensorStruct with generic field types.
    /// Creates a dynamic TensorStruct definition with fields of concrete type,
    /// processes them, and returns a result of the generic type.
    /// </summary>
    [Module]
    public partial class GenericTensorStructWithGenericFields
    {
        public static Tensor<T> Inline<T>(Tensor<T> input) where T : FloatLike
        {
            // Create a dynamic struct definition with a concrete float32 field
            var fields = new[]
            {
                new TensorStructFieldDef("Data", DataStructure.Tensor, rank: null, DType.Float32),
            };
            var def = new TensorStructDef(fields, "GenericDataStruct");
            var dtype = DType.GetOrCreateForTensorStruct(def);

            // Cast input to float32 for the struct
            var inputF32 = input.Cast<float32>();

            // Create TensorStruct containing the input
            var tensorStruct = InternalOp.TensorStructCreate(dtype, [inputF32]);

            // Extract the field back out
            var extracted = (Tensor<float32>)InternalOp.TensorStructGetField(tensorStruct, "Data", DType.Float32, null, DataStructure.Tensor);

            // Cast back to generic type T
            return extracted.Cast<T>();
        }
    }

    /// <summary>
    /// Simple generic module that works with a TensorStruct variable (not as input/output, but internally).
    /// This is a simpler test that just verifies TensorStruct operations compile and work in module context.
    /// </summary>
    [Module]
    public partial class GenericTensorStructSum
    {
        public static Scalar<T> Inline<T>(Scalar<float32> a, Scalar<float32> b) where T : FloatLike
        {
            // Globals.TensorStruct<T> builds a DispatchProxy implementing the IStruct
            // interface — property access translates to TENSOR_STRUCT_GETFIELD graph nodes.
            var pair = TensorStruct<GenericPairStruct>(a, b);
            var first = pair.First;
            var second = pair.Second;

            // Sum and cast to generic type
            var sum = first + second;
            return sum.Cast<T>();
        }
    }

    [Module]
    public partial class RealGenericTensorStructSum
    {
        public static Scalar<T> Inline<T, U, V>(RealGenericPairStruct<U, V> pair) 
            where T : FloatLike
            where U : IVarType
            where V : IVarType
        {
            var first = pair.First.Cast<float32>();
            var second = pair.Second.Cast<float32>();

            // Sum and cast to generic type
            var sum = first + second;
            return sum.Cast<T>();
        }
    }

    /// <summary>
    /// A module that creates constant values internally and CALLS RealGenericTensorStructSum,
    /// exercising the full tensor struct parameter passing pipeline including struct creation,
    /// proxy construction, and field access. Uses no input parameters — all values are constants.
    /// Expected output: Scalar(5.0f) + Scalar(7.0f) = 12.0f, cast to T.
    /// </summary>
    [Module]
    public partial class RealGenericTensorStructSumCaller
    {
        public static Scalar<T> Inline<T>()
            where T : FloatLike
        {
            // Create constant scalar values
            var first = Scalar(5.0f);
            var second = Scalar(7.0f);

            var proxy = TensorStruct<RealGenericPairStruct<float32, float32>>(first, second);

            // Call RealGenericTensorStructSum with the proxy
            return RealGenericTensorStructSum.Call<T, float32, float32>(proxy);
        }
    }

    public record GenericPairRecord<T, V>(Scalar<T> First, Scalar<V> Second) : IStruct
        where T : IVarType
        where V : IVarType;

    [Module]
    public partial class GenericRecordSum
    {
        public static Scalar<T> Inline<T, U, V>(GenericPairRecord<U, V> pair) 
            where T : FloatLike
            where U : IVarType
            where V : IVarType
        {
            var first = pair.First.Cast<float32>();
            var second = pair.Second.Cast<float32>();

            // Sum and cast to generic type
            var sum = first + second;
            return sum.Cast<T>();
        }
    }

    /// <summary>
    /// A module that creates constant values internally and CALLS GenericRecordSum,
    /// exercising the full tensor struct parameter passing pipeline with records instead of interfaces.
    /// Uses new GenericPairRecord() instead of TensorStruct proxy constructor.
    /// Expected output: Scalar(5.0f) + Scalar(7.0f) = 12.0f, cast to T.
    /// </summary>
    [Module]
    public partial class GenericRecordSumCaller
    {
        public static Scalar<T> Inline<T>()
            where T : FloatLike
        {
            // Create constant scalar values
            var first = Scalar(5.0f);
            var second = Scalar(7.0f);

            // Use new record instead of TensorStruct proxy constructor
            var record = new GenericPairRecord<float32, float32>(first, second);

            // Call GenericRecordSum with the record
            return GenericRecordSum.Call<T, float32, float32>(record);
        }
    }

    #endregion

    #region Simple TensorStruct Input Module

    /// <summary>
    /// Simple non-generic module that takes a GenericPairStruct as an external input and returns
    /// the sum of its fields. Used to test the full pipeline with a TensorStruct as a model input
    /// (not created from constants internally).
    /// </summary>
    [Module]
    public partial class SimplePairSum
    {
        public static Scalar<float32> Inline(GenericPairStruct pair)
        {
            return pair.First + pair.Second;
        }
    }

    #endregion

    #region TensorStruct in Control Flow (FastUnpackTensorStructs coverage)

    /// <summary>
    /// Carries a bare TensorStruct directly as a loop variable across a constant-iteration
    /// loop, exercising <c>FastUnpackTensorStructs.ExpandLoopOpenStructLoopVars</c> and
    /// <c>ExpandLoopCloseStructLoopVars</c> — the per-field variadic expansion path for
    /// TensorStruct (not sequence-of-TensorStruct) loop-var slots.
    /// </summary>
    [Module]
    public partial class TensorStructLoopCarry
    {
        public static Scalar<float32> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var pair = TensorStruct<GenericPairStruct>(a, b);
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                pair = TensorStruct<GenericPairStruct>(pair.First + Scalar(1.0f), pair.Second + Scalar(2.0f));
            }
            return pair.First + pair.Second;
        }
    }

    /// <summary>
    /// Selects between two bare TensorStruct values via <c>IfElse</c>, exercising
    /// <c>FastUnpackTensorStructs.ExpandIfCloseStructBranches</c> — the per-field
    /// expansion of IF_CLOSE branch slots whose values are TensorStructs.
    /// </summary>
    [Module]
    public partial class TensorStructIfElseReturn
    {
        public static Scalar<float32> Inline(Scalar<bit> condition, Scalar<float32> a, Scalar<float32> b)
        {
            var thenPair = TensorStruct<GenericPairStruct>(a, b);
            var elsePair = TensorStruct<GenericPairStruct>(b, a);

            // IfElse takes graph Variable args — unwrap each proxy to its backing struct,
            // then re-wrap the picked result so field access flows through the proxy.
            var pickedVar = Shorokoo.Core.Nodes.Ops.IfElse(
                condition,
                ((ITensorStructProxy)thenPair).BackingTensorStruct,
                ((ITensorStructProxy)elsePair).BackingTensorStruct);
            var picked = AsTensorStruct<GenericPairStruct>(pickedVar);

            return picked.First + picked.Second;
        }
    }

    /// <summary>
    /// Loop with a mix of slot kinds: a plain <c>Scalar&lt;float32&gt;</c> carry and a
    /// TensorStruct carry, so the loop-var variadic expansion has to interleave
    /// pass-through slots and per-field expanded slots. Exercises the plain-tensor
    /// branch alongside the struct branch in <c>ExpandLoopOpenStructLoopVars</c> and
    /// <c>ExpandLoopCloseStructLoopVars</c>.
    /// </summary>
    [Module]
    public partial class MixedTensorStructLoop
    {
        public static Scalar<float32> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var plainAcc = Scalar(0.0f);
            var pair = TensorStruct<GenericPairStruct>(a, b);

            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                plainAcc = plainAcc + pair.First;
                pair = TensorStruct<GenericPairStruct>(pair.First + pair.Second, pair.Second + Scalar(1.0f));
            }

            return plainAcc + pair.First;
        }
    }

    /// <summary>
    /// Threads a TensorStruct through all six SEQUENCE_* ops in a single body:
    /// <c>SEQUENCE_EMPTY</c> + <c>SEQUENCE_INSERT</c> build a one-element struct
    /// sequence; <c>SEQUENCE_CONSTRUCT</c> builds a two-element one; <c>SEQUENCE_AT</c>
    /// reads back; <c>SEQUENCE_ERASE</c> drops the head; <c>SEQUENCE_LENGTH</c>
    /// queries the survivor. Each op's struct-typed branch in
    /// <c>FastUnpackTensorStructs.Process</c> rewrites the single op into one parallel
    /// op per struct field (LENGTH collapses to a single read from field[0]).
    /// </summary>
    [Module]
    public partial class SequenceOpsOnStructs
    {
        public static Scalar<float32> Inline(Scalar<float32> a1, Scalar<float32> a2, Scalar<float32> b1, Scalar<float32> b2)
        {
            var sA = TensorStruct<GenericPairStruct>(a1, a2);
            var sB = TensorStruct<GenericPairStruct>(b1, b2);

            // Sequence ops require IValue args — unwrap each proxy to its backing struct.
            var sAVar = ((ITensorStructProxy)sA).BackingTensorStruct;
            var sBVar = ((ITensorStructProxy)sB).BackingTensorStruct;
            var dtype = sAVar.Type;

            // EMPTY + INSERT path: empty struct sequence, append sA, read it back.
            var empty = OnnxOp.SequenceEmpty(dtype);
            var seq1 = OnnxOp.SequenceInsert(empty, sAVar, null);
            var picked1 = AsTensorStruct<GenericPairStruct>(seq1.At(Scalar(0L)));

            // CONSTRUCT + ERASE + LENGTH + AT path: build two-element seq, drop the
            // head, query length and read the survivor.
            var seq2 = OnnxOp.SequenceConstruct([sAVar, sBVar]);
            var erased = seq2.RemoveAt(Scalar(0L));
            var len = erased.Count.Cast<float32>();
            var picked2 = AsTensorStruct<GenericPairStruct>(erased.At(Scalar(0L)));

            return picked1.First + picked2.Second + len;
        }
    }

    /// <summary>
    /// Carries a <c>Sequence&lt;TensorStruct&gt;</c> across loop iterations
    /// (body re-binds <c>seq</c> via <c>SEQUENCE_INSERT</c>, so the next iter's
    /// SEQUENCE_AT consumes the prior iter's appended sequence). Drives the
    /// sequence-of-struct loop-var branches of
    /// <c>FastUnpackTensorStructs.ExpandLoopOpenStructLoopVars</c> and
    /// <c>ExpandLoopCloseStructLoopVars</c> — per-field parallel sequence
    /// expansion of the LOOP_OPEN/CLOSE variadic loop-var slots.
    /// </summary>
    [Module]
    public partial class SequenceOfStructLoopCarry
    {
        public static Scalar<float32> Inline(Scalar<float32> a1, Scalar<float32> a2)
        {
            var def = StructDefExtractor.ExtractFromType<GenericPairStruct>();
            var dtype = DType.GetOrCreateForTensorStruct(def);

            var seed = InternalOp.TensorStructCreate(dtype, [a1, a2]);
            Variable seq = OnnxOp.SequenceConstruct(seed);

            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var newStruct = InternalOp.TensorStructCreate(dtype, [a1 + Scalar(1.0f), a2 + Scalar(2.0f)]);
                seq = OnnxOp.SequenceInsert(seq, newStruct, null);
            }

            var picked = OnnxOp.SequenceAt(seq, Scalar(0L));
            var first = (Scalar<float32>)InternalOp.TensorStructGetField(picked, "First", DType.Float32, 0, DataStructure.Tensor);
            return first;
        }
    }

    /// <summary>
    /// IfElse whose branches are each a <c>Sequence&lt;TensorStruct&gt;</c>.
    /// Drives the sequence-of-struct branch of
    /// <c>FastUnpackTensorStructs.ExpandIfCloseStructBranches</c> — IF_CLOSE
    /// branch slots whose values are sequence-of-TensorStruct expand to per-field
    /// parallel sequence slots.
    /// </summary>
    [Module]
    public partial class SequenceOfStructIfElseReturn
    {
        public static Scalar<float32> Inline(Scalar<bit> cond, Scalar<float32> a1, Scalar<float32> a2)
        {
            var def = StructDefExtractor.ExtractFromType<GenericPairStruct>();
            var dtype = DType.GetOrCreateForTensorStruct(def);

            var sA = InternalOp.TensorStructCreate(dtype, [a1, a2]);
            var sB = InternalOp.TensorStructCreate(dtype, [a2, a1]);
            Variable thenSeq = OnnxOp.SequenceConstruct(sA, sB);
            Variable elseSeq = OnnxOp.SequenceConstruct(sB, sA);

            Variable picked = Shorokoo.Core.Nodes.Ops.IfElse(cond, thenSeq, elseSeq);
            var firstStruct = OnnxOp.SequenceAt(picked, Scalar(0L));
            var first = (Scalar<float32>)InternalOp.TensorStructGetField(firstStruct, "First", DType.Float32, 0, DataStructure.Tensor);
            return first;
        }
    }

    /// <summary>
    /// IfElse whose branches return a tuple of <c>(TensorStruct, Scalar)</c> —
    /// slot 0 is a struct (expanded per-field), slot 1 is a plain scalar
    /// (passes through). Drives the plain-tensor passthrough <c>else</c>
    /// branch of <c>FastUnpackTensorStructs.ExpandIfCloseStructBranches</c> that
    /// the single-output struct-IfElse modules don't reach because they have
    /// no non-struct slot in the same IF_CLOSE.
    /// </summary>
    [Module]
    public partial class IfElseMixedStructAndPlainSlots
    {
        public static Scalar<float32> Inline(Scalar<bit> cond, Scalar<float32> a1, Scalar<float32> a2)
        {
            var def = StructDefExtractor.ExtractFromType<GenericPairStruct>();
            var dtype = DType.GetOrCreateForTensorStruct(def);

            TensorStruct<GenericPairStruct> thenStruct = InternalOp.TensorStructCreate(dtype, [a1, a2]);
            TensorStruct<GenericPairStruct> elseStruct = InternalOp.TensorStructCreate(dtype, [a2, a1]);
            var thenPlain = a1 + a2;
            var elsePlain = a1 - a2;

            var (pickedStruct, pickedPlain) = cond.IfElse(
                (thenStruct, thenPlain),
                (elseStruct, elsePlain));

            var pickedFirst = (Scalar<float32>)InternalOp.TensorStructGetField(
                pickedStruct, "First", DType.Float32, 0, DataStructure.Tensor);
            return pickedFirst + pickedPlain;
        }
    }

    /// <summary>
    /// Loop carries a TensorStruct as a loop variable AND uses <c>ctx.Scan</c>
    /// to collect a per-iter plain scalar. The loop-var path drives
    /// <c>FastUnpackTensorStructs.ExpandLoopCloseStructLoopVars</c> (because the
    /// LOOP_OPEN/CLOSE has at least one struct-typed loop var), and once that
    /// expansion handler runs it also walks the scan-output slots after the
    /// loop-var slots — exercising the trailing scan-output expansion block
    /// (~FastProcessors.cs L2173-2184). No existing struct-loop module emits a
    /// scan output, and no existing scan-output module uses a struct loop var,
    /// so this is the only combined shape that hits that slot.
    /// </summary>
    [Module]
    public partial class TensorStructLoopCarryWithScanOutput
    {
        public static Tensor<float32> Inline(Scalar<float32> a, Scalar<float32> b)
        {
            var def = StructDefExtractor.ExtractFromType<GenericPairStruct>();
            var dtype = DType.GetOrCreateForTensorStruct(def);

            Variable pair = InternalOp.TensorStructCreate(dtype, [a, b]);
            Variable? scanned = null;
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                var first = (Scalar<float32>)InternalOp.TensorStructGetField(
                    pair, "First", DType.Float32, 0, DataStructure.Tensor);
                var second = (Scalar<float32>)InternalOp.TensorStructGetField(
                    pair, "Second", DType.Float32, 0, DataStructure.Tensor);
                pair = InternalOp.TensorStructCreate(dtype, [first + Scalar(1.0f), second + Scalar(2.0f)]);
                scanned = (Variable)ctx.Scan(first + second);
            }
            return (Tensor<float32>)scanned!;
        }
    }

    #endregion

    #region Constant-iter loop edge cases (FastFoldConstantIterationLoops.UnrollOne coverage)

    /// <summary>
    /// Constant zero-iteration loop. Drives the early <c>iterCount == 0</c> return
    /// in <c>FastFoldConstantIterationLoops.UnrollOne</c> (~L3832-3845 of
    /// <c>FastProcessors.cs</c>): CLOSE's loop-var outputs are routed directly
    /// to OPEN's initializers and the loop body is swept. Scan vars are
    /// deliberately absent — the eligibility gate disqualifies <c>iterCount==0</c>
    /// with <c>nScan&gt;0</c>.
    /// </summary>
    [Module]
    public partial class ZeroIterConstLoopLayer
    {
        public static Scalar<float32> Inline(Scalar<float32> x)
        {
            var acc = x;
            foreach (var ctx in LoopAPI.Iterate(Scalar(0L)))
            {
                acc = acc + Scalar(1.0f);
            }
            return acc;
        }
    }

    /// <summary>
    /// Constant-iter loop with a <c>ctx.Scan</c> output. Drives the scan-output
    /// branch of <c>FastFoldConstantIterationLoops.UnrollOne</c> — the per-iter
    /// <c>UNSQUEEZE</c> + final <c>CONCAT</c> stack that materialises the
    /// rank-(N) scan tensor (~L4263-4274 record per-iter scan keys, ~L4305-4330
    /// emit the UNSQUEEZE chain plus CONCAT).
    /// </summary>
    [Module]
    public partial class ConstLoopWithScanOutput
    {
        public static Tensor<float32> Inline(Scalar<float32> x)
        {
            var acc = x;
            Variable? scanned = null;
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                acc = acc + Scalar(1.0f);
                scanned = (Variable)ctx.Scan(acc);
            }
            return (Tensor<float32>)scanned!;
        }
    }

    /// <summary>
    /// Constant-iter loop whose body break is dynamic (<c>ctx.ContinueWhile</c>
    /// fed by a runtime-bool input). <c>LoopAPI</c> emits the <c>LOOP_OPEN</c>
    /// with <c>condition: null</c>, so OPEN.Inputs[1] is absent and the unroller
    /// has to seed the AND-chain with a fresh <c>CONSTANT(true)</c> node
    /// (~L3907-3929) and emit per-loop-var <c>WHERE</c> + per-iter <c>AND</c>
    /// gating (~L4210-4247).
    /// </summary>
    [Module]
    public partial class ConstLoopWithDynamicBreak
    {
        public static Scalar<float32> Inline(Scalar<float32> x, Scalar<bit> keepGoing)
        {
            var acc = x;
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                acc = acc + Scalar(1.0f);
                ctx.ContinueWhile(keepGoing);
            }
            return acc;
        }
    }

    /// <summary>
    /// Constant-iter loop whose body contains a nested <c>IfElse</c> with a
    /// loop-dependent condition (compares the iteration index against a
    /// constant) and constant-only branches (no <c>acc</c> in either side, so
    /// the resulting <c>IF_CLOSE</c> takes only constant inputs and is NOT
    /// directly loop-dependent). Drives the scope-pair propagation arm of
    /// <c>FastFoldConstantIterationLoops.UnrollOne</c> (~L3710-3719 of
    /// <c>FastProcessors.cs</c>): the unroller's loop-dep analyzer sees a body
    /// <c>IF_CLOSE</c> with no direct loop-dep input but its paired
    /// <c>IF_OPEN</c>'s inputs are loop-dep (via the iter-index cond), so it
    /// must walk back through <c>GraphOpenNodeKey</c> to mark the CLOSE for
    /// cloning. Both <c>acc</c>-dependent branches (the typical pattern) hit
    /// the direct-loop-dep path and skip this walk-back.
    /// </summary>
    [Module]
    public partial class ConstLoopWithNestedIterDependentIf
    {
        public static Scalar<float32> Inline(Scalar<float32> x)
        {
            var acc = x;
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                var isFirst = ctx.IterationIndex < Scalar(1L);
                var bump = isFirst.IfElse(Scalar(10.0f), Scalar(1.0f));
                acc = acc + bump;
            }
            return acc;
        }
    }

    #endregion


    #region Nested-loop submodule call (CombineIterationIndices flatten coverage)

    /// <summary>
    /// Outer constant-iter loop creates a submodule (with its own internal
    /// loop with trainable params inside) and calls it in the same iteration.
    /// <c>MODULE_SET_HYPERPARAMS</c> for the submodule lives inside the outer
    /// loop, so its iteration-indices input is non-empty. The submodule's
    /// body contains an inner loop with <c>InitSimple.Init</c> calls, so
    /// when <c>FastInlineModulesAndFunctions.FastReparentToCallSite</c> walks
    /// the inlined subgraph, each <c>TRAINABLE_PARAM_REF</c>'s child-side
    /// iteration-indices input is also non-empty (a <c>CONCAT</c> of the
    /// inner loop's iter scalars).
    ///
    /// <para>
    /// Drives the "both halves non-empty" branch of
    /// <c>CombineIterationIndices</c> (FastProcessors.cs ~L856-864), the
    /// only path that flattens two pre-existing CONCATs into one. No other
    /// <c>[Module]</c> in the test zoo triggers it — <c>Model&lt;,&gt;</c>
    /// hyperparams and <c>ModelSequence</c> indexing route through
    /// <c>FastReparentToModelVariable</c> (no <c>CombineIterationIndices</c>
    /// call), and direct-create-and-call patterns outside any loop have an
    /// empty <c>parentKey</c>.
    /// </para>
    /// </summary>
    [Module]
    public partial class NestedLoopWithSubmoduleInnerLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var inner = LoopLayer.Model(Scalar(5L), Scalar(2L));
                x = inner.Call(x);
            }
            return x;
        }
    }

    #endregion

    #region Plain module-on-module call (for ONNX save/reload coverage of FUNCTION_INVOKE inlining)

    /// <summary>
    /// Minimal module-on-module pattern: one [Module] invokes another via the
    /// source-generated <c>Call</c> method. Used to surface the FUNCTION_INVOKE
    /// arm of FastInlineModulesAndFunctions through the ONNX save/reload path —
    /// when the non-concrete graph here is saved to ONNX bytes and loaded back,
    /// FastOnnxModelReader turns the inner module call into a FUNCTION_INVOKE
    /// node. ToConcreteArchitecture then dispatches it through the previously-
    /// untested isFunction branch of FastInlineModulesAndFunctions.
    ///
    /// Kept deliberately simple (no hyperparams, no loops, no struct types) so
    /// the save/load path doesn't trip on unrelated edge cases.
    /// </summary>
    [Module]
    public partial class CallsSimplestModule
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => SimplestLayer.Call(input);
    }

    /// <summary>
    /// Plain wrapper over <see cref="HypersLayer"/>.<c>Call</c> with constant
    /// hyperparams. By itself this only exercises the
    /// <c>TRAINABLE_PARAM_REF</c> arm of
    /// <c>FastInlineModulesAndFunctions.FastReparentToCallSite</c> (HypersLayer's
    /// body has the <c>InitSimple.Init</c> trainable param but no nested
    /// MODULE_SET_HYPERPARAMS). Used as the <i>inner</i> level for
    /// <see cref="CallsCallsHypersLayer"/>, whose inlining brings this body
    /// (including its <c>MODULE_SET_HYPERPARAMS</c> for HypersLayer) into the
    /// outer module's graph.
    /// </summary>
    [Module]
    public partial class CallsHypersLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => HypersLayer.Call(Scalar(2.0f), Scalar(0.5f), input);
    }

    /// <summary>
    /// Three-level module-on-module-on-hyperparam-module call. When the outer
    /// module's <c>MODULE_INVOKE</c> for <see cref="CallsHypersLayer"/> is
    /// inlined, <c>FastReparentToCallSite</c> walks
    /// <see cref="CallsHypersLayer"/>'s flattened body, which contains a
    /// <c>MODULE_SET_HYPERPARAMS</c> node for the nested
    /// <see cref="HypersLayer"/> call. Drives the
    /// <c>MODULE_SET_HYPERPARAMS</c> arm at <c>FastProcessors.cs ~L703-727</c>
    /// — the mirror of the <c>TRAINABLE_PARAM_REF</c> arm at L676-702 hit by
    /// <see cref="CallsSimplestModule"/>.
    /// </summary>
    [Module]
    public partial class CallsCallsHypersLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => CallsHypersLayer.Call(input);
    }

    #endregion
}
