namespace Shorokoo.Tests.Modules
{
    #region Trainable Parameter Initializers (New Pattern)
    
    /// <summary>
    /// Trainable parameter initializer using the new single-class pattern.
    /// Replaces TrainableParamInitializers.InitSimple() with InitSimple.Init()
    /// </summary>
    [TrainableParamInitializer]
    public static partial class InitSimple
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            return Globals.TensorFill(shape, 1.0f);
        }
    }
    
    #endregion

    [Module]
    public partial class LoopLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> numOutFeatures, [Hyper] Scalar<int64> numIterations)
        {
            var x = input;
            var numInFeatures = x.ShapeTensor()[-1L];
            var weights = InitSimple.Init([numInFeatures, numOutFeatures]);
            x = x.MatMul(weights);

            foreach (var i in LoopAPI.Iterate(numIterations))
            {
                var weights2 = InitSimple.Init([numOutFeatures, numOutFeatures]);
                x = x.MatMul(weights2);
            }

            return x;
        }
    }

    [Module]
    public partial class FCLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> numOutFeatures)
        {
            var numInFeatures = input.ShapeTensor()[-1L];
            var weights = InitSimple.Init([numOutFeatures, numInFeatures]);
            var bias = InitSimple.Init([numOutFeatures]).Vec();

            var withoutBias = input.MatMul(weights.Transpose(1, 0));
            var withBias = withoutBias + bias;

            var retVal = withBias;
            return retVal;
        }
    }

    [Module]
    public partial class SimplestLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var weights = InitSimple.Init(input.ShapeTensor());
            return input * weights;
        }
    }

    // Returns a rank-pinned Vector<float32>. Constructing the auto-generated
    // Module<Tensor<float32>, Vector<float32>> goes through
    // ModuleHelper.CreateFunctionSignature, which derives a per-output rank override
    // from the compile-time output type — Vector<T> hits the IVector→1 branch and
    // produces OutputRankOverrides = [1]. The C# codegen path on the generated
    // Function relies on that override to emit `Vector<float32>` (rather than
    // `Tensor<float32>`) as the return type, so this is the natural circumstance
    // that exercises rank-override propagation end-to-end.
    [Module]
    public partial class VectorReturnLayer
    {
        public static Vector<float32> Inline(Tensor<float32> input)
        {
            var weights = InitSimple.Init([Scalar(5L)]).Vec();
            return weights;
        }
    }

    [Module]
    public partial class HypersLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<float32> factor, [Hyper] Scalar<float32> bias)
        {
            var weights = InitSimple.Init(input.ShapeTensor());
            var txWeights = weights * factor + bias;
            return input * weights;
        }
    }

    [Module]
    public partial class SimpleWithHyperparam
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> hyperparam)
        {
            var weights = InitSimple.Init(input.ShapeTensor());
            var multiplier = hyperparam.Cast<float32>();
            return input * weights * multiplier;
        }
    }

    [Module]
    public partial class TwoStackLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> numOutFeatures)
        {
            var fc1 = FCLayer.Model(numOutFeatures);
            var fc2 = FCLayer.Model(numOutFeatures);
            
            var x = fc1.Call(input);
            x = fc2.Call(x);
            
            return x;
        }
    }

    [Module]
    public partial class BackbonedLayer
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(
                Tensor<float32> inputA, Tensor<float32> inputB, Model<Tensor<float32>, Tensor<float32>> bb2,
                [Hyper] Model<Tensor<float32>, Tensor<float32>> bb)
        {
            var bba = bb.Call(inputA);
            var bbb = bb.Call(inputB);

            return (bb2.Call(bba + bbb), bb2.Call(bba * bbb));
        }
    }

    [Module]
    public partial class BackboneInput
    {
        public static Tensor<float32> Inline(Tensor<float32> inputA, Tensor<float32> inputB, Model<Tensor<float32>, Tensor<float32>> bb)
        {
            var bba = bb.Call(inputA);
            var bbb = bb.Call(inputB);

            return bba + bbb;
        }
    }

    [Module]
    public partial class BackboneHyper
    {
        public static Tensor<float32> Inline(Tensor<float32> inputA, Tensor<float32> inputB, [Hyper] Model<Tensor<float32>, Tensor<float32>> bb)
        {
            var bba = bb.Call(inputA);
            var bbb = bb.Call(inputB);

            return bba + bbb;
        }
    }

    [Module]
    public partial class BackbonedSquaredLayer
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(
            Tensor<float32> inputA, Tensor<float32> inputB,
            [Hyper] Model<Tensor<float32>, Tensor<float32>> bb,
            [Hyper] Model<Tensor<float32>, Tensor<float32>, Model<Tensor<float32>, Tensor<float32>>, (Tensor<float32>, Tensor<float32>)> bb2)
        {
            var bba = bb2.Call(inputA, inputB, bb);
            var bbb = bb2.Call(inputB, inputA, bb);

            return (bba.Item1 + bbb.Item2, bba.Item2 + bbb.Item1);
        }
    }

    [Module]
    public partial class BackbonerSquared
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> inputA, Tensor<float32> inputB)
        {
            var ts10 = TwoStackLayer.Model(Scalar(10L));
            var ts5 = TwoStackLayer.Model(Scalar(10L));
            var bb1 = BackbonedLayer.Model(ts10);
            var bb2 = BackbonedSquaredLayer.Model(ts5, bb1);
            return bb2.Call(inputA, inputB);
        }
    }

    [Module]
    public partial class SimpleModelSequence
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numOutFeatures)
        {
            var numOutFeatures1 = numOutFeatures * 2;
            var numOutFeatures2 = numOutFeatures1 * 3;
            var model1 = SimplestLayer.Model();

            var sequence = ModelSequence.Create(model1);
            var x = sequence[Scalar(0L)].Call(inputs);
            return x;
        }
    }

    [Module]
    public partial class SimpleModelSequenceSimpleLooped
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs, [Hyper] Scalar<int64> numOutFeatures)
        {
            var numOutFeatures1 = numOutFeatures * 2;
            var numOutFeatures2 = numOutFeatures1 * 3;
            var model1 = SimplestLayer.Model();
            var model2 = SimplestLayer.Model();
            var sequence = ModelSequence.Create(model1, model2);

            var x = inputs;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
                x = sequence[ctx.IterationIndex].Call(x);

            return x;
        }
    }

    [Module]
    public partial class HyperparamModelSequenceSimpleLooped
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var model1 = SimpleWithHyperparam.Model(Scalar(17L));
            var model2 = SimpleWithHyperparam.Model(Scalar(19L));
            var sequence = ModelSequence.Create(model1, model2);

            var x = inputs;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
                x = sequence[ctx.IterationIndex].Call(x);

            return x;
        }
    }

    [Module]
    public partial class ModelsCreatedInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var simplestModels = ModelSequence.Empty(SimplestLayer.Model());
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var model = SimplestLayer.Model();
                simplestModels = simplestModels.Append(model);
            }

            var x = simplestModels[Scalar(1L)].Call(inputs);
            x = simplestModels[Scalar(0L)].Call(x);

            return x;
        }
    }

    [Module]
    public partial class SimplestBackbone
    {
        public static Tensor<float32> Inline(Model<Tensor<float32>, Tensor<float32>> bb, Tensor<float32> input)
        {
            var x = bb.Call(input);
            x = x + InitSimple.Init(x.ShapeTensor());
            return x;
        }
    }

    [Module]
    public partial class HypersBackbone
    {
        public static Tensor<float32> Inline(Model<Tensor<float32>, Tensor<float32>> bb, Tensor<float32> input, [Hyper] Scalar<float32> div, [Hyper] Scalar<float32> avoid)
        {
            var x = bb.Call(input);
            x = x + InitSimple.Init(x.ShapeTensor());
            x = (x / div) - avoid;
            return x;
        }
    }

    [Module]
    public partial class CalledInLoopBackbone
    {
        public static Tensor<float32> Inline(Model<Tensor<float32>, Tensor<float32>> bb, Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                x = bb.Call(x);
            }

            return x;
        }
    }

    [Module]
    public partial class StraightSimplestBackbone
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var backboneModel = SimplestLayer.Model();

            var bb1 = SimplestBackbone.Model();

            return bb1.Call(backboneModel, inputs);
        }
    }

    [Module]
    public partial class SimplestBackboneCalledInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var backboneModel = SimplestLayer.Model();

            var bb1 = SimplestBackbone.Model();
            var bb2 = SimplestBackbone.Model();

            var sequence = ModelSequence.Create(bb1, bb2);

            var x = inputs;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var model = sequence[ctx.IterationIndex];
                x = model.Call(backboneModel, x);
            }

            return x;
        }
    }

    [Module]
    public partial class HypersBackboneCalledInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var backboneModel = HypersLayer.Model(Scalar(1.5f), Scalar(0.5f));

            var bb1 = HypersBackbone.Model(Scalar(1.6f), Scalar(0.6f));
            var bb2 = HypersBackbone.Model(Scalar(1.7f), Scalar(0.7f));

            var sequence = ModelSequence.Create(bb1, bb2);

            var x = inputs;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var model = sequence[ctx.IterationIndex];
                x = model.Call(backboneModel, x);
            }

            return x;
        }
    }

    [Module]
    public partial class SimplestBackboneCalledInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var backboneModel = SimplestLayer.Model();

            var bb1 = SimplestBackbone.Model();
            var bb2 = SimplestBackbone.Model();
            var bb3 = SimplestBackbone.Model();

            var sequence = ModelSequence.Create(bb1, bb2, bb3);

            var x = inputs;
            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx1 in LoopAPI.Iterate(Scalar(2L)))
                {
                    var model = sequence[ctx0.IterationIndex + ctx1.IterationIndex];
                    x = model.Call(backboneModel, x);
                }
            }

            return x;
        }
    }

    [Module]
    public partial class HypersBackboneCalledInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var backboneModel = HypersLayer.Model(Scalar(1.3f), Scalar(0.3f));

            var bb1 = HypersBackbone.Model(Scalar(1.4f), Scalar(0.4f));
            var bb2 = HypersBackbone.Model(Scalar(1.5f), Scalar(0.5f));
            var bb3 = HypersBackbone.Model(Scalar(1.6f), Scalar(0.6f));

            var sequence = ModelSequence.Create(bb1, bb2, bb3);

            var x = inputs;
            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx1 in LoopAPI.Iterate(Scalar(2L)))
                {
                    var model = sequence[ctx0.IterationIndex + ctx1.IterationIndex];
                    x = model.Call(backboneModel, x);
                }
            }

            return x;
        }
    }

    [Module]
    public partial class SimplestBackboneCreatedInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var simplestBackboneModels = ModelSequence.Empty(SimplestBackbone.Model());
            var simplestModel = SimplestLayer.Model();

            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var model = SimplestBackbone.Model();
                simplestBackboneModels = simplestBackboneModels.Append(model);
            }

            var x = simplestBackboneModels[Scalar(1L)].Call(simplestModel, inputs);
            x = simplestBackboneModels[Scalar(0L)].Call(simplestModel, x);
            return x;
        }
    }

    [Module]
    public partial class HypersBackboneCreatedInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var hypersBackboneModels = ModelSequence.Empty(HypersBackbone.Model(Scalar(1.4f), Scalar(0.4f)));
            var simplestModel = HypersLayer.Model(Scalar(1.3f), Scalar(0.3f));

            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var factor = ctx.IterationIndex.Cast<float32>();
                var bias = ctx.IterationIndex.Cast<float32>() / 2.0f;
                var model = HypersBackbone.Model(factor, bias);
                hypersBackboneModels = hypersBackboneModels.Append(model);
            }

            var x = hypersBackboneModels[Scalar(1L)].Call(simplestModel, inputs);
            x = hypersBackboneModels[Scalar(0L)].Call(simplestModel, x);
            return x;
        }
    }

    [Module]
    public partial class SimplestBackboneCreatedInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var simplestBackboneModels = ModelSequence.Empty(SimplestBackbone.Model());
            var simplestModel = SimplestLayer.Model();

            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
                {
                    var model = SimplestBackbone.Model();
                    simplestBackboneModels = simplestBackboneModels.Append(model);
                }
            }

            var x = simplestBackboneModels[Scalar(1L)].Call(simplestModel, inputs);
            x = simplestBackboneModels[Scalar(0L)].Call(simplestModel, x);
            x = simplestBackboneModels[Scalar(3L)].Call(simplestModel, x);
            x = simplestBackboneModels[Scalar(2L)].Call(simplestModel, x);
            return x;
        }
    }

    [Module]
    public partial class HypersBackboneCreatedInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var simplestBackboneModels = ModelSequence.Empty(HypersBackbone.Model(Scalar(1.3f), Scalar(0.3f)));
            var simplestModel = HypersLayer.Model(Scalar(1.4f), Scalar(0.4f));

            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
                {
                    var factor = (ctx0.IterationIndex.Cast<float32>()*5.0f) + ctx.IterationIndex.Cast<float32>();
                    var bias = ((ctx0.IterationIndex.Cast<float32>()*5.0f) + ctx.IterationIndex.Cast<float32>()) / 10.0f;
                    var model = HypersBackbone.Model(factor, bias);
                    simplestBackboneModels = simplestBackboneModels.Append(model);
                }
            }

            var x = simplestBackboneModels[Scalar(1L)].Call(simplestModel, inputs);
            x = simplestBackboneModels[Scalar(0L)].Call(simplestModel, x);
            x = simplestBackboneModels[Scalar(3L)].Call(simplestModel, x);
            x = simplestBackboneModels[Scalar(2L)].Call(simplestModel, x);
            return x;
        }
    }

    [Module]
    public partial class SimplestBackboneCalledAndCreatedInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var simplestBackboneModels = ModelSequence.Empty(SimplestBackbone.Model());
            var simplestModel = SimplestLayer.Model();

            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
                {
                    var model = SimplestBackbone.Model();
                    simplestBackboneModels = simplestBackboneModels.Append(model);
                }
            }

            var x = inputs;
            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx1 in LoopAPI.Iterate(Scalar(2L)))
                {
                    var model = simplestBackboneModels[ctx0.IterationIndex + ctx1.IterationIndex];
                    x = model.Call(simplestModel, x);
                }
                x = simplestBackboneModels[Scalar(3L)].Call(simplestModel, x);
            }

            x = simplestBackboneModels[Scalar(4L)].Call(simplestModel, x);
            x = simplestBackboneModels[Scalar(5L)].Call(simplestModel, x);
            return x;
        }
    }

    [Module]
    public partial class HypersBackboneCalledAndCreatedInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var hypersBackboneModels = ModelSequence.Empty(HypersBackbone.Model(Scalar(1.5f), Scalar(0.5f)));
            var hypersModel = HypersLayer.Model(Scalar(1.3f), Scalar(0.3f));

            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
                {
                    var model = HypersBackbone.Model(Scalar(1.3f), Scalar(0.3f));
                    hypersBackboneModels = hypersBackboneModels.Append(model);
                }
            }

            var x = inputs;
            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx1 in LoopAPI.Iterate(Scalar(2L)))
                {
                    var model = hypersBackboneModels[ctx0.IterationIndex + ctx1.IterationIndex];
                    x = model.Call(hypersModel, x);
                }
                x = hypersBackboneModels[Scalar(3L)].Call(hypersModel, x);
            }

            x = hypersBackboneModels[Scalar(4L)].Call(hypersModel, x);
            x = hypersBackboneModels[Scalar(5L)].Call(hypersModel, x);
            return x;
        }
    }

    [Module]
    public partial class CalledInLoopBackboneCalledInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var backboneModel = SimplestLayer.Model();

            var bb1 = CalledInLoopBackbone.Model();
            var bb2 = CalledInLoopBackbone.Model();
            var sequence = ModelSequence.Create(bb1, bb2);

            var x = inputs;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var model = sequence[ctx.IterationIndex];
                x = model.Call(backboneModel, x);
            }
            return x;
        }
    }

    [Module]
    public partial class CalledInLoopBackboneCreatedInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var calledInLoopBackboneModels = ModelSequence.Empty(SimplestBackbone.Model());
            var simplestModel = SimplestLayer.Model();
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var model = SimplestBackbone.Model();
                calledInLoopBackboneModels = calledInLoopBackboneModels.Append(model);
            }

            var x = calledInLoopBackboneModels[Scalar(1L)].Call(simplestModel, inputs);
            x = calledInLoopBackboneModels[Scalar(0L)].Call(simplestModel, x);
            return x;
        }
    }

    [Module]
    public partial class CustomTrainableParamInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> alpha)
        {
            return InitSimple.Init(shape);
        }
    }

    /// <summary>
    /// A module that conditionally applies different trainable parameters based on input tensor shape.
    /// The input shape acts as an implicit hyperparameter that cannot be constant-folded away,
    /// which exercises the ListAllSpecificModelIdsUsed filtering logic with IF branches.
    /// Both sets of trainable params are created unconditionally, but only one result is used
    /// based on the runtime condition.
    /// </summary>
    [Module]
    public partial class ConditionalTrainableParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> threshold)
        {
            var lastDim = input.ShapeTensor()[-1L];
            var isLarge = lastDim > threshold;

            // Create both sets of trainable params (always, regardless of condition)
            var weightsLarge = InitSimple.Init(input.ShapeTensor());
            var weightsSmall = InitSimple.Init(input.ShapeTensor());

            // Compute both results
            var largeResult = input * weightsLarge;
            var smallResult = input + weightsSmall;

            // Select based on runtime condition
            return isLarge.IfElse(largeResult, smallResult);
        }
    }

    /// <summary>
    /// A module that uses conditional trainable parameters inside a loop with a fixed iteration count.
    /// Combines IF branching with loop iteration, exercising both the IF and LOOP
    /// handlers in ListAllSpecificModelIdsUsed.
    /// </summary>
    [Module]
    public partial class ConditionalTrainableParamInLoopLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> numIterations, [Hyper] Scalar<int64> threshold)
        {
            var x = input;

            foreach (var ctx in LoopAPI.Iterate(numIterations))
            {
                var lastDim = x.ShapeTensor()[-1L];
                var isLarge = lastDim > threshold;

                var weightsLarge = InitSimple.Init(x.ShapeTensor());
                var weightsSmall = InitSimple.Init(x.ShapeTensor());

                var largeResult = x * weightsLarge;
                var smallResult = x + weightsSmall;

                x = isLarge.IfElse(largeResult, smallResult);
            }

            return x;
        }
    }

    /// <summary>
    /// A module that uses conditional trainable parameters inside a loop where the
    /// number of iterations is derived from the input tensor's last dimension size.
    /// This makes the loop count a dynamic, non-constant-foldable value, exercising
    /// the ListAllSpecificModelIdsUsed LOOP handler with a dynamic iteration count.
    /// </summary>
    [Module]
    public partial class ConditionalTrainableParamInDynamicLoopLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> threshold)
        {
            var x = input;
            var numIterations = input.ShapeTensor()[-1L];

            foreach (var ctx in LoopAPI.Iterate(numIterations))
            {
                var lastDim = x.ShapeTensor()[-1L];
                var isLarge = lastDim > threshold;

                var weightsLarge = InitSimple.Init(x.ShapeTensor());
                var weightsSmall = InitSimple.Init(x.ShapeTensor());

                var largeResult = x * weightsLarge;
                var smallResult = x + weightsSmall;

                x = isLarge.IfElse(largeResult, smallResult);
            }

            return x;
        }
    }

    [Module]
    public partial class OptionalHypersLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] OptionalTensor<float32> optScale)
        {
            var weights = InitSimple.Init(input.ShapeTensor());
            return input * weights;
        }
    }

    [Module]
    public partial class OptionalHypersLayerStraight
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var model = OptionalHypersLayer.Model(Scalar(1.5f));
            return model.Call(inputs);
        }
    }

    [Module]
    public partial class OptionalHypersSequenceCalled
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var m1 = OptionalHypersLayer.Model(Scalar(1.5f));
            var m2 = OptionalHypersLayer.Model(Scalar(2.5f));
            var sequence = ModelSequence.Create(m1, m2);
            var x = sequence[Scalar(0L)].Call(inputs);
            x = sequence[Scalar(1L)].Call(x);
            return x;
        }
    }

    [Module]
    public partial class SeqHypersLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] TensorSequence<float32> scales)
        {
            var weights = InitSimple.Init(input.ShapeTensor());
            return input * weights;
        }
    }

    [Module]
    public partial class SeqHypersSequenceCalled
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var m1 = SeqHypersLayer.Model(Globals.TensorSequence<float32>(Scalar(1.5f), Scalar(2.5f)));
            var m2 = SeqHypersLayer.Model(Globals.TensorSequence<float32>(Scalar(3.5f)));
            var sequence = ModelSequence.Create(m1, m2);
            var x = sequence[Scalar(0L)].Call(inputs);
            x = sequence[Scalar(1L)].Call(x);
            return x;
        }
    }

    [Module]
    public partial class OptionalHypersEmptyThenAppend
    {
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var seq = ModelSequence.Empty(OptionalHypersLayer.Model());
            seq = seq.Append(OptionalHypersLayer.Model(Scalar(1.5f)));
            seq = seq.Append(OptionalHypersLayer.Model(Scalar(2.5f)));
            var x = seq[Scalar(0L)].Call(inputs);
            x = seq[Scalar(1L)].Call(x);
            return x;
        }
    }

    [Module]
    public partial class SequenceAtInNestedLoop
    {
        // Two SimplestLayer models assembled into a static sequence before any loop,
        // then accessed via SEQUENCE_AT with a dynamic inner-loop index inside a 2x2
        // nested loop — the minimal B1b scenario.
        public static Tensor<float32> Inline(Tensor<float32> inputs)
        {
            var m1 = SimplestLayer.Model();
            var m2 = SimplestLayer.Model();
            var seq = ModelSequence.Create(m1, m2);

            var x = inputs;
            foreach (var ctx0 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx1 in LoopAPI.Iterate(Scalar(2L)))
                {
                    x = seq[ctx1.IterationIndex].Call(x);
                }
            }
            return x;
        }
    }

    // ===================================================================
    //  Analytic op-semantics / control-flow checks promoted from the
    //  2026-06-12 framework behavior test campaign.
    //  Self-checking Scalar<bit> modules; ok-counting (cmp.IfElse(1,0))
    //  is used wherever a NaN could otherwise slip through a false
    //  comparison in a mismatch count.
    // ===================================================================

    /// <summary>Op semantics recorded by the campaign and not already audited
    /// elsewhere (int64 Mod, Cast truncation, and ArgMax ties/negative axes live
    /// in the Qee*AuditModules): int64 division truncates toward zero
    /// (−7/2 = −3, 7/−2 = −3, −7/−2 = 3); Slice supports negative steps (full
    /// reverse); Gather accepts negative indices (−1 → last element); ReduceSum
    /// of an EMPTY tensor is 0 (the additive identity); softmax of large equal
    /// logits is stable (exactly ⅓ each — no overflow/NaN).</summary>
    [Module]
    public partial class AnalyticOpSemanticsCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)   // x = [10, 20, 30]
        {
            var div = Vector(-7L, 7L, -7L) / Vector(2L, -2L, -2L);
            var divOk = AllWithin((div - Vector(-3L, -3L, 3L)).Cast<float32>(), 0f, 3);

            var rev = (Tensor<float32>)OnnxOp.Slice(
                Tensor(new long[] { 4L }, 1f, 2f, 3f, 4f), Vector(3L), Vector(-5L), Vector(0L), Vector(-1L));
            var revOk = AllWithin(rev - Tensor(new long[] { 4L }, 4f, 3f, 2f, 1f), 0f, 4);

            var last = (Tensor<float32>)OnnxOp.Gather(x, Vector(-1L), axis: 0);
            var gatherOk = AllWithin(last - Tensor(new long[] { 1L }, 30f), 1e-6f, 1);

            var emptySum = ((Tensor<float32>)VectorFill(0L, 1f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var emptyOk = ((emptySum - Scalar(0f)).Abs() <= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));

            var soft = Tensor(new long[] { 3L }, 1000f, 1000f, 1000f).Softmax(axis: -1);
            var softOk = AllWithin(soft - Tensor(new long[] { 3L }, 1f / 3f, 1f / 3f, 1f / 3f), 1e-6f, 3);

            return divOk + revOk + gatherOk + emptyOk + softOk > Scalar(4L);   // all 5 required
        }

        /// <summary>1 iff every |element| ≤ tol (NaN fails the comparison, so it can't pass).</summary>
        private static Scalar<int64> AllWithin(Tensor<float32> diff, float tol, long count)
            => ((diff.Abs() <= Scalar(tol)).Cast<int64>()
                    .Reduce(ReduceKind.Sum, keepDims: false).Scalar() > Scalar(count - 1))
                .IfElse(Scalar(1L), Scalar(0L));
    }

    /// <summary>IfElse must not leak NaN from the unselected branch: the false branch
    /// computes 0/0 (NaN elementwise) but the (runtime, non-foldable) condition selects
    /// the true branch, so the result must equal 2·x exactly.</summary>
    [Module]
    public partial class AnalyticIfElseNaNIsolationCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var zero = x - x;
            var nanBranch = zero / zero;   // 0/0 = NaN, elementwise
            var cond = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar() > Scalar(0f);
            var chosen = cond.IfElse(x * Scalar(2f), nanBranch);
            var okCount = ((chosen - x * Scalar(2f)).Abs() <= Scalar(0f)).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return okCount > Scalar(3L);   // all 4 elements exact (a NaN leak fails the compare)
        }
    }

    /// <summary>LoopAPI.Iterate with a runtime-derived trip count (min(x) = 3, so the
    /// LOOP survives constant folding all the way to the engine): doubling 3 times
    /// must give exactly 8·x.</summary>
    [Module]
    public partial class AnalyticLoopAccumulateCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)   // x = [3,4,5,6] → trips = 3
        {
            var acc = x;
            var trips = x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>();
            foreach (var ctx in LoopAPI.Iterate(trips))
            {
                acc = acc * Scalar(2f);
            }
            var okCount = ((acc - x * Scalar(8f)).Abs() <= Scalar(0f)).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return okCount > Scalar(3L);
        }
    }

    /// <summary>The zero-trip case: Iterate(min(x) = 0) must leave the carried value
    /// untouched (acc == x exactly).</summary>
    [Module]
    public partial class AnalyticLoopZeroTripCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)   // x = [0,5,7] → trips = 0
        {
            var acc = x;
            var trips = x.Reduce(ReduceKind.Min, keepDims: false).Scalar().Cast<int64>();
            foreach (var ctx in LoopAPI.Iterate(trips))
            {
                acc = acc * Scalar(2f);
            }
            var okCount = ((acc - x).Abs() <= Scalar(0f)).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return okCount > Scalar(2L);
        }
    }

    /// <summary>Pins the NaN-safety of the audit modules' mismatch counting
    /// (test-suite-gaps.md #5): a NaN actual must REGISTER as a mismatch. The plain
    /// "(diff > tol)" form is NaN-blind — IEEE comparisons with NaN are false — which
    /// is why every audit FloatMismatch helper is written as Not(diff &lt;= tol). This
    /// module produces NaN via 0/0 and asserts the NaN-safe form counts every element.</summary>
    [Module]
    public partial class AnalyticNaNMismatchGuardCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)   // x = [1, 2] (any 2 floats)
        {
            var nan = (x - x) / (x - x);                       // elementwise 0/0 = NaN
            var diff = (nan - x).Abs();                        // still NaN
            var counted = ((Tensor<bit>)OnnxOp.Not(diff <= Scalar(1e-3f))).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var blind = (diff > Scalar(1e-3f)).Cast<int64>()
                .Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            // The fixed form counts both NaN elements; the old form counts none (NaN-blind).
            return (counted > Scalar(1L)).IfElse(Scalar(1L), Scalar(0L))
                 + (blind < Scalar(1L)).IfElse(Scalar(1L), Scalar(0L)) > Scalar(1L);
        }
    }
}
