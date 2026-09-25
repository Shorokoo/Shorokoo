using Shorokoo.Modules.Initializers;
using Shorokoo.Modules.Layers;

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

    /// <summary>One param whose shape is static (it resolves with no hints at all) beside one whose
    /// shape needs the input's value, so concretizing without hints leaves a mix of resolved and
    /// unresolved param sites.</summary>
    [Module]
    public partial class StaticAndInputShapedParamsLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var scale = InitSimple.Init([Scalar(1L)]);
            var weights = InitSimple.Init(input.ShapeTensor());
            return input * weights * scale;
        }
    }

    // Returns a rank-pinned Vector<float32>. Constructing the auto-generated
    // Module<Tensor<float32>, Vector<float32>> goes through
    // ModuleHelper.CreateFunctionSignature, which declares each output's rank on its
    // output node from the compile-time output type — Vector<T> hits the IVector→1 branch
    // and produces OutputRanks = [1]. The C# codegen path on the generated
    // Function relies on that declared rank to emit `Vector<float32>` (rather than
    // `Tensor<float32>`) as the return type, so this is the natural circumstance
    // that exercises declared-rank propagation end-to-end.
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

    /// <summary>A gated trainable parameter, as in <c>Linear</c>'s <c>useBias</c>: the bias exists
    /// only on the <c>true</c> branch, so a <c>false</c> concretization hint prunes it.</summary>
    [Module]
    public partial class GatedBiasHyperLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<bit> useBias)
        {
            var bias = InitSimple.Init(input.ShapeTensor());
            return useBias.IfElse(input + bias, input);
        }
    }

    /// <summary>One <c>[Hyper]</c> bit gating two <c>IfElse</c>es — one holding a trainable
    /// param, one not.</summary>
    [Module]
    public partial class TwoGatesOneHyperLayer
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> input, [Hyper] Scalar<bit> flag)
        {
            var gain = InitSimple.Init(input.ShapeTensor());
            var paramless = flag.IfElse(input * Scalar(10f), input * Scalar(100f));
            var gated = flag.IfElse(input * gain, input * Scalar(7f));
            return (paramless, gated);
        }
    }

    /// <summary>A <c>[256, 256]</c> trainable param behind a gate: big enough that a dense
    /// stand-in for it is visible against the small constants a graph normally carries.</summary>
    [Module]
    public partial class BigGatedParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<bit> useBig)
        {
            var big = InitSimple.Init(Vector(256L, 256L));
            return useBig.IfElse(input + big.Reduce(ReduceKind.Sum, Vector(0L, 1L), keepDims: false), input);
        }
    }

    /// <summary>A <c>[256, 256]</c> param behind nested gates: with the outer gate off and the
    /// inner on, the branch stays reachable, so the param's stand-in must still cost nothing.</summary>
    [Module]
    public partial class NestedBigGatedParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<bit> outer, [Hyper] Scalar<bit> inner)
        {
            var big = InitSimple.Init(Vector(256L, 256L));
            var innerVal = inner.IfElse(input + big.Reduce(ReduceKind.Sum, Vector(0L, 1L), keepDims: false), input * Scalar(3f));
            return outer.IfElse(innerVal, input);
        }
    }

    /// <summary>Param on the <b>else</b> branch and a <b>computed</b> (non-input) condition — the
    /// mirror of the usual then-branch, bit-valued gate.</summary>
    [Module]
    public partial class ElseBranchComputedGateLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> mode)
        {
            var w = InitSimple.Init(input.ShapeTensor());
            return (mode > Scalar(3L)).IfElse(input * Scalar(5f), input + w);
        }
    }

    /// <summary>An <c>IfElse</c> inside a <c>LoopAPI.Iterate</c> body — the loop and IF nesting
    /// conventions meeting. No trainable params, so nothing about parameter pruning applies.</summary>
    [Module]
    public partial class LoopGateHyperLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters, [Hyper] Scalar<bit> flag)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(iters))
                x = flag.IfElse(x + Scalar(1f), x * Scalar(2f));
            return x;
        }
    }

    /// <summary>An <c>IfElse</c> in a loop body whose taken branch unwraps an optional —
    /// loop-invariant, but only valid when the branch is taken.</summary>
    [Module]
    public partial class LoopOptionalLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters, OptionalTensor<float32> bias)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(iters))
                x = bias.HasValue().IfElse(x + bias.TensorValue(), x * Scalar(2f));
            return x;
        }
    }

    /// <summary>The other nesting: a <c>LoopAPI.Iterate</c> inside an <c>IfElse</c> branch,
    /// with a loop-invariant node in the loop body.</summary>
    [Module]
    public partial class IfLoopBodyLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters, Scalar<bit> flag)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(iters))
                x = x + input * Scalar(3f);
            return flag.IfElse(x, input);
        }
    }

    /// <summary>Three things an <c>IfElse</c>'s branch scoping must leave outside it: a value both
    /// branches read, one a branch reads that is read again afterwards, and the condition.</summary>
    [Module]
    public partial class SharedWorkAroundAnIfLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, Scalar<float32> gate)
        {
            var shared = input * Scalar(2f);
            var reused = input.Relu();
            return (gate > Scalar(1f)).IfElse(shared + reused, shared) / reused;
        }
    }

    /// <summary>A wholly loop-invariant <c>IfElse</c> in a loop body, consumed by an equally
    /// loop-invariant node. Its <c>IF_CLOSE</c> is pinned in the body, so the consumer is not
    /// hoistable however invariant its own data is.</summary>
    [Module]
    public partial class InvariantGateInLoopLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters, Scalar<bit> flag)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(iters))
                x = x + (flag.IfElse(input * Scalar(2f), input * Scalar(3f)) + input);
            return x;
        }
    }

    /// <summary>An <c>IfElse</c> around a loop around an <c>IfElse</c>. Shorokoo/Shorokoo#270.</summary>
    [Module]
    public partial class IfInLoopInIfLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters, Scalar<bit> flag)
        {
            var x = input;
            foreach (var o in LoopAPI.Iterate(iters))
                x = x + (flag.IfElse(input * Scalar(2f), input * Scalar(3f)) + input);
            return flag.IfElse(x, input * Scalar(9f));
        }
    }

    /// <summary>A loop inside an <c>IfElse</c> branch inside a loop. Shorokoo/Shorokoo#270.</summary>
    [Module]
    public partial class LoopInIfInLoopLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters, Scalar<bit> flag)
        {
            var x = input;
            foreach (var o in LoopAPI.Iterate(iters))
            {
                var y = input;
                foreach (var i in LoopAPI.Iterate(iters)) y = y + (input * Scalar(2f));
                x = x + (flag.IfElse(y, input * Scalar(7f)) + input);
            }
            return x;
        }
    }

    /// <summary>A draw scanned over a loop whose trip count is a literal, so the loop unrolls.
    /// Each unrolled iteration must get its own draw.</summary>
    [Module]
    public partial class ConstantTripDrawScanLayer
    {
        public static Tensor<float32> Inline(Scalar<float32> unused)
        {
            Variable? scanned = null;
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
                scanned = (Variable)ctx.Scan((Scalar<float32>)(Variable)
                    OnnxOp.RandomUniform([], high: 1f, low: 0f, dtype: DType.Float32));
            return (Tensor<float32>)scanned!;
        }
    }

    /// <summary>Used only from <see cref="InitializerFirstUsedInLoopBodyLayer"/>, so the
    /// process-wide Function cache is cold when that module is built however the suite is
    /// ordered — which is what the pin over it needs.</summary>
    [TrainableParamInitializer]
    public static partial class InitOnlyUsedInALoopBody
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => TensorFill(shape, 1.0f);
    }

    /// <summary>An initializer first used inside a loop body. Its input markers must be built in
    /// its own trace, not the caller's, or the enclosing loop records them as body nodes on the
    /// first pass alone and the pass-to-pass comparison rejects the body.</summary>
    [Module]
    public partial class InitializerFirstUsedInLoopBodyLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> iters)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(iters))
                x = x + InitOnlyUsedInALoopBody.Init(x.ShapeTensor());
            return x;
        }
    }

    /// <summary>One gate with a trainable param on <b>each</b> branch, both pruned by an
    /// enclosing gate. Whichever branch wins, the other is the one that unlocks the fold.</summary>
    [Module]
    public partial class ParamOnBothBranchesLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<bit> outer, [Hyper] Scalar<bit> flag)
        {
            var a = InitSimple.Init(input.ShapeTensor());
            var b = InitSimple.Init(input.ShapeTensor());
            var inner = flag.IfElse(input + a, input - b);
            return outer.IfElse(inner, input * Scalar(9f));
        }
    }

    /// <summary><see cref="ParamOnBothBranchesLayer"/> under a Softsign, an op the
    /// QuickExecutionEngine lowers before it walks a graph.</summary>
    [Module]
    public partial class ParamOnBothBranchesSoftsignLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<bit> outer, [Hyper] Scalar<bit> flag)
        {
            var a = InitSimple.Init(input.ShapeTensor());
            var b = InitSimple.Init(input.ShapeTensor());
            var inner = flag.IfElse(input + a, input - b);
            return outer.IfElse(inner, input * Scalar(9f)).Softsign();
        }
    }

    /// <summary>A parameter under a Softsign.</summary>
    [Module]
    public partial class SoftsignOfParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => (input * InitSimple.Init(input.ShapeTensor())).Softsign();
    }

    /// <summary>A rank-0 trainable param behind nested gates — the stand-in emitter's
    /// no-dims path, which cannot EXPAND.</summary>
    [Module]
    public partial class NestedRank0GatedParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<bit> outer, [Hyper] Scalar<bit> inner)
        {
            var s = Shorokoo.Modules.Initializers.ScalarOnes.Init();
            var innerVal = inner.IfElse(input * s, input * Scalar(3f));
            return outer.IfElse(innerVal, input);
        }
    }

    /// <summary>Tuple <c>IfElse</c> on one gate: slot 0 holds a trainable param, slot 1 none.
    /// Folding is all-or-nothing across an <c>IF_CLOSE</c>'s slots, so slot 1 pins whether a
    /// pruned param in slot 0 may fold the whole node.</summary>
    [Module]
    public partial class TupleGateHyperLayer
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> input, [Hyper] Scalar<bit> flag)
        {
            var g = InitSimple.Init(input.ShapeTensor());
            return flag.IfElse((input + g, input * Scalar(10f)), (input, input * Scalar(100f)));
        }
    }

    /// <summary>One trainable param read by two gates — a <c>[Hyper]</c> and a runtime bit — so
    /// neither branch exclusively owns it.</summary>
    [Module]
    public partial class SharedDeadParamTwoGatesLayer
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(
            Tensor<float32> input, [Hyper] Scalar<bit> useBias, Scalar<bit> runtimeFlag)
        {
            var bias = InitSimple.Init(input.ShapeTensor());
            var gated = useBias.IfElse(input + bias, input);
            var other = runtimeFlag.IfElse(input + bias, input * Scalar(2f));
            return (gated, other);
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
    public partial class SeqCountShapedParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, TensorSequence<float32> seq)
        {
            var scale = InitSimple.Init([Scalar(1L)]);
            var weights = InitSimple.Init([seq.Count]);
            return input * weights * scale;
        }
    }

    [Module]
    public partial class ValueShapedParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> input, Vector<int64> sizes)
        {
            var weights = InitSimple.Init([sizes[0L]]);
            return input * weights;
        }
    }

    [Module]
    public partial class SeqThenConvTransposeLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, [Hyper] TensorSequence<float32> scales)
            => Convolution.ConvTranspose(x, 2L, kernelSize: [2L, 2L], stride: [2L, 2L], outputShape: [7L, 7L]);
    }

    [Module]
    public partial class OptionalThenConvTransposeLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, OptionalTensor<float32> bias)
            => Convolution.ConvTranspose(x, 2L, kernelSize: [2L, 2L], stride: [2L, 2L], outputShape: [7L, 7L]);
    }

    [Module]
    public partial class SequenceElementAddLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, TensorSequence<float32> s)
            => x + s[Scalar(0L)];
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
                Tensor([4L], 1f, 2f, 3f, 4f), Vector(3L), Vector(-5L), Vector(0L), Vector(-1L));
            var revOk = AllWithin(rev - Tensor([4L], 4f, 3f, 2f, 1f), 0f, 4);

            var last = (Tensor<float32>)OnnxOp.Gather(x, Vector(-1L), axis: 0);
            var gatherOk = AllWithin(last - Tensor([1L], 30f), 1e-6f, 1);

            var emptySum = ((Tensor<float32>)VectorFill(0L, 1f)).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var emptyOk = ((emptySum - Scalar(0f)).Abs() <= Scalar(0f)).IfElse(Scalar(1L), Scalar(0L));

            var soft = Tensor([3L], 1000f, 1000f, 1000f).Softmax(axis: -1);
            var softOk = AllWithin(soft - Tensor([3L], 1f / 3f, 1f / 3f, 1f / 3f), 1e-6f, 3);

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

    /// <summary>Pins the NaN-safety of the audit modules' mismatch counting:
    /// a NaN actual must REGISTER as a mismatch. The plain
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

    /// <summary>A body that hands its argument straight back.</summary>
    [Module]
    public partial class PassThroughSub
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input;
    }

    /// <summary>Calls <see cref="PassThroughSub"/>: the callee's output is one of its own inputs.</summary>
    [Module]
    public partial class CallerOfPassThroughSub
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => PassThroughSub.Call(input) * Scalar(2f);
    }

    /// <summary>Two arguments handed back swapped.</summary>
    [Module]
    public partial class SwapSub
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> a, Tensor<float32> b) => (b, a);
    }

    /// <summary>Calls <see cref="SwapSub"/>: both outputs alias inputs, crosswise.</summary>
    [Module]
    public partial class CallerOfSwapSub
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var (first, second) = SwapSub.Call(input, input * Scalar(2f));
            return first - second;
        }
    }

    /// <summary>One interior value returned twice — a body whose second output repeats its first.</summary>
    [Module]
    public partial class RepeatedOutputSub
    {
        public static (Tensor<float32>, Tensor<float32>) Inline(Tensor<float32> a)
        {
            var t = a * Scalar(3f);
            return (t, t);
        }
    }

    /// <summary>Calls <see cref="RepeatedOutputSub"/>.</summary>
    [Module]
    public partial class CallerOfRepeatedOutputSub
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var (first, second) = RepeatedOutputSub.Call(input);
            return first + second;
        }
    }

    /// <summary>A pass-through call carrying the loop variable: the loop's carry updater is the
    /// carry itself.</summary>
    [Module]
    public partial class LoopCarriedPassThrough
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
                x = PassThroughSub.Call(x);
            return x;
        }
    }

    /// <summary>Machinery-free callee for the initializer-calls-a-module fixtures below.</summary>
    [Module]
    public partial class DoublerSub
    {
        public static Tensor<float32> Inline(Tensor<float32> v) => v * Scalar(2f);
    }

    /// <summary>Callee with a [Hyper].</summary>
    [Module]
    public partial class HyperDoublerSub
    {
        public static Tensor<float32> Inline(Tensor<float32> v, [Hyper] Scalar<float32> k) => v * k;
    }

    /// <summary>Calls a module that itself calls one, so the callee's own body keeps a sub-module call.</summary>
    [Module]
    public partial class CallerOfCallerOfPassThroughSub
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => CallerOfPassThroughSub.Call(input);
    }

    // ---- Initializers that create or reference a model: refused (FW055) when their body is built.

    [TrainableParamInitializer]
    public static partial class InitCallingAModule
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => DoublerSub.Call(Globals.TensorFill(shape, 1.0f));
    }

    [Module]
    public partial class UsesInitCallingAModule
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * InitCallingAModule.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAParamOwningModule
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
            => SimplestLayer.Call(Globals.TensorFill(shape, 1.0f));
    }

    [Module]
    public partial class UsesInitCallingAParamOwningModule
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitCallingAParamOwningModule.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitCallingHyperModule
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
            => HyperDoublerSub.Call(Scalar(2f), Globals.TensorFill(shape, 1.0f));
    }

    [Module]
    public partial class UsesInitCallingHyperModule
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * InitCallingHyperModule.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitCreatingAModel
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            SimplestLayer.Model();
            return Globals.TensorFill(shape, 1.0f);
        }
    }

    [Module]
    public partial class UsesInitCreatingAModel
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * InitCreatingAModel.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAModelFromASequence
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            var seq = ModelSequence.Create(SimplestLayer.Model(), SimplestLayer.Model());
            return seq[Scalar(0L)].Call(Globals.TensorFill(shape, 1.0f));
        }
    }

    [Module]
    public partial class UsesInitCallingAModelFromASequence
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitCallingAModelFromASequence.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitReadingAModelsParam
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
            => SimplestLayer.Model().GetTrainableParam<float32>([1], rank: 1);
    }

    [Module]
    public partial class UsesInitReadingAModelsParam
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * InitReadingAModelsParam.Init(Vector(2L));
    }

    /// <summary>Written without the generator, whose Init has no way to pass a model argument.</summary>
    public static class InitTakingAModel
    {
        public static Tensor<float32> Inline(Vector<int64> shape, Model<Tensor<float32>, Tensor<float32>> m)
            => Globals.TensorFill(shape, 1.0f);

        public static Tensor<float32> Init(Vector<int64> shape, IModel m)
            => Globals.CallTrainableParamInitializer<float32>(
                (Func<Vector<int64>, Model<Tensor<float32>, Tensor<float32>>, Tensor<float32>>)Inline,
                nameof(InitTakingAModel), isTrainable: true, shape, m.ModelVariable);
    }

    [Module]
    public partial class UsesInitTakingAModel
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitTakingAModel.Init(Vector(2L), SimplestLayer.Model());
    }

    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class StateInitCallingAModule
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => DoublerSub.Call(Globals.TensorFill(shape, 1.0f));
    }

    [Module]
    public partial class UsesStateInitCallingAModule
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * StateInitCallingAModule.Init(Vector(2L));
    }

    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class StateInitCreatingAModel
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            SimplestLayer.Model();
            return Globals.TensorFill(shape, 1.0f);
        }
    }

    [Module]
    public partial class UsesStateInitCreatingAModel
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * StateInitCreatingAModel.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAModelCreatingInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => StateInitCreatingAModel.Init(shape);
    }

    [Module]
    public partial class UsesInitCallingAModelCreatingInitializer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitCallingAModelCreatingInitializer.Init(Vector(2L));
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAModuleInALoop
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            var v = Globals.TensorFill(shape, 1.0f);
            foreach (var _ in LoopAPI.Iterate(Scalar(2L))) v = DoublerSub.Call(v);
            return v;
        }
    }

    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class StateInitCallingAModuleCallingInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => InitCallingAModuleInALoop.Init(shape);
    }

    [Module]
    public partial class UsesStateInitCallingAModuleCallingInitializer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * StateInitCallingAModuleCallingInitializer.Init(Vector(2L));
    }

    // ---- Initializers calling initializers: the call is transparent, only the top-level one is a parameter.

    /// <summary>Fills with 2, so a parameter initialized by it is told apart from an InitSimple one.</summary>
    [TrainableParamInitializer]
    public static partial class InitTwos
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => Globals.TensorFill(shape, 2.0f);
    }

    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class StateInitTwos
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => Globals.TensorFill(shape, 2.0f);
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAnotherInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => InitTwos.Init(shape);
    }

    [Module]
    public partial class UsesInitCallingAnotherInitializer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitCallingAnotherInitializer.Init(input.ShapeTensor());
    }

    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class StateInitCallingAnotherInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => InitTwos.Init(shape);
    }

    [Module]
    public partial class UsesStateInitCallingAnotherInitializer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * StateInitCallingAnotherInitializer.Init(input.ShapeTensor());
    }

    [StateInitializer(Ownership = StateOwnership.ModuleOwned)]
    public static partial class StateInitCallingAStateInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => StateInitTwos.Init(shape);
    }

    [Module]
    public partial class UsesStateInitCallingAStateInitializer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * StateInitCallingAStateInitializer.Init(input.ShapeTensor());
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAStateInitializer
    {
        public static Tensor<float32> Inline(Vector<int64> shape) => StateInitTwos.Init(shape);
    }

    [Module]
    public partial class UsesInitCallingAStateInitializer
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitCallingAStateInitializer.Init(input.ShapeTensor());
    }

    [TrainableParamInitializer]
    public static partial class InitCallingAnInitializerCallingAnother
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
            => StateInitCallingAnotherInitializer.Init(shape) * Scalar(1.5f);
    }

    [Module]
    public partial class UsesInitCallingAnInitializerCallingAnother
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
            => input * InitCallingAnInitializerCallingAnother.Init(input.ShapeTensor());
    }

    /// <summary>An initializer handed another parameter's initialized value.</summary>
    [TrainableParamInitializer]
    public static partial class InitDoublingAnotherParam
    {
        public static Tensor<float32> Inline(Vector<int64> shape, Tensor<float32> source)
            => source * Scalar(2.0f);
    }

    /// <summary>Drives InitDoublingAnotherParam: a parameter at 1, and beside it its double.</summary>
    [Module]
    public partial class UsesInitFromAnotherParam
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var source = InitSimple.Init(input.ShapeTensor());
            return input * source * InitDoublingAnotherParam.Init(input.ShapeTensor(), source);
        }
    }

    /// <summary>An initializer whose body loops: its loop's subgraph inputs still need types in the
    /// emitted body.</summary>
    [TrainableParamInitializer]
    public static partial class InitLoopingWithoutACall
    {
        public static Tensor<float32> Inline(Vector<int64> shape)
        {
            var v = Globals.TensorFill(shape, 1.0f);
            foreach (var _ in LoopAPI.Iterate(Scalar(2L))) v = v * Scalar(2f);
            return v;
        }
    }

    /// <summary>Drives InitLoopingWithoutACall.</summary>
    [Module]
    public partial class UsesInitLoopingWithoutACall
    {
        public static Tensor<float32> Inline(Tensor<float32> input) => input * InitLoopingWithoutACall.Init(Vector(2L));
    }

    /// <summary>One [Module] whose width, depth, head count, class count, bias use and head shape
    /// are all [Hyper]s, so each variant of it is a value rather than another class. Mirrors the
    /// worked example in <c>Documentation/defining-models.md</c>.</summary>
    [Module]
    public partial class HyperParameterizedVisionTransformer
    {
        public static Tensor<float32> Inline(
            Tensor<float32> patches,
            [Hyper] Scalar<int64> embedDim,
            [Hyper] Scalar<int64> numHeads,
            [Hyper] Scalar<int64> ffnDim,
            [Hyper] Scalar<int64> numLayers,
            [Hyper] Scalar<int64> numClasses,
            [Hyper] Scalar<bit> useBias,
            [Hyper] Scalar<bit> useMlpHead)
        {
            var proj = XavierUniform.Init([patches.DimTensor(2), embedDim]);
            var pos = XavierUniform.Init([patches.DimTensor(1), embedDim]);
            var x = patches.MatMul(proj) + pos;

            foreach (var ctx in LoopAPI.Iterate(numLayers))
                x = TransformerEncoderLayer.Call(embedDim, numHeads, ffnDim, useBias, x);

            Vector<int64> seqAxis = [Scalar(1L)];
            var pooled = x.Reduce(ReduceKind.Mean, seqAxis, keepDims: false);

            var wHead = XavierUniform.Init([embedDim, numClasses]);
            var wHidden = XavierUniform.Init([embedDim, embedDim]);
            var wOut = XavierUniform.Init([embedDim, numClasses]);
            return useMlpHead.IfElse(
                pooled.MatMul(wHidden).Tanh().MatMul(wOut),
                pooled.MatMul(wHead));
        }
    }

    /// <summary>An output whose rank its flag decides: the input as it is, or flattened.</summary>
    [Module]
    public partial class RankByFlagLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Scalar<bit> flag)
            => flag.IfElse(x, (Tensor<float32>)OnnxOp.Reshape(x, Vector(-1L), false));
    }

    /// <summary>An output whose rank the values of its axes decide.</summary>
    [Module]
    public partial class SqueezeByAxesLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Vector<int64> axes)
            => (Tensor<float32>)OnnxOp.Squeeze(x, axes);
    }

    /// <summary>A struct output.</summary>
    [Module]
    public partial class StructOutputLayer
    {
        public static GenericPairStruct Inline(Scalar<float32> a) => TensorStruct<GenericPairStruct>(a, a);
    }

    /// <summary>A struct output beside a tensor output.</summary>
    [Module]
    public partial class StructInATupleOutputLayer
    {
        public static (GenericPairStruct, Tensor<float32>) Inline(Scalar<float32> a, Tensor<float32> x)
            => (TensorStruct<GenericPairStruct>(a, a), x);
    }

    /// <summary>A sequence of like elements and one of unlike elements.</summary>
    [Module]
    public partial class SequenceOutputsLayer
    {
        public static (TensorSequence<float32>, TensorSequence<float32>) Inline(Scalar<float32> a, Tensor<float32> x)
            => (Globals.TensorSequence<float32>(x, x), Globals.TensorSequence<float32>(x, a));
    }

    /// <summary>A string input split on spaces: an op the QuickExecutionEngine does not compute.</summary>
    [Module]
    public partial class StringSplitLayer
    {
        public static Tensor<utf8> Inline(Tensor<utf8> x)
        {
            var (split, _) = OnnxOp.StringSplit(x, delimiter: " ");
            return (Tensor<utf8>)split;
        }
    }

    /// <summary>A string input dropping a stopword in the default locale, which not every
    /// machine has: an op the QuickExecutionEngine gives a rank but no extent.</summary>
    [Module]
    public partial class StopwordNormalizerLayer
    {
        public static Tensor<utf8> Inline(Tensor<utf8> x) => NN.StringNormalizer(x, caseChangeAction: "LOWER", stopwords: ["a"]);
    }

    /// <summary>A string normalized in a locale no machine has, beside a weighted output.</summary>
    [Module]
    public partial class UnknownLocaleNormalizerBesideAWeightLayer
    {
        public static (Tensor<utf8>, Tensor<float32>) Inline(Tensor<utf8> s, Tensor<float32> y)
            => (NN.StringNormalizer(s, caseChangeAction: "LOWER", locale: "xx_XX", stopwords: ["a"]),
                y * InitSimple.Init(y.ShapeTensor()));
    }

    /// <summary>A tensor scaled by a scalar: an input of no declared rank beside one of rank 0.</summary>
    [Module]
    public partial class TensorTimesScalarLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x, Scalar<float32> s) => x * s;
    }

    /// <summary>An optional output: the optional input it is handed.</summary>
    [Module]
    public partial class OptionalPassThroughLayer
    {
        public static OptionalTensor<float32> Inline(OptionalTensor<float32> bias) => bias;
    }

    /// <summary>An output whose rank a parameter's value decides: the input as it is while the
    /// parameter sums positive, else flattened.</summary>
    [Module]
    public partial class RankByParamLayer
    {
        public static Tensor<float32> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init(x.ShapeTensor());
            var sum = (Scalar<float32>)OnnxOp.ReduceSum(w, null, keepdims: false, noopWithEmptyAxes: null);
            var positive = (Scalar<bit>)OnnxOp.Greater(sum, Scalar(0f));
            return positive.IfElse(x, (Tensor<float32>)OnnxOp.Reshape(x, Vector(-1L), false));
        }
    }

    /// <summary>An output whose dims a parameter's values decide.</summary>
    [Module]
    public partial class NonZeroOfParamLayer
    {
        public static Tensor<int64> Inline(Tensor<float32> x)
            => (Tensor<int64>)OnnxOp.NonZero(InitSimple.Init([Scalar(3L)]) * x);
    }
}
