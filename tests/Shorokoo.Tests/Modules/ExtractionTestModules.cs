namespace Shorokoo.Tests.Modules
{
    // =====================================================
    // Modules for testing ExtractLoopVariableAsSequenceOutput via SimplifyTrainableParamInitializers
    // =====================================================

    /// <summary>
    /// Module with a single trainable parameter inside a loop.
    /// This exercises ExtractLoopVariableAsSequenceOutput for a single parameter extraction.
    /// </summary>
    [Module]
    public partial class SingleTrainableInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                var weights = InitSimple.Init(x.ShapeTensor());
                x = x * weights;
                ctx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module with multiple trainable parameters inside the same loop.
    /// This exercises ExtractLoopVariableAsSequenceOutput for multiple parameters from the same loop.
    /// </summary>
    [Module]
    public partial class MultiTrainablesSameLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var weights1 = InitSimple.Init(x.ShapeTensor());
                var weights2 = InitSimple.Init(x.ShapeTensor());
                x = x * weights1 + weights2;
                ctx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module with trainable parameters in different sequential loops.
    /// This exercises ExtractLoopVariableAsSequenceOutput for parameters from different loops.
    /// </summary>
    [Module]
    public partial class TrainablesInDifferentLoops
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            
            // First loop with trainable parameter
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var weights1 = InitSimple.Init(x.ShapeTensor());
                x = x * weights1;
                ctx.ContinueWhile(Scalar(true));
            }
            
            // Second loop with a different trainable parameter
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var weights2 = InitSimple.Init(x.ShapeTensor());
                x = x + weights2;
                ctx.ContinueWhile(Scalar(true));
            }
            
            return x;
        }
    }

    /// <summary>
    /// Module with trainable parameters in nested loops.
    /// The inner loop contains a trainable parameter.
    /// </summary>
    [Module]
    public partial class TrainableInNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var outerCtx in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var innerCtx in LoopAPI.Iterate(Scalar(2L)))
                {
                    var weights = InitSimple.Init(x.ShapeTensor());
                    x = x * weights;
                    innerCtx.ContinueWhile(Scalar(true));
                }
                outerCtx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module with trainable parameters in both outer and inner loops.
    /// This tests extraction from multiple loop scopes.
    /// </summary>
    [Module]
    public partial class TrainablesInBothLoopLevels
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var outerCtx in LoopAPI.Iterate(Scalar(2L)))
            {
                // Trainable in outer loop
                var outerWeights = InitSimple.Init(x.ShapeTensor());
                x = x * outerWeights;
                
                foreach (var innerCtx in LoopAPI.Iterate(Scalar(2L)))
                {
                    // Trainable in inner loop
                    var innerWeights = InitSimple.Init(x.ShapeTensor());
                    x = x + innerWeights;
                    innerCtx.ContinueWhile(Scalar(true));
                }
                outerCtx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module combining trainable parameters outside loops and inside loops.
    /// </summary>
    [Module]
    public partial class TrainablesMixedLoopAndNoLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            
            // Trainable parameter outside any loop
            var globalWeights = InitSimple.Init(x.ShapeTensor());
            x = x * globalWeights;
            
            // Trainable parameter inside loop
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var loopWeights = InitSimple.Init(x.ShapeTensor());
                x = x + loopWeights;
                ctx.ContinueWhile(Scalar(true));
            }
            
            return x;
        }
    }

    /// <summary>
    /// Module with multiple trainable parameters in a nested loop structure.
    /// Inner loop has two trainable parameters.
    /// </summary>
    [Module]
    public partial class MultiTrainablesNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var outerCtx in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var innerCtx in LoopAPI.Iterate(Scalar(2L)))
                {
                    var weights1 = InitSimple.Init(x.ShapeTensor());
                    var weights2 = InitSimple.Init(x.ShapeTensor());
                    x = x * weights1 + weights2;
                    innerCtx.ContinueWhile(Scalar(true));
                }
                outerCtx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module with trainable parameters in three levels of nested loops.
    /// Tests deeply nested loop extraction.
    /// </summary>
    [Module]
    public partial class TrainableInTripleNestedLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx1 in LoopAPI.Iterate(Scalar(2L)))
            {
                foreach (var ctx2 in LoopAPI.Iterate(Scalar(2L)))
                {
                    foreach (var ctx3 in LoopAPI.Iterate(Scalar(2L)))
                    {
                        var weights = InitSimple.Init(x.ShapeTensor());
                        x = x * weights;
                        ctx3.ContinueWhile(Scalar(true));
                    }
                    ctx2.ContinueWhile(Scalar(true));
                }
                ctx1.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module with trainable parameters using hyperparameters inside a loop.
    /// </summary>
    [Module]
    public partial class TrainableWithHyperInLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<int64> featureSize)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            {
                var weights = InitSimple.Init([featureSize]);
                x = x * weights;
                ctx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }

    /// <summary>
    /// Module with single iteration loop containing trainable parameter.
    /// Edge case testing.
    /// </summary>
    [Module]
    public partial class TrainableInSingleIterationLoop
    {
        public static Tensor<float32> Inline(Tensor<float32> input)
        {
            var x = input;
            foreach (var ctx in LoopAPI.Iterate(Scalar(1L)))
            {
                var weights = InitSimple.Init(x.ShapeTensor());
                x = x * weights;
                ctx.ContinueWhile(Scalar(true));
            }
            return x;
        }
    }
}
