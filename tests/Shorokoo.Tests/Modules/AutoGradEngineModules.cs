namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Phase 4 AD-B2 modules for the autograd ENGINE path-checking semantics in
    /// <c>FastProcessAutoGrad</c> plus the new AD003 attribute-envelope guards.
    /// Same self-checking pattern as <c>AutoGradStructuralModules.cs</c> where the
    /// scenario differentiates successfully; the AD003 scenarios are driven through
    /// <c>Assert.Throws</c> in <c>AutoGradEngineTests</c> (the exception surfaces
    /// from the AUTO_GRAD lowering during <c>AdvancedTestGraph</c>'s concretization,
    /// before anything executes).
    /// </summary>

    // ===================================================================
    //  Engine: unregistered op (dynamic Loop) on the loss→param path
    // ===================================================================

    /// <summary>
    /// A dynamic LOOP (runtime-dependent trip count, so the static unroll pass
    /// can't remove it) sits between the loss and the AUTO_GRAD parameter
    /// <c>x</c>. Loop ops have no registered gradient; the engine must throw
    /// AD003 at lowering instead of silently cutting the chain (which would
    /// hand <c>x</c> a zeros gradient — a silently frozen parameter).
    /// </summary>
    [Module]
    public partial class AutoGradEngineLoopOnParamPathCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var state = x.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            // Trip count derived from x's runtime VALUES: not constant-foldable,
            // so the LOOP survives to the autograd lowering.
            var iter = state.Abs().Cast<int64>() + Scalar(1L);
            foreach (var ctx in LoopAPI.Iterate(iter))
            {
                state = state * Scalar(1.5f);
            }
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, state);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e9f);
        }
    }

    // ===================================================================
    //  Engine: unregistered leaf op (RandomNormal) feeding the loss path
    //  with NO parameter behind it → legitimate gradient leaf, cut silently
    // ===================================================================

    /// <summary>
    /// <c>loss = Σ (x · r)</c> where <c>r</c> is a RandomNormal draw — an op with
    /// no gradient entry at all. Since no AUTO_GRAD parameter lies in <c>r</c>'s
    /// ancestry, the engine must treat it as a legitimate gradient leaf and cut
    /// the chain there (NOT throw), and the gradient of <c>x</c> is exactly
    /// <c>r</c> (same forward tensor, so the comparison is exact regardless of
    /// the sampled values).
    /// </summary>
    [Module]
    public partial class AutoGradEngineRandomLeafCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var r = Globals.RandomNormal(Vector(4L), mean: 0.0f, scale: 1.0f, seed: 7f);
            var loss = (x * r).Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var diff = (grad - r).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-5f);
        }
    }

    // ===================================================================
    //  Slice with steps != 1: gradient now implemented (flat-index scatter)
    // ===================================================================

    /// <summary>
    /// Slice with steps=2 ([1:6:2] of a [6] vector selects x1, x3, x5), so
    /// dL/dx = [0,1,0,1,0,1]. The old Pad-based gradient assumed step 1 and
    /// would have produced the contiguous [0,1,1,1,1,0]; the new flat-index
    /// scatter path (taken whenever a steps input is wired) places the grads
    /// exactly.
    /// </summary>
    [Module]
    public partial class AutoGradEngineSliceStepsCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var sliced = (Tensor<float32>)OnnxOp.Slice(x,
                starts: Vector(1L), ends: Vector(6L), axes: Vector(0L), steps: Vector(2L));
            var loss = sliced.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            var expected = Vector(0f, 1f, 0f, 1f, 0f, 1f).Tensor();
            var diff = (grad - expected).Abs();
            return diff.Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e-4f);
        }
    }

    // ===================================================================
    //  AD003 guards: unsupported attribute combinations must throw loudly
    //  (asserted via Assert.Throws in AutoGradEngineTests — the modules just
    //  put the offending op on a loss→param path)
    // ===================================================================

    /// <summary>Pad with mode='reflect' on the loss path → AD003 at lowering
    /// (the reflect adjoint needs a scatter-add of the border grads; reusing the
    /// constant-mode slice was silently wrong).</summary>
    [Module]
    public partial class AutoGradEnginePadReflectThrowCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var padded = (Tensor<float32>)OnnxOp.Pad(x, Vector(1L, 1L), mode: PadMode.Reflect);
            var loss = padded.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e9f);
        }
    }

    /// <summary>ScatterND with reduction='mul' on the loss path → AD003 at
    /// lowering (the mul reduction's product-quotient adjoint is unimplemented;
    /// falling into the 'none' branch was silently wrong).</summary>
    [Module]
    public partial class AutoGradEngineScatterMulThrowCheck
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var indices = (Tensor<int64>)OnnxOp.Reshape(Vector(1L), Vector(1L, 1L), allowZero: false);
            var updates = (Tensor<float32>)OnnxOp.Expand(Scalar(2f), Vector(1L));
            var scattered = (Tensor<float32>)OnnxOp.ScatterND(x, indices, updates, ScatterNDReduction.Mul);
            var loss = scattered.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            var grad = (Tensor<float32>)Shorokoo.Core.Nodes.AutoDiff.Ops.AutoGrad(x, loss);
            // Never reached: AUTO_GRAD lowering throws AD003 (asserted by the test).
            return grad.Abs().Reduce(ReduceKind.Max, keepDims: false).Scalar() < Scalar(1e9f);
        }
    }
}
