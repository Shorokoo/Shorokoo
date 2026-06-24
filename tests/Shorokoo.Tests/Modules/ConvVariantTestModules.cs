namespace Shorokoo.Tests.Modules
{
    /// <summary>
    /// Self-checking module: a Conv built with the tensor-geometry overload
    /// (<c>NN.Conv</c> with Vector/Scalar geometry → SHRK_CONV, geometry as int64 tensor inputs) must
    /// produce the same output as the standard ONNX Conv with identical static geometry, after
    /// <c>FastLowerAttributeTensorOps</c> resolves the tensor inputs back to static attributes.
    /// Uses non-trivial geometry (stride 2, asymmetric-capable pads, dilation 1) so real values
    /// are exercised, not just defaults. Returns a single Scalar&lt;bit&gt; so AutoTest treats it
    /// as a self-checking graph.
    /// </summary>
    [Module]
    public partial class ConvVariantMatchesStandard
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(3L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();

            var standard = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                pads: [1L, 1L, 1L, 1L], strides: [2L, 2L]);

            var variant = NN.Conv(x, w, b, AutoPad.NotSet,
                pads: Vector(1L, 1L, 1L, 1L), strides: Vector(2L, 2L),
                dilations: Vector(1L, 1L), kernelShape: Vector(3L, 3L), group: Scalar(1L));

            var diff = (standard - variant).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return diff < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// Variant Conv whose <c>kernel_shape</c> is computed from the input tensor's spatial shape
    /// (kernel = [H-2, W-2]) instead of a literal. Because the geometry subgraph reads
    /// <c>Shape(x)</c> of a model input, it cannot be resolved by constant folding alone — it
    /// forces <c>FastLowerAttributeTensorOps</c> down its sample-input (QEE/ORT) resolution path.
    /// For the test sample x:[1,3,5,5] the kernel resolves to [3,3], matching the weight
    /// [2,3,3,3] and the standard reference Conv. Self-checking (returns Scalar&lt;bit&gt;).
    /// </summary>
    [Module]
    public partial class ConvVariantShapeDependentAttrs
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(2L), Scalar(3L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(2L)]).Vec();

            var shape = x.ShapeTensor();
            var h = shape[2];
            var wDim = shape[3];

            // kernel_shape derived from the input spatial dims: [H-2, W-2] = [3,3] for x:[1,3,5,5].
            Vector<int64> kernelShape = [h - Scalar(2L), wDim - Scalar(2L)];

            var standard = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [1L, 1L], group: 1L, kernelShape: [3L, 3L],
                pads: [0L, 0L, 0L, 0L], strides: [1L, 1L]);

            var variant = NN.Conv(x, w, b, AutoPad.NotSet,
                pads: Vector(0L, 0L, 0L, 0L), strides: Vector(1L, 1L),
                dilations: Vector(1L, 1L), kernelShape: kernelShape, group: Scalar(1L));

            var diff = (standard - variant).Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            return diff < Scalar(1e-4f);
        }
    }

    /// <summary>
    /// Variant Conv inside a constant-count loop whose geometry depends on BOTH the input shape and
    /// the loop index: <c>kernel_shape = [H-2, W-2]</c> (from <c>Shape(x)</c>); <c>dilations =
    /// [i+1, i+1]</c> (from the loop index); <c>pads = i+1</c> per side (<c>= dilation *
    /// (kernelH-1)/2</c>, depending on both). Each iteration is a dilated SAME conv over the input,
    /// reduced to a scalar and summed. The loop unrolls in FastSimplify; afterward each unrolled
    /// SHRK_CONV's geometry is resolved by FastLowerAttributeTensorOps — the index part is constant
    /// post-unroll (strategy 1), the shape part needs the sample input (strategy 2/3). Self-checking
    /// against a hand-unrolled standard-Conv reference (relative tolerance), returns Scalar&lt;bit&gt;.
    /// </summary>
    [Module]
    public partial class ConvVariantLoopShapeAndIndexAttrs
    {
        public static Scalar<bit> Inline(Tensor<float32> x)
        {
            var w = InitSimple.Init([Scalar(3L), Scalar(3L), Scalar(3L), Scalar(3L)]);
            var b = InitSimple.Init([Scalar(3L)]).Vec();

            var shape = x.ShapeTensor();
            var kh = shape[2] - Scalar(2L);              // 3 for H=5 (shape-dependent)
            var kw = shape[3] - Scalar(2L);              // 3 for W=5 (shape-dependent)
            var halfK = (kh - Scalar(1L)) / Scalar(2L);  // 1 (shape-dependent)

            // Reference: hand-unrolled dilated SAME convs with literal geometry (dilation/pad 1,2,3).
            var reference = StdConvScalar(x, w, b, 1L)
                          + StdConvScalar(x, w, b, 2L)
                          + StdConvScalar(x, w, b, 3L);

            // Variant: a loop of 3 whose conv geometry comes from the input shape and the loop index.
            var acc = Scalar(0f);
            foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
            {
                var d = ctx.IterationIndex + Scalar(1L);   // 1,2,3 (index-dependent)
                var p = d * halfK;                         // depends on index AND shape
                var conv = NN.Conv(x, w, b, AutoPad.NotSet,
                    pads: [p, p, p, p], strides: Vector(1L, 1L), dilations: [d, d],
                    kernelShape: [kh, kw], group: Scalar(1L));
                acc = acc + conv.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
            }

            return (reference - acc).Abs() < Scalar(1e-3f) * (reference.Abs() + Scalar(1f));
        }

        private static Scalar<float32> StdConvScalar(Tensor<float32> x, Tensor<float32> w, Vector<float32> b, long d)
        {
            var conv = NN.Conv(x, w, b, AutoPad.NotSet,
                dilations: [d, d], group: 1L, kernelShape: [3L, 3L],
                pads: [d, d, d, d], strides: [1L, 1L]);
            return conv.Abs().Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        }
    }
}
