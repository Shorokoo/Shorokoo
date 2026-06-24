using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for matrix multiplication and related linear algebra operations.
/// These are typically the most compute-intensive operations in neural networks.
/// MatMul cost is O(M*N*K) for (M,K)×(K,N) matrices.
/// Conv cost is O(batch * out_channels * spatial_output * kernel_volume * in_channels / groups).
/// </summary>
internal class LinearAlgebraPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>
    {
        MATMUL, GEMM, CONV, CONV_TRANSPOSE, EINSUM
    };

    public OpPerfResult Estimate(OpPerfInput input)
    {
        return input.OpCode switch
        {
            MATMUL => EstimateMatMul(input),
            GEMM => EstimateGemm(input),
            CONV => EstimateConv(input),
            CONV_TRANSPOSE => EstimateConvTranspose(input),
            EINSUM => EstimateEinsum(input),
            _ => OpPerfResult.Zero
        };
    }

    private static OpPerfResult EstimateMatMul(OpPerfInput input)
    {
        var aShape = input.InputShapes[0];
        var bShape = input.InputShapes[1];
        if (aShape is null || bShape is null)
            return OpPerfResult.Zero;

        var aDims = aShape.Shape.Dims;
        var bDims = bShape.Shape.Dims;

        // For MatMul (A×B where A is ...×M×K and B is ...×K×N):
        // FLOPs ≈ batch * M * N * (2K - 1) ≈ batch * M * N * 2K
        long m = aDims.Length >= 2 ? aDims[^2] : 1;
        long k = aDims.Length >= 1 ? aDims[^1] : 1;
        long n = bDims.Length >= 1 ? bDims[^1] : 1;

        // Batch dimensions
        long batch = 1;
        for (int i = 0; i < aDims.Length - 2; i++)
            batch *= aDims[i];

        // 2 * M * N * K multiplications+additions, normalized by 256
        var flops = batch * m * n * 2.0 * k;
        var computeTime = flops / 256.0;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
        };
    }

    private static OpPerfResult EstimateGemm(OpPerfInput input)
    {
        // GEMM: alpha * A × B + beta * C
        // Same as MatMul but with additional scaling + bias add
        var aShape = input.InputShapes[0];
        var bShape = input.InputShapes[1];
        if (aShape is null || bShape is null)
            return OpPerfResult.Zero;

        var aDims = aShape.Shape.Dims;
        var bDims = bShape.Shape.Dims;

        long m = aDims.Length >= 2 ? aDims[0] : 1;
        long k = aDims.Length >= 2 ? aDims[1] : aDims[0];
        long n = bDims.Length >= 2 ? bDims[1] : bDims[0];

        // Check transA/transB attributes
        var transA = GetLongAttr(input, "transA", 0);
        var transB = GetLongAttr(input, "transB", 0);
        if (transA != 0) (m, k) = (k, m);
        if (transB != 0) (k, n) = (n, k);

        var flops = m * n * 2.0 * k + m * n * 2.0; // matmul + scale + bias
        var computeTime = flops / 256.0;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
        };
    }

    private static OpPerfResult EstimateConv(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0]; // N,C,H,W or N,C,D,H,W
        var weightShape = input.InputShapes[1]; // O,C/g,kH,kW or O,C/g,kD,kH,kW
        var outputShape = input.OutputShapes[0];
        if (inputShape is null || weightShape is null || outputShape is null)
            return OpPerfResult.Zero;

        var wDims = weightShape.Shape.Dims;
        var outDims = outputShape.Shape.Dims;

        // Need at least 3D tensors for Conv
        if (wDims.Length < 3 || outDims.Length < 3)
            return OpPerfResult.Zero;

        long group = GetLongAttr(input, "group", 1);

        // Conv FLOPs: batch * outChannels * spatialOutputSize * kernelVolume * (inChannels/groups) * 2
        long batch = outDims[0];
        long outChannels = outDims[1];
        long spatialOutput = 1;
        for (int i = 2; i < outDims.Length; i++)
            spatialOutput *= outDims[i];

        long kernelVolume = 1;
        for (int i = 2; i < wDims.Length; i++)
            kernelVolume *= wDims[i];

        long inChannelsPerGroup = wDims[1];

        var flops = batch * outChannels * spatialOutput * kernelVolume * inChannelsPerGroup * 2.0;
        var computeTime = flops / 256.0;

        // Conv may need im2col workspace
        long workspaceBytes = batch * inChannelsPerGroup * group * kernelVolume * spatialOutput
            * (inputShape.DType.EncodingBitCount / 8);

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = workspaceBytes,
        };
    }

    private static OpPerfResult EstimateConvTranspose(OpPerfInput input)
    {
        // ConvTranspose is roughly similar cost to Conv
        // Use output shape to estimate FLOPs
        var outputShape = input.OutputShapes[0];
        var weightShape = input.InputShapes[1];
        if (outputShape is null || weightShape is null)
            return OpPerfResult.Zero;

        var wDims = weightShape.Shape.Dims;
        var outDims = outputShape.Shape.Dims;

        if (wDims.Length < 3 || outDims.Length < 3)
            return OpPerfResult.Zero;

        long group = GetLongAttr(input, "group", 1);

        long batch = outDims[0];
        long outChannels = outDims[1];
        long spatialOutput = 1;
        for (int i = 2; i < outDims.Length; i++)
            spatialOutput *= outDims[i];

        long kernelVolume = 1;
        for (int i = 2; i < wDims.Length; i++)
            kernelVolume *= wDims[i];

        long inChannelsPerGroup = group > 0 ? wDims[0] / group : wDims[0];

        var flops = batch * outChannels * spatialOutput * kernelVolume * inChannelsPerGroup * 2.0;
        var computeTime = flops / 256.0;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
        };
    }

    private static OpPerfResult EstimateEinsum(OpPerfInput input)
    {
        // Einsum is general — estimate based on output size × average input size
        var outputShape = input.OutputShapes[0];
        if (outputShape is null)
            return OpPerfResult.Zero;

        long totalInputElements = 0;
        int inputCount = 0;
        foreach (var s in input.InputShapes)
        {
            if (s is not null)
            {
                totalInputElements += s.ElementCount;
                inputCount++;
            }
        }

        var avgInputSize = inputCount > 0 ? totalInputElements / inputCount : outputShape.ElementCount;
        var flops = outputShape.ElementCount * avgInputSize * 2.0;
        var computeTime = flops / 256.0;

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = 0,
        };
    }

    private static long GetLongAttr(OpPerfInput input, string name, long defaultValue)
    {
        if (input.Attributes.TryGetValue(name, out var val) && val is long l)
            return l;
        return defaultValue;
    }
}
