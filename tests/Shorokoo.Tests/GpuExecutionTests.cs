using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// Smoke tests that actually execute a small Shorokoo computation graph on the
/// local machine's GPU (NOT skipped).
///
/// These tests assume the host has an ONNX Runtime–compatible GPU available.
/// On a machine without one they will fail with a clear EP-loading error.
/// They are deliberately excluded from the coverage suite (no
/// Purpose=Coverage); run them on a CUDA machine with --filter "Purpose=Hardware".
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
public class GpuExecutionTests
{
    /// <summary>
    /// Explicitly exercises the CUDA execution provider.
    /// </summary>
    [CudaFact]
    public void CudaProvider_AddTwoFloat32Scalars_ReturnsCorrectSum()
    {
        // [CudaFact] already gates on a GPU backend being loaded; confirm the
        // bound factory really is a CUDA one rather than a silent CPU fallback.
        var backend = InferenceBackend.Factory.GetType().Assembly.GetName().Name ?? "";
        Assert.EndsWith("GPU", backend);

        var result = AddTwoScalars(2.0f, 3.0f);
        Assert.Equal(5.0f, result);
    }

    private static float AddTwoScalars(float left, float right)
    {
        var a = InputScalar<float32>();
        var b = InputScalar<float32>();
        var c = a + b;

        var graph = new InternalComputationGraph([a, b], [c]);
        var ctx = new ComputeContext();
        var results = ctx.Execute(
            InternalComputationGraphConverter.ToFastGraph(graph),
            TensorData([], left),
            TensorData([], right));

        return results[0].ToTensorData().As<float32>().AccessMemory<float>()[0];
    }
}
