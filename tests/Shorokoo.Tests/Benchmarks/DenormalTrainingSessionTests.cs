using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Graph;
using Shorokoo.OnnxRuntime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// ONNX Runtime's <c>session.set_denormal_as_zero</c> sets FTZ/DAZ in the constructing thread's
/// MXCSR, under a <c>call_once</c> that makes a process's first session the one that decides. It
/// therefore reaches far past ORT: every later float and double operation on that thread flushes,
/// managed code included, and Shorokoo pins dense uniform draws over subnormal ranges bit-exact.
/// So the factory does not set it, and this class is the guard on that — its own
/// <c>dotnet test</c> invocation exists to give the question a process whose first session is the
/// one it builds.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Benchmark")]
[Collection(SerialMeasurement.Name)]
public class DenormalTrainingSessionTests
{
    /// <summary>
    /// A session built under the rig's profile must not flush the calling thread's denormals.
    /// ONNX Runtime applies <c>session.set_denormal_as_zero</c> to the constructing thread once
    /// per process, under a <c>call_once</c>, so this can only be asked of the process's FIRST
    /// session — which is what this class's own <c>dotnet test</c> invocation provides, and why
    /// the assertion cannot live in the parallel coverage suite.
    ///
    /// <para>Run in a process that has already built one, the flag would never reach this thread
    /// and the assertion would hold for the wrong reason, reporting green while guarding nothing.
    /// So it first requires that no session has been built yet: in a shared process this test
    /// fails rather than passing vacuously, which is the whole point of giving the class an
    /// invocation of its own.</para>
    /// </summary>
    [Fact]
    public void TestTheTrainingProfilesFirstSessionLeavesTheCallingThreadsDenormalsAlone()
    {
        Assert.Equal(0, OrtSessionFactory.SessionsCreated);

        var x = InputTensor<float32>("x", rank: 2);
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            new InternalComputationGraph([x], [OnnxOp.Relu(x)]), prepForOnnx: true);
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, proto);

        using var options = new SessionOptions();
        OrtSessionFactory.Configure(options, ShorokooGraphOptimization.TrainingStep, ShorokooLogSeverity.Fatal);
        using (new InferenceSession(stream.ToArray(), options)) { }

        Assert.Equal(BitConverter.SingleToInt32Bits(1e-40f), BitConverter.SingleToInt32Bits(TimesOne(1e-40f)));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static float TimesOne(float x) => x * 1f;
}
