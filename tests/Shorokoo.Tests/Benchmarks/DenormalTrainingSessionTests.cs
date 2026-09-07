using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json.Linq;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Graph;
using Shorokoo.OnnxRuntime;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// Attention gradients hold denormal floats, and ORT's MatMul on denormal operands runs about
/// ten times slower: in encoder1's training step the batched attention products (inputs
/// 8×4×…) take 6.6 + 3×1.0 ms with denormals live and 0.7 + 3×0.2 ms with
/// <c>session.set_denormal_as_zero</c> (ORT profiler, fresh process each, idle machine).
/// This compares a session under the factory's <see cref="ShorokooGraphOptimization.TrainingStep"/>
/// configuration against one that flushes explicitly. Skipped: ORT's <c>set_denormal_as_zero</c>
/// sets FTZ/DAZ on the constructing thread once per process and leaks it into managed code, so the
/// factory cannot enable it (Shorokoo/Shorokoo#252, open) — the fix has to keep the gradients out of
/// the denormal range or scope the flag to the run.
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

    [Fact(Skip = "Shorokoo/Shorokoo#252: attention gradients are denormal and MLAS GEMM runs ~7× slower; set_denormal_as_zero leaks FTZ/DAZ into the calling thread")]
    public void TestAttentionMatMulsDoNotPayForDenormalsInTheTrainingSession()
    {
        long[] shape = [8L, 128L, 128L];
        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(shape, SyntheticFeed.Floats(shape[0] * shape[1] * shape[2], 0)))];
        var rig = TrainingRig.FromScratch(MemoryPassEncoder1.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.01f);
        var inputs = rig.OptimizationInputShapes;
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(rig.TrainingStepPureGraph.ToInternal(), prepForOnnx: true,
            inputDims: inputs.Select(s => s.Shape.Dims).ToArray());
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, proto);
        var model = stream.ToArray();

        var configured = BatchedMatMulKernelMs(model, inputs, explicitFlush: false);
        var flushed = BatchedMatMulKernelMs(model, inputs, explicitFlush: true);

        Assert.True(configured <= 2.0 * flushed);
    }

    /// <summary>Median over runs of the summed kernel time (ms) of MatMuls whose first input is batched 8×4×….</summary>
    private static double BatchedMatMulKernelMs(byte[] model, (Shape Shape, DType DType)[] inputs, bool explicitFlush)
    {
        var dir = Path.Combine(Path.GetTempPath(), "shorokoo-denormal-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using var options = new SessionOptions();
            if (explicitFlush)
            {
                options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            }
            else
                OrtSessionFactory.Configure(options, ShorokooGraphOptimization.TrainingStep, ShorokooLogSeverity.Fatal);
            options.EnableProfiling = true;
            options.ProfileOutputPathPrefix = Path.Combine(dir, "profile");
            using var session = new InferenceSession(model, options);
            var feeds = new Dictionary<string, OrtValue>();
            for (var i = 0; i < session.InputNames.Count; i++)
                feeds[session.InputNames[i]] = SyntheticFeed.Tensor(inputs[i].Shape, inputs[i].DType, i);
            using var runOptions = new RunOptions();
            for (var run = 0; run < 6; run++)
                foreach (var o in session.Run(runOptions, feeds, session.OutputNames)) o.Dispose();
            GC.KeepAlive(feeds);

            var events = JArray.Parse(File.ReadAllText(session.EndProfiling()));
            var runs = events.Where(e => (string?)e["cat"] == "Session" && (string?)e["name"] == "model_run").Skip(1)
                .Select(r => ((long)r["ts"]!, (long)r["ts"]! + (long)r["dur"]!)).ToList();
            var kernels = events.Where(e => (string?)e["cat"] == "Node" && ((string?)e["name"] ?? "").EndsWith("_kernel_time")
                && (string?)e["args"]?["op_name"] is "MatMul" or "FusedMatMul"
                && (e["args"]?["input_type_shape"]?.ToString(Newtonsoft.Json.Formatting.None) ?? "").Contains("[8,4,")).ToList();
            var perRun = runs.Select(r => kernels.Where(k => (long)k["ts"]! >= r.Item1 && (long)k["ts"]! <= r.Item2).Sum(k => (long)k["dur"]!) / 1000.0)
                .OrderBy(x => x).ToArray();
            return perRun[perRun.Length / 2];
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

}
