using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Jax;
using Shorokoo.Jax.Cpu;
using Shorokoo.Jax.Cuda;
using Shorokoo.PythonHost;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;
using static Shorokoo.Tests.PyTorchBackendCoverageTests;

namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class JaxBackendCoverageTests
{
    private static readonly JaxCpuBackend Jax = new();

    [Fact]
    public void TestEveryJaxElementTypeRoundTripsThroughATensorAndASession()
    {
        foreach (var type in EveryTorchType)
        {
            var bytes = Enumerable.Range(0, 3 * ElementSize(type)).Select(i => (byte)(type == ShorokooTensorElementType.Bool ? i % 2 : i * 7 % 64)).ToArray();
            using var session = Jax.CreateSession(PyTorchBackendCoverageTests.Onnx("Identity", (int)type), default, default, DeviceMemorySettings.Default);
            using var host = Jax.CreateTensorFromRawBytes(type, bytes, [3]);
            using var device = Jax.CreateTensorInBackendMemory(type, bytes, [3]);
            var outputs = session.Run(new Dictionary<string, IShorokooTensorValue> { ["x0"] = host }, ["y"], RunSettings.Default);

            Assert.Equal(type, host.ElementType);
            Assert.Equal([3L], host.Shape);
            Assert.Equal(bytes, Jax.CopyTensorToHost(host));
            Assert.Equal(bytes, Jax.CopyTensorToHost(device));
            Assert.Equal(bytes, Jax.CopyTensorToHost(outputs[0]));
            Assert.Equal(type, outputs[0].ElementType);
            outputs[0].Dispose();
        }
    }

    [Fact]
    public void TestTypedTensorsSpansAndUninitializedTensorsAreTheTensorsTheySayTheyAre()
    {
        using var floats = Jax.CreateTensor([1f, 2f, 3f, 4f], [2, 2]);
        using var longs = Jax.CreateTensor([long.MaxValue, -1L], [2]);
        using var blank = Jax.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Int32, [2, 3]);
        using var empty = Jax.CreateTensor(Array.Empty<float>(), [0, 4]);
        floats.GetTensorMutableDataAsSpan<float>()[3] = 9f;

        Assert.Equal([1f, 2f, 3f, 9f], floats.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal([long.MaxValue, -1L], longs.GetTensorDataAsSpan<long>().ToArray());
        Assert.Equal(6, blank.GetTensorDataAsSpan<int>().Length);
        Assert.Equal([0L, 4L], empty.Shape);
        Assert.True(floats.IsHostAccessible);
        Assert.Throws<NotSupportedException>(() => Jax.CreateTensorFromRawBytes(ShorokooTensorElementType.Int4, [0], [2]));
        Assert.Throws<NotSupportedException>(() => Jax.CreateStringTensor(["a"], [1]));
        Assert.Throws<NotSupportedException>(() => Jax.CreateSequence([]));
        Assert.Throws<ArgumentException>(() => Jax.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, [0, 0], [1]));
    }

    [Fact]
    public void TestAModelGraphRunsOnJaxAndAgreesWithOnnxRuntime()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onOrt = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        using var context = new ComputeContext(Jax);
        var compiled = context.Compile(model);

        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(context.Execute(model, input.Shared())[0]));
        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(compiled.Execute(input.Shared())[0]));
        SideBySideModel.AssertAgree(onOrt, SideBySideModel.Floats(compiled.Execute(input)[0]));
        Assert.True(input.IsDisposed);
        Assert.Equal("Shorokoo.Jax.Cpu", compiled.Backend.Name);
    }

    [Fact]
    public void TestAnAutoGradNodeIsJaxsGradientOfItsLoss()
    {
        float[] w = [0.5f, -1f, 2f], b = [0.1f, 0.2f, -0.3f], x = [1f, 2f, -0.5f], c = [0.3f, -0.2f, 0.1f];
        var step = PyTorchBackendCoverageTests.Graph(["w", "b", "x", "c", "u"], ["w2", "gb", "gp", "gu", "loss"],
            Node("Exp", ["c"], ["p"]),
            Node("Mul", ["w", "x"], ["m"]),
            Node("Add", ["m", "b"], ["s"]),
            Node("Tanh", ["s"], ["t"]),
            Node("Mul", ["t", "p"], ["tp"]),
            Node("ReduceSum", ["tp"], ["loss"]),
            AutoGrad(["loss", "w", "b", "p", "u"], ["gw", "gb", "gp", "gu"]),
            Node("Sub", ["w", "gw"], ["w2"]));
        using var session = Jax.CreateSession(TrainingStep(step), ShorokooGraphOptimization.TrainingStep, default, DeviceMemorySettings.Default);
        var outputs = RunFloats(session, new() { ["w"] = w, ["b"] = b, ["x"] = x, ["c"] = c, ["u"] = [4f] }, ["w2", "gb", "gp", "gu", "loss"]);

        var t = w.Select((wi, i) => MathF.Tanh(wi * x[i] + b[i])).ToArray();
        var gb = t.Select((ti, i) => MathF.Exp(c[i]) * (1 - ti * ti)).ToArray();
        AssertNear(w.Select((wi, i) => wi - gb[i] * x[i]).ToArray(), outputs[0]);
        AssertNear(gb, outputs[1]);
        AssertNear(t, outputs[2]);
        Assert.Equal([0f], outputs[3]);
        AssertNear([t.Select((ti, i) => ti * MathF.Exp(c[i])).Sum()], outputs[4]);
        Assert.True(((IShorokooBackend)Jax).AcceptsTrainingFormat(TrainingFormats.OnnxAutoGrad));
        Assert.True(((IShorokooBackend)Jax).AcceptsTrainingFormat(TrainingFormats.Onnx));
        Assert.False(((IShorokooBackend)Jax).AcceptsTrainingFormat("onnx-autograd/2"));
    }

    private static float[][] RunFloats(IShorokooSession session, Dictionary<string, float[]> feeds, string[] outputs)
    {
        var inputs = feeds.ToDictionary(f => f.Key, f => Jax.CreateTensor(f.Value, [f.Value.Length]));
        try
        {
            var values = session.Run(inputs, outputs, RunSettings.Default);
            var floats = values.Select(v => v.GetTensorDataAsSpan<float>().ToArray()).ToArray();
            foreach (var value in values) value.Dispose();
            return floats;
        }
        finally
        {
            foreach (var input in inputs.Values) input.Dispose();
        }
    }
}
