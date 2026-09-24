using System.Runtime.InteropServices;
using Shorokoo.Core.Backends;
using Shorokoo.PyTorch.Cuda;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

/// <summary>
/// The PyTorch CUDA backend on a card. It runs in the CUDA 13 environment, which the first test
/// provisions (some 6 GB) unless <c>SHOROKOO_PYTHON_ENV</c> names one; and a process has one
/// Python environment, so run this class in a process where no CPU torch backend started first --
/// the Purpose=Hardware filter does.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Hardware")]
public class PyTorchCudaHardwareTests
{
    private static readonly Lazy<TorchCudaBackend> Cuda = new(() => new TorchCudaBackend());

    [TorchCudaFact]
    public void TestATensorLivesOnTheCardAndComesHomeByCopy()
    {
        byte[] bytes = [.. MemoryMarshal.AsBytes<float>([1f, 2f, 3f])];
        using var onCard = Cuda.Value.CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bytes, [3]);
        using var onHost = Cuda.Value.CreateTensorFromRawBytes(ShorokooTensorElementType.Float, bytes, [3]);
        using var blank = Cuda.Value.CreateUninitializedTensorInBackendMemory(ShorokooTensorElementType.Int64, [4]);

        Assert.False(onCard.IsHostAccessible);
        Assert.True(onHost.IsHostAccessible);
        Assert.Equal(bytes, Cuda.Value.CopyTensorToHost(onCard));
        Assert.Throws<InvalidOperationException>(() => onCard.GetTensorDataAsSpan<float>().Length);
        Assert.False(blank.IsHostAccessible);
    }

    [TorchCudaFact]
    public void TestAModelRunsOnTheCardAndAgreesWithTheHost()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onHost = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        using var context = new ComputeContext(Cuda.Value);

        SideBySideModel.AssertAgree(onHost, SideBySideModel.Floats(context.Execute(model, input.Shared())[0]), SideBySideModel.DeviceTolerance);
        SideBySideModel.AssertAgree(onHost, SideBySideModel.Floats(context.Compile(model).Execute(input)[0]), SideBySideModel.DeviceTolerance);
    }

    [TorchCudaFact]
    public void TestARetainedOutputStaysOnTheCardAndTheRestComeHome()
    {
        using var session = Cuda.Value.CreateSession(NegModel(), default, default, DeviceMemorySettings.Default);
        using var x = Cuda.Value.CreateTensor([1f, -2f], [2]);
        var inputs = new Dictionary<string, IShorokooTensorValue> { ["x"] = x };
        using var kept = session.RunRetainingOutputs(inputs, ["y"], new HashSet<string> { "y" }, RunSettings.Default)[0];
        using var fetched = session.Run(inputs, ["y"], RunSettings.Default)[0];

        Assert.True(session.HasDeviceMemory);
        Assert.Equal(SessionOutputPlacement.Device, session.OutputPlacement);
        Assert.False(kept.IsHostAccessible);
        Assert.Equal([-1f, 2f], fetched.GetTensorDataAsSpan<float>().ToArray());
        Assert.Equal(Cuda.Value.CopyTensorToHost(fetched), Cuda.Value.CopyTensorToHost(kept));
    }

    private static byte[] NegModel()
    {
        var node = new Shorokoo.Core.Factory.IR.NodeProto { OpType = "Neg", Name = "neg" };
        node.Inputs.Add("x");
        node.Outputs.Add("y");
        var graph = new Shorokoo.Core.Factory.IR.GraphProto { Name = "g" };
        graph.Inputs.Add(new Shorokoo.Core.Factory.IR.ValueInfoProto { Name = "x" });
        graph.Outputs.Add(new Shorokoo.Core.Factory.IR.ValueInfoProto { Name = "y" });
        graph.Nodes.Add(node);
        var model = new Shorokoo.Core.Factory.IR.ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new Shorokoo.Core.Factory.IR.OperatorSetIdProto { Domain = "", Version = 21 });
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }
}

/// <summary>Runs a test only where an NVIDIA driver answers: the torch CUDA backend brings its own
/// CUDA libraries, so the driver is all the machine has to supply.</summary>
public sealed class TorchCudaFactAttribute : FactAttribute
{
    public TorchCudaFactAttribute()
    {
        if (!NativeLibrary.TryLoad(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1", out var driver))
            Skip = "No NVIDIA driver on this machine, so there is no card for the PyTorch CUDA backend to run on.";
        else
            NativeLibrary.Free(driver);
    }
}
