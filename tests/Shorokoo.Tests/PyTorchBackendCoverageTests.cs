using Shorokoo.Core.Backends;
using Shorokoo.PyTorch;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;

namespace Shorokoo.Tests;

[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class PyTorchBackendCoverageTests
{
    private static readonly TorchCpuBackend Torch = new();

    [Fact]
    public void TestAModelGraphRunsOnTorchAndAgreesWithOnnxRuntime()
    {
        var (model, input) = SideBySideModel.Concrete();
        var onOrt = SideBySideModel.Floats(new ComputeContext().Execute(model, input.Shared())[0]);
        var onTorch = SideBySideModel.Floats(new ComputeContext(Torch).Execute(model, input.Shared())[0]);
        SideBySideModel.AssertAgree(onOrt, onTorch);
    }
}
