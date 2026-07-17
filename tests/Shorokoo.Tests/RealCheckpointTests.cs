using Shorokoo.Onnx;
using Shorokoo.Tests.Utils;

namespace Shorokoo.Tests;

/// <summary>
/// Opt-in checks against a real, third-party pretrained checkpoint that a developer
/// downloads manually (kept out of the repo — see <c>tests/test-data/README.md</c>).
/// These replace the former committed golden-weight parity fixtures.
///
/// Tagged <c>Purpose=Manual</c> so the automated Coverage suite skips them; each test
/// also skips at runtime (via <see cref="RequiresDownloadedModelFactAttribute"/>) when
/// its checkpoint has not been downloaded, so a clean checkout stays green.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
public class RealCheckpointTests
{
    private const string Resnet18Rel = "models/resnet18/resnet18.safetensors";

    private const string Resnet18Hint =
        "curl -L https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors " +
        "-o tests/test-data/models/resnet18/resnet18.safetensors";

    /// <summary>
    /// A real torchvision/timm ResNet18 <c>.safetensors</c> loads through Shorokoo's
    /// <see cref="SafeTensorLoader"/> with its tensor names, shapes and dtypes intact —
    /// the load half of release-test-plan check E-3. (Bit-exact PyTorch parity remains
    /// the manual E-6 exercise; binding third-party parameter names onto a built graph
    /// needs a per-architecture naming scheme, out of scope for this smoke test.)
    /// </summary>
    [RequiresDownloadedModelFact(Resnet18Rel, Resnet18Hint)]
    public void Resnet18Checkpoint_LoadsViaSafeTensorMachinery()
    {
        var path = TestDataPaths.Of("models", "resnet18", "resnet18.safetensors");

        // A truncated or malformed file throws inside the loader.
        var tensors = SafeTensorLoader.LoadSafeTensors(path);
        var byName = tensors.ToDictionary(t => t.Name);

        // A standard ResNet18 state dict carries ~120 tensors.
        Assert.True(tensors.Count >= 100,
            $"expected a full ResNet18 state dict, got {tensors.Count} tensors");

        // Landmark tensors present in every torchvision/timm ResNet18, with canonical
        // shapes — proves names and shapes survived the parse intact.
        AssertTensor(byName, "conv1.weight",          "F32", 64, 3, 7, 7);
        AssertTensor(byName, "bn1.running_mean",      "F32", 64);
        AssertTensor(byName, "layer1.0.conv1.weight", "F32", 64, 64, 3, 3);
        AssertTensor(byName, "layer4.1.conv2.weight", "F32", 512, 512, 3, 3);
        AssertTensor(byName, "fc.weight",             "F32", 1000, 512);
        AssertTensor(byName, "fc.bias",               "F32", 1000);

        // The batchnorm step counter is a rank-0 (empty-shape) I64 scalar when present
        // — exercises the loader's scalar path on a real file.
        var counter = tensors.FirstOrDefault(t => t.Name.EndsWith("num_batches_tracked"));
        if (counter is not null)
        {
            Assert.Equal("I64", counter.DataType);
            Assert.Empty(counter.Shape);
        }

        // The decoded conv1 kernel is real trained data: exact byte count, all finite,
        // not uniformly zero.
        var conv1Bytes = byName["conv1.weight"].Data.AccessRawMemory().ToArray();
        Assert.Equal(64L * 3 * 7 * 7 * sizeof(float), conv1Bytes.Length);
        var conv1 = new float[conv1Bytes.Length / sizeof(float)];
        Buffer.BlockCopy(conv1Bytes, 0, conv1, 0, conv1Bytes.Length);
        Assert.All(conv1, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(conv1, v => v != 0f);

        // Whole-file integrity: every tensor's decoded byte payload equals
        // product(shape) * dtype-size — the full state dict round-trips through the
        // parser, not just the landmark tensors. (Empty shape ⇒ 1 element, rank-0.)
        foreach (var t in tensors)
        {
            var dtypeSize = t.DataType switch { "F32" => 4, "I64" => 8, _ => 0 };
            if (dtypeSize == 0) continue; // this checkpoint only uses F32/I64
            var elements = t.Shape.Aggregate(1L, (acc, d) => acc * d);
            Assert.Equal(elements * dtypeSize, t.Data.AccessRawMemory().Length);
        }

        // The higher-level ModelParamList entry point reads the same file.
        var paramSet = SafeTensorLoader.LoadModelParamSet(path);
        Assert.NotNull(paramSet.Find("conv1.weight"));
        Assert.NotNull(paramSet.Find("fc.weight"));
    }

    private static void AssertTensor(
        IReadOnlyDictionary<string, SafeTensor> byName, string name, string dtype, params long[] shape)
    {
        Assert.True(byName.TryGetValue(name, out var t), $"missing tensor '{name}'");
        Assert.Equal(dtype, t!.DataType);
        Assert.Equal(shape, t.Shape);
    }
}
