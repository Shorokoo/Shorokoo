using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Graph;

namespace Shorokoo.Tests;

/// <summary>
/// PyTorch/torchvision → Shorokoo parameter name mapping for the
/// <see cref="RetinaNet.Models.ResNet18"/> sample model, used to bind a real
/// torchvision/timm ResNet18 <c>.safetensors</c> checkpoint onto the Shorokoo graph
/// (release-test-plan check E-3, the prediction half).
///
/// <para>The Shorokoo side uses <see cref="RetinaNet.Models.FrozenBatchNorm"/> (eval-mode
/// batch norm: <c>y = x*scale + bias</c>), which carries the same four learnable/buffer
/// tensors as torchvision's <c>BatchNorm2d</c> — <c>running_mean</c>, <c>running_var</c>,
/// <c>weight</c>, <c>bias</c> — in this initializer order:
/// <c>ResNetInitZeros#0</c>=running_mean, <c>ResNetInitSimple#0</c>=running_var,
/// <c>ResNetInitSimple#1</c>=weight, <c>ResNetInitZeros#1</c>=bias. Convolutions use
/// bias-free kernels (matching torchvision's <c>bias=False</c> convs), so only the
/// <c>weight</c> binds. The always-computed block shortcut convolution is constant-folded
/// away when <c>downsample=false</c>, so only the first block of layers 2-4 contributes
/// <c>downsample.0/1.*</c> tensors — exactly as in torchvision.</para>
///
/// <para>This is a complete bijection: all 102 Shorokoo trainable parameters map to the
/// 102 weight tensors of a torchvision ResNet18 state dict (the 20 Int64
/// <c>num_batches_tracked</c> counters are inference-irrelevant and intentionally unbound).</para>
/// </summary>
public static class TorchvisionResNet18NamingScheme
{
    // Maps from the Shorokoo BatchNorm-instance / conv-instance index to the torchvision sub-name.
    private static readonly Dictionary<string, string> BnBasic = new() { ["0"] = "bn1", ["1"] = "bn2" };
    private static readonly Dictionary<string, string> BnWithDownsample =
        new() { ["0"] = "bn1", ["1"] = "bn2", ["2"] = "downsample.1" };
    private static readonly Dictionary<string, string> ConvBasic = new() { ["0"] = "conv1", ["1"] = "conv2" };

    /// <summary>
    /// Builds the naming scheme for a concrete ResNet18 architecture graph (the output of
    /// <see cref="InternalComputationGraphExtensions.ToConcreteArchitecture"/>). Bind weights with
    /// <c>arch.ToConcreteModel(weights, scheme)</c>.
    /// </summary>
    public static SimplePatternNamingScheme Create(InternalComputationGraph concreteArch)
    {
        var shorokooIdScheme =
            ModuleParamSetNamingScheme.CreateShorokooNamingScheme(concreteArch.GetConcreteModelParamInfos());
        return new SimplePatternNamingScheme(
            BuildPatterns(), shorokooIdScheme, ModuleParamSetNamingScheme.PyTorchFrameworkId);
    }

    private static List<SimplePatternScheme> BuildPatterns()
    {
        var patterns = new List<SimplePatternScheme>();

        // Emit the four FrozenBatchNorm tensors for a given context. The init class+index
        // literal selects the tensor; {bnName} is the per-context BatchNorm-instance map.
        void AddBn(string ctxPattern, string fmtPrefix, string bnCapture, Dictionary<string, string> bnMap)
        {
            var maps = new Dictionary<string, Dictionary<string, string>> { ["bn"] = bnMap };
            (string init, string suffix)[] tensors =
            [
                ("ResNetInitZeros#0", "running_mean"),
                ("ResNetInitSimple#0", "running_var"),
                ("ResNetInitSimple#1", "weight"),
                ("ResNetInitZeros#1", "bias"),
            ];
            foreach (var (init, suffix) in tensors)
                patterns.Add(new SimplePatternScheme(
                    $"{ctxPattern}.FrozenBatchNorm#{{{bnCapture}}}.{init}",
                    $"{fmtPrefix}.{{{bnCapture}|bn}}.{suffix}", maps));
        }

        // The stem batch norm has a single instance (FrozenBatchNorm#0 → torchvision bn1).
        void AddStemBn()
        {
            (string init, string suffix)[] tensors =
            [
                ("ResNetInitZeros#0", "running_mean"),
                ("ResNetInitSimple#0", "running_var"),
                ("ResNetInitSimple#1", "weight"),
                ("ResNetInitZeros#1", "bias"),
            ];
            foreach (var (init, suffix) in tensors)
                patterns.Add(new SimplePatternScheme(
                    $"TrainableParam#0.ResNetStem#0.FrozenBatchNorm#0.{init}", $"bn1.{suffix}"));
        }

        // ── Stem ───────────────────────────────────────────────────────────────
        patterns.Add(new SimplePatternScheme(
            "TrainableParam#0.ResNetStem#0.Conv2Dk77s22#0.ResNetInitWeight#0", "conv1.weight"));
        AddStemBn();

        // ── Layer 1: BasicStackS11#0, two BasicBlockS11 in a loop (no downsample) ──
        const string l1Ctx = "TrainableParam#0.BasicStackS11#0.Loop#0:{idx}.BasicBlockS11#0";
        patterns.Add(new SimplePatternScheme(
            $"{l1Ctx}.Conv2Dk33s11#{{c}}.ResNetInitWeight#0",
            "layer1.{idx}.{c|conv}.weight",
            new() { ["conv"] = ConvBasic }));
        AddBn(l1Ctx, "layer1.{idx}", "b", BnBasic);

        // ── Layers 2-4: BasicStackS22#{L} → layer{L+2} ───────────────────────────
        // First block: BasicBlockS22#0 (stride-2, with downsample shortcut).
        const string l22b0 = "TrainableParam#0.BasicStackS22#{L}.BasicBlockS22#0";
        patterns.Add(new SimplePatternScheme(
            $"{l22b0}.Conv2Dk33s22#0.ResNetInitWeight#0", "layer{L + 2}.0.conv1.weight"));
        patterns.Add(new SimplePatternScheme(
            $"{l22b0}.Conv2Dk33s11#0.ResNetInitWeight#0", "layer{L + 2}.0.conv2.weight"));
        patterns.Add(new SimplePatternScheme(
            $"{l22b0}.Conv2Dk11s22#0.ResNetInitWeight#0", "layer{L + 2}.0.downsample.0.weight"));
        AddBn(l22b0, "layer{L + 2}.0", "b", BnWithDownsample);

        // Remaining block(s): BasicBlockS11 in a loop; loop idx 0 → torchvision block 1.
        const string l22loop = "TrainableParam#0.BasicStackS22#{L}.Loop#0:{idx}.BasicBlockS11#0";
        patterns.Add(new SimplePatternScheme(
            $"{l22loop}.Conv2Dk33s11#{{c}}.ResNetInitWeight#0",
            "layer{L + 2}.{idx + 1}.{c|conv}.weight",
            new() { ["conv"] = ConvBasic }));
        AddBn(l22loop, "layer{L + 2}.{idx + 1}", "b", BnBasic);

        // ── Classification head ──────────────────────────────────────────────────
        patterns.Add(new SimplePatternScheme(
            "TrainableParam#0.ClassificationHead#0.DenseBasic#0.ResNetInitWeight#0", "fc.weight"));
        patterns.Add(new SimplePatternScheme(
            "TrainableParam#0.ClassificationHead#0.DenseBasic#0.ResNetInitZeros#0", "fc.bias"));

        return patterns;
    }
}
