using Shorokoo.Core.Graph;
using Shorokoo.Graph;
using Shorokoo.Onnx;
using Shorokoo.Runtime;
using Shorokoo.Tests.Utils;
using RetinaNet.Models;
using static Shorokoo.Globals;

namespace Shorokoo.Tests;

/// <summary>
/// Opt-in end-to-end checks that bind a real torchvision/timm ResNet18 checkpoint onto the
/// Shorokoo <see cref="ResNet18"/> graph via <see cref="TorchvisionResNet18NamingScheme"/> and
/// run a forward pass — the <b>prediction</b> half of release-test-plan check E-3 (the load half
/// is <see cref="RealCheckpointTests"/>). Together they make E-3 completable end-to-end.
///
/// <para>Both the checkpoint and the preprocessed sample input are developer-downloaded/generated
/// (git-ignored — see <c>tests/test-data/README.md</c>), so these are tagged
/// <c>Purpose=Manual</c> and skip themselves when the data is absent.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
public partial class RealCheckpointPredictionTests
{
    private const string Resnet18Rel = "models/resnet18/resnet18.safetensors";
    private const string SampleInputRel = "models/resnet18/sample-input-dog.safetensors";

    private const string Hint =
        "Download the checkpoint (huggingface.co/timm/resnet18.a1_in1k) and generate the input " +
        "with tests/test-data/models/resnet18/make-sample-input.py";

    // ResNet18.ComputationGraph declares inputs in this order; the image is last.
    private static InternalComputationGraph BuildConcreteArchitecture(TensorData inputHint)
        => ResNet18.ComputationGraph.ToConcreteArchitecture(ResNet18.ComputationGraph.FromOrderedInputs([
            TensorData(DType.Int64, [], 1000L),   // numClasses
            TensorData(DType.Float32, [], 0.9f),  // bnMomentum (unused at inference)
            TensorData(DType.Float32, [], 1e-5f), // bnEps
            TensorData(DType.Bool, [], true),     // includeTop
            TensorData(DType.Bool, [], false),    // applySoftmax (raw logits)
            inputHint,                            // inputs [1,3,224,224]
        ]));

    /// <summary>
    /// The PyTorch→Shorokoo name map binds every Shorokoo ResNet18 parameter to a distinct tensor
    /// that exists in the real checkpoint, and consumes every weight tensor in the checkpoint
    /// (the 20 Int64 <c>num_batches_tracked</c> counters are inference-irrelevant and unbound).
    /// A complete bijection is the precondition for a faithful forward pass — a single
    /// mis-mapped or dropped weight would silently corrupt the prediction.
    /// </summary>
    [RequiresDownloadedModelFact(Resnet18Rel, Hint)]
    public void NamingScheme_BindsEveryCheckpointWeight_AsCompleteBijection()
    {
        var arch = BuildConcreteArchitecture(TensorDataWithDefaultVals(DType.Float32, [1L, 3L, 224L, 224L]));
        var scheme = TorchvisionResNet18NamingScheme.Create(arch);
        var modelIds = arch.GetConcreteModelParamInfos().ModelIds;

        var checkpointNames = SafeTensorLoader
            .LoadSafeTensors(TestDataPaths.Of("models", "resnet18", "resnet18.safetensors"))
            .Select(t => t.Name).ToHashSet();

        var mapped = modelIds.Select(scheme.ToName).ToList();

        Assert.DoesNotContain(null, mapped);                                  // every param maps
        Assert.Equal(mapped.Count, mapped.Distinct().Count());               // no two params collide
        Assert.All(mapped, name => Assert.Contains(name!, checkpointNames));  // every name is real
        Assert.Equal(102, mapped.Count);                                     // full ResNet18 weight set

        var boundWeights = checkpointNames.Where(n => !n.EndsWith("num_batches_tracked")).ToHashSet();
        Assert.True(boundWeights.SetEquals(mapped!), "every checkpoint weight tensor is bound");
    }

    /// <summary>
    /// Binds the real checkpoint and classifies the canonical PyTorch sample image (a Samoyed):
    /// the top-1 prediction must be ImageNet class 258 (Samoyed). This is the judged top-1
    /// prediction E-3 calls for — real third-party weights, bound by name, producing a correct
    /// classification through Shorokoo's own execution path. (Torch-free: needs only the
    /// downloaded checkpoint and the generated input.)
    /// </summary>
    [RequiresDownloadedModelFact(Hint, Resnet18Rel, SampleInputRel)]
    public void PredictsSamoyed_FromRealCheckpoint()
    {
        const int SamoyedClass = 258;
        var logits = RunForward();
        Assert.Equal(1000, logits.Length);

        var top1 = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]);
        Assert.Equal(SamoyedClass, top1);
    }

    /// <summary>
    /// Full-distribution parity against PyTorch: Shorokoo's logits and softmax probabilities for
    /// every one of the 1000 classes must match the PyTorch reference within a tight numerical
    /// tolerance — not merely agreeing on the argmax. The reference is the baked-in
    /// <see cref="ReferenceLogits"/> (produced by the reference ResNet18 on the same input tensor;
    /// see <c>make-reference-logits.py</c>), so no extra download is needed. Observed agreement is
    /// at the float32 noise floor (|Δlogit| ~1e-5, |Δprob| ~1e-6); the tolerances below leave
    /// headroom for cross-machine kernel variation while still catching any real numerical regression.
    /// </summary>
    [RequiresDownloadedModelFact(Hint, Resnet18Rel, SampleInputRel)]
    public void MatchesPyTorchProbabilities_OnRealCheckpoint()
    {
        var logits = RunForward();
        Assert.Equal(ReferenceLogits.Length, logits.Length);

        var probs = Softmax(logits);
        var referenceProbs = Softmax(ReferenceLogits);

        var maxLogitDiff = Enumerable.Range(0, logits.Length).Max(i => Math.Abs(logits[i] - ReferenceLogits[i]));
        var maxProbDiff = Enumerable.Range(0, probs.Length).Max(i => Math.Abs(probs[i] - referenceProbs[i]));

        Assert.True(maxLogitDiff < 1e-3, $"max |logit diff| = {maxLogitDiff:E4} (expected < 1e-3)");
        Assert.True(maxProbDiff < 1e-4, $"max |prob diff| = {maxProbDiff:E4} (expected < 1e-4)");
    }

    /// <summary>Binds the real checkpoint and runs the forward pass, returning the 1000 raw logits.</summary>
    private static float[] RunForward()
    {
        var input = SafeTensorLoader
            .LoadSafeTensors(TestDataPaths.Of("models", "resnet18", "sample-input-dog.safetensors"))
            .Single(t => t.Name == "input").Data;
        Assert.Equal([1L, 3L, 224L, 224L], input.Shape.Dims);

        var arch = BuildConcreteArchitecture(input);
        var weights = SafeTensorLoader.LoadModelParamSet(TestDataPaths.Of("models", "resnet18", "resnet18.safetensors"));
        var concrete = arch.ToConcreteModel(weights, TorchvisionResNet18NamingScheme.Create(arch));

        var outputs = new ComputeContext().Execute(concrete,
            TensorData(DType.Int64, [], 1000L),
            TensorData(DType.Float32, [], 0.9f),
            TensorData(DType.Float32, [], 1e-5f),
            TensorData(DType.Bool, [], true),
            TensorData(DType.Bool, [], false),
            input);

        return outputs[0].ToTensorData<float32>().AccessMemory().ToArray();
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exp = logits.Select(v => Math.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(e => (float)(e / sum)).ToArray();
    }
}
