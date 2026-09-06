using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.OnnxRuntime;
using Shorokoo.Runtime;
using Shorokoo.Tests.Benchmarks;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Tests;

/// <summary>
/// The <c>Rematerializer</c>'s clones are the same op on the same inputs with the same
/// attributes, so ONNX Runtime's CommonSubexpressionElimination (a level-1 transformer, on at
/// <c>ORT_ENABLE_ALL</c>) would merge every clone back into the node it recomputes, and the
/// activation the pass modelled as freed would stay live. The training step's session is built
/// with <see cref="ShorokooGraphOptimization.TrainingStep"/> to stop that; this
/// checks that clones ORT's plain ORT_ENABLE_ALL merges away survive under that level (fusions
/// may still absorb a Conv or Relu clone into its neighbour, which is not a merge).
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RematerializationRuntimeTests
{
    [Fact]
    public void TestRematerializedClonesSurviveOrtGraphOptimization()
    {
        var values = new float[8 * 3 * 64 * 64];
        for (var i = 0; i < values.Length; i++) values[i] = ((i * 37) % 101) * 0.01f - 0.5f;
        NamedModelParam[] sample = [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([8L, 3L, 64L, 64L], values))];
        var rig = TrainingRig.FromScratch(MemoryPassConv.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.01f);

        var original = rig.PreOptimizationGraph.ToInternal().Nodes.Select(n => n.Key).ToHashSet();
        var optimized = rig.OptimizationResult.OptimizedGraph;
        var names = EmittedNames(optimized);
        var clones = optimized.Nodes
            .Where(n => !original.Contains(n.Key) && n.OpCode is CONV or RELU or ADD or MUL or MATMUL && names.ContainsKey(n.Key))
            .ToArray();
        Assert.NotEmpty(clones);

        var kept = OrtOptimizedNodeNames(optimized, ShorokooGraphOptimization.TrainingStep);
        var merged = OrtOptimizedNodeNames(optimized, ShorokooGraphOptimization.EnableAll);
        var survivingKept = clones.Count(c => kept.Contains(names[c.Key]));
        var survivingMerged = clones.Count(c => merged.Contains(names[c.Key]));
        Assert.True(survivingMerged < clones.Length);
        Assert.True(survivingKept > survivingMerged);
        Assert.All(clones.Where(c => c.OpCode is MATMUL), c => Assert.Contains(names[c.Key], kept));
    }

    private static Dictionary<FastNodeKey, string> EmittedNames(InternalComputationGraph graph)
    {
        var clone = graph.Clone();
        FastLowerAttributeTensorOps.Process(clone);
        FastLowerStateUpdateLinksForInference.Process(clone);
        FastLowerRandomOps.Process(clone);
        FastAddIdentityForOuterScopeValues.Process(clone);
        FastPrepForOnnx.Process(clone);
        FastStripCallStacks.Process(clone);
        return FastUseUniqueNames.ProcessAndReturnMap(clone).ToDictionary(kv => kv.Key, kv => kv.Value.ToString());
    }

    private static HashSet<string> OrtOptimizedNodeNames(InternalComputationGraph graph, ShorokooGraphOptimization level)
    {
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true);
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, proto);

        var path = Path.Combine(Path.GetTempPath(), "shorokoo-remat-" + Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            using var options = new SessionOptions();
            OrtSessionFactory.Configure(options, level, ShorokooLogSeverity.Fatal);
            options.OptimizedModelFilePath = path;
            using var session = new InferenceSession(stream.ToArray(), options);
            using var file = File.OpenRead(path);
            var optimized = ProtoBuf.Serializer.Deserialize<ModelProto>(file);
            return optimized.Graph.Nodes.Select(n => n.Name).ToHashSet();
        }
        finally
        {
            File.Delete(path);
        }
    }
}
