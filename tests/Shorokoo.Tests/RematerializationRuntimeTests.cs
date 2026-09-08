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
///
/// <para>Also the training step's runtime numerics, where a lowered step has to be built and run
/// to ask the question at all — which is why the feed guard below lives here rather than beside
/// the harnesses it guards.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class RematerializationRuntimeTests
{
    [Fact]
    public void TestRematerializedClonesSurviveOrtGraphOptimization()
    {
        var values = new float[1024 * 32];
        for (var i = 0; i < values.Length; i++) values[i] = ((i * 37) % 101) * 0.01f - 0.5f;
        NamedModelParam[] sample = [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData([1024L, 32L], values))];
        var rig = TrainingRig.FromScratch(Modules.CheckpointedNarrowMlpStack.ComputationGraph, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.01f);

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
        Assert.True(survivingKept * 2 >= clones.Length);
    }

    /// <summary>
    /// The training step's own sessions get the profile that keeps its recomputation; a graph
    /// compiled through the ordinary entry point does not.
    /// </summary>
    [Fact]
    public void TestOnlyTheRigsOwnSessionsGetTheTrainingStepProfile()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var graph = new InternalComputationGraph([x], [OnnxOp.Relu(x)]);

        Assert.Equal(ShorokooGraphOptimization.TrainingStep,
            ComputeContext.Default.Compile(graph, inputDims: null, trainingStep: true).Optimization);
        Assert.Equal(ShorokooGraphOptimization.EnableAll,
            ComputeContext.Default.Compile(graph, inputDims: null, trainingStep: false).Optimization);
    }

    /// <summary>
    /// <see cref="SyntheticFeed"/> must keep a training step's attention probabilities normal:
    /// what the profiling harnesses feed is what their kernel times measure.
    /// </summary>
    [Fact]
    public void TestTheProfilingHarnessFeedKeepsAttentionOutOfTheDenormalRange()
    {
        long[] shape = [2L, 32L, 128L];
        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(shape, new float[shape[0] * shape[1] * shape[2]]))];
        var rig = TrainingRig.FromScratch(MemoryPassEncoder1.ComputationGraph, L2Loss.ComputationGraph,
            SGDOptimizer.ComputationGraph, sample, 0.01f);
        var inputs = rig.OptimizationInputShapes;
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(rig.TrainingStepPureGraph.ToInternal(), prepForOnnx: true,
            inputDims: inputs.Select(s => s.Shape.Dims).ToArray());

        var probability = proto.Graph.Nodes.First(n => n.OpType == "Softmax").Outputs[0];
        proto.Graph.Outputs.Add(new ValueInfoProto { Name = probability });
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, proto);

        using var options = new SessionOptions();
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
        using var session = new InferenceSession(stream.ToArray(), options);
        var feeds = new Dictionary<string, OrtValue>();
        for (var i = 0; i < session.InputNames.Count; i++)
            feeds[session.InputNames[i]] = SyntheticFeed.Tensor(inputs[i].Shape, inputs[i].DType, i);
        using var runOptions = new RunOptions();
        using var results = session.Run(runOptions, feeds, [probability]);
        var probabilities = results[0].GetTensorDataAsSpan<float>();

        int denormals = 0, zeros = 0;
        foreach (var p in probabilities)
            if (p == 0f) zeros++;
            else if (MathF.Abs(p) < 1.17549435e-38f) denormals++;
        GC.KeepAlive(feeds);

        Assert.Equal(0, denormals);
        Assert.Equal(0, zeros);
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
