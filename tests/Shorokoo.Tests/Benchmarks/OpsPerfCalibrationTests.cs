using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Shorokoo.Core.AutoDiffCheckpointing;
using Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Graph;
using Shorokoo.Modules.Losses;
using Shorokoo.Modules.Optimizers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// Calibration harness for the <c>OpsPerf</c> compute model: for every memory-pass model family
/// it lowers the training step, records each node's shapes, attributes and the estimators'
/// <c>ComputeTime</c>, and profiles the same ONNX model in ORT (kernel time per node name,
/// median over runs) at both <c>ORT_ENABLE_ALL</c> — what the rig runs — and
/// <c>ORT_DISABLE_ALL</c>, where every node pairs with exactly one kernel. One JSON per family is
/// written to <c>$SHOROKOO_OPSPERF_CALIBRATION_DIR</c> (default: a folder under the temp path);
/// <c>tools/opsperf-calibration/fit.py</c> pairs, fits and reports on those files. Manual: it
/// measures the machine, so it is never part of the coverage suite.
///
/// <para>Run with <c>dotnet test --filter "FullyQualifiedName~OpsPerfCalibrationTests"</c>.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
[Collection(SerialMeasurement.Name)]
public class OpsPerfCalibrationTests
{
    private const int MeasuredRuns = 7;
    private const int Sessions = 2;

    private static readonly (string Family, Func<ComputationGraph> Model, long[] Shape)[] Suite =
    [
        ("mlp",          () => MemoryPassMlp.ComputationGraph,            [64L, 256L]),
        ("mlp-b8",       () => MemoryPassMlp.ComputationGraph,            [8L, 256L]),
        ("mlp-b512",     () => MemoryPassMlp.ComputationGraph,            [512L, 256L]),
        ("conv",         () => MemoryPassConv.ComputationGraph,           [8L, 3L, 64L, 64L]),
        ("conv-b2s32",   () => MemoryPassConv.ComputationGraph,           [2L, 3L, 32L, 32L]),
        ("encoder1",     () => MemoryPassEncoder1.ComputationGraph,       [8L, 128L, 128L]),
        ("encoder1-b2",  () => MemoryPassEncoder1.ComputationGraph,       [2L, 32L, 128L]),
        ("encoder2",     () => MemoryPassEncoder2.ComputationGraph,       [8L, 128L, 128L]),
        ("lstm",         () => MemoryPassLstm.ComputationGraph,           [8L, 32L, 64L]),
        ("attn-dense",   () => SdpaMeanPoolModel.ComputationGraph,        [2L, 4L, 256L, 32L]),
        ("attn-d64",     () => SdpaMeanPoolModel.ComputationGraph,        [2L, 4L, 256L, 64L]),
        ("attn-small",   () => SdpaMeanPoolModel.ComputationGraph,        [1L, 2L, 64L, 32L]),
        ("attn-chunk4",  () => ChunkedSdpaMeanPoolModel.ComputationGraph, [2L, 4L, 256L, 32L]),
    ];

    [Fact]
    public void RecordCalibrationData()
    {
        var dir = Environment.GetEnvironmentVariable("SHOROKOO_OPSPERF_CALIBRATION_DIR")
            ?? Path.Combine(Path.GetTempPath(), "shorokoo-opsperf-calibration");
        Directory.CreateDirectory(dir);
        var only = Environment.GetEnvironmentVariable("SHOROKOO_OPSPERF_CALIBRATION_FAMILIES")?.Split(',');

        foreach (var (family, model, shape) in Suite)
        {
            if (only is not null && !only.Contains(family)) continue;
            var record = Record(family, model(), shape);
            File.WriteAllText(Path.Combine(dir, family + ".json"), JsonConvert.SerializeObject(record));
        }
    }

    private static JObject Record(string family, ComputationGraph model, long[] shape)
    {
        var count = 1L;
        foreach (var d in shape) count *= d;
        NamedModelParam[] sample =
            [new TensorDataModelParam("input", ModelParamType.InputParam, TensorData(shape, new float[count]))];
        var rig = TrainingRig.FromScratch(model, L2Loss.ComputationGraph, SGDOptimizer.ComputationGraph, sample, 0.01f);

        var inputs = rig.OptimizationInputShapes
            .Select(s => Shorokoo.TensorData.CreateFromRawBytes(s.Shape, s.DType, new byte[s.Shape.Count * (s.DType.EncodingBitCount / 8)]))
            .ToArray();
        var pre = rig.PreOptimizationGraph.ToInternal();
        var preShapes = new ShapeInferenceInterpreter(new ComputeContext()).Infer(pre, inputs);

        return new JObject
        {
            ["family"] = family,
            ["shape"] = new JArray(shape),
            ["graphs"] = new JObject
            {
                ["pre"] = RecordGraph(pre, preShapes, rig.OptimizationInputShapes),
                ["post"] = RecordGraph(rig.OptimizationResult.OptimizedGraph, rig.OptimizationResult.ShapeInfo, rig.OptimizationInputShapes),
            },
        };
    }

    private static JObject RecordGraph(InternalComputationGraph graph, ShapeInferenceResult shapes, (Shape Shape, DType DType)[] inputShapes)
    {
        var eval = new GraphEvaluator().Evaluate(graph, shapes);
        var registry = new OpPerfRegistry();
        var emitted = EmittedNames(graph);
        var consumers = new Dictionary<FastTensorKey, List<string>>();
        foreach (var n in graph.Nodes)
            foreach (var k in n.Inputs)
                if (k is not null) (consumers.TryGetValue(k.Value, out var l) ? l : consumers[k.Value] = new List<string>()).Add(n.OpCode);

        var nodes = new JArray();
        for (var i = 0; i < graph.Nodes.Count; i++)
        {
            var n = graph.Nodes[i];
            var attrs = n.Attributes.GetAttributeVals().Where(kv => !kv.Key.StartsWith("shrk_")).ToDictionary(kv => kv.Key, kv => kv.Value);
            var est = n.IsModelInput() ? 0.0 : registry.Estimate(new OpPerfInput
            {
                OpCode = n.OpCode,
                InputShapes = n.Inputs.Select(k => k is null ? null : shapes.GetTensorInfo(k.Value)).ToArray(),
                OutputShapes = n.Outputs.Select(k => k is null ? null : shapes.GetTensorInfo(k.Value)).ToArray(),
                InputMustRemainIntact = n.Inputs.Select(_ => true).ToArray(),
                Attributes = attrs,
                ConsumerOpCodes = n.Outputs.Where(k => k is not null).SelectMany(k => consumers.GetValueOrDefault(k!.Value) ?? new List<string>()).ToArray(),
            }).ComputeTime;
            nodes.Add(new JObject
            {
                ["idx"] = i,
                ["name"] = emitted.GetValueOrDefault(n.Key),
                ["op"] = n.OpCode,
                ["ins"] = new JArray(n.Inputs.Select(k => k is null ? null : Describe(shapes.GetTensorInfo(k.Value)))),
                ["outs"] = new JArray(n.Outputs.Select(k => k is null ? null : Describe(shapes.GetTensorInfo(k.Value)))),
                ["attrs"] = new JObject(n.Attributes.GetAttributeVals()
                    .Where(kv => !kv.Key.StartsWith("shrk_") && kv.Value is not null)
                    .Select(kv => new JProperty(kv.Key, Stringify(kv.Value!)))),
                ["est"] = est,
                ["consumers"] = new JArray(n.Outputs.Where(k => k is not null).SelectMany(k => consumers.GetValueOrDefault(k!.Value) ?? new List<string>())),
                ["scope"] = n.GraphOpenNodeKey is { IsEmpty: false } open ? open.ToString() : null,
            });
        }

        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(graph, prepForOnnx: true, inputDims: inputShapes.Select(s => s.Shape.Dims).ToArray());
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, proto);
        var bytes = stream.ToArray();

        return new JObject
        {
            ["totalEstimate"] = eval.TotalComputeTime,
            ["nodes"] = nodes,
            ["kernels"] = new JObject
            {
                ["enableAll"] = Profile(bytes, inputShapes, GraphOptimizationLevel.ORT_ENABLE_ALL),
                ["disableAll"] = Profile(bytes, inputShapes, GraphOptimizationLevel.ORT_DISABLE_ALL),
            },
        };
    }

    // The ONNX builder clones the graph, runs these pre-passes, then renumbers every key to N1, N2, …
    // (FastUseUniqueNames); ORT's profiler reports kernels under those names. Replaying the same
    // passes on our own clone yields the same numbering, and the returned map ties each original
    // node to the name its kernel will carry (null for nodes a pass removed).
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

    private static JToken? Describe(TensorShapeInfo? info)
        => info is null ? null : new JObject { ["dims"] = new JArray(info.Shape.Dims), ["dtype"] = info.DType.ToString(), ["bytes"] = info.MemoryBytes };

    private static string Stringify(object value) => value switch
    {
        long[] xs => "[" + string.Join(",", xs) + "]",
        float[] xs => "[" + string.Join(",", xs) + "]",
        string[] xs => "[" + string.Join(",", xs) + "]",
        TensorData td => "tensor" + "[" + string.Join(",", td.Shape.Dims) + "]",
        _ => value.ToString() ?? "",
    };

    // ----- profiling -----------------------------------------------------------------

    /// <summary>
    /// Per kernel name: op type, ORT's shapes, executions per run, and the per-run summed duration
    /// (µs) as the median over runs — taken in <see cref="Sessions"/> independent sessions and
    /// reported as the smaller, since a session can land in a pathological allocation layout that
    /// every one of its runs then pays for.
    /// </summary>
    private static JArray Profile(byte[] model, (Shape Shape, DType DType)[] inputShapes, GraphOptimizationLevel level)
    {
        var best = new Dictionary<string, JObject>();
        for (var s = 0; s < Sessions; s++)
            foreach (var k in ProfileSession(model, inputShapes, level).Cast<JObject>())
            {
                var name = (string)k["name"]!;
                if (!best.TryGetValue(name, out var have) || (double)k["us"]! < (double)have["us"]!) best[name] = k;
            }
        return new JArray(best.Values);
    }

    private static JArray ProfileSession(byte[] model, (Shape Shape, DType DType)[] inputShapes, GraphOptimizationLevel level)
    {
        var dir = Path.Combine(Path.GetTempPath(), "shorokoo-opsperf-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using var options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            options.GraphOptimizationLevel = level;
            options.EnableProfiling = true;
            options.ProfileOutputPathPrefix = Path.Combine(dir, "profile");
            if (Environment.GetEnvironmentVariable("SHOROKOO_OPSPERF_DENORMAL_AS_ZERO") is "1")
                options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            using var session = new InferenceSession(model, options);
            var feeds = new Dictionary<string, OrtValue>();
            for (var i = 0; i < session.InputNames.Count; i++)
                feeds[session.InputNames[i]] = SyntheticFeed.Tensor(inputShapes[i].Shape, inputShapes[i].DType, i);
            using var runOptions = new RunOptions();
            for (var run = 0; run <= MeasuredRuns; run++)
                foreach (var o in session.Run(runOptions, feeds, session.OutputNames)) o.Dispose();
            GC.KeepAlive(feeds);

            var events = JArray.Parse(File.ReadAllText(session.EndProfiling()));
            var kernels = events
                .Where(e => (string?)e["cat"] == "Node" && ((string?)e["name"] ?? "").EndsWith("_kernel_time"))
                .ToList();
            var runs = events
                .Where(e => (string?)e["cat"] == "Session" && (string?)e["name"] == "model_run")
                .Skip(1)
                .Select(r => ((long)r["ts"]!, (long)r["ts"]! + (long)r["dur"]!))
                .ToList();

            var perName = new Dictionary<string, (JToken Sample, List<long> Sums, List<int> Counts)>();
            foreach (var (start, end) in runs)
            {
                var sums = new Dictionary<string, long>();
                var counts = new Dictionary<string, int>();
                foreach (var k in kernels)
                {
                    var ts = (long)k["ts"]!;
                    if (ts < start || ts > end) continue;
                    var name = ((string)k["name"]!)[..^"_kernel_time".Length];
                    sums[name] = sums.GetValueOrDefault(name) + (long)k["dur"]!;
                    counts[name] = counts.GetValueOrDefault(name) + 1;
                    if (!perName.ContainsKey(name)) perName[name] = (k, new List<long>(), new List<int>());
                }
                foreach (var (name, sum) in sums)
                {
                    perName[name].Sums.Add(sum);
                    perName[name].Counts.Add(counts[name]);
                }
            }

            var result = new JArray();
            foreach (var (name, (sample, sums, counts)) in perName)
            {
                var args = sample["args"];
                result.Add(new JObject
                {
                    ["name"] = name,
                    ["op"] = (string?)args?["op_name"],
                    ["ins"] = args?["input_type_shape"],
                    ["outs"] = args?["output_type_shape"],
                    ["count"] = counts.Max(),
                    ["us"] = Median(sums),
                    ["runs"] = new JArray(sums),
                });
            }
            return result;
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static double Median(List<long> xs)
    {
        var s = xs.OrderBy(x => x).ToArray();
        return s.Length % 2 == 1 ? s[s.Length / 2] : (s[s.Length / 2 - 1] + s[s.Length / 2]) / 2.0;
    }
}
