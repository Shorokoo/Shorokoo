using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Shorokoo.Core.Backends;
using Shorokoo.Core.Factory.IR;
using Shorokoo.OnnxRuntime;
using OrtElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// The memory-reuse scenario, run under every configuration that bears on it, its figures written
/// to <c>$SHOROKOO_MEMORY_REUSE_DIR</c> (default: a folder under the temp path), one Markdown report
/// and one JSON file per fact.
///
/// <para><b>The scenario.</b> Inputs A and B, each <c>[2N, M]</c> float, both consumed by the run.
/// <c>A_half = A[0:N]</c> and <c>B_half = B[N:2N]</c>, each a <c>Slice</c> whose bounds the run
/// computes from A's shape. C is a <c>[N, M]</c> tensor of 2s (<c>ConstantOfShape</c> of A_half's
/// shape) through <c>Neg</c>, <c>Abs</c>, <c>Sigmoid</c>; <c>L0 = Concat(C, B_half)</c> on axis 0,
/// then <c>Sigmoid</c>, <c>Neg</c>, <c>Abs</c> give L. Outputs L, A_half, B_half. Every dimension
/// is symbolic, so ONNX Runtime cannot fold the fill and its chain into a constant.</para>
///
/// <para><b>The ideal</b> is no memory beyond A and B: A_half stays where it is in A, B_half is
/// copied into A's second half, C is made in B's first half — so that B holds the concatenation
/// with no copy, B_half's rows being in place already — and every unary op runs in place.</para>
///
/// <para><b>Configurations</b>, on the host and on the card: ONNX Runtime's own arena or
/// Shorokoo's allocator; graph optimizations on or off; the memory pattern on or off; and five
/// approaches. <i>Plain</i>: ONNX Runtime allocates every output. <i>BindFinal</i>: A_half, B_half
/// and L bound into the consumed inputs. <i>BindAll</i>: every intermediate made a graph output and
/// bound to its final place. <i>Redirect</i>: Shorokoo's allocator serving the run's requests out of
/// the consumed inputs, as a recording run mapped them. <i>Shipped</i>: the backend's own session
/// and <c>RunConsuming</c>. Every request the allocator serves and every block it takes back is
/// logged in order; an arena session is read through its arena's figures.</para>
///
/// <para>Run with <c>dotnet test -p:ShorokooGpuTests=true --filter
/// "FullyQualifiedName~MemoryReuseScenarioTests"</c>; the card's fact needs that build, the host's
/// and the Python one run in either.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
[Collection(SerialMeasurement.Name)]
public class MemoryReuseScenarioTests
{
    private const long N = 4096;
    private const long M = 2048;
    private const long HalfCount = N * M;
    private const long WholeCount = 2 * N * M;
    private const long HalfBytes = HalfCount * sizeof(float);
    private const long WholeBytes = WholeCount * sizeof(float);
    private const long Big = 1L << 20;
    private const int Runs = 3;

    private static readonly string[] Outputs = ["L", "A_half", "B_half"];
    private static readonly string[] Intermediates = ["C0", "C1", "C2", "C3", "L0", "L1", "L2"];

    private static string OutputDirectory()
    {
        var dir = Environment.GetEnvironmentVariable("SHOROKOO_MEMORY_REUSE_DIR")
            ?? Path.Combine(Path.GetTempPath(), "shorokoo-memory-reuse");
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void RecordTheScenarioOnTheHost() => Record(card: false, "host");

    [CudaFact]
    public void RecordTheScenarioOnTheCard() => Record(card: true, "card");

    // ---- the scenario as an ONNX model ----

    /// <summary>
    /// How the model states its shapes, which decides what ONNX Runtime can know of them when it
    /// plans the run. <i>Computed</i>: A and B are <c>[rows, cols]</c> and the run computes
    /// <c>rows / 2</c> for the slices, so the planner knows no dimension of a half, the fill or the
    /// concatenation. <i>Paired</i>: A and B are <c>[2, n, m]</c> and each half is the slice
    /// <c>[0:1]</c> or <c>[1:2]</c> of the leading dimension, <c>[1, n, m]</c> — the same bytes in
    /// the same order, with every shape symbolic yet known to the planner. <i>Static</i>: A and B
    /// are <c>[2N, M]</c> with constant slice bounds, so the whole of C's chain is computable when
    /// the session is built. <i>Tracked</i>: as Paired, with the fill made as <c>A_half * 0 + 2</c>
    /// rather than by <c>ConstantOfShape</c>, whose shape the planner cannot follow from a shape the
    /// run computes — so every shape in the graph is symbolic and known to the planner.
    /// </summary>
    internal enum Shapes { Computed, Paired, Static, Tracked }

    private static readonly Shapes[] AllShapes = [Shapes.Computed, Shapes.Paired, Shapes.Tracked, Shapes.Static];

    private static bool Leading(Shapes shapes) => shapes is Shapes.Paired or Shapes.Tracked;

    private static long[] WholeShape(Shapes shapes) => Leading(shapes) ? [2, N, M] : [2 * N, M];

    private static long[] HalfShape(Shapes shapes) => Leading(shapes) ? [1, N, M] : [N, M];

    internal static byte[] Scenario(Shapes shapes, bool exposeIntermediates)
    {
        var graph = new GraphProto { Name = "memory_reuse" };
        string?[] dims = shapes switch
        {
            Shapes.Computed => ["rows", "cols"],
            Shapes.Paired or Shapes.Tracked => ["2", "n", "m"],
            _ => [(2 * N).ToString(CultureInfo.InvariantCulture), M.ToString(CultureInfo.InvariantCulture)],
        };
        graph.Inputs.Add(FloatValue("A", dims));
        graph.Inputs.Add(FloatValue("B", dims));
        graph.Initializers.Add(Int64s("zero", 0));
        switch (shapes)
        {
            case Shapes.Computed:
                graph.Initializers.Add(Int64s("two", 2));
                graph.Nodes.Add(Node("shape_A", "Shape", "A", "rows", Int("start", 0), Int("end", 1)));
                graph.Nodes.Add(Node("half_rows", "Div", "rows two", "half"));
                graph.Nodes.Add(Node("slice_A", "Slice", "A zero half zero", "A_half"));
                graph.Nodes.Add(Node("slice_B", "Slice", "B half rows zero", "B_half"));
                break;
            case Shapes.Paired or Shapes.Tracked:
                graph.Initializers.Add(Int64s("one", 1));
                graph.Initializers.Add(Int64s("two", 2));
                graph.Nodes.Add(Node("slice_A", "Slice", "A zero one zero", "A_half"));
                graph.Nodes.Add(Node("slice_B", "Slice", "B one two zero", "B_half"));
                break;
            default:
                graph.Initializers.Add(Int64s("half", N));
                graph.Initializers.Add(Int64s("rows", 2 * N));
                graph.Nodes.Add(Node("slice_A", "Slice", "A zero half zero", "A_half"));
                graph.Nodes.Add(Node("slice_B", "Slice", "B half rows zero", "B_half"));
                break;
        }
        if (shapes == Shapes.Tracked)
        {
            graph.Initializers.Add(new TensorProto { Name = "zero_f", data_type = (int)TensorProto.DataType.Float, Dims = [], FloatDatas = [0f] });
            graph.Initializers.Add(new TensorProto { Name = "two_f", data_type = (int)TensorProto.DataType.Float, Dims = [], FloatDatas = [2f] });
            graph.Nodes.Add(Node("fill_mul", "Mul", "A_half zero_f", "c_zero"));
            graph.Nodes.Add(Node("fill_C", "Add", "c_zero two_f", "C0"));
        }
        else
        {
            graph.Nodes.Add(Node("shape_C", "Shape", "A_half", "c_shape"));
            graph.Nodes.Add(Node("fill_C", "ConstantOfShape", "c_shape", "C0", new AttributeProto
            {
                Name = "value", Type = AttributeProto.AttributeType.Tensor,
                T = new TensorProto { data_type = (int)TensorProto.DataType.Float, Dims = [1], FloatDatas = [2f] },
            }));
        }
        graph.Nodes.Add(Node("c_neg", "Neg", "C0", "C1"));
        graph.Nodes.Add(Node("c_abs", "Abs", "C1", "C2"));
        graph.Nodes.Add(Node("c_sigmoid", "Sigmoid", "C2", "C3"));
        graph.Nodes.Add(Node("concat_L", "Concat", "C3 B_half", "L0", Int("axis", 0)));
        graph.Nodes.Add(Node("l_sigmoid", "Sigmoid", "L0", "L1"));
        graph.Nodes.Add(Node("l_neg", "Neg", "L1", "L2"));
        graph.Nodes.Add(Node("l_abs", "Abs", "L2", "L"));
        foreach (var name in exposeIntermediates ? [.. Outputs, .. Intermediates] : Outputs)
            graph.Outputs.Add(FloatValue(name, null));
        var model = new ModelProto { IrVersion = 10, Graph = graph, ProducerName = "memory-reuse-scenario" };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });
        using var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, model);
        return stream.ToArray();
    }

    private static TensorProto Int64s(string name, long value)
        => new() { Name = name, data_type = (int)TensorProto.DataType.Int64, Dims = [1], Int64Datas = [value] };

    /// <summary>A float value, of the shape <paramref name="dims"/> names — a number for a fixed
    /// dimension, a name for a symbolic one — or of no stated shape.</summary>
    private static ValueInfoProto FloatValue(string name, string?[]? dims)
    {
        var tensor = new TypeProto.Tensor { ElemType = (int)TensorProto.DataType.Float };
        if (dims is not null)
        {
            tensor.Shape = new TensorShapeProto();
            foreach (var dim in dims)
                tensor.Shape.Dims.Add(long.TryParse(dim, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                    ? new TensorShapeProto.Dimension { DimValue = value }
                    : new TensorShapeProto.Dimension { DimParam = dim! });
        }
        return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = tensor } };
    }

    private static AttributeProto Int(string name, long value)
        => new() { Name = name, Type = AttributeProto.AttributeType.Int, I = value };

    private static NodeProto Node(string name, string op, string inputs, string outputs, params AttributeProto[] attributes)
    {
        var node = new NodeProto { Name = name, OpType = op, Domain = "" };
        node.Inputs.AddRange(inputs.Split(' '));
        node.Outputs.AddRange(outputs.Split(' '));
        node.Attributes.AddRange(attributes);
        return node;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] Pattern(bool b)
    {
        var values = new float[WholeCount];
        for (long i = 0; i < values.Length; i++)
            values[i] = b ? 0.5f - (i % 997) * 0.002f : (i % 1013) * 0.001f;
        return values;
    }

    // ---- the consumed inputs, in memory of the harness's own ----

    /// <summary>A and B, allocated once outside every allocator under test and filled again before
    /// each run, and the views a run binds over them.</summary>
    private sealed class Inputs : IDisposable
    {
        internal readonly bool OnCard;
        internal readonly IntPtr A;
        internal readonly IntPtr B;
        internal readonly float[] APattern = Pattern(b: false);
        internal readonly float[] BPattern = Pattern(b: true);
        private readonly OrtMemoryInfo? _cardInfo;

        /// <summary>The card's memory, as a binding names where ONNX Runtime is to leave an output;
        /// null on the host.</summary>
        internal OrtMemoryInfo? CardInfo => _cardInfo;

        internal Inputs(bool onCard)
        {
            OnCard = onCard;
            _cardInfo = onCard ? new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default) : null;
            A = Allocate();
            B = Allocate();
            Assert.NotEqual(IntPtr.Zero, A);
            Assert.NotEqual(IntPtr.Zero, B);
        }

        private IntPtr Allocate()
            => OnCard ? CudaInterop.Allocate(0, WholeBytes) : Marshal.AllocHGlobal((nint)WholeBytes);

        internal void Refill()
        {
            Write(A, APattern);
            Write(B, BPattern);
        }

        private void Write(IntPtr to, float[] from)
        {
            if (OnCard) Assert.Equal(0, CudaInterop.CopyHostToDevice(MemoryMarshal.AsBytes(from.AsSpan()), to));
            else Marshal.Copy(from, 0, to, from.Length);
        }

        internal float[] Read(IntPtr from, long count)
        {
            var values = new float[count];
            if (OnCard) Assert.Equal(0, CudaInterop.CopyDeviceToHost(from, MemoryMarshal.AsBytes(values.AsSpan())));
            else Marshal.Copy(from, values, 0, values.Length);
            return values;
        }

        internal OrtValue View(IntPtr at, long[] shape)
            => OrtValue.CreateTensorValueWithData(
                _cardInfo ?? OrtMemoryInfo.DefaultInstance, OrtElementType.Float, shape, at,
                shape.Aggregate((long)sizeof(float), (a, d) => a * d));

        /// <summary>Where <paramref name="address"/> is, in the inputs' words.</summary>
        internal string Where(IntPtr address)
        {
            long offset;
            if ((offset = (long)address - (long)A) >= 0 && offset < WholeBytes) return $"A+{offset >> 20}MiB";
            if ((offset = (long)address - (long)B) >= 0 && offset < WholeBytes) return $"B+{offset >> 20}MiB";
            return "own";
        }

        public void Dispose()
        {
            if (OnCard)
            {
                CudaInterop.Release(0, A);
                CudaInterop.Release(0, B);
            }
            else
            {
                Marshal.FreeHGlobal(A);
                Marshal.FreeHGlobal(B);
            }
            _cardInfo?.Dispose();
        }

        /// <summary>Whether the three outputs, at those addresses, hold what the scenario computes
        /// from A and B as they were filled.</summary>
        internal bool Correct(IntPtr l, IntPtr aHalf, IntPtr bHalf)
        {
            var lValues = Read(l, WholeCount);
            var aValues = Read(aHalf, HalfCount);
            var bValues = Read(bHalf, HalfCount);
            var c = Sigmoid(Sigmoid(2f));
            for (long i = 0; i < HalfCount; i++)
            {
                if (aValues[i] != APattern[i] || bValues[i] != BPattern[HalfCount + i]) return false;
                if (MathF.Abs(lValues[i] - c) > 1e-4f) return false;
                if (MathF.Abs(lValues[HalfCount + i] - Sigmoid(BPattern[HalfCount + i])) > 1e-4f) return false;
            }
            return true;
        }
    }

    // ---- configurations ----

    private enum AllocatorKind { Arena, Shorokoo }

    private enum Approach { Plain, BindFinal, BindAll, Redirect, Shipped }

    private sealed record Config(bool Card, Shapes Shapes, AllocatorKind Allocator, bool Optimized, bool MemoryPattern, Approach Approach, bool Parallel = false)
    {
        public override string ToString()
            => $"{(Card ? "card" : "host")}/{Shapes}/{Allocator}/{(Optimized ? "opt-all" : "opt-off")}/{(MemoryPattern ? "pattern-on" : "pattern-off")}/{Approach}{(Parallel ? "/parallel" : "")}";
    }

    private static IEnumerable<Config> Configurations(bool card)
    {
        foreach (var shapes in AllShapes)
        {
            foreach (var approach in (Approach[])[Approach.Plain, Approach.BindFinal, Approach.BindAll])
                foreach (var allocator in (AllocatorKind[])[AllocatorKind.Arena, AllocatorKind.Shorokoo])
                    foreach (var optimized in (bool[])[true, false])
                        foreach (var pattern in (bool[])[true, false])
                            yield return new Config(card, shapes, allocator, optimized, pattern, approach);
            yield return new Config(card, shapes, AllocatorKind.Shorokoo, true, true, Approach.Plain, Parallel: true);
            yield return new Config(card, shapes, AllocatorKind.Shorokoo, true, false, Approach.Redirect);
            yield return new Config(card, shapes, AllocatorKind.Shorokoo, false, false, Approach.Redirect);
            yield return new Config(card, shapes, AllocatorKind.Shorokoo, true, true, Approach.Shipped);
        }
    }

    /// <summary>Where each output a bound approach binds goes: into which input, from which byte,
    /// at what shape.</summary>
    private static (string Name, bool InA, long Offset, long[] Shape)[] Bindings(Approach approach, Shapes shapes)
    {
        var whole = WholeShape(shapes);
        var half = HalfShape(shapes);
        (string, bool, long, long[])[] finals =
        [
            ("L", false, 0, whole),
            ("A_half", true, 0, half),
            ("B_half", true, HalfBytes, half),
        ];
        if (approach == Approach.BindFinal) return finals;
        return
        [
            .. finals,
            ("C0", false, 0, half), ("C1", false, 0, half), ("C2", false, 0, half), ("C3", false, 0, half),
            ("L0", false, 0, whole), ("L1", false, 0, whole), ("L2", false, 0, whole),
        ];
    }

    // ---- what a run did ----

    /// <summary>What the allocators were asked and given back during one run, in order.</summary>
    private sealed class Trace
    {
        private readonly object _gate = new();
        private readonly List<(CachingAllocator.Event Event, long Ticks)> _events = [];
        private long _start;

        internal void Start()
        {
            _start = Stopwatch.GetTimestamp();
            CachingAllocator.Observer = e =>
            {
                var now = Stopwatch.GetTimestamp();
                lock (_gate) _events.Add((e, now));
            };
        }

        internal static void Stop() => CachingAllocator.Observer = null;

        /// <summary>Each event, with the microseconds from <see cref="Start"/> to it.</summary>
        internal List<(CachingAllocator.Event Event, double Micros)> Events
        {
            get
            {
                lock (_gate) return [.. _events.Select(e => (e.Event, (e.Ticks - _start) * 1e6 / Stopwatch.Frequency))];
            }
        }

        internal double MicrosSinceStart() => (Stopwatch.GetTimestamp() - _start) * 1e6 / Stopwatch.Frequency;
    }

    /// <summary>One run as it was observed, before its events are put to the kernels that made
    /// them.</summary>
    private sealed record RawRun(
        int Run,
        List<(CachingAllocator.Event Event, double Micros)> Events,
        double Micros,
        Dictionary<string, IntPtr> Outputs,
        bool Correct,
        Dictionary<string, string>? ArenaBefore,
        Dictionary<string, string>? ArenaAfter);

    /// <summary>The kernels of one run as ONNX Runtime profiled them: microseconds from the run's
    /// start, and the run's length.</summary>
    private sealed record ProfiledRun(double Micros, List<(string Node, double Start, double End)> Nodes)
    {
        /// <summary>The kernel an event <paramref name="micros"/> into the harness's measure of
        /// the run happened in or after, the two clocks aligned on the middle of the slack between
        /// the harness's measure and the profile's.</summary>
        internal string? At(double micros, double runMicros, bool allocation)
        {
            if (Nodes.Count == 0) return null;
            var at = micros - Math.Max(0, (runMicros - Micros) / 2);
            if (allocation)
            {
                var within = Nodes.FirstOrDefault(n => at >= n.Start - 2 && at <= n.End + 2);
                if (within.Node is not null) return Short(within.Node);
                var nearest = Nodes.MinBy(n => Math.Min(Math.Abs(at - n.Start), Math.Abs(at - n.End)));
                return Short(nearest.Node) + "?";
            }
            var last = Nodes.LastOrDefault(n => n.Start <= at);
            return last.Node is null ? "before the kernels" : (at <= last.End ? "in " : "after ") + Short(last.Node);
        }

        private static string Short(string node) => node.Split(':')[0];
    }

    private sealed record RunRecord(
        int Run,
        int Requests,
        long RequestedBytes,
        int BigRequests,
        long PeakBytes,
        long HeldAtEndBytes,
        int FreshBlocks,
        int Redirected,
        List<string> Sequence,
        Dictionary<string, string> OutputPlaces,
        bool Correct,
        double Milliseconds,
        Dictionary<string, string>? ArenaBefore,
        Dictionary<string, string>? ArenaAfter,
        List<IntPtr> BigAddresses,
        List<long> BigSizes);

    private static RunRecord Summarize(RawRun raw, Inputs inputs, ProfiledRun? profile)
    {
        var (run, timed, micros, outputs, correct, arenaBefore, arenaAfter) = raw;
        var milliseconds = micros / 1000;
        string Kernel(double at, bool allocation) => profile?.At(at, micros, allocation) is { } node ? $" [{node}]" : "";
        var live = new Dictionary<IntPtr, (long Bytes, string Label)>();
        var sequence = new List<string>();
        List<IntPtr> bigAddresses = [];
        List<long> bigSizes = [];
        int requests = 0, big = 0, fresh = 0, redirected = 0, small = 0;
        long requested = 0, inUse = 0, peak = 0;
        foreach (var (e, at) in timed)
        {
            if (e.Allocation)
            {
                if (e.Redirected)
                {
                    redirected++;
                    if (e.Requested >= Big) sequence.Add($"+{Mib(e.Requested)} -> {inputs.Where(e.Address)} (redirected){Kernel(at, true)}");
                    continue;
                }
                requests++;
                requested += e.Requested;
                if (e.Fresh) fresh++;
                inUse += e.Requested;
                peak = Math.Max(peak, inUse);
                if (e.Requested >= Big)
                {
                    big++;
                    bigAddresses.Add(e.Address);
                    bigSizes.Add(e.Requested);
                    var label = $"#{big}";
                    live[e.Address] = (e.Requested, label);
                    var where = inputs.Where(e.Address);
                    sequence.Add($"+{Mib(e.Requested)} {label}{(e.Fresh ? " fresh" : " cached")}{(where == "own" ? "" : " at " + where)}{Kernel(at, true)}");
                }
                else
                {
                    small++;
                    live[e.Address] = (e.Requested, "small");
                }
            }
            else if (e.Redirected)
            {
                if (sequence.Count > 0) sequence.Add($"-redirected {inputs.Where(e.Address)}{Kernel(at, false)}");
            }
            else if (live.Remove(e.Address, out var block))
            {
                inUse -= block.Bytes;
                if (block.Label != "small") sequence.Add($"-{Mib(block.Bytes)} {block.Label}{Kernel(at, false)}");
            }
            else if (e.Requested >= Big)
                sequence.Add($"-{Mib(e.Requested)} block from before the run{Kernel(at, false)}");
        }
        if (small > 0) sequence.Add($"({small} requests under 1 MiB)");
        var places = outputs.ToDictionary(o => o.Key, o =>
        {
            var where = inputs.Where(o.Value);
            if (where != "own") return where;
            var index = bigAddresses.LastIndexOf(o.Value);
            return index >= 0 ? $"#{index + 1}" : "own";
        });
        return new RunRecord(run, requests, requested, big, peak, inUse, fresh, redirected, sequence, places, correct,
            milliseconds, arenaBefore, arenaAfter, bigAddresses, bigSizes);
    }

    private static string Mib(long bytes) => (bytes / (double)(1 << 20)).ToString("0.##", CultureInfo.InvariantCulture) + "MiB";

    // ---- running a configuration ----

    private sealed record ConfigRecord(string Config, List<RunRecord> Runs, List<string> NodeOrder, string? Failure);

    private static void Record(bool card, string name)
    {
        var dir = OutputDirectory();
        _ = CachingAllocator.ForHost();
        if (card) _ = CachingAllocator.ForCard(0);
        foreach (var shapes in AllShapes)
        {
            File.WriteAllBytes(Path.Combine(dir, $"scenario-{shapes}.onnx"), Scenario(shapes, exposeIntermediates: false));
            File.WriteAllBytes(Path.Combine(dir, $"scenario-{shapes}-exposed.onnx"), Scenario(shapes, exposeIntermediates: true));
        }

        List<ConfigRecord> records = [];
        using (var inputs = new Inputs(card))
        {
            foreach (var config in Configurations(card))
            {
                try
                {
                    var model = Scenario(config.Shapes, exposeIntermediates: config.Approach == Approach.BindAll);
                    records.Add(config.Approach == Approach.Shipped
                        ? RunShipped(config, model)
                        : RunConfig(config, model, inputs, dir));
                }
                catch (Exception failure)
                {
                    records.Add(new ConfigRecord(config.ToString(), [], [], failure.GetType().Name + ": " + failure.Message));
                }
            }
        }
        File.WriteAllText(Path.Combine(dir, $"memory-reuse-{name}.json"), JsonConvert.SerializeObject(records, Formatting.Indented));
        File.WriteAllText(Path.Combine(dir, $"memory-reuse-{name}.md"), Report(name, records));
        Assert.All(records, r => Assert.Null(r.Failure));
    }

    private static InferenceSession NewSession(byte[] model, Config config, string profilePrefix)
    {
        using var options = new SessionOptions();
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        options.GraphOptimizationLevel = config.Optimized ? GraphOptimizationLevel.ORT_ENABLE_ALL : GraphOptimizationLevel.ORT_DISABLE_ALL;
        options.EnableMemoryPattern = config.MemoryPattern;
        options.ExecutionMode = config.Parallel ? ExecutionMode.ORT_PARALLEL : ExecutionMode.ORT_SEQUENTIAL;
        if (config.Allocator == AllocatorKind.Shorokoo) options.AddSessionConfigEntry("session.use_env_allocators", "1");
        options.ProfileOutputPathPrefix = profilePrefix;
        options.EnableProfiling = true;
        if (config.Card) OrtBackend.AppendCuda(options, 0, DeviceMemorySettings.Default);
        return new InferenceSession(model, options);
    }

    private static Dictionary<string, string>? ArenaFigures(InferenceSession session, Config config)
    {
        if (config.Allocator != AllocatorKind.Arena) return null;
        using var cardInfo = config.Card ? new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default) : null;
        using var allocator = new OrtAllocator(session, cardInfo ?? OrtMemoryInfo.DefaultInstance);
        return OrtArenaStats.ReadRaw(allocator) is { } raw ? new Dictionary<string, string>(raw) : null;
    }

    private static ConfigRecord RunConfig(Config config, byte[] model, Inputs inputs, string dir)
    {
        var profilePrefix = Path.Combine(dir, "profile-" + config.ToString().Replace('/', '-'));
        CachingAllocator.Account? host = null, cardAccount = null;
        if (config.Allocator == AllocatorKind.Shorokoo)
        {
            host = CachingAllocator.ForHost().Open("scenario");
            cardAccount = config.Card ? CachingAllocator.ForCard(0).Open("scenario") : null;
        }
        InferenceSession session;
        using (CachingAllocator.Charge(host, cardAccount)) session = NewSession(model, config, profilePrefix);
        List<RawRun> raws = [];
        List<ProfiledRun> profiled;
        IntPtr[]? script = null;
        try
        {
            for (int run = 1; run <= Runs; run++)
            {
                var raw = RunOnce(session, config, inputs, run, host, cardAccount, script);
                raws.Add(raw);
                if (config.Approach == Approach.Redirect && script is null) script = Script(Summarize(raw, inputs, null), inputs);
            }
        }
        finally
        {
            var profile = session.EndProfiling();
            session.Dispose();
            if (host is not null) host.Allocator.Close(host);
            if (cardAccount is not null) cardAccount.Allocator.Close(cardAccount);
            profiled = ProfiledRuns(profile);
            try { File.Delete(profile); } catch (IOException) { }
        }
        var runs = raws.Select((raw, i) => Summarize(raw, inputs, i < profiled.Count ? profiled[i] : null)).ToList();
        var order = profiled.Count > 0 ? profiled[^1].Nodes.Select(n => n.Node).ToList() : [];
        return new ConfigRecord(config.ToString(), runs, order, null);
    }

    /// <summary>
    /// Where a redirected run serves each request of a megabyte or more, in the order the recording
    /// run made them: A_half's into A's first half, B_half's into A's second, L's and every other
    /// request into B — C's chain in B's first half, the concatenation and L's chain over all of B.
    /// </summary>
    private static IntPtr[] Script(RunRecord recording, Inputs inputs)
    {
        var script = new IntPtr[recording.BigAddresses.Count];
        for (int k = 0; k < script.Length; k++)
        {
            var place = recording.OutputPlaces.FirstOrDefault(p => p.Value == $"#{k + 1}").Key;
            script[k] = place switch
            {
                "A_half" => inputs.A,
                "B_half" => inputs.A + (nint)HalfBytes,
                _ => inputs.B,
            };
        }
        return script;
    }

    private static RawRun RunOnce(
        InferenceSession session, Config config, Inputs inputs, int run,
        CachingAllocator.Account? host, CachingAllocator.Account? card, IntPtr[]? script)
    {
        inputs.Refill();
        using var a = inputs.View(inputs.A, WholeShape(config.Shapes));
        using var b = inputs.View(inputs.B, WholeShape(config.Shapes));
        using var runOptions = new RunOptions();
        // An output ONNX Runtime allocates is left on the card through a binding, as the backend
        // leaves one: a plain run would copy it to the host.
        var bound = config.Approach is Approach.BindFinal or Approach.BindAll;
        OrtIoBinding? binding = null;
        List<(string Name, OrtValue Value, IntPtr At)> views = [];
        if (bound || config.Card)
        {
            binding = session.CreateIoBinding();
            binding.BindInput("A", a);
            binding.BindInput("B", b);
        }
        if (bound)
        {
            foreach (var (name, inA, offset, shape) in Bindings(config.Approach, config.Shapes))
            {
                var at = (inA ? inputs.A : inputs.B) + (nint)offset;
                var view = inputs.View(at, shape);
                views.Add((name, view, at));
                binding!.BindOutput(name, view);
            }
        }
        else if (binding is not null)
            foreach (var name in Outputs) binding.BindOutputToDevice(name, inputs.CardInfo!);

        Func<bool, long, IntPtr>? redirect = null;
        if (script is not null)
        {
            var next = 0;
            redirect = (onCard, bytes) => onCard == config.Card && bytes >= Big && next < script.Length ? script[next++] : IntPtr.Zero;
        }

        var arenaBefore = ArenaFigures(session, config);
        var trace = new Trace();
        IDisposableReadOnlyCollection<OrtValue>? fetched = null;
        double micros;
        using (CachingAllocator.Charge(host, card))
        {
            CachingAllocator.Redirect = redirect;
            trace.Start();
            try
            {
                if (binding is null) fetched = session.Run(runOptions, ["A", "B"], [a, b], Outputs);
                else if (!bound) fetched = session.RunWithBoundResults(runOptions, binding);
                else
                {
                    session.RunWithBinding(runOptions, binding);
                    binding.SynchronizeBoundOutputs();
                }
            }
            finally
            {
                micros = trace.MicrosSinceStart();
                Trace.Stop();
                CachingAllocator.Redirect = null;
            }
        }
        var arenaAfter = ArenaFigures(session, config);

        var addresses = new Dictionary<string, IntPtr>();
        if (fetched is not null)
            for (int i = 0; i < Outputs.Length; i++) addresses[Outputs[i]] = OrtBackend.AddressOf(fetched[i]);
        else
            foreach (var (name, _, at) in views.Where(v => Outputs.Contains(v.Name))) addresses[name] = at;
        var correct = inputs.Correct(addresses["L"], addresses["A_half"], addresses["B_half"]);
        var raw = new RawRun(run, trace.Events, micros, addresses, correct, arenaBefore, arenaAfter);

        fetched?.Dispose();
        foreach (var (_, view, _) in views) view.Dispose();
        binding?.Dispose();
        return raw;
    }

    /// <summary>The backend's own path: a session of the stock backend for the device, fed tensors
    /// it placed, run through <c>RunConsuming</c>, which releases them as it returns.</summary>
    private static ConfigRecord RunShipped(Config config, byte[] model)
    {
        OrtBackend backend = config.Card ? new CardBackend() : new HostBackend();
        using var session = backend.CreateSession(model, ShorokooGraphOptimization.EnableAll, ShorokooLogSeverity.Error, DeviceMemorySettings.Default);
        var aBytes = MemoryMarshal.AsBytes(Pattern(b: false).AsSpan()).ToArray();
        var bBytes = MemoryMarshal.AsBytes(Pattern(b: true).AsSpan()).ToArray();
        List<RunRecord> runs = [];
        using var reader = new Inputs(config.Card);
        for (int run = 1; run <= Runs; run++)
        {
            var a = (OrtTensorValue)((IShorokooBackend)backend).CreateTensorInBackendMemory(ShorokooTensorElementType.Float, aBytes, WholeShape(config.Shapes));
            var b = (OrtTensorValue)((IShorokooBackend)backend).CreateTensorInBackendMemory(ShorokooTensorElementType.Float, bBytes, WholeShape(config.Shapes));
            var aAt = OrtBackend.AddressOf(a.Inner);
            var bAt = OrtBackend.AddressOf(b.Inner);
            var trace = new Trace();
            trace.Start();
            IReadOnlyList<IShorokooTensorValue> results;
            double micros;
            try
            {
                results = session.RunConsuming(
                    new Dictionary<string, IShorokooTensorValue> { ["A"] = a, ["B"] = b }, [a, b], Outputs, RunSettings.Default);
            }
            finally
            {
                micros = trace.MicrosSinceStart();
                Trace.Stop();
            }
            var addresses = new Dictionary<string, IntPtr>();
            for (int i = 0; i < Outputs.Length; i++) addresses[Outputs[i]] = OrtBackend.AddressOf(((OrtTensorValue)results[i]).Inner);
            var correct = reader.Correct(addresses["L"], addresses["A_half"], addresses["B_half"]);
            var places = new ShippedPlaces(aAt, bAt);
            var record = Summarize(new RawRun(run, trace.Events, micros, addresses, correct, null, null), reader, null);
            record = record with
            {
                OutputPlaces = addresses.ToDictionary(o => o.Key, o => places.Where(o.Value) ?? record.OutputPlaces[o.Key]),
                Sequence = [.. record.Sequence, $"(consumed inputs: A at {aAt:X}, B at {bAt:X})"],
            };
            runs.Add(record);
            foreach (var value in results) ((IShorokooBackend)backend).Release(value);
        }
        return new ConfigRecord(config.ToString(), runs, [], null);
    }

    private sealed record ShippedPlaces(IntPtr A, IntPtr B)
    {
        internal string? Where(IntPtr address)
        {
            long offset;
            if ((offset = (long)address - (long)A) >= 0 && offset < WholeBytes) return $"consumed A+{offset >> 20}MiB";
            if ((offset = (long)address - (long)B) >= 0 && offset < WholeBytes) return $"consumed B+{offset >> 20}MiB";
            return null;
        }
    }

    private sealed class HostBackend : OrtBackend;

    private sealed class CardBackend() : OrtBackend(cudaDeviceId: 0);

    /// <summary>The kernels of the last run the profile at <paramref name="path"/> recorded, in the
    /// order they ran: node, operator, provider.</summary>
    private static List<ProfiledRun> ProfiledRuns(string path)
    {
        List<ProfiledRun> runs = [];
        if (!File.Exists(path)) return runs;
        var events = JArray.Parse(File.ReadAllText(path));
        var kernels = events
            .Where(e => (string?)e["cat"] == "Node" && ((string?)e["name"])?.EndsWith("_kernel_time", StringComparison.Ordinal) == true)
            .Select(e => (Ts: (long)e["ts"]!, Dur: (long)e["dur"]!,
                Name: $"{((string)e["name"]!).Replace("_kernel_time", "")}:{e["args"]?["op_name"]}@{((string?)e["args"]?["provider"])?.Replace("ExecutionProvider", "")}"))
            .OrderBy(k => k.Ts)
            .ToList();
        foreach (var run in events.Where(e => (string?)e["name"] == "model_run").Select(e => (Ts: (long)e["ts"]!, Dur: (long)e["dur"]!)).OrderBy(r => r.Ts))
            runs.Add(new ProfiledRun(run.Dur, [.. kernels
                .Where(k => k.Ts >= run.Ts && k.Ts <= run.Ts + run.Dur)
                .Select(k => (k.Name, (double)(k.Ts - run.Ts), (double)(k.Ts - run.Ts + k.Dur)))]));
        return runs;
    }

    // ---- the report ----

    private static string Report(string name, List<ConfigRecord> records)
    {
        var text = new StringBuilder();
        text.AppendLine(CultureInfo.InvariantCulture, $"# Memory reuse scenario on the {name}");
        text.AppendLine();
        text.AppendLine(CultureInfo.InvariantCulture, $"Inputs A and B are [{2 * N}, {M}] float, {Mib(WholeBytes)} each; halves {Mib(HalfBytes)}.");
        text.AppendLine("Figures are what ONNX Runtime asked of Shorokoo's allocator during the run (beyond the inputs, outputs included),");
        text.AppendLine("or for an arena session what its arena reports. Run 1 is the session's first run.");
        text.AppendLine();
        text.AppendLine("| configuration | run | requests >= 1 MiB | bytes asked | peak in use | fresh blocks | redirected | arena allocs / in use / max in use / reserved | L / A_half / B_half | correct |");
        text.AppendLine("|---|---|---|---|---|---|---|---|---|---|");
        foreach (var record in records)
        {
            if (record.Failure is not null)
            {
                text.AppendLine(CultureInfo.InvariantCulture, $"| {record.Config} | - | failed: {record.Failure.Replace('|', '/').Replace('\n', ' ')} | | | | | | | |");
                continue;
            }
            foreach (var run in record.Runs)
            {
                var arena = run.ArenaAfter is { } after
                    ? $"{Delta(run.ArenaBefore, after, "NumAllocs")} / {Mib(Figure(after, "InUse"))} / {Mib(Figure(after, "MaxInUse"))} / {Mib(Figure(after, "TotalAllocated"))}"
                    : "";
                text.AppendLine(CultureInfo.InvariantCulture,
                    $"| {record.Config} | {run.Run} | {run.BigRequests} | {Mib(run.RequestedBytes)} | {Mib(run.PeakBytes)} | {run.FreshBlocks} | {run.Redirected} | {arena} | {run.OutputPlaces["L"]} / {run.OutputPlaces["A_half"]} / {run.OutputPlaces["B_half"]} | {run.Correct} |");
            }
        }
        text.AppendLine();
        text.AppendLine("## Sequences");
        foreach (var record in records.Where(r => r.Failure is null))
        {
            text.AppendLine();
            text.AppendLine(CultureInfo.InvariantCulture, $"### {record.Config}");
            if (record.NodeOrder.Count > 0) text.AppendLine(CultureInfo.InvariantCulture, $"kernels: {string.Join(", ", record.NodeOrder)}");
            foreach (var run in record.Runs)
                text.AppendLine(CultureInfo.InvariantCulture, $"- run {run.Run} ({run.Milliseconds:0.0} ms): {string.Join(", ", run.Sequence)}");
        }
        return text.ToString();
    }

    private static long Figure(Dictionary<string, string> figures, string name)
        => figures.TryGetValue(name, out var value) ? long.Parse(value, CultureInfo.InvariantCulture) : -1;

    private static string Delta(Dictionary<string, string>? before, Dictionary<string, string> after, string name)
        => before is null ? Figure(after, name).ToString(CultureInfo.InvariantCulture)
            : (Figure(after, name) - Figure(before, name)).ToString(CultureInfo.InvariantCulture);
}
