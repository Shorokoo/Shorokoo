using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory.IR;
using Shorokoo.OnnxRuntime;

namespace Shorokoo.Tests;

/// <summary>
/// Temporary probe for Shorokoo/Shorokoo#377 item 2: does ORT route a session's initializers
/// through the arena these statistics read? Two earlier CPU probes disagreed; this settles it by
/// reading the arena either side of session construction with a weight of a size nothing else
/// would produce.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
public class ArenaInitializerProbe
{
    private const int Rows = 1024;
    private const int Cols = 1024;
    private const long WeightBytes = (long)Rows * Cols * sizeof(float);   // 4 MiB exactly

    [Fact]
    public void ProbeWhetherInitializersLandInTheSessionArena()
    {
        Console.WriteLine($"weight = {WeightBytes:N0} bytes");

        Report("CPU", () => new SessionOptions(), () => null);

        try
        {
            using var probe = new SessionOptions();
            probe.AppendExecutionProvider_CUDA(0);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nCUDA unavailable in this deployment: {ex.GetType().Name}: {ex.Message}");
            return;
        }

        Report("CUDA", () =>
        {
            var o = new SessionOptions();
            o.AppendExecutionProvider_CUDA(0);
            return o;
        },
        () => new OrtMemoryInfo("Cuda", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default));
    }

    /// <summary><paramref name="makeInfo"/> returns null for "ORT's own default", which is a
    /// shared singleton this must not dispose — only an info built here is ours to release.
    /// </summary>
    private static void Report(
        string label, Func<SessionOptions> makeOptions, Func<OrtMemoryInfo?> makeInfo)
    {
        Console.WriteLine($"\n=== {label} ===");
        using var options = makeOptions();
        using var session = new InferenceSession(ModelWithInitializer(), options);

        var owned = makeInfo();
        try
        {
            using var allocator = new OrtAllocator(session, owned ?? OrtMemoryInfo.DefaultInstance);
            Print("at construction", OrtArenaStats.Read(allocator));

            var input = new float[Rows];
            using (var x = OrtValue.CreateTensorValueFromMemory(input, [1L, Rows]))
            using (var run = new RunOptions())
            using (session.Run(run, new Dictionary<string, OrtValue> { ["x"] = x }, session.OutputNames))
            {
            }

            Print("after one run", OrtArenaStats.Read(allocator));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  {ex.GetType().Name}: {ex.Message}");
        }
        finally
        {
            owned?.Dispose();
            GC.KeepAlive(session);
        }
    }

    private static void Print(string when, Shorokoo.Core.Backends.ArenaStatistics? s)
    {
        if (s is not { } a) { Console.WriteLine($"  {when}: null"); return; }
        Console.WriteLine(
            $"  {when,-16} InUse={a.InUseBytes,12:N0}  MaxInUse={a.MaxInUseBytes,12:N0}  " +
            $"Total={a.TotalAllocatedBytes,12:N0}  NumAllocs={a.AllocationCount,6}  " +
            $"MaxAlloc={a.MaxAllocSizeBytes,12:N0}  Reserves={a.ReserveCount,4}  " +
            $"Extensions={a.ArenaExtensionCount,4}");
    }

    /// <summary>MatMul of a [1, Rows] input against a [Rows, Cols] initializer — 4 MiB of weight
    /// and nothing else of any size, so an arena figure either side of construction is
    /// unambiguous.</summary>
    private static byte[] ModelWithInitializer()
    {
        static TypeProto FloatMatrix(long rows, long cols) => new()
        {
            TensorType = new TypeProto.Tensor
            {
                ElemType = (int)TensorProto.DataType.Float,
                Shape = new TensorShapeProto
                {
                    Dims =
                    {
                        new TensorShapeProto.Dimension { DimValue = rows },
                        new TensorShapeProto.Dimension { DimValue = cols },
                    },
                },
            },
        };

        var graph = new GraphProto { Name = "shorokoo_arena_initializer_probe" };
        graph.Inputs.Add(new ValueInfoProto { Name = "x", Type = FloatMatrix(1, Rows) });
        graph.Outputs.Add(new ValueInfoProto { Name = "y", Type = FloatMatrix(1, Cols) });
        graph.Initializers.Add(new TensorProto
        {
            Name = "W",
            Dims = [Rows, Cols],
            data_type = (int)TensorProto.DataType.Float,
            RawData = new byte[WeightBytes],
        });
        graph.Nodes.Add(new NodeProto { OpType = "MatMul", Inputs = { "x", "W" }, Outputs = { "y" } });

        var model = new ModelProto { IrVersion = 10, Graph = graph };
        model.OpsetImports.Add(new OperatorSetIdProto { Domain = "", Version = 21 });

        using var serialized = new MemoryStream();
        ProtoBuf.Serializer.Serialize(serialized, model);
        return serialized.ToArray();
    }
}
