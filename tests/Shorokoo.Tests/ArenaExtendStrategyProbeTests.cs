using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.OnnxRuntime;

namespace Shorokoo.Tests;

/// <summary>
/// The measurement behind <see cref="ArenaExtendStrategy.Auto"/>: how much memory ORT's arena
/// ends up holding under each extend strategy, for a session whose allocation sizes settle and
/// for one whose sizes keep growing. A developer workflow rather than a check — it reads
/// process-wide native allocation, which the parallel coverage suite would drown — so it is
/// <c>Purpose=Manual</c> and prints a table:
///
/// <code>
/// dotnet test tests/Shorokoo.Tests/Shorokoo.Tests.csproj \
///   --filter "FullyQualifiedName~ArenaExtendStrategyProbeTests" \
///   --logger "console;verbosity=detailed" --no-build
/// </code>
///
/// Recorded on ORT 1.26, CPU EP, 4 chained [rows, 512] x [512, 512] matmuls, arena uncapped:
///
/// | workload | SameAsRequested | NextPowerOfTwo |
/// |---|---|---|
/// | ten runs at one shape | 11 MiB | 16 MiB |
/// | four runs, each shape larger | 23 MiB | 15 MiB |
///
/// Neither strategy wins outright, which is why the choice is made per session. Exact-size
/// extension tracks a settled series of sizes instead of doubling past it, and strands every
/// region it outgrows when the sizes keep climbing — at a 16 MiB cap the growing workload fails
/// outright under it while ORT's doubling runs in 15 MiB. A training step is the settled case:
/// its shapes are fixed when the step is compiled and repeat for the life of the run.
///
/// <para>The CPU and CUDA arenas are the same <c>BFCArena</c> with the same strategy enum, so the
/// shape of the result carries over; the figures do not, and the ratio grows with the number of
/// distinct allocation sizes a step makes. The strategy reaches the CPU arena only through an
/// env-registered allocator, which is why this drives ORT directly rather than going through
/// <see cref="ComputeContext"/>.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
public class ArenaExtendStrategyProbeTests
{
    [StructLayout(LayoutKind.Sequential)]
    private struct MallInfo2
    {
        public nuint Arena, OrdBlks, SmBlks, HBlks, HBlkHd, UsmBlks, FsmBlks, UordBlks, FordBlks, KeepCost;
    }

    [DllImport("libc", EntryPoint = "mallinfo2")]
    private static extern MallInfo2 MallInfo2Get();

    /// <summary>Bytes glibc holds for native callers: in-use blocks plus mmapped ones. ORT takes
    /// its arena regions through malloc; the .NET GC does not, so the managed heap stays out.</summary>
    private static long NativeHeldBytes()
    {
        var m = MallInfo2Get();
        return (long)m.UordBlks + (long)m.HBlkHd;
    }

    private static byte[] ModelBytes()
    {
        var x = InputTensor<float32>("x", rank: 2);
        var w = InputTensor<float32>("w", rank: 2);
        var a = x.MatMul(w);
        var b = (a * a).MatMul(w);
        var c = (b + a).MatMul(w);
        var d = (c * b).MatMul(w);
        var proto = FastOnnxModelBuilder.BuildInternalOnnxModel(
            new InternalComputationGraph([x, w], [d]), prepForOnnx: true);
        var stream = new MemoryStream();
        ProtoBuf.Serializer.Serialize(stream, proto);
        return stream.ToArray();
    }

    private static string RunUnder(byte[] model, ArenaExtendStrategy strategy, long arenaLimit, int[] rows, int k)
    {
        var memInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        var cfg = new OrtArenaCfg(
            (uint)arenaLimit, strategy is ArenaExtendStrategy.NextPowerOfTwo ? 0 : 1, 1024 * 1024, -1);
        OrtEnv.Instance().CreateAndRegisterAllocator(memInfo, cfg);
        var before = NativeHeldBytes();
        long held;
        try
        {
            using var options = new SessionOptions();
            OrtSessionFactory.Configure(options, ShorokooGraphOptimization.EnableBasic, ShorokooLogSeverity.Fatal);
            options.AddSessionConfigEntry("session.use_env_allocators", "1");
            using var session = new InferenceSession(model, options);
            var wData = new float[k * k];
            foreach (var n in rows)
            {
                using var xv = OrtValue.CreateTensorValueFromMemory(new float[(long)n * k], [n, k]);
                using var wv = OrtValue.CreateTensorValueFromMemory(wData, [k, k]);
                var feeds = new Dictionary<string, OrtValue>
                {
                    [session.InputNames[0]] = xv,
                    [session.InputNames[1]] = wv,
                };
                using var runOptions = new RunOptions();
                using var results = session.Run(runOptions, feeds, session.OutputNames);
                GC.KeepAlive(feeds);
            }
            held = NativeHeldBytes() - before;
        }
        catch (OnnxRuntimeException)
        {
            return "does not fit";
        }
        finally
        {
            OrtEnv.Instance().UnregisterAllocator(memInfo);
        }
        return $"{held / (1024 * 1024)} MiB";
    }

    [Fact]
    public void ProbeWhatEachArenaExtendStrategyEndsUpHolding()
    {
        const int k = 512;
        const long uncapped = 3072L << 20;
        var model = ModelBytes();
        int[] settled = [.. Enumerable.Repeat(1024, 10)];
        int[] growing = [256, 512, 1024, 2048];

        RunUnder(model, ArenaExtendStrategy.SameAsRequested, uncapped, [256], k);   // warm up

        foreach (var (name, rows, limit) in new (string, int[], long)[]
                 {
                     ("settled sizes, uncapped", settled, uncapped),
                     ("growing sizes, uncapped", growing, uncapped),
                     ("growing sizes, 16 MiB arena", growing, 16L << 20),
                 })
            foreach (var strategy in new[] { ArenaExtendStrategy.SameAsRequested, ArenaExtendStrategy.NextPowerOfTwo })
                Console.WriteLine($"{name,-28} {strategy,-16} {RunUnder(model, strategy, limit, rows, k)}");
    }
}
