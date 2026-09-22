using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Backends;
using Shorokoo.OnnxRuntime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// The measurement behind <c>DeviceMemorySettings.ArenaExtend</c>'s default: what ORT's arena ends up
/// holding under each extend strategy, for a series of allocation sizes that settles and for ones
/// that do not. Manual: it measures the machine, so it is never part of the coverage suite, and it
/// prints a table rather than gating anything.
///
/// <para>Run with <c>dotnet test --filter "FullyQualifiedName~ArenaExtendStrategyProbeTests"
/// --logger "console;verbosity=detailed"</c>. Linux and glibc only — it reads the arena off
/// <c>mallinfo2</c>, and says so rather than failing anywhere else.</para>
///
/// <para>The CPU and CUDA arenas are the same <c>BFCArena</c> with the same strategy enum, so the
/// shape of the result carries over; the figures do not, and the ratio grows with the number of
/// distinct allocation sizes a step makes. The strategy reaches the CPU arena only through an
/// env-registered allocator, which is why this drives ORT directly rather than going through
/// <see cref="ComputeContext"/>. The figures this prints, and what they decide, are recorded once
/// — in <c>Documentation/inference.md</c> under "Device memory (GPU backends)" — rather than also
/// here, so that a machine disagreeing with them has only one place to correct.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
[Collection(SerialMeasurement.Name)]
public class ArenaExtendStrategyProbeTests
{
    private const int K = 512;
    private const long Uncapped = 3072L << 20;

    [StructLayout(LayoutKind.Sequential)]
    private struct MallInfo2
    {
        public nuint Arena, OrdBlks, SmBlks, HBlks, HBlkHd, UsmBlks, FsmBlks, UordBlks, FordBlks, KeepCost;
    }

    [DllImport("libc", EntryPoint = "mallinfo2")]
    private static extern MallInfo2 MallInfo2Get();

    /// <summary>Bytes glibc holds for native callers: in-use blocks of its main arena plus every
    /// mmapped one. ORT takes its arena regions through malloc and the .NET GC does not, so the
    /// managed heap stays out of the figure — but an ORT thread-pool thread allocating below the
    /// mmap threshold lands in a secondary glibc arena this cannot see, which is why the figures
    /// move by a MiB or two between runs.</summary>
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

    /// <summary>
    /// Bytes the arena holds after feeding <paramref name="rows"/> to one session, or null if the
    /// arena could not serve the workload within <paramref name="arenaLimit"/>. The baseline is
    /// taken after the session exists, so what is measured is the arena the runs grow and not the
    /// session that was going to be built either way.
    /// </summary>
    private static long? RunUnder(byte[] model, ArenaExtendStrategy strategy, long arenaLimit, int[] rows)
    {
        // OrtArenaCfg and OrtMemoryInfo are SafeHandles that ORT takes as bare IntPtrs, so a bare
        // local would be unreachable garbage for the whole of CreateAndRegisterAllocator.
        Assert.InRange(arenaLimit, 1, uint.MaxValue);
        using var memInfo = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
        using var cfg = new OrtArenaCfg(
            (uint)arenaLimit, strategy is ArenaExtendStrategy.NextPowerOfTwo ? 0 : 1, 1024 * 1024, -1);
        OrtEnv.Instance().CreateAndRegisterAllocator(memInfo, cfg);
        try
        {
            using var options = new SessionOptions();
            OrtBackend.Configure(options, ShorokooGraphOptimization.EnableBasic, ShorokooLogSeverity.Fatal);
            options.AddSessionConfigEntry("session.use_env_allocators", "1");
            using var session = new InferenceSession(model, options);
            var before = NativeHeldBytes();
            var wData = new float[K * K];
            foreach (var n in rows)
            {
                using var xv = OrtValue.CreateTensorValueFromMemory(new float[(long)n * K], [n, K]);
                using var wv = OrtValue.CreateTensorValueFromMemory(wData, [K, K]);
                var feeds = new Dictionary<string, OrtValue>
                {
                    [session.InputNames[0]] = xv,
                    [session.InputNames[1]] = wv,
                };
                using var runOptions = new RunOptions();
                using var results = session.Run(runOptions, feeds, session.OutputNames);
                GC.KeepAlive(feeds);
            }
            return NativeHeldBytes() - before;
        }
        catch (OnnxRuntimeException ex) when (ex.Message.Contains("BFCArena"))
        {
            return null;
        }
        finally
        {
            OrtEnv.Instance().UnregisterAllocator(memInfo);
        }
    }

    [Fact]
    public void ProbeWhatEachArenaExtendStrategyEndsUpHolding()
    {
        try
        {
            MallInfo2Get();
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            Console.WriteLine("mallinfo2 is unavailable here (glibc 2.33+ on Linux only); nothing measured.");
            return;
        }

        var model = ModelBytes();
        var rng = new Random(7);
        int[] sizes = [256, 512, 1024, 2048];
        int[] growing = [256, 512, 1024, 2048];

        (string Name, int[] Rows, long Limit)[] cases =
        [
            ("one shape, ten runs", [.. Enumerable.Repeat(1024, 10)], Uncapped),
            ("alternating 2048/512, twenty runs", [.. Enumerable.Range(0, 20).Select(i => i % 2 == 0 ? 2048 : 512)], Uncapped),
            ("largest first, then settled", [2048, 1024, 512, 256, .. Enumerable.Repeat(1024, 10)], Uncapped),
            ("shuffled from four sizes, twenty runs", [.. Enumerable.Range(0, 20).Select(_ => sizes[rng.Next(sizes.Length)])], Uncapped),
            ("growing, then settled", [.. growing, .. Enumerable.Repeat(2048, 10)], Uncapped),
            ("growing 256 to 2048", growing, Uncapped),
            ("growing 256 to 2048, 16 MiB arena", growing, 16L << 20),
        ];

        RunUnder(model, ArenaExtendStrategy.SameAsRequested, Uncapped, [256]);   // warm up

        foreach (var (name, rows, limit) in cases)
        {
            var same = RunUnder(model, ArenaExtendStrategy.SameAsRequested, limit, rows);
            var pow2 = RunUnder(model, ArenaExtendStrategy.NextPowerOfTwo, limit, rows);
            Console.WriteLine($"{name,-38} SameAsRequested {Show(same),-14} NextPowerOfTwo {Show(pow2)}");
        }

        static string Show(long? held) => held is { } bytes ? $"{bytes / (1024 * 1024)} MiB" : "does not fit";
    }
}
