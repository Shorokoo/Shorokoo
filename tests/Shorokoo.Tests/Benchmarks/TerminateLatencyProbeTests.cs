using System.Diagnostics;
using Shorokoo.Core.Inference.Abstractions;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Benchmarks;

/// <summary>
/// The measurement behind what <see cref="RunSettings.CancellationToken"/> is worth: how long a
/// run takes to return once its token is cancelled, and whether it stops at all. ORT reads its
/// terminate flag between nodes, so the wait is whatever is left of the kernel in flight — and a
/// graph whose one node is a large matmul has no boundary to stop at, which is the case this
/// drives alongside the chains. Manual: it measures the machine, so it is never part of the
/// coverage suite, and it prints a table rather than gating anything.
///
/// <para>Run with <c>dotnet test --filter "FullyQualifiedName~TerminateLatencyProbeTests"
/// --logger "console;verbosity=detailed"</c>.</para>
///
/// <para>Each case is a chain of <c>kernels</c> square matmuls, which separates the two questions:
/// chains of the same total length but different kernel sizes say whether the wait tracks one
/// kernel or the whole run, and a one-kernel chain says what happens when there is no boundary.
/// <c>run took</c> against <c>uninterrupted</c> is the reading that matters — a terminated run is
/// one that came back early. The figures this prints, and what they decide, are recorded once —
/// in <c>Documentation/inference.md</c> — rather than also here, so that a machine disagreeing
/// with them has only one place to correct.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Manual")]
[Collection(SerialMeasurement.Name)]
public class TerminateLatencyProbeTests
{
    private sealed record Chain(ComputeContext Context, CompiledGraph Compiled, TensorData X, TensorData W)
        : IDisposable
    {
        public void Dispose() => Context.Dispose();
    }

    private static Chain Build(int size, int kernels)
    {
        var x = InputTensor<float32>("x", rank: 2);
        var w = InputTensor<float32>("w", rank: 2);
        var y = x;
        for (int i = 0; i < kernels; i++) y = y.MatMul(w);

        var context = new ComputeContext();
        var compiled = context.Compile(new InternalComputationGraph([x, w], [y]));
        var values = new float[(long)size * size];
        for (int i = 0; i < values.Length; i++) values[i] = 1f / (1 + (i % 97));
        return new Chain(
            context, compiled, TensorData([size, size], values), TensorData([size, size], values));
    }

    /// <summary>The shortest of <paramref name="samples"/> undisturbed runs, after one to warm
    /// up. The shortest rather than the mean: every disturbance this machine offers makes a run
    /// longer, so the floor is the reading that is about the work.</summary>
    private static TimeSpan Uninterrupted(Chain chain, int samples)
    {
        var best = TimeSpan.MaxValue;
        for (int i = 0; i <= samples; i++)
        {
            var started = Stopwatch.GetTimestamp();
            chain.Compiled.Execute([chain.X, chain.W]);
            var took = Stopwatch.GetElapsedTime(started);
            if (i > 0 && took < best) best = took;
        }
        return best;
    }

    /// <summary>Cancels <paramref name="delay"/> into a run and reports how long the run took in
    /// all, how long it took to come back after the flag was set, and how it ended.</summary>
    private static (TimeSpan Took, TimeSpan? AfterFlag, string Ending) CancelDuringARun(
        Chain chain, TimeSpan delay)
    {
        using var cts = new CancellationTokenSource();
        var settings = new RunSettings { CancellationToken = cts.Token };
        var finished = new ManualResetEventSlim();
        var ending = "ran to completion";

        var started = Stopwatch.GetTimestamp();
        var run = Task.Run(() =>
        {
            try { chain.Compiled.Execute([chain.X, chain.W], settings); }
            catch (OperationCanceledException) { ending = "stopped"; }
            catch (Exception e) { ending = e.GetType().Name; }
            finished.Set();
        });

        if (finished.Wait(delay))
        {
            run.GetAwaiter().GetResult();
            return (Stopwatch.GetElapsedTime(started), null, ending + " before the flag");
        }

        var flagged = Stopwatch.GetTimestamp();
        cts.Cancel();
        run.GetAwaiter().GetResult();
        return (Stopwatch.GetElapsedTime(started), Stopwatch.GetElapsedTime(flagged), ending);
    }

    [Fact]
    public void ProbeHowLongARunTakesToReturnAfterItsTokenIsCancelled()
    {
        (int Size, int Kernels)[] cases =
        [
            (4096, 1),
            (2048, 1),
            (4096, 3),
            (2048, 8),
            (1024, 32),
            (512, 128),
        ];
        double[] fractions = [0.1, 0.5, 0.9];

        foreach (var (size, kernels) in cases)
        {
            using var chain = Build(size, kernels);
            var total = Uninterrupted(chain, samples: 3);
            var perKernel = total / kernels;

            foreach (var fraction in fractions)
            {
                var (took, afterFlag, ending) = CancelDuringARun(chain, total * fraction);
                Console.WriteLine(
                    $"{size}^3 x {kernels,-4} uninterrupted {Ms(total),9}  kernel {Ms(perKernel),9}  "
                    + $"flag at {Ms(total * fraction),9}  run took {Ms(took),9}  "
                    + $"back after {(afterFlag is { } l ? Ms(l) : "-"),9}  {ending}");
            }
        }

        static string Ms(TimeSpan t) => $"{t.TotalMilliseconds:F0} ms";
    }
}
