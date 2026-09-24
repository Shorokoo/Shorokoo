using Shorokoo.Core.Backends;
using Shorokoo.PyTorch;
using Shorokoo.PyTorch.Cpu;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Utils;

/// <summary>
/// The PyTorch CPU backend's half of <see cref="QeeAudit"/>: an audit module that passed on ONNX
/// Runtime is run again on torch, and every output must agree — bit for bit, or within
/// <see cref="AutoTest.Tolerance"/> for a floating-point one.
///
/// <para><b>Which modules run.</b> Every one whose operators the backend translates, decided by the
/// backend itself: session creation refuses a model using an operator absent from its operator
/// table (<see cref="TorchUnsupportedReason.UnknownOperator"/>), and that refusal alone skips the
/// module. So adding an operator turns on every audit that was waiting for it, with nothing to
/// list here. Any other failure — an attribute the translation does not handle, a wrong value, an
/// exception in the run — fails the audit, so no module whose operators are all translated can be
/// skipped quietly.</para>
///
/// <para><b>Known disagreements.</b> A module whose operators are all translated but which torch
/// gets wrong for a reason not yet fixed is listed in <see cref="KnownDisagreements"/> with the
/// reason. It is still run, and must still disagree: once it agrees, the audit fails until the
/// entry is removed, so the list cannot go stale.</para>
/// </summary>
internal static class QeeAuditOnTorch
{
    private static readonly Lazy<TorchCpuBackend> Backend = new(() => new TorchCpuBackend());

    // An entry reads: [typeof(SomeAuditCheck)] = "what torch gets wrong, and the issue tracking it",
    private static readonly Dictionary<Type, string> KnownDisagreements = new()
    {
    };

    public static bool Agrees<TModule>(InternalComputationGraph model, TensorData[] inputs)
    {
        IData[] feeds = [.. inputs.Select(static t => (IData)t.Shared())];
        var reference = ComputeContext.Default.Execute(model, feeds).Select(p => p.ToTensorData()).ToArray();
        bool agrees;
        try
        {
            var onTorch = new ComputeContext(Backend.Value).Execute(model, feeds).Select(p => p.ToTensorData()).ToArray();
            agrees = reference.Length == onTorch.Length && reference.Zip(onTorch).All(pair => Same(pair.First, pair.Second));
        }
        catch (Exception ex) when (IsUntranslatedOperator(ex))
        {
            return true;
        }
        catch (Exception) when (KnownDisagreements.ContainsKey(typeof(TModule)))
        {
            agrees = false;
        }
        return agrees != KnownDisagreements.ContainsKey(typeof(TModule));
    }

    private static bool IsUntranslatedOperator(Exception? ex)
    {
        for (; ex is not null; ex = ex.InnerException)
            if (ex is TorchUnsupportedModelException { Reason: TorchUnsupportedReason.UnknownOperator })
                return true;
        return false;
    }

    private static bool Same(TensorData expected, TensorData actual)
    {
        if (expected.DType != actual.DType || !expected.Shape.Equals(actual.Shape)) return false;
        if (expected.DType.IsSameElementTypeAs(DType.Utf8)) return expected.Data.SequenceEqual(actual.Data);
        var want = Widened(expected);
        var got = Widened(actual);
        if (want is null || got is null) return expected.CopyRawMemory().AsSpan().SequenceEqual(actual.CopyRawMemory());
        return want.Zip(got).All(p => p.First.Equals(p.Second)
            || Math.Abs(p.First - p.Second) <= AutoTest.Tolerance * Math.Max(1.0, Math.Abs(p.First)));
    }

    private static double[]? Widened(TensorData data)
    {
        var raw = data.CopyRawMemory();
        if (data.DType == DType.Float32) return [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(raw).ToArray().Select(v => (double)v)];
        if (data.DType == DType.Float64) return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(raw).ToArray();
        if (data.DType == DType.Float16) return [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Float16>(raw).ToArray().Select(v => (double)(float)v)];
        if (data.DType == DType.BFloat16) return [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, BFloat16>(raw).ToArray().Select(v => (double)(float)v)];
        return null;
    }
}
