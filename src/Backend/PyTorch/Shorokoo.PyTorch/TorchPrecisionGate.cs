namespace Shorokoo.PyTorch;

/// <summary>
/// Keeps torch runs on cards that set torch's TensorFloat-32 switches differently apart, and lets
/// runs that set them alike run beside one another: the switches are the whole process's, read as
/// each kernel is launched, so a run in one precision must not overlap a run in the other, and two
/// runs in one precision may.
///
/// <para>Any number of runs in one precision hold the gate at once. A run in the other waits until
/// they have all left; and while it waits, a run arriving in the precision held waits too, so
/// neither precision keeps the other out: as the last holder leaves, the turn passes to every run
/// waiting in the other precision, which enter together.</para>
/// </summary>
internal sealed class TorchPrecisionGate
{
    private readonly object _gate = new();

    // Per precision -- 0 full, 1 TensorFloat-32 -- the runs waiting to enter, and how many of them
    // the turn last handed to that precision lets in.
    private readonly int[] _waiting = new int[2];
    private readonly int[] _passes = new int[2];
    private int _held;
    private int _holders;

    /// <summary>Enters as a run in TensorFloat-32 where <paramref name="tensorFloat32"/>, and in full
    /// precision otherwise, waiting its turn.</summary>
    internal void Enter(bool tensorFloat32)
    {
        var precision = tensorFloat32 ? 1 : 0;
        lock (_gate)
        {
            if (!Joins(precision))
            {
                _waiting[precision]++;
                while (!(_passes[precision] > 0 && (_holders == 0 || _held == precision))) Monitor.Wait(_gate);
                _waiting[precision]--;
                _passes[precision]--;
            }
            _held = precision;
            _holders++;
        }
    }

    /// <summary>Enters as <see cref="Enter"/> does where that would not wait, and answers whether it
    /// entered.</summary>
    internal bool TryEnter(bool tensorFloat32)
    {
        var precision = tensorFloat32 ? 1 : 0;
        lock (_gate)
        {
            if (!Joins(precision)) return false;
            _held = precision;
            _holders++;
            return true;
        }
    }

    /// <summary>Leaves, as a run that entered in <paramref name="tensorFloat32"/>'s precision.</summary>
    internal void Exit(bool tensorFloat32)
    {
        var precision = tensorFloat32 ? 1 : 0;
        lock (_gate)
        {
            if (_holders == 0 || _held != precision)
                throw new InvalidOperationException("No run in that precision holds the gate.");
            if (--_holders > 0) return;
            // The last holder hands the turn on: to the rest of its own precision's turn where some of
            // those have yet to enter, else to the other precision's waiting runs, else to its own.
            var other = 1 - precision;
            if (_passes[precision] == 0)
            {
                if (_waiting[other] > 0) _passes[other] = _waiting[other];
                else _passes[precision] = _waiting[precision];
            }
            Monitor.PulseAll(_gate);
        }
    }

    /// <summary>How many runs in <paramref name="tensorFloat32"/>'s precision wait to enter.</summary>
    internal int Waiting(bool tensorFloat32)
    {
        lock (_gate) return _waiting[tensorFloat32 ? 1 : 0];
    }

    /// <summary>Whether a run in <paramref name="precision"/> arriving now enters at once: nothing
    /// in the other precision holds the gate, waits for it, or has the turn.</summary>
    private bool Joins(int precision)
    {
        var other = 1 - precision;
        return _waiting[other] == 0 && _passes[other] == 0 && (_holders == 0 || _held == precision);
    }
}
