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
    /// precision otherwise, waiting its turn — unless <paramref name="token"/> is cancelled first, or
    /// the thread is interrupted, either of which gives the wait up and lets the others on.</summary>
    /// <exception cref="OperationCanceledException"><paramref name="token"/> was cancelled while
    /// the run waited.</exception>
    internal void Enter(bool tensorFloat32, CancellationToken token = default)
    {
        var precision = tensorFloat32 ? 1 : 0;
        // Wakes the waiters as the token is cancelled. Disposed only once the lock is let go: disposing
        // waits out a callback, which may itself be waiting for the lock.
        using var wake = token.CanBeCanceled
            ? token.Register(static gate => ((TorchPrecisionGate)gate!).WakeAll(), this)
            : default;
        lock (_gate)
        {
            if (!Joins(precision))
            {
                _waiting[precision]++;
                var entering = false;
                try
                {
                    while (!(_passes[precision] > 0 && (_holders == 0 || _held == precision)))
                    {
                        token.ThrowIfCancellationRequested();
                        Monitor.Wait(_gate);
                    }
                    _passes[precision]--;
                    entering = true;
                }
                finally
                {
                    _waiting[precision]--;
                    if (!entering) GaveUp();
                }
            }
            _held = precision;
            _holders++;
        }
    }

    private void WakeAll()
    {
        lock (_gate) Monitor.PulseAll(_gate);
    }

    /// <summary>Settles the turn once a waiting run has gone without entering: no pass is left for a
    /// run that no longer waits, and where nothing of the other precision waits, the waiting runs of
    /// the one held join it.</summary>
    private void GaveUp()
    {
        for (int precision = 0; precision < 2; precision++) _passes[precision] = Math.Min(_passes[precision], _waiting[precision]);
        if (_holders == 0) HandOn(_held);
        else if (_waiting[1 - _held] == 0 && _passes[1 - _held] == 0) _passes[_held] = _waiting[_held];
        Monitor.PulseAll(_gate);
    }

    /// <summary>Hands the turn on, with nothing holding the gate: to the rest of the turn of the
    /// precision last held where some of those have yet to enter, else to the other precision's
    /// waiting runs, else to its own.</summary>
    private void HandOn(int last)
    {
        var other = 1 - last;
        if (_passes[last] > 0 || _passes[other] > 0) return;
        if (_waiting[other] > 0) _passes[other] = _waiting[other];
        else _passes[last] = _waiting[last];
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
            HandOn(precision);
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
