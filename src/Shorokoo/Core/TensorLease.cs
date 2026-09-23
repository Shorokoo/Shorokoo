using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Shorokoo.Core.Backends;

namespace Shorokoo.Runtime
{
    /// <summary>
    /// What a run will hold in its compute context's memory outside the arena of the session that
    /// runs it, planned before it takes anything (<see cref="RunFeeds.Plan"/>).
    /// </summary>
    /// <param name="Attached">The bytes of the live tensors attached to the context there.</param>
    /// <param name="AttachedTensors">How many tensors those are.</param>
    /// <param name="Added">The bytes the run adds to them: what it reads there in place that the
    /// context has not counted, and the copies it makes there.</param>
    internal readonly record struct DevicePlan(long Attached, int AttachedTensors, long Added)
    {
        /// <summary>All of it: the discount the session's arena limit is cut from the budget by.</summary>
        internal long Outside => Attached + Added;
    }

    /// <summary>
    /// A run's reader lock on one tensor or sequence, taken by the compute context running it, from
    /// <c>ComputeContext.Lock</c>. While it is held the tensor cannot be deleted — a delete is
    /// refused, declined, or negotiated through <see cref="Eviction"/> — nor consumed by another
    /// run, and the lease holds the tensor itself, strongly, so nothing a run is reading can be
    /// collected under it either.
    ///
    /// <para>The one obligation a holder has is to listen for that signal on everything it has
    /// locked and to stop as soon as any of them is raised. A run discharges it by handing the
    /// linked signal to the backend as <c>RunSettings.CancellationToken</c>; a holder that ignores
    /// it is not unsafe, only slow — the delete then waits for the read to end on its own.</para>
    ///
    /// <para>Disposing is dropping the lock, and it is a count rather than a flag: the same
    /// tensor locked twice is released twice.</para>
    /// </summary>
    internal sealed class TensorLease : IDisposable
    {
        private readonly ComputeContext _holder;

        // Held strongly for as long as the lease is: a run keeps what it reads alive for the whole
        // run, so a tensor with a lock on it is never garbage.
        private readonly object _target;

        // Who is reading, as the tensor names the read to a run it has to refuse.
        private readonly object? _reader;
        private int _released;

        internal TensorLease(ComputeContext holder, TensorData tensor, CancellationToken eviction, object? reader)
        {
            _holder = holder;
            _target = tensor;
            _reader = reader;
            Eviction = eviction;
        }

        internal TensorLease(ComputeContext holder, TensorDataSequence sequence, object? reader)
        {
            _holder = holder;
            _target = sequence;
            _reader = reader;
        }

        /// <summary>Raised when something asks for the tensor to be deleted. Stop reading it and
        /// drop this lease: the deleter is waiting for exactly that. A sequence is never deleted
        /// that way, so its lease has no signal to raise.</summary>
        internal CancellationToken Eviction { get; }

        /// <summary>What this lease holds.</summary>
        internal object Target => _target;

        /// <summary>Drops the lock. The memory goes now if the delete it was waiting for has
        /// already been asked for. Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0) return;
            GC.SuppressFinalize(this);
            // The tensor's lock first, then the context's count, and neither under the other: the
            // two gates are never nested, so a delete waiting for this lease cannot end up behind
            // a gate the run is queued on.
            switch (_target)
            {
                case TensorData tensor: tensor.ReleaseReadLock(_reader); break;
                case TensorDataSequence sequence: sequence.ReleaseReadLock(_reader); break;
            }
            _holder.ReleaseLease(_target);
        }

#if DEBUG
        // Collecting a locked tensor is a bug: a run holds a strong reference to everything it
        // reads for its whole duration, through this lease, so the only way a locked tensor becomes
        // garbage is with its lease, undisposed. This is where that is observable. Debug builds only
        // -- the release build carries no finalizer on anything here for it.
        ~TensorLease()
        {
            if (Volatile.Read(ref _released) == 0)
                Debug.Fail(
                    $"A reader lock on {_target} was collected without being released: the run that "
                    + "took it never gave it back, and the tensor it locked became garbage while "
                    + "still locked.");
        }
#endif
    }

    /// <summary>
    /// A run, named the way a message about it names it — which graph, on which context — for a
    /// tensor it consumed to say who took it, and a tensor it is reading to say who holds it.
    /// Described on first use: most runs are never mentioned in a message.
    /// </summary>
    internal sealed class RunIdentity
    {
        private readonly Func<string> _describe;
        private string? _text;

        internal RunIdentity(Func<string> describe) => _describe = describe;

        /// <inheritdoc/>
        public override string ToString() => _text ??= _describe();
    }

    /// <summary>
    /// What one run does with its inputs, decided in one place for both run paths: which it reads
    /// and which it consumes, what it is fed for each, and what it gives up when it returns.
    ///
    /// <list type="bullet">
    /// <item><b>How each is fed.</b> A tensor or sequence fed as it is is consumed; one fed through
    /// <c>.Shared()</c> is read; one fed through <c>.TryConsume()</c> is consumed if no other run is
    /// reading it when this one starts, and read otherwise. The same one fed more than once in the
    /// call is held once and bound to every input it feeds: shared if any occurrence is, else
    /// consumed if any is bare, else tried.</item>
    /// <item><b>Held before anything is built.</b> A read takes a reader lock and attaches the
    /// running context; a consumption takes the tensor — it is dead from here, whatever the run
    /// then does. Everything is checked first, so a run refused because one of its feeds is dead
    /// or is being read by another run takes nothing at all.</item>
    /// <item><b>What the session is fed.</b> The tensor's own value where the run's backend can
    /// address it; otherwise a copy in memory it can. A read reuses the copy the tensor holds (and
    /// locks it, and attaches it); a consumption takes that copy, or makes one, and the tensor's own
    /// memory is released at once.</item>
    /// <item><b>Handed over.</b> What is consumed goes to the backend in the call
    /// (<see cref="IShorokooSession.RunConsuming(IReadOnlyDictionary{string, IShorokooTensorValue}, IReadOnlyCollection{IShorokooTensorValue}, IReadOnlyList{string}, IReadOnlySet{string}, RunSettings, out IReadOnlyList{string})"/>)
    /// and is the backend's from then on, on every path — to release, or to write an output into.
    /// Consumed memory that never reached that call is still this run's, and is released when the
    /// run gives up.</item>
    /// <item><b>Under a device-memory budget</b>, what the run will hold in the context's memory is
    /// planned before anything is taken (<see cref="Plan"/>), the session is chosen against it, and
    /// each copy the run then makes is admitted against that plan — so a run the budget cannot fit
    /// is refused having taken nothing.</item>
    /// </list>
    /// </summary>
    internal sealed class RunFeeds : IDisposable
    {
        // How many times a read goes back for a copy a concurrent write retired between its being
        // found and its being locked. Each round needs another write landing in that window.
        private const int CopyRetries = 8;

        private readonly ComputeContext _context;
        private readonly IShorokooBackend _backend;
        private readonly RunIdentity _run;
        private readonly List<TensorLease> _leases = [];

        // The memory the running backend computes in, which is what a budget on the context counts.
        private readonly MemorySpace _space;

        // What Prepare made of the inputs: one target per distinct tensor or sequence, by what it
        // feeds, and the one each input feeds.
        private IReadOnlyList<NamedModelParam>? _inputs;
        private List<Target>? _targets;
        private Dictionary<object, Target>? _bySubject;
        private Target[]? _targetOf;

        // Under a budget, once admitted: the arena limit of the session this run will use, the plan
        // it was chosen by, and the bytes of the copies the run has had to make that the plan did not
        // count.
        private bool _admitted;
        private long _arenaLimit;
        private DevicePlan _plan;
        private long _unplannedBytes;

        // Everything this run took, whatever it then does with the memory: released when the run
        // gives up unless the backend was handed it, or it was released already.
        private readonly List<TensorData> _takenTensors = [];
        private readonly List<TensorDataSequence> _takenSequences = [];

        // What the backend is handed: tensors consumed where they are, and the copies consumed in
        // the place of the ones that could not be -- and, by value, which tensor each value was, so
        // an output the backend wrote into one can be told whose memory it now lives in.
        private readonly List<TensorData> _handedTensors = [];
        private readonly List<TensorDataSequence> _handedSequences = [];
        private readonly List<IShorokooTensorValue> _consumed = [];
        private readonly Dictionary<IShorokooTensorValue, TensorData> _handedByValue =
            new(ReferenceEqualityComparer.Instance);

        private bool _handedOver;
        private int _held;

        /// <param name="context">The context running, which takes the locks and is attached to
        /// what the run reads.</param>
        /// <param name="backend">The backend the session runs on: what a feed has to be
        /// addressable by, and what builds a copy of one that is not.</param>
        /// <param name="run">The run, for the messages that name it.</param>
        internal RunFeeds(ComputeContext context, IShorokooBackend backend, RunIdentity run)
        {
            _context = context;
            _backend = backend;
            _run = run;
            _space = backend.MemorySpace;
            Budget = context.BudgetIn(_space);
        }

        /// <summary>The device-memory budget this run is under — its context's, where the memory
        /// its backend computes in is a device's — or null where there is none.</summary>
        internal long? Budget { get; }

        /// <summary>How many inputs are held, by a lock or by consumption.</summary>
        internal int Held => _held;

        /// <summary>The locks held, whose eviction signals the run listens for.</summary>
        internal IReadOnlyList<TensorLease> Leases => _leases;

        /// <summary>The values handed to the backend, each once.</summary>
        internal IReadOnlyCollection<IShorokooTensorValue> Consumed => _consumed;

        /// <summary>
        /// <see cref="Prepare"/> and then <see cref="Feed(Func{string, string})"/>, for a run with
        /// nothing to decide in between.
        /// </summary>
        internal Dictionary<string, IShorokooTensorValue> Feed(
            IReadOnlyList<NamedModelParam> inputs, Func<string, string> sessionNameOf)
        {
            Prepare(inputs);
            return Feed(sessionNameOf);
        }

        /// <summary>
        /// Works out what each input feeds — one target per distinct tensor or sequence, in the order
        /// they first appear — and refuses, before anything is taken, every feed that cannot be held.
        /// </summary>
        /// <exception cref="ObjectDisposedException">A feed is dead. Nothing was taken.</exception>
        /// <exception cref="InvalidOperationException">A feed to be consumed is being read by
        /// another run, or is of a kind nothing knows how to hold. Nothing was taken.</exception>
        internal void Prepare(IReadOnlyList<NamedModelParam> inputs)
        {
            ArgumentNullException.ThrowIfNull(inputs);

            var byFeed = new Dictionary<object, Target>(ReferenceEqualityComparer.Instance);
            var targets = new List<Target>();
            var targetOf = new Target[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var subject = SubjectOf(input);
                if (!byFeed.TryGetValue(subject, out var target))
                {
                    target = new Target(subject);
                    byFeed.Add(subject, target);
                    targets.Add(target);
                }
                target.Add(input.Described, input.Sharing);
                targetOf[i] = target;
            }

            // Everything that can refuse the run is asked before anything is taken, so a refused run
            // spends nothing. A feed can still die or be locked by another thread in between, which
            // is then a refusal part-way; what was taken by then stays taken, and is released when
            // the run gives up.
            foreach (var target in targets) target.RefuseIfCannotBeHeld(_run);

            _inputs = inputs;
            _targets = targets;
            _bySubject = byFeed;
            _targetOf = targetOf;
        }

        /// <summary>
        /// Holds every input <see cref="Prepare"/> was given and builds what the session is fed for
        /// each, keyed by the session's name for it (<paramref name="sessionNameOf"/>). See the class
        /// for the rules.
        /// </summary>
        /// <exception cref="ObjectDisposedException">A feed died after it was prepared.</exception>
        /// <exception cref="InvalidOperationException">A feed to be consumed came to be read by
        /// another run after it was prepared, or a copy the run had to make that its plan did not
        /// foresee would take its context past its device-memory budget.</exception>
        internal Dictionary<string, IShorokooTensorValue> Feed(Func<string, string> sessionNameOf)
        {
            var inputs = _inputs ?? throw new InvalidOperationException("A run was fed before it was prepared.");
            var targets = _targets!;
            var targetOf = _targetOf!;

            // Reads first, then consumptions: a lock refused because its tensor died in between is
            // then refused before this run has taken anything.
            foreach (var target in targets)
                if (target.Mode == FeedMode.Shared) Lock(target);
            foreach (var target in targets)
                if (target.Mode != FeedMode.Shared) TakeOrLock(target);

            for (int i = 0; i < inputs.Count; i++)
            {
                inputs[i].Held();
                _held++;
            }

            foreach (var target in targets) target.Value = Build(target);

            var fed = new Dictionary<string, IShorokooTensorValue>(inputs.Count);
            for (int i = 0; i < inputs.Count; i++)
                fed[sessionNameOf(inputs[i].ParamName)] = targetOf[i].Value!;
            return fed;
        }

        /// <summary>
        /// What this run will hold in its context's memory for as long as it runs, outside the arena
        /// of the session that runs it — worked out before anything is taken, so the session can be
        /// chosen, and the run refused, while nothing has been spent.
        ///
        /// <para>That is every live tensor attached to the context there, and what the run adds to
        /// it: each tensor it is fed that it reads in place there and the context has not yet
        /// counted, and each copy it will have to make there of one it cannot read where it is —
        /// or the copy already held, where one is. A sequence is read through the host and adds
        /// nothing. What is in <paramref name="excludingArena"/> — the arena of the session about to
        /// run, where its own earlier runs left their outputs — is inside that session's limit
        /// already, and is left out.</para>
        /// </summary>
        internal DevicePlan Plan(object? excludingArena)
        {
            var targets = _targets ?? throw new InvalidOperationException("A run was planned before it was prepared.");
            var counted = new HashSet<TensorData>(ReferenceEqualityComparer.Instance);
            long attached = 0;
            var attachedTensors = 0;
            foreach (var tensor in _context.AttachedTensorsIn(_space, excludingArena))
            {
                counted.Add(tensor);
                attached += tensor.ByteCount;
                attachedTensors++;
            }

            long added = 0;
            foreach (var target in targets)
            {
                target.PlannedFresh = false;
                target.Copy = null;
                // A sequence's value is built in host memory whatever the provider.
                if (target.Subject is not TensorData tensor) continue;
                TensorData? resident = tensor;
                if (!tensor.FeedsInPlace(_backend))
                {
                    var where = TensorData.RunMemoryOf(_backend, tensor.DType);
                    if (where.Space != _space) continue;
                    // Read through the copy the tensor holds there, or through a fresh one.
                    resident = target.Copy = tensor.CopyHeldAt(where);
                    if (resident is null)
                    {
                        target.PlannedFresh = true;
                        added += tensor.ByteCount;
                        continue;
                    }
                }
                if (resident.Space != _space) continue;
                if (excludingArena is not null && ReferenceEquals(resident.Arena, excludingArena)) continue;
                if (counted.Add(resident)) added += resident.ByteCount;
            }
            return new DevicePlan(attached, attachedTensors, added);
        }

        /// <summary>
        /// Records the session this run was admitted to — its arena limit, and the plan it was
        /// chosen by — so each copy the run then makes in the context's memory is held to it.
        /// </summary>
        internal void Admit(long arenaLimit, DevicePlan plan)
        {
            _admitted = true;
            _arenaLimit = arenaLimit;
            _plan = plan;
        }

        /// <summary>
        /// The refusal of a run when what it would hold in its context's memory leaves nothing of the
        /// context's budget for the arena it computes in. Thrown before anything is taken.
        /// </summary>
        internal InvalidOperationException NoRoom(long limit, DevicePlan plan)
            => new(
                $"{_run} would hold {Figure(plan.Outside)} bytes of {_space} outside its own arena — "
                + $"{Figure(plan.Attached)} bytes of the {Figure(plan.AttachedTensors)} tensor(s) "
                + $"attached to its compute context there, and {Figure(plan.Added)} bytes more that it "
                + "reads there or copies there to read — which leaves nothing of the context's "
                + $"{Figure(limit)}-byte device-memory budget (DeviceMemorySettings.LimitBytes) for the "
                + "arena the run computes in. Nothing it was fed has been taken. Delete what the "
                + "context no longer needs, feed less at once, or give the context a larger budget.");

        private static string Figure(long value) => ComputeContext.Figure(value);

        /// <summary>
        /// Admits a copy of <paramref name="source"/> this run is about to make in its context's
        /// memory, against the plan its session was chosen by.
        ///
        /// <para>A copy the plan counted always fits. So does one made in place of the copy the run
        /// meant to read — retired by a write, or taken, before the run held it — whose memory has
        /// gone back: the new one only takes its room. Where that memory is still held, because
        /// another run is reading the retired copy, both are on the card at once, which the plan did
        /// not count; that is refused where it would leave the session's arena less than its
        /// limit.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">The copy does not fit.</exception>
        private void AdmitCopy(TensorData source)
        {
            if (!_admitted || Budget is not { } limit) return;
            if (TensorData.RunMemoryOf(_backend, source.DType).Space != _space) return;
            if (_bySubject is null || !_bySubject.TryGetValue(source, out var target)) return;
            if (target.PlannedFresh)
            {
                target.PlannedFresh = false;
                return;
            }
            if (target.Copy is { IsDisposed: true, IsLocked: false }) return;

            var bytes = source.ByteCount;
            _unplannedBytes += bytes;
            var outside = _plan.Outside + _unplannedBytes;
            if (outside <= limit - _arenaLimit) return;
            throw new InvalidOperationException(
                $"{_run} has to copy {source.Describe()} into {_space} to read it again — "
                + $"{Figure(bytes)} bytes — while the copy it meant to read, retired before this run "
                + $"held it, is still held by another run: with both it would hold {Figure(outside)} bytes "
                + $"there outside its own arena, which with the {Figure(_arenaLimit)} bytes its "
                + $"session's arena may take is more than its compute context's {Figure(limit)}-byte "
                + "device-memory budget (DeviceMemorySettings.LimitBytes). What it had taken by then "
                + "stays taken. Delete what the context no longer needs, or give the context a larger "
                + "budget.");
        }

        /// <summary>
        /// Calls into the backend with what this run consumed handed over. From the moment
        /// <paramref name="run"/> is called that memory is the backend's alone, however it returns:
        /// the tensors it was are marked so, and nothing here touches it again.
        /// </summary>
        internal T HandOver<T>(Func<IReadOnlyCollection<IShorokooTensorValue>, T> run)
        {
            _handedOver = true;
            try
            {
                return run(_consumed);
            }
            finally
            {
                foreach (var tensor in _handedTensors) tensor.HandedToBackend();
                foreach (var sequence in _handedSequences) sequence.HandedToBackend();
            }
        }

        /// <summary>
        /// Gives up everything held, however the run ended: every lock dropped — each copy's before
        /// the tensor's it was made of, so a run that finds a tensor free finds its copies free too —
        /// and whatever the run took and did not hand to the backend released.
        /// </summary>
        public void Dispose()
        {
            for (int i = _leases.Count - 1; i >= 0; i--) _leases[i].Dispose();
            // A release is once only, and does nothing for what was handed over, so these can be
            // asked of everything.
            foreach (var tensor in _handedTensors) tensor.ReleaseTaken();
            foreach (var sequence in _handedSequences) sequence.ReleaseTaken();
            foreach (var tensor in _takenTensors) tensor.ReleaseTaken();
            foreach (var sequence in _takenSequences) sequence.ReleaseTaken();
            Debug.Assert(_handedOver || _consumed.Count == 0 || _handedTensors.Count + _handedSequences.Count > 0);
        }

        /// <summary>
        /// The tensor or sequence <paramref name="input"/> feeds. An absent optional feeds nothing,
        /// and a session cannot be fed one: its own refusal, naming the engine that can, is thrown
        /// here, before anything is taken.
        /// </summary>
        private object SubjectOf(NamedModelParam input) => input switch
        {
            TensorDataModelParam tensor => tensor.ToTensorData(),
            OptionalTensorDataModelParam { Data: { HasValue: true, Value: { } present } } => present,
            OptionalTensorDataModelParam absent => RefuseAbsent(absent),
            TensorDataSequenceModelParam sequence => sequence.ToTensorDataSequence(),
            _ => throw new InvalidOperationException(
                $"Input '{input.ParamName}' was fed to a run as a {input.GetType().Name}, which nothing "
                + "knows how to hold. Every fed input has to be held for the length of the run, or "
                + "deleting it from another thread frees what the run is reading."),
        };

        private object RefuseAbsent(OptionalTensorDataModelParam absent)
        {
            absent.ToTensorValue(_backend);
            throw new InvalidOperationException($"Input '{absent.ParamName}' is an absent optional.");
        }

        /// <summary>Takes a reader lock on the target for this run, attaching the running
        /// context.</summary>
        private void Lock(Target target)
        {
            _leases.Add(target.Subject switch
            {
                TensorData tensor => _context.Lock(tensor, _run),
                TensorDataSequence sequence => _context.Lock(sequence, _run),
                _ => throw new UnreachableException(),
            });
        }

        /// <summary>
        /// Takes the target for this run: consumed, if nothing is reading it — and refused if
        /// something is, unless it was only to be tried, in which case it is read instead.
        /// </summary>
        private void TakeOrLock(Target target)
        {
            var death = TensorDeath.ConsumedBy(_run, target.Inputs, tried: target.Mode == FeedMode.TryConsume);
            var outcome = target.Subject switch
            {
                TensorData tensor => tensor.TryTake(death),
                TensorDataSequence sequence => sequence.TryTake(death),
                _ => throw new UnreachableException(),
            };
            switch (outcome)
            {
                case TakeOutcome.Taken:
                    target.Consumed = true;
                    target.Death = death;
                    if (target.Subject is TensorData taken) _takenTensors.Add(taken);
                    else _takenSequences.Add((TensorDataSequence)target.Subject);
                    return;
                case TakeOutcome.Locked when target.Mode == FeedMode.TryConsume:
                    Lock(target);
                    return;
                case TakeOutcome.Locked:
                    throw target.BeingRead(_run);
                default:
                    throw target.Refusal();
            }
        }

        /// <summary>The value the session is fed for <paramref name="target"/>, which this run now
        /// holds.</summary>
        private IShorokooTensorValue Build(Target target)
            => target.Subject switch
            {
                TensorData tensor when target.Consumed => Consume(target, tensor, target.Death!),
                TensorData tensor => Read(target, tensor),
                TensorDataSequence sequence when target.Consumed => Consume(sequence, target.Death!),
                TensorDataSequence sequence => Read(sequence),
                _ => throw new UnreachableException(),
            };

        private IShorokooTensorValue Consume(Target target, TensorData tensor, TensorDeath death)
        {
            if (tensor.FeedsInPlace(_backend))
                return Hand(tensor, tensor.UncheckedValue(_backend));

            // Incompatible memory: the contents go into the run's memory, the tensor is spent, and
            // the copy is what is consumed. Its own memory is released now rather than after the
            // run, which is the point of consuming it.
            var copy = target.Copy = tensor.TakeRunCopy(_backend, AdmitCopy, death);
            var value = Hand(copy, copy.UncheckedValue(_backend));
            tensor.ReleaseTaken();
            return value;
        }

        private IShorokooTensorValue Hand(TensorData tensor, IShorokooTensorValue value)
        {
            _handedTensors.Add(tensor);
            _consumed.Add(value);
            _handedByValue[value] = tensor;
            return value;
        }

        /// <summary>
        /// The arena the memory behind <paramref name="value"/> was in, where it is a value this run
        /// handed to the backend — the record of the tensor it was, consumed where it stood or copied
        /// for the run — or null where it was in no arena, or is not one this run handed over.
        /// What an output the backend wrote into that memory is in.
        /// </summary>
        internal object? ArenaOfHanded(IShorokooTensorValue value)
            => _handedByValue.TryGetValue(value, out var tensor) ? tensor.Arena : null;

        private IShorokooTensorValue Read(Target target, TensorData tensor)
        {
            if (tensor.FeedsInPlace(_backend)) return tensor.UncheckedValue(_backend);

            // Incompatible memory, read: the copy the tensor holds in the run's memory, made on the
            // first such read. Locked for the run like the tensor, and attached to the context that
            // reads it, which is what it now occupies memory for.
            for (int attempt = 0; ; attempt++)
            {
                var copy = target.Copy = tensor.SharedCopyFor(_backend, AdmitCopy);
                try
                {
                    _leases.Add(_context.Lock(copy, _run));
                    return copy.UncheckedValue(_backend);
                }
                catch (ObjectDisposedException) when (copy.IsDisposed && attempt < CopyRetries)
                {
                    // Retired by a write between being found and being locked: make the next one.
                }
            }
        }

        private IShorokooTensorValue Consume(TensorDataSequence sequence, TensorDeath death)
        {
            if (sequence.FeedsInPlace(_backend))
            {
                _handedSequences.Add(sequence);
                _consumed.Add(sequence.UncheckedValue);
                return sequence.UncheckedValue;
            }

            var copy = sequence.TakeRunCopy(_backend, death);
            _handedSequences.Add(copy);
            _consumed.Add(copy.UncheckedValue);
            sequence.ReleaseTaken();
            return copy.UncheckedValue;
        }

        private IShorokooTensorValue Read(TensorDataSequence sequence)
        {
            if (sequence.FeedsInPlace(_backend)) return sequence.UncheckedValue;

            for (int attempt = 0; ; attempt++)
            {
                var copy = sequence.SharedCopyFor(_backend);
                try
                {
                    _leases.Add(_context.Lock(copy, _run));
                    return copy.UncheckedValue;
                }
                catch (ObjectDisposedException) when (copy.IsDisposed && attempt < CopyRetries)
                {
                }
            }
        }

        /// <summary>How a run treats one feed.</summary>
        private enum FeedMode
        {
            Consume,
            Shared,
            TryConsume,
        }

        /// <summary>One tensor or sequence this run is fed, however many inputs it feeds.</summary>
        private sealed class Target(object subject)
        {
            private readonly List<string> _names = [];
            private bool _anyShared;
            private bool _anyBare;

            internal object Subject { get; } = subject;

            internal bool Consumed { get; set; }

            internal TensorDeath? Death { get; set; }

            internal IShorokooTensorValue? Value { get; set; }

            /// <summary>Under a budget: whether the plan counted a fresh copy of it, not yet
            /// made.</summary>
            internal bool PlannedFresh { get; set; }

            /// <summary>The copy this run reads it through, as far as the run knows: the one it held
            /// when the run was planned, and then the one each read found or made.</summary>
            internal TensorData? Copy { get; set; }

            /// <summary>What the run does with it, over every occurrence: shared if any occurrence
            /// is, else consumed if any is bare, else tried.</summary>
            internal FeedMode Mode
                => _anyShared ? FeedMode.Shared : _anyBare ? FeedMode.Consume : FeedMode.TryConsume;

            /// <summary>The input or inputs it feeds, as a message names them.</summary>
            internal string Inputs
                => _names.Count == 1
                    ? _names[0]
                    : $"{string.Join(", ", _names.Take(_names.Count - 1))} and {_names[^1]}";

            internal void Add(string described, SharedInputMode? sharing)
            {
                if (!_names.Contains(described)) _names.Add(described);
                switch (sharing)
                {
                    case null: _anyBare = true; break;
                    case SharedInputMode.Shared: _anyShared = true; break;
                }
            }

            /// <summary>Refuses, before anything is taken, a feed that is dead — nothing can be read
            /// or consumed — or one to be consumed that another run is reading.</summary>
            internal void RefuseIfCannotBeHeld(RunIdentity run)
            {
                switch (Subject)
                {
                    case TensorData { IsDisposed: true }:
                    case TensorDataSequence { IsDisposed: true }:
                        throw Refusal();
                    case TensorData tensor when Mode == FeedMode.Consume && tensor.IsLocked:
                    case TensorDataSequence sequence when Mode == FeedMode.Consume && sequence.IsLocked:
                        throw BeingRead(run);
                }
            }

            /// <summary>What an access to it throws now that it is dead.</summary>
            internal Exception Refusal() => Subject switch
            {
                TensorData tensor => tensor.Death!.Refusal(tensor, tensor.Describe()),
                TensorDataSequence sequence => sequence.Death!.Refusal(sequence, sequence.Describe()),
                _ => throw new UnreachableException(),
            };

            /// <summary>
            /// The refusal of a consumption that would take memory another run is reading, naming
            /// that run where it said who it was, and the ways around it.
            /// </summary>
            internal InvalidOperationException BeingRead(RunIdentity run)
            {
                var (what, reader) = Subject switch
                {
                    TensorData tensor => (tensor.Describe(), tensor.DescribeReader()),
                    TensorDataSequence sequence => (sequence.Describe(), sequence.DescribeReader()),
                    _ => throw new UnreachableException(),
                };
                return new InvalidOperationException(
                    $"{what} is being read by {reader ?? "another run"}, so {run} cannot consume it as "
                    + $"{Inputs}: fed as it is, it is given to the run it feeds, which would take its "
                    + "memory from under the one reading it. Pass it as .Shared() to read it alongside "
                    + "that run -- or pass the struct, sequence or checkpoint holding it that way -- "
                    + "or as .TryConsume() to have it consumed only when nothing else is reading it; "
                    + "or wait for the other run to return.");
            }
        }
    }
}
