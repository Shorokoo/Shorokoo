using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.ExceptionServices;
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
    /// <param name="InArena">The bytes of what it was fed that the runtime copies into the arena
    /// itself — at least that much of the arena the run needs.</param>
    internal readonly record struct DevicePlan(long Attached, int AttachedTensors, long Added, long InArena = 0)
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
        private readonly ILifetimeOwner _target;

        // Who is reading, as the tensor names the read to a run it has to refuse.
        private readonly object? _reader;
        private int _released;

        internal TensorLease(ComputeContext holder, ILifetimeOwner target, CancellationToken eviction, object? reader)
        {
            _holder = holder;
            _target = target;
            _reader = reader;
            Eviction = eviction;
        }

        /// <summary>Raised when something asks for the tensor to be deleted. Stop reading it and
        /// drop this lease: the deleter is waiting for exactly that. A sequence is never deleted
        /// that way, so its signal is never raised.</summary>
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
            // a gate the run is queued on. The count is dropped whatever the release does: a
            // release that throws has still let go of the lock, and a count left behind would
            // refuse the context's disposal for good.
            try
            {
                _target.Life.ReleaseReadLock(_reader);
            }
            finally
            {
                _holder.ReleaseLease(_target);
            }
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
    /// consumed if any is bare, else tried. A list sequence's own elements are held with it, as it
    /// is fed — the same tensor fed on its own as well counts as one more occurrence — so an
    /// element another run is reading refuses the sequence's consumption as it would its own, and
    /// one this run reads outlives the sequence it consumes.</item>
    /// <item><b>Held before anything is built.</b> A read takes a reader lock and attaches the
    /// running context; a consumption takes the tensor — it is dead from here, whatever the run
    /// then does. Everything is checked first, so a run refused because one of its feeds is dead
    /// or is being read by another run takes nothing at all.</item>
    /// <item><b>What the session is fed.</b> The tensor's own value where the run's backend can
    /// address it. Otherwise a read reuses the copy the tensor holds in the run's memory (and locks
    /// it, and attaches it), or makes one. A consumption takes that copy where nothing else is
    /// reading it; failing that, on a device whose run no output of may be written into the tensor
    /// (<see cref="WrittenInto"/>), it goes to the session in host memory — as it is, where it is a
    /// host value of the running runtime already — for the runtime to copy into the arena it
    /// computes in; and otherwise into a fresh copy in the run's memory. The tensor's own memory is
    /// released as soon as every value the run is fed has been built.</item>
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

        // What Prepare made of the inputs: one target per distinct tensor or sequence -- those an
        // input feeds, and the elements of a list sequence one feeds -- and the one each input feeds.
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
        private readonly List<ILifetimeOwner> _taken = [];

        // What the backend is handed, by value: the tensors and sequences consumed where they are, and
        // the copies consumed in the place of the ones that could not be -- each once, and each the one
        // its value was, so an output the backend wrote into a value can be told whose memory it now
        // lives in.
        private readonly Dictionary<IShorokooTensorValue, ILifetimeOwner> _handed = new(ReferenceEqualityComparer.Instance);

        // What marking the handed-over memory as the backend's threw -- retiring a copy of it that a
        // backend failed to release -- kept for the run's end rather than thrown in its place.
        private Exception? _handOverFailure;

        /// <param name="context">The context running, which takes the locks and is attached to
        /// what the run reads.</param>
        /// <param name="backend">The backend the session runs on: what a feed has to be
        /// addressable by, and what builds a copy of one that is not.</param>
        /// <param name="run">The run, for the messages that name it.</param>
        /// <param name="budget">The context's device-memory budget as the run found it — the one
        /// reading of it that the run's gate and its plan are both decided by.</param>
        internal RunFeeds(ComputeContext context, IShorokooBackend backend, RunIdentity run, long? budget)
        {
            _context = context;
            _backend = backend;
            _run = run;
            _space = backend.MemorySpace;
            Budget = budget;
        }

        /// <summary>The device-memory budget this run is under — its context's, where the memory
        /// its backend computes in is a device's — or null where there is none.</summary>
        internal long? Budget { get; }

        /// <summary>
        /// The inputs, by the names the run is fed them under, whose consumed memory an output of
        /// this run may be written into: those the session pairs with an output the run keeps in the
        /// device memory it computes in, since an output can only be written into memory where it is
        /// produced. Where the run cannot read a consumed feed of one of these where it is, it is fed
        /// through a copy the framework makes in the run's memory, which the output can then take. A
        /// consumed feed of any other input the run cannot read where it is goes to the session as
        /// the host has it, for the runtime to copy into the arena the session's limit covers —
        /// rather than into memory outside it, which a budget would cut that limit for — and an
        /// output the run fetches back can then be written into it there. None for a run that keeps
        /// no output on the device, or aliases nothing.
        /// </summary>
        internal IReadOnlySet<string> WrittenInto { get; set; } = System.Collections.Frozen.FrozenSet<string>.Empty;

        /// <summary>The locks held, whose eviction signals the run listens for.</summary>
        internal IReadOnlyList<TensorLease> Leases => _leases;

        /// <summary>
        /// Works out what each input feeds — one target per distinct tensor or sequence, in the order
        /// they first appear, and one more for each element of a list sequence fed — and refuses,
        /// before anything is taken, every feed that cannot be held.
        /// </summary>
        /// <exception cref="ObjectDisposedException">A feed is dead. Nothing was taken.</exception>
        /// <exception cref="InvalidOperationException">A feed to be consumed is being read by
        /// another run, or is of a kind nothing knows how to hold. Nothing was taken.</exception>
        internal void Prepare(IReadOnlyList<NamedModelParam> inputs)
        {
            ArgumentNullException.ThrowIfNull(inputs);

            var bySubject = new Dictionary<object, Target>(ReferenceEqualityComparer.Instance);
            var targets = new List<Target>();
            var targetOf = new Target[inputs.Count];
            Target TargetOf(ILifetimeOwner subject)
            {
                if (bySubject.TryGetValue(subject, out var target)) return target;
                target = new Target(subject);
                bySubject.Add(subject, target);
                targets.Add(target);
                return target;
            }

            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var target = targetOf[i] = TargetOf(SubjectOf(input));
                var name = new FeedName(input.Label, input.ParamName, Element: -1);
                target.Feeds(name, input.FeedMode);
                // A list sequence's elements are its own, and a run reading or consuming it reads or
                // consumes them with it -- the sequence value it is fed is built from them. Each is
                // held as this input feeds the sequence, beside whatever else feeds it.
                if (target.Subject is TensorDataSequence { OwnElements: { } elements })
                    for (int j = 0; j < elements.Count; j++)
                        TargetOf(elements[j]).Holds(name with { Element = j }, input.FeedMode, target);
            }

            // Everything that can refuse the run is asked before anything is taken, so a refused run
            // spends nothing. A feed can still die or be locked by another thread in between, which
            // is then a refusal part-way; what was taken by then stays taken, and is released when
            // the run gives up.
            foreach (var target in targets) target.RefuseIfCannotBeHeld(_run, _backend);

            _inputs = inputs;
            _targets = targets;
            _bySubject = bySubject;
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
            // then refused before this run has taken anything. A list sequence's elements come after
            // the sequences holding them, which a tried element follows: read where one of them is.
            foreach (var target in targets)
                if (target.Mode == FeedMode.Shared) Lock(target);
            foreach (var target in targets)
                if (target.Mode != FeedMode.Shared && !target.IsHeld) TakeOrLock(target);
            foreach (var target in targets)
                if (target.Mode != FeedMode.Shared && target.IsHeld)
                {
                    if (target.Mode == FeedMode.TryConsume && target.HeldByARead) Lock(target);
                    else TakeOrLock(target);
                }

            foreach (var input in inputs) input.Held();

            foreach (var target in targets)
                if (target.IsFed) target.Value = Build(target);

            // Every value is built, so what a consumption copied from, and the elements a consumed
            // sequence held only for its value to be built from, are nobody's any more: their memory
            // goes now rather than after the run, which is the point of consuming them.
            foreach (var target in targets)
                if (target.Consumed && (target.Copied || !target.IsFed)) target.Life.ReleaseTaken();

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
        /// or the copy already held, where one is. A tensor it consumes that goes to the session
        /// from the host (<see cref="ThroughTheHost"/>) is copied into the arena by the runtime, once
        /// for each input it feeds, so it is counted there instead (<see cref="DevicePlan.InArena"/>);
        /// so is a tried one nothing else is reading, which the run is about to take. A sequence is
        /// read through the host and adds nothing, and so do the elements it is built from, which the
        /// run holds without putting them on the books. What is in
        /// <paramref name="excludingArena"/> — the arena of the session about to run, where its own
        /// earlier runs left their outputs — is inside that session's limit already, and is left
        /// out.</para>
        ///
        /// <para>The route each feed takes is decided here, and <see cref="Feed"/> follows it: the
        /// copy held, the arena, or a fresh copy.</para>
        /// </summary>
        internal DevicePlan Plan(object? excludingArena)
        {
            var targets = _targets ?? throw new InvalidOperationException("A run was planned before it was prepared.");
            var (attached, attachedTensors) = _context.AttachedIn(excludingArena);

            // What the run adds is what is not on the books already, each once however many inputs
            // it is read through.
            long added = 0;
            long inArena = 0;
            HashSet<TensorData>? adding = null;
            bool Adds(TensorData tensor)
                => tensor.Space == _space
                   && !(excludingArena is not null && ReferenceEquals(tensor.Arena, excludingArena))
                   && !_context.Attaches(tensor)
                   && (adding ??= new(ReferenceEqualityComparer.Instance)).Add(tensor);
            foreach (var target in targets)
            {
                target.PlannedFresh = false;
                target.PlannedArena = false;
                target.Copy = null;
                // A sequence's value is built in host memory whatever the provider.
                if (!target.IsFed || target.Subject is not TensorData tensor) continue;
                TensorData? resident = tensor;
                if (!tensor.FeedsInPlace(_backend))
                {
                    // A read locks the tensor as well as its copy, and the lock attaches it to the
                    // context: where it is in the context's memory too -- another runtime's
                    // allocation on the same card -- the books carry both for the run. A tried feed
                    // may yet be read, so it is counted as one.
                    if (target.Mode != FeedMode.Consume && Adds(tensor)) added += tensor.ByteCount;
                    var where = TensorData.RunMemoryOf(_backend, tensor.DType);
                    if (where.Space != _space) continue;
                    // Read through the copy the tensor holds there, or through a fresh one.
                    resident = target.Copy = tensor.CopyHeldAt(where);
                    if (resident is null)
                    {
                        // Handed to the runtime through the host, it is copied into the arena, which
                        // the session's limit covers: nothing held outside it.
                        if (target.WillBeTaken && ThroughTheHost(target, where))
                        {
                            target.PlannedArena = true;
                            inArena += tensor.ByteCount * target.FedInputs;
                            continue;
                        }
                        target.PlannedFresh = true;
                        added += tensor.ByteCount;
                        continue;
                    }
                }
                if (Adds(resident)) added += resident.ByteCount;
            }
            return new DevicePlan(attached, attachedTensors, added, inArena);
        }

        /// <summary>
        /// The arena limit a new session gets for this run under a budget of <paramref name="limit"/>
        /// bytes — the budget less everything the run will hold outside that session's arena, which
        /// starts empty — with the run admitted against it.
        /// </summary>
        /// <exception cref="InvalidOperationException">What the run would hold leaves the arena
        /// nothing. Nothing has been taken.</exception>
        internal long AdmitFresh(long limit)
        {
            var plan = Plan(excludingArena: null);
            var arena = ComputeContext.ArenaLimitWithin(limit, plan.Outside) ?? throw NoRoom(limit, plan);
            Admit(arena, plan);
            return arena;
        }

        /// <summary>
        /// Records the session this run was admitted to — its arena limit, and the plan it was
        /// chosen by — so each copy the run then makes in the context's memory is held to it.
        /// Refuses the run, before it takes anything, where what the runtime is to copy into that
        /// arena of what it was fed does not fit in it by itself.
        /// </summary>
        /// <exception cref="InvalidOperationException">The arena cannot hold what the run was fed.
        /// Nothing has been taken.</exception>
        internal void Admit(long arenaLimit, DevicePlan plan)
        {
            if (plan.InArena > arenaLimit)
                throw new InvalidOperationException(
                    $"{_run} would have its runtime copy {Figure(plan.InArena)} bytes it was fed into the "
                    + $"arena it computes in, which its compute context's device-memory budget "
                    + "(DeviceMemorySettings.LimitBytes) leaves "
                    + $"{Figure(arenaLimit)} bytes, with {Figure(plan.Outside)} bytes of {_space} held "
                    + "outside it. Nothing it was fed has been taken. Delete what the context no longer "
                    + "needs, feed less at once, or give the context a larger budget.");
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
        /// everything it was is marked so, and nothing here touches it again.
        ///
        /// <para>Marking one retires the copies runs made of it, and a copy whose release throws does
        /// not stop the rest being marked, nor take the place of what the call returned or threw: it
        /// is thrown when the run gives up (<see cref="Dispose(Exception)"/>), and only where the run
        /// itself did not fail.</para>
        /// </summary>
        internal T HandOver<T>(Func<IReadOnlyCollection<IShorokooTensorValue>, T> run)
        {
            try
            {
                return run(_handed.Keys);
            }
            finally
            {
                // Every one of them, whatever marking one of them does: one left unmarked would be
                // released again when the run gives up, after its backend has released it.
                foreach (var owner in _handed.Values)
                {
                    try
                    {
                        owner.Life.HandedToBackend();
                    }
                    catch (Exception e)
                    {
                        _handOverFailure ??= e;
                    }
                }
            }
        }

        /// <summary>
        /// Gives up everything held, however the run ended: every lock dropped — each copy's before
        /// the tensor's it was made of, so a run that finds a tensor free finds its copies free too —
        /// and whatever the run took and did not hand to the backend released.
        ///
        /// <para>Every one of them is given up even where giving one up throws, since what is left
        /// held is held for good. Where the run itself failed — <paramref name="failed"/> — that is
        /// what its caller has to hear, so a release failing on the way out is not allowed to take
        /// its place; otherwise the first such failure is thrown once everything is given up.</para>
        /// </summary>
        /// <param name="failed">What the run failed with, or null where it returned.</param>
        internal void Dispose(Exception? failed)
        {
            Exception? releasing = _handOverFailure;
            for (int i = _leases.Count - 1; i >= 0; i--)
            {
                try
                {
                    _leases[i].Dispose();
                }
                catch (Exception e)
                {
                    releasing ??= e;
                }
            }
            // A release is once only, and does nothing for what was handed over, so these can be
            // asked of everything.
            foreach (var owner in (IEnumerable<ILifetimeOwner>)[.. _handed.Values, .. _taken])
            {
                try
                {
                    owner.Life.ReleaseTaken();
                }
                catch (Exception e)
                {
                    releasing ??= e;
                }
            }
            if (releasing is not null && failed is null) ExceptionDispatchInfo.Throw(releasing);
        }

        /// <summary><see cref="Dispose(Exception)"/> for a run that returned.</summary>
        public void Dispose() => Dispose(failed: null);

        /// <summary>
        /// The tensor or sequence <paramref name="input"/> feeds. An absent optional feeds nothing,
        /// and a session cannot be fed one: its own refusal, naming the engine that can, is thrown
        /// here, before anything is taken. Nor can it be fed a struct whole: its refusal says to feed
        /// the fields, which <c>Execute</c> expands a struct into.
        /// </summary>
        private ILifetimeOwner SubjectOf(NamedModelParam input) => input switch
        {
            TensorDataModelParam tensor => tensor.ToTensorData(),
            OptionalTensorDataModelParam { Data: { HasValue: true, Value: { } present } } => present,
            OptionalTensorDataModelParam absent => RefuseAbsent(absent),
            TensorDataSequenceModelParam sequence => sequence.ToTensorDataSequence(),
            TensorStructModelParam whole => RefuseWhole(whole),
            _ => throw new InvalidOperationException(
                $"Input '{input.ParamName}' was fed to a run as a {input.GetType().Name}, which nothing "
                + "knows how to hold. Every fed input has to be held for the length of the run, or "
                + "deleting it from another thread frees what the run is reading."),
        };

        private ILifetimeOwner RefuseAbsent(OptionalTensorDataModelParam absent)
        {
            absent.ToTensorValue(_backend);
            throw new InvalidOperationException($"Input '{absent.ParamName}' is an absent optional.");
        }

        private static ILifetimeOwner RefuseWhole(TensorStructModelParam whole)
        {
            whole.ToTensorValue();
            throw new InvalidOperationException($"Input '{whole.ParamName}' is a struct.");
        }

        /// <summary>Takes a reader lock on the target for this run, attaching the running context
        /// to a tensor the run is fed — not to an element it holds only for a sequence it is fed,
        /// which it reads through the host and never in the context's memory.</summary>
        private void Lock(Target target) => _leases.Add(_context.Lock(target.Subject, _run, attach: target.IsFed));

        /// <summary>
        /// Takes the target for this run: consumed, if nothing is reading it — and refused if
        /// something is, unless it was only to be tried, in which case it is read instead.
        /// </summary>
        private void TakeOrLock(Target target)
        {
            var death = TensorDeath.ConsumedBy(_run, target.Inputs, tried: target.Mode == FeedMode.TryConsume);
            switch (target.Life.TryTake(death))
            {
                case TakeOutcome.Taken:
                    target.Consumed = true;
                    target.Death = death;
                    _taken.Add(target.Subject);
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
                TensorData tensor when target.Consumed => Consume(target, tensor),
                TensorData tensor => Read(target, tensor),
                TensorDataSequence sequence when target.Consumed => Consume(target, sequence),
                TensorDataSequence sequence => Read(sequence),
                _ => throw new UnreachableException(),
            };

        private IShorokooTensorValue Consume(Target target, TensorData tensor)
        {
            if (tensor.FeedsInPlace(_backend))
                return Hand(tensor, tensor.UncheckedValue(_backend));

            // The route the plan counted, where there is one: a copy that turned up in the run's
            // memory since is left to the tensor, which releases it as the run releases the tensor.
            var where = TensorData.RunMemoryOf(_backend, tensor.DType);
            if (target.PlannedArena
                || (!_admitted && ThroughTheHost(target, where) && tensor.CopyHeldAt(where) is null))
                return ConsumeThroughTheHost(target, tensor);

            // Incompatible memory: the contents go into the run's memory, the tensor is spent, and a
            // copy is what is consumed -- the one it holds there, where nothing else is reading it.
            // Its own memory is released as soon as every value is built rather than after the run,
            // which is the point of consuming it.
            if (tensor.TakeCopyAt(where, target.Death!) is { } held)
                return ConsumeCopy(target, held);

            // None it can take -- none is held, or another run is reading the one that is: through
            // the host where no output may be written into it, and otherwise a fresh copy.
            if (ThroughTheHost(target, where))
                return ConsumeThroughTheHost(target, tensor);
            return ConsumeCopy(target, tensor.TakeRunCopy(_backend, AdmitCopy, target.Death!));
        }

        private IShorokooTensorValue ConsumeCopy(Target target, TensorData copy)
        {
            target.Copy = copy;
            target.Copied = true;
            return Hand(copy, copy.UncheckedValue(_backend));
        }

        /// <summary>
        /// A consumed tensor handed to the session in host memory, for the runtime to copy into the
        /// arena it computes in: as it is, where it is a value of the running backend's runtime in
        /// host memory already, and otherwise through a host copy of that runtime's.
        /// </summary>
        private IShorokooTensorValue ConsumeThroughTheHost(Target target, TensorData tensor)
        {
            if (tensor.Location == TensorData.HostMemoryOf(_backend))
                return Hand(tensor, tensor.UncheckedValue(_backend));
            return ConsumeCopy(target, tensor.HostRunCopy(_backend, target.Death!));
        }

        private IShorokooTensorValue Consume(Target target, TensorDataSequence sequence)
        {
            if (sequence.FeedsInPlace(_backend)) return Hand(sequence, sequence.UncheckedValue);

            var copy = sequence.TakeRunCopy(_backend, target.Death!);
            target.Copied = true;
            return Hand(copy, copy.UncheckedValue);
        }

        private IShorokooTensorValue Hand(ILifetimeOwner owner, IShorokooTensorValue value)
        {
            _handed.Add(value, owner);
            return value;
        }

        /// <summary>
        /// Whether a consumed feed the run cannot read where it is goes to the session as the host has
        /// it — a value of the running backend's runtime in host memory, which the runtime copies into
        /// its own arena — rather than through a copy the framework makes at <paramref name="where"/>,
        /// the run's memory: where that memory is a device's, and no output may be written into it.
        /// </summary>
        private bool ThroughTheHost(Target target, MemoryLocation where)
            => !where.Space.IsHost && !target.MayBeWrittenInto(WrittenInto);

        /// <summary>
        /// The arena the memory behind <paramref name="value"/> was in, where it is a value this run
        /// handed to the backend — the record of the tensor it was, consumed where it stood or copied
        /// for the run — or null where it was in no arena, or is not one this run handed over.
        /// What an output the backend wrote into that memory is in.
        /// </summary>
        internal object? ArenaOfHanded(IShorokooTensorValue value)
            => _handed.TryGetValue(value, out var owner) && owner is TensorData tensor ? tensor.Arena : null;

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

        /// <summary>
        /// One input a target is held for, as a message names it: the caller's label for the input,
        /// or its name — and, for an element of a list sequence the input feeds, which one.
        /// </summary>
        private readonly record struct FeedName(string? Label, string ParamName, int Element)
        {
            public override string ToString()
            {
                var input = Label ?? $"input '{ParamName}'";
                return Element < 0 ? input : $"element {Element} of {input}";
            }
        }

        /// <summary>One tensor or sequence this run holds, however many inputs it feeds.</summary>
        private sealed class Target(ILifetimeOwner subject)
        {
            private readonly List<FeedName> _names = [];
            private List<Target>? _holders;
            private bool _anyShared;
            private bool _anyBare;

            internal ILifetimeOwner Subject { get; } = subject;

            internal Lifetime Life => Subject.Life;

            /// <summary>Whether an input feeds it, rather than only a list sequence an input feeds
            /// holding it — the one kind the session is handed a value for.</summary>
            internal bool IsFed { get; private set; }

            /// <summary>Whether it is an element of a list sequence this run is fed.</summary>
            internal bool IsHeld => _holders is not null;

            /// <summary>Whether a list sequence holding it is one this run reads rather than
            /// consumes: its value is built from this, which a read leaves alive.</summary>
            internal bool HeldByARead => _holders?.Exists(static holder => !holder.Consumed) == true;

            /// <summary>Whether the run is about to take it, as far as can be told before it does:
            /// consumed, or tried with nothing else reading it now.</summary>
            internal bool WillBeTaken
                => Mode == FeedMode.Consume || (Mode == FeedMode.TryConsume && !Life.IsLocked);

            /// <summary>How many of the run's inputs it feeds, each of which the session reads a
            /// value of its own for.</summary>
            internal int FedInputs => _names.Count(static name => name.Element < 0);

            internal bool Consumed { get; set; }

            /// <summary>Whether what the session is fed for it, consumed, is a copy — so its own
            /// memory is nobody's once every value is built.</summary>
            internal bool Copied { get; set; }

            internal TensorDeath? Death { get; set; }

            internal IShorokooTensorValue? Value { get; set; }

            /// <summary>Under a budget: whether the plan counted a fresh copy of it, not yet
            /// made.</summary>
            internal bool PlannedFresh { get; set; }

            /// <summary>Under a budget: whether the plan counted it into the session's arena, to
            /// be handed to the session in host memory for the runtime to copy there.</summary>
            internal bool PlannedArena { get; set; }

            /// <summary>The copy this run reads it through, as far as the run knows: the one it held
            /// when the run was planned, and then the one each read found or made.</summary>
            internal TensorData? Copy { get; set; }

            /// <summary>What the run does with it, over every occurrence: shared if any occurrence
            /// is, else consumed if any is bare, else tried.</summary>
            internal FeedMode Mode
                => _anyShared ? FeedMode.Shared : _anyBare ? FeedMode.Consume : FeedMode.TryConsume;

            /// <summary>The input or inputs it is held for, as a message names them.</summary>
            internal string Inputs
                => _names.Count == 1
                    ? _names[0].ToString()
                    : $"{string.Join(", ", _names.Take(_names.Count - 1))} and {_names[^1]}";

            /// <summary>Whether one of the inputs it feeds is one an output may be written into the
            /// consumed memory of.</summary>
            internal bool MayBeWrittenInto(IReadOnlySet<string> writtenInto)
                => writtenInto.Count > 0 && _names.Exists(name => name.Element < 0 && writtenInto.Contains(name.ParamName));

            /// <summary>Records an input that feeds it.</summary>
            internal void Feeds(FeedName name, SharedInputMode? sharing)
            {
                IsFed = true;
                Holds(name, sharing);
            }

            /// <summary>Records an input it is held for: one that feeds it, or feeds the list
            /// sequence holding it.</summary>
            internal void Holds(FeedName name, SharedInputMode? sharing)
            {
                if (!_names.Contains(name)) _names.Add(name);
                switch (sharing)
                {
                    case null: _anyBare = true; break;
                    case SharedInputMode.Shared: _anyShared = true; break;
                }
            }

            /// <summary>Records an input feeding <paramref name="sequence"/>, a list sequence this is
            /// one of the elements of.</summary>
            internal void Holds(FeedName name, SharedInputMode? sharing, Target sequence)
            {
                Holds(name, sharing);
                (_holders ??= []).Add(sequence);
            }

            /// <summary>Refuses, before anything is taken, a feed that is dead — nothing can be read
            /// or consumed — one to be consumed that another run is reading, and a tensor the run
            /// can neither be handed where it is nor have copied.</summary>
            internal void RefuseIfCannotBeHeld(RunIdentity run, IShorokooBackend backend)
            {
                if (Life.Death is not null) throw Refusal();
                if (Mode == FeedMode.Consume && Life.IsLocked) throw BeingRead(run);
                if (Subject is TensorData tensor && !(IsFed && tensor.FeedsInPlace(backend)) && !tensor.CanBeCopiedOut)
                    throw new InvalidOperationException(
                        $"{tensor.Describe()} cannot be fed to {run} as {Inputs}: the run cannot be handed "
                        + "its memory where it is, and nothing can copy it out, because it is in an "
                        + "execution provider's own memory and the backend that made it was never "
                        + "recorded. Nothing the run was fed has been taken. Wrap the value with "
                        + "TensorData.Create(shape, dtype, value, backend) to say which backend made it.");
            }

            /// <summary>What an access to it throws now that it is dead.</summary>
            internal Exception Refusal() => Life.Death!.Refusal(Subject, Subject.Describe());

            /// <summary>
            /// The refusal of a consumption that would take memory another run is reading, naming
            /// that run where it said who it was, and the ways around it.
            /// </summary>
            internal InvalidOperationException BeingRead(RunIdentity run)
                => new(
                    $"{Subject.Describe()} is being read by {Life.DescribeReader() ?? "another run"}, "
                    + $"so {run} cannot consume it as {Inputs}: fed as it is, it is given to the run it "
                    + "feeds, which would take its memory from under the one reading it. Pass it as "
                    + ".Shared() to read it alongside that run -- or pass the struct, sequence or "
                    + "checkpoint holding it that way -- or as .TryConsume() to have it consumed only "
                    + "when nothing else is reading it; or wait for the other run to return.");
        }
    }
}
