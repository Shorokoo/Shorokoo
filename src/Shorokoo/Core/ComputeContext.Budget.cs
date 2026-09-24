using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using Shorokoo.Core.Backends;

namespace Shorokoo.Runtime
{
    /// <summary>
    /// A compute context's device-memory budget: <see cref="DeviceMemorySettings.LimitBytes"/> on a
    /// context whose memory is a device's, kept by counting what is attached to the context there.
    ///
    /// <list type="bullet">
    /// <item><b>What it counts.</b> The bytes of the live tensors attached to the context that are in
    /// its memory (<see cref="ReadDeviceMemoryUse"/>), plus — while one of its runs executes — that
    /// run's arena, which the session was built to cap at what the attached tensors leave.</item>
    /// <item><b>Transfers.</b> <see cref="TensorData.To"/>, <see cref="TensorData.CopyTo"/>,
    /// <see cref="AllocateUninitialized(Shape, DType)"/> and the copies a run makes of memory it
    /// cannot read where it is are refused, naming the budget, what is attached and what was asked
    /// for, when what is attached plus what they would add would pass the limit.</item>
    /// <item><b>Runs.</b> A session's arena limit is the budget less what the context holds in its
    /// memory outside that arena for the length of the run — the <i>discount</i>. A session is kept
    /// while its limit is within what the budget allows, and built again when the discount has grown
    /// past what it left room for; see <see cref="ArenaLimitWithin"/>. An output a run wrote into
    /// memory it consumed (<see cref="OutputAlias"/>) is where that memory was — outside the arena,
    /// or inside it where the consumed tensor was the session's own earlier output — and is counted
    /// there, once.</item>
    /// <item><b>One at a time.</b> Under a budget, the context's runs, the sessions it builds and what
    /// is placed in its memory are serialized, and every run shrinks its arena when it ends.</item>
    /// </list>
    ///
    /// <para>A context with no <c>LimitBytes</c>, and one whose memory is the host's, keeps no budget:
    /// nothing here touches what it does.</para>
    /// </summary>
    public partial class ComputeContext
    {
        /// <summary>
        /// Into how many parts a budget is cut for the room a session's arena leaves: a session is
        /// built with the budget less the discount rounded up to the next whole part above it — a
        /// sixty-fourth of the budget — so that it is kept until the discount grows past that, and
        /// rebuilt at most this many times as the discount climbs through the parts, and once more
        /// each time what is left halves in the last one (see <see cref="ArenaLimitWithin"/>).
        /// </summary>
        internal const int BudgetParts = 64;

        // Serializes what spends this context's budget. Built on the first budgeted use: a context
        // with no budget never has one.
        private BudgetGate? _budgetGate;

        /// <summary>
        /// What is attached to this context in its own memory, against its device-memory budget:
        /// the bytes of the live tensors on its books that are in its memory — on a GPU backend,
        /// the card's — how many they are, and the budget, where one is in force.
        ///
        /// <para>That is what the budget counts. A transfer onto this context is refused when what is
        /// attached plus what it would add passes <see cref="DeviceMemoryUse.LimitBytes"/>, and a
        /// session compiled or run here gets an arena limited to what is left. A tensor on two
        /// contexts' books counts on both; one that dies, is collected or is detached
        /// (<see cref="Detach"/>) drops out. <see cref="Host"/> keeps no books and reads as
        /// nothing.</para>
        ///
        /// <para>It is a reading: it walks this context's list, and nothing is remembered.</para>
        /// </summary>
        public DeviceMemoryUse ReadDeviceMemoryUse()
        {
            if (_isHost) return default;
            var space = MemorySpace;
            var (bytes, tensors) = AttachedIn(space);
            return new DeviceMemoryUse(bytes, tensors, BudgetIn(space));
        }

        /// <summary>
        /// The budget in force on this context's memory when that memory is
        /// <paramref name="space"/>: <see cref="DeviceMemorySettings.LimitBytes"/> where the space is
        /// a device's, and null where it is the host's, which a device-memory budget does not
        /// govern.
        /// </summary>
        internal long? BudgetIn(MemorySpace space) => _isHost || space.IsHost ? null : DeviceMemory.LimitBytes;

        /// <summary>
        /// The bytes of the live tensors attached to this context in <paramref name="space"/>, and
        /// how many they are — leaving out those in <paramref name="excludingArena"/>, the arena of
        /// the session about to run, whose limit already covers them.
        /// </summary>
        internal (long Bytes, int Tensors) AttachedIn(MemorySpace space, object? excludingArena = null)
        {
            long bytes = 0;
            var tensors = 0;
            foreach (var tensor in AttachedTensorsIn(space, excludingArena))
            {
                bytes += tensor.ByteCount;
                tensors++;
            }
            return (bytes, tensors);
        }

        /// <summary>The live tensors attached to this context in <paramref name="space"/>, those in
        /// <paramref name="excludingArena"/> left out.</summary>
        internal List<TensorData> AttachedTensorsIn(MemorySpace space, object? excludingArena)
        {
            var found = new List<TensorData>();
            foreach (var tensor in _attached.Snapshot())
            {
                if (tensor.IsDisposed || tensor.Space != space) continue;
                if (excludingArena is not null && ReferenceEquals(tensor.Arena, excludingArena)) continue;
                found.Add(tensor);
            }
            return found;
        }

        /// <summary>
        /// Enters this context's budget gate when memory in <paramref name="space"/> is under a
        /// budget, and answers null — having entered nothing — when it is not. From here to the
        /// matching <see cref="BudgetGate.Exit"/>, no run of this context executes and nothing else
        /// is placed in its memory. The thread holding the gate may enter it again (see
        /// <see cref="BudgetGate"/>).
        /// </summary>
        /// <exception cref="OperationCanceledException"><paramref name="cancellation"/> was cancelled
        /// while this waited for the gate.</exception>
        internal BudgetGate? EnterBudget(MemorySpace space, CancellationToken cancellation)
        {
            if (BudgetIn(space) is null) return null;
            var gate = Volatile.Read(ref _budgetGate)
                       ?? Interlocked.CompareExchange(ref _budgetGate, new BudgetGate(), null)
                       ?? _budgetGate!;
            gate.Enter(cancellation);
            return gate;
        }

        /// <summary>
        /// Attaches <paramref name="tensors"/> to this context, as <see cref="TensorData.To"/> does
        /// with what the context can read where it is — refusing first, having attached none of
        /// them, when the ones not yet on its books would take its budget past its limit.
        /// </summary>
        /// <exception cref="InvalidOperationException">The budget cannot take them.</exception>
        internal void AttachAllWithinBudget(IReadOnlyList<TensorData> tensors, string operation)
        {
            if (_isHost) return;
            var space = MemorySpace;
            var gate = EnterBudget(space, CancellationToken.None);
            try
            {
                // Again, now that this may have waited at the gate for a run of this context: a
                // tensor consumed or deleted meanwhile is refused, not handed back dead.
                foreach (var tensor in tensors) tensor.ThrowIfDisposed();
                if (gate is not null)
                {
                    long adding = 0;
                    var seen = new HashSet<TensorData>(ReferenceEqualityComparer.Instance);
                    foreach (var tensor in tensors)
                        if (tensor.Space == space && !tensor.IsDisposed && !_attached.Contains(tensor)
                            && seen.Add(tensor))
                            adding += tensor.ByteCount;
                    RefusePlacementOverBudget(space, adding, () => tensors.Count == 1
                        ? $"{operation}(context) of {tensors[0].Describe()}"
                        : $"{operation}(context) of a sequence of {tensors.Count} tensors");
                }
                foreach (var tensor in tensors) Attach(tensor);
            }
            finally
            {
                gate?.Exit();
            }
        }

        /// <summary>
        /// Places what <paramref name="place"/> puts on this context — a struct's fields, or a
        /// sequence's elements — as one placement under its device-memory budget: refused whole,
        /// having placed nothing, where the <paramref name="bytes"/> it adds would take the budget
        /// past its limit, and taken off the books whole where a part of it fails, so a composite
        /// that is refused leaves nothing of itself counted. The gate is held throughout, so nothing
        /// else is placed, and no run of this context runs, in between.
        ///
        /// <para><paramref name="bytes"/> is null where what the composite adds cannot be told
        /// without making it — a copy of a runtime's sequence, whose elements are only minted by
        /// reading them — and then each part is held to the budget as it is placed.</para>
        /// </summary>
        /// <exception cref="InvalidOperationException">The budget cannot take the composite. Nothing
        /// was placed.</exception>
        internal T PlaceAll<T>(long? bytes, Func<string> asked, Func<T> place)
        {
            if (_isHost) return place();
            var space = MemorySpace;
            var gate = EnterBudget(space, CancellationToken.None);
            if (gate is null) return place();
            try
            {
                if (bytes is { } adding) RefusePlacementOverBudget(space, adding, asked);
                // Only this placement attaches to this context while the gate is held -- its runs,
                // compiles and every other placement wait at it -- so whatever the list gains in
                // between is this placement's.
                var before = new HashSet<TensorData>(_attached.Snapshot(), ReferenceEqualityComparer.Instance);
                try
                {
                    return place();
                }
                catch
                {
                    foreach (var tensor in _attached.Snapshot())
                        if (!before.Contains(tensor))
                            lock (_gate) _attached.Remove(tensor);
                    throw;
                }
            }
            finally
            {
                gate.Exit();
            }
        }

        /// <summary>
        /// Refuses to place <paramref name="bytes"/> more in this context's memory, in
        /// <paramref name="space"/>, where its budget cannot take them alongside what is attached to
        /// it there. The caller holds the budget gate, so nothing else is placed in between.
        /// </summary>
        /// <param name="space">The memory being placed in: the context's own.</param>
        /// <param name="bytes">What the placement adds to what is attached there.</param>
        /// <param name="asked">What asked, as the refusal names it — built only for a refusal.</param>
        /// <exception cref="InvalidOperationException">The budget cannot take them.</exception>
        internal void RefusePlacementOverBudget(MemorySpace space, long bytes, Func<string> asked)
        {
            if (bytes <= 0 || BudgetIn(space) is not { } limit) return;
            var (attached, tensors) = AttachedIn(space);
            if (attached + bytes <= limit) return;
            throw new InvalidOperationException(
                $"{asked()} asks this compute context for {Figure(bytes)} bytes of {space}, which its "
                + "device-memory budget cannot give: the budget (DeviceMemorySettings.LimitBytes) is "
                + $"{Figure(limit)} bytes, and {Figure(attached)} bytes of it are attached to the "
                + $"context there, in {Figure(tensors)} tensor(s), leaving {Figure(limit - attached)}. "
                + "Delete what the context no longer needs, or give it a larger budget.");
        }

        /// <summary>
        /// The refusal of a compile when what is attached to this context leaves no room in its
        /// budget for the new session's arena: a session holds its weights in that arena from the
        /// moment it is built, so it needs a limit above zero.
        /// </summary>
        private InvalidOperationException NoRoomToCompile(MemorySpace space, long limit, long attached, int tensors)
            => new(
                "Compiling a graph on this compute context builds a session whose arena has to fit "
                + $"in the context's device-memory budget, and the {Figure(attached)} bytes of the "
                + $"{Figure(tensors)} tensor(s) attached to it in {space} leave nothing of the "
                + $"{Figure(limit)} bytes it has (DeviceMemorySettings.LimitBytes). Delete what the "
                + "context no longer needs, or give it a larger budget.");

        /// <summary>A figure as a budget refusal prints it: invariant digits, so a message reads the
        /// same whatever culture the process runs in.</summary>
        internal static string Figure(long value) => value.ToString(CultureInfo.InvariantCulture);

        /// <summary>
        /// The arena limit a session is built with under a budget of <paramref name="limit"/>
        /// bytes while <paramref name="outside"/> bytes of the context's memory are held outside
        /// that arena, or null when they leave it nothing.
        ///
        /// <para>The budget less the discount rounded up to the next whole
        /// <see cref="BudgetParts"/>th of the budget above it. Rounding up is the headroom that keeps
        /// a session: one is kept while its limit is within what the budget allows, which it stays
        /// until the discount grows past the part it rounded up to. So a steady run keeps its
        /// session, and one whose discount keeps climbing rebuilds once per part it climbs through —
        /// at most <see cref="BudgetParts"/> times — rather than on every run. It costs the arena
        /// less than one part of the budget.</para>
        ///
        /// <para>In the budget's last part, where that rounding would leave nothing, the limit is the
        /// largest halving of a part that fits in what is left: a discount climbing on through it
        /// then rebuilds once each time what is left halves — a handful of times — where the room
        /// exactly would rebuild on every run it grew by a byte.</para>
        ///
        /// <para>The limit only ever comes down. A session is not rebuilt when the discount falls,
        /// since a limit below what the budget allows breaches nothing, and a policy that also
        /// raised it would rebuild on every run of a loop whose discount moved both ways.</para>
        /// </summary>
        internal static long? ArenaLimitWithin(long limit, long outside)
        {
            if (outside >= limit) return null;
            var part = Math.Max(1L, limit / BudgetParts);
            var headroom = part - Math.Max(0L, outside) % part;
            if (outside < limit - headroom) return limit - outside - headroom;
            var room = limit - outside;
            var arena = part;
            while (arena > room) arena /= 2;
            return arena;
        }
    }

    /// <summary>
    /// What serializes everything that spends a budgeted compute context's device memory — its runs,
    /// the sessions it builds, and what is placed in its memory — so that no two of them count the
    /// same room. Held by one thread at a time.
    ///
    /// <para>The holder may enter it again, and a nested entry is counted rather than waited for:
    /// a thread waiting on a gate it holds itself would wait for ever. Nothing the framework does
    /// while holding the gate enters it twice — a run's own copies are admitted against the run's
    /// plan rather than through the gate — so this only keeps a nested entry from becoming a
    /// hang.</para>
    /// </summary>
    internal sealed class BudgetGate
    {
        private readonly SemaphoreSlim _semaphore = new(1, 1);

        // The managed thread holding the gate, or zero; written only by the holder, so another
        // thread can never read its own id here by accident.
        private int _owner;
        private int _depth;

        /// <summary>Waits for the gate, unless this thread holds it already.</summary>
        /// <exception cref="OperationCanceledException"><paramref name="cancellation"/> was cancelled
        /// while this waited.</exception>
        internal void Enter(CancellationToken cancellation)
        {
            var me = Environment.CurrentManagedThreadId;
            if (Volatile.Read(ref _owner) == me)
            {
                _depth++;
                return;
            }
            _semaphore.Wait(cancellation);
            Volatile.Write(ref _owner, me);
            _depth = 1;
        }

        /// <summary>Leaves the gate, letting the next holder in once this thread has left it as often
        /// as it entered.</summary>
        internal void Exit()
        {
            if (--_depth > 0) return;
            Volatile.Write(ref _owner, 0);
            _semaphore.Release();
        }
    }
}
