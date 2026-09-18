using System;

namespace Shorokoo
{
    /// <summary>
    /// A tensor handed to a run rather than lent to it — the result of
    /// <see cref="TensorData.Donate"/>, and an <see cref="IData"/> like any other feed, so it goes
    /// straight into <c>Execute</c> or <c>Run</c>.
    ///
    /// <para>What makes it a donation is what happens to the handles. The tensor it was made from
    /// is spent from the moment it is made, exactly as a cross-space <c>TransferTo</c> source is,
    /// and this carries the only handle left on those bytes. Feeding it gives that handle up too,
    /// so for the length of the run the run's own lock is the only thing naming the allocation —
    /// and the bytes go back to the allocator the moment it lets go, rather than waiting for a
    /// caller who has no name for them left.</para>
    ///
    /// <para>It is consumed by the feed, so it feeds once: a second run given the same donation
    /// is refused. If some other handle still names the same bytes — one <c>GiveAccessTo</c>
    /// handed out — they stay alive for it, because donating gives up this handle and says
    /// nothing about anyone else's.</para>
    /// </summary>
    public sealed class TensorDonation : IData, IDisposable
    {
        internal TensorDonation(TensorData tensor) => Tensor = tensor;

        /// <summary>The donated handle: the run's to give up, and nobody else's to hold.</summary>
        internal TensorData Tensor { get; }

        /// <inheritdoc/>
        public DType DType => Tensor.DType;

        /// <summary>The shape of the tensor being donated.</summary>
        public Shape Shape => Tensor.Shape;

        /// <summary>
        /// Takes back a donation nothing was ever fed, releasing the handle it carries — which is
        /// the last one, so the bytes go. It is what the donated tensor's own <c>Dispose</c> would
        /// have been had it not been given away, and it is the only deterministic release an unfed
        /// donation has.
        ///
        /// <para>Harmless after a feed, and after a second call: the run has already given the
        /// handle up by then, and letting go of a handle twice lets go of nothing.</para>
        /// </summary>
        public void Dispose() => Tensor.Dispose();

        /// <inheritdoc/>
        public override string ToString() => $"donated {Tensor}";
    }
}
