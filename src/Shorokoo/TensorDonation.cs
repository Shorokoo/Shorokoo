using System;

namespace Shorokoo
{
    /// <summary>
    /// A tensor handed to a run rather than lent to it — the result of
    /// <see cref="TensorData.Donate"/>, and an <see cref="IData"/> like any other feed, so it goes
    /// straight into <c>Execute</c> or <c>Run</c>.
    ///
    /// <para>What makes it a donation is what the run does with it. A run fed a tensor reads it,
    /// holding a reader lock for as long as it runs and leaving the tensor alive afterwards; a run
    /// fed a donation <b>consumes</b> the tensor when it starts — the tensor dies there, saying which
    /// run took it — and releases its memory the moment the run returns, rather than whenever the
    /// caller lets go of it.</para>
    ///
    /// <para>Nothing happens to the tensor until a run starts with it: a run refused before it
    /// starts takes nothing, so the donation can be fed again. It feeds once — a second run given
    /// the same donation is refused, the tensor being dead by then — and a run is refused it while
    /// another run is reading the tensor, since consuming memory another run is reading would take
    /// it from under that run.</para>
    /// </summary>
    public sealed class TensorDonation : IData, IDisposable
    {
        internal TensorDonation(TensorData tensor) => Tensor = tensor;

        /// <summary>The donated tensor, which the run it is fed to consumes.</summary>
        internal TensorData Tensor { get; }

        /// <inheritdoc/>
        public DType DType => Tensor.DType;

        /// <summary>The shape of the tensor being donated.</summary>
        public Shape Shape => Tensor.Shape;

        /// <summary>
        /// Takes back a donation nothing was ever fed by deleting the tensor it carries — which is
        /// what giving it away was for. Harmless after a feed, and after a second call: the run has
        /// consumed the tensor by then, and a dead tensor is left as it is.
        /// </summary>
        /// <exception cref="InvalidOperationException">A run is reading the tensor.</exception>
        public void Dispose() => Tensor.Dispose();

        /// <inheritdoc/>
        public override string ToString() => $"donated {Tensor}";
    }
}
