namespace Shorokoo.Core.Utils
{
    /// <summary>
    /// A stream that, when <see cref="IsLengthOnly"/>, only counts the bytes written through it:
    /// it keeps none of them and passes none on. A writer whose payload is expensive to produce —
    /// tensors in a device's memory, which would have to be copied off the card — reports the
    /// payload's length to such a stream with <see cref="Advance"/> instead of writing it.
    /// </summary>
    internal interface ILengthOnlyStream
    {
        /// <summary>Whether the stream only counts, so that <see cref="Advance"/> stands in for
        /// writing.</summary>
        bool IsLengthOnly { get; }

        /// <summary>Counts <paramref name="count"/> bytes as written, exactly as writing that many
        /// would — including refusing them where writing them would be refused.</summary>
        void Advance(long count);
    }
}
