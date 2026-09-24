using System;

namespace Shorokoo
{
    /// <summary>
    /// What a run does with a feed that is not handed to it as it is — the two modes of
    /// <see cref="SharedInput"/>.
    /// </summary>
    public enum SharedInputMode
    {
        /// <summary>
        /// Read it. The run takes a reader lock on it for as long as it runs, and it is alive and
        /// unchanged afterwards. What <c>.Shared()</c> asks for.
        /// </summary>
        Shared = 0,

        /// <summary>
        /// Consume it if nothing else is reading it when the run starts, and read it exactly as
        /// <see cref="Shared"/> does otherwise. What <c>.TryConsume()</c> asks for.
        /// </summary>
        TryConsume = 1,
    }

    /// <summary>
    /// A feed a run is not simply given: a tensor, sequence, struct or optional wrapped with a
    /// <see cref="Mode"/> that says what the run does with it instead. <c>t.Shared()</c> and
    /// <c>t.TryConsume()</c> make one, and it is an <see cref="IData"/> like any other feed, so it
    /// goes straight into <c>Execute</c>, <c>TrainStep</c> and every other call that turns
    /// <see cref="IData"/> arguments into run inputs. <c>Run</c> takes parameters rather than
    /// feeds, and a parameter carries the same choice as its <c>FeedMode</c>.
    ///
    /// <para><b>Why it exists.</b> A tensor fed to a run as it is — bare — is <b>consumed</b> by
    /// it: the run takes the tensor when it starts, and the tensor is dead from then on, its memory
    /// given to the run and released as soon as the run no longer needs it. That is what a feed
    /// built for one run wants, and it is the default. A feed you mean to use again says so:</para>
    ///
    /// <list type="table">
    /// <item><term><c>t.Shared()</c></term><description>The run takes a reader lock on
    /// <c>t</c> and reads it; <c>t</c> is alive and unchanged afterwards.</description></item>
    /// <item><term><c>t.TryConsume()</c></term><description>Decided when the run starts, not when
    /// this is called: if no other run is reading <c>t</c> then, the run consumes it; otherwise it
    /// reads it exactly as <c>Shared</c> does.</description></item>
    /// </list>
    ///
    /// <para>On a composite — a struct, a sequence — the mode applies to every member, except a
    /// struct's field given a mode of its own when the struct was built, which keeps it unless the
    /// struct is fed <c>.Shared()</c>: that reads every field. The same
    /// tensor fed more than once in one call is taken at most once and bound to every input it
    /// feeds: if any occurrence is shared it is not consumed, otherwise a bare occurrence consumes
    /// it, and otherwise it follows <see cref="SharedInputMode.TryConsume"/>.</para>
    ///
    /// <para>It is not a tensor: it holds no memory and cannot be read. <see cref="Value"/> is what
    /// it wraps.</para>
    /// </summary>
    public sealed class SharedInput : IData
    {
        internal SharedInput(IData value, SharedInputMode mode)
        {
            ArgumentNullException.ThrowIfNull(value);
            if (value is not (TensorData or TensorDataSequence or TensorDataStruct or OptionalTensorData))
                throw new ArgumentException(
                    $"A {value.GetType().Name} cannot be fed shared: only a tensor, a sequence, a struct "
                    + "or an optional is a feed a run can read.", nameof(value));
            if (!Enum.IsDefined(mode))
                throw new ArgumentOutOfRangeException(nameof(mode), mode, "Not a SharedInputMode.");
            Value = value;
            Mode = mode;
        }

        /// <summary>The feed this wraps: a <see cref="TensorData"/>,
        /// <see cref="TensorDataSequence"/>, <see cref="TensorDataStruct"/> or
        /// <see cref="OptionalTensorData"/>.</summary>
        public IData Value { get; }

        /// <summary>Whether the run reads <see cref="Value"/>, or consumes it when nothing else is
        /// reading it.</summary>
        public SharedInputMode Mode { get; }

        /// <inheritdoc/>
        public DType DType => Value.DType;

        /// <inheritdoc/>
        public override string ToString()
            => $"{(Mode == SharedInputMode.Shared ? "shared" : "try-consume")} {Value}";
    }
}
