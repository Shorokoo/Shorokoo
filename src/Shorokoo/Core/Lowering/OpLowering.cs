using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core.Lowering;

/// <summary>
/// The one thing an <see cref="OpLowering"/> is allowed to do: emit a primitive operator. The
/// engine running the lowering supplies the emitter and so decides what emitting means — the
/// QuickExecutionEngine computes the primitive's value on the spot
/// (<see cref="RuntimeTensorEmitter"/>), while the graph side builds a real node for it
/// (<see cref="VariableEmitter"/>).
///
/// <para><typeparamref name="T"/> is whatever that engine calls a value: an
/// <c>IRuntimeTensor</c> for the QuickExecutionEngine, a <c>Variable</c> for the graph. A
/// lowering never names either, which is what lets one decomposition serve both.</para>
/// </summary>
/// <typeparam name="T">The emitting engine's value type.</typeparam>
internal interface IOpEmitter<T> where T : class
{
    /// <summary>
    /// Emits one <paramref name="opCode"/> operator over <paramref name="inputs"/> (a null entry
    /// is an omitted optional input) and returns its <paramref name="outputCount"/> outputs.
    /// <paramref name="attrs"/> are the emitted operator's OWN attributes, named as its node
    /// definition declares them — never the attribute bag of the operator being lowered.
    /// Throws when the emitted operator cannot be produced, e.g. because this engine has no
    /// implementation of that op code.
    /// </summary>
    T[] Emit(string opCode, T?[] inputs, (string Name, object? Value)[] attrs, int outputCount);
}

/// <summary>
/// How one operator is computed out of simpler ones — written once here, run by every engine
/// that lacks a direct implementation of it.
///
/// <para><b>Why.</b> An operator that is merely a composition of simpler operators used to be
/// written up to three times over: as the authoring-layer decomposition that keeps the exported
/// ONNX at the opset the framework emits, as a QuickExecutionEngine kernel, and as a gradient
/// rule. Nothing was shared, so the three could drift. A registered lowering is that
/// decomposition stated once, in terms neither engine owns.</para>
///
/// <para><b>Operator lowering is not graph lowering.</b> "Lowering" elsewhere in this codebase
/// means the graph concretization pipeline — <c>ToConcreteArchitecture</c> and the
/// <c>FastLower*</c> passes, which rewrite the graph that is then exported (see
/// <c>Documentation/debugging.md</c>). Operator lowering rewrites nothing: it is an
/// engine-internal fallback consulted while an engine is already running, and it leaves the
/// graph exactly as it found it. A <c>Softsign</c> node still exports as a <c>Softsign</c>
/// node.</para>
///
/// <para>A lowering body is ordinary C#, so it may branch and loop over ranks or attributes; it
/// simply cannot touch any engine's concrete value type — everything it produces comes back
/// from <see cref="IOpEmitter{T}.Emit"/>.</para>
/// </summary>
internal abstract class OpLowering
{
    /// <summary>The op code this lowering expresses (e.g. "Softsign").</summary>
    public abstract string OpCode { get; }

    /// <summary>
    /// Computes the operator's outputs — one per declared output, in declaration order — from
    /// <paramref name="inputs"/> (a null entry is an omitted optional input) and
    /// <paramref name="attributes"/>, the bag of the operator BEING lowered. Every intermediate
    /// value is obtained from <paramref name="emitter"/>, and intermediates are the lowering's
    /// own locals: nothing the lowering emits is visible to the engine beyond the outputs it
    /// returns.
    ///
    /// <para>An override has to restate <c>where T : class</c>. C# normally forbids repeating an
    /// inherited constraint, but a reference-type one is the exception, and without it the
    /// compiler reads the <c>T?</c> here as <c>Nullable&lt;T&gt;</c> and finds no method to
    /// override.</para>
    /// </summary>
    public abstract T[] Lower<T>(IOpEmitter<T> emitter, T?[] inputs, OnnxCSharpAttributes attributes)
        where T : class;
}
