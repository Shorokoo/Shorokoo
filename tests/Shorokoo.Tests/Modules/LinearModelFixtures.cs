using Shorokoo.Modules.Initializers;

namespace Shorokoo.Tests.Modules;

[TrainableParamInitializer]
public static partial class InitScalarWeight
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitBnRunningMean
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 0.0f);
    }
}

[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitBnRunningVar
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

/// <summary>
/// Optimizer-owned ones-fill state initializer. The ones value is deliberately different from
/// <see cref="Shorokoo.Modules.Optimizers.OptimizerStateZeros"/> so tests can prove optimizer
/// state really is initialized by its [StateInitializer] (not blanket zero-filled by the rig).
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class InitOptStateOnes
{
    public static Tensor<float32> Inline(Vector<int64> shape)
    {
        return Globals.TensorFill(shape, 1.0f);
    }
}

/// <summary>
/// Plain SGD plus a step-counter state created from <see cref="InitOptStateOnes"/>: the counter
/// starts at 1 and increments by 1 each step, while the parameter update ignores it. Lets tests
/// assert both the initializer-driven initial value and the per-step state round-trip.
/// </summary>
[Module]
public partial class StepCountingSgdOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate)
    {
        var stepCounter = InitOptStateOnes.Init(currentParam.ShapeTensor());
        Globals.StateUpdate(stepCounter, stepCounter + Scalar(1f));
        return currentParam - learningRate * grad;
    }
}

/// <summary>
/// Optimizer-owned state initializer that fills the parameter's shape with a supplied scalar value
/// — deliberately <b>reads a hyperparameter</b> (see <see cref="InitFromHyperOptimizer"/>) so its
/// state-init graph consumes a hyper input. Used to exercise the §2.5 value route (state init sees
/// the hyper's real value at the initial counters, not 0f) and the D5 fail-loud path.
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class InitToScalarFill
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<float32> value)
    {
        // A param-shaped tensor filled with the (graph-valued) scalar: ones * value broadcasts it,
        // and — crucially — reads `value`, so the split-off state-init graph consumes that hyper input.
        return Globals.TensorFill(shape, 1.0f) * value;
    }
}

/// <summary>
/// SGD whose optimizer state is initialized to the learning-rate hyperparameter's value (via
/// <see cref="InitToScalarFill"/>), then carried unchanged. Because its state initializer reads the
/// LR hyper, the fresh optimizer state equals the LR at the initial counters — so a test can read
/// that state back and prove the value route feeds the real scheduled value (not the old hardcoded
/// 0f), and that a runtime LR triggers the D5 fail-loud unless supplied explicitly.
/// </summary>
[Module]
public partial class InitFromHyperOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate)
    {
        var s = InitToScalarFill.Init(currentParam.ShapeTensor(), learningRate);   // state init reads LR
        Globals.StateUpdate(s, s);                                                 // carried unchanged
        return currentParam - learningRate * grad;
    }
}

/// <summary>Impure scheduler module — carries a trainable parameter; rig build must reject it (D4).</summary>
[Module]
public partial class ParamScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step)
    {
        var w = InitScalarWeight.Init(Vector(1L));
        var wScalar = w.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return step.Cast<float32>() * Scalar(0f) + wScalar;
    }
}

/// <summary>Impure scheduler module — carries module state (a StateUpdate); rig build must reject it (D4).</summary>
[Module]
public partial class StateScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step)
    {
        var s = InitBnRunningMean.Init(Vector(1L));
        Globals.StateUpdate(s, s + Scalar(1f));
        var sScalar = s.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return step.Cast<float32>() * Scalar(0f) + sScalar;
    }
}

/// <summary>
/// Multi-counter scheduler module (D1): consumes both the <c>step</c> and <c>epoch</c> reserved
/// counters, so a test can prove the rig feeds each named counter from the checkpoint. Value is
/// <c>0.5 − 0.01·step − 0.1·epoch</c> — pure arithmetic over both counters.
/// </summary>
[Module]
public partial class StepEpochScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step, Scalar<int64> epoch)
        => Scalar(0.5f) - step.Cast<float32>() * Scalar(0.01f) - epoch.Cast<float32>() * Scalar(0.1f);
}

/// <summary>
/// SGD whose hyperparameters span four dtypes — <c>float32</c>, <c>int32</c>, <c>bit</c> and
/// <c>float64</c> — so the whole hyperparameter pipeline (authoring default, generated
/// hyperparameter set, packing, graph binding, persistence) is exercised off <c>float32</c> (#125).
/// The update is <c>param·(1 − decay·lr) − sign·lr·scale·grad</c>, with <c>sign</c> <c>+1</c> when
/// <c>descend</c> is true and <c>−1</c> otherwise, so every hyperparameter observably moves the weight.
/// </summary>
[Module]
public partial class MixedDTypeHyperOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate,
        [Hyper(2)] Scalar<int32> gradScale,
        [Hyper(true)] Scalar<bit> descend,
        [Hyper(0.25)] Scalar<float64> decay)
    {
        var sign = descend.Cast<float32>() * Scalar(2f) - Scalar(1f);
        return currentParam * (Scalar(1f) - decay.Cast<float32>() * learningRate)
             - sign * learningRate * gradScale.Cast<float32>() * grad;
    }
}

/// <summary>
/// Optimizer-owned state initializer filling the parameter's shape from an <c>int32</c> scalar, so a
/// non-<c>float32</c> hyperparameter reaches the split-off state-init graph (#125).
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class InitToIntScalarFill
{
    public static Tensor<float32> Inline(Vector<int64> shape, Scalar<int32> value)
        => Globals.TensorFill(shape, 1.0f) * value.Cast<float32>();
}

/// <summary>
/// SGD whose optimizer state is initialized from an <c>int32</c> hyperparameter — the non-float
/// counterpart of <see cref="InitFromHyperOptimizer"/>, proving the §2.5 value route carries a
/// declared dtype other than <c>float32</c> into state init.
/// </summary>
[Module]
public partial class InitFromIntHyperOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate,
        [Hyper(3)] Scalar<int32> stateSeed)
    {
        var s = InitToIntScalarFill.Init(currentParam.ShapeTensor(), stateSeed);
        Globals.StateUpdate(s, s);
        return currentParam - learningRate * grad;
    }
}

/// <summary>Scheduler module producing an <c>int32</c> value, for a non-float32 scheduled hyperparameter.</summary>
[Module]
public partial class IntStepScheduler
{
    public static Scalar<int32> Inline(Scalar<int64> step)
        => (step + Scalar(2L)).Cast<int32>();
}

/// <summary>
/// SGD whose learning rate is a <b>vector</b> hyperparameter, broadcast over the parameter — a
/// non-scalar hyperparameter carried end to end (#125). The update is
/// <c>param − (perElementRate · gain) · grad</c>, with <c>gain</c> a plain scalar alongside it so a
/// rig mixes shapes as well as dtypes.
/// </summary>
[Module]
public partial class VectorRateOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper] Vector<float32> perElementRate,
        [Hyper(1f)] Scalar<float32> gain)
        => currentParam - perElementRate * gain * grad;
}

/// <summary>
/// Scheduler module producing a rank-1 <c>float32</c> value — a decaying per-element learning rate —
/// so a non-scalar hyperparameter can be driven in-graph from the step counter.
/// </summary>
[Module]
public partial class VectorRateScheduler
{
    public static Vector<float32> Inline(Scalar<int64> step)
        => Globals.Vector(0.1f, 0.2f, 0.4f, 0.8f) - step.Cast<float32>() * Scalar(0.01f);
}

/// <summary>
/// Optimizer-owned state initializer seeded from a rank-1 hyperparameter, so a non-scalar
/// hyperparameter reaches the split-off state-init graph.
/// </summary>
[StateInitializer(Ownership = StateOwnership.OptimizerOwned)]
public static partial class InitToVectorSum
{
    public static Tensor<float32> Inline(Vector<int64> shape, Vector<float32> value)
        => Globals.TensorFill(shape, 1.0f) * value.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
}

/// <summary>
/// SGD whose optimizer state is initialized from the sum of a <b>vector</b> hyperparameter, proving
/// the §2.5 value route carries a non-scalar hyperparameter into state init.
/// </summary>
[Module]
public partial class InitFromVectorHyperOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper] Vector<float32> perElementRate)
    {
        var s = InitToVectorSum.Init(currentParam.ShapeTensor(), perElementRate);
        Globals.StateUpdate(s, s);
        return currentParam - perElementRate * grad;
    }
}

/// <summary>Impure scheduler module — draws RNG; rig build must reject it (D4).</summary>
[Module]
public partial class RngScheduler
{
    public static Scalar<float32> Inline(Scalar<int64> step)
    {
        var r = Globals.RandomUniform(Vector(1L));
        var rScalar = r.Reduce(ReduceKind.Sum, keepDims: false).Scalar();
        return step.Cast<float32>() * Scalar(0f) + rScalar;
    }
}

/// <summary>
/// Optimizer that misuses a module-owned state initializer for its state; the TrainingRig must
/// reject the graph with guidance towards StateOwnership.OptimizerOwned.
/// </summary>
[Module]
public partial class ModuleOwnedStateOptimizer
{
    public static Tensor<float32> Inline(
        Tensor<float32> currentParam,
        Tensor<float32> grad,
        [Hyper(0.1f)] Scalar<float32> learningRate)
    {
        var state = InitBnRunningMean.Init(currentParam.ShapeTensor());
        Globals.StateUpdate(state, state + Scalar(1f));
        return currentParam - learningRate * grad;
    }
}

/// <summary>
/// Model that misuses an optimizer-owned state initializer for module state; the TrainingRig
/// must reject the graph with guidance towards StateOwnership.ModuleOwned.
/// </summary>
[Module]
public partial class OptimizerOwnedStateModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var weight = InitScalarWeight.Init(Vector(1L));
        var state = InitOptStateOnes.Init(Vector(1L));
        Globals.StateUpdate(state, state + Scalar(1f));
        return input * weight;
    }
}

[Module]
public partial class ScalarMultiplyModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        Vector<int64> weightShape = Vector(1L);
        var weight = InitScalarWeight.Init(weightShape);
        return input * weight;
    }
}

/// <summary>Module-owned rank-0 state: a call counter, one float rather than a param-shaped buffer.</summary>
[StateInitializer(Ownership = StateOwnership.ModuleOwned)]
public static partial class InitScalarCallCount
{
    public static Scalar<float32> Inline() => Scalar(0.0f);
}

/// <summary>
/// The rank-0 twin of <see cref="ScalarMultiplyModel"/>: two trainable rank-0 parameters (a gain
/// seeded at 1 and a bias seeded at 0) plus rank-0 module-owned state — none of the three carrying
/// a shape input.
/// </summary>
[Module]
public partial class Rank0ScalarModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var calls = InitScalarCallCount.Init();
        Globals.StateUpdate(calls, calls + Scalar(1f));
        return input * ScalarOnes.Init() + ScalarZeros.Init() + calls * Scalar(0f);
    }
}

/// <summary>Scales its input by one trainable rank-0 gain seeded at 1.</summary>
[Module]
public partial class Rank0GainSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => input * ScalarOnes.Init();
}

/// <summary>
/// Two trainable rank-0 parameters in creation order: a bias seeded at 0 is parameter [1], a gain
/// seeded at 1 is parameter [2]. A reference to [2] that mis-resolved to the module's FIRST
/// initializer would read 0 instead of 1.
/// </summary>
[Module]
public partial class Rank0BiasThenGainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var bias = ScalarZeros.Init();
        var gain = ScalarOnes.Init();
        return input * gain + bias;
    }
}

/// <summary>
/// Two rank-0 parameters created inside a 3-trip loop body: the per-iteration realization path
/// must give each iteration slot its own shapeless parameter.
/// </summary>
[Module]
public partial class Rank0ParamsInLoopModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var x = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            x = x * ScalarOnes.Init() + ScalarZeros.Init();
            ctx.ContinueWhile(Scalar(true));
        }
        return x;
    }
}

/// <summary><see cref="Rank0GainSubModel"/> called plainly — the naming baseline for
/// <see cref="Rank0GainWithRefModel"/>.</summary>
[Module]
public partial class Rank0GainNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => Rank0GainSubModel.Call(input);
}

/// <summary><see cref="Rank0GainNoRefModel"/> plus a read-only reference to the sub-model's rank-0
/// parameter, contributing nothing to the output.</summary>
[Module]
public partial class Rank0GainWithRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = Rank0GainSubModel.Model();
        return m.Call(input) + m.GetTrainableParam<float32>([1], rank: 0) * Scalar(0f);
    }
}

/// <summary>Rank-1 counterpart of <see cref="Rank0GainSubModel"/>: a shaped initializer, so its
/// definition carries a shape input a reference to it has not.</summary>
[Module]
public partial class Rank1GainSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => input * Ones.Init([Scalar(2L)]);
}

/// <summary><see cref="Rank1GainSubModel"/> called plainly — the naming baseline for
/// <see cref="Rank1GainWithRefModel"/>.</summary>
[Module]
public partial class Rank1GainNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => Rank1GainSubModel.Call(input);
}

/// <summary><see cref="Rank1GainNoRefModel"/> plus a read-only reference to the sub-model's rank-1
/// parameter, contributing nothing to the output.</summary>
[Module]
public partial class Rank1GainWithRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = Rank1GainSubModel.Model();
        return m.Call(input) + m.GetTrainableParam<float32>([1], rank: 1) * Scalar(0f);
    }
}

/// <summary>One nested and one flat parameter, called plainly — the naming baseline for
/// <see cref="MixedDepthGainWithRefsModel"/>.</summary>
[Module]
public partial class MixedDepthGainNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => Rank1GainNoRefModel.Call(input) + Rank1GainSubModel.Call(input);
}

/// <summary>
/// <see cref="MixedDepthGainNoRefModel"/> plus a read-only reference to each parameter. The two
/// referenced models sit at different depths, so composing the shallow reference's relative id
/// onto the deep model's base yields <c>[1, 1]</c> — a strict prefix of the nested parameter's
/// own <c>[1, 1, 1]</c>, and the id a prefix-shortest template lookup would settle on first.
/// </summary>
[Module]
public partial class MixedDepthGainWithRefsModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var deep = Rank1GainNoRefModel.Model();
        var flat = Rank1GainSubModel.Model();
        return deep.Call(input) + flat.Call(input)
             + deep.GetTrainableParam<float32>([1, 1], rank: 1) * Scalar(0f)
             + flat.GetTrainableParam<float32>([1], rank: 1) * Scalar(0f);
    }
}

/// <summary>One model handle called twice — the two calls share the one weight.</summary>
[Module]
public partial class SharedModelCalledTwiceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = Rank1GainSubModel.Model();
        return m.Call(input) + m.Call(input);
    }
}

/// <summary><see cref="Rank1GainSubModel"/> plus module-owned state, so a call site contributes a
/// state parameter as well as a trainable one.</summary>
[Module]
public partial class StatefulGainSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seen = InitRunningMean.Init(input.ShapeTensor());
        Globals.StateUpdate(seen, seen + Scalar(1f));
        return input * Ones.Init([Scalar(2L)]) + seen;
    }
}

/// <summary><see cref="StatefulGainSubModel"/> called once — the naming baseline for
/// <see cref="StatefulGainCalledTwiceModel"/>.</summary>
[Module]
public partial class StatefulGainNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => StatefulGainSubModel.Call(input);
}

/// <summary>One stateful model handle called twice — the two calls share its one weight and its
/// one piece of state.</summary>
[Module]
public partial class StatefulGainCalledTwiceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = StatefulGainSubModel.Model();
        return m.Call(input) + m.Call(input);
    }
}

/// <summary>A gain module taking a hyperparameter, so its model struct is wider than
/// <see cref="Rank1GainSubModel"/>'s and it owns two parameters rather than one.</summary>
[Module]
public partial class HyperScaledGainSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> input, [Hyper] Scalar<float32> scale)
        => input * Ones.Init([Scalar(2L)]) * scale + Zeros.Init([Scalar(2L)]);
}

/// <summary><see cref="HyperScaledGainSubModel"/> called directly — the baseline for what
/// indexing it out of a sequence must produce.</summary>
[Module]
public partial class HyperScaledGainNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => HyperScaledGainSubModel.Model(Scalar(2f)).Call(input);
}

/// <summary>A sequence of two different modules, indexed at the second one — which owns two
/// parameters where the first owns one.</summary>
[Module]
public partial class HeterogeneousHyperSequenceAtOneModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => ModelSequence.Create<Model<Tensor<float32>, Tensor<float32>>>(
               Rank1GainSubModel.Model(), HyperScaledGainSubModel.Model(Scalar(2f)))[Scalar(1L)].Call(input);
}

/// <summary>A reference to a parameter of a model that is never called, so the graph holds the
/// reference with no definition behind it.</summary>
[Module]
public partial class RefWithoutDefinitionModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = Rank1GainSubModel.Model();
        return input + m.GetTrainableParam<float32>([1], rank: 1) * Scalar(0f);
    }
}

/// <summary>A model called inside a 3-trip loop body — its parameter's template carries the
/// loop's generalized slot. The naming baseline for <see cref="Rank1GainRefInLoopModel"/>.</summary>
[Module]
public partial class Rank1GainInLoopNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = Rank1GainSubModel.Model();
        var x = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            x = m.Call(x);
            ctx.ContinueWhile(Scalar(true));
        }
        return x;
    }
}

/// <summary><see cref="Rank1GainInLoopNoRefModel"/> plus a read-only reference taken inside the
/// loop body, contributing nothing to the output.</summary>
[Module]
public partial class Rank1GainRefInLoopModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var m = Rank1GainSubModel.Model();
        var x = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            x = m.Call(x) + m.GetTrainableParam<float32>([1], rank: 1) * Scalar(0f);
            ctx.ContinueWhile(Scalar(true));
        }
        return x;
    }
}

/// <summary>Calls whatever model it is handed, so its callee's parameters arrive through a
/// <c>[Hyper] Model&lt;&gt;</c> rather than being created in its own body.</summary>
[Module]
public partial class HyperModelHost
{
    public static Tensor<float32> Inline(Tensor<float32> input,
        [Hyper] Model<Tensor<float32>, Tensor<float32>> inner) => inner.Call(input);
}

/// <summary><see cref="Rank1GainSubModel"/> reached through a <c>[Hyper] Model&lt;&gt;</c> — the
/// same parameter and the same forward as calling it directly.</summary>
[Module]
public partial class HyperModelGainModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => HyperModelHost.Model(Rank1GainSubModel.Model()).Call(input);
}

/// <summary><see cref="Rank1GainSubModel"/> reached through a <c>[Hyper] Model&lt;&gt;</c> of a host
/// taken out of a <c>ModelSequence</c> at a constant position.</summary>
[Module]
public partial class HyperModelGainFromSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Create(HyperModelHost.Model(Rank1GainSubModel.Model()));
        return seq[Scalar(0L)].Call(input);
    }
}

/// <summary>The one <see cref="Rank1GainSubModel"/> left in a <c>ModelSequence</c> after the
/// other is removed — the baseline for <see cref="HyperModelGainFromErasedSequenceModel"/>.</summary>
[Module]
public partial class GainFromErasedSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Create(Rank1GainSubModel.Model(), Rank1GainSubModel.Model())
            .RemoveAt(Scalar(0L));
        return seq[Scalar(0L)].Call(input);
    }
}

/// <summary><see cref="Rank1GainSubModel"/> reached through a <c>[Hyper] Model&lt;&gt;</c> of the
/// one host left in a <c>ModelSequence</c> after the other is removed.</summary>
[Module]
public partial class HyperModelGainFromErasedSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Create(HyperModelHost.Model(Rank1GainSubModel.Model()),
                                       HyperModelHost.Model(Rank1GainSubModel.Model()))
            .RemoveAt(Scalar(0L));
        return seq[Scalar(0L)].Call(input);
    }
}

/// <summary>Two <see cref="Rank1GainSubModel"/>s reached out of a <c>ModelSequence</c> indexed by
/// the loop's iteration index — the baseline for <see cref="HyperModelGainFromDynamicSequenceModel"/>,
/// with no host indirection.</summary>
[Module]
public partial class GainFromDynamicSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Create(Rank1GainSubModel.Model(), Rank1GainSubModel.Model());
        var x = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            x = seq[ctx.IterationIndex].Call(x);
        return x;
    }
}

/// <summary><see cref="GainFromDynamicSequenceModel"/> with each element wrapped in a
/// <see cref="HyperModelHost"/>, so the sub-model arrives through a <c>[Hyper] Model&lt;&gt;</c> of
/// a host the loop index picks out of the sequence.</summary>
[Module]
public partial class HyperModelGainFromDynamicSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Create(HyperModelHost.Model(Rank1GainSubModel.Model()),
                                       HyperModelHost.Model(Rank1GainSubModel.Model()));
        var x = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            x = seq[ctx.IterationIndex].Call(x);
        return x;
    }
}

/// <summary><see cref="HyperModelGainFromDynamicSequenceModel"/> with the hosts appended to an
/// empty <c>ModelSequence</c> rather than constructed into one.</summary>
[Module]
public partial class HyperModelGainFromAppendedSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Empty(HyperModelHost.Model(Rank1GainSubModel.Model()))
            .Append(HyperModelHost.Model(Rank1GainSubModel.Model()))
            .Append(HyperModelHost.Model(Rank1GainSubModel.Model()));
        var x = input;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L)))
            x = seq[ctx.IterationIndex].Call(x);
        return x;
    }
}

/// <summary><see cref="HeterogeneousSequenceAtOneModel"/> extended inside a loop, so the sequence
/// leaves the loop as a loop variable and cannot be laid out in order — the position is a constant
/// but the element it names is not reachable.</summary>
[Module]
public partial class HeterogeneousThroughLoopSequenceModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var seq = ModelSequence.Create<Model<Tensor<float32>, Tensor<float32>>>(
            Rank1GainSubModel.Model(), TwoParamGainSubModel.Model());
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L))) seq = seq.Append(Rank1GainSubModel.Model());
        return seq[Scalar(1L)].Call(input);
    }
}

/// <summary>A rank-1 gain plus a second, zero-valued parameter, so which of two bodies a call
/// reached shows in the parameter ids and not only in the name they are filed under.</summary>
[Module]
public partial class TwoParamGainSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => input * Ones.Init([Scalar(2L)]) + Zeros.Init([Scalar(2L)]);
}

/// <summary><see cref="TwoParamGainSubModel"/> called plainly — what indexing element 1 of
/// <see cref="HeterogeneousSequenceAtOneModel"/> should reach.</summary>
[Module]
public partial class TwoParamGainNoRefModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => TwoParamGainSubModel.Call(input);
}

/// <summary>Two different modules of one <c>Model&lt;&gt;</c> signature in a sequence, indexed at
/// the second. A <c>ModelSequence</c> names element 0's module, so the element indexed and the
/// module named disagree.</summary>
[Module]
public partial class HeterogeneousSequenceAtOneModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => ModelSequence.Create<Model<Tensor<float32>, Tensor<float32>>>(
               Rank1GainSubModel.Model(), TwoParamGainSubModel.Model())[Scalar(1L)].Call(input);
}

/// <summary>A stateful module whose own body calls another stateful module, so a call to it
/// closes two nested state scopes rather than one.</summary>
[Module]
public partial class NestedStatefulSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var mine = InitRunningMean.Init(t.ShapeTensor());
        Globals.StateUpdate(mine, mine + Scalar(10f));
        return StatefulGainSubModel.Call(t) + mine;
    }
}

/// <summary><see cref="NestedStatefulSubModel"/> called twice — the inner module's own call-site
/// markers must not be mistaken for the outer parameter's.</summary>
[Module]
public partial class NestedStatefulCalledTwiceModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = NestedStatefulSubModel.Model();
        return m.Call(t) + m.Call(t);
    }
}

/// <summary>Two state parameters owned by one module, so one call site closes two updates at
/// once and each has to be tracked against its own field.</summary>
[Module]
public partial class TwoStateFieldsSubModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var a = InitRunningMean.Init(t.ShapeTensor());
        var b = InitRunningMean.Init(t.ShapeTensor());
        Globals.StateUpdate(a, a + Scalar(1f));
        Globals.StateUpdate(b, b + Scalar(100f));
        return t * Ones.Init([Scalar(2L)]) + a + b;
    }
}

/// <summary><see cref="TwoStateFieldsSubModel"/> called twice.</summary>
[Module]
public partial class TwoStateFieldsCalledTwiceModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = TwoStateFieldsSubModel.Model();
        return m.Call(t) + m.Call(t);
    }
}

/// <summary>A stateful model called from both arms of an <c>IfElse</c>, on a condition derived at
/// runtime so neither arm folds away. Only one arm runs, so the two updates are alternatives
/// rather than a sequence and cannot be composed.</summary>
[Module]
public partial class StatefulCalledFromBothIfArmsModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = StatefulGainSubModel.Model();
        return (t.ShapeTensor()[0] > Scalar(0L))
            .IfElse(() => m.Call(t), () => m.Call(t) * Scalar(2f));
    }
}

/// <summary>One stateful model handle called twice with the second call's output discarded, so the
/// call reaches the graph through nothing but the state update it registers.</summary>
[Module]
public partial class StatefulCallDiscardedModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = StatefulGainSubModel.Model();
        var used = m.Call(t);
        _ = m.Call(t);
        return used;
    }
}

/// <summary>A stateful model called once inside a loop body, so its one call site is the loop's
/// and its update registers once for the step, whatever the trip count.</summary>
[Module]
public partial class StatefulCalledOnceInALoopModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = StatefulGainSubModel.Model();
        var x = t;
        foreach (var ctx in LoopAPI.Iterate(Scalar(3L)))
        {
            x = m.Call(x);
            ctx.ContinueWhile(Scalar(true));
        }
        return x;
    }
}

/// <summary>A trainable parameter inside a loop whose trip count is not a compile-time constant, so
/// the loop is not unrolled before the training graph is built.</summary>
[Module]
public partial class GainInRolledLoopModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var x = t;
        foreach (var ctx in LoopAPI.Iterate(t.ShapeTensor()[0])) x = x * Ones.Init([Scalar(2L)]);
        return x;
    }
}

/// <summary>Module-owned state updated inside a loop whose trip count is not a compile-time
/// constant, so the update's value stays inside the body the training graph cannot unroll.</summary>
[Module]
public partial class StatefulGainInRolledLoopModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = StatefulGainSubModel.Model();
        var x = t;
        foreach (var ctx in LoopAPI.Iterate(t.ShapeTensor()[0]))
        {
            x = m.Call(x);
            ctx.ContinueWhile(Scalar(true));
        }
        return x;
    }
}

/// <summary>The same loop with a constant trip count, which unrolls and trains.</summary>
[Module]
public partial class GainInConstantTripLoopModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var x = t;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L))) x = x * Ones.Init([Scalar(2L)]);
        return x;
    }
}

/// <summary>Draws one uniform sample of its own, so every model built from it owns an RNG
/// feed.</summary>
[Module]
public partial class DrawingSub
{
    public static Tensor<float32> Inline(Tensor<float32> t) => t + Globals.RandomUniform(Vector(2L));
}

/// <summary>Calls the model handed to it as a hyperparameter, so that call site only enters the
/// graph when this body is spliced.</summary>
[Module]
public partial class DrawViaHyperModel
{
    public static Tensor<float32> Inline(
        Tensor<float32> t, [Hyper] Model<Tensor<float32>, Tensor<float32>> m)
        => m.Call(t);
}

/// <summary>One <see cref="DrawingSub"/> model called twice, once directly and once from inside
/// <see cref="DrawViaHyperModel"/>, so the two call sites are inlined a pass apart.</summary>
[Module]
public partial class DrawTwiceOneCallThroughHyperModel
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = DrawingSub.Model();
        return DrawViaHyperModel.Call(m, t) - m.Call(t);
    }
}

/// <summary>Two <see cref="DrawingSub"/> models called one after the other — one RNG stream
/// each.</summary>
[Module]
public partial class DrawTwoDirect
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var a = DrawingSub.Model();
        var b = DrawingSub.Model();
        return b.Call(a.Call(t));
    }
}

/// <summary><see cref="DrawTwoDirect"/> with the two models reached out of a
/// <c>ModelSequence</c>.</summary>
[Module]
public partial class DrawTwoFromSequence
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Create(DrawingSub.Model(), DrawingSub.Model());
        return seq[Scalar(1L)].Call(seq[Scalar(0L)].Call(t));
    }
}

/// <summary><see cref="DrawTwoDirect"/> with the two models appended to a <c>ModelSequence</c>
/// rather than constructed into one.</summary>
[Module]
public partial class DrawTwoFromAppendedSequence
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Empty(DrawingSub.Model())
            .Append(DrawingSub.Model()).Append(DrawingSub.Model());
        return seq[Scalar(1L)].Call(seq[Scalar(0L)].Call(t));
    }
}

/// <summary><see cref="DrawTwoFromAppendedSequence"/> with the appending done inside a loop, so
/// the sequence itself is a loop variable rather than a value this graph lays out.</summary>
[Module]
public partial class DrawTwoFromSequenceAppendedInLoop
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Empty(DrawingSub.Model());
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L))) seq = seq.Append(DrawingSub.Model());
        return seq[Scalar(1L)].Call(seq[Scalar(0L)].Call(t));
    }
}

/// <summary>One <see cref="DrawingSub"/> model called on every trip of a loop.</summary>
[Module]
public partial class DrawInLoopDirect
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var m = DrawingSub.Model();
        var x = t;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L))) x = m.Call(x);
        return x;
    }
}

/// <summary><see cref="DrawInLoopDirect"/> with the model picked out of a <c>ModelSequence</c> by
/// the loop's iteration index, so its identity is only known at run time.</summary>
[Module]
public partial class DrawInLoopFromSequence
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Create(DrawingSub.Model(), DrawingSub.Model());
        var x = t;
        foreach (var ctx in LoopAPI.Iterate(Scalar(2L))) x = seq[ctx.IterationIndex].Call(x);
        return x;
    }
}

/// <summary>The last <see cref="DrawingSub"/> model of a <c>ModelSequence</c>, named by counting
/// back from the end.</summary>
[Module]
public partial class DrawFromSequenceAtNegativeIndex
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Create(DrawingSub.Model(), DrawingSub.Model());
        return seq[Scalar(-1L)].Call(t);
    }
}

/// <summary>A <see cref="DrawingSub"/> model inserted at the front of a <c>ModelSequence</c> and
/// read back from there — the third created, at the first position.</summary>
[Module]
public partial class DrawFromSequenceAfterInsertAt
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Create(DrawingSub.Model(), DrawingSub.Model())
            .InsertAt(DrawingSub.Model(), Scalar(0L));
        return seq[Scalar(0L)].Call(t);
    }
}

/// <summary>Two <see cref="DrawingSub"/> models left in a <c>ModelSequence</c> after one is
/// removed.</summary>
[Module]
public partial class DrawTwoFromErasedSequence
{
    public static Tensor<float32> Inline(Tensor<float32> t)
    {
        var seq = ModelSequence.Create(DrawingSub.Model(), DrawingSub.Model(), DrawingSub.Model())
            .RemoveAt(Scalar(0L));
        return seq[Scalar(1L)].Call(seq[Scalar(0L)].Call(t));
    }
}

/// <summary>Two <see cref="DrawingSub"/> models picked out of a <c>ModelSequence</c> at positions
/// only known at run time — different models on the two calls, whichever way round.</summary>
[Module]
public partial class DrawTwoAtRuntimePositions
{
    public static Tensor<float32> Inline(Tensor<float32> t, Scalar<int64> i)
    {
        var seq = ModelSequence.Create(DrawingSub.Model(), DrawingSub.Model());
        return seq[i].Call(t) - seq[Scalar(1L) - i].Call(t);
    }
}

/// <summary><see cref="DrawTwoAtRuntimePositions"/> reached through another module, so the models'
/// ids carry that module's prefix.</summary>
[Module]
public partial class DrawTwoAtRuntimePositionsNested
{
    public static Tensor<float32> Inline(Tensor<float32> t, Scalar<int64> i)
        => DrawTwoAtRuntimePositions.Model().Call(t, i);
}

/// <summary>
/// An initializer that states its shape nowhere the pipeline can read it: it takes no input, so
/// there is no shape vector, and returns <c>Tensor</c> rather than <c>Scalar</c>, so the declared
/// rank is unknown — the shape lives only inside the body.
/// </summary>
[TrainableParamInitializer]
public static partial class InitShapelessZeros
{
    public static Tensor<float32> Inline() => Globals.TensorFill(Vector(4L), 0.5f);
}

[Module]
public partial class ShapelessInitModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => input * InitShapelessZeros.Init();
}

/// <summary>An initializer whose Inline hands its input straight back.</summary>
[TrainableParamInitializer]
public static partial class InitIdentityScalar
{
    public static Scalar<float32> Inline(Scalar<float32> value) => value;
}

[Module]
public partial class IdentityInitModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => input * InitIdentityScalar.Init(Scalar(2f));
}

/// <summary>The shape-first form of <see cref="InitIdentityScalar"/>'s pass-through body.</summary>
[TrainableParamInitializer]
public static partial class InitIdentityShaped
{
    public static Tensor<float32> Inline(Vector<int64> shape, Tensor<float32> seed) => seed;
}

[Module]
public partial class IdentityShapedInitModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
        => input * InitIdentityShaped.Init(Vector(2L), Globals.TensorFill(Vector(2L), 2.0f));
}

[Module]
public partial class ScalarMultiplyWithBatchNormModel
{
    public static Tensor<float32> Inline(Tensor<float32> input)
    {
        var scalarShape = Vector(1L);

        var runningMean = InitBnRunningMean.Init(scalarShape);
        var runningVar = InitBnRunningVar.Init(scalarShape);

        Vector<int64> weightShape = Vector(1L);
        var weight = InitScalarWeight.Init(weightShape);

        Vector<int64> batchAxis = [Scalar(0L)];
        var batchMean = input.Reduce(ReduceKind.Mean, batchAxis, keepDims: false);
        var diff = input - batchMean;
        var batchVar = (diff * diff).Reduce(ReduceKind.Mean, batchAxis, keepDims: false);

        var epsilon = Scalar(1e-5f);
        var normalized = diff / (batchVar + epsilon).Sqrt();

        var momentum = Scalar(0.1f);
        var batchMeanVec = batchMean.Reshape(scalarShape);
        var batchVarVec = batchVar.Reshape(scalarShape);
        var updatedMean = runningMean * (Scalar(1f) - momentum) + batchMeanVec * momentum;
        var updatedVar = runningVar * (Scalar(1f) - momentum) + batchVarVec * momentum;
        Globals.StateUpdate(runningMean, updatedMean);
        Globals.StateUpdate(runningVar, updatedVar);

        return normalized * weight;
    }
}

