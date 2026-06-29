using System;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Modules.Initializers;
using Shorokoo.Onnx;
using static Shorokoo.Globals;

namespace Shorokoo.Modules.Layers;

/// <summary>The point nonlinearity of a vanilla (Elman) <see cref="Recurrent.RNN"/> cell.</summary>
/// <remarks>
/// Maps onto the ONNX <c>RNN</c> op's <c>activations</c> string attribute.
/// <see cref="Tanh"/> is the default (and the op's own default, so it is passed as
/// the no-activations fast path that keeps the recurrence trainable);
/// <see cref="Relu"/> is a non-default activation that <b>builds and runs (forward
/// inference) but is not differentiable</b> — back-propagation through time throws
/// AD003 at lowering. See <see cref="Recurrent.RNN"/>.
/// </remarks>
public enum RnnNonlinearity
{
    /// <summary>Hyperbolic-tangent recurrence <c>h_t = tanh(W·x_t + R·h_{t-1} + b)</c> (default; trainable).</summary>
    Tanh,
    /// <summary>ReLU recurrence <c>h_t = relu(W·x_t + R·h_{t-1} + b)</c>. Forward / inference-grade only (BPTT throws AD003).</summary>
    Relu,
}

/// <summary>Sequence-processing direction of a <see cref="Recurrent.RNN"/>, mirroring the ONNX <c>RNN</c> op's <c>direction</c> attribute.</summary>
/// <remarks>
/// <see cref="Forward"/> and <see cref="Reverse"/> are single-direction and
/// <b>trainable</b>; <see cref="Bidirectional"/> runs both directions and
/// concatenates their per-step outputs on the feature axis, but is <b>forward /
/// inference-grade only</b> — its back-propagation through time throws AD003 at
/// lowering. See <see cref="Recurrent.RNN"/>.
/// </remarks>
public enum RnnDirection
{
    /// <summary>Process the sequence from first step to last (default; trainable).</summary>
    Forward,
    /// <summary>Process the sequence from last step to first (trainable).</summary>
    Reverse,
    /// <summary>Run a forward and a reverse pass and concatenate them on the feature axis (<c>D = 2</c>). Inference-grade only (BPTT throws AD003).</summary>
    Bidirectional,
}

/// <summary>
/// Recurrent-layer graph-building helpers. The first member is the vanilla
/// (Elman) <see cref="RNN"/>; LSTM and GRU will join this class and reuse the
/// shared recurrent contract (weight ownership over the ONNX <c>[num_dir, …]</c>
/// layout, the <see cref="RecurrentUniform"/> init, the nonlinearity/direction
/// knob mapping, the <c>numLayers</c> stacking, and the <c>batchFirst</c>
/// in-graph transpose).
/// </summary>
/// <remarks>
/// <para>
/// Like <see cref="Convolution"/> and <see cref="Pooling"/>, these are
/// <b>static, plain-C#-argument</b> graph-building helpers, not <c>[Module]</c>s.
/// Every knob is shape-determining or topology-determining and is therefore baked
/// at build time regardless (so the <c>[Hyper]</c> machinery would buy nothing),
/// and the <c>nonlinearity</c>/<c>direction</c> enums cannot be expressed as the
/// scalar-only <c>[Hyper]</c> type anyway. The owned weights are still created via
/// <see cref="RecurrentUniform"/>'s <c>Init</c> — which emits trainable-parameter
/// nodes into the composed graph exactly as <see cref="Convolution"/>'s <c>Conv</c>
/// owns its weight — so the (single-direction, tanh) layer trains end-to-end.
/// </para>
/// </remarks>
public static class Recurrent
{
    /// <summary>
    /// Vanilla (Elman) recurrent layer: <c>h_t = act(W·x_t + R·h_{t-1} + b)</c>,
    /// with <c>act</c> tanh or relu, over a multi-layer / uni- or bi-directional
    /// stack. Mirrors PyTorch <c>nn.RNN</c> (defaults: tanh, forward, 1 layer,
    /// sequence-first, bias on; zeroed initial state <c>h_0</c>).
    /// </summary>
    /// <param name="x">
    /// Input sequence. <c>[L, N, inputSize]</c> by default, or <c>[N, L, inputSize]</c>
    /// when <paramref name="batchFirst"/> is true. <c>inputSize</c> is read in-graph
    /// from the last axis (the layer is lazy in input size, like
    /// <see cref="Convolution"/>'s <c>Conv</c> <c>inChannels</c>).
    /// </param>
    /// <param name="hiddenSize">Hidden state size <c>H</c> (the per-direction feature width). Required, like PyTorch.</param>
    /// <param name="nonlinearity">
    /// Cell activation. <see cref="RnnNonlinearity.Tanh"/> (default) is trainable;
    /// <see cref="RnnNonlinearity.Relu"/> is <b>forward / inference-grade only</b>
    /// (its back-propagation through time throws AD003 at lowering).
    /// </param>
    /// <param name="direction">
    /// <see cref="RnnDirection.Forward"/> (default) and <see cref="RnnDirection.Reverse"/>
    /// are trainable; <see cref="RnnDirection.Bidirectional"/> (<c>D = 2</c>) is
    /// <b>forward / inference-grade only</b> (BPTT throws AD003).
    /// </param>
    /// <param name="numLayers">Number of stacked RNN layers (each layer's output sequence feeds the next). Default 1.</param>
    /// <param name="batchFirst">
    /// When true, <paramref name="x"/> and the returned <c>y</c> are
    /// batch-first (<c>[N, L, …]</c>); realized by an in-graph transpose around a
    /// <c>layout=0</c> op (ORT-CPU rejects <c>layout=1</c> and autodiff only supports
    /// <c>layout=0</c>). The returned <c>hN</c> stays <c>[D·numLayers, N, H]</c>
    /// regardless, matching PyTorch.
    /// </param>
    /// <param name="bias">
    /// When true (default) a single trainable bias <c>[D, H]</c> per layer is owned
    /// and fed to the op as <c>B = concat(bias, zeros)</c> on axis 1 (so the ONNX
    /// input-bias <c>Wb</c> carries it and the recurrent-bias <c>Rb</c> is 0 — the
    /// two ONNX/PyTorch biases collapse into one, as Keras/Flax do). When false the
    /// op is given no bias.
    /// </param>
    /// <returns>
    /// <c>y</c> — the full output sequence in PyTorch layout
    /// <c>[L, N, D·H]</c> (or <c>[N, L, D·H]</c> when <paramref name="batchFirst"/>),
    /// i.e. every step's hidden state; and <c>hN</c> — the final hidden state per
    /// direction and layer, shaped <c>[D·numLayers, N, H]</c>. Slice <c>y[-1]</c> or
    /// read <c>hN</c> for the "last output only" use.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Per layer and with <c>D = direction == Bidirectional ? 2 : 1</c>, the owned
    /// trainable parameters are <c>W [D, H, in]</c> (input→hidden),
    /// <c>R [D, H, H]</c> (hidden→hidden), and <c>bias [D, H]</c>, all
    /// <see cref="RecurrentUniform"/>-initialized (PyTorch's <c>U(−1/√H, 1/√H)</c>).
    /// The single gate means there is no PyTorch↔ONNX gate-reorder (LSTM/GRU add
    /// only that). The default initial state <c>h_0</c> is zero (an omitted op
    /// input).
    /// </para>
    /// <para>
    /// <b>Autodiff caveat (loud).</b> Only single-direction (forward or reverse),
    /// <see cref="RnnNonlinearity.Tanh"/>, <c>layout=0</c> RNNs are trainable.
    /// <see cref="RnnNonlinearity.Relu"/> and <see cref="RnnDirection.Bidirectional"/>
    /// build and run for <b>forward inference</b> but throw AD003 in
    /// back-propagation through time; they are intentionally exposed (not gated),
    /// documented as inference-grade. RNN has no QEE step values — value checks run
    /// on the ORT backend.
    /// </para>
    /// </remarks>
    public static (Tensor<float32> y, Tensor<float32> hN) RNN(
        Tensor<float32> x,
        long hiddenSize,
        RnnNonlinearity nonlinearity = RnnNonlinearity.Tanh,
        RnnDirection direction = RnnDirection.Forward,
        int numLayers = 1,
        bool batchFirst = false,
        bool bias = true)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers must be at least 1.");

        long d = direction == RnnDirection.Bidirectional ? 2L : 1L;
        var onnxDir = direction switch
        {
            RnnDirection.Forward => RNNDirection.Forward,
            RnnDirection.Reverse => RNNDirection.Reverse,
            RnnDirection.Bidirectional => RNNDirection.Bidirectional,
            _ => throw new ArgumentOutOfRangeException(nameof(direction), direction, null),
        };

        // Tanh is the op default (pass null → keeps the autodiff default-activations
        // fast path); Relu is the only non-default and is inference-only.
        string[]? activations = nonlinearity switch
        {
            RnnNonlinearity.Tanh => null,
            RnnNonlinearity.Relu => d == 2L ? new[] { "Relu", "Relu" } : new[] { "Relu" },
            _ => throw new ArgumentOutOfRangeException(nameof(nonlinearity), nonlinearity, null),
        };

        // batchFirst is realized by transposing into [L, N, in]; the op always runs
        // at layout=0 and the final Y is transposed back to [N, L, D*H] below.
        var curX = batchFirst ? x.Transpose(1L, 0L, 2L) : x;

        var hScalar = Scalar(hiddenSize);
        var dScalar = Scalar(d);
        Tensor<float32>? y = null;
        Tensor<float32>? hN = null;

        for (int layer = 0; layer < numLayers; layer++)
        {
            // in_0 is read in-graph from x's last axis; in_ℓ = D*H for stacked layers.
            Scalar<int64> inSize = layer == 0 ? curX.DimTensor(-1) : Scalar(d * hiddenSize);

            var w = RecurrentUniform.Init([dScalar, hScalar, inSize], hScalar);   // [D, H, in]
            var r = RecurrentUniform.Init([dScalar, hScalar, hScalar], hScalar);  // [D, H, H]

            Tensor<float32>? b = null;
            if (bias)
            {
                // Single owned bias [D, H]; fed as B = concat(bias, zeros) on axis 1
                // ([D, 2H]) so Wb = bias and Rb = 0.
                var biasParam = RecurrentUniform.Init([dScalar, hScalar], hScalar);    // [D, H]
                var rbZeros = TensorFill((Vector<int64>)[dScalar, hScalar], 0.0f);
                b = biasParam.Concat(1L, rbZeros);                            // [D, 2H]
            }

            var (yVar, yhVar) = OnnxOp.Rnn(curX, w, r, b, null, null,
                null, null, activations, null, onnxDir, hiddenSize, false);

            var yLayer = (Tensor<float32>)yVar;    // [L, D, N, H]
            var yhLayer = (Tensor<float32>)yhVar;  // [D, N, H]

            // Reshape Y [L, D, N, H] -> [L, N, D*H] for the next layer's X / the return.
            // Transpose to [L, N, D, H] first so the D and H axes are adjacent, then fold.
            var lScalar = yLayer.DimTensor(0);
            var nScalar = yLayer.DimTensor(2);
            var yLNDH = yLayer.Transpose(0L, 2L, 1L, 3L);  // [L, N, D, H]
            curX = yLNDH.Reshape((Vector<int64>)[lScalar, nScalar, dScalar * hScalar]); // [L, N, D*H]
            y = curX;

            // Collect Y_h, concatenating on the leading axis -> [D*numLayers, N, H].
            hN = hN is null ? yhLayer : hN.Value.Concat(0L, yhLayer);
        }

        // y / hN are non-null: numLayers >= 1 guarantees at least one iteration.
        var yOut = batchFirst ? y!.Value.Transpose(1L, 0L, 2L) : y!.Value;  // [N, L, D*H] when batchFirst
        return (yOut, hN!.Value);
    }

    /// <summary>
    /// Long Short-Term Memory layer over a multi-layer / uni- or bi-directional
    /// stack, with the standard gate recurrence
    /// <c>i = σ(W_i·x + R_i·h + b_i)</c>, <c>o = σ(W_o·x + R_o·h + b_o)</c>,
    /// <c>f = σ(W_f·x + R_f·h + b_f)</c>, <c>c̃ = tanh(W_c·x + R_c·h + b_c)</c>,
    /// <c>C_t = f ⊙ C_{t-1} + i ⊙ c̃</c>, <c>H_t = o ⊙ tanh(C_t)</c> (fixed
    /// sigmoid gate / tanh cell activations — there is no nonlinearity knob).
    /// Mirrors PyTorch <c>nn.LSTM</c> (defaults: forward, 1 layer, sequence-first,
    /// bias on; zeroed initial state <c>h_0</c>/<c>c_0</c>).
    /// </summary>
    /// <param name="x">
    /// Input sequence. <c>[L, N, inputSize]</c> by default, or <c>[N, L, inputSize]</c>
    /// when <paramref name="batchFirst"/> is true. <c>inputSize</c> is read in-graph
    /// from the last axis (the layer is lazy in input size, like
    /// <see cref="Convolution"/>'s <c>Conv</c> <c>inChannels</c>).
    /// </param>
    /// <param name="hiddenSize">Hidden state size <c>H</c> (the per-direction feature width). Required, like PyTorch.</param>
    /// <param name="direction">
    /// <see cref="RnnDirection.Forward"/> (default) and <see cref="RnnDirection.Reverse"/>
    /// are trainable; <see cref="RnnDirection.Bidirectional"/> (<c>D = 2</c>) is
    /// <b>forward / inference-grade only</b> (BPTT throws AD003 at lowering). The
    /// flag is exposed (not gated) so a bidirectional <i>inference</i> / ONNX-export
    /// model is reachable; training one raises a loud, attributable exception.
    /// </param>
    /// <param name="numLayers">Number of stacked LSTM layers (each layer's output sequence feeds the next). Default 1.</param>
    /// <param name="batchFirst">
    /// When true, <paramref name="x"/> and the returned <c>y</c> are
    /// batch-first (<c>[N, L, …]</c>); realized by an in-graph transpose around a
    /// <c>layout=0</c> op (ORT-CPU rejects <c>layout=1</c> and autodiff only supports
    /// <c>layout=0</c>). The returned <c>hN</c>/<c>cN</c> stay <c>[D·numLayers, N, H]</c>
    /// regardless, matching PyTorch.
    /// </param>
    /// <param name="bias">
    /// When true (default) a single trainable bias <c>[D, 4H]</c> per layer is owned
    /// and fed to the op as <c>B = concat(bias, zeros)</c> on axis 1 (<c>[D, 8H]</c>,
    /// so the ONNX input-bias <c>Wb</c> carries it and the recurrent-bias <c>Rb</c>
    /// is 0 — the two ONNX/PyTorch biases collapse into one, as Keras/Flax do). When
    /// false the op is given no bias.
    /// </param>
    /// <returns>
    /// <c>y</c> — the full output sequence in PyTorch layout
    /// <c>[L, N, D·H]</c> (or <c>[N, L, D·H]</c> when <paramref name="batchFirst"/>),
    /// i.e. every step's hidden state; <c>hN</c> — the final hidden state per
    /// direction and layer, shaped <c>[D·numLayers, N, H]</c>; and <c>cN</c> — the
    /// final cell state, same shape. Slice <c>y[-1]</c> or read <c>hN</c> for the
    /// "last output only" use; ignore <c>hN</c>/<c>cN</c> for the "sequence only" use
    /// (the tuple covers Keras <c>return_sequences</c>/<c>return_state</c> flexibly).
    /// </returns>
    /// <remarks>
    /// <para>
    /// Per layer and with <c>D = direction == Bidirectional ? 2 : 1</c>, the owned
    /// trainable parameters are <c>W [D, 4H, in]</c> (input→hidden),
    /// <c>R [D, 4H, H]</c> (hidden→hidden), and a single bias <c>[D, 4H]</c>, all
    /// <see cref="RecurrentUniform"/>-initialized with the explicit hidden size
    /// (PyTorch's <c>U(−1/√H, 1/√H)</c>, not <c>1/√(4H)</c>). The four gate blocks
    /// are packed in the ONNX-native <c>i, o, f, c</c> order — the only layout
    /// <see cref="OnnxOp.Lstm"/> understands — with <b>no</b> reorder shim. The
    /// default initial states <c>h_0</c>/<c>c_0</c> are zero (omitted op inputs;
    /// peephole <c>P</c> is null).
    /// </para>
    /// <para>
    /// <b>Port note (gate order).</b> A Shorokoo <c>LSTM</c> weight is in ONNX
    /// <c>i, o, f, c</c> gate order, whereas PyTorch <c>nn.LSTM</c> packs
    /// <c>i, f, g(=c), o</c>. Because the init is uniform across gates, the gate
    /// order is <b>unobservable</b> at initialization (permuting the gate blocks of
    /// a uniform-random tensor yields a statistically identical tensor), so there is
    /// no training-dynamics or correctness difference for a from-scratch model. The
    /// reorder only matters when importing pretrained PyTorch weights (out of scope):
    /// on import, permute the <c>4H</c> rows from <c>i,f,g,o</c> to <c>i,o,f,g</c>
    /// (and split + sum PyTorch's two <c>4H</c> biases into the single <c>4H</c>
    /// owned bias).
    /// </para>
    /// <para>
    /// <b>Autodiff caveat (loud).</b> Only single-direction (forward or reverse),
    /// <c>layout=0</c>, default-activation LSTMs are trainable; this is the trainable
    /// corner and is what we wire. <see cref="RnnDirection.Bidirectional"/> builds
    /// and runs for <b>forward inference</b> but throws AD003 in back-propagation
    /// through time; it is intentionally exposed (not gated), documented as
    /// inference-grade. Peephole, <c>input_forget</c>, <c>clip</c>, custom
    /// activations and variable-length <c>sequence_lens</c> are not exposed (all
    /// throw AD003 in BPTT). LSTM has no QEE step values — value checks run on the
    /// ORT backend.
    /// </para>
    /// </remarks>
    public static (Tensor<float32> y, Tensor<float32> hN, Tensor<float32> cN) LSTM(
        Tensor<float32> x,
        long hiddenSize,
        RnnDirection direction = RnnDirection.Forward,
        int numLayers = 1,
        bool batchFirst = false,
        bool bias = true)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers must be at least 1.");

        long d = direction == RnnDirection.Bidirectional ? 2L : 1L;
        var onnxDir = direction switch
        {
            RnnDirection.Forward => LSTMDirection.Forward,
            RnnDirection.Reverse => LSTMDirection.Reverse,
            RnnDirection.Bidirectional => LSTMDirection.Bidirectional,
            _ => throw new ArgumentOutOfRangeException(nameof(direction), direction, null),
        };

        // batchFirst is realized by transposing into [L, N, in]; the op always runs
        // at layout=0 and the final Y is transposed back to [N, L, D*H] below.
        var curX = batchFirst ? x.Transpose(1L, 0L, 2L) : x;

        var hScalar = Scalar(hiddenSize);
        var fourH = Scalar(4L * hiddenSize);  // 4 gate blocks (i, o, f, c) per direction
        var dScalar = Scalar(d);
        Tensor<float32>? y = null;
        Tensor<float32>? hN = null;
        Tensor<float32>? cN = null;

        for (int layer = 0; layer < numLayers; layer++)
        {
            // in_0 is read in-graph from x's last axis; in_ℓ = D*H for stacked layers.
            Scalar<int64> inSize = layer == 0 ? curX.DimTensor(-1) : Scalar(d * hiddenSize);

            // Gate-packed weights in ONNX-native i,o,f,c order. The bound is keyed on
            // the true hidden size H (not 4H), matching PyTorch's U(-1/√H, 1/√H).
            var w = RecurrentUniform.Init([dScalar, fourH, inSize], hScalar);   // [D, 4H, in]
            var r = RecurrentUniform.Init([dScalar, fourH, hScalar], hScalar);  // [D, 4H, H]

            Tensor<float32>? b = null;
            if (bias)
            {
                // Single owned bias [D, 4H]; fed as B = concat(bias, zeros) on axis 1
                // ([D, 8H]) so Wb = bias and Rb = 0.
                var biasParam = RecurrentUniform.Init([dScalar, fourH], hScalar);    // [D, 4H]
                var rbZeros = TensorFill((Vector<int64>)[dScalar, fourH], 0.0f);
                b = biasParam.Concat(1L, rbZeros);                                   // [D, 8H]
            }

            // h_0 / c_0 default to zero (omitted inputs); peephole P = null.
            // Default sigmoid/tanh activations (activations: null) keep the trainable
            // autodiff fast path; inputForget = false, layout = 0.
            var (yVar, yhVar, ycVar) = OnnxOp.Lstm(curX, w, r, b, null, null, null, null,
                null, null, null, null, onnxDir, hiddenSize, false, false);

            var yLayer = (Tensor<float32>)yVar;    // [L, D, N, H]
            var yhLayer = (Tensor<float32>)yhVar;  // [D, N, H]
            var ycLayer = (Tensor<float32>)ycVar;  // [D, N, H]

            // Reshape Y [L, D, N, H] -> [L, N, D*H] for the next layer's X / the return.
            // Transpose to [L, N, D, H] first so the D and H axes are adjacent, then fold.
            var lScalar = yLayer.DimTensor(0);
            var nScalar = yLayer.DimTensor(2);
            var yLNDH = yLayer.Transpose(0L, 2L, 1L, 3L);  // [L, N, D, H]
            curX = yLNDH.Reshape((Vector<int64>)[lScalar, nScalar, dScalar * hScalar]); // [L, N, D*H]
            y = curX;

            // Collect Y_h / Y_c, concatenating on the leading axis -> [D*numLayers, N, H].
            hN = hN is null ? yhLayer : hN.Value.Concat(0L, yhLayer);
            cN = cN is null ? ycLayer : cN.Value.Concat(0L, ycLayer);
        }

        // y / hN / cN are non-null: numLayers >= 1 guarantees at least one iteration.
        var yOut = batchFirst ? y!.Value.Transpose(1L, 0L, 2L) : y!.Value;  // [N, L, D*H] when batchFirst
        return (yOut, hN!.Value, cN!.Value);
    }

    /// <summary>
    /// Gated Recurrent Unit layer over a multi-layer / uni- or bi-directional
    /// stack, with the standard gate recurrence
    /// <c>z = σ(W_z·x + R_z·h + b_z)</c> (update gate),
    /// <c>r = σ(W_r·x + R_r·h + b_r)</c> (reset gate),
    /// <c>ĥ = tanh(W_h·x + r ⊙ (R_h·h) + b_h)</c> (candidate, reset-after form;
    /// see <paramref name="linearBeforeReset"/>),
    /// <c>H_t = (1 − z) ⊙ ĥ + z ⊙ H_{t-1}</c> (fixed sigmoid gate / tanh candidate
    /// activations — there is no nonlinearity knob, and there is no separate cell
    /// state). Mirrors PyTorch <c>nn.GRU</c> (defaults: forward, 1 layer,
    /// sequence-first, bias on, reset-after; zeroed initial state <c>h_0</c>).
    /// </summary>
    /// <param name="x">
    /// Input sequence. <c>[L, N, inputSize]</c> by default, or <c>[N, L, inputSize]</c>
    /// when <paramref name="batchFirst"/> is true. <c>inputSize</c> is read in-graph
    /// from the last axis (the layer is lazy in input size, like
    /// <see cref="Convolution"/>'s <c>Conv</c> <c>inChannels</c>).
    /// </param>
    /// <param name="hiddenSize">Hidden state size <c>H</c> (the per-direction feature width). Required, like PyTorch.</param>
    /// <param name="direction">
    /// <see cref="RnnDirection.Forward"/> (default) and <see cref="RnnDirection.Reverse"/>
    /// are trainable; <see cref="RnnDirection.Bidirectional"/> (<c>D = 2</c>) is
    /// <b>forward / inference-grade only</b> (BPTT throws AD003 at lowering). The
    /// flag is exposed (not gated) so a bidirectional <i>inference</i> / ONNX-export
    /// model is reachable; training one raises a loud, attributable exception.
    /// </param>
    /// <param name="numLayers">Number of stacked GRU layers (each layer's output sequence feeds the next). Default 1.</param>
    /// <param name="batchFirst">
    /// When true, <paramref name="x"/> and the returned <c>y</c> are
    /// batch-first (<c>[N, L, …]</c>); realized by an in-graph transpose around a
    /// <c>layout=0</c> op (ORT-CPU rejects <c>layout=1</c> and autodiff only supports
    /// <c>layout=0</c>). The returned <c>hN</c> stays <c>[D·numLayers, N, H]</c>
    /// regardless, matching PyTorch.
    /// </param>
    /// <param name="bias">
    /// When true (default) a single trainable bias <c>[D, 3H]</c> per layer is owned
    /// and fed to the op as <c>B = concat(bias, zeros)</c> on axis 1 (<c>[D, 6H]</c>,
    /// so the ONNX input-bias <c>Wb</c> carries it and the recurrent-bias <c>Rb</c>
    /// is 0 — the two ONNX/PyTorch biases collapse into one, as Keras/Flax do). When
    /// false the op is given no bias. With <paramref name="linearBeforeReset"/> true
    /// (the default) the single <c>Wb</c> bias is numerically equivalent to PyTorch's
    /// <c>b_ih + b_hh</c> sum.
    /// </param>
    /// <param name="linearBeforeReset">
    /// Selects how the reset gate enters the candidate. When true (default) the reset
    /// is applied <b>after</b> the recurrent matmul —
    /// <c>ĥ = tanh(W_h·x + r ⊙ (R_h·h + Rb_h) + Wb_h)</c> — matching PyTorch
    /// <c>nn.GRU</c>, Keras <c>reset_after=True</c>, Flax, and cuDNN (the de-facto
    /// standard). When false the reset is applied <b>before</b> the recurrent matmul —
    /// <c>ĥ = tanh(W_h·x + (r ⊙ h)·R_hᵀ + Rb_h + Wb_h)</c> — the original Cho-et-al.
    /// v1 form and the ONNX op's own default. The two forms are numerically distinct
    /// with the same weights; both are trainable.
    /// </param>
    /// <returns>
    /// <c>y</c> — the full output sequence in PyTorch layout
    /// <c>[L, N, D·H]</c> (or <c>[N, L, D·H]</c> when <paramref name="batchFirst"/>),
    /// i.e. every step's hidden state; and <c>hN</c> — the final hidden state per
    /// direction and layer, shaped <c>[D·numLayers, N, H]</c>. Slice <c>y[-1]</c> or
    /// read <c>hN</c> for the "last output only" use; ignore <c>hN</c> for the
    /// "sequence only" use (the tuple covers Keras
    /// <c>return_sequences</c>/<c>return_state</c> flexibly). Unlike
    /// <see cref="LSTM"/> there is no cell state.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Per layer and with <c>D = direction == Bidirectional ? 2 : 1</c>, the owned
    /// trainable parameters are <c>W [D, 3H, in]</c> (input→hidden),
    /// <c>R [D, 3H, H]</c> (hidden→hidden), and a single bias <c>[D, 3H]</c>, all
    /// <see cref="RecurrentUniform"/>-initialized with the explicit hidden size
    /// (PyTorch's <c>U(−1/√H, 1/√H)</c>, not <c>1/√(3H)</c>). The three gate blocks
    /// are packed in the ONNX-native <c>z, r, h</c> order — the only layout
    /// <see cref="OnnxOp.Gru"/> understands — with <b>no</b> reorder shim. The
    /// default initial state <c>h_0</c> is zero (an omitted op input); there is no
    /// cell state.
    /// </para>
    /// <para>
    /// <b>Port note (gate order).</b> A Shorokoo <c>GRU</c> weight is in ONNX
    /// <c>z, r, h</c> gate order, whereas PyTorch <c>nn.GRU</c> packs <c>r, z, n(=h)</c>
    /// (Keras packs <c>r, z</c>). Because the init is uniform across gates, the gate
    /// order is <b>unobservable</b> at initialization (permuting the gate blocks of a
    /// uniform-random tensor yields a statistically identical tensor), so there is no
    /// training-dynamics or correctness difference for a from-scratch model. The
    /// reorder only matters when importing pretrained PyTorch weights (out of scope):
    /// on import, permute the <c>3H</c> rows from <c>r,z,n</c> to <c>z,r,h</c> (swap
    /// the first two gate blocks; the candidate block stays last), and map PyTorch's
    /// two <c>3H</c> biases onto <c>Wb</c>/<c>Rb</c> — exact for the default
    /// <paramref name="linearBeforeReset"/> = true (reset-after) form.
    /// </para>
    /// <para>
    /// <b>Reset-before-vs-after.</b> The default
    /// <paramref name="linearBeforeReset"/> = true matches PyTorch <c>nn.GRU</c>,
    /// Keras <c>reset_after=True</c>, Flax, and cuDNN numerically — which is what a
    /// porting user expects. The ONNX op's own default is the reset-before form
    /// (<c>linear_before_reset = 0</c>); pass <c>linearBeforeReset: false</c> for that
    /// original-Cho-v1 form.
    /// </para>
    /// <para>
    /// <b>Autodiff caveat (loud).</b> Only single-direction (forward or reverse),
    /// <c>layout=0</c>, default-activation GRUs are trainable; this is the trainable
    /// corner and is what we wire. Both <paramref name="linearBeforeReset"/> forms are
    /// differentiable. <see cref="RnnDirection.Bidirectional"/> builds and runs for
    /// <b>forward inference</b> but throws AD003 in back-propagation through time; it
    /// is intentionally exposed (not gated), documented as inference-grade. <c>clip</c>,
    /// custom activations and variable-length <c>sequence_lens</c> are not exposed (all
    /// throw AD003 in BPTT). GRU has no QEE step values — value checks run on the ORT
    /// backend.
    /// </para>
    /// </remarks>
    public static (Tensor<float32> y, Tensor<float32> hN) GRU(
        Tensor<float32> x,
        long hiddenSize,
        RnnDirection direction = RnnDirection.Forward,
        int numLayers = 1,
        bool batchFirst = false,
        bool bias = true,
        bool linearBeforeReset = true)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "numLayers must be at least 1.");

        long d = direction == RnnDirection.Bidirectional ? 2L : 1L;
        var onnxDir = direction switch
        {
            RnnDirection.Forward => GRUDirection.Forward,
            RnnDirection.Reverse => GRUDirection.Reverse,
            RnnDirection.Bidirectional => GRUDirection.Bidirectional,
            _ => throw new ArgumentOutOfRangeException(nameof(direction), direction, null),
        };

        // batchFirst is realized by transposing into [L, N, in]; the op always runs
        // at layout=0 and the final Y is transposed back to [N, L, D*H] below.
        var curX = batchFirst ? x.Transpose(1L, 0L, 2L) : x;

        var hScalar = Scalar(hiddenSize);
        var threeH = Scalar(3L * hiddenSize);  // 3 gate blocks (z, r, h) per direction
        var dScalar = Scalar(d);
        Tensor<float32>? y = null;
        Tensor<float32>? hN = null;

        for (int layer = 0; layer < numLayers; layer++)
        {
            // in_0 is read in-graph from x's last axis; in_ℓ = D*H for stacked layers.
            Scalar<int64> inSize = layer == 0 ? curX.DimTensor(-1) : Scalar(d * hiddenSize);

            // Gate-packed weights in ONNX-native z,r,h order. The bound is keyed on
            // the true hidden size H (not 3H), matching PyTorch's U(-1/√H, 1/√H).
            var w = RecurrentUniform.Init([dScalar, threeH, inSize], hScalar);   // [D, 3H, in]
            var r = RecurrentUniform.Init([dScalar, threeH, hScalar], hScalar);  // [D, 3H, H]

            Tensor<float32>? b = null;
            if (bias)
            {
                // Single owned bias [D, 3H]; fed as B = concat(bias, zeros) on axis 1
                // ([D, 6H]) so Wb = bias and Rb = 0.
                var biasParam = RecurrentUniform.Init([dScalar, threeH], hScalar);    // [D, 3H]
                var rbZeros = TensorFill((Vector<int64>)[dScalar, threeH], 0.0f);
                b = biasParam.Concat(1L, rbZeros);                                    // [D, 6H]
            }

            // h_0 defaults to zero (omitted input); no cell state. Default sigmoid/tanh
            // activations (activations: null) keep the trainable autodiff fast path;
            // clip = null, layout = 0. linearBeforeReset selects reset-after (true,
            // PyTorch-matching default) vs reset-before (false, the ONNX op default).
            var (yVar, yhVar) = OnnxOp.Gru(curX, w, r, b, null, null,
                null, null, null, null, onnxDir, hiddenSize, false, linearBeforeReset);

            var yLayer = (Tensor<float32>)yVar;    // [L, D, N, H]
            var yhLayer = (Tensor<float32>)yhVar;  // [D, N, H]

            // Reshape Y [L, D, N, H] -> [L, N, D*H] for the next layer's X / the return.
            // Transpose to [L, N, D, H] first so the D and H axes are adjacent, then fold.
            var lScalar = yLayer.DimTensor(0);
            var nScalar = yLayer.DimTensor(2);
            var yLNDH = yLayer.Transpose(0L, 2L, 1L, 3L);  // [L, N, D, H]
            curX = yLNDH.Reshape((Vector<int64>)[lScalar, nScalar, dScalar * hScalar]); // [L, N, D*H]
            y = curX;

            // Collect Y_h, concatenating on the leading axis -> [D*numLayers, N, H].
            hN = hN is null ? yhLayer : hN.Value.Concat(0L, yhLayer);
        }

        // y / hN are non-null: numLayers >= 1 guarantees at least one iteration.
        var yOut = batchFirst ? y!.Value.Transpose(1L, 0L, 2L) : y!.Value;  // [N, L, D*H] when batchFirst
        return (yOut, hN!.Value);
    }

    /// <summary>
    /// Single-step vanilla (Elman) cell: <c>h' = act(W·x + R·h + b)</c>, with
    /// <c>act</c> tanh or relu. Mirrors PyTorch <c>nn.RNNCell</c> — one timestep of
    /// the recurrence, with the previous hidden state handed in and the new one
    /// returned, the building block a user composes into a custom unrolled loop
    /// (scheduled sampling, attention-augmented decoders, beam search).
    /// </summary>
    /// <param name="x">Step input <c>[N, inputSize]</c>. <c>inputSize</c> is read in-graph from the last axis (lazy in input size, like <see cref="RNN"/>).</param>
    /// <param name="h">Previous hidden state <c>[N, hiddenSize]</c>. Required (the user owns the loop and seeds step 0 with an explicit zero tensor).</param>
    /// <param name="hiddenSize">Hidden state size <c>H</c>. Required, like PyTorch.</param>
    /// <param name="nonlinearity">
    /// Cell activation. <see cref="RnnNonlinearity.Tanh"/> (default) is trainable;
    /// <see cref="RnnNonlinearity.Relu"/> is <b>forward / inference-grade only</b>
    /// (its back-propagation through time throws AD003 at lowering) — the same loud,
    /// documented limit <see cref="RNN"/> carries.
    /// </param>
    /// <param name="bias">
    /// When true (default) a single trainable bias <c>[1, H]</c> is owned and fed to
    /// the op as <c>B = concat(bias, zeros)</c> on axis 1 (so <c>Wb</c> carries it and
    /// <c>Rb</c> is 0). When false the op is given no bias.
    /// </param>
    /// <returns>The new hidden state <c>h' [N, hiddenSize]</c> (the <c>num_dir</c> axis stripped), ready to thread straight back as the next step's <paramref name="h"/>.</returns>
    /// <remarks>
    /// <para>
    /// Implemented as one step of the ONNX <c>RNN</c> op at sequence-length 1: <c>x</c>
    /// is unsqueezed to <c>[seq=1, N, in]</c>, <c>h</c> to <c>[num_dir=1, N, H]</c> and
    /// passed as <c>initial_h</c>, and the op's final hidden <c>Y_h</c> is squeezed
    /// back to <c>[N, H]</c>. This reuses the validated gate math, the
    /// <see cref="RecurrentUniform"/> init, and the bias collapse of <see cref="RNN"/>
    /// verbatim — a cell is <b>definitionally</b> one step of the layer.
    /// </para>
    /// <para>
    /// The owned parameters are <c>W [1, H, in]</c>, <c>R [1, H, H]</c> and (when
    /// <paramref name="bias"/>) <c>bias [1, H]</c>, all
    /// <see cref="RecurrentUniform"/>-initialized (PyTorch's <c>U(−1/√H, 1/√H)</c>).
    /// Single-direction only (no <c>num_dir</c>/bidirectional knob — a single-step cell
    /// has no sequence to reverse). The default (tanh) cell is fully trainable;
    /// <see cref="RnnNonlinearity.Relu"/> is inference-only as above.
    /// </para>
    /// </remarks>
    public static Tensor<float32> RNNCell(
        Tensor<float32> x,
        Tensor<float32> h,
        long hiddenSize,
        RnnNonlinearity nonlinearity = RnnNonlinearity.Tanh,
        bool bias = true)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");

        // Tanh is the op default (pass null → keeps the autodiff default-activations
        // fast path); Relu is the only non-default and is inference-only.
        string[]? activations = nonlinearity switch
        {
            RnnNonlinearity.Tanh => null,
            RnnNonlinearity.Relu => new[] { "Relu" },
            _ => throw new ArgumentOutOfRangeException(nameof(nonlinearity), nonlinearity, null),
        };

        var hScalar = Scalar(hiddenSize);
        var dScalar = Scalar(1L);          // single direction
        var inSize = x.DimTensor(-1);      // read in-graph from x's last axis

        var w = RecurrentUniform.Init([dScalar, hScalar, inSize], hScalar);   // [1, H, in]
        var r = RecurrentUniform.Init([dScalar, hScalar, hScalar], hScalar);  // [1, H, H]

        Tensor<float32>? b = null;
        if (bias)
        {
            // Single owned bias [1, H]; fed as B = concat(bias, zeros) on axis 1
            // ([1, 2H]) so Wb = bias and Rb = 0.
            var biasParam = RecurrentUniform.Init([dScalar, hScalar], hScalar);    // [1, H]
            var rbZeros = TensorFill((Vector<int64>)[dScalar, hScalar], 0.0f);
            b = biasParam.Concat(1L, rbZeros);                            // [1, 2H]
        }

        // Reuse the sequence op at seq=1: X [1, N, in], initial_h [1, N, H].
        var bigX = x.Unsqueeze(0L);   // [seq=1, N, in]
        var h0 = h.Unsqueeze(0L);     // [num_dir=1, N, H]

        var (_, yhVar) = OnnxOp.Rnn(bigX, w, r, b, null, h0,
            null, null, activations, null, RNNDirection.Forward, hiddenSize, false);

        // Y_h is [num_dir=1, N, H]; squeeze the num_dir axis -> [N, H].
        return ((Tensor<float32>)yhVar).Squeeze(Vector(0L));
    }

    /// <summary>
    /// Single-step LSTM cell: the four gates over <c>(h, c)</c> —
    /// <c>i = σ(W_i·x + R_i·h + b_i)</c>, <c>o = σ(W_o·x + R_o·h + b_o)</c>,
    /// <c>f = σ(W_f·x + R_f·h + b_f)</c>, <c>c̃ = tanh(W_c·x + R_c·h + b_c)</c>,
    /// <c>c' = f ⊙ c + i ⊙ c̃</c>, <c>h' = o ⊙ tanh(c')</c>. Mirrors PyTorch
    /// <c>nn.LSTMCell</c> — one timestep, with the previous <c>(h, c)</c> handed in and
    /// the new <c>(h', c')</c> returned.
    /// </summary>
    /// <param name="x">Step input <c>[N, inputSize]</c>. <c>inputSize</c> is read in-graph from the last axis (lazy in input size, like <see cref="LSTM"/>).</param>
    /// <param name="h">Previous hidden state <c>[N, hiddenSize]</c>. Required.</param>
    /// <param name="c">Previous cell state <c>[N, hiddenSize]</c>. Required.</param>
    /// <param name="hiddenSize">Hidden state size <c>H</c>. Required, like PyTorch.</param>
    /// <param name="bias">
    /// When true (default) a single trainable bias <c>[1, 4H]</c> is owned and fed to
    /// the op as <c>B = concat(bias, zeros)</c> on axis 1 (<c>[1, 8H]</c>, so <c>Wb</c>
    /// carries it and <c>Rb</c> is 0). When false the op is given no bias.
    /// </param>
    /// <returns>The new <c>(h', c')</c>, each <c>[N, hiddenSize]</c> (the <c>num_dir</c> axis stripped), in PyTorch carry order; thread both straight back into the next step.</returns>
    /// <remarks>
    /// <para>
    /// Implemented as one step of the ONNX <c>LSTM</c> op at sequence-length 1: <c>x</c>
    /// is unsqueezed to <c>[seq=1, N, in]</c>, <c>h</c>/<c>c</c> to <c>[num_dir=1, N, H]</c>
    /// and passed as <c>initial_h</c>/<c>initial_c</c>, and the op's final
    /// <c>Y_h</c>/<c>Y_c</c> are squeezed back to <c>[N, H]</c>. This reuses the
    /// validated gate math, the ONNX-native <c>i, o, f, c</c> gate packing, the
    /// <see cref="RecurrentUniform"/> init, and the bias collapse of <see cref="LSTM"/>
    /// verbatim — a cell is <b>definitionally</b> one step of the layer, with no second
    /// gate implementation to keep in sync.
    /// </para>
    /// <para>
    /// The owned parameters are <c>W [1, 4H, in]</c>, <c>R [1, 4H, H]</c> and (when
    /// <paramref name="bias"/>) <c>bias [1, 4H]</c>, all
    /// <see cref="RecurrentUniform"/>-initialized with the true hidden size
    /// (<c>U(−1/√H, 1/√H)</c>, not <c>1/√(4H)</c>). Single-direction only, fully
    /// trainable (default sigmoid/tanh activations, <c>layout=0</c>). See
    /// <see cref="LSTM"/> for the PyTorch <c>i,f,g,o</c>→ONNX <c>i,o,f,c</c> import note.
    /// </para>
    /// </remarks>
    public static (Tensor<float32> h, Tensor<float32> c) LSTMCell(
        Tensor<float32> x,
        Tensor<float32> h,
        Tensor<float32> c,
        long hiddenSize,
        bool bias = true)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");

        var hScalar = Scalar(hiddenSize);
        var fourH = Scalar(4L * hiddenSize);  // 4 gate blocks (i, o, f, c)
        var dScalar = Scalar(1L);             // single direction
        var inSize = x.DimTensor(-1);         // read in-graph from x's last axis

        // Gate-packed weights in ONNX-native i,o,f,c order. The bound is keyed on the
        // true hidden size H (not 4H), matching PyTorch's U(-1/√H, 1/√H).
        var w = RecurrentUniform.Init([dScalar, fourH, inSize], hScalar);   // [1, 4H, in]
        var r = RecurrentUniform.Init([dScalar, fourH, hScalar], hScalar);  // [1, 4H, H]

        Tensor<float32>? b = null;
        if (bias)
        {
            // Single owned bias [1, 4H]; fed as B = concat(bias, zeros) on axis 1
            // ([1, 8H]) so Wb = bias and Rb = 0.
            var biasParam = RecurrentUniform.Init([dScalar, fourH], hScalar);    // [1, 4H]
            var rbZeros = TensorFill((Vector<int64>)[dScalar, fourH], 0.0f);
            b = biasParam.Concat(1L, rbZeros);                                   // [1, 8H]
        }

        // Reuse the sequence op at seq=1: X [1, N, in], initial_h/initial_c [1, N, H].
        // peephole P = null; default sigmoid/tanh activations (activations: null) keep
        // the trainable autodiff fast path; inputForget = false, layout = 0.
        var bigX = x.Unsqueeze(0L);   // [seq=1, N, in]
        var h0 = h.Unsqueeze(0L);     // [num_dir=1, N, H]
        var c0 = c.Unsqueeze(0L);     // [num_dir=1, N, H]

        var (_, yhVar, ycVar) = OnnxOp.Lstm(bigX, w, r, b, null, h0, c0, null,
            null, null, null, null, LSTMDirection.Forward, hiddenSize, false, false);

        // Y_h / Y_c are [num_dir=1, N, H]; squeeze the num_dir axis -> [N, H].
        var hOut = ((Tensor<float32>)yhVar).Squeeze(Vector(0L));
        var cOut = ((Tensor<float32>)ycVar).Squeeze(Vector(0L));
        return (hOut, cOut);
    }

    /// <summary>
    /// Single-step GRU cell: reset/update/candidate over <c>h</c> —
    /// <c>z = σ(W_z·x + R_z·h + b_z)</c> (update),
    /// <c>r = σ(W_r·x + R_r·h + b_r)</c> (reset),
    /// <c>ĥ = tanh(W_h·x + r ⊙ (R_h·h + Rb_h) + Wb_h)</c> (candidate, reset-after form;
    /// see <paramref name="linearBeforeReset"/>),
    /// <c>h' = (1 − z) ⊙ ĥ + z ⊙ h</c>. Mirrors PyTorch <c>nn.GRUCell</c> — one
    /// timestep, with the previous hidden state handed in and the new one returned.
    /// </summary>
    /// <param name="x">Step input <c>[N, inputSize]</c>. <c>inputSize</c> is read in-graph from the last axis (lazy in input size, like <see cref="GRU"/>).</param>
    /// <param name="h">Previous hidden state <c>[N, hiddenSize]</c>. Required.</param>
    /// <param name="hiddenSize">Hidden state size <c>H</c>. Required, like PyTorch.</param>
    /// <param name="bias">
    /// When true (default) a single trainable bias <c>[1, 3H]</c> is owned and fed to
    /// the op as <c>B = concat(bias, zeros)</c> on axis 1 (<c>[1, 6H]</c>, so <c>Wb</c>
    /// carries it and <c>Rb</c> is 0). When false the op is given no bias.
    /// </param>
    /// <param name="linearBeforeReset">
    /// Selects how the reset gate enters the candidate. When true (default) the reset
    /// is applied <b>after</b> the recurrent matmul, matching PyTorch <c>nn.GRUCell</c>,
    /// Keras <c>reset_after=True</c>, Flax and cuDNN; when false the original Cho-v1
    /// reset-before form (the ONNX op's own default). Both are trainable. See
    /// <see cref="GRU"/>.
    /// </param>
    /// <returns>The new hidden state <c>h' [N, hiddenSize]</c> (the <c>num_dir</c> axis stripped), ready to thread straight back as the next step's <paramref name="h"/>.</returns>
    /// <remarks>
    /// <para>
    /// Implemented as one step of the ONNX <c>GRU</c> op at sequence-length 1: <c>x</c>
    /// is unsqueezed to <c>[seq=1, N, in]</c>, <c>h</c> to <c>[num_dir=1, N, H]</c> and
    /// passed as <c>initial_h</c>, and the op's final hidden <c>Y_h</c> is squeezed back
    /// to <c>[N, H]</c>. This reuses the validated gate math, the ONNX-native
    /// <c>z, r, h</c> gate packing, the <c>linearBeforeReset</c> candidate form, the
    /// <see cref="RecurrentUniform"/> init, and the bias collapse of <see cref="GRU"/>
    /// verbatim — a cell is <b>definitionally</b> one step of the layer.
    /// </para>
    /// <para>
    /// The owned parameters are <c>W [1, 3H, in]</c>, <c>R [1, 3H, H]</c> and (when
    /// <paramref name="bias"/>) <c>bias [1, 3H]</c>, all
    /// <see cref="RecurrentUniform"/>-initialized with the true hidden size
    /// (<c>U(−1/√H, 1/√H)</c>, not <c>1/√(3H)</c>). Single-direction only, fully
    /// trainable in both <paramref name="linearBeforeReset"/> forms. See
    /// <see cref="GRU"/> for the PyTorch <c>r,z,n</c>→ONNX <c>z,r,h</c> import note.
    /// </para>
    /// </remarks>
    public static Tensor<float32> GRUCell(
        Tensor<float32> x,
        Tensor<float32> h,
        long hiddenSize,
        bool bias = true,
        bool linearBeforeReset = true)
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, "hiddenSize must be positive.");

        var hScalar = Scalar(hiddenSize);
        var threeH = Scalar(3L * hiddenSize);  // 3 gate blocks (z, r, h)
        var dScalar = Scalar(1L);              // single direction
        var inSize = x.DimTensor(-1);          // read in-graph from x's last axis

        // Gate-packed weights in ONNX-native z,r,h order. The bound is keyed on the
        // true hidden size H (not 3H), matching PyTorch's U(-1/√H, 1/√H).
        var w = RecurrentUniform.Init([dScalar, threeH, inSize], hScalar);   // [1, 3H, in]
        var r = RecurrentUniform.Init([dScalar, threeH, hScalar], hScalar);  // [1, 3H, H]

        Tensor<float32>? b = null;
        if (bias)
        {
            // Single owned bias [1, 3H]; fed as B = concat(bias, zeros) on axis 1
            // ([1, 6H]) so Wb = bias and Rb = 0.
            var biasParam = RecurrentUniform.Init([dScalar, threeH], hScalar);    // [1, 3H]
            var rbZeros = TensorFill((Vector<int64>)[dScalar, threeH], 0.0f);
            b = biasParam.Concat(1L, rbZeros);                                    // [1, 6H]
        }

        // Reuse the sequence op at seq=1: X [1, N, in], initial_h [1, N, H]. Default
        // sigmoid/tanh activations (activations: null) keep the trainable autodiff fast
        // path; clip = null, layout = 0. linearBeforeReset selects reset-after (true,
        // PyTorch-matching default) vs reset-before (false, the ONNX op default).
        var bigX = x.Unsqueeze(0L);   // [seq=1, N, in]
        var h0 = h.Unsqueeze(0L);     // [num_dir=1, N, H]

        var (_, yhVar) = OnnxOp.Gru(bigX, w, r, b, null, h0,
            null, null, null, null, GRUDirection.Forward, hiddenSize, false, linearBeforeReset);

        // Y_h is [num_dir=1, N, H]; squeeze the num_dir axis -> [N, H].
        return ((Tensor<float32>)yhVar).Squeeze(Vector(0L));
    }
}
