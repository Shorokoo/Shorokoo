using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for recurrent neural network operations (GRU, LSTM, RNN).
/// These ops process sequences timestep by timestep with matrix multiplications at each step.
/// Cost is O(sequence_length × batch × hidden_size² × gates).
/// </summary>
internal class RecurrentPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>
    {
        GRU, LSTM, RNN
    };

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var inputShape = input.InputShapes[0]; // [seq_len, batch, input_size]
        if (inputShape is null)
            return OpPerfResult.Zero;

        var dims = inputShape.Shape.Dims;
        if (dims.Length < 3)
            return OpPerfResult.Zero;

        long seqLen = dims[0];
        long batch = dims[1];
        long inputSize = dims[2];
        long hiddenSize = GetLongAttr(input, "hidden_size", inputSize);

        int gates = input.OpCode switch
        {
            LSTM => 4, // i, f, o, c gates
            GRU => 3,  // z, r, h gates
            RNN => 1,  // single hidden state
            _ => 1
        };

        // Per timestep: gates × (input_size × hidden_size + hidden_size × hidden_size) × 2 FLOPs
        var flopsPerTimestep = gates * (inputSize * hiddenSize + hiddenSize * hiddenSize) * 2.0 * batch;
        var totalFlops = seqLen * flopsPerTimestep;
        var computeTime = totalFlops / 256.0;

        // RNNs need workspace for intermediate gate values
        long directions = 1; // Simplified — could be bidirectional
        var workspaceBytes = seqLen * batch * hiddenSize * gates * directions
            * (inputShape.DType.EncodingBitCount / 8);

        return new OpPerfResult
        {
            ComputeTime = computeTime,
            ExtraMemoryBytes = workspaceBytes,
        };
    }

    private static long GetLongAttr(OpPerfInput input, string name, long defaultValue)
    {
        if (input.Attributes.TryGetValue(name, out var val) && val is long l)
            return l;
        return defaultValue;
    }
}
