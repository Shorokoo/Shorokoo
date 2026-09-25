namespace Shorokoo.PyTorch.Translation.Operators;

// Recurrent networks, each an explicit loop over the time steps; the semantics are in
// shorokoo_torch/ops_rnn.py.
internal static partial class OperatorTable
{
    static partial void RegisterRnn(Registry table)
    {
        const string M = "ops_rnn.";
        string[] common = ["activation_alpha", "activation_beta", "activations", "clip", "direction", "hidden_size", "layout"];
        table.Map("RNN", M + "rnn", common, outputs: true);
        table.Map("GRU", M + "gru", [.. common, "linear_before_reset"], outputs: true);
        table.Map("LSTM", M + "lstm", [.. common, "input_forget"], outputs: true);
    }
}
