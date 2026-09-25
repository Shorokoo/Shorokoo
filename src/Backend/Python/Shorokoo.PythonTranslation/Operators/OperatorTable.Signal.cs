namespace Shorokoo.PythonTranslation.Operators;

// Signal processing; the semantics are in each support package's ops_signal.py.
internal static partial class OperatorTable
{
    static partial void RegisterSignal(Registry table)
    {
        const string M = "ops_signal.";
        table.Map("DFT", M + "dft", ["axis", "inverse", "onesided"], opset: true);
        table.Map("STFT", M + "stft", ["onesided"]);
        table.Map("HannWindow", M + "hann_window", ["output_datatype", "periodic"], gradient: GradientRule.NotDifferentiable);
        table.Map("HammingWindow", M + "hamming_window", ["output_datatype", "periodic"], gradient: GradientRule.NotDifferentiable);
        table.Map("BlackmanWindow", M + "blackman_window", ["output_datatype", "periodic"], gradient: GradientRule.NotDifferentiable);
        table.Map("MelWeightMatrix", M + "mel_weight_matrix", ["output_datatype"], gradient: GradientRule.NotDifferentiable);
    }
}
