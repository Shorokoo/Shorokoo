using Microsoft.ML.OnnxRuntime;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.OnnxRuntime;

internal sealed class OrtInferenceSession : IShorokooInferenceSession
{
    private readonly InferenceSession _session;

    public OrtInferenceSession(InferenceSession session) { _session = session; }

    public IReadOnlyList<string> InputNames => _session.InputNames;
    public IReadOnlyList<string> OutputNames => _session.OutputNames;

    public IReadOnlyList<IShorokooTensorValue> Run(
        IReadOnlyDictionary<string, IShorokooTensorValue> inputs,
        IReadOnlyList<string> outputNames)
    {
        var ortInputs = new Dictionary<string, OrtValue>(inputs.Count);
        foreach (var (k, v) in inputs)
            ortInputs[k] = ((OrtTensorValue)v).Inner;

        using var runOptions = new RunOptions();
        var results = _session.Run(runOptions, ortInputs, outputNames);

        var wrapped = new List<IShorokooTensorValue>(results.Count);
        foreach (var r in results) wrapped.Add(new OrtTensorValue(r));
        return wrapped;
    }

    public void Dispose() => _session.Dispose();
}
