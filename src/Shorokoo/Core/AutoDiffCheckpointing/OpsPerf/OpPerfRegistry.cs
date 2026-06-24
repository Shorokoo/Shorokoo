namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Registry that maps operation codes to their performance estimators.
/// Provides a centralized lookup for op perf estimation during graph evaluation.
/// </summary>
internal class OpPerfRegistry
{
    private readonly Dictionary<string, IOpPerf> _registry = new();
    private readonly DefaultOpPerf _defaultPerf = new();

    public OpPerfRegistry()
    {
        Register(new UnaryElementwisePerf());
        Register(new BinaryElementwisePerf());
        Register(new ReductionPerf());
        Register(new TensorManipulationPerf());
        Register(new LinearAlgebraPerf());
        Register(new PoolingNormPerf());
        Register(new RecurrentPerf());
        Register(new MiscPerf());
    }

    private void Register(IOpPerf perf)
    {
        foreach (var opCode in perf.SupportedOpCodes)
            _registry[opCode] = perf;
    }

    /// <summary>
    /// Gets the performance estimator for a given operation code.
    /// Returns a default estimator for unregistered ops.
    /// </summary>
    public IOpPerf GetEstimator(string opCode)
        => _registry.TryGetValue(opCode, out var perf) ? perf : _defaultPerf;

    /// <summary>
    /// Estimates performance for a given operation.
    /// </summary>
    public OpPerfResult Estimate(OpPerfInput input)
        => GetEstimator(input.OpCode).Estimate(input);
}

/// <summary>
/// Default performance estimator for operations without a specialized estimator.
/// Provides a conservative estimate based on output size.
/// </summary>
internal class DefaultOpPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>();

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var outputShape = input.OutputShapes.FirstOrDefault(s => s is not null);
        if (outputShape is null)
            return OpPerfResult.Zero;

        // Default: assume cost proportional to output size
        return new OpPerfResult
        {
            ComputeTime = outputShape.ElementCount / 256.0,
            ExtraMemoryBytes = 0,
        };
    }
}
