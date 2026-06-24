using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.AutoDiffCheckpointing.OpsPerf;

/// <summary>
/// Performance estimator for tensor shape manipulation operations.
/// These ops change the shape or layout of tensor data without performing
/// arithmetic. Most are zero-copy (metadata-only) or require a memory copy.
/// </summary>
internal class TensorManipulationPerf : IOpPerf
{
    public IReadOnlySet<string> SupportedOpCodes { get; } = new HashSet<string>
    {
        RESHAPE, FLATTEN, SQUEEZE, UNSQUEEZE, EXPAND, TRANSPOSE,
        CONCAT, SPLIT, SLICE, PAD, TILE, GATHER, GATHER_ELEMENTS, GATHER_ND,
        SCATTER_ELEMENTS, SCATTER_ND, SHAPE, SIZE, NON_ZERO,
        ONE_HOT, DEPTH_TO_SPACE, SPACE_TO_DEPTH, TRILU,
        COMPRESS, REVERSE_SEQUENCE, UNIQUE, EYE_LIKE,
        CONSTANT_OF_SHAPE, RANGE, CENTER_CROP_PAD, CLIP,
    };

    public OpPerfResult Estimate(OpPerfInput input)
    {
        var opCode = input.OpCode;

        // Shape/Size are metadata-only ops with negligible cost
        if (opCode == SHAPE || opCode == SIZE)
            return OpPerfResult.Zero;

        var outputShape = input.OutputShapes[0];
        if (outputShape is null)
            return OpPerfResult.Zero;

        var outputElements = outputShape.ElementCount;

        switch (opCode)
        {
            case RESHAPE:
            case FLATTEN:
            case SQUEEZE:
            case UNSQUEEZE:
            {
                // Zero-copy reshape — just a view change, no data movement
                // Output reuses the input buffer unconditionally
                return new OpPerfResult
                {
                    ComputeTime = 0,
                    ExtraMemoryBytes = 0,
                    InPlaceBufferReuse = new Dictionary<int, int> { [0] = 0 }
                };
            }

            case EXPAND:
            {
                // May require actual memory copy for broadcasting
                var inputShape = input.InputShapes[0];
                if (inputShape is not null && inputShape.ElementCount == outputElements)
                {
                    // No actual expansion needed — same size
                    return new OpPerfResult
                    {
                        ComputeTime = 0,
                        ExtraMemoryBytes = 0,
                        InPlaceBufferReuse = new Dictionary<int, int> { [0] = 0 }
                    };
                }
                // Real broadcast: cost is proportional to output size (memory copy)
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 0.5, // Memory-bound copy
                    ExtraMemoryBytes = 0,
                };
            }

            case TRANSPOSE:
            {
                // Data movement with non-contiguous access pattern
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 1.5, // Cache-unfriendly copy
                    ExtraMemoryBytes = 0,
                };
            }

            case CONCAT:
            {
                // Copy all inputs into output buffer
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 0.5, // Memory copy
                    ExtraMemoryBytes = 0,
                };
            }

            case SPLIT:
            {
                // Copy portions of input to separate output buffers
                var inputShape = input.InputShapes[0];
                var inputElements = inputShape?.ElementCount ?? outputElements;
                return new OpPerfResult
                {
                    ComputeTime = inputElements / 256.0 * 0.5,
                    ExtraMemoryBytes = 0,
                };
            }

            case SLICE:
            {
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 0.5,
                    ExtraMemoryBytes = 0,
                };
            }

            case PAD:
            {
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 0.5,
                    ExtraMemoryBytes = 0,
                };
            }

            case TILE:
            {
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 0.5,
                    ExtraMemoryBytes = 0,
                };
            }

            case GATHER:
            case GATHER_ELEMENTS:
            case GATHER_ND:
            {
                // Random access reads — cache-unfriendly
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 2.0,
                    ExtraMemoryBytes = 0,
                };
            }

            case SCATTER_ELEMENTS:
            case SCATTER_ND:
            {
                // Random access writes plus read-modify-write for reduction modes
                var updateShape = input.InputShapes.Length > 2 ? input.InputShapes[2] : null;
                var updateElements = updateShape?.ElementCount ?? outputElements;
                // Check if output can reuse the data input buffer (first input)
                var canInPlace = !input.InputMustRemainIntact[0]
                    && input.InputShapes[0] is not null
                    && input.InputShapes[0]!.ElementCount == outputElements
                    && input.InputShapes[0]!.DType == outputShape.DType;
                return new OpPerfResult
                {
                    ComputeTime = updateElements / 256.0 * 3.0,
                    ExtraMemoryBytes = 0,
                    InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
                };
            }

            case CLIP:
            {
                var inputShape = input.InputShapes[0];
                if (inputShape is null) return OpPerfResult.Zero;
                var canInPlace = !input.InputMustRemainIntact[0]
                    && inputShape.ElementCount == outputElements
                    && inputShape.DType == outputShape.DType;
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 1.0,
                    ExtraMemoryBytes = 0,
                    InPlaceBufferReuse = canInPlace ? new Dictionary<int, int> { [0] = 0 } : new Dictionary<int, int>()
                };
            }

            default:
            {
                // Default: assume memory copy proportional to output
                return new OpPerfResult
                {
                    ComputeTime = outputElements / 256.0 * 1.0,
                    ExtraMemoryBytes = 0,
                };
            }
        }
    }
}
