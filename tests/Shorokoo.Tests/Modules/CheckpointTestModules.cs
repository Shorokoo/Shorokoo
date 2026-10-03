using Shorokoo.Modules.Layers;

namespace Shorokoo.Tests.Modules;

// ---------------------------------------------------------------------------
// Twins for [Module(Checkpoint = true)]: the same MLP block with and without the
// hint, stacked three deep behind a linear head. Parameter init is keyed by the
// numeric ModelId path, so a checkpointed stack and its plain twin start from
// identical weights and must train identically; only the training-step graph
// the memory-aware pass emits differs. The narrow block (width 32) is fed a large
// batch so activations, not weights, dominate the peak; the tiny one keeps the
// whole step under the memory pass's size threshold.
// ---------------------------------------------------------------------------

internal static class MlpStackFixture
{
    internal static Tensor<float32> Block(Tensor<float32> x, long width)
    {
        var h = Linear.Model(Scalar(width), Scalar(true)).Call(x).Relu();
        return Linear.Model(Scalar(width), Scalar(true)).Call(h).Relu();
    }

    internal static Tensor<float32> Head(Tensor<float32> h)
        => Linear.Model(Scalar(16L), Scalar(true)).Call(h);
}

[Module(Checkpoint = true)]
public partial class CheckpointedTinyMlpBlock
{
    public static Tensor<float32> Inline(Tensor<float32> x) => MlpStackFixture.Block(x, 8);
}

[Module]
public partial class PlainTinyMlpBlock
{
    public static Tensor<float32> Inline(Tensor<float32> x) => MlpStackFixture.Block(x, 8);
}

[Module]
public partial class CheckpointedTinyMlpStack
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, F]
    {
        var h = CheckpointedTinyMlpBlock.Call(x);
        h = CheckpointedTinyMlpBlock.Call(h);
        h = CheckpointedTinyMlpBlock.Call(h);
        return MlpStackFixture.Head(h);
    }
}

[Module]
public partial class PlainTinyMlpStack
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, F]
    {
        var h = PlainTinyMlpBlock.Call(x);
        h = PlainTinyMlpBlock.Call(h);
        h = PlainTinyMlpBlock.Call(h);
        return MlpStackFixture.Head(h);
    }
}

[Module(Checkpoint = true)]
public partial class CheckpointedNarrowMlpBlock
{
    public static Tensor<float32> Inline(Tensor<float32> x) => MlpStackFixture.Block(x, 32);
}

[Module]
public partial class PlainNarrowMlpBlock
{
    public static Tensor<float32> Inline(Tensor<float32> x) => MlpStackFixture.Block(x, 32);
}

[Module]
public partial class CheckpointedNarrowMlpStack
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, F]
    {
        var h = CheckpointedNarrowMlpBlock.Call(x);
        h = CheckpointedNarrowMlpBlock.Call(h);
        h = CheckpointedNarrowMlpBlock.Call(h);
        return MlpStackFixture.Head(h);
    }
}

[Module]
public partial class PlainNarrowMlpStack
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, F]
    {
        var h = PlainNarrowMlpBlock.Call(x);
        h = PlainNarrowMlpBlock.Call(h);
        h = PlainNarrowMlpBlock.Call(h);
        return MlpStackFixture.Head(h);
    }
}


// ---------------------------------------------------------------------------
// The memory-pass benchmark's encoders and dense attention with each layer (or the whole
// attention) a [Module(Checkpoint = true)] segment, for measuring what recomputing them buys.
// ---------------------------------------------------------------------------

[Module(Checkpoint = true)]
public partial class CheckpointedEncoderLayer
{
    public static Tensor<float32> Inline(Tensor<float32> x)
        => TransformerEncoderLayer.Model(Scalar(128L), Scalar(4L), Scalar(512L), Scalar(true)).Call(x);
}

[Module]
public partial class CheckpointedEncoder1
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, L, E]
    {
        Vector<int64> seq = [Scalar(1L)];
        return CheckpointedEncoderLayer.Call(x).Reduce(ReduceKind.Mean, seq, keepDims: false);
    }
}

[Module]
public partial class CheckpointedEncoder2
{
    public static Tensor<float32> Inline(Tensor<float32> x)   // [N, L, E]
    {
        var y = CheckpointedEncoderLayer.Call(CheckpointedEncoderLayer.Call(x));
        Vector<int64> seq = [Scalar(1L)];
        return y.Reduce(ReduceKind.Mean, seq, keepDims: false);
    }
}

[Module(Checkpoint = true)]
public partial class CheckpointedMeanPooledAttention
{
    public static Tensor<float32> Inline(Tensor<float32> input)   // [N, H, L, d]
        => AttentionTestGraphs.MeanPooledAttention(input, queryChunks: 1);
}
