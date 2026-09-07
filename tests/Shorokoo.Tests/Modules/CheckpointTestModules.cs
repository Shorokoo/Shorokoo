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

