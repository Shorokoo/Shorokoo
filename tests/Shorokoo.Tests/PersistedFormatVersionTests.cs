using System.IO.Compression;
using System.Text.Json;

namespace Shorokoo.Tests;

/// <summary>
/// Every Shorokoo-controlled save-format version is 1 and stays there (#108/#109, #144/#146):
/// nothing Shorokoo writes has been released, so a bump records an edit to unreleased code
/// rather than a version anyone had, and every "read the older shape" path is dead weight.
///
/// <para>A failure here is the intended behaviour of a version bump, not a test to update in
/// passing: either revert the bump and express the change within version 1, or update this test
/// together with the rule in <c>Documentation/skpt-checkpoints.md</c>.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class PersistedFormatVersionTests
{
    [Fact]
    public void TestEveryPersistedFormatConstantMagicAndWrittenFileCarriesVersionOne()
    {
        Assert.Equal(1, SrkFileFormat.CurrentVersion);                  // .srk container
        Assert.Equal(1, SkptFileFormat.CurrentVersion);                 // .skpt manifest
        Assert.Equal(1, SkptFileFormat.TrainingCheckpointVersion);      // .skpt training block
        Assert.Equal(1, SkptFileFormat.TrainingRigVersion);             // .skpt rig block
        Assert.Equal(1L, TrainingCheckpoint.CheckpointFormatVersion);   // flat checkpoint marker

        // The .srk version is also the magic's last byte — the pre-header rejection reads it.
        Assert.Equal((byte)1, SrkFileFormat.Magic[3]);
        Assert.Equal(SrkFileFormat.CurrentVersion, SrkFileFormat.Magic[3]);

        // A writer hardcoding a different number would slip past the constants, so parse the
        // version out of real bytes on disk without going through them at all.
        var dir = Path.Combine(Path.GetTempPath(), $"shrk_fmtver_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var moduleGraph = ScalarMultiplyModel.ComputationGraph;
            var model = moduleGraph
                .ToConcreteArchitecture(moduleGraph.FromOrderedInputs([TensorData([2], 1.0f, 2.0f)]))
                .ToConcreteModel();

            var srkPath = Path.Combine(dir, "model.srk");
            CompressedFormatUtils.SaveFastGraphToFile(srkPath, model, compressed: false);
            var srkBytes = File.ReadAllBytes(srkPath);
            Assert.Equal((byte)1, srkBytes[3]);

            int headerLen = srkBytes[4] | (srkBytes[5] << 8);
            using (var header = JsonDocument.Parse(srkBytes.AsSpan(6, headerLen).ToArray()))
                Assert.Equal(1, header.RootElement.GetProperty("srkVersion").GetInt32());

            var skptPath = Path.Combine(dir, "model.skpt");
            Persistence.From(model).WithModel().WithWeights().Save(skptPath);
            using (var archive = ZipFile.OpenRead(skptPath))
            using (var entry = archive.GetEntry(SkptFileFormat.ConfigEntryName)!.Open())
            using (var manifest = JsonDocument.Parse(entry))
                Assert.Equal(1, manifest.RootElement.GetProperty("skptVersion").GetInt32());
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
