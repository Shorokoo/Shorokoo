using System.IO;
using System.Linq;
using System.Text.Json;
using System.IO.Compression;
using Shorokoo.Core.Utils;

namespace Shorokoo.Tests;

/// <summary>
/// The persisted-format version guard: <b>every Shorokoo-controlled save-format version is
/// 1</b>, and stays there until a breaking change is a deliberate, reviewed act.
///
/// <para>Nothing Shorokoo writes has been released, so no file written by older code exists
/// anywhere. That makes a version number above 1 a record of edits to unreleased code rather
/// than a version anyone ever had — and it makes every "read the older shape" path dead
/// weight. The rule is therefore: one version per format, no compatibility shims, and a
/// breaking change regenerates rather than migrates.</para>
///
/// <para><b>Why this test exists.</b> The rule has been applied twice. Issue #108 (PR #109,
/// 2026-07-28) reset every format version to 1 and deleted the pre-release compat shims.
/// Within a week two of them had drifted back — the flat checkpoint marker to 3 and the .skpt
/// rig block to 2 — and #144 (PR #146) reset them again. Nothing caught the regression in
/// between, because every existing assertion compares a written file against the constant
/// (<c>Assert.Equal(SkptFileFormat.CurrentVersion, manifest.SkptVersion)</c>), which passes at
/// any value. This test pins the constants to the literal 1 so a third drift cannot land
/// silently.</para>
///
/// <para><b>If this test fails</b>, that is the intended behaviour of a version bump: it is not
/// a test to update in passing. Either the bump is unintended — revert it and express the
/// change within version 1, which is almost always possible while nothing is released — or it
/// is a deliberate decision to start versioning this format, in which case update this test
/// together with the rule in <c>Documentation/skpt-checkpoints.md</c>.</para>
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class PersistedFormatVersionTests
{
    private const string Rule =
        "Every persisted format is version 1 (see #108/#109 and #144/#146). Nothing is " +
        "released, so a bump records an edit to unreleased code rather than a version anyone " +
        "had. Express the change within version 1, or update this test deliberately.";

    [Fact]
    public void TestEveryPersistedFormatVersionIsOne()
    {
        Assert.Equal(1, SrkFileFormat.CurrentVersion);                  // .srk container
        Assert.Equal(1, SkptFileFormat.CurrentVersion);                 // .skpt manifest
        Assert.Equal(1, SkptFileFormat.TrainingCheckpointVersion);      // .skpt training block
        Assert.Equal(1, SkptFileFormat.TrainingRigVersion);             // .skpt rig block
        Assert.Equal(1L, TrainingCheckpoint.CheckpointFormatVersion);   // flat checkpoint marker
    }

    [Fact]
    public void TestSrkContainerMagicCarriesVersionOne()
    {
        // The .srk version is also the magic's last byte, so a bump that missed the magic (or
        // vice versa) would make the two disagree — the pre-header rejection reads the byte.
        Assert.Equal((byte)1, SrkFileFormat.Magic[3]);
        Assert.Equal(SrkFileFormat.CurrentVersion, SrkFileFormat.Magic[3]);
    }

    [Fact]
    public void TestWrittenFilesLiterallyCarryVersionOne()
    {
        // The constants above are what the writers reference, but a writer that hardcoded a
        // different number would slip past them. Assert the bytes on disk instead: parse the
        // version out of a real .skpt manifest and a real .srk header without going through
        // the constants at all.
        var dir = Path.Combine(Path.GetTempPath(), $"shrk_fmtver_{System.Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var model = BuildTinyConcreteModel();

            var srkPath = Path.Combine(dir, "model.srk");
            CompressedFormatUtils.SaveFastGraphToFile(srkPath, model, compressed: false);
            var srkBytes = File.ReadAllBytes(srkPath);
            Assert.Equal((byte)1, srkBytes[3]);   // magic's version byte

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

    /// <summary>The smallest concrete model that round-trips through both containers — the same
    /// fixture the .srk/.skpt round-trip tests use.</summary>
    private static ComputationGraph BuildTinyConcreteModel()
    {
        var moduleGraph = ScalarMultiplyModel.ComputationGraph;
        return moduleGraph
            .ToConcreteArchitecture(moduleGraph.FromOrderedInputs([TensorData([2], 1.0f, 2.0f)]))
            .ToConcreteModel();
    }
}
