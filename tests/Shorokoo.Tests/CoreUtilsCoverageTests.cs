using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Shorokoo.Core.Factory.OpsFactories;
using Shorokoo.Core.Inference.Abstractions;

namespace Shorokoo.Tests;

/// <summary>
/// Coverage-purpose tests for small framework utilities that the graph-level suites
/// never touch directly: the internal LINQ-ish helpers in
/// <c>Shorokoo.Core.Utils.Extensions</c> (AddAll/Iterate/FindIndexOf/NotNull(s)/Convert),
/// the <see cref="NodeKey"/>/<see cref="TensorKey"/> identity structs, the
/// <see cref="ShorokooException"/> hierarchy, the OpsFactories
/// <see cref="Helpers"/> dtype sets + attribute-type mapping, and the
/// <see cref="InferenceBackend"/> deployment-folder discovery fallback (the
/// OS-derived backend selection, a live ORT execution check, and the multi-backend
/// CPU-vs-GPU selection policy), and the <see cref="AtomicFileWriter"/>
/// temp-and-rename commit protocol (crash-window fault injection, stale-temp sweep,
/// and the directory shape). Plain xunit facts.
/// </summary>
[Trait("Domain", "Core")]
[Trait("Purpose", "Coverage")]
public class CoreUtilsCoverageTests
{
    [Fact]
    public void TestExtensionsCollectionHelpersCoverage()
    {
        var list = new List<int> { 1 };
        list.AddAll(new[] { 2, 3 });
        Assert.Equal(new[] { 1, 2, 3 }, list);

        var dctTuples = new Dictionary<string, int>();
        dctTuples.AddAll(new[] { ("a", 1), ("b", 2) });
        Assert.Equal(2, dctTuples.Count);

        var dctRefTuples = new Dictionary<string, int>();
        dctRefTuples.AddAll(new[] { Tuple.Create("x", 9) });
        Assert.Equal(9, dctRefTuples["x"]);

        Assert.Equal([("a", 0), ("b", 1)], new[] { "a", "b" }.Iterate().ToArray());

        Assert.Equal((Index)1, new[] { "a", "b", "c" }.FindIndexOf("b"));
        Assert.Equal(^0, new[] { "a", "b", "c" }.FindIndexOf("z"));
    }

    [Fact]
    public void TestExtensionsNullHelpersCoverage()
    {
        Assert.Equal(new[] { "a", "b" }, new string?[] { "a", null, "b" }.NotNulls().ToArray());
        Assert.Equal(new[] { 1, 3 }, Extensions.NotNulls<int>(new int?[] { 1, null, 3 }).ToArray());
        Assert.Equal(new[] { 2, 4 }, Extensions.NotNull<int>(new int?[] { null, 2, 4 }).ToArray());
        Assert.Equal(new[] { "x" }, new string?[] { "x" }.AssertNotNulls().ToArray());

        Assert.Equal("ok", ((string?)"ok").AssertNotNull());
        Assert.Equal(7, ((int?)7).AssertNotNull());
        Assert.Equal("ok", ((string?)"ok").NotNull());
        Assert.Equal(7, ((int?)7).NotNull());
        Assert.Throws<InvalidTensorOperationException>(() => ((string?)null).NotNull());
        Assert.Throws<InvalidTensorOperationException>(() => ((int?)null).NotNull());
    }

    [Fact]
    public void TestExtensionsTupleHelpersCoverage()
    {
        ITuple tuple = (1, 2, 3);
        Assert.Equal(3, tuple.ToEnumerable().Cast<object>().Count());
        Assert.Equal(new[] { 1, 2, 3 }, tuple.Cast<int>().ToArray());
    }

    [Fact]
    public void TestExtensionsConvertCoverage()
    {
        // int-source: identity + the three widening/narrowing iterators + unsupported target.
        var ints = new[] { 1, 2 };
        Assert.Same(ints, ints.Convert<int>());
        Assert.Equal(new uint[] { 1, 2 }, ints.Convert<uint>().ToArray());
        Assert.Equal(new long[] { 1, 2 }, ints.Convert<long>().ToArray());
        Assert.Equal(new ulong[] { 1, 2 }, ints.Convert<ulong>().ToArray());
        Assert.Throws<UnsupportedDTypeException>(() => ints.Convert<float>());

        // uint-source.
        var uints = new uint[] { 3, 4 };
        Assert.Equal(new[] { 3, 4 }, uints.Convert<int>().ToArray());
        Assert.Same(uints, uints.Convert<uint>());
        Assert.Equal(new long[] { 3, 4 }, uints.Convert<long>().ToArray());
        Assert.Equal(new ulong[] { 3, 4 }, uints.Convert<ulong>().ToArray());
        Assert.Throws<UnsupportedDTypeException>(() => uints.Convert<float>());

        // long-source.
        var longs = new long[] { 5, 6 };
        Assert.Equal(new[] { 5, 6 }, longs.Convert<int>().ToArray());
        Assert.Equal(new uint[] { 5, 6 }, longs.Convert<uint>().ToArray());
        Assert.Same(longs, longs.Convert<long>());
        Assert.Equal(new ulong[] { 5, 6 }, longs.Convert<ulong>().ToArray());
        Assert.Throws<UnsupportedDTypeException>(() => longs.Convert<float>());

        // ulong-source.
        var ulongs = new ulong[] { 7, 8 };
        Assert.Equal(new[] { 7, 8 }, ulongs.Convert<int>().ToArray());
        Assert.Equal(new uint[] { 7, 8 }, ulongs.Convert<uint>().ToArray());
        Assert.Equal(new long[] { 7, 8 }, ulongs.Convert<long>().ToArray());
        Assert.Same(ulongs, ulongs.Convert<ulong>());
        Assert.Throws<UnsupportedDTypeException>(() => ulongs.Convert<float>());

        // Generic Convert<TIn, TOut>: implicit-operator path (long → Shape) and
        // Convert.ChangeType fallback path (int → double).
        var shapes = new long[] { 2L, 3L }.Convert<long, Shape>().ToArray();
        Assert.Equal(new Shape(2L), shapes[0]);
        Assert.Equal(new Shape(3L), shapes[1]);
        Assert.Equal(new[] { 1.0, 2.0 }, new[] { 1, 2 }.Convert<int, double>().ToArray());
    }

    [Fact]
    public void TestNodeKeyCoverage()
    {
        var key = NodeKey.New();
        Assert.False(key.IsEmpty);
        Assert.True(NodeKey.Empty.IsEmpty);

        var parsed = NodeKey.Parse(key.Id.ToString());
        Assert.Equal(key, parsed);
        Assert.True(key == parsed);
        Assert.False(key != parsed);
        Assert.True(key.Equals((object)parsed));
        Assert.Equal(key.GetHashCode(), parsed.GetHashCode());
        Assert.Equal(0, key.CompareTo(parsed));
        Assert.Equal(key.Id.ToString("N"), key.ToString());

        Assert.True(NodeKey.TryParse(key.Id.ToString(), out var tryParsed));
        Assert.Equal(key, tryParsed);
        Assert.False(NodeKey.TryParse("not-a-guid", out var failed));
        Assert.True(failed.IsEmpty);
    }

    [Fact]
    public void TestTensorKeyCoverage()
    {
        var node = NodeKey.New();
        var key = new TensorKey(node, 2);
        Assert.False(key.IsEmpty);
        Assert.False(key.IsConnectingTensor);
        Assert.True(TensorKey.Empty.IsEmpty);
        Assert.True(TensorKey.ForConnectingTensor(node).IsConnectingTensor);

        var roundTripped = TensorKey.Parse(key.ToString());
        Assert.Equal(key, roundTripped);
        Assert.True(key == roundTripped);
        Assert.False(key != roundTripped);
        Assert.True(key.Equals((object)roundTripped));
        Assert.Equal(key.GetHashCode(), roundTripped.GetHashCode());
        Assert.Equal(0, key.CompareTo(roundTripped));
        Assert.True(key.CompareTo(new TensorKey(node, 3)) < 0);

        Assert.Throws<FormatException>(() => TensorKey.Parse("missing-colon"));
        Assert.True(TensorKey.TryParse(key.ToString(), out var tryParsed));
        Assert.Equal(key, tryParsed);
        Assert.False(TensorKey.TryParse(null, out _));
        Assert.False(TensorKey.TryParse("", out _));
        Assert.False(TensorKey.TryParse("a:b:c", out _));
        Assert.False(TensorKey.TryParse("nope:1", out _));
    }

    [Fact]
    public void TestShorokooExceptionsCoverage()
    {
        var inner = new InvalidOperationException("inner");

        var dtype = new UnsupportedDTypeException("E001", "float99", "Cast", "extra context");
        Assert.Equal("E001", dtype.ErrorCode);
        Assert.Equal("float99", dtype.DTypeName);
        Assert.Equal("Cast", dtype.Operation);
        Assert.Contains("[E001]", dtype.Message);

        var tensorOp = new InvalidTensorOperationException("E002", "Reshape", "t0", "bad dims");
        Assert.Equal("Reshape", tensorOp.Operation);
        Assert.Equal("t0", tensorOp.TensorInfo);

        var node = new OnnxNodeException("E003", "Add", "add_1", "boom");
        Assert.Equal("Add", node.NodeType);
        Assert.Equal("add_1", node.NodeName);

        var module = new ModuleException("E004", "MyModule", "broken");
        Assert.Equal("MyModule", module.ModuleName);
        var moduleInner = new ModuleException("E004", "MyModule", "broken", inner);
        Assert.Same(inner, moduleInner.InnerException);

        var ctx = new ComputeContextException("E005", "cpu", "session failed");
        Assert.Equal("cpu", ctx.ContextInfo);
        var ctxInner = new ComputeContextException("E005", "cpu", "session failed", inner);
        Assert.Same(inner, ctxInner.InnerException);

        var model = new ModelException("E006", "model.onnx", "load failed");
        Assert.Equal("model.onnx", model.ModelInfo);
        var modelInner = new ModelException("E006", "model.onnx", "load failed", inner);
        Assert.Same(inner, modelInner.InnerException);

        var autodiff = new AutoDiffNotSupportedException("E007", "Det", "no gradient");
        Assert.Equal("Det", autodiff.OpName);
        Assert.Contains("Det", autodiff.Message);

        var reflection = new ReflectionException("E008", "Invoke", "MyType", "missing method");
        Assert.Equal("Invoke", reflection.MethodInfo);
        Assert.Equal("MyType", reflection.TypeInfo);
    }

    [Fact]
    public void TestOpsFactoriesHelpersCoverage()
    {
        // The dtype-set fields are consumed by op factories; touching each runs the
        // static initializers.
        Assert.Contains(DType.Float32, Helpers.Numeric14);
        Assert.Contains(DType.Float32, Helpers.Numeric13);
        Assert.Contains(DType.Float32, Helpers.Numeric6);
        Assert.Contains(DType.Float32, Helpers.Numeric1);
        Assert.Contains(DType.String, Helpers.All2);
        Assert.Contains(DType.BFloat16, Helpers.All13);

        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Int, AttributeType.Bool.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Ints, AttributeType.Bools.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Int, AttributeType.Long.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Ints, AttributeType.Longs.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Int, AttributeType.DType.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Ints, AttributeType.DTypes.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Float, AttributeType.Float.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Floats, AttributeType.Floats.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Graph, AttributeType.Graph.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.String, AttributeType.String.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Strings, AttributeType.Strings.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.String, AttributeType.Enum.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Strings, AttributeType.Enums.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.Tensor, AttributeType.Tensor.ToProto());
        Assert.Equal(Core.Factory.IR.AttributeProto.AttributeType.TypeProto, AttributeType.TypeProto.ToProto());
        Assert.Throws<UnsupportedDTypeException>(() => ((AttributeType)(-1)).ToProto());
    }

    [Fact]
    public void TestInferenceBackendDiscoveryCoverage()
    {
        // No backend is set explicitly in this suite, so accessing Factory exercises
        // the deployment-folder auto-discovery fallback. It must resolve to the
        // platform backend deployed alongside the test host -- derived from the
        // running OS rather than hardcoded, so this holds on Windows and Linux alike
        // (the WinCPU coverage run no longer trips on a Linux-only expectation).
        var factory = InferenceBackend.Factory;
        Assert.NotNull(factory);

        var name = factory.GetType().Assembly.GetName().Name ?? "";
        var expectedPrefix = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "Shorokoo.Win"
            : "Shorokoo.Linux";
        Assert.StartsWith(expectedPrefix, name);

        // Discovery is sticky: a second access returns the same instance.
        Assert.Same(factory, InferenceBackend.Factory);

        // ...and the discovered backend actually executes: run 2 + 3 through ORT.
        var sum = OnnxEngine.Eval(Scalar(2f) + Scalar(3f));
        Assert.Equal(5f, sum.As<float32>().AccessMemory()[0]);
    }

    [Fact]
    public void TestInferenceBackendSelectionPolicyCoverage()
    {
        // The selection policy that runs when more than one backend is deployed
        // for the current OS. (The suite only ships one backend, so this exercises
        // the multi-candidate paths directly.)
        var cpu = ("Shorokoo.LinuxCPU", false);
        var gpu = ("Shorokoo.LinuxGPU", true);

        // Nothing deployed -> no choice.
        Assert.Null(InferenceBackend.SelectBackend([], cudaAvailable: true));

        // A single backend is taken as-is, regardless of the CUDA state.
        Assert.Equal(cpu, InferenceBackend.SelectBackend([cpu], cudaAvailable: true)!.Value);
        Assert.Equal(gpu, InferenceBackend.SelectBackend([gpu], cudaAvailable: false)!.Value);

        // With both accessible, CUDA presence decides: GPU when present, CPU when not.
        Assert.Equal(gpu, InferenceBackend.SelectBackend([cpu, gpu], cudaAvailable: true)!.Value);
        Assert.Equal(cpu, InferenceBackend.SelectBackend([cpu, gpu], cudaAvailable: false)!.Value);
    }

    [Fact]
    public void TestVariableHandleConversionCoverage()
    {
        // A graph node carries the structural kind, runtime dtype and rank; wrapping it in a typed
        // value handle (the implicit Variable->handle operators, via IValue.RequireKind/
        // RequireDType/RequireRank) must enforce that all three are compatible.
        Variable scalarNode = InputScalar<float32>("a");        // Tensor kind, rank 0, float32
        Variable vectorNode = InputVector<float32>("b");        // Tensor kind, rank 1, float32
        Variable rank2Node = InputTensor<float32>("c", rank: 2);
        Variable seqNode = OnnxOp.SequenceEmpty(DType.Float32);
        Variable optNode = OnnxOp.Optional(null, DataStructure.Tensor, DType.Float32);

        // Valid conversions preserve structure + rank.
        Assert.Equal(0, ((Variable)(Scalar<float32>)scalarNode).Rank);
        Assert.Equal(1, ((Variable)(Vector<float32>)vectorNode).Rank);
        Assert.Equal(2, ((Variable)(Tensor<float32>)rank2Node).Rank);
        Assert.Equal(DataStructure.Sequence, ((Variable)(TensorSequence<float32>)seqNode).Structure());
        Assert.Equal(DataStructure.Optional, ((Variable)(OptionalTensor<float32>)optNode).Structure());

        // Structure must always match.
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Tensor<float32>)seqNode);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(TensorSequence<float32>)scalarNode);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(OptionalTensor<float32>)scalarNode);

        // No implicit dtype reinterpretation (use Cast to convert).
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Tensor<float64>)scalarNode);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Scalar<int64>)scalarNode);

        // A known-mismatching rank is an error.
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Scalar<float32>)rank2Node);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Vector<float32>)rank2Node);
        Assert.Throws<InvalidTensorOperationException>(() => (object)(Scalar<float32>)vectorNode);

        // An UNKNOWN-rank node is adapted to a fixed-rank handle with an Identity rank-conversion node.
        Variable unranked = InputTensor<float32>("u"); Assert.Null(unranked.Rank);
        var sFromNull = (Variable)(Scalar<float32>)unranked;
        Assert.Equal(0, sFromNull.Rank); Assert.Equal(OpCodes.IDENTITY, sFromNull.OwningNode.OpCode);
        Variable unranked2 = InputTensor<float32>("u2");
        var vFromNull = (Variable)(Vector<float32>)unranked2;
        Assert.Equal(1, vFromNull.Rank); Assert.Equal(OpCodes.IDENTITY, vFromNull.OwningNode.OpCode);

        // Vec()/Scalar() reinterpret a tensor handle to a fixed-rank Vector/Scalar, validating rank
        // exactly like the Variable->handle operators: a known mismatch is an error (you cannot coerce
        // a rank-0 scalar into a vector, or a rank-1 vector into a scalar), a matching rank passes
        // through, and an unknown rank is materialised with an Identity rank-conversion.
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Tensor<float32>)rank2Node).Vec());
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Tensor<float32>)rank2Node).Scalar());
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Scalar<float32>)scalarNode).Vec());
        Assert.Throws<InvalidTensorOperationException>(() => (object)((Vector<float32>)vectorNode).Scalar());
        Assert.Equal(1, ((Variable)((Tensor<float32>)vectorNode).Vec()).Rank);
        Assert.Equal(0, ((Variable)((Tensor<float32>)scalarNode).Scalar()).Rank);
        Variable unrankedVec = InputTensor<float32>("uv");
        var vecAdapted = (Variable)((Tensor<float32>)unrankedVec).Vec();
        Assert.Equal(1, vecAdapted.Rank); Assert.Equal(OpCodes.IDENTITY, vecAdapted.OwningNode.OpCode);

        // Cast<V> is the explicit dtype CONVERSION (inserts a Cast node) — to change dtype you must Cast;
        // there is no reinterpret.
        Assert.Equal(DType.Float64, ((Variable)scalarNode.Cast<float64>()).Type);
    }

    [Fact]
    public void TestTensorSequenceInsertAtInterfaceCoverage()
    {
        // ITensorSequence.InsertAt accepts any ITensor element. A Vector<T>/Scalar<T> is an ITensor
        // but not a Tensor<T>, so the element must convert through its backing Variable (the validating
        // Variable→Tensor<T> operator) — a direct (Tensor<T>)element unbox would throw
        // InvalidCastException for these non-Tensor handle structs.
        ITensorSequence seq = TensorSequence<float32>(InputTensor<float32>("e0", rank: 1));

        ITensor vectorElem = InputVector<float32>("v");   // a Vector<float32> boxed as ITensor
        ITensor scalarElem = InputScalar<float32>("s");   // a Scalar<float32> boxed as ITensor

        var afterVec = seq.InsertAt(vectorElem, Scalar(0L));
        var afterBoth = afterVec.InsertAt(scalarElem, Scalar(0L));

        Assert.Equal(DataStructure.Sequence, afterBoth.Structure());
    }

    // ---- AtomicFileWriter: temp-and-rename commit, stale sweep ----

    /// <summary>Fresh scratch directory for one atomic-writer scenario; deleted by the caller.</summary>
    private static string NewScratchDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"shrk_atomic_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void WriteVia(string path, string content, Action<string>? onWarning = null) =>
        AtomicFileWriter.WriteFile(
            path, s => s.Write(System.Text.Encoding.UTF8.GetBytes(content)), onWarning);

    /// <summary>
    /// The single-file commit protocol: a successful save lands the content with no staged
    /// <c>.tmp-</c> sibling left behind; a failure injected in the crash window between write
    /// and rename (and a failure thrown by the content writer itself) leaves the previous
    /// content untouched at the target; and stale temps planted by a "crashed" earlier save
    /// are swept on the next successful save of the same target — but temps belonging to a
    /// different target are not.
    /// </summary>
    [Fact]
    public void TestAtomicFileWriterCrashWindowCoverage()
    {
        var dir = NewScratchDir();
        var target = Path.Combine(dir, "state.bin");
        try
        {
            WriteVia(target, "v1");
            Assert.Equal("v1", File.ReadAllText(target));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // Crash in the commit window (after write+flush, before rename): the old content
            // survives and the writer disposes its own staged copy. The hook filters on the
            // scratch dir so concurrent tests elsewhere are unaffected.
            AtomicFileWriter.CommitFaultInjection = p =>
            {
                if (p.StartsWith(dir, StringComparison.Ordinal)) throw new IOException("injected crash");
            };
            try
            {
                Assert.Throws<IOException>(() => WriteVia(target, "v2"));
            }
            finally { AtomicFileWriter.CommitFaultInjection = null; }
            Assert.Equal("v1", File.ReadAllText(target));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // A failure inside the content writer behaves the same way.
            Assert.Throws<InvalidOperationException>(() => AtomicFileWriter.WriteFile(
                target, _ => throw new InvalidOperationException("writer boom")));
            Assert.Equal("v1", File.ReadAllText(target));

            // A hard crash leaves a stale temp behind (planted here by hand, matching the
            // staged-name shape ".tmp-<target>-<32-hex-guid>"). The next successful save of the
            // same target sweeps it — but only precise matches: a temp for a *different* target,
            // one whose name merely shares this target's as a prefix, and a ".tmp-" file whose
            // suffix isn't a GUID all survive.
            var staleOurs   = Path.Combine(dir, $".tmp-state.bin-{Guid.NewGuid():N}");
            var staleOther  = Path.Combine(dir, $".tmp-other.bin-{Guid.NewGuid():N}");
            var stalePrefix = Path.Combine(dir, $".tmp-state.bin-2-{Guid.NewGuid():N}"); // target "state.bin-2"
            var staleNonGuid = Path.Combine(dir, ".tmp-state.bin-notahexguidxxxxxxxxxxx");
            File.WriteAllText(staleOurs, "partial");
            File.WriteAllText(staleOther, "partial");
            File.WriteAllText(stalePrefix, "partial");
            File.WriteAllText(staleNonGuid, "partial");
            WriteVia(target, "v2");
            Assert.Equal("v2", File.ReadAllText(target));
            Assert.False(File.Exists(staleOurs));
            Assert.True(File.Exists(staleOther));
            Assert.True(File.Exists(stalePrefix));
            Assert.True(File.Exists(staleNonGuid));

            // A concurrent writer's in-flight temp (still held open with a deny-all share) is
            // never swept: the sweep can't acquire it, so it skips rather than destroying a live
            // save's data. Simulate the live writer by holding a precise-match temp open.
            var liveTemp = Path.Combine(dir, $".tmp-state.bin-{Guid.NewGuid():N}");
            using (new FileStream(liveTemp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                WriteVia(target, "v3");
                Assert.True(File.Exists(liveTemp)); // held open → skipped
            }
            Assert.Equal("v3", File.ReadAllText(target));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>
    /// Up-front validation fails before any data is written: a target whose directory does
    /// not exist (the staged copy must live in that directory so the commit rename never
    /// crosses filesystems), an empty path, a path that reuses the reserved staging prefix,
    /// and a null content writer.
    /// </summary>
    [Fact]
    public void TestAtomicFileWriterValidationCoverage()
    {
        var dir = NewScratchDir();
        try
        {
            var missingDir = Path.Combine(dir, "does-not-exist");
            var ex = Assert.Throws<DirectoryNotFoundException>(
                () => WriteVia(Path.Combine(missingDir, "state.bin"), "v1"));
            Assert.Contains("does not exist", ex.Message);
            Assert.False(Directory.Exists(missingDir));       // nothing was created or written

            Assert.Throws<ArgumentException>(() => WriteVia("", "v1"));
            Assert.Throws<ArgumentException>(() => WriteVia("   ", "v1"));
            Assert.Throws<ArgumentException>(() => WriteVia(Path.Combine(dir, ".tmp-state.bin"), "v1"));
            Assert.Throws<ArgumentNullException>(() => AtomicFileWriter.WriteFile(Path.Combine(dir, "s.bin"), null!));
            Assert.Empty(Directory.GetFileSystemEntries(dir));

            Assert.True(AtomicFileWriter.IsTempName(".tmp-run-42"));
            Assert.False(AtomicFileWriter.IsTempName("run-42"));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>
    /// The directory shape for container formats: a commit atomically renames the fully
    /// populated staged directory into place (with its files flushed); the target existing
    /// beforehand is rejected up front (a rename cannot atomically replace a non-empty
    /// directory) without invoking the populate callback; and a failure while populating (or in
    /// the commit window) removes the staged directory.
    /// </summary>
    [Fact]
    public void TestAtomicFileWriterWriteDirectoryCoverage()
    {
        var dir = NewScratchDir();
        try
        {
            var target = Path.Combine(dir, "run-001");
            AtomicFileWriter.WriteDirectory(target, temp =>
            {
                File.WriteAllText(Path.Combine(temp, "weights.bin"), "w");
                Directory.CreateDirectory(Path.Combine(temp, "sub"));
                File.WriteAllText(Path.Combine(temp, "sub", "meta.json"), "{}");
            });
            Assert.Equal("w", File.ReadAllText(Path.Combine(target, "weights.bin")));
            Assert.Equal("{}", File.ReadAllText(Path.Combine(target, "sub", "meta.json")));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // Existing target: rejected before any staging happens.
            bool populated = false;
            Assert.Throws<IOException>(() => AtomicFileWriter.WriteDirectory(target, _ => populated = true));
            Assert.False(populated);

            // Populate failure and commit-window crash both leave no staged debris and no target.
            var target2 = Path.Combine(dir, "run-002");
            Assert.Throws<InvalidOperationException>(() => AtomicFileWriter.WriteDirectory(
                target2, _ => throw new InvalidOperationException("populate boom")));
            AtomicFileWriter.CommitFaultInjection = p =>
            {
                if (p.StartsWith(dir, StringComparison.Ordinal)) throw new IOException("injected crash");
            };
            try
            {
                Assert.Throws<IOException>(() => AtomicFileWriter.WriteDirectory(
                    target2, temp => File.WriteAllText(Path.Combine(temp, "weights.bin"), "w2")));
            }
            finally { AtomicFileWriter.CommitFaultInjection = null; }
            Assert.False(Directory.Exists(target2));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));

            // Fresh-name-per-version pattern: a second distinct target commits alongside the
            // first, each fully populated, with no staged debris left behind.
            AtomicFileWriter.WriteDirectory(
                target2, temp => File.WriteAllText(Path.Combine(temp, "weights.bin"), "w2"));
            Assert.Equal("w", File.ReadAllText(Path.Combine(target, "weights.bin")));
            Assert.Equal("w2", File.ReadAllText(Path.Combine(target2, "weights.bin")));
            Assert.Empty(Directory.GetFileSystemEntries(dir, ".tmp-*"));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
