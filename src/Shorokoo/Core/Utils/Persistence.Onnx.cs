using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Shorokoo.Core;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Onnx;
using IR = Shorokoo.Core.Factory.IR;

namespace Shorokoo
{
    /// <summary>
    /// The ONNX exchange boundary, mirroring the safetensors boundary in
    /// <c>Persistence.SafeTensors.cs</c>: <see cref="ExportOnnx"/> writes a concrete model
    /// to a standard, externally-loadable ("vanilla" dialect) <c>.onnx</c>, and
    /// <see cref="ImportOnnx"/> turns a foreign vanilla <c>.onnx</c> back into a native
    /// runnable <see cref="ComputationGraph"/> — with each ONNX initializer's name adopted
    /// as the parameter identifier at the boundary (optionally translated through the same
    /// <see cref="ModuleParamSetNamingScheme"/> surface the safetensors import uses).
    /// <see cref="ImportOnnxToCheckpoint"/> lands the imported model straight in a native
    /// <c>.skpt</c> in one call, via the standard container writer.
    ///
    /// <para>Importing a vanilla ONNX is lossy by design — a concrete model's module
    /// structure and hyper defaults are not expressible in the vanilla dialect, so the
    /// result is a value-faithful concrete model, not a structural round-trip. For a
    /// structural round-trip, use the native <c>.skpt</c> container (<see cref="From"/> /
    /// <see cref="Load(string)"/>).</para>
    /// </summary>
    public static partial class Persistence
    {
        /// <summary>
        /// Exports <paramref name="concreteModel"/> to a standard, externally-loadable
        /// ("vanilla" dialect) <c>.onnx</c> file — every emitted node is a stock ONNX op or a
        /// call to an emitted function, so the file loads in any conforming ONNX runtime. The
        /// graph must be a weight-filled <see cref="GraphKind.ConcreteModel"/>; a graph still
        /// carrying Shorokoo-internal orchestration ops is refused (naming the offending ops)
        /// rather than written as a file only a Shorokoo runtime could load. The write is
        /// atomic (staged to a temp file beside <paramref name="filePath"/> and committed by
        /// rename); the target's directory must already exist.
        /// </summary>
        /// <param name="concreteModel">The weight-filled concrete model to export.</param>
        /// <param name="filePath">Target path; a <c>.onnx</c> extension is conventional.</param>
        /// <param name="opset">Default-domain opset stamp; raised automatically when the
        /// graph uses ops introduced in a later opset.</param>
        public static void ExportOnnx(
            ComputationGraph concreteModel,
            string filePath,
            OpSetVersion opset = OpSetVersion.OPS_21)
        {
            if (concreteModel is null) throw new ArgumentNullException(nameof(concreteModel));
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("ONNX path cannot be null or empty.", nameof(filePath));

            // BuildOnnxModel enforces the concrete-model kind and the vanilla-dialect
            // guarantee, naming offending ops on failure — no need to duplicate that here.
            var model = FastOnnxModelBuilder.BuildOnnxModel(concreteModel, opset);
            AtomicFileWriter.WriteFile(
                filePath, stream => ProtoBuf.Serializer.Serialize(stream, model));
        }

        /// <summary>
        /// Imports a foreign vanilla <c>.onnx</c> file into a native runnable
        /// <see cref="ComputationGraph"/> through the existing ONNX reader
        /// (<see cref="OnnxModelImporter"/>). Models using the standard ONNX external-data
        /// mechanism (initializer bytes in a <c>.data</c> side file) load transparently —
        /// <c>location</c> keys resolve against <paramref name="filePath"/>'s directory.
        ///
        /// <para>At the boundary, each foreign initializer's ONNX name is adopted as its
        /// parameter identifier so the imported model can be named, checkpointed and reloaded
        /// natively (a Shorokoo-produced ONNX already carries canonical identifiers, which are
        /// kept as-is). When a <paramref name="namingScheme"/> is supplied — the same
        /// <see cref="ModuleParamSetNamingScheme"/> surface <see cref="ImportSafeTensors"/>
        /// takes — each foreign initializer name is translated through it and the result used
        /// as the identifier; a name the scheme does not cover keeps its original ONNX name,
        /// which is always a valid identifier. Two initializers resolving to one identifier
        /// fail loudly, naming both.</para>
        ///
        /// <para>Fails loudly, naming the offending op and the file, on a construct the reader
        /// cannot ingest (an op outside the vanilla ONNX dialect Shorokoo reads, or a node in
        /// an unknown domain); a truncated or garbage file fails loudly naming the file.</para>
        /// </summary>
        /// <param name="filePath">Path of the <c>.onnx</c> file to import.</param>
        /// <param name="namingScheme">Optional scheme translating foreign initializer names to
        /// canonical Shorokoo identifiers; null adopts the ONNX names verbatim.</param>
        /// <returns>The imported native graph (a concrete model for a vanilla export).</returns>
        public static ComputationGraph ImportOnnx(
            string filePath,
            ModuleParamSetNamingScheme? namingScheme = null)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("ONNX path cannot be null or empty.", nameof(filePath));
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"'{filePath}': ONNX file not found.", filePath);

            byte[] bytes;
            try
            {
                bytes = File.ReadAllBytes(filePath);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                throw new InvalidDataException($"'{filePath}': could not be read ({e.Message}).", e);
            }

            IR.ModelProto model;
            try
            {
                using var ms = new MemoryStream(bytes, writable: false);
                model = ProtoBuf.Serializer.Deserialize<IR.ModelProto>(ms);
            }
            catch (Exception e) when (e is ProtoBuf.ProtoException
                or EndOfStreamException
                or IndexOutOfRangeException
                or ArgumentException
                or OverflowException
                or FormatException
                or InvalidOperationException)
            {
                throw new InvalidDataException(
                    $"'{filePath}': not a readable ONNX model — failed to parse the protobuf " +
                    $"({e.GetType().Name}: {e.Message}). The file is corrupt or not an ONNX model.", e);
            }

            if (model.Graph is null)
                throw new InvalidDataException(
                    $"'{filePath}': the ONNX model carries no graph — the file is empty, corrupt, " +
                    "or not an ONNX model.");

            // Boundary gate: reject anything outside the vanilla ONNX dialect Shorokoo reads,
            // naming the op and the file, before the reader (whose deeper failure on an unknown
            // op is a bare lookup error).
            ValidateVanillaDialect(model, filePath);

            InternalComputationGraph graph;
            GraphKind? taggedKind;
            try
            {
                var fullDir = Path.GetDirectoryName(Path.GetFullPath(filePath));
                (graph, taggedKind) = OnnxModelImporter.FromModelProtoWithKindTag(model, fullDir);
            }
            catch (Exception e) when (e is ProtoBuf.ProtoException
                or EndOfStreamException
                or IndexOutOfRangeException
                or ArgumentOutOfRangeException
                or OverflowException
                or FormatException
                or KeyNotFoundException)
            {
                throw new InvalidDataException(
                    $"'{filePath}': the ONNX model could not be imported " +
                    $"({e.GetType().Name}: {e.Message}). The file is corrupt or uses a construct " +
                    "Shorokoo's importer cannot ingest.", e);
            }

            // Assign identifiers on the mutable internal graph, then freeze — a
            // ComputationGraph is immutable, so the naming must happen before it is wrapped.
            AdoptInitializerIdentifiers(graph, namingScheme, filePath);
            return new ComputationGraph(graph, taggedKind ?? SrkFileFormat.DetectStage(graph));
        }

        /// <summary>
        /// Imports a foreign vanilla <c>.onnx</c> (exactly as
        /// <see cref="ImportOnnx(string, ModuleParamSetNamingScheme?)"/>, including the
        /// boundary naming and all fail-loud checks) and lands the imported concrete model
        /// straight in a native <c>.skpt</c> checkpoint in one call, via the standard
        /// <c>Persistence.From(model).WithModel().WithWeights().Save(...)</c> writer — so
        /// "foreign ONNX → native checkpoint" is a single step. The <c>.skpt</c> write is
        /// atomic; nothing is written when the import fails. Returns the imported concrete
        /// model (the same graph <see cref="Load(string)"/> reproduces from the written checkpoint).
        /// </summary>
        /// <param name="onnxPath">Path of the <c>.onnx</c> file to import.</param>
        /// <param name="checkpointPath">Target path of the <c>.skpt</c> checkpoint to write.</param>
        /// <param name="namingScheme">Optional scheme translating foreign initializer names to
        /// canonical Shorokoo identifiers; null adopts the ONNX names verbatim.</param>
        public static ComputationGraph ImportOnnxToCheckpoint(
            string onnxPath,
            string checkpointPath,
            ModuleParamSetNamingScheme? namingScheme = null)
        {
            if (string.IsNullOrWhiteSpace(checkpointPath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(checkpointPath));

            var model = ImportOnnx(onnxPath, namingScheme);
            if (model.Kind != GraphKind.ConcreteModel)
                throw new InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                    "Persistence.ImportOnnxToCheckpoint", "a 'concrete-model' graph", model.Kind,
                    "Only a weight-filled concrete model lands as a .skpt; a vanilla ONNX that " +
                    "imports as another kind cannot be checkpointed here."));
            From(model).WithModel().WithWeights().Save(checkpointPath);
            return model;
        }

        /// <summary>
        /// Gives each foreign initializer (a <c>MODEL_PARAM_DATA</c> node) a valid, unique
        /// canonical parameter identifier at the import boundary. A Shorokoo-produced ONNX
        /// already stamps a canonical identifier template on every initializer (and on the RNG
        /// identity parameter) — those are kept verbatim. A third-party ONNX imports its
        /// initializers as constants with no identifier and the raw ONNX name in their friendly
        /// name; each is promoted here to a canonical <c>[modelId]:TrainableParam#0.name#0</c>
        /// identifier so the model can be named, checkpointed and reloaded natively — the .skpt
        /// container's parameter identifiers are canonical templates, not arbitrary strings.
        ///
        /// <para>The identifier's canonical <b>name</b> is, by default, the ONNX initializer
        /// name (escaped into a single template part). When a <paramref name="namingScheme"/>
        /// is supplied — the same <see cref="ModuleParamSetNamingScheme"/> surface
        /// <see cref="ImportSafeTensors"/> takes — the ONNX name is translated through it and
        /// the result used as the canonical name; a name the scheme does not cover keeps its
        /// ONNX name. Two initializers resolving to one canonical name fail loudly, naming both,
        /// so no weight can silently overwrite another in the checkpoint.</para>
        /// </summary>
        private static void AdoptInitializerIdentifiers(
            InternalComputationGraph graph, ModuleParamSetNamingScheme? namingScheme, string filePath)
        {
            // Canonical name (the template-string portion, without the synthetic [modelId]) ->
            // the ONNX initializer it came from, so a collision names both sources.
            var sourceByCanonicalName = new Dictionary<string, string>(StringComparer.Ordinal);
            // Distinct synthetic ModelId per promoted parameter. A foreign model has no other
            // templated parameters, so a simple counter cannot collide; a Shorokoo export's
            // parameters are already templated and are skipped below.
            int nextModelId = 1;

            foreach (var node in graph.Nodes)
            {
                if (node.OpCode != InternalOpCodes.MODEL_PARAM_DATA) continue;
                // The RNG identity parameter is model definition, not a weight; it carries the
                // reserved identifier a Shorokoo export stamps and must keep it untouched.
                if (node.IdentifierTemplate == FastWireRngKeyDerivation.RngSeedIdentifierTemplate)
                    continue;
                // A Shorokoo-produced ONNX already carries a canonical identifier template; keep it.
                if (!string.IsNullOrEmpty(node.IdentifierTemplate)) continue;

                var onnxName = node.FriendlyName;
                if (string.IsNullOrEmpty(onnxName))
                    throw new InvalidDataException(
                        $"'{filePath}': an ONNX initializer has no name, so its weight cannot be " +
                        "identified for a native checkpoint. Every initializer must be named.");

                // Canonical name: the scheme's translation of the ONNX name, else the ONNX name
                // itself escaped into a single trainable-parameter template part.
                string canonicalName;
                string? mapped = null;
                if (namingScheme is not null)
                {
                    try
                    {
                        mapped = namingScheme.ToName(onnxName);
                    }
                    // A scheme that cannot translate this string (no rule matches, or the scheme
                    // maps ModelIds rather than strings) simply does not cover the name.
                    catch (Exception e) when (e is InvalidOperationException or NotSupportedException)
                    {
                        mapped = null;
                    }
                }
                canonicalName = !string.IsNullOrEmpty(mapped)
                    ? mapped!
                    : $"TrainableParam#0.{ModelParamIdentifierTemplatePart.EscapePartName(onnxName)}#0";

                if (sourceByCanonicalName.TryGetValue(canonicalName, out var otherSource))
                    throw new InvalidDataException(
                        $"'{filePath}': ONNX initializers '{otherSource}' and '{onnxName}' both resolve " +
                        $"to parameter name '{canonicalName}'" +
                        (namingScheme is null ? "" : " under the naming scheme") +
                        " — names must be unique or one weight would overwrite the other in a native " +
                        "checkpoint.");
                sourceByCanonicalName[canonicalName] = onnxName;

                var template = $"[{nextModelId}]:{canonicalName}";
                try
                {
                    // Validate the synthesized template parses as a canonical identifier — a
                    // scheme may produce a name that is not a legal template part.
                    _ = new ModelParamIdentifierTemplate(template);
                }
                catch (Exception e) when (e is ArgumentException or IndexOutOfRangeException or FormatException)
                {
                    throw new InvalidDataException(
                        $"'{filePath}': ONNX initializer '{onnxName}' maps to parameter name " +
                        $"'{canonicalName}', which is not a valid Shorokoo parameter identifier " +
                        $"({e.Message}). Adjust the naming scheme to produce a canonical name of the " +
                        "form 'Category#i.Name#j'.", e);
                }
                nextModelId++;

                node.IdentifierTemplate = template;
                node.FriendlyName = onnxName;
            }
        }

        /// <summary>
        /// Rejects a model carrying any construct the ONNX reader cannot ingest — an op outside
        /// the vanilla ONNX dialect Shorokoo reads, or a node in an unknown domain — naming the
        /// first offending op, its node, and the file. Walks the main graph, its nested
        /// control-flow subgraphs, and every function body. Nodes in the emitted
        /// <c>Functions</c> domain are accepted when the model declares a matching function.
        /// </summary>
        private static void ValidateVanillaDialect(IR.ModelProto model, string filePath)
        {
            var declaredFunctions = new HashSet<string>(
                model.Functions?.Select(f => f.Name) ?? Enumerable.Empty<string>(),
                StringComparer.Ordinal);

            void CheckNode(IR.NodeProto node)
            {
                var domain = node.Domain;
                bool ok = domain switch
                {
                    "" or "ai.onnx" or "ai.onnx.ml"
                        => Definitions.VanillaOpNames.Contains(node.OpType),
                    "Functions" => declaredFunctions.Contains(node.OpType),
                    _ => false,
                };
                if (!ok)
                {
                    var where = string.IsNullOrEmpty(node.Name) ? "" : $" (node '{node.Name}')";
                    var dom = string.IsNullOrEmpty(domain) ? "" : $" in domain '{domain}'";
                    throw new InvalidDataException(
                        $"'{filePath}': ONNX op '{node.OpType}'{where}{dom} is not a construct Shorokoo's " +
                        "importer can ingest. ImportOnnx accepts a vanilla ONNX model (standard ops and " +
                        "emitted function calls); a Shorokoo internal-dialect file loads via Persistence.Load.");
                }

                foreach (var attr in node.Attributes)
                {
                    if (attr.G is not null) CheckGraph(attr.G);
                    foreach (var g in attr.Graphs) CheckGraph(g);
                }
            }

            void CheckGraph(IR.GraphProto graph)
            {
                foreach (var node in graph.Nodes) CheckNode(node);
            }

            CheckGraph(model.Graph);
            if (model.Functions is not null)
                foreach (var fn in model.Functions)
                    foreach (var node in fn.Nodes)
                        CheckNode(node);
        }
    }
}
