using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Shorokoo.Core;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Utils;
using Shorokoo.Graph;
using Shorokoo.Onnx;

namespace Shorokoo
{
    /// <summary>
    /// The safetensors weight-exchange boundary: <see cref="ExportSafeTensors"/> writes a
    /// concrete model's weights to a single standard .safetensors file (canonical Shorokoo
    /// parameter names by default, or third-party names via a naming scheme), and
    /// <see cref="ImportSafeTensors(ComputationGraph, string, ModuleParamSetNamingScheme?)"/>
    /// binds a foreign .safetensors file onto a concrete architecture with the scheme applied
    /// at the boundary — strictly: every source tensor must land on a parameter, every
    /// parameter must be covered, and dtype/shape must match, each failure naming the
    /// offending tensor. <see cref="ImportSafeTensorsToCheckpoint"/> lands the imported
    /// result straight in a native .skpt in one call.
    /// </summary>
    public static partial class Checkpoint
    {
        /// <summary>
        /// Exports the model's weights to a single standard .safetensors file. The graph must
        /// be a weight-filled <see cref="GraphKind.ConcreteModel"/>; every weight parameter is
        /// written (the RNG identity parameter is model definition, not a weight, and is not
        /// exported). With no <paramref name="namingScheme"/>, tensors carry the parameters'
        /// canonical Shorokoo ids — the same names <see cref="CheckpointBuilder.Save"/> records
        /// in a .skpt, and the names <see cref="ImportSafeTensors(ComputationGraph, string,
        /// ModuleParamSetNamingScheme?)"/> binds with no scheme. With a scheme (e.g. a
        /// <see cref="SimplePatternNamingScheme"/> targeting PyTorch/timm names), each
        /// canonical id is translated via
        /// <see cref="ModuleParamSetNamingScheme.ToName(string)"/>; a parameter the scheme does
        /// not cover, or two parameters colliding on one exported name, fail loudly naming the
        /// parameters — weights are never silently dropped or overwritten on export.
        ///
        /// <para>The write is atomic (staged to a temp file beside
        /// <paramref name="filePath"/> and committed by rename); the target's directory must
        /// already exist. The output is plain safetensors — loadable by any safetensors
        /// implementation.</para>
        /// </summary>
        /// <param name="concreteModel">The weight-filled concrete model to export.</param>
        /// <param name="filePath">Target path; a <c>.safetensors</c> extension is conventional.</param>
        /// <param name="namingScheme">Optional scheme translating canonical parameter ids to
        /// third-party tensor names; null exports canonical Shorokoo names.</param>
        public static void ExportSafeTensors(
            ComputationGraph concreteModel,
            string filePath,
            ModuleParamSetNamingScheme? namingScheme = null)
        {
            if (concreteModel is null) throw new ArgumentNullException(nameof(concreteModel));
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("SafeTensors path cannot be null or empty.", nameof(filePath));
            if (concreteModel.Kind != GraphKind.ConcreteModel)
                throw new InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                    "Checkpoint.ExportSafeTensors", "a 'concrete-model' graph", concreteModel.Kind,
                    "Only a weight-filled concrete model has weight tensors to export. Lower the " +
                    "graph with ToConcreteArchitecture(inputHints, ...).ToConcreteModel(...) first."));

            var weightNodes = CheckpointBuilder.CollectWeightNodes(
                concreteModel.ToInternal(), "Checkpoint.ExportSafeTensors");
            if (weightNodes.Count == 0)
                throw new InvalidOperationException(
                    "Checkpoint.ExportSafeTensors: the model has no weight parameters — there is " +
                    "nothing to export.");

            var tensors = new List<SafeTensor>(weightNodes.Count);
            var paramIdByName = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var node in weightNodes)
            {
                // The node identifier is the serialized "[ModelId]:Cat.Mod.Param" form; the
                // canonical parameter name — what the canonical naming scheme produces and what
                // ImportSafeTensors binds with no scheme — is the template-string portion.
                var paramId = Core.Nodes.Processors.Training.FastDiscoverParamsHelpers
                    .ExtractTemplateString(node.IdentifierTemplate!);
                var name = namingScheme is null ? paramId : namingScheme.ToName(paramId);
                if (string.IsNullOrEmpty(name))
                    throw new InvalidOperationException(
                        $"Checkpoint.ExportSafeTensors: parameter '{paramId}' maps to no tensor " +
                        "name under the naming scheme — every weight must be named to be " +
                        "exported. Add a rule covering it, or pass no scheme to export canonical " +
                        "Shorokoo names.");
                if (paramIdByName.TryGetValue(name, out var otherParam))
                    throw new InvalidOperationException(
                        $"Checkpoint.ExportSafeTensors: parameters '{otherParam}' and '{paramId}' " +
                        $"both map to tensor name '{name}' under the naming scheme; exported " +
                        "tensor names must be unique or one weight would silently overwrite the other.");
                paramIdByName[name] = paramId;

                var data = node.GetTensorData()!;
                tensors.Add(new SafeTensor(
                    name, data, SafeTensorLoader.DTypeToSafeTensorDType(data.DType), data.Shape.Dims));
            }

            AtomicFileWriter.WriteFile(
                filePath, stream => SafeTensorLoader.SaveSafeTensorsToStream(stream, tensors));
        }

        /// <summary>
        /// Imports a .safetensors file onto a concrete architecture, applying
        /// <paramref name="namingScheme"/> at the boundary to map the file's tensor names onto
        /// the architecture's parameters, and returns the bound
        /// <see cref="GraphKind.ConcreteModel"/>. The binding itself is the standard
        /// <see cref="ComputationGraph.ToConcreteModel(ModelParamList, ModuleParamSetNamingScheme)"/>
        /// path — what import adds is <b>strictness</b>: where plain <c>ToConcreteModel</c>
        /// silently drops names that do not resolve, import fails loudly, naming the offending
        /// tensor, on (a) a source tensor that maps to no parameter, (b) a required parameter
        /// with no source tensor, (c) two source tensors mapping to one parameter, and (d) a
        /// dtype or shape mismatch after mapping. All validation happens before any binding, so
        /// a failed import never yields a partially bound model.
        ///
        /// <para>With no scheme, the file must carry canonical Shorokoo parameter ids (what
        /// <see cref="ExportSafeTensors"/> writes by default). For a foreign file
        /// (PyTorch/timm names), pass the same kind of scheme
        /// <c>ToConcreteModel(weights, scheme)</c> takes — built with the ModelId format DSL
        /// (<see cref="ModelIdNamingScheme"/>) or the pattern DSL
        /// (<see cref="SimplePatternNamingScheme"/>).</para>
        ///
        /// <para>Truncation and corruption of the file are refused by the safetensors loader's
        /// declared-vs-actual size checks, naming the file. A safetensors
        /// <c>__metadata__</c> block is metadata, not a tensor, and is ignored.</para>
        /// </summary>
        /// <param name="concreteArchitecture">The architecture to bind onto (from
        /// <see cref="ComputationGraph.ToConcreteArchitecture"/>).</param>
        /// <param name="filePath">Path of the .safetensors file to import.</param>
        /// <param name="namingScheme">Optional scheme mapping the architecture's parameters to
        /// the file's tensor names; null expects canonical Shorokoo names.</param>
        /// <returns>The concrete model with every parameter bound from the file.</returns>
        public static ComputationGraph ImportSafeTensors(
            ComputationGraph concreteArchitecture,
            string filePath,
            ModuleParamSetNamingScheme? namingScheme = null)
        {
            if (concreteArchitecture is null) throw new ArgumentNullException(nameof(concreteArchitecture));
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("SafeTensors path cannot be null or empty.", nameof(filePath));
            if (concreteArchitecture.Kind != GraphKind.ConcreteArchitecture)
                throw new InvalidOperationException(SrkFileFormat.KindMismatchMessage(
                    "Checkpoint.ImportSafeTensors", "a 'concrete-architecture' graph",
                    concreteArchitecture.Kind,
                    "Import binds the file's tensors onto unbound parameters. Lower a module " +
                    "graph with ToConcreteArchitecture(inputHints, ...) first; a weight-filled " +
                    "concrete model has no unbound parameters to import into."));

            // The loader's fail-loud checks (declared header length and every tensor's
            // data_offsets validated against the actual byte count) run here, naming filePath.
            var tensors = SafeTensorLoader.LoadSafeTensors(filePath);

            var paramInfos = concreteArchitecture.GetConcreteModelParamInfos();
            namingScheme ??= ModuleParamSetNamingScheme.CreateShorokooNamingScheme(paramInfos);

            // One expected file name per distinct parameter (the same parameter appears once
            // per invocation of its module; all copies share one ModelId and one tensor).
            // Two parameters colliding on one mapped name is a scheme defect and fails here,
            // before any file name is looked up against the ambiguous map.
            var distinctParams = paramInfos.ParamInfos
                .GroupBy(p => p.ModelId).Select(g => g.First()).ToList();
            var paramByMappedName = new Dictionary<string, ConcreteModelParamInfo>(StringComparer.Ordinal);
            var mappedNames = new string?[distinctParams.Count];
            for (int i = 0; i < distinctParams.Count; i++)
            {
                var info = distinctParams[i];
                string? mappedName;
                try
                {
                    mappedName = namingScheme.ToName(info);
                }
                catch (InvalidOperationException)
                {
                    // A ModelIdNamingScheme throws when no format rule matches the ModelId;
                    // for import that simply means the scheme does not cover this parameter.
                    mappedName = null;
                }
                mappedNames[i] = mappedName;
                if (mappedName is null) continue;
                if (paramByMappedName.TryGetValue(mappedName, out var other))
                    throw new InvalidDataException(
                        $"'{filePath}': parameters '{other.ToShorokooIdString()}' and " +
                        $"'{info.ToShorokooIdString()}' both map to source tensor name " +
                        $"'{mappedName}' under the naming scheme — the mapping is ambiguous, so " +
                        "one tensor would bind two parameters. Fix the scheme so parameter names " +
                        "stay unique.");
                paramByMappedName[mappedName] = info;
            }

            // (a) Every source tensor must land on a parameter.
            var boundNames = new HashSet<string>(StringComparer.Ordinal);
            foreach (var tensor in tensors)
            {
                if (!paramByMappedName.TryGetValue(tensor.Name, out var info))
                {
                    var hint = tensor.Name == TrainingCheckpoint.CheckpointMarkerName
                        ? " The file is a Shorokoo training checkpoint (it carries the " +
                          $"'{TrainingCheckpoint.CheckpointMarkerName}' marker), not a plain " +
                          "weights file — resume it with TrainingRig.LoadCheckpoint instead."
                        : " Does the file belong to this model (and does the scheme cover its names)?";
                    throw new InvalidDataException(
                        $"'{filePath}': source tensor '{tensor.Name}' maps to no parameter of " +
                        $"this architecture under the naming scheme.{hint}");
                }
                // Distinct safetensors names are distinct header keys, so a repeat here is
                // impossible from a well-formed file; guard anyway so a hand-built duplicate
                // fails named instead of binding whichever copy parses last.
                if (!boundNames.Add(tensor.Name))
                    throw new InvalidDataException(
                        $"'{filePath}': the file contains two tensors named '{tensor.Name}' — " +
                        "safetensors names must be unique.");

                // (d) Dtype/shape must match after mapping — same comparison Checkpoint.Load
                // applies when rebinding a .skpt.
                if (info.DType.ToIVarType() != tensor.Data.DType.ToIVarType()
                    || !info.Shape.Dims.SequenceEqual(tensor.Data.Shape.Dims))
                    throw new InvalidDataException(
                        $"'{filePath}': tensor '{tensor.Name}' (dtype {tensor.Data.DType}, shape " +
                        $"[{string.Join(",", tensor.Data.Shape.Dims)}]) does not match parameter " +
                        $"'{info.ToShorokooIdString()}' (dtype {info.DType}, shape " +
                        $"[{string.Join(",", info.Shape.Dims)}]).");
            }

            // (b) Every required parameter must have a source tensor. Report in the
            // architecture's parameter order so the first missing parameter is deterministic.
            for (int i = 0; i < distinctParams.Count; i++)
            {
                var info = distinctParams[i];
                if (mappedNames[i] is not string expectedName)
                    throw new InvalidDataException(
                        $"'{filePath}': required model parameter '{info.ToShorokooIdString()}' " +
                        "maps to no source tensor name under the naming scheme — add a rule " +
                        "covering it. Every parameter must bind for the import to be complete.");
                if (!boundNames.Contains(expectedName))
                    throw new InvalidDataException(
                        $"'{filePath}': required model parameter '{info.ToShorokooIdString()}' " +
                        $"has no source tensor in the file (expected tensor '{expectedName}'). " +
                        "Does the file belong to this model?");
            }

            // Bind through the standard path — the same ModelParamList / ToConcreteModel
            // binding the checkpoint machinery uses; validation above guarantees nothing is
            // silently dropped by it.
            var weights = new ModelParamList(
                tensors.Select(t => new KeyValuePair<string, TensorData>(t.Name, t.Data)),
                ModelParamType.TrainableParam);
            return concreteArchitecture.ToConcreteModel(weights, namingScheme);
        }

        /// <summary>
        /// Imports a foreign .safetensors file onto a concrete architecture (exactly as
        /// <see cref="ImportSafeTensors(ComputationGraph, string, ModuleParamSetNamingScheme?)"/>,
        /// including all fail-loud mapping checks) and lands the bound result as a native .skpt
        /// checkpoint in one call, via the standard
        /// <c>Checkpoint.From(model).WithModel().WithWeights().Save(...)</c> writer — so
        /// "foreign safetensors → native checkpoint" is a single step. The .skpt write is
        /// atomic; nothing is written when the import fails. Returns the bound concrete model
        /// (the same graph <see cref="Load"/> reproduces from the written checkpoint).
        /// </summary>
        /// <param name="concreteArchitecture">The architecture to bind onto.</param>
        /// <param name="safeTensorsPath">Path of the .safetensors file to import.</param>
        /// <param name="checkpointPath">Target path of the .skpt checkpoint to write.</param>
        /// <param name="namingScheme">Optional scheme mapping the architecture's parameters to
        /// the file's tensor names; null expects canonical Shorokoo names.</param>
        public static ComputationGraph ImportSafeTensorsToCheckpoint(
            ComputationGraph concreteArchitecture,
            string safeTensorsPath,
            string checkpointPath,
            ModuleParamSetNamingScheme? namingScheme = null)
        {
            if (string.IsNullOrWhiteSpace(checkpointPath))
                throw new ArgumentException("Checkpoint path cannot be null or empty.", nameof(checkpointPath));

            var model = ImportSafeTensors(concreteArchitecture, safeTensorsPath, namingScheme);
            From(model).WithModel().WithWeights().Save(checkpointPath);
            return model;
        }
    }
}
