using Shorokoo.Core;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Core.Graph
{
    /// <summary>
    /// Captured trainable-parameter info for a single specific model id: the
    /// id, its target Function, and the initializer-parameter <see cref="TensorData"/>
    /// values (shape vector + any other initializer inputs) observed at the
    /// call site that produced it.
    /// </summary>
    internal struct TrainableParamInfo
    {
        public readonly ImmutableArray<TensorData> TrainableParamInputParamValues { get; init; }
        public readonly ModelId SpecificModelId { get; init; }
        public readonly Function TargetFn { get; init; }
        /// <summary>
        /// The parameter's shape. A shaped initializer takes the shape vector as its first
        /// initializer input, so it is read straight off that value. A <b>rank-0</b> initializer
        /// (<c>Inline() =&gt; Scalar(0f)</c>) declares the scalar shape in its return type instead,
        /// and takes no shape input — any input it does take is a seed value, not a shape vector —
        /// so its declared rank decides first. Those are the only two ways an initializer states
        /// its shape; one that bakes a shape into its body and takes no input states it nowhere,
        /// and is rejected rather than guessed at.
        /// </summary>
        public Shape Shape
        {
            get
            {
                var declaredRank = TargetFn.OutputRankOverrides.Length > 0
                    ? TargetFn.OutputRankOverrides[0] : null;
                if (declaredRank == 0) return Shape.Scalar;
                if (TrainableParamInputParamValues.IsDefaultOrEmpty)
                    throw new InvalidOperationException(
                        $"Parameter initializer '{TargetFn.DefaultName}' takes no initializer input and "
                        + "does not return a rank-0 Scalar<T>, so the shape of the parameter it creates "
                        + "is not stated anywhere the pipeline can read it. An initializer must either "
                        + "take the parameter's shape as its first Inline parameter "
                        + "(Inline(Vector<int64> shape, ...)) or declare the scalar shape by returning "
                        + "Scalar<T>; a shape baked into the body of a no-input Inline is neither.");
                var shapeTensorData = TrainableParamInputParamValues.First();
                var shapeLongs = shapeTensorData.As<int64>().AccessMemory().ToArray();
                return new Shape(shapeLongs);
            }
        }
    }

    /// <summary>
    /// Map of generalized model-id templates to <see cref="ModelParamIdentifierTemplate"/>s,
    /// with helpers to walk a specific model id back to its generalized form
    /// and resolve a per-id template.
    /// </summary>
    internal struct IdTemplateInfos
    {
        public readonly ImmutableDictionary<ModelId, ModelParamIdentifierTemplate> IdTemplates { get; init; }

        private ImmutableDictionary<ModelId, bool>? isNextPartLoop;
        private ImmutableDictionary<ModelId, bool> getIsNextPartLoop()
        {
            if (isNextPartLoop is null)
            {
                var dctIsNextPartLoop = new Dictionary<ModelId, bool>();
                var maxLength = IdTemplates.Keys.Select(x => x.Vals.Length).Max();
                for (var numIds = 1; numIds <= maxLength; numIds++)
                {
                    var toAdd =
                    IdTemplates.Keys.Where(x => x.Vals.Length > numIds)
                                .Select(x => (startPart: new ModelId(x.Vals.Take(numIds).ToArray()), fullId: x))
                                .DistinctBy(x => x.startPart)
                                .Select(x => (x.startPart, x.fullId.Vals[numIds] == -1));

                    dctIsNextPartLoop.AddAll(toAdd);
                }

                this.isNextPartLoop = dctIsNextPartLoop.ToImmutableDictionary();
            }

            return isNextPartLoop;
        }

        public ModelId ToGeneralModelId(ModelId specificModelId)
        {
            var curGeneralModelId = new ModelId(specificModelId.Vals[0]);
            var dctIsNextPartLoop = getIsNextPartLoop();

            while (!IdTemplates.ContainsKey(curGeneralModelId))
            {
                if (!dctIsNextPartLoop.ContainsKey(curGeneralModelId))
                    throw new ArgumentException(nameof(specificModelId), "The provided specific model id is not in this infos.");

                var isLoop = dctIsNextPartLoop[curGeneralModelId];
                curGeneralModelId = isLoop ?
                                        new ModelId([.. curGeneralModelId.Vals, -1]) :
                                        new ModelId([.. curGeneralModelId.Vals, specificModelId.Vals[curGeneralModelId.Vals.Length]]);
            }

            return curGeneralModelId;
        }

        public ModelParamIdentifierTemplate GetSpecificIdentifierTemplate(ModelId specificModelId)
        {
            var generalModelId = ToGeneralModelId(specificModelId);
            var generalIdTemplate = this.IdTemplates[generalModelId];

            return generalIdTemplate.ToSpecificTemplate(specificModelId);
        }
    }
}
