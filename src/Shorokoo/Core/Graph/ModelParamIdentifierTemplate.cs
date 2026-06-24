using Microsoft.CodeAnalysis.CSharp.Syntax;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;

namespace Shorokoo.Core.Graph
{
    public enum ModelParamIdentifierTemplatePartType
    {
        Module,
        Loop,
        Param,
        Category
    }

    public class ModelParamIdentifierTemplatePart : IEquatable<ModelParamIdentifierTemplatePart>
    {
        public static ModelParamIdentifierTemplatePart ModuleCategory = new ModelParamIdentifierTemplatePart(ModelParamIdentifierTemplatePartType.Category, "Module", 0, null);
        public static ModelParamIdentifierTemplatePart TrainableParamCategory = new ModelParamIdentifierTemplatePart(ModelParamIdentifierTemplatePartType.Category, "TrainableParam", 0, null);
        public static ModelParamIdentifierTemplatePart ModelStateCategory = new ModelParamIdentifierTemplatePart(ModelParamIdentifierTemplatePartType.Category, "ModelState", 0, null);

        /// <summary>
        /// The type of this part of the identifier path.
        /// </summary>
        public ModelParamIdentifierTemplatePartType Type { get; private set; }

        /// <summary>
        /// The name of this part of the identifier path.
        /// </summary>
        public string Name { get; private set; }

        /// <summary>
        /// If name is not unique in this scope, this index is used to deduplicate it.
        /// When unique, this is 0.
        /// This is also a 0-based index.
        /// </summary>
        public int DeduplicationIndex { get; private set; }

        /// <summary>
        /// For Loop type only, the index of the loop iteration this part refers to.
        /// </summary>
        public int? LoopIndex { get; private set; }

        public ModelParamIdentifierTemplatePart(ModelParamIdentifierTemplatePartType type, string name, int deduplicationIndex = 0, int? loopIndex = null)
        {
            Type = type;
            Name = name ?? throw new ArgumentNullException(nameof(name));
            DeduplicationIndex = deduplicationIndex;
            LoopIndex = loopIndex;
        }

        public static string EscapePartName(string partName) =>
            partName
                .Replace(@"\", @"\s")
                .Replace(@"#", @"\h")
                .Replace(@":", @"\c")
                .Replace(@".", @"\d");

        public static string UnescapePartName(string partName) =>
            partName
                .Replace(@"\h", "#")
                .Replace(@"\c", ":")
                .Replace(@"\d", ".")
                .Replace(@"\s", @"\");

        public static ModelParamIdentifierTemplatePart Parse(string partString, bool isLastPart = false, bool isFirstPart = false)
        {
            if (string.IsNullOrWhiteSpace(partString))
                throw new ArgumentException("Part string cannot be null or whitespace.", nameof(partString));

            var isLoop = partString.StartsWith("Loop#");
            ModelParamIdentifierTemplatePartType type;

            if (isLastPart)
                type = ModelParamIdentifierTemplatePartType.Param;
            else if (isFirstPart)
                type = ModelParamIdentifierTemplatePartType.Category;
            else if (isLoop)
                type = ModelParamIdentifierTemplatePartType.Loop;
            else
                type = ModelParamIdentifierTemplatePartType.Module;

            var subparts = partString.Split(['#', ':']);
            if (subparts.Length < 2 || subparts.Length > 3)
                throw new ArgumentException($"Invalid part string format {partString}");

            var escapedName = subparts[0];
            var name = UnescapePartName(escapedName);

            var dedupeIndexString = subparts[1];
            if (!int.TryParse(dedupeIndexString, out int deduplicationIndex))
                throw new ArgumentException($"Invalid part string format: invalid deduplication index in part string '{partString}'", nameof(partString));

            int? loopIndex = null;
            if (isLoop)
            {
                if (subparts.Length != 3)
                    throw new ArgumentException("Invalid loop part string format: missing loop index.", nameof(partString));
                var loopIndexString = subparts[2];
                if (!int.TryParse(loopIndexString, out var theLoopIndex))
                    throw new ArgumentException($"Invalid part string format: invalid loop index in part string '{partString}'", nameof(partString));

                loopIndex = theLoopIndex;
            }
            else if (subparts.Length != 2)
                throw new ArgumentException($"Invalid part string format {partString}");


            return new ModelParamIdentifierTemplatePart(type, name, deduplicationIndex, loopIndex);
        }

        public override string ToString()
        {
            string retval;
            if (this.Type == ModelParamIdentifierTemplatePartType.Loop)
                return retval = $"Loop#{this.DeduplicationIndex}:{this.LoopIndex}";

            var escapedName = EscapePartName(this.Name);

            retval = $"{escapedName}#{this.DeduplicationIndex}";

            Debug.Assert(Parse(
                    retval, 
                    isFirstPart: this.Type == ModelParamIdentifierTemplatePartType.Category,
                    isLastPart: this.Type == ModelParamIdentifierTemplatePartType.Param).Equals(this));
            return retval;
        }

        public override bool Equals(object? obj)
        {
            if (obj is not ModelParamIdentifierTemplatePart part)
                return false;

            return Equals(part);
        }

        public bool Equals(ModelParamIdentifierTemplatePart? part)
        {
            if (part == null)
                return false;

            return
                part.Name == this.Name &&
                part.DeduplicationIndex == this.DeduplicationIndex &&
                part.LoopIndex == this.LoopIndex &&
                part.Type == this.Type;
        }

        public override int GetHashCode()
            => HashCode.Combine(this.Name, this.DeduplicationIndex, this.LoopIndex, this.Type);

        public static bool operator ==(ModelParamIdentifierTemplatePart? left, ModelParamIdentifierTemplatePart? right)
            => (left is null && right is null) || (left is not null && left.Equals(right));

        public static bool operator !=(ModelParamIdentifierTemplatePart? left, ModelParamIdentifierTemplatePart? right)
            => !(left == right);
    }

    public class ModelParamIdentifierTemplate : IEquatable<ModelParamIdentifierTemplate>
    {
        /// <summary>
        /// The generalized model id for this ModelParamIdentifierTemplate.
        /// </summary>
        public ModelId ModelIdTemplate { get; private set; }
        public bool IsModule => Parts[0] == ModelParamIdentifierTemplatePart.ModuleCategory;
        public string Category => Parts[0].Name; 
        public ImmutableArray<ModelParamIdentifierTemplatePart> Parts { get; private set; }

        /// <summary>
        /// The ModelParamIdentifierTemplate is specific if it has no unspecified loop iterations.
        /// 
        /// Note that a ModelParamIdentifierTemplate that is neither specific nor generalized will have some but not all of its loop
        /// iterations specified.
        /// 
        /// Also note that both IsGenerilized and IsSpecific are true when there are no loop iteration indices in this ModelParamIdentifierTemplate.
        /// </summary>
        public bool IsSpecific => Parts.Where(x => x.Type == ModelParamIdentifierTemplatePartType.Loop).All(x => x.LoopIndex != -1);

        /// <summary>
        /// The ModelParamIdentifierTemplate is specific if all loop iterations are unspecified (set to -1).
        /// 
        /// Note that a ModelParamIdentifierTemplate that is neither specific nor generalized will have some but not all of its loop
        /// iterations specified.
        /// 
        /// Also note that both IsGenerilized and IsSpecific are true when there are no loop iteration indices in this ModelParamIdentifierTemplate.
        /// </summary>
        public bool IsGeneralized => Parts.Where(x => x.Type == ModelParamIdentifierTemplatePartType.Loop).All(x => x.LoopIndex == -1);

        /// <summary>
        /// The specific model id for this ModelParamIdentifierTemplate as specified in the template.
        /// Note that SpecificModelId will still contain unspecified iteration indices if IsSpecific is false.
        /// 
        /// Also note that SpecificModelId is equal to TemplateModelId when IsGeneralized is true.
        /// </summary>
        public ModelId SpecificModelId => ModelIdTemplate.ApplyIterationIndices(LoopIndices.ToArray());

        internal ImmutableArray<int> LoopIndices => Parts.Where(x => x.Type == ModelParamIdentifierTemplatePartType.Loop).Select(x => x.LoopIndex.AssertNotNull()).ToImmutableArray();

        public ModelParamIdentifierTemplate(ModelParamIdentifierTemplate basePath, ModelParamIdentifierTemplate subModulePath)
        {
            this.ModelIdTemplate = new ModelId(basePath.ModelIdTemplate, subModulePath.ModelIdTemplate);
            this.Parts = [subModulePath.Parts[0], ..basePath.Parts[1..], ..subModulePath.Parts[1..]];
            Debug.Assert(this.Category != "TrainableParam" || this.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Param);
            Debug.Assert(this.Category != "Module" || this.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Module);
            Debug.Assert(this.Parts.Length == 2 || this.Parts[1..^1].All(x => x.Type == ModelParamIdentifierTemplatePartType.Module || x.Type == ModelParamIdentifierTemplatePartType.Loop));
            Debug.Assert(ModelIdTemplate.NumIterationIds == this.Parts.Count(x => x.Type == ModelParamIdentifierTemplatePartType.Loop));
        }
        private ModelParamIdentifierTemplate(ModelId modelId, ImmutableArray<ModelParamIdentifierTemplatePart> parts)
        {
            ModelIdTemplate = modelId;
            Parts = parts;
            Debug.Assert(this.Category != "TrainableParam" || this.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Param);
            Debug.Assert(this.Category != "Module" || this.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Module);
            Debug.Assert(this.Parts.Length == 2 || this.Parts[1..^1].All(x => x.Type == ModelParamIdentifierTemplatePartType.Module || x.Type == ModelParamIdentifierTemplatePartType.Loop));
            Debug.Assert(ModelIdTemplate.NumIterationIds == this.Parts.Count(x => x.Type == ModelParamIdentifierTemplatePartType.Loop));
        }

        public ModelParamIdentifierTemplate(string identifierTemplateString)
        {
            if (identifierTemplateString[0] != '[')
                throw new ArgumentException("Invalid identifier template string format: missing opening '['");

            var closingBracketIndex = identifierTemplateString.IndexOf("]:");
            if (closingBracketIndex == -1)
                throw new ArgumentException("Invalid identifier template string format: missing closing ']'");

            var splitLocation = identifierTemplateString.IndexOf(':');
            if (splitLocation == -1)
                throw new ArgumentException($"Invalid identifier template string format: {identifierTemplateString}");

            var modelIdString = identifierTemplateString[..splitLocation];
            var partsString = identifierTemplateString[(splitLocation+1)..];

            ModelIdTemplate = new ModelId(modelIdString);

            var partStrings = partsString.Split(new char[] { '.' }, StringSplitOptions.RemoveEmptyEntries);

            if (partStrings.Length < 2)
                throw new ArgumentException("Invalid identifier template string format: must have at least one category and module ']'");

            var categoryPart = ModelParamIdentifierTemplatePart.Parse(partStrings[0], isFirstPart: true);
            (var modulePartStrings, var trainableParamPartString) =
            (categoryPart == ModelParamIdentifierTemplatePart.ModuleCategory) ?
                (partStrings[1..], 
                 null) :
                
                (partStrings.Length > 2 ? partStrings[1..^1] : [],
                 partStrings[^1]);

            var parts = modulePartStrings.Select(x => ModelParamIdentifierTemplatePart.Parse(x)).ToList();
            parts.Insert(0, categoryPart);

            if (trainableParamPartString != null)
                parts.Add(ModelParamIdentifierTemplatePart.Parse(trainableParamPartString, isLastPart: true));

            this.Parts = parts.ToImmutableArray();
            Debug.Assert(this.Category != "TrainableParam" || this.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Param);
            Debug.Assert(this.Category != "Module" || this.Parts[^1].Type == ModelParamIdentifierTemplatePartType.Module);
            Debug.Assert(this.Parts.Length == 2 || this.Parts[1..^1].All(x => x.Type == ModelParamIdentifierTemplatePartType.Module || x.Type == ModelParamIdentifierTemplatePartType.Loop));
            Debug.Assert(ModelIdTemplate.NumIterationIds == this.Parts.Count(x => x.Type == ModelParamIdentifierTemplatePartType.Loop));
        }


        private static ModelParamIdentifierTemplate LocalTemplate(ModelParamIdentifierTemplatePart categoryPart, ModelId modelId, string moduleName, int deduplicationId, ImmutableArray<int> loopDedupeIds)
        {
            var loopParts = new List<ModelParamIdentifierTemplatePart>();
            foreach (var loopDedupeId in loopDedupeIds)
                loopParts.Add(new(ModelParamIdentifierTemplatePartType.Loop, "Loop", loopDedupeId, -1));

            var partType = categoryPart == ModelParamIdentifierTemplatePart.ModuleCategory ?
                                 ModelParamIdentifierTemplatePartType.Module :
                                 ModelParamIdentifierTemplatePartType.Param;

            return new ModelParamIdentifierTemplate(modelId,
                [categoryPart, ..loopParts,
                 new (partType, moduleName, deduplicationId)]);
        }

        public static ModelParamIdentifierTemplate LocalModule(ModelId modelId, string moduleName, int deduplicationId, ImmutableArray<int> loopDedupeIds)
            => LocalTemplate(new(ModelParamIdentifierTemplatePartType.Category, "Module"), modelId, moduleName, deduplicationId, loopDedupeIds);

        public static ModelParamIdentifierTemplate LocalTrainableParam(ModelId modelId, string moduleName, int deduplicationId, ImmutableArray<int> loopDedupeIds)
            => LocalTemplate(new(ModelParamIdentifierTemplatePartType.Category, "TrainableParam"), modelId, moduleName, deduplicationId, loopDedupeIds);

        public static ModelParamIdentifierTemplate LocalStateParam(ModelId modelId, string moduleName, int deduplicationId, ImmutableArray<int> loopDedupeIds)
            => LocalTemplate(new(ModelParamIdentifierTemplatePartType.Category, "StateParam"), modelId, moduleName, deduplicationId, loopDedupeIds);

        public ModelParamIdentifierTemplate ToGeneralizedTemplate()
        {
            if (this.IsGeneralized)
                return this;

            var newParts = new List<ModelParamIdentifierTemplatePart>();
            for (int i = 0; i < this.Parts.Length; i++)
            {
                var part = this.Parts[i];
                if (part.Type != ModelParamIdentifierTemplatePartType.Loop)
                {
                    newParts.Add(part);
                    continue;
                }

                newParts.Add(new ModelParamIdentifierTemplatePart(part.Type, part.Name, part.DeduplicationIndex, loopIndex: -1));
            }

            return new ModelParamIdentifierTemplate(this.ModelIdTemplate, newParts.ToImmutableArray());
        }

        public ModelParamIdentifierTemplate ToSpecificTemplate(ModelId modelId)
        {
            if (this.IsSpecific)
                return this;

            var loopIndices = modelId.IterationIndexValues(this.ModelIdTemplate.IterationIdLocations);
            var loopIndexIndex = 0;
            var newParts = new List<ModelParamIdentifierTemplatePart>();
            for (int i = 0; i < this.Parts.Length; i++)
            {
                var part = this.Parts[i];
                if (part.Type != ModelParamIdentifierTemplatePartType.Loop)
                {
                    newParts.Add(part);
                    continue;
                }

                newParts.Add(new ModelParamIdentifierTemplatePart(part.Type, part.Name, part.DeduplicationIndex, loopIndices[loopIndexIndex]));
                loopIndexIndex++;
            }

            return new ModelParamIdentifierTemplate(this.ModelIdTemplate, newParts.ToImmutableArray());
        }

        public override string ToString()
        {
            var retval = $"[{ModelIdTemplate.ToString()}]:{this.ToTemplateString()}";
            Debug.Assert(new ModelParamIdentifierTemplate(retval).Equals(this));
            return retval;
        }

        public string ToTemplateString()
        {
            return string.Join(".", Parts.Select(x => x.ToString()));
        }

        public override bool Equals(object? obj)
        {
            if (obj is not ModelParamIdentifierTemplate template)
                return false;

            return this.Equals(template);
        }

        public override int GetHashCode()
        {
            var hashcode = HashCode.Combine(this.ModelIdTemplate);
            foreach (var part in this.Parts)
                hashcode = HashCode.Combine(hashcode, part.GetHashCode());

            return hashcode;
        }

        public bool Equals(ModelParamIdentifierTemplate? template)
        {
            if (template is null)
                return false;
            return
                this.ModelIdTemplate == template.ModelIdTemplate &&
                this.Parts.SequenceEqual(template.Parts);
        }

        public static bool operator ==(ModelParamIdentifierTemplate? left, ModelParamIdentifierTemplate? right)
            => (left is null && right is null) || (left is not null && left.Equals(right));

        public static bool operator !=(ModelParamIdentifierTemplate? left, ModelParamIdentifierTemplate? right)
            => !(left == right);
    }
}
