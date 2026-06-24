using ProtoBuf;
using Shorokoo;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using static Shorokoo.Core.ModuleParamSetNamingScheme;

namespace Shorokoo.Core
{
    public class ConcreteModelParamInfo
    {
        public DType DType { get; private set; }
        public Shape Shape { get; private set; }
        public ModelId ModelId { get; private set; }
        public ModelParamIdentifierTemplate ParamIdentifier { get; private set; }

        public ConcreteModelParamInfo(Node trainableParamNode)
        {
            this.DType = trainableParamNode.Outputs[0]!.DType;
            this.Shape = new Shape(trainableParamNode.Attributes.GetLongsVal(OnnxOpAttributeNames.ShrkAttrShape).AssertNotNull());
            this.ModelId = new ModelId(trainableParamNode.Attributes.GetIntsVal(OnnxOpAttributeNames.ShrkAttrLocalModelId).AssertNotNull());
            this.ParamIdentifier = trainableParamNode.IdentifierTemplate.AssertNotNull();
        }

        public override string ToString()
        {
            return $"({Shape}|{ModelId}|{ParamIdentifier}";
        }

        public string ToShorokooIdString() =>
            this.ParamIdentifier.ToSpecificTemplate(this.ModelId).ToTemplateString();
    }

    public class ConcreteModelParamInfos
    {
        public ConcreteModelParamInfos(ImmutableArray<ConcreteModelParamInfo> paramNames)
        {
            this.ParamInfos = paramNames.OrderBy(x => x.ModelId).ToImmutableArray();
        }

        public ImmutableArray<ConcreteModelParamInfo> ParamInfos { get; private set; }
        public ImmutableArray<ModelId> ModelIds => this.ParamInfos.Select(x => x.ModelId).ToImmutableArray();
    }

    /// <summary>
    /// Represents a format specification for converting ModelIds to third-party param names.
    /// Uses the DSL described in Documentation/param-naming-format-dsl.md.
    /// </summary>
    public class ModelIdFormat
    {
        /// <summary>
        /// Format string with placeholders like {0}, {1 + 2}, {2|conv,bn,fc}.
        /// </summary>
        public string Format { get; }
        
        /// <summary>
        /// Optional match pattern like "[1, 2, *, *]" or "[1, 3|4|5, 1, *, *, *, *]".
        /// If null, matches all ModelIds.
        /// </summary>
        public string? Match { get; }
        
        /// <summary>
        /// Named maps for value lookups.
        /// </summary>
        public ImmutableDictionary<string, ImmutableDictionary<int, string>> Maps { get; }

        public ModelIdFormat(string format, string? match = null, Dictionary<string, Dictionary<int, string>>? maps = null)
        {
            Format = format;
            Match = match;
            Maps = maps?.ToImmutableDictionary(
                kv => kv.Key, 
                kv => kv.Value.ToImmutableDictionary()
            ) ?? ImmutableDictionary<string, ImmutableDictionary<int, string>>.Empty;
        }

        /// <summary>
        /// Checks if a ModelId matches this format's match pattern.
        /// </summary>
        public bool Matches(ModelId modelId)
        {
            if (Match == null || Match == "*")
                return true;

            var pattern = Match.Trim();
            if (!pattern.StartsWith("[") || !pattern.EndsWith("]"))
                return false;

            var patternParts = pattern[1..^1].Split(',').Select(p => p.Trim()).ToArray();
            var vals = modelId.Vals;

            if (patternParts.Length != vals.Length)
                return false;

            for (int i = 0; i < patternParts.Length; i++)
            {
                var part = patternParts[i];
                if (part == "*" || part == "-1")
                    continue;

                // Handle OR patterns like "3|4|5"
                if (part.Contains('|'))
                {
                    var options = part.Split('|').Select(o => int.Parse(o.Trim())).ToArray();
                    if (!options.Contains(vals[i]))
                        return false;
                }
                else
                {
                    if (int.Parse(part) != vals[i])
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Converts a ModelId to a param name using this format.
        /// </summary>
        public string ToName(ModelId modelId)
        {
            return EvaluateFormat(Format, modelId.Vals.ToArray());
        }

        public static string UnescapeString(string input)
        {
            return input.Replace("\\o", "{")
                    .Replace("\\c", "}")
                    .Replace("\\s", "\\");
        }

        public static string EscapeString(string input)
        {
            return input.Replace("\\", "\\s")
                        .Replace("}", "\\c")
                        .Replace("{", "\\o");
        }

        private string EvaluateFormat(string format, int[] vals)
        {
            var result = new System.Text.StringBuilder();
            int i = 0;

            while (i < format.Length)
            {
                var nextOpenIdx = format.IndexOf('{', i);
                if (nextOpenIdx == -1)
                    nextOpenIdx = format.Length;

                if (nextOpenIdx > i)
                {
                    // Append literal text before the next placeholder
                    result.Append(UnescapeString(format[i..nextOpenIdx]));
                    i = nextOpenIdx;
                }
                else 
                {
                    Debug.Assert(nextOpenIdx == i);
                    var closeIdx = FindMatchingBrace(format, i);
                    if (closeIdx == -1)
                        throw new FormatException($"Unmatched '{{' in format string at position {i}");

                    var placeholder = format[(i + 1)..closeIdx];
                    var evaluated = EvaluatePlaceholder(placeholder, vals);
                    result.Append(evaluated);
                    i = closeIdx + 1;
                }
            }

            return result.ToString();
        }

        private int FindMatchingBrace(string format, int openIdx)
        {
            int depth = 1;
            for (int i = openIdx + 1; i < format.Length; i++)
            {
                if (format[i] == '{') depth++;
                else if (format[i] == '}')
                {
                    depth--;
                    if (depth == 0) return i;
                }
            }
            return -1;
        }

        private string EvaluatePlaceholder(string placeholder, int[] vals)
        {
            // Handle simple index: {N} or {N + M} or {N - M}
            // Handle inline maps: {N|val0,val1,val2}
            // Handle named maps: {N|mapName}
            // Handle range maps: {N|ranges|outputs}

            var pipeIdx = placeholder.IndexOf('|');
            if (pipeIdx == -1)
            {
                // Simple index with optional offset
                var value = EvaluateIndexExpression(placeholder.Trim(), vals);
                return value.ToString();
            }

            var indexPart = placeholder[..pipeIdx].Trim();
            var mapPart = placeholder[(pipeIdx + 1)..];
            var value2 = EvaluateIndexExpression(indexPart, vals);

            // Check if this is a range map (contains second |)
            var secondPipeIdx = mapPart.IndexOf('|');
            if (secondPipeIdx != -1)
            {
                // Range map: {N|ranges|outputs}
                var rangesPart = mapPart[..secondPipeIdx];
                var outputsPart = mapPart[(secondPipeIdx + 1)..];
                return EvaluateRangeMap(value2, rangesPart, outputsPart, vals);
            }

            // Check if this is a named map
            if (Maps.TryGetValue(mapPart.Trim(), out var namedMap))
            {
                if (!namedMap.TryGetValue(value2, out var mappedValue))
                    throw new KeyNotFoundException($"Key {value2} not found in map '{mapPart.Trim()}'");
                return mappedValue;
            }

            // Inline map: {N|val0,val1,val2}
            var mapValues = mapPart.Split(',').Select(v => v.Trim()).ToArray();
            if (value2 < 0 || value2 >= mapValues.Length)
                throw new IndexOutOfRangeException($"Index {value2} is out of range for inline map with {mapValues.Length} values");

            var mappedResult = mapValues[value2];
            
            // Check for recursive format strings (contains {)
            if (mappedResult.Contains('{'))
            {
                return EvaluateFormat(mappedResult, vals);
            }

            return mappedResult;
        }

        private int EvaluateIndexExpression(string expr, int[] vals)
        {
            expr = expr.Trim();

            // Handle addition: N + M
            var plusIdx = expr.IndexOf('+');
            if (plusIdx != -1)
            {
                var left = expr[..plusIdx].Trim();
                var right = expr[(plusIdx + 1)..].Trim();
                var idx = int.Parse(left);
                var offset = int.Parse(right);
                return vals[idx] + offset;
            }

            // Handle subtraction: N - M
            var minusIdx = expr.IndexOf('-');
            if (minusIdx != -1 && minusIdx > 0)
            {
                var left = expr[..minusIdx].Trim();
                var right = expr[(minusIdx + 1)..].Trim();
                var idx = int.Parse(left);
                var offset = int.Parse(right);
                return vals[idx] - offset;
            }

            // Simple index
            var index = int.Parse(expr);
            return vals[index];
        }
        private static string[] SplitTopLevel(string input)
        {
            var result = new List<string>();
            int depth = 0;
            int start = 0;

            for (int i = 0; i < input.Length; i++)
            {
                char c = input[i];

                if (c == '{')
                    depth++;
                else if (c == '}')
                    depth--;
                else if (c == ',' && depth == 0)
                {
                    // top-level comma → split here
                    result.Add(input.Substring(start, i - start).Trim());
                    start = i + 1;
                }
            }

            // last segment
            result.Add(input.Substring(start).Trim());

            return result.ToArray();
        }

        private string EvaluateRangeMap(int value, string rangesPart, string outputsPart, int[] vals)
        {
            var ranges = rangesPart.Split(',');
            var outputs = SplitTopLevel(outputsPart);

            if (ranges.Length != outputs.Length)
                throw new FormatException($"Range count ({ranges.Length}) must match output count ({outputs.Length})");

            for (int i = 0; i < ranges.Length; i++)
            {
                var range = ranges[i].Trim();
                if (MatchesRange(value, range))
                {
                    var output = outputs[i];
                    // Check for recursive format strings
                    if (output.Contains('{'))
                    {
                        return EvaluateFormat(output, vals);
                    }
                    return output;
                }
            }

            throw new KeyNotFoundException($"Value {value} does not match any range in '{rangesPart}'");
        }

        private bool MatchesRange(int value, string range)
        {
            // Handle single value: "1"
            if (!range.Contains(':'))
            {
                return int.Parse(range.Trim()) == value;
            }

            // Handle range with step: "start::step" or "start:end" or "start:end:step"
            var parts = range.Split(':');
            if (parts.Length == 2 && string.IsNullOrEmpty(parts[1]))
            {
                // "start:" means start and above
                var start = int.Parse(parts[0].Trim());
                return value >= start;
            }
            else if (parts.Length == 2)
            {
                // "start:end" means start to end (inclusive)
                var start = int.Parse(parts[0].Trim());
                var end = int.Parse(parts[1].Trim());
                return value >= start && value <= end;
            }
            else if (parts.Length == 3)
            {
                var start = string.IsNullOrEmpty(parts[0]) ? 0 : int.Parse(parts[0].Trim());
                var end = string.IsNullOrEmpty(parts[1]) ? int.MaxValue : int.Parse(parts[1].Trim());
                var step = int.Parse(parts[2].Trim());

                // Check if value is in the arithmetic sequence
                if (value < start || value > end) return false;
                return (value - start) % step == 0;
            }

            return false;
        }
    }

    public class SimplePatternNamingScheme : ModuleParamSetNamingScheme
    {
        public ImmutableArray<SimplePatternScheme> Patterns { get; }

        public ModelIdNamingScheme ModelIdToShorokooIdScheme { get; }

        private Dictionary<string, ModelId>? dctReverseCache = null;

        public SimplePatternNamingScheme(IEnumerable<SimplePatternScheme> patterns, ModelIdNamingScheme modelIdToShorokooIdScheme, string frameworkId) : base(frameworkId)
        {
            this.Patterns = patterns.ToImmutableArray();
            this.ModelIdToShorokooIdScheme = modelIdToShorokooIdScheme;
        }

        private void buildReverseCache(ImmutableArray<ModelId> candidates)
        {
            // WARNING: if there is an issue here, the root cause is most certainly somewhere 
            // deep in the core of Shorokoo implementation from how the candidates were generated.
            this.dctReverseCache = candidates.ToDictionary(x => this.ToName(x)!, x => x);
        }

        private string? toName(string shorokooId)
        {
            foreach (var pattern in Patterns)
            {
                if (pattern.Matches(shorokooId))
                    return pattern.ToName(shorokooId);
            }

            return null;
        }

        public override string? ToName(ModelId modelId)
        {
            var shorokooId = this.ModelIdToShorokooIdScheme.ToName(modelId);
            return toName(shorokooId);
        }

        public override string? ToName(ConcreteModelParamInfo shorokooParam)
        {
            return this.toName(shorokooParam.ToShorokooIdString());
        }

        public override ModelId? ToModelId(string paramName, ImmutableArray<ModelId> candidates)
        {
            if (dctReverseCache is not null && dctReverseCache.TryGetValue(paramName, out var modelId))
                return modelId;

            buildReverseCache(candidates);

            if (dctReverseCache.AssertNotNull().TryGetValue(paramName, out modelId))
                return modelId;

            return null;
        }
    }

    public class ModelIdNamingScheme : ModuleParamSetNamingScheme
    {
        public ImmutableArray<ModelIdFormat> Patterns { get; }

        private Dictionary<string, ModelId>? dctReverseCache = null;

        public ModelIdNamingScheme(IEnumerable<ModelIdFormat> patterns, string frameworkId) : base(frameworkId)
        {
            this.Patterns = patterns.ToImmutableArray();
        }

        private void buildReverseCache(ImmutableArray<ModelId> candidates)
        {
            this.dctReverseCache = candidates.ToDictionary(x => this.ToName(x), x => x);
        }

        public override string ToName(ModelId modelId)
        {
            foreach (var pattern in Patterns)
            {
                if (pattern.Matches(modelId))
                    return pattern.ToName(modelId);
            }

            throw new InvalidOperationException($"No matching pattern for ModelId [{string.Join(",", modelId.Vals)}]");
        }

        public override string ToName(ConcreteModelParamInfo shorokooParam) 
            => ToName(shorokooParam.ModelId);

        public override ModelId? ToModelId(string paramName, ImmutableArray<ModelId> candidates)
        {
            if (dctReverseCache is not null && dctReverseCache.TryGetValue(paramName, out var modelId))
                return modelId;

            buildReverseCache(candidates);

            if (dctReverseCache.AssertNotNull().TryGetValue(paramName, out modelId))
                return modelId;

            // Return null for unmatched param names - the caller filters these out
            return null;
        }
    }

    /// <summary>
    /// Naming scheme for a module targeting a specific framework, e.g., PyTorch, TensorFlow, etc.
    /// </summary>
    public abstract class ModuleParamSetNamingScheme
    {
        public const string ShorokooFrameworkId = "Shorokoo.Auto.Default";
        public const string PyTorchFrameworkId = "Shorokoo.PyTorch";
        public const string TensorFlowFrameworkId = "Shorokoo.TensorFlow";
        public const string FlaxFrameworkId = "Shorokoo.Flax";

        /// <summary>
        /// The framework affinity for this naming scheme.
        /// E.g. if set to PyTorchFrameworkId, this naming scheme is intended to
        /// match model parameter names created and understood by PyTorch models.
        /// </summary>
        public string FrameworkId { get; private set; }

        public ModuleParamSetNamingScheme(string frameworkId)
        {
            FrameworkId = frameworkId;
        }

        public abstract string? ToName(ModelId modelId);

        public abstract string? ToName(ConcreteModelParamInfo shorokooParam);

        public abstract ModelId? ToModelId(string paramName, ImmutableArray<ModelId> candidatess);

        /// <summary>
        /// Creates a naming scheme from a ModelIdNamingScheme using the ModelId to Name DSL.
        /// </summary>
        public static ModuleParamSetNamingScheme FromModelIdFormats(ModelIdNamingScheme modelIdScheme, string frameworkId)
        {
            return modelIdScheme;
        }

        public static ModelIdNamingScheme CreateShorokooNamingScheme(ConcreteModelParamInfos concreteModelIdentifierTemplates)
        {
            var paramNamingSchemes = new List<ModelIdFormat>();

            foreach (var paramInfo in concreteModelIdentifierTemplates.ParamInfos.DistinctBy(x => x.ModelId))
            {
                var param = paramInfo.ParamIdentifier;
                var stringParts = new List<string>();
                var modelIdIndex = 0;
                var templateModelId = param.SpecificModelId;
                foreach (var part in param.Parts)
                {
                    if (part.Type == ModelParamIdentifierTemplatePartType.Category)
                    {
                        // The category part does not have an associated ModelId value.
                        stringParts.Add(ModelIdFormat.EscapeString(part.ToString()));
                    }
                    else if (part.Type == ModelParamIdentifierTemplatePartType.Loop)
                    {
                        // The loop part has 2 associated ModelId values.
                        Debug.Assert(templateModelId.Vals[modelIdIndex] != -1);
                        modelIdIndex++;

                        if (templateModelId.Vals[modelIdIndex] == -1)
                            stringParts.Add($"Loop#{part.DeduplicationIndex}:{{{modelIdIndex}}}");
                        else
                            stringParts.Add(ModelIdFormat.EscapeString(part.ToString()));

                        modelIdIndex++;
                    }
                    else
                    {
                        Debug.Assert(templateModelId.Vals[modelIdIndex] != -1);
                        stringParts.Add(ModelIdFormat.EscapeString(part.ToString()));
                        modelIdIndex++;
                    }
                }

                var formatTemplate = String.Join(".", stringParts);
                // Use the actual ModelId from the ConcreteModelParamInfo for pattern matching,
                // as the identifier template's ModelId may be shorter when the template wasn't
                // fully composed (e.g., for model sequence access paths).
                var actualModelId = paramInfo.ModelId;
                var matchTemplate = $"[{string.Join(",", actualModelId.Vals.Select(x => x == -1 ? "*" : x.ToString()))}]";
                paramNamingSchemes.Add(new ModelIdFormat(
                    match: matchTemplate,
                    format: formatTemplate));
            }

            return new ModelIdNamingScheme(paramNamingSchemes.ToImmutableArray(), ShorokooFrameworkId);
        }

        /// <summary>
        /// Creates a naming scheme from a SimplePatternNamingScheme using the Simple Pattern Matching DSL.
        /// </summary>
        public static ModuleParamSetNamingScheme FromSimplePatterns(SimplePatternNamingScheme simplePatternScheme, string frameworkId)
        {
            return simplePatternScheme;
        }
    }

    #region Simple Pattern Matching DSL

    /// <summary>
    /// Semantic element type for Shorokoo ID parsing.
    /// </summary>
    public enum SemanticElementType
    {
        Word,   // Contiguous letters
        Number, // Contiguous digits (as a unit)
        Dot,    // The '.' character
        Hash,   // The '#' character
        Colon   // The ':' character
    }

    /// <summary>
    /// A single semantic element from a Shorokoo ID.
    /// </summary>
    public record SemanticElement(string Value, SemanticElementType Type);

    /// <summary>
    /// A pattern for matching Shorokoo IDs and converting them to third-party param names.
    /// Uses the DSL described in Documentation/param-naming-pattern-dsl.md.
    /// </summary>
    public class SimplePatternScheme
    {
        /// <summary>
        /// The pattern to match against Shorokoo IDs.
        /// </summary>
        public string Pattern { get; }

        /// <summary>
        /// The format string for generating param names.
        /// </summary>
        public string Format { get; }

        /// <summary>
        /// Named maps for value lookups.
        /// </summary>
        public ImmutableDictionary<string, ImmutableDictionary<string, string>> Maps { get; }

        public SimplePatternScheme(string pattern, string format, Dictionary<string, Dictionary<string, string>>? maps = null)
        {
            Pattern = pattern;
            Format = format;
            Maps = maps?.ToImmutableDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.ToImmutableDictionary()
            ) ?? ImmutableDictionary<string, ImmutableDictionary<string, string>>.Empty;
        }

        /// <summary>
        /// Parses a Shorokoo ID into semantic elements.
        /// </summary>
        public static List<SemanticElement> ParseSemanticElements(string input)
        {
            var elements = new List<SemanticElement>();
            int i = 0;

            while (i < input.Length)
            {
                var c = input[i];

                if (c == '.')
                {
                    elements.Add(new SemanticElement(".", SemanticElementType.Dot));
                    i++;
                }
                else if (c == '#')
                {
                    elements.Add(new SemanticElement("#", SemanticElementType.Hash));
                    i++;
                }
                else if (c == ':')
                {
                    elements.Add(new SemanticElement(":", SemanticElementType.Colon));
                    i++;
                }
                else if (char.IsDigit(c))
                {
                    // Capture entire number (all contiguous digits)
                    int start = i;
                    while (i < input.Length && char.IsDigit(input[i]))
                        i++;
                    elements.Add(new SemanticElement(input[start..i], SemanticElementType.Number));
                }
                else if (char.IsLetter(c))
                {
                    // Capture word (all contiguous letters)
                    int start = i;
                    while (i < input.Length && char.IsLetter(input[i]))
                        i++;
                    elements.Add(new SemanticElement(input[start..i], SemanticElementType.Word));
                }
                else
                {
                    // Unknown character, skip (or treat as individual element)
                    i++;
                }
            }

            return elements;
        }

        /// <summary>
        /// Attempts to match the pattern against a Shorokoo ID and extract captures.
        /// </summary>
        public bool TryMatch(string shorokooId, out Dictionary<string, string> captures)
        {
            captures = new Dictionary<string, string>();
            var inputElements = ParseSemanticElements(shorokooId);
            var patternTokens = ParsePattern(Pattern);

            return MatchTokens(patternTokens, inputElements, 0, 0, captures);
        }

        /// <summary>
        /// Checks if this pattern matches the given Shorokoo ID.
        /// </summary>
        public bool Matches(string shorokooId)
        {
            return TryMatch(shorokooId, out _);
        }

        /// <summary>
        /// Converts a Shorokoo ID to a third-party param name using this pattern.
        /// </summary>
        public string ToName(string shorokooId)
        {
            if (!TryMatch(shorokooId, out var captures))
                throw new InvalidOperationException($"Shorokoo ID '{shorokooId}' does not match pattern '{Pattern}'");

            return EvaluateFormat(Format, captures);
        }

        private List<PatternToken> ParsePattern(string pattern)
        {
            var tokens = new List<PatternToken>();
            int i = 0;
            var literalBuilder = new System.Text.StringBuilder();

            void FlushLiteral()
            {
                if (literalBuilder.Length > 0)
                {
                    tokens.Add(new PatternToken(PatternTokenType.Literal, literalBuilder.ToString()));
                    literalBuilder.Clear();
                }
            }

            while (i < pattern.Length)
            {
                // Handle escape sequences
                if (pattern[i] == '\\' && i + 1 < pattern.Length)
                {
                    var next = pattern[i + 1];
                    if (next == 'o')
                    {
                        literalBuilder.Append('{');
                        i += 2;
                        continue;
                    }
                    else if (next == 's')
                    {
                        literalBuilder.Append('\\');
                        i += 2;
                        continue;
                    }
                }

                // Handle placeholders/captures
                if (pattern[i] == '{')
                {
                    FlushLiteral();
                    
                    var closeIdx = pattern.IndexOf('}', i);
                    if (closeIdx == -1)
                        throw new FormatException($"Unmatched '{{' in pattern at position {i}");

                    var content = pattern[(i + 1)..closeIdx];
                    tokens.Add(ParsePlaceholder(content));
                    i = closeIdx + 1;
                }
                else
                {
                    // Literal character - accumulate into builder
                    literalBuilder.Append(pattern[i]);
                    i++;
                }
            }

            FlushLiteral();

            return tokens;
        }

        private PatternToken ParsePlaceholder(string content)
        {
            if (content == "*")
                return new PatternToken(PatternTokenType.Wildcard, null);

            // Parse capture: {name}, {name:n}, {name|range}
            var pipeIdx = content.IndexOf('|');
            var colonIdx = content.IndexOf(':');

            string name;
            int count = 1;
            RangeConstraint? range = null;

            if (pipeIdx != -1)
            {
                // Has range constraint
                name = content[..pipeIdx].Trim();
                var rangeStr = content[(pipeIdx + 1)..].Trim();
                range = ParseRangeConstraint(rangeStr);
            }
            else if (colonIdx != -1)
            {
                // Has element count
                name = content[..colonIdx].Trim();
                count = int.Parse(content[(colonIdx + 1)..].Trim());
            }
            else
            {
                name = content.Trim();
            }

            return new PatternToken(PatternTokenType.Capture, name, count, range);
        }

        private RangeConstraint ParseRangeConstraint(string rangeStr)
        {
            // Handle: "1:3", "2:", ":5", "1::2", "0::2"
            var parts = rangeStr.Split(':');

            if (parts.Length == 2 && !string.IsNullOrEmpty(parts[0]) && string.IsNullOrEmpty(parts[1]))
            {
                // "start:" - open-ended
                var start = int.Parse(parts[0]);
                return new RangeConstraint(start, int.MaxValue, 1);
            }
            else if (parts.Length == 2 && string.IsNullOrEmpty(parts[0]) && !string.IsNullOrEmpty(parts[1]))
            {
                // ":end" - from 0
                var end = int.Parse(parts[1]);
                return new RangeConstraint(0, end, 1);
            }
            else if (parts.Length == 2)
            {
                // "start:end"
                var start = int.Parse(parts[0]);
                var end = int.Parse(parts[1]);
                return new RangeConstraint(start, end, 1);
            }
            else if (parts.Length == 3)
            {
                // "start:end:step" or "start::step"
                var start = string.IsNullOrEmpty(parts[0]) ? 0 : int.Parse(parts[0]);
                var end = string.IsNullOrEmpty(parts[1]) ? int.MaxValue : int.Parse(parts[1]);
                var step = int.Parse(parts[2]);
                return new RangeConstraint(start, end, step);
            }

            throw new FormatException($"Invalid range constraint: {rangeStr}");
        }

        private bool MatchTokens(List<PatternToken> tokens, List<SemanticElement> elements, int tokenIdx, int elemIdx, Dictionary<string, string> captures)
        {
            while (tokenIdx < tokens.Count)
            {
                var token = tokens[tokenIdx];

                switch (token.Type)
                {
                    case PatternTokenType.Literal:
                        // Match literal text against elements
                        var literalElements = ParseSemanticElements(token.Value!);
                        for (int i = 0; i < literalElements.Count; i++)
                        {
                            if (elemIdx >= elements.Count)
                                return false;
                            if (elements[elemIdx].Value != literalElements[i].Value)
                                return false;
                            elemIdx++;
                        }
                        tokenIdx++;
                        break;

                    case PatternTokenType.Wildcard:
                        // Greedy match - try matching rest of pattern with remaining elements
                        // Try progressively more elements for wildcard
                        for (int wildEnd = elemIdx; wildEnd <= elements.Count; wildEnd++)
                        {
                            var capturesCopy = new Dictionary<string, string>(captures);
                            if (MatchTokens(tokens, elements, tokenIdx + 1, wildEnd, capturesCopy))
                            {
                                foreach (var kv in capturesCopy)
                                    captures[kv.Key] = kv.Value;
                                return true;
                            }
                        }
                        return false;

                    case PatternTokenType.Capture:
                        // Capture n semantic elements
                        var count = token.Count;
                        if (elemIdx + count > elements.Count)
                            return false;

                        var capturedValue = string.Join("", elements.Skip(elemIdx).Take(count).Select(e => e.Value));

                        // Check range constraint if present
                        if (token.Range != null && int.TryParse(capturedValue, out var numValue))
                        {
                            if (!token.Range.Matches(numValue))
                                return false;
                        }

                        captures[token.Value!] = capturedValue;
                        elemIdx += count;
                        tokenIdx++;
                        break;
                }
            }

            // Success if we consumed all elements
            return elemIdx == elements.Count;
        }

        private string EvaluateFormat(string format, Dictionary<string, string> captures)
        {
            var result = new System.Text.StringBuilder();
            int i = 0;

            while (i < format.Length)
            {
                // Handle escape sequences
                if (format[i] == '\\' && i + 1 < format.Length)
                {
                    var next = format[i + 1];
                    if (next == 'o')
                    {
                        result.Append('{');
                        i += 2;
                        continue;
                    }
                    else if (next == 's')
                    {
                        result.Append('\\');
                        i += 2;
                        continue;
                    }
                }

                // Handle placeholders
                if (format[i] == '{')
                {
                    var closeIdx = format.IndexOf('}', i);
                    if (closeIdx == -1)
                        throw new FormatException($"Unmatched '{{' in format string at position {i}");

                    var placeholder = format[(i + 1)..closeIdx];
                    var evaluated = EvaluatePlaceholder(placeholder, captures);
                    result.Append(evaluated);
                    i = closeIdx + 1;
                }
                else
                {
                    result.Append(format[i]);
                    i++;
                }
            }

            return result.ToString();
        }

        private string EvaluatePlaceholder(string placeholder, Dictionary<string, string> captures)
        {
            // Handle: {name}, {name + N}, {name - N}, {name|mapName}, {name|lower}
            var pipeIdx = placeholder.IndexOf('|');
            if (pipeIdx != -1)
            {
                var namePart = placeholder[..pipeIdx].Trim();
                var mapPart = placeholder[(pipeIdx + 1)..].Trim();

                var value = EvaluateNameExpression(namePart, captures);

                // Check for built-in transforms
                if (mapPart == "lower")
                    return value.ToLowerInvariant();

                // Look up in map
                if (Maps.TryGetValue(mapPart, out var map))
                {
                    if (!map.TryGetValue(value, out var mappedValue))
                        throw new KeyNotFoundException($"Key '{value}' not found in map '{mapPart}'");
                    return mappedValue;
                }

                throw new KeyNotFoundException($"Map '{mapPart}' not found");
            }

            // Simple reference with optional offset
            return EvaluateNameExpression(placeholder.Trim(), captures);
        }

        private string EvaluateNameExpression(string expr, Dictionary<string, string> captures)
        {
            expr = expr.Trim();

            // Handle addition: name + N
            var plusIdx = expr.IndexOf('+');
            if (plusIdx != -1)
            {
                var name = expr[..plusIdx].Trim();
                var offset = int.Parse(expr[(plusIdx + 1)..].Trim());
                var value = int.Parse(captures[name]);
                return (value + offset).ToString();
            }

            // Handle subtraction: name - N
            var minusIdx = expr.IndexOf('-');
            if (minusIdx != -1)
            {
                var name = expr[..minusIdx].Trim();
                var offset = int.Parse(expr[(minusIdx + 1)..].Trim());
                var value = int.Parse(captures[name]);
                return (value - offset).ToString();
            }

            // Simple name lookup
            if (!captures.TryGetValue(expr, out var result))
                throw new KeyNotFoundException($"Capture '{expr}' not found");
            return result;
        }

        private enum PatternTokenType
        {
            Literal,
            Wildcard,
            Capture
        }

        private record PatternToken(PatternTokenType Type, string? Value, int Count = 1, RangeConstraint? Range = null);

        private record RangeConstraint(int Start, int End, int Step)
        {
            public bool Matches(int value)
            {
                if (value < Start || value > End)
                    return false;
                if (Step == 1)
                    return true;
                return (value - Start) % Step == 0;
            }
        }
    }

    #endregion

}
