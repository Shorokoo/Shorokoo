using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Microsoft.CodeAnalysis.Operations;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.WebSockets;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;
using System.Xml.Linq;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Graph
{
    public record FrameworkId
    {
        public string Name { get; }

        private FrameworkId(string value) => Name = value;

        public static readonly FrameworkId Unknown = new("Unknown");
        public static readonly FrameworkId Shorokoo = new("Shorokoo.Default");
        public static readonly FrameworkId PyTorch = new("PyTorch.Default");
        public static readonly FrameworkId TensorFlow = new("TensorFlow.Default");
        public static readonly FrameworkId Flax = new("Flax.Default");
    }

    /// <summary>
    /// The following have model ids:
    ///  - Models
    ///  - Loops that contain models or trainable parameters.
    ///  - Loop iterations
    ///  - Trainable Parameters
    /// 
    /// Model Ids ultimately serve to track the identity of trainable parameters.
    /// 
    /// In intermediate processing steps they track the identity of concrete models, loops and loop iterations.
    /// 
    /// For the variables produced from following nodes, ModelIds are relative to the parent model.
    ///   - MODEL_MODEL_REF
    ///   - MODEL_ID_REF
    ///   - MODEL_PARAM_MODEL_REF
    ///   - MODEL_PARAM_ID_REF
    /// 
    /// For all other variables, ModelIds are relative to the function they are in (or global if in the main graph).
    /// 
    /// ModelIds are paths of Id. When a MODEL_INVOKE on a model known at inline time, then the model ids in the called module are 
    /// parented to the model id of the model that was called.
    /// 
    /// Loops have their own ModelId and each iteration of the loop has its own ModelId as well. A variable in a loop has a placeholder -1 to indicate the iteration index.
    /// 
    /// E.g.:
    /// ModuleA uses submodule ModuleB. ModuleB contains a loop that executes n times and each iteration initializes a new trainable param.
    /// 
    /// The global computation graph calls ModuleA twice once with 2 iterations on ModuleB and once with 3 iterations on ModuleB.
    /// 
    /// In the global graph, the two ModuleA models will have ModelId:
    ///   - [1]
    ///   - [2]
    ///   
    /// In ModuleA, ModuleB will have ModelId:
    ///   - [1]
    ///   
    /// In ModuleB, the loop will have ModelId:
    ///   - [1]
    ///   
    /// In ModuleB, the loop iteration will have ModelId:
    ///   - [1, -1]
    ///   
    /// In ModuleB, the trainable parameter will have ModelId:
    ///   - [1, -1, 1]
    ///   
    /// After inlining both ModuleA and ModuleB, because all models can be tracked back to their hardcoded ModelIds we will get:
    /// 
    /// In the global graph:
    /// Two loops with ModelId:
    ///   - [1,1,1] (First ModuleA/ModuleB/Loop)
    ///   - [2,1,1] (Second ModuleA/ModuleB/Loop)
    ///   
    /// Two loop iterations with ModelId:
    ///   - [1,1,1,-1] (First ModuleA/ModuleB/Loop/Iteration)
    ///   - [2,1,1,-1] (Second ModuleA/ModuleB/Loop/Iteration)
    /// 
    /// Two trainable param variables with ModelId:
    ///   - [1,1,1,-1,1] (First ModuleA/ModuleB/Loop/Iteration/TrainableParam)
    ///   - [2,1,1,-1,1] (Second ModuleA/ModuleB/Loop/Iteration/TrainableParam)
    /// 
    /// After unrolling there will be 5 trainable param variables with the following ModelIds:
    ///   - [1,1,1,0,1]
    ///   - [1,1,1,1,1]
    ///   - [2,1,1,0,1]
    ///   - [2,1,1,1,1]
    ///   - [2,1,1,2,1]
    /// 
    /// </summary>
    public struct ModelId : IComparable<ModelId>
    {
        public ImmutableArray<int> Vals = ImmutableArray<int>.Empty;

        public bool IsIterationModelId => this.Vals.Contains(-1);
        public int NumIterationIds => this.Vals.Count(x => x == -1);
        public int[] IterationIdLocations =>
                    this.Vals.Select((value, index) => new { value, index })
                    .Where(x => x.value == -1)
                    .Select(x => x.index)
                    .ToArray();

        private void check()
        {
            // Debug.Assert(this.Vals.All(x => x != 0));
        }

        public ModelId()
        {
            Vals = [];
            check();
        }

        public ModelId(params int[] modelId)
        {
            Vals = modelId.ToImmutableArray();
            check();
        }

        public static ModelId FromLongVals(long[] modelIdVals)
            => new ModelId(modelIdVals.Select(x => (int)x).ToArray());

        public ModelId(params ModelId[] parts)
        {
            Vals = [.. parts.SelectMany(x => x.Vals)];
            check();
        }

        public ModelId(string modelIdString)
        {
            if (modelIdString.Length == 0 || modelIdString == "[]")
            {
                Vals = [];
                return;
            }

            if (modelIdString[0] == '[')
            {
                if (modelIdString[^1] != ']')
                    throw new ArgumentException($"Invalid ModelId string format: {modelIdString}", nameof(modelIdString));

                modelIdString = modelIdString[1..^1];
            }

            var splits = modelIdString.Split(',').Select(x => x.Trim()).ToArray();

            if (splits.Any(x => !int.TryParse(x, out _)))
                throw new ArgumentException($"Invalid ModelId string format: {modelIdString}", nameof(modelIdString));

            Vals = splits.Select(x => int.Parse(x)).ToImmutableArray();
        }

        public bool IsInIterationModelId(ModelId iterationModelId, bool allowPartialIterationIndex)
        {
            Debug.Assert(allowPartialIterationIndex || !this.IsIterationModelId);
            Debug.Assert(allowPartialIterationIndex || iterationModelId.IsIterationModelId);

            if (this.Vals.Length != iterationModelId.Vals.Length)
                return false;

            for (int i = 0; i < this.Vals.Length; i++)
            {
                if (iterationModelId.Vals[i] != -1 &&
                    iterationModelId.Vals[i] != this.Vals[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Applies the given iteration indices to this ModelId.
        /// The current model Id must have as many -1 values as there are indices provided.
        /// </summary>
        /// <param name="iterationIndices"></param>
        /// <returns></returns>
        public ModelId ApplyIterationIndices(int[] iterationIndices)
        {
            if (this.NumIterationIds != iterationIndices.Length)
                throw new InvalidTensorOperationException(ErrorCodes.VG001, "Iteration Indices Application", $"indices count {iterationIndices.Length} vs expected {this.NumIterationIds}", "Must have as many indices values as there are -1s in Vals");

            if (iterationIndices.Length == 0)
                return this;

            var retval = new List<int>();
            var indexIntoIterationIndices = 0;
            for (int i = 0; i < Vals.Length; i++)
            {
                if (Vals[i] == -1)
                    retval.Add(iterationIndices[indexIntoIterationIndices++]);
                else
                    retval.Add(this.Vals[i]);
            }

            return new ModelId(retval.ToArray());
        }

        public ModelId ApplyFirstIterationIndex(int iterationIndex)
        {
            if (!this.IsIterationModelId)
                throw new InvalidTensorOperationException(ErrorCodes.VG002, "Iteration Index Application", $"ModelId {this}", "ModelId must be an iteration model id");

            var retval = new List<int>();
            var firstIterationIndex = this.IterationIdLocations[0];
            return new ModelId(
                    [.. this.Vals.Take(firstIterationIndex),
                    iterationIndex,
                    .. this.Vals.Skip(firstIterationIndex + 1)]);
        }

        public int[] IterationIndexMaxCounts(ModelId[] modelIds)
        {
            var thisvals = this.Vals;
            Debug.Assert(modelIds.All(x => !x.IsIterationModelId && x.Vals.Length == thisvals.Length));

            var iterationLocations = this.IterationIdLocations;

            var iterationIndexMaxCounts = new int[iterationLocations.Length];
            for (int i = 0; i < iterationLocations.Length; i++)
                iterationIndexMaxCounts[i] = modelIds.Max(x => x.Vals[iterationLocations[i]]) + 1;

            return iterationIndexMaxCounts;
        }

        public int[] IterationIndexValues(int[] iterationIndexLocations)
        {
            var vals = this.Vals;
            return iterationIndexLocations.Select(x => vals[x]).ToArray();
        }

        public int[] IterationIndexValues(ModelId modelIdLoopTemplate)
            => this.IterationIndexValues(modelIdLoopTemplate.IterationIdLocations);

        // Hashcode and equals implementations
        public override int GetHashCode()
        {
            return Vals.Aggregate(0, (current, item) => current ^ item.GetHashCode());
        }

        public override bool Equals(object? obj)
        {
            if (obj is ModelId other)
            {
                return Vals.SequenceEqual(other.Vals);
            }
            return false;
        }

        public static bool operator ==(ModelId? left, ModelId? right)
        {
            if (left is null && right is null)
                return true;

            if (left is null || right is null)
                return false;

            return left.Equals(right);
        }

        public static bool operator !=(ModelId? left, ModelId? right)
        {
            return !(left == right);
        }

        public override string ToString()
        {
            return string.Join(",", Vals);
        }

        /// <summary>
        /// Checks whether this ModelId starts with the ids found in x. A -1 in either will match any value.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public bool StartsWith(ModelId x)
        {
            if (this.Vals.Length < x.Vals.Length)
                return false;

            for (int i = 0; i < x.Vals.Length; i++)
            {
                if (this.Vals[i] == -1 || x.Vals[i] == -1)
                    continue;

                if (this.Vals[i] != x.Vals[i])
                    return false;
            }

            return true;
        }

        public int CompareTo(ModelId other)
        {
            int minLength = Math.Min(this.Vals.Length, other.Vals.Length);
            for (int i = 0; i < minLength; i++)
            {
                if (this.Vals[i] < other.Vals[i])
                    return -1;
                else if (this.Vals[i] > other.Vals[i])
                    return 1;
            }

            if (this.Vals.Length < other.Vals.Length)
                return -1;
            else if (this.Vals.Length > other.Vals.Length)
                return 1;
            return 0;
        }

        internal long[] ToLongVals() => this.Vals.Select(x => (long)x).ToArray(); 

        public static bool operator <(ModelId left, ModelId right)
        {
            return left.CompareTo(right) < 0;
        }

        public static bool operator <=(ModelId left, ModelId right)
        {
            return left.CompareTo(right) <= 0;
        }

        public static bool operator >(ModelId left, ModelId right)
        {
            return left.CompareTo(right) > 0;
        }

        public static bool operator >=(ModelId left, ModelId right)
        {
            return left.CompareTo(right) >= 0;
        }
    }
}
