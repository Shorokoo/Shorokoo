using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using static Shorokoo.Globals;
using System.Collections.Immutable;

namespace Shorokoo.Core
{
    public class Function
    {
        public class FunctionNotFoundException : ShorokooException
        {
            public string FunctionName { get; }

            public FunctionNotFoundException(string errorCode, string functionName, string additionalContext = "") 
                : base(errorCode, $"Function '{functionName}' was not found. {additionalContext}".Trim())
            {
                FunctionName = functionName;
            }
        }

        /// <summary>
        /// Metadata property on FunctionProto.
        /// The value is a FunctionType specifying the purpose of the function.
        /// Use FromComponentTypeName and ToComponentTypeName to convert between string and FunctionType.
        /// </summary>
        public const string IRFunctionTypeParamName = "ComponentType";

        /// <summary>
        /// Metadata property on FunctionProto to preserve the original module name (DefaultName).
        /// This is used when the ONNX function name differs from the original module name
        /// (e.g., when name conflicts require unique names like "fn_0" in the ONNX file).
        /// </summary>
        public const string IRFunctionFriendlyName = "ModuleFriendlyName";

        /// <summary>
        /// Metadata property on ValueInfoProto for Scalar tensors of DType.Model or DType.Module.
        /// The value is the name of the target function for the Model or Module.
        /// </summary>
        public const string IRFunctionSignatureParamName = "Signature";

        /// <summary>
        /// Metadata property on ValueInfoProto for inputs to a Function or the main computation graph.
        /// The value specifies whether that input is a hyperparameter, ready input or model input.
        /// Use
        /// </summary>
        public const string IRInputTypeName = "InputType";

        /// <summary>
        /// Metadata property on ValueInfoProto for a defaulted hyperparameter input: the
        /// <c>[Hyper(defaultValue)]</c> default, stored (invariant-culture) so it survives
        /// ONNX serialization. Read back onto the input node's <c>ShrkAttrDefaultValue</c> attribute.
        /// </summary>
        public const string IRDefaultValue = "DefaultValue";

        public string DefaultName { get; private set; }
        public string FriendlyName { get; private set; }

        public string ModelSignatureString => _signatures.Value.modelSignature;
        public string ModuleSignatureString => _signatures.Value.moduleSignature;
        private Lazy<(string moduleSignature, string modelSignature)> _signatures
            => __signatures ??= new Lazy<(string, string)>(() => OriginalFastGraph.GetSignatureStrings());
        private Lazy<(string moduleSignature, string modelSignature)>? __signatures;

        public FunctionType FunctionType { get; private set; }

        /// <summary>
        /// For <see cref="Shorokoo.Core.Nodes.OnnxNodes.FunctionType.StateParamInitializer"/> functions,
        /// who updates the state this initializer creates: the owning module's own forward logic
        /// (<see cref="StateOwnership.ModuleOwned"/>, e.g. BatchNorm running stats) or an optimizer
        /// module (<see cref="StateOwnership.OptimizerOwned"/>, e.g. Adam moments). <c>null</c> for
        /// non-state-initializer functions. The TrainingRig uses this to reject module-owned state
        /// inside optimizer graphs and optimizer-owned state inside model graphs.
        /// </summary>
        public StateOwnership? StateOwnership { get; private set; }

        private Function[]? directlyReferencedFunctions = null;
        public Function[] DirectlyReferencedFunctions
        {
            get
            {
                if (directlyReferencedFunctions is null)
                {
                    directlyReferencedFunctions = OriginalFastGraph.Nodes
                                        .Select(n => n.TargetFunction).NotNulls().ToArray();
                }

                return directlyReferencedFunctions;
            }
        }

        private Function[]? referencedFunctions = null;
        public Function[] ReferencedFunctions
        {
            get
            {
                if (referencedFunctions is null)
                {
                    referencedFunctions = DirectlyReferencedFunctions
                                    .SelectMany(x => x.ReferencedFunctions)
                                    .Concat(this.DirectlyReferencedFunctions).ToArray();
                }

                return referencedFunctions;
            }
        }

        internal ImmutableArray<Variable> Inputs { get { EnsureConvertedSnapshot(); return _inputs; } }
        internal ImmutableArray<Variable> HyperparamInputs { get { EnsureConvertedSnapshot(); return _hyperparamInputs; } }
        internal ImmutableArray<Variable> NonHyperparamInputs { get { EnsureConvertedSnapshot(); return _nonHyperparamInputs; } }
        internal ImmutableArray<Variable> Outputs { get { EnsureConvertedSnapshot(); return _outputs; } }
        internal ImmutableArray<int?> OutputRankOverrides { get { EnsureConvertedSnapshot(); return _outputRankOverrides; } }

        /// <summary>
        /// Primary representation of the function body.
        /// </summary>
        public FastComputationGraph OriginalFastGraph { get; }

        // Cached Variable views derived from a one-shot rebuild of OriginalFastGraph.
        // FastComputationGraphConverter.BuildNodes reconstructs the underlying
        // Node/Variable objects without wrapping them in a ComputationGraph; the
        // resulting inputs/outputs are stored here and Function never holds a CG handle.
        private bool _convertedSnapshotComputed;
        private ImmutableArray<Variable> _inputs;
        private ImmutableArray<Variable> _hyperparamInputs;
        private ImmutableArray<Variable> _nonHyperparamInputs;
        private ImmutableArray<Variable> _outputs;
        private ImmutableArray<int?> _outputRankOverrides;

        // Materializes the Variable snapshot directly from OriginalFastGraph via
        // BuildNodes. Shielded from any active outer LoopAPI context: rebuilding
        // nodes fires Node ctors which would otherwise leak into an enclosing
        // LoopAPI.Iterate body's pass-equality tracking.
        private void EnsureConvertedSnapshot()
        {
            if (_convertedSnapshotComputed) return;
            LoopAPI.PushLooperContext();
            try
            {
                var built = FastComputationGraphConverter.BuildNodes(this.OriginalFastGraph);
                _inputs = built.inputs;
                _hyperparamInputs = built.inputs
                    .Where(x => x.InputType == Shorokoo.Core.Nodes.NodeDefinitions.InputType.Hyperparam)
                    .ToImmutableArray();
                _nonHyperparamInputs = built.inputs
                    .Where(x => x.InputType != Shorokoo.Core.Nodes.NodeDefinitions.InputType.Hyperparam)
                    .ToImmutableArray();
                _outputs = built.outputs;
                _outputRankOverrides = OriginalFastGraph.OutputRankOverrides is null
                    ? built.outputs.Select(x => x.Rank).ToImmutableArray()
                    : OriginalFastGraph.OutputRankOverrides.ToImmutableArray();
                _convertedSnapshotComputed = true;
            }
            finally { LoopAPI.PopLooperContext(); }
        }

        public Function(FastComputationGraph fastGraph, FunctionType functionType, string? defaultName, string? friendlyName,
            StateOwnership? stateOwnership = null)
        {
            this.OriginalFastGraph = fastGraph;
            this.FunctionType = functionType;
            this.DefaultName = defaultName ?? friendlyName ?? "UnnamedFunction";
            this.FriendlyName = friendlyName ?? defaultName ?? "UnnamedFunction";
            this.StateOwnership = functionType == FunctionType.StateParamInitializer
                ? stateOwnership ?? Shorokoo.Modules.StateOwnership.ModuleOwned
                : null;
        }

        public Variable[] Call(params Variable?[] tensors)
            => InternalOp.FunctionInvoke(tensors,
                    this.Outputs.Select(x => x.Structure()).ToArray(),
                    this.Outputs.Select(x => x.DType).ToArray(),
                    this.OutputRankOverrides.Select(x => x ?? -1).ToArray(),
                    targetFn: this,
                    genericTypeArgs: null);

        public static InputType FromInputTypeName(string? name = null)
        {
            if (name is null)
                return InputType.ReadyInput;
            else if (name == nameof(InputType.Hyperparam))
                return InputType.Hyperparam;
            else if (name == nameof(InputType.ModelInput))
                return InputType.ModelInput;
            else if (name == nameof(InputType.ReadyInput))
                return InputType.ReadyInput;
            else if (name == nameof(InputType.GenericType))
                return InputType.GenericType;

            throw new UnsupportedDTypeException(ErrorCodes.FW005, name ?? "null", "ToInputType",
                    $"Input type name '{name}' is not supported for conversion to InputType");
        }

        public static string? ToInputTypeName(InputType? inputType)
        {
            if (inputType is null)
                return null;

            switch (inputType)
            {
                case InputType.Hyperparam:
                    return nameof(InputType.Hyperparam);
                case InputType.ModelInput:
                    return nameof(InputType.ModelInput);
                case InputType.ReadyInput:
                    return nameof(InputType.ReadyInput);
                case InputType.GenericType:
                    return nameof(InputType.GenericType);
                default:
                    throw new UnsupportedDTypeException(ErrorCodes.FW005, inputType?.ToString() ?? "null", "ToInputTypeName", 
                        $"InputType '{inputType}' is not supported for name conversion");
            }
        }

        public static FunctionType FromComponentTypeName(string? name = null)
        {
            if (name is null)
                return FunctionType.Function;
            else if (name == nameof(FunctionType.Function))
                return FunctionType.Function;
            else if (name == nameof(FunctionType.Module))
                return FunctionType.Module;
            else if (name == nameof(FunctionType.TrainableParamInitializer))
                return FunctionType.TrainableParamInitializer;
            else if (name == nameof(FunctionType.StateParamInitializer))
                return FunctionType.StateParamInitializer;
            else if (name == nameof(FunctionType.ModuleSignature))
                return FunctionType.ModuleSignature;

            throw new UnsupportedDTypeException(ErrorCodes.FW005, name ?? "null", "FromComponentTypeName", 
                $"Component type name '{name}' is not supported for FunctionType conversion");
        }

        public static string ToComponentTypeName(FunctionType functionType)
        {
            switch (functionType)
            {
                case FunctionType.Function:
                    return nameof(FunctionType.Function);
                case FunctionType.Module:
                    return nameof(FunctionType.Module);
                case FunctionType.TrainableParamInitializer:
                    return nameof(FunctionType.TrainableParamInitializer);
                case FunctionType.StateParamInitializer:
                    return nameof(FunctionType.StateParamInitializer);
                case FunctionType.ModuleSignature:
                    return nameof(FunctionType.ModuleSignature);
                default:
                    throw new UnsupportedDTypeException(ErrorCodes.FW005, functionType.ToString(), "ToComponentTypeName", 
                        $"FunctionType '{functionType}' is not supported for component type name conversion");
            }
        }

        private Shorokoo.Graph.FastComputationGraph? fastFlattenedGraph = null;
        /// <summary>
        /// Returns a FastCG representation of this function's body with every
        /// inlinable <c>MODEL_INVOKE</c> / <c>FUNCTION_INVOKE</c> recursively expanded;
        /// any remaining invokes (e.g. a call on a non-hyper signature-only model
        /// variable) are left in place for the caller's inline pass to resolve once
        /// the concrete model is substituted. The result is cached so each function
        /// is flattened at most once per process.
        ///
        /// Produced entirely on the Fast path — this method clones the primary
        /// <see cref="OriginalFastGraph"/> body and runs
        /// <see cref="Shorokoo.Core.Nodes.Processors.Fast.FastInlineModulesAndFunctions.Process"/>
        /// on the result. Sub-function flattening recurses through this same method,
        /// so no <c>ComputationGraph</c>-side inliner is invoked.
        ///
        /// Callers must <see cref="Shorokoo.Graph.FastComputationGraph.Clone"/> the
        /// returned graph (and re-key it) before splicing it into a parent graph, to
        /// avoid both aliasing the cache and NodeKey collisions when the same function
        /// is inlined at multiple call sites.
        /// </summary>
        internal Shorokoo.Graph.FastComputationGraph GetFastFlattenedGraph()
        {
            if (this.fastFlattenedGraph is null)
            {
                var fast = this.OriginalFastGraph.Clone();
                Shorokoo.Core.Nodes.Processors.Fast.FastInlineModulesAndFunctions.Process(fast);
                this.fastFlattenedGraph = fast;
            }

            Debug.Assert(this.fastFlattenedGraph.Inputs.Count == this.OriginalFastGraph.Inputs.Count);
            return this.fastFlattenedGraph;
        }

    }
}
