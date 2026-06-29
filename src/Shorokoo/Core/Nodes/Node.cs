
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Shorokoo.Core.Graph;
using Shorokoo.Core;
using Shorokoo;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Net.NetworkInformation;
using static Shorokoo.Core.InternalGlobals;
using static Shorokoo.Globals;
using Shorokoo.Core.Nodes.OnnxNodes;

namespace Shorokoo.Core.Nodes
{
    public class Node
    {
        private static long NextOrderingHintNumber = 0;

        public long OrderingHintNumber { get; private set; }

        /// <summary>
        /// A globally unique identifier for this node.
        /// NodeKeys remain stable across graph transformations that preserve node identity.
        /// </summary>
        public NodeKey Key { get; private set; }

        public string OpCode { get; }

        public string OpName => OpCode;

        public OnnxCSharpAttributes Attributes { get; }

        public NodeDefinition NodeDef { get; }

        public BestGraphAttribute[] Subgraphs => this.GraphAttributeNames.Order().Select(Attributes.GetGraphVal).AssertNotNulls().ToArray();

        public TensorData? GetTensorData() => this.NodeDef.OpName != InternalOpCodes.MODEL_PARAM_DATA ? null : Attributes.GetTensorVal(OnnxOpAttributeNames.ShrkAttrTensorData);

        /// <summary>
        /// Gets the IsTrainable value for MODEL_PARAM_DATA, TRAINABLE_PARAM_X_REF nodes, and function call nodes.
        /// Returns true for trainable parameters, false for state parameters.
        /// For function call nodes, derives the value from the FunctionType of the target function.
        /// Throws if the attribute is not present on MODEL_PARAM_DATA/TRAINABLE_PARAM_X_REF nodes.
        /// </summary>
        public bool GetIsTrainable()
        {
            // For function call nodes with trainable/state param initializer, use the FunctionType
            if (this.TargetFunction != null)
            {
                if (this.TargetFunction.FunctionType == FunctionType.TrainableParamInitializer)
                    return true;
                if (this.TargetFunction.FunctionType == FunctionType.StateParamInitializer)
                    return false;
            }
            
            // For MODEL_PARAM_DATA and TRAINABLE_PARAM_X_REF nodes, check the attribute
            var attrVals = this.Attributes.GetAttributeVals();
            if (attrVals.TryGetValue(OnnxOpAttributeNames.ShrkAttrIsTrainable, out var val) && val is bool boolVal)
                return boolVal;
            throw new InvalidOperationException($"Missing '{OnnxOpAttributeNames.ShrkAttrIsTrainable}' attribute on node '{this.OpCode}'. All trainable/state parameter nodes must specify whether they are trainable.");
        }

        public bool IsOpenNode { get; protected set; }
        public bool IsCloseNode { get; protected set; }

        public bool IsModelInput => NodeDef.OpName == InternalOpCodes.MODEL_TENSOR_INPUT || 
                                    NodeDef.OpName == InternalOpCodes.MODEL_OPTIONAL_INPUT || 
                                    NodeDef.OpName == InternalOpCodes.MODEL_SEQUENCE_INPUT ||
                                    NodeDef.OpName == InternalOpCodes.MODEL_TENSORSTRUCT_INPUT ||
                                    NodeDef.OpName == InternalOpCodes.GENERIC_TYPE_INPUT;

        /// <summary>
        /// Returns true if this is an initialized model parameter (MODEL_PARAM_DATA).
        /// 
        /// This could be a trainable parameter or a state parameter, use GetIsTrainable to determine which.
        /// </summary>
        public bool IsModelParamData => NodeDef.OpName == InternalOpCodes.MODEL_PARAM_DATA;

        /// <summary>
        /// Returns true if this is a model param initializer of any kind.
        /// This could be a standard module param initializer node (TRAINABLE_PARAM_REF).
        /// A param initializer applied to a reference model instance:
        ///  - TRAINABLE_PARAM_MODEL_REF, the reference model instance is identified by a tensor.
        ///  - TRAINABLE_PARAM_ID_REF, the reference model instance is identified by a model id.
        /// 
        /// Or it could be a function call node that calls a model parameter initializer function such as a TrainableParamInitializer or a StateParamInitializer.
        /// 
        /// Note that this returns false for an initialized model parameter (MODEL_PARAM_DATA).
        /// </summary>
        public bool IsModelParamInitializer
        {
            get
            {
                if (this.NodeDef.OpName == InternalOpCodes.TRAINABLE_PARAM_REF ||
                    this.NodeDef.OpName == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF ||
                    this.NodeDef.OpName == InternalOpCodes.TRAINABLE_PARAM_ID_REF)
                    Debug.Assert(this.TargetFunction?.FunctionType == FunctionType.TrainableParamInitializer ||
                                 this.TargetFunction?.FunctionType == FunctionType.StateParamInitializer);

                return this.TargetFunction?.FunctionType == FunctionType.TrainableParamInitializer ||
                       this.TargetFunction?.FunctionType == FunctionType.StateParamInitializer;
            }
        }

        /// <summary>
        /// Returns true if this node's target function is a trainable param initializer (not state param initializer).
        /// </summary>
        public bool IsTrainableParamInitializerOnly => this.TargetFunction?.FunctionType == FunctionType.TrainableParamInitializer;

        /// <summary>
        /// Returns true if this node's target function is a state param initializer.
        /// </summary>
        public bool IsStateParamInitializer => this.TargetFunction?.FunctionType == FunctionType.StateParamInitializer;

        public bool IsFunction => this.NodeDef.OpName == InternalOpCodes.FUNCTION_INVOKE;

        public bool IsGraphNode => IsOpenNode || IsCloseNode;

        public string[] GraphAttributeNames => NodeDef.AttributeDefs.Where(x => x.Type == AttributeType.Graph).Select(x => x.AttributeName).ToArray();

        public Node? GraphOpenNode { get; internal set; }

        public string DefaultName => this.Key.ToString();

        public string? FriendlyName { get; protected set; }

        // public virtual ImmutableArray<Variable?> Inputs => FullInputs.OrderBy(x => x.Key, comparer: StringComparer.Ordinal).SelectMany(x => x.Value).ToImmutableArray();

        private ImmutableArray<Variable?>? inputs = null;

        /// <summary>
        /// All inputs to this node, except the connecting tensor of a close node.
        /// </summary>
        public virtual ImmutableArray<Variable?> Inputs
        {
            get
            {
                if (inputs is not null)
                    return inputs.AssertNotNull();

                inputs = FullInputs.OrderBy(x => x.Key, comparer: StringComparer.Ordinal).SelectMany(x => x.Value).ToImmutableArray();
                return inputs.AssertNotNull();
            }
        }

        /// <summary>
        /// All inputs to this node, including the connecting tensor of a close node.
        /// </summary>
        public virtual ImmutableArray<Variable?> AllInputs => !this.IsCloseNode ? this.Inputs : this.Inputs.Insert(0, this.ConnectingTensor);

        public virtual Variable?[] InputsWithConnectingTensor => this.IsCloseNode ? this.Inputs.Append(this.ConnectingTensor).ToArray() : this.Inputs.ToArray();

        public virtual Variable?[] Outputs => FullOutputs.OrderBy(x => x.Key, comparer: StringComparer.Ordinal).SelectMany(x => x.Value).ToArray();


        /// <summary>
        /// All outputs of this node, except the connecting tensor of an open node.
        /// </summary>
        public virtual ImmutableArray<Variable?> OutputsImmutable => Outputs.ToImmutableArray();

        /// <summary>
        /// All outputs of this node, including the connecting tensor of an open node.
        /// </summary>
        public virtual ImmutableArray<Variable?> AllOutputs => !this.IsOpenNode ? this.OutputsImmutable : this.OutputsImmutable.Insert(0, this.ConnectingTensor);

        public Variable? ConnectingTensor;

        /// <summary>
        /// Input variables classified by the attribute name of the graph attribute they correspond to.
        /// For input variables not associated with a graph (the typical case) the input variables are found in the empty string entry.
        /// REMINDER: The outputs of a graph attribute are found here as the inputs of the close node.
        /// </summary>
        public virtual ImmutableDictionary<string, Variable?[]> FullInputs { get; protected set; }

        public virtual ImmutableDictionary<string, ImmutableArray<Variable?>> FullInputsImmutable
            => FullInputs.ToImmutableDictionary(x => x.Key, x => x.Value.ToImmutableArray());

        /// <summary>
        /// Output variables classified by the attribute name of the graph attribute they correspond to.
        /// For output variables not associated with a graph (the typical case) the output variables are found in the empty string entry.
        /// REMINDER: The inputs of a graph attribute are found here as the outputs of the open node.
        /// </summary>
        public virtual ImmutableDictionary<string, Variable?[]> FullOutputs { get; protected set; }
        public ImmutableDictionary<string, ImmutableArray<Variable?>> FullOutputsImmutable
            => FullOutputs.ToImmutableDictionary(x => x.Key, x => x.Value.ToImmutableArray());

        public int NumTrailingNullInputs { get; protected set; } = 0;

        public Variable?[] InputsWithTrailingNulls => [.. Inputs, .. Enumerable.Repeat((Variable?)null, this.NumTrailingNullInputs)];

        public string? StackTrace { get; protected set; }

        public Function? TargetFunction { get; protected set; }

        public ModelParamIdentifierTemplate? IdentifierTemplate { get; protected set; }

        public bool IsRelativeModelIdOperator => (  OpCode == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF ||
                                                    OpCode == InternalOpCodes.TRAINABLE_PARAM_ID_REF ||
                                                    OpCode == InternalOpCodes.SUBMODEL);

        /// <summary>
        /// Nodes whose output tensor has a model id.
        /// </summary>
        [Obsolete]
        public bool IsModuleNode => this.Outputs.Any(x => x?.Type == DType.Model) ||
                                    this.OpCode == InternalOpCodes.MODEL_PARAM_DATA ||
                                    this.IsModelParamInitializer;

        public bool IsModuleCall => this.OpCode == InternalOpCodes.MODEL_INVOKE;

        public string FullNodeOpName => this.OpCode == OpCodes.IF_OPEN ? OpCodes.IF :
                                        this.OpCode == OpCodes.IF_CLOSE ? OpCodes.IF :
                                        this.OpCode == OpCodes.LOOP_OPEN ? OpCodes.LOOP :
                                        this.OpCode == OpCodes.LOOP_CLOSE ? OpCodes.LOOP :
                                        this.OpCode == OpCodes.SCAN_OPEN ? OpCodes.SCAN :
                                        this.OpCode == OpCodes.SCAN_CLOSE ? OpCodes.SCAN :
                                        this.OpCode == OpCodes.SEQUENCE_MAP_OPEN ? OpCodes.SEQUENCE_MAP :
                                        this.OpCode == OpCodes.SEQUENCE_MAP_CLOSE ? OpCodes.SEQUENCE_MAP :
                                        this.OpCode;

        public bool IsGraphOpenNode => this.IsOpenNode;

        public bool IsStateUpdateLink => this.OpCode == InternalOpCodes.STATE_UPDATE_LINK;

        public bool IsWithStateDeps => this.OpCode == InternalOpCodes.WITH_STATE_DEPS;

        public Node(NodeDefinition nodeDef, OnnxCSharpAttributes? attributes, ImmutableDictionary<string, Variable?[]> inputs, ImmutableDictionary<string, OutputTensorInfo[]> outputs, string? stackTrace, string? defaultName, string? identifierTemplateString, Function? targetFunction = null, Node? openNode = null, NodeKey? existingKey = null, long? existingOrderingHint = null)
        {

            if (inputs.SelectMany(x => x.Value).NotNulls().Any(x => !x.IsValid))
            {
                var invalidInputs = inputs.SelectMany(x => x.Value).NotNulls().Where(x => !x.IsValid).ToArray();
                var inputNames = string.Join(", ", invalidInputs.Select(x => x.GetType().Name));
                throw new OnnxNodeException(ErrorCodes.NOD001, nodeDef.OpName, defaultName ?? "Unknown",
                    $"Invalid input variables detected: {inputNames}. {ErrorMessage}");
            }

            this.OrderingHintNumber = existingOrderingHint ?? Interlocked.Increment(ref NextOrderingHintNumber);
            this.Key = existingKey ?? NodeKey.New();
            this.FriendlyName = defaultName;
            this.NodeDef = nodeDef;
            this.IsOpenNode = nodeDef.IsOpenNode;
            this.IsCloseNode = nodeDef.IsCloseNode;
            this.OpCode = nodeDef.OpName;
            this.Attributes = attributes ?? OnnxCSharpAttributes.FromCSharpVals(new Dictionary<string, object?>(), new NodeDefAttributeDef[0].ToImmutableList());
            this.FullInputs = inputs;

            this.TargetFunction = targetFunction;
            this.GraphOpenNode = openNode;

            this.IdentifierTemplate = identifierTemplateString == null ? null : new ModelParamIdentifierTemplate(identifierTemplateString);

            if (this.IsOpenNode)
            {
                this.ConnectingTensor = new Variable(DType.Invalid, this, null, null, DataStructure.Tensor);
            }

            if (this.IsCloseNode && this.GraphOpenNode is not null)
            {
                this.ConnectingTensor = this.GraphOpenNode.ConnectingTensor;
            }

            // For operators that create a new model variable, the ModuleFn is stored in the
            // node's TargetFunction (rather than flowing from the attribute through the
            // operator-definition system).
            var targetFnIsModuleFn = this.OpCode == OpCodes.SEQUENCE_CONSTRUCT || this.OpCode == OpCodes.SEQUENCE_EMPTY ||
                                     this.OpCode == InternalOpCodes.CREATE_MODULE ||
                                     this.OpCode == InternalOpCodes.MODEL_TENSOR_INPUT ||
                                     this.OpCode == InternalOpCodes.SUBMODEL;

            var moduleFnOverride = targetFnIsModuleFn ? targetFunction : null;
            if (this.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
                moduleFnOverride = ((Variable)this.Inputs[0]!).ModuleFn;
            if (this.OpCode == OpCodes.SEQUENCE_AT && this.Inputs[0] is Variable inputVar)
                moduleFnOverride = inputVar.ModuleFn;

            this.FullOutputs = outputs.ToImmutableDictionary(x => x.Key,
                        x => x.Value.Select(x => x.Name == string.Empty ? null : 
                            this.MakeOutput(x.DType, 
                                x.ModuleFn ?? moduleFnOverride, 
                                x.Structure, x.Rank, x.Name)).ToArray());

            this.StackTrace = stackTrace is null ? new StackTrace(fNeedFileInfo: true).ToString() : stackTrace;

            // Loops do a lot of strange things that override the normal way nodes are constructed.
            (this.FullInputs, this.FullOutputs) = LoopAPI.ProcessNode(this);

            this.inputs = null;

            // Assign TensorKeys to all outputs after FullOutputs is finalized
            this.AssignTensorKeys();

            if (this.OpCode == InternalOpCodes.MODULE_SET_HYPERPARAMS)
            {
                var moduleNode = this.Inputs[0]!.OwningNode;
                Debug.Assert(moduleNode.TargetFunction is not null);
                Debug.Assert(this.Inputs.Length == moduleNode.TargetFunction.HyperparamInputs.Length + 2);
            }

            if (this.NodeDef.OpName == InternalOpCodes.TRAINABLE_PARAM_REF ||
                this.NodeDef.OpName == InternalOpCodes.TRAINABLE_PARAM_MODEL_REF ||
                this.NodeDef.OpName == InternalOpCodes.TRAINABLE_PARAM_ID_REF)
                Debug.Assert(this.TargetFunction?.FunctionType == FunctionType.TrainableParamInitializer ||
                             this.TargetFunction?.FunctionType == FunctionType.StateParamInitializer);

            Debug.Assert(!(this.OpCode == InternalOpCodes.MODEL_INVOKE && this.Inputs[0]!.ModuleFn is null));
            Debug.Assert(this.Outputs.NotNulls().All(x => x.ModuleFn is null || x.Type == DType.Model || x.Type == DType.Module || x.Type == DType.Int64));
        }

        /// <summary>
        /// Assigns TensorKeys to all output tensors and the connecting tensor (if any).
        /// Called at the end of the constructor after FullOutputs is finalized.
        /// </summary>
        private void AssignTensorKeys()
        {
            // Assign keys to connecting tensor first (uses index -1)
            if (this.IsOpenNode && this.ConnectingTensor != null)
            {
                ((Variable)this.ConnectingTensor).SetKey(TensorKey.ForConnectingTensor(this.Key));
            }

            // Assign keys to all outputs in order
            int outputIndex = 0;
            foreach (var group in this.FullOutputs.OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                foreach (var output in group.Value)
                {
                    if (output != null)
                    {
                        // Set the key on the non-generic Variable base (previously a dynamic
                        // dispatch, because the base used to be the generic ImmutableVariable<T>).
                        ((Variable)output).SetKey(new TensorKey(this.Key, outputIndex));
                    }
                    outputIndex++;
                }
            }
        }

        public Variable?[] TrimNulls(Variable?[] inputs)
            => inputs.Reverse().SkipWhile(x => x is null).Reverse().ToArray();

        protected Variable MakeOutput(DType type, Function? moduleFn, DataStructure structure, int? rank, string? name)
        {
            switch(structure)
            {
                case DataStructure.Tensor:
                    if (rank == 0)
                        return InternalGlobals.Scalar(type, this, moduleFn, name);
                    else if (rank == 1)
                        return InternalGlobals.Vector(null, type, this, moduleFn, name);
                    else
                        return InternalGlobals.Tensor(null, type, this, moduleFn, name, rank: rank);
                case DataStructure.Sequence:
                    return InternalGlobals.TensorSequence(type, this, moduleFn, name);
                case DataStructure.Optional:
                    return InternalGlobals.OptionalTensor(type, this, moduleFn, name);
                case DataStructure.TensorStruct:
                    return Shorokoo.Core.InternalGlobals.TensorStruct(type, this, moduleFn, name);
            }

            throw new UnsupportedDTypeException(ErrorCodes.NOD002, structure.ToString(), "MakeOutput", 
                $"DataStructure '{structure}' is not supported for output creation. Supported types: Tensor, Sequence, Optional, TensorStruct");
        }

        public override string ToString()
        {
            return "Node: " + this.NodeDef.OpName;
        }

        private const string ErrorMessage = @"Cannot use a variable outside of a loop that was assigned inside a loop but not previously used inside that same loop.

This is an unfortunate limitation of the current Shorokoo implementation.

For the curious, this is because Shorokoo is unable to retrieve
the initial value of the variable and therefore cannot make a variable available that will have the correct value after the loop in the event that the
loop executes zero time.

In addition Shorokoo is currently unable to determine whether the loop is guaranteed to execute at least once, and therefore reacts conservatively to 
avoid bugs.

E.g.
Example 1:
var x = 1;
loop 
{
    x = x + 1; // No problem here, x is used as input before it is assigned to x.
}

x = -x; // This is OK

Example 2:
var x = 1;
var y = 2;
loop 
{
    y = x + 1; // y is assigned to before being used as input. Using y outside of the loop is forbidden.
    x = y + 1; // No problem here, x is assigned to after being used in the previous line.
}

x = -x; // This is OK
y = -y; // Because of how y was constructed, it cannot be used outside the loop. Error is reported here.

Fix 1:
var x = 1;
var y = 2;
loop 
{
    Loop.Init(y); // Shorokoo identifies y's initial value here.
    y = x + 1; // No problem here, y is assigned to after being recorded by Loop.Init.
    x = y + 1; // No problem here, x is assigned to after being used in the previous line.
}

x = -x; // This is OK
y = -y; // This is OK


Fix 2:
var x = 1;
var y = 2;
loop 
{
    OnnxOp.Identity(y); // Shorokoo identifies y's initial value here. Even if the Identity operator is eventually optimized away in the computation graph.
    y = x + 1; // No problem here, y is assigned to after being used by the Identity operator.
    x = y + 1; // No problem here, x is assigned to after being used in the previous line.
}

x = -x; // This is OK
y = -y; // This is OK
";
    }
}
