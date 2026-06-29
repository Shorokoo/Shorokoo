using Shorokoo.Core;
using Shorokoo;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Utils;
using Shorokoo.Core.Nodes;
using System.Collections.Immutable;
using System.Data;
using System.Diagnostics;
using System.Xml.Linq;

namespace Shorokoo
{
    using FullInputs = ImmutableDictionary<string, Variable?[]>;
    using FullOutputs = ImmutableDictionary<string, Variable?[]>;

    /// <summary>
    /// The body of the loop executes four times.
    /// First Pass:
    /// We build out the graph structure to be able to locate where variables from before the loop are used.
    /// 
    /// Second Pass:
    /// We follow the graph structure to locate where first pass "zombie" variables are used.
    /// This indicates the use of a loop variable.
    /// The corresponding variable from the first pass is the open loop node input.
    /// At the end of the second pass we build the Open Loop Node.
    /// 
    /// Third Pass:
    /// Here we build out the proper contents of the loop body.
    /// At those locations where a loop variable was identified, we use the appropriate Open Loop Node output variable.
    /// All tensors created in the body will be marked as inputs to the loop close node.
    /// In addition scanned tensors and the exit condition tensor are collected and used as inputs to the loop close node.
    /// At the end of the third pass, we build the Close Loop Node.
    /// Unused loop variables can be trimed later as a post processing operation on the VirtualGraph.
    /// 
    /// Fourth Pass:
    /// We override all returned variables during the execution of the loop body to be the corresponding output variable of the close loop node
    /// instead of the original output that was used as input to the close loop node.
    /// 
    /// For inner loops, nothing is tracked or modified during the first two passes of the outer loop.
    /// Then all four passes are executed on the outer loops' THIRD pass.
    /// On the outer loop's fourth's pass, the inner loop's fourth pass is executed again.
    /// 
    /// </summary>
    public class Looper
    {
        private Node? OpenLoopNode;
        private Node? CloseLoopNode;

        private List<Node> firstPassLoopBody = new List<Node>();
        private List<Node> secondPassLoopBody = new List<Node>();
        private List<Node> thirdPassLoopBody = new List<Node>();
        private List<Node> fourthPassLoopBody = new List<Node>();

        public int LoopDepth { get; private set; }
        public int CurrentPass { get; private set; }

        private Scalar<bit>? continueWhileTensor;
        private Scalar<int64>? maxNumIterations;
        // private Dictionary<Variable, LoopScanVariable> scanVariablesByScanInput = new Dictionary<Variable, LoopScanVariable>();

        private Dictionary<(int NodeIndex, int InputIndex), LoopVariable> loopVariableByNodeInputLocation = new Dictionary<(int NodeIndex, int InputIndex), LoopVariable>();
        private Dictionary<(int NodeIndex, int OutputIndex), LoopVariable> loopVariableByNodeOutputLocation = new Dictionary<(int NodeIndex, int OutputIndex), LoopVariable>();
        // private Dictionary<Variable, LoopVariable> openNodeInputs = new Dictionary<Variable, LoopVariable>();

        // Open node outputs except the first two: the vestigal condition variable (always set to scalar true) and the iteration index variable.
        private Dictionary<Variable, LoopVariable> openNodeOutputs = new Dictionary<Variable, LoopVariable>();
        private Dictionary<Variable, LoopVariable> thirdPassOutputs = new Dictionary<Variable, LoopVariable>();
        private Dictionary<Variable, LoopVariable> closeNodeOutputs = new Dictionary<Variable, LoopVariable>();
        private Dictionary<Variable, LoopVariable> innerLoopCloseNodeOutputs = new Dictionary<Variable, LoopVariable>();

        // private HashSet<Variable> zombieScanVariableOutputs = new HashSet<Variable>();

        private HashSet<Variable> allExternalInputs = new HashSet<Variable>();
        private HashSet<Variable> allExternalInputExceptLoopVariables = new HashSet<Variable>();
        private Dictionary<(int nodeIndex, int outputIndex), Variable> secondPassOuputZombieVariables = new Dictionary<(int nodeIndex, int outputIndex), Variable>();

        private Dictionary<(int NodeIndex, int InputIndex), LoopVariableInput> variableInputs = new Dictionary<(int NodeIndex, int InputIndex), LoopVariableInput>();
        private Dictionary<(int NodeIndex, int OutputIndex), LoopVariableOutput> variableOutputs = new Dictionary<(int NodeIndex, int OutputIndex), LoopVariableOutput>();
        private Dictionary<Variable, LoopVariableOutput> firstPassVariableOutputs = new Dictionary<Variable, LoopVariableOutput>();

        /// <summary>
        /// A special purpose loop variable that lets us output the iteration index from the Loop Close Node.
        /// It essentially will hold the number of iterations the loop has executed.
        /// </summary>`
        public LoopVariable? IterationIndexLoopVariable { get; private set; }

        public Looper(int depth)
        {
            this.LoopDepth = depth;
            this.CurrentPass = 0;
        }

        public void SetMaxNumIterations(Scalar<int64>? maxNumIterations)
        {
            Debug.Assert(this.CurrentPass == 1);
            Debug.Assert(this.maxNumIterations is null);
            this.maxNumIterations = maxNumIterations;
        }

        public Tensor<T> Scan<T>(Vector<T> v) where T : IVarType => Scan<T>((Tensor<T>)v);
        public Tensor<T> Scan<T>(Tensor<T> toScan) where T : IVarType
        {
            // Key the loop-variable dictionaries by the Immutable* graph value (they are populated
            // with node outputs, which are immutables); a struct handle would not match.
            Variable retVal = OnnxOp.LoopScanZombie(toScan);

            if (this.CurrentPass == 1)
            {
                var loopVariableOuputForScanVariable = firstPassVariableOutputs[retVal];
                loopVariableOuputForScanVariable.SetIsLocalScanVariable(toScan);
            }

            if (this.CurrentPass == 3)
            {
                var loopVariableForScanVariable = thirdPassOutputs[retVal];
                Debug.Assert(loopVariableForScanVariable.IsLocalScanVariable);
                loopVariableForScanVariable.SetLocalScanVariableInput(toScan);
            }

            return retVal;
        }

        public Vector<T> Scan<T>(Scalar<T> toScan) where T : IVarType
        {
            return (Variable)this.Scan<T>((Tensor<T>)toScan);
        }

        public void ContinueWhile(Scalar<bit> breakWhenTensor)
        {
            Debug.Assert(this.continueWhileTensor is null && this.CurrentPass == 3);
            this.continueWhileTensor = breakWhenTensor;
        }

        private static ImmutableDictionary<string, Variable?[]> applyMapping(ImmutableDictionary<string, Variable?[]> original, List<(Variable from, Variable to)> mapping)
        {
            var retval = new Dictionary<string, Variable?[]>();
            var mappingDct = new Dictionary<Variable, Variable>();
            foreach (var (from, to) in mapping)
                mappingDct[from] = to;

            // Built with a loop rather than ToDictionary: the mapping can contain
            // duplicate 'from' entries, and the last one wins.

            foreach (var key in original.Keys)
                retval[key] = original[key].Select(x =>
                    x is null ? null :
                    !mappingDct.ContainsKey(x) ? x :
                    mappingDct[x]).ToArray();

            return retval.ToImmutableDictionary();
        }

        // private static ImmutableDictionary<string, Variable[]> applyNotNullMapping(ImmutableDictionary<string, Variable[]> original, List<(Variable from, Variable to)> mapping)
        // {
        //     var retval = new Dictionary<string, Variable[]>();
        //     var mappingDct = mapping.ToDictionary(x => x.from, x => x.to);
        //     foreach (var key in original.Keys)
        //         retval[key] = original[key].Select(x => mappingDct.ContainsKey(x) ? mappingDct[x] : x).ToArray();
        // 
        //     return retval.ToImmutableDictionary();
        // }

        private void checkMatch(int nodeIndex, Node nextPass)
        {
            var prevPassLoopBody = this.CurrentPass == 2 ? firstPassLoopBody :
                                   this.CurrentPass == 3 ? secondPassLoopBody :
                                   thirdPassLoopBody;

            var curPassLoopBody = this.CurrentPass == 2 ? secondPassLoopBody :
                                  this.CurrentPass == 3 ? thirdPassLoopBody :
                                  fourthPassLoopBody;

            if (curPassLoopBody.Count > nodeIndex)
                throw new InvalidTensorOperationException(ErrorCodes.FW011, "Loop Body Validation", $"node index {nodeIndex}", "Current pass loop body count exceeds node index");

            if (prevPassLoopBody.Count <= nodeIndex)
                throw new InvalidTensorOperationException(ErrorCodes.FW012, "Loop Body Construction", $"node index {nodeIndex}", "Previous pass loop body count is insufficient for node index");

            var previousPass = prevPassLoopBody[nodeIndex];

            if (previousPass.OpCode != nextPass.OpCode)
            {
                throw new InvalidTensorOperationException(ErrorCodes.FW013, "Loop Pass Validation", $"previous: {previousPass.OpCode}, next: {nextPass.OpCode}", "OpCode mismatch between loop passes");
            }

            if (previousPass.Inputs.Length != nextPass.Inputs.Length)
                throw new InvalidTensorOperationException(ErrorCodes.FW014, "Loop Pass Validation", $"previous inputs: {previousPass.Inputs.Length}, next inputs: {nextPass.Inputs.Length}", "Input count mismatch between loop passes");

            if (previousPass.Inputs.Zip(nextPass.Inputs).Any(x => (x.First is null) != (x.Second is null)))
                throw new InvalidTensorOperationException(ErrorCodes.FW015, "Loop Pass Validation", "input nullness comparison", "Input nullness mismatch between loop passes");

            if (previousPass.Outputs.Length != nextPass.Outputs.Length)
                throw new InvalidTensorOperationException(ErrorCodes.FW016, "Loop Pass Validation", $"previous outputs: {previousPass.Outputs.Length}, next outputs: {nextPass.Outputs.Length}", "Output count mismatch between loop passes");

            if (previousPass.Outputs.Zip(nextPass.Outputs).Any(x => (x.First is null) != (x.Second is null)))
                throw new InvalidTensorOperationException(ErrorCodes.FW017, "Loop Pass Validation", "output nullness comparison", "Output nullness mismatch between loop passes");

            foreach (var attrdef in previousPass.NodeDef.AttributeDefs)
            {
                var attrName = attrdef.AttributeName;
                if (previousPass.Attributes.IsDefaultValue(attrName) !=
                    nextPass.Attributes.IsDefaultValue(attrName))
                    throw new InvalidTensorOperationException(ErrorCodes.FW018, "Loop Attribute Validation", $"attribute '{attrName}'", "Attribute default value status mismatch between loop passes");

                //     .Equals(nextPass.Attributes.GetAttributeObj(attrName).AssertNotNull()))
                //     throw new InvalidOperationException("");
            }
        }

        public static Variable?[] Inputs(FullInputs fullInputs)
            => fullInputs.OrderBy(x => x.Key).SelectMany(x => x.Value).ToArray();

        public static Variable?[] Outputs(FullOutputs fullInputs)
            => fullInputs.OrderBy(x => x.Key).SelectMany(x => x.Value).ToArray();

        private Dictionary<int, Scalar<int64>> dctLoopIndexVariables = new();
        public Scalar<int64> GetLoopIndexVariable()
        {
            if (dctLoopIndexVariables.ContainsKey(CurrentPass))
                return dctLoopIndexVariables[CurrentPass];

            Scalar<int64> indexVariable = OnnxOp.LoopIndexVariable();
            dctLoopIndexVariables[CurrentPass] = indexVariable;
            return indexVariable;
        }

        public (FullInputs newInputs, FullOutputs newOutputs) ProcessNode(Node node, FullInputs originalInputs, FullOutputs originalOutputs)
        {
            Debug.Assert(this.CurrentPass >= 1 && this.CurrentPass <= 4);
            var nodeInputs = Inputs(originalInputs);
            var nodeOutputs = Outputs(originalOutputs);

            if (this.CurrentPass == 1)
            {
                var nodeIndex = this.firstPassLoopBody.Count;
                this.firstPassLoopBody.Add(node);

                for (int inputIndex = 0; inputIndex < nodeInputs.Length; inputIndex++)
                {
                    var input = nodeInputs[inputIndex];
                    if (input is null) continue;

                    var loopVariable = new LoopVariableInput(input, nodeIndex, inputIndex);
                    this.variableInputs[loopVariable.Key] = loopVariable;
                    if (!this.firstPassVariableOutputs.ContainsKey(input))
                        this.allExternalInputs.Add(input);
                }

                for (int outputIndex = 0; outputIndex < nodeOutputs.Length; outputIndex++)
                {
                    var output = nodeOutputs[outputIndex];

                    if (output is null)
                        continue;

                    var loopVariable = new LoopVariableOutput(output, nodeIndex, outputIndex);
                    this.variableOutputs[loopVariable.Key] = loopVariable;
                    Debug.Assert(!this.firstPassVariableOutputs.ContainsKey(output));
                    this.firstPassVariableOutputs[output] = loopVariable;
                }

                // Keep everything as is. The outputs produced here will be checked against the inputs used in the second pass
                // to identify loop variables.
                return (originalInputs, originalOutputs);
            }

            if (this.CurrentPass == 2)
            {
                // Invalid scan variable inputs
                // if (nodeInputs.Any(x => this.zombieScanVariableOutputs.Contains(x)))
                //     throw new InvalidOperationException("Cannot use output of scan variables inside the loop.");

                var nodeIndex = this.secondPassLoopBody.Count;
                checkMatch(nodeIndex, node);
                this.secondPassLoopBody.Add(node);

                for (int inputIndex = 0; inputIndex < nodeInputs.Length; inputIndex++)
                {
                    var input = nodeInputs[inputIndex];
                    if (input is null) continue;

                    var loopInputVariable = this.variableInputs[(nodeIndex, inputIndex)];
                    loopInputVariable.SetSecondPassInput(input);
                }

                for (int outputIndex = 0; outputIndex < nodeOutputs.Length; outputIndex++)
                {
                    if (this.variableOutputs.ContainsKey((nodeIndex, outputIndex)))
                    {
                        var output = nodeOutputs[outputIndex];
                        Debug.Assert(output is not null);

                        var outputVariable = this.variableOutputs[(nodeIndex, outputIndex)];
                        outputVariable.SetSecondPassZombieOutput(output);
                        this.secondPassOuputZombieVariables[outputVariable.Key] = output;
                    }
                    else
                        Debug.Assert(nodeOutputs[outputIndex] is null);
                }

                // Keep everything as is. All outputs here are unused and unneeded.
                return (originalInputs, originalOutputs);
            }

            if (this.CurrentPass == 3)
            {
                // Invalid scan variable inputs
                // if (nodeInputs.Any(x => this.zombieScanVariableOutputs.Contains(x)))
                //     throw new InvalidOperationException("Cannot use output of scan variables inside the loop.");

                var nodeIndex = this.thirdPassLoopBody.Count;
                checkMatch(nodeIndex, node);
                this.thirdPassLoopBody.Add(node);

                // Here we construct the body of the loop, these are the "real" nodes that will be used in the final graph.

                // There are six kinds of inputs here:

                // Leave as-is:
                // 1. Non loop variables defined outside any loop.
                // 2. Variables defined during the current iteration of the loop.
                // 3. The iteration index
                // 4. Null inputs
                // 5. Variables defined by the output of a loop open node of an enclosing loop.

                // Use the loopOpenNodeOutput of the corresponding loop variable.
                // 6. Variables defined in the previous iteration of the loop.

                // Variables of type 1, 2 and 5 must be kept as is. Variables of type 3 must use the corresponding openNodeOutput variable.
                List<(Variable from, Variable to)> mapping = new List<(Variable from, Variable to)>();
                for (int inputIndex = 0; inputIndex < nodeInputs.Length; inputIndex++)
                {
                    var inputVariable = nodeInputs[inputIndex];

                    // Case 4: Leave null as-is
                    if (inputVariable is null)
                        continue;

                    if (this.loopVariableByNodeInputLocation.ContainsKey((nodeIndex, inputIndex)))
                    {
                        // Case 6: Grab the open node output of the corresponding loop variable.
                        var loopVariable = this.loopVariableByNodeInputLocation[(nodeIndex, inputIndex)];
                        mapping.Add((inputVariable, loopVariable.OpenNodeOutput.AssertNotNull()));

                        //if (loopVariable.IsLocalScanVariable)
                        //    loopVariable.Set
                    }
                    else if (Object.ReferenceEquals(inputVariable, OpenLoopNode.AssertNotNull().Outputs[0]))
                    {
                        // Case 3: use the Loop Index Variable as is.
                    }
                    else if (thirdPassOutputs.ContainsKey(inputVariable) || innerLoopCloseNodeOutputs.ContainsKey(inputVariable))
                    {
                        // Case 2:  
                    }
                    else
                    {
                        // Case 5: The real value we want to use here was captured during the first pass.
                        var originalInput = this.variableInputs[(nodeIndex, inputIndex)].FirstPassInput;
                        mapping.Add((inputVariable, originalInput));
                    }
                }

                List<(Variable from, Variable to)> outputMapping = new List<(Variable from, Variable to)>();
                for (int outputIndex = 0; outputIndex < nodeOutputs.Length; outputIndex++)
                {
                    var outputVariable = nodeOutputs[outputIndex];
                    if (outputVariable is null)
                        continue;

                    if (this.loopVariableByNodeOutputLocation.ContainsKey((nodeIndex, outputIndex)))
                    {
                        var loopVariable = this.loopVariableByNodeOutputLocation[(nodeIndex, outputIndex)];
                        if ((nodeIndex, outputIndex) == (0, 0))
                        {
                            Debug.Assert(loopVariable.FirstPassOutput.OwningNode.NodeDef.FullNodeOpName == OpCodes.LOOP_INDEX_VARIABLE);
                            Scalar<int64> actualIterationIndexVariable = (Variable)this.OpenLoopNode.AssertNotNull().Outputs[0].AssertNotNull();
                            loopVariable.SetThirdPassOutput(actualIterationIndexVariable);
                            outputMapping.Add((outputVariable, actualIterationIndexVariable));
                        }
                        else
                            loopVariable.SetThirdPassOutput(outputVariable);

                        this.thirdPassOutputs[outputVariable] = loopVariable;
                    }
                    else
                        Debug.Assert(outputVariable is null /* ||
                            outputVariable.Type == DType.Model ||
                            outputVariable.Type == DType.Module */);
                }

                return (applyMapping(originalInputs, mapping), applyMapping(originalOutputs, outputMapping));
            }

            if (this.CurrentPass == 4)
            {
                // Invalid scan variable inputs
                // if (nodeInputs.Any(x => this.zombieScanVariableOutputs.Contains(x)))
                //     throw new InvalidOperationException("Cannot use output of scan variables inside the loop.");

                var nodeIndex = this.fourthPassLoopBody.Count;

                checkMatch(nodeIndex, node);
                this.fourthPassLoopBody.Add(node);

                // Nodes constructed in the fourth pass are discared and never used.
                // We use this pass to overwrite the variables returned to the caller for all operations inside the
                // loop's body with the corresponding variables from the output of the loop close node.
                // This way the caller can use these variables freely outside the loop.

                List<(Variable from, Variable to)> mapping = new List<(Variable from, Variable to)>();
                for (int outputIndex = 0; outputIndex < nodeOutputs.Length; outputIndex++)
                {
                    var outputVariable = nodeOutputs[outputIndex];
                    if (outputVariable is not null && this.loopVariableByNodeOutputLocation.ContainsKey((nodeIndex, outputIndex)))
                    {
                        var loopVariable = this.loopVariableByNodeOutputLocation[(nodeIndex, outputIndex)];
                        if (loopVariable.CloseNodeOutput is not null)
                            mapping.Add((outputVariable, loopVariable.FourthPassOutput.AssertNotNull()));
                        else
                            loopVariable.SetInvalidFourthPassOutput(outputVariable);
                    }
                    else
                        Debug.Assert(outputVariable is null/* ||
                            outputVariable.Type == DType.Model ||
                            outputVariable.Type == DType.Module */);
                }

                return (originalInputs, applyMapping(originalOutputs, mapping));
            }

            throw new InvalidTensorOperationException(ErrorCodes.FW019, "Loop State Processing", "loop state resolution", "Failed to resolve loop state - unable to determine appropriate input/output mapping");
        }


        public void StartFirstPass()
        {
            if (this.CurrentPass != 0)
                throw new InvalidTensorOperationException(ErrorCodes.FW020, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start first pass - current pass must be 0");

            this.CurrentPass = 1;
        }

        public void StartSecondPass()
        {
            if (this.CurrentPass != 1)
                throw new InvalidTensorOperationException(ErrorCodes.FW021, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start second pass - current pass must be 1");

            this.CurrentPass = 2;
        }

        public void BuildLoopOpenNode()
        {
            if (this.CurrentPass != 2)
                throw new InvalidTensorOperationException(ErrorCodes.FW022, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot build loop open node - current pass must be 2");

            // Building the loop open node is primarily about identifying all the loop variables.

            // There are five kinds of loop variables to watch out for to make sure we identify them all:
            // 1. The simple loop variable, it has one initialization value, it is used once in an operation inside the loop
            //    and it is overriden later with the output of an (same or different) operation inside the loop
            //    such that in the next iteration of the loop the new variable is used instead of the initialization value.
            //
            // 2. The multi use loop variable. The initialization value is used in two or more operation. The variable that it is
            //    replaced with is used in the same for for all these operations.
            //
            // 3. The mixed use loop variable. The initialization value is used for two or more operation. The variable that is is
            //    replaced with is different for each of these operations.
            //    This kind of situation should be treated as multiple different loop variables that all have the same initialization
            //    tensor.
            //
            // 4. Unused loop variable. It is a variable that is generated as an output to an operation inside the loop, but it is not
            //    directly used in subsequent iteration of the loop.
            //
            // 5. The iteration index. This variable is provided as output from the open node and could be used after the loop to
            //    identify the number of iterations performed.

            // Case 1 is automatically addressed.

            // The VariableInput.Key splits off initialization value for each place they are used. This helps to address Case 3,
            // but causes duplication for Case 2.
            var loopVariableInputs = this.variableInputs.Values.Where(x =>
                                        x.SecondPassInput is not null &&
                                        this.firstPassVariableOutputs.ContainsKey(x.SecondPassInput));

            // Group the items for Case 2 into a single "canonicalLoopVariable".
            var canonicalLoopVariables = loopVariableInputs.GroupBy(x => (x.FirstPassInput, x.SecondPassInput.AssertNotNull())).ToDictionary(x => x.Key, x => x.ToHashSet());

            var loopVariablesWithInitializers = new List<LoopVariable>();
            foreach (var canonicalLoopVariable in canonicalLoopVariables)
            {
                (var firstPassInputVariable, var firstPassOutputVariable) = canonicalLoopVariable.Key;
                Debug.Assert(canonicalLoopVariable.Value.All(x => x.SecondPassInput == firstPassOutputVariable));

                var inputKeys = canonicalLoopVariable.Value.Select(x => x.Key).ToList();
                var outputKey = this.firstPassVariableOutputs[firstPassOutputVariable].Key;
                var secondPassOutputVariable = this.secondPassOuputZombieVariables[outputKey];

                var loopVariable = new LoopVariable(firstPassInputVariable, firstPassOutputVariable, secondPassOutputVariable, isLocalScanVariable: false, inputKeys, outputKey);
                loopVariablesWithInitializers.Add(loopVariable);
            }

            // This addresses Cases 1, 2 and 3. Cases 4 and 5 do not have any loop variable inputs.
            this.loopVariableByNodeInputLocation = loopVariablesWithInitializers.SelectMany(loopVariable =>
                                                        loopVariable.InputKeys.Select(inputKey => (inputKey, loopVariable)))
                                                        .ToDictionary(x => x.inputKey, x => x.loopVariable);

            // this.openNodeInputs = loopVariablesWithInitializers.ToDictionary(x => x.OpenNodeInput.AssertNotNull());

            // Identify loop variables that fall under Case 4. Which is all output variables that don't already have a specified loop variable.
            var identifiedFirstPassOutputLoopVariables = this.loopVariableByNodeInputLocation.Values.Select(x => x.FirstPassOutput).ToHashSet();

            var outputOnlyOutputVariables = this.firstPassVariableOutputs.Values.Where(x => !identifiedFirstPassOutputLoopVariables.Contains(x.FirstPassOutput)).ToList();
            var outputOnlyLoopVariables = outputOnlyOutputVariables
                                                     .Where(x => !x.IsLocalScanVariable /* &&
                                                            x.FirstPassOutput.Type != DType.Model &&
                                                            x.FirstPassOutput.Type != DType.Module */)
                                                .Select(x => new LoopVariable(null, x.FirstPassOutput, x.SecondPassOutput.AssertNotNull(), x.IsLocalScanVariable, [], x.Key))
                                                .ToList();

            var scanLoopVariables = outputOnlyOutputVariables.Where(x => x.IsLocalScanVariable)
                                                .Select(x => new LoopVariable(x.ScanInput, x.FirstPassOutput, x.SecondPassOutput.AssertNotNull(), x.IsLocalScanVariable, [], x.Key))
                                                .ToList();

            // var iterationIndexLoopVariable = new LoopVariable(null, this.IterationIndexFirstPassZombie.AssertNotNull(), this.IterationIndexSecondPassZombie.AssertNotNull(), [], (-1, -1));
            // this.IterationIndexLoopVariable = iterationIndexLoopVariable;

            var allNonScanLoopVariables = loopVariablesWithInitializers.Concat(outputOnlyLoopVariables).ToList();
            var allLoopVariables = allNonScanLoopVariables.Concat(scanLoopVariables).ToList();
            this.loopVariableByNodeOutputLocation = allLoopVariables.ToDictionary(x => x.OutputKey);

            var loopVariableInitializers = loopVariablesWithInitializers.Select(x => x.OpenNodeInputInitializer.AssertNotNull()).ToArray();

            Debug.Assert(this.OpenLoopNode is null);
            this.OpenLoopNode = OnnxOp.LoopOpen(maxNumIterations: this.maxNumIterations, condition: null, loopVariableInitializers: loopVariableInitializers);

            // The input's first two variables are the condition (kinda pointless) and the max number of iterations
            // The output's first two variables are the condition (always set to true and completely useless) and the current iteration number.
            Debug.Assert(this.OpenLoopNode.Inputs.Length == this.OpenLoopNode.Outputs.Length ||
                    (this.OpenLoopNode.Inputs.Length == 1 && this.OpenLoopNode.Outputs.Length == 2));
            Debug.Assert(this.OpenLoopNode.Outputs.Length == loopVariablesWithInitializers.Count + 2);
            var inputOutputMapper = loopVariablesWithInitializers.Zip(this.OpenLoopNode.Inputs.Skip(2).Zip(this.OpenLoopNode.Outputs.Skip(2).AssertNotNulls()));
            foreach ((var loopVariable, (var input, var output)) in inputOutputMapper)
            {
                Debug.Assert(Object.ReferenceEquals(loopVariable.OpenNodeInputInitializer, input));
                loopVariable.SetOpenNodeOutput(output);
                this.openNodeOutputs[output] = loopVariable;
            }

            var nonLoopExternalInputs = this.allExternalInputs.ToHashSet();
            foreach (var loopVariable in loopVariablesWithInitializers)
                nonLoopExternalInputs.Remove(loopVariable.OpenNodeInputInitializer.AssertNotNull());

            this.allExternalInputExceptLoopVariables = nonLoopExternalInputs;
        }

        public void StartThirdPass()
        {
            if (this.CurrentPass != 2)
                throw new InvalidTensorOperationException(ErrorCodes.FW020, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start third pass - current pass must be 2");

            this.CurrentPass = 3;
        }

        public void MapInnerLoopCloseNodeOutputsToOuterLoopThirdPassOutputs(ImmutableList<(Variable outerThirdPassOuput, Variable innerCloseNodeOutput)> mappings)
        {
            foreach (var mapping in mappings)
            {
                if (this.thirdPassOutputs.ContainsKey(mapping.outerThirdPassOuput))
                {
                    var loopVariable = this.thirdPassOutputs[mapping.outerThirdPassOuput];
                    loopVariable.SetInnerLoopCloseNodeOutput(mapping.innerCloseNodeOutput);
                    this.innerLoopCloseNodeOutputs[mapping.innerCloseNodeOutput] = loopVariable;
                }
            }
        }

        public ImmutableList<(Variable outerThirdPassOuput, Variable innerCloseNodeOutput)> BuildLoopCloseNode()
        {
            if (this.CurrentPass != 3)
                throw new InvalidTensorOperationException(ErrorCodes.FW021, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot build loop close node - current pass must be 3");

            // Debug.Assert(this.IterationIndexLoopVariable is not null);
            // this.IterationIndexLoopVariable.SetThirdPassOutput(IterationIndexLoopVariable.OpenNodeOutput.AssertNotNull());
            // this.thirdPassOutputs[this.IterationIndexLoopVariable.CloseNodeInput.AssertNotNull()] = this.IterationIndexLoopVariable;
            // Debug.Assert(this.thirdPassOutputs.ContainsKey(this.IterationIndexLoopVariable.CloseNodeInput.AssertNotNull()));

            var closeNodeLoopVariables = this.OpenLoopNode.AssertNotNull().Outputs.Skip(2).AssertNotNulls().Select(x => this.openNodeOutputs[x].AssertNotNull()).ToArray();
            var closeNodeLoopInputs = closeNodeLoopVariables.Select(x => x.CloseNodeInput.AssertNotNull()).ToArray();
            Debug.Assert(thirdPassOutputs.Values.Where(x => !x.IsLocalScanVariable && x.OpenNodeInputInitializer is not null).Count() == closeNodeLoopInputs.Length);

            var closeNodeScanLoopVariables = thirdPassOutputs.Values.Where(x => x.IsLocalScanVariable).ToArray();
            var closeNodeScanInputs = closeNodeScanLoopVariables.Select(x => x.CloseNodeInput.AssertNotNull()).ToArray();

            var loopCloseNode = OnnxOp.LoopClose(
                            this.continueWhileTensor ?? Globals.Scalar(true),
                            closeNodeLoopInputs,
                            closeNodeScanInputs,
                            this.OpenLoopNode.AssertNotNull());

            var numLoopOutputs = closeNodeLoopInputs.Length;
            var numScanOutputs = closeNodeScanInputs.Length;
            var numBreakConditions = 1;

            Debug.Assert(closeNodeLoopVariables.Length == numLoopOutputs);
            Debug.Assert(closeNodeScanLoopVariables.Length == numScanOutputs);
            Debug.Assert(this.OpenLoopNode.AssertNotNull().Outputs.Length == this.openNodeOutputs.Count + 2);
            Debug.Assert(this.OpenLoopNode.AssertNotNull().Outputs.Length == numLoopOutputs + 2);
            Debug.Assert(loopCloseNode.Inputs.Length == numBreakConditions + numLoopOutputs + numScanOutputs);
            Debug.Assert(loopCloseNode.Outputs.Length == numLoopOutputs + numScanOutputs);
            foreach (((var closeNodeInputVariable, var closeNodeOutputVariable), var loopVariable) in
                            loopCloseNode.Inputs.Skip(1).Zip(loopCloseNode.Outputs.AssertNotNulls()).Zip(closeNodeLoopVariables.Concat(closeNodeScanLoopVariables)))
            {
                if (loopVariable.OpenNodeInputInitializer is not null)
                {
                    loopVariable.SetCloseNodeOutput(closeNodeOutputVariable);
                    this.closeNodeOutputs[closeNodeOutputVariable] = loopVariable;
                }

                Debug.Assert(Object.ReferenceEquals(loopVariable.CloseNodeInput, closeNodeInputVariable));
            }

            this.CloseLoopNode = loopCloseNode;

            return this.closeNodeOutputs.Values.Select(x => (x.FirstPassOutput, x.CloseNodeOutput.AssertNotNull())).ToImmutableList();
        }

        public void StartFourthPass()
        {
            if (this.CurrentPass != 3)
                throw new InvalidTensorOperationException(ErrorCodes.FW020, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start fourth pass - current pass must be 3");

            this.CurrentPass = 4;
        }

        public void Terminate()
        {
            if (this.CurrentPass != 4)
                throw new InvalidTensorOperationException(ErrorCodes.FW021, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot terminate loop - current pass must be 4");

            this.CurrentPass = 5;

            foreach (var loopVariable in this.loopVariableByNodeOutputLocation.Values.Where(x => x.OpenNodeInputInitializer is null))
                loopVariable.InvalidFourthPassOutput.AssertNotNull().IsValid = false;
        }
    }

    public static class LoopAPI
    {
        // [ThreadStatic] field initializers only run on the thread that first touches the
        // type, so on every other thread the backing field is null. Use lazy-initialized
        // properties so each thread gets its own instance on first access.
        [ThreadStatic]
        private static List<Looper>? _looperStack;
        private static List<Looper> LooperStack
        {
            get => _looperStack ??= new List<Looper>();
            set => _looperStack = value;
        }

        [ThreadStatic]
        private static Stack<List<Looper>>? _looperContexts;
        private static Stack<List<Looper>> LooperContexts
            => _looperContexts ??= new Stack<List<Looper>>();

        /// <summary>
        /// Gets the interation indices of all outerloop ordered from outermost to innerermost.
        /// </summary>
        internal static ImmutableList<Scalar<int64>> IterationIndices => LooperStack.Select(x => x.GetLoopIndexVariable()).ToImmutableList();

        internal static void PushLooperContext()
        {
            LooperContexts.Push(LooperStack);
            LooperStack = new List<Looper>();
        }
        internal static void PopLooperContext()
        {
            Debug.Assert(LooperStack.Count == 0); // Really shouldn't be popping an active loop.
            LooperStack = LooperContexts.Pop();
        }

        internal static (FullInputs newInputs, FullOutputs newOutputs) ProcessNode(Node node)
        {
            // No active loop.
            if (LooperStack.Count == 0)
                return (node.FullInputs, node.FullOutputs);

            var activeIdx = LooperStack.FindIndex(x => x.CurrentPass == 0) - 1;
            if (activeIdx < 0) activeIdx = LooperStack.Count - 1;
            var activeL = LooperStack[activeIdx];

            // Nodes involved in the creation of the loop, these, in principle are not part of the loop body.
            if (node.NodeDef.FullNodeOpName == OpCodes.LOOP ||
                node.NodeDef.FullNodeOpName == OpCodes.LOOP_FAKE_INPUT)
                return (node.FullInputs, node.FullOutputs);

            // This node has no inputs, therefore it cannot be part of the loop body.
            // This is important because constants are occasionally created during the creation of the loop that should
            // not be processed as part of the loop body.
            if (node.Inputs.Length == 0 &&
                node.NodeDef.FullNodeOpName != OpCodes.LOOP_INDEX_VARIABLE)
                return (node.FullInputs, node.FullOutputs);

            FullInputs retvalInputs = node.FullInputs;
            FullOutputs retvalOutputs = node.FullOutputs;

            var activeLooperIndex = LooperStack.FindIndex(x => x.CurrentPass == 0) - 1;
            if (activeLooperIndex < 0) activeLooperIndex = LooperStack.Count - 1;

            var activeLooper = LooperStack[activeLooperIndex];
            if (activeLooper.CurrentPass == 1 && activeLooperIndex > 0)
            {
                // We process an inner loop's pass 1 at the same time as its outer loop's pass 3.
                // This enables the inner loop to capture the outer loop's OpenNode outputs used as inputs to nodes
                // part of the inner loop's body.
                Debug.Assert(LooperStack.Take(activeLooperIndex).All(x => x.CurrentPass == 3));
                var outerLooper = LooperStack[activeLooperIndex - 1];
                Debug.Assert(outerLooper.CurrentPass == 3);
                (retvalInputs, retvalOutputs) = outerLooper.ProcessNode(node, retvalInputs, retvalOutputs);
            }

            // The active loop should be on pass 1, 2 or 4
            // For pass 3: Because, inner loops get activated while their outer loops is on pass 3,
            // a loop is only activated during pass 3 for nodes that do not belong to inner-more loops.
            Debug.Assert(activeLooper.CurrentPass == 1 ||
                        activeLooper.CurrentPass == 2 ||
                        (activeLooper.CurrentPass == 3 && (activeLooperIndex == LooperStack.Count - 1)) ||
                        activeLooper.CurrentPass == 4);

            // If the current node is in the scope of inner loops of the active loop, then these
            // inner loops should be on pass 0.
            Debug.Assert(LooperStack.Count <= activeLooperIndex + 1 ||
                            LooperStack.Skip(activeLooperIndex + 1).All(x => x.CurrentPass == 0));

            // If the active loop is inside the scope of outer loops, then these outer loops
            // should on pass 3.
            Debug.Assert(activeLooperIndex == 0 ||
                        LooperStack.Take(activeLooperIndex).All(x => x.CurrentPass == 3));


            (retvalInputs, retvalOutputs) = activeLooper.ProcessNode(node, retvalInputs, retvalOutputs);

            return (retvalInputs, retvalOutputs);
        }

        private static IEnumerable<(Action<Scalar<bit>> breakWhen, Looper looper, Scalar<int64> iterationIndex)> LoopFull(Scalar<int64>? maxNumIterations)
        {
            var looper = new Looper(LooperStack.Count);
            LooperStack.Add(looper);

            try
            {
                // Do nothing if there is an outer loop that is in its first or second pass.
                if (looper.LoopDepth != 0 && (LooperStack[looper.LoopDepth - 1].CurrentPass < 3 || LooperStack[looper.LoopDepth - 1].CurrentPass == 4))
                {
                    yield return ((x) => { }, looper, looper.GetLoopIndexVariable());
                }
                else
                {
                    // First pass, track the nodes that are part of the body of the loop.
                    looper.StartFirstPass();
                    looper.SetMaxNumIterations(maxNumIterations);
                    yield return ((x) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(LooperStack.Count == looper.LoopDepth + 1);

                    // Second pass, identify the loop variables
                    looper.StartSecondPass();
                    yield return ((x) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(LooperStack.Count == looper.LoopDepth + 1);

                    looper.BuildLoopOpenNode();

                    // Third pass, build the actual loop body
                    looper.StartThirdPass();
                    yield return (looper.ContinueWhile, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(LooperStack.Count == looper.LoopDepth + 1);

                    var outerLoopMappings = looper.BuildLoopCloseNode();
                    if (looper.LoopDepth > 0)
                        LooperStack[looper.LoopDepth - 1].MapInnerLoopCloseNodeOutputsToOuterLoopThirdPassOutputs(outerLoopMappings);

                    // Fourth pass, make loop output variables available to the caller.
                    looper.StartFourthPass();
                    yield return ((x) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(LooperStack.Count == looper.LoopDepth + 1);

                    looper.Terminate();
                }

                Debug.Assert(LooperStack.Count == looper.LoopDepth + 1);
                Debug.Assert(LooperStack[looper.LoopDepth] == looper);
            }
            finally
            {
                // Always remove the looper from the stack, even if an exception occurred
                if (LooperStack.Count > looper.LoopDepth && LooperStack[looper.LoopDepth] == looper)
                {
                    LooperStack.RemoveAt(looper.LoopDepth);
                }
            }
        }

        /// <summary>
        /// Call this function with any C# variable that is used after the loop and assigned to within the loop 
        /// but not used inside the loop before the assignment.
        /// 
        /// <code>
        /// e.g.
        /// var x = Scalar(12L);
        /// var y = Scalar(0L);
        /// foreach(var ctx in LoopAPI.Iterate(numIterations)
        /// {
        ///     LoopAPI.Init(x); // Shorokoo is made aware of x's default value here.
        ///     y = y + 1; // y is used before the assignment operation
        ///     x = y;     // x is not used before the assignment operation. So LoopAPI.Init is required.
        ///     x = x + 1; // It's too late the variable x has already been overwritten. Shorokoo cannot determine x's default value from this line.
        /// }
        /// 
        /// // Now Shorokoo knows that x's default value is 12L, and can build the loop node of the computation graph accordingly in the event the loop executes 0 times.
        /// var z = x + 3;
        /// </code>
        /// </summary>
        /// <param name="toInits"></param>
        public static void Init(params Variable[] toInits)
        {
            foreach (var toInit in toInits)
                OnnxOp.Identity(toInit, toInit.Rank);
        }

        public static IEnumerable<IterationContext> Iterate(Scalar<int64> maxNumIterations)
        {
            foreach ((var continueWhile, var scanner, var iterationIndex) in LoopFull(maxNumIterations))
                yield return new IterationContext(continueWhile, scanner, iterationIndex);
        }
    }

    public class IterationContext
    {
        private Looper looper;
        private Action<Scalar<bit>> continueWhile;

        public Scalar<int64> IterationIndex { get; private set; }

        internal IterationContext(Action<Scalar<bit>> continueWhile, Looper looper, Scalar<int64> iterationIndex)
        {
            this.continueWhile = continueWhile;
            this.looper = looper;
            this.IterationIndex = iterationIndex;
        }

        public Tensor<T> Scan<T>(Tensor<T> tensor) where T : IVarType => looper.Scan(tensor);
        public Tensor<T> Scan<T>(Vector<T> v) where T : IVarType => looper.Scan((Tensor<T>)v);

        public Vector<T> Scan<T>(Scalar<T> scalar) where T : IVarType => looper.Scan(scalar);

        public void Break(Scalar<bit> exitLoopWhenTrue) => continueWhile(!exitLoopWhenTrue);

        public void ContinueWhile(Scalar<bit> exitLoopWhenFalse) => continueWhile(exitLoopWhenFalse);
    }

    public class LoopVariableInput
    {
        public LoopVariableInput(Variable input, int nodeIndex, int inputIndex)
        {
            this.Key = (nodeIndex, inputIndex);
            this.FirstPassInput = input;
        }

        public (int NodeIndex, int InputIndex) Key { get; private set; }
        public Variable FirstPassInput { get; private set; }
        public Variable? SecondPassInput { get; private set; }

        public LoopVariableOutput? LoopVariableConnection { get; private set; }

        public void SetSecondPassInput(Variable input)
        {
            Debug.Assert(this.SecondPassInput is null);
            this.SecondPassInput = input;
        }
    }

    public class LoopVariableOutput
    {
        public LoopVariableOutput(Variable output, int nodeIndex, int outputIndex)
        {
            this.Key = (nodeIndex, outputIndex);
            this.FirstPassOutput = output;
        }

        public (int NodeIndex, int OutputIndex) Key { get; private set; }

        public Variable FirstPassOutput { get; private set; }
        public Variable? SecondPassOutput { get; private set; }

        public Variable? ScanInput { get; private set; }

        public void SetSecondPassZombieOutput(Variable output)
        {
            Debug.Assert(this.SecondPassOutput is null);
            this.SecondPassOutput = output;
        }

        public void SetIsLocalScanVariable(Variable scanInput)
        {
            Debug.Assert(!this.IsLocalScanVariable);
            Debug.Assert(this.ScanInput is null);
            // Store the graph node (these are matched against node outputs elsewhere).
            this.ScanInput = scanInput;
            this.IsLocalScanVariable = true;
        }

        public bool IsLocalScanVariable { get; private set; }
    }

    public class LoopVariable
    {
        public LoopVariable(Variable? initializerVariable, Variable firstPassZombie, Variable secondPassZombie, bool isLocalScanVariable, List<(int NodeIndex, int InputIndex)> inputKeys, (int NodeIndex, int OutputIndex) outputKey)
        {
            this.InputKeys = inputKeys.ToImmutableList();
            this.OutputKey = outputKey;
            this.OpenNodeInputInitializer = initializerVariable;
            this.FirstPassOutput = firstPassZombie;
            this.SecondPassZombie = secondPassZombie;
            this.IsLocalScanVariable = isLocalScanVariable;
        }

        public bool IsLocalScanVariable { get; private set; }

        public ImmutableList<(int NodeIndex, int InputIndex)> InputKeys { get; private set; }
        public (int NodeIndex, int InputIndex) OutputKey { get; private set; }

        /// <summary>
        /// The initial value available before the loop.
        /// </summary>
        public Variable? OpenNodeInputInitializer { get; private set; }

        /// <summary>
        /// The tensor used inside the Loop Body / Loop Graph
        /// </summary>
        public Variable? OpenNodeOutput { get; private set; }

        public Variable? ScanVariableThirdPassInput { get; private set; }

        /// <summary>
        /// The updated value after each execution of the Loop Body / Loop Graph
        /// If this LoopVariable corresponds to the Loop Index or the output of a node where the current loop
        /// is the innermost loop, then this is simply that output/ loop index variable.
        /// If this is the output of a node that belong to a nested inner loop (at any level), then this
        /// will correspond to the CloseNode output variable of the outermost of these nested inner loops.
        /// </summary>
        public Variable? CloseNodeInput => InnerLoopCloseNodeOutput ?? this.ScanVariableThirdPassInput ?? ThirdPassOutput;

        /// <summary>
        /// The final value after the loop exits.
        /// </summary>
        public Variable? CloseNodeOutput { get; private set; }

        /// <summary>
        /// Variable created during the first pass over the loop's body.
        /// If there is an outer then this is also the variable recorded as that outer loop's ThirdPassOutput.
        /// </summary>
        public Variable FirstPassOutput { get; private set; }

        /// <summary>
        /// Variable created during the third pass over the loop's body.
        /// If there is an inner loop then this is also the variable recorded as that inner loop's FirstPassOutput
        /// </summary>
        public Variable? ThirdPassOutput { get; private set; }

        public Variable? InvalidFourthPassOutput { get; private set; }

        public Variable? FourthPassOutput => CloseNodeOutput ?? InvalidFourthPassOutput;

        /// <summary>
        /// When there is an inner loop, then this will hold that inner's loop corresponding Close Node Output
        /// it is to be used here as the close node input.
        /// </summary>
        public Variable? InnerLoopCloseNodeOutput { get; private set; }

        /// <summary>
        /// Temporary variable created during the second pass over the loop's body. This variable should eventually get discarded.
        /// </summary>
        public Variable? SecondPassZombie { get; private set; }

        public void SetOpenNodeOutput(Variable openNodeOutput)
        {
            Debug.Assert(this.OpenNodeOutput is null);
            this.OpenNodeOutput = openNodeOutput;
        }

        public void SetThirdPassOutput(Variable thirdPassOutput)
        {
            Debug.Assert(this.ThirdPassOutput is null);
            this.ThirdPassOutput = thirdPassOutput;
        }

        public void SetCloseNodeOutput(Variable closeNodeOutput)
        {
            Debug.Assert(this.CloseNodeOutput is null);
            this.CloseNodeOutput = closeNodeOutput;
        }

        public void SetInnerLoopCloseNodeOutput(Variable innerLoopCloseNodeOutput)
        {
            Debug.Assert(this.InnerLoopCloseNodeOutput is null);
            this.InnerLoopCloseNodeOutput = innerLoopCloseNodeOutput;
        }

        public void SetLocalScanVariableInput<T>(Tensor<T> toScan) where T : IVarType
        {
            Debug.Assert(this.IsLocalScanVariable && this.ScanVariableThirdPassInput is null);
            this.ScanVariableThirdPassInput = toScan;
        }

        public void SetInvalidFourthPassOutput(Variable fourthPassOutput)
        {
            Debug.Assert(this.InvalidFourthPassOutput is null);
            this.InvalidFourthPassOutput = fourthPassOutput;
        }
    }

}