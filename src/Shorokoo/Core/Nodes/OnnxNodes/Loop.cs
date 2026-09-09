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
    /// 
    /// Lag Pass (the second pass traced a second time):
    /// A local holding what another local held one iteration ago advances by only one lag step
    /// per pass, so after two passes it still reads its pre-loop value and looks like nothing.
    /// One more tracking pass moves it onto a first-pass body output, one pass behind the carry
    /// it trails, which is what identifies it. At the end of the lag pass we build the Open Loop
    /// Node.
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
        private List<Node> lagPassLoopBody = new List<Node>();
        private List<Node> thirdPassLoopBody = new List<Node>();
        private List<Node> fourthPassLoopBody = new List<Node>();

        public int LoopDepth { get; private set; }
        public int CurrentPass { get; private set; }

        /// <summary>Which of the two traces of pass 2 is running: 0 for the second pass proper,
        /// 1 for the lag pass that follows it. Meaningless outside pass 2.</summary>
        private int secondPassRound;

        /// <summary>Counts every trace of the body, so per-trace caches cannot collide across
        /// the two traces of pass 2.</summary>
        private int traceRound;

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

        /// <summary>Third-pass outputs this looper recorded while a NESTED loop was tracing its own
        /// first pass. Such a value is produced inside that loop's body, so it has no value of its
        /// own at the end of one of THIS loop's iterations unless the nested loop carries it out.</summary>
        private HashSet<Variable> nestedBodyThirdPassOutputs = new HashSet<Variable>();

        // private HashSet<Variable> zombieScanVariableOutputs = new HashSet<Variable>();

        private HashSet<Variable> allExternalInputs = new HashSet<Variable>();
        private HashSet<Variable> allExternalInputExceptLoopVariables = new HashSet<Variable>();
        private Dictionary<(int nodeIndex, int outputIndex), Variable> secondPassOuputZombieVariables = new Dictionary<(int nodeIndex, int outputIndex), Variable>();

        private Dictionary<(int NodeIndex, int InputIndex), LoopVariableInput> variableInputs = new Dictionary<(int NodeIndex, int InputIndex), LoopVariableInput>();
        private Dictionary<(int NodeIndex, int OutputIndex), LoopVariableOutput> variableOutputs = new Dictionary<(int NodeIndex, int OutputIndex), LoopVariableOutput>();
        private Dictionary<Variable, LoopVariableOutput> firstPassVariableOutputs = new Dictionary<Variable, LoopVariableOutput>();

        /// <summary>
        /// Input locations of the <c>Identity</c> nodes <see cref="LoopAPI.Init"/> emits. A read at
        /// one of these is a user declaration that the variable is carried, which is what lets the
        /// second pass tell a genuine reassignment from the variable substitutions the surrounding
        /// machinery performs between passes (an inner loop's first pass runs during its outer
        /// loop's third, so plain inequality between the passes means nothing on its own).
        /// </summary>
        private HashSet<(int NodeIndex, int InputIndex)> initDeclarationInputs = new HashSet<(int NodeIndex, int InputIndex)>();

        /// <summary>How many of the open node's loop variables are lag carries. They own no body
        /// node output, so they are absent from <see cref="thirdPassOutputs"/>.</summary>
        private int lagCarryCount;


        /// <summary>Set by <see cref="LoopAPI.Init"/> immediately before it emits its
        /// <c>Identity</c>, and consumed by the next node the first pass records.</summary>
        internal bool NextNodeIsInitDeclaration { get; set; }

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
                // An ENCLOSING loop's ctx.Scan called from inside a nested loop's body reaches
                // here once per pass of that inner loop, but this looper only records nodes on
                // the inner loop's first pass (LoopAPI.ProcessNode). The zombie the later inner
                // passes create was never registered here, and those passes are throwaway: bind
                // on the one that was recorded and let the rest fall through untouched.
                if (!thirdPassOutputs.TryGetValue(retVal, out var loopVariableForScanVariable))
                    return retVal;
                Debug.Assert(loopVariableForScanVariable.IsLocalScanVariable);

                // Bind the scan to the value the BODY reads this iteration. For a carry read
                // before the body updates it, that is NOT the caller's C# local: the body is
                // traced four times and the local is never rebound between passes, so by the
                // third pass it holds what the first two passes advanced it to — nodes that live
                // in the OUTER graph. ProcessNode has already rewritten the zombie node's own
                // input to the carry's open-node output, so take the rewrite from there.
                //
                // Only that rewrite. ProcessNode's other one is the outer-scope fallback, which
                // resolves an input to the first pass's node; for a body value the looper does
                // not track — anything built from a zero-input op, which node interception skips
                // — that fallback names the pass-1 node outside the loop, and binding it would
                // hoist e.g. a scanned RandomNormal out of the body and stack one draw N times.
                // Every other case leaves the input alone, and where the fallback is an identity
                // — the scanned tensor really is loop-invariant — it agrees with the local anyway.
                var inBodyScanInput = retVal.OwningNode.Inputs[0].AssertNotNull();
                loopVariableForScanVariable.SetLocalScanVariableInput(
                    this.openNodeOutputs.ContainsKey(inBodyScanInput) ? inBodyScanInput : toScan);
            }

            return retVal;
        }

        public Vector<T> Scan<T>(Scalar<T> toScan) where T : IVarType
        {
            return (Variable)this.Scan<T>((Tensor<T>)toScan);
        }

        /// <summary>
        /// Binds the loop's exit condition to the value the BODY reads this iteration, the way
        /// <see cref="Scan{T}(Tensor{T})"/> binds a scan input. <paramref name="wrapped"/> is an
        /// Identity the iteration context put around <paramref name="asWritten"/> purely so the
        /// read sits at a node input: a bare local — a bit carry read before the body updates it —
        /// names an OUTER-graph node by this pass, and ProcessNode rewrites node inputs, not
        /// locals. Only the loop-carry rewrite is taken, so a condition the body already produces
        /// keeps its own node and the Identity is left dead for the graph to prune.
        /// </summary>
        public void ContinueWhile(Scalar<bit> wrapped, Scalar<bit> asWritten)
        {
            Debug.Assert(this.continueWhileTensor is null && this.CurrentPass == 3);
            var inBodyCondition = ((Variable)wrapped).OwningNode.Inputs[0].AssertNotNull();
            this.continueWhileTensor =
                this.openNodeOutputs.ContainsKey(inBodyCondition) ? inBodyCondition : asWritten;
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
            var prevPassLoopBody = this.CurrentPass == 2 ? (this.secondPassRound == 0 ? firstPassLoopBody : secondPassLoopBody) :
                                   this.CurrentPass == 3 ? lagPassLoopBody :
                                   thirdPassLoopBody;

            var curPassLoopBody = this.CurrentPass == 2 ? (this.secondPassRound == 0 ? secondPassLoopBody : lagPassLoopBody) :
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
            // Keyed by trace round rather than pass: pass 2 is traced twice, and returning the
            // first trace's variable on the second would skip a body node the other passes have.
            if (dctLoopIndexVariables.ContainsKey(traceRound))
                return dctLoopIndexVariables[traceRound];

            Scalar<int64> indexVariable = OnnxOp.LoopIndexVariable();
            dctLoopIndexVariables[traceRound] = indexVariable;
            return indexVariable;
        }

        public (FullInputs newInputs, FullOutputs newOutputs) ProcessNode(
            Node node, FullInputs originalInputs, FullOutputs originalOutputs, bool fromNestedLoopBody = false)
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
                    if (this.NextNodeIsInitDeclaration)
                        this.initDeclarationInputs.Add(loopVariable.Key);
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

                this.NextNodeIsInitDeclaration = false;

                // Keep everything as is. The outputs produced here will be checked against the inputs used in the second pass
                // to identify loop variables.
                return (originalInputs, originalOutputs);
            }

            if (this.CurrentPass == 2)
            {
                // Invalid scan variable inputs
                // if (nodeInputs.Any(x => this.zombieScanVariableOutputs.Contains(x)))
                //     throw new InvalidOperationException("Cannot use output of scan variables inside the loop.");

                var isLagPass = this.secondPassRound == 1;

                var nodeIndex = isLagPass ? this.lagPassLoopBody.Count : this.secondPassLoopBody.Count;
                checkMatch(nodeIndex, node);
                (isLagPass ? this.lagPassLoopBody : this.secondPassLoopBody).Add(node);

                for (int inputIndex = 0; inputIndex < nodeInputs.Length; inputIndex++)
                {
                    var input = nodeInputs[inputIndex];
                    if (input is null) continue;

                    var loopInputVariable = this.variableInputs[(nodeIndex, inputIndex)];
                    if (isLagPass)
                        loopInputVariable.SetLagPassInput(input);
                    else
                        loopInputVariable.SetSecondPassInput(input);
                }

                // The lag pass keeps the second pass's zombie outputs: they are what the open
                // node's loop variables are built from, and the lag pass only adds input reads.
                if (!isLagPass)
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
                        // A read that reaches Case 5 names a value from OUTSIDE this loop — except
                        // when it names one of this loop's own first-pass body outputs. That means
                        // the local trails a body value by more passes than identification saw: one
                        // extra trace catches a lag of one, and a longer chain advances a step per
                        // pass, so it is still reading its pre-loop value when the open node is
                        // built and lands here instead. Left alone it would take the fallback below
                        // and be pinned to that pre-loop value — silently, on every engine.
                        if (this.firstPassVariableOutputs.ContainsKey(inputVariable))
                            throw new UnsupportedLoopVariableAssignmentException(
                                ErrorCodes.FW049, LagChainTooDeepGuidance);

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
                        if (fromNestedLoopBody)
                            this.nestedBodyThirdPassOutputs.Add(outputVariable);
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
            this.traceRound++;
        }

        public void StartSecondPass()
        {
            if (this.CurrentPass != 1)
                throw new InvalidTensorOperationException(ErrorCodes.FW021, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start second pass - current pass must be 1");

            this.CurrentPass = 2;
            this.traceRound++;
        }

        public void StartLagPass()
        {
            if (this.CurrentPass != 2 || this.secondPassRound != 0)
                throw new InvalidTensorOperationException(ErrorCodes.FW021, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start lag pass - current pass must be the first trace of pass 2");

            this.secondPassRound = 1;
            this.traceRound++;
        }

        public void BuildLoopOpenNode()
        {
            if (this.CurrentPass != 2 || this.secondPassRound != 1)
                throw new InvalidTensorOperationException(ErrorCodes.FW022, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot build loop open node - current pass must be the lag pass");

            using var loopNodeBuild = GraphTrace.Loopers.EnterLoopNodeBuild();

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

            // A variable declared with LoopAPI.Init whose read changed between the passes without
            // becoming a body output was reassigned by the body to a value computed OUTSIDE the loop.
            // The loop itself can carry it — a Shorokoo close node may take that outside tensor as its
            // input directly, and the ONNX export inserts the Identity vanilla Loop needs — but the
            // variable cannot be handed back. After the loop the user's variable *is* that outside
            // value, the same Variable the rest of the graph may hold, and the fourth pass rebinds a
            // body value by re-tracing the node that produced it: a bare assignment produces no node,
            // so there is nothing to rebind and no way to tell this use of the value from any other.
            // Until this was caught the carry was dropped silently and a zero-iteration loop returned
            // the body's value instead of the pre-loop one, contradicting what LoopAPI.Init promises.
            //
            // A lag carry reads the same way on the second pass — the trailed carry's value from
            // before the loop — and is told apart by the lag pass, where the read has moved onto a
            // body output. Without that exclusion a lagged local declared with LoopAPI.Init was
            // refused as an outside assignment, and pointed at LoopAPI.Carry for a shape that
            // needs no wrapping at all.
            var reassignedFromOutsideTheBody = this.variableInputs.Values
                .FirstOrDefault(x => this.initDeclarationInputs.Contains(x.Key)
                                     && x.SecondPassInput is not null
                                     && !this.firstPassVariableOutputs.ContainsKey(x.SecondPassInput)
                                     && !Object.ReferenceEquals(x.SecondPassInput, x.FirstPassInput)
                                     && !(x.LagPassInput is not null
                                          && this.firstPassVariableOutputs.ContainsKey(x.LagPassInput)));
            if (reassignedFromOutsideTheBody is not null)
                throw new UnsupportedLoopVariableAssignmentException(
                    ErrorCodes.FW023, CarryAssignedFromOutsideGuidance);

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

            // 6. The lag-1 carry: a local holding what another carry held one iteration ago. Its
            //    read does not move between the first two passes — each pass advances such a local
            //    by a single lag step, so it still holds the pre-loop value and the filter above
            //    never fires — and the lag pass is what exposes it: by then the read has landed on
            //    a FIRST-pass body output, one pass behind the carry it trails. Left unidentified,
            //    every read of it resolved through ProcessNode's outer-scope case to the value it
            //    held before the loop.
            var lagVariableInputs = this.variableInputs.Values.Where(x =>
                                        x.LagPassInput is not null &&
                                        x.SecondPassInput is not null &&
                                        !this.firstPassVariableOutputs.ContainsKey(x.SecondPassInput) &&
                                        this.firstPassVariableOutputs.ContainsKey(x.LagPassInput));

            var carryByOutputKey = new Dictionary<(int NodeIndex, int OutputIndex), LoopVariable>();
            foreach (var carry in loopVariablesWithInitializers)
                carryByOutputKey[carry.OutputKey] = carry;

            var lagLoopVariables = new List<LoopVariable>();
            foreach (var lagGroup in lagVariableInputs.GroupBy(x => (x.FirstPassInput, x.LagPassInput.AssertNotNull())))
            {
                (var lagInitializer, var trailedFirstPassOutput) = lagGroup.Key;

                // The value assigned is the trailed local's value at the point of the assignment,
                // which precedes that local's own update — otherwise the second pass would already
                // have seen the read move and identified an ordinary carry. So it is the trailed
                // carry's value at the START of the iteration: its open-node output. A local that
                // is not carried has no such value to hand back, and cannot be trailed.
                var trailedOutputKey = this.firstPassVariableOutputs[trailedFirstPassOutput].Key;
                if (!carryByOutputKey.TryGetValue(trailedOutputKey, out var trailedCarry))
                    throw new UnsupportedLoopVariableAssignmentException(
                        ErrorCodes.FW047, LagCarryOfUncarriedValueGuidance);

                lagLoopVariables.Add(LoopVariable.Lag(
                    lagInitializer, trailedCarry, trailedFirstPassOutput, lagGroup.Select(x => x.Key).ToList()));
            }
            // A nested loop cannot hand a lag carry on to its enclosing loop. Every other carry
            // travels out through the close node and is matched to the enclosing loop's own
            // variable by the body output that produced it (MapInnerLoopCloseNodeOutputs...); a
            // lag carry has no body output of its own — it shares the trailed carry's — so there
            // is nothing to match it by, and the enclosing loop re-derives it as a lag carry of
            // its OWN trailed carry, resetting it at every enclosing iteration.
            if (this.LoopDepth > 0 && lagLoopVariables.Count > 0)
                throw new UnsupportedLoopVariableAssignmentException(
                    ErrorCodes.FW048, LagCarryInNestedLoopGuidance);

            loopVariablesWithInitializers.AddRange(lagLoopVariables);
            this.lagCarryCount = lagLoopVariables.Count;

            // This addresses Cases 1, 2, 3 and 6. Cases 4 and 5 do not have any loop variable inputs.
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

            var allNonScanLoopVariables = loopVariablesWithInitializers.Where(x => !x.IsLagCarry)
                                                .Concat(outputOnlyLoopVariables).ToList();
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

            foreach (var lagVariable in lagLoopVariables)
                lagVariable.BindLagSource();

            var nonLoopExternalInputs = this.allExternalInputs.ToHashSet();
            foreach (var loopVariable in loopVariablesWithInitializers)
                nonLoopExternalInputs.Remove(loopVariable.OpenNodeInputInitializer.AssertNotNull());

            this.allExternalInputExceptLoopVariables = nonLoopExternalInputs;
        }

        internal const string InnerScanReadAfterEnclosingLoopGuidance =
            "an inner loop's ctx.Scan output was read after the ENCLOSING loop."
            + "\n"
            + "\nThe stacked tensor is produced once per iteration of the enclosing loop, and that "
            + "loop has no initial value to return for it when it runs zero times, so it cannot "
            + "carry it out."
            + "\n"
            + "\nConsume the inner loop's scan output inside the enclosing loop's body, or scan on "
            + "the enclosing loop's own context instead.";

        internal const string LagCarryReadAfterLoopGuidance =
            "a variable holding the previous iteration's value of a loop carry was read after the "
            + "loop."
            + "\n"
            + "\nThe loop computes that value, but a bare assignment creates no node in the body, so "
            + "there is nothing for the loop to point the variable at afterwards — it still names "
            + "the value the body was working on."
            + "\n"
            + "\nWrap the assignment so the body produces it:"
            + "\n"
            + "\n    foreach (var ctx in LoopAPI.Iterate(trips))"
            + "\n    {"
            + "\n        sum  = sum + prev;"
            + "\n        prev = LoopAPI.Carry(acc);   // not `prev = acc;`"
            + "\n        acc  = acc + Scalar(1.0f);"
            + "\n    }"
            + "\n"
            + "\nReading the lagged value only inside the body needs no wrapping.";

        internal const string BodyValueReadAfterLoopGuidance =
            "a value produced inside a loop body is used after the loop but is not one of the "
            + "loop's outputs."
            + "\n"
            + "\nThe body assigns it before ever reading it, so Shorokoo cannot recover the value "
            + "the loop should return when it runs zero times. Read it once in the body before the "
            + "first assignment — LoopAPI.Init(x) does exactly that.";

        internal const string IterationIndexReadAfterLoopGuidance =
            "a loop's ctx.IterationIndex was used after the loop."
            + "\n"
            + "\nThe iteration index is the loop's own per-iteration counter and has no value once "
            + "the loop has exited. Carry the value you want out of the body instead:"
            + "\n"
            + "\n    var last = Scalar(-1L);"
            + "\n    foreach (var ctx in LoopAPI.Iterate(trips))"
            + "\n    {"
            + "\n        LoopAPI.Init(last);"
            + "\n        last = ctx.IterationIndex;"
            + "\n    }";

        private const string OuterScanOfUncarriedNestedValueGuidance =
            "an enclosing loop's ctx.Scan was given a value produced inside a nested loop that the "
            + "nested loop does not carry out."
            + "\n"
            + "\nThe enclosing loop records once per one of ITS iterations, so it needs the value "
            + "the nested loop ends with. A body-local the nested loop assigns before ever reading "
            + "does not survive that loop, so there is nothing to record."
            + "\n"
            + "\nDeclare it in the nested body before the first assignment:"
            + "\n"
            + "\n    foreach (var ctx1 in LoopAPI.Iterate(innerTrips))"
            + "\n    {"
            + "\n        LoopAPI.Init(t);"
            + "\n        t = acc * Scalar(2.0f);"
            + "\n        scanned = ctx0.Scan(t);"
            + "\n    }"
            + "\n"
            + "\nOr scan on the nested loop's own context instead.";

        private const string LagChainTooDeepGuidance =
            "a loop body read a variable holding the value another held more than one iteration "
            + "ago."
            + "\n"
            + "\nA carry trailing another by a single iteration is identified; a longer chain — "
            + "prev2 = prev; prev = acc; — is not, and would read its pre-loop value on every "
            + "iteration."
            + "\n"
            + "\nGive each step of the chain a body node of its own:"
            + "\n"
            + "\n    sum   = sum + prev2;"
            + "\n    prev2 = LoopAPI.Carry(prev);"
            + "\n    prev  = LoopAPI.Carry(acc);"
            + "\n    acc   = acc + Scalar(1.0f);";

        private const string LagCarryInNestedLoopGuidance =
            "a nested loop body assigned a variable the previous iteration's value of one of that "
            + "loop's carries."
            + "\n"
            + "\nA lagged variable shares the body node of the carry it trails, so the enclosing "
            + "loop has nothing to carry it out by, and would reset it at every one of its own "
            + "iterations."
            + "\n"
            + "\nWrap the assignment so the lagged value gets a body node of its own:"
            + "\n"
            + "\n    prev = LoopAPI.Carry(acc);   // not `prev = acc;`"
            + "\n"
            + "\nLag without wrapping is supported in an outermost loop only.";

        private const string LagCarryOfUncarriedValueGuidance =
            "the loop body assigned a variable the previous iteration's value of another variable "
            + "that the loop does not itself carry."
            + "\n"
            + "\nA lagged variable hands back the value the variable it trails held at the start of "
            + "the iteration, so that variable has to be a loop carry — read in the body before it "
            + "is assigned. Declare it with LoopAPI.Init, or compute the lagged value in the body "
            + "from a carry instead.";

        private const string CarryAssignedFromOutsideGuidance =
            "the loop body assigned a variable declared with LoopAPI.Init a value computed outside "
            + "the loop."
            + "\n"
            + "\nInstead of:"
            + "\n"
            + "\n    var carry = n + Scalar(5L);"
            + "\n    foreach (var ctx in LoopAPI.Iterate(trips))"
            + "\n    {"
            + "\n        LoopAPI.Init(carry);"
            + "\n        carry = n;"
            + "\n    }"
            + "\n    return carry;"
            + "\n"
            + "\nWrite:"
            + "\n"
            + "\n    var carry = n + Scalar(5L);"
            + "\n    foreach (var ctx in LoopAPI.Iterate(trips))"
            + "\n    {"
            + "\n        LoopAPI.Init(carry);"
            + "\n        carry = LoopAPI.Carry(n);"
            + "\n    }"
            + "\n    return carry;"
            + "\n"
            + "\nOr move the assignment out of the loop.";

        public void StartThirdPass()
        {
            if (this.CurrentPass != 2 || this.secondPassRound != 1)
                throw new InvalidTensorOperationException(ErrorCodes.FW020, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start third pass - current pass must be the lag pass");

            this.CurrentPass = 3;
            this.traceRound++;
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

            using var loopNodeBuild = GraphTrace.Loopers.EnterLoopNodeBuild();

            // Debug.Assert(this.IterationIndexLoopVariable is not null);
            // this.IterationIndexLoopVariable.SetThirdPassOutput(IterationIndexLoopVariable.OpenNodeOutput.AssertNotNull());
            // this.thirdPassOutputs[this.IterationIndexLoopVariable.CloseNodeInput.AssertNotNull()] = this.IterationIndexLoopVariable;
            // Debug.Assert(this.thirdPassOutputs.ContainsKey(this.IterationIndexLoopVariable.CloseNodeInput.AssertNotNull()));

            var closeNodeLoopVariables = this.OpenLoopNode.AssertNotNull().Outputs.Skip(2).AssertNotNulls().Select(x => this.openNodeOutputs[x].AssertNotNull()).ToArray();
            var closeNodeLoopInputs = closeNodeLoopVariables.Select(x => x.CloseNodeInput.AssertNotNull()).ToArray();
            Debug.Assert(thirdPassOutputs.Values.Where(x => !x.IsLocalScanVariable && x.OpenNodeInputInitializer is not null).Count()
                            + this.lagCarryCount == closeNodeLoopInputs.Length);

            // A scan bound to a value a NESTED loop's body produces — this loop's ctx.Scan called
            // from inside an inner loop — records the value this body ends the iteration with,
            // which is the inner loop's close-node output. The binding was taken while the inner
            // loop was still on its own first pass, so it names the body value the inner loop
            // later carries out; follow that mapping now the inner loop has closed. A scan of a
            // value this body produces itself has no such mapping and keeps its binding.
            foreach (var scanVariable in thirdPassOutputs.Values.Where(x => x.IsLocalScanVariable))
            {
                if (scanVariable.ScanVariableThirdPassInput is not { } scanned) continue;
                if (!this.thirdPassOutputs.TryGetValue(scanned, out var producer)) continue;

                if (producer.InnerLoopCloseNodeOutput is { } innerClose)
                {
                    scanVariable.RebindLocalScanVariableInput(innerClose);
                    continue;
                }

                // The nested loop discards this value, so it has none at the end of one of this
                // loop's iterations. The binding still names the nested loop's throwaway FIRST
                // pass, which sits inline in this body and recomputes from this loop's open-node
                // outputs — a graph that runs and stacks the wrong numbers.
                if (this.nestedBodyThirdPassOutputs.Contains(scanned))
                    throw new UnsupportedLoopVariableAssignmentException(
                        ErrorCodes.FW050, OuterScanOfUncarriedNestedValueGuidance);
            }

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

            // A lag carry's FirstPassOutput is the trailed carry's, which already appears in this
            // mapping under its own close output; leaving it in would overwrite that entry.
            return this.closeNodeOutputs.Values.Where(x => !x.IsLagCarry)
                        .Select(x => (x.FirstPassOutput, x.CloseNodeOutput.AssertNotNull())).ToImmutableList();
        }

        public void StartFourthPass()
        {
            if (this.CurrentPass != 3)
                throw new InvalidTensorOperationException(ErrorCodes.FW020, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot start fourth pass - current pass must be 3");

            this.CurrentPass = 4;
            this.traceRound++;
        }

        public void Terminate()
        {
            if (this.CurrentPass != 4)
                throw new InvalidTensorOperationException(ErrorCodes.FW021, "Loop Phase Validation", $"current pass: {this.CurrentPass}", "Cannot terminate loop - current pass must be 4");

            this.CurrentPass = 5;

            foreach (var loopVariable in this.loopVariableByNodeOutputLocation.Values.Where(x => x.OpenNodeInputInitializer is null))
            {
                var invalid = loopVariable.InvalidFourthPassOutput.AssertNotNull();
                invalid.IsValid = false;

                // Two of these are not what the generic answer (LoopAPI.Init) addresses, and the
                // enclosing loop cannot tell them apart from an ordinary output-only body value
                // without asking what produced them: a nested loop's scan output, which has no
                // pre-loop value to declare, and the iteration index, which is scoped to one
                // iteration by construction.
                invalid.InvalidReason = loopVariable.FirstPassOutput.OwningNode.OpCode switch
                {
                    OpCodes.LOOP_SCAN_VARIABLE => InnerScanReadAfterEnclosingLoopGuidance,
                    OpCodes.LOOP_INDEX_VARIABLE => IterationIndexReadAfterLoopGuidance,
                    _ => BodyValueReadAfterLoopGuidance,
                };
            }

            // A lag carry's own value is handed back by the close node, but the user's local does
            // not name it: a bare assignment produces no body node, so the fourth pass has nothing
            // to remap and the local still names the trailed carry's body value. That value must
            // not leave the loop — the same limitation LoopAPI.Carry exists for, and the same
            // remedy.
            foreach (var lagVariable in this.loopVariableByNodeInputLocation.Values.Where(x => x.IsLagCarry).Distinct())
            {
                var trailedBodyValue = lagVariable.TrailedCarry.AssertNotNull().ThirdPassOutput;
                if (trailedBodyValue is not null)
                {
                    trailedBodyValue.IsValid = false;
                    trailedBodyValue.InvalidReason = LagCarryReadAfterLoopGuidance;
                }
            }
        }

        /// <summary>
        /// Translates a variable observed inside this loop's body (during the third pass) to
        /// its post-loop value, for trace-time recorders that defer an in-body registration
        /// to after the loop (in-loop <c>Globals.StateUpdate</c>). Only a carried loop
        /// variable has a well-defined final value — its close-node output, which falls back
        /// to the initializer when the loop runs zero iterations; every other body value is
        /// reported by kind rather than guessed at. Callable once the close node is built
        /// (from the fourth pass on).
        /// </summary>
        internal (PostLoopTranslation kind, Variable? postLoop) TranslateToPostLoop(Variable inBody)
        {
            var loopVariable =
                this.thirdPassOutputs.TryGetValue(inBody, out var asThirdPass) ? asThirdPass :
                this.innerLoopCloseNodeOutputs.TryGetValue(inBody, out var asInnerClose) ? asInnerClose :
                null;

            if (loopVariable is null)
            {
                // Values the loop machinery scopes to a single iteration have no post-loop
                // meaning: an open-node output is the value at the START of an iteration
                // (the close output is the value after the last body execution — a different
                // thing), and the raw open-node outputs cover the iteration index.
                if (this.openNodeOutputs.ContainsKey(inBody) ||
                    (this.OpenLoopNode is not null &&
                     this.OpenLoopNode.Outputs.Any(x => Object.ReferenceEquals(x, inBody))))
                    return (PostLoopTranslation.LoopScoped, null);
                return (PostLoopTranslation.External, inBody);
            }

            if (loopVariable.IsLocalScanVariable)
                return (PostLoopTranslation.ScanVariable, null);
            if (loopVariable.CloseNodeOutput is null)
                return (PostLoopTranslation.NotALoopOutput, null);
            return (PostLoopTranslation.Translated, loopVariable.CloseNodeOutput);
        }
    }

    /// <summary>How <see cref="Looper.TranslateToPostLoop"/> resolved an in-body variable.</summary>
    internal enum PostLoopTranslation
    {
        /// <summary>Not produced inside this loop's body — usable after the loop as-is.</summary>
        External,
        /// <summary>A carried loop variable's body value — translated to its close-node output.</summary>
        Translated,
        /// <summary>A body value that never surfaces as a loop output (its fourth-pass output is invalidated).</summary>
        NotALoopOutput,
        /// <summary>A scanned value — its close output is the stacked per-iteration tensor, not a final value.</summary>
        ScanVariable,
        /// <summary>Loop machinery scoped to one iteration (open-node output / iteration index).</summary>
        LoopScoped,
    }

    /// <summary>
    /// The loop-tracing state of one trace: the in-progress <see cref="Looper"/>s, outermost
    /// first, plus the pass queries derived from them. Everything that understands loop
    /// passes lives here, next to the machinery that drives them; the ambient trace merely
    /// carries an instance (see <see cref="Shorokoo.Core.GraphTrace"/>).
    /// </summary>
    internal sealed class LooperStack : IReadOnlyList<Looper>
    {
        private readonly List<Looper> _loopers = new List<Looper>();

        public int Count => _loopers.Count;
        public Looper this[int index] => _loopers[index];
        public IEnumerator<Looper> GetEnumerator() => _loopers.GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

        internal void Add(Looper looper) => _loopers.Add(looper);
        internal void RemoveAt(int index) => _loopers.RemoveAt(index);

        /// <summary>Whether the trace is currently inside a <c>LoopAPI.Iterate</c> body.</summary>
        internal bool InLoopBody => _loopers.Count > 0;

        /// <summary>
        /// The looper whose pass currently drives node interception, and its stack index:
        /// the innermost looper that has started tracing (inner loops sit on the stack at
        /// pass 0 while an outer loop is mid-pass). Null when no loop is active.
        /// </summary>
        internal (Looper looper, int index)? Active
        {
            get
            {
                if (_loopers.Count == 0) return null;
                var activeIndex = _loopers.FindIndex(x => x.CurrentPass == 0) - 1;
                if (activeIndex < 0) activeIndex = _loopers.Count - 1;
                return (_loopers[activeIndex], activeIndex);
            }
        }

        /// <summary>
        /// Whether trace-time actions that must fire <b>exactly once per source occurrence</b>
        /// (e.g. <see cref="Shorokoo.Rng.Pin(object[])"/>) should record right now. A loop body
        /// is executed once per construction pass — first (track), second (identify loop vars),
        /// third (build the real body), fourth (expose outputs) — so a naive record inside a
        /// loop body would fire up to four times, three of them against throwaway nodes. Only
        /// the pass that builds the surviving body nodes (the active looper's third pass) is
        /// canonical; at module level (no active loop) recording is always canonical. This is
        /// THE canonical-pass query: every recorder asks here, rather than re-deriving the
        /// active-looper selection.
        /// </summary>
        internal bool InCanonicalRecordingScope
            => Active is not { } active || active.looper.CurrentPass == 3;

        /// <summary>
        /// The loopers whose bodies enclose the current recording point, outermost first:
        /// the active looper and its ancestors. A snapshot for deferred in-loop recorders
        /// (in-loop <c>Globals.StateUpdate</c>) — the <see cref="Looper"/> objects outlive
        /// their stack entries, so the chain stays walkable at loop termination. Empty when
        /// no loop is active.
        /// </summary>
        internal ImmutableList<Looper> ActiveChain
            => Active is { } active
                ? _loopers.Take(active.index + 1).ToImmutableList()
                : ImmutableList<Looper>.Empty;

        private int _loopNodeBuildDepth;

        /// <summary>
        /// Whether the loop's own construction is building nodes right now, rather than a body
        /// being traced. Such a node belongs to no body: it exists once, on the pass that builds
        /// it, so recording it would desync the pass-to-pass body comparison
        /// (<c>Looper.checkMatch</c>) — most visibly the continue-condition constant
        /// <see cref="Looper.BuildLoopCloseNode"/> creates on the third pass alone.
        ///
        /// <para>The loop index variable is machinery too, but it is a body value by
        /// construction — every pass makes one, at the same position — so it is created outside
        /// this scope and tracked like any other body node.</para>
        /// </summary>
        internal bool BuildingLoopNodes => _loopNodeBuildDepth > 0;

        /// <summary>Marks its lifetime as <see cref="BuildingLoopNodes"/>.</summary>
        internal LoopNodeBuildScope EnterLoopNodeBuild() => new(this);

        internal readonly struct LoopNodeBuildScope : IDisposable
        {
            private readonly LooperStack _stack;
            internal LoopNodeBuildScope(LooperStack stack)
            {
                _stack = stack;
                stack._loopNodeBuildDepth++;
            }
            public void Dispose() => _stack._loopNodeBuildDepth--;
        }

        /// <summary>Iteration-index variables of all in-progress loops, outermost first.</summary>
        internal ImmutableList<Scalar<int64>> IterationIndices
            => _loopers.Select(x => x.GetLoopIndexVariable()).ToImmutableList();
    }

    public static class LoopAPI
    {
        // All trace-time loop state lives in the ambient trace's LooperStack, reached
        // through GraphTrace: the graph builder enters a (module-build) trace per body
        // trace, and a standalone Iterate outside any build enters an isolated trace for
        // the duration of the iteration (see LoopFull).

        /// <summary>
        /// Gets the interation indices of all outerloop ordered from outermost to innerermost.
        /// </summary>
        internal static ImmutableList<Scalar<int64>> IterationIndices
            => GraphTrace.IsTracing
                ? GraphTrace.Loopers.IterationIndices
                : ImmutableList<Scalar<int64>>.Empty;

        internal static (FullInputs newInputs, FullOutputs newOutputs) ProcessNode(Node node)
        {
            // No trace in progress, or no active loop: node creation passes through untouched.
            if (!GraphTrace.IsTracing)
                return (node.FullInputs, node.FullOutputs);
            var looperStack = GraphTrace.Loopers;
            if (looperStack.Active is not { } active)
                return (node.FullInputs, node.FullOutputs);

            var (activeLooper, activeLooperIndex) = active;

            // Nodes involved in the creation of the loop, these, in principle are not part of the loop body.
            if (node.NodeDef.FullNodeOpName == OpCodes.LOOP ||
                node.NodeDef.FullNodeOpName == OpCodes.LOOP_FAKE_INPUT)
                return (node.FullInputs, node.FullOutputs);

            // Nodes the loop's own construction creates are not body nodes. This used to be
            // approximated by "no inputs, therefore not part of the body", which held for the
            // constants that motivated it but not in general: a draw the body itself created has
            // no inputs either, and went untracked, so its consumers resolved through the
            // outer-scope case to the first pass's node — emitted before LOOP_OPEN, leaving every
            // iteration reading the one draw (Shorokoo/Shorokoo#262). Ask which code is building
            // instead of guessing from the node's shape.
            if (looperStack.BuildingLoopNodes)
                return (node.FullInputs, node.FullOutputs);

            // A module input marker names a graph input of the function being built, and a
            // function's inputs are created before its body runs — so no looper of its own trace
            // can be active for one. Reaching here means an ENCLOSING trace's looper is, i.e. the
            // build was not shielded and is about to be recorded as that caller's loop body. The
            // symptom is remote from the cause (a pass-to-pass mismatch four passes later, on the
            // first call only), so say it here instead.
            Debug.Assert(!Shorokoo.Core.Factory.FastOpsetResolver.IsModelInputOpCode(node.OpCode),
                "LoopAPI.ProcessNode: a module input marker reached an enclosing loop body — the " +
                "function build that created it was not shielded by GraphTrace.EnterModuleBuild " +
                "or EnterIsolated.");

            FullInputs retvalInputs = node.FullInputs;
            FullOutputs retvalOutputs = node.FullOutputs;

            if (activeLooper.CurrentPass == 1 && activeLooperIndex > 0)
            {
                // We process an inner loop's pass 1 at the same time as its outer loop's pass 3.
                // This enables the inner loop to capture the outer loop's OpenNode outputs used as inputs to nodes
                // part of the inner loop's body.
                Debug.Assert(looperStack.Take(activeLooperIndex).All(x => x.CurrentPass == 3));
                var outerLooper = looperStack[activeLooperIndex - 1];
                Debug.Assert(outerLooper.CurrentPass == 3);
                (retvalInputs, retvalOutputs) = outerLooper.ProcessNode(
                    node, retvalInputs, retvalOutputs, fromNestedLoopBody: true);
            }

            // The active loop should be on pass 1, 2 or 4
            // For pass 3: Because, inner loops get activated while their outer loops is on pass 3,
            // a loop is only activated during pass 3 for nodes that do not belong to inner-more loops.
            Debug.Assert(activeLooper.CurrentPass == 1 ||
                        activeLooper.CurrentPass == 2 ||
                        (activeLooper.CurrentPass == 3 && (activeLooperIndex == looperStack.Count - 1)) ||
                        activeLooper.CurrentPass == 4);

            // If the current node is in the scope of inner loops of the active loop, then these
            // inner loops should be on pass 0.
            Debug.Assert(looperStack.Count <= activeLooperIndex + 1 ||
                            looperStack.Skip(activeLooperIndex + 1).All(x => x.CurrentPass == 0));

            // If the active loop is inside the scope of outer loops, then these outer loops
            // should on pass 3.
            Debug.Assert(activeLooperIndex == 0 ||
                        looperStack.Take(activeLooperIndex).All(x => x.CurrentPass == 3));


            (retvalInputs, retvalOutputs) = activeLooper.ProcessNode(node, retvalInputs, retvalOutputs);

            return (retvalInputs, retvalOutputs);
        }

        private static IEnumerable<(Action<Scalar<bit>, Scalar<bit>> breakWhen, Looper looper, Scalar<int64> iterationIndex)> LoopFull(Scalar<int64>? maxNumIterations)
        {
            // A loop traced inside a module build records into that build's trace. A standalone
            // trace (hand-built graphs, e.g. InternalComputationGraph construction in tests) gets its
            // own isolated trace for the duration of the iteration — disposed when the loop
            // completes (or its enumerator is abandoned) — so the looper state never outlives
            // the loop and never bleeds into an unrelated ambient trace.
            using var standalone = GraphTrace.EnterIsolatedIfNone();
            var looperStack = GraphTrace.Loopers;

            var looper = new Looper(looperStack.Count);
            looperStack.Add(looper);

            try
            {
                // Do nothing if there is an outer loop that is in its first or second pass.
                if (looper.LoopDepth != 0 && (looperStack[looper.LoopDepth - 1].CurrentPass < 3 || looperStack[looper.LoopDepth - 1].CurrentPass == 4))
                {
                    yield return ((x, y) => { }, looper, looper.GetLoopIndexVariable());
                }
                else
                {
                    // First pass, track the nodes that are part of the body of the loop.
                    looper.StartFirstPass();
                    looper.SetMaxNumIterations(maxNumIterations);
                    yield return ((x, y) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(looperStack.Count == looper.LoopDepth + 1);

                    // Second pass, identify the loop variables
                    looper.StartSecondPass();
                    yield return ((x, y) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(looperStack.Count == looper.LoopDepth + 1);

                    // Lag pass, identify the carries that trail another carry by one iteration.
                    looper.StartLagPass();
                    yield return ((x, y) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(looperStack.Count == looper.LoopDepth + 1);

                    looper.BuildLoopOpenNode();

                    // Third pass, build the actual loop body
                    looper.StartThirdPass();
                    yield return (looper.ContinueWhile, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(looperStack.Count == looper.LoopDepth + 1);

                    var outerLoopMappings = looper.BuildLoopCloseNode();
                    if (looper.LoopDepth > 0)
                        looperStack[looper.LoopDepth - 1].MapInnerLoopCloseNodeOutputsToOuterLoopThirdPassOutputs(outerLoopMappings);

                    // Fourth pass, make loop output variables available to the caller.
                    looper.StartFourthPass();
                    yield return ((x, y) => { }, looper, looper.GetLoopIndexVariable());
                    Debug.Assert(looperStack.Count == looper.LoopDepth + 1);

                    looper.Terminate();
                }

                Debug.Assert(looperStack.Count == looper.LoopDepth + 1);
                Debug.Assert(looperStack[looper.LoopDepth] == looper);
            }
            finally
            {
                // Always remove the looper from the stack, even if an exception occurred.
                // (The standalone scope's `using` disposes after this finally, so the
                // looper is gone before its trace exits.)
                if (looperStack.Count > looper.LoopDepth && looperStack[looper.LoopDepth] == looper)
                {
                    looperStack.RemoveAt(looper.LoopDepth);
                }
            }

            // An in-loop Globals.StateUpdate defers its registration to here — it is sugar
            // for registering the post-loop value of the updated tensor. Resolution needs
            // the OUTERMOST loop fully closed (the close-node outputs it translates to must
            // exist at every nesting level) and its looper off the stack (so the deferred
            // STATE_UPDATE_LINK nodes are created at module scope instead of being
            // intercepted as loop-body nodes). Nothing can be pending outside a module
            // build: recording is gated on one.
            if (looper.LoopDepth == 0 && GraphTrace.IsModuleBuildTracing)
                GraphTrace.StateUpdates.ResolvePendingInLoop();
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
            {
                // The declaration belongs to the innermost loop enclosing this call — the deepest
                // looper on the stack, which is not necessarily the *active* one. An enclosing loop
                // traces the inner loop's body inline during its own first two passes (the inner
                // looper is still on pass 0 then), and the variable it sees at this site changes
                // between those passes because the inner loop rebinds it, which is not a user
                // reassignment. Marking only the innermost looper keeps the declaration where the
                // user wrote it, and keeps this per-trace rather than process-wide.
                var loopers = GraphTrace.Loopers;
                var declaring = loopers.Count > 0 ? loopers[loopers.Count - 1] : null;
                if (declaring is not null) declaring.NextNodeIsInitDeclaration = true;
                OnnxOp.Identity(toInit, toInit.Rank);
                if (declaring is not null) declaring.NextNodeIsInitDeclaration = false;
            }
        }

        /// <summary>
        /// Marks <paramref name="value"/> as a loop carry's new value when that value was computed
        /// <em>outside</em> the loop body. A bare assignment of such a value creates no node in the
        /// body, and the loop's result is delivered by re-tracing the node that produced the body's
        /// value and remapping its output — so there is nothing to remap, and the assigned variable
        /// would keep referring to the outside value after the loop. This wraps it in a body-local
        /// node so the carry behaves like any other:
        /// <code>
        /// foreach (var ctx in LoopAPI.Iterate(trips))
        /// {
        ///     LoopAPI.Init(carry);
        ///     carry = LoopAPI.Carry(n);   // not `carry = n`
        /// }
        /// </code>
        /// A value the body already computes needs no wrapping.
        /// </summary>
        /// <param name="value">The value to carry, computed outside the loop body.</param>
        public static Scalar<T> Carry<T>(Scalar<T> value) where T : IVarType
            => (Scalar<T>)OnnxOp.Identity(value, rank: 0);

        /// <inheritdoc cref="Carry{T}(Scalar{T})"/>
        public static Vector<T> Carry<T>(Vector<T> value) where T : IVarType
            => (Vector<T>)OnnxOp.Identity(value, rank: 1);

        /// <inheritdoc cref="Carry{T}(Scalar{T})"/>
        public static Tensor<T> Carry<T>(Tensor<T> value) where T : IVarType
            => (Tensor<T>)OnnxOp.Identity(value, rank: null);

        public static IEnumerable<IterationContext> Iterate(Scalar<int64> maxNumIterations)
        {
            foreach ((var continueWhile, var scanner, var iterationIndex) in LoopFull(maxNumIterations))
                yield return new IterationContext(continueWhile, scanner, iterationIndex);
        }
    }

    public class IterationContext
    {
        private Looper looper;
        private Action<Scalar<bit>, Scalar<bit>> continueWhile;

        public Scalar<int64> IterationIndex { get; private set; }

        internal IterationContext(Action<Scalar<bit>, Scalar<bit>> continueWhile, Looper looper, Scalar<int64> iterationIndex)
        {
            this.continueWhile = continueWhile;
            this.looper = looper;
            this.IterationIndex = iterationIndex;
        }

        public Tensor<T> Scan<T>(Tensor<T> tensor) where T : IVarType => looper.Scan(tensor);
        public Tensor<T> Scan<T>(Vector<T> v) where T : IVarType => looper.Scan((Tensor<T>)v);

        public Vector<T> Scan<T>(Scalar<T> scalar) where T : IVarType => looper.Scan(scalar);

        public void Break(Scalar<bit> exitLoopWhenTrue) => Continue(!exitLoopWhenTrue);

        public void ContinueWhile(Scalar<bit> exitLoopWhenFalse) => Continue(exitLoopWhenFalse);

        /// <summary>
        /// Hands the looper the condition twice: once wrapped in an Identity, so the read sits at
        /// a node input ProcessNode can rewrite, and once as written, for the cases where that
        /// rewrite is not the one wanted. The wrap runs on every pass, so the four traces stay
        /// aligned; the looper binds one of the two and the Identity is dead either way.
        /// </summary>
        private void Continue(Scalar<bit> exitLoopWhenFalse)
            => continueWhile((Scalar<bit>)OnnxOp.Identity(exitLoopWhenFalse, rank: 0), exitLoopWhenFalse);
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

        /// <summary>The value read here on the lag pass — the second pass traced once more.</summary>
        public Variable? LagPassInput { get; private set; }

        public LoopVariableOutput? LoopVariableConnection { get; private set; }

        public void SetSecondPassInput(Variable input)
        {
            Debug.Assert(this.SecondPassInput is null);
            this.SecondPassInput = input;
        }

        public void SetLagPassInput(Variable input)
        {
            Debug.Assert(this.LagPassInput is null);
            this.LagPassInput = input;
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

        /// <summary>
        /// A carry that trails another carry by one iteration. It owns no body node output of its
        /// own: its value at the end of each iteration is the trailed carry's value at the
        /// <em>start</em> of that iteration, which is that carry's open-node output.
        /// </summary>
        public static LoopVariable Lag(
            Variable initializer, LoopVariable trailed, Variable trailedFirstPassOutput,
            List<(int NodeIndex, int InputIndex)> inputKeys)
            => new LoopVariable(initializer, trailedFirstPassOutput, trailed.SecondPassZombie.AssertNotNull(),
                                isLocalScanVariable: false, inputKeys, trailed.OutputKey)
            { IsLagCarry = true, TrailedCarry = trailed };

        /// <summary>The carry this one trails, for a lag carry; null otherwise. See <see cref="Lag"/>.</summary>
        public LoopVariable? TrailedCarry { get; private init; }

        /// <summary>See <see cref="Lag"/>.</summary>
        public bool IsLagCarry { get; private init; }

        /// <summary>The trailed carry's open-node output, bound once the open node exists.</summary>
        public Variable? LagSourceOpenNodeOutput { get; private set; }

        public void BindLagSource()
        {
            Debug.Assert(this.IsLagCarry && this.LagSourceOpenNodeOutput is null);
            this.LagSourceOpenNodeOutput = this.TrailedCarry.AssertNotNull().OpenNodeOutput.AssertNotNull();
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
        public Variable? CloseNodeInput => LagSourceOpenNodeOutput ?? InnerLoopCloseNodeOutput ?? this.ScanVariableThirdPassInput ?? ThirdPassOutput;

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

        public void SetLocalScanVariableInput(Variable toScan)
        {
            Debug.Assert(this.IsLocalScanVariable && this.ScanVariableThirdPassInput is null);
            this.ScanVariableThirdPassInput = toScan;
        }

        /// <summary>
        /// Replaces the bound scan input once a nested loop has closed and its close-node output
        /// is known. See <see cref="Looper.BuildLoopCloseNode"/>.
        /// </summary>
        public void RebindLocalScanVariableInput(Variable toScan)
        {
            Debug.Assert(this.IsLocalScanVariable && this.ScanVariableThirdPassInput is not null);
            this.ScanVariableThirdPassInput = toScan;
        }

        public void SetInvalidFourthPassOutput(Variable fourthPassOutput)
        {
            Debug.Assert(this.InvalidFourthPassOutput is null);
            this.InvalidFourthPassOutput = fourthPassOutput;
        }
    }

}