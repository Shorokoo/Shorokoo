using System.Reflection;
using Shorokoo.Core.Factory.CSharpFactory;
using Shorokoo.Runtime;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.Processors.Helpers;

namespace Shorokoo.Tests.Utils
{
    public static class AutoTest
    {
        private static IValue[] tupleToArray(object tuple)
        {
            return [..tuple.GetType().GetFields()
                        .Where(f => f.Name.StartsWith("Item"))
                        .OrderBy(f => f.Name)
                                .Select(f => f.GetValue(tuple))
                                .Cast<IValue>(),
                    ..tuple.GetType().GetFields()
                        .Where(f => f.Name == "Rest")
                        .Select(f => f.GetValue(tuple))
                        .NotNulls()
                        .SelectMany(tupleToArray)];
        }

        public static bool TestGraph(FastComputationGraph graph, ComputeContext? context = null, bool testOnnxRoundtrip = true, bool testCsRoundtrip = true, TensorData[]? sampleInputs = null, bool testQuickEngineExecution = false)
        {

            byte[][] originalResults;
            byte[][]? onnxResults = null;
            byte[][]? csResults = null;

            context ??= ComputeContext.Default;
            var inputData = (IData[])(sampleInputs ?? Array.Empty<TensorData>());
            var resultA = context.Execute(graph, inputData);
            var originalTensorData = resultA.Select(x => x.ToTensorData()).ToArray();
            originalResults = originalTensorData.Select(td => td.AccessRawMemory().ToArray()).ToArray();

            // Convention: a graph whose sole output is a Scalar<bit> is treated as a
            // self-checking computation — the bit must be 1 (true). Lets module-shaped
            // coverage tests embed their result validation inside the module's Inline
            // method and keep the xUnit test as a one-liner.
            if (originalTensorData.Length == 1
                && originalTensorData[0].DType == DType.Bool
                && originalTensorData[0].Shape.Dims.Length == 0
                && originalResults[0].Length > 0
                && originalResults[0][0] == 0)
                return false;

            if (testOnnxRoundtrip)
            {
                var data = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
                var onnxRoundtrip = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);
                var resultB = context.Execute(onnxRoundtrip, inputData);
                onnxResults = resultB.Select(x => x.ToTensorData().AccessRawMemory().ToArray()).ToArray();
            }

            // CS roundtrip relies on a no-input C# lambda; skip the compile + execute when
            // the graph takes runtime inputs since BuildLambda<TResult> doesn't surface a
            // way to supply them. Codegen (BuildFullGraph) is always safe to run — it just
            // produces text — so always exercise it to cover the per-op MakeXxx handlers.
            if (testCsRoundtrip)
                new CSharpModelBuilder().BuildFullGraph(graph, "testModel");

            if (testCsRoundtrip && graph.Inputs.Count == 0)
            {
                var csharpLambda = new CSharpModelBuilder().BuildLambda<object>(graph, "testModel");
                var csharpResults = csharpLambda();

                IValue[] csharpOutputs =
                                (csharpResults is IValue singleOut) ? [singleOut] :
                                (csharpResults is IValue[] arrayOut) ? arrayOut :
                                tupleToArray(csharpResults); // Treat it as a tuple.

                var csRoundtrip = new FastComputationGraph([], [.. csharpOutputs.Select(o => o.ToVariable())]);
                var resultC = context.Execute(csRoundtrip);
                csResults = resultC.Select(x => x.ToTensorData().AccessRawMemory().ToArray()).ToArray();
            }
            else
            {
                testCsRoundtrip = false;
            }

            var targetNum = originalResults.Length;

            var goodOnnx = !testOnnxRoundtrip || (onnxResults is not null && onnxResults.Length == targetNum && onnxResults.Zip(originalResults).Count(x => x.First.SequenceEqual(x.Second)) == targetNum);
            var goodCs = !testCsRoundtrip || (csResults is not null && csResults.Length == targetNum && csResults.Zip(originalResults).Count(x => x.First.SequenceEqual(x.Second)) == targetNum);

            if (!goodOnnx || !goodCs)
                return false;

            if (testQuickEngineExecution && !RunQuickEngineExecution(graph, sampleInputs))
                return false;

            return true;
        }

        /// <summary>
        /// Runs <see cref="QuickExecutionEngine"/> on the graph with the given sample inputs and
        /// asserts every declared output has been resolved to a non-Invalid <see cref="DType"/>.
        /// Used as an extra validation layer alongside the ONNX/CS roundtrips so coverage tests
        /// exercise the QEE op path on the same module graphs the ONNX path runs.
        /// </summary>
        private static bool RunQuickEngineExecution(FastComputationGraph graph, TensorData[]? sampleInputs)
        {
            var qee = new QuickExecutionEngine();
            var store = sampleInputs is null
                ? qee.Run(graph)
                : qee.Run(graph, sampleInputs);

            foreach (var outKey in graph.Outputs)
            {
                if (!store.TryGetValue(outKey, out var rt) || rt.DType == DType.Invalid)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// One-liner-friendly entry point for module-graph tests. Reflects on
        /// <typeparamref name="TModule"/> for its source-generated <c>ComputationGraph</c>
        /// property, lowers the module graph to a concrete model using
        /// <paramref name="hyperparamInputs"/> + <paramref name="runtimeInputs"/> as input
        /// hints, then runs the resulting graph through <see cref="TestGraph"/> with the
        /// same inputs supplied at execution time.
        ///
        /// The split between hyperparam and runtime inputs is for the caller's benefit —
        /// after <c>ToConcreteArchitecture</c>, hyperparam inputs remain as ordinary graph
        /// inputs and are supplied alongside runtime inputs at execution time. Both are
        /// needed as architecture-time hints so trainable params whose shapes derive from
        /// either set of inputs can be materialized.
        /// </summary>
        public static bool AdvancedTestGraph<TModule>(
            TensorData[] hyperparamInputs,
            TensorData[] runtimeInputs,
            ComputeContext? context = null,
            bool testOnnxRoundtrip = true,
            bool testCsRoundtrip = true,
            bool testQuickEngineExecution = true,
            Dictionary<string, DType>? genericTypes = null)
        {
            var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException(
                    $"{typeof(TModule).FullName} has no public static ComputationGraph property");
            var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;

            return AdvancedTestGraph(moduleGraph, hyperparamInputs, runtimeInputs,
                context, testOnnxRoundtrip, testCsRoundtrip, testQuickEngineExecution, genericTypes);
        }

        /// <summary>
        /// Graph-first overload of <see cref="AdvancedTestGraph{TModule}"/> for module graphs
        /// that don't come from a source-generated static <c>ComputationGraph</c> property —
        /// e.g. codegen-free modules built via <see cref="Shorokoo.Modules.ModuleFactory"/>.
        /// </summary>
        public static bool AdvancedTestGraph(
            FastComputationGraph moduleGraph,
            TensorData[] hyperparamInputs,
            TensorData[] runtimeInputs,
            ComputeContext? context = null,
            bool testOnnxRoundtrip = true,
            bool testCsRoundtrip = true,
            bool testQuickEngineExecution = true,
            Dictionary<string, DType>? genericTypes = null)
        {
            // Generic-method modules build their ComputationGraph with IGenericType placeholder
            // DTypes + leading GENERIC_TYPE_INPUT inputs. Apply the caller-supplied type
            // specialization (if any) via FastChangeGenericTypeSpecialization, then concretize
            // via FastToConcreteDataType — the latter removes the generic input slots and strips
            // the param-name tags from DType attributes.
            if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
            {
                if (genericTypes is not null && genericTypes.Count > 0)
                    Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
                moduleGraph = Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType.Process(moduleGraph);
            }

            var allInputs = new TensorData[hyperparamInputs.Length + runtimeInputs.Length];
            Array.Copy(hyperparamInputs, 0, allInputs, 0, hyperparamInputs.Length);
            Array.Copy(runtimeInputs, 0, allInputs, hyperparamInputs.Length, runtimeInputs.Length);

            var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));
            var concreteModel = concreteArch.ToConcreteModel();

            return TestGraph(
                concreteModel,
                context: context,
                testOnnxRoundtrip: testOnnxRoundtrip,
                testCsRoundtrip: testCsRoundtrip,
                sampleInputs: allInputs,
                testQuickEngineExecution: testQuickEngineExecution);
        }

        /// <summary>
        /// Variant of <see cref="AdvancedTestGraph{TModule}"/> that ALSO roundtrips the
        /// raw pre-concretization moduleGraph AND the post-architecture
        /// (pre-materialization) concreteArch through ONNX save/load before final
        /// concretization. Exercises load-time paths that the concrete-model roundtrip
        /// inside TestGraph can't reach:
        /// <list type="bullet">
        ///   <item>moduleGraph roundtrip — load-time MODEL_INVOKE / SequenceConstruct
        ///         / FunctionProto reconstruction.</item>
        ///   <item>concreteArch roundtrip — load-time TRAINABLE_PARAM reconstruction
        ///         (<c>BuildFastTrainableParamNodeFromProto</c>) because at this
        ///         stage trainable params still carry their initializer-fn
        ///         TargetFunction (not yet materialized as constants).</item>
        /// </list>
        /// </summary>
        public static bool AdvancedTestGraphWithModuleGraphRoundtrip<TModule>(
            TensorData[] hyperparamInputs,
            TensorData[] runtimeInputs,
            ComputeContext? context = null,
            bool testOnnxRoundtrip = true,
            bool testCsRoundtrip = true,
            bool testQuickEngineExecution = true,
            Dictionary<string, DType>? genericTypes = null)
        {
            var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException(
                    $"{typeof(TModule).FullName} has no public static ComputationGraph property");
            var moduleGraph = (FastComputationGraph)prop.GetValue(null)!;

            if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
            {
                if (genericTypes is not null && genericTypes.Count > 0)
                    Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
                moduleGraph = Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType.Process(moduleGraph);
            }

            var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
            moduleGraph = CompressedFormatUtils.LoadFastGraphFromBinary(data, isCompressed: true);

            var allInputs = new TensorData[hyperparamInputs.Length + runtimeInputs.Length];
            Array.Copy(hyperparamInputs, 0, allInputs, 0, hyperparamInputs.Length);
            Array.Copy(runtimeInputs, 0, allInputs, hyperparamInputs.Length, runtimeInputs.Length);

            var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));

            // concreteArch roundtrip: at this stage trainable params still carry their
            // initializer-fn TargetFunction (the FastConvertTrainableParamIdRefToTrainableParam
            // pass has run but ToConcreteModel hasn't materialized them as constants yet).
            // Saving these triggers FastOpsetResolver's isParamInitializerFn branch which
            // rewrites the opcode to the initializer-fn name; on reload, the function-name
            // opcode dispatches into BuildFastTrainableParamNodeFromProto.
            var archData = CompressedFormatUtils.SaveFastGraphToBinary(concreteArch, compressed: true);
            concreteArch = CompressedFormatUtils.LoadFastGraphFromBinary(archData, isCompressed: true);

            var concreteModel = concreteArch.ToConcreteModel();

            return TestGraph(
                concreteModel,
                context: context,
                testOnnxRoundtrip: testOnnxRoundtrip,
                testCsRoundtrip: testCsRoundtrip,
                sampleInputs: allInputs,
                testQuickEngineExecution: testQuickEngineExecution);
        }

    }
}
