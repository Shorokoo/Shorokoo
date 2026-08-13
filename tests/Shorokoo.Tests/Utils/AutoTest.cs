using System.Reflection;
using System.Runtime.InteropServices;
using Shorokoo.Core.Inference.Abstractions;
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

        /// <summary>Value cap for the QEE pass — see <see cref="RunQuickEngineExecution"/>. Sized to
        /// clear coverage-module intermediates, not to be unbounded.</summary>
        private const int QeeValueCap = 65536;

        /// <summary>Default tolerance for <c>expected</c>, applied as
        /// <c>|actual - want| &lt;= Tolerance * max(1, |want|)</c> — absolute near zero, relative once
        /// the magnitude grows, so one number covers both a 0.5 activation and a 1e30 bound.</summary>
        public const double Tolerance = 1e-5;

        /// <summary>
        /// Widens a result tensor's elements to double so one <c>expected</c> array can cover
        /// outputs of any numeric dtype. Bools read as 1/0.
        /// </summary>
        private static double[] Flatten(TensorData td)
        {
            var raw = td.AccessRawMemory();
            var dt = td.DType;
            if (dt == DType.Float32) return [.. MemoryMarshal.Cast<byte, float>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.Float64) return [.. MemoryMarshal.Cast<byte, double>(raw).ToArray()];
            if (dt == DType.Int64) return [.. MemoryMarshal.Cast<byte, long>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.Int32) return [.. MemoryMarshal.Cast<byte, int>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.Int16) return [.. MemoryMarshal.Cast<byte, short>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.Int8) return [.. MemoryMarshal.Cast<byte, sbyte>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.UInt64) return [.. MemoryMarshal.Cast<byte, ulong>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.UInt32) return [.. MemoryMarshal.Cast<byte, uint>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.UInt16) return [.. MemoryMarshal.Cast<byte, ushort>(raw).ToArray().Select(v => (double)v)];
            if (dt == DType.UInt8 || dt == DType.Bool) return [.. raw.ToArray().Select(v => (double)v)];
            if (dt == DType.Float16) return [.. MemoryMarshal.Cast<byte, Float16>(raw).ToArray().Select(v => (double)(float)v)];
            if (dt == DType.BFloat16) return [.. MemoryMarshal.Cast<byte, BFloat16>(raw).ToArray().Select(v => (double)(float)v)];
            throw new NotSupportedException($"expected-value check does not cover {dt}");
        }

        /// <summary>Readonly-graph entry point: TestGraph never mutates, so it borrows the
        /// wrapped internal graph directly.</summary>
        public static bool TestGraph(ComputationGraph graph, ComputeContext? context = null, bool testOnnxRoundtrip = true, bool testCsRoundtrip = true, TensorData[]? sampleInputs = null, bool testQuickEngineExecution = false, double[]? expected = null, double tolerance = Tolerance)
            => TestGraph(graph.ToInternal(), context, testOnnxRoundtrip, testCsRoundtrip, sampleInputs, testQuickEngineExecution, expected, tolerance);

        public static bool TestGraph(InternalComputationGraph graph, ComputeContext? context = null, bool testOnnxRoundtrip = true, bool testCsRoundtrip = true, TensorData[]? sampleInputs = null, bool testQuickEngineExecution = false, double[]? expected = null, double tolerance = Tolerance)
        {

            byte[][] originalResults;
            byte[][]? onnxResults = null;
            byte[][]? csResults = null;

            context ??= ComputeContext.Default;
            var inputData = (IData[])(sampleInputs ?? Array.Empty<TensorData>());
            var resultA = context.Execute(graph, inputData);
            var originalTensorData = resultA.Select(x => x.ToTensorData()).ToArray();
            originalResults = originalTensorData.Select(td => td.AccessRawMemory().ToArray()).ToArray();

            // Convention: a graph whose sole output is a bool is treated as a self-checking
            // computation — every bit must be true. Lets module-shaped coverage tests embed
            // their result validation inside the module's Inline method and keep the xUnit
            // test as a one-liner. Any shape counts, not just rank 0: a check module miswired
            // to return e.g. a [1]-shaped bool would otherwise degrade silently to
            // roundtrip-only validation and pass without its value check ever running.
            if (originalTensorData.Length == 1
                && originalTensorData[0].DType == DType.Bool
                && originalResults[0].Length > 0
                && Array.IndexOf<byte>(originalResults[0], 0) >= 0)
                return false;

            // The other half of result validation, for modules that return values rather than a
            // verdict bit: without this the roundtrips only agree with each other, so a module
            // computing the wrong answer passes on every engine. Compared NaN-safely — the check
            // is written as !(diff <= tol) so a NaN actual fails instead of slipping through.
            if (expected is not null)
            {
                var actual = originalTensorData.SelectMany(Flatten).ToArray();
                if (actual.Length != expected.Length)
                    return false;
                for (int i = 0; i < actual.Length; i++)
                    if (!(Math.Abs(actual[i] - expected[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i]))))
                        return false;
            }

            if (testOnnxRoundtrip)
            {
                var data = CompressedFormatUtils.SaveFastGraphToBinary(graph, compressed: true);
                var onnxRoundtrip = CompressedFormatUtils.LoadFastGraphFromBinary(data);
                var resultB = context.Execute(onnxRoundtrip.ToInternal(), inputData);
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

                var csRoundtrip = new InternalComputationGraph([], [.. csharpOutputs.Select(o => o.ToVariable())]);
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
        ///
        /// On top of the dtype pass, the self-checking-<c>Scalar</c> convention is enforced here
        /// too: a sole bool output the QEE actually computed must be all-true, exactly as the ORT
        /// path requires. The value check is conditional on the QEE having a value at all, so a
        /// module whose bit it cannot reach still gets the dtype pass rather than a false failure.
        ///
        /// The engine is given a raised <see cref="QuickExecutionEngine.MaxDataElements"/>, because
        /// the default 256 is what silences most of these bits rather than any gap in op coverage:
        /// one broadcast to 512 elements part-way through a module drops that tensor's values, and
        /// every op downstream inherits the loss all the way to the verdict. Coverage-module
        /// intermediates are small, so the cap only has to clear them, not be unbounded.
        /// </summary>
        private static bool RunQuickEngineExecution(InternalComputationGraph graph, TensorData[]? sampleInputs)
        {
            var qee = new QuickExecutionEngine { MaxDataElements = QeeValueCap };
            var store = sampleInputs is null
                ? qee.Run(graph)
                : qee.Run(graph, sampleInputs);

            IRuntimeTensor? soleOutput = null;
            foreach (var outKey in graph.Outputs)
            {
                if (!store.TryGetValue(outKey, out var rt) || rt.DType == DType.Invalid)
                    return false;
                soleOutput = rt;
            }

            if (graph.Outputs.Count == 1
                && soleOutput is RuntimeTensor { BoolData: { } bits } bitOut
                && bitOut.DType == DType.Bool
                && bits.Length > 0
                && bits.IndexOf(false) >= 0)
                return false;

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
            Dictionary<string, DType>? genericTypes = null,
            RngConfig? rngConfig = null,
            double[]? expected = null,
            double tolerance = Tolerance)
        {
            var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException(
                    $"{typeof(TModule).FullName} has no public static ComputationGraph property");
            var moduleGraph = (ComputationGraph)prop.GetValue(null)!;

            return AdvancedTestGraph(moduleGraph, hyperparamInputs, runtimeInputs,
                context, testOnnxRoundtrip, testCsRoundtrip, testQuickEngineExecution, genericTypes, rngConfig,
                expected, tolerance);
        }

        /// <summary>
        /// Graph-first overload of <see cref="AdvancedTestGraph{TModule}"/> for module graphs
        /// that don't come from a source-generated static <c>ComputationGraph</c> property —
        /// e.g. codegen-free modules built via <see cref="Shorokoo.Modules.ModuleFactory"/>.
        /// </summary>
        public static bool AdvancedTestGraph(
            ComputationGraph moduleGraph,
            TensorData[] hyperparamInputs,
            TensorData[] runtimeInputs,
            ComputeContext? context = null,
            bool testOnnxRoundtrip = true,
            bool testCsRoundtrip = true,
            bool testQuickEngineExecution = true,
            Dictionary<string, DType>? genericTypes = null,
            RngConfig? rngConfig = null,
            double[]? expected = null,
            double tolerance = Tolerance)
            // Copy: the generic-type specialization below mutates the module graph in place.
            => AdvancedTestGraph(moduleGraph.ToInternal(), hyperparamInputs, runtimeInputs,
                context, testOnnxRoundtrip, testCsRoundtrip, testQuickEngineExecution, genericTypes, rngConfig,
                expected, tolerance);

        public static bool AdvancedTestGraph(
            InternalComputationGraph moduleGraph,
            TensorData[] hyperparamInputs,
            TensorData[] runtimeInputs,
            ComputeContext? context = null,
            bool testOnnxRoundtrip = true,
            bool testCsRoundtrip = true,
            bool testQuickEngineExecution = true,
            Dictionary<string, DType>? genericTypes = null,
            RngConfig? rngConfig = null,
            double[]? expected = null,
            double tolerance = Tolerance)
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
            // Deterministic per-parameter init (master seed 0) — same as real models. Closed-form
            // reference checks reference the layer's realized weights via IModel.GetTrainableParam
            // rather than re-running an initializer, so they no longer depend on tied init.
            var concreteModel = concreteArch.ToConcreteModel(rngConfig ?? RngConfig.Default);

            return TestGraph(
                concreteModel,
                context: context,
                testOnnxRoundtrip: testOnnxRoundtrip,
                testCsRoundtrip: testCsRoundtrip,
                sampleInputs: allInputs,
                testQuickEngineExecution: testQuickEngineExecution,
                expected: expected,
                tolerance: tolerance);
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
        ///   <item>concreteArch roundtrip — load-time MODEL_PARAM reconstruction
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
            Dictionary<string, DType>? genericTypes = null,
            double[]? expected = null,
            double tolerance = Tolerance)
        {
            var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException(
                    $"{typeof(TModule).FullName} has no public static ComputationGraph property");
            var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();

            if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
            {
                if (genericTypes is not null && genericTypes.Count > 0)
                    Shorokoo.Core.Nodes.Processors.Fast.FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
                moduleGraph = Shorokoo.Core.Nodes.Processors.Fast.FastToConcreteDataType.Process(moduleGraph);
            }

            var data = CompressedFormatUtils.SaveFastGraphToBinary(moduleGraph, compressed: true);
            moduleGraph = CompressedFormatUtils.LoadFastGraphCore(data, "<roundtrip>", null).Graph;

            var allInputs = new TensorData[hyperparamInputs.Length + runtimeInputs.Length];
            Array.Copy(hyperparamInputs, 0, allInputs, 0, hyperparamInputs.Length);
            Array.Copy(runtimeInputs, 0, allInputs, hyperparamInputs.Length, runtimeInputs.Length);

            var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));

            // concreteArch roundtrip: at this stage trainable params still carry their
            // initializer-fn TargetFunction (the FastConvertModelParamIdRefToModelParam
            // pass has run but ToConcreteModel hasn't materialized them as constants yet).
            // Saving these triggers FastOpsetResolver's isParamInitializerFn branch which
            // rewrites the opcode to the initializer-fn name; on reload, the function-name
            // opcode dispatches into BuildFastTrainableParamNodeFromProto.
            var archData = CompressedFormatUtils.SaveFastGraphToBinary(concreteArch, compressed: true);
            concreteArch = CompressedFormatUtils.LoadFastGraphCore(archData, "<roundtrip>", null).Graph;

            // Deterministic per-parameter init (see AdvancedTestGraph).
            var concreteModel = concreteArch.ToConcreteModel(RngConfig.Default);

            return TestGraph(
                concreteModel,
                context: context,
                testOnnxRoundtrip: testOnnxRoundtrip,
                testCsRoundtrip: testCsRoundtrip,
                sampleInputs: allInputs,
                testQuickEngineExecution: testQuickEngineExecution,
                expected: expected,
                tolerance: tolerance);
        }

    }
}
