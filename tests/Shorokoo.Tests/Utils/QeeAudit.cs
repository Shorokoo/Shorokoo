using System.Reflection;
using Shorokoo.Core.Graph;
using Shorokoo.Core.Inference;
using Shorokoo.Core.Nodes.Processors.Fast;
using Shorokoo.Core.Nodes.Processors.Helpers;
using Shorokoo.Runtime;

namespace Shorokoo.Tests.Utils
{
    /// <summary>How the QEE-only pass judges a lowered module's outputs.</summary>
    public enum QeeStrictness
    {
        /// <summary>No QEE-only pass.</summary>
        None,
        /// <summary>Every output must be a concretely computed rank-0/[1] bool that is true.</summary>
        AllBitsTrue,
        /// <summary>Bool outputs must be a concretely computed true bit; every other output only needs a resolved dtype.</summary>
        BitsTrueRestTyped,
    }

    /// <summary>
    /// Driver for the self-checking QEE audit modules. Lowers a module once
    /// (ComputationGraph → ToConcreteArchitecture → ToConcreteModel) and runs both audit
    /// passes on that single concrete model: the strict QuickExecutionEngine-only
    /// self-check, then the full <c>AutoTest.TestGraph</c> pipeline (ORT execute, ONNX
    /// save/load roundtrip, C# codegen, QEE dtype pass).
    /// </summary>
    public static class QeeAudit
    {
        /// <summary>Strict QEE self-check + the full AutoTest pipeline.</summary>
        public static bool Check<TModule>(params TensorData[] runtimeInputs)
            => CheckWith<TModule>(runtimeInputs);

        /// <summary>Strict QEE self-check only — for graphs ORT cannot run.</summary>
        public static bool QeeOnly<TModule>(params TensorData[] runtimeInputs)
            => CheckWith<TModule>(runtimeInputs, autoTest: false);

        /// <summary>QEE self-check only, allowing non-bool outputs that just need a resolved dtype.</summary>
        public static bool QeeOnlyTyped<TModule>(params TensorData[] runtimeInputs)
            => CheckWith<TModule>(runtimeInputs, autoTest: false, qee: QeeStrictness.BitsTrueRestTyped);

        /// <summary>AutoTest pipeline only — for modules whose bit QEE cannot compute.</summary>
        public static bool OrtOnly<TModule>(params TensorData[] runtimeInputs)
            => CheckWith<TModule>(runtimeInputs, qee: QeeStrictness.None);

        public static bool CheckWith<TModule>(
            TensorData[] runtimeInputs,
            TensorData[]? hyperparamInputs = null,
            bool autoTest = true,
            QeeStrictness qee = QeeStrictness.AllBitsTrue,
            bool testOnnxRoundtrip = true,
            bool testCsRoundtrip = true,
            bool testQuickEngineExecution = true,
            Dictionary<string, DType>? genericTypes = null,
            RngConfig? rngConfig = null)
        {
            TensorData[] allInputs = hyperparamInputs is null || hyperparamInputs.Length == 0
                ? runtimeInputs
                : [.. hyperparamInputs, .. runtimeInputs];
            var concreteModel = Lower<TModule>(allInputs, genericTypes, rngConfig);

            if (qee != QeeStrictness.None && !QeePass(concreteModel, allInputs, qee))
                return false;

            return !autoTest || AutoTest.TestGraph(
                concreteModel,
                testOnnxRoundtrip: testOnnxRoundtrip,
                testCsRoundtrip: testCsRoundtrip,
                sampleInputs: allInputs,
                testQuickEngineExecution: testQuickEngineExecution);
        }

        /// <summary>QEE output tensors in declaration order — for outputs whose shapes are
        /// intentionally unknown (rank-only degradation) or hold strings.</summary>
        public static IRuntimeTensor[] Outputs<TModule>(params TensorData[] runtimeInputs)
        {
            var concreteModel = Lower<TModule>(runtimeInputs, null, null);
            var store = RunQee(concreteModel, runtimeInputs);
            return [.. concreteModel.Outputs.Select(k => store.TryGetValue(k, out var rt)
                ? rt
                : throw new InvalidOperationException($"missing output {k}"))];
        }

        private static InternalComputationGraph Lower<TModule>(
            TensorData[] allInputs, Dictionary<string, DType>? genericTypes, RngConfig? rngConfig)
        {
            var prop = typeof(TModule).GetProperty("ComputationGraph", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException(
                    $"{typeof(TModule).FullName} has no public static ComputationGraph property");
            var moduleGraph = ((ComputationGraph)prop.GetValue(null)!).ToInternal();

            if (moduleGraph.Nodes.Any(n => n.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT))
            {
                if (genericTypes is { Count: > 0 })
                    FastChangeGenericTypeSpecialization.Process(moduleGraph, genericTypes);
                moduleGraph = FastToConcreteDataType.Process(moduleGraph);
            }

            var concreteArch = moduleGraph.ToConcreteArchitecture(moduleGraph.FromOrderedInputs([.. allInputs]));
            return concreteArch.ToConcreteModel(rngConfig ?? RngConfig.Default);
        }

        private static bool QeePass(
            InternalComputationGraph concreteModel, TensorData[] inputs, QeeStrictness strictness)
        {
            var store = RunQee(concreteModel, inputs);
            foreach (var outKey in concreteModel.Outputs)
            {
                if (!store.TryGetValue(outKey, out var rt)) return false;
                if (strictness == QeeStrictness.AllBitsTrue)
                {
                    if (rt is not RuntimeTensor { BoolData: { Length: 1 } bits } plain
                        || plain.DType != DType.Bool || !bits[0])
                        return false;
                }
                else
                {
                    if (rt.DType == DType.Invalid) return false;
                    if (rt is RuntimeTensor bitOut && bitOut.DType == DType.Bool
                        && (bitOut.BoolData is not { Length: 1 } lenient || !lenient[0]))
                        return false;
                }
            }
            return true;
        }

        private static Dictionary<FastTensorKey, IRuntimeTensor> RunQee(
            InternalComputationGraph concreteModel, TensorData[] inputs)
        {
            var qee = new QuickExecutionEngine();
            return inputs.Length == 0 ? qee.Run(concreteModel) : qee.Run(concreteModel, inputs);
        }

        public static TensorData F32(long[] dims, params float[] vals) => Globals.TensorData(dims, vals);
        public static TensorData I64(long[] dims, params long[] vals) => Globals.TensorData(dims, vals);
        public static TensorData I32(long[] dims, params int[] vals) => Globals.TensorData(dims, vals);
        public static TensorData I8(long[] dims, params sbyte[] vals) => Globals.TensorData(dims, vals);
        public static TensorData U8(long[] dims, params byte[] vals) => Globals.TensorData(dims, vals);
        public static TensorData Bits(long[] dims, params bool[] vals) => Globals.TensorData(dims, vals);
        public static TensorData Strs(long[] dims, params string[] vals) => Globals.TensorData(dims, vals);
        public static TensorData F32Zeros(long[] dims) => Globals.TensorDataWithDefaultVals(DType.Float32, dims);
        public static TensorData I8Zeros(long[] dims) => Globals.TensorDataWithDefaultVals(DType.Int8, dims);
    }
}
