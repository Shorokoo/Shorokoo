using System;
using System.Reflection;
using Shorokoo.Core;
using Shorokoo.Graph;

namespace Shorokoo.Modules
{
    /// <summary>
    /// Codegen-free entry point for defining Shorokoo modules from plain delegates — the
    /// functional equivalent of what the <c>[Module]</c> source generator emits for a partial
    /// class with a static <c>Inline</c> method.
    ///
    /// <para><b>Usage.</b> Write the module body as a static method (or non-capturing
    /// <c>static</c> lambda) whose parameters are the flattened hyperparameter/input tensors —
    /// exactly the shape of a <c>[Module]</c> class's <c>Inline</c> method — then:</para>
    /// <code>
    /// static Tensor&lt;float32&gt; Body(Tensor&lt;float32&gt; x)
    ///     =&gt; x * InitSimple.Init(x.ShapeTensor());
    ///
    /// var module = ModuleFactory.FromFunc&lt;Tensor&lt;float32&gt;, Tensor&lt;float32&gt;&gt;(Body);
    /// var model  = module.SetHyperparams();        // codegen's Foo.Model()
    /// var y      = model.Call(x);                  // codegen's Foo.Call(x)
    /// var graph  = ModuleFactory.ComputationGraph( // codegen's Foo.ComputationGraph
    ///                 (Func&lt;Tensor&lt;float32&gt;, Tensor&lt;float32&gt;&gt;)Body);
    /// </code>
    ///
    /// <para><b>Hyperparameters.</b> Mark the leading parameters with <c>[Hyper]</c> (on the
    /// static method, or on an explicitly-typed lambda's parameters) and use
    /// <see cref="FromFuncWithHypers{THyper, TIn, TOut}"/> and friends; bind them with
    /// <c>module.SetHyperparams(...)</c>.</para>
    ///
    /// <para><b>Constraint.</b> The delegate must be a static method group or a non-capturing
    /// (<c>static</c>) lambda. The body is invoked once by reflection to trace the graph and the
    /// resulting <see cref="Function"/> is cached by the body's <see cref="MethodInfo"/>, so
    /// captured locals would be baked invisibly into the graph and shared across closures.
    /// Inside the body, everything available to <c>[Module]</c> <c>Inline</c> methods works
    /// unchanged: <c>[TrainableParamInitializer]</c> classes /
    /// <see cref="Globals.CallTrainableParamInitializer(Delegate, Variable[])"/>,
    /// <see cref="Globals.StateUpdate{T}(T, T)"/>, <c>LoopAPI.Iterate</c>, and calls into other
    /// modules.</para>
    /// </summary>
    public static class ModuleFactory
    {
        // ───────────────────────────── no hyperparameters ─────────────────────────────

        /// <summary>Creates a module with no runtime inputs from a delegate body.</summary>
        /// <param name="fn">Static method group or non-capturing lambda producing the output.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static CallbackModule<TOut> FromFunc<TOut>(Func<TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 0);
            return new CallbackModule<TOut>(fn, referenceMethod: null, name: name);
        }

        /// <summary>Creates a single-input module from a delegate body.</summary>
        /// <param name="fn">Static method group or non-capturing lambda mapping the input to the output.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<TIn, TOut> FromFunc<TIn, TOut>(Func<TIn, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 0);
            return new Module<TIn, TOut>(fn, referenceMethod: null, name: name);
        }

        /// <summary>
        /// Creates a two-input module from a delegate body with flattened parameters.
        /// The returned module's input type is the tuple <c>(T1, T2)</c>; bind a
        /// <c>Model&lt;T1, T2, TOut&gt;</c> via <c>SetHyperparams&lt;...&gt;()</c> for a
        /// two-argument <c>Call</c>.
        /// </summary>
        /// <param name="fn">Static method group or non-capturing lambda mapping the inputs to the output.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<(T1, T2), TOut> FromFunc<T1, T2, TOut>(Func<T1, T2, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 0);
            return new Module<(T1, T2), TOut>(ins => fn(ins.Item1, ins.Item2), fn, name);
        }

        /// <summary>Creates a three-input module from a delegate body with flattened parameters.</summary>
        /// <param name="fn">Static method group or non-capturing lambda mapping the inputs to the output.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<(T1, T2, T3), TOut> FromFunc<T1, T2, T3, TOut>(Func<T1, T2, T3, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 0);
            return new Module<(T1, T2, T3), TOut>(ins => fn(ins.Item1, ins.Item2, ins.Item3), fn, name);
        }

        /// <summary>Creates a four-input module from a delegate body with flattened parameters.</summary>
        /// <param name="fn">Static method group or non-capturing lambda mapping the inputs to the output.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<(T1, T2, T3, T4), TOut> FromFunc<T1, T2, T3, T4, TOut>(Func<T1, T2, T3, T4, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 0);
            return new Module<(T1, T2, T3, T4), TOut>(ins => fn(ins.Item1, ins.Item2, ins.Item3, ins.Item4), fn, name);
        }

        // ──────────────── with hyperparameters ([Hyper]-marked trailing params) ────────────────
        //
        // Convention: the leading delegate parameter is the single runtime input and the trailing
        // parameters are hyperparameters (annotated [Hyper], exactly like a [Module] Inline
        // method, which is inputs-first / hyperparameters-last). For hyperparameters combined with
        // MULTIPLE runtime inputs, construct the Module<THypers, TInputs, TOutputs> base directly
        // with a wrapper lambda, e.g.
        //   new Module<Scalar<float32>, (Variable, Tensor<float32>), Tensor<float32>>(
        //       (h, ins) => Body(ins.Item1, ins.Item2, h), Body);

        /// <summary>
        /// Creates a module with one runtime input and one hyperparameter. The delegate's last
        /// parameter must be annotated <c>[Hyper]</c> (on the static method, or on an
        /// explicitly-typed lambda parameter). Bind it with <c>module.SetHyperparams(h)</c>.
        /// </summary>
        /// <param name="fn">Static method group or non-capturing lambda: <c>(input, [Hyper] hyper) =&gt; output</c>.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<THyper, TIn, TOut> FromFuncWithHypers<TIn, THyper, TOut>(Func<TIn, THyper, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 1);
            return new Module<THyper, TIn, TOut>((h, x) => fn(x, h), fn, name);
        }

        /// <summary>
        /// Creates a module with one runtime input and two hyperparameters. The delegate's last
        /// two parameters must be annotated <c>[Hyper]</c>. Bind them with
        /// <c>module.SetHyperparams((h1, h2))</c>.
        /// </summary>
        /// <param name="fn">Static method group or non-capturing lambda: <c>(input, [Hyper] h1, [Hyper] h2) =&gt; output</c>.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<(TH1, TH2), TIn, TOut> FromFuncWithHypers<TIn, TH1, TH2, TOut>(Func<TIn, TH1, TH2, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 2);
            return new Module<(TH1, TH2), TIn, TOut>((h, x) => fn(x, h.Item1, h.Item2), fn, name);
        }

        /// <summary>
        /// Creates a module with one runtime input and three hyperparameters. The delegate's last
        /// three parameters must be annotated <c>[Hyper]</c>. Bind them with
        /// <c>module.SetHyperparams((h1, h2, h3))</c>.
        /// </summary>
        /// <param name="fn">Static method group or non-capturing lambda: <c>(input, [Hyper] h1, [Hyper] h2, [Hyper] h3) =&gt; output</c>.</param>
        /// <param name="name">Optional module name (defaults to the body's declaring class name).</param>
        public static Module<(TH1, TH2, TH3), TIn, TOut> FromFuncWithHypers<TIn, TH1, TH2, TH3, TOut>(Func<TIn, TH1, TH2, TH3, TOut> fn, string? name = null)
        {
            ValidateModuleDelegate(fn, hyperCount: 3);
            return new Module<(TH1, TH2, TH3), TIn, TOut>((h, x) => fn(x, h.Item1, h.Item2, h.Item3), fn, name);
        }

        // ───────────────────────────── computation graph ─────────────────────────────

        /// <summary>
        /// Returns the module body's <see cref="FastComputationGraph"/> — the codegen-free
        /// equivalent of the source-generated static <c>ComputationGraph</c> property, suitable
        /// for <c>TrainingRig.FromScratch</c>, ONNX export, and the concretization pipeline.
        /// Like the generated property, the underlying build is cached (per body
        /// <see cref="MethodInfo"/>) and a fresh clone is returned on every call, so callers may
        /// freely mutate the result.
        /// </summary>
        /// <param name="fn">The module body (static method group or non-capturing lambda, flattened parameters).</param>
        /// <param name="name">Optional module name used if the body wasn't already built/cached.</param>
        public static FastComputationGraph ComputationGraph(Delegate fn, string? name = null)
        {
            if (fn is null)
                throw new ArgumentNullException(nameof(fn));
            EnsureFlattenedParameters(fn);

            // CreateTargetFunction validates the non-capturing constraint, builds the body graph
            // through the same GraphBuilder machinery the source generator uses, and caches the
            // resulting Function by the body's MethodInfo (shared with FromFunc-created modules).
            var function = ModuleHelper.CreateTargetFunction(fn, defaultName: name);
            return function.OriginalFastGraph.Clone();
        }

        // ───────────────────────────── validation ─────────────────────────────

        /// <summary>
        /// Validates a module-body delegate: non-capturing, flattened (tuple-free) parameters,
        /// and a [Hyper]-annotation layout matching the factory overload's hyperparameter count.
        /// </summary>
        private static void ValidateModuleDelegate(Delegate fn, int hyperCount)
        {
            if (fn is null)
                throw new ArgumentNullException(nameof(fn));

            ModuleHelper.EnsureNonCapturingDelegate(fn.Target, fn.Method.Name);
            EnsureFlattenedParameters(fn);

            var parameters = fn.Method.GetParameters();
            var firstHyperIndex = parameters.Length - hyperCount;
            for (int i = 0; i < parameters.Length; i++)
            {
                var isHyper = parameters[i].GetCustomAttribute<HyperAttribute>() is not null;

                if (i >= firstHyperIndex && !isHyper)
                    throw new ArgumentException(
                        $"Parameter '{parameters[i].Name}' must be annotated [Hyper]: this FromFuncWithHypers overload " +
                        $"treats the last {hyperCount} parameter(s) as hyperparameters, and the graph builder reads the " +
                        "[Hyper] attribute off the delegate's parameters to distinguish SetHyperparams-bound inputs from " +
                        "Call-time inputs. Annotate the static method's parameter, or use an explicitly-typed lambda: " +
                        "static (Tensor<float32> x, [Hyper] Scalar<float32> h) => ...",
                        nameof(fn));

                if (i < firstHyperIndex && isHyper)
                    throw new ArgumentException(
                        $"Parameter '{parameters[i].Name}' is annotated [Hyper], but this factory overload declares only " +
                        $"the last {hyperCount} parameter(s) as hyperparameters. Hyperparameters must come last; use a " +
                        "FromFuncWithHypers overload whose hyperparameter count matches the [Hyper] annotations.",
                        nameof(fn));
            }
        }

        /// <summary>
        /// Rejects tuple-typed delegate parameters with a targeted message — module bodies take
        /// flattened parameters (one per tensor), mirroring <c>[Module]</c> <c>Inline</c> methods.
        /// </summary>
        private static void EnsureFlattenedParameters(Delegate fn)
        {
            foreach (var parameter in fn.Method.GetParameters())
            {
                if (ModuleHelper.IsValueTuple(parameter.ParameterType))
                    throw new ArgumentException(
                        $"Delegate parameter '{parameter.Name}' is a tuple. Module bodies take flattened parameters " +
                        "(one per tensor), like a [Module] Inline method; use the multi-parameter FromFunc / " +
                        "FromFuncWithHypers overloads instead of grouping inputs into a tuple.",
                        nameof(fn));
            }
        }
    }
}
