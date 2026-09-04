using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System.Diagnostics;
using static Shorokoo.Globals;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Training;

namespace Shorokoo.Modules
{
    /// <summary>
    /// Marks a partial class whose static <c>Inline</c> method defines a module body. The source
    /// generator emits the module plumbing from it (e.g. <c>Model()</c>, <c>Call</c>, and the
    /// static <c>ComputationGraph</c> property).
    /// </summary>
    public class ModuleAttribute : Attribute { }

    /// <summary>
    /// Marks a static partial class whose static <c>Inline</c> method initializes a trainable
    /// parameter (typically shape-only: <c>Inline(Vector&lt;int64&gt; shape)</c>). The source
    /// generator wires it through <c>Globals.CallTrainableParamInitializer</c>.
    /// </summary>
    public class TrainableParamInitializerAttribute : Attribute { }

    /// <summary>
    /// State ownership types for state initializers.
    /// </summary>
    public enum StateOwnership
    {
        /// <summary>
        /// State that is updated by the module's own logic during forward passes.
        /// </summary>
        ModuleOwned,

        /// <summary>
        /// State that is updated by an external optimizer.
        /// </summary>
        OptimizerOwned
    }

    /// <summary>
    /// Marks a static partial class as a state initializer with specified ownership.
    /// The class must contain a public static Inline method.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class StateInitializerAttribute : Attribute
    {
        /// <summary>Who updates the state: the module's own logic or an external optimizer.</summary>
        public StateOwnership Ownership { get; set; } = StateOwnership.ModuleOwned;
    }

    /// <summary>
    /// Marks an <c>Inline</c> parameter as a hyperparameter (a scalar configured when the module is
    /// instantiated, not a runtime input). An <c>Inline</c> method declares its tensor inputs first and
    /// its hyperparameters last, so every <c>[Hyper]</c> parameter must come after all input parameters;
    /// a non-hyper parameter following a <c>[Hyper]</c> one is a malformed signature: the source
    /// generator reports warning MSG002 and generates no <c>Model</c>/<c>Call</c> for the method.
    /// (The generated <c>Call</c> shortcut takes its arguments the other way round — hyperparameters
    /// first, then inputs.)
    ///
    /// For optimizer modules, an optional <see cref="DefaultValue"/> supplies the default used by the
    /// source-generated strongly-typed hyperparameter set (e.g. <c>AdamWOptimizerHyperparameters</c>),
    /// keeping the default next to the declaration as the single source of truth.
    ///
    /// <para>The parameter's own <c>Scalar&lt;T&gt;</c> declaration is the source of truth for the
    /// hyperparameter's dtype; the default is formatted at that dtype by the source generator. Attribute
    /// arguments are compile-time constants and a single <see cref="float"/> parameter cannot carry an
    /// <see cref="int"/>, <see cref="double"/> or <see cref="bool"/>, hence one constructor per host
    /// literal type. A dtype with no natural C# literal (e.g. <c>float16</c>) takes no default — declare
    /// it as a bare <c>[Hyper]</c> and bind it explicitly.</para>
    /// </summary>
    public class HyperAttribute : Attribute
    {
        /// <summary>Declares a hyperparameter with no default value.</summary>
        public HyperAttribute() { }

        /// <summary>Declares a hyperparameter defaulting to the given floating-point value.</summary>
        public HyperAttribute(float defaultValue) : this((object)defaultValue) { }

        /// <summary>Declares a hyperparameter defaulting to the given double-precision value.</summary>
        public HyperAttribute(double defaultValue) : this((object)defaultValue) { }

        /// <summary>Declares a hyperparameter defaulting to the given integer value.</summary>
        public HyperAttribute(int defaultValue) : this((object)defaultValue) { }

        /// <summary>Declares a hyperparameter defaulting to the given 64-bit integer value.</summary>
        public HyperAttribute(long defaultValue) : this((object)defaultValue) { }

        /// <summary>Declares a hyperparameter defaulting to the given boolean value.</summary>
        public HyperAttribute(bool defaultValue) : this((object)defaultValue) { }

        private HyperAttribute(object defaultValue)
        {
            DefaultValue = defaultValue;
            HasDefault = true;
        }

        /// <summary>The default value, boxed at the host literal type the constructor took.</summary>
        public object? DefaultValue { get; }

        /// <summary>Whether a default value was supplied.</summary>
        public bool HasDefault { get; }
    }
}
