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
    /// Marks a partial class whose static <c>Inline</c> method defines a module body that is
    /// compiled <b>statically</b> (read as syntax, never executed) to MLIR-flavored assembly text
    /// and parsed into the graph at runtime. Unlike <c>[Module]</c> — which traces the body — this
    /// lets native C# <c>if</c>/<c>while</c> lower to graph control flow. The body is restricted to
    /// the ModuleV2 subset (see <c>src/docs/design/mlir-assembly-parser.md</c>). The generator emits
    /// a static <c>ComputationGraph</c> property that parses the embedded text.
    /// </summary>
    public class ModuleV2Attribute : Attribute { }

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
    /// instantiated, not a runtime input). Hyperparameters must come before all input parameters.
    ///
    /// For optimizer modules, an optional <see cref="DefaultValue"/> supplies the default used by the
    /// source-generated strongly-typed hyperparameter set (e.g. <c>AdamWOptimizerHyperparameters</c>),
    /// keeping the default next to the declaration as the single source of truth.
    /// </summary>
    public class HyperAttribute : Attribute
    {
        /// <summary>Declares a hyperparameter with no default value.</summary>
        public HyperAttribute() { }

        /// <summary>Declares a hyperparameter with the given default value.</summary>
        public HyperAttribute(float defaultValue)
        {
            DefaultValue = defaultValue;
            HasDefault = true;
        }

        /// <summary>The default value, when one was supplied via <see cref="HyperAttribute(float)"/>.</summary>
        public float DefaultValue { get; }

        /// <summary>Whether a default value was supplied.</summary>
        public bool HasDefault { get; }
    }
}
