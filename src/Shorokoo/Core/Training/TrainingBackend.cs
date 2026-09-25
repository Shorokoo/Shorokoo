namespace Shorokoo
{
    /// <summary>
    /// The formats a training step is handed to an execution backend in. A backend says which it
    /// runs through <see cref="Shorokoo.Core.Backends.IShorokooBackend.AcceptsTrainingFormat"/>;
    /// every backend runs <see cref="Onnx"/>.
    /// </summary>
    public static class TrainingFormats
    {
        /// <summary>
        /// A training step whose gradient Shorokoo has already computed: ordinary ONNX, the forward
        /// pass, the backward pass and the optimizer update all written out as operators. Every
        /// backend runs it.
        /// </summary>
        public const string Onnx = "onnx";

        /// <summary>
        /// A training step whose gradient is left to the execution backend: ONNX in which one node,
        /// <c>ai.shorokoo.training::AutoGrad</c> (version 1), takes the loss followed by the tensors
        /// to differentiate it with respect to, and returns one gradient per such tensor. Only a
        /// backend that computes gradients itself runs it.
        /// </summary>
        public const string OnnxAutoGrad = "onnx-autograd/1";

        /// <summary>The operator domain of the gradient node in an <see cref="OnnxAutoGrad"/>
        /// step, imported by the model at <see cref="AutoGradDomainVersion"/>.</summary>
        public const string AutoGradDomain = "ai.shorokoo.training";

        /// <summary>The version of <see cref="AutoGradDomain"/> an <see cref="OnnxAutoGrad"/> step
        /// imports.</summary>
        public const int AutoGradDomainVersion = 1;

        /// <summary>The <c>op_type</c> of the gradient node in an <see cref="OnnxAutoGrad"/> step:
        /// inputs <c>[loss, wrt_0, …, wrt_n-1]</c>, outputs <c>[grad_0, …, grad_n-1]</c>, where
        /// <c>grad_i</c> is the gradient of the sum of <c>loss</c> with respect to <c>wrt_i</c>,
        /// shaped and typed as <c>wrt_i</c>, and zero where <c>loss</c> does not depend on it. It has
        /// no attributes and no subgraph: the forward pass it differentiates is the loss's ancestry
        /// in the graph around it.</summary>
        public const string AutoGradOpType = "AutoGrad";

        /// <summary>The model metadata key an <see cref="OnnxAutoGrad"/> step records its format
        /// under, with the format itself as the value.</summary>
        public const string MetadataKey = "shrk_training_format";
    }

    /// <summary>
    /// Who computes the gradient of a <see cref="TrainingRig"/>'s training step.
    /// <see cref="Shorokoo"/>, the default, differentiates the step itself and hands every backend
    /// ordinary ONNX. <see cref="Native"/> leaves the gradient to the execution backend of the
    /// rig's <see cref="TrainingRig.RuntimeContext"/>, which must accept
    /// <see cref="TrainingFormats.OnnxAutoGrad"/>. Everything else about the step — the model, the
    /// loss, the optimizer, schedules, hyperparameters and random draws — is Shorokoo's either way.
    /// </summary>
    public sealed class TrainingBackend
    {
        /// <summary>Shorokoo's own automatic differentiation: the step every backend runs
        /// (<see cref="TrainingFormats.Onnx"/>).</summary>
        public static TrainingBackend Shorokoo { get; } = new("Shorokoo", TrainingFormats.Onnx);

        /// <summary>The gradient is computed by the runtime context's execution backend
        /// (<see cref="TrainingFormats.OnnxAutoGrad"/>).</summary>
        public static TrainingBackend Native { get; } = new("Native", TrainingFormats.OnnxAutoGrad);

        /// <summary>The backend's name, as <see cref="ToString"/> shows it.</summary>
        public string Name { get; }

        /// <summary>The format the rig's training step is handed to the execution backend in; one
        /// of <see cref="TrainingFormats"/>.</summary>
        public string Format { get; }

        /// <summary>Whether Shorokoo lowers the step's gradient itself, which is the default path
        /// exactly as it has always been.</summary>
        internal bool LowersAutoGrad => Format == TrainingFormats.Onnx;

        private TrainingBackend(string name, string format)
        {
            Name = name;
            Format = format;
        }

        /// <summary>The name and the format, e.g. <c>Native (onnx-autograd/1)</c>.</summary>
        public override string ToString() => $"{Name} ({Format})";
    }
}
