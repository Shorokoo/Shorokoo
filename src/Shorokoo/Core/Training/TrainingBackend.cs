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
