using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Core.Factory
{
    /// <summary>
    /// Writes the <c>activation_alpha</c> / <c>activation_beta</c> lists of every <c>RNN</c>,
    /// <c>GRU</c> and <c>LSTM</c> node in full, in a form every reader agrees on.
    ///
    /// <para>ONNX consumes the two lists in the order of the node's activation functions, each
    /// function taking only the values it uses — <c>Affine</c> an alpha and a beta,
    /// <c>LeakyRelu</c> an alpha, <c>Tanh</c> neither — and falling back to the default of the
    /// ONNX operator of its name once a list runs out. ONNX Runtime reads them otherwise. Its
    /// <c>RNN</c> kernel takes <c>alpha[d]</c> and <c>beta[d]</c> for direction <c>d</c>, whatever
    /// the activations, so a bidirectional node whose list is shorter than two — valid whenever
    /// only one of its activations takes that argument — is read past its end, a native assertion
    /// that ends the process. Its <c>GRU</c> and <c>LSTM</c> kernels consume the lists in order but
    /// fall back to 0, not to the operator's default (<c>Affine</c>'s alpha and
    /// <c>ThresholdedRelu</c>'s alpha are 1).</para>
    ///
    /// <para>So the lists are rewritten to what the spec reads out of them: one value per
    /// activation that takes one, the default filled in where the given list ran out, and the
    /// attribute dropped where no activation takes it. An <c>RNN</c> has one activation per
    /// direction, and its lists are laid out one value per direction, so that the position ONNX
    /// Runtime reads and the order ONNX consumes in pick the same value: with at most two
    /// activations the one list can always satisfy both, an activation that takes nothing holding
    /// the value the next one reads in order. Spec-equivalent throughout, so it is applied to every
    /// model that leaves Shorokoo — for a backend or as an exported file.</para>
    ///
    /// <para>A node is left as written where the rewrite cannot know what it means: an
    /// activation ONNX does not name, a list of activations of the wrong length, or an attribute
    /// that refers to a function's attribute.</para>
    /// </summary>
    internal static class RecurrentActivationArguments
    {
        private const string Alpha = "activation_alpha";
        private const string Beta = "activation_beta";

        /// <summary>Whether each activation takes an alpha and a beta, and the defaults of the
        /// ONNX operator of its name. ONNX's <c>ScaledTanh</c> states no default; 1 is used for
        /// both, the identity-like scaling, as the PyTorch backend reads it.</summary>
        private static readonly Dictionary<string, (bool TakesAlpha, bool TakesBeta, float Alpha, float Beta)> Arguments =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ["Relu"] = (false, false, 0f, 0f),
                ["Tanh"] = (false, false, 0f, 0f),
                ["Sigmoid"] = (false, false, 0f, 0f),
                ["Softsign"] = (false, false, 0f, 0f),
                ["Softplus"] = (false, false, 0f, 0f),
                ["Affine"] = (true, true, 1f, 0f),
                ["LeakyRelu"] = (true, false, 0.01f, 0f),
                ["ThresholdedRelu"] = (true, false, 1f, 0f),
                ["ScaledTanh"] = (true, true, 1f, 1f),
                ["HardSigmoid"] = (true, true, 0.2f, 0.5f),
                ["Elu"] = (true, false, 1f, 0f),
            };

        /// <summary>Each operator's activations for one direction when the node names none.</summary>
        private static readonly Dictionary<string, string[]> DefaultActivations = new(StringComparer.Ordinal)
        {
            ["RNN"] = ["Tanh"],
            ["GRU"] = ["Sigmoid", "Tanh"],
            ["LSTM"] = ["Sigmoid", "Tanh", "Tanh"],
        };

        /// <summary>Rewrites every recurrent node of <paramref name="model"/>: the main graph, the
        /// subgraphs nested in it, and every function body.</summary>
        public static void Normalize(ModelProto model)
        {
            if (model.Graph is { } graph)
                FastOnnxModelBuilder.ForEachGraphRecursive(graph, g => g.Nodes.ForEach(Normalize));
            foreach (var function in model.Functions)
            {
                var body = new GraphProto();
                body.Nodes.AddRange(function.Nodes);
                FastOnnxModelBuilder.ForEachGraphRecursive(body, g => g.Nodes.ForEach(Normalize));
            }
        }

        private static void Normalize(NodeProto node)
        {
            if (node.Domain.Length != 0 || !DefaultActivations.TryGetValue(node.OpType, out var perDirection))
                return;
            var alpha = Find(node, Alpha);
            var beta = Find(node, Beta);
            var activations = Find(node, "activations");
            var direction = Find(node, "direction");
            if (alpha is null && beta is null && activations is null) return;
            AttributeProto?[] given = [alpha, beta, activations, direction];
            if (given.Any(a => a is not null && !string.IsNullOrEmpty(a.RefAttrName)))
                return;

            int directions = direction?.S is { } d && Encoding.UTF8.GetString(d) == "bidirectional" ? 2 : 1;
            string[] names = activations is null
                ? [.. Enumerable.Repeat(perDirection, directions).SelectMany(n => n)]
                : [.. activations.Strings.Select(s => Encoding.UTF8.GetString(s))];
            if (names.Length != perDirection.Length * directions || names.Any(n => !Arguments.ContainsKey(n)))
                return;

            var alphas = Consumed(names, alpha?.Floats, n => (Arguments[n].TakesAlpha, Arguments[n].Alpha));
            var betas = Consumed(names, beta?.Floats, n => (Arguments[n].TakesBeta, Arguments[n].Beta));
            bool positional = node.OpType == "RNN";
            Write(node, Alpha, positional ? Positional(alphas) : Ordered(alphas));
            Write(node, Beta, positional ? Positional(betas) : Ordered(betas));
        }

        /// <summary>Per activation, the value ONNX gives it from <paramref name="given"/> read in
        /// order, or null for one that takes none.</summary>
        private static float?[] Consumed(string[] names, float[]? given, Func<string, (bool Takes, float Default)> argument)
        {
            var values = new float?[names.Length];
            int next = 0;
            for (int i = 0; i < names.Length; i++)
            {
                var (takes, fallback) = argument(names[i]);
                if (takes) values[i] = given is not null && next < given.Length ? given[next++] : fallback;
            }
            return values;
        }

        private static float[] Ordered(float?[] values) => [.. values.OfType<float>()];

        /// <summary>One value per activation: its own where it takes one, else the value the
        /// in-order reading takes at that position, so both readings agree (see the class
        /// remarks; an RNN has at most two activations).</summary>
        private static float[] Positional(float?[] values)
        {
            var ordered = Ordered(values);
            if (ordered.Length == 0) return [];
            return [.. values.Select((v, i) => v ?? (i < ordered.Length ? ordered[i] : 0f))];
        }

        private static void Write(NodeProto node, string name, float[] values)
        {
            node.Attributes.RemoveAll(a => a.Name == name);
            if (values.Length > 0)
                node.Attributes.Add(new AttributeProto { Name = name, Type = AttributeProto.AttributeType.Floats, Floats = values });
        }

        private static AttributeProto? Find(NodeProto node, string name)
            => node.Attributes.FirstOrDefault(a => a.Name == name);
    }
}
