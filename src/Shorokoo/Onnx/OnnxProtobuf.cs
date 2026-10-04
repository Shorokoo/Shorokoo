using System.IO;
using ProtoBuf.Meta;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// How a model read in from outside is parsed: through a protobuf type model of its own, under
    /// which a message may nest at most <see cref="MaxDepth"/> levels below the model.
    ///
    /// <para>protobuf-net's default type model allows 512 levels and refuses the 512th by throwing,
    /// at a depth where its readers have used up nearly all of a thread's stack, so unwinding that
    /// exception overflows what is left and the process dies with it. A file of a few kilobytes is
    /// enough. Under this one the refusal comes while there is stack to spare: a hundred is
    /// protobuf's own default recursion limit, under which ONNX's reference implementation parses a
    /// model, so no model it reads nests any deeper.</para>
    /// </summary>
    internal static class OnnxProtobuf
    {
        /// <summary>The level below the model at which a message is refused, as protobuf-net
        /// counts it: the model's own fields are one level down, so 99 levels are read.</summary>
        internal const int MaxDepth = 100;

        private static readonly RuntimeTypeModel _model = CreateModel();

        private static RuntimeTypeModel CreateModel()
        {
            var model = RuntimeTypeModel.Create("Shorokoo.Onnx");
            model.MaxDepth = MaxDepth;
            return model;
        }

        /// <summary>The model in <paramref name="stream"/>. One nested <see cref="MaxDepth"/> levels
        /// deep or more throws <see cref="System.InvalidOperationException"/>; a truncated or
        /// otherwise malformed one throws as protobuf-net refuses it.</summary>
        internal static ModelProto ReadModel(Stream stream) => _model.Deserialize<ModelProto>(stream);
    }
}
