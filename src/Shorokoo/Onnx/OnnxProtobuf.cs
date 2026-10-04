using System;
using System.IO;
using ProtoBuf;
using ProtoBuf.Meta;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// How a model read in from outside is parsed: through a protobuf type model of its own, under
    /// which a message may nest at most <see cref="MaxDepth"/> levels below the model.
    ///
    /// <para>protobuf-net's default type model allows 512 levels and refuses the 512th by throwing,
    /// at a depth where its readers have used up nearly all of a thread's default stack, so
    /// unwinding that exception overflows what is left and the process dies with it. A file of a
    /// few kilobytes is enough. Under this one the refusal comes while there is stack to spare: a
    /// hundred is protobuf's own default recursion limit, under which ONNX's reference
    /// implementation parses a model, so no model it reads nests any deeper.</para>
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

        /// <summary>
        /// The model in <paramref name="stream"/>. Bytes protobuf-net refuses throw
        /// <see cref="ProtoException"/>, as every malformed protobuf does — a model nested
        /// <see cref="MaxDepth"/> levels deep or more among them, which protobuf-net itself refuses
        /// with an <see cref="InvalidOperationException"/> no caller would take for a malformed
        /// file — or <see cref="EndOfStreamException"/> where they end early.
        /// </summary>
        internal static ModelProto ReadModel(Stream stream)
        {
            try
            {
                return _model.Deserialize<ModelProto>(stream);
            }
            catch (InvalidOperationException e) when (e is not ObjectDisposedException)
            {
                throw new ProtoException($"The model's protobuf could not be read: {e.Message}", e);
            }
        }
    }
}
