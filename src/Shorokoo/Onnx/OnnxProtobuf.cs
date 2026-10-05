using System;
using System.IO;
using ProtoBuf;
using ProtoBuf.Meta;
using Shorokoo.Core.Factory.IR;

namespace Shorokoo.Onnx
{
    /// <summary>
    /// How a model's bytes are parsed, wherever they come from — a file, a caller's buffer, a
    /// runtime's own output: through a protobuf type model of its own, under which a message may
    /// nest at most <see cref="MaxDepth"/> levels below the model.
    ///
    /// <para>protobuf-net's default type model allows 512 levels and refuses the 512th by throwing,
    /// at a depth where its readers have used up nearly all of a thread's default stack, so
    /// unwinding that exception overflows what is left and the process dies with it. A file of a
    /// few kilobytes is enough. Under this one the refusal comes while there is stack to spare: a
    /// hundred levels is protobuf's own default recursion limit, under which ONNX's reference
    /// implementation parses a model, so this reads exactly the models it reads.</para>
    /// </summary>
    internal static class OnnxProtobuf
    {
        /// <summary>The deepest a message is read below the model: the model's own fields are one
        /// level down. protobuf reads a message 100 levels down and refuses one 101 levels down.</summary>
        internal const int MaxDepth = 100;

        private static readonly RuntimeTypeModel _model = CreateModel();

        private static RuntimeTypeModel CreateModel()
        {
            var model = RuntimeTypeModel.Create("Shorokoo.Onnx");
            // protobuf-net counts the model itself as a level, and refuses the level its limit
            // names.
            model.MaxDepth = MaxDepth + 1;
            return model;
        }

        /// <summary>
        /// The model in <paramref name="stream"/>. Bytes protobuf-net refuses throw
        /// <see cref="ProtoException"/>, as every malformed protobuf does — a model nested more than
        /// <see cref="MaxDepth"/> levels deep among them, which protobuf-net itself refuses with an
        /// <see cref="InvalidOperationException"/> no caller would take for a malformed file — or
        /// <see cref="EndOfStreamException"/> where they end early.
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
