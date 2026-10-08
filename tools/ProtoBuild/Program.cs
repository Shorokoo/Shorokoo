using System.Reflection;

namespace ProtoBuild
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Dummy Project to build onnx.proto");

            _ = typeof(Onnx.AttributeProto);
        }
    }
}
