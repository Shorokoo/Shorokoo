using System;
using System.Collections.Generic;
using System.Text;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Graph;
using Shorokoo.Modules;
using Shorokoo.Onnx;

namespace Shorokoo.Core
{
    internal static class InternalExtensions
    {
        public static void ThrowIsNumLike<T>(this Tensor<T> toCheck) where T : IVarType
        {
            if (!typeof(NumLike).IsAssignableFrom(typeof(T)))
                throw new InvalidTensorOperationException(ErrorCodes.FW008, "ThrowIsNumLike", typeof(T).Name, 
                    $"Type '{typeof(T).Name}' is not assignable from NumLike");
        }
    }
}
