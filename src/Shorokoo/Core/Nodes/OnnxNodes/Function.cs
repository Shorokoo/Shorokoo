using Shorokoo;
using Shorokoo.Modules;
using Microsoft.VisualBasic;
using Shorokoo.Core.Factory;
using Shorokoo.Core.Factory.IR;
using Shorokoo.Onnx;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;

namespace Shorokoo.Core.Nodes.OnnxNodes
{
    public enum FunctionType
    {
        Function,
        Module,
        TrainableParamInitializer,
        StateParamInitializer,
        ModuleSignature
    }


    // This is one of the more problematic nodes.
    //
    // Pre opset 18, the squeezed indices are a hardcoded attribute.
    // Start opset 18, the squeezed indices are a tensor that could come from anywhere.
    // public class FunctionNode : Node
    // {
    //     //public override string OpCode => "#invalid#";
    // 
    //     public FunctionType FunctionType => TargetFunction.FunctionType;
    // 
    //     public Function TargetFunction { get; private set; }
    // 
    //     public FunctionNode(Function function, NodeDefinition nodeDef, OnnxCSharpAttributes? attributes, ImmutableDictionary<string, Variable?[]> inputs, ImmutableDictionary<string, OutputTensorInfo[]> outputs, string? stackTrace) :
    //             base(nodeDef, attributes, inputs, outputs, stackTrace, function)
    //     {
    //         this.TargetFunction = function;
    //     }
    // }

    //public class ModuleNode : Node
    //{
    //    public Shorokoo.Framework.BaseBestModule? TargetModule { get; private set; }
    //    public Function TargetFunction { get; private set; }
    //
    //    public ModuleNode(Shorokoo.Framework.BaseBestModule? module, OnnxFunction targetFunction, NodeDefinition nodeDef, OnnxCSharpAttributes? attributes, ImmutableDictionary<string, Variable?[]> inputs, ImmutableDictionary<string, OutputTensorInfo[]> outputs, string? stackTrace) :
    //            base(nodeDef, attributes, inputs, outputs, stackTrace)
    //    {
    //        this.TargetModule = module;
    //        this.TargetFunction = targetFunction;
    //    }
    //
    //    public override NodeInfo[] ForOpsetMany(VirtualGraph graph, OpSetVersion opSet, ErrorReports errors)
    //    {
    //        var vfn = graph.GetVFunction(this.TargetFunction);
    //        OnnxCSharpAttributes toUseAttrs = this.Attributes;
    //        if (this.Attributes.AttributeDefs.Any(x => x.AttributeName == OnnxOpAttributeNames.InternalAttrFunctionName))
    //        {
    //            var functionName = vfn.Name;
    //            var functionDomainName = vfn.Domain;
    //            toUseAttrs = toUseAttrs.SetAttributes(
    //                    (OnnxOpAttributeNames.InternalAttrFunctionName, functionName),
    //                    (OnnxOpAttributeNames.InternalAttrDomainName, functionDomainName));
    //        }
    //
    //        if (this.Attributes.AttributeDefs.Any(x => x.AttributeName == OnnxOpAttributeNames.InternalAttrModuleLocalId))
    //        {
    //            var localId = graph.GetVTensor(this.Outputs[0]).ModuleId;
    //            toUseAttrs = toUseAttrs.SetAttributes(
    //                    (OnnxOpAttributeNames.InternalAttrModuleLocalId, localId));
    //        }
    //
    //        return [new NodeInfo(graph, this.Inputs, this.Outputs, toUseAttrs.ToProto(), this.StackTrace, opSet, OpCode, this.NodeDef.Domain)];
    //    }
    //}
}
