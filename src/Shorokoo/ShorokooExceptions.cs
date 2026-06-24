using System;
using System.IO;
using System.Runtime.CompilerServices;
using Shorokoo.Core;

namespace Shorokoo
{
    /// <summary>
    /// Base exception for all Shorokoo-specific errors
    /// </summary>
    public abstract class ShorokooException : Exception
    {
        /// <summary>The stable Shorokoo error code (e.g. "CC002"), also prefixed to the message.</summary>
        public string ErrorCode { get; }
        
        /// <summary>Creates the exception with a stable error code prefixed to the message.</summary>
        protected ShorokooException(string errorCode, string message,
            [CallerFilePath] string callerFilePath = "",
            [CallerMemberName] string callerMemberName = "",
            [CallerLineNumber] int callerLineNumber = 0) : base($"[{errorCode}] {message}")
        {
            ErrorCode = errorCode;
        }
        
        /// <summary>Creates the exception with an error code, message and inner exception.</summary>
        protected ShorokooException(string errorCode, string message, Exception innerException,
            [CallerFilePath] string callerFilePath = "",
            [CallerMemberName] string callerMemberName = "",
            [CallerLineNumber] int callerLineNumber = 0) 
            : base($"[{errorCode}] {message}", innerException)
        {
            ErrorCode = errorCode;
        }
    }

    /// <summary>
    /// Exception thrown when a DType operation is not supported or invalid
    /// </summary>
    public class UnsupportedDTypeException : ShorokooException
    {
        /// <summary>Name of the offending dtype.</summary>
        public string DTypeName { get; }
        /// <summary>The operation that rejected the dtype.</summary>
        public string Operation { get; }

        /// <summary>Creates the exception for <paramref name="dTypeName"/> rejected by <paramref name="operation"/>.</summary>
        public UnsupportedDTypeException(string errorCode, string dTypeName, string operation, string additionalContext = "",
            [CallerFilePath] string callerFilePath = "",
            [CallerMemberName] string callerMemberName = "",
            [CallerLineNumber] int callerLineNumber = 0)
            : base(errorCode, $"DType '{dTypeName}' is not supported for operation '{operation}'. {additionalContext}".Trim(),
                   callerFilePath, callerMemberName, callerLineNumber)
        {
            DTypeName = dTypeName;
            Operation = operation;
        }
    }

    /// <summary>
    /// Exception thrown when tensor operations are invalid
    /// </summary>
    public class InvalidTensorOperationException : ShorokooException
    {
        /// <summary>Description of the tensor(s) involved.</summary>
        public string TensorInfo { get; }
        /// <summary>The operation that failed.</summary>
        public string Operation { get; }

        /// <summary>Creates the exception for <paramref name="operation"/> failing on <paramref name="tensorInfo"/>.</summary>
        public InvalidTensorOperationException(string errorCode, string operation, string tensorInfo, string reason,
            [CallerFilePath] string callerFilePath = "",
            [CallerMemberName] string callerMemberName = "",
            [CallerLineNumber] int callerLineNumber = 0)
            : base(errorCode, $"Invalid tensor operation '{operation}' on {tensorInfo}: {reason}",
                   callerFilePath, callerMemberName, callerLineNumber)
        {
            TensorInfo = tensorInfo;
            Operation = operation;
        }
    }

    /// <summary>
    /// Exception thrown when ONNX node operations fail
    /// </summary>
    public class OnnxNodeException : ShorokooException
    {
        /// <summary>The ONNX op type of the failing node.</summary>
        public string NodeType { get; }
        /// <summary>The name of the failing node.</summary>
        public string NodeName { get; }

        /// <summary>Creates the exception for a failing ONNX node.</summary>
        public OnnxNodeException(string errorCode, string nodeType, string nodeName, string reason)
            : base(errorCode, $"ONNX node '{nodeName}' of type '{nodeType}' failed: {reason}")
        {
            NodeType = nodeType;
            NodeName = nodeName;
        }
    }

    /// <summary>
    /// Exception thrown when module operations fail
    /// </summary>
    public class ModuleException : ShorokooException
    {
        /// <summary>The name of the failing module.</summary>
        public string ModuleName { get; }

        /// <summary>Creates the exception for a failing module operation.</summary>
        public ModuleException(string errorCode, string moduleName, string reason)
            : base(errorCode, $"Module '{moduleName}' operation failed: {reason}")
        {
            ModuleName = moduleName;
        }
        
        /// <summary>Creates the exception for a failing module operation, with an inner exception.</summary>
        public ModuleException(string errorCode, string moduleName, string reason, Exception innerException)
            : base(errorCode, $"Module '{moduleName}' operation failed: {reason}", innerException)
        {
            ModuleName = moduleName;
        }
    }

    /// <summary>
    /// Exception thrown when computation context operations fail
    /// </summary>
    public class ComputeContextException : ShorokooException
    {
        /// <summary>Description of the compute context involved.</summary>
        public string ContextInfo { get; }

        /// <summary>Creates the exception for a failing compute-context operation.</summary>
        public ComputeContextException(string errorCode, string contextInfo, string reason)
            : base(errorCode, $"Compute context operation failed in {contextInfo}: {reason}")
        {
            ContextInfo = contextInfo;
        }
        
        /// <summary>Creates the exception for a failing compute-context operation, with an inner exception.</summary>
        public ComputeContextException(string errorCode, string contextInfo, string reason, Exception innerException)
            : base(errorCode, $"Compute context operation failed in {contextInfo}: {reason}", innerException)
        {
            ContextInfo = contextInfo;
        }
    }

    /// <summary>
    /// Exception thrown when model building or loading fails
    /// </summary>
    public class ModelException : ShorokooException
    {
        /// <summary>Description of the model involved.</summary>
        public string ModelInfo { get; }

        /// <summary>Creates the exception for a failing model operation.</summary>
        public ModelException(string errorCode, string modelInfo, string reason)
            : base(errorCode, $"Model operation failed for {modelInfo}: {reason}")
        {
            ModelInfo = modelInfo;
        }
        
        /// <summary>Creates the exception for a failing model operation, with an inner exception.</summary>
        public ModelException(string errorCode, string modelInfo, string reason, Exception innerException)
            : base(errorCode, $"Model operation failed for {modelInfo}: {reason}", innerException)
        {
            ModelInfo = modelInfo;
        }
    }

    /// <summary>
    /// Exception thrown when <see cref="Globals.StateUpdate{T}(T, T)"/> is called on a tensor
    /// that is not a state variable. State variables must be created by the <c>Init</c> method
    /// of a <c>[StateInitializer]</c> class; the message explains how to declare one.
    /// </summary>
    public class InvalidStateUpdateException : ShorokooException
    {
        /// <summary>Description of the offending tensor (op code / name of its producing node).</summary>
        public string TensorInfo { get; }

        /// <summary>Creates the exception for a StateUpdate call on a non-state variable.</summary>
        public InvalidStateUpdateException(string errorCode, string tensorInfo, string reason)
            : base(errorCode, $"StateUpdate target ({tensorInfo}) is not a state variable: {reason}")
        {
            TensorInfo = tensorInfo;
        }
    }

    /// <summary>
    /// Exception thrown when a gradient is requested for an operator or construct
    /// that Shorokoo's autodiff does not support. The message states whether the
    /// limitation is permanent (mathematically/structurally impossible) or simply
    /// not implemented yet.
    /// </summary>
    public class AutoDiffNotSupportedException : ShorokooException
    {
        /// <summary>The op (or construct) the gradient was requested for.</summary>
        public string OpName { get; }

        /// <summary>Creates the exception for an op without autodiff support.</summary>
        public AutoDiffNotSupportedException(string errorCode, string opName, string reason)
            : base(errorCode, $"Autodiff does not support op '{opName}': {reason}")
        {
            OpName = opName;
        }
    }

    /// <summary>
    /// Exception thrown when reflection or method invocation fails
    /// </summary>
    public class ReflectionException : ShorokooException
    {
        /// <summary>The method that reflection failed on.</summary>
        public string MethodInfo { get; }
        /// <summary>The type that reflection failed on.</summary>
        public string TypeInfo { get; }

        /// <summary>Creates the exception for a failing reflection / invocation operation.</summary>
        public ReflectionException(string errorCode, string methodInfo, string typeInfo, string reason)
            : base(errorCode, $"Reflection operation failed for method '{methodInfo}' on type '{typeInfo}': {reason}")
        {
            MethodInfo = methodInfo;
            TypeInfo = typeInfo;
        }
    }
}